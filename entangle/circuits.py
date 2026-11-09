"""Trainable quantum models (ansatz circuits + single-outcome readout).

Each model is a `Model`: a parameterised circuit with no measurements, plus a
readout mask saying which measured bitstrings count as "class 1". The model's
output is P(class 1) = probability of measuring one of those bitstrings.

Parameters can be shared between gates (the QCNN shares weights within a
layer). Each gate gets its own *occurrence* parameter and `tie[k]` says which
trainable weight drives occurrence k, so the parameter-shift rule still applies
gate by gate.

Why no data re-uploading here (unlike the AGI text classifier)? The input is an
unknown quantum state, and an unknown state cannot be copied (no-cloning). The
quantum analogue of re-uploading is to give the model several *copies* of the
state, which is what the two-copy model does.
"""
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


@dataclass
class Model:
    name: str
    circuit: QuantumCircuit       # acts on num_qubits; parameters = occurrence params
    occ: list                     # occurrence parameters, in tie order
    tie: np.ndarray               # occ k -> weight index
    num_weights: int
    readout: np.ndarray           # bool mask over 2^num_qubits basis states
    copies: int = 1               # copies of the input state consumed per shot
    description: str = ""

    @property
    def num_qubits(self):
        return self.circuit.num_qubits

    def bound(self, weights):
        w = np.asarray(weights)[self.tie]
        return self.circuit.assign_parameters(dict(zip(self.occ, w)))

    def interpret(self, index):
        """Bitstring index -> class (for qiskit-machine-learning's SamplerQNN)."""
        return int(self.readout[index])


class _Builder:
    """Small helper that hands out tied occurrence parameters."""

    def __init__(self, num_qubits):
        self.qc = QuantumCircuit(num_qubits)
        self.occ, self.tie = [], []
        self.num_weights = 0

    def new_weights(self, k):
        start = self.num_weights
        self.num_weights += k
        return list(range(start, start + k))

    def p(self, w):
        par = Parameter(f"t{len(self.occ)}")
        self.occ.append(par)
        self.tie.append(w)
        return par

    def model(self, name, readout, copies=1, description=""):
        return Model(name, self.qc, self.occ, np.array(self.tie, dtype=int), self.num_weights,
                     np.asarray(readout, dtype=bool), copies, description)


def _bit(index, q):
    return (index >> q) & 1


def qubit_mask(num_qubits, q):
    """Readout: class 1 iff qubit q measures 1."""
    return np.array([_bit(i, q) for i in range(2 ** num_qubits)], dtype=bool)


# --------------------------------------------------------------------------- #
# Stage 1 models
# --------------------------------------------------------------------------- #


def one_copy_model(n, reps=3):
    """Hardware-efficient ansatz on the n input qubits, read out qubit 0.

    Layers of RY, RZ on every qubit and a CX chain. Its output is
    Tr(M rho) for one trainable observable M: linear in the state.
    """
    b = _Builder(n)
    for layer in range(reps + 1):
        for q in range(n):
            wy, wz = b.new_weights(2)
            b.qc.ry(b.p(wy), q)
            b.qc.rz(b.p(wz), q)
        if layer < reps:
            for q in range(n - 1):
                b.qc.cx(q, q + 1)
    return b.model(f"QNN 1-copy", qubit_mask(n, 0), copies=1,
                   description=f"{n} qubits, {reps}-layer RY/RZ+CX ansatz, read out qubit 0")


def _su2_block(b, a, c, w=None):
    """Trainable two-qubit block: RY RZ on both, CX, RY RZ on both (8 weights).

    Pass `w` (8 weight indices) to reuse existing weights instead of making new ones."""
    w = list(b.new_weights(8)) if w is None else list(w)
    k = 0
    for layer in range(2):
        for q in (a, c):
            b.qc.ry(b.p(w[k]), q)
            b.qc.rz(b.p(w[k + 1]), q)
            k += 2
        if layer == 0:
            b.qc.cx(a, c)


def two_copy_model(n, readout="any", shared=False):
    """Two copies of the state (2n qubits): a trainable block on each pair (q_i, q_i')
    of matching qubits, then class 1 iff any pair measures 11.

    The block can learn a Bell-basis measurement, in which the singlet (the only
    state that is antisymmetric under swapping the copies) shows up as 11. That is
    the SWAP test: P(singlet on pair i) = (1 - Tr rho_i^2) / 2, which is 0 for every
    qubit of a product state. Output is quadratic in rho, which a single copy
    cannot give.
    """
    m = 2 * n
    b = _Builder(m)
    w = b.new_weights(8) if shared else None   # shared: one block reused on every pair
    for i in range(n):
        _su2_block(b, i, n + i, w)
    singlets = np.array([sum(_bit(k, i) and _bit(k, n + i) for i in range(n))
                         for k in range(2 ** m)])
    if readout == "any":
        mask, rule, name = singlets > 0, "any pair reads 11", "QNN 2-copy"
    elif readout == "parity":
        # SWAP of the whole register = product of pair SWAPs = (-1)^(number of singlets),
        # so P(odd number of pairs read 11) = (1 - Tr rho^2) / 2: the global purity.
        mask, rule, name = singlets % 2 == 1, "an odd number of pairs read 11", "QNN 2-copy (parity)"
    else:
        raise ValueError(readout)
    if shared:
        name += ", shared"
    return b.model(name, mask, copies=2,
                   description=f"{m} qubits (2 copies), "
                               f"{'one shared trainable block' if shared else 'a trainable block'} per qubit pair, "
                               f"class 1 if {rule}")


# --------------------------------------------------------------------------- #
# Stage 2: quantum convolutional neural network
# --------------------------------------------------------------------------- #


def _conv(b, w, a, c):
    """Two-qubit convolution unit (8 shared weights): RY RZ on both, CX, RY RZ on both.

    (The 3-parameter unit from the Qiskit QCNN tutorial was tried first: on the
    Ising ground states, which are symmetric under flipping every spin, its output
    was exactly 0.5 for every input at every weight setting we tried, so it could not learn.)
    """
    for q, k in ((a, 0), (c, 2)):
        b.qc.ry(b.p(w[k]), q)
        b.qc.rz(b.p(w[k + 1]), q)
    b.qc.cx(a, c)
    for q, k in ((a, 4), (c, 6)):
        b.qc.ry(b.p(w[k]), q)
        b.qc.rz(b.p(w[k + 1]), q)


def _pool(b, w, source, sink):
    """Pooling unit (4 shared weights): rotate `source`, CX source -> sink, rotate `sink`.

    Coherent version of "measure the source, rotate the sink depending on the
    result"; the source qubit is not used again.
    """
    b.qc.ry(b.p(w[0]), source)
    b.qc.rz(b.p(w[1]), source)
    b.qc.cx(source, sink)
    b.qc.ry(b.p(w[2]), sink)
    b.qc.rz(b.p(w[3]), sink)


def qcnn_model(n):
    """QCNN: [convolution on neighbours, pooling halves the qubits] until one qubit is left.

    Weights are shared within each convolution / pooling layer (translation
    invariance), so the parameter count grows like log(n), not n.
    Pooling is done coherently (no mid-circuit measurement), which is equivalent
    for the final single-qubit readout.
    """
    b = _Builder(n)
    active = list(range(n))
    while len(active) > 1:
        wc = b.new_weights(8)
        k = len(active)
        pairs = [(active[i], active[i + 1]) for i in range(0, k - 1, 2)]
        pairs += [(active[i], active[i + 1]) for i in range(1, k - 1, 2)]
        if k > 2:
            pairs.append((active[-1], active[0]))  # close the ring
        for a, c in pairs:
            _conv(b, wc, a, c)
        wp = b.new_weights(4)
        keep = []
        for i in range(0, k - 1, 2):
            _pool(b, wp, active[i], active[i + 1])
            keep.append(active[i + 1])
        if k % 2:
            keep.append(active[-1])
        active = keep
    out = active[0]
    return b.model("QCNN", qubit_mask(n, out), copies=1,
                   description=f"{n}-qubit QCNN, {b.num_weights} shared weights, read out qubit {out}")


def hea_model(n, reps=3):
    """Unstructured hardware-efficient ansatz with the same readout, as a QCNN ablation."""
    m = one_copy_model(n, reps)
    m.name = "QNN (generic ansatz)"
    return m
