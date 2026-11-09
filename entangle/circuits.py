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


def _su2_block(b, a, c):
    """Trainable two-qubit block: RY RZ on both, CX, RY RZ on both (8 weights)."""
    for _ in range(2):
        for q in (a, c):
            wy, wz = b.new_weights(2)
            b.qc.ry(b.p(wy), q)
            b.qc.rz(b.p(wz), q)
        if _ == 0:
            b.qc.cx(a, c)


def two_copy_model(n):
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
    for i in range(n):
        _su2_block(b, i, n + i)
    readout = np.array([any(_bit(k, i) and _bit(k, n + i) for i in range(n))
                        for k in range(2 ** m)], dtype=bool)
    return b.model("QNN 2-copy", readout, copies=2,
                   description=f"{m} qubits (2 copies), trainable block per qubit pair, "
                               f"class 1 if any pair reads 11")


# --------------------------------------------------------------------------- #
# Stage 2: quantum convolutional neural network
# --------------------------------------------------------------------------- #


def _conv(b, w, a, c):
    """Two-qubit convolution unit (3 shared weights), as in the Qiskit QCNN tutorial."""
    b.qc.rz(-np.pi / 2, c)
    b.qc.cx(c, a)
    b.qc.rz(b.p(w[0]), a)
    b.qc.ry(b.p(w[1]), c)
    b.qc.cx(a, c)
    b.qc.ry(b.p(w[2]), c)
    b.qc.cx(c, a)
    b.qc.rz(np.pi / 2, a)


def _pool(b, w, source, sink):
    """Pooling unit: entangle `source` into `sink`, then `source` is discarded."""
    b.qc.rz(-np.pi / 2, sink)
    b.qc.cx(sink, source)
    b.qc.rz(b.p(w[0]), source)
    b.qc.ry(b.p(w[1]), sink)
    b.qc.cx(source, sink)
    b.qc.ry(b.p(w[2]), sink)


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
        wc = b.new_weights(3)
        k = len(active)
        pairs = [(active[i], active[i + 1]) for i in range(0, k - 1, 2)]
        pairs += [(active[i], active[i + 1]) for i in range(1, k - 1, 2)]
        if k > 2:
            pairs.append((active[-1], active[0]))  # close the ring
        for a, c in pairs:
            _conv(b, wc, a, c)
        wp = b.new_weights(3)
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
