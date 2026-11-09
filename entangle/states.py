"""Quantum datasets: random product/entangled states and Ising ground states.

Every state here comes from a *circuit* (or, for Ising ground states, from an
exact-diagonalisation state vector that a state-preparation circuit loads).
Quantum models only ever see the state by running that circuit and acting on
the qubits. Classical baselines see either the full state vector (a privileged,
exponentially large description) or measurement snapshots (see measure.py).
"""
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
# Entanglement measures
# --------------------------------------------------------------------------- #


def reduced_purities(psi, n):
    """Tr(rho_i^2) for every single qubit i of an n-qubit pure state."""
    psi = np.asarray(psi).reshape([2] * n)  # axis k = qubit n-1-k (Qiskit order)
    out = np.empty(n)
    for q in range(n):
        ax = n - 1 - q
        m = np.moveaxis(psi, ax, 0).reshape(2, -1)
        rho = m @ m.conj().T
        out[q] = np.real(np.trace(rho @ rho))
    return out


def meyer_wallach(psi, n):
    """Meyer-Wallach global entanglement Q in [0, 1]: 0 iff the state is a product state.

    Q = 2 * (1 - mean_i Tr(rho_i^2)), the average linear entropy of single qubits.
    For 2 qubits, Q equals the squared concurrence.
    """
    return float(2.0 * (1.0 - reduced_purities(psi, n).mean()))


def concurrence(psi):
    """Concurrence of a 2-qubit pure state, |<psi|Y(x)Y|psi*>| = 2|ad - bc|."""
    a, b, c, d = np.asarray(psi).reshape(4)
    return float(2 * abs(a * d - b * c))


# --------------------------------------------------------------------------- #
# Stage 1: random product vs entangled states
# --------------------------------------------------------------------------- #


def prep_template(n, layers=2):
    """Parameterised state-preparation circuit.

    local layer (RZ RY RZ on every qubit), then `layers` x [RZZ entanglers on a
    chain, local layer]. With every entangler angle 0 it prepares a random
    product state; with random entangler angles, an entangled one.
    Returns (circuit, local_params, entangler_params).
    """
    n_loc = 3 * n * (layers + 1)
    n_ent = (n - 1) * layers
    loc = ParameterVector("loc", n_loc)
    ent = ParameterVector("ent", n_ent)
    qc = QuantumCircuit(n, name="prep")
    li, ei = 0, 0

    def local():
        nonlocal li
        for q in range(n):
            qc.rz(loc[li], q)
            qc.ry(loc[li + 1], q)
            qc.rz(loc[li + 2], q)
            li += 3

    local()
    for layer in range(layers):
        pairs = [(q, q + 1) for q in range(n - 1)]
        if layer % 2:  # alternate the order so depth stays low
            pairs = pairs[::-1]
        for a, b in pairs:
            qc.rzz(ent[ei], a, b)
            ei += 1
        local()
    return qc, list(loc), list(ent)


@dataclass
class EntanglementData:
    n: int
    params: np.ndarray        # (N, n_params) values for the prep template, in template order
    states: np.ndarray        # (N, 2^n) complex state vectors (for classical baselines / checks)
    mw: np.ndarray            # Meyer-Wallach Q per state
    y: np.ndarray             # 1 = entangled, 0 = product
    layers: int = 2
    q_min: float = 0.2
    extra: dict = field(default_factory=dict)

    def circuit(self, i):
        qc, loc, ent = prep_template(self.n, self.layers)
        return qc.assign_parameters(dict(zip(loc + ent, self.params[i])))

    def subset(self, idx):
        return EntanglementData(self.n, self.params[idx], self.states[idx], self.mw[idx],
                                self.y[idx], self.layers, self.q_min)


def make_entanglement_data(n, num_states, seed=0, layers=2, q_min=0.2):
    """Balanced dataset: half product states, half entangled states with Q >= q_min.

    Local rotations are random in both classes, so the two classes have the same
    average state (the maximally mixed state); only entanglement tells them apart.
    Entangled states with Q < q_min are rejected so the label has a margin.
    """
    rng = np.random.default_rng(seed)
    qc, loc, ent = prep_template(n, layers)
    order = loc + ent
    rows, states, mws, ys = [], [], [], []
    labels = np.array([0, 1] * (num_states // 2) + [1] * (num_states % 2))
    rng.shuffle(labels)
    for label in labels:
        while True:
            # Haar-ish local rotations: uniform Euler angles with a cos-distributed polar angle
            lv = rng.uniform(-np.pi, np.pi, len(loc)).reshape(-1, 3)
            lv[:, 1] = np.arccos(rng.uniform(-1, 1, len(lv)))
            lv = lv.reshape(-1)
            ev = rng.uniform(-np.pi, np.pi, len(ent)) if label else np.zeros(len(ent))
            vals = np.concatenate([lv, ev])
            psi = Statevector(qc.assign_parameters(dict(zip(order, vals)))).data
            q = meyer_wallach(psi, n)
            if label == 0 or q >= q_min:
                break
        rows.append(vals)
        states.append(psi)
        mws.append(q)
        ys.append(label)
    return EntanglementData(n, np.array(rows), np.array(states), np.array(mws),
                            np.array(ys), layers, q_min)


# --------------------------------------------------------------------------- #
# Stage 2: transverse-field Ising chain ground states
# --------------------------------------------------------------------------- #

_X = np.array([[0, 1], [1, 0]], dtype=float)
_Z = np.array([[1, 0], [0, -1]], dtype=float)
_I = np.eye(2)


def _site_op(op, site, n):
    """Operator `op` on qubit `site` (Qiskit little-endian order) as a dense matrix."""
    mats = [_I] * n
    mats[n - 1 - site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def tfim_hamiltonian(n, h, J=1.0, periodic=False):
    """H = -J sum Z_i Z_{i+1} - h sum X_i on an open (default) or periodic chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim))
    bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if periodic and n > 2 else [])
    for i, j in bonds:
        H -= J * _site_op(_Z, i, n) @ _site_op(_Z, j, n)
    for i in range(n):
        H -= h * _site_op(_X, i, n)
    return H


def tfim_ground_state(n, h, J=1.0, periodic=False):
    """Exact ground state in the even-parity sector (prod X_i = +1).

    For small h the two lowest levels are almost degenerate; restricting to the
    symmetry sector of the true finite-size ground state avoids numerical mixing.
    """
    H = tfim_hamiltonian(n, h, J, periodic)
    # Parity of a Z-basis index under prod X is not diagonal; build the sector
    # from symmetric combinations |b> + |~b> instead.
    dim = 2 ** n
    full = dim - 1
    basis = []
    for b in range(dim):
        if b < (b ^ full):
            v = np.zeros(dim)
            v[b] = v[b ^ full] = 1 / np.sqrt(2)
            basis.append(v)
    B = np.array(basis).T                    # dim x dim/2
    w, v = np.linalg.eigh(B.T @ H @ B)
    psi = B @ v[:, 0]
    psi *= np.sign(psi[np.argmax(np.abs(psi))])  # fix the sign (amplitudes are >= 0)
    return psi.astype(complex), float(w[0])


@dataclass
class PhaseData:
    n: int
    h: np.ndarray            # transverse field per state
    states: np.ndarray       # (N, 2^n) ground states
    y: np.ndarray            # 1 = ordered (h < 1), 0 = disordered (h > 1)

    def circuit(self, i):
        """State-preparation circuit (exact, from the ED state vector)."""
        from qiskit.circuit.library import StatePreparation
        qc = QuantumCircuit(self.n, name="prep")
        qc.append(StatePreparation(self.states[i]), range(self.n))
        return qc

    def subset(self, idx):
        return PhaseData(self.n, self.h[idx], self.states[idx], self.y[idx])


def make_phase_data(n, num_states, seed=0, h_range=(0.1, 2.0), exclude=0.0):
    """Ground states for random h in h_range; label = ordered (h < 1).

    `exclude` drops |h - 1| < exclude (the critical region, where finite chains are
    neither clearly ordered nor disordered).
    """
    rng = np.random.default_rng(seed)
    hs = []
    while len(hs) < num_states:
        h = rng.uniform(*h_range)
        if abs(h - 1) >= exclude:
            hs.append(h)
    hs = np.array(hs)
    states = np.array([tfim_ground_state(n, h)[0] for h in hs])
    return PhaseData(n, hs, states, (hs < 1).astype(int))
