"""Measurement snapshots and the classical features built from them.

A snapshot = one copy of the state, each qubit measured in a chosen Pauli basis
(X, Y or Z), giving one bit per qubit. This is all a classical learner gets in
practice: no amplitudes, just bits.

- random_pauli_snapshots: each snapshot uses a fresh random basis per qubit
  (classical shadows, Huang, Kueng & Preskill 2020).
- xz_snapshots: half the snapshots in the all-Z basis, half in all-X (the
  natural measurements for the Ising chain).

Features: estimated Pauli expectation values <P> from the snapshots whose bases
match P, plus unbiased estimates of the single-qubit purities
Tr(rho_i^2) = (1 + <X_i>^2 + <Y_i>^2 + <Z_i>^2) / 2.
"""
from itertools import combinations, product

import numpy as np

X, Y, Z = 0, 1, 2
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.diag([1, -1j])
_ROT = {X: _H, Y: _H @ _SDG, Z: np.eye(2)}  # rotate basis -> Z before measuring


def basis_probs(psi, n, setting):
    """Outcome distribution over bitstrings when qubit q is measured in basis setting[q]."""
    t = np.asarray(psi, dtype=complex).reshape([2] * n)
    for q, b in enumerate(setting):
        if b != Z:
            ax = n - 1 - q
            t = np.moveaxis(np.tensordot(_ROT[b], t, axes=([1], [ax])), 0, ax)
    return np.abs(t.reshape(-1)) ** 2


def _bits(indices, n):
    return ((np.asarray(indices)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def random_pauli_snapshots(states, n, shots, rng):
    """For each state: (settings (shots, n) in {0:X,1:Y,2:Z}, bits (shots, n))."""
    all_settings = np.array(list(product(range(3), repeat=n)))
    out = []
    for psi in states:
        probs = np.array([basis_probs(psi, n, s) for s in all_settings])
        cum = np.cumsum(probs, axis=1)
        idx = rng.integers(0, len(all_settings), shots)
        u = rng.random(shots)[:, None]
        outcome = np.minimum((cum[idx] < u).sum(axis=1), 2 ** n - 1)
        out.append((all_settings[idx], _bits(outcome, n)))
    return out


def xz_snapshots(states, n, shots, rng):
    """Half the shots in the all-Z basis, half in all-X."""
    out = []
    for psi in states:
        sets, bits = [], []
        for basis, k in ((Z, shots - shots // 2), (X, shots // 2)):
            setting = [basis] * n
            p = basis_probs(psi, n, setting)
            outcome = rng.choice(2 ** n, size=k, p=p / p.sum())
            sets.append(np.tile(setting, (k, 1)))
            bits.append(_bits(outcome, n))
        out.append((np.concatenate(sets), np.concatenate(bits)))
    return out


def snapshots_from_counts(records, n):
    """Hardware results [(setting, counts dict of bitstrings)] -> (settings, bits) arrays.

    Qiskit bitstrings are written with qubit 0 on the right.
    """
    sets, bits = [], []
    for setting, counts in records:
        for bitstring, c in counts.items():
            b = np.array([int(ch) for ch in bitstring[::-1][:n]], dtype=np.int8)
            sets.append(np.tile(setting, (c, 1)))
            bits.append(np.tile(b, (c, 1)))
    return np.concatenate(sets), np.concatenate(bits)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def all_paulis(n, max_weight=None):
    """Pauli strings as tuples of (qubit, basis), up to max_weight non-identity factors."""
    max_weight = n if max_weight is None else max_weight
    out = []
    for w in range(1, max_weight + 1):
        for qs in combinations(range(n), w):
            for bs in product(range(3), repeat=w):
                out.append(tuple(zip(qs, bs)))
    return out


def ising_paulis(n):
    """Z_i, X_i, Z_i Z_j, X_i X_j for all i < j (what XZ snapshots can estimate)."""
    out = []
    for b in (Z, X):
        out += [((i, b),) for i in range(n)]
        out += [((i, b), (j, b)) for i, j in combinations(range(n), 2)]
    return out


def _pauli_matrix(pauli, n):
    mats = {X: np.array([[0, 1], [1, 0]]), Y: np.array([[0, -1j], [1j, 0]]),
            Z: np.diag([1, -1])}
    ops = [np.eye(2)] * n
    for q, b in pauli:
        ops[n - 1 - q] = mats[b]
    out = ops[0]
    for m in ops[1:]:
        out = np.kron(out, m)
    return out


def exact_expectations(states, n, paulis):
    """<P> for every state and Pauli string (the infinite-shot limit of the estimates)."""
    S = np.asarray(states)
    return np.stack([np.real(np.einsum("ni,ij,nj->n", S.conj(), _pauli_matrix(p, n), S))
                     for p in paulis], axis=1)


def estimate_expectations(settings, bits, paulis):
    """Snapshot estimates of <P> and unbiased estimates of <P>^2 for each Pauli string."""
    signs = 1 - 2 * bits.astype(float)          # +1 for outcome 0, -1 for outcome 1
    est, sq = np.zeros(len(paulis)), np.zeros(len(paulis))
    for k, pauli in enumerate(paulis):
        match = np.ones(len(settings), dtype=bool)
        val = np.ones(len(settings))
        for q, b in pauli:
            match &= settings[:, q] == b
            val *= signs[:, q]
        x = val[match]
        m = len(x)
        if m:
            est[k] = x.mean()
        if m >= 2:
            sq[k] = (x.sum() ** 2 - m) / (m * (m - 1))   # unbiased estimate of <P>^2
    return est, sq


def shadow_features(snapshots, n, paulis, purities=True):
    """Feature matrix: [<P> estimates..., single-qubit purity estimates...] per state."""
    idx = {p: k for k, p in enumerate(paulis)}
    rows = []
    for settings, bits in snapshots:
        est, sq = estimate_expectations(settings, bits, paulis)
        pur = [(1 + sum(sq[idx[((q, b),)]] for b in (X, Y, Z) if ((q, b),) in idx)) / 2
               for q in range(n)]
        rows.append(np.concatenate([est, pur]) if purities else est)
    return np.array(rows)


def exact_features(states, n, paulis, purities=True):
    """Same features with exact expectation values (infinite shots)."""
    E = exact_expectations(states, n, paulis)
    if not purities:
        return E
    idx = {p: k for k, p in enumerate(paulis)}
    pur = np.stack([(1 + sum(E[:, idx[((q, b),)]] ** 2 for b in (X, Y, Z) if ((q, b),) in idx)) / 2
                    for q in range(n)], axis=1)
    return np.concatenate([E, pur], axis=1)
