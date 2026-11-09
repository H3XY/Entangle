"""Z2 lattice gauge theory in 2+1 dimensions (Hamiltonian / Kogut-Susskind form).

A toy cousin of Yang-Mills theory: the gauge group is Z2 = {+1, -1} instead of
SU(N), space is a small L x L periodic lattice (a torus), and there is one qubit
per link. The Hamiltonian is

    H = - sum_plaquettes B_p  -  g sum_links X_l,        B_p = Z Z Z Z around plaquette p

B_p is the magnetic term (the lattice version of F_{mu nu}^2's magnetic part),
X_l the electric field on a link. Physical states obey Gauss's law
G_v = prod_{links at v} X_l = +1 at every vertex. We work in the sector with no
electric flux winding around the torus (the sector of the state with every X = +1).

Small g: deconfined (topologically ordered, like the toric code). Large g:
confined (Wilson loops decay with the enclosed area). Wegner duality (1971) maps
the physical sector exactly onto the 2D transverse-field Ising model on the dual
lattice, one spin per plaquette,

    H_dual = - sum_p X_p  -  g sum_links Z_p1 Z_p2      (in the sector prod_p X_p = +1)

with p1, p2 the two plaquettes sharing a link. Its critical point is at
g / 1 = 1 / 3.04438 (2D quantum Ising, Bloete & Deng 2002), so the gauge theory
changes phase at g_c ~= 0.3285 in the infinite lattice. Both phases are gapped;
the gap closes only at g_c. Wilson loops map to products of X_p over the
enclosed plaquettes.
"""
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

G_CRITICAL = 1 / 3.04438  # infinite-lattice confinement transition (via the dual 2D Ising model)


# --------------------------------------------------------------------------- #
# Lattice geometry
# --------------------------------------------------------------------------- #

def h_link(x, y, L):
    return 2 * ((y % L) * L + (x % L))


def v_link(x, y, L):
    return 2 * ((y % L) * L + (x % L)) + 1


def plaquette_index(x, y, L):
    return (y % L) * L + (x % L)


@lru_cache(maxsize=None)
def lattice(L):
    """Links of each plaquette, links at each vertex, and the two plaquettes of each link."""
    plaq_links = [(h_link(x, y, L), v_link(x + 1, y, L), h_link(x, y + 1, L), v_link(x, y, L))
                  for y in range(L) for x in range(L)]
    vertex_links = [(h_link(x, y, L), h_link(x - 1, y, L), v_link(x, y, L), v_link(x, y - 1, L))
                    for y in range(L) for x in range(L)]
    link_plaqs = [[] for _ in range(2 * L * L)]
    for p, links in enumerate(plaq_links):
        for l in links:
            link_plaqs[l].append(p)
    return plaq_links, vertex_links, [tuple(v) for v in link_plaqs]


# --------------------------------------------------------------------------- #
# Dual (Ising) solver: the fast route to the physical spectrum
# --------------------------------------------------------------------------- #

def _dual_hamiltonian(L, g):
    """Sparse H_dual on 2^(L^2) states; bit p of the index = dual spin s_p (Z_p = (-1)^s_p)."""
    P = L * L
    dim = 2 ** P
    idx = np.arange(dim)
    _, _, link_plaqs = lattice(L)
    diag = np.zeros(dim)
    for p1, p2 in link_plaqs:
        diag -= g * (1 - 2 * (((idx >> p1) ^ (idx >> p2)) & 1))
    rows, cols, vals = [idx], [idx], [diag]
    for p in range(P):                               # - X_p flips spin p
        rows.append(idx)
        cols.append(idx ^ (1 << p))
        vals.append(-np.ones(dim))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(dim, dim))


def dual_spectrum(L, g, k=2, penalty=50.0):
    """Lowest k energies and states of the physical (Gauss law, zero winding) sector.

    The sector is prod_p X_p = +1 of the dual model; the odd sector is pushed up by `penalty`.
    """
    P = L * L
    dim = 2 ** P
    idx = np.arange(dim)
    flip_all = sp.csr_matrix((np.ones(dim), (idx, idx ^ (dim - 1))), shape=(dim, dim))
    H = _dual_hamiltonian(L, g) + penalty * (sp.identity(dim) - flip_all) / 2
    if dim <= 512:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = eigsh(H, k=k, which="SA", tol=1e-10)
    order = np.argsort(w)
    return w[order], v[:, order]


def wilson_loop(psi, L, R, T, x0=0, y0=0):
    """<W(C)> for an R x T rectangular loop: the product of X_p over the enclosed plaquettes."""
    mask = 0
    for dy in range(T):
        for dx in range(R):
            mask |= 1 << plaquette_index(x0 + dx, y0 + dy, L)
    idx = np.arange(len(psi))
    return float(np.real(np.vdot(psi, psi[idx ^ mask])))


def electric_field(psi, L):
    """<X_l> averaged over links = <Z_p1 Z_p2> averaged over dual bonds."""
    _, _, link_plaqs = lattice(L)
    idx = np.arange(len(psi))
    prob = np.abs(psi) ** 2
    return float(np.mean([prob @ (1 - 2 * (((idx >> a) ^ (idx >> b)) & 1)) for a, b in link_plaqs]))


# --------------------------------------------------------------------------- #
# Link-qubit picture: the state a quantum device would hold
# --------------------------------------------------------------------------- #

def link_hamiltonian(L, g):
    """Dense H on 2^(2L^2) link qubits (Qiskit order: link l = qubit l). Only for L = 2."""
    n = 2 * L * L
    dim = 2 ** n
    idx = np.arange(dim)
    plaq_links, _, _ = lattice(L)
    H = np.zeros((dim, dim))
    for links in plaq_links:                         # - prod Z: diagonal in the Z basis
        parity = np.zeros(dim, dtype=int)
        for l in links:
            parity ^= (idx >> l) & 1
        H[idx, idx] -= 1 - 2 * parity
    for l in range(n):                               # - g X_l: flips link l
        H[idx, idx ^ (1 << l)] -= g
    return H


def gauss_operators(L):
    """G_v = prod X over the links at each vertex, as bit masks (X flips those bits)."""
    _, vertex_links, _ = lattice(L)
    return [sum(1 << l for l in links) for links in vertex_links]


def dual_to_links(phi, L):
    """Map a dual (plaquette-spin) state to the link-qubit state in the Z basis.

    Dual spins s give the electric configuration x_l = s_p1 XOR s_p2 (X_l = (-1)^x_l),
    i.e. a product state in the X basis. Both s and its complement give the same x.
    A Hadamard on every link then returns Z-basis amplitudes.
    """
    P, n = L * L, 2 * L * L
    _, _, link_plaqs = lattice(L)
    s = np.arange(2 ** P)
    x = np.zeros_like(s)
    for l, (a, b) in enumerate(link_plaqs):
        x |= (((s >> a) ^ (s >> b)) & 1) << l
    amp_x = np.zeros(2 ** n, dtype=complex)
    np.add.at(amp_x, x, phi)
    amp_x /= np.linalg.norm(amp_x)
    t = amp_x.reshape([2] * n)
    Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for ax in range(n):
        t = np.moveaxis(np.tensordot(Hd, t, axes=([1], [ax])), 0, ax)
    psi = t.reshape(-1)
    k = np.argmax(np.abs(psi))
    return psi * (abs(psi[k]) / psi[k])


# --------------------------------------------------------------------------- #
# Dataset for classification (L = 2: 8 link qubits)
# --------------------------------------------------------------------------- #

class GaugeData:
    def __init__(self, g, states, y, L):
        self.g, self.states, self.y, self.L = g, states, y, L
        self.n = 2 * L * L

    def subset(self, idx):
        return GaugeData(self.g[idx], self.states[idx], self.y[idx], self.L)


def make_gauge_data(num_states, seed=0, L=2, g_range=(0.05, 1.0), g_split=G_CRITICAL):
    """Link-qubit ground states for random g; label 1 = confined (g > g_split)."""
    rng = np.random.default_rng(seed)
    gs = rng.uniform(*g_range, num_states)
    states = np.array([dual_to_links(dual_spectrum(L, g, k=1)[1][:, 0], L) for g in gs])
    return GaugeData(gs, states, (gs > g_split).astype(int), L)
