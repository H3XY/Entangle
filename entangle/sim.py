"""Small batched state-vector simulator with adjoint gradients.

Runs a model circuit on many input states at once and returns exact readout
probabilities plus their gradients with respect to every rotation angle, at
the cost of about three forward passes (adjoint method) instead of two passes
per parameter (parameter shift). Gives the same numbers as parameter shift
(checked in tests/test_qnn.py), just much faster for the 8-qubit models.

Supports the gates the models use: rx, ry, rz with trainable or fixed angles,
plus any fixed gate Qiskit can turn into a matrix (cx, h, ...).
"""
import numpy as np
from qiskit.quantum_info import Operator


def _rot(name, t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    if name == "ry":
        G = np.array([[c, -s], [s, c]], dtype=complex)
        dG = 0.5 * np.array([[-s, -c], [c, -s]], dtype=complex)
    elif name == "rz":
        e = np.exp(-0.5j * t)
        G = np.array([[e, 0], [0, e.conjugate()]])
        dG = np.array([[-0.5j * e, 0], [0, 0.5j * e.conjugate()]])
    elif name == "rx":
        G = np.array([[c, -1j * s], [-1j * s, c]])
        dG = 0.5 * np.array([[-s, -1j * c], [-1j * c, -s]])
    else:
        raise ValueError(name)
    return G, dG


class CompiledCircuit:
    """A model circuit turned into a gate list that knows which angles are trainable."""

    def __init__(self, circuit, occ):
        self.m = circuit.num_qubits
        index = {p: k for k, p in enumerate(occ)}
        self.ops = []  # (kind, qubits, payload)
        for inst in circuit.data:
            op = inst.operation
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            if op.name in ("barrier", "measure"):
                continue
            if op.name in ("rx", "ry", "rz"):
                theta = op.params[0]
                if hasattr(theta, "parameters") and theta.parameters:
                    (par,) = theta.parameters
                    if theta != par:
                        raise ValueError("only plain parameters are supported in rotations")
                    self.ops.append(("param", qubits, (op.name, index[par])))
                    continue
                self.ops.append(("fixed", qubits, _rot(op.name, float(theta))[0]))
            else:
                self.ops.append(("fixed", qubits, Operator(op).data))

    # -- tensor helpers -- #
    def _axes(self, qubits):
        return [1 + (self.m - 1 - q) for q in reversed(qubits)]

    def _apply(self, psi, G, qubits):
        k = len(qubits)
        Gt = G.reshape([2] * (2 * k))
        ax = self._axes(qubits)
        out = np.tensordot(Gt, psi, axes=(list(range(k, 2 * k)), ax))
        return np.moveaxis(out, list(range(k)), ax)

    def _gates(self, occ_values):
        gates = []
        for kind, qubits, payload in self.ops:
            if kind == "param":
                name, k = payload
                G, dG = _rot(name, occ_values[k])
                gates.append((G, dG, qubits, k))
            else:
                gates.append((payload, None, qubits, None))
        return gates

    def run(self, states, occ_values):
        """Final states for each input row."""
        psi = np.asarray(states, dtype=complex).reshape([len(states)] + [2] * self.m)
        for G, _, qubits, _ in self._gates(occ_values):
            psi = self._apply(psi, G, qubits)
        return psi.reshape(len(states), -1)

    def probs(self, states, occ_values, mask):
        out = self.run(states, occ_values)
        return np.sum(np.abs(out[:, mask]) ** 2, axis=1)

    def probs_and_grad(self, states, occ_values, mask, weights):
        """p_n = P(readout) per state, and d(sum_n weights_n p_n)/d(angle_k) for every angle."""
        N = len(states)
        gates = self._gates(occ_values)
        phi = np.asarray(states, dtype=complex).reshape([N] + [2] * self.m)
        for G, _, qubits, _ in gates:
            phi = self._apply(phi, G, qubits)
        flat = phi.reshape(N, -1)
        p = np.sum(np.abs(flat[:, mask]) ** 2, axis=1)
        lam = np.where(mask[None, :], flat, 0) * np.asarray(weights)[:, None]
        lam = lam.reshape(phi.shape)
        grad = np.zeros(len(occ_values))
        for G, dG, qubits, k in reversed(gates):
            Gd = G.conj().T
            phi = self._apply(phi, Gd, qubits)
            if k is not None:
                grad[k] += 2 * np.real(np.vdot(lam, self._apply(phi, dG, qubits)))
            lam = self._apply(lam, Gd, qubits)
        return p, grad
