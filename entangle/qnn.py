"""Exact training and evaluation of the quantum models.

The model sees a state only by running the trainable circuit on it and
measuring. For a pure input state |psi> (prepared by a circuit),

    P(class 1) = sum over readout bitstrings b of |<b| U(w) |psi>|^2

which is exactly what a noise-free device returns in the infinite-shot limit.
We compute it with the full unitary U(w) applied to the prepared states, which
is the same as composing prep circuit + model circuit and simulating each one
(tests/test_qnn.py checks this against qiskit-machine-learning's SamplerQNN on
the composed circuits), but is much faster for hundreds of states.

Training follows the AGI project: exact probabilities (no shot noise during
training), cross-entropy loss, L-BFGS-B, gradients from the parameter-shift rule.
Mixed input states are given by a purification: an array A of shape (N, dim, r)
with rho = A A^dagger (r = rank). The model acts on the system qubits only, so
P(class 1) = sum over the r columns of A (and r^2 columns of A (x) A for two
copies) of the pure-state probability. Each column is simulated as its own row.

A 2-parameter logistic calibration sigmoid(a * (p - b)) maps P(class 1) to a class
probability, so models whose raw output lives in a narrow range (the two-copy
model) are judged fairly. It is trained jointly with the circuit weights.
"""
import time

import numpy as np
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

from .sim import CompiledCircuit


def tensor_copies(states, copies):
    """|psi>^{(x) copies} for each row (copy 1 on the lowest qubits)."""
    out = states
    for _ in range(copies - 1):
        out = np.einsum("ni,nj->nij", states, out).reshape(len(states), -1)
    return out


class Batch:
    """Simulator rows for a set of input states, and which state each row belongs to."""

    def __init__(self, states, copies):
        S = np.asarray(states, dtype=complex)
        if S.ndim == 2:                      # pure states: one row each
            self.X = tensor_copies(S, copies)
            self.owner = np.arange(len(S))
        else:                                # purifications (N, dim, r): rows = columns of A^(x)copies
            N, dim, r = S.shape
            A = S
            for _ in range(copies - 1):      # copy 1 on the lowest qubits, as in tensor_copies
                A = np.einsum("nik,njl->nijkl", S, A).reshape(N, dim * A.shape[1], r * A.shape[2])
            self.X = np.transpose(A, (0, 2, 1)).reshape(-1, A.shape[1])
            self.owner = np.repeat(np.arange(N), A.shape[2])
        self.n_states = len(S)

    def total(self, row_values):
        return np.bincount(self.owner, weights=row_values, minlength=self.n_states)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class QuantumClassifier:
    """A Model (see circuits.py) plus trained weights and calibration."""

    def __init__(self, model):
        self.model = model
        self.weights = None
        self.a, self.b = 10.0, 0.5
        self.history = []
        self.train_seconds = 0.0
        self._sim = CompiledCircuit(model.circuit, model.occ)

    # ---- exact outputs ---------------------------------------------------- #
    def _inputs(self, states):
        return Batch(states, self.model.copies)

    def _observable(self, occ_values):
        """M = U^dag P U for the given occurrence-parameter values."""
        qc = self.model.circuit.assign_parameters(dict(zip(self.model.occ, occ_values)))
        U = Operator(qc).data
        Ur = U[self.model.readout]
        return Ur.conj().T @ Ur

    def _p_operator(self, occ_values, X):
        """Reference implementation via the full unitary (used in tests)."""
        M = self._observable(occ_values)
        return X.total(np.real(np.einsum("ni,ij,nj->n", X.X.conj(), M, X.X)))

    def _p(self, occ_values, X):
        return X.total(self._sim.probs(X.X, occ_values, self.model.readout))

    def raw_proba(self, states, weights=None):
        """Exact P(class 1) for each state (infinite shots, no noise)."""
        w = self.weights if weights is None else weights
        return self._p(np.asarray(w)[self.model.tie], self._inputs(states))

    def calibrate(self, p):
        return _sigmoid(self.a * (np.asarray(p) - self.b))

    def predict(self, states, shots=None, rng=None):
        """Class predictions. shots=None: exact; otherwise P(class 1) is estimated
        from `shots` single-shot measurements (each consuming `copies` states)."""
        p = self.raw_proba(states)
        if shots:
            rng = rng or np.random.default_rng(0)
            p = rng.binomial(shots, np.clip(p, 0, 1)) / shots
        return (self.calibrate(p) >= 0.5).astype(int)

    def predict_from_p(self, p):
        return (self.calibrate(p) >= 0.5).astype(int)

    # ---- training --------------------------------------------------------- #
    def _loss_grad(self, params, X, y, shift_rule=False):
        model = self.model
        w, a, b = params[:-2], params[-2], params[-1]
        occ = w[model.tie]
        p = self._p(occ, X)
        q = _sigmoid(a * (p - b))
        eps = 1e-9
        loss = -np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))
        dl_dz = (q - y) / len(y)                  # d loss / d z,   z = a (p - b)
        dl_dp = dl_dz * a
        if shift_rule:                           # parameter-shift rule, one gate at a time
            g_occ = np.empty(len(occ))
            for k in range(len(occ)):
                plus, minus = occ.copy(), occ.copy()
                plus[k] += np.pi / 2
                minus[k] -= np.pi / 2
                g_occ[k] = dl_dp @ (self._p(plus, X) - self._p(minus, X)) / 2
        else:                                    # adjoint method, same result, ~K times faster
            _, g_occ = self._sim.probs_and_grad(X.X, occ, model.readout, dl_dp[X.owner])
        g_w = np.bincount(model.tie, weights=g_occ, minlength=model.num_weights)
        g_a = np.sum(dl_dz * (p - b))
        g_b = -np.sum(dl_dz * a)
        self._last_loss = loss
        return loss, np.concatenate([g_w, [g_a, g_b]])

    def fit(self, states, y, maxiter=150, restarts=3, seed=0, verbose=False, init=None):
        """Train with L-BFGS-B from `restarts` random starts, or from `init` (weights) if given."""
        X = self._inputs(states)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)
        best = None
        t0 = time.time()
        starts = [np.asarray(init, dtype=float)] if init is not None else \
            [rng.uniform(-np.pi, np.pi, self.model.num_weights) for _ in range(restarts)]
        for r, w0 in enumerate(starts):  # keep the best training loss
            p0 = self._p(w0[self.model.tie], X)
            x0 = np.concatenate([w0, [10.0, float(np.median(p0))]])
            hist = []
            res = minimize(self._loss_grad, x0, args=(X, y), jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter},
                           callback=lambda xk: hist.append(float(self._last_loss)))
            if verbose:
                print(f"    restart {r}: loss {res.fun:.4f} after {res.nit} iterations", flush=True)
            if best is None or res.fun < best[0].fun:
                best = (res, hist)
        res, self.history = best
        self.weights, self.a, self.b = res.x[:-2], res.x[-2], res.x[-1]
        self.final_loss = float(res.fun)
        self.train_seconds = time.time() - t0
        return self
