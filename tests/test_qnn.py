"""Correctness checks for the states, the fast simulator and the gradients."""
import os
import sys
import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entangle.circuits import one_copy_model, qcnn_model, two_copy_model  # noqa: E402
from entangle.qnn import QuantumClassifier  # noqa: E402
from entangle.states import (concurrence, make_entanglement_data, meyer_wallach,  # noqa: E402
                             prep_template)


def test_meyer_wallach_matches_concurrence_and_product_states():
    d = make_entanglement_data(2, 40, seed=3)
    assert np.allclose([concurrence(s) ** 2 for s in d.states], d.mw)
    assert np.all(d.mw[d.y == 0] < 1e-12)
    assert np.all(d.mw[d.y == 1] >= d.q_min)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert meyer_wallach(bell, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("build,n", [(one_copy_model, 3), (two_copy_model, 2), (qcnn_model, 4)])
def test_simulator_matches_full_unitary_and_gradients_match_parameter_shift(build, n):
    rng = np.random.default_rng(0)
    model = build(n)
    clf = QuantumClassifier(model)
    states = rng.normal(size=(12, 2 ** n)) + 1j * rng.normal(size=(12, 2 ** n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    X = clf._inputs(states)
    params = np.concatenate([rng.uniform(-3, 3, model.num_weights), [3.0, 0.4]])
    occ = params[:-2][model.tie]
    assert np.allclose(clf._p(occ, X), clf._p_operator(occ, X))
    y = rng.integers(0, 2, 12).astype(float)
    _, g_adjoint = clf._loss_grad(params, X, y)
    _, g_shift = clf._loss_grad(params, X, y, shift_rule=True)
    assert np.allclose(g_adjoint, g_shift, atol=1e-10)


@pytest.mark.parametrize("build", [one_copy_model, two_copy_model])
def test_matches_qiskit_machine_learning_sampler_qnn_on_composed_circuit(build):
    """Prep circuit + model circuit run through SamplerQNN gives the same P(class 1)."""
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.neural_networks import SamplerQNN

    n = 2
    model = build(n)
    prep, loc, ent = prep_template(n)
    qc = QuantumCircuit(model.num_qubits)
    for c in range(model.copies):  # the same prepared state on each copy register
        qc.compose(prep, qubits=range(c * n, (c + 1) * n), inplace=True)
    qc.compose(model.circuit, inplace=True)
    qc.measure_all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qnn = SamplerQNN(circuit=qc, input_params=loc + ent, weight_params=model.occ,
                         sampler=Sampler(), interpret=model.interpret, output_shape=2)
    d = make_entanglement_data(n, 6, seed=5)
    occ = np.random.default_rng(1).uniform(-3, 3, len(model.occ))
    # SamplerQNN orders inputs by the given input_params; data rows are already in loc+ent order
    p_qiskit = np.asarray(qnn.forward(d.params, occ))[:, 1]
    clf = QuantumClassifier(model)
    p_ours = clf._p(occ, clf._inputs(d.states))
    assert np.allclose(p_qiskit, p_ours, atol=1e-8)


def test_two_copy_model_learns_entanglement():
    d = make_entanglement_data(2, 120, seed=11)
    clf = QuantumClassifier(two_copy_model(2)).fit(d.states[:60], d.y[:60], maxiter=100, restarts=2)
    assert np.mean(clf.predict(d.states[60:]) == d.y[60:]) > 0.95


def test_ising_ground_state_is_the_true_ground_state():
    from entangle.states import tfim_ground_state, tfim_hamiltonian
    for h in (0.3, 1.0, 1.7):
        psi, e = tfim_ground_state(5, h)
        H = tfim_hamiltonian(5, h)
        assert e == pytest.approx(np.linalg.eigvalsh(H)[0])
        assert np.allclose(H @ psi, e * psi, atol=1e-8)


def test_qcnn_output_depends_on_ising_state():
    """Regression: the tutorial QCNN blocks gave a constant 0.5 on Ising ground states."""
    from entangle.states import make_phase_data
    d = make_phase_data(4, 8, seed=2)
    clf = QuantumClassifier(qcnn_model(4))
    p = clf.raw_proba(d.states, np.random.default_rng(0).uniform(-3, 3, clf.model.num_weights))
    assert np.ptp(p) > 1e-3


def test_hardware_count_parsing_matches_simulated_snapshots():
    """Counts from a real sampler (Aer, noiseless) give the same features as our own snapshots."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from entangle.measure import (Z, X, exact_features, ising_paulis, measurement_circuit,
                                  shadow_features, snapshots_from_counts)
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.h(2)                               # |+> on qubit 2, |0> on qubit 1, |1> on qubit 0
    settings = [[Z, Z, Z], [X, X, X]]
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    circuits = pm.run([measurement_circuit(qc, s) for s in settings])
    counts = [r.data.meas.get_counts() for r in SamplerV2(mode=backend).run(circuits, shots=200).result()]
    feats = shadow_features([snapshots_from_counts(list(zip(settings, counts)), 3)], 3, ising_paulis(3),
                            purities=False)
    from qiskit.quantum_info import Statevector
    exact = exact_features([Statevector(qc).data], 3, ising_paulis(3), purities=False)
    # single-qubit terms that are +-1 exactly: <Z_0> = -1, <Z_1> = +1, <X_2> = +1
    paulis = ising_paulis(3)
    for term in (((0, Z),), ((1, Z),), ((2, X),)):
        k = paulis.index(term)
        assert feats[0, k] == pytest.approx(exact[0, k])


@pytest.mark.parametrize("build", [one_copy_model, two_copy_model])
def test_mixed_state_batches_match_density_matrix(build):
    """Purified inputs give P = Tr(M rho^(x)copies), and rank-1 purifications match pure inputs."""
    rng = np.random.default_rng(4)
    n, r = 2, 3
    A = rng.normal(size=(5, 2 ** n, r)) + 1j * rng.normal(size=(5, 2 ** n, r))
    A /= np.linalg.norm(A, axis=(1, 2), keepdims=True)
    model = build(n)
    clf = QuantumClassifier(model)
    occ = rng.uniform(-3, 3, len(model.occ))
    p = clf._p(occ, clf._inputs(A))
    M = clf._observable(occ)
    for k in range(5):
        rho = A[k] @ A[k].conj().T
        full = rho if model.copies == 1 else np.kron(rho, rho)
        assert p[k] == pytest.approx(np.real(np.trace(M @ full)))
    psi = A[:, :, 0] / np.linalg.norm(A[:, :, 0], axis=1, keepdims=True)
    assert np.allclose(clf._p(occ, clf._inputs(psi)), clf._p(occ, clf._inputs(psi[:, :, None])))


def test_fast_shadow_purity_equals_pairwise_sum_and_is_unbiased():
    from entangle.measure import (random_pauli_snapshots, shadow_global_purity,
                                  shadow_global_purity_pairs)
    from entangle.states import make_purity_data
    rng = np.random.default_rng(7)
    d = make_purity_data(3, 30, seed=3)
    snaps = random_pauli_snapshots(d.states, 3, 300, rng)
    fast = np.array([shadow_global_purity(s, b) for s, b in snaps])
    slow = np.array([shadow_global_purity_pairs(s, b) for s, b in snaps])
    assert np.allclose(fast, slow)
    assert abs(np.mean(fast - d.purity)) < 0.12  # ~3 standard errors for 30 states x 300 shots


def test_two_copy_parity_readout_measures_global_purity():
    """With every pair block set to a Bell-basis measurement, P(class 1) = (1 - Tr rho^2) / 2."""
    from entangle.states import make_purity_data
    d = make_purity_data(2, 10, seed=1)
    model = two_copy_model(2, "parity", shared=True)
    clf = QuantumClassifier(model)
    # Bell measurement inside the block: RY(pi/2)-free first layer, CX, then H-like RY(-pi/2) on the control
    w = np.zeros(8)
    w[4] = -np.pi / 2
    p = clf.raw_proba(d.states, w)
    assert np.allclose(p, (1 - d.purity) / 2, atol=1e-9)


@pytest.mark.parametrize("g", [0.1, 0.33, 1.2])
def test_gauge_duality_gives_link_eigenstates_obeying_gauss_law(g):
    from entangle.gauge import dual_spectrum, dual_to_links, gauss_operators, link_hamiltonian
    w, v = dual_spectrum(2, g, k=2)
    H = link_hamiltonian(2, g)
    idx = np.arange(2 ** 8)
    for k in range(2):  # ground and first excited state of the physical sector
        psi = dual_to_links(v[:, k], 2)
        assert np.linalg.norm(H @ psi - w[k] * psi) < 1e-9
        for mask in gauss_operators(2):  # G_v flips the link bits at vertex v; G_v psi = psi
            assert np.allclose(psi[idx ^ mask], psi)


def test_wilson_loops_limits():
    """g = 0: every plaquette is +1, so all Wilson loops are 1. Large g: area-law decay."""
    from entangle.gauge import dual_spectrum, wilson_loop
    _, v = dual_spectrum(3, 1e-6, k=1)
    assert wilson_loop(v[:, 0], 3, 2, 2) == pytest.approx(1.0, abs=1e-6)
    _, v = dual_spectrum(3, 2.0, k=1)
    w11, w22 = wilson_loop(v[:, 0], 3, 1, 1), wilson_loop(v[:, 0], 3, 2, 2)
    assert 0 < w22 < w11 ** 3  # decays faster than the perimeter would allow: area law


def test_stage7_circuits_match_simulator():
    """Hardware purity circuits (3-qubit purifications, 2 copies, system qubits 0,1,3,4)
    give the same P(class 1) as the batched simulator on the mixed state."""
    from qiskit.quantum_info import Statevector

    from entangle.circuits import two_copy_model
    from entangle.qnn import QuantumClassifier
    from entangle.states import make_purity_data
    from stage7_hardware_purity import qnn_circuit, shadow_circuit

    data = make_purity_data(2, 4, seed=3)
    model = two_copy_model(2, "parity", shared=True)
    clf = QuantumClassifier(model)
    clf.weights = np.random.default_rng(0).uniform(-np.pi, np.pi, model.num_weights)
    p_sim = clf.raw_proba(data.states)
    for i, A in enumerate(data.states):
        qc = qnn_circuit(model, clf.weights, A).remove_final_measurements(inplace=False)
        probs = Statevector(qc).probabilities([0, 1, 3, 4])   # index bit k = measured qubit k
        assert probs[model.readout].sum() == pytest.approx(p_sim[i], abs=1e-9)
        # shadow circuit in the Z basis reproduces the system marginal of rho = A A^dagger
        sc = shadow_circuit(A, [2, 2]).remove_final_measurements(inplace=False)
        rho = A @ A.conj().T
        assert np.allclose(Statevector(sc).probabilities([0, 1]), np.real(np.diag(rho)), atol=1e-9)


def test_budget_guard_refuses_over_cap():
    from entangle.ibm import Budget
    b = Budget(floor=None, cap=10)
    b.spent = 8
    with pytest.raises(SystemExit):
        b.check(5)
    b.check(1)
