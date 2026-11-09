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
