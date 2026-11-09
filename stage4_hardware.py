"""Stage 4: prepare test states on an IBM Quantum device and classify them there.

Models are trained on the exact simulator (as in stages 1-2). Then, for each test
state, the device runs
  - the quantum model: state-preparation circuit (once per copy) + trained model
    circuit, measured `--copies / copies-per-shot` times
  - the classical baseline's measurements: the same prep circuit measured in fixed
    Pauli bases (all 3^n X/Y/Z settings for the entanglement task, all-Z and all-X
    for the Ising task), `--copies` shots in total split evenly over the settings.
The classical models are trained on simulated snapshots with that same design.
Every result is shown next to the ideal (noise-free, same shots) simulation.

Device noise shifts every model's outputs (for example it makes product states
look mixed, as if entangled), so a model trained only in simulation can fail on
the device. `--calibrate K` also prepares K labelled states on the device and
recalibrates: the quantum models refit only their 2-parameter readout
calibration; the classical models are refit on the device snapshots of those
K states. Both columns (simulator-trained and device-recalibrated) are reported.

Tasks:
  entanglement  stage 1, 2 and 3 qubits (the 2-copy QNN needs 4 and 6 qubits)
  phases        stage 2, a 4-qubit Ising chain (prepared with StatePreparation)

Usage:
    python stage4_hardware.py --fake                     # FakeTorino noise model, no account (~1 min)
    python stage4_hardware.py --fake --backend-fake FakeManilaV2
    python stage4_hardware.py                            # real device: least busy (asks first)
    python stage4_hardware.py --backend ibm_torino --states 10 --copies 512

Credentials for real devices, checked in order (never commit them):
    1. IBM_QUANTUM_TOKEN environment variable (plus optional IBM_QUANTUM_INSTANCE, a CRN)
    2. an account saved with QiskitRuntimeService.save_account(...)
"""
import argparse
import os
import time
import warnings
from itertools import product

import numpy as np
from qiskit import QuantumCircuit

from entangle import report
from entangle.circuits import one_copy_model, qcnn_model, two_copy_model
from sklearn.linear_model import LogisticRegression

from entangle.classical import classical_models, fit_predict
from entangle.measure import (X, Z, all_paulis, fixed_settings_snapshots, ising_paulis,
                              measurement_circuit, shadow_features, snapshots_from_counts)
from entangle.qnn import QuantumClassifier
from entangle.states import make_entanglement_data, make_phase_data

warnings.filterwarnings("ignore", category=DeprecationWarning)


# --------------------------------------------------------------------------- #
# Backend and credentials
# --------------------------------------------------------------------------- #

def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    try:
        return QiskitRuntimeService()  # an account saved earlier
    except Exception as e:
        raise SystemExit("No IBM Quantum credentials. Set IBM_QUANTUM_TOKEN or save an account with "
                         "QiskitRuntimeService.save_account(...); or use --fake.\n"
                         f"Details: {e}")


def get_backend(args, min_qubits):
    if args.fake:
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, args.backend_fake)()
        print(f"Using local fake backend {backend.name} (simulated IBM noise, no account needed)")
        return backend
    service = get_service()
    backend = (service.backend(args.backend) if args.backend else
               service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits))
    print("Selected IBM Quantum backend:", backend.name)
    return backend


def run_circuits(backend, circuits, shots, seed=None):
    """Transpile for the backend and run with SamplerV2; returns counts per circuit."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed)
    isa = pm.run(circuits)
    sampler = SamplerV2(mode=backend)
    if seed is not None and hasattr(backend, "refresh"):  # fake backends: reproducible noise sampling
        sampler.options.simulator.seed_simulator = seed
    job = sampler.run(isa, shots=shots)
    print(f"    job {job.job_id()}: {len(isa)} circuits x {shots} shots", flush=True)
    return [r.data.meas.get_counts() for r in job.result()]


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

def task_specs(args):
    specs = []
    if args.task in ("entanglement", "both"):
        for n in args.qubits_entanglement:
            specs.append(dict(
                name=f"entanglement, {n} qubits", n=n,
                data=lambda N, seed, n=n: make_entanglement_data(n, N, seed=seed),
                models=[one_copy_model(n), two_copy_model(n)],
                settings=[list(s) for s in product(range(3), repeat=n)],
                paulis=all_paulis(n), purities=True, purity_only=True))
    if args.task in ("phases", "both"):
        n = args.qubits_phases
        specs.append(dict(
            name=f"Ising phases, {n} qubits", n=n,
            data=lambda N, seed, n=n: make_phase_data(n, N, seed=seed),
            models=[qcnn_model(n)],
            settings=[[Z] * n, [X] * n],
            paulis=ising_paulis(n), purities=False, purity_only=False))
    return specs


def model_circuit(model, prep, n):
    qc = QuantumCircuit(model.num_qubits)
    for c in range(model.copies):
        qc.compose(prep, qubits=range(c * n, (c + 1) * n), inplace=True)
    qc.compose(model.bound(model._weights), inplace=True)
    qc.measure_all()
    return qc


def p_from_counts(counts, readout):
    total = sum(counts.values())
    return sum(c for b, c in counts.items() if readout[int(b, 2)]) / total


def run_task(spec, backend, args, rng):
    n = spec["n"]
    train = spec["data"](args.train, args.seed)
    test = spec["data"](args.states, args.seed + 1)
    cal = spec["data"](args.calibrate, args.seed + 2) if args.calibrate else None
    k_cal = args.calibrate
    both = test.subset(np.arange(len(test.y)))  # test states first, then calibration states, one job
    if k_cal:
        for key, val in vars(test).items():
            if isinstance(val, np.ndarray):
                setattr(both, key, np.concatenate([val, getattr(cal, key)]))
    nt = len(test.y)
    out = {"task": spec["name"], "qubits": n, "test_states": args.states, "copies": args.copies,
           "models": {}}

    # quantum models
    for model in spec["models"]:
        clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                           restarts=args.restarts, seed=args.seed)
        model._weights = clf.weights
        shots = args.copies // model.copies
        circuits = [model_circuit(model, both.circuit(i), n) for i in range(len(both.y))]
        counts = run_circuits(backend, circuits, shots, args.seed)
        p_all = np.array([p_from_counts(c, model.readout) for c in counts])
        p_dev = p_all[:nt]
        p_ideal = rng.binomial(shots, np.clip(clf.raw_proba(test.states), 0, 1)) / shots
        res = {"kind": "quantum", "qubits_used": model.num_qubits, "shots": shots,
               "ideal": float(np.mean(clf.predict_from_p(p_ideal) == test.y)),
               "device": float(np.mean(clf.predict_from_p(p_dev) == test.y)),
               "p_device": p_dev.tolist(), "p_ideal_exact": clf.raw_proba(test.states).tolist()}
        if k_cal:  # refit only the readout calibration sigmoid(a (p - b)) on device outputs
            lr = LogisticRegression(C=1e4).fit(p_all[nt:, None], cal.y)
            res["device_recalibrated"] = float(np.mean(lr.predict(p_dev[:, None]) == test.y))
        out["models"][model.name] = res
        print(f"  {model.name:<34} ideal {res['ideal']:.3f}   device {res['device']:.3f}"
              + (f"   recalibrated {res['device_recalibrated']:.3f}" if k_cal else ""), flush=True)

    # classical baseline on the device's own measurement snapshots
    settings = spec["settings"]
    per = max(1, args.copies // len(settings))
    feats = lambda snaps: shadow_features(snaps, n, spec["paulis"], spec["purities"])  # noqa: E731
    Xtr = feats(fixed_settings_snapshots(train.states, n, settings, per, rng))
    Xte_ideal = feats(fixed_settings_snapshots(test.states, n, settings, per, rng))
    circuits = [measurement_circuit(both.circuit(i), s) for i in range(len(both.y)) for s in settings]
    counts = run_circuits(backend, circuits, per, args.seed)
    snaps = [snapshots_from_counts(list(zip(settings, counts[i * len(settings):(i + 1) * len(settings)])), n)
             for i in range(len(both.y))]
    X_dev = feats(snaps)
    if spec["purity_only"]:  # the strongest stage-1 classical features: single-qubit purities
        Xtr, Xte_ideal, X_dev = Xtr[:, -n:], Xte_ideal[:, -n:], X_dev[:, -n:]
    Xte_dev, Xcal_dev = X_dev[:nt], X_dev[nt:]
    for name, m in classical_models(args.seed).items():
        pred_ideal = fit_predict(m, Xtr, train.y, Xte_ideal)
        pred_dev = m.predict(Xte_dev)
        res = {"kind": "classical", "shots": per * len(settings), "settings": len(settings),
               "ideal": float(np.mean(pred_ideal == test.y)), "device": float(np.mean(pred_dev == test.y))}
        if k_cal:  # refit on the device snapshots of the calibration states
            m2 = classical_models(args.seed)[name]
            res["device_recalibrated"] = float(np.mean(fit_predict(m2, Xcal_dev, cal.y, Xte_dev) == test.y))
        out["models"][f"{name} (snapshots)"] = res
        print(f"  {name + ' (snapshots)':<34} ideal {res['ideal']:.3f}   device {res['device']:.3f}"
              + (f"   recalibrated {res['device_recalibrated']:.3f}" if k_cal else ""), flush=True)
    return out


def write_outputs(results, backend_name, args):
    tag = "fake" if args.fake else "ibm"
    report.save_json(f"stage4_hardware_{tag}.json", {"settings": vars(args), "backend": backend_name,
                                                      "results": results})
    rows = []
    for r in results:
        for name, m in r["models"].items():
            rows.append([r["task"], name, m["shots"], report.fmt(m["ideal"]), report.fmt(m["device"]),
                         report.fmt(m.get("device_recalibrated"))])
    md = (f"# Stage 4: classifying states prepared on {backend_name}\n\n"
          f"{'Local noisy simulation of the device (fake backend).' if args.fake else 'Real IBM Quantum hardware.'} "
          f"{args.states} test states per task (balanced), {args.copies} copies of each state per model. "
          f"Ideal = noise-free simulation with the same number of shots. Recalibrated = after refitting "
          f"on {args.calibrate} labelled states prepared on the device (quantum models: only the "
          f"2-parameter readout calibration; classical models: refit on device snapshots).\n\n"
          + report.md_table(["Task", "Model", "Shots per state", "Ideal", backend_name,
                             f"{backend_name}, recalibrated"], rows) + "\n")
    report.save_text(f"stage4_hardware_{tag}.md", md)

    fig, axes = report.axes_grid(len(results), f"Stage 4: ideal vs {backend_name} "
                                 "(bars, light to dark: ideal, device, device recalibrated)", width=4.4)
    for ax, r in zip(axes, results):
        names = [k for k in r["models"] if r["models"][k]["kind"] == "quantum"]
        cls = max((k for k in r["models"] if r["models"][k]["kind"] == "classical"),
                  key=lambda k: r["models"][k].get("device_recalibrated", r["models"][k]["device"]))
        names.append(cls)
        x = np.arange(len(names))
        colors = [report.STYLE["classical_snap" if r["models"][k]["kind"] == "classical" else
                               ("quantum" if k in ("QNN 2-copy", "QCNN") else "quantum_alt")]["color"]
                  for k in names]
        keys = [("ideal", 0.3, "ideal"), ("device", 0.65, "device"),
                ("device_recalibrated", 1.0, "device, recalibrated")]
        for j, (key, alpha, label) in enumerate(keys):
            ax.bar(x + (j - 1) * 0.27, [r["models"][k].get(key, 0) for k in names], width=0.25,
                   color=colors, alpha=alpha, label=label, edgecolor=report.SURFACE, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(" (snapshots)", "\n(snapshots)") for k in names], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color=report.INK2, linewidth=1, linestyle=":")
        ax.set_title(r["task"], fontsize=10, loc="left")
    axes[0].set_ylabel("test accuracy")
    report.finish(fig, axes, f"stage4_hardware_{tag}.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="local noisy simulation of an IBM device; no account")
    p.add_argument("--backend-fake", default="FakeTorino", help="fake backend class (with --fake)")
    p.add_argument("--backend", help="real IBM backend name (default: least busy)")
    p.add_argument("--task", choices=["entanglement", "phases", "both"], default="both")
    p.add_argument("--qubits-entanglement", type=int, nargs="+", default=[2, 3])
    p.add_argument("--qubits-phases", type=int, default=4)
    p.add_argument("--states", type=int, default=20, help="test states per task")
    p.add_argument("--copies", type=int, default=1024, help="copies of each test state per model")
    p.add_argument("--calibrate", type=int, default=20,
                   help="labelled states prepared on the device to recalibrate readouts (0 = off)")
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yes", action="store_true", help="skip the confirmation before using real hardware")
    args = p.parse_args()

    specs = task_specs(args)
    min_qubits = max(m.num_qubits for s in specs for m in s["models"])
    if not args.fake and not args.yes:
        n_circ = sum((args.states + args.calibrate) * (len(s["models"]) + len(s["settings"])) for s in specs)
        ans = input(f"This sends about {n_circ} circuits (~{args.copies} shots per state per model) to IBM "
                    f"Quantum and uses your account's quantum time. Continue? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("Cancelled.")
    backend = get_backend(args, min_qubits)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    results = []
    for spec in specs:
        print(f"\n{spec['name']}")
        results.append(run_task(spec, backend, args, rng))
    write_outputs(results, backend.name, args)
    tag = "fake" if args.fake else "ibm"
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage4_hardware_{tag}.{{json,md,png}}")


if __name__ == "__main__":
    main()
