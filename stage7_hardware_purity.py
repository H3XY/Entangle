"""Stage 7: the stage 5 purity test (pure vs mixed, 2 qubits) on an IBM Quantum device.

The headline of stage 5 is that a 2-copy measurement tells pure from mixed states
with far fewer copies than any single-copy strategy. This stage runs the 2-qubit
version on hardware, with the classical-shadow baseline measured on the same device.

States. A 2-qubit mixed state of rank 2 cannot be prepared on 2 qubits alone, so
every state is prepared as a purification on 3 qubits (2 system + 1 ancilla) and the
ancilla is never measured: |Psi> = sum_k A[:, k] (x) |k>_ancilla, rho = A A^dagger.
Pure states use the same 3-qubit preparation with the ancilla left in |0>; the
transpiled preparations have the same two-qubit gate count for both classes (we
record it), so circuit depth does not give the label away.

Per state the device runs
  - QNN 2-copy: two 3-qubit preparations (6 qubits) + the trained 2-copy parity
    model (shared block, trained exactly in simulation) on the 4 system qubits;
    class 1 if an odd number of pairs read 11. `--copies / 2` shots.
  - Classical shadows: the 3-qubit preparation measured in all 9 X/Y/Z bases on the
    2 system qubits, `--copies / 9` shots each, turned into the classical-shadow
    estimate of the global purity and the single-qubit purities.
Every result is shown next to the ideal (noise-free, same shots) simulation. As in
stage 4, `--calibrate K` labelled states are also run on the device and used to
recalibrate (QNN: 2-parameter readout calibration; classical: refit on device data).
From the device counts we also subsample fewer shots to get accuracy vs copies.

Repeat runs reuse the same test states and the same physical qubits (`--layout`,
printed by the first run), so they differ only in device noise and drift.

Usage:
    python stage7_hardware_purity.py --fake                        # FakeMarrakesh noise model (local)
    python stage7_hardware_purity.py --backend ibm_marrakesh        # real device (asks first)
    python stage7_hardware_purity.py --backend ibm_marrakesh --layout 1 2 3 4 5 6 --yes

Budget guard on real devices: `--floor` (default 125 s) is the least quantum time
the account must keep after each job and `--cap` the most this run may spend; see
entangle/ibm.py. Job IDs and billed seconds go to results/ibm_jobs.json.
"""
import argparse
import json
import os
import time
import warnings
from itertools import product

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from sklearn.linear_model import LogisticRegression

from entangle import ibm, report
from entangle.circuits import two_copy_model
from entangle.classical import classical_models, fit_predict
from entangle.measure import X, Y, fixed_settings_snapshots, purity_features, snapshots_from_counts
from entangle.qnn import QuantumClassifier
from entangle.states import make_purity_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
N = 2
SETTINGS = [list(s) for s in product(range(3), repeat=N)]


def prep_circuit(A):
    """3-qubit purification: system qubits 0, 1; ancilla qubit 2."""
    psi = np.concatenate([A[:, 0], A[:, 1]])
    qc = QuantumCircuit(N + 1, name="prep")
    qc.append(StatePreparation(psi / np.linalg.norm(psi)), range(N + 1))
    return qc


def qnn_circuit(model, weights, A):
    q = QuantumRegister(2 * (N + 1), "q")
    c = ClassicalRegister(2 * N, "meas")
    qc = QuantumCircuit(q, c)
    prep = prep_circuit(A)
    qc.compose(prep, qubits=[0, 1, 2], inplace=True)
    qc.compose(prep, qubits=[3, 4, 5], inplace=True)
    system = [0, 1, 3, 4]                  # model qubit k -> circuit qubit system[k]
    qc.compose(model.bound(weights), qubits=system, inplace=True)
    qc.measure(system, list(range(2 * N)))
    return qc


def shadow_circuit(A, setting):
    q = QuantumRegister(N + 1, "q")
    c = ClassicalRegister(N, "meas")
    qc = QuantumCircuit(q, c)
    qc.compose(prep_circuit(A), qubits=[0, 1, 2], inplace=True)
    for k, b in enumerate(setting):
        if b == X:
            qc.h(k)
        elif b == Y:
            qc.sdg(k)
            qc.h(k)
    qc.measure(range(N), range(N))
    return qc


def p_from_counts(counts, readout):
    total = sum(counts.values())
    return sum(v for b, v in counts.items() if readout[int(b, 2)]) / total


def subsample(counts, k, rng):
    """k shots drawn without replacement from a counts dict."""
    keys = list(counts)
    draw = rng.multivariate_hypergeometric([counts[b] for b in keys], k)
    return {b: int(d) for b, d in zip(keys, draw) if d}


def wilson(acc, n, z=1.96):
    centre = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def run(args, backend, budget):
    rng = np.random.default_rng(args.seed + 17)
    train = make_purity_data(N, args.train, seed=args.seed)
    test = make_purity_data(N, args.test, seed=args.seed + 1)
    cal = make_purity_data(N, args.calibrate, seed=args.seed + 2)
    A_all = np.concatenate([test.states, cal.states])
    y_all = np.concatenate([test.y, cal.y])
    nt = len(test.y)

    model = two_copy_model(N, "parity", shared=True)
    clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter, restarts=args.restarts,
                                       seed=args.seed)
    p_exact = clf.raw_proba(A_all)
    shots_q = args.copies // 2
    per = args.copies // len(SETTINGS)

    # circuits, mapped to fixed physical qubits
    qnn = [qnn_circuit(model, clf.weights, A) for A in A_all]
    sh = [shadow_circuit(A, s) for A in A_all for s in SETTINGS]
    layout = args.layout
    if layout is None:
        probe = ibm.transpile(backend, qnn[:1], args.seed)[0]
        layout = list(probe.layout.initial_index_layout(filter_ancillas=True))
        print(f"  chosen physical qubits (copy 1: system, system, ancilla; copy 2: same): {layout}")
    isa = (ibm.transpile(backend, qnn, args.seed, initial_layout=layout)
           + ibm.transpile(backend, sh, args.seed, initial_layout=layout[:N + 1]))
    shots = [shots_q] * len(qnn) + [per] * len(sh)
    print(f"  {len(isa)} circuits: {len(qnn)} QNN x {shots_q} shots, {len(sh)} shadow x {per} shots")
    counts, info = ibm.run_pubs(backend, isa, shots, seed=args.seed, simulator=args.fake, budget=budget,
                                label=f"stage7 {args.run_label}", transpiled=True)
    cq, cs = counts[:len(qnn)], counts[len(qnn):]
    twoq = info["two_qubit_gates"]
    prep_twoq = {"pure": float(np.mean([twoq[len(qnn) + i * 9] for i in range(len(A_all)) if y_all[i] == 0])),
                 "mixed": float(np.mean([twoq[len(qnn) + i * 9] for i in range(len(A_all)) if y_all[i] == 1]))}

    out = {"run_label": args.run_label, "backend": backend.name, "fake": args.fake, "layout": layout,
           "job": {k: v for k, v in info.items() if k not in ("layouts", "two_qubit_gates")},
           "test_states": nt, "calibration_states": args.calibrate, "copies": args.copies,
           "two_qubit_gates_in_shadow_circuit": prep_twoq,
           "two_qubit_gates_in_qnn_circuit": {"mean": float(np.mean(twoq[:len(qnn)])),
                                              "min": int(min(twoq[:len(qnn)])), "max": int(max(twoq[:len(qnn)]))},
           "calibration_snapshot": ibm.calibration_snapshot(backend, layout),
           "test_y": test.y.tolist(), "true_purity": test.purity.tolist(), "models": {}}

    # ---- QNN 2-copy
    p_dev_all = np.array([p_from_counts(c, model.readout) for c in cq])
    p_dev = p_dev_all[:nt]
    p_ideal = rng.binomial(shots_q, np.clip(p_exact[:nt], 0, 1)) / shots_q
    lr = LogisticRegression(C=1e4).fit(p_dev_all[nt:, None], cal.y)
    q = {"kind": "quantum", "shots": shots_q,
         "exact": float(np.mean(clf.predict_from_p(p_exact[:nt]) == test.y)),
         "ideal": float(np.mean(clf.predict_from_p(p_ideal) == test.y)),
         "device": float(np.mean(clf.predict_from_p(p_dev) == test.y)),
         "device_recalibrated": float(np.mean(lr.predict(p_dev[:, None]) == test.y)),
         "p_exact": p_exact[:nt].tolist(), "p_device": p_dev.tolist(),
         "mean_shift_p": float(np.mean(p_dev - p_exact[:nt])),
         "mean_shift_p_by_class": {str(k): float(np.mean((p_dev - p_exact[:nt])[test.y == k])) for k in (0, 1)},
         # with an ideal Bell-basis block, P = (1 - Tr rho^2) / 2, so 1 - 2P estimates the purity
         "purity_from_device": (1 - 2 * p_dev).tolist()}
    out["models"]["QNN 2-copy (parity, shared)"] = q

    # ---- classical shadows
    Xtr = purity_features(fixed_settings_snapshots(train.states, N, SETTINGS, per, rng), N)
    Xte_ideal = purity_features(fixed_settings_snapshots(test.states, N, SETTINGS, per, rng), N)
    snaps = [snapshots_from_counts(list(zip(SETTINGS, cs[i * 9:(i + 1) * 9])), N) for i in range(len(A_all))]
    X_dev = purity_features(snaps, N)
    out["shadow_purity_from_device"] = X_dev[:nt, 0].tolist()
    for name, m in classical_models(args.seed).items():
        pred_ideal = fit_predict(m, Xtr, train.y, Xte_ideal)
        pred_dev = m.predict(X_dev[:nt])
        m2 = classical_models(args.seed)[name]
        pred_recal = fit_predict(m2, X_dev[nt:], cal.y, X_dev[:nt])
        out["models"][f"{name} (classical shadows)"] = {
            "kind": "classical", "shots": per * len(SETTINGS),
            "ideal": float(np.mean(pred_ideal == test.y)), "device": float(np.mean(pred_dev == test.y)),
            "device_recalibrated": float(np.mean(pred_recal == test.y))}
    for k, v in out["models"].items():
        for key in ("ideal", "device", "device_recalibrated"):
            v[f"{key}_wilson95"] = wilson(v[key], nt)
        print(f"  {k:<44} ideal {v['ideal']:.3f}   device {v['device']:.3f}   "
              f"recalibrated {v['device_recalibrated']:.3f}", flush=True)

    # ---- accuracy vs copies, by subsampling the device counts (recalibrated readouts)
    sweep = {"copies": [], "QNN 2-copy": [], "Classical shadows (logistic regression)": []}
    for c in [c for c in (16, 32, 64, 128, 256, 512, 1024, 2048) if c <= args.copies]:
        accs_q, accs_c = [], []
        for _ in range(args.resamples):
            pq = np.array([p_from_counts(subsample(cnt, c // 2, rng), model.readout) for cnt in cq])
            lr = LogisticRegression(C=1e4).fit(pq[nt:, None], cal.y)
            accs_q.append(np.mean(lr.predict(pq[:nt, None]) == test.y))
            k = max(1, c // len(SETTINGS))
            if k * len(SETTINGS) >= 18:
                sn = [snapshots_from_counts(list(zip(SETTINGS, [subsample(x, k, rng) for x in cs[i * 9:(i + 1) * 9]])), N)
                      for i in range(len(A_all))]
                Xd = purity_features(sn, N)
                m = classical_models(args.seed)["Logistic regression"]
                accs_c.append(np.mean(fit_predict(m, Xd[nt:], cal.y, Xd[:nt]) == test.y))
        sweep["copies"].append(c)
        sweep["QNN 2-copy"].append([float(np.mean(accs_q)), float(np.std(accs_q))])
        sweep["Classical shadows (logistic regression)"].append(
            [float(np.mean(accs_c)), float(np.std(accs_c))] if accs_c else None)
    out["device_copies_sweep"] = sweep
    return out


def write_outputs(tag):
    path = report.path(f"stage7_hardware_purity_{tag}.json")
    runs = json.load(open(path))["runs"]
    name_q = "QNN 2-copy (parity, shared)"
    models = list(runs[0]["models"])
    rows = []
    for k in models:
        for key in ("ideal", "device", "device_recalibrated"):
            vals = [r["models"][k][key] for r in runs]
            per_run = ", ".join(f"{v:.3f} [{r['models'][k][key + '_wilson95'][0]:.2f}, "
                                f"{r['models'][k][key + '_wilson95'][1]:.2f}]" for v, r in zip(vals, runs))
            rows.append([k, key.replace("_", " "), f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.3f}",
                         per_run])
    r0 = runs[0]
    shift = [[r["run_label"], r["job"]["job_id"], r["job"]["submitted_utc"][:19],
              "–" if r["job"]["quantum_seconds"] is None else f"{r['job']['quantum_seconds']:.0f}",
              f"{r['models'][name_q]['mean_shift_p']:+.4f}",
              f"{r['models'][name_q]['mean_shift_p_by_class']['0']:+.4f} / {r['models'][name_q]['mean_shift_p_by_class']['1']:+.4f}",
              " ".join(map(str, r["layout"]))] for r in runs]
    sw_rows = []
    for r in runs:
        sw = r["device_copies_sweep"]
        for model in ("QNN 2-copy", "Classical shadows (logistic regression)"):
            sw_rows.append([r["run_label"], model] + ["–" if v is None else f"{v[0]:.3f}" for v in sw[model]])
    md = "\n".join([
        f"# Stage 7: pure or mixed, on {r0['backend']}" + (" (fake backend, simulated noise)" if r0["fake"] else ""), "",
        f"2-qubit states (3-qubit purifications), {r0['test_states']} test states (balanced), {r0['copies']} copies "
        f"of each state per model, {len(runs)} independent run(s) on the same test states and physical qubits. "
        f"Recalibrated = refit on {r0['calibration_states']} labelled states prepared on the device. "
        "Mean ± std over runs; per run with a Wilson 95% interval.", "",
        report.md_table(["Model", "Setting", "Mean ± std over runs", "Per run [95% CI]"], rows), "",
        "## Jobs and readout shift", "",
        "Mean shift = mean over test states of (device P(class 1) - exact P(class 1)) for the QNN; "
        "by class: pure / mixed.", "",
        report.md_table(["Run", "Job ID", "Submitted (UTC)", "Quantum s", "Mean shift", "By class", "Physical qubits"],
                        shift), "",
        f"Two-qubit gates per transpiled circuit: shadow (prep only) pure {r0['two_qubit_gates_in_shadow_circuit']['pure']:.1f}, "
        f"mixed {r0['two_qubit_gates_in_shadow_circuit']['mixed']:.1f}; QNN circuit mean "
        f"{r0['two_qubit_gates_in_qnn_circuit']['mean']:.1f}.", "",
        "## Accuracy vs copies, subsampled from the device counts (recalibrated)", "",
        report.md_table(["Run", "Model"] + [str(c) for c in r0["device_copies_sweep"]["copies"]], sw_rows), ""])
    report.save_text(f"stage7_hardware_purity_{tag}.md", md)

    fig, axes = report.axes_grid(2, f"Stage 7: pure or mixed on {r0['backend']}", width=5.2)
    ax = axes[0]
    cop = r0["device_copies_sweep"]["copies"]
    for model, fam, dashed in (("Classical shadows (logistic regression)", "classical_snap", False),
                               ("QNN 2-copy", "quantum", True)):
        m = [np.mean([r["device_copies_sweep"][model][i][0] for r in runs]) if r0["device_copies_sweep"][model][i]
             else np.nan for i in range(len(cop))]
        report.line(ax, cop, m, f"{model}, device", fam, dashed=dashed)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cop)
    ax.set_xticklabels([str(c) for c in cop], fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color=report.INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("copies of each test state")
    ax.set_ylabel("test accuracy (device, recalibrated)")
    ax.set_title("Accuracy vs copies on the device", fontsize=10, loc="left")
    ax = axes[1]
    true = np.array(r0["true_purity"])
    for r, alpha in zip(runs, (1.0, 0.6, 0.4, 0.3)):
        st = report.STYLE["quantum"]
        ax.scatter(true, r["models"][name_q]["purity_from_device"], s=18, color=st["color"], alpha=alpha,
                   edgecolor=report.SURFACE, linewidth=0.5, label=f"QNN, {r['run_label']}")
        st = report.STYLE["classical_snap"]
        ax.scatter(true, r["shadow_purity_from_device"], s=18, color=st["color"], alpha=alpha, marker="D",
                   edgecolor=report.SURFACE, linewidth=0.5, label=f"shadows, {r['run_label']}")
    ax.plot([0.45, 1.02], [0.45, 1.02], color=report.INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("true purity Tr rho^2")
    ax.set_ylabel("purity measured on the device")
    ax.set_title("Measured vs true purity", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, labelcolor=report.INK2, loc="upper left")
    report.finish(fig, axes, f"stage7_hardware_purity_{tag}.png", legend_ax=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--backend-fake", default="FakeMarrakesh")
    p.add_argument("--backend", default="ibm_marrakesh")
    p.add_argument("--test", type=int, default=60)
    p.add_argument("--calibrate", type=int, default=20)
    p.add_argument("--copies", type=int, default=1024)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--resamples", type=int, default=20, help="subsamples per copy count in the copies sweep")
    p.add_argument("--layout", type=int, nargs=6, help="physical qubits (from a previous run)")
    p.add_argument("--run-label", default="run 1")
    p.add_argument("--tag", help="output name suffix (default: fake backend name or 'ibm')")
    p.add_argument("--new", action="store_true", help="start a fresh results file instead of appending")
    p.add_argument("--floor", type=float, default=125.0, help="least quantum seconds the account must keep")
    p.add_argument("--cap", type=float, default=170.0, help="most quantum seconds this run may spend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()
    tag = args.tag or (args.backend_fake.lower() if args.fake else "ibm")
    est = ibm.estimate_seconds((args.test + args.calibrate) * (args.copies // 2 + (args.copies // 9) * 9))
    if not args.fake and not args.yes:
        ans = input(f"This sends {(args.test + args.calibrate) * 10} circuits to {args.backend} "
                    f"(~{est:.0f} s of quantum time). Continue? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("Cancelled.")
    backend = ibm.get_backend(args.fake, args.backend_fake, args.backend, min_qubits=6)
    budget = None if args.fake else ibm.Budget(floor=args.floor, cap=args.cap)
    t0 = time.time()
    res = run(args, backend, budget)
    path = report.path(f"stage7_hardware_purity_{tag}.json")
    data = json.load(open(path)) if os.path.exists(path) and not args.new else {"runs": []}
    data["runs"].append(dict(res, settings=vars(args)))
    report.save_json(f"stage7_hardware_purity_{tag}.json", data)
    write_outputs(tag)
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage7_hardware_purity_{tag}.{{json,md,png}}")


if __name__ == "__main__":
    main()
