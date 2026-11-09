"""Stage 2: phases of matter. Ordered or disordered ground state of the transverse-field Ising chain?

H = -sum Z_i Z_{i+1} - h sum X_i on an open chain of 4-8 qubits. Ground states
with h < 1 are ordered (ferromagnetic), h > 1 disordered (paramagnetic). States
come from exact diagonalisation; a quantum device would load them with a
state-preparation circuit (stage 4 does that).

Models:
  QCNN                 quantum convolutional neural network (weights shared per layer)
  QNN (generic ansatz) hardware-efficient ansatz, same readout (ablation)
  Classical            logistic regression / RBF SVM / MLP on
                       (a) the full state vector
                       (b) measurement snapshots: half the shots in the Z basis, half in X,
                           turned into estimates of <Z_i>, <X_i>, <Z_i Z_j>, <X_i X_j>

Every model gets the same number of copies of each test state (`--copies`).
Accuracy is also reported away from the critical point (|h - 1| >= 0.2), where
finite chains are clearly in one phase.

Usage:
    python stage2_phases.py                    # 4, 6, 8 qubits (about 3 minutes)
    python stage2_phases.py --qubits 4 6 --copies 64
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import hea_model, qcnn_model
from entangle.classical import classical_models, fit_predict, statevector_features
from entangle.measure import exact_features, ising_paulis, shadow_features, xz_snapshots
from entangle.qnn import QuantumClassifier
from entangle.states import make_phase_data

FAR = 0.2  # |h - 1| beyond which a state counts as "away from the critical point"


def scores(pred, data):
    ok = pred == data.y
    far = np.abs(data.h - 1) >= FAR
    return {"all": float(ok.mean()), "away_from_critical": float(ok[far].mean())}


def run_qubits(n, args, log=print):
    rng = np.random.default_rng(args.seed + 7 * n)
    train = make_phase_data(n, args.train, seed=args.seed + 100 * n)
    test = make_phase_data(n, args.test, seed=args.seed + 100 * n + 1)
    res = {"qubits": n, "quantum": {}, "classical": {}}
    curves = {}

    for build in (qcnn_model, hea_model):
        model = build(n)
        clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                           restarts=args.restarts, seed=args.seed)
        ex = scores(clf.predict(test.states), test)
        sh = scores(clf.predict(test.states, shots=args.copies, rng=rng), test)
        res["quantum"][model.name] = {"weights": model.num_weights, "exact": ex, "shots": sh,
                                      "train_seconds": round(clf.train_seconds, 1),
                                      "description": model.description}
        curves[model.name] = clf.calibrate(clf.raw_proba(test.states))
        log(f"  {model.name:<21} exact {ex['all']:.3f} (away from h=1: {ex['away_from_critical']:.3f})   "
            f"{args.copies} shots {sh['all']:.3f}   ({model.num_weights} weights, {clf.train_seconds:.0f}s)")

    def classical(name, Xtr, Xte):
        out = {m: scores(fit_predict(model, Xtr, train.y, Xte), test)
               for m, model in classical_models(args.seed).items()}
        log(f"  classical, {name}: " + ", ".join(f"{k} {v['all']:.3f}" for k, v in out.items()))
        return out

    res["classical"]["full state vector"] = {
        "features": 2 ** (n + 1),
        "exact": classical("full state vector", statevector_features(train.states),
                           statevector_features(test.states))}
    paulis = ising_paulis(n)
    snap_tr = shadow_features(xz_snapshots(train.states, n, args.copies, rng), n, paulis, purities=False)
    snap_te = shadow_features(xz_snapshots(test.states, n, args.copies, rng), n, paulis, purities=False)
    res["classical"]["Z/X snapshots"] = {
        "features": len(paulis), "snapshots": args.copies,
        "exact": classical("Z/X expectation values (exact)",
                           exact_features(train.states, n, paulis, purities=False),
                           exact_features(test.states, n, paulis, purities=False)),
        "shots": classical(f"{args.copies} Z/X snapshots", snap_tr, snap_te)}
    return res, (test.h, curves)


def best(d, key, sub="all"):
    return max(v[sub] for v in d[key].values())


def write_outputs(results, curve, args):
    report.save_json("stage2_phases.json", {"settings": vars(args), "results": results})
    rows = []
    for r in results:
        n = r["qubits"]
        for name, q in r["quantum"].items():
            rows.append([n, name, f"{q['weights']} weights", report.fmt(q["exact"]["all"]),
                         report.fmt(q["exact"]["away_from_critical"]), report.fmt(q["shots"]["all"])])
        sv = r["classical"]["full state vector"]
        for name, a in sv["exact"].items():
            rows.append([n, f"{name}, full state vector", f"{sv['features']} numbers",
                         report.fmt(a["all"]), report.fmt(a["away_from_critical"]), "n/a"])
        sn = r["classical"]["Z/X snapshots"]
        for name in sn["exact"]:
            rows.append([n, f"{name}, Z/X snapshots", f"{sn['features']} features",
                         report.fmt(sn["exact"][name]["all"]),
                         report.fmt(sn["exact"][name]["away_from_critical"]),
                         report.fmt(sn["shots"][name]["all"])])
    table = report.md_table(["Qubits", "Model", "Size", "Test acc. (exact)",
                             f"Exact, |h-1| >= {FAR}", f"Test acc. ({args.copies} copies)"], rows)
    md = (f"# Stage 2: Ising chain phases\n\nOrdered (h < 1) vs disordered (h > 1) ground states, "
          f"h uniform in [0.1, 2]. {args.train} training and {args.test} test states per chain length. "
          f"Exact = infinite shots.\n\n{table}\n")
    report.save_text("stage2_phases.md", md)

    qs = [r["qubits"] for r in results]
    fig, axes = report.axes_grid(3, "Stage 2: ordered or disordered Ising ground state?", width=4.6)
    for ax, key, title in ((axes[0], "exact", "Exact (infinite shots)"),
                           (axes[1], "shots", f"{args.copies} copies of each test state")):
        if key == "exact":
            report.line(ax, qs, [best(r["classical"]["full state vector"], "exact") for r in results],
                        "Classical, full state vector (best of 3)", "classical_state")
        report.line(ax, qs, [best(r["classical"]["Z/X snapshots"], key) for r in results],
                    "Classical, Z/X snapshots (best of 3)", "classical_snap")
        report.line(ax, qs, [r["quantum"]["QNN (generic ansatz)"][key]["all"] for r in results],
                    "QNN, generic ansatz", "quantum_alt")
        report.line(ax, qs, [r["quantum"]["QCNN"][key]["all"] for r in results], "QCNN", "quantum",
                    dashed=True)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("qubits")
        ax.set_xticks(qs)
        ax.set_ylim(0.8, 1.01)
    axes[0].set_ylabel("test accuracy")
    h, cur = curve
    order = np.argsort(h)
    ax = axes[2]
    st = report.STYLE["quantum"]
    ax.plot(h[order], cur["QCNN"][order], color=st["color"], linewidth=2)
    ax.axvline(1.0, color=report.INK2, linewidth=1, linestyle=":")
    ax.set_title(f"QCNN output, {results[-1]['qubits']} qubits", fontsize=10, loc="left")
    ax.set_xlabel("transverse field h")
    ax.set_ylabel("P(ordered)")
    report.finish(fig, axes, "stage2_phases.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--copies", type=int, default=256, help="copies of each test state a model may use")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    results, curve = [], None
    for n in args.qubits:
        print(f"\n{n} qubits")
        r, curve = run_qubits(n, args)
        results.append(r)
    write_outputs(results, curve, args)
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage2_phases.{{json,md,png}}")


if __name__ == "__main__":
    main()
