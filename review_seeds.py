"""Seed-to-seed uncertainty for stages 1, 2, 5 and 6, plus task-difficulty baselines.

The stage scripts each report one seed (stage 5: three). This script reruns the
headline experiments over several seeds (each seed redraws the training and test
states, the shot noise and the model initialisations) and reports per-seed values
and mean ± std. For stage 5 it reads results/stage5_purity.json (already 3 seeds)
and adds per-seed copies-needed numbers.

It also answers "how hard are the phase tasks?" with single-feature baselines: a
logistic regression on ONE exact physical quantity (stage 2: the mean
nearest-neighbour ZZ correlation or the mean X; stage 6: the mean electric field X
or the mean plaquette), and the accuracy of every model on test states near the
critical point only.

Classical "best of" numbers pick the best learner on the test set, which flatters
the classical side (the safe direction for a quantum-vs-classical comparison).

Usage:
    python review_seeds.py --stage 1          # writes results/review_seeds_stage1.{json,md}
    python review_seeds.py --stage 2
    python review_seeds.py --stage 6
    python review_seeds.py --stage 5          # no training, reads stage5_purity.json
"""
import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression

from entangle import report
from entangle.measure import X, Z, exact_expectations


def summary(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "per_seed": [float(x) for x in v]}


def fmt(s):
    return f"{s['mean']:.3f} ± {s['std']:.3f}"


def per_seed_str(s, digits=3):
    return ", ".join(f"{x:.{digits}f}" for x in s["per_seed"])


def one_feature_accuracy(f_train, y_train, f_test, y_test):
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(f_train[:, None], y_train)
    return float(np.mean(lr.predict(f_test[:, None]) == y_test))


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #

def stage1(seeds):
    import stage1_entanglement as s1
    qubits = [2, 3, 4]
    metrics = {}
    for seed in seeds:
        args = SimpleNamespace(qubits=qubits, train=200, test=400, copies=256, q_min=0.2, maxiter=200,
                               restarts=3, seed=seed)
        for n in qubits:
            print(f"stage 1, seed {seed}, {n} qubits", flush=True)
            r = s1.run_qubits(n, args, log=lambda *a: None)
            row = {
                "QNN 2-copy, exact": r["quantum"]["QNN 2-copy"]["test_accuracy_exact"],
                "QNN 2-copy, 256 copies": r["quantum"]["QNN 2-copy"]["test_accuracy_shots"],
                "QNN 1-copy, exact": r["quantum"]["QNN 1-copy"]["test_accuracy_exact"],
                "Classical, full state vector (best of 3)":
                    max(r["classical"]["full state vector"]["test_accuracy_exact"].values()),
                "Classical shadows, exact estimates (best of 6)": s1.best_snapshot(r, "exact"),
                "Classical shadows, 256 snapshots (best of 6)": s1.best_snapshot(r, "shots"),
                "Classical shadows, all Pauli features, 256 snapshots (best of 3)":
                    max(r["classical"]["snapshots"]["test_accuracy_shots"].values()),
                "Classical shadows, purity features, 256 snapshots (best of 3)":
                    max(r["classical"]["snapshots, purities only"]["test_accuracy_shots"].values()),
            }
            for k, v in row.items():
                metrics.setdefault(str(n), {}).setdefault(k, []).append(v)
    return {n: {k: summary(v) for k, v in d.items()} for n, d in metrics.items()}


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #

def ising_scalars(states, n):
    zz = [((i, Z), (i + 1, Z)) for i in range(n - 1)]
    xs = [((i, X),) for i in range(n)]
    E = exact_expectations(states, n, zz + xs)
    return {"mean nearest-neighbour ZZ": E[:, :len(zz)].mean(axis=1), "mean X": E[:, len(zz):].mean(axis=1)}


def stage2(seeds, near=0.2):
    import stage2_phases as s2
    from entangle.states import make_phase_data
    qubits = [4, 6, 8]
    metrics = {}
    for seed in seeds:
        args = SimpleNamespace(qubits=qubits, train=200, test=400, copies=256, maxiter=200, restarts=3,
                               seed=seed)
        for n in qubits:
            print(f"stage 2, seed {seed}, {n} qubits", flush=True)
            r, _ = s2.run_qubits(n, args, log=lambda *a: None)
            train = make_phase_data(n, args.train, seed=seed + 100 * n)
            test = make_phase_data(n, args.test, seed=seed + 100 * n + 1)
            frac_far = float(np.mean(np.abs(test.h - 1) >= near))

            def near_acc(sc):  # accuracy on the states within `near` of h = 1
                return (sc["all"] - frac_far * sc["away_from_critical"]) / (1 - frac_far)
            row = {}
            for name in ("QCNN", "QNN (generic ansatz)"):
                q = r["quantum"][name]
                row[f"{name}, exact"] = q["exact"]["all"]
                row[f"{name}, exact, |h - 1| < {near}"] = near_acc(q["exact"])
                row[f"{name}, 256 copies"] = q["shots"]["all"]
            sv = r["classical"]["full state vector"]["exact"]
            zx = r["classical"]["Z/X snapshots"]
            row["Classical, full state vector (best of 3)"] = max(v["all"] for v in sv.values())
            row["Classical, Z/X expectations exact (best of 3)"] = max(v["all"] for v in zx["exact"].values())
            row[f"Classical, Z/X expectations exact, |h - 1| < {near} (best of 3)"] = \
                max(near_acc(v) for v in zx["exact"].values())
            row["Classical, 256 Z/X snapshots (best of 3)"] = max(v["all"] for v in zx["shots"].values())
            ftr, fte = ising_scalars(train.states, n), ising_scalars(test.states, n)
            for k in ftr:
                row[f"One feature: {k} (exact)"] = one_feature_accuracy(ftr[k], train.y, fte[k], test.y)
            row[f"Fraction of test states with |h - 1| < {near}"] = 1 - frac_far
            for k, v in row.items():
                metrics.setdefault(str(n), {}).setdefault(k, []).append(v)
    return {n: {k: summary(v) for k, v in d.items()} for n, d in metrics.items()}


# --------------------------------------------------------------------------- #
# Stage 6
# --------------------------------------------------------------------------- #

def stage6(seeds, near=0.1):
    import stage6_gauge as s6
    from entangle.gauge import G_CRITICAL, lattice, make_gauge_data
    metrics = {}
    plaq_links, _, _ = lattice(2)
    for seed in seeds:
        print(f"stage 6, seed {seed}", flush=True)
        args = SimpleNamespace(train=200, test=400, copies=256, maxiter=200, restarts=3, seed=seed)
        r = s6.classify(args, log=lambda *a: None)
        train = make_gauge_data(args.train, seed=seed)
        test = make_gauge_data(args.test, seed=seed + 1)
        frac_far = float(np.mean(np.abs(test.g - G_CRITICAL) >= near))

        def near_acc(sc):
            return (sc["all"] - frac_far * sc["away_from_critical"]) / (1 - frac_far)
        row = {}
        for name, q in r["quantum"].items():
            row[f"{name}, exact"] = q["exact"]["all"]
            row[f"{name}, exact, |g - g_c| < {near}"] = near_acc(q["exact"])
            row[f"{name}, 256 copies"] = q["shots"]["all"]
        row["Classical, full state vector (best of 3)"] = \
            max(v["all"] for v in r["classical"]["full state vector"].values())
        row["Classical, 256 Z/X snapshots (best of 3)"] = \
            max(v["all"] for v in r["classical"]["Z/X snapshots"].values())
        row[f"Classical, 256 Z/X snapshots, |g - g_c| < {near} (best of 3)"] = \
            max(near_acc(v) for v in r["classical"]["Z/X snapshots"].values())
        feats = {"mean electric field X": [((l, X),) for l in range(8)],
                 "mean plaquette ZZZZ": [tuple((l, Z) for l in links) for links in plaq_links]}
        for k, paulis in feats.items():
            ftr = exact_expectations(train.states, 8, paulis).mean(axis=1)
            fte = exact_expectations(test.states, 8, paulis).mean(axis=1)
            row[f"One feature: {k} (exact)"] = one_feature_accuracy(ftr, train.y, fte, test.y)
        row[f"Fraction of test states with |g - g_c| < {near}"] = 1 - frac_far
        for k, v in row.items():
            metrics.setdefault("8", {}).setdefault(k, []).append(v)
    return {n: {k: summary(v) for k, v in d.items()} for n, d in metrics.items()}


# --------------------------------------------------------------------------- #
# Stage 5 (from the saved 3-seed run)
# --------------------------------------------------------------------------- #

def stage5(_seeds, target=0.9):
    d = json.load(open(report.path("stage5_purity.json")))
    res, copies = d["results"], d["settings"]["copies"]
    out = {}
    for model in ("QNN 2-copy, shared, transferred from 2 qubits", "Classical shadows (purity estimates)"):
        for n, by_c in res["by_copies"][model].items():
            seeds = len(by_c[str(copies[0])]["runs"])
            need = []
            for s in range(seeds):
                hit = [c for c in copies if by_c[str(c)]["runs"][s] >= target]
                need.append(hit[0] if hit else float("nan"))
            grid = next((c for c in copies if by_c[str(c)]["mean"] >= target), None)
            out.setdefault(n, {})[f"{model}: copies for {target:.0%}"] = {
                "grid_definition": grid, "per_seed": need,
                "median": float(np.median(need)), "min": float(np.min(need)), "max": float(np.max(need))}
    for model, by_n in res["exact"].items():
        for n, s in by_n.items():
            if s:
                out.setdefault(n, {})[f"{model}: exact accuracy"] = summary(s["runs"])
    return out


def write(stage, data, seeds):
    report.save_json(f"review_seeds_stage{stage}.json", {"seeds": list(seeds), "results": data})
    rows = []
    for n, d in data.items():
        for k, s in d.items():
            if "grid_definition" in s:
                rows.append([n, k, f"{s['grid_definition']} (grid)",
                             f"{s['min']:.0f}–{s['max']:.0f}", ", ".join(f"{x:.0f}" for x in s["per_seed"])])
            else:
                rows.append([n, k, fmt(s), f"{s['min']:.3f}–{s['max']:.3f}", per_seed_str(s)])
    head = ["Qubits", "Quantity", "Mean ± std over seeds", "Range", "Per seed"]
    note = {"1": "Entanglement classification (stage 1).",
            "2": "Ising chain phases (stage 2), plus single-feature baselines and near-critical accuracy.",
            "5": "Pure vs mixed (stage 5): per-seed copies needed (smallest power of 2 with accuracy at "
                 "least 90% for that seed) next to the grid definition used in the stage 5 table "
                 "(smallest power of 2 where the mean over seeds reaches 90%).",
            "6": "Z2 gauge theory phases, L = 2, 8 link qubits (stage 6), plus single-feature baselines."}[stage]
    md = (f"# Seed-to-seed uncertainty, stage {stage}\n\n{note} Seeds: {', '.join(map(str, seeds))}. "
          "Each seed redraws training and test states, shot noise and model initialisations. "
          "Std is the sample standard deviation over seeds. 'Best of k' picks the best classical "
          "learner on the test set, which flatters the classical side.\n\n"
          + report.md_table(head, rows) + "\n")
    report.save_text(f"review_seeds_stage{stage}.md", md)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["1", "2", "5", "6"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    t0 = time.time()
    data = {"1": stage1, "2": stage2, "5": stage5, "6": stage6}[args.stage](args.seeds)
    if args.stage == "5":  # the saved stage 5 run used its own seeds
        args.seeds = list(range(json.load(open(report.path("stage5_purity.json")))["settings"]["seeds"]))
    write(args.stage, data, args.seeds)
    print(f"Done in {time.time() - t0:.0f}s: results/review_seeds_stage{args.stage}.{{json,md}}")


if __name__ == "__main__":
    main()
