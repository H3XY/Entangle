"""Stage 5 classical-shadow baseline with validation-based model selection.

The stage 5 table picks the best of three classical learners on the TEST set at each copy
count (flattering the classical side). Here the learner and its hyperparameters are chosen on
a validation split of the training states only, and the chosen model is evaluated once on test:

  - training states (200) split 150 fit / 50 validation (stratified, fixed per seed)
  - candidates: logistic regression C in {0.01, 0.1, 1, 10, 100};
                RBF SVM C in {0.1, 1, 10, 100} x gamma in {scale, 0.1, 1};
                MLP (64, 64) alpha in {1e-4, 1e-3, 1e-2}
  - the best validation accuracy wins (ties: the earlier candidate in that list, i.e. the
    simpler model), it is refit on all 200 training states and scored on the 400 test states

Same data as stage 5 (train seed 100n + 2s, test seed 100n + 2s + 1, s = 0, 1, 2), same
features (shadow estimates of the global purity and the n single-qubit purities from c random
Pauli snapshots). The snapshots themselves are fresh draws, so numbers differ slightly from
stage 5 even under the old rule; the old rule (best of the 3 default learners on test) is
reported on these same draws for a like-for-like comparison, as are all 3 learners separately.

Usage:
    python review_selection.py                 # n = 2..6, 3 seeds (~15 minutes)
"""
import argparse
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from entangle import report
from entangle.classical import classical_models
from entangle.measure import pauli_setting_cdfs, purity_features, random_pauli_snapshots
from entangle.states import make_purity_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def candidates(seed):
    out = [(f"LR C={C}", make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)))
           for C in (0.01, 0.1, 1, 10, 100)]
    out += [(f"SVM C={C} gamma={g}", make_pipeline(StandardScaler(), SVC(C=C, gamma=g)))
            for C in (0.1, 1, 10, 100) for g in ("scale", 0.1, 1)]
    out += [(f"MLP alpha={a}", make_pipeline(StandardScaler(), MLPClassifier((64, 64), alpha=a, max_iter=3000,
                                                                              random_state=seed)))
            for a in (1e-4, 1e-3, 1e-2)]
    return out


def select_and_test(Xtr, ytr, Xte, yte, seed):
    fi, vi = train_test_split(np.arange(len(ytr)), test_size=0.25, stratify=ytr, random_state=seed)
    best = None
    for name, m in candidates(seed):
        acc = np.mean(m.fit(Xtr[fi], ytr[fi]).predict(Xtr[vi]) == ytr[vi])
        if best is None or acc > best[1]:
            best = (name, acc, m)
    name, val, m = best
    return name, float(val), float(np.mean(m.fit(Xtr, ytr).predict(Xte) == yte))


def copies_needed(curve, copies, target):
    for c, a in zip(copies, curve):
        if a >= target:
            return c
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--copies", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--target", type=float, default=0.9)
    args = p.parse_args()
    t0 = time.time()
    learners = list(classical_models(0))
    res = {}
    for n in args.qubits:
        res[str(n)] = {"validated": [], "chosen": [], "old_rule": [], **{k: [] for k in learners}}
        for s in range(args.seeds):
            rng = np.random.default_rng(5000 + 10 * n + s)
            train = make_purity_data(n, 200, seed=100 * n + 2 * s)
            test = make_purity_data(n, 400, seed=100 * n + 2 * s + 1)
            ctr, cte = pauli_setting_cdfs(train.states, n), pauli_setting_cdfs(test.states, n)
            val_curve, chosen, old, per = [], [], [], {k: [] for k in learners}
            for c in args.copies:
                Xtr = purity_features(random_pauli_snapshots(train.states, n, c, rng, ctr), n)
                Xte = purity_features(random_pauli_snapshots(test.states, n, c, rng, cte), n)
                name, val, acc = select_and_test(Xtr, train.y, Xte, test.y, s)
                val_curve.append(acc)
                chosen.append(name)
                accs = {k: float(np.mean(m.fit(Xtr, train.y).predict(Xte) == test.y))
                        for k, m in classical_models(s).items()}
                for k in learners:
                    per[k].append(accs[k])
                old.append(max(accs.values()))
            res[str(n)]["validated"].append(val_curve)
            res[str(n)]["chosen"].append(chosen)
            res[str(n)]["old_rule"].append(old)
            for k in learners:
                res[str(n)][k].append(per[k])
            print(f"  n={n} seed {s}: validated " + " ".join(f"{a:.2f}" for a in val_curve)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

    C = args.copies
    summary = {}
    for n, d in res.items():
        summary[n] = {}
        for rule in ["validated", "old_rule"] + learners:
            curves = np.array(d[rule])
            summary[n][rule] = {"mean_curve": curves.mean(0).tolist(), "std_curve": curves.std(0, ddof=1).tolist(),
                                "grid": copies_needed(curves.mean(0), C, args.target),
                                "per_seed": [copies_needed(cv, C, args.target) for cv in curves]}
    report.save_json("review_selection.json", {"settings": vars(args), "per_seed": res, "summary": summary})

    nm = lambda v: f"> {C[-1]}" if v is None else str(v)  # noqa: E731
    rows = [[n] + [f"{nm(summary[n][r]['grid'])} ({', '.join(nm(v) for v in summary[n][r]['per_seed'])})"
                   for r in ("validated", "old_rule", *learners)] for n in summary]
    curve_rows = []
    for n in summary:
        for r in ("validated", "old_rule", *learners):
            curve_rows.append([n, r] + [f"{m:.3f} ± {s:.3f}" for m, s in
                                        zip(summary[n][r]["mean_curve"], summary[n][r]["std_curve"])])
    chosen_rows = [[n, s, ", ".join(f"{c}: {x}" for c, x in zip(C, res[n]["chosen"][s]))]
                   for n in res for s in range(args.seeds)]
    md = "\n".join([
        "# Stage 5 classical shadows with validation-based model selection", "",
        "Learner and hyperparameters chosen on a 50-state validation split of the 200 training states, then "
        "refit on all 200 and evaluated once on the 400 test states. 'Old rule' = best of the 3 default learners "
        f"on the test set (as in the stage 5 table), on the same snapshot draws. {args.seeds} seeds.", "",
        f"## Copies of each test state needed for {args.target:.0%}", "",
        "Grid definition (smallest power of 2 where the mean over seeds reaches the target), per seed in brackets.", "",
        report.md_table(["Qubits", "Validated selection", "Old rule (best on test)"] + learners, rows), "",
        "## Accuracy vs copies (mean ± std over seeds)", "",
        report.md_table(["Qubits", "Rule"] + [str(c) for c in C], curve_rows), "",
        "## Model chosen on validation, per copy count", "",
        report.md_table(["Qubits", "Seed", "Chosen"], chosen_rows), ""])
    report.save_text("review_selection.md", md)
    print(f"Done in {time.time() - t0:.0f}s: results/review_selection.{{json,md}}")


if __name__ == "__main__":
    main()
