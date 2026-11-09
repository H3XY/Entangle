"""Stage 5: a global property. Is this state pure or mixed?

Data: Haar-random pure states vs random rank-2 mixed states (Tr rho^2 in
[0.5, 0.58]) on n = 2-6 qubits. Both classes average to the maximally mixed
state, and their single-qubit marginals become indistinguishable as n grows:
only the purity of the whole state separates them. This is the setting where
theory proves an exponential gap (Huang, Kueng & Preskill 2021; Chen, Cotler,
Huang & Li 2022): any strategy that measures one copy at a time needs a number
of copies growing exponentially with n, while measuring two copies together
(the SWAP test) needs O(1).

Models:
  QNN 2-copy (parity)         a trainable block on each pair (q_i, q_i'), class 1 if an odd
                              number of pairs read 11. With Bell-basis blocks, P(class 1) =
                              (1 - Tr rho^2) / 2. Trained from scratch (up to --scratch-max qubits).
  QNN 2-copy (parity, shared) one block shared by all pairs, trained from scratch, and also
                              trained at 2 qubits and transferred (fine-tuned briefly) to n qubits
  QNN 1-copy                  linear in rho, so it cannot see purity (control)
  Classical                   logistic regression / RBF SVM / MLP on classical-shadow estimates
                              of the global purity and the single-qubit purities, from random
                              Pauli snapshots (the best known single-copy strategy)

Main output: test accuracy vs copies of each test state, and the number of copies
each approach needs to reach --target accuracy, for every n.

Usage:
    python stage5_purity.py                         # 2-6 qubits, 3 seeds (about 15 minutes)
    python stage5_purity.py --qubits 2 3 4 --seeds 1
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import one_copy_model, two_copy_model
from entangle.classical import classical_models, fit_predict
from entangle.measure import pauli_setting_cdfs, purity_features, random_pauli_snapshots
from entangle.qnn import QuantumClassifier
from entangle.states import make_purity_data

TRANSFER = "QNN 2-copy, shared, transferred from 2 qubits"
SCRATCH = "QNN 2-copy, per-pair blocks, from scratch"
SHARED = "QNN 2-copy, shared block, from scratch"
ONE = "QNN 1-copy"
CLASSICAL = "Classical shadows (purity estimates)"


def best_classical(Xtr, ytr, Xte, yte, seed):
    return max(float(np.mean(fit_predict(m, Xtr, ytr, Xte) == yte)) for m in classical_models(seed).values())


def acc(clf, data, shots, rng):
    return float(np.mean(clf.predict(data.states, shots=shots, rng=rng) == data.y))


def run(args, log=print):
    copies = args.copies
    exact = {k: {n: [] for n in args.qubits} for k in (TRANSFER, SCRATCH, SHARED, ONE)}
    sweep = {k: {n: {c: [] for c in copies} for n in args.qubits} for k in (TRANSFER, ONE, CLASSICAL)}
    seconds = {k: {n: [] for n in args.qubits} for k in (TRANSFER, SCRATCH, SHARED)}

    for seed in range(args.seeds):
        rng = np.random.default_rng(10 + seed)
        small = make_purity_data(2, args.train, seed=700 + seed)
        base = QuantumClassifier(two_copy_model(2, "parity", shared=True)).fit(
            small.states, small.y, maxiter=args.maxiter, restarts=args.restarts, seed=seed)
        for n in args.qubits:
            train = make_purity_data(n, args.train, seed=100 * n + 2 * seed)
            test = make_purity_data(n, args.test, seed=100 * n + 2 * seed + 1)

            t0 = time.time()
            clf = QuantumClassifier(two_copy_model(n, "parity", shared=True)).fit(
                train.states, train.y, maxiter=args.finetune, init=base.weights, seed=seed)
            seconds[TRANSFER][n].append(time.time() - t0)
            exact[TRANSFER][n].append(acc(clf, test, None, rng))
            for c in copies:
                sweep[TRANSFER][n][c].append(acc(clf, test, max(1, c // 2), rng))

            one = QuantumClassifier(one_copy_model(n)).fit(train.states, train.y, maxiter=args.maxiter,
                                                           restarts=args.restarts, seed=seed)
            exact[ONE][n].append(acc(one, test, None, rng))
            for c in copies:
                sweep[ONE][n][c].append(acc(one, test, c, rng))

            if n <= args.scratch_max:
                for key, model in ((SCRATCH, two_copy_model(n, "parity")),
                                   (SHARED, two_copy_model(n, "parity", shared=True))):
                    t0 = time.time()
                    m = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                                     restarts=args.restarts, seed=seed)
                    seconds[key][n].append(time.time() - t0)
                    exact[key][n].append(acc(m, test, None, rng))

            cdf_tr, cdf_te = pauli_setting_cdfs(train.states, n), pauli_setting_cdfs(test.states, n)
            for c in copies:
                Xtr = purity_features(random_pauli_snapshots(train.states, n, c, rng, cdf_tr), n)
                Xte = purity_features(random_pauli_snapshots(test.states, n, c, rng, cdf_te), n)
                sweep[CLASSICAL][n][c].append(best_classical(Xtr, train.y, Xte, test.y, seed))

            log(f"  seed {seed}, {n} qubits: exact " +
                ", ".join(f"{k.split(',')[0] if k == ONE else k} {v[n][-1]:.3f}" for k, v in exact.items() if v[n]))
            for k in sweep:
                log(f"    {k:<48} " + " ".join(f"{sweep[k][n][c][-1]:.2f}" for c in copies))
    return exact, sweep, seconds


def copies_needed(curve, copies, target):
    """Smallest copy count whose mean accuracy reaches target (None if none does)."""
    for c in copies:
        if np.mean(curve[c]) >= target:
            return c
    return None


def summarise(exact, sweep, seconds, args):
    stat = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v)), "runs": v} if v else None  # noqa: E731
    return {
        "exact": {k: {str(n): stat(v[n]) for n in args.qubits} for k, v in exact.items()},
        "by_copies": {k: {str(n): {str(c): stat(v[n][c]) for c in args.copies} for n in args.qubits}
                      for k, v in sweep.items()},
        "copies_needed": {k: {str(n): copies_needed(v[n], args.copies, args.target) for n in args.qubits}
                          for k, v in sweep.items() if k != ONE},
        "train_seconds": {k: {str(n): stat(v[n]) for n in args.qubits} for k, v in seconds.items()},
    }


def write_outputs(res, args):
    report.save_json("stage5_purity.json", {"settings": vars(args), "results": res})
    fmt = lambda s: "–" if s is None else f"{s['mean']:.3f} ± {s['std']:.3f}"  # noqa: E731
    exact_rows = [[n] + [fmt(res["exact"][k][str(n)]) for k in (SCRATCH, SHARED, TRANSFER, ONE)]
                  for n in args.qubits]
    need_rows = [[n] + [str(res["copies_needed"][k][str(n)] or f"> {args.copies[-1]}")
                        for k in (TRANSFER, CLASSICAL)] for n in args.qubits]
    sweep_rows = []
    for n in args.qubits:
        for k in (TRANSFER, CLASSICAL, ONE):
            sweep_rows.append([n, k] + [f"{res['by_copies'][k][str(n)][str(c)]['mean']:.3f}" for c in args.copies])
    md = "\n".join([
        "# Stage 5: pure or mixed? (a global property)", "",
        f"Haar-random pure states vs random rank-2 mixed states. {args.train} training and {args.test} test "
        f"states per qubit count; mean ± std over {args.seeds} seeds. Classical columns: best of logistic "
        "regression, RBF SVM and MLP on classical-shadow estimates of the global and single-qubit purities.", "",
        "## Exact accuracy (infinite shots)", "",
        report.md_table(["Qubits", SCRATCH, SHARED, TRANSFER, ONE], exact_rows), "",
        f"Per-pair from scratch and shared from scratch are trained only up to {args.scratch_max} qubits.", "",
        f"## Copies of each test state needed for {args.target:.0%} accuracy", "",
        report.md_table(["Qubits", "QNN 2-copy (transferred)", "Classical shadows"], need_rows), "",
        "## Accuracy vs copies of each test state", "",
        report.md_table(["Qubits", "Model"] + [str(c) for c in args.copies], sweep_rows), ""])
    report.save_text("stage5_purity.md", md)

    fig, axes = report.axes_grid(2, "Stage 5: pure or mixed? Copies of each state needed", width=5.4)
    ax = axes[0]
    for k, fam in ((CLASSICAL, "classical_snap"), (TRANSFER, "quantum")):
        ns, need = [], []
        for n in args.qubits:
            v = res["copies_needed"][k][str(n)]
            if v is not None:
                ns.append(n)
                need.append(v)
        report.line(ax, ns, need, "QNN 2-copy (SWAP test, learned)" if k == TRANSFER else
                    "Classical shadows (best of 3 learners)", fam, dashed=(fam == "quantum"))
    ax.set_yscale("log", base=2)
    ax.set_xticks(args.qubits)
    ax.set_xlabel("qubits")
    ax.set_ylabel(f"copies for {args.target:.0%} accuracy")
    ax.set_title(f"Copies needed for {args.target:.0%} test accuracy", fontsize=10, loc="left")
    ax = axes[1]
    n = args.qubits[-1]
    for k, fam, label in ((CLASSICAL, "classical_snap", "Classical shadows (best of 3 learners)"),
                          (ONE, "quantum_alt", "QNN 1-copy"),
                          (TRANSFER, "quantum", "QNN 2-copy (SWAP test, learned)")):
        s = res["by_copies"][k][str(n)]
        report.line(ax, args.copies, [s[str(c)]["mean"] for c in args.copies], label, fam,
                    err=[s[str(c)]["std"] for c in args.copies], dashed=(fam == "quantum"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.copies)
    ax.set_xticklabels([str(c) for c in args.copies], fontsize=8)
    ax.set_xlabel("copies of each test state")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color=report.INK2, linewidth=1, linestyle=":")
    ax.set_title(f"Accuracy vs copies, {n} qubits", fontsize=10, loc="left")
    report.finish(fig, axes, "stage5_purity.png", legend_ax=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--copies", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--target", type=float, default=0.9, help="accuracy for the copies-needed metric")
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--finetune", type=int, default=30, help="L-BFGS-B iterations after transfer")
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--scratch-max", type=int, default=5, help="largest n for the from-scratch models")
    args = p.parse_args()
    t0 = time.time()
    exact, sweep, seconds = run(args)
    res = summarise(exact, sweep, seconds, args)
    write_outputs(res, args)
    print(f"\nCopies needed for {args.target:.0%}: " + "; ".join(
        f"{k}: " + ", ".join(f"n={n} {v}" for n, v in res["copies_needed"][k].items()) for k in res["copies_needed"]))
    print(f"Done in {time.time() - t0:.0f}s. Results: results/stage5_purity.{{json,md,png}}")


if __name__ == "__main__":
    main()
