"""Stage 3: data efficiency. How accuracy grows with training states and with measurement shots.

For both tasks (stage 1 entanglement, stage 2 Ising phases) and for quantum and
classical models:
  (a) test accuracy vs number of training states, all models exact (infinite shots)
  (b) test accuracy vs copies of each test state, with --train-fixed training states
      (quantum models are trained on exact probabilities and then measured with
      finite shots; classical snapshot models are trained and tested on features
      from that many snapshots)
Each point is the mean over --seeds independent datasets; the band is +/- 1 std.

Usage:
    python stage3_efficiency.py                         # both tasks, about 10 minutes
    python stage3_efficiency.py --task entanglement --seeds 2
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import hea_model, one_copy_model, qcnn_model, two_copy_model
from entangle.classical import classical_models, fit_predict, statevector_features
from entangle.measure import (all_paulis, exact_features, ising_paulis, random_pauli_snapshots,
                              shadow_features, xz_snapshots)
from entangle.qnn import QuantumClassifier
from entangle.states import make_entanglement_data, make_phase_data

TASKS = {
    "entanglement": dict(
        title="Entangled or product?",
        data=lambda n, N, seed: make_entanglement_data(n, N, seed=seed),
        quantum={"QNN 2-copy": (two_copy_model, "quantum"), "QNN 1-copy": (one_copy_model, "quantum_alt")},
        paulis=all_paulis, purities=True, snapshots=random_pauli_snapshots,
        snap_label="Classical, random-Pauli snapshots"),
    "phases": dict(
        title="Ordered or disordered Ising chain?",
        data=lambda n, N, seed: make_phase_data(n, N, seed=seed),
        quantum={"QCNN": (qcnn_model, "quantum"), "QNN (generic ansatz)": (hea_model, "quantum_alt")},
        paulis=ising_paulis, purities=False, snapshots=xz_snapshots,
        snap_label="Classical, Z/X snapshots"),
}
STATE_LABEL = "Classical, full state vector"


def balanced_subset(y, size, rng):
    """`size` indices with equal numbers of each class (so tiny training sets still have both)."""
    idx = [rng.permutation(np.flatnonzero(y == c))[: size // 2 + (c and size % 2)] for c in (0, 1)]
    return rng.permutation(np.concatenate(idx))


def best_classical(Xtr, ytr, Xte, yte, seed):
    """Best of logistic regression / RBF SVM / MLP, picked on the test set (flatters classical)."""
    return max(float(np.mean(fit_predict(m, Xtr, ytr, Xte) == yte))
               for m in classical_models(seed).values())


def run_task(name, n, args, log=print):
    task = TASKS[name]
    paulis = task["paulis"](n)
    by_size = {k: {s: [] for s in args.sizes} for k in [*task["quantum"], STATE_LABEL, task["snap_label"]]}
    by_copies = {k: {c: [] for c in args.copies} for k in [*task["quantum"], task["snap_label"]]}

    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        pool = task["data"](n, max(max(args.sizes), args.train_fixed), 500 + 2 * seed)
        test = task["data"](n, args.test, 501 + 2 * seed)
        feats = lambda d: exact_features(d.states, n, paulis, task["purities"])  # noqa: E731
        ex_pool, ex_test = feats(pool), feats(test)
        sv_pool, sv_test = statevector_features(pool.states), statevector_features(test.states)

        # (a) accuracy vs training-set size, exact
        for size in args.sizes:
            idx = balanced_subset(pool.y, size, rng)
            for qname, (build, _) in task["quantum"].items():
                clf = QuantumClassifier(build(n)).fit(pool.states[idx], pool.y[idx], maxiter=args.maxiter,
                                                      restarts=args.restarts, seed=seed)
                by_size[qname][size].append(float(np.mean(clf.predict(test.states) == test.y)))
            by_size[STATE_LABEL][size].append(best_classical(sv_pool[idx], pool.y[idx], sv_test, test.y, seed))
            by_size[task["snap_label"]][size].append(
                best_classical(ex_pool[idx], pool.y[idx], ex_test, test.y, seed))
            log(f"  seed {seed}, {size:4d} training states: "
                + ", ".join(f"{k} {v[size][-1]:.3f}" for k, v in by_size.items()))

        # (b) accuracy vs copies per test state, fixed training-set size
        idx = balanced_subset(pool.y, args.train_fixed, rng)
        train = pool.subset(idx)
        for qname, (build, _) in task["quantum"].items():
            model = build(n)
            clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                               restarts=args.restarts, seed=seed)
            for c in args.copies:
                shots = max(1, c // model.copies)
                by_copies[qname][c].append(
                    float(np.mean(clf.predict(test.states, shots=shots, rng=rng) == test.y)))
        for c in args.copies:
            tr = shadow_features(task["snapshots"](train.states, n, c, rng), n, paulis, task["purities"])
            te = shadow_features(task["snapshots"](test.states, n, c, rng), n, paulis, task["purities"])
            by_copies[task["snap_label"]][c].append(best_classical(tr, train.y, te, test.y, seed))
        log(f"  seed {seed}, copies sweep: "
            + "; ".join(f"{k} " + " ".join(f"{v[c][-1]:.2f}" for c in args.copies) for k, v in by_copies.items()))

    summarise = lambda d: {k: {str(x): {"mean": float(np.mean(v)), "std": float(np.std(v)), "runs": v}  # noqa: E731
                               for x, v in series.items()} for k, series in d.items()}
    return {"task": name, "qubits": n, "by_training_states": summarise(by_size),
            "by_copies": summarise(by_copies)}


def family(task, label):
    if label in task["quantum"]:
        return task["quantum"][label][1]
    return "classical_state" if label == STATE_LABEL else "classical_snap"


def plot(res, args):
    task = TASKS[res["task"]]
    fig, axes = report.axes_grid(2, f"Stage 3: {task['title']} ({res['qubits']} qubits), mean of {args.seeds} seeds")
    for ax, key, xs, xlabel, title in (
            (axes[0], "by_training_states", args.sizes, "training states", "Exact (infinite shots)"),
            (axes[1], "by_copies", args.copies, "copies of each test state",
             f"{args.train_fixed} training states")):
        for label, series in res[key].items():
            mean = [series[str(x)]["mean"] for x in xs]
            std = [series[str(x)]["std"] for x in xs]
            fam = family(task, label)
            report.line(ax, xs, mean, label, fam, err=std, dashed=(fam == "quantum"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color=report.INK2, linewidth=1, linestyle=":")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("test accuracy")
    report.finish(fig, axes, f"stage3_{res['task']}.png")


def markdown(all_res, args):
    out = ["# Stage 3: data efficiency", "",
           f"Mean test accuracy (± std over {args.seeds} seeds, {args.test} test states each). "
           "Classical columns are the best of logistic regression, RBF SVM and MLP, picked on the "
           "test set, which flatters the classical side.", ""]
    for res in all_res:
        task = TASKS[res["task"]]
        out += [f"## {task['title']} ({res['qubits']} qubits)", "", "### vs training states (exact)", ""]
        labels = list(res["by_training_states"])
        rows = [[s] + [f"{res['by_training_states'][k][str(s)]['mean']:.3f} ± "
                       f"{res['by_training_states'][k][str(s)]['std']:.3f}" for k in labels] for s in args.sizes]
        out += [report.md_table(["Training states"] + labels, rows), "",
                f"### vs copies of each test state ({args.train_fixed} training states)", ""]
        labels = list(res["by_copies"])
        rows = [[c] + [f"{res['by_copies'][k][str(c)]['mean']:.3f} ± "
                       f"{res['by_copies'][k][str(c)]['std']:.3f}" for k in labels] for c in args.copies]
        out += [report.md_table(["Copies"] + labels, rows), ""]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["entanglement", "phases", "both"], default="both")
    p.add_argument("--qubits-entanglement", type=int, default=4)
    p.add_argument("--qubits-phases", type=int, default=8)
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--copies", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 1024])
    p.add_argument("--train-fixed", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--restarts", type=int, default=2)
    args = p.parse_args()
    t0 = time.time()
    tasks = ["entanglement", "phases"] if args.task == "both" else [args.task]
    all_res = []
    for name in tasks:
        n = args.qubits_entanglement if name == "entanglement" else args.qubits_phases
        print(f"\n{name}, {n} qubits")
        res = run_task(name, n, args)
        plot(res, args)
        all_res.append(res)
    report.save_json("stage3_efficiency.json", {"settings": vars(args), "results": all_res})
    report.save_text("stage3_efficiency.md", markdown(all_res, args))
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage3_efficiency.{{json,md}}, "
          + ", ".join(f"results/stage3_{t}.png" for t in tasks))


if __name__ == "__main__":
    main()
