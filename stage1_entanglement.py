"""Stage 1: is this state entangled? Quantum models vs classical baselines, 2-4 qubits.

Data: random states from a parameterised circuit (random local rotations, RZZ
entanglers). Product states have all entangler angles 0; entangled states have
random ones and Meyer-Wallach entanglement Q >= 0.2 (see entangle/states.py).

Models:
  QNN 1-copy    trainable circuit on the state, read out one qubit
  QNN 2-copy    two copies of the state, trainable block per qubit pair (can learn the SWAP test)
  Classical     logistic regression / RBF SVM / MLP on
                (a) the full state vector (2^(n+1) real numbers; not available in a real experiment)
                (b) measurement snapshots: random-Pauli-basis shots (classical shadows) turned into
                    Pauli expectation and single-qubit purity estimates

Every model gets the same copy budget at test time: the 1-copy QNN uses
`--copies` shots, the 2-copy QNN `--copies / 2` shots (2 copies per shot), and the
classical snapshot models `--copies` snapshots per state.

Usage:
    python stage1_entanglement.py                       # 2, 3, 4 qubits (about 2 minutes)
    python stage1_entanglement.py --qubits 2 3 --copies 128
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import one_copy_model, two_copy_model
from entangle.classical import evaluate_all, statevector_features
from entangle.measure import all_paulis, exact_features, random_pauli_snapshots, shadow_features
from entangle.qnn import QuantumClassifier
from entangle.states import make_entanglement_data


def run_qubits(n, args, log=print):
    rng = np.random.default_rng(args.seed + 7 * n)
    train = make_entanglement_data(n, args.train, seed=args.seed + 100 * n, q_min=args.q_min)
    test = make_entanglement_data(n, args.test, seed=args.seed + 100 * n + 1, q_min=args.q_min)
    res = {"qubits": n, "train_states": args.train, "test_states": args.test,
           "copies_per_prediction": args.copies, "quantum": {}, "classical": {}}

    for build in (one_copy_model, two_copy_model):
        model = build(n)
        clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                           restarts=args.restarts, seed=args.seed)
        shots = args.copies // model.copies
        acc_exact = float(np.mean(clf.predict(test.states) == test.y))
        acc_shots = float(np.mean(clf.predict(test.states, shots=shots, rng=rng) == test.y))
        res["quantum"][model.name] = {
            "qubits_used": model.num_qubits, "weights": model.num_weights,
            "train_accuracy": float(np.mean(clf.predict(train.states) == train.y)),
            "test_accuracy_exact": acc_exact, "test_accuracy_shots": acc_shots,
            "shots": shots, "train_seconds": round(clf.train_seconds, 1),
            "description": model.description,
        }
        log(f"  {model.name:<12} exact {acc_exact:.3f}   {shots:>5} shots {acc_shots:.3f}   "
            f"({model.num_weights} weights, {clf.train_seconds:.0f}s)")

    # (a) full state vector
    acc = evaluate_all(statevector_features(train.states), train.y,
                       statevector_features(test.states), test.y, args.seed)
    res["classical"]["full state vector"] = {"features": 2 ** (n + 1), "test_accuracy_exact": acc}
    log(f"  classical, full state vector ({2 ** (n + 1)} numbers): "
        + ", ".join(f"{k} {v:.3f}" for k, v in acc.items()))

    # (b) measurement snapshots
    paulis = all_paulis(n)
    snap_tr = shadow_features(random_pauli_snapshots(train.states, n, args.copies, rng), n, paulis)
    snap_te = shadow_features(random_pauli_snapshots(test.states, n, args.copies, rng), n, paulis)
    acc_sh = evaluate_all(snap_tr, train.y, snap_te, test.y, args.seed)
    acc_ex = evaluate_all(exact_features(train.states, n, paulis), train.y,
                          exact_features(test.states, n, paulis), test.y, args.seed)
    res["classical"]["snapshots"] = {"features": len(paulis) + n, "snapshots": args.copies,
                                     "test_accuracy_exact": acc_ex, "test_accuracy_shots": acc_sh}
    # physics-informed variant: only the n single-qubit purity estimates (the last n columns)
    ex_tr, ex_te = exact_features(train.states, n, paulis), exact_features(test.states, n, paulis)
    pur_ex = evaluate_all(ex_tr[:, -n:], train.y, ex_te[:, -n:], test.y, args.seed)
    pur_sh = evaluate_all(snap_tr[:, -n:], train.y, snap_te[:, -n:], test.y, args.seed)
    res["classical"]["snapshots, purities only"] = {"features": n, "snapshots": args.copies,
                                                    "test_accuracy_exact": pur_ex,
                                                    "test_accuracy_shots": pur_sh}
    log(f"  classical, {args.copies} snapshots, purity features only: "
        + ", ".join(f"{k} {v:.3f}" for k, v in pur_sh.items()))
    log(f"  classical, snapshots ({len(paulis) + n} features), exact: "
        + ", ".join(f"{k} {v:.3f}" for k, v in acc_ex.items()))
    log(f"  classical, {args.copies} snapshots: " + ", ".join(f"{k} {v:.3f}" for k, v in acc_sh.items()))
    return res


def best_snapshot(r, key):
    """Best classical snapshot model (3 learners x 2 feature sets), picked on the test set.

    Picking on the test set flatters the classical side, which is the safe direction
    for a quantum-vs-classical comparison."""
    return max(max(r["classical"][k][f"test_accuracy_{key}"].values())
               for k in ("snapshots", "snapshots, purities only"))


def write_outputs(results, args):
    report.save_json("stage1_entanglement.json", {"settings": vars(args), "results": results})

    rows = []
    for r in results:
        n, c = r["qubits"], r["copies_per_prediction"]
        for name, q in r["quantum"].items():
            rows.append([n, name, f"{q['qubits_used']} qubits, {q['weights']} weights",
                         report.fmt(q["test_accuracy_exact"]),
                         f"{report.fmt(q['test_accuracy_shots'])} ({q['shots']} shots)"])
        sv = r["classical"]["full state vector"]
        for name, a in sv["test_accuracy_exact"].items():
            rows.append([n, f"{name}, full state vector", f"{sv['features']} numbers", report.fmt(a), "n/a"])
        for kind in ("snapshots", "snapshots, purities only"):
            sn = r["classical"][kind]
            for name in sn["test_accuracy_exact"]:
                rows.append([n, f"{name}, {kind}", f"{sn['features']} features",
                             report.fmt(sn["test_accuracy_exact"][name]),
                             f"{report.fmt(sn['test_accuracy_shots'][name])} ({c} snapshots)"])
    table = report.md_table(["Qubits", "Model", "Size", "Test acc. (exact)",
                             f"Test acc. ({args.copies} copies of each state)"], rows)
    md = (f"# Stage 1: entanglement classification\n\n"
          f"Product vs entangled (Meyer-Wallach Q >= {args.q_min}) random states. "
          f"{args.train} training and {args.test} test states per qubit count, balanced. "
          f"Exact = infinite shots. The copy budget is the number of copies of each test "
          f"state a model may use (the 2-copy QNN uses 2 per shot).\n\n{table}\n")
    report.save_text("stage1_entanglement.md", md)

    qs = [r["qubits"] for r in results]
    fig, axes = report.axes_grid(2, "Stage 1: entangled or product? Test accuracy by qubit count")
    for ax, key, title in ((axes[0], "exact", "Exact (infinite shots)"),
                           (axes[1], "shots", f"{args.copies} copies of each test state")):
        if key == "exact":  # the state vector is the same with or without shots
            report.line(ax, qs, [max(r["classical"]["full state vector"]["test_accuracy_exact"].values())
                                 for r in results], "Classical, full state vector (best of 3)",
                        "classical_state")
        report.line(ax, qs, [best_snapshot(r, key) for r in results],
                    "Classical, snapshots (best of 6)", "classical_snap")
        report.line(ax, qs, [r["quantum"]["QNN 1-copy"][f"test_accuracy_{key}"] for r in results],
                    "QNN 1-copy", "quantum_alt")
        report.line(ax, qs, [r["quantum"]["QNN 2-copy"][f"test_accuracy_{key}"] for r in results],
                    "QNN 2-copy", "quantum", dashed=True)  # dashed so overlapping lines stay visible
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("qubits")
        ax.set_xticks(qs)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(0.5, color=report.INK2, linewidth=1, linestyle=":")
    axes[0].set_ylabel("test accuracy")
    report.finish(fig, axes, "stage1_entanglement.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--train", type=int, default=200, help="training states per qubit count")
    p.add_argument("--test", type=int, default=400, help="test states per qubit count")
    p.add_argument("--copies", type=int, default=256, help="copies of each test state a model may use")
    p.add_argument("--q-min", type=float, default=0.2, help="minimum Meyer-Wallach Q of entangled states")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    results = []
    for n in args.qubits:
        print(f"\n{n} qubits")
        results.append(run_qubits(n, args))
    write_outputs(results, args)
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage1_entanglement.{{json,md,png}}")


if __name__ == "__main__":
    main()
