"""Stage 8: pure vs mixed with shallow random circuits (a hardware-sized version of stage 5).

Stage 5 uses Haar-random states, which need circuits whose depth grows exponentially with
n, so only n = 2 ran on hardware (stage 7). Here every state comes from a SHALLOW circuit:

  pure   U |0...0>
  mixed  U rho_0 U^dagger, rho_0 = |0><0| on every qubit except one random qubit s, which starts in
         w|0><0| + (1 - w)|1><1| (w uniform in [0.3, 0.7], so Tr rho^2 is in [0.5, 0.58] as in stage 5)

with U a random brickwork circuit of `depth` layers [Haar-random single-qubit gate on every
qubit, CZ on alternating neighbour pairs of a line], drawn fresh for every state. On hardware a
state is n + 1 qubits: ancilla RY, CX(ancilla -> s), then U on the system qubits; pure states run
the same circuit with RY(0), so the gate count does not depend on the label. Depth 0 (product
states) is the easy control; "haar" reruns the stage 5 states with this script's code.

Purity is unitary-invariant, so the 2-copy SWAP-test model does not care about the depth. The
single-copy side does: at shallow depth the mixedness stays inside the light cone of qubit s, so
local measurements may see it. The theory behind stage 5's exponential gap assumes (approximate)
2-designs, which shallow circuits only become at depth growing with n. This stage measures
where the gap survives.

Models (each seed, each n):
  QNN 2-copy (parity, shared)   trained at 2 qubits on Haar states and fine-tuned at n on Haar
                                states (as in stage 5), then applied unchanged to every depth
  Shadows: global + local       best of LR / RBF SVM / MLP on classical-shadow estimates of the
                                global purity and the n single-qubit purities (stage 5's baseline)
  Shadows: all subsystems       best of the 3 on shadow estimates of Tr rho_A^2 for every subset A
                                of the qubits (2^n - 1 features). A pure state has
                                Tr rho_A^2 = Tr rho_Abar^2 for every bipartition; mixed states break
                                this, and at shallow depth small subsystems already show it.
Both classical baselines pick the best learner on the test set, which flatters them (see
review_selection.py). Diagnostics: the same learners on EXACT (infinite-copy) local and
proper-subsystem purities, i.e. how much of the label is visible without the global purity.

Copies: c copies = c preparations. QNN: c/2 shots (2 preparations each). Shadows: c snapshots.

`--plan` (no simulation, no account): transpiles the hardware circuits of one pure and one mixed
state per (n, depth) for FakeMarrakesh and reports two-qubit gate counts and the estimated
quantum seconds of a stage-7-sized run (60 test + 20 calibration states, 1024 copies each).

Usage:
    python stage8_shallow_purity.py                              # n = 2..6, depths 0 1 2 4 8 haar, 3 seeds
    python stage8_shallow_purity.py --qubits 2 3 4 --depths 1 2 --seeds 1
    python stage8_shallow_purity.py --plan --qubits 2 3 4 --depths 1 2 4
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from entangle import report
from entangle.circuits import two_copy_model
from entangle.classical import classical_models, fit_predict
from entangle.measure import pauli_setting_cdfs, purity_features, random_pauli_snapshots, subset_purity_features
from entangle.qnn import QuantumClassifier
from entangle.states import make_purity_data, make_shallow_purity_data

QNN = "QNN 2-copy (parity, shared)"
LOCAL = "Shadows: global + local purities"
SUBSETS = "Shadows: all subsystem purities"
METHODS = (QNN, LOCAL, SUBSETS)
EXACT_LOCAL = "Exact single-qubit purities"
EXACT_SUBSETS = "Exact proper-subsystem purities"


def depth_key(d):
    return "haar" if d == "haar" else int(d)


def make_data(n, num, depth, seed):
    if depth == "haar":
        return make_purity_data(n, num, seed=seed)
    return make_shallow_purity_data(n, num, depth, seed=seed)


def exact_subset_purities(states, n):
    """Tr(rho_A^2) for every non-empty subset A (bitmask order, last = whole state)."""
    out = np.empty((len(states), 2 ** n - 1))
    for k, A in enumerate(states):
        t = A.reshape([2] * n + [A.shape[1]])
        for m in range(1, 2 ** n):
            keep = [n - 1 - q for q in range(n) if m >> q & 1]       # tensor axes of the qubits in A
            rest = [ax for ax in range(n + 1) if ax not in keep]       # traced out, incl. purification
            M = np.transpose(t, keep + rest).reshape(2 ** len(keep), -1)
            rho = M @ M.conj().T
            out[k, m - 1] = np.real(np.vdot(rho, rho))
    return out


def best_classical(Xtr, ytr, Xte, yte, seed):
    return max(float(np.mean(fit_predict(m, Xtr, ytr, Xte) == yte)) for m in classical_models(seed).values())


def job(task):
    """One (seed, n): train the QNN once, then every depth."""
    seed, n, args = task
    t0 = time.time()
    small = make_purity_data(2, args.train, seed=700 + seed)                     # as in stage 5
    base = QuantumClassifier(two_copy_model(2, "parity", shared=True)).fit(
        small.states, small.y, maxiter=args.maxiter, restarts=args.restarts, seed=seed)
    haar_train = make_purity_data(n, args.train, seed=100 * n + 2 * seed)
    clf = QuantumClassifier(two_copy_model(n, "parity", shared=True)).fit(
        haar_train.states, haar_train.y, maxiter=args.finetune, init=base.weights, seed=seed)
    out = {"seed": seed, "n": n, "train_seconds": time.time() - t0, "depths": {}}
    for d in args.depths:
        dseed = 80000 + 1000 * n + 10 * seed + (99 if d == "haar" else d)
        train, test = make_data(n, args.train, d, dseed), make_data(n, args.test, d, dseed + 500)
        rng = np.random.default_rng(dseed + 7)
        r = {"exact": {QNN: float(np.mean(clf.predict(test.states) == test.y))}, "by_copies": {}}
        r["by_copies"][QNN] = [float(np.mean(clf.predict(test.states, shots=max(1, c // 2), rng=rng) == test.y))
                               for c in args.copies]
        Etr, Ete = exact_subset_purities(train.states, n), exact_subset_purities(test.states, n)
        singles = [(1 << q) - 1 for q in range(n)]
        r["exact"][EXACT_LOCAL] = best_classical(Etr[:, singles], train.y, Ete[:, singles], test.y, seed)
        r["exact"][EXACT_SUBSETS] = (best_classical(Etr[:, :-1], train.y, Ete[:, :-1], test.y, seed)
                                     if n > 1 else None)
        cdf_tr, cdf_te = pauli_setting_cdfs(train.states, n), pauli_setting_cdfs(test.states, n)
        r["by_copies"][LOCAL], r["by_copies"][SUBSETS] = [], []
        for c in args.copies:
            str_ = random_pauli_snapshots(train.states, n, c, rng, cdf_tr)
            ste = random_pauli_snapshots(test.states, n, c, rng, cdf_te)
            r["by_copies"][LOCAL].append(best_classical(purity_features(str_, n), train.y,
                                                        purity_features(ste, n), test.y, seed))
            r["by_copies"][SUBSETS].append(best_classical(subset_purity_features(str_, n), train.y,
                                                          subset_purity_features(ste, n), test.y, seed))
        out["depths"][str(d)] = r
    out["seconds"] = time.time() - t0
    return out


def crossing(copies, acc, target):
    """Copies where accuracy first reaches target, interpolated linearly in log2(copies)."""
    for i, a in enumerate(acc):
        if a >= target:
            if i == 0:
                return float(copies[0])
            a0, a1 = acc[i - 1], a
            x0, x1 = np.log2(copies[i - 1]), np.log2(copies[i])
            return float(2 ** (x0 + (target - a0) / (a1 - a0) * (x1 - x0)))
    return None


def nm_copies(v):
    return "-" if v is None else f"{v:.0f}"


def summarise(jobs, args):
    C = args.copies
    res = {}
    for n in args.qubits:
        res[str(n)] = {}
        for d in args.depths:
            runs = [j["depths"][str(d)] for j in jobs if j["n"] == n]
            s = {"exact": {k: float(np.mean([r["exact"][k] for r in runs]))
                           if runs[0]["exact"][k] is not None else None for k in runs[0]["exact"]},
                 "by_copies": {}, "copies_needed": {}, "copies_needed_per_seed": {}}
            for m in METHODS:
                curves = np.array([r["by_copies"][m] for r in runs])
                s["by_copies"][m] = {"mean": curves.mean(0).tolist(), "std": curves.std(0).tolist()}
                s["copies_needed"][m] = crossing(C, curves.mean(0), args.target)
                s["copies_needed_per_seed"][m] = [crossing(C, cv, args.target) for cv in curves]
            res[str(n)][str(d)] = s
    return res


def write_outputs(res, jobs, args, plan=None):
    report.save_json("stage8_shallow_purity.json", {"settings": vars(args), "summary": res, "runs": jobs,
                                                    "plan": plan})
    C = args.copies
    nm = lambda v: f"> {C[-1]}" if v is None else f"{v:.0f}"  # noqa: E731
    need_rows, exact_rows, curve_rows = [], [], []
    for n in args.qubits:
        for d in args.depths:
            s = res[str(n)][str(d)]
            need_rows.append([n, d] + [f"{nm(s['copies_needed'][m])} ("
                                       + ", ".join(nm(v) for v in s["copies_needed_per_seed"][m]) + ")"
                                       for m in METHODS])
            exact_rows.append([n, d] + [report.fmt(s["exact"][k]) for k in (QNN, EXACT_LOCAL, EXACT_SUBSETS)])
            for m in METHODS:
                curve_rows.append([n, d, m] + [f"{a:.3f}" for a in s["by_copies"][m]["mean"]])
    md = ["# Stage 8: pure or mixed? Shallow random circuits", "",
          f"Pure U|0..0> vs mixed U rho_0 U^dagger, U a random brickwork circuit of the given depth (CZ layers), "
          f"fresh per state; 'haar' = stage 5's Haar-random states. {args.train} training and {args.test} test "
          f"states per (n, depth); {args.seeds} seeds. The QNN is trained on Haar states only (2 qubits, "
          "fine-tuned at n) and applied unchanged to every depth. Classical: best of LR / RBF SVM / MLP "
          "chosen on the test set (flatters the classical side). c copies = c preparations "
          "(QNN: c/2 shots of two preparations; shadows: c snapshots).", "",
          f"## Copies of each test state needed for {args.target:.0%} accuracy", "",
          "Mean curve over seeds, interpolated linearly in log2(copies); per seed in brackets.", "",
          report.md_table(["Qubits", "Depth"] + list(METHODS), need_rows), "",
          "## How visible is the label without the global purity? (exact, infinite copies)", "",
          "Test accuracy of the best classical learner on exact purities of single qubits, or of every "
          "proper subsystem (all subsets except the whole state). 0.5 = invisible.", "",
          report.md_table(["Qubits", "Depth", "QNN (exact)", EXACT_LOCAL, EXACT_SUBSETS], exact_rows), "",
          "## Accuracy vs copies (mean over seeds)", "",
          report.md_table(["Qubits", "Depth", "Model"] + [str(c) for c in C], curve_rows), ""]
    if plan:
        md += plan_md(plan)
    report.save_text("stage8_shallow_purity.md", "\n".join(md))
    plot(res, args)


def plot(res, args):
    fams = {QNN: ("quantum", True), LOCAL: ("classical_snap", False), SUBSETS: ("classical_snap_alt", False)}
    depths = [d for d in args.depths]
    fig, axes = report.axes_grid(2, "Stage 8: pure or mixed with shallow circuits. Copies needed", width=5.4)
    ax = axes[0]
    n = args.qubits[-1]
    xs = list(range(len(depths)))
    for m, (fam, dashed) in fams.items():
        ys = [res[str(n)][str(d)]["copies_needed"][m] for d in depths]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        report.line(ax, [p[0] for p in pts], [p[1] for p in pts], m, fam, dashed=dashed)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yscale("log", base=2)
    ax.set_xlabel("circuit depth (CZ layers)")
    ax.set_ylabel(f"copies for {args.target:.0%} accuracy")
    ax.set_title(f"{n} qubits, by depth", fontsize=10, loc="left")
    ax = axes[1]
    d = args.mid_depth if args.mid_depth in depths else depths[len(depths) // 2]
    for m, (fam, dashed) in fams.items():
        pts = [(q, res[str(q)][str(d)]["copies_needed"][m]) for q in args.qubits]
        pts = [p for p in pts if p[1] is not None]
        report.line(ax, [p[0] for p in pts], [p[1] for p in pts], m, fam, dashed=dashed)
    ax.set_xticks(args.qubits)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("qubits")
    ax.set_ylabel(f"copies for {args.target:.0%} accuracy")
    ax.set_title(f"Depth {d}, by qubits", fontsize=10, loc="left")
    report.finish(fig, axes, "stage8_shallow_purity.png")


# --------------------------------------------------------------------------- #
# Hardware plan (transpile only)
# --------------------------------------------------------------------------- #

def hardware_circuits(model, weights, prep, n):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    q = QuantumRegister(2 * (n + 1), "q")
    c = ClassicalRegister(2 * n, "meas")
    qnn = QuantumCircuit(q, c)
    qnn.compose(prep, qubits=list(range(n + 1)), inplace=True)
    qnn.compose(prep, qubits=list(range(n + 1, 2 * n + 2)), inplace=True)
    system = list(range(n)) + list(range(n + 1, 2 * n + 1))
    qnn.compose(model.bound(weights), qubits=system, inplace=True)
    qnn.measure(system, list(range(2 * n)))
    sh = QuantumCircuit(QuantumRegister(n + 1, "q"), ClassicalRegister(n, "meas"))
    sh.compose(prep, inplace=True)
    for k in range(n):                       # worst-case setting: every qubit in Y
        sh.sdg(k)
        sh.h(k)
    sh.measure(range(n), range(n))
    return qnn, sh


def plan(args):
    from entangle import ibm
    backend = ibm.get_backend(True, "FakeMarrakesh")
    small = make_purity_data(2, 100, seed=700)          # the 2-qubit shared block, as in stage 5
    weights = QuantumClassifier(two_copy_model(2, "parity", shared=True)).fit(
        small.states, small.y, maxiter=args.maxiter, restarts=1).weights
    rows = []
    for n in args.qubits:
        model = two_copy_model(n, "parity", shared=True)
        for d in args.depths:
            if d == "haar":
                continue
            data = make_shallow_purity_data(n, 2 * args.plan_states, d, seed=4242)
            counts = {}
            for label in (0, 1):
                idx = np.flatnonzero(data.y == label)[:args.plan_states]
                circs = [c for i in idx for c in hardware_circuits(model, weights, data.circuit(i), n)]
                isa = ibm.transpile(backend, circs, seed=7)
                tq, ts = isa[0::2], isa[1::2]
                used = max(len({c.find_bit(qb).index for inst in c.data for qb in inst.qubits}) for c in tq)
                counts[label] = ([c.num_nonlocal_gates() for c in tq], [c.depth() for c in tq],
                                 [c.num_nonlocal_gates() for c in ts], used)
            states, copies = 80, 1024
            settings = 3 ** n
            per = max(1, copies // settings)
            shots = states * (copies // 2 + per * settings)
            ms = lambda v: [float(np.mean(v)), float(np.std(v))]  # noqa: E731
            rows.append({"n": n, "depth": d, "states_per_class": len(counts[0][0]),
                         "qnn_2q_pure": ms(counts[0][0]), "qnn_2q_mixed": ms(counts[1][0]),
                         "qnn_depth_pure": ms(counts[0][1]), "qnn_depth_mixed": ms(counts[1][1]),
                         "shadow_2q_pure": ms(counts[0][2]), "shadow_2q_mixed": ms(counts[1][2]),
                         "physical_qubits": max(counts[0][3], counts[1][3]), "circuits": states * (1 + settings),
                         "shots": shots, "estimated_seconds": round(ibm.estimate_seconds(shots), 1)})
            r = rows[-1]
            print(f"  n={n} depth {d}: QNN 2q gates {pm(r['qnn_2q_pure'])} / {pm(r['qnn_2q_mixed'])} (pure / mixed), "
                  f"depth {pm(r['qnn_depth_pure'])} / {pm(r['qnn_depth_mixed'])}, shadow 2q "
                  f"{pm(r['shadow_2q_pure'])} / {pm(r['shadow_2q_mixed'])}, ~{r['estimated_seconds']} s", flush=True)
    return rows


def pm(v):
    return f"{v[0]:.1f} ± {v[1]:.1f}"


def plan_md(rows):
    return ["## Hardware plan (FakeMarrakesh transpile, nothing run)", "",
            "Mean ± std over random pure and mixed states (count per class in the JSON), transpiled (optimization level 1) for FakeMarrakesh. "
            "Estimated seconds: a stage-7-sized run of 80 states (60 test + 20 calibration) at 1024 copies "
            "each (512 QNN shots + 3^n shadow settings x floor(1024 / 3^n) shots), using the per-shot rate "
            "measured in this project's billed jobs (entangle/ibm.py).", "",
            report.md_table(["Qubits", "Depth", "QNN 2q gates (pure / mixed)", "QNN depth (pure / mixed)",
                             "Shadow 2q gates (pure / mixed)", "Physical qubits", "Circuits", "Shots",
                             "Est. quantum s"],
                            [[r["n"], r["depth"], f"{pm(r['qnn_2q_pure'])} / {pm(r['qnn_2q_mixed'])}",
                              f"{pm(r['qnn_depth_pure'])} / {pm(r['qnn_depth_mixed'])}",
                              f"{pm(r['shadow_2q_pure'])} / {pm(r['shadow_2q_mixed'])}", r["physical_qubits"],
                              r["circuits"], r["shots"], r["estimated_seconds"]] for r in rows]), ""]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--depths", nargs="+", default=["0", "1", "2", "4", "8", "haar"])
    p.add_argument("--copies", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--target", type=float, default=0.9)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--finetune", type=int, default=30)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--mid-depth", type=depth_key, default=2, help="depth of the by-qubits panel")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--plan-states", type=int, default=20, help="--plan: states per class to transpile")
    p.add_argument("--plan", action="store_true", help="only transpile the hardware circuits and estimate cost")
    args = p.parse_args()
    args.depths = [depth_key(d) for d in args.depths]
    t0 = time.time()
    if args.plan:
        rows = plan(args)
        report.save_json("stage8_hardware_plan.json", {"settings": vars(args), "plan": rows})
        report.save_text("stage8_hardware_plan.md", "\n".join(["# Stage 8 hardware plan", ""] + plan_md(rows)[2:]))
        print(f"Done in {time.time() - t0:.0f}s: results/stage8_hardware_plan.{{json,md}}")
        return
    tasks = sorted(product(range(args.seeds), args.qubits), key=lambda t: -t[1])   # big n first
    jobs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for j in pool.map(job, [(s, n, args) for s, n in tasks]):
            jobs.append(j)
            line = "; ".join(f"d={d}: " + "/".join(nm_copies(crossing(args.copies, r["by_copies"][m], args.target))
                                                   for m in METHODS) for d, r in j["depths"].items())
            print(f"  seed {j['seed']}, n={j['n']} ({j['seconds']:.0f}s; QNN/local/subsets copies) {line}",
                  flush=True)
    jobs.sort(key=lambda j: (j["n"], j["seed"]))
    res = summarise(jobs, args)
    write_outputs(res, jobs, args)
    print(f"Done in {time.time() - t0:.0f}s: results/stage8_shallow_purity.{{json,md,png}}")


if __name__ == "__main__":
    main()
