"""Barren-plateau check for the stage 5 purity models: gradient variance vs qubit count.

Stage 5 found that the 2-copy parity model with a separate block per qubit pair,
trained from random starts, sometimes fails at 5 qubits (0.85 +- 0.21), while the
same circuit with one block shared by all pairs does not. That was described as a
barren plateau. This script measures it directly instead of inferring it.

For each qubit count n and each model (per-pair blocks / one shared block) we draw
random weights uniformly in [-pi, pi) (the same distribution `fit` starts from)
and compute, exactly, the gradient of the class signal

    D(w) = mean P(class 1 | mixed states) - mean P(class 1 | pure states)

over a fixed batch of stage 5 states. A barren plateau shows up as Var_w[dD/dw_k]
shrinking exponentially with n. We report the variance of each weight's gradient
(averaged over weights), the variance of D itself, and the fraction of random
starts whose gradient norm is below 1e-3, over several data seeds, plus a fitted
log2 slope per added qubit.

Usage:
    python review_gradients.py                  # n = 2..7, 100 draws, 3 seeds (about 20 minutes)
    python review_gradients.py --qubits 2 3 4 --draws 50
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import two_copy_model
from entangle.qnn import Batch
from entangle.sim import CompiledCircuit
from entangle.states import make_purity_data

MODELS = {"per-pair blocks": False, "shared block": True}


def gradient_stats(n, shared, states, y, draws, rng):
    model = two_copy_model(n, "parity", shared=shared)
    sim = CompiledCircuit(model.circuit, model.occ)
    X = Batch(states, 2)
    sign = np.where(y == 1, 1.0 / np.sum(y == 1), -1.0 / np.sum(y == 0))   # D = sum_s sign_s p_s
    row_w = sign[X.owner]
    grads, signals = [], []
    for _ in range(draws):
        w = rng.uniform(-np.pi, np.pi, model.num_weights)
        p_rows, g_occ = sim.probs_and_grad(X.X, w[model.tie], model.readout, row_w)
        grads.append(np.bincount(model.tie, weights=g_occ, minlength=model.num_weights))
        signals.append(float(row_w @ p_rows))
    G = np.array(grads)
    norms = np.linalg.norm(G, axis=1)
    return {"weights": model.num_weights,
            "grad_var_mean": float(np.mean(np.var(G, axis=0))),
            "grad_var_first": float(np.var(G[:, 0])),
            "signal_var": float(np.var(signals)),
            "frac_small_grad": float(np.mean(norms < 1e-3)),
            "median_grad_norm": float(np.median(norms))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--draws", type=int, default=100, help="random weight vectors per (n, model, seed)")
    p.add_argument("--states", type=int, default=16, help="stage 5 states in the batch")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()
    t0 = time.time()
    res = {name: {str(n): [] for n in args.qubits} for name in MODELS}
    for seed in range(args.seeds):
        for n in args.qubits:
            data = make_purity_data(n, args.states, seed=900 + 10 * n + seed)
            for name, shared in MODELS.items():
                st = gradient_stats(n, shared, data.states, data.y, args.draws,
                                    np.random.default_rng(1000 * seed + n))
                res[name][str(n)].append(st)
                print(f"  seed {seed}, n={n}, {name:<16} Var[dD/dw] {st['grad_var_mean']:.2e}   "
                      f"Var[D] {st['signal_var']:.2e}   |grad|<1e-3: {st['frac_small_grad']:.2f}", flush=True)

    summary = {}
    for name in MODELS:
        ns = np.array(args.qubits)
        v = np.array([[s["grad_var_mean"] for s in res[name][str(n)]] for n in args.qubits])
        slope, _ = np.polyfit(ns, np.log2(v.mean(axis=1)), 1)
        summary[name] = {"log2_slope_per_qubit": float(slope),
                         "decay_factor_per_qubit": float(2 ** -slope)}
    report.save_json("review_gradients.json", {"settings": vars(args), "per_seed": res, "summary": summary})

    def ms(name, n, key):
        v = [s[key] for s in res[name][str(n)]]
        return f"{np.mean(v):.2e} ± {np.std(v):.1e}"
    rows = [[n, name, res[name][str(n)][0]["weights"], ms(name, n, "grad_var_mean"),
             ms(name, n, "signal_var"),
             f"{np.mean([s['frac_small_grad'] for s in res[name][str(n)]]):.2f}"]
            for n in args.qubits for name in MODELS]
    md = "\n".join([
        "# Gradient variance of the stage 5 purity models (barren-plateau check)", "",
        f"Signal D(w) = mean P(class 1) over mixed states minus over pure states, on {args.states} stage 5 "
        f"states. Weights drawn uniformly in [-pi, pi) ({args.draws} draws), exact gradients. Mean ± std "
        f"over {args.seeds} data seeds.", "",
        report.md_table(["Qubits", "Model", "Weights", "Var[dD/dw], mean over weights", "Var[D]",
                         "Fraction of starts with gradient norm < 1e-3"], rows), "",
        "Fitted decay of the gradient variance (log2-linear fit over n):", "",
        report.md_table(["Model", "log2 slope per qubit", "variance shrinks by a factor of, per qubit"],
                        [[k, f"{v['log2_slope_per_qubit']:.2f}", f"{v['decay_factor_per_qubit']:.2f}"]
                         for k, v in summary.items()]), ""])
    report.save_text("review_gradients.md", md)

    fig, axes = report.axes_grid(1, "Gradient variance at random initialisation, stage 5 parity model",
                                 width=5.6)
    ax = axes[0]
    for name, fam in (("per-pair blocks", "quantum_alt"), ("shared block", "quantum")):
        m = [np.mean([s["grad_var_mean"] for s in res[name][str(n)]]) for n in args.qubits]
        e = [np.std([s["grad_var_mean"] for s in res[name][str(n)]]) for n in args.qubits]
        report.line(ax, args.qubits, m, name, fam, err=e)
    ax.set_yscale("log", base=2)
    ax.set_xticks(args.qubits)
    ax.set_xlabel("qubits per copy (n)")
    ax.set_ylabel("Var[dD/dw], mean over weights")
    report.finish(fig, axes, "review_gradients.png", legend_ax=0)
    print(f"Done in {time.time() - t0:.0f}s. Slopes: " +
          ", ".join(f"{k} {v['log2_slope_per_qubit']:.2f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
