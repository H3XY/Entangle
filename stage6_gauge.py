"""Stage 6: a toy lattice gauge theory. Mass gap, confinement and phase classification.

Z2 lattice gauge theory in 2+1 dimensions on an L x L torus, one qubit per link
(see entangle/gauge.py). This is NOT Yang-Mills theory and nothing here bears
on the Millennium problem's proof; it is the simplest lattice gauge theory, used
to make the ideas concrete: Gauss's law, Wilson loops, confinement, and a gap
that can close at a phase transition. See docs/YANG_MILLS.md.

Part A (exact spectra, L = 2, 3, 4):
  - energy gap E1 - E0 of the physical sector vs coupling g
  - Wilson loops W(R x T) and the Creutz ratio chi = -ln[W(2x2) W(1x1) / W(1x2)^2],
    an estimate of the string tension: ~0 when deconfined, > 0 when confined
Part B (classification, L = 2, 8 link qubits):
  - deconfined (g < g_c) vs confined (g > g_c) ground states: QCNN and generic QNN
    vs classical models on the state vector or on Z/X-basis snapshots turned into
    gauge-invariant observables (plaquettes, 1x2 Wilson loops, electric fields)

Usage:
    python stage6_gauge.py                # about 2 minutes
    python stage6_gauge.py --skip-classify
"""
import argparse
import time

import numpy as np

from entangle import report
from entangle.circuits import hea_model, qcnn_model
from entangle.classical import classical_models, fit_predict, statevector_features
from entangle.gauge import (G_CRITICAL, dual_spectrum, electric_field, lattice, make_gauge_data,
                            wilson_loop)
from entangle.measure import X, Z, shadow_features, xz_snapshots
from entangle.qnn import QuantumClassifier

FAR = 0.1  # |g - g_c| beyond which a state counts as away from the transition


def spectra(args, log=print):
    gs = np.round(np.linspace(args.g_min, args.g_max, args.g_points), 4)
    out = {}
    for L in args.sizes:
        rows = []
        t0 = time.time()
        for g in gs:
            w, v = dual_spectrum(L, g, k=2)
            psi = v[:, 0]
            row = {"g": float(g), "E0": float(w[0]), "gap": float(w[1] - w[0]),
                   "electric_field": electric_field(psi, L), "W11": wilson_loop(psi, L, 1, 1)}
            if L >= 3:
                W12 = 0.5 * (wilson_loop(psi, L, 1, 2) + wilson_loop(psi, L, 2, 1))
                W22 = wilson_loop(psi, L, 2, 2)
                row.update(W12=W12, W22=W22)
                ratio = W22 * row["W11"] / W12 ** 2 if W12 > 0 and W22 > 0 else None
                row["creutz"] = float(-np.log(ratio)) if ratio else None
            rows.append(row)
        out[str(L)] = rows
        mins = min(rows, key=lambda r: r["gap"])
        log(f"  L={L} ({2 * L * L} link qubits): min gap {mins['gap']:.3f} at g={mins['g']:.3f}  "
            f"({time.time() - t0:.1f}s)")
    return out


def gauge_paulis(L):
    """Gauge-invariant observables measurable with all-Z / all-X snapshots."""
    plaq_links, _, _ = lattice(L)
    paulis = [tuple((l, Z) for l in links) for links in plaq_links]           # plaquettes B_p
    for y in range(L):                                                         # 1x2 and 2x1 Wilson loops
        for x in range(L):
            p = plaq_links[y * L + x]
            for q in (plaq_links[y * L + (x + 1) % L], plaq_links[((y + 1) % L) * L + x]):
                boundary = sorted(set(p) ^ set(q))
                if boundary:
                    paulis.append(tuple((l, Z) for l in boundary))
    n = 2 * L * L
    paulis += [((l, X),) for l in range(n)]                                    # electric fields
    paulis += [((a, X), (b, X)) for a in range(n) for b in range(a + 1, n)]    # their correlations
    return list(dict.fromkeys(paulis))


def classify(args, log=print):
    rng = np.random.default_rng(args.seed)
    train = make_gauge_data(args.train, seed=args.seed)
    test = make_gauge_data(args.test, seed=args.seed + 1)
    far = np.abs(test.g - G_CRITICAL) >= FAR
    score = lambda pred: {"all": float(np.mean(pred == test.y)),  # noqa: E731
                          "away_from_critical": float(np.mean((pred == test.y)[far]))}
    res = {"qubits": train.n, "quantum": {}, "classical": {}}
    for build in (qcnn_model, hea_model):
        model = build(train.n)
        clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=args.maxiter,
                                           restarts=args.restarts, seed=args.seed)
        res["quantum"][model.name] = {"weights": model.num_weights,
                                      "exact": score(clf.predict(test.states)),
                                      "shots": score(clf.predict(test.states, shots=args.copies, rng=rng))}
        log(f"  {model.name:<21} exact {res['quantum'][model.name]['exact']['all']:.3f}   "
            f"{args.copies} shots {res['quantum'][model.name]['shots']['all']:.3f}")
    best = lambda Xtr, Xte: {k: score(fit_predict(m, Xtr, train.y, Xte))  # noqa: E731
                             for k, m in classical_models(args.seed).items()}
    res["classical"]["full state vector"] = best(statevector_features(train.states),
                                                 statevector_features(test.states))
    paulis = gauge_paulis(2)
    feats = lambda d: shadow_features(xz_snapshots(d.states, d.n, args.copies, rng), d.n, paulis,  # noqa: E731
                                      purities=False)
    res["classical"]["Z/X snapshots"] = best(feats(train), feats(test))
    for k, v in res["classical"].items():
        log(f"  classical, {k}: " + ", ".join(f"{m} {s['all']:.3f}" for m, s in v.items()))
    return res


def write_outputs(spec, cls, args):
    report.save_json("stage6_gauge.json", {"settings": vars(args), "g_critical": G_CRITICAL,
                                           "spectra": spec, "classification": cls})
    lines = ["# Stage 6: Z2 lattice gauge theory (a toy, not Yang-Mills)", "",
             f"Infinite-lattice transition at g_c = {G_CRITICAL:.4f} (dual 2D quantum Ising model). "
             "Physical sector: Gauss's law at every vertex, no electric flux winding the torus.", "",
             "## Energy gap and string tension", ""]
    pick = [g for g in (0.1, 0.2, 0.3, 0.33, 0.4, 0.6, 1.0, 1.5)]
    rows = []
    for L, data in spec.items():
        for g in pick:
            r = min(data, key=lambda r: abs(r["g"] - g))
            rows.append([L, f"{r['g']:.3f}", f"{r['gap']:.3f}", f"{r['W11']:.3f}",
                         "–" if r.get("creutz") is None else f"{r['creutz']:.3f}",
                         f"{r['electric_field']:.3f}"])
    lines += [report.md_table(["L", "g", "gap E1 - E0", "W(1x1)", "Creutz ratio", "electric field (mean X)"], rows), ""]
    lines += ["Minimum gap per lattice size:", ""]
    lines += [report.md_table(["L", "link qubits", "min gap", "at g"],
                              [[L, 2 * int(L) ** 2, f"{m['gap']:.3f}", f"{m['g']:.3f}"]
                               for L, d in spec.items() for m in [min(d, key=lambda r: r['gap'])]]), ""]
    if cls:
        lines += ["## Classifying deconfined vs confined ground states (L = 2, 8 link qubits)", ""]
        rows = [[k, report.fmt(v["exact"]["all"]), report.fmt(v["exact"]["away_from_critical"]),
                 report.fmt(v["shots"]["all"])] for k, v in cls["quantum"].items()]
        for kind, v in cls["classical"].items():
            for m, s in v.items():
                rows.append([f"{m}, {kind}", report.fmt(s["all"]), report.fmt(s["away_from_critical"]),
                             report.fmt(s["all"]) if kind != "full state vector" else "n/a"])
        lines += [report.md_table(["Model", "Test acc. (exact)", f"Exact, g at least {FAR} from g_c",
                                   f"Test acc. ({args.copies} copies)"], rows), "",
                  "For the snapshot models the exact and copies columns are the same run "
                  f"({args.copies} snapshots per state)."]
    report.save_text("stage6_gauge.md", "\n".join(lines) + "\n")

    fig, axes = report.axes_grid(3, "Stage 6: Z2 lattice gauge theory on an L x L torus", width=4.6)
    # lattice size is ordered, so one hue from light (small L) to dark (large L)
    shades = dict(zip(spec, ["#9ec5f0", "#4a90e2", "#1c5aa8", "#0d3466"]))
    for L, data in spec.items():
        axes[0].plot([r["g"] for r in data], [r["gap"] for r in data], color=shades[L], linewidth=2,
                     label=f"L = {L}")
        creutz = [(r["g"], r["creutz"]) for r in data if r.get("creutz") is not None]
        if creutz:
            axes[2].plot([c[0] for c in creutz], [c[1] for c in creutz], color=shades[L], linewidth=2,
                         label=f"L = {L}")
    for line_ax in (axes[0], axes[2]):
        line_ax.axvline(G_CRITICAL, color=report.INK2, linewidth=1, linestyle=":")
        line_ax.text(G_CRITICAL, 0.98, " g_c", transform=line_ax.get_xaxis_transform(), color=report.INK2,
                     fontsize=8, va="top")
        line_ax.legend(frameon=False, fontsize=8, labelcolor=report.INK2,
                       loc="upper center" if line_ax is axes[0] else "upper left")
        line_ax.set_xlabel("coupling g")
    axes[0].set_ylabel("gap E1 - E0")
    axes[0].set_title("Energy gap closes only near g_c", fontsize=10, loc="left")
    axes[2].set_ylabel("Creutz ratio (string tension)")
    axes[2].set_title("Confinement switches on past g_c", fontsize=10, loc="left")
    ax = axes[1]
    Ls = [int(L) for L in spec]
    for g, fam, label in ((0.1, "classical_state", "g = 0.1 (deconfined)"),
                          (G_CRITICAL, "quantum_alt", "g = g_c"),
                          (1.0, "classical_snap", "g = 1.0 (confined)")):
        gaps = [min(spec[str(L)], key=lambda r: abs(r["g"] - g))["gap"] for L in Ls]
        report.line(ax, Ls, gaps, label, fam)
    ax.set_xticks(Ls)
    ax.set_xlabel("lattice size L")
    ax.set_ylabel("gap E1 - E0")
    ax.set_ylim(bottom=0)
    ax.set_title("Gap vs lattice size", fontsize=10, loc="left")
    report.finish(fig, axes, "stage6_gauge.png", legend_ax=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4], help="lattice sizes L (L <= 4)")
    p.add_argument("--g-min", type=float, default=0.02)
    p.add_argument("--g-max", type=float, default=1.5)
    p.add_argument("--g-points", type=int, default=75)
    p.add_argument("--skip-classify", action="store_true")
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--copies", type=int, default=256)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    print("Part A: spectra")
    spec = spectra(args)
    cls = None
    if not args.skip_classify:
        print("\nPart B: classifying deconfined vs confined (L = 2, 8 link qubits)")
        cls = classify(args)
    write_outputs(spec, cls, args)
    print(f"\nDone in {time.time() - t0:.0f}s. Results: results/stage6_gauge.{{json,md,png}}")


if __name__ == "__main__":
    main()
