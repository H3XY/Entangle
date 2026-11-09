"""Stage 7 hardware purity test: recalibration, copy accounting and intervals, from saved counts.

No device is used: everything is recomputed from results/ibm_raw (the counts IBM returned,
and the same ISA circuits replayed on FakeMarrakesh; see fetch_ibm_results.py).

Data. As in stage7_hardware_purity.py (seed 0): 200 training states (seed 0, simulation only),
60 test states (seed 1) and 20 calibration states (seed 2), each balanced pure / mixed, drawn
independently (Haar-random pure states; rank-2 mixed states). The calibration and test states
are prepared on the device in the same job; the training states never are. We check that no
calibration state equals a training or test state.

Copy accounting (the same for both methods). "c copies" = c state preparations of the test state:
  - 2-copy QNN: c/2 shots, each shot uses two fresh preparations (one per copy register)
  - classical shadows: floor(c/9) shots in each of the 9 X/Y/Z bases (9 * floor(c/9) <= c)
The device ran 512 QNN shots (1,024 preparations) and 113 shots x 9 bases (1,017) per state.
Smaller c are drawn WITHOUT replacement from those device shots (multivariate hypergeometric).

Methods and what recalibration refits (on the 20 labelled device calibration states):
  QNN 2-copy               1 feature p = P(odd number of pairs read 11); logistic regression, 2 parameters
  Shadows, global purity   1 feature: shadow estimate of Tr rho^2; logistic regression, 2 parameters
                           (same flexibility as the QNN)
  Shadows, 3 features      global + 2 single-qubit purity estimates; logistic regression, 4 parameters
                           (the stage 7 headline curve)
"Before" = the same model trained in simulation (QNN: exact training + its 2-parameter calibration;
shadows: the logistic regression trained on simulated snapshots with the same design and c),
applied to device data with nothing refitted.

Uncertainty, per run and copy count c:
  - shots: R subsamples of the device shots
  - test states: bootstrap over the 60 test states (percentile 95% interval)
  - choice of calibration states: (i) bootstrap the 20 calibration states; (ii) re-split the 80
    device states (60 test + 20 calibration) at random into 20 for calibration and 60 for testing
  - copies for 90%: log2-linear interpolation of the mean curve where it crosses 0.9, with a
    bootstrap interval (test states and subsamples resampled together), and the shadows/QNN ratio.

Usage:
    python review_purity_hw.py                 # ~5 minutes
"""
import argparse
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from entangle import ibm, report
from entangle.circuits import two_copy_model
from entangle.measure import fixed_settings_snapshots, purity_features, snapshots_from_counts
from entangle.qnn import QuantumClassifier
from entangle.states import make_purity_data
from stage7_hardware_purity import N, SETTINGS, p_from_counts, subsample

COPIES = [16, 32, 64, 128, 256, 512, 1024]
METHODS = ["QNN 2-copy", "Shadows, global purity (2 params)", "Shadows, 3 features (4 params)"]


def lr():
    return LogisticRegression(C=1e4, max_iter=5000)


def fit_ok(y):
    return len(np.unique(y)) == 2


def load_counts(source, label):
    ibm.set_replay(source)
    rec, counts, _ = ibm.REPLAY[label]
    return rec, counts


def features(cq, cs, c, rng, readout):
    """Per state: QNN p, shadow global purity, shadow [global, local, local] from c copies."""
    k = c // len(SETTINGS)
    pq = np.array([p_from_counts(subsample(x, c // 2, rng), readout) for x in cq])
    if k * len(SETTINGS) < 18:  # the global purity estimator needs at least 2 snapshots
        return pq, None
    snaps = [snapshots_from_counts(list(zip(SETTINGS, [subsample(x, k, rng) for x in cs[i * 9:(i + 1) * 9]])), N)
             for i in range(len(cq))]
    return pq, purity_features(snaps, N)


def crossing(copies, acc, target=0.9):
    """Copies where the curve first reaches target, by log2-linear interpolation (nan if never)."""
    acc = np.asarray(acc, dtype=float)
    for i in range(len(copies)):
        if acc[i] >= target:
            if i == 0:
                return float(copies[0])
            x0, x1 = np.log2(copies[i - 1]), np.log2(copies[i])
            if np.isnan(acc[i - 1]):
                return float(copies[i])
            t = (target - acc[i - 1]) / (acc[i] - acc[i - 1])
            return float(2 ** (x0 + t * (x1 - x0)))
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resamples", type=int, default=30, help="shot subsamples per copy count")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap replicates over test states")
    p.add_argument("--cal-reps", type=int, default=200, help="replicates for calibration-state uncertainty")
    args = p.parse_args()
    t0 = time.time()
    seed = 0
    train = make_purity_data(N, 200, seed=seed)
    test = make_purity_data(N, 60, seed=seed + 1)
    cal = make_purity_data(N, 20, seed=seed + 2)
    rho = lambda A: A @ A.conj().T  # noqa: E731
    dists = [np.linalg.norm(rho(a) - rho(b)) for a in cal.states for b in np.concatenate([train.states, test.states])]
    model = two_copy_model(N, "parity", shared=True)
    clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=150, restarts=2, seed=seed)
    y_te, y_cal = test.y, cal.y
    y_all = np.concatenate([y_te, y_cal])
    nt = len(y_te)
    rng = np.random.default_rng(123)

    # simulation-trained shadow classifiers for each copy count ("before" recalibration)
    sim = {}
    for c in COPIES:
        k = c // len(SETTINGS)
        if k * len(SETTINGS) < 18:
            continue
        Xtr = purity_features(fixed_settings_snapshots(train.states, N, SETTINGS, k, rng), N)
        sim[c] = (lr().fit(Xtr[:, :1], train.y), lr().fit(Xtr, train.y))

    out = {"settings": vars(args), "copies": COPIES, "methods": METHODS,
           "data": {"train": "make_purity_data(2, 200, seed=0)", "test": "make_purity_data(2, 60, seed=1)",
                    "calibration": "make_purity_data(2, 20, seed=2)",
                    "min_distance_calibration_to_train_or_test": float(min(dists))},
           "copy_accounting": {"qnn_shots_per_copy_count": "c/2 shots, 2 preparations each",
                               "shadow_shots_per_copy_count": "floor(c/9) shots in each of 9 bases",
                               "device_preparations_per_state": {"QNN": 1024, "shadows": 1017},
                               "subsampling": "without replacement from the device shots"},
           "runs": {}}

    for source, src_label in (("ibm", "ibm_marrakesh"), ("FakeMarrakesh", "FakeMarrakesh (identical circuits)")):
        for run in ("run 1", "run 2"):
            rec, counts = load_counts(source, f"stage7 {run}")
            cq, cs = counts[:80], counts[80:]
            key = f"{src_label}, {run}"
            res = {m: {"before": [], "after": [], "after_ci": [], "before_ci": [], "cal_boot": [],
                       "cal_resplit": []} for m in METHODS}
            draws = {m: [] for m in METHODS}   # per c: (R, nt) correct-prediction matrices after recalibration
            for c in COPIES:
                correct_after = {m: [] for m in METHODS}
                correct_before = {m: [] for m in METHODS}
                for _ in range(args.resamples):
                    pq, X = features(cq, cs, c, rng, model.readout)
                    feats = {METHODS[0]: pq[:, None]}
                    if X is not None:
                        feats[METHODS[1]], feats[METHODS[2]] = X[:, :1], X
                    for m, F in feats.items():
                        after = lr().fit(F[nt:], y_cal).predict(F[:nt])
                        if m == METHODS[0]:
                            before = clf.predict_from_p(pq[:nt])
                        else:
                            before = sim[c][0 if m == METHODS[1] else 1].predict(F[:nt])
                        correct_after[m].append(after == y_te)
                        correct_before[m].append(before == y_te)
                for m in METHODS:
                    if not correct_after[m]:
                        for k in ("before", "after", "before_ci", "after_ci"):
                            res[m][k].append(None)
                        draws[m].append(None)
                        continue
                    A, B = np.array(correct_after[m], float), np.array(correct_before[m], float)
                    draws[m].append(A)
                    idx = rng.integers(0, nt, (args.boot, nt))
                    res[m]["after"].append(float(A.mean()))
                    res[m]["before"].append(float(B.mean()))
                    res[m]["after_ci"].append(np.percentile(A.mean(0)[idx].mean(1), [2.5, 97.5]).tolist())
                    res[m]["before_ci"].append(np.percentile(B.mean(0)[idx].mean(1), [2.5, 97.5]).tolist())
                # calibration-state uncertainty (fresh subsample per replicate)
                boot, resplit = {m: [] for m in METHODS}, {m: [] for m in METHODS}
                for _ in range(args.cal_reps):
                    pq, X = features(cq, cs, c, rng, model.readout)
                    feats = {METHODS[0]: pq[:, None]}
                    if X is not None:
                        feats[METHODS[1]], feats[METHODS[2]] = X[:, :1], X
                    bi = nt + rng.integers(0, len(y_cal), len(y_cal))
                    perm = rng.permutation(len(y_all))
                    ci, ti = perm[:20], perm[20:]
                    for m, F in feats.items():
                        if fit_ok(y_all[bi]):
                            boot[m].append(np.mean(lr().fit(F[bi], y_all[bi]).predict(F[:nt]) == y_te))
                        if fit_ok(y_all[ci]):
                            resplit[m].append(np.mean(lr().fit(F[ci], y_all[ci]).predict(F[ti]) == y_all[ti]))
                for m in METHODS:
                    res[m]["cal_boot"].append([float(np.mean(boot[m]))] + np.percentile(boot[m], [2.5, 97.5]).tolist()
                                              if boot[m] else None)
                    res[m]["cal_resplit"].append([float(np.mean(resplit[m]))] +
                                                 np.percentile(resplit[m], [2.5, 97.5]).tolist() if resplit[m] else None)
            # copies for 90%, with bootstrap over test states and subsamples
            cross = {}
            for m in METHODS:
                curve = [np.nan if a is None else a for a in res[m]["after"]]
                reps = []
                for _ in range(args.boot):
                    ti = rng.integers(0, nt, nt)
                    curve_b = [np.nan if d is None else d[rng.integers(0, len(d), len(d))][:, ti].mean() for d in draws[m]]
                    reps.append(crossing(COPIES, curve_b))
                reps = np.array(reps)
                cross[m] = {"point": crossing(COPIES, curve), "reps": reps}
            q = cross[METHODS[0]]["reps"]
            ratios = {}
            for m in METHODS[1:]:
                r = cross[m]["reps"] / q
                ok = np.isfinite(r)
                ratios[m] = {"point": cross[m]["point"] / cross[METHODS[0]]["point"],
                             "ci": np.percentile(r[ok], [2.5, 97.5]).tolist() if ok.any() else None}
            res["copies_for_90"] = {m: {"point": cross[m]["point"],
                                        "ci": np.percentile(cross[m]["reps"][np.isfinite(cross[m]["reps"])],
                                                            [2.5, 97.5]).tolist()} for m in METHODS}
            res["ratio_to_qnn"] = ratios
            out["runs"][key] = res
            print(f"  {key}: copies for 90% " + ", ".join(
                f"{m.split(',')[0] if m == METHODS[0] else m} {v['point']:.0f} [{v['ci'][0]:.0f}, {v['ci'][1]:.0f}]"
                for m, v in res["copies_for_90"].items()), flush=True)

    report.save_json("review_purity_hw.json", out)
    write_md(out)
    print(f"Done in {time.time() - t0:.0f}s: results/review_purity_hw.{{json,md}}")


def write_md(out):
    C = out["copies"]
    f = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
    fci = lambda v, ci: "–" if v is None else f"{v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"  # noqa: E731
    lines = ["# Stage 7 hardware purity test: recalibration, copy accounting, intervals", "",
             "Recomputed offline from the saved device counts (results/ibm_raw). FakeMarrakesh rows replay the "
             "identical transpiled circuits (same physical qubits 4, 6, 5 / 8, 17, 7) with the same shots.", "",
             "## Data and copy accounting", "",
             f"- Training: {out['data']['train']} (simulation only). Test: {out['data']['test']}. Calibration: "
             f"{out['data']['calibration']}. All balanced pure / mixed, drawn independently; smallest distance "
             f"||rho_cal - rho_other|| between a calibration state and any training or test state: "
             f"{out['data']['min_distance_calibration_to_train_or_test']:.3f} (so none coincide).",
             "- c copies = c preparations of the state for both methods: the QNN uses c/2 shots of a circuit "
             "holding 2 copies; shadows use floor(c/9) shots in each of 9 bases. On the device: 512 QNN shots "
             "(1,024 preparations) and 9 x 113 shadow shots (1,017) per state. Smaller c are subsampled "
             "without replacement from those shots.",
             "- Recalibration refits, on the 20 device calibration states: QNN, a logistic regression on p "
             "(2 parameters); shadows, global purity, the same (2 parameters); shadows, 3 features, a logistic "
             "regression on [global, local, local] purity estimates (4 parameters). 'Before' = the model "
             "trained in simulation, nothing refitted.", ""]
    for key, res in out["runs"].items():
        lines += [f"## {key}", "",
                  f"Mean over {out['settings']['resamples']} shot subsamples; brackets: 95% bootstrap interval "
                  "over the 60 test states.", "",
                  report.md_table(["Method", "Recalibration"] + [str(c) for c in C],
                                  [[m, "before"] + [fci(v, ci) if v is not None else "–"
                                                    for v, ci in zip(res[m]["before"], res[m]["before_ci"])]
                                   for m in METHODS] +
                                  [[m, "after"] + [fci(v, ci) if v is not None else "–"
                                                   for v, ci in zip(res[m]["after"], res[m]["after_ci"])]
                                   for m in METHODS]), "",
                  "Uncertainty over the choice of calibration states (mean [2.5%, 97.5%]): bootstrap of the 20 "
                  "calibration states / random 20-60 re-split of all 80 device states:", "",
                  report.md_table(["Method", "Scheme"] + [str(c) for c in C],
                                  [[m, s] + [("–" if v is None else f"{v[0]:.3f} [{v[1]:.2f}, {v[2]:.2f}]")
                                             for v in res[m][k]]
                                   for m in METHODS for s, k in (("bootstrap 20", "cal_boot"),
                                                                 ("re-split 80", "cal_resplit"))]), "",
                  "Copies for 90% (log2-linear interpolation of the 'after' curve; 95% bootstrap interval) and "
                  "ratio to the QNN:", "",
                  report.md_table(["Method", "Copies for 90%", "Ratio to QNN"],
                                  [[m, f"{v['point']:.0f} [{v['ci'][0]:.0f}, {v['ci'][1]:.0f}]",
                                    "1" if m == METHODS[0] else
                                    (f"{res['ratio_to_qnn'][m]['point']:.2f} [{res['ratio_to_qnn'][m]['ci'][0]:.2f}, "
                                     f"{res['ratio_to_qnn'][m]['ci'][1]:.2f}]")]
                                   for m, v in res["copies_for_90"].items()]), ""]
    report.save_text("review_purity_hw.md", "\n".join(lines))


if __name__ == "__main__":
    main()
