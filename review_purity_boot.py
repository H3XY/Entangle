"""Stage 7 copies-for-90%: hierarchical bootstrap per job, and the matching noise-free ratio.

review_purity_hw.py gives intervals from a bootstrap over test states with 30 overlapping shot
subsamples, and keeps the 20-state recalibration fixed. This script propagates every source of
sampling uncertainty instead. Each device job (run 1, run 2) is treated on its own, since the
two jobs are the only independent executions. In every replicate:

  1. resample the 60 test states and the 20 calibration states with replacement
     (calibration resamples must contain both classes)
  2. for every selected state, resample its raw device shots with replacement
     (512 QNN shots; 113 shots in each of the 9 shadow bases)
  3. for each copy count c, subsample WITHOUT replacement from those resampled shots
     (QNN: c/2 shots; shadows: floor(c/9) per basis), for test and calibration states alike
  4. refit the 2-parameter recalibration (logistic regression on the QNN output p, or on the
     shadow global-purity estimate) on the resampled calibration states, score the test states
  5. read off copies-for-90% by log2-linear interpolation of the accuracy curve, for the QNN and
     for shadows (2 parameters), and their ratio

Calibration shot budget: the primary analysis uses the same c for the calibration states as for
the test states (as in stage 7 and review_purity_hw.py). A sensitivity variant calibrates on the
full device budget (1,024 copies) of each calibration state.

Noise-free reference, same rule and same design: the identical replicate procedure with shots
drawn from the exact outcome distributions of the same 60 test and 20 calibration states
(QNN: binomial in P(class 1); shadows: the same 9 fixed bases), i.e. stage 7 without a device.
Also reported: the stage 5 simulation at n = 2 (different test states, 3-feature best-of-3
shadows) put through the same interpolation instead of the power-of-2 grid.

Usage:
    python review_purity_boot.py                  # 1,000 replicates per job (~15 minutes)
"""
import argparse
import json
import time

import numpy as np

from entangle import report
from entangle.circuits import two_copy_model
from entangle.measure import basis_probs, purity_features, snapshots_from_counts
from entangle.qnn import QuantumClassifier
from entangle.states import make_purity_data
from review_purity_hw import crossing, load_counts, lr
from stage7_hardware_purity import N, SETTINGS

COPIES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]   # 4 and 8 so the QNN threshold is not clipped at 16
FULL = 1024


class Shots:
    """Device or simulated shots of one state: QNN outcomes and 9 per-basis outcome histograms."""

    def __init__(self, q_keys, q_counts, s_keys, s_counts):
        self.q_keys, self.q_counts = q_keys, np.asarray(q_counts)
        self.s_keys, self.s_counts = s_keys, [np.asarray(c) for c in s_counts]

    @classmethod
    def from_device(cls, cq, cs_list):
        qk = list(cq)
        sk = [list(c) for c in cs_list]
        return cls(qk, [cq[k] for k in qk], sk, [[c[k] for k in keys] for c, keys in zip(cs_list, sk)])

    def resample(self, rng):
        """Bootstrap the raw shots: same totals, drawn with replacement."""
        q = rng.multinomial(self.q_counts.sum(), self.q_counts / self.q_counts.sum())
        s = [rng.multinomial(c.sum(), c / c.sum()) for c in self.s_counts]
        return Shots(self.q_keys, q, self.s_keys, s)

    def features(self, c, rng, readout):
        """(QNN p, shadow global purity) from c copies, subsampled without replacement."""
        q = rng.multivariate_hypergeometric(self.q_counts, c // 2)
        p = sum(n for k, n in zip(self.q_keys, q) if readout[int(k, 2)]) / (c // 2)
        k = c // len(SETTINGS)
        if k * len(SETTINGS) < 18:
            return p, np.nan
        recs = []
        for setting, keys, counts in zip(SETTINGS, self.s_keys, self.s_counts):
            draw = rng.multivariate_hypergeometric(counts, k)
            recs.append((setting, {b: int(n) for b, n in zip(keys, draw) if n}))
        return p, float(purity_features([snapshots_from_counts(recs, N)], N)[0, 0])


def simulated_shots(A, p_exact, rng):
    """Noise-free shots of one state with the stage 7 design and budget (512 QNN, 9 x 113 shadow)."""
    q1 = rng.binomial(512, p_exact)
    s_keys, s_counts = [], []
    for setting in SETTINGS:
        probs = basis_probs(A, N, setting)
        draw = rng.multinomial(113, probs / probs.sum())
        s_keys.append([format(i, f"0{N}b") for i in range(2 ** N)])
        s_counts.append(draw)
    return q1, s_keys, s_counts


def replicate(test_shots, cal_shots, y_te, y_cal, readout, rng, resample_shots=True, cal_full=False):
    nt, nc = len(test_shots), len(cal_shots)
    ti = rng.integers(0, nt, nt)
    while True:
        ci = rng.integers(0, nc, nc)
        if len(np.unique(y_cal[ci])) == 2:
            break
    te = [test_shots[i].resample(rng) if resample_shots else test_shots[i] for i in ti]
    ca = [cal_shots[i].resample(rng) if resample_shots else cal_shots[i] for i in ci]
    yt, yc = y_te[ti], y_cal[ci]
    if cal_full:
        Fc_full = np.array([s.features(FULL, rng, readout) for s in ca])
    curves = {"QNN": [], "Shadows": []}
    for c in COPIES:
        Ft = np.array([s.features(c, rng, readout) for s in te])
        Fc = Fc_full if cal_full else np.array([s.features(c, rng, readout) for s in ca])
        for j, m in enumerate(("QNN", "Shadows")):
            if np.isnan(Ft[0, j]):
                curves[m].append(np.nan)
                continue
            curves[m].append(float(np.mean(lr().fit(Fc[:, j:j + 1], yc).predict(Ft[:, j:j + 1]) == yt)))
    tq, ts = crossing(COPIES, curves["QNN"]), crossing(COPIES, curves["Shadows"])
    return tq, ts, curves


def summarise(reps, sim_ratio=None):
    tq = np.array([r[0] for r in reps], float)
    ts = np.array([r[1] for r in reps], float)
    tq_i = np.where(np.isnan(tq), np.inf, tq)      # never reached 90% within 1,024 copies
    ts_i = np.where(np.isnan(ts), np.inf, ts)
    ratio = ts_i / tq_i
    out = {"replicates": len(reps),
           "qnn_threshold": {"median": float(np.median(tq_i)), "ci95": np.percentile(tq_i, [2.5, 97.5]).tolist(),
                             "never_reached": int(np.isnan(tq).sum())},
           "shadow_threshold": {"median": float(np.median(ts_i)), "ci95": np.percentile(ts_i, [2.5, 97.5]).tolist(),
                                "never_reached": int(np.isnan(ts).sum())},
           "ratio": {"median": float(np.nanmedian(ratio)), "ci95": np.nanpercentile(ratio, [2.5, 97.5]).tolist(),
                     "one_sided_upper95": float(np.nanpercentile(ratio, 95)),
                     "p_ratio_below_1": float(np.nanmean(ratio < 1))}}
    if sim_ratio is not None:
        out["ratio"]["p_ratio_below_sim"] = float(np.nanmean(ratio < sim_ratio))
        out["ratio"]["sim_ratio"] = sim_ratio
    return out, ratio


def stage5_interpolated(n=2):
    """Stage 5 simulation at n qubits: same interpolation rule on the mean curves."""
    d = json.load(open(report.path("stage5_purity.json")))
    C = d["settings"]["copies"]
    res = d["results"]["by_copies"]
    q = [res["QNN 2-copy, shared, transferred from 2 qubits"][str(n)][str(c)]["mean"] for c in C]
    s = [res["Classical shadows (purity estimates)"][str(n)][str(c)]["mean"] for c in C]
    tq, ts = crossing(C, q), crossing(C, s)
    return {"copies": C, "qnn_threshold": tq, "shadow_threshold": ts, "ratio": ts / tq}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--sens-reps", type=int, default=300, help="replicates for the full-budget calibration variant")
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(2026)
    train = make_purity_data(N, 200, seed=0)
    test = make_purity_data(N, 60, seed=1)
    cal = make_purity_data(N, 20, seed=2)
    model = two_copy_model(N, "parity", shared=True)
    clf = QuantumClassifier(model).fit(train.states, train.y, maxiter=150, restarts=2, seed=0)
    readout = model.readout
    out = {"settings": vars(args), "copies": COPIES,
           "calibration_budget": "primary: same c as the test states; sensitivity: full 1,024 copies",
           "stage5_interpolated_n2": stage5_interpolated()}

    # noise-free reference with the stage 7 design, states and budget
    A_all = np.concatenate([test.states, cal.states])
    p_exact = clf.raw_proba(A_all)
    sim_reps = []
    for r in range(args.reps):
        shots = []
        for A, pe in zip(A_all, p_exact):
            q1, sk, sc = simulated_shots(A, pe, rng)
            shots.append(Shots(["1", "0"], [q1, 512 - q1], sk, sc))
        sim_readout = np.array([False, True])     # key "1" = class 1 (p is all we need)
        sim_reps.append(replicate(shots[:60], shots[60:], test.y, cal.y, sim_readout, rng, resample_shots=False))
        if r % 200 == 0:
            print(f"  simulation replicate {r}", flush=True)
    sim_summary, sim_ratios = summarise(sim_reps)
    sim_ratio = sim_summary["ratio"]["median"]
    out["noise_free_same_design"] = sim_summary
    print(f"  noise-free: QNN {sim_summary['qnn_threshold']['median']:.0f}, shadows "
          f"{sim_summary['shadow_threshold']['median']:.0f}, ratio {sim_ratio:.2f} "
          f"{sim_summary['ratio']['ci95']}", flush=True)

    out["jobs"] = {}
    for run in ("run 1", "run 2"):
        rec, counts = load_counts("ibm", f"stage7 {run}")
        cq, cs = counts[:80], counts[80:]
        shots = [Shots.from_device(cq[i], cs[i * 9:(i + 1) * 9]) for i in range(80)]
        reps = []
        for r in range(args.reps):
            reps.append(replicate(shots[:60], shots[60:], test.y, cal.y, readout, rng))
        summ, dev_ratios = summarise(reps, sim_ratio)
        # both ratios uncertain: compare every device replicate with every noise-free replicate
        d, s_ = dev_ratios[np.isfinite(dev_ratios)], sim_ratios[np.isfinite(sim_ratios)]
        summ["ratio"]["p_below_sim_distribution"] = float(np.mean(d[:, None] < s_[None, :]))
        diff = np.log2(rng.choice(d, 20000)) - np.log2(rng.choice(s_, 20000))
        summ["ratio"]["log2_ratio_minus_sim_ci95"] = np.percentile(diff, [2.5, 97.5]).tolist()
        sens = [replicate(shots[:60], shots[60:], test.y, cal.y, readout, rng, cal_full=True)
                for _ in range(args.sens_reps)]
        summ["calibration_full_budget"], _ = summarise(sens, sim_ratio)
        summ["job_id"] = rec["job_id"]
        out["jobs"][run] = summ
        print(f"  {run} ({rec['job_id']}): QNN {summ['qnn_threshold']['median']:.0f} {summ['qnn_threshold']['ci95']}, "
              f"shadows {summ['shadow_threshold']['median']:.0f} {summ['shadow_threshold']['ci95']}, ratio "
              f"{summ['ratio']['median']:.2f} {summ['ratio']['ci95']}, P(ratio < sim) "
              f"{summ['ratio']['p_ratio_below_sim']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    report.save_json("review_purity_boot.json", out)
    write_md(out)
    print(f"Done in {time.time() - t0:.0f}s: results/review_purity_boot.{{json,md}}")


def write_md(out):
    ci = lambda s: f"{s['median']:.0f} [{s['ci95'][0]:.0f}, {s['ci95'][1]:.0f}]"  # noqa: E731
    rci = lambda s: f"{s['median']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]"  # noqa: E731
    sim = out["noise_free_same_design"]
    rows = [["Noise-free, same states and design", ci(sim["qnn_threshold"]), ci(sim["shadow_threshold"]),
             rci(sim["ratio"]), "–", "–", "–", "–"]]
    for run, s in out["jobs"].items():
        rows.append([f"ibm_marrakesh {run} ({s['job_id']})", ci(s["qnn_threshold"]), ci(s["shadow_threshold"]),
                     rci(s["ratio"]), f"{s['ratio']['one_sided_upper95']:.2f}",
                     f"{s['ratio']['p_ratio_below_sim']:.3f}", f"{s['ratio']['p_below_sim_distribution']:.3f}",
                     "[{:.2f}, {:.2f}]".format(*s["ratio"]["log2_ratio_minus_sim_ci95"])])
    sens_rows = [[f"ibm_marrakesh {run}", ci(s["calibration_full_budget"]["qnn_threshold"]),
                  ci(s["calibration_full_budget"]["shadow_threshold"]), rci(s["calibration_full_budget"]["ratio"]),
                  f"{s['calibration_full_budget']['ratio']['p_ratio_below_sim']:.3f}"]
                 for run, s in out["jobs"].items()]
    st5 = out["stage5_interpolated_n2"]
    md = "\n".join([
        "# Stage 7 copies-for-90%: hierarchical bootstrap per job", "",
        f"{out['settings']['reps']} replicates per job. Each replicate resamples the 60 test and 20 calibration "
        "states, resamples each state's raw device shots with replacement, subsamples c copies without "
        "replacement, refits the 2-parameter recalibration on the resampled calibration states, and "
        "interpolates the 90% threshold (log2-linear). Medians with 95% percentile intervals. Shadows = "
        "2-parameter readout on the shadow global-purity estimate (same flexibility as the QNN).", "",
        "Calibration shot budget (primary): the calibration states use the same c as the test states.", "",
        report.md_table(["Source", "QNN copies for 90%", "Shadow copies for 90%", "Ratio shadows / QNN",
                         "One-sided 95% upper bound on ratio", "P(ratio < noise-free median)",
                         "P(ratio < noise-free ratio), both uncertain", "95% CI of log2(device / noise-free ratio)"],
                        rows), "",
        f"Noise-free ratio used for the comparison: {sim['ratio']['median']:.2f} (median of the noise-free "
        "replicates; same states, design, budget, recalibration and interpolation rule as the device runs).", "",
        f"Stage 5 simulation at n = 2 (different test states; 3-feature best-of-3 shadows) through the same "
        f"interpolation: QNN {st5['qnn_threshold']:.1f}, shadows {st5['shadow_threshold']:.1f}, ratio "
        f"{st5['ratio']:.2f} (the power-of-2 grid gives 32 vs 128 = 4).", "",
        "## Sensitivity: calibration on the full device budget", "",
        f"Calibration states at 1,024 copies, test states at c ({out['settings']['sens_reps']} replicates).", "",
        report.md_table(["Source", "QNN copies for 90%", "Shadow copies for 90%", "Ratio",
                         "P(ratio < noise-free ratio)"], sens_rows), ""])
    report.save_text("review_purity_boot.md", md)


if __name__ == "__main__":
    main()
