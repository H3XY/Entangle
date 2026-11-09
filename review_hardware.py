"""Summary of every hardware run: real ibm_marrakesh vs its FakeMarrakesh noise model.

Reads results/stage4_hardware_<tag>.json (one file per run), results/stage7_hardware_purity_<tag>.json
(all runs in one file) and results/ibm_jobs.json, and writes results/review_hardware.md:
  - accuracy per model: mean ± std over independent runs, with a Wilson 95% interval for
    the pooled test states
  - the mean shift of the quantum models' outputs (device minus exact), real vs fake
  - the calibration of the physical qubits used (medians of T1, T2, readout and CZ error)
  - every billed job with its ID, time and quantum seconds

Usage:
    python review_hardware.py
"""
import glob
import json
import os
import re

import numpy as np

from entangle import report


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def calib_summary(snap):
    if not snap:
        return {}
    q = snap["qubits"].values()

    def med(key):
        v = [e[key] for e in q if e.get(key) is not None]
        return float(np.median(v)) if v else None
    cz = list(snap["two_qubit"].values())
    return {"qubits": len(snap["qubits"]), "T1_us": med("T1_us"), "T2_us": med("T2_us"),
            "readout_error": med("measure_error"), "sx_error": med("sx_error"),
            "cz_error": float(np.median(cz)) if cz else None, "cz_pairs": len(cz),
            "taken_utc": snap.get("taken_utc", "")[:19]}


def stage4_runs(prefix):
    runs = []
    for f in sorted(glob.glob(report.path(f"stage4_hardware_{prefix}_run*.json"))):
        runs.append(json.load(open(f)))
    return runs


def stage4_rows(runs, label):
    rows, shifts = [], []
    if not runs:
        return rows, shifts
    tasks = [r["task"] for r in runs[0]["results"]]
    for ti, task in enumerate(tasks):
        models = runs[0]["results"][ti]["models"]
        nt = runs[0]["results"][ti]["test_states"]
        for name in models:
            row = [label, task, name]
            for key in ("ideal", "device", "device_recalibrated"):
                vals = [r["results"][ti]["models"][name].get(key) for r in runs]
                k = round(sum(vals) * nt)
                lo, hi = wilson(k, nt * len(vals))
                row.append(f"{ms(vals)} [{lo:.2f}, {hi:.2f}]")
            rows.append(row)
            m0 = models[name]
            if m0["kind"] == "quantum":
                s = [float(np.mean(np.array(r["results"][ti]["models"][name]["p_device"])
                                   - np.array(r["results"][ti]["models"][name]["p_ideal_exact"]))) for r in runs]
                shifts.append([label, task, name, ", ".join(f"{x:+.4f}" for x in s)])
    return rows, shifts


def main():
    lines = ["# Hardware summary: ibm_marrakesh vs FakeMarrakesh", "",
             "Every number is test accuracy. Mean ± std over independent runs (same test states, same "
             "transpiler seed); in brackets a Wilson 95% interval for all test predictions pooled over runs. "
             "Each cell: ideal (noise-free simulation, same shots), device, device recalibrated on labelled "
             "device states.", ""]

    # ---- stage 7
    lines += ["## Stage 7: pure or mixed (2 qubits, 2-copy test vs classical shadows)", ""]
    rows, shift_rows, sweep_rows = [], [], []
    for tag, label in (("ibm", "ibm_marrakesh"), ("fakemarrakesh", "FakeMarrakesh")):
        f = report.path(f"stage7_hardware_purity_{tag}.json")
        if not os.path.exists(f):
            continue
        runs = json.load(open(f))["runs"]
        nt = runs[0]["test_states"]
        for name in runs[0]["models"]:
            row = [label, name, len(runs)]
            for key in ("ideal", "device", "device_recalibrated"):
                vals = [r["models"][name][key] for r in runs]
                lo, hi = wilson(round(sum(vals) * nt), nt * len(vals))
                row.append(f"{ms(vals)} [{lo:.2f}, {hi:.2f}]")
            rows.append(row)
        q = "QNN 2-copy (parity, shared)"
        shift_rows.append([label, ", ".join(f"{r['models'][q]['mean_shift_p']:+.4f}" for r in runs),
                           ", ".join(f"{r['models'][q]['mean_shift_p_by_class']['0']:+.4f} / "
                                     f"{r['models'][q]['mean_shift_p_by_class']['1']:+.4f}" for r in runs)])
        cop = runs[0]["device_copies_sweep"]["copies"]
        for model in ("QNN 2-copy", "Classical shadows (logistic regression)"):
            vals = []
            for i in range(len(cop)):
                v = [r["device_copies_sweep"][model][i] for r in runs]
                vals.append("–" if v[0] is None else ms([x[0] for x in v]))
            sweep_rows.append([label, model] + vals)
    if rows:
        lines += [report.md_table(["Device", "Model", "Runs", "Ideal", "Device", "Device, recalibrated"], rows), "",
                  "Accuracy vs copies of each test state, subsampled from the device counts and recalibrated "
                  "(mean ± std over runs):", "",
                  report.md_table(["Device", "Model"] + [str(c) for c in cop], sweep_rows), "",
                  "Mean shift of the QNN output, device minus exact P(class 1), per run (all; pure / mixed):", "",
                  report.md_table(["Device", "All test states", "Pure / mixed"], shift_rows), ""]

    # ---- stage 4
    lines += ["## Stage 4: entanglement (2-copy QNN) and Ising phases (QCNN) vs classical snapshots", ""]
    all_rows, all_shift = [], []
    for prefix, label in (("ibm", "ibm_marrakesh"), ("fakemarrakesh", "FakeMarrakesh")):
        r, s = stage4_rows(stage4_runs(prefix), label)
        all_rows += r
        all_shift += s
    if all_rows:
        lines += [report.md_table(["Device", "Task", "Model", "Ideal", "Device", "Device, recalibrated"], all_rows), "",
                  "Mean shift of the quantum models' output, device minus exact P(class 1), per run:", "",
                  report.md_table(["Device", "Task", "Model", "Per run"], all_shift), ""]

    # ---- calibration
    cal_rows = []
    for f in sorted(glob.glob(report.path("stage4_hardware_ibm_run*.json"))):
        c = calib_summary(json.load(open(f)).get("calibration_snapshot"))
        if c:
            cal_rows.append([os.path.basename(f), c["taken_utc"], c["qubits"], f"{c['T1_us']:.0f}",
                             f"{c['T2_us']:.0f}", f"{c['readout_error']:.4f}",
                             "–" if c["cz_error"] is None else f"{c['cz_error']:.4f} ({c['cz_pairs']} pairs)"])
    f7 = report.path("stage7_hardware_purity_ibm.json")
    if os.path.exists(f7):
        for r in json.load(open(f7))["runs"]:
            c = calib_summary(r["calibration_snapshot"])
            cal_rows.append([f"stage 7 {r['run_label']}", c["taken_utc"], c["qubits"], f"{c['T1_us']:.0f}",
                             f"{c['T2_us']:.0f}", f"{c['readout_error']:.4f}",
                             "–" if c["cz_error"] is None else f"{c['cz_error']:.4f} ({c['cz_pairs']} pairs)"])
    if cal_rows:
        lines += ["## Calibration of the physical qubits used (medians)", "",
                  "Full per-qubit values (T1, T2, readout, sx and CZ errors) are in each results JSON under "
                  "`calibration_snapshot`.", "",
                  report.md_table(["Run", "Snapshot (UTC)", "Qubits", "T1 (us)", "T2 (us)", "Readout error",
                                   "CZ error"], cal_rows), ""]

    # ---- jobs
    jf = report.path("ibm_jobs.json")
    if os.path.exists(jf):
        jobs = json.load(open(jf))
        total = sum(j["quantum_seconds"] or 0 for j in jobs)
        lines += ["## Billed jobs", "",
                  report.md_table(["Job ID", "Label", "Submitted (UTC)", "Circuits", "Shots", "Estimate (s)",
                                   "Quantum s"],
                                  [[j["job_id"], j["label"], j["submitted_utc"][:19], j["circuits"],
                                    j["total_shots"], j["estimated_seconds"],
                                    "–" if j["quantum_seconds"] is None else f"{j['quantum_seconds']:.0f}"]
                                   for j in jobs]), "",
                  f"Total billed in this campaign: {total:.0f} s. (The earlier 280-circuit run of 8–9 Nov "
                  "predates the job log.)", ""]
    report.save_text("review_hardware.md", "\n".join(lines))
    print("Wrote results/review_hardware.md")


if __name__ == "__main__":
    main()
