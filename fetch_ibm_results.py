"""Download the raw data of every logged IBM job, and replay the same circuits on a fake backend.

No quantum time is used. For each job in results/ibm_jobs.json this saves
  results/ibm_raw/<job_id>.counts.json   measured counts per circuit (as IBM returned them)
  results/ibm_raw/<job_id>.qpy.gz          the exact transpiled (ISA) circuits that ran, with shots
so every hardware number can be recomputed offline. With --replay FakeMarrakesh it then
runs those identical ISA circuits (same physical qubits, same gates) with the same shots on
the fake backend's noise model and saves results/ibm_raw/<job_id>.<fake>.counts.json.

Usage:
    python fetch_ibm_results.py                         # download
    python fetch_ibm_results.py --replay FakeMarrakesh  # download (if missing) and replay
"""
import argparse
import gzip
import json
import os
import warnings

from qiskit import qpy

from entangle import ibm, report

warnings.filterwarnings("ignore", category=DeprecationWarning)
RAW = report.path("ibm_raw")


def raw_path(job_id, suffix):
    os.makedirs(RAW, exist_ok=True)
    return os.path.join(RAW, f"{job_id}.{suffix}")


def download(service, job_id):
    cpath, qpath = raw_path(job_id, "counts.json"), raw_path(job_id, "qpy.gz")
    if os.path.exists(cpath) and os.path.exists(qpath):
        return
    job = service.job(job_id)
    pubs = job.inputs["pubs"]
    circuits, shots = [p[0] for p in pubs], [int(p[2]) for p in pubs]
    counts = [r.data.meas.get_counts() for r in job.result()]
    with gzip.open(qpath, "wb") as f:
        qpy.dump(circuits, f)
    json.dump({"job_id": job_id, "backend": job.backend().name, "shots": shots, "counts": counts},
              open(cpath, "w"))
    print(f"  {job_id}: {len(circuits)} circuits")


def load_circuits(job_id):
    with gzip.open(raw_path(job_id, "qpy.gz"), "rb") as f:
        return qpy.load(f)


def replay(job_id, fake_name, seed):
    from qiskit_ibm_runtime import SamplerV2, fake_provider
    out = raw_path(job_id, f"{fake_name.lower()}.counts.json")
    if os.path.exists(out):
        return
    backend = getattr(fake_provider, fake_name)()
    circuits = load_circuits(job_id)
    shots = json.load(open(raw_path(job_id, "counts.json")))["shots"]
    sampler = SamplerV2(mode=backend)
    sampler.options.simulator.seed_simulator = seed
    res = sampler.run([(c, None, s) for c, s in zip(circuits, shots)]).result()
    json.dump({"job_id": job_id, "backend": backend.name, "replay_of": job_id, "shots": shots,
               "counts": [r.data.meas.get_counts() for r in res]}, open(out, "w"))
    print(f"  replayed {job_id} on {backend.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", help="fake backend class to replay the identical circuits on")
    p.add_argument("--seed", type=int, default=0, help="base simulator seed for the replay (+ job index)")
    args = p.parse_args()
    jobs = json.load(open(report.path(ibm.JOB_LOG)))
    service = ibm.get_service()
    for j in jobs:
        download(service, j["job_id"])
    if args.replay:
        for k, j in enumerate(jobs):
            replay(j["job_id"], args.replay, args.seed + k)


if __name__ == "__main__":
    main()
