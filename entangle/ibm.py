"""IBM Quantum access shared by the hardware stages: credentials, backends, a
quantum-time budget guard, a job log, and calibration snapshots.

Credentials, checked in order (never commit them):
    1. IBM_QUANTUM_TOKEN environment variable (plus optional IBM_QUANTUM_INSTANCE, a CRN)
    2. an account saved with QiskitRuntimeService.save_account(...)

Budget guard (real devices only): before every job we read the account's remaining
quantum time with QiskitRuntimeService().usage() and refuse to submit if

    remaining - estimated cost of this job < floor

The estimate comes from a measured job on ibm_marrakesh (200 circuits x 1024 shots
cost 57 s): about 0.26 ms per shot plus about 3 s per job. A session cap (`cap`)
also limits the total this process may spend. Every job (fake or real) is appended
to results/ibm_jobs.json with its ID, time, size and the quantum seconds IBM bills.
"""
import datetime
import json
import logging
import os

from . import report

# qiskit-ibm-runtime's transpiler plugins fail to import against Qiskit 1.x (we do not use them)
logging.getLogger("stevedore.extension").setLevel(logging.CRITICAL)

SECONDS_PER_SHOT = 57.0 / (200 * 1024)
SECONDS_PER_JOB = 3.0
JOB_LOG = "ibm_jobs.json"


def get_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    try:
        return QiskitRuntimeService()  # an account saved earlier
    except Exception as e:
        raise SystemExit("No IBM Quantum credentials. Set IBM_QUANTUM_TOKEN or save an account with "
                         "QiskitRuntimeService.save_account(...); or use --fake.\n"
                         f"Details: {e}")


def get_backend(fake, backend_fake=None, backend=None, min_qubits=1):
    if fake:
        from qiskit_ibm_runtime import fake_provider
        b = getattr(fake_provider, backend_fake)()
        print(f"Using local fake backend {b.name} (simulated IBM noise, no account needed)")
        return b
    service = get_service()
    b = (service.backend(backend) if backend else
         service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits))
    print("Selected IBM Quantum backend:", b.name)
    return b


def estimate_seconds(total_shots, jobs=1):
    return SECONDS_PER_SHOT * total_shots + SECONDS_PER_JOB * jobs


class Budget:
    """Stops real-device jobs that would take the account below `floor` seconds or this
    process past `cap` seconds (both None = no limit)."""

    def __init__(self, floor=None, cap=None):
        self.floor, self.cap, self.spent = floor, cap, 0.0
        self._service = None

    def remaining(self):
        if self._service is None:
            self._service = get_service()
        return float(self._service.usage()["usage_remaining_seconds"])

    def check(self, estimate):
        if self.cap is not None and self.spent + estimate > self.cap:
            raise SystemExit(f"Budget cap: this job (~{estimate:.0f} s) would take this run to "
                             f"{self.spent + estimate:.0f} s, over the cap of {self.cap:.0f} s. Not submitted.")
        if self.floor is not None:
            left = self.remaining()
            if left - estimate < self.floor:
                raise SystemExit(f"Budget floor: {left:.0f} s remain on the account; this job (~{estimate:.0f} s) "
                                 f"would leave less than the floor of {self.floor:.0f} s. Not submitted.")
            print(f"    budget: {left:.0f} s remain, job estimate ~{estimate:.0f} s, floor {self.floor:.0f} s",
                  flush=True)


def _quantum_seconds(job, tries=10, wait=3.0):
    """Billed quantum seconds. IBM fills this in a few seconds after the result, so retry."""
    import time
    for _ in range(tries):
        try:
            qs = float(job.metrics()["usage"]["quantum_seconds"])
            if qs > 0:
                return qs
        except Exception:
            pass
        time.sleep(wait)
    return None


def log_job(entry):
    path = report.path(JOB_LOG)
    jobs = json.load(open(path)) if os.path.exists(path) else []
    jobs.append(entry)
    report.save_json(JOB_LOG, jobs)


def transpile(backend, circuits, seed=None, initial_layout=None):
    """ISA circuits for the backend (fixed seed_transpiler so repeat runs map the same way)."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    # translation_method="translator": Qiskit's standard basis translation. Real IBM backends
    # otherwise ask for qiskit-ibm-runtime's own plugin, which fails to import against Qiskit 1.x.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed,
                                      translation_method="translator", initial_layout=initial_layout)
    return pm.run(circuits)


def run_pubs(backend, circuits, shots, seed=None, simulator=False, budget=None, label="",
             initial_layout=None, transpiled=False):
    """Run circuits in ONE SamplerV2 job. `shots` is an int or one int per circuit.
    Pass `transpiled=True` for circuits already mapped with `transpile`.

    Returns (counts per circuit, info dict with job ID, layouts, timing, quantum seconds).
    """
    from qiskit_ibm_runtime import SamplerV2
    isa = list(circuits) if transpiled else transpile(backend, circuits, seed, initial_layout)
    shot_list = [shots] * len(isa) if isinstance(shots, int) else [int(s) for s in shots]
    total = int(sum(shot_list))
    est = estimate_seconds(total)
    if budget is not None and not simulator:
        budget.check(est)
    sampler = SamplerV2(mode=backend)
    if seed is not None and simulator:  # fake backends only: reproducible noise sampling
        sampler.options.simulator.seed_simulator = seed  # (real devices reject this option)
    submitted = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job = sampler.run([(c, None, s) for c, s in zip(isa, shot_list)])
    print(f"    job {job.job_id()}: {len(isa)} circuits, {total} shots (estimate ~{est:.0f} s)", flush=True)
    result = job.result()
    counts = [r.data.meas.get_counts() for r in result]
    qs = None if simulator else _quantum_seconds(job)
    if budget is not None and not simulator:
        budget.spent += qs if qs is not None else est
    layouts = [list(c.layout.final_index_layout()) if c.layout is not None else None for c in isa]
    two_q = [sum(v for k, v in c.count_ops().items() if k in ("cz", "cx", "ecr")) for c in isa]
    info = {"job_id": job.job_id(), "backend": backend.name, "simulator": simulator, "label": label,
            "submitted_utc": submitted, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "circuits": len(isa), "total_shots": total, "estimated_seconds": round(est, 1),
            "quantum_seconds": qs, "seed_transpiler": seed,
            "physical_qubits": sorted({q for lay in layouts if lay for q in lay})}
    if not simulator:  # fake-backend jobs are not billed and not logged
        log_job(info)
    if qs is not None:
        print(f"    billed quantum time: {qs:.1f} s", flush=True)
    return counts, dict(info, layouts=layouts, two_qubit_gates=two_q)


def calibration_snapshot(backend, qubits):
    """T1, T2, readout and gate errors of the given physical qubits from backend.target."""
    t = backend.target
    qs = sorted(set(qubits))
    snap = {"backend": backend.name, "taken_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "qubits": {}, "two_qubit": {}}
    try:
        props = backend.properties()
        snap["properties_last_update"] = str(getattr(props, "last_update_date", None))
    except Exception:
        pass
    for q in qs:
        qp = t.qubit_properties[q] if t.qubit_properties else None
        entry = {"T1_us": None if qp is None or qp.t1 is None else qp.t1 * 1e6,
                 "T2_us": None if qp is None or qp.t2 is None else qp.t2 * 1e6}
        for op in ("measure", "sx", "x"):
            if op in t.operation_names and (q,) in t[op] and t[op][(q,)] is not None:
                entry[f"{op}_error"] = t[op][(q,)].error
        snap["qubits"][str(q)] = entry
    for op in ("cz", "cx", "ecr"):
        if op not in t.operation_names:
            continue
        for pair, prop in t[op].items():
            if pair[0] in qs and pair[1] in qs and prop is not None:
                snap["two_qubit"][f"{op} {pair[0]}-{pair[1]}"] = prop.error
    return snap
