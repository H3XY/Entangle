# Hardware summary: ibm_marrakesh vs FakeMarrakesh

FakeMarrakesh rows replay the IDENTICAL transpiled circuits of the real jobs (same physical qubits, gates and shots; fetch_ibm_results.py --replay). Per-qubit calibration: review_calibration.md.

Every number is test accuracy. Mean ± std over independent runs (same test states, same transpiler seed); in brackets a Wilson 95% interval for all test predictions pooled over runs. Each cell: ideal (noise-free simulation, same shots), device, device recalibrated on labelled device states.

## Stage 7: pure or mixed (2 qubits, 2-copy test vs classical shadows)

| Device | Model | Runs | Ideal | Device | Device, recalibrated |
|---|---|---|---|---|---|
| ibm_marrakesh | QNN 2-copy (parity, shared) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| ibm_marrakesh | Logistic regression (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 0.992 ± 0.012 [0.95, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| ibm_marrakesh | SVM (RBF) (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| ibm_marrakesh | MLP (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| FakeMarrakesh | QNN 2-copy (parity, shared) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| FakeMarrakesh | Logistic regression (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 0.992 ± 0.012 [0.95, 1.00] |
| FakeMarrakesh | SVM (RBF) (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |
| FakeMarrakesh | MLP (classical shadows) | 2 | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] | 1.000 ± 0.000 [0.97, 1.00] |

Accuracy vs copies of each test state, subsampled from the device counts and recalibrated (mean ± std over runs):

| Device | Model | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| ibm_marrakesh | QNN 2-copy | 0.773 ± 0.026 | 0.856 ± 0.012 | 0.929 ± 0.005 | 0.976 ± 0.002 | 0.999 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| ibm_marrakesh | Classical shadows (logistic regression) | – | 0.653 ± 0.008 | 0.790 ± 0.018 | 0.907 ± 0.012 | 0.963 ± 0.001 | 0.986 ± 0.005 | 1.000 ± 0.000 |
| FakeMarrakesh | QNN 2-copy | 0.799 ± 0.007 | 0.902 ± 0.000 | 0.950 ± 0.002 | 0.992 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| FakeMarrakesh | Classical shadows (logistic regression) | – | 0.680 ± 0.010 | 0.815 ± 0.015 | 0.917 ± 0.005 | 0.967 ± 0.004 | 0.987 ± 0.004 | 0.992 ± 0.012 |

Mean shift of the QNN output, device minus exact P(class 1), per run (all; pure / mixed):

| Device | All test states | Pure / mixed |
|---|---|---|
| ibm_marrakesh | +0.0376, +0.0410 | +0.0537 / +0.0214, +0.0623 / +0.0196 |
| FakeMarrakesh | +0.0218, +0.0194 | +0.0300 / +0.0136, +0.0298 / +0.0089 |

## Stage 4: entanglement (2-copy QNN) and Ising phases (QCNN) vs classical snapshots

| Device | Task | Model | Ideal | Device | Device, recalibrated |
|---|---|---|---|---|---|
| ibm_marrakesh | entanglement, 2 qubits | QNN 2-copy | 1.000 ± 0.000 [0.95, 1.00] | 0.800 ± 0.035 [0.70, 0.87] | 0.988 ± 0.018 [0.93, 1.00] |
| ibm_marrakesh | entanglement, 2 qubits | Logistic regression (snapshots) | 1.000 ± 0.000 [0.95, 1.00] | 0.938 ± 0.018 [0.86, 0.97] | 0.950 ± 0.035 [0.88, 0.98] |
| ibm_marrakesh | entanglement, 2 qubits | SVM (RBF) (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.875 ± 0.035 [0.78, 0.93] | 0.950 ± 0.035 [0.88, 0.98] |
| ibm_marrakesh | entanglement, 2 qubits | MLP (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.900 ± 0.000 [0.81, 0.95] | 0.975 ± 0.000 [0.91, 0.99] |
| ibm_marrakesh | Ising phases, 4 qubits | QCNN | 0.950 ± 0.000 [0.88, 0.98] | 0.925 ± 0.035 [0.85, 0.97] | 0.925 ± 0.035 [0.85, 0.97] |
| ibm_marrakesh | Ising phases, 4 qubits | Logistic regression (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.950 ± 0.035 [0.88, 0.98] | 0.962 ± 0.018 [0.90, 0.99] |
| ibm_marrakesh | Ising phases, 4 qubits | SVM (RBF) (snapshots) | 0.950 ± 0.000 [0.88, 0.98] | 0.962 ± 0.018 [0.90, 0.99] | 0.900 ± 0.035 [0.81, 0.95] |
| ibm_marrakesh | Ising phases, 4 qubits | MLP (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.950 ± 0.035 [0.88, 0.98] | 0.950 ± 0.000 [0.88, 0.98] |
| FakeMarrakesh | entanglement, 2 qubits | QNN 2-copy | 1.000 ± 0.000 [0.95, 1.00] | 0.962 ± 0.018 [0.90, 0.99] | 0.975 ± 0.000 [0.91, 0.99] |
| FakeMarrakesh | entanglement, 2 qubits | Logistic regression (snapshots) | 1.000 ± 0.000 [0.95, 1.00] | 0.988 ± 0.018 [0.93, 1.00] | 0.988 ± 0.018 [0.93, 1.00] |
| FakeMarrakesh | entanglement, 2 qubits | SVM (RBF) (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.975 ± 0.000 [0.91, 0.99] | 0.988 ± 0.018 [0.93, 1.00] |
| FakeMarrakesh | entanglement, 2 qubits | MLP (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.975 ± 0.000 [0.91, 0.99] | 1.000 ± 0.000 [0.95, 1.00] |
| FakeMarrakesh | Ising phases, 4 qubits | QCNN | 0.950 ± 0.000 [0.88, 0.98] | 0.988 ± 0.018 [0.93, 1.00] | 0.975 ± 0.035 [0.91, 0.99] |
| FakeMarrakesh | Ising phases, 4 qubits | Logistic regression (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.975 ± 0.000 [0.91, 0.99] | 0.950 ± 0.071 [0.88, 0.98] |
| FakeMarrakesh | Ising phases, 4 qubits | SVM (RBF) (snapshots) | 0.950 ± 0.000 [0.88, 0.98] | 0.938 ± 0.018 [0.86, 0.97] | 0.950 ± 0.035 [0.88, 0.98] |
| FakeMarrakesh | Ising phases, 4 qubits | MLP (snapshots) | 0.975 ± 0.000 [0.91, 0.99] | 0.975 ± 0.000 [0.91, 0.99] | 0.950 ± 0.071 [0.88, 0.98] |

Mean shift of the quantum models' output, device minus exact P(class 1), per run:

| Device | Task | Model | Per run |
|---|---|---|---|
| ibm_marrakesh | entanglement, 2 qubits | QNN 2-copy | +0.0284, +0.0316 |
| ibm_marrakesh | Ising phases, 4 qubits | QCNN | -0.0066, -0.0085 |
| FakeMarrakesh | entanglement, 2 qubits | QNN 2-copy | +0.0161, +0.0152 |
| FakeMarrakesh | Ising phases, 4 qubits | QCNN | -0.0068, -0.0086 |

## Calibration of the physical qubits used (medians)

Full per-qubit values (T1, T2, readout, sx and CZ errors) are in each results JSON under `calibration_snapshot`.

| Run | Snapshot (UTC) | Qubits | T1 (us) | T2 (us) | Readout error | CZ error |
|---|---|---|---|---|---|---|
| stage4_hardware_ibm_run1.json | 2026-11-09T18:29:29 | 7 | 236 | 136 | 0.0176 | 0.0021 (10 pairs) |
| stage4_hardware_ibm_run2.json | 2026-11-09T18:31:06 | 7 | 236 | 136 | 0.0176 | 0.0021 (10 pairs) |
| stage 7 run 1 | 2026-11-09T18:16:24 | 6 | 205 | 152 | 0.0071 | 0.0021 (10 pairs) |
| stage 7 run 2 | 2026-11-09T18:17:40 | 6 | 205 | 152 | 0.0071 | 0.0021 (10 pairs) |

## Billed jobs

| Job ID | Label | Submitted (UTC) | Circuits | Shots | Estimate (s) | Quantum s |
|---|---|---|---|---|---|---|
| dbp0r10qtr0c738dj0fg | stage7 pilot | 2026-11-09T18:14:27 | 80 | 3040 | 3.8 | 3 |
| dbp0rd0ou4as73csr3t0 | stage7 run 1 | 2026-11-09T18:15:15 | 800 | 122320 | 37.0 | 36 |
| dbp0s1lt7fqc7388jap0 | stage7 run 2 | 2026-11-09T18:16:36 | 800 | 122320 | 37.0 | 36 |
| dbp11agqtr0c738djefg | stage4 run 1 entanglement, 2 qubits QNN 2-copy | 2026-11-09T18:27:52 | 56 | 14336 | 7.0 | 6 |
| dbp11eoou4as73csrgp0 | stage4 run 1 entanglement, 2 qubits snapshots | 2026-11-09T18:28:10 | 504 | 28224 | 10.9 | 11 |
| dbp11mc4148s739tpbpg | stage4 run 1 Ising phases, 4 qubits QCNN | 2026-11-09T18:28:41 | 56 | 28672 | 11.0 | 10 |
| dbp11rlt7fqc7388jns0 | stage4 run 1 Ising phases, 4 qubits snapshots | 2026-11-09T18:29:02 | 112 | 28672 | 11.0 | 10 |
| dbp124gqtr0c738djg4g | stage4 run 2 entanglement, 2 qubits QNN 2-copy | 2026-11-09T18:29:37 | 56 | 14336 | 7.0 | 6 |
| dbp128k4148s739tpcv0 | stage4 run 2 entanglement, 2 qubits snapshots | 2026-11-09T18:29:53 | 504 | 28224 | 10.9 | 11 |
| dbp12g5t7fqc7388jp20 | stage4 run 2 Ising phases, 4 qubits QCNN | 2026-11-09T18:30:24 | 56 | 28672 | 11.0 | 10 |
| dbp12lc4148s739tpdm0 | stage4 run 2 Ising phases, 4 qubits snapshots | 2026-11-09T18:30:45 | 112 | 28672 | 11.0 | 10 |

Total billed in this campaign: 149 s. (The earlier 280-circuit run of 8–9 Nov predates the job log.)
