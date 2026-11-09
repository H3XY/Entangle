# Stage 7: pure or mixed, on ibm_marrakesh

2-qubit states (3-qubit purifications), 4 test states (balanced), 256 copies of each state per model, 1 independent run(s) on the same test states and physical qubits. Recalibrated = refit on 4 labelled states prepared on the device. Mean ± std over runs; per run with a Wilson 95% interval.

| Model | Setting | Mean ± std over runs | Per run [95% CI] |
|---|---|---|---|
| QNN 2-copy (parity, shared) | ideal | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| QNN 2-copy (parity, shared) | device | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| QNN 2-copy (parity, shared) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| Logistic regression (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| Logistic regression (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| Logistic regression (classical shadows) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| SVM (RBF) (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| SVM (RBF) (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| SVM (RBF) (classical shadows) | device recalibrated | 0.750 ± 0.000 | 0.750 [0.30, 0.95] |
| MLP (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| MLP (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |
| MLP (classical shadows) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.51, 1.00] |

## Jobs and readout shift

Mean shift = mean over test states of (device P(class 1) - exact P(class 1)) for the QNN; by class: pure / mixed.

| Run | Job ID | Submitted (UTC) | Quantum s | Mean shift | By class | Physical qubits |
|---|---|---|---|---|---|---|
| pilot | dbp0r10qtr0c738dj0fg | 2026-11-09T18:14:27 | 0 | +0.0445 | +0.0341 / +0.0549 | 4 6 5 8 17 7 |

Two-qubit gates per transpiled circuit: shadow (prep only) pure 7.0, mixed 7.0; QNN circuit mean 19.0.

## Accuracy vs copies, subsampled from the device counts (recalibrated)

| Run | Model | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| pilot | QNN 2-copy | 0.625 | 0.875 | 1.000 | 1.000 | 1.000 |
| pilot | Classical shadows (logistic regression) | – | 0.500 | 0.875 | 1.000 | 1.000 |
