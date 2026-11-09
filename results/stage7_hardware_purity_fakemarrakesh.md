# Stage 7: pure or mixed, on fake_marrakesh (fake backend, simulated noise)

2-qubit states (3-qubit purifications), 60 test states (balanced), 1024 copies of each state per model, 2 independent run(s) on the same test states and physical qubits. Recalibrated = refit on 20 labelled states prepared on the device. Mean ± std over runs; per run with a Wilson 95% interval.

| Model | Setting | Mean ± std over runs | Per run [95% CI] |
|---|---|---|---|
| QNN 2-copy (parity, shared) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| QNN 2-copy (parity, shared) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| QNN 2-copy (parity, shared) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| SVM (RBF) (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| SVM (RBF) (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| SVM (RBF) (classical shadows) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| MLP (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| MLP (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| MLP (classical shadows) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |

## Jobs and readout shift

Mean shift = mean over test states of (device P(class 1) - exact P(class 1)) for the QNN; by class: pure / mixed.

| Run | Job ID | Submitted (UTC) | Quantum s | Mean shift | By class | Physical qubits |
|---|---|---|---|---|---|---|
| run 1 | b795639e-0332-4008-9d14-8f5c223725ed | 2026-11-09T16:26:29 | – | +0.0151 | +0.0216 / +0.0086 | 14 12 13 18 10 11 |
| run 2 | d2dca310-06fc-4f98-b625-550be4540516 | 2026-11-09T16:26:50 | – | +0.0166 | +0.0228 / +0.0104 | 14 12 13 18 10 11 |

Two-qubit gates per transpiled circuit: shadow (prep only) pure 7.0, mixed 7.0; QNN circuit mean 19.0.

## Accuracy vs copies, subsampled from the device counts (recalibrated)

| Run | Model | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| run 1 | QNN 2-copy | 0.837 | 0.914 | 0.973 | 0.991 | 1.000 | 1.000 | 1.000 |
| run 1 | Classical shadows (logistic regression) | – | 0.685 | 0.827 | 0.918 | 0.971 | 0.995 | 1.000 |
| run 2 | QNN 2-copy | 0.839 | 0.902 | 0.958 | 0.993 | 1.000 | 1.000 | 1.000 |
| run 2 | Classical shadows (logistic regression) | – | 0.657 | 0.800 | 0.903 | 0.957 | 0.980 | 1.000 |
