# Stage 7: pure or mixed, on fake_marrakesh (fake backend, simulated noise)

2-qubit states (3-qubit purifications), 60 test states (balanced), 1024 copies of each state per model, 2 independent run(s) on the same test states and physical qubits. Recalibrated = refit on 20 labelled states prepared on the device. Mean ± std over runs; per run with a Wilson 95% interval.

| Model | Setting | Mean ± std over runs | Per run [95% CI] |
|---|---|---|---|
| QNN 2-copy (parity, shared) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| QNN 2-copy (parity, shared) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| QNN 2-copy (parity, shared) | device recalibrated | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | ideal | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | device | 1.000 ± 0.000 | 1.000 [0.94, 1.00], 1.000 [0.94, 1.00] |
| Logistic regression (classical shadows) | device recalibrated | 0.992 ± 0.012 | 1.000 [0.94, 1.00], 0.983 [0.91, 1.00] |
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
| run 1 | dbp0rd0ou4as73csr3t0 | 2026-11-09T18:15:15 | 36 | +0.0218 | +0.0300 / +0.0136 | 4 6 5 8 17 7 |
| run 2 | dbp0s1lt7fqc7388jap0 | 2026-11-09T18:16:36 | 36 | +0.0194 | +0.0298 / +0.0089 | 4 6 5 8 17 7 |

Two-qubit gates per transpiled circuit: shadow (prep only) pure 7.0, mixed 7.0; QNN circuit mean 19.0.

## Accuracy vs copies, subsampled from the device counts (recalibrated)

| Run | Model | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| run 1 | QNN 2-copy | 0.794 | 0.902 | 0.951 | 0.991 | 1.000 | 1.000 | 1.000 |
| run 1 | Classical shadows (logistic regression) | – | 0.672 | 0.826 | 0.913 | 0.970 | 0.990 | 1.000 |
| run 2 | QNN 2-copy | 0.804 | 0.902 | 0.948 | 0.993 | 1.000 | 1.000 | 1.000 |
| run 2 | Classical shadows (logistic regression) | – | 0.687 | 0.805 | 0.920 | 0.965 | 0.984 | 0.983 |
