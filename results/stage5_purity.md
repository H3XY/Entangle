# Stage 5: pure or mixed? (a global property)

Haar-random pure states vs random rank-2 mixed states. 200 training and 400 test states per qubit count; mean ± std over 3 seeds. Classical columns: best of logistic regression, RBF SVM and MLP on classical-shadow estimates of the global and single-qubit purities.

## Exact accuracy (infinite shots)

| Qubits | QNN 2-copy, per-pair blocks, from scratch | QNN 2-copy, shared block, from scratch | QNN 2-copy, shared, transferred from 2 qubits | QNN 1-copy |
|---|---|---|---|---|
| 2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.512 ± 0.031 |
| 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.521 ± 0.035 |
| 4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.482 ± 0.009 |
| 5 | 0.852 ± 0.210 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.522 ± 0.021 |
| 6 | – | – | 1.000 ± 0.000 | 0.474 ± 0.026 |

Per-pair from scratch and shared from scratch are trained only up to 5 qubits.

## Copies of each test state needed for 90% accuracy

| Qubits | QNN 2-copy (transferred) | Classical shadows |
|---|---|---|
| 2 | 32 | 128 |
| 3 | 16 | 256 |
| 4 | 32 | 512 |
| 5 | 32 | 1024 |
| 6 | 32 | 4096 |

## Accuracy vs copies of each test state

| Qubits | Model | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | QNN 2-copy, shared, transferred from 2 qubits | 0.616 | 0.696 | 0.799 | 0.891 | 0.935 | 0.970 | 0.992 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | Classical shadows (purity estimates) | 0.531 | 0.533 | 0.584 | 0.633 | 0.751 | 0.840 | 0.938 | 0.981 | 0.998 | 1.000 | 1.000 | 1.000 |
| 2 | QNN 1-copy | 0.522 | 0.521 | 0.514 | 0.501 | 0.503 | 0.500 | 0.505 | 0.514 | 0.508 | 0.504 | 0.503 | 0.505 |
| 3 | QNN 2-copy, shared, transferred from 2 qubits | 0.617 | 0.717 | 0.808 | 0.927 | 0.961 | 0.981 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | Classical shadows (purity estimates) | 0.522 | 0.525 | 0.544 | 0.598 | 0.649 | 0.733 | 0.842 | 0.928 | 0.988 | 0.995 | 1.000 | 1.000 |
| 3 | QNN 1-copy | 0.495 | 0.505 | 0.503 | 0.497 | 0.506 | 0.524 | 0.507 | 0.517 | 0.524 | 0.518 | 0.522 | 0.524 |
| 4 | QNN 2-copy, shared, transferred from 2 qubits | 0.624 | 0.701 | 0.822 | 0.892 | 0.943 | 0.971 | 0.994 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | Classical shadows (purity estimates) | 0.512 | 0.522 | 0.549 | 0.538 | 0.592 | 0.676 | 0.733 | 0.809 | 0.913 | 0.978 | 0.999 | 1.000 |
| 4 | QNN 1-copy | 0.487 | 0.492 | 0.492 | 0.487 | 0.496 | 0.480 | 0.484 | 0.494 | 0.481 | 0.482 | 0.488 | 0.481 |
| 5 | QNN 2-copy, shared, transferred from 2 qubits | 0.613 | 0.724 | 0.828 | 0.872 | 0.948 | 0.979 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | Classical shadows (purity estimates) | 0.509 | 0.525 | 0.509 | 0.512 | 0.555 | 0.593 | 0.676 | 0.743 | 0.805 | 0.901 | 0.985 | 0.998 |
| 5 | QNN 1-copy | 0.492 | 0.527 | 0.532 | 0.517 | 0.502 | 0.503 | 0.507 | 0.515 | 0.511 | 0.514 | 0.522 | 0.517 |
| 6 | QNN 2-copy, shared, transferred from 2 qubits | 0.615 | 0.687 | 0.787 | 0.873 | 0.924 | 0.983 | 0.998 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | Classical shadows (purity estimates) | 0.529 | 0.510 | 0.500 | 0.525 | 0.519 | 0.553 | 0.615 | 0.668 | 0.712 | 0.815 | 0.893 | 0.973 |
| 6 | QNN 1-copy | 0.482 | 0.507 | 0.502 | 0.470 | 0.506 | 0.497 | 0.492 | 0.484 | 0.479 | 0.474 | 0.484 | 0.475 |
