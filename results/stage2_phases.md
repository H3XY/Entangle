# Stage 2: Ising chain phases

Ordered (h < 1) vs disordered (h > 1) ground states, h uniform in [0.1, 2]. 200 training and 400 test states per chain length. Exact = infinite shots.

| Qubits | Model | Size | Test acc. (exact) | Exact, |h-1| >= 0.2 | Test acc. (256 copies) |
|---|---|---|---|---|---|
| 4 | QCNN | 24 weights | 0.998 | 1.000 | 0.968 |
| 4 | QNN (generic ansatz) | 32 weights | 1.000 | 1.000 | 0.965 |
| 4 | Logistic regression, full state vector | 32 numbers | 0.993 | 1.000 | n/a |
| 4 | SVM (RBF), full state vector | 32 numbers | 0.988 | 1.000 | n/a |
| 4 | MLP, full state vector | 32 numbers | 1.000 | 1.000 | n/a |
| 4 | Logistic regression, Z/X snapshots | 20 features | 0.988 | 1.000 | 0.980 |
| 4 | SVM (RBF), Z/X snapshots | 20 features | 0.975 | 0.994 | 0.963 |
| 4 | MLP, Z/X snapshots | 20 features | 0.980 | 1.000 | 0.965 |
| 6 | QCNN | 36 weights | 1.000 | 1.000 | 0.985 |
| 6 | QNN (generic ansatz) | 48 weights | 1.000 | 1.000 | 0.975 |
| 6 | Logistic regression, full state vector | 128 numbers | 1.000 | 1.000 | n/a |
| 6 | SVM (RBF), full state vector | 128 numbers | 1.000 | 1.000 | n/a |
| 6 | MLP, full state vector | 128 numbers | 1.000 | 1.000 | n/a |
| 6 | Logistic regression, Z/X snapshots | 42 features | 0.993 | 1.000 | 0.990 |
| 6 | SVM (RBF), Z/X snapshots | 42 features | 0.993 | 1.000 | 0.980 |
| 6 | MLP, Z/X snapshots | 42 features | 0.990 | 1.000 | 0.980 |
| 8 | QCNN | 36 weights | 0.993 | 1.000 | 0.980 |
| 8 | QNN (generic ansatz) | 64 weights | 0.993 | 1.000 | 0.978 |
| 8 | Logistic regression, full state vector | 512 numbers | 0.990 | 1.000 | n/a |
| 8 | SVM (RBF), full state vector | 512 numbers | 0.985 | 1.000 | n/a |
| 8 | MLP, full state vector | 512 numbers | 0.995 | 1.000 | n/a |
| 8 | Logistic regression, Z/X snapshots | 72 features | 0.995 | 1.000 | 0.980 |
| 8 | SVM (RBF), Z/X snapshots | 72 features | 0.990 | 1.000 | 0.975 |
| 8 | MLP, Z/X snapshots | 72 features | 0.985 | 1.000 | 0.958 |
