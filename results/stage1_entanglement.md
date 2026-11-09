# Stage 1: entanglement classification

Product vs entangled (Meyer-Wallach Q >= 0.2) random states. 200 training and 400 test states per qubit count, balanced. Exact = infinite shots. The copy budget is the number of copies of each test state a model may use (the 2-copy QNN uses 2 per shot).

| Qubits | Model | Size | Test acc. (exact) | Test acc. (256 copies of each state) |
|---|---|---|---|---|
| 2 | QNN 1-copy | 2 qubits, 16 weights | 0.492 | 0.495 (256 shots) |
| 2 | QNN 2-copy | 4 qubits, 16 weights | 1.000 | 0.998 (128 shots) |
| 2 | Logistic regression, full state vector | 8 numbers | 0.460 | n/a |
| 2 | SVM (RBF), full state vector | 8 numbers | 0.685 | n/a |
| 2 | MLP, full state vector | 8 numbers | 0.647 | n/a |
| 2 | Logistic regression, snapshots | 17 features | 1.000 | 0.945 (256 snapshots) |
| 2 | SVM (RBF), snapshots | 17 features | 0.998 | 0.920 (256 snapshots) |
| 2 | MLP, snapshots | 17 features | 0.988 | 0.917 (256 snapshots) |
| 2 | Logistic regression, snapshots, purities only | 2 features | 1.000 | 0.938 (256 snapshots) |
| 2 | SVM (RBF), snapshots, purities only | 2 features | 1.000 | 0.940 (256 snapshots) |
| 2 | MLP, snapshots, purities only | 2 features | 1.000 | 0.940 (256 snapshots) |
| 3 | QNN 1-copy | 3 qubits, 24 weights | 0.500 | 0.485 (256 shots) |
| 3 | QNN 2-copy | 6 qubits, 24 weights | 1.000 | 0.995 (128 shots) |
| 3 | Logistic regression, full state vector | 16 numbers | 0.547 | n/a |
| 3 | SVM (RBF), full state vector | 16 numbers | 0.552 | n/a |
| 3 | MLP, full state vector | 16 numbers | 0.525 | n/a |
| 3 | Logistic regression, snapshots | 66 features | 0.983 | 0.930 (256 snapshots) |
| 3 | SVM (RBF), snapshots | 66 features | 0.985 | 0.925 (256 snapshots) |
| 3 | MLP, snapshots | 66 features | 0.912 | 0.875 (256 snapshots) |
| 3 | Logistic regression, snapshots, purities only | 3 features | 1.000 | 0.970 (256 snapshots) |
| 3 | SVM (RBF), snapshots, purities only | 3 features | 1.000 | 0.965 (256 snapshots) |
| 3 | MLP, snapshots, purities only | 3 features | 1.000 | 0.970 (256 snapshots) |
| 4 | QNN 1-copy | 4 qubits, 32 weights | 0.505 | 0.492 (256 shots) |
| 4 | QNN 2-copy | 8 qubits, 32 weights | 1.000 | 1.000 (128 shots) |
| 4 | Logistic regression, full state vector | 32 numbers | 0.492 | n/a |
| 4 | SVM (RBF), full state vector | 32 numbers | 0.525 | n/a |
| 4 | MLP, full state vector | 32 numbers | 0.535 | n/a |
| 4 | Logistic regression, snapshots | 259 features | 0.920 | 0.875 (256 snapshots) |
| 4 | SVM (RBF), snapshots | 259 features | 0.958 | 0.885 (256 snapshots) |
| 4 | MLP, snapshots | 259 features | 0.708 | 0.640 (256 snapshots) |
| 4 | Logistic regression, snapshots, purities only | 4 features | 1.000 | 0.988 (256 snapshots) |
| 4 | SVM (RBF), snapshots, purities only | 4 features | 1.000 | 0.988 (256 snapshots) |
| 4 | MLP, snapshots, purities only | 4 features | 1.000 | 0.985 (256 snapshots) |
