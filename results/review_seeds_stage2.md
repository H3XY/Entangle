# Seed-to-seed uncertainty, stage 2

Ising chain phases (stage 2), plus single-feature baselines and near-critical accuracy. Seeds: 0, 1, 2, 3, 4. Each seed redraws training and test states, shot noise and model initialisations. Std is the sample standard deviation over seeds. 'Best of k' picks the best classical learner on the test set, which flatters the classical side.

| Qubits | Quantity | Mean ± std over seeds | Range | Per seed |
|---|---|---|---|---|
| 4 | QCNN, exact | 0.993 ± 0.008 | 0.980–0.998 | 0.998, 0.980, 0.998, 0.990, 0.998 |
| 4 | QCNN, exact, |h - 1| < 0.2 | 0.963 ± 0.037 | 0.906–0.989 | 0.987, 0.906, 0.989, 0.944, 0.989 |
| 4 | QCNN, 256 copies | 0.964 ± 0.007 | 0.955–0.973 | 0.968, 0.960, 0.968, 0.973, 0.955 |
| 4 | QNN (generic ansatz), exact | 0.993 ± 0.008 | 0.980–1.000 | 1.000, 0.980, 0.998, 0.990, 0.998 |
| 4 | QNN (generic ansatz), exact, |h - 1| < 0.2 | 0.966 ± 0.040 | 0.906–1.000 | 1.000, 0.906, 0.989, 0.944, 0.989 |
| 4 | QNN (generic ansatz), 256 copies | 0.967 ± 0.005 | 0.963–0.975 | 0.965, 0.963, 0.975, 0.968, 0.963 |
| 4 | Classical, full state vector (best of 3) | 0.993 ± 0.009 | 0.980–1.000 | 1.000, 0.980, 1.000, 0.990, 0.998 |
| 4 | Classical, Z/X expectations exact (best of 3) | 0.979 ± 0.009 | 0.965–0.988 | 0.988, 0.975, 0.983, 0.985, 0.965 |
| 4 | Classical, Z/X expectations exact, |h - 1| < 0.2 (best of 3) | 0.901 ± 0.033 | 0.853–0.935 | 0.935, 0.882, 0.921, 0.915, 0.853 |
| 4 | Classical, 256 Z/X snapshots (best of 3) | 0.965 ± 0.014 | 0.945–0.980 | 0.980, 0.968, 0.958, 0.973, 0.945 |
| 4 | One feature: mean nearest-neighbour ZZ (exact) | 0.994 ± 0.005 | 0.988–1.000 | 0.995, 0.988, 0.998, 0.990, 1.000 |
| 4 | One feature: mean X (exact) | 0.994 ± 0.005 | 0.988–0.998 | 0.998, 0.988, 0.998, 0.990, 0.998 |
| 4 | Fraction of test states with |h - 1| < 0.2 | 0.208 ± 0.024 | 0.177–0.238 | 0.193, 0.213, 0.223, 0.177, 0.238 |
| 6 | QCNN, exact | 0.998 ± 0.003 | 0.993–1.000 | 1.000, 0.998, 0.993, 1.000, 1.000 |
| 6 | QCNN, exact, |h - 1| < 0.2 | 0.991 ± 0.014 | 0.968–1.000 | 1.000, 0.988, 0.968, 1.000, 1.000 |
| 6 | QCNN, 256 copies | 0.979 ± 0.007 | 0.970–0.985 | 0.985, 0.973, 0.970, 0.980, 0.985 |
| 6 | QNN (generic ansatz), exact | 0.998 ± 0.003 | 0.993–1.000 | 1.000, 1.000, 0.993, 1.000, 1.000 |
| 6 | QNN (generic ansatz), exact, |h - 1| < 0.2 | 0.994 ± 0.014 | 0.968–1.000 | 1.000, 1.000, 0.968, 1.000, 1.000 |
| 6 | QNN (generic ansatz), 256 copies | 0.979 ± 0.004 | 0.975–0.983 | 0.975, 0.983, 0.975, 0.983, 0.978 |
| 6 | Classical, full state vector (best of 3) | 0.998 ± 0.003 | 0.993–1.000 | 1.000, 1.000, 0.993, 1.000, 1.000 |
| 6 | Classical, Z/X expectations exact (best of 3) | 0.994 ± 0.002 | 0.993–0.998 | 0.993, 0.998, 0.993, 0.993, 0.993 |
| 6 | Classical, Z/X expectations exact, |h - 1| < 0.2 (best of 3) | 0.967 ± 0.012 | 0.957–0.988 | 0.957, 0.988, 0.968, 0.961, 0.963 |
| 6 | Classical, 256 Z/X snapshots (best of 3) | 0.980 ± 0.008 | 0.970–0.990 | 0.990, 0.975, 0.970, 0.985, 0.980 |
| 6 | One feature: mean nearest-neighbour ZZ (exact) | 0.998 ± 0.004 | 0.990–1.000 | 1.000, 1.000, 0.990, 1.000, 1.000 |
| 6 | One feature: mean X (exact) | 0.998 ± 0.003 | 0.993–1.000 | 1.000, 1.000, 0.993, 1.000, 1.000 |
| 6 | Fraction of test states with |h - 1| < 0.2 | 0.204 ± 0.024 | 0.172–0.238 | 0.172, 0.213, 0.238, 0.193, 0.203 |
| 8 | QCNN, exact | 0.995 ± 0.006 | 0.985–1.000 | 0.993, 0.998, 1.000, 1.000, 0.985 |
| 8 | QCNN, exact, |h - 1| < 0.2 | 0.974 ± 0.033 | 0.923–1.000 | 0.960, 0.988, 1.000, 1.000, 0.923 |
| 8 | QCNN, 256 copies | 0.981 ± 0.008 | 0.968–0.988 | 0.980, 0.988, 0.968, 0.985, 0.988 |
| 8 | QNN (generic ansatz), exact | 0.995 ± 0.006 | 0.985–1.000 | 0.993, 0.998, 1.000, 1.000, 0.985 |
| 8 | QNN (generic ansatz), exact, |h - 1| < 0.2 | 0.974 ± 0.033 | 0.923–1.000 | 0.960, 0.988, 1.000, 1.000, 0.923 |
| 8 | QNN (generic ansatz), 256 copies | 0.978 ± 0.006 | 0.968–0.983 | 0.978, 0.980, 0.968, 0.983, 0.983 |
| 8 | Classical, full state vector (best of 3) | 0.996 ± 0.006 | 0.985–1.000 | 0.995, 0.998, 1.000, 1.000, 0.985 |
| 8 | Classical, Z/X expectations exact (best of 3) | 0.993 ± 0.003 | 0.990–0.998 | 0.995, 0.993, 0.990, 0.998, 0.990 |
| 8 | Classical, Z/X expectations exact, |h - 1| < 0.2 (best of 3) | 0.965 ± 0.014 | 0.949–0.985 | 0.973, 0.964, 0.957, 0.985, 0.949 |
| 8 | Classical, 256 Z/X snapshots (best of 3) | 0.985 ± 0.004 | 0.980–0.990 | 0.980, 0.988, 0.990, 0.985, 0.983 |
| 8 | One feature: mean nearest-neighbour ZZ (exact) | 0.995 ± 0.006 | 0.985–1.000 | 0.993, 0.998, 1.000, 1.000, 0.985 |
| 8 | One feature: mean X (exact) | 0.995 ± 0.006 | 0.985–1.000 | 0.993, 0.998, 1.000, 1.000, 0.985 |
| 8 | Fraction of test states with |h - 1| < 0.2 | 0.196 ± 0.025 | 0.162–0.230 | 0.188, 0.208, 0.230, 0.162, 0.195 |
