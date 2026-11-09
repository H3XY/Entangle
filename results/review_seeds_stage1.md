# Seed-to-seed uncertainty, stage 1

Entanglement classification (stage 1). Seeds: 0, 1, 2, 3, 4. Each seed redraws training and test states, shot noise and model initialisations. Std is the sample standard deviation over seeds. 'Best of k' picks the best classical learner on the test set, which flatters the classical side.

| Qubits | Quantity | Mean ± std over seeds | Range | Per seed |
|---|---|---|---|---|
| 2 | QNN 2-copy, exact | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 2 | QNN 2-copy, 256 copies | 0.991 ± 0.006 | 0.983–0.998 | 0.998, 0.983, 0.990, 0.998, 0.990 |
| 2 | QNN 1-copy, exact | 0.494 ± 0.017 | 0.468–0.512 | 0.492, 0.495, 0.502, 0.468, 0.512 |
| 2 | Classical, full state vector (best of 3) | 0.715 ± 0.018 | 0.685–0.730 | 0.685, 0.725, 0.723, 0.713, 0.730 |
| 2 | Classical shadows, exact estimates (best of 6) | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 2 | Classical shadows, 256 snapshots (best of 6) | 0.952 ± 0.008 | 0.945–0.963 | 0.945, 0.945, 0.958, 0.950, 0.963 |
| 2 | Classical shadows, all Pauli features, 256 snapshots (best of 3) | 0.935 ± 0.011 | 0.920–0.948 | 0.945, 0.930, 0.948, 0.920, 0.935 |
| 2 | Classical shadows, purity features, 256 snapshots (best of 3) | 0.951 ± 0.009 | 0.940–0.963 | 0.940, 0.945, 0.958, 0.950, 0.963 |
| 3 | QNN 2-copy, exact | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 3 | QNN 2-copy, 256 copies | 0.992 ± 0.004 | 0.988–0.995 | 0.995, 0.988, 0.995, 0.993, 0.988 |
| 3 | QNN 1-copy, exact | 0.497 ± 0.020 | 0.472–0.522 | 0.500, 0.482, 0.507, 0.472, 0.522 |
| 3 | Classical, full state vector (best of 3) | 0.569 ± 0.024 | 0.540–0.598 | 0.552, 0.598, 0.588, 0.568, 0.540 |
| 3 | Classical shadows, exact estimates (best of 6) | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 3 | Classical shadows, 256 snapshots (best of 6) | 0.975 ± 0.005 | 0.970–0.983 | 0.970, 0.970, 0.975, 0.975, 0.983 |
| 3 | Classical shadows, all Pauli features, 256 snapshots (best of 3) | 0.947 ± 0.021 | 0.920–0.973 | 0.930, 0.955, 0.958, 0.920, 0.973 |
| 3 | Classical shadows, purity features, 256 snapshots (best of 3) | 0.975 ± 0.005 | 0.970–0.983 | 0.970, 0.970, 0.975, 0.975, 0.983 |
| 4 | QNN 2-copy, exact | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 4 | QNN 2-copy, 256 copies | 0.996 ± 0.005 | 0.988–1.000 | 1.000, 0.988, 0.995, 1.000, 1.000 |
| 4 | QNN 1-copy, exact | 0.494 ± 0.030 | 0.443–0.520 | 0.505, 0.495, 0.520, 0.443, 0.507 |
| 4 | Classical, full state vector (best of 3) | 0.524 ± 0.025 | 0.490–0.545 | 0.535, 0.507, 0.490, 0.545, 0.545 |
| 4 | Classical shadows, exact estimates (best of 6) | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000, 1.000, 1.000 |
| 4 | Classical shadows, 256 snapshots (best of 6) | 0.988 ± 0.006 | 0.978–0.993 | 0.988, 0.990, 0.993, 0.990, 0.978 |
| 4 | Classical shadows, all Pauli features, 256 snapshots (best of 3) | 0.903 ± 0.015 | 0.885–0.922 | 0.885, 0.890, 0.910, 0.922, 0.907 |
| 4 | Classical shadows, purity features, 256 snapshots (best of 3) | 0.988 ± 0.006 | 0.978–0.993 | 0.988, 0.990, 0.993, 0.990, 0.978 |
