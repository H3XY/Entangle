# Seed-to-seed uncertainty, stage 6

Z2 gauge theory phases, L = 2, 8 link qubits (stage 6), plus single-feature baselines. Seeds: 0, 1, 2, 3, 4. Each seed redraws training and test states, shot noise and model initialisations. Std is the sample standard deviation over seeds. 'Best of k' picks the best classical learner on the test set, which flatters the classical side.

| Qubits | Quantity | Mean ± std over seeds | Range | Per seed |
|---|---|---|---|---|
| 8 | QCNN, exact | 0.999 ± 0.001 | 0.998–1.000 | 1.000, 1.000, 0.998, 1.000, 0.998 |
| 8 | QCNN, exact, |g - g_c| < 0.1 | 0.996 ± 0.006 | 0.988–1.000 | 1.000, 1.000, 0.988, 1.000, 0.990 |
| 8 | QCNN, 256 copies | 0.971 ± 0.007 | 0.963–0.978 | 0.963, 0.968, 0.970, 0.978, 0.978 |
| 8 | QNN (generic ansatz), exact | 0.998 ± 0.002 | 0.995–1.000 | 1.000, 1.000, 0.995, 1.000, 0.998 |
| 8 | QNN (generic ansatz), exact, |g - g_c| < 0.1 | 0.993 ± 0.010 | 0.976–1.000 | 1.000, 1.000, 0.976, 1.000, 0.990 |
| 8 | QNN (generic ansatz), 256 copies | 0.971 ± 0.007 | 0.965–0.980 | 0.975, 0.965, 0.980, 0.965, 0.968 |
| 8 | Classical, full state vector (best of 3) | 1.000 ± 0.001 | 0.998–1.000 | 1.000, 1.000, 1.000, 1.000, 0.998 |
| 8 | Classical, 256 Z/X snapshots (best of 3) | 0.983 ± 0.003 | 0.980–0.988 | 0.980, 0.985, 0.980, 0.988, 0.983 |
| 8 | Classical, 256 Z/X snapshots, |g - g_c| < 0.1 (best of 3) | 0.924 ± 0.014 | 0.905–0.937 | 0.913, 0.935, 0.905, 0.937, 0.929 |
| 8 | One feature: mean electric field X (exact) | 0.998 ± 0.002 | 0.995–1.000 | 1.000, 1.000, 0.995, 1.000, 0.998 |
| 8 | One feature: mean plaquette ZZZZ (exact) | 0.998 ± 0.002 | 0.995–1.000 | 1.000, 1.000, 0.995, 1.000, 0.998 |
| 8 | Fraction of test states with |g - g_c| < 0.1 | 0.222 ± 0.019 | 0.198–0.245 | 0.230, 0.230, 0.210, 0.198, 0.245 |
