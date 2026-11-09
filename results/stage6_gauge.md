# Stage 6: Z2 lattice gauge theory (a toy, not Yang-Mills)

Infinite-lattice transition at g_c = 0.3285 (dual 2D quantum Ising model). Physical sector: Gauss's law at every vertex, no electric flux winding the torus.

## Energy gap and string tension

| L | g | gap E1 - E0 | W(1x1) | Creutz ratio | electric field (mean X) |
|---|---|---|---|---|---|
| 2 | 0.100 | 3.481 | 0.990 | – | 0.103 |
| 2 | 0.200 | 3.084 | 0.953 | – | 0.222 |
| 2 | 0.300 | 2.861 | 0.882 | – | 0.365 |
| 2 | 0.320 | 2.841 | 0.863 | – | 0.395 |
| 2 | 0.400 | 2.853 | 0.775 | – | 0.518 |
| 2 | 0.600 | 3.448 | 0.542 | – | 0.755 |
| 2 | 1.000 | 5.895 | 0.291 | – | 0.922 |
| 2 | 1.500 | 9.598 | 0.180 | – | 0.969 |
| 3 | 0.100 | 3.271 | 0.994 | -0.000 | 0.055 |
| 3 | 0.200 | 2.540 | 0.971 | -0.000 | 0.131 |
| 3 | 0.300 | 1.939 | 0.901 | -0.001 | 0.268 |
| 3 | 0.320 | 1.860 | 0.876 | -0.001 | 0.308 |
| 3 | 0.400 | 1.792 | 0.736 | 0.000 | 0.502 |
| 3 | 0.600 | 3.164 | 0.441 | 0.159 | 0.810 |
| 3 | 1.000 | 7.096 | 0.253 | 0.719 | 0.936 |
| 3 | 1.500 | 11.467 | 0.167 | 1.211 | 0.972 |
| 4 | 0.100 | 3.210 | 0.995 | 0.000 | 0.051 |
| 4 | 0.200 | 2.350 | 0.976 | 0.001 | 0.114 |
| 4 | 0.300 | 1.511 | 0.915 | 0.008 | 0.232 |
| 4 | 0.320 | 1.387 | 0.888 | 0.013 | 0.275 |
| 4 | 0.400 | 1.377 | 0.706 | 0.094 | 0.526 |
| 4 | 0.600 | 3.543 | 0.430 | 0.531 | 0.818 |
| 4 | 1.000 | 7.286 | 0.253 | 1.040 | 0.936 |
| 4 | 1.500 | 11.514 | 0.167 | 1.440 | 0.972 |

Minimum gap per lattice size:

| L | link qubits | min gap | at g |
|---|---|---|---|
| 2 | 8 | 2.829 | 0.360 |
| 3 | 18 | 1.767 | 0.380 |
| 4 | 32 | 1.266 | 0.360 |

## Classifying deconfined vs confined ground states (L = 2, 8 link qubits)

| Model | Test acc. (exact) | Exact, g at least 0.1 from g_c | Test acc. (256 copies) |
|---|---|---|---|
| QCNN | 1.000 | 1.000 | 0.963 |
| QNN (generic ansatz) | 1.000 | 1.000 | 0.975 |
| Logistic regression, full state vector | 1.000 | 1.000 | n/a |
| SVM (RBF), full state vector | 1.000 | 1.000 | n/a |
| MLP, full state vector | 1.000 | 1.000 | n/a |
| Logistic regression, Z/X snapshots | 0.980 | 1.000 | 0.980 |
| SVM (RBF), Z/X snapshots | 0.980 | 1.000 | 0.980 |
| MLP, Z/X snapshots | 0.968 | 0.997 | 0.968 |

For the snapshot models the exact and copies columns are the same run (256 snapshots per state).
