# Gradient variance of the stage 5 purity models (barren-plateau check)

Signal D(w) = mean P(class 1) over mixed states minus over pure states, on 16 stage 5 states. Weights drawn uniformly in [-pi, pi) (100 draws), exact gradients. Mean ± std over 3 data seeds.

| Qubits | Model | Weights | Var[dD/dw], mean over weights | Var[D] | Fraction of starts with gradient norm < 1e-3 |
|---|---|---|---|---|---|
| 2 | per-pair blocks | 16 | 9.52e-04 ± 1.0e-04 | 3.46e-03 ± 2.2e-04 | 0.00 |
| 2 | shared block | 8 | 2.60e-03 ± 1.8e-04 | 5.31e-03 ± 3.7e-04 | 0.00 |
| 3 | per-pair blocks | 24 | 2.52e-04 ± 2.7e-06 | 9.64e-04 ± 7.4e-05 | 0.00 |
| 3 | shared block | 8 | 1.18e-03 ± 4.6e-05 | 1.50e-03 ± 1.5e-04 | 0.00 |
| 4 | per-pair blocks | 32 | 6.55e-05 ± 3.3e-06 | 2.68e-04 ± 3.6e-05 | 0.00 |
| 4 | shared block | 8 | 5.58e-04 ± 6.2e-05 | 5.64e-04 ± 8.6e-05 | 0.00 |
| 5 | per-pair blocks | 40 | 1.73e-05 ± 6.5e-07 | 7.52e-05 ± 5.5e-06 | 0.00 |
| 5 | shared block | 8 | 3.49e-04 ± 6.0e-05 | 1.88e-04 ± 4.4e-05 | 0.00 |
| 6 | per-pair blocks | 48 | 4.16e-06 ± 8.1e-08 | 2.02e-05 ± 6.7e-07 | 0.00 |
| 6 | shared block | 8 | 2.23e-04 ± 9.0e-05 | 1.42e-04 ± 8.2e-05 | 0.00 |
| 7 | per-pair blocks | 56 | 1.04e-06 ± 3.1e-08 | 4.15e-06 ± 9.3e-07 | 0.00 |
| 7 | shared block | 8 | 1.58e-04 ± 9.4e-05 | 6.51e-05 ± 4.0e-05 | 0.00 |

Fitted decay of the gradient variance (log2-linear fit over n):

| Model | log2 slope per qubit | shrink factor per qubit | Per-seed slopes | 95% CI over seeds (t) | 95% CI, bootstrap over draws |
|---|---|---|---|---|---|
| per-pair blocks | -1.97 | 3.91 | -1.95, -1.94, -2.01 | [-2.06, -1.87] | [-1.98, -1.95] |
| shared block | -0.80 | 1.74 | -1.05, -0.76, -0.79 | [-1.26, -0.47] | [-1.01, -0.69] |

The fit is over n = 2, 3, 4, 5, 6, 7 only: it describes an exponential-like decay over the measured range and does not establish the asymptotic rate.
