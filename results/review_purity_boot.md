# Stage 7 copies-for-90%: hierarchical bootstrap per job

1000 replicates per job. Each replicate resamples the 60 test and 20 calibration states, resamples each state's raw device shots with replacement, subsamples c copies without replacement, refits the 2-parameter recalibration on the resampled calibration states, and interpolates the 90% threshold (log2-linear). Medians with 95% percentile intervals. Shadows = 2-parameter readout on the shadow global-purity estimate (same flexibility as the QNN).

Calibration shot budget (primary): the calibration states use the same c as the test states.

| Source | QNN copies for 90% | Shadow copies for 90% | Ratio shadows / QNN | One-sided 95% upper bound on ratio | P(ratio < noise-free median) | P(ratio < noise-free ratio), both uncertain | 95% CI of log2(device / noise-free ratio) |
|---|---|---|---|---|---|---|---|
| Noise-free, same states and design | 16 [8, 40] | 108 [60, 181] | 6.25 [2.48, 14.87] | – | – | – | – |
| ibm_marrakesh run 1 (dbp0rd0ou4as73csr3t0) | 48 [24, 91] | 128 [64, 256] | 2.89 [1.26, 7.13] | 6.50 | 0.937 | 0.866 | [-2.83, 0.84] |
| ibm_marrakesh run 2 (dbp0s1lt7fqc7388jap0) | 55 [27, 108] | 128 [80, 256] | 2.54 [1.06, 6.28] | 5.66 | 0.973 | 0.902 | [-3.07, 0.63] |

Noise-free ratio used for the comparison: 6.25 (median of the noise-free replicates; same states, design, budget, recalibration and interpolation rule as the device runs).

Stage 5 simulation at n = 2 (different test states; 3-feature best-of-3 shadows) through the same interpolation: QNN 18.5, shadows 97.7, ratio 5.29 (the power-of-2 grid gives 32 vs 128 = 4).

## Sensitivity: calibration on the full device budget

Calibration states at 1,024 copies, test states at c (300 replicates).

| Source | QNN copies for 90% | Shadow copies for 90% | Ratio | P(ratio < noise-free ratio) |
|---|---|---|---|---|
| ibm_marrakesh run 1 | 42 [22, 81] | 128 [62, 210] | 3.09 [1.32, 6.91] | 0.947 |
| ibm_marrakesh run 2 | 49 [25, 86] | 128 [85, 256] | 2.97 [1.32, 7.40] | 0.960 |
