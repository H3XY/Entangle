# Stage 7 hardware purity test: recalibration, copy accounting, intervals

Recomputed offline from the saved device counts (results/ibm_raw). FakeMarrakesh rows replay the identical transpiled circuits (same physical qubits 4, 6, 5 / 8, 17, 7) with the same shots.

## Data and copy accounting

- Training: make_purity_data(2, 200, seed=0) (simulation only). Test: make_purity_data(2, 60, seed=1). Calibration: make_purity_data(2, 20, seed=2). All balanced pure / mixed, drawn independently; smallest distance ||rho_cal - rho_other|| between a calibration state and any training or test state: 0.318 (so none coincide).
- c copies = c preparations of the state for both methods: the QNN uses c/2 shots of a circuit holding 2 copies; shadows use floor(c/9) shots in each of 9 bases. On the device: 512 QNN shots (1,024 preparations) and 9 x 113 shadow shots (1,017) per state. Smaller c are subsampled without replacement from those shots.
- Recalibration refits, on the 20 device calibration states: QNN, a logistic regression on p (2 parameters); shadows, global purity, the same (2 parameters); shadows, 3 features, a logistic regression on [global, local, local] purity estimates (4 parameters). 'Before' = the model trained in simulation, nothing refitted.

## ibm_marrakesh, run 1

Mean over 30 shot subsamples; brackets: 95% bootstrap interval over the 60 test states.

| Method | Recalibration | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | before | 0.794 [0.76, 0.83] | 0.877 [0.85, 0.90] | 0.937 [0.92, 0.95] | 0.984 [0.98, 0.99] | 0.998 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | before | – | 0.677 [0.65, 0.71] | 0.803 [0.77, 0.83] | 0.905 [0.88, 0.93] | 0.961 [0.94, 0.98] | 0.993 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | before | – | 0.670 [0.63, 0.71] | 0.807 [0.77, 0.85] | 0.901 [0.87, 0.93] | 0.963 [0.95, 0.98] | 0.978 [0.95, 0.99] | 1.000 [1.00, 1.00] |
| QNN 2-copy | after | 0.777 [0.76, 0.80] | 0.869 [0.85, 0.89] | 0.932 [0.92, 0.95] | 0.981 [0.97, 0.99] | 0.996 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | after | – | 0.673 [0.65, 0.70] | 0.813 [0.79, 0.84] | 0.908 [0.89, 0.93] | 0.968 [0.95, 0.98] | 0.989 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | after | – | 0.653 [0.62, 0.69] | 0.783 [0.75, 0.81] | 0.870 [0.85, 0.89] | 0.956 [0.93, 0.97] | 0.989 [0.98, 1.00] | 1.000 [1.00, 1.00] |

Uncertainty over the choice of calibration states (mean [2.5%, 97.5%]): bootstrap of the 20 calibration states / random 20-60 re-split of all 80 device states:

| Method | Scheme | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | bootstrap 20 | 0.759 [0.58, 0.87] | 0.847 [0.73, 0.93] | 0.930 [0.85, 0.98] | 0.976 [0.92, 1.00] | 0.996 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | re-split 80 | 0.764 [0.62, 0.87] | 0.859 [0.77, 0.95] | 0.933 [0.85, 0.98] | 0.980 [0.93, 1.00] | 0.997 [0.97, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | bootstrap 20 | – | 0.638 [0.40, 0.77] | 0.793 [0.67, 0.90] | 0.899 [0.80, 0.98] | 0.970 [0.92, 1.00] | 0.994 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | re-split 80 | – | 0.640 [0.43, 0.78] | 0.797 [0.68, 0.90] | 0.907 [0.82, 0.98] | 0.973 [0.92, 1.00] | 0.995 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | bootstrap 20 | – | 0.631 [0.43, 0.77] | 0.755 [0.57, 0.88] | 0.865 [0.70, 0.97] | 0.958 [0.88, 1.00] | 0.993 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | re-split 80 | – | 0.639 [0.47, 0.77] | 0.778 [0.63, 0.87] | 0.888 [0.77, 0.97] | 0.967 [0.92, 1.00] | 0.993 [0.97, 1.00] | 1.000 [1.00, 1.00] |

Copies for 90% (log2-linear interpolation of the 'after' curve; 95% bootstrap interval) and ratio to the QNN:

| Method | Copies for 90% | Ratio to QNN |
|---|---|---|
| QNN 2-copy | 45 [33, 56] | 1 |
| Shadows, global purity (2 params) | 121 [102, 157] | 2.69 [2.01, 3.98] |
| Shadows, 3 features (4 params) | 163 [126, 201] | 3.63 [2.56, 5.13] |

## ibm_marrakesh, run 2

Mean over 30 shot subsamples; brackets: 95% bootstrap interval over the 60 test states.

| Method | Recalibration | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | before | 0.768 [0.74, 0.80] | 0.864 [0.84, 0.89] | 0.932 [0.92, 0.95] | 0.971 [0.96, 0.98] | 0.993 [0.99, 1.00] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | before | – | 0.654 [0.62, 0.68] | 0.811 [0.78, 0.84] | 0.909 [0.89, 0.93] | 0.956 [0.94, 0.97] | 0.994 [0.99, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | before | – | 0.666 [0.63, 0.71] | 0.808 [0.77, 0.85] | 0.912 [0.89, 0.94] | 0.963 [0.95, 0.98] | 0.985 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | after | 0.748 [0.73, 0.77] | 0.853 [0.83, 0.87] | 0.918 [0.90, 0.94] | 0.972 [0.96, 0.98] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | after | – | 0.647 [0.62, 0.68] | 0.804 [0.78, 0.83] | 0.915 [0.90, 0.93] | 0.978 [0.97, 0.99] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | after | – | 0.647 [0.61, 0.68] | 0.787 [0.76, 0.82] | 0.914 [0.90, 0.93] | 0.974 [0.96, 0.98] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] |

Uncertainty over the choice of calibration states (mean [2.5%, 97.5%]): bootstrap of the 20 calibration states / random 20-60 re-split of all 80 device states:

| Method | Scheme | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | bootstrap 20 | 0.733 [0.50, 0.85] | 0.831 [0.70, 0.93] | 0.908 [0.82, 0.98] | 0.970 [0.92, 1.00] | 0.996 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | re-split 80 | 0.741 [0.58, 0.85] | 0.839 [0.73, 0.93] | 0.919 [0.85, 0.98] | 0.972 [0.92, 1.00] | 0.996 [0.97, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | bootstrap 20 | – | 0.620 [0.35, 0.77] | 0.789 [0.67, 0.88] | 0.890 [0.78, 0.97] | 0.973 [0.93, 1.00] | 0.997 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | re-split 80 | – | 0.635 [0.45, 0.77] | 0.794 [0.67, 0.90] | 0.900 [0.82, 0.97] | 0.975 [0.92, 1.00] | 0.998 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | bootstrap 20 | – | 0.613 [0.38, 0.77] | 0.761 [0.62, 0.90] | 0.866 [0.70, 0.95] | 0.965 [0.92, 1.00] | 0.995 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | re-split 80 | – | 0.635 [0.48, 0.75] | 0.775 [0.63, 0.88] | 0.881 [0.77, 0.95] | 0.965 [0.88, 1.00] | 0.997 [0.98, 1.00] | 1.000 [1.00, 1.00] |

Copies for 90% (log2-linear interpolation of the 'after' curve; 95% bootstrap interval) and ratio to the QNN:

| Method | Copies for 90% | Ratio to QNN |
|---|---|---|
| QNN 2-copy | 53 [42, 69] | 1 |
| Shadows, global purity (2 params) | 117 [103, 138] | 2.20 [1.63, 2.91] |
| Shadows, 3 features (4 params) | 119 [105, 143] | 2.24 [1.65, 3.05] |

## FakeMarrakesh (identical circuits), run 1

Mean over 30 shot subsamples; brackets: 95% bootstrap interval over the 60 test states.

| Method | Recalibration | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | before | 0.803 [0.76, 0.84] | 0.894 [0.87, 0.92] | 0.957 [0.94, 0.97] | 0.992 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | before | – | 0.665 [0.64, 0.69] | 0.812 [0.78, 0.84] | 0.924 [0.90, 0.94] | 0.976 [0.96, 0.99] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | before | – | 0.682 [0.65, 0.71] | 0.824 [0.79, 0.86] | 0.936 [0.92, 0.95] | 0.974 [0.96, 0.99] | 0.995 [0.99, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | after | 0.803 [0.78, 0.82] | 0.892 [0.88, 0.91] | 0.957 [0.95, 0.97] | 0.992 [0.99, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | after | – | 0.666 [0.64, 0.69] | 0.804 [0.78, 0.82] | 0.923 [0.90, 0.94] | 0.978 [0.97, 0.99] | 0.998 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | after | – | 0.658 [0.62, 0.69] | 0.794 [0.76, 0.83] | 0.909 [0.89, 0.93] | 0.968 [0.96, 0.98] | 0.997 [0.99, 1.00] | 1.000 [1.00, 1.00] |

Uncertainty over the choice of calibration states (mean [2.5%, 97.5%]): bootstrap of the 20 calibration states / random 20-60 re-split of all 80 device states:

| Method | Scheme | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | bootstrap 20 | 0.800 [0.72, 0.88] | 0.880 [0.77, 0.97] | 0.954 [0.87, 1.00] | 0.988 [0.95, 1.00] | 0.999 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | re-split 80 | 0.803 [0.70, 0.90] | 0.887 [0.80, 0.97] | 0.954 [0.87, 1.00] | 0.989 [0.95, 1.00] | 0.999 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | bootstrap 20 | – | 0.624 [0.38, 0.77] | 0.802 [0.67, 0.92] | 0.910 [0.82, 0.98] | 0.974 [0.92, 1.00] | 0.997 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | re-split 80 | – | 0.629 [0.45, 0.75] | 0.802 [0.68, 0.90] | 0.908 [0.82, 0.97] | 0.973 [0.93, 1.00] | 0.996 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | bootstrap 20 | – | 0.627 [0.40, 0.77] | 0.772 [0.55, 0.88] | 0.889 [0.75, 0.98] | 0.965 [0.90, 1.00] | 0.995 [0.97, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | re-split 80 | – | 0.645 [0.48, 0.77] | 0.787 [0.67, 0.90] | 0.899 [0.78, 0.97] | 0.965 [0.90, 1.00] | 0.993 [0.97, 1.00] | 1.000 [1.00, 1.00] |

Copies for 90% (log2-linear interpolation of the 'after' curve; 95% bootstrap interval) and ratio to the QNN:

| Method | Copies for 90% | Ratio to QNN |
|---|---|---|
| QNN 2-copy | 35 [29, 42] | 1 |
| Shadows, global purity (2 params) | 112 [98, 134] | 3.20 [2.55, 4.17] |
| Shadows, 3 features (4 params) | 121 [104, 153] | 3.46 [2.69, 4.78] |

## FakeMarrakesh (identical circuits), run 2

Mean over 30 shot subsamples; brackets: 95% bootstrap interval over the 60 test states.

| Method | Recalibration | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | before | 0.798 [0.75, 0.84] | 0.877 [0.85, 0.90] | 0.959 [0.95, 0.97] | 0.998 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | before | – | 0.702 [0.68, 0.73] | 0.823 [0.80, 0.85] | 0.928 [0.91, 0.95] | 0.978 [0.97, 0.99] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | before | – | 0.698 [0.67, 0.73] | 0.828 [0.79, 0.86] | 0.927 [0.90, 0.95] | 0.978 [0.97, 0.99] | 0.996 [0.99, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | after | 0.786 [0.76, 0.81] | 0.874 [0.86, 0.89] | 0.952 [0.94, 0.96] | 0.996 [0.99, 1.00] | 0.999 [1.00, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | after | – | 0.700 [0.68, 0.73] | 0.824 [0.81, 0.84] | 0.921 [0.90, 0.94] | 0.979 [0.97, 0.99] | 0.997 [0.99, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | after | – | 0.688 [0.66, 0.72] | 0.802 [0.77, 0.83] | 0.903 [0.88, 0.92] | 0.972 [0.96, 0.98] | 0.998 [1.00, 1.00] | 1.000 [1.00, 1.00] |

Uncertainty over the choice of calibration states (mean [2.5%, 97.5%]): bootstrap of the 20 calibration states / random 20-60 re-split of all 80 device states:

| Method | Scheme | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| QNN 2-copy | bootstrap 20 | 0.795 [0.68, 0.90] | 0.876 [0.77, 0.97] | 0.953 [0.88, 1.00] | 0.988 [0.95, 1.00] | 0.999 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| QNN 2-copy | re-split 80 | 0.800 [0.68, 0.88] | 0.887 [0.80, 0.97] | 0.959 [0.90, 1.00] | 0.990 [0.95, 1.00] | 1.000 [0.98, 1.00] | 1.000 [1.00, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | bootstrap 20 | – | 0.632 [0.45, 0.78] | 0.799 [0.70, 0.88] | 0.905 [0.80, 0.98] | 0.974 [0.92, 1.00] | 0.998 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, global purity (2 params) | re-split 80 | – | 0.648 [0.42, 0.78] | 0.804 [0.68, 0.90] | 0.908 [0.80, 0.97] | 0.976 [0.92, 1.00] | 0.998 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | bootstrap 20 | – | 0.633 [0.43, 0.77] | 0.771 [0.58, 0.88] | 0.887 [0.75, 0.97] | 0.974 [0.92, 1.00] | 0.997 [0.98, 1.00] | 1.000 [1.00, 1.00] |
| Shadows, 3 features (4 params) | re-split 80 | – | 0.654 [0.50, 0.78] | 0.794 [0.67, 0.90] | 0.896 [0.78, 0.98] | 0.970 [0.90, 1.00] | 0.997 [0.98, 1.00] | 1.000 [1.00, 1.00] |

Copies for 90% (log2-linear interpolation of the 'after' curve; 95% bootstrap interval) and ratio to the QNN:

| Method | Copies for 90% | Ratio to QNN |
|---|---|---|
| QNN 2-copy | 40 [32, 47] | 1 |
| Shadows, global purity (2 params) | 110 [95, 134] | 2.74 [2.21, 3.63] |
| Shadows, 3 features (4 params) | 125 [104, 160] | 3.10 [2.38, 4.28] |
