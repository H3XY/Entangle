# Stage 8 hardware plan

Mean ± std over random pure and mixed states (count per class in the JSON), transpiled (optimization level 1) for FakeMarrakesh. Estimated seconds: a stage-7-sized run of 80 states (60 test + 20 calibration) at 1024 copies each (512 QNN shots + 3^n shadow settings x floor(1024 / 3^n) shots), using the per-shot rate measured in this project's billed jobs (entangle/ibm.py).

| Qubits | Depth | QNN 2q gates (pure / mixed) | QNN depth (pure / mixed) | Shadow 2q gates (pure / mixed) | Physical qubits | Circuits | Shots | Est. quantum s |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 10.1 ± 1.4 / 10.7 ± 1.5 | 31.1 ± 2.9 / 33.3 ± 3.0 | 2.0 ± 0.0 / 2.0 ± 0.0 | 6 | 800 | 122320 | 37.0 |
| 2 | 2 | 14.0 ± 0.0 / 14.0 ± 0.0 | 41.0 ± 0.0 / 42.0 ± 0.0 | 3.0 ± 0.0 / 3.0 ± 0.0 | 6 | 800 | 122320 | 37.0 |
| 2 | 4 | 18.0 ± 0.0 / 18.0 ± 0.0 | 53.0 ± 0.0 / 54.0 ± 0.0 | 5.0 ± 0.0 / 5.0 ± 0.0 | 6 | 800 | 122320 | 37.0 |
| 3 | 1 | 12.4 ± 1.2 / 12.1 ± 1.4 | 30.8 ± 3.6 / 30.7 ± 4.3 | 2.0 ± 0.0 / 2.0 ± 0.0 | 8 | 2240 | 120880 | 36.6 |
| 3 | 2 | 21.0 ± 0.0 / 21.0 ± 0.0 | 45.9 ± 6.6 / 45.3 ± 7.4 | 3.0 ± 0.0 / 3.0 ± 0.0 | 8 | 2240 | 120880 | 36.6 |
| 3 | 4 | 26.9 ± 1.4 / 27.1 ± 1.4 | 53.6 ± 0.5 / 54.4 ± 0.5 | 5.0 ± 0.0 / 5.0 ± 0.0 | 8 | 2240 | 120880 | 36.6 |
| 4 | 1 | 17.8 ± 2.4 / 18.4 ± 2.2 | 32.4 ± 4.5 / 34.6 ± 4.2 | 3.0 ± 0.0 / 3.0 ± 0.0 | 11 | 6560 | 118720 | 36.0 |
| 4 | 2 | 30.4 ± 4.5 / 32.9 ± 5.2 | 52.8 ± 8.5 / 57.0 ± 8.7 | 4.0 ± 0.0 / 4.0 ± 0.0 | 11 | 6560 | 118720 | 36.0 |
| 4 | 4 | 44.0 ± 1.4 / 44.0 ± 1.4 | 62.8 ± 11.4 / 63.8 ± 11.4 | 7.0 ± 0.0 / 7.0 ± 0.0 | 10 | 6560 | 118720 | 36.0 |
