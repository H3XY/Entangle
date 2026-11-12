# Calibration of the physical qubits used: ibm_marrakesh vs FakeMarrakesh

Real rows: backend.target read right after each run (9 Nov 2026). FakeMarrakesh rows: the fake backend's stored calibration for the same physical qubits (its snapshot date is fixed in qiskit-ibm-runtime, not today's). The FakeMarrakesh results replay the identical transpiled circuits.

## stage 7 (purity)

| Source | Qubit | T1 (us) | T2 (us) | Readout error | sx error |
|---|---|---|---|---|---|
| ibm_marrakesh | 4 | 161 | 190 | 0.0225 | 3.57e-04 |
| ibm_marrakesh | 5 | 305 | 216 | 0.0078 | 5.74e-04 |
| ibm_marrakesh | 6 | 175 | 114 | 0.0005 | 1.78e-04 |
| ibm_marrakesh | 7 | 262 | 200 | 0.0176 | 2.12e-04 |
| ibm_marrakesh | 8 | 152 | 113 | 0.0054 | 2.94e-04 |
| ibm_marrakesh | 17 | 236 | 34 | 0.0063 | 4.42e-04 |
| FakeMarrakesh | 4 | 240 | 224 | 0.0195 | 3.53e-04 |
| FakeMarrakesh | 5 | 325 | 374 | 0.0056 | 7.47e-05 |
| FakeMarrakesh | 6 | 284 | 212 | 0.0027 | 1.92e-04 |
| FakeMarrakesh | 7 | 290 | 107 | 0.0054 | 1.52e-04 |
| FakeMarrakesh | 8 | 253 | 112 | 0.0063 | 9.06e-05 |
| FakeMarrakesh | 17 | 153 | 44 | 0.0105 | 1.42e-04 |

| Source | CZ pair | Error |
|---|---|---|
| ibm_marrakesh | cz 17-7 | 0.0021 |
| ibm_marrakesh | cz 4-5 | 0.0021 |
| ibm_marrakesh | cz 5-4 | 0.0021 |
| ibm_marrakesh | cz 5-6 | 0.0016 |
| ibm_marrakesh | cz 6-5 | 0.0016 |
| ibm_marrakesh | cz 6-7 | 0.0022 |
| ibm_marrakesh | cz 7-17 | 0.0021 |
| ibm_marrakesh | cz 7-6 | 0.0022 |
| ibm_marrakesh | cz 7-8 | 0.0028 |
| ibm_marrakesh | cz 8-7 | 0.0028 |
| FakeMarrakesh | cz 17-7 | 0.0026 |
| FakeMarrakesh | cz 4-5 | 0.0024 |
| FakeMarrakesh | cz 5-4 | 0.0024 |
| FakeMarrakesh | cz 5-6 | 0.0017 |
| FakeMarrakesh | cz 6-5 | 0.0017 |
| FakeMarrakesh | cz 6-7 | 0.0023 |
| FakeMarrakesh | cz 7-17 | 0.0026 |
| FakeMarrakesh | cz 7-6 | 0.0023 |
| FakeMarrakesh | cz 7-8 | 0.0022 |
| FakeMarrakesh | cz 8-7 | 0.0022 |

## stage 4 (entanglement + Ising)

| Source | Qubit | T1 (us) | T2 (us) | Readout error | sx error |
|---|---|---|---|---|---|
| ibm_marrakesh | 0 | 300 | 28 | 0.0200 | 5.65e-04 |
| ibm_marrakesh | 1 | 160 | 136 | 0.0239 | 3.30e-04 |
| ibm_marrakesh | 4 | 161 | 190 | 0.0225 | 3.57e-04 |
| ibm_marrakesh | 5 | 305 | 216 | 0.0078 | 5.74e-04 |
| ibm_marrakesh | 6 | 175 | 114 | 0.0005 | 1.78e-04 |
| ibm_marrakesh | 7 | 262 | 200 | 0.0176 | 2.12e-04 |
| ibm_marrakesh | 17 | 236 | 34 | 0.0063 | 4.42e-04 |
| FakeMarrakesh | 0 | 366 | 61 | 0.0112 | 1.35e-04 |
| FakeMarrakesh | 1 | 183 | 187 | 0.0054 | 1.03e-04 |
| FakeMarrakesh | 4 | 240 | 224 | 0.0195 | 3.53e-04 |
| FakeMarrakesh | 5 | 325 | 374 | 0.0056 | 7.47e-05 |
| FakeMarrakesh | 6 | 284 | 212 | 0.0027 | 1.92e-04 |
| FakeMarrakesh | 7 | 290 | 107 | 0.0054 | 1.52e-04 |
| FakeMarrakesh | 17 | 153 | 44 | 0.0105 | 1.42e-04 |

| Source | CZ pair | Error |
|---|---|---|
| ibm_marrakesh | cz 0-1 | 0.0027 |
| ibm_marrakesh | cz 1-0 | 0.0027 |
| ibm_marrakesh | cz 17-7 | 0.0021 |
| ibm_marrakesh | cz 4-5 | 0.0021 |
| ibm_marrakesh | cz 5-4 | 0.0021 |
| ibm_marrakesh | cz 5-6 | 0.0016 |
| ibm_marrakesh | cz 6-5 | 0.0016 |
| ibm_marrakesh | cz 6-7 | 0.0022 |
| ibm_marrakesh | cz 7-17 | 0.0021 |
| ibm_marrakesh | cz 7-6 | 0.0022 |
| FakeMarrakesh | cz 0-1 | 0.0017 |
| FakeMarrakesh | cz 1-0 | 0.0017 |
| FakeMarrakesh | cz 17-7 | 0.0026 |
| FakeMarrakesh | cz 4-5 | 0.0024 |
| FakeMarrakesh | cz 5-4 | 0.0024 |
| FakeMarrakesh | cz 5-6 | 0.0017 |
| FakeMarrakesh | cz 6-5 | 0.0017 |
| FakeMarrakesh | cz 6-7 | 0.0023 |
| FakeMarrakesh | cz 7-17 | 0.0026 |
| FakeMarrakesh | cz 7-6 | 0.0023 |
