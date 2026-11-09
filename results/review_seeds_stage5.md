# Seed-to-seed uncertainty, stage 5

Pure vs mixed (stage 5): per-seed copies needed (smallest power of 2 with accuracy at least 90% for that seed) next to the grid definition used in the stage 5 table (smallest power of 2 where the mean over seeds reaches 90%). Seeds: 0, 1, 2. Each seed redraws training and test states, shot noise and model initialisations. Std is the sample standard deviation over seeds. 'Best of k' picks the best classical learner on the test set, which flatters the classical side.

| Qubits | Quantity | Mean ± std over seeds | Range | Per seed |
|---|---|---|---|---|
| 2 | QNN 2-copy, shared, transferred from 2 qubits: copies for 90% | 32 (grid) | 16–64 | 64, 16, 16 |
| 2 | Classical shadows (purity estimates): copies for 90% | 128 (grid) | 128–128 | 128, 128, 128 |
| 2 | QNN 2-copy, shared, transferred from 2 qubits: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 2 | QNN 2-copy, per-pair blocks, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 2 | QNN 2-copy, shared block, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 2 | QNN 1-copy: exact accuracy | 0.512 ± 0.038 | 0.485–0.555 | 0.485, 0.495, 0.555 |
| 3 | QNN 2-copy, shared, transferred from 2 qubits: copies for 90% | 16 (grid) | 16–16 | 16, 16, 16 |
| 3 | Classical shadows (purity estimates): copies for 90% | 256 (grid) | 256–256 | 256, 256, 256 |
| 3 | QNN 2-copy, shared, transferred from 2 qubits: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 3 | QNN 2-copy, per-pair blocks, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 3 | QNN 2-copy, shared block, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 3 | QNN 1-copy: exact accuracy | 0.521 ± 0.043 | 0.492–0.570 | 0.492, 0.500, 0.570 |
| 4 | QNN 2-copy, shared, transferred from 2 qubits: copies for 90% | 32 (grid) | 16–32 | 32, 16, 16 |
| 4 | Classical shadows (purity estimates): copies for 90% | 512 (grid) | 512–512 | 512, 512, 512 |
| 4 | QNN 2-copy, shared, transferred from 2 qubits: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 4 | QNN 2-copy, per-pair blocks, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 4 | QNN 2-copy, shared block, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 4 | QNN 1-copy: exact accuracy | 0.482 ± 0.011 | 0.475–0.495 | 0.475, 0.495, 0.477 |
| 5 | QNN 2-copy, shared, transferred from 2 qubits: copies for 90% | 32 (grid) | 16–32 | 16, 32, 16 |
| 5 | Classical shadows (purity estimates): copies for 90% | 1024 (grid) | 1024–2048 | 2048, 1024, 2048 |
| 5 | QNN 2-copy, shared, transferred from 2 qubits: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 5 | QNN 2-copy, per-pair blocks, from scratch: exact accuracy | 0.852 ± 0.257 | 0.555–1.000 | 0.555, 1.000, 1.000 |
| 5 | QNN 2-copy, shared block, from scratch: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 5 | QNN 1-copy: exact accuracy | 0.522 ± 0.025 | 0.495–0.545 | 0.545, 0.525, 0.495 |
| 6 | QNN 2-copy, shared, transferred from 2 qubits: copies for 90% | 32 (grid) | 16–64 | 64, 32, 16 |
| 6 | Classical shadows (purity estimates): copies for 90% | 4096 (grid) | 2048–4096 | 2048, 4096, 4096 |
| 6 | QNN 2-copy, shared, transferred from 2 qubits: exact accuracy | 1.000 ± 0.000 | 1.000–1.000 | 1.000, 1.000, 1.000 |
| 6 | QNN 1-copy: exact accuracy | 0.474 ± 0.032 | 0.438–0.495 | 0.495, 0.490, 0.438 |
