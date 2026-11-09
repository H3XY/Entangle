# Stage 4: classifying states prepared on fake_torino

Local noisy simulation of the device (fake backend). 20 test states per task (balanced), 1024 copies of each state per model. Ideal = noise-free simulation with the same number of shots. Recalibrated = after refitting on 20 labelled states prepared on the device (quantum models: only the 2-parameter readout calibration; classical models: refit on device snapshots).

| Task | Model | Shots per state | Ideal | fake_torino | fake_torino, recalibrated |
|---|---|---|---|---|---|
| entanglement, 2 qubits | QNN 1-copy | 1024 | 0.600 | 0.550 | 0.550 |
| entanglement, 2 qubits | QNN 2-copy | 512 | 1.000 | 0.950 | 1.000 |
| entanglement, 2 qubits | Logistic regression (snapshots) | 1017 | 1.000 | 0.500 | 0.900 |
| entanglement, 2 qubits | SVM (RBF) (snapshots) | 1017 | 1.000 | 0.500 | 0.950 |
| entanglement, 2 qubits | MLP (snapshots) | 1017 | 0.950 | 0.500 | 0.900 |
| entanglement, 3 qubits | QNN 1-copy | 1024 | 0.550 | 0.500 | 0.600 |
| entanglement, 3 qubits | QNN 2-copy | 512 | 1.000 | 0.500 | 1.000 |
| entanglement, 3 qubits | Logistic regression (snapshots) | 999 | 1.000 | 0.550 | 1.000 |
| entanglement, 3 qubits | SVM (RBF) (snapshots) | 999 | 1.000 | 0.500 | 1.000 |
| entanglement, 3 qubits | MLP (snapshots) | 999 | 1.000 | 0.500 | 1.000 |
| Ising phases, 4 qubits | QCNN | 1024 | 1.000 | 1.000 | 0.950 |
| Ising phases, 4 qubits | Logistic regression (snapshots) | 1024 | 1.000 | 1.000 | 0.950 |
| Ising phases, 4 qubits | SVM (RBF) (snapshots) | 1024 | 1.000 | 0.900 | 0.950 |
| Ising phases, 4 qubits | MLP (snapshots) | 1024 | 1.000 | 0.950 | 0.950 |
