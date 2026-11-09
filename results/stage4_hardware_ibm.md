# Stage 4: classifying states prepared on ibm_marrakesh

Real IBM Quantum hardware. 10 test states per task (balanced), 512 copies of each state per model. Ideal = noise-free simulation with the same number of shots. Recalibrated = after refitting on 10 labelled states prepared on the device (quantum models: only the 2-parameter readout calibration; classical models: refit on device snapshots).

| Task | Model | Shots per state | Ideal | ibm_marrakesh | ibm_marrakesh, recalibrated |
|---|---|---|---|---|---|
| entanglement, 2 qubits | QNN 1-copy | 512 | 0.500 | 0.500 | 0.500 |
| entanglement, 2 qubits | QNN 2-copy | 256 | 1.000 | 0.900 | 1.000 |
| entanglement, 2 qubits | Logistic regression (snapshots) | 504 | 1.000 | 0.900 | 1.000 |
| entanglement, 2 qubits | SVM (RBF) (snapshots) | 504 | 1.000 | 0.900 | 1.000 |
| entanglement, 2 qubits | MLP (snapshots) | 504 | 1.000 | 0.900 | 1.000 |
| Ising phases, 4 qubits | QCNN | 512 | 0.900 | 1.000 | 0.900 |
| Ising phases, 4 qubits | Logistic regression (snapshots) | 512 | 1.000 | 1.000 | 0.900 |
| Ising phases, 4 qubits | SVM (RBF) (snapshots) | 512 | 1.000 | 1.000 | 0.900 |
| Ising phases, 4 qubits | MLP (snapshots) | 512 | 1.000 | 1.000 | 0.800 |
