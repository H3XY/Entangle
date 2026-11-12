# Stage 4: classifying states prepared on ibm_marrakesh

Real IBM Quantum hardware. 40 test states per task (balanced), 512 copies of each state per model. Ideal = noise-free simulation with the same number of shots. Recalibrated = after refitting on 16 labelled states prepared on the device (quantum models: only the 2-parameter readout calibration; classical models: refit on device snapshots).

| Task | Model | Shots per state | Ideal | ibm_marrakesh | ibm_marrakesh, recalibrated |
|---|---|---|---|---|---|
| entanglement, 2 qubits | QNN 2-copy | 256 | 1.000 | 0.825 | 1.000 |
| entanglement, 2 qubits | Logistic regression (snapshots) | 504 | 1.000 | 0.925 | 0.925 |
| entanglement, 2 qubits | SVM (RBF) (snapshots) | 504 | 0.975 | 0.900 | 0.925 |
| entanglement, 2 qubits | MLP (snapshots) | 504 | 0.975 | 0.900 | 0.975 |
| Ising phases, 4 qubits | QCNN | 512 | 0.950 | 0.900 | 0.900 |
| Ising phases, 4 qubits | Logistic regression (snapshots) | 512 | 0.975 | 0.925 | 0.950 |
| Ising phases, 4 qubits | SVM (RBF) (snapshots) | 512 | 0.950 | 0.950 | 0.875 |
| Ising phases, 4 qubits | MLP (snapshots) | 512 | 0.975 | 0.925 | 0.950 |
