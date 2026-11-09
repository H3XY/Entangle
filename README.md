# Entangle

An honest quantum machine-learning benchmark on data that is **quantum to begin with**.

The [AGI project](https://github.com/H3XY/AGI) squeezed text into 4 rotation angles
and found that a quantum classifier never beat classical models on it. That is
expected: when the data is classical, a classical computer already has all of it.
Here the inputs are quantum states, so a quantum model can act on them directly
while a classical model only ever sees measurement outcomes. That is where quantum
ML has a real argument. This repo tests the argument, with classical baselines
next to every quantum number.

## Results in one paragraph

On detecting entanglement, a **two-copy** QNN learns the SWAP test and is perfect
at 2–4 qubits. A one-copy QNN stays at chance however it is trained, because its
output is linear in the state. Generic classical learners on the full state vector
also do poorly with 200 examples. But the fair classical competitor, a learner on
the same number of random-Pauli measurement snapshots with purity features, is
close behind: 0.94–0.99 against the QNN's 0.995–1.000. On Ising phases everything
works (0.96–1.00) and the QCNN has no edge. On a simulated IBM device, noise breaks
quantum and classical models alike until both are recalibrated on 20 states
prepared on the device. A small run on real IBM hardware (`ibm_marrakesh`) behaved
the same way, with milder noise. **The quantum benefit we measure is real but small. It
comes from the two-copy measurement, not from the trainable circuit.** It shows
up as shot efficiency: with 64 copies of each test state, the two-copy QNN
classifies entanglement at 0.96, against 0.79 for classical shadows (stage 3).

## How it works

```
state-prep circuit ──► |psi>                     (the data: never written down as numbers for the QNN)
                        │
   quantum model:       ├─► trainable circuit ─► measure ─► P(class 1) ─► sigmoid(a(p − b)) ─► label
                        │   (1 copy, 2 copies, or a QCNN)
   classical baselines: ├─► measure in X/Y/Z bases ─► Pauli-expectation & purity estimates ─► sklearn
                        └─► full state vector (2^(n+1) numbers; not available in a real experiment)
```

- **Quantum input, no encoding.** The QNN's only access to a state is running a
  circuit on it and measuring. Training uses the exact output probabilities, the
  infinite-shot limit, computed by a small batched simulator
  ([entangle/sim.py](entangle/sim.py)) with adjoint gradients. Tests check it
  against the full unitary, against parameter-shift gradients, and against
  `qiskit-machine-learning`'s `SamplerQNN` run on the composed prep + model
  circuits.
- **Lessons reused from AGI.** Single-qubit or single-event readout, exact
  probabilities for training, L-BFGS-B on cross-entropy, several random restarts,
  and classical baselines on every table.
- **Lesson that did not transfer: data re-uploading.** You cannot re-upload an
  unknown quantum state, because no-cloning forbids copies. The quantum analogue
  is to ask for more copies of the state, and that is exactly what makes the
  two-copy model work.
- **Same copy budget for everyone.** At test time each model gets the same number
  of copies of each state. The 1-copy QNN uses them as shots. The 2-copy QNN uses
  2 per shot. The classical models use 1 per snapshot.

### Stage 1: entangled or product?

Random 2–4 qubit states come from a circuit of random local rotations and RZZ
entanglers ([entangle/states.py](entangle/states.py)). Product states have every
entangler angle at 0. Entangled states have Meyer–Wallach entanglement Q ≥ 0.2,
where Q is the mean linear entropy of single qubits and equals the squared
concurrence for 2 qubits. Local rotations are random, so both classes have the
same average state, and only entanglement tells them apart.

| Model | What it computes |
|---|---|
| QNN 1-copy | Tr(Mρ) for one trainable observable M. Linear in ρ, so it cannot separate the classes. |
| QNN 2-copy | A trainable 2-qubit block on each pair (qubit i, its copy), with class 1 if any pair reads 11. It can learn Bell-basis measurement: P(singlet on pair i) = (1 − Tr ρᵢ²)/2. |
| Classical, state vector | Logistic regression, RBF SVM and MLP on [Re ψ, Im ψ]. |
| Classical, snapshots | The same learners on all 4ⁿ−1 Pauli expectation estimates plus single-qubit purity estimates, or on the purities alone. |

![Stage 1](results/stage1_entanglement.png)

| Qubits | QNN 1-copy | QNN 2-copy | Classical, state vector (best) | Classical, snapshots (best) |
|---|---|---|---|---|
| 2 | 0.49 / 0.50 | 1.00 / 0.998 | 0.69 | 1.00 / 0.94 |
| 3 | 0.50 / 0.49 | 1.00 / 0.995 | 0.55 | 1.00 / 0.97 |
| 4 | 0.50 / 0.49 | 1.00 / 1.000 | 0.54 | 1.00 / 0.99 |

Each cell is exact / 256 copies per test state, with 200 training and 400 test
states. Full table: [results/stage1_entanglement.md](results/stage1_entanglement.md).

How it scales:
- The 2-copy QNN needs 2n qubits and 8n weights, and stays perfect.
- The full-state classical models get worse as n grows: 2^(n+1) inputs, a highly
  nonlinear target, and only 200 examples. The labels themselves are computed
  classically from the state vector, so this is not "classical can't". It is
  "generic learning from amplitudes is data-hungry". Stage 3 shows how hungry.
- The snapshot models get *better* with n on the purity features, because there
  are more qubits to average over. With all 4ⁿ−1 Pauli features they get worse,
  from too many noisy features.

### Stage 2: phases of the transverse-field Ising chain

Ground states of H = −Σ ZᵢZᵢ₊₁ − h Σ Xᵢ on open chains of 4, 6 and 8 qubits come
from exact diagonalisation in the even-parity sector, with h uniform in [0.1, 2].
The label is ordered (h < 1) vs disordered (h > 1).

- **QCNN** ([entangle/circuits.py](entangle/circuits.py)): convolution on
  neighbours, then pooling that halves the active qubits, repeated until one is
  left. Weights are shared per layer: 36 weights at 8 qubits.
- **Generic QNN**: the same readout with an unstructured ansatz (64 weights at 8 qubits).
- **Classical**: the same learners on the state vector, or on Z-basis and X-basis
  snapshots, turned into ⟨Zᵢ⟩, ⟨Xᵢ⟩, ⟨ZᵢZⱼ⟩ and ⟨XᵢXⱼ⟩ estimates.

![Stage 2](results/stage2_phases.png)

| Qubits | QCNN | Generic QNN | Classical, state vector (best) | Classical, Z/X snapshots (best) |
|---|---|---|---|---|
| 4 | 0.998 / 0.968 | 1.000 / 0.965 | 1.000 | 0.988 / 0.980 |
| 6 | 1.000 / 0.985 | 1.000 / 0.975 | 1.000 | 0.993 / 0.990 |
| 8 | 0.993 / 0.980 | 0.993 / 0.978 | 0.995 | 0.995 / 0.980 |

Each cell is exact / 256 copies. Every model is perfect away from the critical
point (|h − 1| ≥ 0.2). Full table: [results/stage2_phases.md](results/stage2_phases.md).

Ising phases are easy from ordinary Z/X measurements, because the ZZ correlations
give the answer away. The QCNN matches classical models and uses about half the
weights of the generic ansatz, but it has no accuracy advantage. One finding
from the build: the 3-parameter conv and pool units from the Qiskit QCNN tutorial
output exactly 0.5 on these spin-flip-symmetric states, so they cannot learn.
General RY/RZ + CX units fix that, and a regression test covers it.

### Stage 3: data efficiency

Accuracy against the number of training states (left, exact) and against the
number of copies of each test state (right, 200 training states). Each point is
the mean of 3 independent datasets, and the band is ±1 std.
Full tables: [results/stage3_efficiency.md](results/stage3_efficiency.md).

![Stage 3, entanglement](results/stage3_entanglement.png)

**Entanglement (4 qubits).** This is where the two-copy QNN earns its keep.
Measured per copy of the state, it is far more efficient than classical
snapshots:

| Copies per test state | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| QNN 2-copy | 0.68 | 0.85 | 0.96 | 0.99 | 1.00 |
| Classical, random-Pauli snapshots | 0.54 | 0.58 | 0.79 | 0.91 | 0.93 |

The reason: a Bell-pair measurement on two copies reads out a purity deficit in
one shot. Classical shadows must estimate ⟨X⟩², ⟨Y⟩² and ⟨Z⟩² from shots split
across random bases. Against training states, though, the classical snapshot
model learns *faster*: 0.83 vs 0.67 at 16 states. The QNN's non-convex training
only reliably finds the SWAP test from about 64 states upward. The 1-copy QNN and
the full-state classical models stay near chance even at 256 training states.

![Stage 3, Ising phases](results/stage3_phases.png)

**Ising phases (8 qubits).** All models reach about 0.92 from just 4 training
states and about 0.99 from 32. With few shots, classical Z/X snapshots do
better: 0.87 vs the QCNN's 0.79 at 4 copies. The ZZ correlations are directly
measurable, while the QCNN's single readout bit is a noisier summary. The curves
meet by 256 copies.

### Stage 4: hardware

Each test state is prepared on the device, then classified there.
- **Quantum models:** the prep circuit (once per copy) is followed by the trained
  model circuit.
- **Classical baseline:** the prep circuit is measured in fixed Pauli bases: all
  3ⁿ settings for entanglement, all-Z and all-X for Ising.

Models are trained in simulation. Noise shifts every model's outputs. For
example, noise makes product states look mixed, which reads as "entangled". So
the script also prepares 20 labelled calibration states on the device. The
quantum models refit only their 2-parameter readout calibration on them, and the
classical models are refit on the device snapshots.

![Stage 4 on a fake IBM backend](results/stage4_hardware_fake.png)

Local noisy simulation of `ibm_torino` (`FakeTorino`), 20 test states per task,
1024 copies per state per model:

| Task | QNN 1-copy | QNN 2-copy / QCNN | Classical, snapshots (best) |
|---|---|---|---|
| Entanglement, 2 qubits | 0.60 → 0.55 → 0.55 | 1.00 → 0.95 → 1.00 | 1.00 → 0.50 → 0.95 |
| Entanglement, 3 qubits | 0.55 → 0.50 → 0.60 | 1.00 → 0.50 → 1.00 | 1.00 → 0.55 → 1.00 |
| Ising phases, 4 qubits | – | 1.00 → 1.00 → 0.95 | 1.00 → 1.00 → 0.95 |

Each cell reads ideal → device → device recalibrated. The classical column takes
the best learner in each column. Without recalibration, the classical purity
models call every state "entangled", and so does the 6-qubit two-copy circuit.
Noise lowers measured purity exactly the way entanglement does. Twenty labelled
device states fix both. See
[results/stage4_hardware_fake.md](results/stage4_hardware_fake.md). With 20 test
states, one state is 0.05 of accuracy, and reruns can differ by about that much.
**Real hardware: `ibm_marrakesh`** (156-qubit IBM Heron, open plan, 8–9 Nov 2026).
A small run: 10 test states and 10 calibration states per task, 512 copies of
each state per model, 280 circuits in 5 jobs:

```bash
python stage4_hardware.py --qubits-entanglement 2 --states 10 --calibrate 10 --copies 512
```

![Stage 4 on ibm_marrakesh](results/stage4_hardware_ibm.png)

| Task | QNN 1-copy | QNN 2-copy / QCNN | Classical, snapshots (best) |
|---|---|---|---|
| Entanglement, 2 qubits | 0.50 → 0.50 → 0.50 | 1.00 → 0.90 → 1.00 | 1.00 → 0.90 → 1.00 |
| Ising phases, 4 qubits | – | 0.90 → 1.00 → 0.90 | 1.00 → 1.00 → 0.90 |

Each cell reads ideal → device → device recalibrated. See
[results/stage4_hardware_ibm.md](results/stage4_hardware_ibm.md).

- The real device was gentler than the FakeTorino noise model. Without any
  recalibration, the 2-copy QNN and the classical snapshot models both kept
  0.90 on entanglement, where the fake device dropped the classical models to
  0.50.
- Recalibrating restored entanglement to 1.00 for both. On Ising it *lowered*
  accuracy by one or two states, because refitting on only 10 labelled states
  can overfit.
- With 10 test states, one state is 0.10 of accuracy. These numbers show that
  the pipeline works end to end on hardware, and that quantum and classical
  models degrade similarly. They are too small to rank the models.

## Usage

Requires Python 3.11+ (developed on 3.12). Everything runs locally on CPU, and
nothing is downloaded beyond the pip packages.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python stage1_entanglement.py          # ~15 s   2-4 qubit entanglement classifier
python stage2_phases.py                # ~2 min  4-8 qubit Ising phases, QCNN
python stage3_efficiency.py            # ~9 min  accuracy vs training states and vs shots
python stage4_hardware.py --fake       # ~1 min  IBM device noise model, no account needed
python -m pytest -q tests              # correctness checks
./security/audit_deps.sh               # pip-audit scan -> results/dependency_audit.md
```

Each script takes `--help`. Examples: `--qubits 2 3`, `--copies 64`, `--seeds 5`,
`--task phases`.

### Real IBM Quantum hardware

Create an account at https://quantum.cloud.ibm.com, then either export your key:

```bash
export IBM_QUANTUM_TOKEN="<your API key>"
export IBM_QUANTUM_INSTANCE="<your instance CRN>"   # optional
```

or save it once with
`QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=..., set_as_default=True)`.
**Never commit tokens.** `.gitignore` excludes `.env` files and `qiskit-ibm.json`.

```bash
python stage4_hardware.py                              # least busy device; asks before submitting
python stage4_hardware.py --backend ibm_torino --states 10 --copies 512 --task entanglement
```

Results go to `results/stage4_hardware_ibm.*`. The full default run is about
1,720 circuits; the small run above is 280.

## Repo layout

```
stage1_entanglement.py   stage2_phases.py   stage3_efficiency.py   stage4_hardware.py
entangle/
  states.py      random product/entangled states, Meyer-Wallach Q, concurrence, Ising ground states
  circuits.py    1-copy QNN, 2-copy QNN, QCNN, generic ansatz (tied weights, readout masks)
  sim.py         batched state-vector simulator with adjoint gradients
  qnn.py         training (exact probabilities, L-BFGS-B, readout calibration) and shot sampling
  measure.py     Pauli snapshots (random, Z/X, fixed bases), feature estimates, measurement circuits
  classical.py   logistic regression / RBF SVM / MLP baselines
  report.py      JSON, Markdown and chart output
tests/           simulator vs full unitary, adjoint vs parameter shift, vs SamplerQNN, physics checks
security/        pip-audit dependency scan
results/         JSON, Markdown tables and charts for every stage, dependency audit
```

## Limitations

- **Small systems.** We test 2–4 qubits for entanglement (8 with two copies) and
  4–8 for Ising, all simulated exactly. Proven quantum advantages in learning
  from experiments (Huang et al., *Science* 2022) appear for properties such as
  *global* purity, where classical snapshots need exponentially many shots.
  Single-qubit purities, used here, are cheap for classical shadows too. That is
  why the classical gap is small.
- **The two-copy model is physics-informed.** Its pairwise blocks and
  "any pair reads 11" readout are designed so that the SWAP test is expressible.
  Training finds it reliably, but a generic ansatz on 2n qubits might not.
- **Two copies at once** need both copies in quantum memory together. That is
  harder than measuring copies one at a time, which is all the classical
  baseline needs.
- **Training is noise-free.** Models are trained on exact probabilities and only
  tested with shots and noise. On the device they need recalibration on a few
  labelled states, and so do the classical baselines.
- **Classical baselines are generic** (sklearn), but picking the best of them on
  the test set flatters the classical side. A hand-written classical estimator
  can compute Q from the state vector exactly, and that is how the labels are made.
- **Ising states are loaded with `StatePreparation`.** That is exact, but it is
  not how a real experiment would make them. For stage 4 we keep the chain at 4
  qubits so the circuits stay shallow.
- **Qiskit is pinned below 2**, as in AGI. `SamplerQNN`'s exact reference
  `Sampler` (used in the tests) is removed in Qiskit 2.
- **Stage 4 numbers are small-sample**: 20 test states on the fake backend and
  10 on real hardware. Only the smallest configuration (2-qubit entanglement,
  4-qubit Ising) has been run on a real device.

## License

MIT
