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
the same way, with milder noise. On those tasks, which depend on single-qubit
properties, **the quantum benefit is real but small**. It shows up as shot
efficiency: with 64 copies of each test state, the two-copy QNN classifies
entanglement at 0.96, against 0.79 for classical shadows (stage 3).

**On a global property the gap becomes exponential (stage 5).** To tell pure states
from mixed ones at 90% accuracy, the best single-copy classical method needs 128
copies of each state at 2 qubits and 4,096 at 6 qubits. That is roughly double
per added qubit, as theory predicts. The two-copy QNN needs 16–32 copies at
every size. In every stage the advantage comes from measuring two copies
together, not from the trainable circuit.

**Stage 6** is a toy lattice gauge theory: Z2 gauge theory with one qubit per
link. It shows confinement and an energy gap that closes only at a phase
transition. It is *not* Yang–Mills theory and says nothing toward the
Millennium problem's proof; [docs/YANG_MILLS.md](docs/YANG_MILLS.md) explains
what that problem asks and what is known.

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

### Stage 5: a global property (pure or mixed?)

Stages 1–3 use labels that depend on single-qubit properties, which classical
shadows estimate cheaply. Stage 5 tests the case where theory proves an
exponential separation (Huang, Kueng & Preskill 2021; Chen, Cotler, Huang & Li
2022): telling **Haar-random pure states** from **random rank-2 mixed states**
(Tr ρ² between 0.5 and 0.58), on 2–6 qubits. Both classes average to the
maximally mixed state, and their single-qubit marginals become alike as n grows.
Only the purity of the whole state separates them.

- **QNN 2-copy (parity).** The same pairwise blocks as stage 1, with class 1 if an
  *odd* number of pairs read 11. The SWAP of the whole register is the product of
  the pair SWAPs, so with Bell-basis blocks P(class 1) = (1 − Tr ρ²)/2 (checked in
  `tests/`).
- **Classical shadows.** Random-Pauli snapshots turned into an unbiased estimate of
  the *global* purity, plus single-qubit purities, with the best of 3 learners.
  This is the best known single-copy strategy. A rank-4 factorisation makes the
  estimator about 6ⁿ·4n operations instead of S²·n.
- Mixed states are handled as purifications (ρ = AA†) in the simulator and the
  snapshot sampler.

![Stage 5](results/stage5_purity.png)

Copies of each test state needed for 90% test accuracy (200 training states,
mean of 3 seeds):

| Qubits | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| QNN 2-copy (learned SWAP test) | 32 | 16 | 32 | 32 | 32 |
| Classical shadows (best of 3) | 128 | 256 | 512 | 1,024 | 4,096 |

At 6 qubits and 64 copies, the QNN scores 0.98 and classical shadows 0.55. The
one-copy QNN is at chance (0.47–0.52) at every size and every copy count.
Full tables: [results/stage5_purity.md](results/stage5_purity.md).

**Training ran into a barren plateau.** With a separate trainable block per
pair, training from a random start works up to 4 qubits but fails at 5 in one of
three seeds (0.85 ± 0.21). The parity readout multiplies contributions from every
pair, so random-start gradients shrink exponentially with n, the textbook barren
plateau of global cost functions. Two fixes work:
- sharing one block across all pairs (1.00 at 5 qubits from scratch)
- training that block at 2 qubits and transferring it to larger n (1.00 at every
  size, with 30 fine-tuning steps)

The transferred model is the one in the table. So the exponential advantage at
*measurement* time comes with a *training* hazard that has to be designed around.

### Stage 6: a toy lattice gauge theory (not Yang–Mills)

Z2 lattice gauge theory in 2+1 dimensions on an L × L torus, one qubit per link
(8, 18 and 32 qubits for L = 2, 3, 4) ([entangle/gauge.py](entangle/gauge.py)):

    H = − Σ_plaquettes Z Z Z Z − g Σ_links X

Physical states obey Gauss's law at every vertex. Spectra come from Wegner
duality, an exact map to the 2D quantum Ising model with one spin per
plaquette. The tests check it against direct simulation of the link qubits.
The infinite-lattice transition is at g_c ≈ 0.3285.

![Stage 6](results/stage6_gauge.png)

| | L = 2 | L = 3 | L = 4 |
|---|---|---|---|
| Gap at g = 0.1 (deconfined) | 3.48 | 3.27 | 3.21 |
| Gap at g = 1.0 (confined) | 5.90 | 7.10 | 7.29 |
| Minimum gap (near g_c) | 2.83 | 1.77 | 1.27 |
| Creutz ratio (string tension) at g = 1.0 | – | 0.72 | 1.04 |

- **Both phases have a gap** that settles as the lattice grows.
- **Near the transition the gap shrinks** with every size, consistent with it
  closing in infinite volume.
- **Confinement switches on past g_c.** The Creutz ratio, an estimate of the
  string tension, turns positive there, and Wilson loops decay with the
  enclosed area.
- **Phase classification** of the 8-qubit ground states works for the QCNN
  (1.00 exact, 0.96 with 256 copies) and for classical learners on Z/X
  snapshots of plaquettes and electric fields (0.97–0.98) alike. As with Ising,
  there is no quantum edge.

Full tables: [results/stage6_gauge.md](results/stage6_gauge.md).

The connection to Yang–Mills is conceptual only. A mass gap is not automatic:
here it closes at a phase transition. The conjecture for 4D SU(N) is that no
such transition separates the provably gapped strong-coupling regime from the
continuum. Z2 has no gluons, no asymptotic freedom and no continuum limit, and
every finite lattice has a gap anyway. See
[docs/YANG_MILLS.md](docs/YANG_MILLS.md) for the problem statement, the known
rigorous results, and why no computation here amounts to progress on a proof.

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
python stage5_purity.py                # ~35 min pure vs mixed, 2-6 qubits, copies-needed scaling
python stage6_gauge.py                 # ~1 min  Z2 lattice gauge theory: gap, confinement, phases
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
stage1_entanglement.py ... stage6_gauge.py    one CLI per stage
docs/YANG_MILLS.md       the Millennium problem: statement, known results, what stage 6 does not show
entangle/
  states.py      product/entangled states, Meyer-Wallach Q, concurrence, Ising ground states, pure/mixed states
  circuits.py    1-copy QNN, 2-copy QNN, QCNN, generic ansatz (tied weights, readout masks)
  sim.py         batched state-vector simulator with adjoint gradients
  qnn.py         training (exact probabilities, L-BFGS-B, readout calibration), mixed-state batches, shots
  measure.py     Pauli snapshots, feature estimates, shadow global-purity estimator, measurement circuits
  classical.py   logistic regression / RBF SVM / MLP baselines
  gauge.py       Z2 lattice gauge theory: lattice, Wegner-dual solver, Wilson loops, link-qubit states
  report.py      JSON, Markdown and chart output
tests/           simulator vs full unitary, adjoint vs parameter shift, vs SamplerQNN, physics checks
security/        pip-audit dependency scan
results/         JSON, Markdown tables and charts for every stage, dependency audit
```

## Limitations

- **Small systems.** We test 2–4 qubits for entanglement (8 with two copies),
  4–8 for Ising and 2–6 for purity (12 with two copies), all simulated exactly.
  The exponential trend in stage 5 is measured over five sizes. It matches the
  theory, but it is not a proof of scaling, which the cited papers provide.
- **Stage 5 states are Haar-random,** chosen because the theory is cleanest for
  them. Preparing Haar-random states needs circuits whose depth grows
  exponentially, so stage 5 is simulation only. A hardware version would use
  shallow random circuits (approximate designs) instead.
- **The two-copy model is physics-informed.** Its pairwise blocks and its
  readouts ("any pair reads 11", "an odd number of pairs read 11") are designed
  so that the SWAP test is expressible. Training finds it, with weight sharing or
  transfer for the global readout. A generic ansatz on 2n qubits might not.
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
