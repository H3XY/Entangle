# The Yang–Mills mass gap: what it asks, what is known, and what this repo does (and doesn't) show

**Short version.** This repo contains no proof of the Yang–Mills mass gap, and
no attempt at one. The problem has been open since the 1970s. A credible
attack needs new mathematics in constructive quantum field theory, not more
computation. Stage 6 builds the simplest lattice gauge theory on a handful of
qubits, to make the ideas concrete. It shows how a gap can *close*, which is
why proving that it stays open is hard.

## What the problem asks

The Clay Mathematics Institute statement (Jaffe & Witten, 2000):

> For any compact simple gauge group G, prove that a non-trivial quantum
> Yang–Mills theory exists on ℝ⁴ and has a mass gap Δ > 0.

There are two parts, and both are hard.

1. **Existence.** Construct the quantum theory mathematically. It must be a
   well-defined quantum field theory in four spacetime dimensions, satisfying
   axioms at least as strong as the Wightman or Osterwalder–Schrader axioms.
   No interacting quantum field theory in four dimensions has ever been
   constructed this way.
2. **Mass gap.** Show that the energy spectrum above the vacuum starts at some
   Δ > 0. Every excitation (a "glueball") costs at least Δ, even though the
   classical theory has massless waves (gluons).

## Why it is hard

- **Perturbation theory says the opposite.** Expanding around the free theory
  gives massless gluons. The gap is a non-perturbative effect: no finite order
  of the standard expansion shows it.
- **Two regimes must be connected.** Yang–Mills theory is *asymptotically free*:
  its coupling shrinks at short distances. The continuum theory therefore lives
  at weak lattice coupling, while the gap and confinement are easiest to prove
  at strong lattice coupling. A proof must control the theory across the whole
  range, in infinite volume, and through the continuum limit.
- **Four dimensions are marginal.** The constructive program built interacting
  φ⁴ theories in 2 and 3 dimensions. In 4 dimensions, φ⁴ turns out to be
  *trivial*: its continuum limit is a free theory. That is a warning that 4D
  constructions can collapse.
- **Any finite lattice has a gap.** A finite lattice has finitely many states
  per site, so its spectrum is discrete and the gap is positive automatically.
  All numerical results, including ours, are of this kind. The content of the
  problem is a gap that stays bounded away from zero *uniformly* as the volume
  goes to infinity and the lattice spacing goes to zero.

## What is known

**Rigorous results** (partial, and not for the full 4D continuum theory):
- **Strong coupling.** Lattice gauge theories at strong coupling confine and
  have a mass gap. Wilson loops obey an area law, from convergent cluster
  expansions (Osterwalder & Seiler 1978). This is the lattice regime *far* from
  the continuum limit.
- **2D Yang–Mills is exactly solvable** (Migdal 1975; Witten 1991). But in two
  dimensions there are no propagating gluons, so it is a much simpler theory.
- **Balaban's renormalization-group program** (1980s) proved ultraviolet
  stability bounds for lattice Yang–Mills in 3 and 4 dimensions, in finite
  volume. It is one of the deepest partial results toward existence, but it
  does not construct the infinite-volume theory or prove a gap.
- **Constructive QFT.** φ⁴ was constructed in 2D and 3D (Glimm, Jaffe and
  others). Triviality of φ⁴ in 4D (Aizenman 1981, Fröhlich 1982, and
  Aizenman & Duminil-Copin 2021 for the marginal case) shows how delicate 4D is.
- **A probabilistic line of work** on lattice gauge theories, led by Sourav
  Chatterjee and others since the mid-2010s, has given new rigorous results on
  Wilson loops and large-N limits. It has not given a 4D continuum gap.

**Strong non-rigorous evidence:**
- **Lattice Monte Carlo** (from Creutz 1980 onward) finds no bulk phase
  transition between strong and weak coupling for SU(2) and SU(3) in 4D. That
  suggests the provably gapped strong-coupling regime connects smoothly to the
  continuum.
- **Glueball masses** have been computed on the lattice (for example Morningstar
  & Peardon 1999 for SU(3)). Physics strongly supports confinement: no free
  gluons are observed.

## What stage 6 does

The model is Z2 lattice gauge theory in 2+1 dimensions, on L × L tori with
L = 2, 3, 4. There is one qubit per link: 8, 18 and 32 qubits. It is the
Kogut–Susskind Hamiltonian form of Wegner's 1971 model.

- **Gauss's law** is built in. Physical states satisfy it at every vertex, and
  the tests check this.
- **Wegner duality** maps the physical sector exactly to the 2D quantum Ising
  model. The tests check this against direct simulation of the link qubits.
- **Confinement.** At large coupling g the Creutz ratio, an estimate of the
  string tension, turns positive and Wilson loops decay with area. At small g
  the string tension is zero (deconfined, topologically ordered phase).
- **The gap.** Both phases have a gap that settles as L grows. At g = 0.1 it
  goes 3.48 → 3.27 → 3.21 for L = 2, 3, 4, and at g = 1.0 it goes
  5.90 → 7.10 → 7.29. Near the transition
  (g_c ≈ 0.33) the minimum gap shrinks from 2.83 to 1.77 to 1.27 for L = 2, 3, 4,
  consistent with it closing in infinite volume.
- **Phase classification** (8 qubits) works with the QCNN and with classical
  learners on Z/X snapshots alike, at 0.96–1.00. As in stage 2, there is no
  quantum edge.

The lesson that carries over: **a mass gap is not automatic.** In this toy
theory the gap closes at a phase transition. The conjectured picture for 4D
SU(N) is that *no* such transition separates the provably gapped
strong-coupling regime from the continuum. Turning that picture into a theorem,
in infinite volume and in the continuum, is the problem.

## What stage 6 does not show

- **The gauge group is wrong.** Z2 is abelian and discrete: there are no
  gluons, no gauge fields with self-interaction of the Yang–Mills kind, no
  asymptotic freedom and no continuum limit with a running coupling.
- **The dimension is wrong.** It is 2+1, not 3+1.
- **The sizes are tiny.** 4 × 4 is the largest lattice, and finite lattices
  always have a gap (see above).
- **Numbers are not proofs.** Even perfect numerics on any finite lattice
  cannot establish a uniform infinite-volume, continuum-limit bound.

## Where quantum computers fit

Hamiltonian lattice gauge theory is one of the main targets of quantum
simulation. Real-time dynamics and theories at finite density suffer from the
sign problem in classical Monte Carlo, and quantum hardware could simulate
them directly. That would give *numerical evidence*: spectra, string breaking,
dynamics. It would not give proofs. A proof of the mass gap is a mathematical
achievement, and no amount of simulation substitutes for it.

## References

From memory, for orientation. Check each one before citing it formally.

- A. Jaffe, E. Witten, *Quantum Yang–Mills theory*, Clay Mathematics Institute
  Millennium Problem description (2000).
- K. G. Wilson, *Confinement of quarks*, Phys. Rev. D 10, 2445 (1974).
- J. Kogut, L. Susskind, *Hamiltonian formulation of Wilson's lattice gauge
  theories*, Phys. Rev. D 11, 395 (1975).
- F. J. Wegner, *Duality in generalized Ising models and phase transitions
  without local order parameters*, J. Math. Phys. 12, 2259 (1971).
- S. Elitzur, *Impossibility of spontaneously breaking local symmetries*,
  Phys. Rev. D 12, 3978 (1975).
- K. Osterwalder, E. Seiler, *Gauge field theories on a lattice*, Ann. Phys.
  110, 440 (1978).
- M. Creutz, *Monte Carlo study of quantized SU(2) gauge theory*, Phys. Rev. D
  21, 2308 (1980).
- E. Fradkin, S. Shenker, *Phase diagrams of lattice gauge theories with Higgs
  fields*, Phys. Rev. D 19, 3682 (1979).
- E. Witten, *On quantum gauge theories in two dimensions*, Commun. Math. Phys.
  141, 153 (1991).
- J. Glimm, A. Jaffe, *Quantum Physics: A Functional Integral Point of View*
  (Springer, 2nd ed. 1987).
- C. Morningstar, M. Peardon, *The glueball spectrum from an anisotropic
  lattice study*, Phys. Rev. D 60, 034509 (1999).
- H. W. J. Blöte, Y. Deng, *Cluster Monte Carlo simulation of the transverse
  Ising model*, Phys. Rev. E 66, 066110 (2002) (source of g_c).
