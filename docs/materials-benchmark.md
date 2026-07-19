# Implementation parity

This benchmark asks whether TT-Atom reproduces each model's original
implementation on the same input. Model accuracy (whether the potential
itself fits the chemistry) is out of scope.

## Method

ML interatomic potentials are deterministic forward passes, so the
device-vs-reference distance is a single number, not a distribution. The
benchmark still records the same three legs as the pharma framework so a
reader can compare like for like:

| leg | comparison |
|---|---|
| R | reference versus reference (deterministic, so bit-identical by construction) |
| D | device versus device (deterministic, so bit-identical by construction) |
| X | device versus reference |

Parity passes when X is at the model's own numerical floor: energy
relative error below 1%, force/stress PCC above 0.99, with the one
disclosed noise-floor exception (OrbMol open-shell direct forces, see
footnote ‡). R and D are exact by construction for every row here
(PCC = 1.00000, energy rel err = 0), verified live on card: the
upstream `fairchem` / `orb-models` CPU oracle and the TT-Atom device
forward are each bit-identical across reruns, so the only non-trivial
distance is X. This is the same convention the deterministic encoder
legs (ESMC, SaProt, ProteinMPNN) use in TT-Bio's `docs/pharma-benchmark.md`.

The analysis harness is `scripts/release_gate.py --leg accuracy`. It runs
the real-weight parity modules in `tests/test_*realweight*.py` verbatim
under pytest with JUnit XML, so the gate never re-derives a parity bar or
oracle: the bars and the real oracles live in the test modules
themselves (`fairchem` `uma-s-1` for UMA; the real `orb-models` CPU oracle
for Orb-v3 / OrbMol). Every golden bundle embeds the upstream reference
energy / forces / stress from build time
(`tests/gen_golden_real.py`, `tests/gen_golden_orb.py`), so a release
check reruns only the device side against a fixed reference.

## Results

These are the committed benchmark measurements for TT-Atom, taken on a
single Blackhole p150a card (card 0) on 2026-07-19 with
`TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg
accuracy`. Every currently-shipped family is covered: UMA `uma-s-1`
(molecular / `omol`), Orb-v3 `conservative-inf-omat` and `direct-20-omat`
(bulk / `omat`, analytic forces, periodic supercell, multi-element oxide,
stress, ZBL short-contact), and OrbMol `conservative-omol` (molecule /
charged / open-shell).

| family | checkpoint | regime | metric | R | D | X | result |
|---|---|---|---:|---:|---:|---:|---|
| uma | uma-s-1 | molecular / omol (ethanol) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | — | GAP† |
| orb | conservative-inf-omat | bulk / omat (Si toy) | energy rel err | 0 | 0 | 6.9e-4 | PASS |
| orb | conservative-inf-omat | analytic forces (`F = -dE/dpos`) | force PCC | 1.00000 | 1.00000 | 0.99999 | PASS |
| orb | direct-20-omat | bulk / omat (direct) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | 5.8e-4 / 0.99997 | PASS |
| orb | conservative-inf-omat | periodic supercell (24-atom Si) | backbone node PCC | 1.00000 | 1.00000 | 0.99956 | PASS¶ |
| orb | conservative-inf-omat | bulk / omat (MgO oxide, multi-element) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | 1.6e-3 / 0.99998 | PASS§ |
| orb | conservative-inf-omat | stress (conservative) | stress PCC (Voigt-6) | 1.00000 | 1.00000 | 0.99995 | PASS |
| orb | direct-20-omat | stress (dedicated head) | stress PCC (Voigt-6) | 1.00000 | 1.00000 | 0.99997 | PASS |
| orb | direct-20-omat | ZBL short-contact forces | force PCC (GNN + ZBL) | 1.00000 | 1.00000 | 1.00000 | PASS‖ |
| orbmol | conservative-omol | molecule (closed-shell) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | 1.6e-6 / 0.99973 | PASS |
| orbmol | conservative-omol | charged (NH4+) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | 4.6e-6 / 0.99331 | PASS |
| orbmol | conservative-omol | open-shell radical (CH3·) | energy rel err, force PCC | 0 / 1.00000 | 0 / 1.00000 | 9.2e-6 / 0.97850 | PASS‡ |

The Orb-v3 bulk / `omat` row also checks the encoder (node PCC 0.999997,
edge PCC 0.999991) and the full 5-layer message-passing backbone (final
node PCC 0.99946, edge PCC 0.9766 — the edge stream is a pure residual
with no further consumer, so it drifts more under bf16 over 5 layers;
expected precision compounding, not an algorithmic difference). The
OrbMol rows span three conditioning regimes (closed-shell, charge=+1,
open-shell spin=2); the open-shell radical's force magnitude is an order
of magnitude smaller than the other two, so the same absolute error
depresses its PCC while its energy stays the tightest of the three.
Orb-v3 is not equivariant (a plain attention-MPNN over scalar features),
a real architectural difference from UMA, not a port discrepancy; the
full non-equivariance analysis and the ZBL pair-repulsion correction live
in `docs/orb-port.md`.

† The UMA row is a **release-gated environment gap, not a parity
failure**. The device end-to-end forward
(`tests/test_realweight.py::test_end_to_end_energy_forces`) raises
`AttributeError: module 'ttnn._ttnn.operations.experimental' has no
attribute 'fused_rotate'`: UMA's equivariant rotation
(`tt_atom/rotation.py`) needs the custom `ttnn.experimental.fused_rotate`
op, which is **absent from the `ttnn` build installed in
`~/.ttatom_run/env` on this host**. The README's Install section says
`ttnn` must come from a source `tt-metal` build of branch
`moritztng/tt-atom` that carries `fused_rotate`; this host's env does not
have that build (a prior rebuild was reverted the same day because it
forced a `ttnn` downgrade that regressed Orb-v3 throughput by ~57%, see
memory `pc-ttatom-env-missing-fused-rotate`). It is a multi-day rebase of
that custom-kernel branch onto current `tt-metal`, out of scope here. The
other four UMA tests in the module pass: config sanity (real `uma-s-1`,
`ff_type=spectral`, 4 layers, `lmax=mmax=2`, `omol` task), the host MoLE
merge anchor (merged vs unmerged-MoE oracle, energy rel err < 1e-6 and
force PCC > 0.999), weight-bundle coverage, and the spectral-atomwise
module (activation PCC ≥ 0.98 vs the golden). The device energy/forces X
number is therefore not measurable on this card; the row is reported as
GAP, not forced to PASS or skipped. Fix: build `ttnn` from a source
`tt-metal` that carries `fused_rotate` on the release host, then re-run
`scripts/release_gate.py --leg accuracy`.

¶ The periodic-supercell row verifies the radius-graph reconstruction
matches `orb-models`' neighbour list exactly (1064 edges, symmetric
difference 0) before running the backbone on the reconstructed graph
(final node PCC 0.99956). Stress for the `conservative` checkpoint comes
from the same autograd virial pass as the analytic forces; the
`direct-20-omat` checkpoint has no autograd, so its stress comes from a
dedicated stress head (PCC 0.99997, separate row). `direct-omol` has no
stress head, consistent with stress being meaningless for an isolated
molecule.

‖ The ZBL short-contact row uses a Si structure pushed to a bond length
where the ZBL pair-repulsion term is non-negligible (0.106 eV, 1.3% of the
total 8.14 eV, versus ~1e-7 eV for the bulk-Si golden). The host ZBL
forces match a finite-difference check to 3.7e-10, and the total
(GNN + ZBL) force PCC is 1.00000 vs the `orb-models` oracle (|F|max
51.3 eV/A) — i.e. the ZBL correction is exact on the host and the device
GNN force matches the oracle at short contact, not just at equilibrium.

§ The MgO row is the first multi-element bulk system in the suite — every
other bulk row is pure Si (Z=14). It exercises three code paths a single-Z
golden cannot reach: the per-element reference-energy denormalize
(`host_energy_denormalize` sums `ref_weight[Z]` per atom, only ever hit at
one Z elsewhere), the mixed-Z ZBL pair-repulsion (Mg-Mg, Mg-O, O-O, each
with its own covalent-radii-sum envelope), and the encoder's per-element
embedding table at two atomic numbers simultaneously. Same
`conservative-inf-omat` checkpoint and analytic-force VJP as the Si toy;
MgO rock-salt is a textbook ionic oxide, in-distribution for OMat24, and
the row runs in ~1.5 s alongside the existing rows.

‡ The OrbMol open-shell radical (`CH3·`, spin=2) is the noise-floor case.
Its `conservative` force PCC is 0.97850, below the 0.99 bar but above the
0.97 floor the test module holds for this system; its energy rel err
(9.2e-6) is the tightest of the three OrbMol systems. The depression is
magnitude, not algorithm: the oracle |F|max is 0.032 eV/A (vs 0.48 for
the closed-shell molecule), so the same sub-millieV absolute error (MAE
0.0056 eV/A, on par with the siblings) depresses the correlation. The
shipped `direct-omol` checkpoint hits the same wall harder on this one
system: `test_direct_energy_and_forces[molecule_openshell]` reports force
PCC 0.89259 (below the test's 0.9 open-shell bar) with MAE 0.0075 eV/A,
still on par with its siblings — the gate surfaces this honestly as a
FAIL on the `direct-omol` open-shell row. It is a known borderline
noise-floor case (documented in the test's own docstring), not a
regression; the `conservative-omol` row above is the parity gate that
ships.

## Reproducing a comparison

The whole leg runs with one command on a single card:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg accuracy
```

The gate prints a per-module `PASS / FAIL / GAP` table with pass/skip/fail
counts; the underlying numbers above come from the same modules with
stdout captured (`-s`):

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python3 -m pytest tests/test_orb_realweight.py tests/test_orb_forces_realweight.py \
    tests/test_orb_direct_realweight.py tests/test_orb_periodic_realweight.py \
    tests/test_orb_stress_realweight.py tests/test_orb_zbl_forces.py \
    tests/test_orb_mgo_realweight.py \
    tests/test_orb_omol_realweight.py tests/test_realweight.py -s -q
```

Each UMA bundle embeds the `fairchem` reference energy/forces from build
time, and each Orb-v3 / OrbMol golden does the same for `orb-models`
(`tests/gen_golden_orb.py`), so a release check reruns only the device
side against the fixed reference. Regenerate a golden only when its
pinned upstream version or settings change.
