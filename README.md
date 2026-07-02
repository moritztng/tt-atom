# TT-Atom

**Run Meta's [UMA](https://huggingface.co/facebook/UMA) machine-learning interatomic potential on
[Tenstorrent](https://tenstorrent.com) — a drop-in replacement for the fairchem ASE calculator.**

TT-Atom runs the UMA / eSCN-MD backbone — energy **and conservative analytic forces** — fully
device-resident on Tenstorrent via [`ttnn`](https://github.com/tenstorrent/tt-metal), behind a
clean [ASE](https://wiki.fysik.dtu.dk/ase/) `Calculator` that mirrors fairchem's. Molecules **and**
periodic materials, validated against the released `uma-s-1` checkpoint. Moving off fairchem is a
one-line change.

## Migrating from fairchem — one line

fairchem:

```python
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

calc = FAIRChemCalculator(load_predict_unit("uma-s-1.pt"), task_name="omol")
atoms.calc = calc
```

TT-Atom (same ASE surface, runs on the Tenstorrent card):

```python
from tt_atom import TTAtomCalculator

calc = TTAtomCalculator("uma_s_ethanol.npz", task_name="omol")   # a TT-Atom bundle
atoms.calc = calc
```

Everything downstream — `atoms.get_potential_energy()`, `get_forces()`, ASE optimizers, MD — is
unchanged. The one difference is the model file: instead of a fairchem `.pt` you pass a **TT-Atom
bundle**, produced once with `tt-atom convert-checkpoint` (below). The bundle is specific to a
composition + charge/spin + task because UMA's MoLE experts are merged at convert time (the exact
released inference path for a fixed composition) — ideal for a relaxation or MD run, where those
are constant.

## Quickstart

Two environments are required because `ttnn` needs `numpy<2` while `fairchem` needs `numpy>=2`:

```bash
# 1) runtime env (ttnn + TT-Atom) — where you run relaxations/MD
pip install -e .                     # numpy<2, torch (CPU), ase; install ttnn separately (below)

# 2) reference env (only to convert a checkpoint / regenerate goldens)
python -m venv refenv && refenv/bin/pip install "fairchem-core>=2.10"   # numpy>=2, SEPARATE venv
```

**Installing ttnn.** `ttnn` is the Tenstorrent runtime; it is not on PyPI. Install the `ttnn` /
`tt-metal` wheel that matches your card and `tt-kmd` driver (see the
[tt-metal releases](https://github.com/tenstorrent/tt-metal)). `import tt_atom` never imports
`ttnn`, so the package installs and imports fine on a machine without a card.

Convert a UMA checkpoint to a bundle (**in the reference env**), then use it (**in the runtime env**):

```bash
# convert (reference env): MoLE-merge uma-s-1 for a composition + task, embed a reference for verify
refenv/bin/python tools/export_weights.py --uma-s-1 \
    --molecule CH3CH2OH --task omol --charge 0 --spin 1 --out uma_s_ethanol.npz

tt-atom verify uma_s_ethanol.npz          # runtime env: device parity vs the embedded reference
tt-atom relax  uma_s_ethanol.npz --molecule CH3CH2OH --trace
tt-atom md     uma_s_ethanol.npz --molecule CH3CH2OH --steps 200 --temp 300
```

Or from Python:

```python
from ase.build import molecule
from ase.optimize import FIRE
from tt_atom import TTAtomCalculator

atoms = molecule("CH3CH2OH")
atoms.info.update(charge=0, spin=1)
atoms.calc = TTAtomCalculator("uma_s_ethanol.npz", trace=True)   # trace = ~2x on the MD/relax loop
FIRE(atoms).run(fmax=0.05)                                       # analytic forces -> real relaxation
print(atoms.get_potential_energy())
```

Examples: [`relax.py`](examples/relax.py), [`md.py`](examples/md.py),
[`periodic.py`](examples/periodic.py) (a bulk crystal), [`batch.py`](examples/batch.py) (multi-card).

## Accuracy — validated against fairchem uma-s-1

TT-Atom does **not** ship weights (the `facebook/UMA` checkpoints are gated under the FAIR
Chemistry License — bring your own). What is validated is that the device model reproduces the
**released `uma-s-1`** checkpoint via the official fairchem reference. Measured on a single
Blackhole **p150**, reproducible with `tests/test_realweight.py` (molecules) and
`tests/test_periodic.py` (materials):

| task | system | graph | energy rel. err | force PCC | force MAE (eV/Å) |
|---|---|---|---:|---:|---:|
| **omol** | ethanol | aperiodic | **1.8e-7** | **0.99958** | 3.4e-3 |
| **omat** | bulk Si (diamond) | periodic `[T,T,T]` | **3.0e-4** | **0.99999** | 6.5e-3 |
| **oc20** | Cu(100)+H slab | mixed `[T,T,F]` | **8.6e-5** | **1.00000** | 9.7e-4 |

All meet the drop-in bar (energy rel < 1e-3, force PCC > 0.99). The periodic neighbour list
reproduces fairchem's `radius_graph_pbc` **edge-for-edge** (same edges, same image offsets).
`odac` and `omc` use the identical data-driven path (a per-dataset token + per-task energy
normalizer read from the checkpoint) — enable them by exporting with `--task odac` / `--task omc`.

**MoLE merge anchor:** the host expert-merge reproduces the unmerged 32-expert MoE oracle to
energy rel 1.3e-12 / force PCC 1.0 — the merge is exact, not an approximation.

**Model coverage.** `uma-s-1` (lmax=mmax=2) is the validated default. `uma-m-1p1` exports cleanly
but is **not supported**: it uses `lmax=4/mmax=2` spherical-harmonic coefficient subselection, a
code path TT-Atom does not implement; the calculator raises a clear error rather than mis-running.

## How it maps cleanly to Tenstorrent

eSCN/eSEN replaces the irregular Clebsch–Gordan tensor products of typical equivariant nets with
the **SO(2) convolution trick**: after rotating each edge into its local frame with a Wigner-D
matrix, the SO(3) tensor product collapses into **per-order (per-`m`) dense GEMMs**. ~85–90% of the
compute is then plain matmul — what the hardware wants — while the equivariant "hard part" (Wigner
construction, the radius graph) is <1% of the work and stays on host.

- **`so2.py`** — SO(2) convolution as flat per-`m` 2-D GEMMs in a tile-aligned `[E, 9·C]` layout.
- **`rotation.py`** — per-edge Wigner rotation as a **sparse multiply-accumulate** over its fixed
  nonzero pattern, one launch over all edges (replacing a launch-bound batched matmul).
- **`geometry.py`** — host radius graph, aperiodic **and** cell-aware minimum-image (PBC).
- **`forces.py`** — **analytic** `F = −dE/dx` by a hand-written reverse pass through the device
  graph; the cheap geometric `d(Wigner)/dx` is finished by `torch.autograd` on host. Not finite differences.
- **`trace.py`** — trace-captured, device-resident forward+backward for the MD/relaxation loop.
- **`model.py` / `device.py`** — full residency, program cache, `bf16` weights with `HiFi4` + `fp32`
  dest accumulation (`packer_l1_acc`), matmul PCC ≈ 1.0 vs torch.
- **`batch.py`** — one-process-per-card fan-out for multi-card throughput.
- **`disjoint.py`** — disjoint-union (block-diagonal) graph batching: concatenate K systems into
  one graph and evaluate them in a *single* device forward (the fairchem/PyG way).

## Performance (measured, p150 Blackhole)

**Single-eval device engine vs 16-thread PyTorch CPU** (full config, random weights — the
architecture is weight-independent). The device latency is nearly flat in system size, so the
speedup grows with the system:

| atoms | edges | TT device (ms) | CPU 16-thr (ms) | device speedup |
|------:|------:|---------------:|----------------:|---------------:|
| 54  | 786  | 27.1 | 28.4 | 1.1× |
| 128 | 2234 | 29.0 | 70.6 | **2.4×** |
| 250 | 4834 | 35.1 | 184.7 | **5.3×** |

**Trace-captured MD/relaxation loop (real uma-s-1, ethanol).** Profiling shows the device
forward+backward is ~96% of a step and is host-*dispatch*-bound at these graph sizes, so capturing
the device op-stream once and replaying it (`trace=True`) is the e2e lever:

| path | per step (E+F) | FIRE relaxation |
|---|---:|---|
| eager | 86 ms | 46 steps, 119.5 ms/step |
| **traced** | **40 ms** (2.14×) | identical 46-step trajectory, same final E, **51.4 ms/step (2.33×)** |

Forces from the traced path are **bit-for-bit** the eager analytic forces — the trace only removes
host dispatch. `tt-atom relax --trace` / `md --trace` use it.

**Multi-card throughput** scales near-linearly — **3.95× on 4 cards** (validated on a 4-card
QuietBox, `qb1`; the fan-out path runs on any card count, single-card here).

**Single-card batched throughput (many small systems).** For a small molecule the device forward
is *dispatch*-bound: one ethanol (9 atoms) costs ~31 ms end-to-end almost all of which is host
overhead (build geometry, upload, launch, read back), so one-at-a-time is flat at ~32 systems/s no
matter how you slice it. Disjoint-union batching (below) pays that overhead once for K systems:

| K (ethanol) | one-at-a-time | batched | speedup |
|---:|---:|---:|---:|
| 2 | 32 sys/s | 56 sys/s | 1.7× |
| 8 | 33 sys/s | 181 sys/s | 5.6× |
| 32 | 32 sys/s | 413 sys/s | **12.8×** |
| 64 | 32 sys/s | 439 sys/s | **13.5×** |

Crossover is K = 2 (K = 1 is parity), and throughput peaks around K = 32–64 then declines
(366/283/207 sys/s at K = 128/256/512): the scatter-add is a dense matmul `S[N, E]` and the
block-diagonal batch makes it `[ΣN, ΣE]`, which grows **O(K²)** (both nodes and edges scale with
K), so the wasted zeros eventually dominate. No OOM through K = 512 on one p150 (ΣN = 4608,
ΣE = 36864, ~680 MB of scatter matrices); the useful batch is ≲ 64 for tiny molecules (and
correspondingly fewer for larger ones). All numbers measured on one p150, real uma-s-1, energy-only.

Honest notes: a `--fast` (`bfloat8_b`) mode exists but gives no speedup (the forward is
data-movement bound, not flop bound) and slightly worse accuracy — `bf16` is the default. Rotation
is the device hotspot (~60% of a block) and already flat-layout; its cost is dispatch, recovered by
trace, not compute headroom.

## Batched inference (disjoint-union, many small systems)

To evaluate many independent small systems on one card, concatenate them into a single
block-diagonal graph and run one device forward — exactly how fairchem/PyG batch (`Batch.from_data_list`):
no leading batch dimension, one big graph, per-system energies recovered by a segment-sum. Every
eSCN-MD op is per-node or per-edge, so the whole backbone is batch-transparent once each system's
edges carry a node offset; only the energy readout changes (sum → segment-sum) and the analytic
forces need no change at all (block-diagonal ⇒ each atom's force is `−dE_(its system)/dx`).

```python
from tt_atom import TTAtomCalculator
from ase.build import molecule

calc = TTAtomCalculator("uma_s_ethanol.npz")        # single-system API is unchanged
batch = [molecule("CH3CH2OH") for _ in range(32)]   # e.g. an MD ensemble / conformer set
for a in batch:
    a.rattle(stdev=0.05); a.info.update(charge=0, spin=1)

out = calc.evaluate_batch(batch)                    # ONE device forward
out["energy"]      # np.ndarray [K]  — per-system energies (eV)
out["forces"]      # list[np.ndarray [N_k, 3]] — per-system forces
```

Validated bit-close against fairchem's own batched merged inference (`data_list_collater` +
`predict`) on the real uma-s-1 checkpoint: **energy rel err 3.5e-6, force PCC 0.99999** (K = 8
ethanol conformers). One caveat, inherited from UMA: a merged bundle bakes the MoLE expert routing
for **one reduced composition** (fairchem's `merge_MOLE_model` asserts the same), so a batch shares
that composition — conformers, an MD ensemble, an active-learning set of one molecule. A genuinely
mixed-composition batch needs the unmerged model, which TT-Atom does not run. See
[`examples/evaluate_batch.py`](examples/evaluate_batch.py).

## Reproduce

```bash
python -m pytest tests/ -q                          # 26 tests (parity, forces, periodic, trace, batch)
tt-atom verify  uma_s_ethanol.npz                   # device vs embedded fairchem reference
python benchmarks/bench_trace.py --weights uma_s_ethanol.npz     # eager vs traced e2e
python benchmarks/bench_throughput.py --weights model.npz --cells 2 3 4 5
python benchmarks/bench_batch.py --weights uma_s_ethanol.npz --ks 1 2 4 8 16 32 64  # batched vs one-at-a-time
```

## Layout

```
tt_atom/   model · so2 · rotation · forces · geometry(PBC) · grid · spectral · norm · activation
           · weights · calculator · trace · batch(multi-card) · disjoint(batching) · device · cli
tests/     per-module + end-to-end parity · analytic-force VJP · periodic · trace · real-weight · batch
benchmarks/throughput (CPU-vs-TT) · multi-card · trace (eager-vs-traced) · batching · chart generator
examples/  relax · md · periodic (crystal) · batch (multi-card) · evaluate_batch (disjoint-union)
tools/     fairchem checkpoint -> WeightBundle exporter (embeds a reference for `tt-atom verify`)
```

## License

TT-Atom (this code) is **Apache-2.0** (see [LICENSE](LICENSE), [NOTICE](NOTICE)). It depends on
`fairchem-core` for reference/weights but vendors none of it. The UMA / eSEN model weights are
**separately licensed** (FAIR Chemistry License) and are **not** included or redistributed here.
