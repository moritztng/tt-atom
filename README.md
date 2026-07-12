# TT-Atom

![Caffeine molecular dynamics, uma-s-1 on a Tenstorrent Blackhole card](assets/caffeine_md.gif)

Run ML interatomic potentials on [Tenstorrent](https://tenstorrent.com): Meta's [UMA](https://huggingface.co/facebook/UMA) and Orbital Materials' [Orb-v3 / OrbMol](https://github.com/orbital-materials/orb-models), both behind the same [ASE](https://wiki.fysik.dtu.dk/ase/) calculator interface (bring your own checkpoint). Energy, forces and stress for molecules and periodic materials.

## Install

TT-Atom is the custom-kernel-only, highest-performance build for `uma-s`. Its per-edge rotation runs through a custom tt-metal kernel that the pip `ttnn` wheel does not carry, so `ttnn` comes from a **source tt-metal build**. The op is pre-integrated on the [`moritztng/tt-atom`](https://github.com/tenstorrent/tt-metal/tree/moritztng/tt-atom) branch of tt-metal, so the build is a plain clone-and-build — no patching. You need a Tenstorrent card and its driver.

Orb-v3/OrbMol are non-equivariant (see [Model coverage](#model-coverage)) and run on stock `ttnn` ops — if you only use those models, skip step 1 and `pip install ttnn` from PyPI.

**1. Build and install tt-metal with the op** (branch `moritztng/tt-atom`):

```bash
git clone --recursive -b moritztng/tt-atom https://github.com/tenstorrent/tt-metal.git
cd tt-metal
export TT_METAL_HOME=$PWD
./build_metal.sh --build-type Release          # full build (tens of minutes)
pip install -e .                               # tt-metal's own dev-install path
```

The branch is the validated base `b5522097b39` plus the `fused_rotate` op library — nothing else. Its source and contract are mirrored in [`custom_kernels/README.md`](custom_kernels/README.md) as the authoritative backup (and for re-integrating onto a newer tt-metal commit).

`TT_METAL_HOME` must stay exported at **runtime** too — the JIT-compiled kernels load from `$TT_METAL_HOME/build_Release`, so don't delete that directory after installing.

On some boards/firmware this base commit's UMD misreads the board ID as a dual-chip P300 (`Board ... has 1 chips, but expected 2 chips for board type p300` -> `TT_FATAL: Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type`), which blocks opening *any* device, single-card included. If you hit that, export `TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto` before opening a device — this also needs to be set in the parent process before constructing `tt_atom.batch.MultiCard`, since its per-card worker processes inherit it.

**2. Install TT-Atom into the same venv:**

```bash
git clone https://github.com/moritztng/tt-atom.git
pip install -e ./tt-atom                        # numpy<2, torch (CPU), ase — NOT ttnn
```

**3. Verify the op is loaded:**

```bash
python -c "import ttnn; e=ttnn._ttnn.operations.experimental; print(hasattr(e,'fused_rotate'), hasattr(e,'fused_rotate_gc'))"   # -> True True
```

`uma-s` (lmax=mmax=2) is the validated target; other checkpoints (e.g. uma-m) raise a clear error. `import tt_atom` never imports ttnn, so it imports fine on a machine without a card.

## Quickstart

```bash
tt-atom run structure.xyz
```

```python
from ase.io import read
from tt_atom import UMA, Orb

atoms = read("structure.xyz")
atoms.calc = UMA(atoms)                                    # or: Orb(atoms)
atoms.get_potential_energy()
atoms.get_forces()
```

`UMA(atoms)` uses `uma-s-1`, infers the task (`omat` if the cell is periodic, else `omol`), and builds a device-resident model for that composition on first use. Later calls load it from cache. `Orb(atoms)` is the symmetric one-liner for `orb-v3-conservative-inf-omat` (pass `checkpoint=` for one of the other three, see [Model coverage](#model-coverage)) — its weights aren't composition-specific, so the cache is per-checkpoint, not per-structure. Everything downstream is plain ASE either way.

## Relax and MD

```bash
tt-atom run structure.xyz --relax --out relaxed.xyz
tt-atom run structure.xyz --md --steps 200 --temp 300
```

Add `--trace` (or `UMA(atoms, trace=True)`) to replay the captured device graph over the loop. About 2x on relax/MD, forces stay bit-identical.

## What it supports

- Models: UMA's `uma-s-1` and all four Orb-v3/OrbMol checkpoints (`orb-v3-{conservative,direct}-{omat,omol}`).
  See [Model coverage](#model-coverage) for what else exists upstream and why this build doesn't run it.
- Tasks: UMA — `omol`, `omat`, `oc20`, `odac`, `omc`. Orb-v3/OrbMol — `omat`, `omol`.
- Systems: isolated molecules and periodic cells, both model families. Charge and spin: `UMA(atoms,
  charge=-1, spin=2)` (all UMA tasks); `Orb(atoms, checkpoint="orb-v3-conservative-omol", charge=-1,
  spin=2)` (OrbMol checkpoints only — the omat checkpoints were never trained with conditioning and
  ignore both).
- Properties: energy always. Conservative analytic forces (`F = -dE/dpos`) for UMA and Orb-v3's
  `conservative` checkpoints; a direct MLP force head (no autograd, the fast path) for Orb-v3's
  `direct` checkpoints. Stress for UMA (always) and Orb-v3 (`conservative` via the same autograd
  pass; `direct-20-omat` via a dedicated stress head — `direct-omol` has none, consistent with
  stress not being meaningful for isolated molecules), so variable-cell relaxation works for either
  family (see [`examples/relax_cell.py`](examples/relax_cell.py)). Orb-v3 is honestly **not**
  equivariant (see [Model coverage](#model-coverage)) — a real architectural difference from UMA,
  not a gap in this port.

## Model coverage

### UMA

Meta has released two UMA sizes: `uma-s-1` (`.1`/`.2`) and `uma-m-1p1` — there is no `uma-l`. The
[paper](https://arxiv.org/abs/2506.23971) scales capacity via mixture-of-linear-experts on the
small and medium models rather than shipping a third, larger dense tier, and
[facebook/UMA](https://huggingface.co/facebook/UMA) carries checkpoints for only those two.

Of the two that exist, only `uma-s-1` runs on this build; `uma-m-1p1` raises a clear
`RuntimeError` naming the shape rather than silently running slow or wrong (`tests/test_umam.py`
anchors this contract) — see [`custom_kernels/README.md`](custom_kernels/README.md)'s "`fused_rotate`
contract" section for why. A hypothetical `uma-l`, sized above `uma-m`, would hit the same limit,
and is moot anyway since the checkpoint doesn't exist to test it against.

### Orb-v3 / OrbMol

[Orbital Materials](https://github.com/orbital-materials/orb-models) ships four public,
ungated checkpoints, all of which run on this build:

| checkpoint | family | notes |
|---|---|---|
| `orb-v3-conservative-inf-omat` | Orb-v3 | analytic forces (`F = -dE/dpos`), stress via the same autograd pass |
| `orb-v3-direct-20-omat` | Orb-v3 | forces are a direct MLP head — no autograd, the fast checkpoint; dedicated stress head |
| `orb-v3-conservative-omol` | OrbMol | aperiodic molecules, charge + spin conditioning, no stress head |
| `orb-v3-direct-omol` | OrbMol | forces are a direct MLP head, charge + spin conditioning, no stress head |

**Orb-v3 is honestly NOT equivariant** — it's a plain attention-MPNN over scalar features (real
spherical harmonics are used only as a fixed per-edge descriptor, never carried as a rotated
tensor representation). None of UMA's four custom kernels (`fused_rotate`/`fused_rotate_gc`/
`fused_gate`/`fused_ln_bw`) apply, so **Orb-v3/OrbMol run on stock `ttnn` ops** — no source
tt-metal build is needed if you only use these models (see [Install](#install) and
`docs/orb-port.md`'s "Architecture verdict" for the full read of the upstream source that
established this).

Unlike UMA, Orb has no MoLE (or any) expert routing baked in at merge time — the raw checkpoint
weights are valid for *any* composition/charge/spin, so `Orb(atoms)` caches its (much cheaper)
weight export once per *checkpoint name*, not per structure (`tt_atom.orb_weight_cache`, mirrors
`tt_atom.bundle_cache`'s refenv-subprocess pattern but without the per-composition merge). The
`max_num_neighbors` truncation Orb's own reference applies per atom (20 for the `-20` checkpoints,
120 otherwise) is not implemented here; rather than silently return a different neighbour list on
a denser structure, `Orb(atoms)` raises a clear error naming the degree and the checkpoint's cap
(same philosophy as `uma-m`'s shape error above) — use the `-inf`/`omol` checkpoints (cap 120) for
denser systems, or a smaller cell.

## Accuracy

Every model/task is checked on-device against its own real upstream reference (fairchem for UMA,
`orb-models` for Orb-v3/OrbMol) run on the same structure.

| model | task | system | energy rel. err | force PCC | stress |
|---|---|---|---:|---:|---:|
| uma-s-1 | omol | ethanol | 2e-7 | 0.9996 | |
| uma-s-1 | omat | bulk Si | 3e-4 | 0.99999 | PCC 0.99999 |
| uma-s-1 | oc20 | Cu(100) + H slab | 9e-5 | 1.0000 | |
| uma-s-1 | odac | MgO framework | 2e-4 | 0.99999 | |
| uma-s-1 | omc | solid CO2 | 8e-5 | 1.0000 | |
| orb-v3-conservative-inf-omat | omat | bulk Si | 1.19e-4 | 0.999975 | PCC >0.999 |
| orb-v3-direct-20-omat | omat | bulk Si | 5.79e-4 | 0.999966 | PCC >0.99 (dedicated stress head) |
| orb-v3-conservative-omol | omol | H2O / NH4+ / CH3• | 1.6e-6 – 9.2e-6 | 0.97 – 0.9997 | n/a (no stress head) |
| orb-v3-direct-omol | omol | H2O / NH4+ / CH3• | 1.7e-6 – 3.9e-5 | 0.93 – 0.998 | n/a |

The OrbMol rows span three systems (closed-shell, charged, open-shell radical) — the low end of
each range is the open-shell radical, whose forces PCC is depressed by its own tiny force
magnitude (an absolute bf16 noise floor against a signal an order of magnitude smaller than the
other two systems), not a conditioning bug: its energy — which has no such magnitude sensitivity —
is the tightest of all rows here. Full per-system breakdowns, the non-equivariance analysis, and
Orb-v3's ZBL pair-repulsion correction live in [`docs/orb-port.md`](docs/orb-port.md).

Dynamics are stable: UMA's NVE energy drift is about 1 meV/atom/ps. These numbers are from `ttnn`
0.68.0. Op numerics can shift slightly between `ttnn` versions, so confirm parity on the version
you actually run:

Reproduce it yourself. Every UMA bundle embeds the fairchem reference energy/forces from build
time; Orb-v3/OrbMol goldens do the same for `orb-models` (`tests/gen_golden_orb.py`):

```bash
tt-atom verify model.npz     # UMA: device output vs the embedded fairchem reference
pytest tests/                # full parity suite against both models' upstream goldens
```

## Throughput

Both models are dispatch-bound at typical MD/relaxation sizes, so the same two levers apply —
batch many systems into one device pass, or trace-capture a fixed-topology loop to cut host
dispatch overhead — though only UMA's batching is wired into a calculator method today:

| mechanism | UMA | Orb-v3 |
|---|---|---|
| Batch independent systems (`calc.evaluate_batch`) | ~13x vs looping on one card (many small molecules) | backbone verified batch-transparent (bit-exact row-independence), no `evaluate_batch` wired up yet — see `docs/orb-port.md` |
| Multi-card fan-out (`tt_atom.batch.MultiCard`) | proven (one process per card) | inherits the same card-count-agnostic scheduler; not independently wall-clock-benchmarked yet |
| Trace-captured single-system MD/relax step | ~2x, bit-identical forces (`trace=True`) | 1.30–1.51x, bit-exact vs eager (`tt_atom.orb_trace.OrbTracedEngine`) — shrinks at larger graphs since only host *dispatch* is removed, not the host geometry recompute (see `docs/orb-port.md`) |
| Trace-captured batched MD ensemble | K=4: 4.2x (59→246 sys/s); K=16: 2.6x (207→528 sys/s), approaching the K≥128 eager plateau (~700 sys/s) | not implemented (no batched Orb calculator path yet) |
| `--fast` (bf8 weights + Wigner coefs) | `fast=` kept for reproducibility; the real bandwidth win is the separate edge-activation dataflow (`TT_ATOM_BF8_EDGE`) plus on-device edge-degree (`TT_ATOM_DEVICE_EDE`) and the fused radial-LayerNorm backward — see below | measured **dead end**, 0.99–1.01x — Orb's forward is dispatch-bound, not weight-bandwidth-bound, so halving weight bytes does nothing (`fast=` stays threaded through for reproducibility only) |
| Source-build perf wins (`device_ede` / `bf8_edge` / `fused_lnbw`) | ~2x traced MD step at large systems (512 atoms: 389→194 ms; 216: 158→85 ms), force PCC 0.9997. Size-dependent — `device_ede`/`bf8_edge` regress small molecules (~0.85x at 9 atoms) so they're opt-in (`TT_ATOM_DEVICE_EDE=1`, `TT_ATOM_BF8_EDGE=1`) for bulk/large MD; `fused_lnbw` is a pure kernel fuse and defaults on. All three need the source `ttnn` build and no-op safely on stock `ttnn` | n/a — Orb runs on stock `ttnn` and doesn't use these kernels |

UMA batching:

```python
out = calc.evaluate_batch(list_of_atoms)              # out["energy"], out["forces"]
out = calc.evaluate_batch(replicas, trace=True)       # per-step in an MD ensemble loop
```

To use several cards, fan systems across them with `tt_atom.batch` (one process per card, either
model).

## Compared to upstream (fairchem / orb-models)

TT-Atom is an inference runtime, not a rewrite of either upstream project. It reuses the released
weights and matches them.

|  | Upstream (fairchem for UMA, orb-models for Orb-v3/OrbMol) | TT-Atom |
|--|:--------:|:-------:|
| Hardware | GPU, CPU | Tenstorrent |
| Energy, forces, stress | ✅ | ✅ (Orb: `conservative` via autograd+virial, `direct` via dedicated MLP heads) |
| Molecules, periodic (PBC) | ✅ | ✅ |
| Charge/spin conditioning | ✅ (UMA, all tasks; OrbMol only for Orb) | ✅ (`charge=`/`spin=` kwargs, same shape for both models) |
| Tasks / checkpoints | UMA: omol/omat/oc20/odac/omc; Orb-v3/OrbMol: omat/omol | `uma-s-1`; all 4 public Orb-v3/OrbMol checkpoints |
| ASE relax and MD | ✅ | ✅ (plus a traced loop, both models) |
| Batched inference | ✅ | ✅ UMA (one composition per batch); Orb backbone verified batch-transparent, not yet wired into a calculator method |
| LAMMPS interface | ✅ (fairchem; not verified whether orb-models ships one) | ❌ |
| Training, fine-tuning | ✅ | ❌ (inference only) |

## Bundles and the reference environment

The model is a "bundle": UMA weights merged for one composition. `UMA(atoms)` builds and caches bundles for you, so most users never touch this. To build one yourself:

```bash
refenv/bin/python tools/export_weights.py --uma-s-1 --xyz structure.xyz --task omol --out model.npz
```

then `TTAtomCalculator("model.npz")`.

Building a bundle needs `fairchem` to read the checkpoint and merge the experts. `fairchem` wants `numpy>=2`, which cannot share a process with `ttnn`'s `numpy<2`, so keep it in its own venv:

```bash
python -m venv refenv && refenv/bin/pip install "fairchem-core>=2.10"
```

`UMA(atoms)` and `tt-atom run` call it automatically the first time they see a new composition, then cache the result. Set `TT_ATOM_REFENV` to its python if it is not found automatically. Cached runs never need it.

Orb has no MoLE (or any) expert routing, so its weights aren't composition-specific — `Orb(atoms)`
caches one plain weight export **per checkpoint name** (not per structure), also via the same
reference env (`orb-models` installs into it alongside `fairchem-core` with no conflicts):

```bash
refenv/bin/python tools/export_orb_weights.py --ckpt conservative-inf-omat --out weights.npz
```

then `OrbCalculator(weights.npz)`. `Orb(atoms)` calls this automatically on first use of a given
`checkpoint=`; a cache hit needs no reference env, same as UMA's.

## License

MIT for this code, which reimplements the UMA / eSCN-MD architecture from [fairchem](https://github.com/facebookresearch/fairchem) (also MIT) and the Orb-v3 architecture from [orb-models](https://github.com/orbital-materials/orb-models) (Apache-2.0). It depends on `ttnn` (Apache-2.0) and `ase` (LGPL-2.1+). The UMA weights are separately licensed under the [FAIR Chemistry License](https://huggingface.co/facebook/UMA), are gated, and are not included — bring your own. The Orb-v3/OrbMol weights are Apache-2.0 and ungated; `Orb(atoms)` downloads them itself on first use of a given checkpoint.
