# TT-Atom

![Caffeine molecular dynamics, uma-s-1 on a Tenstorrent Blackhole card](assets/caffeine_md.gif)

Run Meta's [UMA](https://huggingface.co/facebook/UMA) interatomic potential on [Tenstorrent](https://tenstorrent.com). Energy, forces and stress for molecules and periodic materials, behind an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator. Bring your own UMA checkpoint.

## Install

TT-Atom is the custom-kernel-only, highest-performance build for `uma-s`. Its per-edge Wigner rotation runs as a custom tt-metal kernel that the pip `ttnn` wheel does not carry, so `ttnn` comes from a **source tt-metal build**. The op is pre-integrated on the [`moritztng/tt-atom`](https://github.com/tenstorrent/tt-metal/tree/moritztng/tt-atom) branch of tt-metal, so the build is a plain clone-and-build — no patching. You need a Tenstorrent card and its driver.

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
from tt_atom import UMA

atoms = read("structure.xyz")
atoms.calc = UMA(atoms)
atoms.get_potential_energy()
atoms.get_forces()
```

`UMA(atoms)` uses `uma-s-1`, infers the task (`omat` if the cell is periodic, else `omol`), and builds a device-resident model for that composition on first use. Later calls load it from cache. Everything downstream is plain ASE.

## Relax and MD

```bash
tt-atom run structure.xyz --relax --out relaxed.xyz
tt-atom run structure.xyz --md --steps 200 --temp 300
```

Add `--trace` (or `UMA(atoms, trace=True)`) to replay the captured device graph over the loop. About 2x on relax/MD, forces stay bit-identical.

## What it supports

- Models: `uma-s-1` (default). See [Model coverage](#model-coverage) for what else exists upstream
  and why this build doesn't run it.
- Tasks: `omol`, `omat`, `oc20`, `odac`, `omc`.
- Systems: isolated molecules and periodic cells. Charge and spin via `UMA(atoms, charge=-1, spin=2)`.
- Properties: energy, conservative analytic forces, and stress, so variable-cell relaxation works (see [`examples/relax_cell.py`](examples/relax_cell.py)).

## Model coverage

Meta has released two UMA sizes: `uma-s-1` (`.1`/`.2`) and `uma-m-1p1` — there is no `uma-l`. The
[paper](https://arxiv.org/abs/2506.23971) scales capacity via mixture-of-linear-experts on the
small and medium models rather than shipping a third, larger dense tier, and
[facebook/UMA](https://huggingface.co/facebook/UMA) carries checkpoints for only those two.

Of the two that exist, only `uma-s-1` runs on this build. `uma-s` is square (lmax=mmax=2), so its
per-edge Wigner rotation is a 9x9 tile that fits the fused kernel's L1 CB budget. `uma-m-1p1` uses
mmax<lmax spherical-harmonic subselection, so its rotation is rectangular (25<->19, W=256) — that
overflows the kernel's L1 budget, and this build has no MAC fallback, so it raises a clear
`RuntimeError` naming the shape rather than silently running slow or wrong
(`tests/test_umam.py` anchors this contract). A hypothetical `uma-l`, sized above `uma-m`, would
need L1 headroom `uma-m` already overflows, so it isn't a new question, just a bigger version of
the one above — and moot, since the checkpoint doesn't exist to test it against.

## Accuracy

Every task is checked on-device against the released `uma-s-1` checkpoint run through fairchem on the same structure.

| task | system | energy rel. err | force PCC | stress PCC |
|------|--------|----------------:|----------:|-----------:|
| omol | ethanol         | 2e-7 | 0.9996  |     |
| omat | bulk Si         | 3e-4 | 0.99999 | 0.99999 |
| oc20 | Cu(100) + H slab| 9e-5 | 1.0000  |     |
| odac | MgO framework   | 2e-4 | 0.99999 |     |
| omc  | solid CO2       | 8e-5 | 1.0000  |     |

Dynamics are stable: NVE energy drift is about 1 meV/atom/ps. These numbers are from `ttnn` 0.68.0. Op numerics can shift slightly between `ttnn` versions, so confirm parity on the version you actually run:

Reproduce it yourself. Every bundle embeds the fairchem reference energy and forces from build time, so:

```bash
tt-atom verify model.npz     # device output vs the embedded fairchem reference
pytest tests/                # full parity suite against fairchem goldens
```

## Throughput

Batch independent systems into a single device pass:

```python
out = calc.evaluate_batch(list_of_atoms)   # out["energy"], out["forces"]
```

For many small molecules this is roughly 13x over looping on one card. To use several cards, fan systems across them with `tt_atom.batch` (one process per card).

For a **batched MD ensemble / relaxation** — K fixed-composition replicas evolving with a stable neighbour list — add `trace=True` to capture the batched device graph once and replay it (forces stay bit-identical; it re-captures whenever the neighbour list changes):

```python
out = calc.evaluate_batch(replicas, trace=True)   # per-step in the ensemble loop
```

At small per-system sizes the eager batched forward is host-dispatch-bound below saturation, so the trace lets a *modest* ensemble reach near-peak throughput: measured on one p150 (uma-s-1, 9-atom molecules) K=4 gives 4.2x (59→246 systems/s), K=16 2.6x (207→528 sys/s) — approaching the K≥128 eager device-bound plateau (~700 sys/s) at a fraction of the batch size. Leave it `False` for one-shot screening, where a fresh batch each call would re-capture every time.

## Compared to fairchem

TT-Atom is an inference runtime, not a rewrite of fairchem. It reuses the released weights and matches them.

|  | fairchem | TT-Atom |
|--|:--------:|:-------:|
| Hardware | GPU, CPU | Tenstorrent |
| Energy, forces, stress | ✅ | ✅ |
| Molecules, periodic (PBC) | ✅ | ✅ |
| Tasks (omol/omat/oc20/odac/omc) | ✅ | ✅ |
| Models | uma-s, uma-m | uma-s-1 |
| ASE relax and MD | ✅ | ✅ (plus a traced loop) |
| Batched inference | ✅ | ✅ (one composition per batch) |
| LAMMPS interface | ✅ | ❌ |
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

## License

MIT for this code, which reimplements the UMA / eSCN-MD architecture from [fairchem](https://github.com/facebookresearch/fairchem) (also MIT). It depends on `ttnn` (Apache-2.0) and `ase` (LGPL-2.1+). The UMA weights are separately licensed under the [FAIR Chemistry License](https://huggingface.co/facebook/UMA), are gated, and are not included. Bring your own.
