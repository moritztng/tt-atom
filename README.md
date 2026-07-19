# TT-Atom

![Caffeine molecular dynamics, uma-s-1 on a Tenstorrent Blackhole card](assets/caffeine_md.gif)

Run ML interatomic potentials on [Tenstorrent](https://tenstorrent.com): Meta's [UMA](https://huggingface.co/facebook/UMA) and Orbital Materials' [Orb-v3 / OrbMol](https://github.com/orbital-materials/orb-models), both behind the same [ASE](https://wiki.fysik.dtu.dk/ase/) calculator interface (bring your own checkpoint). Energy, forces and stress for molecules and periodic materials.

## Install

You need a Tenstorrent card and its driver. UMA's equivariant rotation uses a custom op that the pip `ttnn` wheel doesn't carry, so UMA needs a **source tt-metal build**. Orb-v3/OrbMol run on stock `ttnn` ops, so if you only use those, `pip install ttnn` from PyPI and skip step 1.

**1. Build and install tt-metal with the op** (branch `moritztng/tt-atom`):

```bash
git clone --recursive -b moritztng/tt-atom https://github.com/tenstorrent/tt-metal.git
cd tt-metal
export TT_METAL_HOME=$PWD
./build_metal.sh --build-type Release          # full build (tens of minutes)
pip install -e .                               # tt-metal's own dev-install path
```

Keep `TT_METAL_HOME` exported at runtime too, and don't delete `$TT_METAL_HOME/build_Release` after installing. The JIT-compiled kernels load from there. See [`custom_kernels/README.md`](custom_kernels/README.md) for the op's source and how to re-integrate it onto a newer tt-metal commit. If opening a device fails with a `p300` / `mesh graph descriptor` error, see [Troubleshooting](#troubleshooting).

**2. Install TT-Atom into the same venv:**

```bash
git clone https://github.com/moritztng/tt-atom.git
pip install -e ./tt-atom                        # numpy<2, torch (CPU), ase (not ttnn)
```

**3. Verify the op is loaded:**

```bash
python -c "import ttnn; e=ttnn._ttnn.operations.experimental; print(hasattr(e,'fused_rotate'), hasattr(e,'fused_rotate_gc'))"   # -> True True
```

`uma-s-1` is the validated UMA target; other checkpoints (e.g. `uma-m`) raise a clear error.

## Quickstart

```bash
tt-atom run structure.xyz
```

```python
from ase.build import molecule
from tt_atom import Calculator

atoms = molecule("H2O")                          # any ASE Atoms (e.g. ase.io.read("file.xyz"))
atoms.calc = Calculator(atoms, "orb-v3-conservative-omol")   # an Orb checkpoint, by name
# atoms.calc = Calculator(atoms, "uma-s-1")                  # UMA, by name (same as Calculator(atoms))
atoms.get_potential_energy()
atoms.get_forces()
```

One entry point, one call, the model picked by name (like fairchem's `FAIRChemCalculator` or Hugging Face's `AutoModel.from_pretrained`). You never need to know whether it's a UMA or an Orb under the hood. The name selects the family: any `uma-*` routes to the equivariant eSCN-MD engine, any `orb-v3-*` to the Orb backbone (see [Model coverage](#model-coverage)). With no name, `Calculator(atoms)` is the default, `uma-s-1`.

UMA infers the task (`omat` if the cell is periodic, else `omol`) and builds a model for that composition on first use, then loads it from cache. Orb weights aren't composition-specific, so its cache is per checkpoint, not per structure. The example leads with an Orb checkpoint because its weights are ungated and it runs on stock `ttnn`; UMA needs the gated `uma-s-1` weights and a source `tt-metal` build (see [Install](#install)). Everything downstream is plain ASE either way.

## Relax and MD

```bash
tt-atom run structure.xyz --relax --out relaxed.xyz
tt-atom run structure.xyz --md --steps 200 --temp 300
```

Add `--trace` (or `Calculator(atoms, trace=True)`, UMA only) to reuse the captured device graph across steps. About 2x on relax/MD, forces bit-identical.

## What it supports

- Models: UMA's `uma-s-1` and all four Orb-v3/OrbMol checkpoints (`orb-v3-{conservative,direct}-{omat,omol}`).
  See [Model coverage](#model-coverage) for what else exists upstream and why this build doesn't run it.
- Tasks: UMA: `omol`, `omat`, `oc20`, `odac`, `omc`. Orb-v3/OrbMol: `omat`, `omol`.
- Systems: isolated molecules and periodic cells, both model families. Charge and spin: `Calculator(
  atoms, charge=-1, spin=2)` (all UMA tasks); `Calculator(atoms, "orb-v3-conservative-omol",
  charge=-1, spin=2)` (OrbMol checkpoints only; the Orb-v3 omat checkpoints were never trained with
  conditioning and ignore both).
- Properties: energy always. Conservative analytic forces (`F = -dE/dpos`) for UMA and Orb-v3's
  `conservative` checkpoints; a direct MLP force head (no autograd, the fast path) for Orb-v3's
  `direct` checkpoints. Stress for UMA (always) and Orb-v3 (`conservative` via the same autograd
  pass; `direct-20-omat` via a dedicated stress head; `direct-omol` has none, consistent with
  stress not being meaningful for isolated molecules), so variable-cell relaxation works for either
  family (see [`examples/relax_cell.py`](examples/relax_cell.py)). Orb-v3 is not equivariant
  (see [Model coverage](#model-coverage)), a real architectural difference from UMA, not a gap in
  this port.

## Model coverage

### UMA

Meta has released two UMA sizes: `uma-s-1` (`.1`/`.2`) and `uma-m-1p1` (there is no `uma-l`). The
[paper](https://arxiv.org/abs/2506.23971) scales capacity via mixture-of-linear-experts on the
small and medium models rather than shipping a third dense tier, and
[facebook/UMA](https://huggingface.co/facebook/UMA) carries checkpoints for only those two.

Only `uma-s-1` runs on this build; `uma-m-1p1` raises a clear `RuntimeError` naming the shape
rather than silently running slow or wrong (`tests/test_umam.py` anchors this contract). See
[`custom_kernels/README.md`](custom_kernels/README.md) for why.

### Orb-v3 / OrbMol

[Orbital Materials](https://github.com/orbital-materials/orb-models) ships four public,
ungated checkpoints, all of which run on this build:

| checkpoint | family | notes |
|---|---|---|
| `orb-v3-conservative-inf-omat` | Orb-v3 | analytic forces (`F = -dE/dpos`), stress via the same autograd pass |
| `orb-v3-direct-20-omat` | Orb-v3 | forces are a direct MLP head (no autograd, the fast checkpoint); dedicated stress head |
| `orb-v3-conservative-omol` | OrbMol | aperiodic molecules, charge + spin conditioning, no stress head |
| `orb-v3-direct-omol` | OrbMol | forces are a direct MLP head, charge + spin conditioning, no stress head |

**Orb-v3 is not equivariant**: it's a plain attention-MPNN over scalar features, with no rotated
tensor representation. None of UMA's custom kernels apply, so **Orb-v3/OrbMol run on stock `ttnn`
ops**, and no source tt-metal build is needed if you only use these models (see [Install](#install);
`docs/orb-port.md` has the full architecture read).

Orb caps each atom's neighbour count per the checkpoint (20 for the `-20` checkpoints, 120
otherwise). A structure that exceeds it raises a clear error rather than silently returning a
different neighbour list; use the `-inf`/`omol` checkpoints (cap 120) or a smaller cell for denser
systems.

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

The OrbMol rows span three systems (closed-shell, charged, open-shell radical). The low end of
each force-PCC range is the open-shell radical, whose force magnitude is an order of magnitude
smaller than the other two, so the same absolute error depresses its correlation; its energy is
the tightest of all rows. Full per-system breakdowns, the non-equivariance analysis, and Orb-v3's
ZBL pair-repulsion correction live in [`docs/orb-port.md`](docs/orb-port.md).

Dynamics are stable: UMA's NVE energy drift is about 1 meV/atom/ps. Op numerics can shift between
`ttnn` versions, so confirm parity on yours.

For the full device-vs-reference parity framework (R / D / X noise-floor legs across every shipped
family and regime), see [`docs/materials-benchmark.md`](docs/materials-benchmark.md).

Reproduce it yourself. Every UMA bundle embeds the fairchem reference energy/forces from build
time; Orb-v3/OrbMol goldens do the same for `orb-models` (`tests/gen_golden_orb.py`):

```bash
tt-atom verify model.npz     # UMA: device output vs the embedded fairchem reference
pytest tests/                # full parity suite against both models' upstream goldens
```

## Throughput

Both models are dispatch-bound at typical MD/relaxation sizes, so the same two levers apply:
batch many systems into one device pass, or trace-capture a fixed-topology loop to cut host
dispatch overhead. Both ship `calc.evaluate_batch`:

| mechanism | UMA | Orb-v3 |
|---|---|---|
| Batch independent systems (`calc.evaluate_batch`) | ~13x vs looping on one card (many small molecules) | ~19x (conservative-omol); ~12x (direct-omol), at K=128 9-atom molecules |
| Multi-card fan-out (`tt_atom.batch.MultiCard`) | one process per card | inherits the same scheduler |
| Trace-captured single-system MD/relax step | ~2x, bit-identical forces (`trace=True`) | 1.30–1.51x; energy bit-exact, analytic force finish within 1e-6 of eager |
| Trace-captured batched MD ensemble | K=4: 4.2x; K=16: 2.6x | not implemented |
| Source-build perf flags | ~2x traced MD step at large systems; opt-in env vars; regress small molecules. Details in [`custom_kernels/README.md`](custom_kernels/README.md) | n/a (stock `ttnn`) |
| bf8 (`fast=` / `examples/orb_md.py --fast`) | no win from weights alone; the real lever is the edge-activation dataflow above | 1.21–1.23x at 512–2016 atoms by compressing hidden MLP activations; release-gated accuracy trade-off ([details](docs/orb-port.md)) |

Batching (either model):

```python
out = calc.evaluate_batch(list_of_atoms)              # out["energy"], out["forces"]
out = calc.evaluate_batch(replicas, trace=True)       # per-step in an MD ensemble loop (UMA)
```

UMA bakes one MoLE bundle per reduced composition, so a batch shares that composition (conformers,
or an MD ensemble of one molecule). Orb has no per-composition routing, so its batch may mix
compositions, charges, and spins freely; the only constraint is the checkpoint's per-atom
`max_num_neighbors` cap, enforced per-system inside the batch.

To use several cards, fan systems across them with `tt_atom.batch` (one process per card, either
model).

## Compared to upstream (fairchem / orb-models)

TT-Atom is an inference runtime, not a rewrite of either upstream project. It reuses the released
weights and matches them.

|  | Upstream (fairchem for UMA, orb-models for Orb-v3/OrbMol) | TT-Atom |
|---|:---:|:---:|
| Hardware | GPU, CPU | Tenstorrent |
| Energy, forces, stress | ✅ | ✅ (Orb: `conservative` via autograd+virial, `direct` via dedicated MLP heads) |
| Molecules, periodic (PBC) | ✅ | ✅ |
| Charge/spin conditioning | ✅ (UMA, all tasks; OrbMol only for Orb) | ✅ (`charge=`/`spin=` kwargs, same shape for both models) |
| Tasks / checkpoints | UMA: omol/omat/oc20/odac/omc; Orb-v3/OrbMol: omat/omol | `uma-s-1`; all 4 public Orb-v3/OrbMol checkpoints |
| ASE relax and MD | ✅ | ✅ (plus a traced loop, both models) |
| Batched inference | ✅ | ✅ both models: UMA (one composition per batch), Orb (any mix of compositions/charge/spin), `calc.evaluate_batch` |
| LAMMPS interface | ✅ (fairchem; not verified whether orb-models ships one) | ❌ |
| Training, fine-tuning | ✅ | ❌ (inference only) |

## Bundles and the reference environment

`Calculator(atoms)` builds and caches model bundles for you, so most users never touch this. To
build a UMA bundle yourself:

```bash
refenv/bin/python tools/export_weights.py --uma-s-1 --xyz structure.xyz --task omol --out model.npz
```

then `TTAtomCalculator("model.npz")`. Building one needs `fairchem` to read the checkpoint and
merge the experts. `fairchem` needs `numpy>=2`, which can't share a process with `ttnn`'s
`numpy<2`, so keep it in its own venv:

```bash
python -m venv refenv && refenv/bin/pip install "fairchem-core>=2.10"
```

`Calculator(atoms)` and `tt-atom run` call it automatically the first time they see a new
composition, then cache the result. Set `TT_ATOM_REFENV` to its python if it isn't found
automatically. Cached runs never need it.

Orb weights aren't composition-specific, so an Orb bundle is one plain weight export per
checkpoint name, built in the same reference env (`orb-models` installs alongside `fairchem-core`
with no conflicts):

```bash
refenv/bin/python tools/export_orb_weights.py --ckpt conservative-inf-omat --out weights.npz
```

then `OrbCalculator(weights.npz)`. `Calculator(atoms, "orb-...")` calls this automatically on first
use of a given checkpoint name; a cache hit needs no reference env, same as UMA's.

## Troubleshooting

On some boards/firmware the tt-metal base commit's UMD misreads the board ID as a dual-chip P300
and refuses to open any device (single-card included), with `Custom fabric mesh graph descriptor
path must be specified for CUSTOM cluster type`. Export this before opening a device:

```bash
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
```

Set it in the parent process before constructing `tt_atom.batch.MultiCard` too, since its per-card
worker processes inherit it.

## License

MIT for this code, which reimplements the UMA / eSCN-MD architecture from [fairchem](https://github.com/facebookresearch/fairchem) (also MIT) and the Orb-v3 architecture from [orb-models](https://github.com/orbital-materials/orb-models) (Apache-2.0). It depends on `ttnn` (Apache-2.0) and `ase` (LGPL-2.1+). The UMA weights are separately licensed under the [FAIR Chemistry License](https://huggingface.co/facebook/UMA), are gated, and are not included (bring your own). The Orb-v3/OrbMol weights are Apache-2.0 and ungated; `Calculator(atoms, "orb-...")` downloads them itself on first use of a given checkpoint.
