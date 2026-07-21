# TT-Atom

![Caffeine molecular dynamics, uma-s-1 on a Tenstorrent Blackhole card](assets/caffeine_md.gif)

Run ML interatomic potentials on [Tenstorrent](https://tenstorrent.com): Meta's [UMA](https://huggingface.co/facebook/UMA) and Orbital Materials' [Orb-v3 / OrbMol](https://github.com/orbital-materials/orb-models), both behind the same [ASE](https://wiki.fysik.dtu.dk/ase/) calculator interface (bring your own checkpoint). Energy, forces and stress for molecules and periodic materials.

## Install

You need a Tenstorrent card and its driver, and a Linux build environment (Ubuntu 22.04 / 24.04 is what we test on). UMA's equivariant rotation uses a custom op that the pip `ttnn` wheel doesn't carry, so UMA needs a **source tt-metal build**. Orb-v3/OrbMol run on stock `ttnn`.

This release is validated against tt-metal commit **`8d759240fdd763a38e3abdc8344076f584dc4f4d`** on branch `moritztng/tt-atom`. The branch moves, so pin that commit for a reproducible build. (To move to a newer tt-metal commit later, re-integrate the op following [`custom_kernels/README.md`](custom_kernels/README.md).)

**1. Create the runtime environment:**

```bash
python -m venv venv
. venv/bin/activate
```

**2. Install `ttnn`:**

For Orb-v3/OrbMol only:

```bash
pip install ttnn
```

For UMA, build the pinned source with its custom op:

```bash
git clone --recursive -b moritztng/tt-atom https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout 8d759240fdd763a38e3abdc8344076f584dc4f4d   # the validated commit (above)
git submodule update --recursive --init
export TT_METAL_HOME=$PWD

sudo ./install_dependencies.sh                   # one-time: cmake, ninja, clang-20, the SFPI kernel compiler, ...
./build_metal.sh --build-type Release            # full build (tens of minutes)
pip install -e .                                 # tt-metal's own dev-install path
```

See [`docs/install.md`](docs/install.md) for source-build and device-open troubleshooting.

**3. Install the TT-Atom wheel from [GitHub Releases](https://github.com/moritztng/tt-atom/releases)
into the same venv:**

```bash
pip install ./tt_atom-0.2.1-py3-none-any.whl    # numpy<2, torch (CPU), ase (not ttnn)
```

**4. For UMA, verify the op is loaded:**

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

These CLI simulation commands currently use UMA. For either model family, use the Python
`Calculator` interface shown above with standard ASE optimizers and MD drivers.

Add `--trace` (or `Calculator(atoms, trace=True)`, UMA only) to reuse the device graph across
steps; see [`custom_kernels/README.md`](custom_kernels/README.md) for measured performance.

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

Orb-v3/OrbMol run on stock `ttnn`; see [`docs/orb-port.md`](docs/orb-port.md) for architecture and
verification details.

Orb caps each atom's neighbour count per the checkpoint (20 for the `-20` checkpoints, 120
otherwise). A structure that exceeds it raises a clear error rather than silently returning a
different neighbour list; use the `-inf`/`omol` checkpoints (cap 120) or a smaller cell for denser
systems.

## Accuracy

Every supported family is release-gated on-device against its upstream reference. See
[`docs/materials-benchmark.md`](docs/materials-benchmark.md) for results and reproduction.

## Throughput

Both families support batching, trace replay, and multi-card fan-out. See
[`docs/orb-port.md`](docs/orb-port.md) and [`custom_kernels/README.md`](custom_kernels/README.md)
for measured performance and usage constraints.

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

`Calculator(atoms)` exports and caches model bundles on first use. See
[`docs/install.md`](docs/install.md) for the separate reference environment and manual export.

## Troubleshooting

See [`docs/install.md`](docs/install.md) for source-build, SFPI, cache, and mesh-descriptor fixes.

## License

MIT for this code, which reimplements the UMA / eSCN-MD architecture from [fairchem](https://github.com/facebookresearch/fairchem) (also MIT) and the Orb-v3 architecture from [orb-models](https://github.com/orbital-materials/orb-models) (Apache-2.0). It depends on `ttnn` (Apache-2.0) and `ase` (LGPL-2.1+). The UMA weights are separately licensed under the [FAIR Chemistry License](https://huggingface.co/facebook/UMA), are gated, and are not included (bring your own). The Orb-v3/OrbMol weights are Apache-2.0 and ungated; `Calculator(atoms, "orb-...")` downloads them itself on first use of a given checkpoint.
