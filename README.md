# TT-Atom

![Caffeine molecular dynamics, uma-s-1 on a Tenstorrent Blackhole card](assets/caffeine_md.gif)

Run ML interatomic potentials on [Tenstorrent](https://tenstorrent.com): Meta's [UMA](https://huggingface.co/facebook/UMA), behind an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator (bring your own checkpoint), and Orbital Materials' [Orb-v3 / OrbMol](#orb-v3-and-orbmol). Energy, forces and stress for molecules and periodic materials.

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

- Models: `uma-s-1` (default), `uma-s-1.2`. See [Model coverage](#model-coverage) for what else
  exists upstream and why this build doesn't run it.
- Tasks: `omol`, `omat`, `oc20`, `odac`, `omc`.
- Systems: isolated molecules and periodic cells. Charge and spin via `UMA(atoms, charge=-1, spin=2)`.
- Properties: energy, conservative analytic forces, and stress, so variable-cell relaxation works (see [`examples/relax_cell.py`](examples/relax_cell.py)).

## Model coverage

Meta has released two UMA sizes: `uma-s-1` (`.1`/`.2`) and `uma-m-1p1` — there is no `uma-l`. The
[paper](https://arxiv.org/abs/2506.23971) scales capacity via mixture-of-linear-experts on the
small and medium models rather than shipping a third, larger dense tier, and
[facebook/UMA](https://huggingface.co/facebook/UMA) carries checkpoints for only those two.

Of the two that exist, only `uma-s` runs on this build (both `uma-s-1` and `uma-s-1.2`). `uma-s` is
square (lmax=mmax=2), so its per-edge Wigner rotation is a 9x9 tile that fits the fused kernel's L1 CB
budget. `uma-m-1p1` uses mmax<lmax spherical-harmonic subselection, so its rotation is rectangular
(25<->19, W=256) — that overflows the kernel's L1 budget, and this build has no MAC fallback, so it
raises a clear `RuntimeError` naming the shape rather than silently running slow or wrong
(`tests/test_umam.py` anchors this contract). A hypothetical `uma-l`, sized above `uma-m`, would
need L1 headroom `uma-m` already overflows, so it isn't a new question, just a bigger version of
the one above — and moot, since the checkpoint doesn't exist to test it against.

### uma-s-1.2

`uma-s-1.2` adds fairchem's charge-balanced channels: the `l=0` charge channels are re-balanced to the system charge after every block. TT-Atom applies this automatically — point `UMA` at the checkpoint (gated; bring your own):

```python
atoms.calc = UMA(atoms, checkpoint="uma-s-1p2.pt")
```

Parity with fairchem — forces, energy, and stress across 757 molecular and periodic systems, plus a CPU throughput comparison — is written up in [`docs/uma-s-1p2-validation.md`](docs/uma-s-1p2-validation.md).

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
| Models | uma-s, uma-m | uma-s-1, uma-s-1.2 |
| ASE relax and MD | ✅ | ✅ (plus a traced loop) |
| Batched inference | ✅ | ✅ (one composition per batch) |
| LAMMPS interface | ✅ | ❌ |
| Training, fine-tuning | ✅ | ❌ (inference only) |

## Orb-v3 and OrbMol

Alongside UMA, TT-Atom also runs [Orb-v3](https://github.com/orbital-materials/orb-models)
(Orbital Materials) and **OrbMol**, its charge/spin-conditioned molecular variant:

| checkpoint | family | notes |
|---|---|---|
| `orb-v3-conservative-inf-omat` | Orb-v3 | analytic forces (`F = -dE/dpos`), stress |
| `orb-v3-direct-20-omat` | Orb-v3 | forces are a direct MLP head — no autograd, the fast checkpoint |
| `orb-v3-conservative-omol` | OrbMol | aperiodic molecules, charge + spin conditioning, no stress head |
| `orb-v3-direct-omol` | OrbMol | forces are a direct MLP head, charge + spin conditioning |

**Orb-v3 is NOT equivariant** — it's a plain attention-MPNN over scalar features (real spherical
harmonics are used only as a fixed per-edge descriptor, never carried as a rotated tensor
representation). None of UMA's four custom kernels (`fused_rotate`/`fused_rotate_gc`/`fused_gate`/
`fused_ln_bw`) apply, so **Orb-v3/OrbMol run on stock `ttnn` ops** — no source tt-metal build is
needed if you only use these models (see `docs/orb-port.md`'s "Architecture verdict").

Unlike UMA, there is no `Orb(atoms)` ASE one-liner yet — checkpoints load through the same
golden/bundle export used for parity testing (`tests/gen_golden_orb.py`, run in the `fairchem`/
`orb-models` reference env), then the device modules are called directly:

```bash
refenv/bin/python tests/gen_golden_orb.py --ckpt orb-v3-conservative-inf-omat \
    --out model.npz   # --system for a custom structure; see the script's --help
```

```python
from tt_atom.device import open_device
from tt_atom.orb_weights import OrbWeights
from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead, host_cutoff, _to_dev
from tt_atom.orb_forces import energy_and_forces

device = open_device(0)
gw = OrbWeights.load("model.npz")
cfg, w = gw.config, gw.weights
L = cfg["num_message_passing_steps"]

encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                  latent_dim=cfg["latent_dim"], hidden_dim=1024)
layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                     latent_dim=cfg["latent_dim"], hidden_dim=1024) for i in range(L)]
ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)

energy, forces = energy_and_forces(
    encoder, layers, ehead, device,
    pos=gw.inp("pos"), senders=gw.inp("senders").long(), receivers=gw.inp("receivers").long(),
    atomic_numbers=gw.inp("atomic_numbers").long(), node_feat=gw.host("node_feat"),
)
```

For a fixed-topology MD/relaxation loop, capture the graph once with `tt_atom.orb_trace`'s
`OrbTracedEngine` and replay it per step (bit-exact vs eager, 1.3–4.8x measured depending on
system size and environment — see `docs/orb-port.md` and `CHANGELOG.md`) instead of calling
`energy_and_forces` fresh every step.

Full accuracy tables, the non-equivariance analysis, and reproduction commands for both checkpoint
families live in [`docs/orb-port.md`](docs/orb-port.md).

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
