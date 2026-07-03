# TT-Atom

Run Meta's [UMA](https://huggingface.co/facebook/UMA) interatomic potential on [Tenstorrent](https://tenstorrent.com). Energy, forces and stress for molecules and periodic materials, behind an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator. Bring your own UMA checkpoint.

## Install

```bash
pip install git+https://github.com/moritztng/tt-atom.git
```

You also need `ttnn`, the Tenstorrent runtime. It is not on PyPI, so install the wheel matching your card and `tt-kmd` driver from [tt-metal](https://github.com/tenstorrent/tt-metal). TT-Atom pins `numpy<2` to match it. `import tt_atom` works without a card.

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

- Models: `uma-s-1` (default), `uma-m-1p1`.
- Tasks: `omol`, `omat`, `oc20`, `odac`, `omc`.
- Systems: isolated molecules and periodic cells. Charge and spin via `UMA(atoms, charge=-1, spin=2)`.
- Properties: energy, conservative analytic forces, and stress, so variable-cell relaxation works (see [`examples/relax_cell.py`](examples/relax_cell.py)).

Checked on-device against the released checkpoints (energy rel < 1e-3, force PCC > 0.99). For instance `uma-s-1` ethanol gives E rel 2e-7 and force PCC 0.9996, Si `omat` stress PCC 0.99999, and NVE energy drift is around 1 meV/atom/ps.

## Throughput

Batch independent systems into a single device pass:

```python
out = calc.evaluate_batch(list_of_atoms)   # out["energy"], out["forces"]
```

For many small molecules this is roughly 13x over looping on one card. To use several cards, fan systems across them with `tt_atom.batch` (one process per card).

## Bundles and the reference environment

The model is a "bundle": UMA weights merged for one composition. `UMA(atoms)` builds and caches bundles for you, so most users never touch this. To build one yourself:

```bash
tt-atom convert-checkpoint --uma-s-1 --xyz structure.xyz --task omol --out model.npz
```

then `TTAtomCalculator("model.npz")`.

Building a bundle needs `fairchem` to read the checkpoint and merge the experts. `fairchem` wants `numpy>=2`, which cannot share a process with `ttnn`'s `numpy<2`, so keep it in its own venv:

```bash
python -m venv refenv && refenv/bin/pip install "fairchem-core>=2.10"
```

`UMA(atoms)` and `tt-atom run` call it automatically the first time they see a new composition, then cache the result. Set `TT_ATOM_REFENV` to its python if it is not found automatically. Cached runs never need it.

## License

Apache-2.0 for this code. UMA weights are separately licensed under the [FAIR Chemistry License](https://huggingface.co/facebook/UMA) and are not included.
