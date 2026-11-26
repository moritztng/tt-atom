"""Generate float64 golden tensors for the TT-Atom MACE port (reference env only).

    ~/.ttatom_run/maceref/bin/python tests/gen_golden_mace.py \
        --model medium-0b3 --out ~/.ttatom_run/goldens_real/si_mace_mp0b3.npz

Runs stock mace-torch on CPU in float64 and saves, for one rattled Si supercell (the same system
the Orb and UMA goldens use): the graph inputs, energy / forces / stress, the output of every
top-level module (node embedding, radial embedding, spherical harmonics, each interaction,
product and readout) for bottom-up parity, and the full state dict.

Only MIT-licensed weights are allowed here, because goldens carry the weights. Per
ACEsuit/mace-foundations (checked 2026-11-26): MP-0a/0b/0b2/0b3 and MPA-0 are MIT; OMAT-0,
MATPES, OMOL-0, MH, MDP and every MACE-OFF23 model are under the Academic Software License and
must never be written into a file we could redistribute.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from ase.build import bulk

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from npz_atomic import config_array, savez_atomic  # noqa: E402

MIT_MODELS = {"small", "medium", "large", "small-0b", "medium-0b", "small-0b2", "medium-0b2",
              "large-0b2", "medium-0b3", "medium-mpa-0"}


def arr(t):
    return t.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium-0b3", choices=sorted(MIT_MODELS))
    ap.add_argument("--reps", default="2,2,2", help="Si diamond supercell repeats")
    ap.add_argument("--rattle", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mace.calculators import mace_mp
    from mace.tools import torch_geometric, utils
    from mace import data

    torch.set_default_dtype(torch.float64)
    calc = mace_mp(model=args.model, device="cpu", default_dtype="float64")
    model = calc.models[0].eval()

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * tuple(int(x) for x in args.reps.split(","))
    atoms.rattle(stdev=args.rattle, seed=args.seed)
    atoms.calc = calc
    E = float(atoms.get_potential_energy())
    F = np.asarray(atoms.get_forces())
    S = np.asarray(atoms.get_stress())

    # Re-run the model directly on the same batch the calculator builds, with hooks on every
    # top-level module, so the activations belong to exactly this energy.
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cfg = data.config_from_atoms(atoms)
    ds = [data.AtomicData.from_config(cfg, z_table=z_table, cutoff=float(model.r_max),
                                      heads=getattr(calc, "available_heads", None))]
    batch = next(iter(torch_geometric.dataloader.DataLoader(ds, batch_size=1, shuffle=False)))

    acts: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_mod, _inp, out):
            for i, o in enumerate(out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(o, torch.Tensor):
                    acts[f"{name}.out{i}"] = arr(o)
        return fn

    handles = []
    for name in ("node_embedding", "radial_embedding", "spherical_harmonics", "pair_repulsion_fn"):
        if hasattr(model, name):
            handles.append(getattr(model, name).register_forward_hook(hook(name)))
    # Each block and each of its direct sub-blocks (linear_up, conv_tp_weights, conv_tp, linear,
    # skip_tp, density_fn; the product's symmetric contraction and linear), so a parity failure
    # names the sub-block that carries it.
    for kind in ("interactions", "products", "readouts"):
        for i, m in enumerate(getattr(model, kind)):
            handles.append(m.register_forward_hook(hook(f"{kind}.{i}")))
            for sub, c in m.named_children():
                handles.append(c.register_forward_hook(hook(f"{kind}.{i}.{sub}")))
    out = model(batch.to_dict(), training=False, compute_force=True, compute_stress=True)
    for h in handles:
        h.remove()
    E2 = float(out["energy"].sum())
    assert abs(E2 - E) < 1e-9 * max(1.0, abs(E)), f"hooked rerun energy {E2} != calculator {E}"

    saved = {
        "config": config_array(dict(model=args.model, reps=args.reps, rattle=args.rattle,
                                    seed=args.seed, r_max=float(model.r_max),
                                    num_interactions=int(model.num_interactions),
                                    dtype="float64", licence="MIT")),
        "in@atomic_numbers": atoms.get_atomic_numbers(),
        "in@pos": atoms.get_positions(),
        "in@cell": np.asarray(atoms.get_cell()),
        "in@edge_index": arr(batch.edge_index),
        "in@shifts": arr(batch.shifts),
        "out@energy": np.array([E]),
        "out@forces": F,
        "out@stress": S,
    }
    saved.update({f"a@{k}": v for k, v in acts.items()})
    saved.update({f"w@{k}": arr(v) for k, v in model.state_dict().items()})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    savez_atomic(args.out, **saved)
    print(f"wrote {args.out}: natoms={len(atoms)} nedges={batch.edge_index.shape[1]} "
          f"E={E:.10f} eV  n_acts={len(acts)}  n_weights={sum(k.startswith('w@') for k in saved)}")


if __name__ == "__main__":
    main()
