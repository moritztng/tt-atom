"""Energy and forces of TensorNet-MatPES with the whole network on a chip, graded against the float64 model.

The host builds the neighbour list and the per-edge geometry in float64 (matgl's own functions) and closes the
chain from the edge gradients to forces. Embedding and readout run in float32, the interaction layers in --dtype.

    python benchmarks/tensornet_gate_b.py --dtype bfloat16 --server "ssh host run_server.sh 11" --out b.json
"""
import argparse
import json

import numpy as np
import torch

from census_tensornet import build
from tensornet_gate_a import Remote
from tt_atom.tensornet import embed as E
from tt_atom.tensornet import layer as L
from tt_atom.tensornet import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="diamond Si supercell, 8 n^3 atoms")
    ap.add_argument("--model", default="materialyze/TensorNet-PES-MatPES-PBE-2025.2")
    ap.add_argument("--dtype", default="bfloat16", help="dtype of the interaction layers")
    ap.add_argument("--min-width", type=int, default=32, help="floor of the padded degree")
    ap.add_argument("--steps", type=int, default=3, help="repeat the device step; the last one is graded and timed")
    ap.add_argument("--server", default="python benchmarks/tensornet_layer_server.py --cpu")
    ap.add_argument("--rattle", type=float, default=0.0, help="extra Gaussian displacement in A, for a second cell of another energy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    import matgl
    from matgl.ext.ase import PESCalculator
    from matgl.graph._compute import compute_pair_vector_and_distance

    pot = matgl.load_model(args.model).double()
    pot.calc_stresses = False
    m, graphs = pot.model, []
    hook = m.register_forward_hook(lambda mod, a, kw, out: graphs.append(kw.get("g", a[0] if a else None)), with_kwargs=True)
    atoms = build("si", args.n)
    atoms.positions += np.random.default_rng(1).normal(0, args.rattle, atoms.positions.shape)
    atoms.calc = PESCalculator(pot)
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces().copy()
    hook.remove()
    g = graphs[0]
    z, ei, n = g.node_type, g.edge_index.long(), len(atoms)

    pos = g.pos.detach().double().requires_grad_(True)
    vec, r = compute_pair_vector_and_distance(pos, ei, g.pbc_offshift.detach().double())
    rbf = m.bond_expansion(r)
    cut = torch.where(r <= m.cutoff, 0.5 * (torch.cos(torch.pi * r / m.cutoff) + 1), 0.0)[:, None]
    geo = E.geometry(vec)
    edge, send, table, d = L.slots(ei, n, min_width=args.min_width)
    pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge].detach()
    real = edge < ei.shape[1]
    unpad = lambda t: torch.zeros(ei.shape[1], *t.shape[1:], dtype=torch.float64).index_copy(0, edge[real], t[real].double())

    remote = Remote(args.server)
    remote.call(op="model", w=M.weights(m, torch.float64), n=n, d=d, units=m.units, send=send, table=table, dtype=args.dtype)
    msg = dict(op="energy", z=pad(E.pair(m.tensor_embedding, z, ei)), rbf=pad(rbf), cut=pad(cut), geo=pad(geo))
    for _ in range(args.steps):
        out = remote.call(**msg)
    std, mean = float(pot.data_std), float(pot.data_mean)
    grad = torch.autograd.grad((rbf * unpad(out["grbf"])).sum() + (cut * unpad(out["gcut"])).sum() + (geo * unpad(out["ggeo"])).sum(), pos)[0]
    f1 = -(std * grad).numpy()
    e1 = std * float(out["e"].sum()) + mean * 1 + float(pot.element_refs.property_offset.double()[z].sum())
    res = {"model": args.model, "atoms": n, "edges": ei.shape[1], "slots": n * d, "layer_dtype": args.dtype,
           "embed_readout_dtype": "float32", "rattle_A": args.rattle, "e_ref_eV_per_atom": e0 / n,
           "dE_meV_per_atom": 1e3 * (e1 - e0) / n,
           "force_mae_meV_A": 1e3 * float(np.abs(f1 - f0).mean()), "force_max_meV_A": 1e3 * float(np.abs(f1 - f0).max()),
           "mean_abs_force_eV_A": float(np.abs(f0).mean()), "device_step_s_untuned": out["seconds"], **remote.call(op="close")}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
