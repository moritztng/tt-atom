"""Generate golden reference tensors for TT-Atom parity tests.

Run with the *reference* environment (fairchem-core, numpy>=2), NOT the ttnn env:

    ~/.ttatom_run/refenv/bin/python tests/gen_golden.py --tiny --out tests/data/golden_tiny.npz

This instantiates the fairchem ``eSCNMDBackbone`` (the eSEN / eSCN-MD / UMA backbone) with
deterministic random weights, runs a forward + autograd-force pass on a small system, and
saves to a single ``.npz``:

  * inputs            (atomic_numbers, pos, edge_index, ...)
  * host geometric    (wigner_and_M_mapping[_inv], edge_envelope, x_edge, sys_node_embedding,
                       x_message_init) -- the terms TT-Atom precomputes on host
  * weights           (the full state_dict, key -> array, under the ``w@`` prefix)
  * activations       (per-module inputs/outputs under the ``a@`` prefix)
  * outputs           (node_embedding, energy, forces)
  * config            (JSON string of the backbone config)

The ttnn port (different env, numpy<2) loads this npz and checks PCC per-module and
end-to-end. .npz arrays are numpy-version agnostic, which is exactly why we decouple the
two environments through disk rather than importing fairchem next to ttnn.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ase.build import molecule, bulk
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone


TINY = dict(
    sphere_channels=32, lmax=2, mmax=2, num_layers=2, hidden_channels=32,
    edge_channels=16, num_distance_basis=32,
)
# Representative "uma-s-like" config (used for perf / full-size goldens).
FULL = dict(
    sphere_channels=128, lmax=2, mmax=2, num_layers=2, hidden_channels=128,
    edge_channels=128, num_distance_basis=512,
)
COMMON = dict(
    max_num_elements=100, cutoff=5.0, max_neighbors=300, otf_graph=False,
    direct_forces=False, regress_forces=True, regress_stress=False,
    norm_type="rms_norm_sh", act_type="gate", ff_type="grid",
    use_dataset_embedding=True, dataset_list=["omat"], distance_function="gaussian",
)


def build_system(kind: str):
    if kind == "molecule":
        atoms = molecule("CH3CH2OH")          # ethanol, 9 atoms, aperiodic
        atoms.info["charge"] = 0
        atoms.info["spin"] = 0
        return atoms
    if kind == "bulk":
        atoms = bulk("Si", "diamond", a=5.43) * (2, 1, 1)   # small periodic cell
        atoms.rattle(stdev=0.1, seed=1)                      # break symmetry -> nonzero forces
        atoms.info["charge"] = 0
        atoms.info["spin"] = 0
        return atoms
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="use the tiny config (committed golden)")
    ap.add_argument("--system", default="molecule", choices=["molecule", "bulk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = dict(COMMON)
    cfg.update(TINY if args.tiny else FULL)

    backbone = eSCNMDBackbone(**cfg).eval()
    # energy head: matches MLP_Energy_Head (sum of per-node MLP on l=0 channel)
    sc, hc = cfg["sphere_channels"], cfg["hidden_channels"]
    energy_block = torch.nn.Sequential(
        torch.nn.Linear(sc, hc), torch.nn.SiLU(),
        torch.nn.Linear(hc, hc), torch.nn.SiLU(),
        torch.nn.Linear(hc, 1),
    )

    atoms = build_system(args.system)
    # aperiodic molecules need a vacuum box for the periodic neighbour-list builder
    mol_box = 12.0 if args.system == "molecule" else None
    data = AtomicData.from_ase(
        atoms, r_edges=True, radius=cfg["cutoff"], max_neigh=cfg["max_neighbors"],
        molecule_cell_size=mol_box, task_name="omat", target_dtype=torch.float32,
    )

    saved: dict[str, np.ndarray] = {}

    def npy(t):
        return t.detach().to(torch.float32).cpu().numpy() if t.dtype.is_floating_point \
            else t.detach().cpu().numpy()

    # ---- capture per-module activations via hooks ----------------------------------
    acts: dict[str, np.ndarray] = {}

    def save_io(name):
        def hook(mod, inp, out):
            for i, t in enumerate(inp):
                if torch.is_tensor(t):
                    acts[f"{name}.in{i}"] = npy(t)
            outs = out if isinstance(out, tuple) else (out,)
            for i, t in enumerate(outs):
                if torch.is_tensor(t):
                    acts[f"{name}.out{i}"] = npy(t)
        return hook

    handles = []
    handles.append(backbone.edge_degree_embedding.register_forward_hook(save_io("edge_degree")))
    handles.append(backbone.norm.register_forward_hook(save_io("final_norm")))
    for li, blk in enumerate(backbone.blocks):
        handles.append(blk.register_forward_hook(save_io(f"block{li}")))
        handles.append(blk.norm_1.register_forward_hook(save_io(f"block{li}.norm_1")))
        handles.append(blk.edge_wise.register_forward_hook(save_io(f"block{li}.edgewise")))
        handles.append(blk.edge_wise.so2_conv_1.register_forward_hook(save_io(f"block{li}.so2_1")))
        handles.append(blk.edge_wise.so2_conv_2.register_forward_hook(save_io(f"block{li}.so2_2")))
        handles.append(blk.norm_2.register_forward_hook(save_io(f"block{li}.norm_2")))
        handles.append(blk.atom_wise.register_forward_hook(save_io(f"block{li}.atomwise")))

    # capture the host geometric terms produced inside forward by wrapping the method
    orig_wigner = backbone._get_rotmat_and_wigner
    captured = {}

    def wrap_wigner(edge_distance_vecs):
        w, winv = orig_wigner(edge_distance_vecs)
        captured["wigner"] = npy(w)
        captured["wigner_inv"] = npy(winv)
        return w, winv

    backbone._get_rotmat_and_wigner = wrap_wigner

    # ---- forward + autograd forces -------------------------------------------------
    out = backbone(data)
    node_emb = out["node_embedding"]
    node_energy = energy_block(node_emb.narrow(1, 0, 1).squeeze(1)).view(-1)
    energy = torch.zeros(len(data["natoms"]))
    energy.index_add_(0, data["batch"], node_energy)
    forces = -torch.autograd.grad(energy.sum(), data["pos"])[0]

    for h in handles:
        h.remove()

    # ---- assemble npz --------------------------------------------------------------
    saved["config"] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)
    # inputs
    saved["in@atomic_numbers"] = npy(data["atomic_numbers"])
    saved["in@pos"] = npy(data["pos"])
    saved["in@edge_index"] = npy(data["edge_index"])
    saved["in@cell"] = npy(data["cell"])
    saved["in@batch"] = npy(data["batch"])
    saved["in@natoms"] = npy(data["natoms"])
    saved["in@charge"] = npy(data["charge"])
    saved["in@spin"] = npy(data["spin"])
    # host geometric terms TT-Atom precomputes
    saved["host@wigner"] = captured["wigner"]
    saved["host@wigner_inv"] = captured["wigner_inv"]
    saved["host@x_edge"] = acts["block0.edgewise.in1"]          # x_edge fed to edgewise
    saved["host@edge_envelope"] = acts["block0.edgewise.in6"]   # edge_envelope arg
    saved["host@x_message_init"] = acts["edge_degree.out0"]     # node feats after edge-degree emb
    # SO3 grid transform matrices (fixed) for GridAtomwise; and the per-node system embedding
    sg = backbone.SO3_grid["lmax_lmax"]
    saved["host@to_grid_mat"] = npy(sg.to_grid_mat)
    saved["host@from_grid_mat"] = npy(sg.from_grid_mat)
    csd = backbone.csd_embedding(data["charge"], data["spin"], data.get("dataset", default=None))
    saved["host@sys_node_embedding"] = npy(csd[data["batch"]])
    # outputs
    saved["out@node_embedding"] = npy(node_emb)
    saved["out@energy"] = npy(energy)
    saved["out@forces"] = npy(forces)
    # weights
    for k, v in backbone.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in energy_block.state_dict().items():
        saved[f"w@energy_block.{k}"] = npy(v)
    # activations
    for k, v in acts.items():
        saved[f"a@{k}"] = v

    np.savez(args.out, **saved)
    print(f"wrote {args.out}")
    print(f"  config: {cfg}")
    print(f"  natoms={int(data['natoms'].sum())} nedges={data['edge_index'].shape[1]} "
          f"node_emb={tuple(node_emb.shape)}")
    print(f"  energy={energy.tolist()}  |F|max={forces.abs().max().item():.4f}")
    print(f"  n weight tensors={sum(1 for k in saved if k.startswith('w@'))} "
          f"n activations={len(acts)}")


if __name__ == "__main__":
    main()
