"""Generate REAL-weight golden tensors for TT-Atom uma-s-1 parity (refenv only).

Run with the *reference* environment (fairchem-core, numpy>=2), NOT the ttnn env:

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system molecule --task omol --out ~/.ttatom_run/goldens_real/ethanol_omol.npz

Unlike ``gen_golden.py`` (random weights, self-chosen config), this loads the gated
``facebook/UMA`` ``uma-s-1`` checkpoint and reproduces the *released* model:

  * EMA weights (the predict unit uses ``use_ema=True``);
  * MoLE merge on host -> a plain ``eSCNMDBackbone`` (fairchem's own
    ``merge_MOLE_model``, the exact inference path with ``merge_mole=True``);
  * ``ff_type=spectral`` atomwise, ``num_layers=4``, ``num_distance_basis=64``, ``cutoff=6``;
  * the per-task energy normalizer (``E = rmsd*E_raw + sum_i refs[Z_i]``) and force
    scale (``F = rmsd * F_raw``).

It also records the *unmerged* MoE oracle energy+forces (a fresh predict unit with the
default ``merge_mole=False`` settings) so the merge can be validated to PCC>0.999 (the host
correctness anchor). Goldens are dumped to disk (NOT committed) and consumed by the ttnn env.

The bundle layout mirrors ``gen_golden.py`` so the ttnn loader / tests share one code path;
spectral-atomwise activations replace the grid ones, and ``scale@*`` carries the normalizer.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from ase.build import molecule, bulk
from huggingface_hub import hf_hub_download
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def npy(t):
    return t.detach().to(torch.float32).cpu().numpy() if t.dtype.is_floating_point \
        else t.detach().cpu().numpy()


def build_system(kind: str, task: str):
    if kind == "molecule":
        atoms = molecule("CH3CH2OH")              # ethanol, 9 atoms, aperiodic
        atoms.info.update(charge=0, spin=1)       # omol default (closed-shell singlet)
        return atoms
    if kind == "bulk":
        atoms = bulk("Si", "diamond", a=5.43) * (2, 1, 1)
        atoms.rattle(stdev=0.1, seed=1)
        atoms.info.update(charge=0, spin=1)
        return atoms
    if kind == "slab":
        from ase.build import fcc100, add_adsorbate       # Cu(100) slab + H adsorbate (oc20)
        atoms = fcc100("Cu", (2, 2, 2), vacuum=8.0)
        add_adsorbate(atoms, "H", height=1.5, position="hollow")
        atoms.rattle(stdev=0.05, seed=2)
        atoms.info.update(charge=0, spin=1)               # pbc = [True, True, False] (mixed)
        return atoms
    if kind == "mof":
        # odac (DAC / MOFs): a metal-oxide framework fragment (MgO), fully periodic. A minimal
        # inorganic stand-in that exercises the odac dataset token + normalizer on the periodic path.
        atoms = bulk("MgO", "rocksalt", a=4.21) * (2, 1, 1)
        atoms.rattle(stdev=0.08, seed=3)
        atoms.info.update(charge=0, spin=1)
        return atoms
    if kind == "molcrystal":
        # omc (molecular crystals): solid CO2 (dry ice) as a periodic cubic cell dense enough that
        # periodic images fall within the 6 A cutoff (tests the cell-aware neighbour list).
        co2 = molecule("CO2")
        co2.set_cell([5.0, 5.0, 5.0])
        co2.set_pbc(True)
        co2.center()
        co2.rattle(stdev=0.05, seed=4)
        co2.info.update(charge=0, spin=1)
        return co2
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="molecule",
                    choices=["molecule", "bulk", "slab", "mof", "molcrystal"])
    ap.add_argument("--task", default="omol")
    ap.add_argument("--ckpt", default="uma-s-1", help="UMA checkpoint name (e.g. uma-s-1, uma-m-1p1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = hf_hub_download("facebook/UMA", f"checkpoints/{args.ckpt}.pt")
    atoms = build_system(args.system, args.task)

    # ---- ground-truth oracle: unmerged MoE (the released inference default) --------------
    pu_oracle = load_predict_unit(ckpt, inference_settings="default", device="cpu")
    calc_oracle = FAIRChemCalculator(pu_oracle, task_name=args.task)
    atoms.calc = calc_oracle
    E_oracle = float(atoms.get_potential_energy())
    F_oracle = atoms.get_forces().astype(np.float32)
    # stress (ASE Voigt-6) only for a fully-periodic cell; zeros as a sentinel otherwise
    S_oracle = (atoms.get_stress().astype(np.float32)
                if bool(np.all(atoms.get_pbc())) else np.zeros(6, dtype=np.float32))
    print(f"oracle (unmerged MoE): E={E_oracle:.6f} eV  |F|max={np.abs(F_oracle).max():.4f} "
          f"stress={S_oracle}")

    # ---- merged plain backbone: host MoLE merge (the correctness anchor) ------------------
    settings = InferenceSettings(
        tf32=False, activation_checkpointing=True, merge_mole=True,
        compile=False, external_graph_gen=False, internal_graph_gen_version=2,
    )
    pu = load_predict_unit(ckpt, inference_settings=settings, device="cpu")
    calc = FAIRChemCalculator(pu, task_name=args.task)
    atoms2 = build_system(args.system, args.task)
    atoms2.calc = calc
    E_merged = float(atoms2.get_potential_energy())     # triggers the merge in _lazy_init
    F_merged = atoms2.get_forces().astype(np.float32)
    print(f"merged (host MoLE):    E={E_merged:.6f} eV  |F|max={np.abs(F_merged).max():.4f}")

    hydra = pu.model.module
    backbone = hydra.backbone                            # plain eSCNMDBackbone after merge
    head = hydra.output_heads["energyandforcehead"].head
    energy_block = head.energy_block
    assert type(backbone).__name__ == "eSCNMDBackbone", \
        f"backbone was not merged to a plain backbone (got {type(backbone).__name__})"

    # per-task normalizer (energy scale + element references) -------------------------------
    etask = pu.model.module.tasks[f"{args.task}_energy"]
    rmsd = float(etask.normalizer.rmsd)
    mean = float(etask.normalizer.mean)
    elem_refs = etask.element_references.element_references.detach().cpu().numpy().astype(np.float64)

    # ---- hooks to capture per-module activations on the MERGED backbone -------------------
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
    # the real backbone pre-fuses the polynomial envelope into wigner_inv (no separate edgewise
    # arg), so capture it straight off the PolynomialEnvelope module (TT-Atom applies it separately)
    handles.append(backbone.envelope.register_forward_hook(save_io("envelope")))
    handles.append(backbone.edge_degree_embedding.register_forward_hook(save_io("edge_degree")))
    handles.append(backbone.norm.register_forward_hook(save_io("final_norm")))
    for li, blk in enumerate(backbone.blocks):
        handles.append(blk.register_forward_hook(save_io(f"block{li}")))
        handles.append(blk.norm_1.register_forward_hook(save_io(f"block{li}.norm_1")))
        handles.append(blk.edge_wise.register_forward_hook(save_io(f"block{li}.edgewise")))
        handles.append(blk.edge_wise.so2_conv_1.register_forward_hook(save_io(f"block{li}.so2_1")))
        handles.append(blk.edge_wise.so2_conv_2.register_forward_hook(save_io(f"block{li}.so2_2")))
        handles.append(blk.norm_2.register_forward_hook(save_io(f"block{li}.norm_2")))
        aw = blk.atom_wise                                  # SpectralAtomwise
        handles.append(aw.register_forward_hook(save_io(f"block{li}.atomwise")))
        handles.append(aw.scalar_mlp.register_forward_hook(save_io(f"block{li}.aw_scalar")))
        handles.append(aw.so3_linear_1.register_forward_hook(save_io(f"block{li}.aw_so3lin1")))
        handles.append(aw.act.register_forward_hook(save_io(f"block{li}.aw_gate")))
        handles.append(aw.so3_linear_2.register_forward_hook(save_io(f"block{li}.aw_so3lin2")))

    captured = {}
    orig_wigner = backbone._get_rotmat_and_wigner

    def wrap_wigner(edge_distance_vecs):
        w, winv = orig_wigner(edge_distance_vecs)
        captured["wigner"] = npy(w)
        captured["wigner_inv"] = npy(winv)
        return w, winv

    backbone._get_rotmat_and_wigner = wrap_wigner

    orig_graph = backbone._generate_graph

    def wrap_graph(dd):
        gd = orig_graph(dd)
        captured["edge_distance_vec"] = npy(gd["edge_distance_vec"])
        captured["edge_distance"] = npy(gd["edge_distance"])
        captured["edge_index"] = npy(gd["edge_index"])
        return gd

    backbone._generate_graph = wrap_graph

    # re-run via the calculator to fire hooks on the merged path; capture the data object too
    data = calc.a2g(atoms2)
    captured["data"] = data
    atoms2.calc.calculate(atoms2, ["energy", "forces"], ["positions"])  # re-run -> fills acts

    for h in handles:
        h.remove()
    backbone._get_rotmat_and_wigner = orig_wigner
    backbone._generate_graph = orig_graph

    # ---- node embedding + raw/denorm energy from the merged backbone directly -------------
    data = data.clone()
    data.pos.requires_grad_(True)
    emb = backbone(data)
    node_emb = emb["node_embedding"]
    scalar = node_emb.narrow(1, 0, 1).squeeze(1)            # l=0 channel [N, C]
    node_energy = energy_block(scalar).view(-1)
    n_sys = len(data["natoms"])
    E_raw = torch.zeros(n_sys, dtype=node_energy.dtype).index_add(0, data["batch"], node_energy)
    refs_t = torch.from_numpy(elem_refs).to(node_energy.dtype)
    ref_per_atom = refs_t[data["atomic_numbers"].long()]
    E_final = E_raw * rmsd + mean
    E_final = E_final.index_add(0, data["batch"], ref_per_atom)
    F_final = -torch.autograd.grad(E_final.sum(), data.pos)[0]
    print(f"recompute (merged bb):  E={E_final.sum().item():.6f} eV "
          f"|F|max={F_final.abs().max().item():.4f}  E_raw={E_raw.sum().item():.6f}")

    # ---- assemble npz ----------------------------------------------------------------------
    saved: dict[str, np.ndarray] = {}
    # config the ttnn port needs (real uma-s-1 values)
    bb = backbone
    out_cfg = dict(
        sphere_channels=bb.sphere_channels, lmax=bb.lmax, mmax=bb.mmax,
        num_layers=len(bb.blocks), hidden_channels=bb.hidden_channels,
        num_distance_basis=int(bb.distance_expansion.offset.numel()),
        cutoff=float(bb.cutoff), ff_type="spectral", act_type="gate",
        norm_type="rms_norm_sh", chg_spin_emb_type=bb.chg_spin_emb_type, task=args.task,
    )
    saved["config"] = np.frombuffer(json.dumps(out_cfg).encode(), dtype=np.uint8)

    di = captured["data"]
    saved["in@atomic_numbers"] = npy(di["atomic_numbers"])
    saved["in@pos"] = npy(data.pos)
    saved["in@edge_index"] = captured["edge_index"]
    saved["in@cell"] = npy(di["cell"])
    saved["in@pbc"] = np.asarray(atoms2.get_pbc(), dtype=bool)
    saved["in@batch"] = npy(di["batch"])
    saved["in@natoms"] = npy(di["natoms"])
    saved["in@charge"] = npy(di["charge"])
    saved["in@spin"] = npy(di["spin"])

    saved["host@wigner"] = captured["wigner"]
    saved["host@wigner_inv"] = captured["wigner_inv"]
    saved["host@edge_distance_vec"] = captured["edge_distance_vec"]
    saved["host@edge_distance"] = captured["edge_distance"]
    saved["host@to_m"] = npy(backbone.mappingReduced.to_m)
    saved["host@gauss_offset"] = npy(backbone.distance_expansion.offset)
    saved["host@gauss_coeff"] = np.array([backbone.distance_expansion.coeff], dtype=np.float32)
    saved["host@x_edge"] = acts["block0.edgewise.in1"]
    saved["host@edge_envelope"] = acts["envelope.out0"].reshape(-1, 1, 1)
    saved["host@x_message_init"] = acts["edge_degree.out0"]
    sg = backbone.SO3_grid["lmax_lmax"]
    saved["host@to_grid_mat"] = npy(sg.to_grid_mat)
    saved["host@from_grid_mat"] = npy(sg.from_grid_mat)
    csd = backbone.csd_embedding(di["charge"], di["spin"], di.get("dataset", default=None))
    saved["host@sys_node_embedding"] = npy(csd[di["batch"]])

    # energy normalizer / element references
    saved["scale@rmsd"] = np.array([rmsd], dtype=np.float64)
    saved["scale@mean"] = np.array([mean], dtype=np.float64)
    saved["scale@elem_refs"] = elem_refs

    # outputs
    saved["out@node_embedding"] = npy(node_emb)
    saved["out@energy_raw"] = npy(E_raw)
    saved["out@energy"] = np.array([E_final.sum().item()], dtype=np.float64)
    saved["out@forces"] = npy(F_final)
    saved["out@energy_oracle"] = np.array([E_oracle], dtype=np.float64)
    saved["out@forces_oracle"] = F_oracle
    saved["out@stress_oracle"] = S_oracle
    saved["out@energy_merged_oracle"] = np.array([E_merged], dtype=np.float64)
    saved["out@forces_merged_oracle"] = F_merged

    # weights (merged plain backbone + energy head)
    for k, v in backbone.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in energy_block.state_dict().items():
        saved[f"w@energy_block.{k}"] = npy(v)

    for k, v in acts.items():
        saved[f"a@{k}"] = v

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out}")
    print(f"  config: {out_cfg}")
    n_edges = captured["edge_index"].shape[1]
    print(f"  natoms={int(di['natoms'].sum())} nedges={n_edges} node_emb={tuple(node_emb.shape)}")
    print(f"  n weight tensors={sum(1 for k in saved if k.startswith('w@'))} "
          f"n activations={len(acts)}")


if __name__ == "__main__":
    main()
