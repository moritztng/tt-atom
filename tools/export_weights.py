"""Export a TT-Atom weight bundle from a fairchem checkpoint (or random init).

Run in the *fairchem* environment (fairchem-core, numpy>=2) — NOT the ttnn env — because it
instantiates the reference ``eSCNMDBackbone`` to obtain both the learned ``state_dict`` and the
fixed geometric buffers (Jd, to_m, SO3 grid matrices, gaussian basis) that a bare checkpoint
does not contain. The resulting ``.npz`` is loaded by ``tt_atom.weights.WeightBundle`` in the
ttnn env. This two-step export is what lets the two incompatible numpy worlds coexist.

    # random-weight demo bundle (architecture only, no checkpoint):
    ~/.ttatom_run/refenv/bin/python tools/export_weights.py --out model.npz

    # plain backbone checkpoint (no MoLE), fresh energy head:
    ~/.ttatom_run/refenv/bin/python tools/export_weights.py --checkpoint plain.pt --out model.npz

    # real released uma-s-1 (gated facebook/UMA; MoLE-merged on host for a fixed composition):
    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tools/export_weights.py \
        --uma-s-1 --molecule CH3CH2OH --task omol --charge 0 --spin 1 --out uma_s_ethanol.npz

For ``--uma-s-1`` the bundle is composition-specific (MoLE routing is fixed at merge time), so it
is valid for systems with the same reduced composition / charge / spin / dataset — exactly the
constant-composition regime of a relaxation or MD run. The exported bundle carries the per-task
energy normalizer (``scale@*``) and the real energy head. No weights are committed/redistributed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fairchem.core.models.uma.escn_md import eSCNMDBackbone

TINY = dict(sphere_channels=32, lmax=2, mmax=2, num_layers=2, hidden_channels=32,
            edge_channels=16, num_distance_basis=32)
FULL = dict(sphere_channels=128, lmax=2, mmax=2, num_layers=2, hidden_channels=128,
            edge_channels=128, num_distance_basis=512)
COMMON = dict(max_num_elements=100, cutoff=5.0, max_neighbors=300, otf_graph=False,
              direct_forces=False, regress_forces=True, regress_stress=False,
              norm_type="rms_norm_sh", act_type="gate", ff_type="grid",
              use_dataset_embedding=True, dataset_list=["omat"], distance_function="gaussian")


def npy(t):
    return t.detach().to(torch.float32).cpu().numpy()


def export_uma_s_1(args):
    """Export the released uma-s-1 checkpoint: host MoLE-merge to a plain backbone for the given
    composition, then write a clean WeightBundle (weights + fixed buffers + energy normalizer)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from ase.build import molecule
    from huggingface_hub import hf_hub_download
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    ckpt = args.checkpoint or hf_hub_download("facebook/UMA", "checkpoints/uma-s-1.pt")
    settings = InferenceSettings(tf32=False, activation_checkpointing=True, merge_mole=True,
                                 compile=False, external_graph_gen=False, internal_graph_gen_version=2)
    pu = load_predict_unit(ckpt, inference_settings=settings, device="cpu")
    calc = FAIRChemCalculator(pu, task_name=args.task)
    if args.xyz:
        from ase.io import read as _read
        atoms = _read(args.xyz)
    else:
        atoms = molecule(args.molecule)
    atoms.info.update(charge=args.charge, spin=args.spin)
    atoms.calc = calc
    E_ref = float(atoms.get_potential_energy())           # triggers the host MoLE merge
    F_ref = atoms.get_forces().astype(np.float32)

    bb = pu.model.module.backbone                         # plain eSCNMDBackbone after merge
    assert type(bb).__name__ == "eSCNMDBackbone", "merge did not produce a plain backbone"
    energy_block = pu.model.module.output_heads["energyandforcehead"].head.energy_block
    etask = pu.model.module.tasks[f"{args.task}_energy"]

    cfg = dict(sphere_channels=bb.sphere_channels, lmax=bb.lmax, mmax=bb.mmax,
               num_layers=len(bb.blocks), hidden_channels=bb.hidden_channels,
               num_distance_basis=int(bb.distance_expansion.offset.numel()),
               cutoff=float(bb.cutoff), ff_type="spectral", act_type="gate",
               norm_type="rms_norm_sh", chg_spin_emb_type=bb.chg_spin_emb_type, task=args.task,
               # charge_balanced_channels (uma-s-1.2): l=0 scalar channels re-balanced to the
               # system charge after each block. cs==ce (default) => disabled (uma-s-1).
               charge_channel_start=int(getattr(bb, "charge_channel_start", 0)),
               charge_channel_end=int(getattr(bb, "charge_channel_end", 0)))

    saved = {"config": np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)}
    for k, v in bb.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in energy_block.state_dict().items():
        saved[f"w@energy_block.{k}"] = npy(v)
    sg = bb.SO3_grid["lmax_lmax"]
    saved["host@to_m"] = npy(bb.mappingReduced.to_m)
    saved["host@to_grid_mat"] = npy(sg.to_grid_mat)
    saved["host@from_grid_mat"] = npy(sg.from_grid_mat)
    saved["host@gauss_offset"] = npy(bb.distance_expansion.offset)
    saved["host@gauss_coeff"] = np.array([bb.distance_expansion.coeff], dtype=np.float32)
    saved["scale@rmsd"] = np.array([float(etask.normalizer.rmsd)], dtype=np.float64)
    saved["scale@mean"] = np.array([float(etask.normalizer.mean)], dtype=np.float64)
    saved["scale@elem_refs"] = etask.element_references.element_references.detach().cpu().numpy().astype(np.float64)

    # embed the fairchem reference E/F for this composition so `tt-atom verify` can close the
    # roundtrip on device (the two numpy worlds cannot share a process, so we carry the numbers).
    saved["ref@energy"] = np.array([E_ref], dtype=np.float64)
    saved["ref@forces"] = F_ref
    saved["ref@pos"] = npy(torch.as_tensor(atoms.get_positions()))
    saved["ref@atomic_numbers"] = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    saved["ref@charge"] = np.array([float(args.charge)], dtype=np.float64)
    saved["ref@spin"] = np.array([float(args.spin)], dtype=np.float64)
    saved["ref@cell"] = npy(torch.as_tensor(atoms.get_cell().array))
    saved["ref@pbc"] = np.asarray(atoms.get_pbc(), dtype=bool)

    np.savez(args.out, **saved)
    print(f"wrote {args.out}  ({sum(1 for k in saved if k.startswith('w@'))} weight tensors, "
          f"uma-s-1 merged for {args.xyz or args.molecule} charge={args.charge} spin={args.spin} task={args.task})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="fairchem state_dict .pt (optional)")
    ap.add_argument("--uma-s-1", action="store_true", help="export the released uma-s-1 (MoLE-merged)")
    ap.add_argument("--xyz", default=None, help="structure file (overrides --molecule; for compositions not in ASE g2)")
    ap.add_argument("--molecule", default="CH3CH2OH", help="ASE molecule name for uma-s-1 routing")
    ap.add_argument("--task", default="omol")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.uma_s_1:
        export_uma_s_1(args)
        return

    torch.manual_seed(args.seed)
    cfg = dict(COMMON)
    cfg.update(TINY if args.tiny else FULL)
    bb = eSCNMDBackbone(**cfg).eval()
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("backbone.", "", 1): v for k, v in sd.items()}
        missing, unexpected = bb.load_state_dict(sd, strict=False)
        print(f"loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    sc, hc = cfg["sphere_channels"], cfg["hidden_channels"]
    energy = torch.nn.Sequential(torch.nn.Linear(sc, hc), torch.nn.SiLU(),
                                 torch.nn.Linear(hc, hc), torch.nn.SiLU(), torch.nn.Linear(hc, 1))

    saved = {"config": np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)}
    for k, v in bb.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in energy.state_dict().items():
        saved[f"w@energy_block.{k}"] = npy(v)
    sg = bb.SO3_grid["lmax_lmax"]
    saved["host@to_m"] = npy(bb.mappingReduced.to_m)
    saved["host@to_grid_mat"] = npy(sg.to_grid_mat)
    saved["host@from_grid_mat"] = npy(sg.from_grid_mat)
    saved["host@gauss_offset"] = npy(bb.distance_expansion.offset)
    saved["host@gauss_coeff"] = np.array([bb.distance_expansion.coeff], dtype=np.float32)

    np.savez(args.out, **saved)
    print(f"wrote {args.out}  ({sum(1 for k in saved if k.startswith('w@'))} weight tensors)")


if __name__ == "__main__":
    main()
