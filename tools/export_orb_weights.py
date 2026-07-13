"""Export a TT-Atom Orb-v3/OrbMol weight bundle from a real checkpoint (refenv only).

Unlike UMA (``tools/export_weights.py``), Orb has no MoLE expert routing to merge — the raw
checkpoint weights are valid for *any* composition/charge/spin, so this is a plain one-time
per-checkpoint export (no structure, no forward pass), read back by ``tt_atom.orb_weights.OrbWeights``
in the ttnn env. This is what lets an Orb ``Calculator`` skip the per-composition bundle dance a
UMA ``Calculator`` needs (see README's "Model coverage").

Run with the reference (numpy>=2, has ``orb-models``) env:

    ~/.ttatom_run/refenv/bin/python tools/export_orb_weights.py \
        --ckpt conservative-inf-omat --out ~/.cache/tt_atom/orb_weights/conservative-inf-omat.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from orb_models.forcefield import pretrained

CKPTS = {
    "conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
    "direct-20-omat": pretrained.orb_v3_direct_20_omat,
    "conservative-omol": pretrained.orb_v3_conservative_omol,
    "direct-omol": pretrained.orb_v3_direct_omol,
}


def npy(t):
    return t.detach().to(torch.float32).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CKPTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    orbff = CKPTS[args.ckpt](device="cpu", precision="float32-highest")
    orbff.eval()
    gns = orbff.model  # MoleculeGNS

    # latent_dim is the encoder's output width -- read directly off its own weight tensor
    # (mirrors gen_golden_orb.py, which reads it off a captured activation instead; no
    # activations are captured here since there's no golden forward pass to hook).
    latent_dim = int(gns.state_dict()["_encoder._node_fn.mlp.NN-2.weight"].shape[0])
    cfg = dict(
        latent_dim=latent_dim,
        num_message_passing_steps=gns.num_message_passing_steps,
        edge_embed_size=gns.edge_embed_size,
        node_embed_size=gns.node_embed_size,
        lmax=gns.angular_transform._lmax,
        num_bases=gns.rbf_transform.num_bases,
        outer_product_with_cutoff=gns.outer_product_with_cutoff,
        checkpoint=args.ckpt,
        task="omol" if "omol" in args.ckpt else "omat",
        has_charge_spin_cond=gns.conditioner is not None,
        cutoff=float(orbff.system_config.radius),
        max_num_neighbors=int(orbff.system_config.max_num_neighbors),
    )

    saved: dict[str, np.ndarray] = {"config": np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)}
    for k, v in gns.state_dict().items():
        saved[f"w@{k}"] = npy(v)
    for k, v in orbff.heads["energy"].state_dict().items():
        saved[f"w@energy_head.{k}"] = npy(v)
    if "forces" in orbff.heads:
        for k, v in orbff.heads["forces"].state_dict().items():
            saved[f"w@forces_head.{k}"] = npy(v)
    if "stress" in orbff.heads:
        for k, v in orbff.heads["stress"].state_dict().items():
            saved[f"w@stress_head.{k}"] = npy(v)
    if orbff.pair_repulsion:
        for k, v in orbff.pair_repulsion_fn.state_dict().items():
            saved[f"w@pair_repulsion.{k}"] = npy(v)
    if gns.conditioner is not None:
        for k, v in gns.conditioner.state_dict().items():
            saved[f"w@conditioner.{k}"] = npy(v)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out}  (config={cfg})")
    print(f"  n weight tensors={sum(1 for k in saved if k.startswith('w@'))}")


if __name__ == "__main__":
    main()
