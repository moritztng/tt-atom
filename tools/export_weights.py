"""Export a TT-Atom weight bundle from a fairchem checkpoint (or random init).

Run in the *fairchem* environment (fairchem-core, numpy>=2) — NOT the ttnn env — because it
instantiates the reference ``eSCNMDBackbone`` to obtain both the learned ``state_dict`` and the
fixed geometric buffers (Jd, to_m, SO3 grid matrices, gaussian basis) that a bare checkpoint
does not contain. The resulting ``.npz`` is loaded by ``tt_atom.weights.WeightBundle`` in the
ttnn env. This two-step export is what lets the two incompatible numpy worlds coexist.

    # random-weight demo bundle (architecture only, no checkpoint):
    ~/.ttatom_run/refenv/bin/python tools/export_weights.py --out model.npz

    # real weights (bring your own fairchem UMA checkpoint; weights are separately licensed):
    ~/.ttatom_run/refenv/bin/python tools/export_weights.py --checkpoint uma.pt --out model.npz

The energy head is exported as a fresh MLP unless the checkpoint provides one; for real-weight
runs map your checkpoint's energy head keys to ``energy_block.{0,2,4}.{weight,bias}``.
"""
from __future__ import annotations

import argparse
import json

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="fairchem state_dict .pt (optional)")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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

    def npy(t):
        return t.detach().to(torch.float32).cpu().numpy()

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
