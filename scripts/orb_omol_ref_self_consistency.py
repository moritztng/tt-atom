"""Confirm the orb-models CPU reference is deterministic (self-consistency PCC = 1.0):
two independent predict() calls on the same open-shell system must give bit-identical
forces. This is the reference's own seed-to-seed floor -- it's 1.0 (eval mode, fp32-highest,
no sampling/dropout), so the device's 0.8926 is below the reference's own consistency, and
the relevant floor is the bf16 precision floor (see notes/noise_floor_analysis.py).

    ~/.ttatom_run/refenv/bin/python scripts/orb_omol_ref_self_consistency.py
"""
from __future__ import annotations
import math, numpy as np, torch
from ase import Atoms
from orb_models.forcefield import pretrained
from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs


def build_openshell():
    r = 1.079
    pos = [[0.0, 0.0, 0.0]]
    for k in range(3):
        t = 2 * math.pi * k / 3
        pos.append([r * math.cos(t), r * math.sin(t), 0.0])
    a = Atoms("CH3", positions=pos); a.info.update(charge=0, spin=2); return a


def run(tag):
    torch.manual_seed(0)
    f = pretrained.orb_v3_direct_omol(device="cpu", precision="float32-highest"); f.eval()
    g = ase_atoms_to_atom_graphs(build_openshell(), f.system_config, device=torch.device("cpu"))
    r = f.predict(g, split=False)
    return r["forces"].detach().to(torch.float32).cpu().numpy()


def main():
    f1 = run("direct"); f2 = run("direct")
    print("max abs diff over 2 reference re-runs:", float(np.abs(f1 - f2).max()))
    print("bit-identical:", np.array_equal(f1, f2))
    print("self-consistency PCC:", float(np.corrcoef(f1.ravel(), f2.ravel())[0, 1]))


if __name__ == "__main__":
    main()
