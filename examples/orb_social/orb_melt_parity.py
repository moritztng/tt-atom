"""On-device Orb-v3 vs orb-models CPU reference parity, on frames of the actual Si melt trajectory.

Confirms the Tenstorrent trajectory is the *real* Orb-v3 (not a degraded port): energy +
conservative forces are compared against orb-models on CPU for selected frames spanning the
melt — a cold lattice frame, a thermal solid frame, and a liquid frame. The device path uses a
fresh calculator per frame so the neighbour list matches that exact geometry (isolating model
parity from the MD's rebuildable-topology optimisation).

Two-stage (device env = py3.10+ttnn, reference env = py3.11+orb_models):

    # 1. CPU reference (refenv)
    ~/.ttatom_run/refenv/bin/python examples/orb_social/orb_melt_parity.py ref \
        --traj si_melt.extxyz --frames 0 350 700 --out parity_ref.npz
    # 2. on-device + compare (device env)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=<tt-atom> ~/.ttatom_run/env/bin/python examples/orb_social/orb_melt_parity.py device \
        --traj si_melt.extxyz --frames 0 350 700 --weights <golden.npz> \
        --ref parity_ref.npz --out parity.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from ase.io import read

MODEL = "orb-v3-conservative-inf-omat"


def _pcc(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def run_ref(args):
    import torch
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    orbff = pretrained.ORB_PRETRAINED_MODELS[MODEL](device="cpu", precision="float32-high")
    calc = ORBCalculator(orbff, device="cpu")
    frames = read(args.traj, index=":")
    out = {}
    for fi in args.frames:
        at = frames[fi]
        at.calc = calc
        e = float(at.get_potential_energy())
        f = at.get_forces().astype(np.float64)
        out[f"e_{fi}"] = np.array([e])
        out[f"f_{fi}"] = f
        print(f"[ref ] frame {fi:4d}  E={e:.4f} eV ({e/len(at):.4f} eV/atom)  "
              f"|F|max={np.abs(f).max():.4f} eV/A", flush=True)
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


def run_device(args):
    import sys
    sys.path.insert(0, args.md_dir)
    from orb_melt_md import OrbMeltCalculator

    ref = np.load(args.ref)
    frames = read(args.traj, index=":")
    rows = []
    for fi in args.frames:
        at = frames[fi]
        calc = OrbMeltCalculator(args.weights, device_id=args.device_id)   # fresh -> topology per frame
        try:
            at.calc = calc
            e = float(at.get_potential_energy())
            f = at.get_forces().astype(np.float64)
        finally:
            calc.close()
        e_ref = float(ref[f"e_{fi}"][0])
        f_ref = ref[f"f_{fi}"]
        n = len(at)
        row = {
            "frame": fi,
            "n_atoms": n,
            "e_dev": e, "e_ref": e_ref,
            "e_abs_diff_meV_per_atom": abs(e - e_ref) / n * 1e3,
            "f_pcc": _pcc(f, f_ref),
            "f_max_abs_diff": float(np.abs(f - f_ref).max()),
            "f_max_abs_ref": float(np.abs(f_ref).max()),
        }
        rows.append(row)
        print(f"[dev ] frame {fi:4d}  E_dev={e:.4f}  E_ref={e_ref:.4f}  "
              f"dE={row['e_abs_diff_meV_per_atom']:.3f} meV/atom  "
              f"F_PCC={row['f_pcc']:.6f}  F_maxdiff={row['f_max_abs_diff']:.4f} eV/A "
              f"(ref |F|max={row['f_max_abs_ref']:.3f})", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"model": MODEL, "frames": rows}, fh, indent=2)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--traj", required=True)
    r.add_argument("--frames", type=int, nargs="+", required=True)
    r.add_argument("--out", required=True)
    d = sub.add_parser("device")
    d.add_argument("--traj", required=True)
    d.add_argument("--frames", type=int, nargs="+", required=True)
    d.add_argument("--weights", required=True)
    d.add_argument("--ref", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--device-id", type=int, default=0)
    d.add_argument("--md-dir", default=".")
    args = ap.parse_args()
    (run_ref if args.mode == "ref" else run_device)(args)


if __name__ == "__main__":
    main()
