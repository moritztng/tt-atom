"""On-device Orb-v3 vs orb-models CPU reference parity, on the actual MD system.

The MD demo runs a 216-atom Si diamond supercell. This checks that the Tenstorrent
trajectory is the *real* Orb-v3, not a degraded port, by comparing energy + conservative
forces against orb-models running on CPU, for selected frames of the trajectory.

Two-stage (the device env is py3.10+ttnn, the reference env is py3.11+orb_models):

    # 1. CPU reference (refenv)
    ~/.ttatom_run/refenv/bin/python orb_parity.py ref \
        --traj si216_final.extxyz --frames 0 33 99 --out parity_ref.npz
    # 2. on-device + compare (device env)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=<tt-atom> ~/.ttatom_run/env/bin/python orb_parity.py device \
        --traj si216_final.extxyz --frames 0 33 99 --weights <golden.npz> \
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
    """orb-models conservative-inf-omat on CPU (the reference the port must match)."""
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
              f"|F|max={np.abs(f).max():.4f} eV/A")
    np.savez(args.out, **out)
    print(f"wrote {args.out}")


def run_device(args):
    """On-device Orb-v3 energy+forces for the same frames, then compare to the CPU reference."""
    import sys
    sys.path.insert(0, args.md_dir)
    from orb_md_device import OrbDeviceCalculator

    ref = np.load(args.ref)
    frames = read(args.traj, index=":")
    rows = []
    for fi in args.frames:
        at = frames[fi]
        # fresh calculator per frame so the neighbour list matches THIS geometry
        # (isolates model parity from the MD's frozen-topology optimisation).
        calc = OrbDeviceCalculator(args.weights, device_id=args.device_id)
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
            "e_per_atom_dev": e / n, "e_per_atom_ref": e_ref / n,
            "e_abs_diff_meV_per_atom": abs(e - e_ref) / n * 1e3,
            "f_pcc": _pcc(f, f_ref),
            "f_max_abs_diff": float(np.abs(f - f_ref).max()),
            "f_rms_ref": float(np.sqrt((f_ref**2).mean())),
            "f_max_abs_ref": float(np.abs(f_ref).max()),
        }
        rows.append(row)
        print(f"[dev ] frame {fi:4d}  E_dev={e:.4f}  E_ref={e_ref:.4f}  "
              f"dE={row['e_abs_diff_meV_per_atom']:.3f} meV/atom  "
              f"F_PCC={row['f_pcc']:.6f}  F_maxdiff={row['f_max_abs_diff']:.4f} eV/A "
              f"(ref |F|max={row['f_max_abs_ref']:.3f})")
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
    d.add_argument("--md-dir", default="/home/moritz/.coworker/artifacts/orb-social")
    args = ap.parse_args()
    (run_ref if args.mode == "ref" else run_device)(args)


if __name__ == "__main__":
    main()
