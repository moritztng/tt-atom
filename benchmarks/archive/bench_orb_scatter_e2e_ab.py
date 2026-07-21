"""Controlled A/B for the row-major scatter optimization: old (TILE concat) vs new (RM concat),
same process, same warmup, repeated to expose run-to-run variance. One OrbDeviceCalculator per
cell (each captures its own trace with the active scatter path).

    source .source_env.sh
    TT_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH $PYREFENV benchmarks/bench_orb_scatter_e2e_ab.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz
"""
from __future__ import annotations

import argparse
import os
import statistics

import numpy as np
from ase.build import bulk


def _measure(weights, device_id, nx, ny, nz, *, warmup, steps, seed):
    from examples.orb_md import OrbDeviceCalculator
    atoms0 = bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz)
    calc = OrbDeviceCalculator(weights, device_id=device_id)
    atoms = atoms0.copy()
    atoms.calc = calc
    rng = np.random.default_rng(seed)
    atoms.set_positions(atoms.get_positions())
    _ = atoms.get_potential_energy(); _ = atoms.get_forces()
    for _ in range(warmup - 1):
        p = atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape)
        atoms.set_positions(p)
        _ = atoms.get_potential_energy(); _ = atoms.get_forces()
    calc.step_ms.clear()
    for _ in range(steps):
        p = atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape)
        atoms.set_positions(p)
        _ = atoms.get_potential_energy(); _ = atoms.get_forces()
    med = float(np.median(list(calc.step_ms)))
    calc.close()
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    sizes = [("3x3x3", 3, 3, 3), ("4x4x4", 4, 4, 4), ("5x5x5", 5, 5, 5), ("6x6x7", 6, 6, 7)]
    print(f"reps={args.reps} warmup={args.warmup} steps={args.steps}\n", flush=True)
    print(f"{'size':8s} {'N':>5s} {'E':>7s} {'old_ms':>9s} {'new_ms':>9s} {'speedup':>8s} "
          f"{'gap_closed':>11s}", flush=True)
    results = []
    for tag, nx, ny, nz in sizes:
        N = len(bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz))
        old_runs, new_runs = [], []
        for r in range(args.reps):
            os.environ["TT_ATOM_ORB_SCATTER_RM"] = "0"
            old_runs.append(_measure(args.weights, args.device_id, nx, ny, nz,
                                     warmup=args.warmup, steps=args.steps, seed=args.seed + r))
            os.environ["TT_ATOM_ORB_SCATTER_RM"] = "1"
            new_runs.append(_measure(args.weights, args.device_id, nx, ny, nz,
                                     warmup=args.warmup, steps=args.steps, seed=args.seed + r))
        old = float(statistics.median(old_runs))
        new = float(statistics.median(new_runs))
        # H200 reference times from the prior fair GPU leg (benchmarks/orb_perf_dollar_gpu_v0.7.0.json)
        h200 = {"3x3x3": 16.85, "4x4x4": 19.43, "5x5x5": 44.47, "6x6x7": 70.51}[tag]
        gap_closed = 100.0 * (old - new) / (old - h200)
        results.append({"tag": tag, "N": N, "old_ms": old, "new_ms": new,
                        "speedup": old / new, "h200_ms": h200, "gap_closed_pct": gap_closed,
                        "old_runs": [round(x, 3) for x in old_runs],
                        "new_runs": [round(x, 3) for x in new_runs]})
        print(f"{tag:8s} {N:5d} {'':7s} {old:9.3f} {new:9.3f} {old/new:7.3f}x {gap_closed:10.2f}%",
              flush=True)
    import json
    with open("benchmarks/orb_scatter_e2e_ab.json", "w") as f:
        json.dump({"warmup": args.warmup, "steps": args.steps, "reps": args.reps,
                   "results": results}, f, indent=2)
    print("\nwrote benchmarks/orb_scatter_e2e_ab.json", flush=True)


if __name__ == "__main__":
    main()
