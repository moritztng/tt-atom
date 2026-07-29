#!/usr/bin/env python3
"""Multi-card relax/MD scaling harness — measures wall-clock fan-out across N cards.

Runs a batch of independent structures through MultiCardSim relax at 1, 2, 4 (configured)
cards and prints the per-card wall-clock + the speedup vs 1 card. Each card owns a full
Calculator + FIRE loop for its shard, so the work is embarrassingly parallel: the ceiling is
Nx modulo the fixed per-worker device-open + program-cache warmup cost (the same Amdahl shape
as the measured tt-bio esmc/predict multicard fanouts — see the coworker memories).

Usage:
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/multicard_sim_scaling.py \\
        --model orb-v3-direct-omol --devices 0,1,2,3 --n-systems 24 --steps 20

On a 1-card host (pc), this only measures the 1-card baseline (the multi-card rows need a
multi-card host — qb1 has 4 Blackhole cards). The harness prints honest numbers: it runs
each card-count once and reports wall-clock; no fabricated scaling.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from ase.build import molecule


def _systems(n, seed=0):
    out = []
    rng = np.random.default_rng(seed)
    mols = ["H2O", "CH3CH2OH", "C6H6", "CH3COOH"]
    for i in range(n):
        a = molecule(mols[i % len(mols)])
        a.info.update(charge=0, spin=0)
        a.rattle(stdev=0.05, seed=int(rng.integers(0, 10**6)))
        out.append(dict(pos=a.get_positions().tolist(), Z=a.get_atomic_numbers().tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="orb-v3-direct-omol")
    ap.add_argument("--devices", default="0", help="comma-separated card ids available")
    ap.add_argument("--n-systems", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--mode", default="relax", choices=["relax", "md"])
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    from tt_atom.batch import MultiCardSim

    all_devs = tuple(int(d) for d in args.devices.split(","))
    systems = _systems(args.n_systems)
    sim = dict(mode=args.mode, fmax=args.fmax, steps=args.steps, dt=1.0, temp=300.0, seed=42)
    print(f"scaling: {args.n_systems} systems, model={args.model}, mode={args.mode}, "
          f"steps={args.steps}, available cards={all_devs}")

    rows = []
    for n in [1, 2, 4]:
        if n > len(all_devs):
            continue
        devs = all_devs[:n]
        t0 = time.monotonic()
        with MultiCardSim(args.model, device_ids=devs, sim_params=sim) as pool:
            results = pool.run([dict(s) for s in systems])
        wall = time.monotonic() - t0
        ok = sum(1 for r in results if r.get("ok"))
        rows.append(dict(n=n, devices=devs, wall_s=round(wall, 2), ok=ok))
        print(f"  {n} card(s) {devs}: {wall:.1f}s  ({ok}/{len(systems)} ok)")

    if len(rows) >= 2:
        base = rows[0]["wall_s"]
        print("\nspeedup vs 1 card:")
        for r in rows:
            r["speedup"] = round(base / r["wall_s"], 2) if r["wall_s"] > 0 else None
            print(f"  {r['n']} card(s): {r['speedup']}x  ({r['wall_s']}s)")
    else:
        print("\nonly 1 card-count measured on this host; multi-card scaling needs more cards")
    print("\n" + json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
