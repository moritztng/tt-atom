"""Multi-card throughput scaling for TT-Atom.

Evaluates a fixed pool of independent systems on 1..N cards (one worker process per card,
weights resident) and reports aggregate Medges/s and the scaling factor at each card count.
All numbers are measured here; nothing is hardcoded. Results -> benchmarks/results/multicard.json.

Run:  ~/.ttatom_run/env/bin/python benchmarks/bench_multicard.py --weights /tmp/tt_full.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
from ase.build import bulk

from tt_atom.batch import MultiCard

RESULTS = pathlib.Path(__file__).parent / "results"


def make_systems(n_systems, cells):
    # Same-size systems (constant edge count) so the device program cache stays warm: this
    # measures steady-state throughput capacity. Production screening of differently-sized
    # systems would bucket/pad edges to a few fixed sizes for the same effect.
    systems = []
    for i in range(n_systems):
        a = bulk("Si", "diamond", a=5.43) * (cells, cells, cells)
        a.rattle(stdev=0.1, seed=1)                        # identical geometry -> constant E
        systems.append((a.get_positions().astype(np.float32), a.get_atomic_numbers()))
    return systems


def measure(weights, device_ids, systems, fast):
    with MultiCard(weights, device_ids=device_ids, fast=fast) as mc:
        mc.energies(systems[: len(device_ids)])            # warmup (compile per worker)
        t0 = time.perf_counter()
        _, total_edges = mc.energies(systems)
        dt = time.perf_counter() - t0
    return total_edges / dt / 1e6, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cells", type=int, default=4)            # ~128 atoms / ~2200 edges each
    ap.add_argument("--systems", type=int, default=64)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-cards", type=int, default=4)
    args = ap.parse_args()

    systems = make_systems(args.systems, args.cells)
    natoms = len(systems[0][1])
    rows = []
    for n in range(1, args.max_cards + 1):
        ids = list(range(n))
        medges, dt = measure(args.weights, ids, systems, args.fast)
        rows.append(dict(cards=len(ids), device_ids=ids, medges_per_s=medges, wall_s=dt))
        print(f"{len(ids)} card(s): {medges:6.3f} Medges/s  ({args.systems} systems x ~{natoms} atoms in {dt:.2f}s)")

    base = rows[0]["medges_per_s"]
    for r in rows:
        r["scaling_vs_1card"] = r["medges_per_s"] / base
    print(f"{args.max_cards}-card scaling: {rows[-1]['scaling_vs_1card']:.2f}x")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("multicard_fast.json" if args.fast else "multicard.json")
    out.write_text(json.dumps(dict(systems=args.systems, cells=args.cells, natoms=natoms,
                                   fast=args.fast, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
