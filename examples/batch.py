"""Multi-card batch throughput on Tenstorrent via TT-Atom's fan-out.

The evaluation of one system is independent of every other, so throughput scales by running one
worker process per Tenstorrent card (each pinned to its own device with the model resident) while
the parent streams systems to a shared queue. Near-linear scaling was validated on a 4-card
QuietBox (qb1): 3.95x on 4 cards. On a single-card host this still runs — it just uses one card.

    ~/.ttatom_run/env/bin/python examples/batch.py --device-ids 0 --n 32
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
from ase.build import molecule

from tt_atom.batch import MultiCard

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "model_tiny_demo.npz"))
    ap.add_argument("--device-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=32, help="number of systems to evaluate")
    args = ap.parse_args()

    base = molecule("CH3CH2OH")
    rng = np.random.default_rng(0)
    systems = []
    for _ in range(args.n):
        pos = base.get_positions() + rng.normal(scale=0.05, size=base.get_positions().shape)
        systems.append((pos, base.get_atomic_numbers()))

    with MultiCard(args.weights, device_ids=tuple(args.device_ids)) as pool:
        t = time.perf_counter()
        energies, total_edges = pool.energies(systems)
        dt = time.perf_counter() - t

    print(f"evaluated {args.n} systems on {len(args.device_ids)} card(s) in {dt:.2f}s "
          f"({args.n / dt:.1f} systems/s, {total_edges} edges total)")
    print(f"first energies: {[round(float(e), 4) for e in energies[:4]]}")


if __name__ == "__main__":
    main()
