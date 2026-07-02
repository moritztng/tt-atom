"""Disjoint-union (block-diagonal) batched inference on ONE card.

Evaluate K independent systems in a SINGLE device forward — the fairchem/PyG way: concatenate
them into one big block-diagonal graph, run once, recover per-system energies by segment-sum
(and per-system forces from the one shared analytic backward). This is the throughput lever for
many small systems, where one-at-a-time is host-dispatch-bound; see benchmarks/bench_batch.py.

The single-system TTAtomCalculator API is unchanged — batching is just the extra
``evaluate_batch`` method. A merged uma-s-1 bundle bakes the MoLE routing for one reduced
composition, so the batch shares that composition (e.g. an MD ensemble / conformer set of one
molecule); pass a real merged bundle for physical energies.

    ~/.ttatom_run/env/bin/python examples/evaluate_batch.py --weights uma_s_ethanol.npz --n 32
"""
from __future__ import annotations

import argparse
import pathlib
import time

from ase.build import molecule

from tt_atom import TTAtomCalculator

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "model_tiny_demo.npz"))
    ap.add_argument("--mol", default="CH3CH2OH")
    ap.add_argument("--n", type=int, default=32, help="number of systems in the batch")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    systems = []
    for i in range(args.n):
        a = molecule(args.mol)
        a.rattle(stdev=0.05, seed=i)                 # a conformer set (one composition)
        a.info.update(charge=0, spin=1)
        systems.append(a)

    calc = TTAtomCalculator(args.weights, device_id=args.device_id)
    try:
        calc.evaluate_batch(systems)                 # warm the program cache for this shape
        t = time.perf_counter()
        out = calc.evaluate_batch(systems)
        dt = time.perf_counter() - t
    finally:
        calc.close()

    E = out["energy"]
    print(f"evaluated {args.n} systems in ONE device forward in {dt*1e3:.1f} ms "
          f"({args.n / dt:.1f} systems/s)")
    print(f"per-system energies (eV): {[round(float(e), 4) for e in E[:4]]} ...")
    print(f"forces[0] shape: {out['forces'][0].shape}")


if __name__ == "__main__":
    main()
