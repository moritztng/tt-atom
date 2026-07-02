"""Disjoint-union batching throughput on ONE card: K small systems batched vs one-at-a-time.

Batching's win is in the dispatch-bound regime — MANY SMALL systems, where per-call host
overhead (build geometry, upload, launch, read back) dominates device compute. Concatenating K
systems into one block-diagonal graph pays that overhead once instead of K times.

This is a strict apples-to-apples: both paths call the SAME code (``energy_and_forces_batch``,
energy-only). "one-at-a-time" runs it K times with a 1-system batch; "batched" runs it once with
a K-system batch. So the only variable is the disjoint union. We report systems/s for each, the
batched speedup, the crossover K (where batched first overtakes), and the batch-size ceiling
(largest K before device OOM).

    ~/.ttatom_run/env/bin/python benchmarks/bench_batch.py --weights ~/.ttatom_run/uma_s_ethanol.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ase.build import molecule

from tt_atom import device as D
from tt_atom.model import Backbone
from tt_atom.geometry import HostGeometry
from tt_atom.weights import WeightBundle
from tt_atom import forces, disjoint

RESULTS = pathlib.Path(__file__).parent / "results"


def conformers(k, mol, seed0=10):
    out = []
    for i in range(k):
        a = molecule(mol)
        a.rattle(stdev=0.08, seed=seed0 + i)
        a.info.update(charge=0, spin=1)
        out.append(a)
    return out


def time_it(fn, iters):
    fn()                                        # warm (program-cache fill for this shape)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mol", default="CH3CH2OH", help="ASE molecule name (small system)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    b = WeightBundle.load(args.weights)
    cfg, w = b.config, b.weights
    C = cfg["sphere_channels"]
    dev = D.open_device(args.device_id)
    bb = Backbone(w, dev, cfg, b.to_grid_mat, b.from_grid_mat)
    geo = HostGeometry(w, cfg, b.to_m, b.gauss_offset, b.gauss_coeff, gamma=0.0)

    natoms = len(molecule(args.mol))
    print(f"molecule {args.mol}: {natoms} atoms/system")

    rows = []
    ceiling = None
    for k in args.ks:
        systems = conformers(k, args.mol)
        try:
            # one-at-a-time: same code, K single-system batches
            def seq():
                for a in systems:
                    bg1 = disjoint.assemble([a], cfg["cutoff"], w, C, task=b.task)
                    forces.energy_and_forces_batch(bb, geo, bg1, compute_forces=False)

            # batched: one K-system disjoint-union forward
            bgK = disjoint.assemble(systems, cfg["cutoff"], w, C, task=b.task)

            def bat():
                forces.energy_and_forces_batch(bb, geo, bgK, compute_forces=False)

            seq_s = time_it(seq, max(2, args.iters // 2))
            bat_s = time_it(bat, args.iters)
        except RuntimeError as e:                # device OOM etc. -> record the ceiling and stop
            print(f"K={k}: FAILED ({str(e).splitlines()[0][:80]}) -> batch-size ceiling below {k}")
            ceiling = k
            break

        Etot = bgK.edge_index.shape[1]
        seq_thru = k / seq_s
        bat_thru = k / bat_s
        rows.append(dict(K=k, natoms_total=int(bgK.pos.shape[0]), nedges_total=Etot,
                         seq_ms=seq_s * 1e3, batched_ms=bat_s * 1e3,
                         seq_sys_per_s=seq_thru, batched_sys_per_s=bat_thru,
                         speedup=seq_s / bat_s))
        print(f"K={k:4d}  Ntot={bgK.pos.shape[0]:5d} Etot={Etot:6d}  "
              f"seq={seq_s*1e3:8.2f}ms ({seq_thru:7.1f} sys/s)  "
              f"batched={bat_s*1e3:8.2f}ms ({bat_thru:7.1f} sys/s)  x{seq_s/bat_s:.2f}")

    import ttnn
    ttnn.close_device(dev)

    speedups = [r for r in rows if r["speedup"] > 1.0]
    crossover = speedups[0]["K"] if speedups else None
    best = max(rows, key=lambda r: r["speedup"]) if rows else None
    summary = dict(molecule=args.mol, natoms_per_system=natoms, crossover_K=crossover,
                   batch_ceiling=ceiling,
                   best_speedup=best["speedup"] if best else None,
                   best_K=best["K"] if best else None)
    print("\nSUMMARY:", json.dumps(summary))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "batch_throughput.json"
    out.write_text(json.dumps(dict(config=cfg, summary=summary, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
