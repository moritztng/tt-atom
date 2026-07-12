"""Disjoint-union batching throughput for Orb on ONE card: ``evaluate_batch`` vs looping
``calculate`` -- the Orb-family counterpart to ``benchmarks/bench_batch.py``.

Same discipline as the UMA benchmark: many small systems, real timing, real card. "looping" runs
the single-system ``calculate`` K times (the path a user would use without batching); "batched"
runs ``evaluate_batch`` once on the K-system disjoint union. Both compute energy + forces, so the
only variable is the disjoint union. We report systems/s for each, the batched speedup, and the
batch-size ceiling (largest K before device OOM).

    TT_VISIBLE_DEVICES=0 ~/.ttatom_run/env/bin/python \
        benchmarks/bench_orb_evaluate_batch.py --checkpoint orb-v3-conservative-omol --mol CH3CH2OH
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from ase.build import molecule

from tt_atom.orb_calculator import OrbCalculator

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
    ap.add_argument("--checkpoint", default="orb-v3-conservative-omol",
                    help="Orb checkpoint name (small-molecule: conservative-omol / direct-omol)")
    ap.add_argument("--mol", default="CH3CH2OH", help="ASE molecule name (small system)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    calc = OrbCalculator.from_checkpoint(args.checkpoint, device_id=args.device_id)
    natoms = len(molecule(args.mol))
    print(f"checkpoint {args.checkpoint}  molecule {args.mol}: {natoms} atoms/system")

    rows = []
    ceiling = None
    for k in args.ks:
        systems = conformers(k, args.mol)
        try:
            def seq():
                for a in systems:
                    a.calc = calc
                    a.get_potential_energy()
                    a.get_forces()
                    a.calc.results.clear()      # force a real recompute each iter (no ASE cache)

            def bat():
                calc.evaluate_batch(systems)

            seq_s = time_it(seq, max(2, args.iters // 2))
            bat_s = time_it(bat, args.iters)
        except RuntimeError as e:                # device OOM etc. -> record the ceiling and stop
            print(f"K={k}: FAILED ({str(e).splitlines()[0][:80]}) -> batch-size ceiling below {k}")
            ceiling = k
            break

        seq_thru = k / seq_s
        bat_thru = k / bat_s
        rows.append(dict(K=k, natoms_total=int(natoms * k),
                         seq_ms=seq_s * 1e3, batched_ms=bat_s * 1e3,
                         seq_sys_per_s=seq_thru, batched_sys_per_s=bat_thru,
                         speedup=seq_s / bat_s))
        print(f"K={k:4d}  Ntot={natoms*k:5d}  "
              f"seq={seq_s*1e3:8.2f}ms ({seq_thru:7.1f} sys/s)  "
              f"batched={bat_s*1e3:8.2f}ms ({bat_thru:7.1f} sys/s)  x{seq_s/bat_s:.2f}")

    calc.close()

    speedups = [r for r in rows if r["speedup"] > 1.0]
    crossover = speedups[0]["K"] if speedups else None
    best = max(rows, key=lambda r: r["speedup"]) if rows else None
    summary = dict(checkpoint=args.checkpoint, molecule=args.mol, natoms_per_system=natoms,
                   crossover_K=crossover, batch_ceiling=ceiling,
                   best_speedup=best["speedup"] if best else None,
                   best_K=best["K"] if best else None)
    print("\nSUMMARY:", json.dumps(summary))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "orb_batch_throughput.json"
    out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
