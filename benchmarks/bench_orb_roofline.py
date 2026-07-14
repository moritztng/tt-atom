"""Stage-level timing for Orb-v3's traced conservative MD step.

This measures the exact path used by ``bench_orb_perf_dollar_tt.py`` while separating
host geometry, host-to-device refresh, traced device forward/backward, device-to-host
readback, and the final host geometry VJP.  Explicit synchronization between stages
makes the attribution additive; ``end_to_end_ms`` separately measures the normal
pipelined path without those diagnostic barriers.

    PATH=~/.ttatom_run/env/bin:$PATH TT_VISIBLE_DEVICES=0 python3 \
        benchmarks/bench_orb_roofline.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --sizes 3x3x3,6x6x7 --steps 20 --out benchmarks/orb_roofline.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bench_orb_perf_dollar_tt import SIZES


def _summary(samples):
    return {
        "median": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "raw": [round(float(x), 4) for x in samples],
    }


def _one_size(args, size):
    import ttnn

    from examples.orb_md import OrbDeviceCalculator

    if args.ablate_scatter:
        from tt_atom import scatter

        def _zero_segment_sum(ttnn_module, msg, gather_dev, dmax, n, width):
            del gather_dev, dmax
            return ttnn_module.multiply(
                ttnn_module.slice(msg, [0, 0], [n, width]), 0.0
            )

        scatter.segment_sum = _zero_segment_sum

    tag, nx, ny, nz = size
    atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
    calc = OrbDeviceCalculator(args.weights, device_id=0, fast=args.fast)
    atoms.calc = calc
    rng = np.random.default_rng(args.seed)

    # Build the topology, compile, capture, and prime the program/trace caches.
    atoms.set_positions(atoms.get_positions())
    atoms.get_potential_energy()
    atoms.get_forces()
    for _ in range(args.warmup - 1):
        atoms.set_positions(atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape))
        atoms.get_potential_energy()
        atoms.get_forces()

    eng = calc.engine
    stages = {
        "host_geometry_ms": [],
        "h2d_refresh_ms": [],
        "trace_replay_ms": [],
        "d2h_readback_ms": [],
        "host_force_vjp_ms": [],
        "diagnostic_sum_ms": [],
        "end_to_end_ms": [],
    }

    try:
        # Additive attribution with a synchronization boundary after refresh.
        for _ in range(args.steps):
            pos = torch.tensor(
                atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape),
                dtype=torch.float64,
            ).requires_grad_(True)

            t0 = time.perf_counter()
            ctx = eng._prepare(pos)
            t1 = time.perf_counter()

            eng._refresh(ctx)
            ttnn.synchronize_device(calc.device)
            t2 = time.perf_counter()

            ttnn.execute_trace(calc.device, eng.tid, cq_id=0, blocking=True)
            t3 = time.perf_counter()

            raw_pred = ttnn.to_torch(eng.raw_pred_t).double().view(())
            g_edge_feat = ttnn.to_torch(eng.g_edge_feat_t).float()
            g_cutoff = ttnn.to_torch(eng.g_cutoff_t).float()
            t4 = time.perf_counter()

            from tt_atom.orb_geometry import host_edge_features_vjp
            host_edge_features_vjp(
                ctx[2], eng.senders, eng.receivers, eng.Z.shape[0], g_edge_feat, g_cutoff,
                r_max=eng.r_max, num_bases=eng.num_bases)
            t5 = time.perf_counter()
            # Keep the scalar read live so an optimizer cannot elide it.
            assert torch.isfinite(raw_pred)

            values = [
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                (t3 - t2) * 1e3,
                (t4 - t3) * 1e3,
                (t5 - t4) * 1e3,
            ]
            for key, value in zip(list(stages)[:5], values):
                stages[key].append(value)
            stages["diagnostic_sum_ms"].append(sum(values))

        # Normal path, no diagnostic synchronization between refresh and replay.
        for _ in range(args.steps):
            pos = torch.tensor(
                atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape),
                dtype=torch.float64,
            )
            t0 = time.perf_counter()
            eng(pos)
            stages["end_to_end_ms"].append((time.perf_counter() - t0) * 1e3)
    finally:
        calc.close()

    summaries = {name: _summary(values) for name, values in stages.items()}
    total = summaries["diagnostic_sum_ms"]["median"]
    for name in list(stages)[:5]:
        summaries[name]["share_of_diagnostic_sum_pct"] = 100.0 * summaries[name]["median"] / total

    return {
        "tag": tag,
        "N": len(atoms),
        "edges": int(calc.n_edges),
        "warmup": args.warmup,
        "timed_steps": args.steps,
        "stages": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--element", default="Si")
    parser.add_argument("--a", type=float, default=5.43)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true", help="bf8 weights and hidden MLP activations")
    parser.add_argument("--ablate-scatter", action="store_true",
                        help="profiling only: replace segment sums with shape-compatible zeros")
    parser.add_argument("--sizes", default="3x3x3,6x6x7")
    parser.add_argument("--out", default="benchmarks/orb_roofline.json")
    args = parser.parse_args()

    wanted = set(args.sizes.split(","))
    plan = [size for size in SIZES if size[0] in wanted]
    records = []
    for size in plan:
        print(f"profiling {size[0]}...", flush=True)
        record = _one_size(args, size)
        records.append(record)
        med = {key: value["median"] for key, value in record["stages"].items()}
        print(
            f"N={record['N']} E={record['edges']} "
            f"geometry={med['host_geometry_ms']:.2f} h2d={med['h2d_refresh_ms']:.2f} "
            f"replay={med['trace_replay_ms']:.2f} d2h={med['d2h_readback_ms']:.2f} "
            f"host-vjp={med['host_force_vjp_ms']:.2f} "
            f"e2e={med['end_to_end_ms']:.2f} ms",
            flush=True,
        )

    payload = {
        "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
        "model": "orb-v3-conservative-inf-omat",
        "method": "explicitly synchronized additive stage attribution plus normal-path end-to-end",
        "fast": args.fast,
        "scatter_ablation": args.ablate_scatter,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
