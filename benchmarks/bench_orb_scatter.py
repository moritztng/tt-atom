"""Compare Orb's gather/reduce segment sum with ttnn.scatter_add.

The synthetic graph has Orb's measured periodic-Si degree (46 edges/atom).  Both
implementations return tiled ``[N, 256]`` tensors so timings include any layout
conversion required by their caller.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _time(device, fn, warmup=5, steps=30):
    import ttnn

    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(steps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="216,512,1000,2016")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", default="benchmarks/orb_scatter_ablation.json")
    args = parser.parse_args()

    import ttnn

    from tt_atom.device import open_device
    from tt_atom.scatter import build_gather, segment_sum

    device = open_device(0)
    records = []
    try:
        for n in map(int, args.sizes.split(",")):
            e = 46 * n
            width = args.width
            # Balanced but shuffled degree-46 graph, matching periodic Si's degree.
            idx = torch.arange(e, dtype=torch.int64) % n
            idx = idx[torch.randperm(e, generator=torch.Generator().manual_seed(1))]
            msg_host = torch.randn(e, width, generator=torch.Generator().manual_seed(2))
            msg = ttnn.from_torch(
                msg_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

            gather, dmax = build_gather(idx, n, e)
            gather_dev = ttnn.from_torch(
                torch.from_numpy(gather),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            zeros = ttnn.from_torch(
                torch.zeros(n, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            scatter_index = ttnn.from_torch(
                idx.to(torch.int32).view(e, 1).expand(e, width).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            gather_fn = lambda: segment_sum(ttnn, msg, gather_dev, dmax, n, width)
            scatter_fn = lambda: ttnn.scatter_add(zeros, 0, scatter_index, msg)
            ref = ttnn.to_torch(gather_fn()).float()
            got = ttnn.to_torch(scatter_fn()).float()
            pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
            max_abs = (ref - got).abs().max().item()
            gather_ms = _time(device, gather_fn, steps=args.steps)
            scatter_ms = _time(device, scatter_fn, steps=args.steps)
            records.append({
                "N": n, "edges": e, "width": width,
                "gather_reduce_ms": gather_ms, "scatter_add_ms": scatter_ms,
                "gather_over_scatter_speedup": gather_ms / scatter_ms,
                "pcc": pcc, "max_abs": max_abs,
            })
            print(
                f"N={n} E={e} gather_reduce={gather_ms:.3f} ms "
                f"scatter_add={scatter_ms:.3f} ms speedup={gather_ms/scatter_ms:.2f}x "
                f"PCC={pcc:.8f} maxabs={max_abs:.6g}",
                flush=True,
            )
    finally:
        ttnn.close_device(device)
    with open(args.out, "w") as f:
        json.dump({
            "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }, f, indent=2)


if __name__ == "__main__":
    main()
