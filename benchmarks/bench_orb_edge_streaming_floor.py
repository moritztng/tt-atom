"""Measure the optimistic floor for edge-tiled L1 streaming.

This intentionally omits the conservative-path cache export and aggregation.  It asks the
first gating question for the megakernel: can stock matmuls process L1-resident edge chunks
fast enough to beat the full-edge DRAM path before any scatter or cache-export overhead is
added?  A candidate that loses this lower-bound test cannot win as a Python-level chunked
pipeline.

    TT_VISIBLE_DEVICES=0 python3 benchmarks/bench_orb_edge_streaming_floor.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import torch

from bench_orb_edge_mlp import EDGE_COUNTS, _pcc


def _median_ms(ttnn, device, fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": float(statistics.median(samples)),
        "raw_ms": [round(value, 4) for value in samples],
    }


def _trace_median_ms(ttnn, device, fn, warmup, iterations):
    fn()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        for _ in range(warmup):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - start) * 1e3)
        return {
            "median_ms": float(statistics.median(samples)),
            "raw_ms": [round(value, 4) for value in samples],
        }, output
    finally:
        ttnn.release_trace(device, trace_id)


def _rmsnorm_l1(mlp, x):
    ttnn = mlp.ttnn
    l1 = ttnn.L1_MEMORY_CONFIG
    ms = ttnn.mean(ttnn.multiply(x, x, memory_config=l1), dim=-1, keepdim=True, memory_config=l1)
    inv = ttnn.rsqrt(ttnn.add(ms, mlp.norm.eps, memory_config=l1), memory_config=l1)
    return ttnn.multiply(
        ttnn.multiply(x, inv, memory_config=l1),
        mlp.norm.w,
        memory_config=l1,
    )


def _chunked_l1_floor(mlp, x, chunk_rows):
    """Optimistic candidate: only final chunks leave L1; no VJP caches or aggregation."""
    ttnn = mlp.ttnn
    l1 = ttnn.L1_MEMORY_CONFIG
    outputs = []
    rows = x.shape[0]
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        xc = ttnn.slice(x, [start, 0], [end, x.shape[1]], memory_config=l1)
        a0 = ttnn.linear(
            xc,
            mlp.w[0],
            bias=mlp.b[0],
            compute_kernel_config=mlp.kcfg,
            dtype=mlp.hidden_dtype,
            memory_config=l1,
        )
        h0 = ttnn.silu(a0, memory_config=l1)
        a1 = ttnn.linear(
            h0,
            mlp.w[1],
            bias=mlp.b[1],
            compute_kernel_config=mlp.kcfg,
            dtype=mlp.hidden_dtype,
            memory_config=l1,
        )
        h1 = ttnn.silu(a1, memory_config=l1)
        h2 = ttnn.linear(
            h1,
            mlp.w[2],
            bias=mlp.b[2],
            compute_kernel_config=mlp.kcfg,
            dtype=ttnn.bfloat16,
            memory_config=l1,
        )
        # A production kernel must write each finished edge chunk to its full-edge DRAM
        # output/cache slot before reusing the L1 scratch.  This explicit copy is slower
        # than the desired direct writer but avoids retaining every chunk in L1 and lets
        # larger, better-utilized chunks establish a conservative implementation bound.
        out_l1 = _rmsnorm_l1(mlp, h2)
        outputs.append(ttnn.to_memory_config(out_l1, ttnn.DRAM_MEMORY_CONFIG))
    return ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sizes", default=",".join(EDGE_COUNTS))
    parser.add_argument("--chunks", default="2048,4096,8192")
    parser.add_argument("--trace-chunks", default="2048")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="benchmarks/orb_edge_streaming_floor.json")
    args = parser.parse_args()

    import ttnn
    from tt_atom.device import open_device
    from tt_atom.orb_model import AttentionInteractionLayer, _to_dev
    from tt_atom.orb_weights import OrbWeights

    torch.manual_seed(args.seed)
    chunks = [int(value) for value in args.chunks.split(",") if value]
    trace_chunks = {int(value) for value in args.trace_chunks.split(",") if value}
    device = open_device(0)
    records = []
    try:
        weights = OrbWeights.load(args.weights)
        layer = AttentionInteractionLayer(
            weights.weights,
            "gnn_stacks.0",
            device,
            latent_dim=256,
            hidden_dim=1024,
        )
        mlp = layer.edge_mlp
        for tag in [name for name in args.sizes.split(",") if name]:
            rows = EDGE_COUNTS[tag]
            x_host = torch.randn(rows, 768, dtype=torch.bfloat16) * 0.25
            x = _to_dev(x_host, device, ttnn.bfloat16)
            baseline_out = mlp(x)
            baseline = _median_ms(ttnn, device, lambda: mlp(x), args.warmup, args.iterations)
            baseline_trace, _ = _trace_median_ms(
                ttnn, device, lambda: mlp(x), args.warmup, args.iterations)
            candidates = []
            for chunk_rows in chunks:
                try:
                    candidate_out = _chunked_l1_floor(mlp, x, chunk_rows)
                    ttnn.synchronize_device(device)
                    parity = _pcc(ttnn.to_torch(baseline_out), ttnn.to_torch(candidate_out))
                    timing = _median_ms(
                        ttnn,
                        device,
                        lambda c=chunk_rows: _chunked_l1_floor(mlp, x, c),
                        args.warmup,
                        args.iterations,
                    )
                    candidate = {
                        "chunk_rows": chunk_rows,
                        "timing": timing,
                        "optimistic_speedup": baseline["median_ms"] / timing["median_ms"],
                        "pcc": parity,
                    }
                    if chunk_rows in trace_chunks:
                        traced, _ = _trace_median_ms(
                            ttnn,
                            device,
                            lambda c=chunk_rows: _chunked_l1_floor(mlp, x, c),
                            args.warmup,
                            args.iterations,
                        )
                        candidate["traced_timing"] = traced
                        candidate["traced_optimistic_speedup"] = (
                            baseline_trace["median_ms"] / traced["median_ms"])
                    candidates.append(candidate)
                    traced_text = (
                        f" traced={candidate['traced_timing']['median_ms']:.3f} ms "
                        f"traced_speedup={candidate['traced_optimistic_speedup']:.3f}x"
                        if "traced_timing" in candidate else ""
                    )
                    print(
                        f"{tag} E={rows} chunk={chunk_rows}: "
                        f"baseline={baseline['median_ms']:.3f} ms "
                        f"floor={timing['median_ms']:.3f} ms "
                        f"speedup={baseline['median_ms'] / timing['median_ms']:.3f}x"
                        f"{traced_text} PCC={parity:.6f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    candidates.append(
                        {"chunk_rows": chunk_rows, "error": str(exc).splitlines()[0]})
                    print(f"{tag} E={rows} chunk={chunk_rows}: ERROR {exc}", flush=True)
            records.append({
                "tag": tag,
                "edges": rows,
                "baseline": baseline,
                "baseline_trace": baseline_trace,
                "candidates": candidates,
            })
    finally:
        ttnn.close_device(device)

    payload = {
        "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
        "scope": "optimistic stock-ttnn L1-chunked edge-MLP floor; excludes VJP cache export and aggregation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
