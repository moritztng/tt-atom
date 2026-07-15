"""Test whether tt-metal's fuseable minimal-matmul is viable for Orb's edge MLP.

The intended megakernel needs a matmul implementation whose writer can be extended
with dual cache/L1 outputs.  ``minimal_matmul`` is the available extension point.
This benchmark compares its fused-SiLU path with the production edge MLP and with
the best traced 2,560-edge stock-op L1 floor.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import torch

from bench_orb_edge_mlp import EDGE_COUNTS, _pcc
from bench_orb_edge_streaming_floor import _median_ms, _rmsnorm_l1, _trace_median_ms


def _minimal_mlp(mlp, x, memory_config):
    ttnn = mlp.ttnn
    silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    h0 = ttnn.experimental.minimal_matmul(
        x, mlp.w[0], bias_tensor=mlp.b[0], fused_activation=silu,
        memory_config=memory_config, dtype=mlp.hidden_dtype,
        compute_kernel_config=mlp.kcfg)
    h1 = ttnn.experimental.minimal_matmul(
        h0, mlp.w[1], bias_tensor=mlp.b[1], fused_activation=silu,
        memory_config=memory_config, dtype=mlp.hidden_dtype,
        compute_kernel_config=mlp.kcfg)
    h2 = ttnn.experimental.minimal_matmul(
        h1, mlp.w[2], bias_tensor=mlp.b[2],
        memory_config=memory_config, dtype=ttnn.bfloat16,
        compute_kernel_config=mlp.kcfg)
    return _rmsnorm_l1(mlp, h2) if memory_config == ttnn.L1_MEMORY_CONFIG else mlp.norm(h2)


def _minimal_contract_mlp(mlp, x):
    """Preserve the production pre-SiLU cache contract; only swap matmul factory."""
    ttnn = mlp.ttnn
    a0 = ttnn.experimental.minimal_matmul(
        x, mlp.w[0], bias_tensor=mlp.b[0],
        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=mlp.hidden_dtype,
        compute_kernel_config=mlp.kcfg)
    h0 = ttnn.silu(a0)
    a1 = ttnn.experimental.minimal_matmul(
        h0, mlp.w[1], bias_tensor=mlp.b[1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=mlp.hidden_dtype,
        compute_kernel_config=mlp.kcfg)
    h1 = ttnn.silu(a1)
    h2 = ttnn.experimental.minimal_matmul(
        h1, mlp.w[2], bias_tensor=mlp.b[2],
        memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        compute_kernel_config=mlp.kcfg)
    return mlp.norm(h2)


def _minimal_chunked(mlp, x, chunk_rows):
    ttnn = mlp.ttnn
    outputs = []
    for start in range(0, x.shape[0], chunk_rows):
        end = min(start + chunk_rows, x.shape[0])
        xc = ttnn.slice(
            x, [start, 0], [end, x.shape[1]], memory_config=ttnn.L1_MEMORY_CONFIG)
        out = _minimal_mlp(mlp, xc, ttnn.L1_MEMORY_CONFIG)
        outputs.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
    return ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sizes", default=",".join(EDGE_COUNTS))
    parser.add_argument("--chunk-rows", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--out", default="benchmarks/orb_minimal_matmul.json")
    args = parser.parse_args()

    import ttnn
    from tt_atom.device import open_device
    from tt_atom.orb_model import AttentionInteractionLayer, _to_dev
    from tt_atom.orb_weights import OrbWeights

    torch.manual_seed(17)
    device = open_device(0)
    records = []
    try:
        weights = OrbWeights.load(args.weights)
        layer = AttentionInteractionLayer(
            weights.weights, "gnn_stacks.0", device, latent_dim=256, hidden_dim=1024)
        mlp = layer.edge_mlp
        for tag in [name for name in args.sizes.split(",") if name]:
            rows = EDGE_COUNTS[tag]
            x = _to_dev(
                torch.randn(rows, 768, dtype=torch.bfloat16) * 0.25,
                device,
                ttnn.bfloat16,
            )
            baseline_out = mlp(x)
            baseline_trace, _ = _trace_median_ms(
                ttnn, device, lambda: mlp(x), args.warmup, args.iterations)

            full_out = _minimal_mlp(mlp, x, ttnn.DRAM_MEMORY_CONFIG)
            full = _median_ms(
                ttnn, device,
                lambda: _minimal_mlp(mlp, x, ttnn.DRAM_MEMORY_CONFIG),
                args.warmup, args.iterations)
            contract_out = _minimal_contract_mlp(mlp, x)
            contract = _median_ms(
                ttnn, device, lambda: _minimal_contract_mlp(mlp, x),
                args.warmup, args.iterations)
            chunked_trace, chunked_out = _trace_median_ms(
                ttnn, device,
                lambda: _minimal_chunked(mlp, x, args.chunk_rows),
                args.warmup, args.iterations)
            base_t = ttnn.to_torch(baseline_out)
            record = {
                "tag": tag,
                "edges": rows,
                "baseline_trace": baseline_trace,
                "minimal_full": full,
                "minimal_full_speedup": baseline_trace["median_ms"] / full["median_ms"],
                "minimal_full_pcc": _pcc(base_t, ttnn.to_torch(full_out)),
                "minimal_contract": contract,
                "minimal_contract_speedup": (
                    baseline_trace["median_ms"] / contract["median_ms"]),
                "minimal_contract_pcc": _pcc(base_t, ttnn.to_torch(contract_out)),
                "chunk_rows": args.chunk_rows,
                "minimal_chunked_trace": chunked_trace,
                "minimal_chunked_speedup": (
                    baseline_trace["median_ms"] / chunked_trace["median_ms"]),
                "minimal_chunked_pcc": _pcc(base_t, ttnn.to_torch(chunked_out)),
            }
            records.append(record)
            print(
                f"{tag} E={rows}: baseline={baseline_trace['median_ms']:.3f} ms "
                f"minimal_full={full['median_ms']:.3f} ms "
                f"({record['minimal_full_speedup']:.3f}x, PCC={record['minimal_full_pcc']:.6f}) "
                f"minimal_contract={contract['median_ms']:.3f} ms "
                f"({record['minimal_contract_speedup']:.3f}x, "
                f"PCC={record['minimal_contract_pcc']:.6f}) "
                f"minimal_chunked={chunked_trace['median_ms']:.3f} ms "
                f"({record['minimal_chunked_speedup']:.3f}x, "
                f"PCC={record['minimal_chunked_pcc']:.6f})",
                flush=True,
            )
    finally:
        ttnn.close_device(device)

    with open(args.out, "w") as handle:
        json.dump({
            "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
