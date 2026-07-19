"""Profile Orb-v3's edge MLP and its available Linear+SiLU fusion.

This is a device microbenchmark, not an end-to-end result.  It uses the real first
message-passing layer weights and the production edge counts.  The fused candidate is
``ttnn.linear(..., activation="silu")``, which applies SiLU in the matmul epilogue and
therefore avoids materializing the pre-activation for a separate SiLU dispatch.

The conservative model cannot use that candidate as-is: its analytic VJP needs both
pre-SiLU tensors.  The benchmark quantifies the maximum forward-only benefit and reports
that storage contract explicitly so it cannot be mistaken for a shippable MD speedup.

    TT_VISIBLE_DEVICES=0 python3 benchmarks/bench_orb_edge_mlp.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --out benchmarks/orb_edge_mlp_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

import torch


EDGE_COUNTS = {
    "3x3x3": 9936,
    "4x4x4": 23552,
    "5x5x5": 46000,
    "6x6x7": 92736,
}


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
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "raw_ms": [round(value, 4) for value in samples],
    }


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _fused_forward(mlp, x):
    """Forward-only MLP using the matmul epilogue activation."""
    ttnn = mlp.ttnn
    h0 = ttnn.linear(
        x,
        mlp.w[0],
        bias=mlp.b[0],
        activation="silu",
        compute_kernel_config=mlp.kcfg,
        dtype=mlp.hidden_dtype,
    )
    h1 = ttnn.linear(
        h0,
        mlp.w[1],
        bias=mlp.b[1],
        activation="silu",
        compute_kernel_config=mlp.kcfg,
        dtype=mlp.hidden_dtype,
    )
    h2 = ttnn.linear(
        h1,
        mlp.w[2],
        bias=mlp.b[2],
        compute_kernel_config=mlp.kcfg,
        dtype=ttnn.bfloat16,
    )
    return mlp.norm(h2)


def _profile_stages(ttnn, device, mlp, x, iterations):
    samples = {name: [] for name in ("linear0", "silu0", "linear1", "silu1", "linear2", "rmsnorm")}
    for _ in range(iterations):
        values = {}
        for name, fn in (
            ("linear0", lambda: ttnn.linear(
                x, mlp.w[0], bias=mlp.b[0], compute_kernel_config=mlp.kcfg,
                dtype=mlp.hidden_dtype)),
            ("silu0", lambda: ttnn.silu(values["linear0"])),
            ("linear1", lambda: ttnn.linear(
                values["silu0"], mlp.w[1], bias=mlp.b[1], compute_kernel_config=mlp.kcfg,
                dtype=mlp.hidden_dtype)),
            ("silu1", lambda: ttnn.silu(values["linear1"])),
            ("linear2", lambda: ttnn.linear(
                values["silu1"], mlp.w[2], bias=mlp.b[2], compute_kernel_config=mlp.kcfg,
                dtype=ttnn.bfloat16)),
            ("rmsnorm", lambda: mlp.norm(values["linear2"])),
        ):
            start = time.perf_counter()
            values[name] = fn()
            ttnn.synchronize_device(device)
            samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: float(statistics.median(values)) for name, values in samples.items()}


def _profile_backward_stages(ttnn, device, mlp, g_out, iterations):
    from tt_atom.orb_forces import _mm, rmsnorm_bw, silu_bw

    samples = {
        name: [] for name in
        ("rmsnorm_bw", "linear2_bw", "silu1_bw", "linear1_bw", "silu0_bw", "linear0_bw")
    }
    for _ in range(iterations):
        values = {}
        for name, fn in (
            ("rmsnorm_bw", lambda: rmsnorm_bw(mlp.norm, g_out)),
            ("linear2_bw", lambda: _mm(ttnn, values["rmsnorm_bw"], mlp.w[2], mlp.kcfg)),
            ("silu1_bw", lambda: silu_bw(ttnn, values["linear2_bw"], mlp._cache_a1)),
            ("linear1_bw", lambda: _mm(ttnn, values["silu1_bw"], mlp.w[1], mlp.kcfg)),
            ("silu0_bw", lambda: silu_bw(ttnn, values["linear1_bw"], mlp._cache_a0)),
            ("linear0_bw", lambda: _mm(ttnn, values["silu0_bw"], mlp.w[0], mlp.kcfg)),
        ):
            start = time.perf_counter()
            values[name] = fn()
            ttnn.synchronize_device(device)
            samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: float(statistics.median(values)) for name, values in samples.items()}


def _logical_work(rows, *, dtype_bytes):
    in_dim, hidden, out_dim = 768, 1024, 256
    flops = 2 * rows * (in_dim * hidden + hidden * hidden + hidden * out_dim)

    # Tensor-value transfers for the exact current forward sequence.  Weights are
    # counted once per op, while activation traffic follows every materialized op.
    linear_values = (
        rows * in_dim + in_dim * hidden + rows * hidden
        + rows * hidden + hidden * hidden + rows * hidden
        + rows * hidden + hidden * out_dim + rows * out_dim
    )
    silu_values = 4 * rows * hidden  # two activations, each read + write
    # RMSNorm implementation: square, mean, add, rsqrt, x*inv, affine multiply.
    rms_values = rows * (2 * out_dim + out_dim + out_dim + 1 + 1 + 1 + 1 + out_dim + 1 + out_dim + out_dim + out_dim)
    weights_bytes = dtype_bytes * (in_dim * hidden + hidden * hidden + hidden * out_dim)
    current_bytes = dtype_bytes * (linear_values + silu_values + rms_values)
    # Epilogue SiLU removes the pre-activation write+read for both hidden layers.
    epilogue_saved_bytes = dtype_bytes * (4 * rows * hidden)
    return {
        "matmul_flops": int(flops),
        "current_logical_bytes": int(current_bytes),
        "weight_bytes_included": int(weights_bytes),
        "epilogue_silu_saved_bytes": int(epilogue_saved_bytes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sizes", default=",".join(EDGE_COUNTS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--stage-iterations", type=int, default=3)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="benchmarks/orb_edge_mlp_profile.json")
    args = parser.parse_args()

    import ttnn
    from tt_atom.device import open_device
    from tt_atom.orb_model import AttentionInteractionLayer, _to_dev
    from tt_atom.orb_weights import OrbWeights

    torch.manual_seed(args.seed)
    device = open_device(0)
    records = []
    try:
        gw = OrbWeights.load(args.weights)
        layer = AttentionInteractionLayer(
            gw.weights, "gnn_stacks.0", device, latent_dim=256, hidden_dim=1024, fast=args.fast)
        mlp = layer.edge_mlp
        wanted = [name for name in args.sizes.split(",") if name]
        for name in wanted:
            rows = EDGE_COUNTS[name]
            # A bounded random input is representative of RMS-normalized residual features
            # and avoids relying on a particular topology while retaining real model weights.
            x_host = torch.randn(rows, 768, dtype=torch.bfloat16) * 0.25
            x = _to_dev(x_host, device, ttnn.bfloat16)

            base_out = mlp(x)
            fused_out = _fused_forward(mlp, x)
            ttnn.synchronize_device(device)
            base_t = ttnn.to_torch(base_out)
            fused_t = ttnn.to_torch(fused_out)
            parity = {
                "pcc": _pcc(base_t, fused_t),
                "max_abs": float((base_t.float() - fused_t.float()).abs().max()),
                "mean_abs": float((base_t.float() - fused_t.float()).abs().mean()),
            }

            baseline = _median_ms(
                ttnn, device, lambda: mlp(x), args.warmup, args.iterations)
            fused = _median_ms(
                ttnn, device, lambda: _fused_forward(mlp, x), args.warmup, args.iterations)
            # Restore the exact baseline forward caches consumed by the analytic VJP.
            mlp(x)
            g_out = _to_dev(
                torch.randn(rows, 256, dtype=torch.bfloat16) * 0.125,
                device,
                ttnn.bfloat16,
            )
            from tt_atom.orb_forces import mlpnorm_bw
            backward = _median_ms(
                ttnn, device, lambda: mlpnorm_bw(mlp, g_out), args.warmup, args.iterations)
            stages = _profile_stages(ttnn, device, mlp, x, args.stage_iterations)
            backward_stages = _profile_backward_stages(
                ttnn, device, mlp, g_out, args.stage_iterations)
            work = _logical_work(rows, dtype_bytes=1 if args.fast else 2)
            record = {
                "tag": name,
                "edges": rows,
                "baseline": baseline,
                "linear_silu_epilogue": fused,
                "forward_only_speedup": baseline["median_ms"] / fused["median_ms"],
                "parity": parity,
                "backward": backward,
                "synchronized_baseline_stage_ms": stages,
                "synchronized_backward_stage_ms": backward_stages,
                "logical_work": work,
                "effective_matmul_tflops": work["matmul_flops"] / baseline["median_ms"] / 1e9,
                "effective_logical_bandwidth_gbs": work["current_logical_bytes"] / baseline["median_ms"] / 1e6,
            }
            records.append(record)
            print(
                f"{name}: E={rows} baseline={baseline['median_ms']:.3f} ms "
                f"epilogue={fused['median_ms']:.3f} ms "
                f"speedup={record['forward_only_speedup']:.3f}x PCC={parity['pcc']:.6f}",
                flush=True,
            )
    finally:
        ttnn.close_device(device)

    payload = {
        "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
        "model": "orb-v3-conservative-inf-omat",
        "scope": "first interaction layer edge MLPNorm with real weights",
        "precision": "bf8 hidden/weights" if args.fast else "bf16",
        "candidate": "ttnn.linear SiLU epilogue; forward-only because conservative VJP requires pre-SiLU tensors",
        "orb_fused_silu_bw": os.environ.get("TT_ATOM_ORB_FUSED_SILU_BW", "auto"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
