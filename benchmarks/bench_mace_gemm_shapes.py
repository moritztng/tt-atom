"""Rate of stock ttnn.matmul on the GEMM shapes a MACE step issues (bench_mace_tt.py --census), bfloat16 operands
and a float32 result as in tt_blocks.mm. One chip; prints GFLOP/s per shape and the step time those rates imply."""
import argparse
import json
import time

import numpy as np
import torch

from tt_atom import device as D

# (left, right, calls per step at 1000 Si atoms, 46000 edges), from bench_mace_tt.py --eager --census
SHAPES = [((128000, 816), (816, 136), 4), ((128000, 136), (136, 816), 4), ((1000, 2048), (2048, 2048), 12),
          ((1000, 5120), (5120, 2048), 3), ((1000, 2048), (2048, 5120), 3), ((46000, 2816), (2816, 64), 3),
          ((46000, 64), (64, 2816), 3), ((128000, 816), (816, 16), 4), ((128000, 16), (16, 816), 4),
          ((1000, 46, 2816), (1000, 2816, 16), 3), ((1000, 46, 16), (1000, 16, 2816), 3),
          ((1000, 16, 46), (1000, 46, 2816), 3), ((1000, 352, 40), (1000, 40, 128), 3),
          ((1000, 40, 352), (1000, 352, 128), 3), ((46000, 64), (64, 64), 24)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="benchmarks/mace_gemm_shapes_tt.json")
    args = ap.parse_args()
    import ttnn

    rows = []
    with D.device(0) as dev:
        cfg = D.compute_kernel_config()
        for a, b, calls in SHAPES:
            ta, tb = (ttnn.from_torch(torch.randn(s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for s in (a, b))
            ts = []
            for _ in range(args.reps + 1):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                out = ttnn.matmul(ta, tb, compute_kernel_config=cfg, dtype=ttnn.float32)
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
                ttnn.deallocate(out)
            ms, flops = float(np.median(ts[1:])) * 1e3, 2 * int(np.prod(a)) * b[-1]
            rows.append({"left": a, "right": b, "calls_per_step": calls, "ms": ms, "gflop_per_s": flops / ms / 1e6})
            print(f"{a} @ {b}: {ms:.2f} ms, {flops / ms / 1e6:.0f} GFLOP/s, x{calls} = {ms * calls:.1f} ms/step", flush=True)
            ttnn.deallocate(ta), ttnn.deallocate(tb)
    print(f"these GEMMs alone: {sum(r['ms'] * r['calls_per_step'] for r in rows):.1f} ms/step")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
