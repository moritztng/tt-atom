"""One TensorNet message pass on one chip with stock ttnn ops, timed and graded against float64.

The pass is what each interaction does three times over (scalar, vector and tensor part held as [N, 128*9]):
gather the sender rows, scale by the per-edge weights, sum onto the receiver. Senders are gathered straight into
the receiver's padded slots, so the scatter is a sum over the degree axis and no second gather is needed.

    python benchmarks/bench_tensornet_pass.py --atoms 4096 --degree 28
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from bench_orb_perf_dollar_tt import AiclkSampler
from tt_atom import device as D
from tt_atom import scatter


def one_pass(ttnn, x, idx, w, n, d, width):
    g = scatter.embed_rows(ttnn, idx, x)                                     # [n*d, width], sender rows
    m = ttnn.multiply(ttnn.to_layout(g, ttnn.TILE_LAYOUT), w)
    red = lambda p: ttnn.sum(ttnn.reshape(ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT), (n, d, width)), dim=1)
    return scatter.linear32(ttnn, red, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=4096)
    ap.add_argument("--degree", type=int, default=28)
    ap.add_argument("--width", type=int, default=1152)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default="tensornet_pass.json")
    args = ap.parse_args()
    import ttnn

    torch.set_num_threads(1)
    n, d, width = args.atoms, args.degree, args.width
    g = torch.Generator().manual_seed(0)
    send = torch.randint(0, n, (n * d,), generator=g)
    x64 = torch.randn(n, width, generator=g, dtype=torch.float64)
    w64 = torch.rand(n * d, width, generator=g, dtype=torch.float64)
    ref = (x64[send] * w64).reshape(n, d, width).sum(1)
    rows = []
    with D.device(0) as dev:
        nodes = sorted({os.readlink(f"/proc/self/fd/{f}").rsplit("/", 1)[1] for f in os.listdir("/proc/self/fd")
                        if os.path.exists(f"/proc/self/fd/{f}") and os.readlink(f"/proc/self/fd/{f}").startswith("/dev/tenstorrent/")})
        idx = ttnn.from_torch(send.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        for name, dt in (("bfloat16", ttnn.bfloat16), ("float32", ttnn.float32)):
            x = ttnn.from_torch(x64.float(), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            w = ttnn.from_torch(w64.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            ms, out = [], None
            with AiclkSampler() as clk:
                for _ in range(args.reps + 1):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    out = one_pass(ttnn, x, idx, w, n, d, width)
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3)
            got = ttnn.to_torch(out).double().reshape(n, width)
            err = (got - ref).abs()
            clock = {k: v for k, v in clk.summary().items() if k in nodes}
            rows.append({"dtype": name, "atoms": n, "edges": n * d, "width": width, "ms_median": float(np.median(ms[1:])),
                         "ms_min": min(ms[1:]), "ms_raw": ms, "rel_err_mean": float(err.mean() / ref.abs().mean()),
                         "rel_err_max": float(err.max() / ref.abs().max()), "opened_nodes": nodes,
                         "aiclk_mhz_during": clock, "host_load1": os.getloadavg()[0]})
            print(json.dumps(rows[-1]), flush=True)
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=1)
            ttnn.deallocate(x), ttnn.deallocate(w)


if __name__ == "__main__":
    main()
