"""Micro-profile tt_atom.scatter.segment_sum internals and cheaper alternatives at 2016-atom
scale, to decide whether the aggregation pole is fixable with existing ttnn ops (layout/memory
choices) or only with a new custom scatter-add kernel.

    source .source_env.sh
    TT_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH $PYREFENV benchmarks/bench_orb_scatter_internals.py
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
from ase.build import bulk


def _med(fn, ttnn, device, *, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    xs = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        xs.append((time.perf_counter() - t) * 1e3)
    return float(statistics.median(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--nz", type=int, default=7)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    import ttnn
    from tt_atom.device import open_device
    from tt_atom.scatter import build_gather, segment_sum
    from tt_atom.geometry import radius_graph

    device = open_device(0, trace_region_size=400_000_000)
    try:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (args.nx, args.ny, args.nz)
        N = len(atoms)
        pos0 = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        edge_index, _ = radius_graph(pos0, 6.0, cell=cell, pbc=[True, True, True])
        src, tgt = edge_index[0], edge_index[1]
        E = int(src.shape[0])
        C = 256
        gather_flat, Dmax = build_gather(tgt, N, E)
        gather_dev = ttnn.from_torch(torch.from_numpy(gather_flat), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        msg_host = torch.randn(E, C, dtype=torch.bfloat16) * 0.25
        msg = ttnn.from_torch(msg_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        print(f"N={N} E={E} C={C} Dmax={Dmax} N*Dmax={N*Dmax} "
              f"(N*Dmax*C*2 = {N*Dmax*C*2/1e6:.1f} MB, E*C*2 = {E*C*2/1e6:.1f} MB)", flush=True)

        # Current implementation end-to-end.
        cur = _med(lambda: segment_sum(ttnn, msg, gather_dev, Dmax, N, C), ttnn, device, iters=args.iters)
        print(f"current segment_sum:          {cur:.3f} ms  ({E*C*2*2/cur/1e6:.1f} GB/s in+out)", flush=True)

        # Sub-step breakdown of the current implementation.
        def zrow():
            return ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)
        z = _med(zrow, ttnn, device, iters=args.iters)
        print(f"  zrow (slice+multiply):       {z:.3f} ms", flush=True)

        def concat_pad():
            zr = ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)
            return ttnn.to_layout(ttnn.concat([msg, zr], dim=0), ttnn.ROW_MAJOR_LAYOUT)
        cp = _med(concat_pad, ttnn, device, iters=args.iters)
        print(f"  concat+to_layout RM:         {cp:.3f} ms", flush=True)

        mpad = ttnn.to_layout(ttnn.concat([msg, ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)],
                                          dim=0), ttnn.ROW_MAJOR_LAYOUT)
        def emb():
            return ttnn.embedding(gather_dev, mpad)
        e = _med(emb, ttnn, device, iters=args.iters)
        print(f"  embedding gather:            {e:.3f} ms", flush=True)

        g = ttnn.embedding(gather_dev, mpad)
        def reshape_tile():
            return ttnn.to_layout(ttnn.reshape(g, (N, Dmax, C)), ttnn.TILE_LAYOUT)
        rt = _med(reshape_tile, ttnn, device, iters=args.iters)
        print(f"  reshape+to_layout TILE:      {rt:.3f} ms", flush=True)

        gt = ttnn.to_layout(ttnn.reshape(g, (N, Dmax, C)), ttnn.TILE_LAYOUT)
        def summ():
            return ttnn.sum(gt, dim=1)
        sm = _med(summ, ttnn, device, iters=args.iters)
        print(f"  sum over Dmax:               {sm:.3f} ms", flush=True)

        # Alternative A: sum in ROW_MAJOR (no tile round-trip).
        g2 = ttnn.embedding(gather_dev, mpad)
        g2r = ttnn.reshape(g2, (N, Dmax, C))   # already ROW_MAJOR
        def sum_rm():
            return ttnn.sum(g2r, dim=1)
        smrm = _med(sum_rm, ttnn, device, iters=args.iters)
        print(f"alt A: sum ROW_MAJOR (no tile rt): {smrm:.3f} ms", flush=True)

        # Alternative B: ttnn.reduce instead of sum.
        # Alternative C: avoid the zero-pad concat -- use a prebuilt padded msg with an extra zero
        # row uploaded once (trace-safe via copy_host_to_device? no, constant). Here just measure
        # the gather+sum without re-concating each call by reusing mpad.
        def gather_sum_reuse():
            gg = ttnn.embedding(gather_dev, mpad)
            return ttnn.sum(ttnn.reshape(gg, (N, Dmax, C)), dim=1)
        gsr = _med(gather_sum_reuse, ttnn, device, iters=args.iters)
        print(f"alt C: embedding+reshape+sum (reuse mpad, RM sum): {gsr:.3f} ms", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
