"""Quantify layout-only fixes for the aggregation scatter pole (no custom C++ kernel).

The current segment_sum forces a TILE->ROW_MAJOR conversion of the 47 MB [E,C] message because
ttnn.embedding requires ROW_MAJOR input. This measures several alternatives that keep the
reduction in ROW_MAJOR and/or share one conversion across the two (sent/recv) scatters, to bound
the layout-only win before deciding whether a custom scatter-add kernel is warranted.

    source .source_env.sh
    TT_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH $PYREFENV benchmarks/bench_orb_scatter_layout.py
"""
from __future__ import annotations

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
    import ttnn
    from tt_atom.device import open_device
    from tt_atom.scatter import build_gather
    from tt_atom.geometry import radius_graph

    device = open_device(0, trace_region_size=400_000_000)
    try:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (6, 6, 7)
        N = len(atoms)
        pos0 = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        edge_index, _ = radius_graph(pos0, 6.0, cell=cell, pbc=[True, True, True])
        src, tgt = edge_index[0], edge_index[1]
        E = int(src.shape[0])
        C = 256
        gflat, Dmax = build_gather(tgt, N, E)
        gdev = ttnn.from_torch(torch.from_numpy(gflat), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        msg_host = torch.randn(E, C, dtype=torch.bfloat16) * 0.25
        msg = ttnn.from_torch(msg_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        print(f"N={N} E={E} C={C} Dmax={Dmax}", flush=True)

        # 0: current implementation (baseline).
        from tt_atom.scatter import segment_sum
        cur = _med(lambda: segment_sum(ttnn, msg, gdev, Dmax, N, C), ttnn, device, iters=10)
        print(f"[0] current segment_sum:                {cur:.3f} ms", flush=True)

        # Separate concat vs to_layout RM.
        def concat_only():
            zr = ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)
            return ttnn.concat([msg, zr], dim=0)            # stays TILE
        co = _med(concat_only, ttnn, device, iters=10)
        print(f"    concat only (TILE):                  {co:.3f} ms", flush=True)
        ct = ttnn.concat([msg, ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)], dim=0)
        def to_rm():
            return ttnn.to_layout(ct, ttnn.ROW_MAJOR_LAYOUT)
        tr = _med(to_rm, ttnn, device, iters=10)
        print(f"    to_layout TILE->RM on [E+1,C]:        {tr:.3f} ms", flush=True)

        # 1: skip the post-gather to_layout TILE -- sum in ROW_MAJOR, convert only the small [N,C] out.
        mpad = ttnn.to_layout(ttnn.concat([msg, ttnn.multiply(ttnn.slice(msg, [0, 0], [1, C]), 0.0)],
                                          dim=0), ttnn.ROW_MAJOR_LAYOUT)
        def v1():
            g = ttnn.embedding(gdev, mpad)
            g = ttnn.reshape(g, (N, Dmax, C))               # RM
            out = ttnn.sum(g, dim=1)                         # [N,C] RM
            return ttnn.to_layout(out, ttnn.TILE_LAYOUT)     # small [N,C]
        v1t = _med(v1, ttnn, device, iters=10)
        print(f"[1] sum RM + convert small out:          {v1t:.3f} ms", flush=True)

        # 2: pad instead of concat (avoid the zrow op).
        def v2():
            mp = ttnn.pad(msg, [E + 1, C], [0, 0], 0.0)      # pad a zero row, TILE
            mpr = ttnn.to_layout(mp, ttnn.ROW_MAJOR_LAYOUT)
            g = ttnn.embedding(gdev, mpr)
            g = ttnn.reshape(g, (N, Dmax, C))
            out = ttnn.sum(g, dim=1)
            return ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        try:
            v2t = _med(v2, ttnn, device, iters=10)
            print(f"[2] pad (no zrow) + sum RM:              {v2t:.3f} ms", flush=True)
        except Exception as exc:
            print(f"[2] pad path failed: {type(exc).__name__}: {exc}", flush=True)

        # 3: share one TILE->RM conversion across two scatters (sent+recv), as the layer does.
        # Represents: convert updated_edges to RM once, then two (RM-mul + gather + sum).
        msg_rm = ttnn.to_layout(msg, ttnn.ROW_MAJOR_LAYOUT)
        # add a zero row once
        zrow_host = torch.zeros(1, C, dtype=torch.bfloat16)
        zrow = ttnn.from_torch(zrow_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        def v3_one():
            mp = ttnn.concat([msg_rm, zrow], dim=0)          # RM concat (cheap, RM)
            g = ttnn.embedding(gdev, mp)
            g = ttnn.reshape(g, (N, Dmax, C))
            out = ttnn.sum(g, dim=1)
            return ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        # cost of the shared conversion (amortized over 2 scatters):
        conv = _med(lambda: ttnn.to_layout(msg, ttnn.ROW_MAJOR_LAYOUT), ttnn, device, iters=10)
        v3one = _med(v3_one, ttnn, device, iters=10)
        print(f"[3] shared: TILE->RM once = {conv:.3f} ms; one RM scatter = {v3one:.3f} ms", flush=True)
        print(f"    [3] two scatters total ~ {conv + 2*v3one:.3f} ms vs current 2x = {2*cur:.3f} ms", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
