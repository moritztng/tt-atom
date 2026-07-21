"""Parity + speed A/B for the new scatter.segment_sum vs the old TILE-concat path.

    source .source_env.sh
    TT_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH $PYREFENV benchmarks/bench_orb_scatter_ab.py
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


def _segment_sum_old(ttnn, msg, gather_dev, Dmax, N, W):
    zrow = ttnn.multiply(ttnn.slice(msg, [0, 0], [1, W]), 0.0)
    mpad = ttnn.to_layout(ttnn.concat([msg, zrow], dim=0), ttnn.ROW_MAJOR_LAYOUT)
    g = ttnn.embedding(gather_dev, mpad)
    g = ttnn.to_layout(ttnn.reshape(g, (N, Dmax, W)), ttnn.TILE_LAYOUT)
    return ttnn.sum(g, dim=1)


def main():
    import ttnn
    from tt_atom.device import open_device
    from tt_atom.scatter import build_gather, segment_sum
    from tt_atom.geometry import radius_graph

    device = open_device(0, trace_region_size=400_000_000)
    try:
        for nx, ny, nz in [(3, 3, 3), (6, 6, 7)]:
            atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz)
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
            torch.manual_seed(17)
            msg_host = torch.randn(E, C, dtype=torch.bfloat16) * 0.25
            msg = ttnn.from_torch(msg_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            out_old = ttnn.to_torch(_segment_sum_old(ttnn, msg, gdev, Dmax, N, C)).float()
            out_new = ttnn.to_torch(segment_sum(ttnn, msg, gdev, Dmax, N, C)).float()
            diff = (out_old - out_new).abs()
            pcc = float(torch.corrcoef(torch.stack([out_old.reshape(-1), out_new.reshape(-1)]))[0, 1])
            # Reference: dense scatter-add on host.
            ref = torch.zeros(N, C)
            ref.index_add_(0, tgt, msg_host.float())
            pcc_ref = float(torch.corrcoef(torch.stack([ref.reshape(-1), out_new.reshape(-1)]))[0, 1])
            print(f"N={N} E={E} Dmax={Dmax}: new-vs-old PCC={pcc:.6f} max_abs={diff.max():.6e} "
                  f"| new-vs-host-scatter PCC={pcc_ref:.6f}", flush=True)

            t_old = _med(lambda: _segment_sum_old(ttnn, msg, gdev, Dmax, N, C), ttnn, device, iters=15)
            t_new = _med(lambda: segment_sum(ttnn, msg, gdev, Dmax, N, C), ttnn, device, iters=15)
            print(f"  old={t_old:.3f} ms  new={t_new:.3f} ms  speedup={t_old/t_new:.2f}x", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
