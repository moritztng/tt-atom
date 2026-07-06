"""Grid feed-forward (``GridAtomwise``) — the per-node MLP in the eSCN-MD block.

Spherical-harmonic coefficients are projected to a real-space S2 grid with a fixed linear
map, a pointwise 3-layer MLP runs over channels at every grid point, then the result is
projected back to coefficients. The two projections are constant buffers (topology-free),
so on device this is just two transpose-matmuls around a channelwise MLP.

Reference: ``fairchem ... escn_md_block.py:GridAtomwise`` + ``common/so3.py:SO3_Grid``::

    x_grid = einsum("bai, zic -> zbac", to_grid_mat, x)   # to_grid
    x_grid = grid_mlp(x_grid)                               # pointwise over channels
    x      = einsum("bai, zbac -> zic", from_grid_mat, x_grid)  # from_grid
"""
from __future__ import annotations

from .device import compute_kernel_config


def _to_dev(t, device, dtype):
    import ttnn

    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class GridAtomwise:
    def __init__(self, weights, prefix, device, to_grid_mat, from_grid_mat, *, fast=False):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.kcfg = compute_kernel_config()
        wdtype = ttnn.bfloat8_b if fast else ttnn.bfloat16

        # to_grid_mat / from_grid_mat: [b, a, nsph]; flatten the (b, a) grid -> [npts, nsph].
        b, a, nsph = to_grid_mat.shape
        self.npts = b * a
        self.nsph = nsph
        tg = to_grid_mat.reshape(self.npts, nsph)          # x_grid = tg @ x  (over nsph)
        fg = from_grid_mat.reshape(self.npts, nsph)         # x = fg^T @ x_grid (over npts)
        # device matmuls operate on [.., C, k] @ [k, n]; store the right-multiply operands.
        self.tg_T = _to_dev(tg.T.contiguous(), device, wdtype)   # [nsph, npts]
        self.fg = _to_dev(fg.contiguous(), device, wdtype)       # [npts, nsph]

        self.w0 = _to_dev(weights[f"{prefix}.grid_mlp.0.weight"].T.contiguous(), device, wdtype)
        self.w2 = _to_dev(weights[f"{prefix}.grid_mlp.2.weight"].T.contiguous(), device, wdtype)
        self.w4 = _to_dev(weights[f"{prefix}.grid_mlp.4.weight"].T.contiguous(), device, wdtype)

    def __call__(self, x):
        """x: ttnn ``[N, nsph, C]`` -> ``[N, nsph, C]``."""
        ttnn = self.ttnn
        from .device import l1_if_fits, L1_NODE_BUDGET
        # gate residency on the grid tensor [N, npts, C] fitting L1 (else DRAM at large N). Use the
        # tile-padded npts (42 -> next mult of 32) since the 3D grid tensor pads the point dim.
        _npts_pad = ((self.npts + 31) // 32) * 32
        L1 = l1_if_fits(ttnn, x.shape[0], _npts_pad * x.shape[2], budget=L1_NODE_BUDGET)
        # The whole grid transform (to_grid -> pointwise MLP -> from_grid) is a BW-bound chain of
        # matmuls/transposes over the large [N, npts, C] real-space grid tensor. Keeping those
        # intermediates L1-resident cuts the chain's DRAM traffic to just the input read + final
        # write -- measured ~20% faster end-to-end at N>=128 (node PCC ~1.0 vs the DRAM path; the
        # only change is the matmul reduction grid, bf16-equivalent). Final output spills to DRAM
        # so the block's residual add stays on the small DRAM-interleaved node tensor.
        # to_grid:  x_grid[z, p, c] = sum_i tg[p, i] x[z, i, c]
        xt = ttnn.transpose(x, 1, 2, memory_config=L1)                   # [N, C, nsph]
        g = ttnn.matmul(xt, self.tg_T, compute_kernel_config=self.kcfg, memory_config=L1)  # [N, C, npts]
        g = ttnn.transpose(g, 1, 2, memory_config=L1)                    # [N, npts, C]
        # pointwise MLP over channels (no bias, SiLU between)
        a1 = ttnn.matmul(g, self.w0, compute_kernel_config=self.kcfg, memory_config=L1)
        g = ttnn.silu(a1)
        a2 = ttnn.matmul(g, self.w2, compute_kernel_config=self.kcfg, memory_config=L1)
        g = ttnn.silu(a2)
        self._cache_a1, self._cache_a2 = a1, a2              # for the analytic-force VJP
        g = ttnn.matmul(g, self.w4, compute_kernel_config=self.kcfg, memory_config=L1)     # [N, npts, C]
        # from_grid: x[z, i, c] = sum_p fg[p, i] x_grid[z, p, c]
        gt = ttnn.transpose(g, 1, 2, memory_config=L1)                   # [N, C, npts]
        o = ttnn.matmul(gt, self.fg, compute_kernel_config=self.kcfg)    # [N, C, nsph]
        return ttnn.transpose(o, 1, 2)                       # [N, nsph, C]
