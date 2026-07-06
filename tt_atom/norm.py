"""Equivariant RMS layer norm over spherical-harmonic features (``rms_norm_sh``).

Mirrors ``fairchem ... nn/layer_norm.py:EquivariantRMSNormArraySphericalHarmonicsV2``:
center the l=0 channel, compute a degree-balanced RMS over all coefficients, scale by a
per-degree affine weight, and add an l=0 bias. All reductions/elementwise -- no matmul.
"""
from __future__ import annotations

import torch


def _l_of_coeff(lmax):
    return [l for l in range(lmax + 1) for _ in range(2 * l + 1)]


class RMSNormSH:
    def __init__(self, weights, prefix, device, *, lmax, num_channels, eps=1e-5):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.lmax = lmax
        self.C = num_channels
        self.eps = eps
        self.nsph = (lmax + 1) ** 2

        lc = _l_of_coeff(lmax)
        # degree-balance weight per coefficient: (1/(2l+1)) / (lmax+1)
        bdw = torch.tensor([1.0 / (2 * l + 1) / (lmax + 1) for l in lc]).view(1, self.nsph, 1)
        self.bdw = ttnn.from_torch(bdw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        aw = weights[f"{prefix}.affine_weight"]               # [lmax+1, C]
        aw_exp = aw[torch.tensor(lc)].view(1, self.nsph, self.C)   # [1, nsph, C]
        self.aw = ttnn.from_torch(aw_exp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ab = weights[f"{prefix}.affine_bias"].view(1, 1, self.C)
        self.ab = ttnn.from_torch(ab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x):
        """x: ttnn ``[N, nsph, C]`` -> ``[N, nsph, C]``."""
        ttnn = self.ttnn
        N = x.shape[0]
        from .device import l1_if_fits, L1_NODE_BUDGET
        self._cache_x = x                                     # for the analytic-force VJP
        # concat relocations only -> bit-identical (node PCC 1.0); keeps the norm's [N,nsph,C]
        # working set on-chip while it fits L1 (falls back to DRAM at large N). Use the tile-padded
        # width (nsph -> next mult of 32) since the 3D tensor pads the coeff dim.
        L1 = l1_if_fits(ttnn, N, ((self.nsph + 31) // 32) * 32 * self.C, budget=L1_NODE_BUDGET)
        # center l=0 across channels
        l0 = ttnn.slice(x, [0, 0, 0], [N, 1, self.C])
        l0_mean = ttnn.mean(l0, dim=2, keepdim=True)          # [N,1,1]
        l0c = ttnn.subtract(l0, l0_mean)
        rest = ttnn.slice(x, [0, 1, 0], [N, self.nsph, self.C])
        x = ttnn.concat([l0c, rest], dim=1, memory_config=L1)

        # degree-balanced component RMS
        fn = ttnn.multiply(x, x)
        fn = ttnn.multiply(fn, self.bdw)
        fn = ttnn.sum(fn, dim=1, keepdim=True)                # [N,1,C]
        fn = ttnn.mean(fn, dim=2, keepdim=True)               # [N,1,1]
        fn = ttnn.rsqrt(ttnn.add(fn, self.eps))

        out = ttnn.multiply(x, ttnn.multiply(fn, self.aw))    # broadcast [N,1,1]*[1,nsph,C]
        # add bias to l=0 only
        l0 = ttnn.add(ttnn.slice(out, [0, 0, 0], [N, 1, self.C]), self.ab)
        rest = ttnn.slice(out, [0, 1, 0], [N, self.nsph, self.C])
        return ttnn.concat([l0, rest], dim=1, memory_config=L1)
