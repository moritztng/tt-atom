"""Equivariant RMS layer norm over spherical-harmonic features (``rms_norm_sh``).

Mirrors ``fairchem ... nn/layer_norm.py:EquivariantRMSNormArraySphericalHarmonicsV2``:
center the l=0 channel, compute a degree-balanced RMS over all coefficients, scale by a
per-degree affine weight, and add an l=0 bias. All reductions/elementwise -- no matmul.
"""
from __future__ import annotations

import os

import torch

# RMSNormSH runs on 3D [N, nsph, C] whose tiny coefficient dim (nsph=9) tile-pads to 32 -- a ~3.5x
# blowup on every reduction/elementwise. The whole norm is a scalar-per-node RMS + per-(coeff,chan)
# affine, so it reformulates cleanly in flat [N, nsph*C]: the degree-balanced RMS folds into ONE
# weighted sum (wvec = bdw[coeff]/C) and the affine into flat multiplies -- no pad. Gated for A/B.
_NORM_FLAT = os.environ.get("TT_ATOM_NORM_FLAT", "1") == "1"


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

        # flat-layout constants (see module docstring): wvec[j] = bdw[coeff(j)]/C, awvec = aw flat,
        # abf = affine bias as [1,C]. All [1, nsph*C] (or [1,C]) for broadcast against flat [N,nsph*C].
        self.flat = _NORM_FLAT
        if self.flat:
            wvec = torch.tensor([1.0 / (2 * l + 1) / (lmax + 1) for l in lc]).view(self.nsph, 1)
            wvec = (wvec.expand(self.nsph, self.C).reshape(1, self.nsph * self.C) / self.C)
            self.wvec = ttnn.from_torch(wvec.contiguous(), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device)
            self.awvec = ttnn.from_torch(aw_exp.reshape(1, self.nsph * self.C).contiguous(),
                                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            self.abf = ttnn.from_torch(weights[f"{prefix}.affine_bias"].view(1, self.C),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, x):
        """x: ttnn ``[N, nsph, C]`` -> ``[N, nsph, C]``."""
        ttnn = self.ttnn
        N = x.shape[0]
        if self.flat:
            return self._call_flat(x)
        from .device import l1_if_fits, L1_NODE_BUDGET
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
        # cache the centered input + rsqrt scale for the analytic-force VJP (rmsnorm_bw then skips
        # recomputing the centering + degree-balanced RMS -- fewer backward device ops, bit-exact)
        self._cache_xc, self._cache_inv = x, fn

        out = ttnn.multiply(x, ttnn.multiply(fn, self.aw))    # broadcast [N,1,1]*[1,nsph,C]
        # add bias to l=0 only
        l0 = ttnn.add(ttnn.slice(out, [0, 0, 0], [N, 1, self.C]), self.ab)
        rest = ttnn.slice(out, [0, 1, 0], [N, self.nsph, self.C])
        return ttnn.concat([l0, rest], dim=1, memory_config=L1)

    def _call_flat(self, x):
        """Flat-layout RMSNormSH ([N, nsph*C]) -- no 3D coeff tile-pad. Bit-compatible with the 3D
        path. Caches the centered flat input xc and rsqrt scale inv [N,1] for the flat backward."""
        ttnn = self.ttnn
        N, C, W = x.shape[0], self.C, self.nsph * self.C
        xf = ttnn.reshape(x, (N, W))
        # center l=0 (first C cols) across channels
        l0 = ttnn.slice(xf, [0, 0], [N, C])
        l0c = ttnn.subtract(l0, ttnn.mean(l0, dim=1, keepdim=True))       # [N,C]
        xc = ttnn.concat([l0c, ttnn.slice(xf, [0, C], [N, W])], dim=1)    # [N, W]
        # degree-balanced RMS as one weighted sum: ms = sum_j wvec_j * xc_j^2
        ms = ttnn.sum(ttnn.multiply(ttnn.multiply(xc, xc), self.wvec), dim=1, keepdim=True)  # [N,1]
        inv = ttnn.rsqrt(ttnn.add(ms, self.eps))
        self._cache_xc, self._cache_inv = xc, inv
        out = ttnn.multiply(xc, ttnn.multiply(inv, self.awvec))          # [N,1]*[1,W] broadcast
        # bias on l=0 only
        l0o = ttnn.add(ttnn.slice(out, [0, 0], [N, C]), self.abf)
        out = ttnn.concat([l0o, ttnn.slice(out, [0, C], [N, W])], dim=1)
        return ttnn.reshape(out, (N, self.nsph, C))
