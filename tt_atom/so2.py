"""SO(2) convolution — the compute heart of eSEN / eSCN-MD.

The SO(2) trick turns the SO(3) tensor product into a set of per-order (per-m) dense GEMMs,
which is exactly why this architecture maps cleanly onto Tenstorrent. We keep everything in a
flattened 2D ``[E, (lmax+1)**2 * C]`` layout where each order ``m`` occupies a contiguous,
tile-aligned column block, so the whole module is column slices + matmuls + elementwise ops
(no awkward tile-dim-1 slicing). Validated to PCC ~1.0 against the fairchem reference.

Reference: ``fairchem/core/models/uma/nn/so2_layers.py`` (``SO2_Convolution``). The l<->m
reordering (``to_m``) is folded into the host-side Wigner matrix, so on device we just split
features by m directly.
"""
from __future__ import annotations

import torch

from .device import compute_kernel_config


def _to_dev(t: torch.Tensor, device, dtype):
    import ttnn

    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class RadialMLP:
    """Linear -> (LayerNorm -> SiLU) x2 -> Linear. Produces per-m radial weights from the
    invariant edge embedding. Mirrors ``fairchem ... nn/radial.py:RadialMLP``."""

    def __init__(self, weights, prefix, device, wdtype):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        # net.0 Linear, net.1 LayerNorm, net.3 Linear, net.4 LayerNorm, net.6 Linear
        self.w0 = _to_dev(weights[f"{prefix}.net.0.weight"].T.contiguous(), device, wdtype)
        self.b0 = _to_dev(weights[f"{prefix}.net.0.bias"], device, wdtype)
        self.ln1w = _to_dev(weights[f"{prefix}.net.1.weight"], device, ttnn.bfloat16)
        self.ln1b = _to_dev(weights[f"{prefix}.net.1.bias"], device, ttnn.bfloat16)
        self.w3 = _to_dev(weights[f"{prefix}.net.3.weight"].T.contiguous(), device, wdtype)
        self.b3 = _to_dev(weights[f"{prefix}.net.3.bias"], device, wdtype)
        self.ln4w = _to_dev(weights[f"{prefix}.net.4.weight"], device, ttnn.bfloat16)
        self.ln4b = _to_dev(weights[f"{prefix}.net.4.bias"], device, ttnn.bfloat16)
        self.w6 = _to_dev(weights[f"{prefix}.net.6.weight"].T.contiguous(), device, wdtype)
        self.b6 = _to_dev(weights[f"{prefix}.net.6.bias"], device, wdtype)
        self.kcfg = compute_kernel_config()

    def __call__(self, x_edge):
        ttnn = self.ttnn
        x = ttnn.linear(x_edge, self.w0, bias=self.b0, compute_kernel_config=self.kcfg)
        x = ttnn.layer_norm(x, weight=self.ln1w, bias=self.ln1b, epsilon=1e-5)
        x = ttnn.silu(x)
        x = ttnn.linear(x, self.w3, bias=self.b3, compute_kernel_config=self.kcfg)
        x = ttnn.layer_norm(x, weight=self.ln4w, bias=self.ln4b, epsilon=1e-5)
        x = ttnn.silu(x)
        return ttnn.linear(x, self.w6, bias=self.b6, compute_kernel_config=self.kcfg)


class SO2Convolution:
    def __init__(self, weights, prefix, device, *, sphere_channels_in, m_output_channels,
                 lmax, mmax, extra_m0_output_channels=0, fast=False):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.Cin = sphere_channels_in
        self.H = m_output_channels
        self.lmax, self.mmax = lmax, mmax
        self.extra = extra_m0_output_channels
        self.kcfg = compute_kernel_config()
        wdtype = ttnn.bfloat8_b if fast else ttnn.bfloat16

        self.num_coef = [lmax - m + 1 for m in range(mmax + 1)]   # coeffs per order m
        # flattened column offsets of each m-block in the [E, (lmax+1)^2 * Cin] input
        self.in_offsets = [0]
        w0 = (lmax + 1) * self.Cin
        self.in_offsets.append(w0)
        for m in range(1, mmax + 1):
            self.in_offsets.append(self.in_offsets[-1] + 2 * self.num_coef[m] * self.Cin)

        self.has_radial = f"{prefix}.rad_func.net.0.weight" in weights
        self.rad_prefix = f"{prefix}.rad_func"
        self.rad = RadialMLP(weights, self.rad_prefix, device, wdtype) if self.has_radial else None
        # radial output is split per-m into widths num_coef[m]*Cin
        self.rad_sizes = [self.num_coef[m] * self.Cin for m in range(mmax + 1)]

        # m=0 dense linear (has bias)
        self.w_m0 = _to_dev(weights[f"{prefix}.fc_m0.weight"].T.contiguous(), device, wdtype)
        self.b_m0 = _to_dev(weights[f"{prefix}.fc_m0.bias"], device, wdtype)
        # m>0 linears (no bias)
        self.w_m = [
            _to_dev(weights[f"{prefix}.so2_m_conv.{m-1}.fc.weight"].T.contiguous(), device, wdtype)
            for m in range(1, mmax + 1)
        ]

    def __call__(self, x, x_edge=None):
        """x: ttnn ``[E, (lmax+1)**2, Cin]``; returns ``[E, (lmax+1)**2, H]`` (+ extra_m0
        gating features ``[E, extra]`` when configured)."""
        ttnn = self.ttnn
        E = x.shape[0]
        nsph = (self.lmax + 1) ** 2
        xf = ttnn.reshape(x, (E, nsph * self.Cin))

        if self.has_radial:
            rad = self.rad(x_edge)                       # [E, sum(num_coef*Cin)]
            off = 0
            rms = []
            for m in range(self.mmax + 1):
                rms.append(ttnn.slice(rad, [0, off], [E, off + self.rad_sizes[m]]))
                off += self.rad_sizes[m]
            mult = [rms[0]] + sum(([rms[m], rms[m]] for m in range(1, self.mmax + 1)), [])
            mult = ttnn.concat(mult, dim=1)
            self._cache_xin, self._cache_mult = xf, mult     # for the analytic-force VJP
            xf = ttnn.multiply(xf, mult)

        out_blocks = []

        # m = 0
        x0 = ttnn.slice(xf, [0, self.in_offsets[0]], [E, self.in_offsets[1]])
        x0 = ttnn.linear(x0, self.w_m0, bias=self.b_m0, compute_kernel_config=self.kcfg)
        extra = None
        if self.extra:
            extra = ttnn.slice(x0, [0, 0], [E, self.extra])
            x0 = ttnn.slice(x0, [0, self.extra], [E, x0.shape[1]])
        out_blocks.append(x0)                            # [E, H*(lmax+1)]

        # m > 0 -- two flat 2D matmuls on the real/imag halves. The earlier [E,2,nc*Cin]
        # reshape made the length-2 part-dim a tile dim (padded 2->32, a 16x data blowup and a
        # per-edge batched matmul); slicing the halves and running two plain GEMMs is ~80x
        # faster on device for the same math (validated) and keeps the radial layout intact.
        for m in range(1, self.mmax + 1):
            nc = self.num_coef[m]
            K = nc * self.Cin                            # half width (real or imag)
            Hh = self.w_m[m - 1].shape[1] // 2           # out_half = H*nc
            blk = ttnn.slice(xf, [0, self.in_offsets[m]], [E, self.in_offsets[m + 1]])  # [E,2K]
            real = ttnn.slice(blk, [0, 0], [E, K])
            imag = ttnn.slice(blk, [0, K], [E, 2 * K])
            fr = ttnn.matmul(real, self.w_m[m - 1], compute_kernel_config=self.kcfg)    # [E,2Hh]
            fi = ttnn.matmul(imag, self.w_m[m - 1], compute_kernel_config=self.kcfg)
            r0 = ttnn.slice(fr, [0, 0], [E, Hh])
            r1 = ttnn.slice(fr, [0, Hh], [E, 2 * Hh])
            i0 = ttnn.slice(fi, [0, 0], [E, Hh])
            i1 = ttnn.slice(fi, [0, Hh], [E, 2 * Hh])
            out_blocks.append(ttnn.subtract(r0, i1))     # real coeffs
            out_blocks.append(ttnn.add(i0, r1))          # imag coeffs

        out = ttnn.concat(out_blocks, dim=1)             # [E, H*(lmax+1)^2]
        out = ttnn.reshape(out, (E, nsph, self.H))
        return (out, extra) if self.extra else out
