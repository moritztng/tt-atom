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

import os

import torch

from .device import compute_kernel_config

# The whole SO(2) convolution (m=0 dense linear + every m>0 real/imag mixing) is ONE linear map
# from the post-radial input [E, nsph*Cin] to [extra | out]. The m>0 cross terms
#   out_real = real@Wa - imag@Wb ,  out_imag = real@Wb + imag@Wa
# fold into a single constant block-structured weight, collapsing ~27 slice/matmul/combine ops
# (per conv, per pass) into one ttnn.linear. The device is op-count/DRAM-glue bound (matmul
# compute floor is ~5 ms/step), so the ~3x extra MACs from the block-diagonal zeros are cheap
# relative to the eliminated intermediate traffic. Bit-compatible column ordering; gated so the
# per-m path stays available for A/B.
_SO2_FUSED = os.environ.get("TT_ATOM_SO2_FUSED", "1") == "1"


def _to_dev(t: torch.Tensor, device, dtype):
    import ttnn

    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class RadialMLP:
    """Linear -> (LayerNorm -> SiLU) x2 -> Linear. Produces per-m radial weights from the
    invariant edge embedding. Mirrors ``fairchem ... nn/radial.py:RadialMLP``."""

    def __init__(self, weights, prefix, device, wdtype, out_scale=1.0):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.eps = 1e-5
        # net.0 Linear, net.1 LayerNorm, net.3 Linear, net.4 LayerNorm, net.6 Linear
        self.w0 = _to_dev(weights[f"{prefix}.net.0.weight"].T.contiguous(), device, wdtype)
        self.b0 = _to_dev(weights[f"{prefix}.net.0.bias"], device, wdtype)
        self.ln1w = _to_dev(weights[f"{prefix}.net.1.weight"], device, ttnn.bfloat16)
        self.ln1b = _to_dev(weights[f"{prefix}.net.1.bias"], device, ttnn.bfloat16)
        self.w3 = _to_dev(weights[f"{prefix}.net.3.weight"].T.contiguous(), device, wdtype)
        self.b3 = _to_dev(weights[f"{prefix}.net.3.bias"], device, wdtype)
        self.ln4w = _to_dev(weights[f"{prefix}.net.4.weight"], device, ttnn.bfloat16)
        self.ln4b = _to_dev(weights[f"{prefix}.net.4.bias"], device, ttnn.bfloat16)
        # ``out_scale`` folds a downstream constant (e.g. the edge-degree 1/rescale) into the final
        # linear in fp32 before the bf16 cast, so it lands inside the matmul's fp32 accumulation
        # instead of a lossy bf16 elementwise multiply (0.2 is not representable in bf16). w6 is the
        # scale factor applied to a linear's output, so scaling both weight and bias is exact.
        self.out_scale = float(out_scale)
        self.w6 = _to_dev(weights[f"{prefix}.net.6.weight"].T.contiguous() * self.out_scale,
                          device, wdtype)
        self.b6 = _to_dev(weights[f"{prefix}.net.6.bias"] * self.out_scale, device, wdtype)
        # broadcast copies of the LN scales for the hand-written backward ([1, n])
        n1 = weights[f"{prefix}.net.1.weight"].shape[0]
        n4 = weights[f"{prefix}.net.4.weight"].shape[0]
        self.ln1w_b = _to_dev(weights[f"{prefix}.net.1.weight"].reshape(1, n1), device, ttnn.bfloat16)
        self.ln4w_b = _to_dev(weights[f"{prefix}.net.4.weight"].reshape(1, n4), device, ttnn.bfloat16)
        self.kcfg = compute_kernel_config()

    def __call__(self, x_edge):
        ttnn = self.ttnn
        a0 = ttnn.linear(x_edge, self.w0, bias=self.b0, compute_kernel_config=self.kcfg)
        n1 = ttnn.layer_norm(a0, weight=self.ln1w, bias=self.ln1b, epsilon=self.eps)
        s1 = ttnn.silu(n1)
        a3 = ttnn.linear(s1, self.w3, bias=self.b3, compute_kernel_config=self.kcfg)
        n4 = ttnn.layer_norm(a3, weight=self.ln4w, bias=self.ln4b, epsilon=self.eps)
        s2 = ttnn.silu(n4)
        # cache pre-norm / pre-silu activations for the analytic-force VJP (device radial backward)
        self._cache = (a0, n1, a3, n4)
        return ttnn.linear(s2, self.w6, bias=self.b6, compute_kernel_config=self.kcfg)

    def _ln_bw(self, g_out, x, w_b):
        """VJP of ``F.layer_norm`` (grad wrt the LN input ``x``, affine scale folded in).
        ``x`` is the cached forward input; ``w_b`` the LN scale as [1, n] for broadcast."""
        ttnn = self.ttnn
        mu = ttnn.mean(x, dim=1, keepdim=True)
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=1, keepdim=True)
        rstd = ttnn.rsqrt(ttnn.add(var, self.eps))
        y = ttnn.multiply(xc, rstd)                       # normalized activation
        g_y = ttnn.multiply(g_out, w_b)                   # broadcast LN scale over rows
        m1 = ttnn.mean(g_y, dim=1, keepdim=True)
        m2 = ttnn.mean(ttnn.multiply(g_y, y), dim=1, keepdim=True)
        inner = ttnn.subtract(ttnn.subtract(g_y, m1), ttnn.multiply(y, m2))
        return ttnn.multiply(rstd, inner)

    def bw(self, g_out):
        """Adjoint of the radial MLP: given ``g_out`` at the output, return g wrt ``x_edge``.
        Mirrors :meth:`__call__` on device (transpose-matmuls + hand-written LN/SiLU backward),
        replacing the host ``torch.autograd`` radial finish (~100 ms at N=128 on the force path)."""
        ttnn = self.ttnn
        from .forces import silu_bw, _mm
        a0, n1, a3, n4 = self._cache
        g_s2 = _mm(ttnn, g_out, self.w6, self.kcfg)        # [E, hidden]
        g_n4 = silu_bw(ttnn, g_s2, n4)
        g_a3 = self._ln_bw(g_n4, a3, self.ln4w_b)
        g_s1 = _mm(ttnn, g_a3, self.w3, self.kcfg)
        g_n1 = silu_bw(ttnn, g_s1, n1)
        g_a0 = self._ln_bw(g_n1, a0, self.ln1w_b)
        return _mm(ttnn, g_a0, self.w0, self.kcfg)         # [E, x_edge dim]


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

        # fused single-matmul weight (see module docstring). Built from the SAME weights, so the
        # output column ordering is bit-identical to the per-m path: [extra | m0coeffs | m1r | m1i
        # | ... ]. self.fused_extra_out is the extra-gating width sliced off the front.
        self.fused_w = self.fused_b = None
        self.fused_extra_out = extra_m0_output_channels
        if _SO2_FUSED:
            self._build_fused(weights, prefix, wdtype)

    def _build_fused(self, weights, prefix, wdtype):
        import ttnn

        lmax, mmax, Cin = self.lmax, self.mmax, self.Cin
        w_m0 = weights[f"{prefix}.fc_m0.weight"].T.contiguous().float()       # [in0, out0=640]
        b_m0 = weights[f"{prefix}.fc_m0.bias"].float()                        # [out0]
        in_total = self.in_offsets[-1]
        out0 = w_m0.shape[1]
        hh = [weights[f"{prefix}.so2_m_conv.{m-1}.fc.weight"].shape[0] // 2
              for m in range(1, mmax + 1)]                                    # Hh per m
        out_total = out0 + sum(2 * h for h in hh)
        W = torch.zeros(in_total, out_total, dtype=torch.float32)
        b = torch.zeros(out_total, dtype=torch.float32)
        # m = 0 block (rows [0:in_offsets[1]] -> cols [0:out0]); carries extra+coeffs and bias
        W[0:self.in_offsets[1], 0:out0] = w_m0
        b[0:out0] = b_m0
        off = out0
        for m in range(1, mmax + 1):
            Hh = hh[m - 1]
            K = self.num_coef[m] * Cin
            r0 = self.in_offsets[m]; i0 = r0 + K                              # real / imag input col starts
            wm = weights[f"{prefix}.so2_m_conv.{m-1}.fc.weight"].T.contiguous().float()  # [K, 2Hh]
            Wa = wm[:, :Hh]; Wb = wm[:, Hh:2 * Hh]
            # out_real = real@Wa - imag@Wb ; out_imag = real@Wb + imag@Wa
            W[r0:r0 + K, off:off + Hh] = Wa
            W[i0:i0 + K, off:off + Hh] = -Wb
            W[r0:r0 + K, off + Hh:off + 2 * Hh] = Wb
            W[i0:i0 + K, off + Hh:off + 2 * Hh] = Wa
            off += 2 * Hh
        self.fused_w = _to_dev(W.contiguous(), self.device, wdtype)
        self.fused_b = _to_dev(b.contiguous(), self.device, wdtype)

    def __call__(self, x, x_edge=None):
        """x: ttnn ``[E, (lmax+1)**2, Cin]`` or flat ``[E, (lmax+1)**2 * Cin]``; returns flat
        ``[E, (lmax+1)**2 * H]`` (+ extra_m0 gating features ``[E, extra]`` when configured)."""
        ttnn = self.ttnn
        E = x.shape[0]
        nsph = (self.lmax + 1) ** 2
        xf = ttnn.reshape(x, (E, nsph * self.Cin)) if len(x.shape) == 3 else x

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

        # fused single-matmul path: whole conv is one linear (see module docstring)
        if self.fused_w is not None:
            full = ttnn.linear(xf, self.fused_w, bias=self.fused_b, compute_kernel_config=self.kcfg)
            if self.extra:
                extra = ttnn.slice(full, [0, 0], [E, self.extra])
                out = ttnn.slice(full, [0, self.extra], [E, full.shape[1]])
                return out, extra
            return full

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

        out = ttnn.concat(out_blocks, dim=1)             # flat [E, (lmax+1)^2 * H], m-primed
        return (out, extra) if self.extra else out
