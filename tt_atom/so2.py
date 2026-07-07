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
# Route the radial-MLP LayerNorm backward (_ln_bw) through the custom fused reduction kernel
# (ttnn.experimental.fused_ln_bw): one kernel launch computes mean/rstd + dx with W L1-resident,
# vs ~15 ttnn ops. Biggest single fuseable glue (~14 ms/step, x~10 calls). Needs source-ttnn build.
_FUSED_LNBW = os.environ.get("TT_ATOM_FUSED_LNBW") == "1"
_RED_CACHE: dict = {}


def _red_tile(ttnn, device, W):
    """[32,32] reduction selector: column 0 = 1/W (matmul rowsum-to-col0 = row mean). Cached."""
    key = (id(device), W)
    t = _RED_CACHE.get(key)
    if t is None:
        r = torch.zeros(32, 32); r[:, 0] = 1.0 / W
        t = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _RED_CACHE[key] = t
    return t


def _to_dev(t: torch.Tensor, device, dtype):
    import ttnn

    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class RadialMLP:
    """Linear -> (LayerNorm -> SiLU) x2 -> Linear. Produces per-m radial weights from the
    invariant edge embedding. Mirrors ``fairchem ... nn/radial.py:RadialMLP``."""

    def __init__(self, weights, prefix, device, wdtype, out_scale=1.0, dup_index=None):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.eps = 1e-5
        # ``dup_index`` (list of original output rows) duplicates/reorders net.6's output so the MLP
        # emits the SO2 per-m multiplier ``mult`` [E, nsph*Cin] directly (real/imag blocks repeated),
        # folding the so2 slice+concat mult-build into the constant weight. The backward is automatic:
        # matmul with the duplicated weight sums the repeated rows' gradients (== the old collapse).
        self._dup_index = dup_index
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
        w6 = weights[f"{prefix}.net.6.weight"] * self.out_scale            # [out, hidden]
        b6 = weights[f"{prefix}.net.6.bias"] * self.out_scale             # [out]
        if dup_index is not None:
            idx = torch.as_tensor(dup_index, dtype=torch.long)
            w6 = w6[idx]; b6 = b6[idx]                                    # duplicate/reorder rows
        self.w6 = _to_dev(w6.T.contiguous(), device, wdtype)
        self.b6 = _to_dev(b6, device, wdtype)
        # broadcast copies of the LN scales for the hand-written backward ([1, n])
        n1 = weights[f"{prefix}.net.1.weight"].shape[0]
        n4 = weights[f"{prefix}.net.4.weight"].shape[0]
        self.ln1w_b = _to_dev(weights[f"{prefix}.net.1.weight"].reshape(1, n1), device, ttnn.bfloat16)
        self.ln4w_b = _to_dev(weights[f"{prefix}.net.4.weight"].reshape(1, n4), device, ttnn.bfloat16)
        self.kcfg = compute_kernel_config()

    def __call__(self, x_edge):
        ttnn = self.ttnn
        # x_edge arrives ROW_MAJOR (cheap trace refresh); tilize on device here (see GraphContext.x_edge)
        if x_edge.layout != ttnn.TILE_LAYOUT:
            x_edge = ttnn.to_layout(x_edge, ttnn.TILE_LAYOUT)
        a0 = ttnn.linear(x_edge, self.w0, bias=self.b0, compute_kernel_config=self.kcfg)
        n1 = ttnn.layer_norm(a0, weight=self.ln1w, bias=self.ln1b, epsilon=self.eps)
        s1 = ttnn.silu(n1)
        a3 = ttnn.linear(s1, self.w3, bias=self.b3, compute_kernel_config=self.kcfg)
        n4 = ttnn.layer_norm(a3, weight=self.ln4w, bias=self.ln4b, epsilon=self.eps)
        s2 = ttnn.silu(n4)
        # cache pre-norm / pre-silu activations for the analytic-force VJP (device radial backward)
        self._cache = (a0, n1, a3, n4)
        return ttnn.linear(s2, self.w6, bias=self.b6, compute_kernel_config=self.kcfg)

    def _silu_ln_bw(self, g, n, x, w_b):
        """Fused SiLU-bw + LayerNorm-bw in ONE kernel launch: computes
        ``dx = ln_bw( silu'(n) * g * gamma, x )`` where ``gy = g * silu'(n) * gamma`` is built
        in-kernel (folds the external ``silu_bw`` op AND the affine-scale multiply). ``n`` is the
        cached pre-silu activation (LN output); ``x`` the cached LN input; ``w_b`` the LN scale [1,W]."""
        import struct
        ttnn = self.ttnn
        W = x.shape[-1]
        red = _red_tile(ttnn, self.device, W)
        eps_bits = struct.unpack("<I", struct.pack("<f", self.eps))[0]
        op = ttnn._ttnn.operations.experimental.fused_ln_bw
        return op(g, x, red, n, w_b, W, eps_bits)

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
        if _FUSED_LNBW and a3.shape[-1] % 32 == 0:
            g_a3 = self._silu_ln_bw(g_s2, n4, a3, self.ln4w_b)   # one kernel: silu-bw + LN-bw
        else:
            g_a3 = self._ln_bw(silu_bw(ttnn, g_s2, n4), a3, self.ln4w_b)
        g_s1 = _mm(ttnn, g_a3, self.w3, self.kcfg)
        if _FUSED_LNBW and a0.shape[-1] % 32 == 0:
            g_a0 = self._silu_ln_bw(g_s1, n1, a0, self.ln1w_b)
        else:
            g_a0 = self._ln_bw(silu_bw(ttnn, g_s1, n1), a0, self.ln1w_b)
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
        # radial output is split per-m into widths num_coef[m]*Cin
        self.rad_sizes = [self.num_coef[m] * self.Cin for m in range(mmax + 1)]
        # dup_index makes the radial MLP emit the full mult [E, nsph*Cin] directly: m=0 block once,
        # each m>0 block twice (real|imag). Maps output positions -> original radial-output rows.
        # Only for the fused path (its backward relies on the dup weight summing repeated rows); the
        # non-fused A/B path keeps the plain [E, sum rad_sizes] output + slice/concat mult build.
        dup_index = None
        self._rad_dup = _SO2_FUSED and self.has_radial
        if self._rad_dup:
            off, dup_index = 0, list(range(self.rad_sizes[0]))
            off = self.rad_sizes[0]
            for m in range(1, mmax + 1):
                blk = list(range(off, off + self.rad_sizes[m]))
                dup_index += blk + blk                                   # real then imag
                off += self.rad_sizes[m]
        self.rad = (RadialMLP(weights, self.rad_prefix, device, wdtype, dup_index=dup_index)
                    if self.has_radial else None)

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
        """Per-m fused weights: m=0 one linear [in0->640]; each m>0 ONE dense matmul on the
        contiguous [real|imag] input with the [[Wa,Wb],[-Wb,Wa]] block [2K->2Hh]. No block-diagonal
        zeros (unlike a single whole-conv matmul) so no MAC blowup at large E, yet ~5 ops per conv
        instead of ~15 (kills the real/imag slices + subtract/add combine)."""
        lmax, mmax, Cin = self.lmax, self.mmax, self.Cin
        w_m0 = weights[f"{prefix}.fc_m0.weight"].T.contiguous().float()       # [in0, 640]
        b_m0 = weights[f"{prefix}.fc_m0.bias"].float()
        self.fused_wm0 = _to_dev(w_m0.contiguous(), self.device, wdtype)
        self.fused_bm0 = _to_dev(b_m0.contiguous(), self.device, wdtype)
        self.fused_m0_out = w_m0.shape[1]                                     # 640 (extra + coeffs)
        self.fused_wm = []                                                   # [2K, 2Hh] per m>0
        self.fused_out_w = [self.fused_m0_out - self.extra]                  # coeff out width per block
        for m in range(1, mmax + 1):
            wm = weights[f"{prefix}.so2_m_conv.{m-1}.fc.weight"].T.contiguous().float()  # [K, 2Hh]
            K, twoHh = wm.shape[0], wm.shape[1]
            Hh = twoHh // 2
            Wa = wm[:, :Hh]; Wb = wm[:, Hh:2 * Hh]
            Wblk = torch.zeros(2 * K, 2 * Hh, dtype=torch.float32)
            Wblk[0:K, 0:Hh] = Wa; Wblk[0:K, Hh:2 * Hh] = Wb                   # real -> [out_real|out_imag]
            Wblk[K:2 * K, 0:Hh] = -Wb; Wblk[K:2 * K, Hh:2 * Hh] = Wa         # imag -> [out_real|out_imag]
            self.fused_wm.append(_to_dev(Wblk.contiguous(), self.device, wdtype))
            self.fused_out_w.append(2 * Hh)
        self.fused_w = True                                                  # sentinel: fused path on

    def __call__(self, x, x_edge=None):
        """x: ttnn ``[E, (lmax+1)**2, Cin]`` or flat ``[E, (lmax+1)**2 * Cin]``; returns flat
        ``[E, (lmax+1)**2 * H]`` (+ extra_m0 gating features ``[E, extra]`` when configured)."""
        ttnn = self.ttnn
        E = x.shape[0]
        nsph = (self.lmax + 1) ** 2
        xf = ttnn.reshape(x, (E, nsph * self.Cin)) if len(x.shape) == 3 else x

        if self.has_radial:
            rad = self.rad(x_edge)
            if self._rad_dup:
                mult = rad                               # [E, nsph*Cin] directly (dup weight)
            else:
                off, rms = 0, []
                for m in range(self.mmax + 1):
                    rms.append(ttnn.slice(rad, [0, off], [E, off + self.rad_sizes[m]]))
                    off += self.rad_sizes[m]
                mult = ttnn.concat([rms[0]] + sum(([rms[m], rms[m]]
                                    for m in range(1, self.mmax + 1)), []), dim=1)
            self._cache_xin, self._cache_mult = xf, mult     # for the analytic-force VJP
            xf = ttnn.multiply(xf, mult)

        # per-m fused path: m0 one linear + one dense matmul per m>0 (see _build_fused)
        if self.fused_w is not None:
            x0 = ttnn.slice(xf, [0, self.in_offsets[0]], [E, self.in_offsets[1]])
            full0 = ttnn.linear(x0, self.fused_wm0, bias=self.fused_bm0,
                                compute_kernel_config=self.kcfg)             # [E, 640]
            extra = None
            if self.extra:
                extra = ttnn.slice(full0, [0, 0], [E, self.extra])
                m0 = ttnn.slice(full0, [0, self.extra], [E, self.fused_m0_out])
            else:
                m0 = full0
            blocks = [m0]
            for m in range(1, self.mmax + 1):
                blk = ttnn.slice(xf, [0, self.in_offsets[m]], [E, self.in_offsets[m + 1]])  # [E,2K]
                blocks.append(ttnn.matmul(blk, self.fused_wm[m - 1], compute_kernel_config=self.kcfg))
            out = ttnn.concat(blocks, dim=1)                                 # [E, nsph*H]
            return (out, extra) if self.extra else out

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
