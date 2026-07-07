"""Spectral feed-forward (``SpectralAtomwise``) — the per-node FF used by uma-s-1.

Where the grid FF (``grid.py``) projects to an S2 grid and runs a channelwise MLP, the spectral
FF stays in the spherical-harmonic basis: two ``SO3_Linear`` layers (a per-degree dense GEMM that
shares one weight matrix across the ``2l+1`` coefficients of each degree ``l``, with a bias on
``l=0`` only) with a gate nonlinearity between them. The gate is driven by scalar features
produced from the ``l=0`` channel by a small MLP.

Reference: ``fairchem ... escn_md_block.py:SpectralAtomwise`` + ``nn/so3_layers.py:SO3_Linear``::

    gating = SiLU(scalar_mlp(x[:, 0]))            # [N, lmax*hidden]
    x = so3_linear_1(x)                            # [N, nsph, hidden]
    x = GateActivation(m_prime=False)(gating, x)   # SiLU on l=0, sigmoid-gate on l>=1
    x = so3_linear_2(x)                            # [N, nsph, sphere_channels]

Unlike the SO(2) path the coefficients here stay in natural (l, m) order (no m-primed Wigner
reorder), so the gate's per-degree expansion is the plain ``[0]*3 + [1]*5`` map for lmax=2.
"""
from __future__ import annotations

import os

import torch

from .device import compute_kernel_config

# SO3_Linear shares one [cin,cout] weight per degree l across that degree's 2l+1 coefficients.
# The per-degree path slices x into 3D [N, 2l+1, cin] blocks and runs a batched matmul -- but the
# tiny coefficient dim (1/3/5) tile-pads to 32, a ~6-32x row blowup that makes this tiny-N module
# cost ~100 ms/step. Folding it into ONE flat 2D matmul with a block-diagonal-by-coefficient
# weight [nsph*cin, nsph*cout] kills the padding entirely (bit-compatible ordering). Gated for A/B.
_SPECTRAL_FUSED = os.environ.get("TT_ATOM_SPECTRAL_FUSED", "1") == "1"


def _to_dev(t, device, dtype):
    import ttnn

    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class SpectralAtomwise:
    def __init__(self, weights, prefix, device, *, sphere_channels, hidden_channels,
                 lmax, mmax, fast=False):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.C = sphere_channels
        self.H = hidden_channels
        self.lmax = lmax
        self.mmax = mmax
        self.nsph = (lmax + 1) ** 2
        self.kcfg = compute_kernel_config()
        wdtype = ttnn.bfloat8_b if fast else ttnn.bfloat16

        # scalar_mlp: Linear(C -> lmax*H) + SiLU, on the l=0 channel only
        self.smlp_w = _to_dev(weights[f"{prefix}.scalar_mlp.0.weight"].T.contiguous(), device, wdtype)
        self.smlp_b = _to_dev(weights[f"{prefix}.scalar_mlp.0.bias"], device, wdtype)

        # SO3_Linear weight is [lmax+1, out, in]; store per-degree [in, out] for x @ W
        def so3(name, cin, cout):
            W = weights[f"{prefix}.{name}.weight"]                     # [lmax+1, out, in]
            blocks = [_to_dev(W[l].T.contiguous(), device, wdtype) for l in range(lmax + 1)]
            b = _to_dev(weights[f"{prefix}.{name}.bias"].view(1, 1, cout), device, wdtype)
            return blocks, b

        self.l1_w, self.l1_b = so3("so3_linear_1", self.C, self.H)     # C -> H
        self.l2_w, self.l2_b = so3("so3_linear_2", self.H, self.C)     # H -> C

        # fused block-diagonal-by-coefficient weights (see module docstring). One flat 2D matmul.
        self.l1_wf = self.l1_bf = self.l2_wf = self.l2_bf = self.gate_exp_w = None
        if _SPECTRAL_FUSED:
            self.l1_wf, self.l1_bf = self._build_so3_fused("so3_linear_1", weights, prefix,
                                                           self.C, self.H, wdtype)
            self.l2_wf, self.l2_bf = self._build_so3_fused("so3_linear_2", weights, prefix,
                                                           self.H, self.C, wdtype)
            # gate expand (natural-order): sigmoid gate row (l-1) broadcasts over degree-l's 2l+1
            # coeffs. Ex [lmax*H, (nsph-1)*H] 0/1 selector -> one matmul (fwd) + transpose (bw),
            # replacing the per-degree 3D [N,2l+1,H] slices (coeff tile-pad).
            H = self.H
            ex = torch.zeros(lmax * H, (self.nsph - 1) * H)
            c = 0
            for l in range(1, lmax + 1):
                for _ in range(2 * l + 1):
                    ex[(l - 1) * H:l * H, c * H:(c + 1) * H] = torch.eye(H)
                    c += 1
            self.gate_exp_w = ttnn.from_torch(ex.contiguous(), dtype=ttnn.bfloat16,
                                              layout=ttnn.TILE_LAYOUT, device=device)

    def _build_so3_fused(self, name, weights, prefix, cin, cout, wdtype):
        """Block-diagonal weight [nsph*cin, nsph*cout]: coeff c (degree l(c)) uses W_l. Bias on
        coeff 0 (l=0) only."""
        W = weights[f"{prefix}.{name}.weight"]           # [lmax+1, out, in]
        bias = weights[f"{prefix}.{name}.bias"].float()  # [cout]
        deg = []                                          # degree of each coefficient
        for l in range(self.lmax + 1):
            deg += [l] * (2 * l + 1)
        Wbd = torch.zeros(self.nsph * cin, self.nsph * cout, dtype=torch.float32)
        for c, l in enumerate(deg):
            Wbd[c * cin:(c + 1) * cin, c * cout:(c + 1) * cout] = W[l].T.float()   # [in,out]
        bbd = torch.zeros(self.nsph * cout, dtype=torch.float32)
        bbd[:cout] = bias                                 # l=0 == coeff 0
        return _to_dev(Wbd.contiguous(), self.device, wdtype), _to_dev(bbd.contiguous(), self.device, wdtype)

    def _so3_linear_fused(self, x, wf, bf, cout):
        """x [N, nsph, cin] -> [N, nsph, cout] via one flat 2D matmul on the block-diagonal weight."""
        ttnn = self.ttnn
        N, cin = x.shape[0], x.shape[2]
        xf = ttnn.reshape(x, (N, self.nsph * cin))
        out = ttnn.linear(xf, wf, bias=bf, compute_kernel_config=self.kcfg)
        return ttnn.reshape(out, (N, self.nsph, cout))

    def _so3_linear(self, x, w_blocks, bias):
        """Per-degree SO3_Linear: x [N, nsph, cin] -> [N, nsph, cout]. One shared GEMM per l,
        bias added on l=0 only."""
        ttnn = self.ttnn
        N, cin = x.shape[0], x.shape[2]
        outs, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            xb = ttnn.slice(x, [0, start, 0], [N, start + n, cin])     # [N, n, cin]
            ob = ttnn.matmul(xb, w_blocks[l], compute_kernel_config=self.kcfg)   # [N, n, cout]
            if l == 0:
                ob = ttnn.add(ob, bias)                                # bias on l=0 only
            outs.append(ob)
            start += n
        return ttnn.concat(outs, dim=1)

    def _gate(self, gating, x):
        """GateActivation (m_prime=False): SiLU on l=0, sigmoid-gate per degree on l>=1.
        gating: flat [N, lmax*H]; x: [N, nsph, H]."""
        ttnn = self.ttnn
        N, H = x.shape[0], self.H
        sg = ttnn.sigmoid(gating)                                      # [N, lmax*H]
        scalar = ttnn.silu(ttnn.slice(x, [0, 0, 0], [N, 1, H]))        # l=0
        parts, start = [scalar], 1
        for l in range(1, self.lmax + 1):
            n = 2 * l + 1
            xb = ttnn.slice(x, [0, start, 0], [N, start + n, H])       # [N, n, H]
            gl = ttnn.slice(sg, [0, (l - 1) * H], [N, l * H])          # [N, H] for degree l
            gl = ttnn.reshape(gl, (N, 1, H))
            parts.append(ttnn.multiply(xb, gl))                        # broadcast over the n coeffs
            start += n
        return ttnn.concat(parts, dim=1)

    def __call__(self, x):
        """x: ttnn ``[N, nsph, C]`` -> ``[N, nsph, C]``."""
        ttnn = self.ttnn
        N = x.shape[0]
        if self.gate_exp_w is not None:
            return self._call_flat(x)
        scalar = ttnn.reshape(ttnn.slice(x, [0, 0, 0], [N, 1, self.C]), (N, self.C))
        a_scalar = ttnn.linear(scalar, self.smlp_w, bias=self.smlp_b,
                               compute_kernel_config=self.kcfg)        # [N, lmax*H] pre-SiLU
        gating = ttnn.silu(a_scalar)
        if self.l1_wf is not None:
            h = self._so3_linear_fused(x, self.l1_wf, self.l1_bf, self.H)
        else:
            h = self._so3_linear(x, self.l1_w, self.l1_b)              # [N, nsph, H]
        # cached for the analytic-force VJP (gating is post-SiLU = sigmoid input)
        self._cache_x, self._cache_a_scalar = x, a_scalar
        self._cache_gating, self._cache_h = gating, h
        g = self._gate(gating, h)                                     # [N, nsph, H]
        if self.l2_wf is not None:
            return self._so3_linear_fused(g, self.l2_wf, self.l2_bf, self.C)
        return self._so3_linear(g, self.l2_w, self.l2_b)              # [N, nsph, C]

    def _call_flat(self, x):
        """Fully-flat SpectralAtomwise ([N, nsph*C]) -- so3_linears are block-diagonal flat matmuls
        and the per-degree gate expands via one 0/1 matmul (no 3D coeff tile-pad). Bit-compatible.
        Caches flat h and the gating for the flat backward (spectral_bw)."""
        ttnn = self.ttnn
        N, C, H, nsph = x.shape[0], self.C, self.H, self.nsph
        xf = ttnn.reshape(x, (N, nsph * C))
        scalar = ttnn.slice(xf, [0, 0], [N, C])                       # l=0 block
        a_scalar = ttnn.linear(scalar, self.smlp_w, bias=self.smlp_b, compute_kernel_config=self.kcfg)
        gating = ttnn.silu(a_scalar)                                  # [N, lmax*H]
        h = ttnn.linear(xf, self.l1_wf, bias=self.l1_bf, compute_kernel_config=self.kcfg)  # [N, nsph*H]
        self._cache_xf, self._cache_a_scalar = xf, a_scalar
        self._cache_gating, self._cache_hf = gating, h
        # gate: SiLU on l=0 block, sigmoid-gate (expanded per degree) on the vector blocks
        sg = ttnn.sigmoid(gating)
        scalar_h = ttnn.silu(ttnn.slice(h, [0, 0], [N, H]))
        gate_exp = ttnn.matmul(sg, self.gate_exp_w, compute_kernel_config=self.kcfg)  # [N,(nsph-1)*H]
        vec = ttnn.multiply(ttnn.slice(h, [0, H], [N, nsph * H]), gate_exp)
        g = ttnn.concat([scalar_h, vec], dim=1)                       # [N, nsph*H]
        out = ttnn.linear(g, self.l2_wf, bias=self.l2_bf, compute_kernel_config=self.kcfg)
        return ttnn.reshape(out, (N, nsph, C))
