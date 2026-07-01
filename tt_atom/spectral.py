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

from .device import compute_kernel_config


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
        scalar = ttnn.reshape(ttnn.slice(x, [0, 0, 0], [N, 1, self.C]), (N, self.C))
        a_scalar = ttnn.linear(scalar, self.smlp_w, bias=self.smlp_b,
                               compute_kernel_config=self.kcfg)        # [N, lmax*H] pre-SiLU
        gating = ttnn.silu(a_scalar)
        h = self._so3_linear(x, self.l1_w, self.l1_b)                  # [N, nsph, H]
        # cached for the analytic-force VJP (gating is post-SiLU = sigmoid input)
        self._cache_x, self._cache_a_scalar = x, a_scalar
        self._cache_gating, self._cache_h = gating, h
        g = self._gate(gating, h)                                     # [N, nsph, H]
        return self._so3_linear(g, self.l2_w, self.l2_b)              # [N, nsph, C]
