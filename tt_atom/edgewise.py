"""Edgewise message passing — the SO(2) message block of eSCN-MD.

For every edge: gather source/target node features, rotate into the edge frame with the
host Wigner matrix, run the two SO(2) convolutions with a gate in between, apply the radial
envelope, rotate back, and scatter-add the messages onto target nodes. Gathers are row
selects (``ttnn.embedding``); the scatter-add is a fixed one-hot matmul ``S @ messages``
(``S[n, e] = 1`` iff edge ``e`` targets node ``n``) — its transpose is exactly the gather,
which is what the analytic-force backward needs.

Reference: ``fairchem ... escn_md_block.py:Edgewise.forward_chunk``.
"""
from __future__ import annotations

from .device import compute_kernel_config
from .so2 import SO2Convolution
from .activation import GateActivation


class Edgewise:
    def __init__(self, weights, prefix, device, *, sphere_channels, hidden_channels,
                 lmax, mmax, fast=False):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.C = sphere_channels
        self.kcfg = compute_kernel_config()
        extra = lmax * hidden_channels                # gate scalar channels
        self.so2_1 = SO2Convolution(
            weights, f"{prefix}.so2_conv_1", device,
            sphere_channels_in=2 * sphere_channels, m_output_channels=hidden_channels,
            lmax=lmax, mmax=mmax, extra_m0_output_channels=extra, fast=fast)
        self.gate = GateActivation(device, lmax=lmax, mmax=mmax, num_channels=hidden_channels)
        self.so2_2 = SO2Convolution(
            weights, f"{prefix}.so2_conv_2", device,
            sphere_channels_in=hidden_channels, m_output_channels=sphere_channels,
            lmax=lmax, mmax=mmax, extra_m0_output_channels=0, fast=fast)

    def __call__(self, x, graph):
        """x: ttnn ``[N, nsph, C]``; ``graph`` carries the on-device geometric terms."""
        ttnn = self.ttnn
        N, nsph = x.shape[0], x.shape[1]
        xf = ttnn.reshape(x, (N, nsph * self.C))          # [N, 9C] row-major gather operand
        xf = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
        xs = ttnn.reshape(ttnn.embedding(graph.src_idx, xf), (graph.E, nsph, self.C))
        xt = ttnn.reshape(ttnn.embedding(graph.tgt_idx, xf), (graph.E, nsph, self.C))
        xs = ttnn.to_layout(xs, ttnn.TILE_LAYOUT)
        xt = ttnn.to_layout(xt, ttnn.TILE_LAYOUT)

        m_cat = ttnn.concat([xs, xt], dim=2)              # [E, 9, 2C]
        m = ttnn.matmul(graph.wigner, m_cat, compute_kernel_config=self.kcfg)  # rotate to edge frame
        m, gating = self.so2_1(m, graph.x_edge)
        m = self.gate(gating, m)
        m = self.so2_2(m, graph.x_edge)
        m_so2 = m
        m = ttnn.multiply(m, graph.edge_envelope)         # [E,1,1] broadcast
        m_env = m
        m = ttnn.matmul(graph.wigner_inv, m, compute_kernel_config=self.kcfg)  # rotate back
        self._cache_mcat, self._cache_mso2, self._cache_menv = m_cat, m_so2, m_env

        # scatter-add to target nodes:  out[N,9,C] = S[N,E] @ m[E,9C]
        mf = ttnn.reshape(m, (graph.E, nsph * self.C))
        out = ttnn.matmul(graph.scatter, mf, compute_kernel_config=self.kcfg)
        return ttnn.reshape(out, (N, nsph, self.C))
