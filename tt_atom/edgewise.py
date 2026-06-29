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
        """x: ttnn ``[N, nsph, C]``; ``graph`` carries the on-device geometric terms.

        Runs flat ``[E, nsph*C]`` end to end: gather (row select) -> per-coordinate concat of
        source|target -> rotate to edge frame (sparse MAC) -> SO(2) conv x2 + gate -> radial
        envelope -> rotate back -> scatter-add to targets (one-hot matmul)."""
        ttnn = self.ttnn
        from . import rotation
        N, nsph, C = x.shape[0], x.shape[1], self.C
        E = graph.E
        dev = self.device
        xf = ttnn.to_layout(ttnn.reshape(x, (N, nsph * C)), ttnn.ROW_MAJOR_LAYOUT)  # gather operand
        xs = ttnn.to_layout(ttnn.reshape(ttnn.embedding(graph.src_idx, xf), (E, nsph, C)), ttnn.TILE_LAYOUT)
        xt = ttnn.to_layout(ttnn.reshape(ttnn.embedding(graph.tgt_idx, xf), (E, nsph, C)), ttnn.TILE_LAYOUT)
        m_cat = ttnn.reshape(ttnn.concat([xs, xt], dim=2), (E, nsph * 2 * C))   # flat, [xs_i|xt_i] per coord

        m_rot = rotation.rotate(ttnn, m_cat, graph.rot_fwd_ij, graph.rot_fwd_coef, nsph, 2 * C, dev)
        m, gating = self.so2_1(m_rot, graph.x_edge)       # flat in/out throughout
        m = self.gate(gating, m)
        m_so2 = self.so2_2(m, graph.x_edge)               # flat [E, nsph*C]
        m_env = ttnn.multiply(m_so2, graph.edge_envelope_f)            # [E,1] broadcast
        m_back = rotation.rotate(ttnn, m_env, graph.rot_inv_ij, graph.rot_inv_coef, nsph, C, dev)
        self._cache_mcat, self._cache_mso2, self._cache_menv = m_cat, m_so2, m_env

        out = ttnn.matmul(graph.scatter, m_back, compute_kernel_config=self.kcfg)   # [N, 9C]
        return ttnn.reshape(out, (N, nsph, C))
