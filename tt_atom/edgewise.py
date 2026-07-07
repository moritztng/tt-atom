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
        from .device import bf8_edge
        N, nsph, C = x.shape[0], x.shape[1], self.C
        E = graph.E
        dev = self.device
        _b8 = bf8_edge()
        xf = ttnn.to_layout(ttnn.reshape(x, (N, nsph * C)), ttnn.ROW_MAJOR_LAYOUT)  # gather operand
        # Keep the src/tgt gathers ROW_MAJOR: the interleave (concat dim=2) + flatten to [E, nsph*2C]
        # is done entirely in ROW_MAJOR (contiguous, no coeff-dim tile padding) with a single
        # to_layout TILE at the end -- avoids the ~18 ms TILE 3D->2D repack of the 9-coeff dim.
        xs = ttnn.reshape(ttnn.embedding(graph.src_idx, xf), (E, nsph, C))   # RM [E, nsph, C]
        xt = ttnn.reshape(ttnn.embedding(graph.tgt_idx, xf), (E, nsph, C))   # RM
        m_cat = ttnn.to_layout(ttnn.reshape(ttnn.concat([xs, xt], dim=2), (E, nsph * 2 * C)),
                               ttnn.TILE_LAYOUT)   # flat [xs_i|xt_i] per coord
        # bf8-edge boundary: from here the whole E-sized edge flow (rotate->so2->gate->so2->
        # rotate_back) runs bf8. bf8 can't be ROW_MAJOR so the cast lands after the RM gather. The
        # device replay is DRAM-bandwidth bound on these [E,nsph*C] activations; bf8 halves traffic.
        if _b8:
            m_cat = ttnn.typecast(m_cat, ttnn.bfloat8_b)

        # rotate node SH (nsph) into the reduced m-space (nred); the SO(2) pipeline runs there
        m_rot = rotation.rotate(ttnn, m_cat, graph.rot_fwd_ij, graph.rot_fwd_coef, nsph, 2 * C,
                                dev, n_out=graph.nred)
        m, gating = self.so2_1(m_rot, graph.x_edge)       # flat in/out throughout (reduced m-space)
        m = self.gate(gating, m)
        m_so2 = self.so2_2(m, graph.x_edge)               # flat [E, nred*C]
        m_env = ttnn.multiply(m_so2, graph.edge_envelope_f)            # [E,1] broadcast
        # rotate the reduced m-space message back to node SH (nsph)
        m_back = rotation.rotate(ttnn, m_env, graph.rot_inv_ij, graph.rot_inv_coef, graph.nred, C,
                                 dev, n_out=nsph)
        self._cache_mcat, self._cache_mso2, self._cache_menv = m_cat, m_so2, m_env

        # scatter-add messages onto target nodes: dense one-hot matmul (small N) or linear O(E)
        # gather+reduce (large N). See GraphContext.linear_scatter / tt_atom/scatter.py.
        odt = ttnn.bfloat16 if bf8_edge() else None
        if graph.linear_scatter:
            from . import scatter
            out = scatter.segment_sum(ttnn, m_back, graph.tgt_gather, graph.Dmax_t, N, nsph * C)
        else:
            # scatter (bf16 one-hot) @ m_back (bf8) -> bf16 node features (back on the bf16 stream)
            out = ttnn.matmul(graph.scatter, m_back, dtype=odt, compute_kernel_config=self.kcfg)
        return ttnn.reshape(out, (N, nsph, C))
