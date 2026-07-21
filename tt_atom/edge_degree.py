"""On-device edge-degree embedding — the node initialisation, moved off host.

The eSCN ``edge_degree_embedding`` builds the initial node features from the invariant edge
embedding: a per-edge radial MLP produces the m=0 block, it is rotated back into node spherical
harmonics with the (inverse) Wigner matrix, scaled by the radial envelope, scatter-added onto
target nodes, and the constant l=0 init (sphere embedding + system embedding) added on top.

Historically this ran on host in ``geometry.HostGeometry`` (its autograd supplied ``dx_init/dpos``
for the analytic force). At N=1000 the radial-MLP forward+backward over E~46k edges was the single
largest per-step host cost (~140 ms). Structurally it is identical to a message-passing layer, so
we run it on device inside the captured trace instead — a few small GEMMs + the fused rotation +
the linear scatter, all machinery the backbone already owns. The host then only differentiates the
cheap geometric terms (wigner_inv, x_edge, envelope), whose adjoints this module's backward
(:func:`edge_degree_bw`) accumulates into the same ``acc`` dict the block backward fills.

Forward mirrors ``geometry.HostGeometry.__call__``'s node-init block; backward mirrors it exactly
in reverse (cf. ``forces.edgewise_bw``). Gated by ``device.device_ede()`` (``TT_ATOM_DEVICE_EDE=1``).
"""
from __future__ import annotations

from .device import compute_kernel_config
from .so2 import RadialMLP


class EdgeDegreeEmbedding:
    """radial MLP -> pad -> rotate-back -> envelope -> scatter/rescale -> + l0, on device."""

    def __init__(self, weights, device, cfg, *, rescale):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.C = cfg["sphere_channels"]
        self.lmax = cfg["lmax"]
        self.m0 = cfg.get("mmax_m0_coeffs", self.lmax + 1)
        self.rescale = float(rescale)
        # fold the 1/rescale node-init scale into the radial MLP's final linear (fp32 -> bf16), so it
        # lands in the matmul's fp32 accumulation instead of a lossy bf16 multiply by 0.2 downstream.
        self.rad = RadialMLP(weights, "edge_degree_embedding.rad_func", device, ttnn.bfloat16,
                             out_scale=1.0 / self.rescale)
        self.kcfg = compute_kernel_config()

    def __call__(self, graph, l0):
        """``x_init`` [N, nsph, C] from ``graph.x_edge`` + rot_inv + envelope and the constant l0."""
        ttnn = self.ttnn
        from . import rotation, scatter

        C = self.C
        nred, nsph, N = graph.nred, graph.nsph, graph.N
        edm = self.rad(graph.x_edge)                            # [E, m0*C]
        # place the m=0 block at the front of the reduced m-space (zeros in the tail coeffs): the
        # flat layout is coeff-major, so padding columns m0*C -> nred*C is exactly F.pad(...,(0,0,0,
        # nred-m0)) on [E, m0, C]. When nred==m0 (rare) this is a no-op.
        if nred > self.m0:
            edm = ttnn.pad(edm, [(0, 0), (0, (nred - self.m0) * C)], value=0.0)
        self._cache_edm = edm                                   # rotate input (for the VJP)
        m_back = rotation.rotate(ttnn, edm, graph.rot_inv_ij, graph.rot_inv_coef, nred, C,
                                 self.device, n_out=nsph)        # [E, nsph*C]
        m_env = ttnn.multiply(m_back, graph.edge_envelope_f)     # [E, nsph*C] * [E,1] broadcast
        self._cache_mback = m_back
        if graph.linear_scatter:
            node = scatter.segment_sum(ttnn, m_env, graph.tgt_gather, graph.Dmax_t, N, nsph * C)
        else:
            node = ttnn.matmul(graph.scatter, m_env, compute_kernel_config=self.kcfg)  # [N, nsph*C]
        node = ttnn.reshape(node, (N, nsph, C))   # 1/rescale already folded into the radial MLP
        return ttnn.add(node, l0)


def edge_degree_bw(ede, graph, g_x_init, acc):
    """VJP of :class:`EdgeDegreeEmbedding`. ``g_x_init`` [N,nsph,C] is the node adjoint after all
    blocks; l0 is constant (pass-through). Accumulates the geometric adjoints (g rot_inv,
    g_envelope) into ``acc`` and appends the radial adjoint so ``backbone_bw`` finishes g_x_edge."""
    ttnn = ede.ttnn
    from . import rotation

    C = ede.C
    N, nsph = g_x_init.shape[0], g_x_init.shape[1]
    E, nred = graph.E, graph.nred
    # x_init = node + l0 (1/rescale folded into the radial MLP); node = scatter(m_env) -> gather
    # the node adjoint back to edges.
    gnf = ttnn.to_layout(ttnn.reshape(g_x_init, (N, nsph * C)), ttnn.ROW_MAJOR_LAYOUT)
    g_menv = ttnn.to_layout(ttnn.embedding(graph.tgt_idx, gnf), ttnn.TILE_LAYOUT)   # [E, nsph*C]
    # m_env = m_back * envelope
    g_mback = ttnn.multiply(g_menv, graph.edge_envelope_f)
    g_env = ttnn.sum(ttnn.multiply(g_menv, ede._cache_mback), dim=1, keepdim=True)  # [E,1]
    # inverse rotation backward (reduced m-space nred -> node SH nsph)
    g_edm, g_rinv = rotation.rotate_bw(ttnn, ede._cache_edm, g_mback, graph.rot_inv_ij,
                                       graph.rot_inv_coef, nred, C, ede.device, n_out=nsph)
    if nred > ede.m0:
        g_edm = ttnn.slice(g_edm, [0, 0], [E, ede.m0 * C])      # unpad -> the m=0 block
    acc["rot_inv"] = g_rinv if acc["rot_inv"] is None else ttnn.add(acc["rot_inv"], g_rinv)
    acc["envelope"] = g_env if acc["envelope"] is None else ttnn.add(acc["envelope"], g_env)
    acc["g_rad"].append((ede, g_edm))    # backbone_bw does ede.rad.bw(g_edm) -> g_x_edge
