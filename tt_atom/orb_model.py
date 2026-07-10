"""Orb-v3 backbone (attention-augmented MPNN), device-resident on Tenstorrent.

Orb is explicitly NON-equivariant (see ``docs/orb-port.md``): no SO(3)/E(3) hidden
representations, no per-edge Wigner rotation. Spherical harmonics are used only once, as a
fixed per-edge scalar descriptor (like the Bessel RBF), concatenated into ordinary latent
vectors. The whole network -- encoder, message passing, decoder -- therefore operates on
plain ``[rows, latent_dim]`` tensors: Linear/SiLU/RMSNorm/segment-sum, no SO(2) convolution,
no equivariant gate, no Wigner rotate. None of ``tt_atom``'s equivariant kernels
(``fused_rotate``, ``fused_gate``, ``fused_ln_bw``, ``so2.py``, ``rotation.py``, ``norm.py``'s
``RMSNormSH``, ``grid.py``/``spectral.py``) apply here; this module reuses only the
architecture-agnostic infra (``device.py``'s compute-kernel policy, ``scatter.py``'s linear
edge->node segment-sum).

The fixed, non-learned per-edge terms (Bessel RBF, spherical-harmonic angular embedding, the
polynomial distance-cutoff envelope used to gate attention) are computed on host exactly like
UMA's wigner/gaussian/envelope buffers, and uploaded once as part of the graph context -- they
are <1% of the compute. Every learned op (the two encoder MLPs, and each interaction layer's
edge/node MLPs + attention gates) runs on device.

Reference: ``orb_models.forcefield.gns.py`` (``Encoder`` / ``AttentionInteractionNetwork``).
"""
from __future__ import annotations

import torch

from .device import compute_kernel_config


def host_cutoff(r: torch.Tensor, r_max: float = 6.0) -> torch.Tensor:
    """Fixed polynomial attention-cutoff envelope (``orb_models...nn_util.get_cutoff``).

    A closed-form function of edge length only (no learned parameters) -- computed on host and
    uploaded as a constant, exactly like UMA's envelope/wigner buffers.
    """
    p = 4
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r / r_max, p)
        + p * (p + 2.0) * torch.pow(r / r_max, p + 1)
        - (p * (p + 1.0) / 2) * torch.pow(r / r_max, p + 2)
    )
    return (envelope * (r < r_max)).unsqueeze(-1)


def _to_dev(t, device, dtype, layout=None):
    import ttnn

    layout = layout or ttnn.TILE_LAYOUT
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


class RMSNorm:
    """Plain ``torch.nn.RMSNorm`` (elementwise-affine, no bias) over the last dim.

    Unlike ``tt_atom.norm.RMSNormSH``, there is no spherical-harmonic degree structure to
    balance -- this is the ordinary transformer-style RMSNorm.
    """

    def __init__(self, weights, prefix, device, dim, eps=1e-6):
        import ttnn

        self.ttnn = ttnn
        self.eps = eps
        self.dim = dim
        self.w = _to_dev(weights[f"{prefix}.weight"].view(1, dim).contiguous(), device, ttnn.bfloat16)

    def __call__(self, x):
        ttnn = self.ttnn
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ms, self.eps))
        return ttnn.multiply(ttnn.multiply(x, inv), self.w)


class MLPNorm:
    """``orb_models...nn_util.mlp_and_layer_norm`` with ``num_mlp_layers=2``: 3 Linears
    (in->hidden->hidden->out, SiLU after the first two, none after the third) + RMSNorm.
    Weight keys ``{prefix}.mlp.NN-{0,1,2}.{weight,bias}`` + ``{prefix}.layer_norm.weight``.
    """

    def __init__(self, weights, prefix, device, in_dim, hidden_dim, out_dim):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.w = []
        self.b = []
        for i in range(3):
            self.w.append(_to_dev(weights[f"{prefix}.mlp.NN-{i}.weight"].T.contiguous(),
                                  device, ttnn.bfloat16))
            self.b.append(_to_dev(weights[f"{prefix}.mlp.NN-{i}.bias"], device, ttnn.bfloat16))
        self.norm = RMSNorm(weights, f"{prefix}.layer_norm", device, out_dim)

    def __call__(self, x):
        ttnn = self.ttnn
        h = ttnn.silu(ttnn.linear(x, self.w[0], bias=self.b[0], compute_kernel_config=self.kcfg))
        h = ttnn.silu(ttnn.linear(h, self.w[1], bias=self.b[1], compute_kernel_config=self.kcfg))
        h = ttnn.linear(h, self.w[2], bias=self.b[2], compute_kernel_config=self.kcfg)
        return self.norm(h)


class Encoder:
    """``gns.Encoder``: separate node/edge MLPNorm blocks, no interaction between them."""

    def __init__(self, weights, device, *, node_in, edge_in, latent_dim, hidden_dim):
        self.node_fn = MLPNorm(weights, "_encoder._node_fn", device, node_in, hidden_dim, latent_dim)
        self.edge_fn = MLPNorm(weights, "_encoder._edge_fn", device, edge_in, hidden_dim, latent_dim)

    def __call__(self, node_features, edge_features):
        return self.node_fn(node_features), self.edge_fn(edge_features)


class AttentionInteractionLayer:
    """``gns.AttentionInteractionNetwork`` for the omat config: no conditioning, sigmoid
    attention gate (not softmax), distance-cutoff-scaled attention. One message-passing step.
    """

    def __init__(self, weights, prefix, device, *, latent_dim, hidden_dim):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.C = latent_dim
        self.edge_mlp = MLPNorm(weights, f"{prefix}._edge_mlp", device, 3 * latent_dim, hidden_dim, latent_dim)
        self.node_mlp = MLPNorm(weights, f"{prefix}._node_mlp", device, 3 * latent_dim, hidden_dim, latent_dim)
        self.receive_attn_w = _to_dev(weights[f"{prefix}._receive_attn.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.receive_attn_b = _to_dev(weights[f"{prefix}._receive_attn.bias"], device, ttnn.bfloat16)
        self.send_attn_w = _to_dev(weights[f"{prefix}._send_attn.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.send_attn_b = _to_dev(weights[f"{prefix}._send_attn.bias"], device, ttnn.bfloat16)

    def __call__(self, nodes, edges, graph):
        """``graph`` supplies ``senders``/``receivers`` gather tables (ttnn embedding-ready,
        [E] row-major uint32) and ``cutoff`` ([E,1] tile, host-precomputed envelope), plus the
        scatter-add gather tables for the node update (``tt_atom.scatter``)."""
        ttnn = self.ttnn
        from . import scatter as _sc

        E, N, C = graph.E, graph.N, self.C

        receive_attn = ttnn.sigmoid(ttnn.linear(edges, self.receive_attn_w, bias=self.receive_attn_b,
                                                compute_kernel_config=self.kcfg))
        send_attn = ttnn.sigmoid(ttnn.linear(edges, self.send_attn_w, bias=self.send_attn_b,
                                             compute_kernel_config=self.kcfg))
        receive_attn = ttnn.multiply(receive_attn, graph.cutoff)
        send_attn = ttnn.multiply(send_attn, graph.cutoff)

        nodes_rm = ttnn.to_layout(nodes, ttnn.ROW_MAJOR_LAYOUT)
        sent_attrs = ttnn.to_layout(ttnn.embedding(graph.senders_idx, nodes_rm), ttnn.TILE_LAYOUT)
        recv_attrs = ttnn.to_layout(ttnn.embedding(graph.receivers_idx, nodes_rm), ttnn.TILE_LAYOUT)
        edge_in = ttnn.concat([edges, sent_attrs, recv_attrs], dim=1)
        updated_edges = self.edge_mlp(edge_in)

        sent_msg = ttnn.multiply(updated_edges, send_attn)
        recv_msg = ttnn.multiply(updated_edges, receive_attn)
        sent_agg = _sc.segment_sum(ttnn, sent_msg, graph.src_gather, graph.Dmax_s, N, C)
        recv_agg = _sc.segment_sum(ttnn, recv_msg, graph.tgt_gather, graph.Dmax_t, N, C)

        node_in = ttnn.concat([nodes, recv_agg, sent_agg], dim=1)
        updated_nodes = self.node_mlp(node_in)

        return ttnn.add(nodes, updated_nodes), ttnn.add(edges, updated_edges)


class OrbGraphContext:
    """Host-precomputed, device-resident geometric terms for one fixed topology (mirrors
    ``tt_atom.model.GraphContext``, but for Orb's plain scatter -- no Wigner rotation buffers).
    """

    def __init__(self, device, *, senders, receivers, cutoff, num_nodes):
        import ttnn
        from . import scatter as _sc

        E = senders.shape[0]
        self.E, self.N = E, num_nodes
        self.senders_idx = _to_dev(senders.to(torch.int32), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.receivers_idx = _to_dev(receivers.to(torch.int32), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.cutoff = _to_dev(cutoff, device, ttnn.bfloat16)

        tgt_g, self.Dmax_t = _sc.build_gather(receivers, num_nodes, E)
        src_g, self.Dmax_s = _sc.build_gather(senders, num_nodes, E)
        self.tgt_gather = _to_dev(torch.from_numpy(tgt_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.src_gather = _to_dev(torch.from_numpy(src_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
