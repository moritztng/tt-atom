"""Analytic forces for ``orb-v3-conservative-inf-omat`` -- reverse-mode VJP through the device
backbone (``F = -dE/dpos``).

Orb is non-equivariant (see ``docs/orb-port.md``): the whole network is plain
Linear/RMSNorm/SiLU/sigmoid/segment-sum, so unlike ``tt_atom/forces.py``'s equivariant UMA
backward there is no rotation adjoint and no local-frame chain rule. Each VJP below mirrors a
forward op in ``tt_atom/orb_model.py`` 1:1: a Linear's backward is the transpose-matmul, RMSNorm's
is the ordinary (non-SH) formula, SiLU/sigmoid go through ttnn's fused ``*_bw`` ops, and
``scatter.segment_sum``'s adjoint is exactly a gather by the *same* sender/receiver index array
used to build its own forward gather table (scatter-add and gather are transposes of each other
over the same index) -- so the backward reuses the graph's existing gather tables and
``senders_idx``/``receivers_idx`` with no new device buffers.

The device VJP produces the adjoint at the two pos-dependent device inputs -- the uploaded edge
feature tensor and the attention cutoff envelope; host ``torch.autograd`` finishes the cheap
``d(edge_feat, cutoff)/dpos`` through ``tt_atom/orb_geometry.py``'s differentiable RBF +
spherical-harmonic + cutoff map (<1% of the compute, exactly like UMA's gaussian/wigner/envelope
host finish in ``tt_atom/forces.py``).
"""
from __future__ import annotations

import torch

from .device import orb_fused_silu_bw


def _mm(ttnn, g, W, kcfg):
    """grad wrt x of ``y = x @ W`` (W stored [in,out]): ``g @ W^T``."""
    return ttnn.matmul(g, W, transpose_b=True, compute_kernel_config=kcfg)


def silu_bw(ttnn, g, x):
    # Stock ttnn.silu_bw materializes sigmoid, grad*sigmoid, rsub, multiply,
    # add, and the final multiply as six DRAM-backed ops.  The source-ttnn
    # build used by TT-Atom already carries fused_gate, whose backward mode
    # computes g*silu'(x) in one device program.  fused_gate reserves one
    # tile for its SO(3) vector branch, so use it for all but the final tile
    # and finish that tile with stock ttnn.  Small/node tensors stay on the
    # stock path because the slice/concat wrapper costs more than it saves.
    rows, width = x.shape[-2], x.shape[-1]

    can_fuse = (
        orb_fused_silu_bw()
        and len(x.shape) == 2
        and rows >= 2048
        and width >= 64
        and width % 32 == 0
        and g.dtype == x.dtype
        and x.dtype in (ttnn.bfloat16, ttnn.bfloat8_b)
    )
    if can_fuse:
        wt = width // 32
        prefix_width = width - 32
        # The one-tile gate is consumed only by the ignored final output tile.
        dummy_gate = ttnn.slice(g, [0, 0], [rows, 32])
        fused = ttnn._ttnn.operations.experimental.fused_gate(
            g, dummy_gate, x, wt, 1, wt - 1, 1)
        tail = ttnn.silu_bw(
            ttnn.slice(g, [0, prefix_width], [rows, width]),
            ttnn.slice(x, [0, prefix_width], [rows, width]),
        )[0]
        return ttnn.concat(
            [ttnn.slice(fused, [0, 0], [rows, prefix_width]), tail], dim=1)
    return ttnn.silu_bw(g, x)[0]


def rmsnorm_bw(norm, g_out):
    """VJP of ``RMSNorm`` (plain, over the last dim -- no spherical-harmonic degree structure,
    unlike ``tt_atom.norm.RMSNormSH``). ``y = x * inv * w``, ``inv = rsqrt(mean_C(x^2) + eps)``:

        g_x = inv*w*g_out - (inv^3/C) * x * sum_C(g_out * w * x)
    """
    ttnn = norm.ttnn
    x, inv, w, C = norm._cache_x, norm._cache_inv, norm.w, norm.dim
    gw = ttnn.multiply(g_out, w)
    s = ttnn.sum(ttnn.multiply(gw, x), dim=-1, keepdim=True)             # [rows,1]
    inv3 = ttnn.multiply(ttnn.multiply(inv, inv), inv)
    term2 = ttnn.multiply(ttnn.multiply(inv3, x), ttnn.multiply(s, 1.0 / C))
    return ttnn.subtract(ttnn.multiply(gw, inv), term2)


def mlpnorm_bw(mlp, g_out):
    """VJP of ``MLPNorm``: 3 Linears (SiLU, SiLU, none) + a trailing ``RMSNorm``."""
    ttnn = mlp.ttnn
    kcfg = mlp.kcfg
    g_h2 = rmsnorm_bw(mlp.norm, g_out)
    g_h1 = _mm(ttnn, g_h2, mlp.w[2], kcfg)
    g_a1 = silu_bw(ttnn, g_h1, mlp._cache_a1)
    g_h0 = _mm(ttnn, g_a1, mlp.w[1], kcfg)
    g_a0 = silu_bw(ttnn, g_h0, mlp._cache_a0)
    return _mm(ttnn, g_a0, mlp.w[0], kcfg)


def attn_layer_bw(layer, graph, g_nodes_out, g_edges_out):
    """VJP of ``AttentionInteractionLayer``. Returns ``(g_nodes_in, g_edges_in, g_cutoff)``,
    the adjoints at this layer's three pos-dependent-eventually inputs (``nodes``, ``edges`` fed
    in, and the shared ``graph.cutoff`` envelope this layer's attention gates consume)."""
    ttnn = layer.ttnn
    from . import scatter as _sc

    c = layer._cache
    C, N, E = layer.C, graph.N, graph.E

    # new_nodes = nodes + updated_nodes ; new_edges = edges + updated_edges: the residual pass-
    # through is a SEPARATE occurrence of nodes/edges from the one inside node_in/edge_in (both
    # feed the loss), so both contributions must be summed -- not just the indirect (MLP) one.
    g_node_in = mlpnorm_bw(layer.node_mlp, g_nodes_out)              # [N, 3C]
    g_nodes_direct = ttnn.slice(g_node_in, [0, 0], [N, C])
    g_recv_agg = ttnn.slice(g_node_in, [0, C], [N, 2 * C])
    g_sent_agg = ttnn.slice(g_node_in, [0, 2 * C], [N, 3 * C])

    # segment_sum adjoint = gather by the SAME index array used to build its own gather table
    g_recv_msg = ttnn.to_layout(
        ttnn.embedding(graph.receivers_idx, ttnn.to_layout(g_recv_agg, ttnn.ROW_MAJOR_LAYOUT)),
        ttnn.TILE_LAYOUT)
    g_sent_msg = ttnn.to_layout(
        ttnn.embedding(graph.senders_idx, ttnn.to_layout(g_sent_agg, ttnn.ROW_MAJOR_LAYOUT)),
        ttnn.TILE_LAYOUT)

    # recv_msg = updated_edges * receive_attn ; sent_msg = updated_edges * send_attn
    g_updated_edges = ttnn.add(g_edges_out, ttnn.add(
        ttnn.multiply(g_recv_msg, c["receive_attn"]), ttnn.multiply(g_sent_msg, c["send_attn"])))
    g_receive_attn = ttnn.sum(ttnn.multiply(g_recv_msg, c["updated_edges"]), dim=1, keepdim=True)
    g_send_attn = ttnn.sum(ttnn.multiply(g_sent_msg, c["updated_edges"]), dim=1, keepdim=True)

    g_edge_in = mlpnorm_bw(layer.edge_mlp, g_updated_edges)          # [E, 3C]
    g_edges_direct = ttnn.slice(g_edge_in, [0, 0], [E, C])
    g_sent_attrs = ttnn.slice(g_edge_in, [0, C], [E, 2 * C])
    g_recv_attrs = ttnn.slice(g_edge_in, [0, 2 * C], [E, 3 * C])

    # embedding-gather adjoint = segment_sum by the SAME index array (transpose pair)
    g_nodes_from_sent = _sc.segment_sum(ttnn, g_sent_attrs, graph.src_gather, graph.Dmax_s, N, C)
    g_nodes_from_recv = _sc.segment_sum(ttnn, g_recv_attrs, graph.tgt_gather, graph.Dmax_t, N, C)

    # receive_attn = sigmoid(ra_lin) * cutoff ; send_attn = sigmoid(sa_lin) * cutoff
    g_ra_sig = ttnn.multiply(g_receive_attn, graph.cutoff)
    g_sa_sig = ttnn.multiply(g_send_attn, graph.cutoff)
    g_cutoff = ttnn.add(ttnn.multiply(g_receive_attn, c["ra_sig"]),
                        ttnn.multiply(g_send_attn, c["sa_sig"]))
    g_ra_lin = ttnn.sigmoid_bw(g_ra_sig, c["ra_lin"])[0]
    g_sa_lin = ttnn.sigmoid_bw(g_sa_sig, c["sa_lin"])[0]
    g_edges_from_attn = ttnn.add(_mm(ttnn, g_ra_lin, layer.receive_attn_w, layer.kcfg),
                                 _mm(ttnn, g_sa_lin, layer.send_attn_w, layer.kcfg))

    g_nodes_in = ttnn.add(g_nodes_out, ttnn.add(g_nodes_direct,
                                                ttnn.add(g_nodes_from_sent, g_nodes_from_recv)))
    g_edges_in = ttnn.add(g_edges_out, ttnn.add(g_edges_direct, g_edges_from_attn))
    return g_nodes_in, g_edges_in, g_cutoff


def energy_bw(ehead):
    """VJP of ``EnergyHead``'s device MLP path (``dE/dE = 1`` seed) -> ``g wrt node_features``
    ``[N, C]`` (broadcasting the ``mean``-backward's ``1/N`` term back out to every node).

    The two constants (the seed and the ``N``-row broadcast-ones column) are created once and
    cached on ``ehead`` so that no device buffer is allocated inside a captured trace region --
    the ttnn trace machinery forbids allocations during capture (it hangs), exactly like
    ``tt_atom/forces.py``'s ``energy_bw`` caches ``bb._bw_seed``/``bb._bw_zeros`` for UMA."""
    ttnn = ehead.ttnn
    kcfg = ehead.kcfg
    N = ehead._cache_N
    if getattr(ehead, "_bw_ones11", None) is None:
        ehead._bw_ones11 = ttnn.ones((1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=ehead.device)
    if getattr(ehead, "_bw_onesN1", None) is None or tuple(ehead._bw_onesN1.shape) != (N, 1):
        ehead._bw_onesN1 = ttnn.ones((N, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=ehead.device)
    g_h = _mm(ttnn, ehead._bw_ones11, ehead.w1, kcfg)         # [1, hidden]
    g_a0 = silu_bw(ttnn, g_h, ehead._cache_a0)
    g_mean = _mm(ttnn, g_a0, ehead.w0, kcfg)                  # [1, C]
    g_mean = ttnn.multiply(g_mean, 1.0 / N)
    return ttnn.matmul(ehead._bw_onesN1, g_mean, compute_kernel_config=kcfg)  # [N, C], every row identical


def backbone_bw(encoder, layers, ehead, graph):
    """Full reverse pass: energy head -> all interaction layers (reversed) -> encoder's edge MLP.
    Returns ``(g_edge_feat, g_cutoff)``, the device adjoints at the two pos-dependent uploaded
    inputs. ``node_feat`` has no ``pos`` dependence (atomic-number embedding only), so its
    adjoint is discarded -- the encoder's node path is never differentiated further.

    The zero-seed edge gradient is cached on ``ehead`` (guarded by ``(E, C)``) for the same
    trace-capture-allocation reason as ``energy_bw``'s cached constants."""
    ttnn = ehead.ttnn

    g_nodes = energy_bw(ehead)
    E, C = graph.E, layers[0].C
    if getattr(ehead, "_bw_edges_zero", None) is None or tuple(ehead._bw_edges_zero.shape) != (E, C):
        ehead._bw_edges_zero = ttnn.zeros((E, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=ehead.device)
    g_edges = ehead._bw_edges_zero
    g_cutoff = None
    for layer in reversed(layers):
        g_nodes, g_edges, g_c = attn_layer_bw(layer, graph, g_nodes, g_edges)
        g_cutoff = g_c if g_cutoff is None else ttnn.add(g_cutoff, g_c)

    g_edge_feat = mlpnorm_bw(encoder.edge_fn, g_edges)        # encoder's edge_fn input adjoint
    return g_edge_feat, g_cutoff


def energy_bw_batch(ehead, seg_mean_T):
    """Batched analogue of ``energy_bw``: the VJP seed at the EnergyHead's node-features input for a
    disjoint-union batch of K systems, where the batched energy is ``E_total = sum_k E_k`` and
    ``E_k = ehead.batch``'s k-th output.

    ``seg_mean_T`` [Ntot, K] is the transpose of the row-normalized segment matrix
    (``seg_mean[k, n] = 1/N_k`` iff atom n in system k): ``seg_mean_T @ g_mean`` broadcasts each
    system's ``g_mean[k]`` back to its own nodes scaled by ``1/N_k`` -- exactly what single-system
    ``energy_bw`` does with ``ones[N,1] @ (g_mean * 1/N)``. The rest of the head VJP (``g_h = w1^T``,
    SiLU backward, ``g_mean = w0^T``) is the per-system-identical matmul, now over ``[K, *]`` instead
    of ``[1, *]``; the pre-SiLU activation ``a0`` is cached by ``EnergyHead.batch``. The K-row ones
    seed is cached on ``ehead`` for the same trace-capture-allocation reason as ``energy_bw``'s
    ``_bw_ones11``."""
    ttnn = ehead.ttnn
    kcfg = ehead.kcfg
    K = seg_mean_T.shape[1]
    if getattr(ehead, "_bw_onesK1", None) is None or tuple(ehead._bw_onesK1.shape) != (K, 1):
        ehead._bw_onesK1 = ttnn.ones((K, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=ehead.device)
    g_h = _mm(ttnn, ehead._bw_onesK1, ehead.w1, kcfg)        # [K, hidden]
    g_a0 = silu_bw(ttnn, g_h, ehead._cache_a0)               # [K, hidden]
    g_mean = _mm(ttnn, g_a0, ehead.w0, kcfg)                 # [K, C]
    return ttnn.matmul(seg_mean_T, g_mean, compute_kernel_config=kcfg)   # [Ntot, C]


def backbone_bw_batch(encoder, layers, ehead, graph, seg_mean_T):
    """Batched ``backbone_bw``: the energy-head seed comes from ``energy_bw_batch`` (per-system);
    every interaction-layer VJP (``attn_layer_bw``) is unchanged because its adjoint algebra only
    touches the graph's gather/scatter tables and per-edge cutoff -- all block-diagonal under a
    disjoint-union batch, so an edge's adjoint can only reach its own system's nodes. Returns
    ``(g_edge_feat, g_cutoff)`` -- the device adjoints at the two pos-dependent uploaded inputs of
    the *whole* batch -- which the host ``torch.autograd.grad`` finish turns into per-atom forces
    (block-diagonal => ``dE_total/dpos[n] = dE_{system(n)}/dpos[n]``)."""
    ttnn = ehead.ttnn

    g_nodes = energy_bw_batch(ehead, seg_mean_T)
    E, C = graph.E, layers[0].C
    if getattr(ehead, "_bw_edges_zero", None) is None or tuple(ehead._bw_edges_zero.shape) != (E, C):
        ehead._bw_edges_zero = ttnn.zeros((E, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                          device=ehead.device)
    g_edges = ehead._bw_edges_zero
    g_cutoff = None
    for layer in reversed(layers):
        g_nodes, g_edges, g_c = attn_layer_bw(layer, graph, g_nodes, g_edges)
        g_cutoff = g_c if g_cutoff is None else ttnn.add(g_cutoff, g_c)

    g_edge_feat = mlpnorm_bw(encoder.edge_fn, g_edges)
    return g_edge_feat, g_cutoff


def energy_and_forces_batch(encoder, layers, ehead, device, *, pos, senders, receivers,
                            atomic_numbers, node_feat, cell_shift, seg_mean, seg_mean_T,
                            r_max=6.0, num_bases=8, cond_nodes=None):
    """Conservative energy + analytic forces for K systems in ONE device forward+backward
    (disjoint-union batched ``orb-v3-conservative-*``). The batched analogue of
    ``energy_and_forces``: one device forward at the union geometry, ``ehead.batch`` for the
    per-system normalized energies, ``backbone_bw_batch`` for the per-system adjoints at the two
    pos-dependent uploaded inputs (``edge_feat``, ``cutoff``), and a host ``torch.autograd.grad``
    finish through the differentiable edge geometry. Block-diagonality makes
    ``dE_total/dpos[n] = dE_{system(n)}/dpos[n]``, so the concatenated forces are each atom's
    own-system force -- no per-system splitting of the backward is needed.

    ``seg_mean`` [K, Ntot] is the row-normalized segment matrix (``seg_mean[k, n] = 1/N_k`` iff
    atom n in system k) -- ``ehead.batch`` matmuls it against the node features for the per-system
    mean. ``seg_mean_T`` [Ntot, K] is its transpose -- ``energy_bw_batch`` matmuls it against
    ``g_mean`` to broadcast each system's head-adjoint back to its own nodes. Both are host tensors,
    uploaded here. ``senders``/``receivers``/``cell_shift`` carry per-system node offsets
    (block-diagonal); ``node_feat``/``cond_nodes`` are the concatenated per-system encoder inputs.
    Returns ``(energy_raw: torch.Tensor[K], forces: torch.Tensor[Ntot, 3])`` in normalized
    (pre-denormalize) space, matching ``energy_and_forces``'s convention so the caller finishes with
    the same host denormalize + ZBL add used by the single-system path."""
    import ttnn

    from .orb_geometry import host_edge_features
    from .orb_model import OrbGraphContext, _to_dev

    pos = pos.detach().clone().requires_grad_(True)
    edge_feat, cutoff, _vectors = host_edge_features(pos, senders, receivers, cell_shift,
                                                    r_max=r_max, num_bases=num_bases)
    Ntot = atomic_numbers.shape[0]
    graph = OrbGraphContext(device, senders=senders, receivers=receivers,
                            cutoff=cutoff.detach().float(), num_nodes=Ntot, cond_nodes=cond_nodes)

    node_dev = _to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev = _to_dev(edge_feat.detach().float(), device, ttnn.bfloat16)
    seg_mean_dev = _to_dev(seg_mean.float(), device, ttnn.bfloat16)
    seg_mean_T_dev = _to_dev(seg_mean_T.float(), device, ttnn.bfloat16)
    nodes, edges = encoder(node_dev, edge_dev)
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)
    raw_pred = ttnn.to_torch(ehead.batch(nodes, seg_mean_dev)).double().view(-1)

    g_edge_feat_dev, g_cutoff_dev = backbone_bw_batch(encoder, layers, ehead, graph, seg_mean_T_dev)
    g_edge_feat = ttnn.to_torch(g_edge_feat_dev).float()
    g_cutoff = ttnn.to_torch(g_cutoff_dev).float()

    forces = -torch.autograd.grad([edge_feat, cutoff], pos, grad_outputs=[g_edge_feat, g_cutoff])[0]
    return raw_pred, forces


def energy_and_forces(encoder, layers, ehead, device, *, pos, senders, receivers, atomic_numbers,
                      node_feat, cell_shift=None, r_max=6.0, num_bases=8, compute_stress=False,
                      cond_nodes=None):
    """Conservative energy + analytic forces ``F = -dE/dpos`` for one system
    (``orb-v3-conservative-inf-omat``). One device forward at the current geometry, one device
    reverse VJP, and a host ``torch.autograd.grad`` finish through the differentiable edge
    geometry (``tt_atom/orb_geometry.py``). ``encoder``/``layers``/``ehead`` are the already-
    constructed device modules (weights loaded once); ``senders``/``receivers``/``cell_shift``
    are the fixed graph topology (host, long/float tensors); ``node_feat`` is the (pos-
    independent) encoder node input. Returns ``(energy_raw: float, forces: torch.Tensor[N,3])``
    in normalized (pre-``host_energy_denormalize``) space -- same convention as ``EnergyHead``'s
    device output, so the caller finishes with the same host denormalize + ZBL add used by the
    energy-only path. When ``compute_stress`` is set, also returns ``virial: torch.Tensor[3,3]``
    (``dE/dstrain`` in the same raw/normalized space as ``forces``) -- the caller divides by the
    cell volume and applies the same denormalize scale as forces (see
    ``host_conservative_force_denormalize``) to get the physical stress tensor.

    ``cond_nodes`` (optional, ``[N, latent_dim]``, ``host_charge_spin_embedding``) is OrbMol's
    charge/spin conditioning input -- forwarded to ``OrbGraphContext`` unchanged. It has no
    ``pos``/``strain`` dependence, so it needs no adjoint of its own: the backward algebra below
    (``backbone_bw``/``attn_layer_bw``) is unmodified, since the conditioning term is just an
    additive shift on ``nodes`` with an identity Jacobian back to this function's own inputs.
    """
    import ttnn

    from .orb_geometry import host_edge_features
    from .orb_model import OrbGraphContext, _to_dev, host_cutoff

    pos = pos.detach().clone().requires_grad_(True)
    strain = torch.zeros(3, 3, dtype=pos.dtype, requires_grad=True) if compute_stress else None
    edge_feat, cutoff, vectors = host_edge_features(pos, senders, receivers, cell_shift,
                                                    r_max=r_max, num_bases=num_bases, strain=strain)

    N = atomic_numbers.shape[0]
    graph = OrbGraphContext(device, senders=senders, receivers=receivers,
                            cutoff=cutoff.detach().float(), num_nodes=N, cond_nodes=cond_nodes)

    node_dev = _to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev = _to_dev(edge_feat.detach().float(), device, ttnn.bfloat16)
    nodes, edges = encoder(node_dev, edge_dev)
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)
    raw_pred = ttnn.to_torch(ehead(nodes)).double().view(())

    g_edge_feat_dev, g_cutoff_dev = backbone_bw(encoder, layers, ehead, graph)
    g_edge_feat = ttnn.to_torch(g_edge_feat_dev).float()
    g_cutoff = ttnn.to_torch(g_cutoff_dev).float()

    inputs = [pos] if strain is None else [pos, strain]
    grads = torch.autograd.grad([edge_feat, cutoff], inputs, grad_outputs=[g_edge_feat, g_cutoff])
    forces = -grads[0]
    if strain is None:
        return float(raw_pred), forces
    return float(raw_pred), forces, grads[1]
