"""Device-side VJP (reverse-mode adjoint) for the PET-MAD backbone (``F = -dE/dpos``).

The pass-5 counterpart to ``tt_atom/orb_forces.py``: a hand-written reverse pass through
``tt_atom/pet_model.py`` that produces the adjoint at the three pos-dependent device
inputs (``edge_vec_cat``, ``cutoff_factors``, ``log_mask``), then a host
``torch.autograd.grad`` finish through ``pet_geometry``'s differentiable edge featurization
turns those adjoints into per-atom forces. The reported ``(E, F)`` pair is now
self-consistent -- the force is the gradient of the SAME device bf16 energy the
calculator reports (closing the pass-4 ~0.026 eV host/device inconsistency).

What is materially new here vs Orb's VJP (everything else has an Orb precedent):

  * the manual-attention log-mask backward. PET's attention is
    ``softmax(QK^T/sqrt(d) + log_mask)`` with a log-additive cutoff mask, not standard
    SDPA and not Orb's sigmoid gate. The softmax-with-additive-bias backward is
    ``g_scores = (g_attn - sum(g_attn * attn, dim=-1, keepdim=True)) * attn``; the mask
    enters additively so ``g_log_mask = sum_heads(g_scores)`` (the bias broadcasts over
    heads).
  * LayerNorm adjoint (``combination_norms``; Orb only has RMSNorm).
  * the NEF reversed-edge scatter: ``new_input_edge = gather(rev_idx, flat(out_edge))``.
    Its adjoint is a scatter-add by the SAME ``rev_idx`` (gather/scatter transpose pair,
    exactly like Orb's ``segment_sum`` adjoint = gather by the same index).

The SwiGLU, RMSNorm, Linear, SiLU adjoints are the same formulas as Orb's
(``orb_forces.py``); we reuse ``orb_fused_silu_bw`` for the SiLU VJP where profitable.
"""
from __future__ import annotations

import torch

from .device import orb_fused_silu_bw


def _mm(ttnn, g, W, kcfg):
    """grad wrt input of ``y = x @ W^T + b`` (W stored ``[in, out]``): ``g @ W``."""
    return ttnn.matmul(g, W, transpose_b=True, compute_kernel_config=kcfg)


def _silu_bw(ttnn, g, x):
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
        dummy_gate = ttnn.slice(g, [0, 0], [rows, 32])
        fused = ttnn._ttnn.operations.experimental.fused_gate(
            g, dummy_gate, x, wt, 1, wt - 1, 1)
        tail = ttnn.silu_bw(
            ttnn.slice(g, [0, prefix_width], [rows, width]),
            ttnn.slice(x, [0, prefix_width], [rows, width]))[0]
        return ttnn.concat(
            [ttnn.slice(fused, [0, 0], [rows, prefix_width]), tail], dim=1)
    return ttnn.silu_bw(g, x)[0]


def _sigmoid_bw(ttnn, g, x):
    return ttnn.sigmoid_bw(g, x)[0]


def rmsnorm_bw(norm, g_out):
    """VJP of ``RMSNorm``: ``y = x * inv * w``, ``inv = rsqrt(mean_C(x^2) + eps)``.

        g_x = inv*w*g_out - (inv^3/C) * x * sum_C(g_out * w * x)
    """
    ttnn = norm.ttnn
    x, inv, w, C = norm._cache_x, norm._cache_inv, norm.w, norm.dim
    gw = ttnn.multiply(g_out, w)
    s = ttnn.sum(ttnn.multiply(gw, x), dim=-1, keepdim=True)
    inv3 = ttnn.multiply(ttnn.multiply(inv, inv), inv)
    term2 = ttnn.multiply(ttnn.multiply(inv3, x), ttnn.multiply(s, 1.0 / C))
    return ttnn.subtract(ttnn.multiply(gw, inv), term2)


def layernorm_bw(norm, g_out):
    """VJP of ``LayerNorm``: ``y = (x - mean) * inv * w + b``,
    ``inv = rsqrt(var + eps)``, ``var = mean_C((x - mean)^2)``.

        g_xc = inv * w * g_out
        g_x  = g_xc - (1/C) sum_C(g_xc) - (inv^3 / C) * xc * sum_C(g_xc * xc)
    """
    ttnn = norm.ttnn
    x = norm._cache_x
    mean = norm._cache_mean
    inv = norm._cache_inv
    w = norm.w
    C = norm.dim
    xc = ttnn.subtract(x, mean)
    g_xc = ttnn.multiply(ttnn.multiply(g_out, inv), w)
    s1 = ttnn.sum(g_xc, dim=-1, keepdim=True)
    s2 = ttnn.sum(ttnn.multiply(g_xc, xc), dim=-1, keepdim=True)
    inv3 = ttnn.multiply(ttnn.multiply(inv, inv), inv)
    return ttnn.subtract(
        ttnn.subtract(g_xc, ttnn.multiply(s1, 1.0 / C)),
        ttnn.multiply(ttnn.multiply(inv3, xc), ttnn.multiply(s2, 1.0 / C)))


def _swiglu_bw(ttnn, g_h, cache, w_in, w_out, kcfg, ones_cache=None):
    """VJP of SwiGLU: ``z = Linear_w_in(x)`` -> split (v, g) -> ``h = v * sigmoid(g)``
    -> ``Linear_w_out(h)``. ``g_h`` is the adjoint at the SwiGLU output (post ``w_out``);
    returns ``g_x`` (adjoint at the SwiGLU input). ``cache`` holds ``z`` (the pre-split
    ``Linear_w_in`` output) and ``sig_g``."""
    z = cache["z"]
    sig_g = cache["sig_g"]
    v, _ = ttnn.chunk(z, 2, dim=-1)
    # h = v * sigmoid(g); g_h is adjoint at h (input to w_out)
    # g_v = g_h * sig_g ; g_g = g_h * v * sig_g * (1 - sig_g)
    g_v = ttnn.multiply(g_h, sig_g)
    # g_g = g_h * v * sig_g * (1 - sig_g). The ``1 - sig_g`` term needs a ones buffer of
    # sig_g's shape; ``ttnn.ones_like`` is an alloc+write that trace capture disallows, so
    # the caller passes a shape-keyed ones cache (``ones_cache``) populated lazily on the
    # first (warmup, pre-capture) call and reused every replay -- bit-identical to
    # ``ones_like`` (same dtype/layout/device), so eager numerics are unchanged.
    sshape = tuple(sig_g.shape)
    if ones_cache is None:
        ones_cache = {}
    if sshape not in ones_cache:
        ones_cache[sshape] = ttnn.ones(sshape, dtype=sig_g.dtype, layout=sig_g.layout,
                                     device=sig_g.device()) if hasattr(sig_g, "device") \
            else ttnn.ones(sshape, dtype=sig_g.dtype, layout=sig_g.layout)
    ones_s = ones_cache[sshape]
    g_g = ttnn.multiply(ttnn.multiply(ttnn.multiply(g_h, v), sig_g),
                       ttnn.subtract(ones_s, sig_g))
    g_z = ttnn.concat([g_v, g_g], dim=-1)
    return _mm(ttnn, g_z, w_in, kcfg)


def _attention_bw(attn, g_new_tokens, log_mask):
    """VJP of ``Attention`` (manual-attention path). Returns ``(g_x, g_log_mask)`` where
    ``g_x`` is the adjoint at the attention input (the normed tokens) and ``g_log_mask``
    is the adjoint at the (head-broadcast) log-additive mask."""
    ttnn = attn.ttnn
    c = attn._cache
    N, S, _ = c["x_shape"]
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    d_model = attn.d_model

    # output_linear: out = Linear(merged, w_out, b_out)
    g_merged = _mm(ttnn, g_new_tokens, attn.w_out, attn.kcfg)        # [N, S, d_model]
    # merge_heads backward: merged [N,S,d] -> [N*heads, S, head_dim]
    g_out_heads = _merge_heads_bw(ttnn, g_merged, N, S, num_heads, head_dim, d_model)
    # out_heads = attn @ v ; attn [N*heads, S, S], v [N*heads, S, head_dim]
    attn_t = ttnn.permute(c["attn"], (0, 2, 1))                       # [N*heads, S, S]
    g_v = ttnn.matmul(attn_t, g_out_heads, compute_kernel_config=attn.kcfg)
    g_attn = ttnn.matmul(g_out_heads, ttnn.permute(c["v"], (0, 2, 1)),
                         compute_kernel_config=attn.kcfg)            # [N*heads, S, S]
    # softmax-with-additive-bias backward:
    #   scores = qk*scale + log_mask ; e = exp(scores - mx) ; attn = e / s
    #   g_scores = (g_attn - sum(g_attn * attn, -1, keepdim)) * attn
    gs_sum = ttnn.sum(ttnn.multiply(g_attn, c["attn"]), dim=-1, keepdim=True)
    g_scores = ttnn.multiply(ttnn.subtract(g_attn, gs_sum), c["attn"])
    # log_mask is the head-broadcast [N*heads, S, S] additive bias -> adjoint = g_scores
    # (the head-broadcast is in the HOST log_mask construction; autograd sums it back
    # through the expand during the host finish, so the device adjoint is just g_scores).
    g_log_mask = g_scores
    # scores = qk * scale + log_mask
    g_qk = ttnn.multiply(g_scores, attn.scale)
    # qk = matmul(q, kT) ; q [N*heads, S, hd], kT [N*heads, hd, S]
    g_q = ttnn.matmul(g_qk, c["k"], compute_kernel_config=attn.kcfg)   # [N*heads, S, hd]
    g_kT = ttnn.matmul(c["q"], g_qk, transpose_a=True,
                       compute_kernel_config=attn.kcfg)               # [N*heads, hd, S]
    g_k = ttnn.permute(g_kT, (0, 2, 1))                                # [N*heads, S, hd]
    # split_heads backward: stack q,k,v -> z ; z = Linear(x, w_in)
    g_z = _split_heads_bw(ttnn, g_q, g_k, g_v, N, S, num_heads, head_dim, d_model)
    g_x = _mm(ttnn, g_z, attn.w_in, attn.kcfg)                         # [N, S, d_model]
    return g_x, g_log_mask


def _merge_heads_bw(ttnn, g_merged, N, S, num_heads, head_dim, d_model):
    """Inverse of ``_merge_heads``: ``[N, S, d_model]`` -> ``[N*heads, S, head_dim]``."""
    g = ttnn.reshape(g_merged, (N, S, num_heads, head_dim))
    g = ttnn.permute(g, (0, 2, 1, 3))                                  # [N, heads, S, hd]
    return ttnn.reshape(g, (N * num_heads, S, head_dim))


def _split_heads_bw(ttnn, g_q, g_k, g_v, N, S, num_heads, head_dim, d_model):
    """Inverse of ``_split_heads`` (which did ``z [N,S,d] -> [N*heads,S,hd]`` for each of
    q,k,v via reshape/permute). The forward ``chunk(z, 3, dim=-1)`` split ``z`` into
    q,k,v (each ``[N,S,d]``) before head-splitting, so the backward re-concatenates the
    three ``[N,S,d]`` adjoints along the last dim."""
    g_q_full = ttnn.reshape(g_q, (N, num_heads, S, head_dim))
    g_q_full = ttnn.permute(g_q_full, (0, 2, 1, 3))
    g_q_full = ttnn.reshape(g_q_full, (N, S, d_model))
    g_k_full = ttnn.reshape(g_k, (N, num_heads, S, head_dim))
    g_k_full = ttnn.permute(g_k_full, (0, 2, 1, 3))
    g_k_full = ttnn.reshape(g_k_full, (N, S, d_model))
    g_v_full = ttnn.reshape(g_v, (N, num_heads, S, head_dim))
    g_v_full = ttnn.permute(g_v_full, (0, 2, 1, 3))
    g_v_full = ttnn.reshape(g_v_full, (N, S, d_model))
    return ttnn.concat([g_q_full, g_k_full, g_v_full], dim=-1)        # [N, S, 3*d]


def _energy_head_bw(ehead, g_raw, cutoff_factors):
    """VJP of ``EnergyHead``. ``g_raw`` is the scalar adjoint at the raw energy (=1 for
    forces). Returns ``(g_node_feat, g_edge_feat, g_cutoff_factors)`` -- the adjoints at
    the head's three pos-eventually inputs (node_feat is pos-independent, but its adjoint
    propagates further into the backbone). ``cutoff_factors`` is the device input tensor;
    its adjoint is accumulated here from the ``edge_pred * cutoff`` product."""
    ttnn = ehead.ttnn
    kcfg = ehead.kcfg
    c = ehead._cache
    N, Dmax = c["node_feat_shape"][0], c["edge_feat_shape"][1]

    # raw = sum(node_pred + edge_pred_sum) ; g_raw is a scalar tensor
    if getattr(ehead, "_bw_ones_n1", None) is None or ehead._bw_ones_n1.shape[0] != N:
        ehead._bw_ones_n1 = ttnn.ones((N, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=ehead.device) if getattr(ehead, "device", None) is not None \
            else ttnn.ones((N, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ones_N1 = ehead._bw_ones_n1
    g_node_pred = ttnn.multiply(ones_N1, g_raw)
    g_edge_pred_sum = ttnn.multiply(ones_N1, g_raw)

    # edge_pred_sum = sum(edge_pred, dim=1) -> broadcast g_edge_pred to [N, Dmax, 1]
    # via ones-multiply (broadcasts [N,1,1] * [1,Dmax,1] -> [N,Dmax,1])
    g_sum_r = ttnn.reshape(g_edge_pred_sum, (N, 1, 1))
    if getattr(ehead, "_bw_ones_dmax", None) is None or ehead._bw_ones_dmax.shape[1] != Dmax:
        ehead._bw_ones_dmax = ttnn.ones((1, Dmax, 1), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=ehead.device)
    g_edge_pred = ttnn.multiply(g_sum_r, ehead._bw_ones_dmax)

    # edge_pred = edge_pred_pre * cutoff_factors
    g_edge_pred_pre = ttnn.multiply(g_edge_pred, cutoff_factors)
    g_cutoff = ttnn.multiply(g_edge_pred, c["edge_pred_pre"])

    # edge_pred_pre = Linear(he, ell)
    g_he = _mm(ttnn, g_edge_pred_pre, ehead.ell_w, kcfg)
    # he = silu(Linear(he2, e2))
    g_he2 = _mm(ttnn, _silu_bw(ttnn, g_he, c["a_e2"]), ehead.e2_w, kcfg)
    g_he0 = _mm(ttnn, _silu_bw(ttnn, g_he2, c["a_e0"]), ehead.e0_w, kcfg)
    g_edge_feat = g_he0

    # node_pred = Linear(h2, nll) ; h2 = silu(Linear(h1, n2)) ; h1 = silu(Linear(node_feat, n0))
    g_h2 = _mm(ttnn, g_node_pred, ehead.nll_w, kcfg)
    g_h1 = _mm(ttnn, _silu_bw(ttnn, g_h2, c["a_n2"]), ehead.n2_w, kcfg)
    g_node_feat = _mm(ttnn, _silu_bw(ttnn, g_h1, c["a_n0"]), ehead.n0_w, kcfg)

    return g_node_feat, g_edge_feat, g_cutoff


def _gnn_layer_bw(layer, g_out_node, g_out_edge, g_next_input_edge, log_mask):
    """VJP of one ``GnnLayer``. Returns ``(g_input_node, g_input_edge, g_edge_vec_cat,
    g_log_mask)`` -- the adjoints at the layer's four (eventually) inputs. ``g_out_edge``
    is the adjoint at the returned ``out_edge_pre`` (post-mlp, pre-combination edge);
    ``g_next_input_edge`` is the adjoint at ``next_input_edge`` (the edge fed to the next
    layer)."""
    ttnn = layer.ttnn
    kcfg = layer.kcfg
    c = layer._cache
    N = c["N"]
    Dmax = c["Dmax"]
    d_pet = layer.d_pet
    d_node = layer.d_node
    # shape-keyed cache for the SwiGLU backward's ``1 - sig_g`` ones buffers, populated
    # lazily on the warmup (pre-capture) call and reused every replay (``ttnn.ones_like``
    # is a trace-blocked alloc+write).
    if not hasattr(layer, "_swiglu_ones"):
        layer._swiglu_ones = {}

    # --- combination backward ---
    # next_input_edge = input_edge + out_edge + comb
    g_input_edge = g_next_input_edge
    g_out_edge = g_next_input_edge
    g_comb = g_next_input_edge

    # comb = Linear(silu(Linear(normed_comb, comb0)), comb2)
    g_h_comb = _mm(ttnn, g_comb, layer.comb2_w, kcfg)
    g_normed_comb = _silu_bw(ttnn, g_h_comb, c["a_comb0"])
    g_concat = layernorm_bw(layer.comb_norm, g_normed_comb)            # [N, Dmax, 2*d]
    # concat([out_edge, new_input_edge], -1)
    g_out_edge_comb = ttnn.slice(g_concat, [0, 0, 0], [N, Dmax, d_pet])
    g_new_input_edge = ttnn.slice(g_concat, [0, 0, d_pet], [N, Dmax, 2 * d_pet])
    g_out_edge = ttnn.add(g_out_edge, g_out_edge_comb)

    # new_input_edge = reshape(embedding(rev_idx, flat)) ; flat = reshape(out_edge, [N*Dmax, d])
    # gather backward = scatter-add by the same rev_idx. Done on DEVICE via
    # ``ttnn.scatter_add`` (bf16) so the backward is fully device-resident and
    # trace-capturable -- the pass-5 host ``torch.index_add_`` roundtrip broke the captured
    # instruction stream (a host op between device ops can't be recorded). ``ttnn.scatter_add``
    # follows ``torch.scatter_add`` semantics (index broadcasts against src along ``dim``), so
    # the [N*Dmax] ``rev_idx`` is broadcast to [N*Dmax, d_pet] once per topology and uploaded as
    # a resident uint32 index buffer; the scatter accumulates at repeated indices in bf16.
    # Verified bf16 PCC 0.999998 vs the float32 host ``index_add`` (probe_trace.py); the
    # end-to-end forces PCC vs golden is unchanged at 0.98990 (the bf16 scatter accumulation
    # sits below the backward's existing bf16 noise floor).
    if getattr(layer, "_rev_idx_b_dev", None) is None or layer._rev_idx_b_dev.shape[0] != N * Dmax:
        rev_idx_b = layer._rev_idx_host.to(torch.int32)[:, None].expand(N * Dmax, d_pet).contiguous()
        layer._rev_idx_b_dev = ttnn.from_torch(rev_idx_b, dtype=ttnn.uint32,
                                               layout=ttnn.ROW_MAJOR_LAYOUT, device=layer.device)
    # persistent zero buffer for the scatter-add input, created lazily on the first call
    # (warmup, before trace capture) and reused every replay. ``ttnn.zeros`` inside the
    # captured body is a host-initiated write that trace capture disallows, so it must be
    # pre-allocated; scatter_add only reads it (stays zero), so one buffer per layer is
    # enough.
    if getattr(layer, "_g_flat_zero", None) is None or layer._g_flat_zero.shape[0] != N * Dmax:
        layer._g_flat_zero = ttnn.zeros((N * Dmax, d_pet), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=layer.device)
    g_new_input_edge_flat = ttnn.reshape(g_new_input_edge, (N * Dmax, d_pet))
    g_flat = ttnn.scatter_add(layer._g_flat_zero, dim=0, index=layer._rev_idx_b_dev,
                              src=g_new_input_edge_flat)
    g_out_edge_from_rev = ttnn.reshape(g_flat, (N, Dmax, d_pet))
    g_out_edge = ttnn.add(g_out_edge, g_out_edge_from_rev)

    # --- transformer-layer edge path backward ---
    # out_edge (returned) = e3 + e4 ; e3 = e1 + e2 ; e4 = swiglu(rmsnorm(e3), mlp)
    g_e3 = g_out_edge
    g_e4 = g_out_edge
    # e4 = swiglu(rmsnorm(e3), mlp)
    g_h_mlp = _mm(ttnn, g_e4, layer.mlp_w_out, kcfg)
    g_normed_e3_mlp = _swiglu_inner_bw(ttnn, g_h_mlp, c["mlp_cache"], layer.mlp_w_in, kcfg,
                                      ones_cache=layer._swiglu_ones)
    g_e3_from_mlp = rmsnorm_bw(layer.norm_mlp, g_normed_e3_mlp)
    g_e3 = ttnn.add(g_e3, g_e3_from_mlp)
    # e3 = e1 + e2
    g_e1 = g_e3
    g_e2 = g_e3

    # --- transformer-layer node path backward ---
    # out_node = reshape(out_node_mid2) ; out_node_mid2 = out_node_mid + swiglu(rmsnorm(out_node_mid), center_mlp)
    # out_node_mid = n_tok + out_node_expanded ; out_node_expanded = Linear(out_node_tok, center_expansion)
    g_out_node_mid2 = ttnn.reshape(g_out_node, (N, 1, d_node))
    g_out_node_mid = g_out_node_mid2
    g_center_mlp_out = g_out_node_mid2
    g_normed_cm = _swiglu_inner_bw(ttnn, _mm(ttnn, g_center_mlp_out, layer.center_mlp_w_out, kcfg),
                                   c["center_mlp_cache"], layer.center_mlp_w_in, kcfg,
                                   ones_cache=layer._swiglu_ones)
    g_out_node_mid_from_mlp = rmsnorm_bw(layer.norm_center, g_normed_cm)
    g_out_node_mid = ttnn.add(g_out_node_mid, g_out_node_mid_from_mlp)
    # out_node_mid = n_tok + out_node_expanded
    g_n_tok = g_out_node_mid
    g_out_node_expanded = g_out_node_mid
    # out_node_expanded = Linear(out_node_tok, center_expansion)
    g_out_node_tok = _mm(ttnn, g_out_node_expanded, layer.center_expansion_w, kcfg)   # [N, 1, d]

    # new_tokens = concat([out_node_tok, e2], dim=1) ; slice backward
    g_new_tokens = ttnn.concat([g_out_node_tok, g_e2], dim=1)        # [N, 1+Dmax, d]
    # new_tokens = attention(normed, log_mask) ; normed = rmsnorm(tokens)
    g_normed, g_log_mask = _attention_bw(layer.attention, g_new_tokens, log_mask)
    g_tokens = rmsnorm_bw(layer.norm_attention, g_normed)
    # tokens = concat([input_node_tok, e1], dim=1)
    g_input_node_tok = ttnn.slice(g_tokens, [0, 0, 0], [N, 1, d_pet])
    g_e1_from_tokens = ttnn.slice(g_tokens, [0, 1, 0], [N, 1 + Dmax, d_pet])
    g_e1 = ttnn.add(g_e1, g_e1_from_tokens)
    # input_node_tok = Linear(node_tok, center_contraction) ; node_tok = reshape(input_node, [N,1,d_node])
    g_node_tok = _mm(ttnn, g_input_node_tok, layer.center_contraction_w, kcfg)       # [N, 1, d_node]
    g_input_node = ttnn.reshape(g_node_tok, (N, d_node))

    # --- CartesianTransformer edge-embed/compress backward ---
    # e1 = Linear(h, compress2) ; h = silu(a_compress0) ; a_compress0 = Linear(edge_tokens, compress0)
    g_h = _mm(ttnn, g_e1, layer.compress2_w, kcfg)
    g_a_compress0 = _silu_bw(ttnn, g_h, c["a_compress0"])
    g_edge_tokens = _mm(ttnn, g_a_compress0, layer.compress0_w, kcfg)
    # edge_tokens = concat([edge_emb, (nbr_emb?), input_edge], -1)
    if layer.is_first:
        # edge_tokens = concat([edge_emb, input_edge], -1)  (2*d)
        g_edge_emb = ttnn.slice(g_edge_tokens, [0, 0, 0], [N, Dmax, d_pet])
        g_input_edge_from_tokens = ttnn.slice(g_edge_tokens, [0, 0, d_pet], [N, Dmax, 2 * d_pet])
    else:
        # edge_tokens = concat([edge_emb, nbr_emb, input_edge], -1)  (3*d)
        g_edge_emb = ttnn.slice(g_edge_tokens, [0, 0, 0], [N, Dmax, d_pet])
        # nbr_emb is pos-independent (embedding of elem_nbr) -> adjoint discarded
        g_input_edge_from_tokens = ttnn.slice(g_edge_tokens, [0, 0, 2 * d_pet], [N, Dmax, 3 * d_pet])
    g_input_edge = ttnn.add(g_input_edge, g_input_edge_from_tokens)
    # edge_emb = Linear(edge_vec_cat, edge_emb_w)
    g_edge_vec_cat = _mm(ttnn, g_edge_emb, layer.edge_emb_w, kcfg)    # [N, Dmax, 4]

    return g_input_node, g_input_edge, g_edge_vec_cat, g_log_mask


def _swiglu_inner_bw(ttnn, g_h, cache, w_in, kcfg, ones_cache=None):
    """Shared SwiGLU VJP body (used by both the node MLP and edge MLP paths). Returns the
    adjoint at the SwiGLU input (the normed input to ``w_in``)."""
    return _swiglu_bw(ttnn, g_h, cache, w_in, None, kcfg, ones_cache=ones_cache)


def backbone_bw(model, bd_dev, g_raw=1.0):
    """Full reverse pass through ``PetModel``. Returns ``(g_edge_vec_cat, g_cutoff_factors,
    g_log_mask)`` -- the device adjoints at the three pos-dependent uploaded inputs, ready
    for the host ``torch.autograd.grad`` finish. ``g_raw`` is the scalar seed at the raw
    energy (1.0 for forces)."""
    ttnn = model.ttnn
    kcfg = model.layers[0].kcfg
    cutoff_factors = bd_dev["cutoff_factors"]                        # [N, Dmax, 1]

    # EnergyHead backward -> adjoints at the final node/edge features
    if getattr(model, "_g_seed_ones", None) is None:
        model._g_seed_ones = ttnn.ones((1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=model.device) if hasattr(model, "device") else \
            ttnn.ones((1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    g_seed = ttnn.multiply(model._g_seed_ones, g_raw)
    g_node, g_edge, g_cutoff = _energy_head_bw(model.energy_head, g_seed, cutoff_factors)
    g_log_mask = None
    g_evc_list = []
    for layer in reversed(model.layers):
        # next_input_edge for this layer is the g_edge propagated from the next layer
        # (for the last layer it's the energy head's g_edge; for earlier layers it's the
        # g_input_edge returned by the layer below). The layer's own out_edge_pre adjoint
        # is g_edge (the energy head's edge adjoint, or the layer-below's input_edge adjoint).
        g_in_node, g_in_edge, g_evc, g_lm = _gnn_layer_bw(
            layer, g_node, g_edge, g_edge, bd_dev["log_mask"])
        g_node = g_in_node
        g_edge = g_in_edge
        g_log_mask = g_lm if g_log_mask is None else ttnn.add(g_log_mask, g_lm)
        # accumulate edge_vec_cat across layers (each layer consumes edge_vec_cat fresh)
        g_evc_list.append(g_evc)
    # sum the per-layer edge_vec_cat adjoints into one device tensor. A clean reduction
    # (no mutable model state) so the backward is trace-capturable: the ``hasattr``/``del``
    # accumulator pattern recorded Python control flow at capture time and left a stale
    # attribute on replay. The sum is a fixed-arity device op stream regardless of layer
    # count, captured once and replayed as-is.
    g_edge_vec_cat = g_evc_list[0]
    for g in g_evc_list[1:]:
        g_edge_vec_cat = ttnn.add(g_edge_vec_cat, g)
    return g_edge_vec_cat, g_cutoff, g_log_mask
