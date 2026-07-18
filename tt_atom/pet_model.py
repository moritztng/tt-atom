"""Device (ttnn) backbone for PET-MAD (UPET, pet-mad-s v1.5.0).

A 1:1 port of ``tt_atom/pet_model_host.py``'s ops to ttnn, run on a Tenstorrent card.
Mirrors the host reference exactly: the same op order, the same RMSNorm / LayerNorm /
SwiGLU / manual-attention semantics, the same NEF reversed-edge gather, the same energy
readout. The only differences are dtype (bf16 operands, fp32 matmul accumulation via
``compute_kernel_config``) and the softmax (a manual ``exp(x-max)/sum`` — ttnn's
``ttnn.softmax`` is an approximate kernel whose row-sum is off by ~3%, see
``tests/test_pet_device.py``'s parity gate).

Inputs are the host geometry's NEF tensors (``tt_atom/pet_geometry.py``) uploaded once per
topology — the per-edge fixed terms (edge vectors/distances, cutoff factors, the
log-additive attention mask, the NEF index tables) are host-computed and uploaded as
constants, exactly like Orb's ``cutoff`` buffer. Every learned op (embeddings, the
per-layer Linears, the attention QKV / scores / context matmuls, the readout MLPs) runs on
device.

Parity target: the canonical-order per-layer fixture
``tests/data/pet_mad_s_si_canon_internals.npz`` (gnn{0,1,2}_node_out / gnn{0,1,2}_edge_out
+ raw_energy), captured from the REAL PET under hooks with the same canonical-ordered
geometry. See ``docs/pet-mad-port.md`` and ``~/.coworker/notes/tt-atom-pet-mad-port-p2.md``.
"""
from __future__ import annotations

import math

import torch

from .device import compute_kernel_config


def _to_dev(t, device, dtype, layout=None):
    import ttnn

    layout = layout or ttnn.TILE_LAYOUT
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _linear(ttnn, x, w_packed, bias, kcfg, dtype):
    """``F.linear``-equivalent. Weights are stored transposed (``[in, out]``, torch
    ``.T.contiguous()``) so ``ttnn.linear`` (which computes ``x @ w^T``) reproduces
    ``F.linear`` exactly — the same convention as ``tt_atom/orb_model.py``."""
    return ttnn.linear(x, w_packed, bias=bias, compute_kernel_config=kcfg, dtype=dtype)


class RMSNorm:
    """``nn.RMSNorm`` (elementwise-affine, no bias) over the last dim — the host
    ``rmsnorm`` op, on device. Caches ``x`` and ``inv`` for the (next pass) force VJP."""

    def __init__(self, weights, prefix, device, dim, eps=1e-6):
        import ttnn

        self.ttnn = ttnn
        self.eps = eps
        self.dim = dim
        self.device = device
        self.w = _to_dev(weights[f"{prefix}.weight"].view(1, dim).contiguous(),
                         device, ttnn.bfloat16)

    def __call__(self, x):
        ttnn = self.ttnn
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ms, self.eps))
        self._cache_x, self._cache_inv = x, inv          # reused by pet_vjp.rmsnorm_bw
        return ttnn.multiply(ttnn.multiply(x, inv), self.w)


class LayerNorm:
    """``nn.LayerNorm`` (elementwise-affine, with bias) over the last dim — the host
    ``layernorm`` op (PET's ``combination_norms``)."""

    def __init__(self, weights, prefix, device, dim, eps=1e-5):
        import ttnn

        self.ttnn = ttnn
        self.eps = eps
        self.dim = dim
        self.w = _to_dev(weights[f"{prefix}.weight"].view(1, dim).contiguous(),
                         device, ttnn.bfloat16)
        self.b = _to_dev(weights[f"{prefix}.bias"].view(1, dim).contiguous(),
                         device, ttnn.bfloat16)
        self.device = device

    def __call__(self, x):
        ttnn = self.ttnn
        mean = ttnn.mean(x, dim=-1, keepdim=True)
        xc = ttnn.subtract(x, mean)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(var, self.eps))
        self._cache_x, self._cache_mean, self._cache_inv = x, mean, inv  # for pet_vjp.layernorm_bw
        return ttnn.add(ttnn.multiply(ttnn.multiply(xc, inv), self.w), self.b)


def _swiglu(ttnn, x, w_in_packed, w_in_bias, w_out_packed, w_out_bias, kcfg, dtype, cache=None):
    """SwiGLU feedforward: ``w_in`` -> split (v, g) -> ``v * sigmoid(g)`` -> ``w_out``.
    PET-MAD v1.5.0 uses SwiGLU everywhere. When ``cache`` is a dict, stores ``z`` (the
    pre-split ``w_in`` output) and ``sig_g`` (= ``sigmoid(g)``) for ``pet_vjp._swiglu_bw``."""
    z = _linear(ttnn, x, w_in_packed, w_in_bias, kcfg, dtype)
    v, g = ttnn.chunk(z, 2, dim=-1)
    sig_g = ttnn.sigmoid(g)
    h = ttnn.multiply(v, sig_g)
    if cache is not None:
        cache["z"], cache["sig_g"] = z, sig_g
    return _linear(ttnn, h, w_out_packed, w_out_bias, kcfg, ttnn.bfloat16)


def _reduce(ttnn, op, x, dim, keepdim=True):
    """ttnn.max/sum return ``(out, indices)`` for some dims; unwrap to the tensor."""
    r = op(x, dim=dim, keepdim=keepdim)
    return r[0] if isinstance(r, tuple) else r


class Attention:
    """``AttentionBlock`` (manual_attention path) on device.

    ``x`` [N, seq, d_model]; ``log_mask`` [N*heads, seq, seq] — the log-additive cutoff
    mask, host-precomputed as ``log(clamp(cf, 1e-15))`` and broadcast over heads (a
    per-topology constant, like Orb's ``cutoff``). Returns [N, seq, d_model].

    Manual softmax (``exp(x-max)/sum``) instead of ``ttnn.softmax`` — the latter is an
    approximate kernel whose row-sum is ~0.97 (one of 32 elements dropped), which would
    attenuate the attention context by 3% and blow the per-layer PCC gate.
    """

    def __init__(self, weights, prefix, device, *, d_model, num_heads, temperature):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.device = device
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / (math.sqrt(self.head_dim) * temperature)
        self.d_model = d_model
        wd = ttnn.bfloat16
        self.w_in = _to_dev(weights[f"{prefix}.input_linear.weight"].T.contiguous(),
                            device, wd)
        self.b_in = _to_dev(weights[f"{prefix}.input_linear.bias"], device, wd)
        self.w_out = _to_dev(weights[f"{prefix}.output_linear.weight"].T.contiguous(),
                             device, wd)
        self.b_out = _to_dev(weights[f"{prefix}.output_linear.bias"], device, wd)

    def _split_heads(self, z):
        """z [N, seq, d_model] -> [N*heads, seq, head_dim]."""
        ttnn = self.ttnn
        N, seq, _ = z.shape
        z = ttnn.reshape(z, (N, seq, self.num_heads, self.head_dim))
        z = ttnn.permute(z, (0, 2, 1, 3))                  # [N, heads, seq, head_dim]
        return ttnn.reshape(z, (N * self.num_heads, seq, self.head_dim))

    def _merge_heads(self, z, N, seq):
        ttnn = self.ttnn
        z = ttnn.reshape(z, (N, self.num_heads, seq, self.head_dim))
        z = ttnn.permute(z, (0, 2, 1, 3))                  # [N, seq, heads, head_dim]
        return ttnn.reshape(z, (N, seq, self.d_model))

    def __call__(self, x, log_mask):
        ttnn = self.ttnn
        N, seq, _ = x.shape
        z = _linear(ttnn, x, self.w_in, self.b_in, self.kcfg, ttnn.bfloat16)  # [N,seq,3*d]
        q, k, v = ttnn.chunk(z, 3, dim=-1)                 # each [N,seq,d]
        q = self._split_heads(q)                           # [N*heads, seq, head_dim]
        k = self._split_heads(k)
        v = self._split_heads(v)
        kT = ttnn.permute(k, (0, 2, 1))                    # [N*heads, head_dim, seq]
        scores = ttnn.matmul(q, kT, compute_kernel_config=self.kcfg)  # [N*heads, seq, seq]
        scores = ttnn.add(ttnn.multiply(scores, self.scale), log_mask)
        mx = _reduce(ttnn, ttnn.max, scores, dim=-1, keepdim=True)
        e = ttnn.exp(ttnn.subtract(scores, mx))
        s = ttnn.sum(e, dim=-1, keepdim=True)
        attn = ttnn.multiply(e, ttnn.reciprocal(s))
        out = ttnn.matmul(attn, v, compute_kernel_config=self.kcfg)  # [N*heads, seq, head_dim]
        # cache for pet_vjp._attention_bw: q, k, v (post head-split), attn, x shape
        self._cache = dict(q=q, k=k, v=v, attn=attn, x_shape=(N, seq, self.d_model))
        out = self._merge_heads(out, N, seq)               # [N, seq, d_model]
        return _linear(ttnn, out, self.w_out, self.b_out, self.kcfg, ttnn.bfloat16)


class GnnLayer:
    """One PET GNN layer = ``CartesianTransformer`` (edge embed/compress + one PreLN
    ``TransformerLayer``) + the reversed-edge ``combination`` MLP. Mirrors
    ``pet_model_host.cartesian_transformer`` + ``transformer_layer`` +
    ``feedforward_featurization``'s per-layer loop body, on device."""

    def __init__(self, weights, device, *, cfg, layer_idx):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.device = device
        self.layer_idx = layer_idx
        self.is_first = layer_idx == 0
        d_pet = cfg["d_pet"]
        d_node = cfg["d_node"]
        wd = ttnn.bfloat16
        p = f"gnn_layers.{layer_idx}."
        self.d_pet = d_pet
        self.d_node = d_node

        def lin(key):
            return (_to_dev(weights[f"{key}.weight"].T.contiguous(), device, wd),
                    _to_dev(weights[f"{key}.bias"], device, wd))

        self.edge_emb_w, self.edge_emb_b = lin(f"{p}edge_embedder")
        if not self.is_first:
            self.nbr_emb_w = _to_dev(weights[f"{p}neighbor_embedder.weight"], device, wd)
        self.compress0_w, self.compress0_b = lin(f"{p}compress.0")
        self.compress2_w, self.compress2_b = lin(f"{p}compress.2")

        tp = f"{p}trans.layers.0."
        self.center_contraction_w, self.center_contraction_b = lin(f"{tp}center_contraction")
        self.center_expansion_w, self.center_expansion_b = lin(f"{tp}center_expansion")
        self.norm_attention = RMSNorm(weights, f"{tp}norm_attention", device, d_pet)
        self.attention = Attention(weights, f"{tp}attention", device,
                                   d_model=d_pet, num_heads=cfg["num_heads"],
                                   temperature=cfg["attention_temperature"])
        self.norm_center = RMSNorm(weights, f"{tp}norm_center_features", device, d_node)
        self.center_mlp_w_in, self.center_mlp_b_in = lin(f"{tp}center_mlp.w_in")
        self.center_mlp_w_out, self.center_mlp_b_out = lin(f"{tp}center_mlp.w_out")
        self.norm_mlp = RMSNorm(weights, f"{tp}norm_mlp", device, d_pet)
        self.mlp_w_in, self.mlp_b_in = lin(f"{tp}mlp.w_in")
        self.mlp_w_out, self.mlp_b_out = lin(f"{tp}mlp.w_out")

        cp = f"combination_norms.{layer_idx}"
        self.comb_norm = LayerNorm(weights, cp, device, 2 * d_pet)
        mp = f"combination_mlps.{layer_idx}"
        self.comb0_w, self.comb0_b = lin(f"{mp}.0")
        self.comb2_w, self.comb2_b = lin(f"{mp}.2")

    def _linear(self, x, wb, dtype):
        return _linear(self.ttnn, x, wb[0], wb[1], self.kcfg, dtype)

    def _swiglu(self, x, w_in, w_out, cache=None):
        return _swiglu(self.ttnn, x, w_in[0], w_in[1], w_out[0], w_out[1],
                       self.kcfg, self.ttnn.bfloat16, cache=cache)

    def __call__(self, input_node, input_edge, edge_vec_cat, elem_nbr_dev,
                 log_mask, rev_idx_dev, rev_idx_host=None):
        """``input_node`` [N, d_node], ``input_edge`` [N, Dmax, d_pet],
        ``edge_vec_cat`` [N, Dmax, 4], ``elem_nbr_dev`` [N*Dmax] uint32 (embedding index
        into ``neighbor_embedder.weight``), ``log_mask`` [N*heads, 1+Dmax, 1+Dmax] (the
        log-additive cutoff mask, host-precomputed and broadcast over heads),
        ``rev_idx_dev`` [N*Dmax] uint32 (the flattened ``reverse_neighbor_index`` — a
        gather index into the flattened edge tensor; padded slots carry indices the host
        geometry already made unique in [0, N*Dmax), so no sentinel pad row is needed).
        ``rev_idx_host`` is the same index as a host ``long`` tensor (cached for the
        NEF-scatter adjoint in ``pet_vjp``). Returns (out_node [N, d_node],
        out_edge [N, Dmax, d_pet])."""
        ttnn = self.ttnn
        N = input_node.shape[0]
        Dmax = input_edge.shape[1]
        self._cache = dict(N=N, Dmax=Dmax)
        self._rev_idx_host = rev_idx_host

        # --- CartesianTransformer: edge embed + compress ---
        edge_emb = self._linear(edge_vec_cat, (self.edge_emb_w, self.edge_emb_b), ttnn.bfloat16)
        if self.is_first:
            edge_tokens = ttnn.concat([edge_emb, input_edge], dim=-1)        # [N,Dmax,2*d]
        else:
            nbr_emb = ttnn.embedding(elem_nbr_dev, self.nbr_emb_w)            # [N*Dmax, d]
            nbr_emb = ttnn.reshape(nbr_emb, (N, Dmax, self.d_pet))
            edge_tokens = ttnn.concat([edge_emb, nbr_emb, input_edge], dim=-1)  # [N,Dmax,3*d]
        a_compress0 = self._linear(edge_tokens, (self.compress0_w, self.compress0_b), ttnn.bfloat16)
        self._cache["a_compress0"] = a_compress0
        h = ttnn.silu(a_compress0)
        edge_tokens = self._linear(h, (self.compress2_w, self.compress2_b), ttnn.bfloat16)  # [N,Dmax,d]

        # --- TransformerLayer (PreLN) ---
        node_tok = ttnn.reshape(input_node, (N, 1, self.d_node))
        input_node_tok = self._linear(node_tok, (self.center_contraction_w, self.center_contraction_b),
                                     ttnn.bfloat16)                          # [N,1,d]
        tokens = ttnn.concat([input_node_tok, edge_tokens], dim=1)           # [N,1+Dmax,d]
        normed = self.norm_attention(tokens)
        new_tokens = self.attention(normed, log_mask)                        # [N,1+Dmax,d]
        out_node_tok, out_edge = ttnn.slice(new_tokens, [0, 0, 0], [N, 1, self.d_pet]), \
            ttnn.slice(new_tokens, [0, 1, 0], [N, 1 + Dmax, self.d_pet])       # [N,1,d],[N,Dmax,d]
        out_node = ttnn.add(node_tok,
                           self._linear(out_node_tok, (self.center_expansion_w, self.center_expansion_b),
                                        ttnn.bfloat16))                     # [N,1,d_node]
        center_mlp_cache = {}
        out_node = ttnn.add(out_node,
                            self._swiglu(self.norm_center(out_node),
                                         (self.center_mlp_w_in, self.center_mlp_b_in),
                                         (self.center_mlp_w_out, self.center_mlp_b_out),
                                         cache=center_mlp_cache))
        out_edge = ttnn.add(edge_tokens, out_edge)
        mlp_cache = {}
        out_edge = ttnn.add(out_edge,
                            self._swiglu(self.norm_mlp(out_edge),
                                         (self.mlp_w_in, self.mlp_b_in),
                                         (self.mlp_w_out, self.mlp_b_out),
                                         cache=mlp_cache))
        self._cache["center_mlp_cache"] = center_mlp_cache
        self._cache["mlp_cache"] = mlp_cache

        # --- combination: reversed-edge gather + LayerNorm + SwiGLU-ish (Linear-SiLU-Linear) ---
        flat = ttnn.reshape(out_edge, (N * Dmax, self.d_pet))                  # [N*Dmax, d]
        rev = ttnn.embedding(rev_idx_dev, flat)                               # [N*Dmax, d]
        new_input_edge = ttnn.reshape(rev, (N, Dmax, self.d_pet))
        concatenated = ttnn.concat([out_edge, new_input_edge], dim=-1)        # [N,Dmax,2*d]
        normed = self.comb_norm(concatenated)
        a_comb0 = self._linear(normed, (self.comb0_w, self.comb0_b), ttnn.bfloat16)
        self._cache["a_comb0"] = a_comb0
        h = ttnn.silu(a_comb0)
        comb = self._linear(h, (self.comb2_w, self.comb2_b), ttnn.bfloat16)    # [N,Dmax,d]

        next_input_edge = ttnn.add(ttnn.add(input_edge, out_edge), comb)
        out_node = ttnn.reshape(out_node, (N, self.d_node))
        return out_node, out_edge, next_input_edge


class EnergyHead:
    """``PET._calculate_last_layer_features`` + ``_calculate_atomic_predictions`` for the
    energy target, on device. ``node_heads.energy.0`` (Linear-SiLU-Linear-SiLU) on node
    features -> ``node_last_layers.energy.0.energy___0`` (Linear(256,1)); same for edges,
    then cutoff-weighted sum over neighbors. Returns the raw (pre-scaler,
    pre-composition) per-structure energy scalar as a device tensor.

    Runs in bf16 (same as the backbone). A float32-operand readout was tried but
    Blackhole's f32 matmul path is only ~bf16-accurate internally (fp32 dest-accum, not
    f32 operands), so it does not tighten the energy; the readout's bf16 contribution
    (~0.008 eV raw) is comparable to the backbone's bf16 drift (~0.0024 eV raw), and the
    combined device energy lands ~0.026 eV from the host reference (measured, see
    ``tests/test_pet_device.py``). The per-layer PCC gate (>=0.999) is checked on the
    backbone outputs upstream of this head."""

    def __init__(self, weights, device):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.device = device
        wd = ttnn.bfloat16

        def lin(key):
            return (_to_dev(weights[f"{key}.weight"].T.contiguous(), device, wd),
                    _to_dev(weights[f"{key}.bias"], device, wd))

        self.n0_w, self.n0_b = lin("node_heads.energy.0.0")
        self.n2_w, self.n2_b = lin("node_heads.energy.0.2")
        self.nll_w, self.nll_b = lin("node_last_layers.energy.0.energy___0")
        self.e0_w, self.e0_b = lin("edge_heads.energy.0.0")
        self.e2_w, self.e2_b = lin("edge_heads.energy.0.2")
        self.ell_w, self.ell_b = lin("edge_last_layers.energy.0.energy___0")

    def _lin(self, x, wb):
        return _linear(self.ttnn, x, wb[0], wb[1], self.kcfg, self.ttnn.bfloat16)

    def __call__(self, node_feat, edge_feat, cutoff_factors):
        """``node_feat`` [N, d_node], ``edge_feat`` [N, Dmax, d_pet], ``cutoff_factors``
        [N, Dmax, 1] (the per-edge cutoff weight; 0 on padded slots, so the padded-edge
        contributions vanish without an explicit mask — same as the host reference's
        ``where``+``*cutoff`` pair, which is redundant given cutoff is already 0 there)."""
        ttnn = self.ttnn
        a_n0 = self._lin(node_feat, (self.n0_w, self.n0_b))
        h = ttnn.silu(a_n0)
        a_n2 = self._lin(h, (self.n2_w, self.n2_b))
        h = ttnn.silu(a_n2)
        node_pred = self._lin(h, (self.nll_w, self.nll_b))                  # [N,1]

        a_e0 = self._lin(edge_feat, (self.e0_w, self.e0_b))
        he = ttnn.silu(a_e0)
        a_e2 = self._lin(he, (self.e2_w, self.e2_b))
        he = ttnn.silu(a_e2)
        edge_pred_pre = self._lin(he, (self.ell_w, self.ell_b))             # [N,Dmax,1]
        edge_pred = ttnn.multiply(edge_pred_pre, cutoff_factors)           # zero on padded
        edge_pred = ttnn.sum(edge_pred, dim=1)                              # [N,1]
        # cache pre-silu activations + pre-multiply edge_pred for pet_vjp._energy_head_bw
        self._cache = dict(a_n0=a_n0, a_n2=a_n2, a_e0=a_e0, a_e2=a_e2,
                           edge_pred_pre=edge_pred_pre,
                           node_feat_shape=tuple(node_feat.shape),
                           edge_feat_shape=tuple(edge_feat.shape))
        return ttnn.sum(ttnn.add(node_pred, edge_pred))                    # scalar


def _host_pet_inputs(bd, cfg):
    """Host-side computation of the device-input tensors (the part of
    :func:`build_device_inputs` that does not touch the device). Returns a dict of host
    tensors: the topology-fixed index tables (``node_idx``, ``elem_nbr``, ``rev_idx`` as
    int32; ``rev_idx_host`` as long) and the pos-dependent feature buffers
    (``edge_vec_cat``, ``cutoff_factors``, ``log_mask`` as float32/bf16-ready), plus the
    three host differentiable copies (``*_host``) that are leaves for the device-VJP host
    finish. Factored out so the trace engine (:mod:`tt_atom.pet_trace`) can refresh only the
    pos-dependent buffers in place without re-uploading the topology tables."""
    N = int(bd["num_nodes"])
    Dmax = int(bd["max_edges_per_node"])
    S = 1 + Dmax
    num_heads = int(cfg["num_heads"])
    cf_dtype = torch.float32

    elem_nodes = bd["element_indices_nodes"].long()
    elem_nbr = bd["element_indices_neighbors"].long()                     # [N, Dmax]
    edge_vec = bd["edge_vectors"]                                           # [N, Dmax, 3]
    edge_dist = bd["edge_distances"]                                        # [N, Dmax]
    cutoff_factors = bd["cutoff_factors"]                                   # [N, Dmax]
    padding_mask = bd["padding_mask"]                                       # [N, Dmax] bool
    rev_idx = bd["reverse_neighbor_index"].long()                          # [N, Dmax]

    edge_vec_cat = torch.cat([edge_vec, edge_dist[..., None]], dim=-1).contiguous()  # [N,Dmax,4]

    # attention cutoff mask (host, exact replica of pet_model_host.cartesian_transformer)
    cutoff_sub = torch.ones(N, dtype=cf_dtype)
    cf = torch.cat([cutoff_sub[:, None], cutoff_factors.to(cf_dtype)], dim=1)      # [N, S]
    total_mask = torch.cat([torch.ones(N, dtype=torch.bool)[:, None], padding_mask], dim=1)
    cf = cf.clone()
    cf[~total_mask] = 0.0
    cf = cf[:, None, :].expand(N, S, S).contiguous()                        # [N, S, S]
    log_mask = torch.log(cf.clamp(min=1e-15))                               # [N, S, S]
    log_mask = log_mask.unsqueeze(1).expand(N, num_heads, S, S).reshape(N * num_heads, S, S).contiguous()

    return dict(
        node_idx=elem_nodes.to(torch.int32),
        elem_nbr=elem_nbr.reshape(-1).to(torch.int32),
        rev_idx=rev_idx.reshape(-1).to(torch.int32),
        rev_idx_host=rev_idx.reshape(-1).contiguous(),                      # host long, for pet_vjp NEF scatter
        edge_vec_cat=edge_vec_cat.detach().contiguous(),
        cutoff_factors=cutoff_factors.to(torch.float32)[..., None].contiguous(),
        log_mask=log_mask.detach().contiguous(),
        # host-side differentiable copies (kept in the autograd graph when pos requires
        # grad) for the device-VJP host finish in pet_forces.device_energy_and_forces.
        edge_vec_cat_host=edge_vec_cat,
        cutoff_factors_host=cutoff_factors.to(torch.float32)[..., None].contiguous(),
        log_mask_host=log_mask,
        N=N, Dmax=Dmax,
    )


def build_device_inputs(bd, cfg, device):
    """Upload the host geometry's NEF tensors as the device-resident per-topology buffers
    the backbone consumes — the device analogue of Orb's ``OrbGraphContext``. All
    pos-dependent fixed terms (edge vectors/distances, cutoff factors, the log-additive
    attention mask, the NEF index tables) are host-computed by ``pet_geometry`` and
    uploaded once; every learned op runs on device.

    Returns a dict of device tensors. ``log_mask`` is ``log(clamp(cf, 1e-15))`` broadcast
    over heads (``cf`` built exactly as ``pet_model_host.cartesian_transformer``: prepend
    the center-token 1, zero the padded slots, broadcast to [N, S, S])."""
    import ttnn

    h = _host_pet_inputs(bd, cfg)
    rm = ttnn.ROW_MAJOR_LAYOUT
    return dict(
        node_idx=_to_dev(h["node_idx"], device, ttnn.uint32, rm),
        elem_nbr=_to_dev(h["elem_nbr"], device, ttnn.uint32, rm),
        edge_vec_cat=_to_dev(h["edge_vec_cat"], device, ttnn.bfloat16),
        cutoff_factors=_to_dev(h["cutoff_factors"], device, ttnn.bfloat16),
        log_mask=_to_dev(h["log_mask"], device, ttnn.bfloat16),
        rev_idx=_to_dev(h["rev_idx"], device, ttnn.uint32, rm),
        rev_idx_host=h["rev_idx_host"],
        edge_vec_cat_host=h["edge_vec_cat_host"],
        cutoff_factors_host=h["cutoff_factors_host"],
        log_mask_host=h["log_mask_host"],
        N=h["N"], Dmax=h["Dmax"],
    )


class PetModel:
    """Full device PET-MAD backbone: node/edge embeddings -> ``num_gnn_layers`` ×
    ``GnnLayer`` -> ``EnergyHead``. Returns the raw (pre-scaler, pre-composition) energy
    scalar; the caller applies ``E = raw * scale + sum_i comp[Z_i]``
    (``tt_atom/pet_weights.py``)."""

    def __init__(self, weights, device, *, cfg):
        import ttnn

        self.ttnn = ttnn
        self.cfg = cfg
        self.device = device
        self.node_emb_w = _to_dev(weights["node_embedders.0.weight"], device, ttnn.bfloat16)
        self.edge_emb_w = _to_dev(weights["edge_embedder.weight"], device, ttnn.bfloat16)
        self.layers = [GnnLayer(weights, device, cfg=cfg, layer_idx=i)
                       for i in range(int(cfg["num_gnn_layers"]))]
        self.energy_head = EnergyHead(weights, device)

    def forward(self, bd_dev, *, return_layers=False):
        """``bd_dev`` = the dict from :func:`build_device_inputs`. Returns the raw energy
        scalar (device tensor). If ``return_layers``, returns ``(raw_energy, [node_out per
        layer], [edge_out per layer])`` for per-layer PCC gating against the canonical
        fixture."""
        ttnn = self.ttnn
        N = bd_dev["N"]
        Dmax = bd_dev["Dmax"]

        node_emb = ttnn.embedding(bd_dev["node_idx"], self.node_emb_w)       # [N, d_node]
        input_edge = ttnn.embedding(bd_dev["elem_nbr"], self.edge_emb_w)     # [N*Dmax, d_pet]
        input_edge = ttnn.reshape(input_edge, (N, Dmax, self.cfg["d_pet"]))

        node_outs, edge_outs = [], []
        for layer in self.layers:
            out_node, out_edge_pre, input_edge = layer(
                node_emb, input_edge, bd_dev["edge_vec_cat"], bd_dev["elem_nbr"],
                bd_dev["log_mask"], bd_dev["rev_idx"],
                rev_idx_host=bd_dev.get("rev_idx_host"))
            node_emb = out_node
            if return_layers:
                node_outs.append(ttnn.to_torch(out_node).float().cpu())
                edge_outs.append(ttnn.to_torch(out_edge_pre).float().cpu())

        raw = self.energy_head(node_emb, input_edge, bd_dev["cutoff_factors"])
        if return_layers:
            return raw, node_outs, edge_outs
        return raw

    def backward(self, bd_dev, g_raw=1.0):
        """Reverse-mode VJP through the forward just run. Returns ``(g_edge_vec_cat,
        g_cutoff_factors, g_log_mask)`` -- the device adjoints at the three pos-dependent
        uploaded inputs (``edge_vec_cat``, ``cutoff_factors``, ``log_mask``), ready for the
        host ``torch.autograd.grad`` finish in ``pet_forces.device_energy_and_forces``.
        ``g_raw`` is the scalar seed at the raw energy (1.0 for forces)."""
        from .pet_vjp import backbone_bw
        return backbone_bw(self, bd_dev, g_raw=g_raw)
