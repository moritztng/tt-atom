"""Standalone pure-torch reference backbone for PET-MAD (UPET, pet-mad-s v1.5.0).

A from-scratch reimplementation of ``metatrain.pet.modules.transformer`` +
``metatrain.pet.model.PET._feedforward_featurization_impl`` + the energy readout, with NO
dependency on metatrain/metatomic/metatensor — so it runs in the ttnn env (numpy<2) exactly
like ``tt_atom/orb_model.py``'s host helpers do for Orb.

Purpose: (1) a verified reference forward (PCC 1.0 vs the real PET, captured under
forward hooks) that the device backbone (``tt_atom/pet_model.py``, next pass) is gated
against; (2) the host-side skeleton whose ops get ported 1:1 to ttnn; (3) the autograd
graph the conservative-force VJP (``tt_atom/pet_forces.py``, next pass) finishes on host
through ``pet_geometry``'s differentiable edge featurization.

Everything operates on plain ``[N, Dmax, *]`` NEF tensors (see ``pet_geometry``). The
attention is PET's ``manual_attention`` (softmax with a log-additive cutoff mask), the
one non-trivial op vs Orb's sigmoid-gated message passing.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rmsnorm(x, weight, eps=1e-6):
    """``nn.RMSNorm`` (elementwise-affine, no bias) over the last dim."""
    ms = (x * x).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + eps)
    return (x * inv) * weight


def layernorm(x, weight, bias, eps=1e-5):
    """``nn.LayerNorm`` (elementwise-affine, with bias) over the last dim — used by
    PET's ``combination_norms`` (LayerNorm(2*d_pet)). The transformer norms use
    ``rmsnorm`` (normalization="RMSNorm")."""
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) * (x - mean)).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + eps)
    return (x - mean) * inv * weight + bias


def feedforward(x, w_in_weight, w_in_bias, w_out_weight, w_out_bias, is_swiglu):
    """``FeedForward`` (metatrain.pet.modules.transformer). SwiGLU: ``w_in`` produces
    ``[2*dim_ff]``, split into value & gate, ``v * sigmoid(g)``; then ``w_out``. Non-SwiGLU
    (SiLU): ``w_in`` -> activation -> ``w_out``. PET-MAD v1.5.0 uses SwiGLU everywhere."""
    if is_swiglu:
        z = F.linear(x, w_in_weight, w_in_bias)
        v, g = z.chunk(2, dim=-1)
        h = v * torch.sigmoid(g)
    else:
        h = F.silu(F.linear(x, w_in_weight, w_in_bias))
    return F.linear(h, w_out_weight, w_out_bias)


def attention(x, cutoff_factors, *, num_heads, temperature, input_linear_weight, input_linear_bias,
              output_linear_weight, output_linear_bias, epsilon=1e-15):
    """``AttentionBlock`` (manual_attention path, the one PET-MAD uses at eval).

    ``x`` [N, seq, d_model=256]; ``cutoff_factors`` [N, seq, seq] — the log-additive
    attention mask (PET clamps the mask at ``epsilon`` before taking ``log``). Returns
    [N, seq, 256]. ``temperature`` is PET's extra scale on top of the standard
    ``1/sqrt(head_dim)``.

    Layout: ``input_linear`` -> [N, seq, 3*d] -> reshape [N, seq, 3, heads, head_dim] ->
    permute to [3, N, heads, seq, head_dim] -> Q, K, V. Scores = QK^T / (sqrt(head_dim)*temp)
    + log(mask); softmax over last axis; @ V; transpose+reshape back; ``output_linear``.
    """
    N, seq, d = x.shape
    head_dim = d // num_heads
    z = F.linear(x, input_linear_weight, input_linear_bias)  # [N, seq, 3*d]
    z = z.reshape(N, seq, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # [3, N, heads, seq, head_dim]
    q, k, v = z[0], z[1], z[2]  # each [N, heads, seq, head_dim]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (math.sqrt(head_dim) * temperature)
    mask = cutoff_factors.clamp(min=epsilon)  # [N, seq, seq]
    attn_weights = attn_weights + torch.log(mask).unsqueeze(1)  # broadcast over heads -> [N, 1, seq, seq]
    attn = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn, v)  # [N, heads, seq, head_dim]
    out = out.transpose(1, 2).reshape(N, seq, d)  # [N, seq, 256]
    return F.linear(out, output_linear_weight, output_linear_bias)


def transformer_layer(node_emb, edge_emb, cutoff_factors, w, *, d_model, d_node, num_heads,
                      temperature, is_first, prefix):
    """``TransformerLayer._forward_pre_ln_impl`` (PreLN, the only type PET-MAD v1.5.0 uses).

    ``node_emb`` [N, 1, d_node=1024], ``edge_emb`` [N, Dmax, d_model=256],
    ``cutoff_factors`` [N, 1+Dmax, 1+Dmax]. ``w`` is the full state_dict; ``prefix`` is the
    layer's full key prefix, e.g. ``gnn_layers.0.trans.layers.0.``. Returns
    (out_node [N, 1, 1024], out_edge [N, Dmax, 256]).

    PET-MAD has ``expanded_node_features=True`` (d_node=1024 != d_model=256), so the
    center path (contraction -> attention -> expansion -> center_mlp) is always active.
    """
    p = prefix
    # center contraction: 1024 -> 256
    input_node = F.linear(node_emb, w[f"{p}center_contraction.weight"], w[f"{p}center_contraction.bias"])
    tokens = torch.cat([input_node, edge_emb], dim=1)  # [N, 1+Dmax, 256]
    normed = rmsnorm(tokens, w[f"{p}norm_attention.weight"])
    new_tokens = attention(normed, cutoff_factors, num_heads=num_heads, temperature=temperature,
                            input_linear_weight=w[f"{p}attention.input_linear.weight"],
                            input_linear_bias=w[f"{p}attention.input_linear.bias"],
                            output_linear_weight=w[f"{p}attention.output_linear.weight"],
                            output_linear_bias=w[f"{p}attention.output_linear.bias"])
    out_node_tok, out_edge = torch.split(new_tokens, [1, new_tokens.shape[1] - 1], dim=1)
    # center expansion: 256 -> 1024, then + center_mlp(rmsnorm(node_emb + expansion(out_node)))
    out_node = node_emb + F.linear(out_node_tok, w[f"{p}center_expansion.weight"], w[f"{p}center_expansion.bias"])
    out_node = out_node + feedforward(rmsnorm(out_node, w[f"{p}norm_center_features.weight"]),
                                       w[f"{p}center_mlp.w_in.weight"], w[f"{p}center_mlp.w_in.bias"],
                                       w[f"{p}center_mlp.w_out.weight"], w[f"{p}center_mlp.w_out.bias"], is_swiglu=True)
    out_edge = edge_emb + out_edge
    out_edge = out_edge + feedforward(rmsnorm(out_edge, w[f"{p}norm_mlp.weight"]),
                                       w[f"{p}mlp.w_in.weight"], w[f"{p}mlp.w_in.bias"],
                                       w[f"{p}mlp.w_out.weight"], w[f"{p}mlp.w_out.bias"], is_swiglu=True)
    return out_node, out_edge


def cartesian_transformer(input_node, input_edge, elem_nbr, edge_vec, padding_mask, edge_dist,
                           cutoff_factors, w, *, cfg, layer_idx):
    """``CartesianTransformer.forward``. ``layer_idx=0`` is the first layer (no
    neighbor_embedder, compress takes 2*d_pet). Returns (out_node [N, 1024],
    out_edge [N, Dmax, 256])."""
    p = f"gnn_layers.{layer_idx}."
    is_first = (layer_idx == 0)
    d_pet = cfg["d_pet"]
    # edge embedding: Linear(4, d_pet) on cat([edge_vec, edge_dist[..., None]])
    edge_emb = torch.cat([edge_vec, edge_dist[..., None]], dim=-1)  # [N, Dmax, 4]
    edge_emb = F.linear(edge_emb, w[f"{p}edge_embedder.weight"], w[f"{p}edge_embedder.bias"])
    if is_first:
        edge_tokens = torch.cat([edge_emb, input_edge], dim=-1)  # [N, Dmax, 2*d_pet]
    else:
        nbr_emb = F.embedding(elem_nbr, w[f"{p}neighbor_embedder.weight"])  # [N, Dmax, d_pet]
        edge_tokens = torch.cat([edge_emb, nbr_emb, input_edge], dim=-1)  # [N, Dmax, 3*d_pet]
    # compress: Sequential(Linear(n_merge*d_pet, d_pet), SiLU, Linear(d_pet, d_pet))
    h = F.silu(F.linear(edge_tokens, w[f"{p}compress.0.weight"], w[f"{p}compress.0.bias"]))
    edge_tokens = F.linear(h, w[f"{p}compress.2.weight"], w[f"{p}compress.2.bias"])  # [N, Dmax, d_pet]

    # build the attention cutoff mask: prepend a 1 (center token), zero padded slots, broadcast to [N, seq, seq]
    N = edge_vec.shape[0]
    cutoff_sub = torch.ones(N, dtype=cutoff_factors.dtype, device=cutoff_factors.device)
    cf = torch.cat([cutoff_sub[:, None], cutoff_factors], dim=1)  # [N, 1+Dmax]
    total_mask = torch.cat([torch.ones(N, dtype=torch.bool, device=padding_mask.device)[:, None], padding_mask], dim=1)
    cf = cf.clone(); cf[~total_mask] = 0.0
    cf = cf[:, None, :].repeat(1, cf.shape[1], 1)  # [N, 1+Dmax, 1+Dmax]

    out_node, out_edge = transformer_layer(
        input_node[:, None, :], edge_tokens, cf, w, d_model=d_pet, d_node=cfg["d_node"],
        num_heads=cfg["num_heads"], temperature=cfg["attention_temperature"], is_first=is_first,
        prefix=f"{p}trans.layers.0.")
    out_node = out_node.squeeze(1)  # [N, 1024]
    return out_node, out_edge


def feedforward_featurization(bd, w, *, cfg):
    """``PET._feedforward_featurization_impl`` (featurizer_type="feedforward", the only
    type pet-mad-s v1.5.0 uses). Runs all ``num_gnn_layers`` GNN layers, mixing forward &
    reversed edge messages at each layer via the combination MLP. Returns the final
    (node_features [N, d_node], edge_features [N, Dmax, d_pet]) — single-element lists
    in the reference (feedforward keeps only the last layer), returned as bare tensors
    here for convenience."""
    elem_nodes = bd["element_indices_nodes"]
    elem_nbr = bd["element_indices_neighbors"]
    edge_vec = bd["edge_vectors"]; edge_dist = bd["edge_distances"]
    cutoff_factors = bd["cutoff_factors"]; padding_mask = bd["padding_mask"]
    rev_idx = bd["reverse_neighbor_index"]

    input_node = F.embedding(elem_nodes, w["node_embedders.0.weight"])  # [N, d_node]
    input_edge = F.embedding(elem_nbr, w["edge_embedder.weight"])  # [N, Dmax, d_pet]
    for i in range(cfg["num_gnn_layers"]):
        out_node, out_edge = cartesian_transformer(
            input_node, input_edge, elem_nbr, edge_vec, padding_mask, edge_dist, cutoff_factors,
            w, cfg=cfg, layer_idx=i)
        # reversed edge messages: flatten, index by reverse_neighbor_index, reshape
        new_input_edge = out_edge.reshape(-1, out_edge.shape[-1])[rev_idx].reshape(out_edge.shape)
        concatenated = torch.cat([out_edge, new_input_edge], dim=-1)  # [N, Dmax, 2*d_pet]
        # combination: LayerNorm(2*d_pet) -> Sequential(Linear(2*d_pet, 2*d_pet), SiLU, Linear(2*d_pet, d_pet))
        normed = layernorm(concatenated, w[f"combination_norms.{i}.weight"], w[f"combination_norms.{i}.bias"])
        h = F.silu(F.linear(normed, w[f"combination_mlps.{i}.0.weight"], w[f"combination_mlps.{i}.0.bias"]))
        comb = F.linear(h, w[f"combination_mlps.{i}.2.weight"], w[f"combination_mlps.{i}.2.bias"])
        input_node = out_node
        input_edge = input_edge + out_edge + comb
    return input_node, input_edge


def energy_raw(node_feat, edge_feat, padding_mask, cutoff_factors, w, *, cfg):
    """``PET._calculate_last_layer_features`` + ``_calculate_atomic_predictions`` for the
    energy target only. ``node_heads[energy][0]`` (Linear-SiLU-Linear-SiLU) on node features
    -> ``node_last_layers[energy][0][key]`` (Linear(256, 1)); same for edges, then mask +
    cutoff-weighted sum over neighbors. Returns the raw (pre-scaler, pre-composition)
    per-structure energy scalar. PET-MAD energy has a single block/key, so the key is
    fixed to ``energy___0``."""
    key = "energy___0"
    node_llf = F.silu(F.linear(node_feat, w["node_heads.energy.0.0.weight"], w["node_heads.energy.0.0.bias"]))
    node_llf = F.silu(F.linear(node_llf, w["node_heads.energy.0.2.weight"], w["node_heads.energy.0.2.bias"]))
    node_pred = F.linear(node_llf, w[f"node_last_layers.energy.0.{key}.weight"], w[f"node_last_layers.energy.0.{key}.bias"])

    edge_llf = F.silu(F.linear(edge_feat, w["edge_heads.energy.0.0.weight"], w["edge_heads.energy.0.0.bias"]))
    edge_llf = F.silu(F.linear(edge_llf, w["edge_heads.energy.0.2.weight"], w["edge_heads.energy.0.2.bias"]))
    edge_pred = F.linear(edge_llf, w[f"edge_last_layers.energy.0.{key}.weight"], w[f"edge_last_layers.energy.0.{key}.bias"])
    edge_pred = torch.where(~padding_mask[..., None], 0.0, edge_pred)
    edge_pred = (edge_pred * cutoff_factors[..., None]).sum(dim=1)  # [N, 1]
    return (node_pred + edge_pred).sum()


def forward_energy(bd, w, *, cfg):
    """Full host forward: geometry -> GNN -> raw energy. Returns the raw (pre-scaler,
    pre-composition) energy scalar — the caller applies ``E = raw * scale + sum_i comp[Z_i]``
    (see ``tools/export_pet_weights.py``)."""
    node_feat, edge_feat = feedforward_featurization(bd, w, cfg=cfg)
    return energy_raw(node_feat, edge_feat, bd["padding_mask"], bd["cutoff_factors"], w, cfg=cfg)



