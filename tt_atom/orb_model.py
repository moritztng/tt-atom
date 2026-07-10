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
        self._cache_x, self._cache_inv = x, inv          # reused by orb_forces.rmsnorm_bw
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
        a0 = ttnn.linear(x, self.w[0], bias=self.b[0], compute_kernel_config=self.kcfg)
        h0 = ttnn.silu(a0)
        a1 = ttnn.linear(h0, self.w[1], bias=self.b[1], compute_kernel_config=self.kcfg)
        h1 = ttnn.silu(a1)
        h2 = ttnn.linear(h1, self.w[2], bias=self.b[2], compute_kernel_config=self.kcfg)
        self._cache_a0, self._cache_a1 = a0, a1           # pre-SiLU activations, for orb_forces.mlpnorm_bw
        return self.norm(h2)


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

        ra_lin = ttnn.linear(edges, self.receive_attn_w, bias=self.receive_attn_b,
                             compute_kernel_config=self.kcfg)
        sa_lin = ttnn.linear(edges, self.send_attn_w, bias=self.send_attn_b,
                             compute_kernel_config=self.kcfg)
        ra_sig = ttnn.sigmoid(ra_lin)
        sa_sig = ttnn.sigmoid(sa_lin)
        receive_attn = ttnn.multiply(ra_sig, graph.cutoff)
        send_attn = ttnn.multiply(sa_sig, graph.cutoff)

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

        # cached forward tensors, reused 1:1 by orb_forces.attn_layer_bw
        self._cache = dict(ra_lin=ra_lin, sa_lin=sa_lin, ra_sig=ra_sig, sa_sig=sa_sig,
                           receive_attn=receive_attn, send_attn=send_attn,
                           updated_edges=updated_edges)
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


class EnergyHead:
    """``forcefield_heads.EnergyHead``'s device-resident MLP path: mean-aggregate the final
    node embedding over the system, then a 2-layer MLP (Linear-SiLU-Linear) to one
    normalized-space scalar per system. Single-system only (no disjoint-union batch dim yet).

    The normalize/denormalize affine + per-element reference-energy lookup are a handful of
    fixed scalars / a 118-row table -- computed on host, see ``host_energy_denormalize``,
    exactly like UMA's ``scale_rmsd``/``scale_mean``/``elem_refs`` (``tt_atom/weights.py``).
    """

    def __init__(self, weights, device, *, latent_dim, hidden_dim):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.kcfg = compute_kernel_config()
        self.w0 = _to_dev(weights["energy_head.mlp.NN-0.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b0 = _to_dev(weights["energy_head.mlp.NN-0.bias"], device, ttnn.bfloat16)
        self.w1 = _to_dev(weights["energy_head.mlp.NN-1.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b1 = _to_dev(weights["energy_head.mlp.NN-1.bias"], device, ttnn.bfloat16)

    def __call__(self, node_features):
        """``node_features``: ttnn ``[N, latent_dim]`` (single system) -> ttnn ``[1, 1]`` raw
        (normalized-space) energy prediction."""
        ttnn = self.ttnn
        N = node_features.shape[0]
        mean = ttnn.mean(node_features, dim=0, keepdim=True)
        a0 = ttnn.linear(mean, self.w0, bias=self.b0, compute_kernel_config=self.kcfg)
        h = ttnn.silu(a0)
        self._cache_a0, self._cache_N = a0, N             # for orb_forces.energy_bw
        return ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.kcfg)

    def batch(self, node_features, seg_mean):
        """Disjoint-union batched readout: ``seg_mean`` [K, Ntot] is a *row-normalized* segment
        matrix (``seg_mean[k, n] = 1/count_k`` iff atom n is in system k, else 0) -- unlike UMA's
        ``Backbone.energy_batch`` (a per-node scalar energy, segment-*summed* per system), Orb's
        ``EnergyHead`` means the node *features* first and only then runs the 2-layer MLP, so the
        adapter is a per-system mean (matmul against ``seg_mean``) feeding the same MLP, batched
        over the K systems -> ``[K, 1]`` raw (normalized-space) energy predictions."""
        ttnn = self.ttnn
        mean = ttnn.matmul(seg_mean, node_features, compute_kernel_config=self.kcfg)  # [K, C]
        h = ttnn.silu(ttnn.linear(mean, self.w0, bias=self.b0, compute_kernel_config=self.kcfg))
        return ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.kcfg)


class ForceHead:
    """``forcefield_heads.ForceHead``'s device-resident MLP path (used by
    ``orb-v3-direct-20-omat``, which predicts forces directly with no energy-autograd VJP):
    a per-node 2-layer MLP (Linear-SiLU-Linear) on the final node embedding -> ``[N, 3]``, then
    net-force removal (subtract the per-system mean predicted force across all nodes -- a fixed
    geometric correction, no learned params). The scalar normalizer inverse (``* sigma + mu``)
    is applied on host, mirroring ``EnergyHead``/UMA's scale convention.

    Single-system only; ``remove_torque_for_nonpbc_systems`` is skipped since it only fires for
    non-periodic (zero-cell) systems -- the ported Si golden is fully periodic.
    """

    def __init__(self, weights, device, *, latent_dim, hidden_dim):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.w0 = _to_dev(weights["forces_head.mlp.NN-0.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b0 = _to_dev(weights["forces_head.mlp.NN-0.bias"], device, ttnn.bfloat16)
        self.w1 = _to_dev(weights["forces_head.mlp.NN-1.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b1 = _to_dev(weights["forces_head.mlp.NN-1.bias"], device, ttnn.bfloat16)

    def __call__(self, node_features):
        """``node_features``: ttnn ``[N, latent_dim]`` -> ttnn ``[N, 3]`` raw (normalized-space,
        mean-removed) per-atom force prediction."""
        ttnn = self.ttnn
        h = ttnn.silu(ttnn.linear(node_features, self.w0, bias=self.b0, compute_kernel_config=self.kcfg))
        pred = ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.kcfg)
        mean = ttnn.mean(pred, dim=0, keepdim=True)
        return ttnn.subtract(pred, mean)


class StressHead:
    """``forcefield_heads.StressHead``'s device-resident MLP path (used by
    ``orb-v3-direct-20-omat``): mean-aggregate the final node embedding over the system (same
    shape/pattern as ``EnergyHead``), then a 2-layer MLP (Linear-SiLU-Linear) to a 6-vector
    (Voigt notation: ``[xx, yy, zz, yz, xz, xy]``) raw prediction. Single-system only.

    Unlike ``EnergyHead``'s single scalar normalizer, the diagonal and off-diagonal components
    have their own ``ScalarNormalizer``s (``host_stress_denormalize`` applies both) -- direct's
    stress has no explicit volume division (unlike the conservative virial): the normalizer
    stats were fit directly against the target's own eV/Å^3 units."""

    def __init__(self, weights, device, *, latent_dim, hidden_dim):
        import ttnn

        self.ttnn = ttnn
        self.kcfg = compute_kernel_config()
        self.w0 = _to_dev(weights["stress_head.mlp.NN-0.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b0 = _to_dev(weights["stress_head.mlp.NN-0.bias"], device, ttnn.bfloat16)
        self.w1 = _to_dev(weights["stress_head.mlp.NN-1.weight"].T.contiguous(), device, ttnn.bfloat16)
        self.b1 = _to_dev(weights["stress_head.mlp.NN-1.bias"], device, ttnn.bfloat16)

    def __call__(self, node_features):
        """``node_features``: ttnn ``[N, latent_dim]`` (single system) -> ttnn ``[1, 6]`` raw
        (normalized-space) Voigt-6 stress prediction."""
        ttnn = self.ttnn
        mean = ttnn.mean(node_features, dim=0, keepdim=True)
        h = ttnn.silu(ttnn.linear(mean, self.w0, bias=self.b0, compute_kernel_config=self.kcfg))
        return ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.kcfg)


def host_force_denormalize(raw_pred: torch.Tensor, *, running_mean: torch.Tensor,
                           running_var: torch.Tensor) -> torch.Tensor:
    """``ForceHead``'s ``ScalarNormalizer.inverse``: ``x * sigma + mu`` (two learned scalars)."""
    sigma = running_var.double().sqrt()
    return raw_pred.double() * sigma + running_mean.double()


def host_stress_denormalize(raw_pred: torch.Tensor, *, diag_mean: torch.Tensor,
                            diag_var: torch.Tensor, offdiag_mean: torch.Tensor,
                            offdiag_var: torch.Tensor) -> torch.Tensor:
    """``StressHead``'s ``denormalize``: the diagonal (``raw_pred[..., :3]``) and off-diagonal
    (``raw_pred[..., 3:]``) Voigt components each go through their own ``ScalarNormalizer``
    (``x * sigma + mu``), concatenated back into one Voigt-6 vector -- no volume division (unlike
    the conservative virial), since these normalizer stats were fit directly against the eV/Å^3
    target."""
    raw = raw_pred.double().reshape(-1, 6)
    diag = raw[:, :3] * diag_var.double().sqrt() + diag_mean.double()
    offdiag = raw[:, 3:] * offdiag_var.double().sqrt() + offdiag_mean.double()
    return torch.cat([diag, offdiag], dim=-1).reshape(raw_pred.shape)


def host_conservative_force_denormalize(raw_forces: torch.Tensor, n_node: int, *,
                                        running_var: torch.Tensor) -> torch.Tensor:
    """Chain-rule scale for ``orb-v3-conservative-inf-omat``'s analytic forces
    (``tt_atom/orb_forces.py``): since ``E_real = (raw_pred*sigma + mu) * n_node + ref_sum`` and
    ``forces = -dE_real/dpos``, the additive ``mu``/``ref_sum`` terms (pos-independent) vanish
    under differentiation and only the multiplicative ``sigma * n_node`` factor survives -- unlike
    ``host_force_denormalize`` (direct-20's ``ForceHead``, its own additive per-component
    normalizer), this reuses the *energy* head's normalizer, the same one
    ``host_energy_denormalize`` uses."""
    sigma = running_var.double().sqrt()
    return raw_forces.double() * sigma * n_node


def host_conservative_stress(virial: torch.Tensor, n_node: int, cell: torch.Tensor, *,
                             running_var: torch.Tensor) -> torch.Tensor:
    """``orb-v3-conservative-inf-omat``'s stress tensor from the raw virial
    (``dE_raw/dstrain``, ``tt_atom/orb_forces.py``'s ``energy_and_forces(..., compute_stress=True)``):
    the same ``sigma * n_node`` chain-rule scale as ``host_conservative_force_denormalize`` (the
    additive ``mu``/``ref_sum`` terms vanish under differentiation regardless of whether the
    derivative is wrt position or strain), then ``stress = virial / volume`` -- Orb's own
    convention (``forcefield_utils.compute_gradient_forces_and_stress``, no extra sign flip,
    unlike forces) -- converted to Voigt-6 ``[xx, yy, zz, yz, xz, yx]`` (off-diagonal terms
    averaged, matching ``torch_full_3x3_to_voigt_6_stress``)."""
    sigma = running_var.double().sqrt()
    volume = torch.linalg.det(cell.double()).abs()
    stress = virial.double() * sigma * n_node / volume
    s01 = 0.5 * (stress[0, 1] + stress[1, 0])
    s02 = 0.5 * (stress[0, 2] + stress[2, 0])
    s12 = 0.5 * (stress[1, 2] + stress[2, 1])
    return torch.stack([stress[0, 0], stress[1, 1], stress[2, 2], s12, s02, s01])


def host_zbl_energy(atomic_numbers: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                    vectors: torch.Tensor, *, p: int = 6) -> torch.Tensor:
    """Ziegler-Biersack-Littmark pair-repulsion energy (``pair_repulsion.ZBLBasis``) -- a fixed
    physical potential (6 universal constants, no learned weights) computed on host directly
    from real atomic numbers + edge vectors, exactly like the attention cutoff envelope.
    Single-system only; returns the mean-aggregated (BC override, see ``pretrained.py``'s
    ``pair_repulsion_fn.node_aggregation = "mean"``) scalar energy to add to the GNN energy.
    """
    import ase.data

    c = torch.tensor([0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.float64).unsqueeze(1)
    d = torch.tensor([3.2, 0.9423, 0.4028, 0.2016], dtype=torch.float64).unsqueeze(1)
    a_exp, a_prefactor = 0.300, 0.4543
    covalent_radii = torch.tensor(ase.data.covalent_radii, dtype=torch.float64)

    Z_u = atomic_numbers[senders].double() + 1
    Z_v = atomic_numbers[receivers].double() + 1
    a = a_prefactor * 0.529 / (Z_u.pow(a_exp) + Z_v.pow(a_exp))

    x = vectors.double().norm(dim=1)
    r_over_a = x / a
    exp_term = torch.exp(-d * r_over_a.unsqueeze(0))
    phi = (c * exp_term).sum(dim=0)

    coulomb_term = 14.3996 * Z_u * Z_v / x
    v_edges_raw = coulomb_term * phi

    r_max = (covalent_radii[atomic_numbers[senders].long()]
             + covalent_radii[atomic_numbers[receivers].long()])
    r_ratio = x / r_max
    mask = (x < r_max).double()
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * r_ratio.pow(p)
        + p * (p + 2.0) * r_ratio.pow(p + 1)
        - (p * (p + 1.0) / 2.0) * r_ratio.pow(p + 2)
    ) * mask

    v_edges = 0.5 * v_edges_raw * envelope
    N = atomic_numbers.shape[0]
    V_ZBL = torch.zeros(N, dtype=torch.float64).index_add_(0, senders, v_edges)
    return V_ZBL.mean()


def host_zbl_forces(atomic_numbers: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                    pos: torch.Tensor, cell_shift: torch.Tensor | None = None) -> torch.Tensor:
    """``dV_ZBL/dr`` via host ``torch.autograd`` on the same closed-form ``host_zbl_energy`` --
    ZBL has zero learned parameters, so (unlike the GNN backbone's device VJP,
    ``tt_atom/orb_forces.py``) there is no device backward to write here at all; this is the
    "straightforward" alternative flagged in ``docs/orb-port.md`` (vs. hand-deriving the closed-
    form ``pair_repulsion.ZBLBasis._polynomial_cutoff_with_derivative``). Needed for
    ``orb-v3-direct-20-omat``'s *total* force whenever ZBL is non-negligible (short contacts,
    surface defects) -- its ``ForceHead`` MLP prediction has no ZBL contribution baked in."""
    pos = pos.detach().clone().double().requires_grad_(True)
    vectors = pos[receivers] - pos[senders]
    if cell_shift is not None:
        vectors = vectors + cell_shift
    energy = host_zbl_energy(atomic_numbers, senders, receivers, vectors)
    return -torch.autograd.grad(energy, pos)[0]


def host_energy_denormalize(raw_pred: torch.Tensor, atomic_numbers: torch.Tensor, n_node: int, *,
                            running_mean: torch.Tensor, running_var: torch.Tensor,
                            ref_weight: torch.Tensor) -> torch.Tensor:
    """``EnergyHead.denormalize``: undo the learned scalar-normalizer affine, undo the
    atom-average, add the per-element linear reference energy. All fixed/tiny (a handful of
    scalars + a 118-length lookup table) -- computed on host, exactly like UMA's
    ``scale_rmsd``/``scale_mean``/``elem_refs`` normalizer (``tt_atom/weights.py``).
    """
    sigma = running_var.double().sqrt()
    x = raw_pred.double() * sigma + running_mean.double()
    x = x * n_node
    ref = ref_weight.double()[atomic_numbers.long()].sum()
    return x + ref
