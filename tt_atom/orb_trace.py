"""Device-resident, trace-captured energy+forces for Orb-v3 at a FIXED topology (MD / relaxation).

Shares the capture/refresh/replay skeleton with the UMA engine via ``trace.TraceEngineBase`` (the
idea is architecture-agnostic: capture the forward+backward op stream once, refresh only the
pos-dependent input buffers in place, replay with zero per-op host dispatch). Only the family-
specific hooks differ: UMA refreshes its equivariant machinery (Wigner rotation coefficients,
``rotation.gather_coef``, ``GraphContext``'s SO(3) buffers), none of which exists for Orb's plain
non-equivariant backbone (see ``tt_atom/orb_model.py``). This supplies those hooks for Orb.

For a fixed topology (``senders``/``receivers``/``cell_shift``), the only pos-dependent device
inputs each step are ``edge_feat`` (the encoder's edge-MLP input) and ``cutoff`` (the attention
gate envelope) -- both produced by ``orb_geometry.host_edge_features`` -- exactly the two
adjoint targets ``orb_forces.energy_and_forces`` differentiates through on host. Both are
refreshed in place via ``copy_host_to_device_tensor`` (the only operand that can overwrite a
resident, trace-captured buffer); the atomic-number-only ``node_feat`` has no ``pos`` dependence
and is uploaded once, never refreshed.

Two modes, selected by which head(s) are given:
  * ``ehead`` only (conservative, e.g. ``orb-v3-conservative-inf-omat``): the trace also captures
    the device reverse VJP (``orb_forces.backbone_bw``); forces are ``-dE/dpos`` via a host
    ``torch.autograd.grad`` finish through the differentiable edge geometry, same as
    ``energy_and_forces``.
  * ``ehead`` + ``fhead`` (direct, e.g. ``orb-v3-direct-20-omat``): forward-only, both heads read
    off the same backbone output -- there is no device backward to capture at all.
"""
from __future__ import annotations

import torch

from .trace import TraceEngineBase, _host_like


class OrbTracedEngine(TraceEngineBase):
    """Captures the device forward(+backward) once for a fixed topology, then replays it.

    Construct with the same operands as ``orb_forces.energy_and_forces`` (minus ``pos``), plus
    the already-constructed ``encoder``/``layers`` device modules and one or two heads; call with
    successive positions. The first call captures the trace; later calls refresh + replay.
    Single-system only (no disjoint-union batch, no stress) -- mirrors the scope of
    ``energy_and_forces``'s non-batched, non-stress path."""

    def __init__(self, encoder, layers, device, *, senders, receivers, atomic_numbers, node_feat,
                 ehead, fhead=None, cell_shift=None, r_max=6.0, num_bases=8):
        super().__init__(device)
        self.encoder = encoder
        self.layers = layers
        self.senders = senders
        self.receivers = receivers
        self.Z = atomic_numbers
        self.node_feat = node_feat
        self.ehead = ehead
        self.fhead = fhead
        self.conservative = fhead is None
        self.shift = cell_shift
        self.r_max = r_max
        self.num_bases = num_bases

    # ------------------------------------------------------------------ capture / refresh

    def _prepare(self, pos):
        from .orb_geometry import host_edge_features

        edge_feat, cutoff, _vectors = host_edge_features(pos, self.senders, self.receivers,
                                                         self.shift, r_max=self.r_max,
                                                         num_bases=self.num_bases)
        return edge_feat, cutoff

    def _refresh(self, ctx):
        edge_feat, cutoff = ctx
        ttnn = self.ttnn
        ttnn.copy_host_to_device_tensor(_host_like(ttnn, self.edge_dev, edge_feat.detach()),
                                        self.edge_dev)
        ttnn.copy_host_to_device_tensor(_host_like(ttnn, self.graph.cutoff, cutoff.detach()),
                                        self.graph.cutoff)

    def _capture(self, ctx):
        edge_feat, cutoff = ctx
        ttnn = self.ttnn
        from .orb_forces import backbone_bw
        from .orb_model import OrbGraphContext, _to_dev

        N = self.Z.shape[0]
        self.graph = OrbGraphContext(self.device, senders=self.senders, receivers=self.receivers,
                                     cutoff=cutoff.detach().float(), num_nodes=N)
        self.node_dev = _to_dev(self.node_feat, self.device, ttnn.bfloat16)
        self.edge_dev = _to_dev(edge_feat.detach().float(), self.device, ttnn.bfloat16)

        def body():
            nodes, edges = self.encoder(self.node_dev, self.edge_dev)
            for layer in self.layers:
                nodes, edges = layer(nodes, edges, self.graph)
            if self.conservative:
                raw_pred = self.ehead(nodes)
                g_edge_feat, g_cutoff = backbone_bw(self.encoder, self.layers, self.ehead, self.graph)
                return raw_pred, g_edge_feat, g_cutoff
            return self.ehead(nodes), self.fhead(nodes)

        body()                                  # warmup: compile all kernels before capture
        ttnn.synchronize_device(self.device)
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        outs = body()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        if self.conservative:
            self.raw_pred_t, self.g_edge_feat_t, self.g_cutoff_t = outs
        else:
            self.raw_pred_t, self.raw_force_t = outs

    # ------------------------------------------------------------------ evaluate

    def _finish(self, pos, ctx):
        """Energy + forces from the replayed outputs. Conservative: analytic ``-dE/dpos``, matching
        ``energy_and_forces``'s return. Direct: raw per-node ForceHead prediction (still normalized-
        space, un-denormalized, matching ``ForceHead.__call__``)."""
        edge_feat, cutoff = ctx
        ttnn = self.ttnn
        raw_pred = ttnn.to_torch(self.raw_pred_t).double().view(())
        if not self.conservative:
            raw_forces = ttnn.to_torch(self.raw_force_t).double()
            return raw_pred, raw_forces

        g_edge_feat = ttnn.to_torch(self.g_edge_feat_t).float()
        g_cutoff = ttnn.to_torch(self.g_cutoff_t).float()
        forces = -torch.autograd.grad([edge_feat, cutoff], pos,
                                      grad_outputs=[g_edge_feat, g_cutoff])[0]
        return float(raw_pred), forces
