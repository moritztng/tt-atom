"""Device-resident, trace-captured energy+forces for the PET-MAD port at a FIXED topology
(MD / relaxation).

The pass-6 counterpart to ``tt_atom/orb_trace.py``: capture the device forward + device VJP
(``PetModel.forward`` + ``PetModel.backward`` -> ``pet_vjp.backbone_bw``) as one ttnn trace
once, then replay it each step with zero per-op host dispatch. Only the three pos-dependent
device inputs (``edge_vec_cat``, ``cutoff_factors``, ``log_mask``) are refreshed in place
via ``copy_host_to_device_tensor``; the topology-fixed index tables (``node_idx``,
``elem_nbr``, ``rev_idx``) are uploaded once at capture. The host ``torch.autograd.grad``
finish through ``pet_geometry``'s differentiable edge featurization runs AFTER replay,
outside the trace -- exactly the Orb pattern (``OrbTracedEngine``).

Prerequisite landed in this pass: ``pet_vjp._gnn_layer_bw`` now does the NEF reversed-edge
gather adjoint on DEVICE via ``ttnn.scatter_add`` (bf16) instead of the pass-5 host
``torch.index_add_`` roundtrip. The host op broke the captured instruction stream (a host
op between device ops cannot be recorded); with it gone, the whole backward is
device-resident and captures cleanly. The eager path uses the same device scatter, so the
trace is bit-exact vs eager -- it only removes host dispatch overhead, no numerics change
(verified: forces PCC vs golden unchanged at 0.98990, traced forces bit-identical to
eager).

Single-system only (no disjoint-union batch, no stress) -- mirrors the scope of
``pet_forces.device_energy_and_forces``. The calculator falls back to a re-capture whenever
the neighbour list changes (an atom crosses the cutoff); only steps that keep the topology
enjoy the replay speedup.
"""
from __future__ import annotations

import torch

from .trace import TraceEngineBase, _host_like


class PetTracedEngine(TraceEngineBase):
    """Captures the PET-MAD device forward+backward once for a fixed topology, then replays.

    Construct with the model operands (minus ``pos``); call with successive positions. The
    first call captures the trace (leaving inputs set to this step's data); later calls
    refresh the pos-dependent buffers + replay. Returns ``(raw_energy, forces)`` in the same
    raw/normalized space as ``pet_forces.device_energy_and_forces`` (the caller multiplies
    forces by ``scale`` to denormalize)."""

    def __init__(self, weights, device, *, cfg, atomic_numbers, cell=None, pbc=None):
        super().__init__(device)
        from .pet_model import PetModel

        self.cfg = cfg
        self.Z = atomic_numbers
        self.cell = cell
        self.pbc = pbc
        self.model = PetModel(weights, device, cfg=cfg)

    # ------------------------------------------------------------------ capture / refresh

    def _prepare(self, pos):
        from .pet_geometry import host_pet_geometry
        from .pet_model import _host_pet_inputs

        # float32 to match the device bf16 forward's host feeder (same as
        # pet_forces.device_energy_and_forces); requires_grad so the host finish can
        # differentiate through pet_geometry's edge featurization.
        p = pos.detach().to(torch.float32).clone().requires_grad_(True)
        bd = host_pet_geometry(p, self.Z, cell=self.cell, pbc=self.pbc, cfg=self.cfg)
        h = _host_pet_inputs(bd, self.cfg)
        return dict(pos=p, h=h)

    def _capture(self, ctx):
        ttnn = self.ttnn
        from .pet_model import _to_dev

        h = ctx["h"]
        N, Dmax = h["N"], h["Dmax"]
        rm = ttnn.ROW_MAJOR_LAYOUT
        # topology-fixed resident buffers: uploaded once, never refreshed.
        self.node_idx = _to_dev(h["node_idx"], self.device, ttnn.uint32, rm)
        self.elem_nbr = _to_dev(h["elem_nbr"], self.device, ttnn.uint32, rm)
        self.rev_idx = _to_dev(h["rev_idx"], self.device, ttnn.uint32, rm)
        self.rev_idx_host = h["rev_idx_host"]
        # pos-dependent resident buffers: refresh targets.
        self.edge_vec_cat = _to_dev(h["edge_vec_cat"], self.device, ttnn.bfloat16)
        self.cutoff_factors = _to_dev(h["cutoff_factors"], self.device, ttnn.bfloat16)
        self.log_mask = _to_dev(h["log_mask"], self.device, ttnn.bfloat16)
        self.bd_dev = dict(
            node_idx=self.node_idx, elem_nbr=self.elem_nbr,
            edge_vec_cat=self.edge_vec_cat, cutoff_factors=self.cutoff_factors,
            log_mask=self.log_mask, rev_idx=self.rev_idx,
            rev_idx_host=self.rev_idx_host,
            edge_vec_cat_host=h["edge_vec_cat_host"],
            cutoff_factors_host=h["cutoff_factors_host"],
            log_mask_host=h["log_mask_host"],
            N=N, Dmax=Dmax,
        )

        def body():
            raw = self.model.forward(self.bd_dev)
            g_evc, g_cutoff, g_lm = self.model.backward(self.bd_dev, g_raw=1.0)
            return raw, g_evc, g_cutoff, g_lm

        body()                                  # warmup: compile kernels + populate lazy buffers
        ttnn.synchronize_device(self.device)
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        outs = body()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.raw_t, self.g_evc_t, self.g_cutoff_t, self.g_lm_t = outs

    def _refresh(self, ctx):
        ttnn = self.ttnn
        h = ctx["h"]
        ttnn.copy_host_to_device_tensor(_host_like(ttnn, self.edge_vec_cat, h["edge_vec_cat"]),
                                        self.edge_vec_cat)
        ttnn.copy_host_to_device_tensor(_host_like(ttnn, self.cutoff_factors, h["cutoff_factors"]),
                                        self.cutoff_factors)
        ttnn.copy_host_to_device_tensor(_host_like(ttnn, self.log_mask, h["log_mask"]),
                                        self.log_mask)

    # ------------------------------------------------------------------ evaluate

    def _finish(self, pos, ctx):
        """Energy + forces from the replayed outputs. The host ``autograd.grad`` finish
        through ``pet_geometry`` is identical to ``pet_forces.device_energy_and_forces`` --
        it runs after replay, outside the captured region."""
        ttnn = self.ttnn
        h = ctx["h"]
        p = ctx["pos"]
        raw = float(ttnn.to_torch(self.raw_t).float().view(-1)[0])
        g_evc = ttnn.to_torch(self.g_evc_t).float()
        g_cutoff = ttnn.to_torch(self.g_cutoff_t).float()
        g_lm = ttnn.to_torch(self.g_lm_t).float()
        forces = -torch.autograd.grad(
            [h["edge_vec_cat_host"], h["cutoff_factors_host"], h["log_mask_host"]],
            p, grad_outputs=[g_evc, g_cutoff, g_lm])[0]
        return raw, forces.detach()
