"""Device-resident, trace-captured energy+forces for a FIXED topology (MD / relaxation).

Profiling the eager path (real uma-s-1, ethanol, p150) shows the device forward+backward is
~96% of a ``calculate()`` call and is host-*dispatch*-bound for these small graphs (hundreds of
tiny ttnn ops), not compute-bound — the host geometry is only ~3 ms. So the e2e lever for an MD
or relaxation loop, where the topology (edge set) is fixed across steps, is a ttnn *trace*: the
device forward+backward instruction stream is captured once and replayed with zero per-op host
dispatch. Each step the cheap host geometry is recomputed from the new positions and its
pos-dependent device inputs (Wigner coefficients, radial edge embedding, envelope, node init)
are refreshed *in place* in the captured buffers; then the trace is replayed.

Measured on p150 (see benchmarks/bench_trace.py): forward-only replay is ~2.6x the eager
forward; the full forward+backward trace gives the reported e2e MD speedup. Forces are bit-for-
bit the eager analytic forces (same op stream) — the trace only removes dispatch overhead.

The engine assumes a fixed topology (fixed ``edge_index`` and edge count). The calculator falls
back to a re-capture whenever the neighbour list changes (an atom crosses the cutoff), so the
result is always correct; only steps that keep the topology enjoy the replay speedup.
"""
from __future__ import annotations

import torch

from . import rotation
from .model import GraphContext


def _host_like(ttnn, dev_tensor, torch_tensor):
    """A HOST ttnn tensor matching ``dev_tensor``'s dtype/layout, holding ``torch_tensor``'s
    data — the only operand ``copy_host_to_device_tensor`` accepts to overwrite a resident (and
    trace-captured) buffer in place.

    Pre-convert a float32 source to bf16 in torch first: ``ttnn.from_torch(float32, dtype=bf16)``
    does a slow scalar host conversion (~3.8 ms for x_edge alone), whereas converting in torch
    (SIMD, ~0.03 ms) then from_torch on the already-bf16 tensor is a plain memcpy — ~10x faster
    per-step write. Both are round-to-nearest-even, so this is bit-identical."""
    if dev_tensor.dtype == ttnn.bfloat16 and torch_tensor.dtype == torch.float32:
        torch_tensor = torch_tensor.to(torch.bfloat16)
    return ttnn.from_torch(torch_tensor, dtype=dev_tensor.dtype, layout=dev_tensor.layout)


class TracedEngine:
    """Captures the device forward+backward once for a fixed topology, then replays it.

    Construct with the same operands as ``forces.energy_and_forces`` (minus ``pos``); call with
    successive positions. The first call captures the trace; later calls refresh + replay."""

    def __init__(self, bb, geo, atomic_numbers, edge_index, sys_node_embedding,
                 edge_cell_shift=None):
        import ttnn

        self.ttnn = ttnn
        self.bb = bb
        self.geo = geo
        self.Z = atomic_numbers
        self.edge_index = edge_index
        self.shift = edge_cell_shift
        self.se = sys_node_embedding
        self.dev = bb.device
        self.C = geo.C
        self.N = atomic_numbers.shape[0]
        self.tid = None
        self._device_ede = bb.edge_degree is not None

    # ------------------------------------------------------------------ capture / refresh

    def _refresh(self, t):
        """Overwrite the pos-dependent resident buffers in place (topology buffers untouched)."""
        ttnn = self.ttnn
        g = self.graph
        # the sparsity pattern is fixed for the topology (cached on the graph at capture); only the
        # coefficient values change per step, so gather them directly instead of re-running pack's
        # amax reduction over [E,nred,nsph] twice per step.
        if not hasattr(self, "_fwd_ii"):
            fi = torch.tensor([i for i, j in g.rot_fwd_ij]); fj = torch.tensor([j for i, j in g.rot_fwd_ij])
            ii = torch.tensor([i for i, j in g.rot_inv_ij]); ij = torch.tensor([j for i, j in g.rot_inv_ij])
            self._fwd_ii, self._fwd_jj, self._inv_ii, self._inv_jj = fi, fj, ii, ij
        cf = rotation.gather_coef(t["wigner"].detach(), self._fwd_ii, self._fwd_jj)
        ci = rotation.gather_coef(t["wigner_inv"].detach(), self._inv_ii, self._inv_jj)
        pairs = [
            (g.rot_fwd_coef, cf),
            (g.rot_inv_coef, ci),
            (g.x_edge, t["x_edge"].detach()),
            # only the envelope [E,1] is consumed on device; the 3D edge_envelope [E,1,1] is dead
            # (its tile pads to [E,32,32] -> ~64 ms/step to re-tilize), so it is not refreshed.
            # Write the ROW_MAJOR bf16 buffer (~0.1 ms); the forward tilizes/casts on device (the
            # bf8 TILE host from_torch here was ~7.8 ms/step -- the largest single refresh cost).
            (g.edge_envelope_rm, t["edge_envelope"].detach().reshape(g.E, 1)),
        ]
        # x_init operand: the host full node init (pos-dependent, refresh) or, with the device
        # edge-degree embedding, the CONSTANT l0 init (pos-independent -> uploaded once, never here).
        if not self._device_ede:
            pairs.append((self.x_init, t["x_init"].detach()))
        for dev_t, src in pairs:
            ttnn.copy_host_to_device_tensor(_host_like(ttnn, dev_t, src), dev_t)

    def _capture(self, t):
        ttnn = self.ttnn
        N, C = self.N, self.C
        self.graph = GraphContext(
            self.dev, edge_index=self.edge_index, wigner=t["wigner"].detach(),
            wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
            edge_envelope=t["edge_envelope"].detach(), num_nodes=N)
        self.se3 = ttnn.from_torch(self.se.reshape(N, 1, C), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=self.dev)
        # x_init operand is the constant l0 node init (device edge-degree on) or the full host
        # x_init (off). l0 is pos-independent so it is uploaded once here and never refreshed.
        init = t["l0"] if self._device_ede else t["x_init"]
        self.x_init = ttnn.from_torch(init.detach(), dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=self.dev)
        from . import forces as Fmod

        def body():
            node_emb, energy = self.bb(self.x_init, self.graph, self.se3)
            acc = Fmod.backbone_bw(self.bb, self.graph, node_emb)
            return energy, acc

        body()                                  # warmup: compile all kernels before capture
        ttnn.synchronize_device(self.dev)
        self.tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.energy_t, self.acc = body()
        ttnn.end_trace_capture(self.dev, self.tid, cq_id=0)
        ttnn.synchronize_device(self.dev)

    # ------------------------------------------------------------------ evaluate

    def __call__(self, pos):
        """Energy + analytic forces at ``pos`` (same topology as construction)."""
        ttnn = self.ttnn
        pos = pos.detach().clone().requires_grad_(True)
        t = self.geo(pos, self.Z, self.edge_index, self.se, edge_cell_shift=self.shift)

        if self.tid is None:
            self._capture(t)            # records the op stream + leaves inputs set to this ``t``
        else:
            self._refresh(t)
        # trace capture records without executing, so a replay is needed to populate the outputs
        # on the capture step too (the inputs already hold this ``t``'s data after _capture).
        ttnn.execute_trace(self.dev, self.tid, cq_id=0, blocking=True)

        E = float(ttnn.to_torch(self.energy_t).reshape(-1)[0])
        acc = self.acc
        nsph = self.graph.nsph
        g_wig = rotation.scatter_coef(ttnn.to_torch(acc["rot_fwd"]).float(),
                                      self.graph.rot_fwd_ij, nsph)
        g_winv = rotation.scatter_coef(ttnn.to_torch(acc["rot_inv"]).float(),
                                       self.graph.rot_inv_ij, nsph)
        g_env = ttnn.to_torch(acc["envelope"]).float().reshape(-1, 1, 1)
        # radial finish is done on device inside the captured trace (backbone_bw); read it back.
        # Only the gaussian block of x_edge = [gaussian | src_emb | tgt_emb] depends on pos, so cast
        # just that block bf16->f32 (the cast dominates readback); embedding cols add zero to dpos.
        # only the gaussian block (first ng cols) of x_edge depends on pos; the src/tgt embedding
        # columns are pos-independent (their adjoint is discarded). Slice on device -> read back
        # only [E, ng] (5x less transfer+cast than the full [E, x_edge_width]).
        ng = self.geo.offset.shape[0]
        W = acc["x_edge"].shape[1]
        gx = ttnn.to_torch(ttnn.slice(acc["x_edge"], [0, 0], [acc["x_edge"].shape[0], ng]))
        g_xe = torch.zeros((gx.shape[0], W), dtype=torch.float32)
        g_xe[:, :ng] = gx.float()
        outs = [t["wigner"], t["wigner_inv"], t["x_edge"], t["edge_envelope"]]
        gouts = [g_wig, g_winv, g_xe, g_env]
        # host x_init adjoint only when x_init is a host term (device edge-degree consumes it on device)
        if not self._device_ede:
            outs = [t["x_init"]] + outs
            gouts = [ttnn.to_torch(acc["x_init"]).float()] + gouts
        g_pos = torch.autograd.grad(outs, pos, grad_outputs=gouts)[0]
        return E, -g_pos

    def close(self):
        if self.tid is not None:
            self.ttnn.release_trace(self.dev, self.tid)
            self.tid = None
