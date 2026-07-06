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
    trace-captured) buffer in place."""
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

    # ------------------------------------------------------------------ capture / refresh

    def _refresh(self, t):
        """Overwrite the pos-dependent resident buffers in place (topology buffers untouched)."""
        ttnn = self.ttnn
        g = self.graph
        _, cf = rotation.pack(t["wigner"].detach())
        _, ci = rotation.pack(t["wigner_inv"].detach())
        pairs = [
            (g.rot_fwd_coef, cf),
            (g.rot_inv_coef, ci),
            (g.x_edge, t["x_edge"].detach()),
            (g.edge_envelope, t["edge_envelope"].detach()),
            (g.edge_envelope_f, t["edge_envelope"].detach().reshape(g.E, 1)),
            (self.x_init, t["x_init"].detach()),
        ]
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
        self.x_init = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16,
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
        g_xi = ttnn.to_torch(acc["x_init"]).float()
        g_wig = rotation.scatter_coef(ttnn.to_torch(acc["rot_fwd"]).float(),
                                      self.graph.rot_fwd_ij, nsph)
        g_winv = rotation.scatter_coef(ttnn.to_torch(acc["rot_inv"]).float(),
                                       self.graph.rot_inv_ij, nsph)
        g_env = ttnn.to_torch(acc["envelope"]).float().reshape(-1, 1, 1)
        # radial finish is done on device inside the captured trace (backbone_bw); read it back
        g_xe = ttnn.to_torch(acc["x_edge"]).float()
        g_pos = torch.autograd.grad(
            [t["x_init"], t["wigner"], t["wigner_inv"], t["x_edge"], t["edge_envelope"]],
            pos, grad_outputs=[g_xi, g_wig, g_winv, g_xe, g_env])[0]
        return E, -g_pos

    def close(self):
        if self.tid is not None:
            self.ttnn.release_trace(self.dev, self.tid)
            self.tid = None
