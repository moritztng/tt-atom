"""MACE-MP-0b3 medium energy and forces on the card: the two interaction + product layers of tt_blocks chained,
then their reverse passes. The per-edge geometry (spherical harmonics, radial basis, ZBL) and its closed-form
derivative run on the host in float64.

Atoms are ordered by species once per structure (the per-element tables and skips need contiguous rows) and the
edges renumbered to match; callers pass and receive tensors in the caller's atom order. Per-species constants
(node embedding, E0) are folded per structure: they depend on nothing but the species.
"""
from __future__ import annotations

import torch

from tt_atom import device as D
from . import backward, tt_blocks
from . import reference as ref

LAYERS = (([(128, 0)], [(128, 0), (128, 1)], False), ([(128, 0), (128, 1)], [(128, 0)], True))


class Mace:
    """Weights are uploaded once per checkpoint and set of present species; the graph tables once per neighbour
    list (an MD step that keeps its list, or the same structure again, reuses both)."""

    def __init__(self, dev, w, dtype, product_dtype=None, trace=False):
        """``product_dtype`` (default ``dtype``) runs the product blocks and readouts at another precision: they
        hold the per-atom energies, where bfloat16's 8-bit mantissa leaves a systematic bias. ``trace`` replays
        energy_forces as a captured trace while the neighbour list stands (open the device with a trace region)."""
        self.dev, self.w, self.dtype, self.trace = dev, w, dtype, trace
        self.pdtype = product_dtype or dtype
        self.scale, self.shift = float(w["scale_shift.scale"]), float(w["scale_shift.shift"])
        self._weights, self._graph = {}, None

    def weights(self, present):
        key = tuple(present.tolist())
        if key not in self._weights:
            w, dev = self.w, self.dev
            self._weights[key] = [(tt_blocks.interaction_setup(dev, w, k, present, irr_in, res, self.dtype),
                                   tt_blocks.product_setup(dev, w, k, present, irr_out, self.pdtype))
                                  for k, (irr_in, irr_out, res) in enumerate(LAYERS)]
        return self._weights[key]

    def graph(self, atomic_numbers, edge_index):
        """Species order, per-species constants and device gather tables for one structure + neighbour list."""
        g = self._graph
        if g is not None and torch.equal(g["Z"], atomic_numbers) and torch.equal(g["ei_in"], edge_index):
            return g
        if g is not None and "trace" in g:
            import ttnn
            ttnn.release_trace(self.dev, g["trace"][0])
        w = self.w
        attrs = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long())
        el = attrs.argmax(1)
        present = el.unique()
        local = torch.searchsorted(present, el)
        order = torch.argsort(local, stable=True)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order))
        attrs = attrs[order]
        self._graph = g = {
            "Z": atomic_numbers.clone(), "ei_in": edge_index.clone(), "present": present, "order": order,
            "feats0": D.to_dev(ref.node_embedding(attrs, w), self.dev, self.dtype),  # [n, 128], species only
            "e0": float((attrs @ w["atomic_energies_fn.atomic_energies"]).sum()),
            "dev": tt_blocks.graph_setup(self.dev, local[order], len(present), inv[edge_index]),
        }
        return g

    def _forward(self, g, sh, sht, ef, tapes=None):
        """-> device [n, 1] sum of the readouts' node energies; ``tapes`` (a list) collects each layer's
        reverse-pass tapes."""
        import ttnn

        cast = lambda a, dt: a if a is None or a.dtype == dt else ttnn.typecast(a, dt)  # noqa: E731
        feats, node_e = g["feats0"], None
        for inter, prod in self.weights(g["present"]):
            tape = None if tapes is None else ({}, {})
            x, sc = tt_blocks.interaction(feats, (sh, sht), ef, inter, g["dev"], tape and tape[0])
            y, e = tt_blocks.product(cast(x, self.pdtype), cast(sc, self.pdtype), prod, g["dev"], tape and tape[1])
            feats = cast(y, self.dtype)
            node_e = e if node_e is None else ttnn.add(node_e, e)
            if tapes is not None:
                tapes.append(tape)
        return node_e

    def _step(self, g, sh, sht, ef):
        """Forward and reverse on the card -> device (node energies [n, 1], dE/d sh [n * D, 16], dE/d ef)."""
        import ttnn

        cast = lambda a, dt: a if a is None or a.dtype == dt else ttnn.typecast(a, dt)  # noqa: E731
        tapes = []
        node_e = self._forward(g, sh, sht, ef, tapes)
        g_feats = g_sh = g_ef = None
        for (inter, prod), (ti, tp) in zip(reversed(self.weights(g["present"])), reversed(tapes)):
            g_x, g_y = tt_blocks.product_bwd(cast(g_feats, self.pdtype), prod, g["dev"], tp)
            g_feats, gs, ge = tt_blocks.interaction_bwd(cast(g_x, self.dtype), cast(g_y, self.dtype) if inter["residual"]
                                                        else None, inter, g["dev"], ti)
            g_sh, g_ef = (gs, ge) if g_sh is None else (ttnn.add(g_sh, gs), ttnn.add(g_ef, ge))
        return node_e, g_sh, g_ef

    def _replay(self, g, hosts):
        """``_step`` as a trace: captured on the first call for this neighbour list, then the three inputs are
        overwritten in place and the recorded op stream replays with no per-op host dispatch."""
        import ttnn

        from tt_atom.trace import _host_like
        if "trace" not in g:
            inputs = [D.to_dev(h, self.dev, self.dtype) for h in hosts]
            self._step(g, *inputs)  # compiles every kernel before the capture
            ttnn.synchronize_device(self.dev)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            outs = self._step(g, *inputs)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            ttnn.synchronize_device(self.dev)
            g["trace"] = (tid, inputs, outs)
        else:
            for d, h in zip(g["trace"][1], hosts):
                ttnn.copy_host_to_device_tensor(_host_like(ttnn, d, h.float()), d)
        ttnn.execute_trace(self.dev, g["trace"][0], cq_id=0, blocking=True)
        return g["trace"][2]

    def _edges(self, g, sh, ef):
        """Per-edge host features -> host [sh [n, D, 16], its transpose, ef [n * D, n_radial]] in the card's
        padded receiver layout. The edge sum multiplies by sh from one side and the reverse pass from the other,
        and the host transposes for free."""
        sh = tt_blocks.pad_edges(g["dev"], sh).reshape(g["dev"]["n"], g["dev"]["D"], -1)
        return [sh, sh.transpose(1, 2).contiguous(), tt_blocks.pad_edges(g["dev"], ef)]

    def _total(self, g, node_e, zbl_node):
        import ttnn

        readout = float(ttnn.to_torch(node_e).double().sum())  # [n, 1] to host: n numbers
        return g["e0"] + self.scale * (float(zbl_node.sum()) + readout) + self.shift * g["dev"]["n"]

    def energy(self, atomic_numbers, edge_index, sh, ef, zbl_node):
        """sh [E, 16], ef [E, n_radial] (torch, edge order of the caller), zbl_node [n] -> total energy (float)."""
        g = self.graph(atomic_numbers, edge_index)
        inputs = [D.to_dev(h, self.dev, self.dtype) for h in self._edges(g, sh, ef)]
        return self._total(g, self._forward(g, *inputs), zbl_node)

    def energy_forces(self, atomic_numbers, pos, edge_index, shifts):
        """pos [n, 3], shifts [E, 3] (float64 torch) -> (total energy (float), forces [n, 3] float64)."""
        import ttnn

        w, g = self.w, self.graph(atomic_numbers, edge_index)
        vec, r = ref.edge_vectors(pos, edge_index, shifts)
        sh, ef = ref.spherical_harmonics(vec), ref.radial_embedding(r, atomic_numbers, edge_index, w)
        hosts = self._edges(g, sh, ef)
        node_e, g_sh, g_ef = self._replay(g, hosts) if self.trace else \
            self._step(g, *(D.to_dev(h, self.dev, self.dtype) for h in hosts))
        energy = self._total(g, node_e, ref.zbl(r, atomic_numbers, edge_index, w, len(pos)))
        g_sh, g_ef = (ttnn.to_torch(a).double()[g["dev"]["slot_of_edge"], :k]
                      for a, k in zip((g_sh, g_ef), (sh.shape[1], ef.shape[1])))
        return energy, -backward.geometry_vjp(w, atomic_numbers, pos, edge_index, shifts, g_sh, g_ef, self.scale)
