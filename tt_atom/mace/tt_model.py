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

    def __init__(self, dev, w, dtype, product_dtype=None):
        """``product_dtype`` (default ``dtype``) runs the product blocks and readouts at another precision: they
        hold the per-atom energies, where bfloat16's 8-bit mantissa leaves a systematic bias."""
        self.dev, self.w, self.dtype = dev, w, dtype
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

    def _forward(self, g, sh, ef, tapes=None):
        """-> sum of the readouts' node energies (float); ``tapes`` (a list) collects each layer's reverse-pass tapes."""
        import ttnn

        cast = lambda a, dt: a if a is None or a.dtype == dt else ttnn.typecast(a, dt)  # noqa: E731
        feats, node_e = g["feats0"], None
        for inter, prod in self.weights(g["present"]):
            tape = None if tapes is None else ({}, {})
            x, sc = tt_blocks.interaction(feats, sh, ef, inter, g["dev"], tape and tape[0])
            y, e = tt_blocks.product(cast(x, self.pdtype), cast(sc, self.pdtype), prod, g["dev"], tape and tape[1])
            feats = cast(y, self.dtype)
            node_e = e if node_e is None else ttnn.add(node_e, e)
            if tapes is not None:
                tapes.append(tape)
        return float(ttnn.to_torch(node_e).double().sum())  # [n, 1] to host: n numbers

    def _total(self, g, readout, zbl_node):
        return g["e0"] + self.scale * (float(zbl_node.sum()) + readout) + self.shift * g["dev"]["n"]

    def energy(self, atomic_numbers, edge_index, sh, ef, zbl_node):
        """sh [E, 16], ef [E, 8] (torch, edge order of the caller), zbl_node [n] -> total energy (float)."""
        g = self.graph(atomic_numbers, edge_index)
        return self._total(g, self._forward(g, *self._edges(g, sh, ef)), zbl_node)

    def _edges(self, g, sh, ef):
        """Per-edge host features -> the card's padded receiver layout."""
        return (tt_blocks.edge_sh(self.dev, g["dev"], sh, self.dtype),
                D.to_dev(tt_blocks.pad_edges(g["dev"], ef), self.dev, self.dtype))

    def energy_forces(self, atomic_numbers, pos, edge_index, shifts):
        """pos [n, 3], shifts [E, 3] (float64 torch) -> (total energy (float), forces [n, 3] float64)."""
        import ttnn

        w, g = self.w, self.graph(atomic_numbers, edge_index)
        vec, r = ref.edge_vectors(pos, edge_index, shifts)
        sh, ef = ref.spherical_harmonics(vec), ref.radial_embedding(r, atomic_numbers, edge_index, w)
        widths, (sh, ef) = (sh.shape[1], ef.shape[1]), self._edges(g, sh, ef)
        tapes = []
        energy = self._total(g, self._forward(g, sh, ef, tapes), ref.zbl(r, atomic_numbers, edge_index, w, len(pos)))
        cast = lambda a, dt: a if a is None or a.dtype == dt else ttnn.typecast(a, dt)  # noqa: E731
        g_feats = g_sh = g_ef = None
        for (inter, prod), (ti, tp) in zip(reversed(self.weights(g["present"])), reversed(tapes)):
            g_x, g_y = tt_blocks.product_bwd(cast(g_feats, self.pdtype), prod, g["dev"], tp)
            g_feats, gs, ge = tt_blocks.interaction_bwd(cast(g_x, self.dtype), cast(g_y, self.dtype) if inter["residual"]
                                                        else None, inter, g["dev"], ti)
            g_sh, g_ef = (gs, ge) if g_sh is None else (ttnn.add(g_sh, gs), ttnn.add(g_ef, ge))
        g_sh, g_ef = (ttnn.to_torch(a).double()[g["dev"]["slot_of_edge"], :k] for a, k in zip((g_sh, g_ef), widths))
        return energy, -backward.geometry_vjp(w, atomic_numbers, pos, edge_index, shifts, g_sh, g_ef, self.scale)
