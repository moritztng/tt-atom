"""MACE-MP-0b3 medium energy on the card: the two interaction + product layers of tt_blocks chained.

Atoms are ordered by species once per structure (the per-element tables and skips need contiguous rows) and the
edges renumbered to match; callers pass and receive tensors in the caller's atom order. Per-species constants
(node embedding, E0) are folded per structure: they depend on nothing but the species.
"""
from __future__ import annotations

import torch

from tt_atom import device as D
from . import dense, tt_blocks
from . import reference as ref

LAYERS = (([(128, 0)], [(128, 0), (128, 1)], False), ([(128, 0), (128, 1)], [(128, 0)], True))


class MaceEnergy:
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

    def energy(self, atomic_numbers, edge_index, sh, ef, zbl_node):
        """sh [E, 16], ef [E, 8] (torch, edge order of the caller), zbl_node [n] -> total energy (float)."""
        import ttnn

        g = self.graph(atomic_numbers, edge_index)
        layers, graph = self.weights(g["present"]), g["dev"]
        dev = lambda a: D.to_dev(a, self.dev, self.dtype)  # noqa: E731
        sh, ef = dev(sh), dev(ef)
        feats, node_e = g["feats0"], None
        for inter, prod in layers:
            x, sc = tt_blocks.interaction(feats, sh, ef, inter, graph)
            if self.pdtype != self.dtype:
                x, sc = (None if a is None else ttnn.typecast(a, self.pdtype) for a in (x, sc))
            y, e = tt_blocks.product(x, sc, prod, graph)
            feats = y if self.pdtype == self.dtype else ttnn.typecast(y, self.dtype)
            node_e = e if node_e is None else ttnn.add(node_e, e)
        readout = float(ttnn.to_torch(node_e).double().sum())  # [n, 1] to host: n numbers
        n = graph["n"]
        return g["e0"] + self.scale * (float(zbl_node.sum()) + readout) + self.shift * n
