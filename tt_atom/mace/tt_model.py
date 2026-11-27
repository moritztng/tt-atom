"""MACE-MP-0b3 medium energy on the card: the two interaction + product layers of tt_blocks chained.

Atoms are ordered by species once at setup (the per-element tables and skips need contiguous rows) and the
edges renumbered to match; callers pass and receive tensors in the caller's atom order. Per-species constants
(node embedding, E0) are folded at setup: they depend on nothing but the species.
"""
from __future__ import annotations

import torch

from tt_atom import device as D
from . import dense, tt_blocks
from . import reference as ref

LAYERS = (([(128, 0)], [(128, 0), (128, 1)], False), ([(128, 0), (128, 1)], [(128, 0)], True))


class MaceEnergy:
    def __init__(self, dev, w, atomic_numbers, edge_index, dtype, product_dtype=None):
        """``product_dtype`` (default ``dtype``) runs the product blocks and readouts at another precision: they
        hold the per-atom energies, where bfloat16's 8-bit mantissa leaves a systematic bias."""
        el = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long()).argmax(1)
        present = el.unique()
        local = torch.searchsorted(present, el)
        self.order = torch.argsort(local, stable=True)
        inv = torch.empty_like(self.order)
        inv[self.order] = torch.arange(len(self.order))
        ei, el_sorted = inv[edge_index], local[self.order]
        attrs = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long())[self.order]
        self.dev, self.dtype, self.n = dev, dtype, len(self.order)
        self.pdtype = product_dtype or dtype
        self.feats0 = D.to_dev(ref.node_embedding(attrs, w), dev, dtype)  # [n, 128], species only
        self.e0 = float((attrs @ w["atomic_energies_fn.atomic_energies"]).sum())
        self.scale, self.shift = float(w["scale_shift.scale"]), float(w["scale_shift.shift"])
        self.layers = [(tt_blocks.interaction_setup(dev, w, k, el_sorted, present, ei, irr_in, res, dtype),
                        tt_blocks.product_setup(dev, w, k, el_sorted, present, irr_out, self.pdtype))
                       for k, (irr_in, irr_out, res) in enumerate(LAYERS)]

    def energy(self, sh, ef, zbl_node):
        """sh [E, 16], ef [E, 8] (torch, edge order of the caller), zbl_node [n] -> total energy (float)."""
        import ttnn

        dev = lambda a: D.to_dev(a, self.dev, self.dtype)  # noqa: E731
        sh, ef = dev(sh), dev(ef)
        feats, node_e = self.feats0, None
        for inter, prod in self.layers:
            x, sc = tt_blocks.interaction(feats, sh, ef, inter)
            if self.pdtype != self.dtype:
                x, sc = (None if a is None else ttnn.typecast(a, self.pdtype) for a in (x, sc))
            y, e = tt_blocks.product(x, sc, prod, self.n)
            feats = y if self.pdtype == self.dtype else ttnn.typecast(y, self.dtype)
            node_e = e if node_e is None else ttnn.add(node_e, e)
        readout = float(ttnn.to_torch(node_e).double().sum())  # [n, 1] to host: n numbers
        return self.e0 + self.scale * (float(zbl_node.sum()) + readout) + self.shift * self.n
