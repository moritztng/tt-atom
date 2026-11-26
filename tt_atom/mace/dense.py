"""MACE blocks recast as the dense ops a device runs: constant gathers, elementwise products, batched matmuls.

Same maths as ``reference.py``, a different association; graded against it in float64.

Symmetric contraction. MACE contracts per node ``sum_nu U_nu . (W_nu[element] x^(x)nu)`` with U carrying nu
equal ell indices. Because x^(x)nu is symmetric, only its distinct monomials matter (16 + 136 + 816 = 968
for ell = 16, nu <= 3, against 16 + 256 + 4096 full products), and ``U_nu @ W_nu[element]`` does not depend
on the node. So at load time each element folds into one [channels, 968, 2L+1] table, and at run time the
block is: build the monomials (one gather + one multiply per order), then one channel-batched matmul per
element. No per-node U contraction is left.
"""
from __future__ import annotations

from itertools import combinations_with_replacement, product

import torch


def monomials(ell, corr):
    """Sorted index tuples of every distinct monomial of degree 1..corr, plus the gather plan that builds
    degree nu from degree nu-1: feature f = features[parent[f]] * x[last[f]] (parent -1 = degree 1)."""
    tuples, parent, last, where = [], [], [], {}
    for nu in range(1, corr + 1):
        for t in combinations_with_replacement(range(ell), nu):
            where[t] = len(tuples)
            tuples.append(t)
            parent.append(where[t[:-1]] if nu > 1 else -1)
            last.append(t[-1])
    return tuples, torch.tensor(parent), torch.tensor(last)


def fold_contraction(U, W):
    """Fold one MACE Contraction into per-element tables T [n_elements, c, F, 2L+1] over the monomials.

    U[nu-1]: [(2L+1)] + [ell]*nu + [K_nu] (no leading axis for L = 0); W[nu-1]: [n_elements, K_nu, c].
    A monomial's entry sums U over every index order that sorts to it, so U need not be symmetric.
    """
    corr, ell = len(U), U[0].shape[-2]
    tuples, _, _ = monomials(ell, corr)
    D = U[0][..., 0, 0].numel()
    blocks = []
    for nu in range(1, corr + 1):
        u = U[nu - 1].reshape(D, ell ** nu, -1)
        mono = [t for t in tuples if len(t) == nu]
        col = {t: m for m, t in enumerate(mono)}
        S = torch.zeros(ell ** nu, len(mono), dtype=u.dtype)
        for f, idx in enumerate(product(range(ell), repeat=nu)):
            S[f, col[tuple(sorted(idx))]] = 1.0
        blocks.append(torch.einsum("dfk,fm,ekc->ecmd", u, S, W[nu - 1]))
    return torch.cat(blocks, dim=2)


def monomial_features(x, parent, last):
    """x [n, c, ell] -> [n, c, F]: each feature is one earlier feature times one component of x."""
    feats = []
    for p, l in zip(parent.tolist(), last.tolist()):
        xl = x[..., l]
        feats.append(xl if p < 0 else feats[p] * xl)
    return torch.stack(feats, dim=-1)


def symmetric_contraction(x, element, T, parent, last):
    """x [n, c, ell], element [n] (index into T's first axis) -> [n, c * (2L+1)], reference.py's layout."""
    P = monomial_features(x, parent, last)
    out = torch.zeros(x.shape[0], x.shape[1], T.shape[-1], dtype=x.dtype)
    for e in element.unique().tolist():
        rows = element == e
        out[rows] = torch.bmm(P[rows].transpose(0, 1), T[e]).transpose(0, 1)  # batched over channels
    return out.reshape(x.shape[0], -1)


def product_tables(w, layer, n_out, elements=None):
    """Fold every output irrep of ``products.<layer>``: [(T, parent, last)] in output order.

    ``elements`` (indices into the model's element table) folds only those, in that order: a table is
    [c, 968, 2L+1] per element, so folding all 89 of MP-0b3 costs ~0.6 GB in float64 for nothing.
    """
    out = []
    for k in range(n_out):
        q = f"products.{layer}.symmetric_contractions.contractions.{k}."
        corr = sum(1 for key in w if key.startswith(q + "U_matrix_"))
        U = [w[f"{q}U_matrix_{nu}"] for nu in range(1, corr + 1)]
        W = [w[f"{q}weights.{corr - 1 - nu}"] for nu in range(1, corr)] + [w[q + "weights_max"]]
        W = W if elements is None else [v[elements] for v in W]
        _, parent, last = monomials(U[0].shape[-2], corr)
        out.append((fold_contraction(U, W), parent, last))
    return out
