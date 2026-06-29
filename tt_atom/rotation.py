"""Per-edge Wigner rotation as a flat sparse multiply-accumulate (the SO(2) frame change).

The edgewise message block rotates node features into the edge frame with a per-edge Wigner
matrix ``W[e]`` (``m_out[e,i,:] = sum_j W[e,i,j] m_in[e,j,:]``). Done as a batched
``[E,9,9]x[E,9,C]`` matmul this is the single most expensive op: each edge is a tiny 9x9 (tile-
padded to 32x32) matmul, so it is launch/overhead bound (~2.9us/edge, flat in E, *not* flop
bound -- LoFi == HiFi4).

But ``W`` has a FIXED sparsity pattern -- the same ``(i,j)`` nonzeros for every edge (block
diagonal in degree ``l``, permuted by the ``to_m`` reordering folded into ``W``). So we do the
rotation as elementwise column-block MACs in flat ``[E, 9*C]`` layout: for each nonzero
``(i,j)``, ``out_block_i += W[:,i,j] * in_block_j``. Every op is then a single launch over all
edges. Measured ~3.8x faster than the batched matmul at PCC 0.99994 (bf16), and it composes
into the flat SO(2) pipeline with no [E,9,C] tile-padding.
"""
from __future__ import annotations

import torch


def pack(wigner: torch.Tensor, tol: float = 1e-6):
    """``[E,n,n]`` Wigner -> (``ij``: list of structural-nonzero (out,in) pairs, ``coef``:
    ``[E, nnz]`` the per-edge values). The pattern is taken from ``amax`` over edges, so it
    includes every entry nonzero for *any* edge in this topology (entries that are ~0 for all
    edges contribute nothing, so dropping them is exact for this topology)."""
    n = wigner.shape[1]
    patt = wigner.abs().amax(dim=0) > tol
    ij = [(int(i), int(j)) for i in range(n) for j in range(n) if patt[i, j]]
    coef = torch.stack([wigner[:, i, j] for (i, j) in ij], dim=1).contiguous()   # [E, nnz]
    return ij, coef


def rotate(ttnn, x_flat, ij, coef, nsph, W, device):
    """``x_flat`` ``[E, nsph*W]`` (W channels per coordinate) -> rotated ``[E, nsph*W]``."""
    E = x_flat.shape[0]
    cols = [ttnn.slice(x_flat, [0, j * W], [E, (j + 1) * W]) for j in range(nsph)]
    out = [None] * nsph
    for k, (i, j) in enumerate(ij):
        c = ttnn.slice(coef, [0, k], [E, k + 1])              # [E,1] broadcast
        term = ttnn.multiply(cols[j], c)
        out[i] = term if out[i] is None else ttnn.add(out[i], term)
    for i in range(nsph):
        if out[i] is None:
            out[i] = ttnn.zeros((E, W), dtype=x_flat.dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.concat(out, dim=1)


def rotate_bw(ttnn, x_in_flat, g_out_flat, ij, coef, nsph, W, device):
    """VJP of :func:`rotate`. Returns (g wrt ``x_in`` flat ``[E,nsph*W]``, g wrt the packed
    coefficients ``[E, nnz]``). The coefficient adjoint is scattered back to a dense ``[E,n,n]``
    on host (:func:`scatter_coef`) to drive the geometric ``dW/dpos`` autograd for the force."""
    E = x_in_flat.shape[0]
    in_cols = [ttnn.slice(x_in_flat, [0, j * W], [E, (j + 1) * W]) for j in range(nsph)]
    gout_cols = [ttnn.slice(g_out_flat, [0, i * W], [E, (i + 1) * W]) for i in range(nsph)]
    g_in = [None] * nsph
    gc = [None] * len(ij)
    for k, (i, j) in enumerate(ij):
        c = ttnn.slice(coef, [0, k], [E, k + 1])
        term = ttnn.multiply(gout_cols[i], c)
        g_in[j] = term if g_in[j] is None else ttnn.add(g_in[j], term)
        gc[k] = ttnn.sum(ttnn.multiply(gout_cols[i], in_cols[j]), dim=1, keepdim=True)   # [E,1]
    for j in range(nsph):
        if g_in[j] is None:
            g_in[j] = ttnn.zeros((E, W), dtype=x_in_flat.dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.concat(g_in, dim=1), ttnn.concat(gc, dim=1)


def scatter_coef(g_coef: torch.Tensor, ij, n: int) -> torch.Tensor:
    """``[E, nnz]`` coefficient adjoints -> dense ``[E, n, n]`` (zeros off-pattern)."""
    E = g_coef.shape[0]
    g = torch.zeros(E, n, n, dtype=g_coef.dtype)
    for k, (i, j) in enumerate(ij):
        g[:, i, j] = g_coef[:, k]
    return g
