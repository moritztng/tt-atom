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
    """``[E, n_out, n_in]`` Wigner -> (``ij``: list of structural-nonzero (out,in) pairs, ``coef``:
    ``[E, nnz]`` the per-edge values). The pattern is taken from ``amax`` over edges, so it
    includes every entry nonzero for *any* edge in this topology (entries that are ~0 for all
    edges contribute nothing, so dropping them is exact for this topology).

    The matrix is rectangular for mmax<lmax checkpoints (uma-m: forward maps the 25-coeff node
    representation to the 19-coeff reduced m-space, ``[E,19,25]``; the inverse maps back)."""
    patt = wigner.abs().amax(dim=0) > tol
    idx = patt.nonzero(as_tuple=False)                          # [nnz,2], row-major (i outer, j inner)
    coef = wigner[:, idx[:, 0], idx[:, 1]].contiguous()         # [E, nnz] — vectorized gather (was a
    ij = [(int(i), int(j)) for i, j in idx.tolist()]            # per-nonzero torch.stack, ~2x slower)
    return ij, coef


def rotate(ttnn, x_flat, ij, coef, n_in, W, device, n_out=None):
    """``x_flat`` ``[E, n_in*W]`` (W channels per coordinate) -> rotated ``[E, n_out*W]``.
    ``n_out`` defaults to ``n_in`` (square rotation, uma-s); pass it for the rectangular
    reduced-m-space rotation (uma-m)."""
    n_out = n_in if n_out is None else n_out
    E = x_flat.shape[0]
    cols = [ttnn.slice(x_flat, [0, j * W], [E, (j + 1) * W]) for j in range(n_in)]
    out = [None] * n_out
    for k, (i, j) in enumerate(ij):
        c = ttnn.slice(coef, [0, k], [E, k + 1])              # [E,1] broadcast
        # fuse the accumulate multiply+add into one addcmul (out[i] += cols[j]*c). Rotation is
        # dispatch/op-count-bound (many tiny ops), so folding the separate add drops both a kernel
        # launch and a DRAM round-trip of the partial sum. Bit-identical to multiply-then-add.
        if out[i] is None:
            out[i] = ttnn.multiply(cols[j], c)
        else:
            out[i] = ttnn.addcmul(out[i], cols[j], c)
    for i in range(n_out):
        if out[i] is None:
            out[i] = ttnn.zeros((E, W), dtype=x_flat.dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.concat(out, dim=1)


def rotate_bw(ttnn, x_in_flat, g_out_flat, ij, coef, n_in, W, device, n_out=None):
    """VJP of :func:`rotate`. Returns (g wrt ``x_in`` flat ``[E,n_in*W]``, g wrt the packed
    coefficients ``[E, nnz]``). The coefficient adjoint is scattered back to a dense
    ``[E, n_out, n_in]`` on host (:func:`scatter_coef`) to drive the geometric ``dW/dpos``
    autograd for the force."""
    n_out = n_in if n_out is None else n_out
    E = x_in_flat.shape[0]
    in_cols = [ttnn.slice(x_in_flat, [0, j * W], [E, (j + 1) * W]) for j in range(n_in)]
    gout_cols = [ttnn.slice(g_out_flat, [0, i * W], [E, (i + 1) * W]) for i in range(n_out)]
    g_in = [None] * n_in
    gc = [None] * len(ij)
    for k, (i, j) in enumerate(ij):
        c = ttnn.slice(coef, [0, k], [E, k + 1])
        term = ttnn.multiply(gout_cols[i], c)
        g_in[j] = term if g_in[j] is None else ttnn.add(g_in[j], term)
        gc[k] = ttnn.sum(ttnn.multiply(gout_cols[i], in_cols[j]), dim=1, keepdim=True)   # [E,1]
    for j in range(n_in):
        if g_in[j] is None:
            g_in[j] = ttnn.zeros((E, W), dtype=x_in_flat.dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.concat(g_in, dim=1), ttnn.concat(gc, dim=1)


def scatter_coef(g_coef: torch.Tensor, ij, n_out: int, n_in: int = None) -> torch.Tensor:
    """``[E, nnz]`` coefficient adjoints -> dense ``[E, n_out, n_in]`` (zeros off-pattern)."""
    n_in = n_out if n_in is None else n_in
    E = g_coef.shape[0]
    g = torch.zeros(E, n_out, n_in, dtype=g_coef.dtype)
    ii = torch.tensor([i for i, j in ij]); jj = torch.tensor([j for i, j in ij])
    g[:, ii, jj] = g_coef                                       # vectorized scatter (was a py loop)
    return g
