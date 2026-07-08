"""Per-edge Wigner rotation as a flat sparse multiply-accumulate (the SO(2) frame change).

The edgewise message block rotates node features into the edge frame with a per-edge Wigner
matrix ``W[e]`` (``m_out[e,i,:] = sum_j W[e,i,j] m_in[e,j,:]``). Done as a batched
``[E,9,9]x[E,9,C]`` matmul this is the single most expensive op: each edge is a tiny 9x9 (tile-
padded to 32x32) matmul, so it is launch/overhead bound (~2.9us/edge, flat in E, *not* flop
bound -- LoFi == HiFi4).

But ``W`` has a FIXED sparsity pattern -- the same ``(i,j)`` nonzeros for every edge (block
diagonal in degree ``l``, permuted by the ``to_m`` reordering folded into ``W``). So the rotation
runs as the custom fused ``ttnn.experimental.fused_rotate`` kernel: one launch reads x once, keeps
all ``nnz`` multiply-accumulates in the dest registers (fp32 accumulate), writes out once --
~4.3x faster than the ~35-dispatch addcmul MAC it replaces (7.01 -> 1.62 ms on the uma-s
rotate shape E=46016), at PCC 0.999995. tt-atom is the
custom-kernel-only uma-s build, so this kernel is the ALWAYS-ON path; a shape the kernel cannot
run (the rectangular reduced-m uma-m rotation) raises rather than silently falling back.

Requires the source-ttnn build that carries the op (see ../custom_kernels/README.md and the
README Install section).
"""
from __future__ import annotations

import torch


def _fused_op(ttnn):
    return ttnn._ttnn.operations.experimental.fused_rotate


def _gc_op(ttnn):
    return ttnn._ttnn.operations.experimental.fused_rotate_gc


# The 32 column-selector tiles ([32, 32*32]; tile c has column c all-ones). Pos/topology-independent
# constant -> built once per device. matmul(prod, sel[c]) = rowsum(prod) placed in output column c.
_SEL_CACHE: dict = {}


def _sel(ttnn, device):
    key = id(device)
    t = _SEL_CACHE.get(key)
    if t is None:
        s = torch.zeros(32, 32 * 32)
        for c in range(32):
            s[:, c * 32 + c] = 1.0
        t = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _SEL_CACHE[key] = t
    return t


# The rotate_bw coefficient adjoint (dE/dcoef) has two on-device forms. The fast one is the custom
# fused mul-reduce kernel (``fused_rotate_gc``): products L1-resident, one accumulating matmul does
# the W-reduction + column placement, reading gout+xin once. But its product CB (32*Wt = 512KB at
# W=256) leaves little L1 headroom, so it clashes with the trace's L1-resident backward tensors
# (grid/spectral) in the small/mid-N regime (measured clash at N=216 E~10k and N=512 E~23k; clash-
# free at large N where those tensors spill to DRAM). Since the gc kernel only wins where the device
# replay dominates (large graphs), gate it to large graphs and use the ones_bd segment-sum GEMM
# below otherwise -- the GEMM is correct at EVERY size and cheap in the small/mid-N dispatch-bound
# regime, so uma-s stays correct with no L1 clash at any size. (A batched-CB gc variant that fits
# small-N exists but its reload-add serial chain regressed large-N ~35%, so the fast full-tile
# kernel + this edge gate is the shippable choice.) Fixed constant, not an env toggle.
_GC_MIN_EDGES = 45000


def _gc_kernel_ok(n_out, n_in, nnz, W, E) -> bool:
    """L1 budget for the gc kernel CBs (see gc_program_factory): 2*(n_out+n_in)*Wt gout/xin +
    32 sel + 32*Wt prod + 2*ceil(nnz/32) out tiles. uma-s fits (~1.16MB); uma-m (W=256, 19x25)
    overflows. Also require a large graph so the CBs co-reside with the trace's L1 tensors."""
    Wt = W // 32
    out_tiles = (nnz + 31) // 32
    tiles = 2 * (n_out + n_in) * Wt + 32 + 32 * Wt + 2 * out_tiles
    return tiles * _TILE_BYTES <= _L1_CB_BUDGET and E >= _GC_MIN_EDGES


# Two hard limits gate the fused_rotate kernel to shapes it can run:
#   (1) The compute kernel fans-in all `d` per-block products into dst[0..d-1] and sums them there,
#       so the max fan-in degree must fit the fp32 DST register file (dst_full_sync_en -> 8 slots).
#   (2) The program factory statically allocates double-buffered CBs for the x, coef and out tiles
#       on each core; their total must fit L1 (1.5 MB). uma-m (W=256, n_in=25, n_out=19) blows past
#       it (~2 MB) and TT_THROWs at program build; uma-s (W=128/256, 9x9) is ~0.4-0.7 MB and fits.
# uma-s always satisfies both (per-edge shape is size-independent); uma-m never does -> it raises.
_MAX_DST = 8
_TILE_BYTES = 2048            # bf16 32x32 tile
_L1_CB_BUDGET = 1_400_000     # < 1.5 MB L1, leaving headroom for runtime/semaphores


def _cb_bytes(n_in, n_out, nnz, W) -> int:
    Wt = W // 32
    return 2 * (n_in * Wt + n_out * Wt + nnz) * _TILE_BYTES


def _kernel_ok(deg, n_in, n_out, nnz, W) -> bool:
    return (len(deg) > 0 and max(deg) <= _MAX_DST
            and _cb_bytes(n_in, n_out, nnz, W) <= _L1_CB_BUDGET)


_UNSUPPORTED = (
    "tt-atom is the custom-kernel-only uma-s build: the fused rotation kernel cannot run this "
    "shape (n_in={n_in}, n_out={n_out}, W={W}, nnz={nnz}). uma-s (square 9x9) is the validated, "
    "supported target; a rectangular reduced-m checkpoint (uma-m: W=256, 19x25) overflows L1 and "
    "is unsupported in this build."
)


_GROUP_CACHE: dict = {}


# id(coef) -> (coef_ref, coef_exp_dev). Holds the coef reference so its python id cannot be recycled
# onto a stale entry. Cleared past a step's worth; reset_expand_cache() drops it before a trace
# capture (see below).
_EXPAND_CACHE: dict = {}


def reset_expand_cache():
    """Drop the expanded-coef cache. The trace engine calls this right before capturing, so the
    on-device coef expansion (repeat_interleave) is recorded INTO the trace and recomputed on every
    replay from the in-place-refreshed compact coef -- instead of reusing a warmup-built tensor the
    replay never refreshes (which would freeze the coefficients at the capture step)."""
    _EXPAND_CACHE.clear()


def _coef_exp(ttnn, coef, dtype=None):
    """Expand compact [E, nnz] device coefficients to [E, nnz*32] ON DEVICE (each nonzero's
    coef broadcast across its 32-column tile). Cached by (id(coef), dtype); holds the coef
    reference so its python id cannot be recycled onto a stale entry. ``dtype`` (bf8-edge) casts
    the expanded coef to match the kernel's bf8 x input -- CACHED, so within a forward+backward
    pass the expansion (and any typecast) runs once, not on every rotate call."""
    # coef is stored ROW_MAJOR (cheap refresh); tilize on device here for the kernel (see
    # GraphContext). Cache keyed on the ORIGINAL (persistent) coef so repeat hits across calls.
    orig = coef
    key = (id(orig), dtype)
    b = _EXPAND_CACHE.get(key)
    if b is not None and b[0] is orig:
        return b[1]
    tiled = ttnn.to_layout(orig, ttnn.TILE_LAYOUT) if orig.layout != ttnn.TILE_LAYOUT else orig
    ce_dev = ttnn.repeat_interleave(tiled, 32, dim=1)           # [E, nnz*32]
    if dtype is not None and ce_dev.dtype != dtype:
        ce_dev = ttnn.typecast(ce_dev, dtype)
    if len(_EXPAND_CACHE) > 64:
        _EXPAND_CACHE.clear()
    _EXPAND_CACHE[key] = (orig, ce_dev)
    return ce_dev


def _group(ij, nblocks, by):
    """Group the (i,j) nonzeros by output block for the fused kernel. ``by='i'`` (forward:
    output block = i, fan-in over j) or ``by='j'`` (backward g_in: output block = j, fan-in
    over i). Returns (deg[nblocks], ks[nnz] coef-tile index, js[nnz] the *other* index)."""
    key = (id(ij), nblocks, by)
    g = _GROUP_CACHE.get(key)
    if g is not None and g[0] is ij:          # identity guard: id(ij) can be recycled onto a stale
        return g[1]                            # entry once the original list is gc'd (wrong nnz).
    deg = [0] * nblocks
    ks: list = []
    other: list = []
    for b in range(nblocks):
        for k, (i, j) in enumerate(ij):
            blk, oth = (i, j) if by == "i" else (j, i)
            if blk == b:
                ks.append(k); other.append(oth); deg[b] += 1
    res = (deg, ks, other)
    if len(_GROUP_CACHE) > 64:
        _GROUP_CACHE.clear()
    _GROUP_CACHE[key] = (ij, res)              # hold the ij reference so its id can't alias a stale entry
    return res


# Cache of block-diagonal ones matrices [nnz*W, nnz] used to turn the rotate_bw coefficient-adjoint
# reductions (nnz per-nonzero dot products) into a single dense GEMM. Keyed by (nnz, W, device id).
_ONES_BD: dict = {}


def _ones_bd(ttnn, device, nnz, W):
    """A [nnz*W, nnz] 0/1 block-diagonal selector: column k is 1 over rows [k*W:(k+1)*W]. Left-
    multiplying a [E, nnz*W] tensor of per-nonzero products by it segment-sums each W-block, i.e.
    computes the nnz row-wise dot products as ONE matmul (matrix engine, fp32 accum) instead of
    nnz separate reductions. 1.0 is exact in bf16 so this is a plain fp32-accumulated sum
    (bit-equivalent to ttnn.sum up to reduction order)."""
    key = (nnz, W, id(device))
    t = _ONES_BD.get(key)
    if t is None:
        blk = torch.block_diag(*[torch.ones(W, 1) for _ in range(nnz)])   # [nnz*W, nnz]
        t = ttnn.from_torch(blk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ONES_BD[key] = t
    return t


def gather_coef(wigner: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor):
    """Gather the packed coefficients ``[E, nnz]`` at a KNOWN (already-derived) sparsity pattern
    ``(ii, jj)``. For a fixed topology the pattern never changes, so re-running :func:`pack`'s
    ``amax`` reduction every step (trace refresh) is wasted work -- cache ``ii, jj`` once and call
    this. Bit-identical to ``pack``'s coef output."""
    return wigner[:, ii, jj].contiguous()


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
    """``x_flat`` ``[E, n_in*W]`` (W channels per coordinate) -> rotated ``[E, n_out*W]`` via the
    fused kernel. ``n_out`` defaults to ``n_in`` (square rotation, uma-s). Raises for a shape the
    kernel cannot run (see :data:`_UNSUPPORTED`)."""
    n_out = n_in if n_out is None else n_out
    deg, ks, js = _group(ij, n_out, "i")
    if not _kernel_ok(deg, n_in, n_out, len(ij), W):
        raise RuntimeError(_UNSUPPORTED.format(n_in=n_in, n_out=n_out, W=W, nnz=len(ij)))
    # bf8-edge: the kernel requires coef in the SAME dtype as its x input; the cast is done once
    # inside _coef_exp and cached (not per rotate call).
    ce_dev = _coef_exp(ttnn, coef, dtype=x_flat.dtype)
    return _fused_op(ttnn)(x_flat, ce_dev, n_in, n_out, W, deg, ks, js)


def rotate_bw(ttnn, x_in_flat, g_out_flat, ij, coef, n_in, W, device, n_out=None):
    """VJP of :func:`rotate`. Returns (g wrt ``x_in`` flat ``[E,n_in*W]``, g wrt the packed
    coefficients ``[E, nnz]``). The coefficient adjoint is scattered back to a dense
    ``[E, n_out, n_in]`` on host (:func:`scatter_coef`) to drive the geometric ``dW/dpos``
    autograd for the force."""
    n_out = n_in if n_out is None else n_out
    E = x_in_flat.shape[0]
    from .device import compute_kernel_config

    deg, ks, is_ = _group(ij, n_in, "j")
    if not _kernel_ok(deg, n_in, n_out, len(ij), W):
        raise RuntimeError(_UNSUPPORTED.format(n_in=n_in, n_out=n_out, W=W, nnz=len(ij)))
    # g_in[j] = sum_{(i,j,k)} coef_k * gout_i is the SAME fused rotation with the pattern grouped
    # by input block j (fan-in over the output rows i).
    ce_dev = _coef_exp(ttnn, coef, dtype=g_out_flat.dtype)      # bf8-edge: cached bf8 coef
    g_in_flat = _fused_op(ttnn)(g_out_flat, ce_dev, n_out, n_in, W, deg, ks, is_)

    # coefficient adjoint gc[k] = sum_W(gout_i * in_j). Large graphs: the custom fused mul-reduce
    # kernel (products L1-resident, one accumulating matmul reduces + places into gc[E,nnz]).
    if _gc_kernel_ok(n_out, n_in, len(ij), W, E):
        is_l = [i for i, j in ij]; js_l = [j for i, j in ij]
        sel = _sel(ttnn, device)
        if g_out_flat.dtype != sel.dtype:                      # bf8-edge: gc needs gout/xin/sel same dtype
            sel = ttnn.typecast(sel, g_out_flat.dtype)
        gc = _gc_op(ttnn)(g_out_flat, x_in_flat, sel, n_out, n_in, W, is_l, js_l)
        return g_in_flat, gc
    # small/mid-N: the gc kernel's product CB would clash with the trace's L1-resident tensors, so
    # segment-sum the nnz per-nonzero dot products as ONE dense GEMM (block-diagonal ones matrix,
    # matrix engine, fp32 accumulate). PCC ~1.0 vs the kernel; correct + cheap at these sizes.
    in_cols = ttnn.split(x_in_flat, W, dim=1)
    gout_cols = ttnn.split(g_out_flat, W, dim=1)
    prods = [ttnn.multiply(gout_cols[i], in_cols[j]) for (i, j) in ij]
    P = ttnn.concat(prods, dim=1)                              # [E, nnz*W]
    gc = ttnn.matmul(P, _ones_bd(ttnn, device, len(ij), W),
                     compute_kernel_config=compute_kernel_config())   # [E, nnz]
    return g_in_flat, gc


def scatter_coef(g_coef: torch.Tensor, ij, n_out: int, n_in: int = None) -> torch.Tensor:
    """``[E, nnz]`` coefficient adjoints -> dense ``[E, n_out, n_in]`` (zeros off-pattern)."""
    n_in = n_out if n_in is None else n_in
    E = g_coef.shape[0]
    g = torch.zeros(E, n_out, n_in, dtype=g_coef.dtype)
    ii = torch.tensor([i for i, j in ij]); jj = torch.tensor([j for i, j in ij])
    g[:, ii, jj] = g_coef                                       # vectorized scatter (was a py loop)
    return g
