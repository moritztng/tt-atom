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

import os

import torch

# Route the per-edge rotation through the custom fused ttnn kernel
# (ttnn.experimental.fused_rotate) instead of the nnz-way addcmul MAC. One kernel launch, one
# DRAM read of x + one write of out; ~14x faster than the MAC form in isolation. Requires the
# source-ttnn build that has the op (see custom_kernels/). Off by default (env-gated).
_FUSED = os.environ.get("TT_ATOM_FUSED_ROTATE") == "1"
# separate backward toggle (defaults to _FUSED); lets us A/B forward-only vs forward+backward.
_FUSED_BW = _FUSED
# id(coef ttnn tensor) -> (coef_ref, coef_exp_dev, deg, ks, js). Holds the coef reference so its
# python id cannot be recycled onto a stale entry. Cleared when it grows past a step's worth.
_FUSED_CACHE: dict = {}


def _fused_op(ttnn):
    return ttnn._ttnn.operations.experimental.fused_rotate


def _gc_op(ttnn):
    return ttnn._ttnn.operations.experimental.fused_rotate_gc


# Route the rotate_bw coefficient adjoint (dE/dcoef) through the custom fused mul-reduce kernel
# instead of the [E, nnz*W] product concat + ones_bd GEMM. The GEMM is memory-bound on the ~823MB
# P concat (measured 111ms/step at N=1000); the kernel keeps the products L1-resident and does the
# W-reduction + column placement in one accumulating matmul, reading gout+xin once. Env-gated (A/B).
_FUSED_GC = os.environ.get("TT_ATOM_FUSED_GC", "1") == "1"

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


# The gc kernel's product CB (32*Wt = 512KB at W=256) leaves little L1 headroom, so it can clash
# with the trace's L1-resident backward tensors (grid/spectral) in the SMALL/MID-N regime (measured
# clash at N=216 E~10k and N=512 E~23k; clash-free at N=1000 E~46k and N=1728 E~79k where the large-N
# path spills those tensors to DRAM). The gc kernel only matters where the device replay dominates
# (large N); small/mid-N MD is dispatch-bound and the ones_bd GEMM there is cheap. So gate the kernel
# to large graphs and fall back to the GEMM below -- no small-N regression, no clash. (A batched-CB
# variant that fits small-N exists but its reload-add serial chain regressed large-N ~35%, so the
# fast full-tile kernel + this edge gate is the shippable choice.)
_GC_MIN_EDGES = int(os.environ.get("TT_ATOM_GC_MIN_EDGES", "45000"))


def _gc_kernel_ok(n_out, n_in, nnz, W, E) -> bool:
    """L1 budget for the gc kernel CBs (see gc_program_factory): 2*(n_out+n_in)*Wt gout/xin +
    32 sel + 32*Wt prod + 2*ceil(nnz/32) out tiles. uma-s fits (~1.16MB); uma-m (W=256, 19x25)
    overflows. Also require a large graph so the CBs co-reside with the trace's L1 tensors."""
    Wt = W // 32
    out_tiles = (nnz + 31) // 32
    tiles = 2 * (n_out + n_in) * Wt + 32 + 32 * Wt + 2 * out_tiles
    return tiles * _TILE_BYTES <= _L1_CB_BUDGET and E >= _GC_MIN_EDGES


# Two hard limits gate the kernel to shapes it can run, with a MAC fallback otherwise (so the fast
# kernel is a drop-in for ANY checkpoint, not just uma-s):
#   (1) The compute kernel fans-in all `d` per-block products into dst[0..d-1] and sums them there,
#       so the max fan-in degree must fit the fp32 DST register file (dst_full_sync_en -> 8 slots).
#   (2) The program factory statically allocates double-buffered CBs for the x, coef and out tiles
#       on each core; their total must fit L1 (1.5 MB). uma-m (W=256, n_in=25, n_out=19) blows past
#       it (~2 MB) and TT_THROWs at program build; uma-s (W=128, 9x9) is ~0.4 MB and fits.
_MAX_DST = 8
_TILE_BYTES = 2048            # bf16 32x32 tile
_L1_CB_BUDGET = 1_400_000     # < 1.5 MB L1, leaving headroom for runtime/semaphores


def _cb_bytes(n_in, n_out, nnz, W) -> int:
    Wt = W // 32
    return 2 * (n_in * Wt + n_out * Wt + nnz) * _TILE_BYTES


def _kernel_ok(deg, n_in, n_out, nnz, W) -> bool:
    return (len(deg) > 0 and max(deg) <= _MAX_DST
            and _cb_bytes(n_in, n_out, nnz, W) <= _L1_CB_BUDGET)


_GROUP_CACHE: dict = {}


def _coef_exp(ttnn, coef, dtype=None):
    """Expand compact [E, nnz] device coefficients to [E, nnz*32] ON DEVICE (each nonzero's
    coef broadcast across its 32-column tile). Cached by (id(coef), dtype); holds the coef
    reference so its python id cannot be recycled onto a stale entry. ``dtype`` (bf8-edge) casts
    the expanded coef to match the kernel's bf8 x input — CACHED, so the per-step trace refresh
    does the typecast once, not on every rotate call (16x/step of a ~100MB tensor)."""
    # coef is stored ROW_MAJOR (cheap refresh); tilize on device here for the kernel (see
    # GraphContext). Cache keyed on the ORIGINAL (persistent) coef so repeat hits across calls.
    orig = coef
    def _tile(c):
        return ttnn.to_layout(c, ttnn.TILE_LAYOUT) if c.layout != ttnn.TILE_LAYOUT else c
    def _build():
        ce = ttnn.repeat_interleave(_tile(orig), 32, dim=1)     # [E, nnz*32]
        if dtype is not None and ce.dtype != dtype:
            ce = ttnn.typecast(ce, dtype)
        return ce
    if os.environ.get("TT_ATOM_FUSED_NOCACHE") == "1":
        return _build()
    key = (id(orig), dtype)
    b = _FUSED_CACHE.get(key)
    if b is not None and b[0] is orig:
        return b[1]
    ce_dev = _build()
    if len(_FUSED_CACHE) > 64:
        _FUSED_CACHE.clear()
    _FUSED_CACHE[key] = (orig, ce_dev)
    return ce_dev


def _group(ij, nblocks, by):
    """Group the (i,j) nonzeros by output block for the fused kernel. ``by='i'`` (forward:
    output block = i, fan-in over j) or ``by='j'`` (backward g_in: output block = j, fan-in
    over i). Returns (deg[nblocks], ks[nnz] coef-tile index, js[nnz] the *other* index)."""
    key = (id(ij), nblocks, by)
    g = _GROUP_CACHE.get(key)
    if g is not None:
        return g
    deg = [0] * nblocks
    ks: list = []
    other: list = []
    for b in range(nblocks):
        for k, (i, j) in enumerate(ij):
            blk, oth = (i, j) if by == "i" else (j, i)
            if blk == b:
                ks.append(k); other.append(oth); deg[b] += 1
    g = (deg, ks, other)
    _GROUP_CACHE[key] = g
    return g


# Cache of block-diagonal ones matrices [nnz*W, nnz] used to turn the rotate_bw coefficient-adjoint
# reductions (nnz per-nonzero dot products) into a single dense GEMM. Keyed by (nnz, W, device id).
_ONES_BD: dict = {}


def _ones_bd(ttnn, device, nnz, W):
    """A [nnz*W, nnz] 0/1 block-diagonal selector: column k is 1 over rows [k*W:(k+1)*W]. Left-
    multiplying a [E, nnz*W] tensor of per-nonzero products by it segment-sums each W-block, i.e.
    computes the nnz row-wise dot products as ONE matmul (matrix engine, fp32 accum) instead of
    nnz separate reductions -- ~1.9x faster at E~46k. 1.0 is exact in bf16 so this is a plain
    fp32-accumulated sum (bit-equivalent to ttnn.sum up to reduction order)."""
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
    """``x_flat`` ``[E, n_in*W]`` (W channels per coordinate) -> rotated ``[E, n_out*W]``.
    ``n_out`` defaults to ``n_in`` (square rotation, uma-s); pass it for the rectangular
    reduced-m-space rotation (uma-m)."""
    n_out = n_in if n_out is None else n_out
    if _FUSED:
        deg, ks, js = _group(ij, n_out, "i")
        if _kernel_ok(deg, n_in, n_out, len(ij), W):
            # bf8-edge: the kernel requires coef in the SAME dtype as its bf8 x input; the cast is
            # done once inside _coef_exp and CACHED (not per rotate call).
            ce_dev = _coef_exp(ttnn, coef, dtype=x_flat.dtype)
            return _fused_op(ttnn)(x_flat, ce_dev, n_in, n_out, W, deg, ks, js)
    E = x_flat.shape[0]
    # non-fused fallback (the fused kernel rejects this pattern — e.g. the rectangular reduced-m
    # uma-m rotation). Splitting a bf8 tensor into [E,1] columns yields MIXED per-column dtypes
    # (bf8 is block-float: a width-1 slice can't carry the tile's shared exponent), so the
    # addcmul MAC below would mix bf8/bf16 operands. Run the MAC in bf16 — cast the bf8-edge
    # activation up once and cast the concatenated result back once (2 casts, vs a per-column
    # fixup); the coef is already bf16 here. bf16 is also the safer dtype for the fallback.
    odt = x_flat.dtype
    if odt == ttnn.bfloat8_b:
        x_flat = ttnn.typecast(x_flat, ttnn.bfloat16)
    if coef.dtype == ttnn.bfloat8_b:                         # --fast stores coef bf8; same split issue
        coef = ttnn.typecast(coef, ttnn.bfloat16)
    cols = ttnn.split(x_flat, W, dim=1)                      # n_in [E,W] blocks in one dispatch
    # split the [E,nnz] coefficients into nnz [E,1] broadcast columns in ONE dispatch instead of
    # nnz per-nonzero ttnn.slice calls. The rotation is the largest single consumer of eager ttnn
    # dispatches (~49% at N=1000) and the eager MD path is host-dispatch bound; the coef slices are
    # ~zero-traffic (each [E,1]) so collapsing them to one op cuts dispatch with no extra DRAM.
    if coef.layout != ttnn.TILE_LAYOUT:                      # coef stored RM (cheap refresh)
        coef = ttnn.to_layout(coef, ttnn.TILE_LAYOUT)
    ccols = ttnn.split(coef, 1, dim=1)
    out = [None] * n_out
    for k, (i, j) in enumerate(ij):
        c = ccols[k]                                          # [E,1] broadcast
        # fuse the accumulate multiply+add into one addcmul (out[i] += cols[j]*c). Rotation is
        # dispatch/op-count-bound (many tiny ops), so folding the separate add drops both a kernel
        # launch and a DRAM round-trip of the partial sum. Bit-identical to multiply-then-add.
        if out[i] is None:
            out[i] = ttnn.multiply(cols[j], c)
        else:
            out[i] = ttnn.addcmul(out[i], cols[j], c)
    for i in range(n_out):
        if out[i] is None:
            # Uncovered output coordinate (no structural nonzero maps to it — happens for the
            # rectangular reduced-m rotation / uma-m, or a geometry whose Wigner has an all-zero
            # row across edges). Emit an on-device zero block via multiply-by-0.0 of an existing
            # [E,W] column rather than ttnn.zeros: a host constant write is forbidden inside a
            # captured trace ("Writes are not supported during trace capture"), whereas this is a
            # plain eltwise. Bit-exact (0.0*finite == 0.0). cols[0] always exists (n_in>=1).
            out[i] = ttnn.multiply(cols[0], 0.0)
    res = ttnn.concat(out, dim=1)
    return ttnn.typecast(res, odt) if res.dtype != odt else res


def rotate_bw(ttnn, x_in_flat, g_out_flat, ij, coef, n_in, W, device, n_out=None):
    """VJP of :func:`rotate`. Returns (g wrt ``x_in`` flat ``[E,n_in*W]``, g wrt the packed
    coefficients ``[E, nnz]``). The coefficient adjoint is scattered back to a dense
    ``[E, n_out, n_in]`` on host (:func:`scatter_coef`) to drive the geometric ``dW/dpos``
    autograd for the force."""
    n_out = n_in if n_out is None else n_out
    E = x_in_flat.shape[0]
    from .device import compute_kernel_config
    if _FUSED_BW and _kernel_ok(_group(ij, n_in, "j")[0], n_in, n_out, len(ij), W):
        # g_in[j] = sum_{(i,j,k)} coef_k * gout_i is the SAME fused rotation with the pattern
        # grouped by input block j (fan-in over the output rows i). The coefficient adjoint gc
        # (below) still needs the per-nonzero products, so those + the ones_bd GEMM are unchanged.
        ce_dev = _coef_exp(ttnn, coef, dtype=g_out_flat.dtype)   # bf8-edge: cached bf8 coef
        deg, ks, is_ = _group(ij, n_in, "j")
        g_in_flat = _fused_op(ttnn)(g_out_flat, ce_dev, n_out, n_in, W, deg, ks, is_)
        if os.environ.get("TT_ATOM_ABLATE_GC") == "1":
            return g_in_flat, ttnn.multiply(coef, 0.0)          # PERF PROBE: skip the gc GEMM
        # coefficient adjoint gc[k] = sum_W(gout_i * in_j): custom fused mul-reduce kernel (products
        # L1-resident, one accumulating matmul reduces+places into gc[E,nnz]) instead of the ~823MB
        # product-concat + ones_bd GEMM. PCC ~1.0 vs the GEMM; the ttnn path stays as the fallback.
        if _FUSED_GC and _gc_kernel_ok(n_out, n_in, len(ij), W, E):
            is_l = [i for i, j in ij]; js_l = [j for i, j in ij]
            sel = _sel(ttnn, device)
            if g_out_flat.dtype != sel.dtype:                # bf8-edge: gc needs gout/xin/sel same dtype
                sel = ttnn.typecast(sel, g_out_flat.dtype)
            gc = _gc_op(ttnn)(g_out_flat, x_in_flat, sel, n_out, n_in, W, is_l, js_l)
            return g_in_flat, gc
        in_cols = ttnn.split(x_in_flat, W, dim=1)
        gout_cols = ttnn.split(g_out_flat, W, dim=1)
        prods = [ttnn.multiply(gout_cols[i], in_cols[j]) for (i, j) in ij]
        P = ttnn.concat(prods, dim=1)                            # [E, nnz*W]
        gc = ttnn.matmul(P, _ones_bd(ttnn, device, len(ij), W),
                         compute_kernel_config=compute_kernel_config())
        return g_in_flat, gc
    # non-fused fallback: as in rotate(), a bf8 split yields mixed per-column dtypes, so run the
    # MAC in bf16 (cast the bf8-edge adjoints/inputs up, cast g_in back at the return).
    odt = g_out_flat.dtype
    if g_out_flat.dtype == ttnn.bfloat8_b:
        g_out_flat = ttnn.typecast(g_out_flat, ttnn.bfloat16)
    if x_in_flat.dtype == ttnn.bfloat8_b:
        x_in_flat = ttnn.typecast(x_in_flat, ttnn.bfloat16)
    if coef.dtype == ttnn.bfloat8_b:
        coef = ttnn.typecast(coef, ttnn.bfloat16)
    in_cols = ttnn.split(x_in_flat, W, dim=1)                    # n_in [E,W] blocks in one dispatch
    gout_cols = ttnn.split(g_out_flat, W, dim=1)                 # n_out [E,W] blocks in one dispatch
    if coef.layout != ttnn.TILE_LAYOUT:                          # coef stored RM (cheap refresh)
        coef = ttnn.to_layout(coef, ttnn.TILE_LAYOUT)
    ccols = ttnn.split(coef, 1, dim=1)                           # nnz [E,1] cols in one dispatch
    g_in = [None] * n_in
    prods = [None] * len(ij)
    for k, (i, j) in enumerate(ij):
        c = ccols[k]
        # fuse the g_in accumulate (g_in[j] += gout_cols[i]*c) into one addcmul (drops a kernel
        # launch + a DRAM round-trip of the partial sum vs multiply-then-add). Bit-identical,
        # mirrors the forward rotate().
        if g_in[j] is None:
            g_in[j] = ttnn.multiply(gout_cols[i], c)
        else:
            g_in[j] = ttnn.addcmul(g_in[j], gout_cols[i], c)
        prods[k] = ttnn.multiply(gout_cols[i], in_cols[j])       # [E,W] per-nonzero product
    # coefficient adjoint gc[k] = sum_W(gout_i * in_j): the nnz row-wise dot products done as ONE
    # dense GEMM (segment-sum by a block-diagonal ones matrix) instead of nnz separate reductions
    # -- ~1.9x faster at E~46k (matrix engine, fp32 accumulate). PCC ~1.0 vs the per-nonzero sums.
    P = ttnn.concat(prods, dim=1)                                # [E, nnz*W]
    gc = ttnn.matmul(P, _ones_bd(ttnn, device, len(ij), W),
                     compute_kernel_config=compute_kernel_config())   # [E, nnz]
    for j in range(n_in):
        if g_in[j] is None:
            # Uncovered input coordinate — see rotate(); emit the zero block on device (trace-safe)
            # instead of ttnn.zeros (a host write forbidden during trace capture). gout_cols[0]
            # always exists (n_out>=1). Bit-exact zero.
            g_in[j] = ttnn.multiply(gout_cols[0], 0.0)
    g_in_flat = ttnn.concat(g_in, dim=1)
    if g_in_flat.dtype != odt:
        g_in_flat = ttnn.typecast(g_in_flat, odt)
    return g_in_flat, gc


def scatter_coef(g_coef: torch.Tensor, ij, n_out: int, n_in: int = None) -> torch.Tensor:
    """``[E, nnz]`` coefficient adjoints -> dense ``[E, n_out, n_in]`` (zeros off-pattern)."""
    n_in = n_out if n_in is None else n_in
    E = g_coef.shape[0]
    g = torch.zeros(E, n_out, n_in, dtype=g_coef.dtype)
    ii = torch.tensor([i for i, j in ij]); jj = torch.tensor([j for i, j in ij])
    g[:, ii, jj] = g_coef                                       # vectorized scatter (was a py loop)
    return g
