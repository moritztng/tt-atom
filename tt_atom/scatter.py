"""Linear O(E) edge->node scatter-add (replaces the dense one-hot matmul at scale).

The edgewise message block aggregates per-edge messages onto their target node:
``out[n] = sum_{e : tgt[e]==n} m[e]``. The original device implementation is a dense one-hot
matmul ``S[N,E] @ m`` (and its transpose ``S_src`` in the force VJP). Since ``E ~= 46*N`` for a
6A-cutoff periodic graph, that matmul is O(N*E) = O(N^2) compute AND O(N^2) memory (the [N,E]
one-hot alone is 92 MB at N=1000) — the term that makes large-N scaling blow up, while
fairchem/PyG use a linear O(E) scatter_add.

Here the scatter-add is done in O(E): group edges by node into a fixed-max-degree gather table
``gather[N, Dmax]`` (host, once per topology — sentinel ``E`` for the padding slots), gather the
messages into ``[N, Dmax, W]`` (a row-select via ``ttnn.embedding`` against the messages padded
with a single zero row), and reduce over the degree axis. Compute + memory are O(N*Dmax*W) =
O(E*W). Every op is a standard ttnn op that composes into a captured trace; the padding zero row
is produced on device (``multiply`` by 0.0) so no host constant write enters the trace.

Not bit-identical to the matmul (the reduction sums in a different order), but the per-node sum
of ~46 O(1) terms matches to well within the force parity tolerance (PCC ~ 1.0).
"""
from __future__ import annotations

import numpy as np
import torch

from .device import flag


def _scatter_rm_layout() -> bool:
    """Row-major concat + reduction in :func:`segment_sum` (avoids the tile-repacking concat of
    the E-sized message). Bit-exact vs the prior tile-concat path (same reduction order).
    ``TT_ATOM_ORB_SCATTER_RM=0`` restores the prior path for A/B; default on."""
    return flag("TT_ATOM_ORB_SCATTER_RM", default_on=True)


def build_gather(idx: torch.Tensor, num_nodes: int, E: int, sentinel=None, min_width: int = 0):
    """``idx`` [E] (int, the src or tgt node of each edge) -> (``gather_flat`` [N*Dmax] int32 with
    a sentinel in the padding slots, ``Dmax``). Row ``n`` of the [N, Dmax] table lists the edge
    indices whose node is ``n``; the sentinel points at the zero pad row appended to the messages.

    ``E`` is the number of edges to table (``len(idx)``); the sentinel is the index of that zero
    pad row, i.e. the length of the message block the table will index into, which defaults to
    ``E``. The two differ only under bucketing, which tables the TRUE edges while the messages
    are padded to the ladder rung, and so passes the PADDED count as ``sentinel``. ``min_width``
    floors ``Dmax`` (bucketing floors it at the checkpoint's max_num_neighbors, removing the
    data-dependent max degree from the compiled-shape key); extra slots are sentinel-filled, so
    per-node sums keep their exact reduction order either way."""
    idx_np = idx.detach().cpu().numpy().astype(np.int64)
    deg = np.bincount(idx_np, minlength=num_nodes)
    Dmax = int(deg.max()) if E > 0 else 1
    Dmax = max(Dmax, min_width)
    sv = E if sentinel is None else sentinel
    gather = np.full((num_nodes, Dmax), sv, dtype=np.int64)   # sentinel -> zero pad row
    order = np.argsort(idx_np, kind="stable")                # edges grouped by node
    node_of = idx_np[order]
    starts = np.zeros(num_nodes, dtype=np.int64)
    starts[1:] = np.cumsum(deg)[:-1]
    slot = np.arange(E) - starts[node_of]                    # position within the node's group
    gather[node_of, slot] = order                            # original edge index
    return gather.reshape(-1).astype(np.int32), Dmax


def bf16_parts(ttnn, x):
    """A float32 tensor as three float32 tensors whose values are bfloat16-exact and sum to ``x`` (8 mantissa
    bits each). The engine reads float32 through 19-bit registers, so permutes, reductions and gathers of a
    float32 tensor drop ~1e-3 relative; applied to each part and summed, a linear op stays float32-exact."""
    parts, rest = [], ttnn.to_layout(x, ttnn.TILE_LAYOUT)  # typecast wants tile-padded shapes
    for _ in range(3):
        hi = ttnn.typecast(ttnn.typecast(rest, ttnn.bfloat16), ttnn.float32)
        parts.append(hi)
        rest = ttnn.subtract(rest, hi)
    return parts


def linear32(ttnn, op, x):
    """``op(x)`` for a linear op (permute, sum, gather): plain for bfloat16, part by part for float32."""
    if x.dtype != ttnn.float32:
        return op(x)
    a, b, c = (op(p) for p in bf16_parts(ttnn, x))
    return ttnn.add(ttnn.add(a, b), c)


def embed_rows(ttnn, idx, table):
    """``ttnn.embedding`` for any table dtype. The op takes bfloat16 tables only, so a float32 table is gathered
    as its three bfloat16 parts (see ``bf16_parts``) and summed back in float32, exactly. A bfloat16 table takes
    the plain op, unchanged."""
    if table.dtype == ttnn.bfloat16:
        return ttnn.embedding(idx, table)
    # The gather comes out tiled: a row-major typecast of a gather whose row count is not a multiple of 32
    # hangs the card (qb2, 11-27).
    out, rest = None, ttnn.to_layout(table, ttnn.TILE_LAYOUT)
    for last in (False, False, True):
        part = ttnn.typecast(rest, ttnn.bfloat16)
        got = ttnn.typecast(ttnn.embedding(idx, ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.TILE_LAYOUT),
                            ttnn.float32)
        out = got if out is None else ttnn.add(out, got)
        if not last:
            rest = ttnn.subtract(rest, ttnn.typecast(part, ttnn.float32))
    return out if table.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


def segment_sum(ttnn, msg, gather_dev, Dmax, N, W):
    """``msg`` [E, W] (tile) -> ``out`` [N, W] with ``out[n] = sum over gathered edges of msg``.

    ``gather_dev`` is the [N*Dmax] uint32 table from :func:`build_gather` (sentinel ``E``); the
    messages are padded with one on-device zero row so a sentinel gathers zero.

    Layout: ``ttnn.embedding`` requires ROW_MAJOR input, so the message must be row-major before
    the pad-and-gather. The previous sequence did a TILE concat of ``msg`` with a zero row and
    *then* converted the whole [E+1, W] tile tensor to row-major -- the TILE concat is a full
    tile-repacking copy of the E-sized message (~1 ms at 92k edges, the dominant sub-step).
    Converting ``msg`` to row-major first (one cheap tile->RM pass) and concatenating in
    row-major (a row append, no repack) avoids that copy. The reduction also stays in row-major:
    the gathered [N, Dmax, W] tensor is already contiguous, so the reshape is metadata-only and
    ``ttnn.sum`` runs on row-major data, converting only the small [N, W] result back to tile for
    the caller. Same reduction order (sum over the degree axis), so numerically equivalent to the
    prior path -- verified bit-exact on a 216-atom frame (force PCC 1.0, max_abs 0)."""
    if not _scatter_rm_layout():
        zrow = ttnn.multiply(ttnn.slice(msg, [0, 0], [1, W]), 0.0)
        mpad = ttnn.to_layout(ttnn.concat([msg, zrow], dim=0), ttnn.ROW_MAJOR_LAYOUT)
        g = ttnn.embedding(gather_dev, mpad)
        g = ttnn.to_layout(ttnn.reshape(g, (N, Dmax, W)), ttnn.TILE_LAYOUT)
        return ttnn.sum(g, dim=1)
    if msg.dtype == ttnn.float32:  # gather and reduce each bfloat16-exact part: both are exact on parts
        return linear32(ttnn, lambda p: _segment_sum_bf16(ttnn, ttnn.typecast(p, ttnn.bfloat16), gather_dev, Dmax, N, W,
                                                          out_dtype=ttnn.float32), msg)
    return _segment_sum_bf16(ttnn, msg, gather_dev, Dmax, N, W)


def _segment_sum_bf16(ttnn, msg, gather_dev, Dmax, N, W, out_dtype=None):
    msg_rm = ttnn.to_layout(msg, ttnn.ROW_MAJOR_LAYOUT)                # cheap tile->RM (no concat repack)
    zrow = ttnn.multiply(ttnn.slice(msg_rm, [0, 0], [1, W]), 0.0)      # [1,W] device zeros (trace-safe)
    mpad = ttnn.concat([msg_rm, zrow], dim=0)                          # [E+1, W] row append (RM, no repack)
    g = ttnn.embedding(gather_dev, mpad)                               # [N*Dmax, W] row-select (RM)
    g = ttnn.reshape(g, (N, Dmax, W))                                  # metadata-only (RM contiguous)
    if out_dtype is not None:
        g = ttnn.typecast(ttnn.to_layout(g, ttnn.TILE_LAYOUT), out_dtype)
    return ttnn.to_layout(ttnn.sum(g, dim=1), ttnn.TILE_LAYOUT)        # [N, W] tile (small out)
