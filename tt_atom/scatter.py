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

import os

import numpy as np
import torch


def _scatter_rm_layout() -> bool:
    """Row-major concat + reduction in :func:`segment_sum` (avoids the tile-repacking concat of
    the E-sized message). Bit-exact vs the prior tile-concat path (same reduction order).
    ``TT_ATOM_ORB_SCATTER_RM=0`` restores the prior path for A/B; default on."""
    return os.environ.get("TT_ATOM_ORB_SCATTER_RM", "1") != "0"



def build_gather(idx: torch.Tensor, num_nodes: int, E: int):
    """``idx`` [E] (int, the src or tgt node of each edge) -> (``gather_flat`` [N*Dmax] int32 with
    sentinel ``E`` in the padding slots, ``Dmax``). Row ``n`` of the [N, Dmax] table lists the edge
    indices whose node is ``n``; the sentinel points at the zero pad row appended to the messages."""
    idx_np = idx.detach().cpu().numpy().astype(np.int64)
    deg = np.bincount(idx_np, minlength=num_nodes)
    Dmax = int(deg.max()) if E > 0 else 1
    gather = np.full((num_nodes, Dmax), E, dtype=np.int64)   # sentinel -> zero pad row
    order = np.argsort(idx_np, kind="stable")                # edges grouped by node
    node_of = idx_np[order]
    starts = np.zeros(num_nodes, dtype=np.int64)
    starts[1:] = np.cumsum(deg)[:-1]
    slot = np.arange(E) - starts[node_of]                    # position within the node's group
    gather[node_of, slot] = order                            # original edge index
    return gather.reshape(-1).astype(np.int32), Dmax


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
    msg_rm = ttnn.to_layout(msg, ttnn.ROW_MAJOR_LAYOUT)                # cheap tile->RM (no concat repack)
    zrow = ttnn.multiply(ttnn.slice(msg_rm, [0, 0], [1, W]), 0.0)      # [1,W] device zeros (trace-safe)
    mpad = ttnn.concat([msg_rm, zrow], dim=0)                          # [E+1, W] row append (RM, no repack)
    g = ttnn.embedding(gather_dev, mpad)                               # [N*Dmax, W] row-select (RM)
    g = ttnn.reshape(g, (N, Dmax, W))                                  # metadata-only (RM contiguous)
    return ttnn.to_layout(ttnn.sum(g, dim=1), ttnn.TILE_LAYOUT)        # [N, W] tile (small out)
