"""Shape bucketing for screening workloads: pad edges (and optionally nodes) to a small
fixed ladder so compiled kernel shapes repeat across differently-sized systems.

Why this exists (measured by benchmarks/bench_compile_pain.py on qb1, orb-v3-conservative-
inf-omat): ttnn compiles kernels per tile-padded shape, and a screening stream of
continuously-sized systems almost never repeats a shape -- a brand-new size costs ~40 s of
compiles on first eval (vs ~1.4 s disk-warm, ~0.1-0.3 s resident-warm), and even crossing ONE
32-row edge-tile boundary at a familiar size costs ~11 s. With edges padded to the ladder
below, a whole stream collapses to <=8 shapes: each compiles once ever, then everything is warm.

The ladder: ratio ~1.55 between consecutive buckets, covering ~1k..22k edges (the production
screening range; bench_multicard's default mix is ~2.2k edges/system). For log-uniform
system sizes the mean padding overhead is ~(r-1)/ln(r) - 1 ~= 23% of one cheap O(E) message-
passing -- deliberately the opposite tradeoff from padding an O(N^3) trunk. Systems above the
top bucket run unpadded (their compile amortizes over a long eval); systems below the bottom
bucket pad up to it (absolute cost negligible at that size).

Bit-exactness (gated by tests/test_bucketing.py: energies AND forces, maxdiff 0.0, padded
vs unpadded): sentinel edges are self-loops on node 0 with displacement exactly r_max along
x, so host_cutoff's (r < r_max) mask gives them an exactly-0.0 envelope -> their
messages are exactly 0.0 on device, and their device adjoints are exactly 0.0 in the force VJP.
The scatter gather tables are built from the TRUE edges only (same per-node reduction order as
unpadded, trailing sentinel slots gather the zero pad row), so real-node sums are bitwise
unchanged. Node padding (when enabled) adds isolated zero-feature nodes: no edge references
them, the energy head's mean is taken over a 0/1 mask so they contribute exactly nothing, and
positions/forces are never padded at all (pos is host-only; device adjoints at padded nodes
are discarded with the padded feature rows).
"""
from __future__ import annotations

import torch

# Edge ladder: ratio ~1.55, tile-multiples, ~1k..22k edges. Mean pad overhead ~23% (log-uniform).
EDGE_BUCKETS = (1024, 1584, 2464, 3808, 5920, 9152, 14208, 22016)

# Node ladder: tile-multiples, ~1.5 ratio, 32..1216 atoms. Only used when node-dim shapes are
# observed to drive compiles (see benchmarks/probe_shapes.py); edge-only bucketing is the default.
NODE_BUCKETS = (32, 64, 128, 192, 320, 512, 768, 1216)


def bucket_size(n: int, ladder=None) -> int:
    """Smallest ladder rung >= n; n itself (unpadded) when above the top rung."""
    ladder = EDGE_BUCKETS if ladder is None else ladder
    for b in ladder:
        if n <= b:
            return b
    return n


def pad_edge_index(senders: torch.Tensor, receivers: torch.Tensor, cell_shift: torch.Tensor,
                   e_bucket: int, r_max: float):
    """Pad edge index + shifts to e_bucket with zero-contributing sentinel edges.

    Sentinel: self-loop on node 0 with displacement exactly (r_max, 0, 0) -- finite geometry
    for the differentiable edge features (no 0/0 in the spherical-harmonic normalize), and an
    exactly-0.0 attention cutoff via host_cutoff's r < r_max mask. No-op when already at
    the bucket."""
    e = senders.shape[0]
    pad = e_bucket - e
    if pad <= 0:
        return senders, receivers, cell_shift
    zeros = torch.zeros(pad, dtype=senders.dtype)
    senders = torch.cat([senders, zeros])
    receivers = torch.cat([receivers, zeros])
    shift_pad = torch.zeros(pad, 3, dtype=cell_shift.dtype)
    shift_pad[:, 0] = r_max
    cell_shift = torch.cat([cell_shift, shift_pad])
    return senders, receivers, cell_shift


def pad_graph(senders: torch.Tensor, receivers: torch.Tensor, cell_shift: torch.Tensor, *,
              r_max: float, max_num_neighbors: int, ladder=None):
    """Pad an edge set to its ladder rung and return the ``OrbGraphContext`` kwargs that keep
    padded evaluation bit-exact: ``(senders, receivers, cell_shift, gather_kwargs)``.

    ``gather_kwargs`` builds the scatter tables from the TRUE edges only (real nodes keep their
    unpadded reduction order; sentinel message rows are never gathered) with the table width
    floored at ``max_num_neighbors`` so the data-dependent max degree stays out of the compiled-
    shape key. The returned tensors feed the DEVICE path only -- host post-processing (ZBL
    corrections, edge vectors) must keep using the true, unpadded edge set: the sentinel's ZBL
    term at exactly r_max is not guaranteed 0.0."""
    e_true = int(senders.shape[0])
    e_bucket = bucket_size(e_true, ladder)
    senders, receivers, cell_shift = pad_edge_index(senders, receivers, cell_shift,
                                                    e_bucket, r_max)
    gkw = dict(gather_edge_count=e_true, gather_width=max_num_neighbors)
    return senders, receivers, cell_shift, gkw


def pad_node_rows(t: torch.Tensor, n_bucket: int) -> torch.Tensor:
    """Pad a [N, ...] host tensor with zero rows to n_bucket (no-op when already there)."""
    n = t.shape[0]
    pad = n_bucket - n
    if pad <= 0:
        return t
    return torch.cat([t, torch.zeros(pad, *t.shape[1:], dtype=t.dtype)])


def node_mask(n_true: int, n_bucket: int) -> torch.Tensor:
    """[n_bucket, 1] float mask: 1.0 on real node rows, 0.0 on padded rows."""
    m = torch.zeros(n_bucket, 1)
    m[:n_true] = 1.0
    return m
