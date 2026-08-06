"""Shape bucketing for screening workloads: pad the edge set to a small fixed ladder so
compiled kernel shapes repeat across differently-sized systems.

Why this exists (measured by benchmarks/bench_compile_pain.py on qb1, orb-v3-conservative-
inf-omat): ttnn compiles kernels per tile-padded shape, and a screening stream of
continuously-sized systems almost never repeats a shape -- a brand-new size costs 40-47 s of
compiles on first eval (vs 1.3-1.7 s disk-warm, ~0.1-0.4 s resident-warm; five supercell sizes,
16..432 atoms, warm-repeat noise under 2%), and even crossing ONE 32-row edge-tile boundary at
a familiar size costs ~11 s. With edges padded to the ladder below, a whole stream collapses to
one shape per ladder rung (9 total; the measured 20-system stream used 7): each compiles once
ever, then everything is warm.

The ladder: ratio ~1.55 between consecutive buckets from 1024 up, covering ~1k..22k edges
(the production screening range; bench_multicard's default mix is ~2.2k edges/system). For
log-uniform system sizes the mean padding overhead is ~(r-1)/ln(r) - 1 ~= 23% of one cheap
O(E) message-passing -- deliberately the opposite tradeoff from padding an O(N^3) trunk.
Systems above the top bucket run unpadded (their compile amortizes over a long eval).

The isolated 256 bottom rung is a CORRECTNESS boundary, not a perf choice: the backward
matmuls switch their fp32 accumulation config at 9 M-tiles (measured on qb1: K=256 [E,C]@[C,H]
true rows are bitwise identical pairwise across M=96..256 and across M=288..1024, but the two
regimes differ). A system at <=8 M-tiles padded past the boundary is NOT bit-exact in the
force VJP, so systems with E <= 256 pad only to 256 (same regime; padding cost at that size
is negligible absolute); everything larger pads into the >=1024 regime on both sides.

Bit-exactness (gated by tests/test_bucketing.py: energies AND forces, maxdiff 0.0, padded
vs unpadded) rests on three constructions:

1. The encoder always runs at the TRUE edge count; zero pad rows enter POST-encoder as a
   device concat of the encoded edge block. This is the load-bearing detail: narrow-K device
   matmuls are NOT M-shape-stable -- the fp32 accumulation order inside a tile can change
   with the M-tile count, flipping bf16 ULPs at TRUE rows (measured on qb1, direct-20-omat,
   2x2x2 Si: padding E 252 -> 1024 pre-encoder moved encoder-output true rows by 2^-9 and
   final forces by 0.015 eV/A; padding post-encoder is maxdiff 0.0). The wider-K layer
   matmuls (K = latent / 3*latent) measured M-stable (true-row diff 0.0 across M 252 -> 1024),
   and the conservative path is bit-exact through 5 layers + the full VJP up to E = 22016.
2. Pad rows are gated to exactly zero by their 0.0 attention-cutoff envelope (sigmoid
   attention x cutoff -> exactly-0.0 messages and device adjoints). Host edge features are
   computed on the TRUE edges only and never re-padded: the fp32 vectorized transcendentals
   in the radial basis/envelope are not length-stable (measured: 2 of 5438 cutoff rows off by
   ~1e-7 when the same true edge set is evaluated inside a 5438 -> 5920 padded array).
3. The scatter gather tables are built from the TRUE edges only (same per-node reduction
   order as unpadded; trailing sentinel slots gather the zero pad row appended after the
   message block), with the table width floored at the checkpoint's max_num_neighbors so the
   data-dependent max degree stays out of the compiled-shape key.

The conservative force VJP slices the edge adjoints back to the true rows on device BEFORE
the encoder backward (every op in between is rowwise, so slice-before == slice-after
bitwise) -- no pad concat enters the host autograd graph at all. Host post-processing (ZBL
energy/forces/stress, edge vectors, virial) always keeps the true, unpadded edge set.

Node-dim tensors are never padded: whether node-dim shape changes drive compiles of their own
at a fixed edge bucket is an open question, measurable with benchmarks/probe_shapes.py.
"""
from __future__ import annotations

import torch

# Edge ladder: 256 (correctness rung: <=8 M-tiles must not cross the backward matmuls'
# 9-tile accumulation-config switch), then ratio ~1.55 tile-multiples, ~1k..22k edges.
# Mean pad overhead ~23% (log-uniform).
EDGE_BUCKETS = (256, 1024, 1584, 2464, 3808, 5920, 9152, 14208, 22016)


def bucket_size(n: int) -> int:
    """Smallest ladder rung >= n; n itself (unpadded) when above the top rung."""
    for b in EDGE_BUCKETS:
        if n <= b:
            return b
    return n


def pad_edge_index(senders: torch.Tensor, receivers: torch.Tensor, e_bucket: int):
    """Pad the edge index to e_bucket with self-loops on node 0. No-op when already there.

    The pad rows are never gathered (tables index the true edges only); the self-loop target
    is irrelevant."""
    e = senders.shape[0]
    pad = e_bucket - e
    if pad <= 0:
        return senders, receivers
    zeros = torch.zeros(pad, dtype=senders.dtype)
    return torch.cat([senders, zeros]), torch.cat([receivers, zeros])


def pad_host_rows(t: torch.Tensor, e_bucket: int) -> torch.Tensor:
    """Zero-pad a host [E, ...] tensor to e_bucket rows. No-op when already there."""
    pad = e_bucket - t.shape[0]
    if pad <= 0:
        return t
    return torch.cat([t, torch.zeros(pad, *t.shape[1:], dtype=t.dtype)])


def pad_device_rows(ttnn, t, e_bucket: int):
    """Zero-pad a device [E, ...] tensor to e_bucket rows (concat a device zeros block -- a
    device op, so the pattern stays trace-capturable). No-op when already there."""
    pad = e_bucket - t.shape[0]
    if pad <= 0:
        return t
    shape = [pad] + [t.shape[i] for i in range(1, len(t.shape))]
    z = ttnn.zeros(shape, dtype=t.dtype, layout=t.layout, device=t.device())
    return ttnn.concat([t, z], dim=0)


def gather_kwargs(e_true: int, max_num_neighbors: int) -> dict:
    """``OrbGraphContext`` kwargs for a bucketed graph: scatter tables built from the TRUE
    edges only (real nodes keep their unpadded reduction order), table width floored at the
    checkpoint's max_num_neighbors so the data-dependent max degree leaves the compiled-shape
    key. The floor applies even unpadded (exact-rung or above-top-rung systems), so every
    bucketing-on system shares the same table-width key; it is bit-exact vs the default
    tables (tests/test_bucketing.py)."""
    return dict(gather_edge_count=e_true, gather_width=max_num_neighbors)


def pad_graph(senders: torch.Tensor, receivers: torch.Tensor, cutoff: torch.Tensor, *,
              max_num_neighbors: int):
    """Host-side prep for one bucketed graph -- the whole pre-upload half of the construction
    above, shared by every bucketing call site (``OrbCalculator.calculate``,
    ``orb_forces.energy_and_forces``, ``batch._run_orb``).

    Returns ``(senders, receivers, cutoff, graph_kwargs, e_bucket)``: the edge index and the
    per-edge cutoff padded to the ladder rung, the ``OrbGraphContext`` kwargs that keep the
    scatter tables on the true edges, and the rung itself (which the caller needs after the
    encoder, for :func:`pad_device_rows`). The caller keeps its own TRUE ``senders``/
    ``receivers``/``cutoff`` for host post-processing and autograd."""
    e_true = senders.shape[0]
    e_bucket = bucket_size(e_true)
    senders, receivers = pad_edge_index(senders, receivers, e_bucket)
    return (senders, receivers, pad_host_rows(cutoff, e_bucket),
            gather_kwargs(e_true, max_num_neighbors), e_bucket)
