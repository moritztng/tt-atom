# benchmarks/archive/

Scripts for **concluded, closed** Orb-v3 perf investigations. Moved here (not deleted)
during the 2026-07-19 benchmarks/ consolidation pass so nothing is lost. Each entry points
at the memory lesson that closed the underlying investigation; un-archive if a genuinely new
angle reopens one.

These scripts are frozen artifacts — they are not maintained for re-running and their imports
may reference each other (not the live tree). Use the live scripts in the parent directory for
current measurements.

## scatter-add as aggregation replacement
- `bench_orb_scatter.py` — `ttnn.scatter_add` vs the gather/reduce segment sum. Closed as an
  8.1x regression; do not revisit as an aggregation optimization.
  Lesson: `tt-atom-orb-hardware-limit-accel` (knowledgebase memory/_global/).

## aggregation-scatter row-major concat layout fix
- `bench_orb_scatter_ab.py`, `bench_orb_scatter_e2e_ab.py` — A/B (micro and end-to-end) for the
  new `scatter.segment_sum` (ROW_MAJOR concat) vs the old TILE-concat path.
- `bench_orb_scatter_internals.py` — micro-profile of `segment_sum` sub-steps that located the
  real cost in the TILE concat (not the `to_layout` conversion intuition guessed first).
- `bench_orb_scatter_layout.py` — quantified layout-only fixes (keep the reduction in
  ROW_MAJOR / share one conversion across sent+recv scatters) before deciding a custom kernel
  was not worth it.
  The ROW_MAJOR-concat fix landed (default-on, `TT_ATOM_ORB_SCATTER_RM=0` restores the old path).
  Lesson: `tt-atom-orb-aggregation-scatter-fusion-result`.

## edge-MLP Linear+SiLU fusion
- `bench_orb_edge_mlp.py` — device microbench of the edge MLP and the
  `ttnn.linear(..., activation="silu")` forward-only fusion candidate. Forward fusion was a
  dead-end (neutral/negative); fused SiLU backward landed a modest 1.03-1.08x.
  Lesson: `tt-atom-orb-edge-mlp-fusion-result`.

## edge-tiled L1-streaming megakernel
- `bench_orb_edge_streaming_floor.py` — optimistic lower-bound floor for streaming edge chunks
  through L1. The megakernel was evaluated and rejected (0.42-0.78x once cache exports are
  included); the minimal_matmul factory was landed instead.
  Lesson: `tt-atom-orb-edge-megakernel-result`.

## minimal_matmul edge-MLP factory
- `bench_orb_minimal_matmul.py` — viability scout for `ttnn.experimental.minimal_matmul` as the
  edge-MLP Linear factory.
- `bench_orb_minimal_matmul_e2e_ab.py` — controlled whole-MD-step A/B for the minimal_matmul
  factory (1.08-1.13x real MD speedup, merged default-on via `TT_ATOM_ORB_MINIMAL_MATMUL`).
  Lesson: `tt-atom-orb-edge-megakernel-result`.
