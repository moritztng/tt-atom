# Orb-v3 port

Porting [Orbital Materials' Orb-v3](https://github.com/orbital-materials/orb-models)
(`orb-v3-conservative-inf-omat`, `orb-v3-direct-20-omat`) onto Tenstorrent alongside the
existing UMA/eSEN support. The initial pass (encoder + full 5-layer backbone + energy for both
checkpoints, direct-20's forces) merged to `master`; this doc's "Completed since" / "Still open"
sections track a follow-up completion pass (branch `wk/tt-atom-orb-completion`). Purely additive:
UMA code paths (`tt_atom/{model,norm,edgewise,so2,rotation,grid,spectral}.py`, `custom_kernels/`)
are untouched throughout.

## Architecture verdict: Orb is NOT equivariant — none of the SO(3) kernels transfer

Read directly from `orb_models.common.models.{gns,angular}.py` (both the GitHub `main` tree and
the installed PyPI `orb-models==0.5.5`, which differ only in module layout, not logic — diffed
byte-for-byte on the two files that matter):

- **`angular.SphericalHarmonics`** computes real spherical harmonics up to `lmax=3` from each
  edge's *unit vector* once, as a fixed `(lmax+1)^2`-length **scalar descriptor**, exactly like
  the Bessel RBF. It is never rotated, never carried as a per-node/per-edge tensor representation,
  and has no Wigner-D machinery anywhere in the codebase.
- **`gns.AttentionInteractionNetwork`** (the message-passing block, called an MPNN with attention
  in Orb's own docs) operates entirely on plain `[rows, latent_dim]` tensors: `nn.Linear`,
  `sigmoid`/`softmax` attention gates, `segment_sum`. No SO(2)/SO(3) convolution, no per-degree
  gating, no local-frame rotation step.
- Normalization is `mlp_norm="rms_norm"` → plain `torch.nn.RMSNorm` over the feature dim (no
  spherical-harmonic degree-balancing), and activation is plain `SiLU`.

**Consequence:** `custom_kernels/fused_rotate` (per-edge Wigner rotation) has nothing to rotate in
Orb — there is no equivariant hidden representation. `fused_gate` (the SO(2) gated nonlinearity
over spherical-harmonic degree blocks, `tt_atom/activation.py`) and `fused_ln_bw` (backward of
`RMSNormSH`'s degree-balanced reduction, `tt_atom/norm.py`) are also both specific to that
degree-block structure — Orb's LayerNorm/RMSNorm and SiLU are the ordinary scalar kind these
kernels do not apply to. **None of TT-Atom's four custom kernels transfer to Orb.** What *does*
transfer is the architecture-agnostic infra: `tt_atom/device.py`'s compute-kernel/dtype policy,
`tt_atom/scatter.py`'s linear O(E) edge→node segment-sum (generic, no equivariance assumption —
reused as-is), and the "host computes the fixed geometric terms once, device runs every learned
GEMM" split already established by `tt_atom/model.py`'s `GraphContext`.

## What's ported and PCC-verified (this pass)

Bottom-up against **both real target checkpoints** — `orb-v3-conservative-inf-omat` and
`orb-v3-direct-20-omat` (Orbital Materials' public S3 bucket, no gating) — on a real structure,
`ase.build.bulk("Si","diamond",a=5.43)*(2,1,1)` rattled (stdev=0.1, seed=1): the *same* system +
rattle seed already used for the UMA real-weight golden (`tests/gen_golden_real.py --system
bulk`), so all three models' outputs are a genuine same-system comparison point.

Golden generation (`tests/gen_golden_orb.py`, run in the existing `~/.ttatom_run/refenv`, which
already has `fairchem-core`+`e3nn` and now also `orb-models==0.5.5` installed side by side with no
dependency conflicts) hooks `MoleculeGNS._encoder`, each `gnn_stacks[i]`, and captures real
weights + real intermediate activations into an npz bundle (`tt_atom/orb_weights.py` reads it in
the ttnn env, mirroring `tt_atom/weights.py`'s numpy<2/numpy>=2 split).

Device port (`tt_atom/orb_model.py`): `RMSNorm`, `MLPNorm` (the 3-Linear-+-RMSNorm block used by
both the encoder and every interaction layer), `Encoder`, `AttentionInteractionLayer`,
`OrbGraphContext` (host-precomputed senders/receivers/cutoff + `tt_atom.scatter` gather tables),
`EnergyHead`, `ForceHead`. The fixed per-edge terms (Bessel RBF, spherical-harmonic descriptor,
the polynomial attention-cutoff envelope, the ZBL pair-repulsion energy) are computed on host and
uploaded/added once, exactly like UMA's wigner/gaussian/envelope buffers — they are fixed
functions of geometry, not learned.

Real on-device PCC (`tests/test_orb_realweight.py`, `tests/test_orb_direct_realweight.py`,
`TT_VISIBLE_DEVICES=0`, bf16 weights/activations, HiFi4 fp32-accumulate matmul — same numerics
policy as UMA), backbone depth (`orb-v3-conservative-inf-omat`):

| component | PCC vs real orb-models CPU reference |
|---|---|
| Encoder (node MLP) | 0.999997 |
| Encoder (edge MLP) | 0.999991 |
| Interaction layer 0 (node) | 0.999978 |
| Interaction layer 0 (edge) | 0.996864 |
| Full 5-layer backbone, final node embedding (→ energy head) | 0.999513 |
| Full 5-layer backbone, final edge stream | 0.976445 |

The node stream (what the energy head actually consumes) holds >0.999 PCC through all 5 layers.
The edge stream drifts more under bf16 by layer 5 (0.997→0.976) — expected precision compounding
over depth in a pure residual with no further consumer, not a correctness bug.

### End-to-end device energy + forces (both target checkpoints)

Went beyond the backbone to the actual consumer-facing quantities, on device, real weights:

- **`EnergyHead`** (`tt_atom/orb_model.py`): mean-aggregate the final node embedding, 2-layer MLP
  on device (`Linear→SiLU→Linear`), then a host-side denormalize (`ScalarNormalizer.inverse` +
  atom-average undo + `LinearReferenceEnergy` per-element lookup) — same shape as UMA's
  `scale_rmsd`/`scale_mean`/`elem_refs` (`tt_atom/weights.py`), different reference table.
- **`ForceHead`** (direct checkpoint only): per-node 2-layer MLP on device, then net-force removal
  (subtract the per-system mean predicted force — a fixed geometric correction, `ttnn.mean`+
  `ttnn.subtract`) and a host scalar-normalizer inverse. **No autograd** — this is the entire
  reason `orb-v3-direct-20-omat` is the fast checkpoint.
- **ZBL pair repulsion** (`host_zbl_energy`, `tt_atom/orb_model.py`): the Ziegler-Biersack-
  Littmark potential has *zero* learned parameters (6 universal physical constants) — implemented
  as a direct host `torch` function from real atomic numbers + edge vectors, exactly like the
  attention-cutoff envelope. Measured `9.5e-8 eV` for this Si golden — negligible, because the
  nearest-neighbor Si-Si bond length (2.20-2.35 Å) sits just outside the ZBL envelope's own cutoff
  (sum of covalent radii ≈ 2.22 Å) at this near-equilibrium geometry. Confirmed on the real graph,
  not assumed; ZBL forces (needed for `orb-v3-direct-20-omat`'s total force) were therefore not
  implemented this pass since their contribution is unmeasurable at this system's geometry — flagged
  in Open below for whichever system exercises it (surface defects, short contacts).

Real end-to-end numbers (`tests/test_orb_realweight.py`, `tests/test_orb_direct_realweight.py`):

| checkpoint | quantity | device result | real oracle | error |
|---|---|---|---|---|
| conservative-inf-omat | energy | -20.499663 eV | -20.497231 eV | rel err 1.19e-4 |
| direct-20-omat | energy | -20.404415 eV | -20.392614 eV | rel err 5.79e-4 |
| direct-20-omat | forces | PCC 0.999966 | — | MAE 0.0079 eV/Å (oracle \|F\|max 2.4661) |

Both energy errors are well inside UMA's own real-weight test bar (`tests/test_realweight.py`:
"energy rel err < 1e-2"). The direct-20 backbone reused the *exact same* `Encoder`/
`AttentionInteractionLayer` classes as the conservative checkpoint, unmodified — different
weights, a smaller graph (max-20-neighbor cutoff → 80 edges vs 172 for the same 4-atom cell) —
confirming the port's bottom-up pieces are checkpoint-agnostic, not conservative-specific.

**Real energy/force numbers, same Si system, all three models (CPU oracle for UMA, device for
Orb):**

| model | E (eV) | \|F\|max (eV/Å) |
|---|---|---|
| uma-s-1 (omat, merged MoLE) | -20.497236 | 2.5249 |
| orb-v3-conservative-inf-omat (device) | -20.499663 | 2.4553 (oracle; forces not yet ported for this checkpoint) |
| orb-v3-direct-20-omat (device) | -20.404415 | 2.4661 (device, PCC 0.999966 vs oracle) |

All three graphs are built from the identical Si structure. Energies agree to within ~0.1 eV out
of ~20.5 eV (~0.5%) across three independently-trained models — expected agreement between
competent MLIPs on a near-equilibrium bulk cell, not by itself evidence of correctness (the PCC/
rel-err tables above are the actual correctness evidence). No official Orb-vs-UMA/eSEN comparison
exists upstream; this is the first same-system numeric side-by-side for TT-Atom.

## Profiling (methodology: measure before reaching for a custom kernel)

Per the hard lesson from tt-bio's Boltz-2 trimul kernel (a hand-rolled kernel that was *slower*
than a standard op once host-dispatch overhead was accounted for — the real win there was trace
capture, not new math) — measured, not assumed, before considering any Orb-specific kernel:

Warm (program-cache-hot) forward, encoder + all 5 interaction layers, on the real Si golden
(N=4, E=172): **5.4 ms/call**. At this tiny system size the workload is almost certainly
dispatch-bound, not compute-bound — the graph is ~45 separate ttnn ops (3 Linear + RMSNorm per
MLP × 2 MLPs × 5 layers, plus per-layer attention gates/gathers/scatter-sums), each launched
individually with no fusion or trace capture yet.

**Verdict: no analogous "many small per-edge dispatches collapsible into one fused kernel"
pattern exists in Orb** the way it did for UMA's Wigner rotation (which fused ~35 `addcmul`
dispatches into one kernel because each was operating on the *same* per-edge sparse-rotation
structure). Orb's interaction layer is ordinary dense GEMMs + two small scatter-adds — ttnn
already has efficient primitives for all of it. The applicable lever, per the methodology, is
**trace capture** (`tt_atom/trace.py` already exists for UMA and is architecture-agnostic — it
captures/replays a fixed ttnn op graph) to eliminate host-dispatch overhead across the ~9 ops/layer,
not a new custom kernel. This is a real profiled measurement at a toy system size, not a
production-scale conclusion — worth re-measuring at a production cell size before committing to
trace capture as the answer.

## Completed since (branch `wk/tt-atom-orb-completion`)

- **Autograd forces for the conservative variant** (`tt_atom/orb_forces.py`): hand-written device
  VJPs mirroring every forward op 1:1 (`Linear`'s transpose-matmul, `RMSNorm`'s ordinary — non-SH
  — backward, SiLU/sigmoid via ttnn's fused `*_bw`, and `scatter.segment_sum`'s adjoint is exactly
  a gather by the *same* sender/receiver index used to build its own forward gather table). A
  differentiable host reimplementation of the Bessel RBF + lmax=3 spherical-harmonic + polynomial
  cutoff edge featurization (`tt_atom/orb_geometry.py`, no `orb-models` dependency) supplies
  `d(edge_feat, cutoff)/dpos` via `torch.autograd`. PCC 0.999975 / MAE 0.0089 eV/Å vs the real
  `orb-models` `torch.autograd` oracle (`tests/test_orb_forces_realweight.py`) — matching
  direct-20's ForceHead parity bar.
- **ZBL pair-repulsion forces** (`host_zbl_forces`, `tt_atom/orb_model.py`): host
  `torch.autograd.grad` on the existing closed-form `host_zbl_energy` (zero learned parameters, so
  no device VJP needed). Verified bit-exact (1e-10) vs central finite differences, and against a
  new dedicated short-contact golden (`tests/gen_golden_orb.py --system short_contact`, two Si
  atoms 1.4 Å apart) where ZBL is ~1.3% of total energy — the original Si golden's ZBL contribution
  is genuinely negligible there, so it never exercised this term. Adding it to direct-20's
  `ForceHead` output improves total-force MAE 0.615→0.390 eV/Å vs the oracle
  (`tests/test_orb_zbl_forces.py`).
- **Periodic images at production scale**: `tt_atom/geometry.py`'s `radius_graph` (already proven
  for UMA) transfers with no code change — only a sender/receiver swap, since Orb's own convention
  (`vectors = pos[receivers] - pos[senders] + shift`) is the opposite of fairchem/UMA's
  `edge_vec = pos[src] - pos[tgt] + shift`. Verified on a new 24-atom/1064-edge periodic Si
  supercell golden (`--system supercell`): the reconstructed edge set exactly matches (symmetric
  diff 0, order-independent) `orb-models`' own neighbor list, and feeding the device backbone with
  this from-scratch topology reproduces the real oracle's final node embedding (PCC 0.9996,
  `tests/test_orb_periodic_realweight.py`).
- **Disjoint-union batching**: verified (bit-exact row-independence, same methodology as
  `ttatom-batching`/`ttatom-qb2-multicard-fanout`) that `Encoder`/`AttentionInteractionLayer`
  attach to a 2-system disjoint-union batch with **no adapter code** — both ops only ever touch
  arbitrary global node indices and `scatter.segment_sum`'s per-edge-group reduction, neither of
  which has a notion of system boundary. One place *does* need an adapter: `EnergyHead` means node
  features first, then runs the MLP (unlike UMA's `Backbone.energy_batch`, a per-node-scalar
  segment-*sum*) — added `EnergyHead.batch` (row-normalized segment-mean matmul), bit-exact vs the
  single-system path (`tests/test_orb_disjoint_batch.py`). Wall-clock multi-card fan-out itself
  (spawning workers across physical cards) was not separately re-benchmarked here — it reuses the
  same card-count-agnostic scheduler already proven for UMA/BoltzGen once per-system independence
  holds (see `predict-multicard-already-exists`/`gen-multicard-already-exists`), and that
  independence is exactly what this test establishes.

## Still open

- A real, verified Orb `TracedEngine`-equivalent (see the trace-capture proof-of-concept below) —
  the one lever this pass's profiling shows would actually help.

## Profiling re-measurement at production scale

Re-measured warm eager forward (`benchmarks/bench_orb_profile.py`) at the toy 4-atom golden vs
the new 24-atom/1064-edge periodic supercell golden (real weights, `conservative-inf-omat`):

| system | N | E | warm forward |
|---|---|---|---|
| toy (bulk Si) | 4 | 172 | 4.167 ms |
| production (supercell) | 24 | 1064 | 4.275 ms |

**Edge count scaled 6.2x, latency scaled 1.03x** — confirms the "dispatch-bound, not compute-
bound" conclusion holds (and strengthens) at production scale: the op count per forward is fixed
(~9 ops/layer x 5 layers + encoder, independent of graph size), so latency barely moves while
compute work grows 6x. Trace capture (eliminating that fixed per-op dispatch overhead) remains the
applicable lever, not a custom kernel.

A quick exploratory attempt to wire up ttnn trace capture for the Orb forward (raw
`begin_trace_capture`/`execute_trace` around `Encoder`+backbone, no refresh logic yet) measured a
1.28x replay speedup (4.29ms eager -> 3.35ms replay) but the replayed output did **not** match the
eager output (max abs diff ~692, far outside bf16 noise) — almost certainly an output-buffer-
identity issue in the naive wiring (UMA's `tt_atom/trace.py` `TracedEngine` handles this carefully
via explicit captured-tensor handles + in-place `copy_host_to_device_tensor` refreshes; that
care was not replicated here). Per this project's correctness bar, an unverified number doesn't
ship: **a real, verified Orb `TracedEngine`-equivalent is not done this pass** — the speedup
direction is directionally promising (and UMA's own trace path measured ~2.6x forward-only), but
someone should port `tt_atom/trace.py`'s pattern properly (a `refresh()` that overwrites
`edge_feat`/`cutoff` in place per MD step, mirroring `orb_forces.energy_and_forces`'s inputs)
rather than trust this quick, broken proof of concept.

## `--fast` (bf8) mode: accuracy-safe, but a measured dead end (branch `wk/tt-atom-orb-bf8-mode`)

Applied UMA's existing weight-dtype policy as-is (`fast=True` -> `ttnn.bfloat8_b` for the
persistent Linear/attention weights in `tt_atom/orb_model.py`'s `MLPNorm`/`AttentionInteractionLayer`/
`EnergyHead`/`ForceHead`/`StressHead`; `compute_kernel_config`'s HiFi4 + fp32 dest-accumulate is
unchanged either way — same split as UMA's `grid.py`/`so2.py`). No new policy invented.

**Accuracy** (`tests/test_orb_bf8_fast.py`, real weights, both checkpoints) holds comfortably
inside UMA's own `--fast` bar (commit `836af75`: "PCC 0.99997, no accuracy loss") and this port's
existing bf16 real-weight thresholds:

| checkpoint | quantity | bf16 (existing) | bf8 weights (`fast=True`) |
|---|---|---|---|
| conservative | energy rel err | 1.19e-4 | 1.03e-3 |
| conservative | forces PCC / MAE | 0.999975 / 0.0089 eV/Å | 0.999963 / 0.0095 eV/Å |
| conservative | stress PCC / max err | (untested at bf16 in isolation) | 0.999920 / 9.25e-4 |
| direct-20 | energy rel err | 5.79e-4 | 5.79e-4 |
| direct-20 | forces PCC / MAE | 0.999966 / 0.0079 eV/Å | 0.999974 / 0.0093 eV/Å |

**Perf** (`benchmarks/bench_orb_profile.py`, warm eager forward, median of 30, real weights):

| case | bf16 | bf8 weights | ratio |
|---|---|---|---|
| conservative toy (N=4, E=172) | 4.231 ms | 4.214 ms | 1.00x |
| conservative production (N=24, E=1064) | 4.434 ms | 4.489 ms | 0.99x |
| direct-20 toy (N=4, E=80) | 4.236 ms | 4.199 ms | 1.01x |

**No measurable win — a dead end, not adopted.** This is exactly what the profiling section above
predicts: the forward is dispatch-bound (fixed ~9 ops/layer regardless of graph size), not
DRAM-bandwidth-bound, so halving the weight tensors' bytes does nothing (matches
`tt_atom/device.py`'s own `bf8_edge()` docstring: "bf8 weights alone = 1.00x" is the exact same
null result UMA itself measured for weight-only bf8). UMA's *real* bf8 win comes from a different,
non-transferable axis — `TT_ATOM_BF8_EDGE`'s bandwidth-bound edge-activation dataflow through the
custom `fused_rotate`/`fused_gate` kernels — and this port's own "Architecture verdict" section
already established that none of those four custom kernels apply to Orb (no equivariant hidden
representation to rotate or gate). So there is no analogous lever here at all, transferable or
otherwise: **no `--fast` CLI flag added for Orb.** The `fast=` kwarg is left threaded through
`tt_atom/orb_model.py` (default `False`, zero behavior change) purely so this null result stays
cheaply reproducible; nothing calls it with `fast=True` outside `tests/test_orb_bf8_fast.py` and
the benchmark.

## Reproducing

```bash
# 1. goldens (real weights, real Si structure) -- refenv (numpy>=2, has orb-models + fairchem)
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt conservative-inf-omat \
    --out ~/.ttatom_run/goldens_real/si_omat_orb.npz
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat \
    --out ~/.ttatom_run/goldens_real/si_omat_orb_direct20.npz

# 1b. goldens for the completion pass: a short-contact system (ZBL forces) and a bigger
# periodic supercell (periodic-image reconstruction)
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat --system short_contact \
    --out ~/.ttatom_run/goldens_real/si_short_contact_orb_direct20.npz
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt conservative-inf-omat --system supercell \
    --out ~/.ttatom_run/goldens_real/si_supercell_orb.npz

# 2. on-device PCC verification -- ttnn env (numpy<2)
TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
    tests/test_orb_realweight.py tests/test_orb_direct_realweight.py \
    tests/test_orb_forces_realweight.py tests/test_orb_zbl_forces.py \
    tests/test_orb_periodic_realweight.py tests/test_orb_disjoint_batch.py -q -s
```
