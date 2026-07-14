# Orb-v3 port

Porting [Orbital Materials' Orb-v3](https://github.com/orbital-materials/orb-models)
(`orb-v3-conservative-inf-omat`, `orb-v3-direct-20-omat`) onto Tenstorrent alongside the
existing UMA/eSEN support. The initial pass (encoder + full 5-layer backbone + energy for both
checkpoints, direct-20's forces) merged to `master`; this doc's "Completed since" / "Still open"
sections track a follow-up completion pass (branch `wk/tt-atom-orb-completion`). Purely additive:
UMA code paths (`tt_atom/{model,norm,edgewise,so2,rotation,grid,spectral}.py`, `custom_kernels/`)
are untouched throughout. A later pass ("OrbMol" below, branch `wk/tt-atom-orbmol-port`) extends
this same backbone to Orbital Materials' molecular/charge+spin-aware checkpoints.

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
  single-system path (`tests/test_orb_disjoint_batch.py`). The batched calculator method is now
  wired up (`OrbCalculator.evaluate_batch`): one device forward for K systems, per-system energies
  via `EnergyHead.batch`, forces either from the batched conservative VJP
  (`orb_forces.energy_and_forces_batch` — `energy_bw_batch` seeds the head adjoint per-system via
  the transposed segment matrix, then the unchanged block-diagonal layer VJP) or the direct
  `ForceHead.batch` (per-system net-force removal). Parity vs looping `calculate` is gated on all
  three checkpoint variants (`tests/test_orb_evaluate_batch.py`, E rel < 1e-2, force PCC > 0.999).
  Wall-clock: ~19x vs looping at K=128 9-atom molecules on `conservative-omol` (1338 vs 71 sys/s),
  ~12x on `direct-omol` (2056 vs 166 sys/s) — `benchmarks/bench_orb_evaluate_batch.py`. Multi-card
  fan-out itself (spawning workers across physical cards) was not separately re-benchmarked here —
  it reuses the same card-count-agnostic scheduler already proven for UMA/BoltzGen once per-system
  independence holds (see `predict-multicard-already-exists`/`gen-multicard-already-exists`), and
  that independence is exactly what this test establishes.

## Still open

Nothing — the trace-capture item below is done (branch `wk/tt-atom-orb-trace-capture`).

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

## Trace capture, done properly (branch `wk/tt-atom-orb-trace-capture`)

Ported `tt_atom/trace.py`'s `TracedEngine` pattern to Orb as `tt_atom/orb_trace.py`'s
`OrbTracedEngine` — the exact fix the section above called for. `TracedEngine` itself is not
reusable as-is (its refresh path is Wigner-rotation/`GraphContext`-specific, none of which exists
for Orb's plain non-equivariant backbone), so this is a from-scratch port of the same *idea*:
capture the forward(+backward) op stream once for a fixed topology, refresh only the two
pos-dependent device inputs in place each step (`edge_feat` and `cutoff`, both produced by
`orb_geometry.host_edge_features` — the same two adjoint targets `orb_forces.energy_and_forces`
differentiates through), then replay. `node_feat` (atomic-number embedding only) has no `pos`
dependence and is uploaded once, never refreshed. Two modes: `ehead` alone captures the
conservative checkpoint's analytic-VJP backward too; `ehead`+`fhead` is forward-only (direct-20
has no device backward at all).

**A real bug found along the way, fixed in `tt_atom/orb_forces.py`:** `energy_bw`/`backbone_bw`
allocated fresh `ttnn.ones`/`ttnn.zeros` constants on *every* call — harmless eager, but exactly
the "the ttnn trace machinery forbids allocations during capture (it hangs)" landmine
`tt_atom/forces.py`'s own `energy_bw` docstring already warns about for UMA. Fixed the same way
UMA does: cache the constants once (guarded by shape) on `ehead` instead of recreating them
per call. Purely a capture-compatibility fix — values are constant, so eager callers see zero
behavior change (all 21 existing Orb tests still pass unmodified).

**Correctness (`tests/test_orb_trace.py`), verified BEFORE any perf number:** replayed output is
BIT-EXACT vs eager (`max abs diff == 0`, not just within a PCC bar) for both checkpoints at both
goldens, including a genuinely different (jittered) `pos` per call, not just a degenerate
identical-input replay:

| checkpoint | system | energy diff | force max abs diff |
|---|---|---|---|
| conservative-inf-omat | toy (N=4, E=172) | 0 | 0 |
| conservative-inf-omat | production (N=24, E=1064) | 0 | 0 |
| direct-20-omat | toy (N=4, E=80) | 0 | 0 |
| direct-20-omat | production (N=24, E=480) | 0 | 0 |

**Perf (`benchmarks/bench_orb_trace.py`, conservative-inf-omat, median of 20, jittered pos each
step — a real MD-loop measurement, not the broken proof-of-concept's identical-input replay):**

| scale | slice | eager | traced/replay | speedup |
|---|---|---|---|---|
| toy (N=4, E=172) | full step (host geometry + device fwd+bw + host force finish) | 13.3 ms | 8.8 ms | 1.51x |
| toy (N=4, E=172) | device-only fwd+bw | 11.9 ms | 6.4 ms | 1.85x |
| production (N=24, E=1064) | full step | 14.8 ms | 11.0 ms | 1.35x |
| production (N=24, E=1064) | device-only fwd+bw | 12.3 ms | 9.4 ms | 1.30x |

**Real, verified, but well short of UMA's own ~2.6x forward-only trace win, and the win shrinks
(not grows) at production scale.** Why: tracing only removes the fixed per-op host *dispatch*
overhead — it does nothing for the host geometry recompute (`host_edge_features`) or the
`copy_host_to_device_tensor` refresh, both of which scale with edge count `E` and run every step
regardless of tracing. At E=1064 that non-traced host work is a much bigger slice of the eager
step than at E=172, so the traced fraction (and thus the speedup) shrinks as the graph grows —
the opposite of what "dispatch-bound" might suggest, because dispatch-bound means the *device*
time barely grows, not that the *host* time doesn't. UMA's own trace path enjoys a bigger win
because its refresh is cheaper per edge (row-major bf16 writes only, see `tt_atom/trace.py`'s
`_refresh` comments) and its equivariant geometry (Wigner coefficients) is a heavier fraction of
eager time to begin with, so removing dispatch buys proportionally more. **Ship it anyway**: even
the production-scale 1.30-1.35x is a real, bit-exact win for a fixed-topology MD/relaxation loop
with no accuracy cost — there's no calculator/CLI surface for Orb yet (unlike UMA's
`trace=`/`--trace`) to wire an opt-in flag into, so `OrbTracedEngine` is exposed the same way
`orb_forces.energy_and_forces` already is: a direct, documented API a caller's own MD/relaxation
loop constructs once per fixed topology and calls per step.

The production trace path now finishes the host geometry derivative with a closed-form VJP
instead of rebuilding a PyTorch autograd graph. It is mathematically equivalent (maximum
normalized-force difference below 3e-7 in the trace parity test) and reduces the current bf16
curve to 45.46 / 107.90 / 210.23 / 424.61 ms per step.

## `--fast` (bf8) mode

Weight-only bf8 remains a dead end. The useful mode also stores the two 1024-wide hidden MLP
activations in bf8, while keeping the 256-wide residual stream in bf16 and matmul accumulation
in fp32. This targets the actual large-graph cost: repeatedly moving hidden edge activations
through DRAM.

On the same traced conservative-force benchmark as the performance-per-dollar comparison,
`--fast` gives 49.29 / 89.39 / 190.79 / 383.06 ms per step at 216 / 512 / 1000 / 2016 atoms.
That is 1.03x / 1.21x / 1.23x / 1.23x faster than bf16. The full measurements are in
`benchmarks/orb_perf_dollar_tt_accel.json`.

Real-weight parity remains within the port's release bar: conservative force PCC 0.999906,
energy relative error 1.84e-3, and stress PCC 0.999590. A 2000-step, 1 ps solid-Si NVE run at
900 K measured -0.005 meV/atom/ps total-energy drift. The mode is still release-gated because
its force MAE is 0.0490 eV/Å versus 0.0089 eV/Å for bf16. Use `examples/orb_md.py --fast` only
when that accuracy/performance trade-off is acceptable.

## OrbMol: the OMol25-trained, charge/spin-conditioned checkpoints (branch `wk/tt-atom-orbmol-port`)

[OrbMol](https://huggingface.co/Orbital-Materials/OrbMol) is Orbital Materials' molecular/
bio-adjacent model -- `orb-v3-conservative-omol` / `orb-v3-direct-omol` in `orb-models==0.5.5`
(`pretrained.orb_v3_conservative_omol`/`orb_v3_direct_omol`; public S3, no gating). Confirmed by
reading `pretrained.py`: **same `MoleculeGNS` backbone** (`Encoder`, 5 `AttentionInteractionNetwork`
layers, `latent_dim=256`, `rms_norm`, `silu`) as the ported omat checkpoints, so `Encoder`/
`EnergyHead`/`ForceHead`/`host_zbl_{energy,forces}`/`orb_forces.energy_and_forces` all reuse
unmodified. Two real differences: `has_charge_spin_cond=True` (below) and `has_stress=False` (no
`StressHead` weights in the checkpoint -- consistent with "stress isn't meaningful for isolated
molecules", nothing to port). `system_config` (`radius=6.0, max_num_neighbors=120`) and molecules
being aperiodic (`pbc=False`) are just config/data, not new code -- `tt_atom/geometry.py`'s
`radius_graph` already takes the aperiodic branch whenever `pbc` is all-`False`, and UMA's own
`bundle_cache`/`disjoint`/`calculator` infra already default to `task="omol"` (aperiodic molecules
with charge/spin) -- this port needed no changes there.

**Charge/spin conditioning** (`nn_util.ChargeSpinConditioner`, read from `gns.py`/`nn_util.py`
byte-for-byte): a **node-only, additive** feature, unrelated to UMA's MoLE-baked-at-merge-time
mechanism (Orb has no MoE at all) -- each of the 5 interaction layers owns its own
`_cond_node_proj` `Linear(256,256)` and computes `nodes = nodes + _cond_node_proj(cond_nodes)` as
the *very first* thing in its forward (before the sender/receiver gather *and* before the
residual add at the end -- both then use the conditioned `nodes`), where `cond_nodes` is one
`sin_emb`-type embedding (`ChargeSpinEmbedding`, closed-form sin/cos of a learned frequency `W`,
zero matmuls, verbatim in `nn_util.py`) of the system's total charge + spin, broadcast to every
node. Ported as `host_charge_spin_embedding` (`tt_atom/orb_model.py`) -- computed on host exactly
like this port's other fixed per-system/per-edge terms (`host_cutoff`, the RBF/spherical-harmonic
edge features) -- and `graph.cond_nodes` (`OrbGraphContext`, optional, `None` for the omat
checkpoints with zero behavior change), consumed by `AttentionInteractionLayer` which
auto-detects conditioning from the weight bundle (`"{prefix}._cond_node_proj.weight" in weights`).
Edge conditioning is unused (`ChargeSpinConditioner(latent_dim)`'s default `emits_edge_embs=False`
in every public checkpoint) so `AttentionInteractionLayer` only implements the node path.

**Forces need no backward changes.** `cond_nodes` is a fixed function of (charge, spin), not of
`pos` -- adding it to `nodes` is an identity-Jacobian shift, so `orb_forces.py`'s existing
hand-written VJPs (`attn_layer_bw`/`backbone_bw`) are correct unmodified; `energy_and_forces`
only gained a passthrough `cond_nodes=` kwarg to reach `OrbGraphContext`. Verified, not just
argued: the conservative checkpoint's on-device analytic forces (below) match the real
`torch.autograd` oracle to the same bar as the omat port's own force test.

**Real on-device parity** (`tests/test_orb_omol_realweight.py`, `TT_VISIBLE_DEVICES=0`, bf16,
real weights, real `orb-models` CPU oracle), three small aperiodic molecules exercising a
closed-shell baseline, a nonzero total charge, and a nonzero spin multiplicity (open-shell):

| system | charge | spin | checkpoint | energy rel err | forces PCC | forces MAE (eV/Å) |
|---|---|---|---|---|---|---|
| H2O | 0 | 1 | conservative | 1.59e-06 | 0.999741 | 0.0062 |
| H2O | 0 | 1 | direct | 1.66e-06 | 0.997977 | 0.0103 |
| NH4+ | +1 | 1 | conservative | 4.55e-06 | 0.994645 | 0.0074 |
| NH4+ | +1 | 1 | direct | 3.86e-05 | 0.994425 | 0.0073 |
| CH3• (radical) | 0 | 2 | conservative | 9.23e-06 | 0.968975 | 0.0041 |
| CH3• (radical) | 0 | 2 | direct | 1.25e-05 | 0.933058 | 0.0057 |

Backbone node-embedding PCC is >0.9998 through all 5 conditioned layers for every system/
checkpoint (the conditioning wiring itself is not the source of any error above). The
open-shell radical's forces PCC is visibly lower (0.93-0.97) despite its MAE being the *smallest*
of the three systems (0.004-0.006 eV/Å, vs 0.006-0.01 for the other two) -- its oracle `|F|max`
is ~0.03-0.05 eV/Å, an order of magnitude smaller than the other systems (~0.09-0.48), so the
same absolute bf16 noise floor produces a much lower correlation coefficient. Not a correctness
issue with the conditioning path (energies, which have no such magnitude sensitivity, are the
tightest of all six rows here); a PCC bar tuned to that system's own signal scale, same reasoning
as this doc's existing edge-stream/ZBL PCC bars. `host_charge_spin_embedding` itself matches the
real `ChargeSpinConditioner`'s captured activation to 5.96e-08 max abs error for all three systems
(bit-level, not statistical, agreement -- it's a closed-form host computation, no device
rounding involved).

**Not ported (genuinely out of scope, not deferred):** `StressHead` (checkpoint has none, nothing
to port). Disjoint-union batching for `cond_nodes` is now wired up — `OrbCalculator.evaluate_batch`
concatenates one `host_charge_spin_embedding` per system and uploads it once, parity-gated on the
OrbMol checkpoint (`tests/test_orb_evaluate_batch.py::test_evaluate_batch_conservative_omol`).

## Performance per dollar: one Blackhole p150 vs an NVIDIA H100-class GPU

> This section was redone fairly on 2026-07-14 (branch `wk/tt-atom-orb-gpu-fair-
> comparison`). An earlier version claimed the p150 was "1.74x faster than an H200" and
> "~40x perf-per-dollar". That compared Tenstorrent's optimized trace/replay path against
> the GPU's stock `orb_models` path with the neighbour list rebuilt every step, and its
> H200 timings (88.4 / 93.2 ms/step) had no committed raw evidence (the worker's teardown
> narration was later proven false). The fair, evidenced redo below compares the
> out-of-box path on each side (TT traced; stock `pip install orb-models` v0.7.0
> `ORBCalculator`) and refutes the old claim: the H200 is faster than the p150 on raw
> throughput at *every* size tested; the p150 wins only on throughput-per-dollar (~3.8-8.5x,
> not ~40x). Raw per-step timings for both legs are committed in
> `benchmarks/orb_perf_dollar_tt_analytic_vjp.json`, `benchmarks/orb_perf_dollar_gpu_v0.7.0.json`
> (headline) and `benchmarks/orb_perf_dollar_gpu.json` (v0.5.5 matched-policy
> transparency).

The question for a buyer: for an Orb-v3 materials-MD workload, how much throughput does
a single Blackhole p150 deliver relative to a single NVIDIA data-centre GPU, and what
does that look like once you divide by what the card costs? The honest answer, comparing
the software a user actually runs out of the box on each side: the NVIDIA H200 is faster
than the p150 on raw throughput at every system size tested. The p150 still wins on
throughput-per-dollar because it costs roughly twenty-three times less, but by ~3.8-8.5x, not
~40x, and that edge shrinks as systems grow. The p150's value proposition here is
price/performance, not raw speed.

### What was measured (out-of-box on each side)

The comparison is the near-out-of-box path a normal user runs on each side -- no
hand-tuning, no custom kernels, no re-architected inference loop. Same model
(`orb-v3-conservative-inf-omat`), same periodic Si diamond supercell, same quantity (one
energy + conservative-force eval per step, `F = -dE/dpos`), warm steady-state, load and
first-call compilation excluded, positions jittered each step so the path is exercised
like a real MD loop. A size sweep (216 / 512 / 1000 / 2016 atoms) so the throughput
trend is visible, not a single point.

| side | card | precision | path (what a user runs) | neighbour list |
|---|---|---|---|---|
| Tenstorrent | Blackhole p150 (one card, device 0) | bf16 weights/activations, fp32-accumulate matmul | `OrbTracedEngine` trace/replay -- the production path `examples/orb_md.py` runs | frozen at the first geometry (free built-in for a fixed-topology solid) |
| NVIDIA | H200 (one GPU, rented on vast.ai) | fp32 (`orb_models` default; bf16 is not a documented flag) | stock `orb_models` `ORBCalculator` from `pip install orb-models` v0.7.0 (compiled backbone + `knn_alchemi` GPU kNN are package defaults) | rebuilt every call (`orb_models` default; no easy user toggle to cache it) |

The two sides behave differently out of the box and that difference is disclosed, not
hidden. The p150's traced path freezes the neighbour list because a solid crystal's
atoms vibrate about their lattice sites and never cross the cutoff, so the topology is
constant and trace-capture replay is bit-exact -- a built-in optimisation a user gets for
free on `examples/orb_md.py`. Stock `orb_models` rebuilds the graph every call and exposes
no documented easy toggle to cache it; that is the GPU user's experience, reported as a
GPU-software limitation, not a TT trick. Forces come from the analytic/autograd backward
on both sides: hand-written device VJPs on Tenstorrent (`tt_atom/orb_forces.py`),
`torch.autograd` on the GPU. Precision is each side's default (TT bf16, GPU fp32); the
bf16-vs-fp32 accuracy axis is already validated elsewhere in this doc (force PCC 0.9999,
~1.2-1.4 meV/atom), so the asymmetry is stated, not relitigated.

A survey was done for a genuinely-easier, officially-supported *faster* GPU path before
settling on stock `orb_models`: there is no NVIDIA NIM container or official
fast-inference server for Orb; the package's own defaults are already the easy fast path
-- the GNS backbone is compiled by default since v0.5.0 (~1.7x at 10k atoms) and
`knn_alchemi` (NVIDIA ALCHEMI GPU kNN) is the default `edge_method` since v0.5.6; bf16 is
not a documented flag (orb-models issue #71: it needs manual Triton rewrites). Matching
the p150's trace/replay by removing the GPU's per-step host dispatch (CUDA graphs /
`torch.compile(reduce-overhead)`) could not be made to work on the stock conservative
regressor and is not faked: `torch.compile(reduce-overhead)` hits graph breaks from a
`float(p)` read in `pair_repulsion` and "outputs still require backward" blocks the
cudagraph fast path (a ~539 ms fallback, worse than eager); manual `torch.cuda.graph`
capture raises `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture` (an un-pinned transfer inside the regressor). So the GPU headline below still
pays per-step host dispatch that the TT traced path does not -- a residual asymmetry that
*disadvantages the GPU*, so the H200 leads despite it.

### Results (out-of-box, what users run)

| system (Si diamond) | N | edges | p150 (bf16, traced) | H200 (fp32, stock `ORBCalculator`) | H200 vs p150 |
|---|---|---|---|---|---|
| 3x3x3 cells | 216 | 9936 | 45.46 ms/step, 22.0 steps/s | 16.85 ms/step, 59.4 steps/s | H200 **2.7x faster** |
| 4x4x4 cells | 512 | 23552 | 107.90 ms/step, 9.3 steps/s | 19.43 ms/step, 51.5 steps/s | H200 **5.6x faster** |
| 5x5x5 cells | 1000 | 46000 | 210.23 ms/step, 4.8 steps/s | 44.47 ms/step, 22.5 steps/s | H200 **4.7x faster** |
| 6x6x7 cells | 2016 | 92736 | 424.61 ms/step, 2.4 steps/s | 70.51 ms/step, 14.2 steps/s | H200 **6.0x faster** |

(Median of 50-60 timed steps after warmup, jittered positions each step. The H200 was
rented on-demand on vast.ai; GPU spend across both legs of the redo was ~$2.05 of a
~$10.36 credit, instances destroyed and teardown verified -- `vastai show instances` ->
`[]`. No H100 on-demand inventory was available the day of the run, so the H200 stands in
for the H100 class, labelled exactly. torch 2.13.0+cu130, orb_models 0.7.0, edge_method
`knn_alchemi` (package default). Raw per-step timings in
`benchmarks/orb_perf_dollar_gpu_v0.7.0.json` and
`benchmarks/orb_perf_dollar_tt_analytic_vjp.json`.)

The H200 leads at every size. Why: the p150 traced step still recomputes the per-edge
geometry on host and uploads it every step (`host_edge_features` +
`copy_host_to_device_tensor`), which scales with edge count E; trace/replay only removes
the fixed per-op device dispatch, not that host work (already noted in the trace-capture
section above). The H200 does the neighbour search and edge featurization on-device, so
its step grows much more slowly with E.

### Matched-policy view (transparency, not the headline)

For transparency, freezing the neighbour list on the GPU too (calling
`regressor.predict` on a frozen batch directly, bypassing `ORBCalculator` -- hand-tuned,
not a user path) isolates the per-step rebuild cost and gives a hardware-vs-hardware
view. Measured on orb_models v0.5.5 (`benchmarks/orb_perf_dollar_gpu.json`,
`benchmarks/orb_perf_dollar_gpu_v0.5.5_crosscheck.json`): 20.0 / 23.2 / 32.5 / 53.8 ms,
i.e. the H200 is 2.3x / 4.7x / 6.5x / 7.9x faster than the p150 traced path. The
stock-out-of-box H200 numbers in the headline table above are the GPU's *slowest*
reasonable out-of-box case (they include the per-call rebuild and the un-removed host
dispatch); the hand-tuned frozen path is faster still, so the p150's out-of-box
perf-per-dollar edge below is, if anything, generous to the p150.

### Hardware cost basis

Prices are single-card street/list prices in USD, cited from public sources (no
fabricated numbers):

| card | price | source |
|---|---|---|
| Tenstorrent Blackhole p150 | $1,399 | tenstorrent.com product page (active list price) |
| NVIDIA H100 PCIe 80GB | ~$25,000-$30,000 | cloudzero.com / jarvislabs.ai 2026 price guides |
| NVIDIA H200 (141GB) | ~$30,000-$40,000 | thundercompute.com / jarvislabs.ai 2026 price guides |

The H100 is the card the UMA perf-per-dollar story is told against (cost ratio ~21x vs
the p150). The H200 measured here is the same price class (cost ratio ~23x) and stands in
for the H100 class; an H100 was not measurable this run because none was available
on-demand on vast.ai the day of the test.

### Perf-per-dollar

Taking the H200 as the measured H100-class stand-in at a representative $32,000 (cost
ratio vs the p150: ~23x), the p150's throughput-per-dollar advantage is cost_ratio /
H200_speedup, using the out-of-box numbers above:

| system | p150 throughput | H200 throughput | H200 raw speedup | cost ratio | p150 perf-per-dollar edge |
|---|---|---|---|---|---|
| 216 atoms | 22.0 steps/s | 59.4 steps/s | 2.7x | ~23x | **~8.5x** |
| 512 atoms | 9.3 steps/s | 51.5 steps/s | 5.6x | ~23x | **~4.1x** |
| 1000 atoms | 4.8 steps/s | 22.5 steps/s | 4.7x | ~23x | **~4.8x** |
| 2016 atoms | 2.4 steps/s | 14.2 steps/s | 6.0x | ~23x | **~3.8x** |

Read plainly: the H200 is the faster card outright at every size, by 2.7x to 6.0x on the
software a user actually runs; the p150 is ~23x cheaper, so it still delivers more
throughput per dollar -- ~8.5x at 216 atoms falling toward ~3.8x near 2000 atoms. The
earlier "~40x per dollar" was wrong by roughly an order of magnitude in the small-N
regime and falls further at larger N. This is one model
(`orb-v3-conservative-inf-omat`), one system family (periodic Si diamond), and the
production out-of-box MD path on each side; it is a perf-per-dollar positioning point,
not a benchmark report, and the p150's edge is price/perf, not raw throughput.

### Reproducing this comparison

```bash
# Tenstorrent side (one Blackhole p150, device 0) -- traced MD step sweep:
TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python \
    benchmarks/bench_orb_perf_dollar_tt.py \
    --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
    --warmup 12 --steps 60 --out benchmarks/orb_perf_dollar_tt_analytic_vjp.json
#   -> 216: 45.46 ms / 22.0 steps/s ; 512: 107.90 ms / 9.3 ; 1000: 210.23 ; 2016: 424.61

# NVIDIA side (one H200, rented on vast.ai) -- out-of-box stock ORBCalculator sweep:
#   on the GPU box: conda create -n orb python=3.12 && conda activate orb \
#       && pip install orb-models ase && apt-get install -y gcc g++   # g++ for triton
python benchmarks/orb_gpu_bench_fair.py --warmup 10 --steps 50 \
    --variants naive_rebuild --out orb_perf_dollar_gpu_v0.7.0.json
#   -> 216: 16.85 ms / 59.4 steps/s ; 512: 19.43 ; 1000: 44.47 ; 2016: 70.51
#   (orb_models 0.7.0, edge_method knn_alchemi default, compiled backbone -- the
#   `pip install orb-models` user path. Add --variants naive_rebuild,frozen_eager for the
#   matched-policy transparency view.)
```

Both harnesses compute the same quantity (one energy + conservative-force eval per step)
on the same Si diamond supercell and the same checkpoint, and both exclude load and
first-call warmup. Raw per-step timings, edge counts, parity, GPU SKU, torch/cuda
versions and git SHA are written to the JSON files -- a prose table alone is not
accepted as evidence this round.

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

# 1c. OrbMol goldens: three small aperiodic molecules (baseline/charged/open-shell), both
# checkpoints
for ck in conservative-omol direct-omol; do
  for sys in molecule molecule_charged molecule_openshell; do
    tag=$(echo $ck | cut -d- -f1)
    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt $ck --system $sys \
        --out ~/.ttatom_run/goldens_real/${sys}_omol_${tag}.npz
  done
done

# 2. on-device PCC verification -- ttnn env (numpy<2)
TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
    tests/test_orb_realweight.py tests/test_orb_direct_realweight.py \
    tests/test_orb_forces_realweight.py tests/test_orb_zbl_forces.py \
    tests/test_orb_periodic_realweight.py tests/test_orb_disjoint_batch.py \
    tests/test_orb_omol_realweight.py tests/test_orb_evaluate_batch.py -q -s
```
