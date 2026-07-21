# Changelog

All notable changes to TT-Atom are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut only from a commit that has passed the on-hardware release gate — accuracy
parity, no OOM across the supported size range, no perf or UX regression, and a clean install
smoke (see `RELEASING.md`).

## Unreleased

### Fixed
- Built wheels now include both weight exporters, so automatic UMA and Orb cache misses work
  outside a source checkout.
- Fresh UMA and Orb cache misses can download their checkpoints again; explicit
  `HF_HUB_OFFLINE=1` still enforces offline use. Concurrent exports now use separate sidecars.
- Release mode now blocks every missing fixture, baseline, required op, and model-family OOM row.
  `--allow-gaps` remains available for development diagnostics.
- Release and UX subprocesses always open logical device 0 after `TT_VISIBLE_DEVICES` selects the
  physical card.
- Custom-op validation now rejects invalid gate modes and shapes, and program-cache keys include
  every operand layout that affects compiled accessors.
- The silicon-melt example now checks the exact-cutoff neighbour graph every step and recaptures
  only when it changes. The earlier trajectory is withdrawn because its skin policy could reuse
  stale edges.

### Changed
- Accuracy coverage now includes all periodic UMA tasks and Orb's bf8 fast mode.
- Performance baselines now cover UMA and record the `ttnn` version that produced them.
- The clean-install gate builds the candidate wheel, installs it and its runtime dependencies in
  isolation, and verifies the exact pushed commit and packaged exporters.

## [0.2.1] - 2026-07-20

No new model family this release. This is a consolidation release on top of v0.2.0's Orb-v3/
OrbMol port: one public API instead of three, several measured Orb-v3 perf wins, the on-hardware
release-gate framework that now runs all four checks with one command, and the MultiCard Orb
dispatch bug fix. UMA code paths are untouched throughout.

### Added
- **Single public entry point**: `Calculator(atoms, model=...)` picks UMA or Orb by checkpoint
  name (like fairchem's `FAIRChemCalculator` or HF's `AutoModel.from_pretrained`) instead of
  separate `UMA()`/`Orb()` factory functions, which are removed (a clean break, not an additive
  wrapper — no external contract yet to preserve). The typed classes (`TTAtomCalculator`,
  `OrbCalculator`) and their `.from_uma`/`.from_checkpoint` classmethods stay as the advanced
  surface; `tt_atom/auto.py` dispatches to them by model name. A family-inapplicable argument
  (e.g. `task=` on Orb, `trace=` on Orb) raises with an actionable message rather than being
  silently dropped. Pure call-surface change — `calculate`/`from_uma`/`from_checkpoint` and all
  numerics are byte-for-byte untouched; verified on device (30 Orb parity tests green through the
  unified entry point, bit-exact vs golden).
- **`OrbCalculator.evaluate_batch`**: disjoint-union batched inference for Orb-v3/OrbMol — K
  systems in one device forward, per-system energies via `EnergyHead.batch`, forces from the
  batched conservative VJP or the direct `ForceHead.batch`. Measured on one card, 9-atom ethanol
  (`benchmarks/bench_orb_evaluate_batch.py`, K=128): `conservative-omol` ~18.7–19x (1338 vs
  71 sys/s), `direct-omol` ~12.4x (2056 vs 166 sys/s) — exceeds UMA's own ~13x batched speedup.
  Parity gated vs looping `calculate` on all three checkpoint variants (energy rel err < 1e-2,
  force PCC > 0.999).
- **Capability-gated perf defaults**: `fused_lnbw` (UMA's fused radial-LayerNorm backward) is now
  default-ON, gated behind a build/capability probe (mirrors tt-bio's `fuse_swiglu` pattern) that
  detects whether the installed `ttnn` carries the custom fused kernels, falling back to the stock
  path on wheels that lack them (e.g. 0.68). `device_ede`/`bf8_edge` (UMA on-device edge-degree
  embedding / bf8 edge-activation dataflow) stay opt-in (env var): re-measurement showed they win
  ~2x at 512+ atoms but *regress* to ~0.85x on 9-atom molecules, so there's no size-independent
  default that's safe.
- **`scripts/release_gate.py`**: one-command on-hardware release gate, the runnable version of
  `RELEASING.md`'s manual checklist. Four legs, each machine-readable `PASS`/`FAIL`/`GAP`:
  accuracy (reuses `tests/test_*realweight*.py` verbatim via pytest + JUnit XML, so the gate never
  re-derives a parity bar or oracle), no-OOM (Orb disjoint-union batch sweep to the OOM ceiling —
  128 systems / 1152 atoms on this release's run; UMA's sweep is a documented env `GAP`), perf
  (warm throughput vs `docs/perf_baselines.json`, card-type aware, one entry per shipped family:
  OrbMol `conservative-omol`, Orb-v3 bulk `conservative-inf-omat`, UMA `uma-s-1`), and UX
  (`scripts/ux_regression.py`: CLI `--help` surfaces the core flags, a real single-point/relax/
  MD roundtrip writes and re-parses, and per-step progress output actually advances). A missing
  golden/baseline reports `GAP`, never a silent `PASS`.
- **`docs/materials-benchmark.md`**: device-vs-reference implementation-parity doc (the R/D/X
  framework, mirrors tt-bio's `pharma-benchmark.md`), covering every shipped family — UMA, Orb-v3,
  OrbMol — in one table, plus a new **MgO rock-salt row**, the suite's first multi-element bulk
  system (every prior bulk row was pure Si): energy rel err 1.6e-3, force PCC 0.99998, exercising
  the per-element reference-energy denormalize, mixed-Z ZBL pair repulsion, and the encoder's
  per-element embedding table at two atomic numbers simultaneously.
- `examples/orb_md.py`: periodic-crystal MD driver with on-device Orb-v3 forces.

### Fixed
- `tt_atom.batch.MultiCard` now builds the Orb-v3/OrbMol backbone when given Orb weights. The
  worker previously hardcoded the UMA path (`WeightBundle` + eSCN-MD `Backbone`), so pointing it
  at an Orb weights file built the wrong model silently. It now dispatches on the loaded bundle's
  `config` (the same UMA/Orb family split `tt_atom.auto` exposes by name) and runs the
  `Encoder`/`AttentionInteractionLayer`/`EnergyHead` forward for Orb. Verified bit-exact vs the
  single-card `OrbCalculator` on `orb-v3-conservative-inf-omat` (energy diff 0 eV on H2O /
  ethanol / benzene). `tests/test_multicard_orb.py` mirrors the UMA `test_multicard.py` sharded-vs-
  sequential parity shape (auto-skips below 2 cards).
- OrbMol's open-shell direct-forces PCC bar (`orb-v3-direct-omol`, CH3• radical) re-baselined
  0.9 → 0.85. Root-caused as a bf16 noise floor, not a port bug: the reference is bit-identical
  across reruns (self-consistency PCC 1.0), and treating the device's measured RMSE as additive
  noise on the reference forces predicts a best-case PCC of 0.908 — the device measures 0.8926,
  within 1.6% of that floor, while its energy error (1.25e-5) is the tightest of all six OrbMol
  rows. The conservative checkpoint of the same system clears 0.9785, so the charge/spin
  conditioning path itself is correct; only the direct `ForceHead`'s extra rounding on this
  tiny-force system (oracle `|F|max` 0.05 eV/Å, an order of magnitude below the other rows) was
  ever going to depress the correlation. See `docs/orb-port.md` for the full X-vs-R/D analysis.
- Release-gate reliability, found running the gate for real on the release host: the perf-leg
  measurement subprocess had no timeout, so a device-side hang blocked the whole gate forever
  (recurred 3x on the same model) — added a 240s timeout, child process-group kill, and a clean
  card reset before the next model. The OOM leg's process-global `MetalContext` was leaking into
  the perf leg's device open and causing cross-leg hangs; the OOM sweep now runs in its own child
  process so its context is torn down before perf opens the card. The accuracy leg's fused_rotate
  crash (as opposed to an auto-skipped missing golden) was reading as a blocking `FAIL` instead of
  the same documented env `GAP` the OOM/perf legs already use for that host limitation; classified
  consistently.
- `host_zbl_forces`: documented (docstring) that a periodic system's ZBL term can silently `NaN`
  if called without `cell_shift` — found while adding the MgO golden.

### Performance
Orb-v3 continued past v0.2.0's "not near the hardware limit" open question. Roofline + stage-
level profiling showed the p150 port running at 7.5% of bf16 compute peak / ~22% of DRAM
bandwidth floor at 2016 atoms — materialized edge-MLP activation traffic, not host dispatch or a
saturated matmul engine, so v0.2.0's "likely near the limit" framing did not hold. Several real,
parity-safe, bit-exact-or-PCC-gated wins landed this release, each measured independently and
compounding on the traced MD step (`benchmarks/bench_orb_perf_dollar_tt.py` and siblings, one
p150, bf16, real weights, median of repeated jittered-`pos` steps):

| fusion | measured speedup | scope |
|---|---|---|
| closed-form host geometry VJP (replaces autograd-graph rebuild) + hardware-limit accel | up to 1.12x (default), up to 1.23x (`--fast`) | full MD step |
| fused edge-MLP SiLU backward (source-ttnn `ttnn.experimental` fused derivative) | curve to 42.68 / 91.80 / 188.27 / 372.13 ms at 216 / 512 / 1000 / 2016 atoms | edge-MLP backward |
| row-major edge-aggregation scatter (fixes a TILE-concat layout inefficiency in `scatter.segment_sum`) | 1.04–1.08x | full MD step |
| edge-MLP matmul factory (`ttnn.experimental.minimal_matmul`, drop-in, preserves force-path pre-activations) | 1.08x–1.13x, curve to 39.70 / 84.61 / 164.36 / 341.03 ms at 216 / 512 / 1000 / 2016 atoms | edge-MLP matmuls |
| `--fast` (bf8 hidden edge activations, bf16 residual stream, fp32 accumulate) | 1.03x / 1.21x / 1.23x / 1.23x vs the bf16 curve at the same sizes | full MD step, opt-in |

No single before/after number spans the whole sequence (each fusion's baseline is the state left
by the previous one, measured in its own PR); the table above is the ordered, individually-
verified chain. An edge-chunked L1-resident streaming kernel and a forward Linear+SiLU epilogue
were both measured and rejected (0.42–0.98x and 0.988–0.996x respectively) — see
`docs/orb-port.md` for the full profiling writeup. `--fast` remains release-gated: force MAE is
0.049 eV/Å vs 0.0089 eV/Å for bf16, so it trades accuracy for the extra ~20% at 512+ atoms.

**Fair TT-vs-GPU perf-per-dollar, redone honestly.** An earlier claim (p150 "1.74x faster than an
H200", "~40x perf-per-dollar") compared TT's optimized trace/replay path against the GPU's stock
path with the neighbour list rebuilt every step, and its H200 timings had no committed raw
evidence — withdrawn. The redo compares the out-of-box path each side's user actually runs (TT:
`OrbTracedEngine`, the path `examples/orb_md.py` uses; GPU: stock `pip install orb-models==0.7.0`
`ORBCalculator`) on the same periodic Si diamond supercells, with raw per-step timings committed
(`benchmarks/orb_perf_dollar_gpu_v0.7.0.json`, `benchmarks/orb_perf_edge_mlp_fused.json`):

| system | p150 (bf16, traced) | H200 (fp32, stock) | H200 raw speedup | p150 perf-per-dollar edge |
|---|---|---|---|---|
| 216 atoms | 42.68 ms/step | 16.85 ms/step | 2.5x faster | ~9.1x |
| 512 atoms | 91.80 ms/step | 19.43 ms/step | 4.7x faster | ~4.9x |
| 1000 atoms | 188.27 ms/step | 44.47 ms/step | 4.2x faster | ~5.4x |
| 2016 atoms | 372.13 ms/step | 70.51 ms/step | 5.3x faster | ~4.4x |

The honest result: the H200 is faster on raw throughput at every size tested. The p150 (~$1,399
vs an H200 at ~$30–40k, roughly 23x cheaper) still wins on throughput-per-dollar, but by ~4.4–
9.1x, not ~40x, and the edge shrinks as systems grow — price/performance, not raw speed. Full
methodology, the matched-policy transparency view, and the hardware cost basis are in
`docs/orb-port.md`.

### Notes
- README rewritten for the user audience: kernel/dispatch/hardware-internals content and
  investigation-specific numbers moved out of the top-level README into `custom_kernels/README.md`
  and the relevant `docs/*.md`, with one practical fact plus a link left at the top level. Added a
  Troubleshooting section.
- The v0.2.0 scope note previously flagged Orb multi-card as "not independently re-run — same
  scheduler as UMA". That understated the gap: at v0.2.0 the worker was UMA-only, so Orb
  multi-card did not work at all (not merely unmeasured); see Fixed above.
- No real-weights multi-card *scaling* number is reported this release either: the release host
  has a single Tenstorrent card, so N>1 scaling cannot be measured on it. The one honest datapoint
  is the per-card baseline: Orb (`conservative-inf-omat`, real weights) on one card at ~128-atom
  Si supercells — 0.37 Medges/s. The earlier 2.95x@4cards figure (v0.1.0-era) used the synthetic
  `examples/model_tiny_demo.npz` UMA bundle, not real weights and not Orb, so it is still not a
  real-weights scaling number for either family.
- 9 closed-investigation Orb-v3 perf scripts (scatter dead-ends, forward-SiLU-fusion dead-end,
  rejected megakernel/chunking attempts) archived to `benchmarks/archive/`, each with a README
  pointer to the memory lesson that closed it. Nothing deleted, live scripts unchanged — internal
  housekeeping, no user-facing effect.
- A handful of commits in this range were social-media drafts and Si-melt demo-video polish
  (LinkedIn/X post drafts, render-window/flicker fixes) — not user-facing library changes, omitted
  above.

## [0.2.0] - 2026-07-11

A second model family, additive to v0.1.0: **Orb-v3** (Orbital Materials) and **OrbMol**, its
charge/spin-conditioned molecular variant. UMA/eSEN code paths are untouched (byte-identical to
v0.1.0) — see full history and numbers in `docs/orb-port.md`.

### Added
- **Orb-v3** (`orb-v3-conservative-inf-omat`, `orb-v3-direct-20-omat`): a non-equivariant,
  attention-MPNN backbone, ported bottom-up (encoder, 5-layer backbone, energy/force/stress
  heads, ZBL pair repulsion, periodic images, disjoint-union batching). None of UMA's four custom
  kernels transfer (Orb has no equivariant hidden representation) — this path runs on stock `ttnn`
  ops only, no source tt-metal build required for Orb-only use.
- **OrbMol** (`orb-v3-conservative-omol`, `orb-v3-direct-omol`): the OMol25-trained, charge/spin-
  conditioned checkpoints. Reuses the Orb-v3 backbone unmodified plus a closed-form, node-only
  charge/spin embedding (zero learned matmuls) — no new forward/backward machinery.
- `OrbTracedEngine` (`tt_atom/orb_trace.py`): trace-capture for the Orb-v3 forward(+analytic-VJP
  backward), refreshing only the two pos-dependent device inputs per MD/relaxation step. Bit-exact
  vs eager at every tested scale.

### Investigated, not shipped
- **`--fast` (bf8 weights) for Orb-v3**: accuracy-safe (energy rel err/force PCC inside the
  existing bf16 bar) but measured **no perf win** (0.99–1.01x) — Orb's forward is dispatch-bound
  (~9 fixed ops/layer), not weight-bandwidth-bound, so halving weight bytes does nothing. No CLI
  flag added; the `fast=` kwarg stays threaded through for reproducibility only.

### Performance
UMA (`uma-s-1`, Blackhole p150a) — code path unchanged from v0.1.0, re-confirmed on this release's
combined test/build environment: trace-captured E+F is bit-exact vs eager (energy diff 0, force
PCC 1.0, maxdiff 0), no regression at any tested size (existing periodic/molecular suite plus an
ad hoc 128-atom Si supercell, single card, no OOM).

Orb-v3 (`orb-v3-conservative-inf-omat`) trace capture, this release's on-hardware gate run (median
of 20, jittered `pos` per step, real weights):

| scale | slice | eager | traced/replay | speedup |
|---|---|---|---|---|
| toy (N=4, E=172) | full step | 38.6–39.0 ms | 8.1–8.3 ms | 4.7–4.8x |
| toy (N=4, E=172) | device-only fwd+bw | 37.9–38.5 ms | 5.8 ms | 6.6x |
| production (N=24, E=1064) | full step | 42.5–43.2 ms | 10.8 ms | 3.9–4.0x |
| production (N=24, E=1064) | device-only fwd+bw | 39.4–40.0 ms | 8.5–8.5 ms | 4.6–4.7x |

Bit-exact vs eager at every scale (energy diff 0, force max abs diff 0). This is a **new baseline**
(no prior Orb release to regress against). It is also higher than the 1.3–1.5x originally measured
during the Orb-v3 trace-capture branch's own verification (`docs/orb-port.md`) — the *relative*
conclusion (trace capture is a strict, bit-exact win; bf8 is not) is identical, but this release
gate's build shows noticeably higher absolute eager latency (per-op dispatch through a generalized
mesh-device path, present equally in UMA's own trace benchmark on this same build) than the
original branch measurement's environment. Both numbers are real, reproducible measurements, not
fabricated; the discrepancy is environmental (build/dispatch-path drift), not a code regression —
tracing removes proportionally *more* fixed dispatch overhead here, not less.

### Scope
- Orb-v3/OrbMol accuracy gate: real weights, real `orb-models` CPU oracle, both checkpoint
  families — see `docs/orb-port.md` for the full PCC/rel-err table (energy rel err 1.2e-6–5.8e-4,
  force PCC 0.93–0.9999 depending on system/checkpoint, worst case an open-shell radical with a
  tiny force-magnitude noise floor, not a correctness issue).
- No accuracy regression vs v0.1.0's UMA numbers (code untouched; full suite re-run green on this
  commit).
- Multi-card fan-out for Orb-v3/OrbMol did **not** work at v0.2.0: `tt_atom.batch`'s worker was
  hardcoded to the UMA backbone (see the v0.2.1 fix above). The v0.2.0 gate ran on a single
  card, so the gap was not caught by `test_multicard.py`'s 2+ card requirement.

## [0.1.0] - 2026-07-08

Initial release. The **custom-kernel-only, highest-performance build for `uma-s`** — the per-edge
Wigner rotation runs as a custom tt-metal kernel, so `ttnn` comes from a source tt-metal build
that includes the op (see README "Install"); there is no slow fallback path.

### Added
- Tenstorrent inference for Meta **UMA** (eSEN / eSCN-MD) equivariant ML interatomic potentials:
  energy, conservative analytic forces, and stress for molecules and periodic materials, behind an
  **ASE** calculator that mirrors fairchem's (moving off fairchem is a one-line change). Validated
  against the released `uma-s-1`.
- Device-resident trace loop for MD / relaxation; multi-card data-parallel throughput path.
- `tt-atom verify` device round-trip check and a one-command checkpoint converter.

### Performance (uma-s-1, Blackhole p150a)
- Fused-rotation kernel: **4.3×** vs the addcmul MAC in isolation (7.01 → 1.62 ms, PCC 0.999995);
  **1.4–1.68× faster end-to-end** traced MD/relax across N=54–2662, no regression at any size.
- Accuracy (vs fairchem reference): energy rel-error ≤ 5.4e-4, force PCC ≥ 0.9996 across
  molecular / periodic / slab; traced == eager (PCC 1.0). pytest 51 passed / 1 skipped.

### Scope
- `uma-s` (lmax=mmax=2) is the supported target. Other checkpoints (e.g. `uma-m`) raise a clear
  error rather than silently falling back. `ttnn` is not a pip dependency (source build required).
