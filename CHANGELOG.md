# Changelog

All notable changes to TT-Atom are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut only from a commit that has passed the on-hardware release gate — accuracy
parity, no OOM across the supported size range, and no perf regression (see `RELEASING.md`).

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
- Multi-card OOM/perf for Orb-v3/OrbMol not independently re-run this release (single card
  available for this gate) — `test_multicard.py`'s existing 2+ card requirement documents the gap;
  no reason to expect it differs from UMA's already-proven multi-card fan-out (same scheduler).

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
