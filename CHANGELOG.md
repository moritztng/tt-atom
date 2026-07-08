# Changelog

All notable changes to TT-Atom are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut only from a commit that has passed the on-hardware release gate — accuracy
parity, no OOM across the supported size range, and no perf regression (see `RELEASING.md`).

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
