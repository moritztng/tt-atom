# Changelog

All notable changes to TT-Atom are recorded here. Versioning is [SemVer](https://semver.org);
releases are cut only from a commit that has passed the on-hardware release gate — accuracy
parity, no OOM across the supported size range, and no perf regression (see `RELEASING.md`).

## [0.1.0] - 2026-07-08

### Added
- Initial release. Tenstorrent (`ttnn`) inference for Meta **UMA** (eSEN / eSCN-MD) equivariant
  ML interatomic potentials: energy, conservative analytic forces, and stress for molecules and
  periodic materials, behind an **ASE** calculator that mirrors fairchem's (moving off fairchem
  is a one-line change). Bring your own UMA checkpoint; validated against the released `uma-s-1`.
- Device-resident trace loop for MD / relaxation; multi-card data-parallel throughput path.
- `tt-atom verify` device round-trip check and a one-command checkpoint converter.
