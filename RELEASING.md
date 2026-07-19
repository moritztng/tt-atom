# Releasing TT-Atom

`master` is the development branch — it may contain experimental, not-yet-validated work.
**A tagged release is a promise to customers that it works.** So a release is cut only from a
commit that has passed the full on-hardware gate below.

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged, by **`scripts/release_gate.py`** — one command,
four legs, machine-readable `PASS` / `FAIL` / `GAP` per leg. Run it on card 0 of the release host:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py
```

The four legs are the four things a tagged release must clear:

1. **ACCURACY / correctness** — numerical parity vs each shipped model family's own reference
   within tolerance (energy rel-error and force/stress PCC), across every supported task and
   graph regime for which a real-weight golden fixture exists, **for every shipped model
   family** — UMA (vs fairchem `uma-s-1`) and Orb-v3 / OrbMol (vs the real `orb-models` CPU
   oracle). The gate reuses the existing `tests/test_*realweight*.py` parity modules verbatim
   (they already encode the bars and the real oracles) by invoking pytest with JUnit XML, so
   the gate never re-derives a parity bar or oracle. A module whose golden fixture is absent
   auto-skips and is reported as `GAP` (missing fixture) — never a silent `PASS`. Run the
   *combined* suite in one pass, not each family in isolation — cross-family interactions (e.g.
   shared device/fixture state) only show up when everything runs together. **No accuracy
   regression** vs the previous release.
2. **No OOM** — runs the supported size range on the card to completion and reports the largest
   size that cleared. Orb family: a disjoint-union batch sweep
   (`OrbCalculator.evaluate_batch` over `K=1..128` small systems in one device forward) — the
   batch ceiling is the OOM frontier. UMA's OOM sweep is a documented `GAP`: UMA's batched
   forward (`TTAtomCalculator.evaluate_batch` → `energy_and_forces_batch` → `edgewise` →
   `rotation.rotate`) goes through the same ALWAYS-ON `fused_rotate` kernel as its accuracy
   leg's end-to-end test, which is absent from this host's `ttnn` build (memory
   `pc-ttatom-env-missing-fused-rotate`); the per-composition bundle itself is not the blocker
   (the bundle cache works and `evaluate_batch` enforces same-composition batching). The gap
   closes automatically once the `fused_rotate` env is rebuilt on the release host; until then
   it is reported, not forced to a number. Any hard size limit is documented in the release
   notes, not discovered by a customer.
3. **No perf regression** — warm steady-state throughput on a fixed small input vs a committed
   per-card, per-model baseline (`docs/perf_baselines.json`), `FAIL` beyond a configurable noise
   margin (default ±15%). One entry per shipped family's throughput path: OrbMol
   `conservative-omol` (molecule batch), Orb-v3 bulk `conservative-inf-omat` (periodic Si batch),
   and UMA `uma-s-1` (molecule batch) — run a subset with `--model <key>` (repeatable). Card-type
   aware (a P300c baseline is never judged against a P150a run), fails loudly on `NO BASELINE`,
   and updates only via `--update-baseline --note "<why>"` (writes only the selected model(s)).
   UMA's batched forward needs the ALWAYS-ON `fused_rotate` kernel, absent from this host's
   `ttnn` build (memory `pc-ttatom-env-missing-fused-rotate`), so on such a host the UMA row
   reports `GAP` (env), not `FAIL` — reported loudly, not skipped. Record the numbers in the
   release notes.
4. **No UX regression** — the user-facing *plumbing* still works, headlessly and fast, on a tiny
   input (`ase.build.molecule("H2O")`). Asserts three things, mirroring tt-bio's
   `scripts/ux_regression.py` in methodology: (a) `tt-atom run --help` (and `relax`/`md`/top-level
   `--help`) exit 0 and list the core flags (`--relax`, `--md`, `--steps`, `--temp`, `--trace`,
   `--out`); (b) a real single-point + relax + MD(`--steps 5`) exit 0, write the `--out` geometry,
   and it parses under `ase.io.read` with finite energy and forces (not NaN/empty); (c) the CLI's
   per-step MD/relax progress stream (`step N  E=...  T=... K`; ASE `FIRE` `Step N` log) advances
   through every real step, not stuck/skipped — the direct analogue of tt-bio's "0 → diffusion"
   live-progress-bug class. Lives in `scripts/ux_regression.py` (sibling, also runnable standalone
   with `--cli-only` in GitHub CI, no card). On hosts whose `ttnn` build lacks the ALWAYS-ON
   `fused_rotate` kernel, the literal `tt-atom run`/`relax`/`md` CLI subcommands (UMA-only) are
   blocked, so legs (b)-(c) drive the identical ASE `FIRE`/`Langevin` + `_log` print pattern via the
   Orb `Calculator` API (stock `ttnn`, no `fused_rotate` dependency); leg (a) tests the literal CLI
   regardless. See `scripts/ux_regression.py` for the per-leg assertions.

If any leg that ran `FAIL`s, it does not ship — fix it or hold the release. `GAP` legs are
reported, not counted as failures (they flag a missing fixture/baseline to close, not a
regression). See `scripts/release_gate.py --help` for per-leg selection (`--leg accuracy|oom|perf|ux`),
a fast smoke (`--quick`), per-model perf selection (`--model <key>`), baseline seeding, and
`--cli-only` (UX leg, no card).

The manual description below is kept only as context on the methodology — the script above is
the actual instruction to follow going forward.

### Manual methodology (context)

1. Accuracy: full test suite green **and** numerical parity vs each model family's own reference
   within tolerance (energy rel-error and force/stress PCC), across every supported task and
   graph regime (molecular, bulk/periodic, slab), for every shipped model family — currently UMA
   (vs fairchem `uma-s-1`) and Orb-v3 / OrbMol (vs the real `orb-models` CPU oracle). Run the
   *combined* suite in one pass, not each family in isolation. No accuracy regression vs the
   previous release.
2. No OOM: run the full supported size range (small molecules → large systems) on the target
   card(s), single- and multi-card, to completion. No out-of-memory.
3. No perf regression: benchmark the release commit against the previous release; latency and
   throughput must not regress beyond noise. Record the numbers in the release notes.

## Cut a release

1. Run the gate above on hardware; capture the accuracy table + benchmark numbers.
2. Bump the version in `pyproject.toml` (SemVer) and add a dated section to `CHANGELOG.md`
   (include the measured accuracy + perf numbers).
3. Tag and push:
   ```bash
   git tag v0.2.0
   git push origin master --tags
   ```
4. CI (`.github/workflows/release.yaml`) builds the sdist + wheel, checks the tag matches the
   `pyproject` version, and publishes a **GitHub Release** with the changelog notes + wheel.

## Distribution: GitHub Releases only (NOT PyPI)

tt-atom is the **custom-kernel-only** build and **requires a source tt-metal/ttnn build** with
the `fused_rotate` op (see the README "Install"). A `pip install tt-atom` wheel therefore can't
run standalone, so publishing to PyPI would be misleading — tt-atom ships via **GitHub Releases**
(source + build instructions + tagged versions). There is intentionally no `pypi-publish` job.
(tt-bio, which *is* pip-installable, does publish to PyPI.)
