# Releasing TT-Atom

`master` is the development branch — it may contain experimental, not-yet-validated work.
**A tagged release is a promise to customers that it works.** So a release is cut only from a
commit that has passed the full on-hardware gate below.

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged, by **`scripts/release_gate.py`** — one command,
three legs, machine-readable `PASS` / `FAIL` / `GAP` per leg. Run it on card 0 of the release host:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py
```

The three legs are the three things a tagged release must clear:

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
   batch ceiling is the OOM frontier. UMA's OOM sweep is a documented `GAP` (per-composition
   bundle path). Any hard size limit is documented in the release notes, not discovered by a
   customer.
3. **No perf regression** — warm steady-state throughput on a fixed small input vs a committed
   per-card baseline (`docs/perf_baselines.json`), `FAIL` beyond a configurable noise margin
   (default ±15%). Card-type-aware (a P300c baseline is never judged against a P150a run), fails
   loudly on `NO BASELINE`, and updates only via `--update-baseline --note "<why>"`. Record the
   numbers in the release notes.

If any leg that ran `FAIL`s, it does not ship — fix it or hold the release. `GAP` legs are
reported, not counted as failures (they flag a missing fixture/baseline to close, not a
regression). See `scripts/release_gate.py --help` for per-leg selection (`--leg accuracy|oom|perf`),
a fast smoke (`--quick`), and baseline seeding.

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
