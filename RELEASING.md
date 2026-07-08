# Releasing TT-Atom

`master` is the development branch — it may contain experimental, not-yet-validated work.
**A tagged release is a promise to customers that it works.** So a release is cut only from a
commit that has passed the full on-hardware gate below.

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged:

1. **Accuracy / correctness** — full test suite green **and** numerical parity vs the reference
   (fairchem `uma-s-1`) within tolerance (energy rel-error and force/stress PCC) across every
   supported task and graph regime (molecular, bulk/periodic, slab). **No accuracy regression**
   vs the previous release.
2. **No OOM** — run the full supported size range (small molecules → large systems) on the
   target card(s), single- and multi-card, to completion. No out-of-memory. Any hard size limit
   is documented in the release notes, not discovered by a customer.
3. **No perf regression** — benchmark the release commit against the previous release; latency
   and throughput must not regress beyond noise. Record the numbers in the release notes.

If any of the three fails, it does not ship — fix it or hold the release.

## Cut a release

1. Run the gate above on hardware; capture the accuracy table + benchmark numbers.
2. Bump the version in `pyproject.toml` (SemVer) and add a dated section to `CHANGELOG.md`
   (include the measured accuracy + perf numbers).
3. Tag and push:
   ```bash
   git tag v0.1.0
   git push origin master --tags
   ```
4. CI (`.github/workflows/release.yml`) builds the sdist + wheel, checks the tag matches the
   `pyproject` version, and publishes a **GitHub Release** with the changelog notes + wheel. If
   PyPI is enabled (below), it also publishes there.

## Enabling PyPI (one-time, maintainer)

Until this is done, releases go to GitHub only. To also publish `pip install tt-atom`:

1. On <https://pypi.org/manage/account/publishing/> add a **pending Trusted Publisher** for
   project `tt-atom`: owner `moritztng`, repository `tt-atom`, workflow `release.yml`,
   environment `pypi`. (No API token is created or stored — GitHub authenticates via OIDC.)
2. In GitHub repo settings → *Secrets and variables → Actions → Variables*, add
   `PYPI_ENABLED = true` to un-gate the `pypi-publish` job.
3. The next `v*` tag publishes to PyPI automatically.
