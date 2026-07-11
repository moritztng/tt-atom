# Releasing TT-Atom

`master` is the development branch — it may contain experimental, not-yet-validated work.
**A tagged release is a promise to customers that it works.** So a release is cut only from a
commit that has passed the full on-hardware gate below.

## Release gate — MUST pass on real Tenstorrent hardware before tagging

GitHub CI only builds and imports the package (no card). Everything that matters is verified
on-device, on the exact commit to be tagged:

1. **Accuracy / correctness** — full test suite green **and** numerical parity vs each model
   family's own reference within tolerance (energy rel-error and force/stress PCC), across every
   supported task and graph regime (molecular, bulk/periodic, slab), **for every shipped model
   family** — currently UMA (vs fairchem `uma-s-1`) and Orb-v3 / OrbMol (vs the real `orb-models`
   CPU oracle). Run the *combined* suite in one pass, not each family in isolation — cross-family
   interactions (e.g. shared device/fixture state) only show up when everything runs together.
   **No accuracy regression** vs the previous release.
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
