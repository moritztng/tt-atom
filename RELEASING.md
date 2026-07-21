# Releasing TT-Atom

`master` is the development branch. Tag only a clean commit that has passed the complete gate on
real Tenstorrent hardware.

## Release gate

Run the four regular legs on card 0:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py
```

Each device leg runs in a fresh process so its device state cannot leak into the next one.

1. **Accuracy** runs the real-weight parity modules in `tests/test_*realweight*.py` against their
   embedded fairchem or `orb-models` references. A missing fixture or required custom op is
   reported as `GAP`, never as a silent pass.
2. **No OOM** sweeps Orb disjoint batches through 128 systems and reports the largest completed
   batch. The UMA row reports `GAP` when the installed `ttnn` lacks `fused_rotate`.
3. **Performance** compares warm throughput for UMA, Orb-v3, and OrbMol with the card-specific
   baselines in `docs/perf_baselines.json`. A missing baseline is a `GAP`; a regression beyond 15%
   is a `FAIL`.
4. **UX** checks the CLI help, parses output geometries, rejects non-finite results, and verifies
   that relaxation and MD progress advances through the run.

`FAIL` blocks a release. Review every `GAP` before tagging and record any accepted environment or
fixture limitation in the release notes.

Useful subsets:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg accuracy
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg perf --model orb-conservative-inf-omat-batch
PYTHONPATH=. python3 scripts/release_gate.py --leg ux --cli-only
```

Refresh a performance baseline only for an intentional, measured change:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py \
  --leg perf --model orb-conservative-inf-omat-batch \
  --update-baseline --note "reason for the new baseline"
```

Review and commit the resulting `docs/perf_baselines.json` diff with that change.

## Clean installation

Once per release, verify the customer installation path:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg install
```

Run this from a clean, committed, pushed worktree. The leg clones the current commit from `origin`,
builds the pinned tt-metal source in a fresh directory and venv, installs that exact TT-Atom
commit, checks both custom rotation ops, runs finite UMA and Orb smoke tests, and verifies a real
CLI relaxation output parses. It fails if the worktree is dirty or the current commit is not
available from `origin`.

The validated tt-metal commit is
`8d759240fdd763a38e3abdc8344076f584dc4f4d` on branch `moritztng/tt-atom`. Keep this value in sync
with `README.md`, `custom_kernels/README.md`, and `scripts/release_gate.py`.

## Cut the release

1. Save the gate output and measured accuracy/performance values.
2. Bump `pyproject.toml` and add a dated section to `CHANGELOG.md`.
3. Commit and rerun the gate on that exact commit.
4. Tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin master vX.Y.Z
   ```

`.github/workflows/release.yaml` verifies that the tag matches the package version, builds the
sdist and wheel, and creates or refreshes the GitHub Release.

## Distribution

TT-Atom is distributed through GitHub Releases, not PyPI. UMA requires a source tt-metal build
with the custom rotation op, so a standalone `pip install tt-atom` package would be misleading.
