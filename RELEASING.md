# Releasing TT-Atom

`master` is the development branch. Tag only a clean commit that has passed the complete gate on
real Tenstorrent hardware.

## Release gate

Run the four regular legs on card 0 in the source-built environment documented in `README.md`:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py
```

Each device leg runs in a fresh process so its device state cannot leak into the next one.

1. **Accuracy** runs the molecular, periodic, and batched UMA parity modules plus the Orb
   real-weight, batched, and fast-mode modules against their embedded fairchem or `orb-models`
   references.
2. **No OOM** sweeps Orb and UMA disjoint batches through 128 systems in separate device
   processes and reports each completed ceiling.
3. **Performance** compares warm throughput for UMA, Orb-v3, and OrbMol with the card-specific
   baselines in `docs/perf_baselines.json`. A missing baseline is a `GAP`; a regression beyond 15%
   is a `FAIL`. Every baseline records the environment it was measured in and a row only gates when
   the run matches it, so run the perf leg in the release environment (see "Performance is gated in
   the release environment" below). Each row is the median of three independent measurement runs,
   taken once no other process holds a card (see "One draw is not a measurement" below).
4. **UX** checks the CLI help, parses output geometries, rejects non-finite results, and verifies
   that relaxation and MD progress advances through the run.

Both `FAIL` and `GAP` block a release. For development diagnostics only, `--allow-gaps` preserves
the measured rows but returns success when a fixture, baseline, weight, or required op is absent.

Useful subsets:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg accuracy
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg perf --model orb-conservative-inf-omat-batch
PYTHONPATH=. python3 scripts/release_gate.py --leg ux --cli-only
```

### Performance is gated in the release environment

All three baselines are measured on the pinned tt-metal source build. That is the environment the
install leg builds, and the only one that can run every model, so a single gate run yields a single
verdict.

Orb-v3 and OrbMol also run on a stock `ttnn` wheel, which is what an Orb-only user installs, and
they are roughly 2.2x faster there: 449 and 396 sys/s on `ttnn` 0.68.0 against 186 and 176 on the
pinned build, same fixtures and same K on the same p150a (2026-09-15). That environment is not
gated because it cannot run UMA at all, so an Orb row measured on a stock wheel reports `GAP`.

A row whose environment does not match its baseline reports `GAP`, not `FAIL`: the number is not
comparable, which is not evidence of a regression. For a source build the environment is keyed on
the tt-metal tree `ttnn` was built from, not on the `ttnn` version string, because an editable
install freezes that string at install time and it goes stale as soon as the checkout moves. A
tree that still matches the pinned commit gates against the pinned baseline whatever its metadata
says.

### One draw is not a measurement

The perf leg waits up to 10 minutes for every other process to let go of a Tenstorrent card, then
runs each model three times in three separate processes and gates the median. A single run against
a fixed threshold decides by luck: on a busy p150a the `uma-s-1-omol-batch` row was bimodal with
the 15% threshold sitting between the modes, and the same unchanged code produced both `PASS` and
`FAIL` across 22 runs. The per-run numbers are printed under each row.

If no quiet window opens in that budget, the leg measures anyway and reports a shortfall as `GAP`
rather than `FAIL`, because a contended run is not evidence of a regression. A `PASS` stands either
way, since contention can only make a run slower. Baselines cannot be seeded against a busy card at
all.

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
builds the pinned tt-metal source in a fresh directory and venv, builds and installs that exact
TT-Atom wheel, verifies both packaged exporters through an isolated Orb cache miss, checks both
custom rotation ops, runs finite UMA and Orb smoke tests, and verifies a real CLI relaxation output
parses. It fails if the worktree is dirty or the current commit is not available from `origin`.

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
sdist and wheel, and creates or refreshes the GitHub Release. Repository protection must limit
tagging to commits with a saved hardware-gate result; GitHub runners do not have TT hardware.

## Distribution

TT-Atom is distributed through GitHub Releases rather than PyPI as project policy. The wheel still
requires a compatible `ttnn` installation; UMA specifically needs the pinned source build with
the custom rotation op.
