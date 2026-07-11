# UMA-S-1.2 on Tenstorrent — validation & performance

This documents the parity and performance of `uma-s-1.2` running on Tenstorrent through TT-Atom,
measured against Meta's fairchem on CPU using the released `uma-s-1p2.pt` checkpoint. It backs the
one-line claim in the README; see [`../README.md`](../README.md) for how to run the model.

`uma-s-1.2` differs from `uma-s-1` by fairchem's **charge-balanced channels**: the `l=0` charge
channels are re-balanced to the system charge (a self-adjoint per-system mean-subtraction, plus a
`charge/natoms` target) after every block. TT-Atom ports this forward and backward through the
single-system, batched, and traced paths. Without it, the force PCC on the released checkpoint
collapses to ~0.83; with it, parity is restored.

## Method

- **Reference:** fairchem-core on CPU — the released `uma-s-1p2.pt`, MoLE-merged per composition,
  through `FAIRChemCalculator`.
- **Device:** TT-Atom on a Tenstorrent Wormhole card, `ttnn` 0.68.0 (bf16 backbone with fp32
  accumulation and an fp32 energy head).
- **Per system:** a static set (equilibrium + seeded rattles) and a seeded NVE trajectory
  (velocity-Verlet). The *identical* geometries are fed to both engines.
- **Metrics:** pooled force PCC (over all geometries), median energy discrepancy per atom
  (meV/atom), stress PCC (periodic), and NVE total-energy drift for each engine.
- **Pass** (per the released `uma-s-1` tolerances) = static **and** trajectory force PCC > 0.99,
  static **and** trajectory median energy < 5 meV/atom, and — when periodic — stress PCC > 0.99.
  Chemical accuracy is ~43 meV/atom (1 kcal/mol), so the 5 meV/atom bar is deliberately strict.

**Why pooled force PCC, not per-geometry.** At a symmetric equilibrium or a perfect crystal
lattice the forces are ~0 by symmetry, so a per-geometry PCC just correlates numerical noise and
reads low despite a tiny absolute error. Pooling over the rattled geometries (where `|F|` is
meaningful) is the honest measure; the artifact is called out explicitly below rather than hidden.

## Coverage

**757 systems**, 2–99 atoms, ~63 elements, molecular + **491 periodic**. Tasks `omol`, `omat`,
`oc20`, `odac`, `omc`. The screen spans 175 `oc20` catalysis surfaces (metal facets × CO/H/O/OH),
40 metal nanoparticles, the 22 S22 non-covalent complexes, open-shell radicals, charged
transition-metal complexes, 120 equation-of-state volume-strain curves, and a broad spread of
elemental / binary / perovskite crystals.

## Forces

Forces reproduce fairchem across the screen with **no genuine errors**: **741 / 757** systems keep
pooled force PCC **> 0.999** (754/757 > 0.99), and the three below 0.99 are all the **zero-force
perfect-lattice artifact** — PCC correlates noise where `|F|≈0` by symmetry, while the absolute error
is tiny (`el_Pb`, `el_Pb_exp`, `s_Pb100_clean`, all traj Fmae < 1.6 meV/Å). Wherever forces are
non-negligible, PCC is ≈ 1.0; the showcase systems are all exactly 1.0000.

The re-screen surfaced one real force regression from `master`'s on-device-backward perf work —
`el_Sn_cmp` (2-atom compressed β-Sn, traj PCC 0.73 / 230 meV/Å): the radial-MLP backward computes its
LayerNorm VJP in **bf16** by default (unlike the fp32-accurate SH-norm/gate backwards), mis-directing
the gradient on that out-of-distribution geometry while the energy stays correct. **This PR fixes it**
by running the small radial backward in fp32 — verified against an fp64 `mirror.radial_mlp` oracle
(device `rad.bw` PCC 0.35 → 1.0); `el_Sn_cmp` is now **1.0000**, at ~4% eager cost. (`master`'s opt-in
`fused_ln_bw` kernel is bf16-only and off by default, so the default path carried the bug.)

## Energy

**691 / 757 pass** the strict bar (force PCC > 0.99, energy < 5 meV/atom, stress PCC > 0.99); passing
systems sit at a median of **1.33 meV/atom**. The misses are the **bf16 backbone energy floor**: the
error scales as `|E_raw| / N`, so it hits the highest-`|raw|`, smallest systems hardest — heavy-metal
diatomics / crystals (`el_Ta_exp` 11, `m_CH` 10, `el_Mo` 9, `el_Tc_exp` 8 meV/atom) and dense
ionic/oxide cells (`x_CaO_rock`). Their **forces are unaffected** (PCC ≈ 1.0) and **every system stays
under chemical accuracy** (≈ 43 meV/atom).

This is a **fundamental bf16-forward precision limit, not a bug** (identical inputs give identical
outputs to within it): the reference-subtracted per-node energy rounds in the bf16 backbone. The fp32
energy head removes the reduction-order term; the residual is the bf16-forward floor. Net, on the
current `master` (custom fused kernels + on-device backward) with these fixes, the pass rate is
**691/757 — above the 672/757 the earlier backbone reached** (the perf work's numerics plus the fp32
head slightly *lower* the floor overall, even as it moves the exact set of borderline systems around).

## Stress (periodic)

Median stress PCC **0.99996**; **488 / 491 ≥ 0.99**. The few lowest are near-zero-stress soft metals /
centrosymmetric cells whose equilibrium stress is itself ~0 (PCC-of-noise).

## Showcase systems

Four end-to-end structures spanning catalysis, nanoparticles, and a charged solvated
transition-metal chelate — all reproduce fairchem forces exactly:

| system | task | atoms | charge / spin | force PCC | energy |
|--------|------|------:|:-------------:|----------:|-------:|
| Pt(111) + CO + O (CO oxidation) | oc20 | 39 | 0 / 1 | **1.0000** | 0.31 meV/atom |
| Cu₅₅ nanoparticle + CO          | omat | 57 | 0 / 1 | **1.0000** | 0.91 meV/atom |
| [Cu(EDTA)]²⁻ chelate            | omol | 33 | −2 / 2 | **1.0000** | 1.18 meV/atom |
| [Cu(EDTA)]²⁻ + 22 H₂O           | omol | 99 | −2 / 2 | **1.0000** | 0.88 meV/atom |

## Performance — CPU vs Tenstorrent

Energy + forces, milliseconds per call, MgO supercells, measured on the current `master` (custom fused
kernels + on-device backward) with this PR's fixes. TT (Wormhole, `ttnn` 0.68.0) eager and traced vs
fairchem on a 16-thread CPU; each point in its own process:

| MgO cell | atoms | TT eager | TT trace | CPU (16-thread) | speedup (CPU / TT-best) |
|----------|------:|---------:|---------:|----------------:|------------------------:|
| 1×1×1 | 8   | 103  | 75   | 138  | 1.8× |
| 2×2×2 | 64  | 317  | 319  | 860  | 2.7× |
| 3×3×3 | 216 | 1059 | 973  | 5326 | **5.5×** |
| 4×4×4 | 512 | 2794 | OOM  | —    | —    |

Molecular water boxes (eager / trace ms): ethanol 90 / 47, H₂O×8 142 / 135, H₂O×27 (81 at) 358 / 347,
H₂O×64 (192 at) 807 / 764.

TT overtakes the 16-core CPU by ~64 atoms and reaches **5.5× at 216 atoms** (trace) — the margin grows
with size. `master`'s kernel work makes this run **~2.2× faster than the pre-perf-pass backbone**
(MgO-216 eager 2307 → 1059 ms) and, notably, restores the trace path at scale: trace now captures
through 216-atom periodic / 192-atom molecular cells (it was capture-bound at large N on the earlier
on-device-backward), and eager reaches the 512-atom cell (2794 ms; only the 512 *trace* still exceeds
single-card DRAM). The trace loop (the MD fast path) is ~1.9× the eager call on small molecules (ethanol
90 → 47 ms), bit-identical forces. The fp32 radial-backward fix here adds ~4%. Below ~10 atoms the CPU
wins — TT's advantage is at real system sizes.

## Reproduce

- **A/B screen:** fairchem-CPU references vs TT-Atom on the card, over the system list above;
  pooled force PCC, per-atom energy, stress PCC, and NVE drift.
- **Gated device parity test (in-repo):** [`../tests/test_realweight_uma_s_1p2.py`](../tests/test_realweight_uma_s_1p2.py)
  checks device energy + analytic forces against the fairchem oracle for a real `uma-s-1.2` golden.
  It auto-skips without the (gated) checkpoint. To run it:

  ```bash
  HF_HUB_OFFLINE=1 <refenv>/bin/python tests/gen_golden_real.py \
      --system molecule --task omol --ckpt uma-s-1p2 \
      --out ~/.ttatom_run/goldens_real/ethanol_omol_uma_s_1p2.npz
  TT_VISIBLE_DEVICES=0 <venv>/bin/python -m pytest tests/test_realweight_uma_s_1p2.py -q
  ```

Numbers are from `ttnn` 0.68.0; op numerics can shift slightly between `ttnn` versions, so confirm
parity on the version you actually run.
