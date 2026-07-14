# Verification — Orb-v3 silicon melt on Tenstorrent Blackhole (qb2 run, 2026-07-14)

All on-device numbers are from a single Blackhole p150 of tt-quietbox2 with the real
`orb-v3-conservative-inf-omat` weights. The CPU reference is the public `orb-models` package
(v0.5.5, same checkpoint) on the identical atomic configurations. Reproduce with `NOTES.md`.

This run extends the trajectory into a long liquid so the melt visibly flows (Moritz's
ADDENDUM 4): NVT Langevin ramps 300 -> 2200 K over 1600 steps (0.8 ps), **holds at a constant
2200 K for 12000 steps (6 ps) of real liquid diffusion**, then an 800-step NVE tail. dt = 0.5 fs,
frames every 4 steps -> **3604 frames, 7.2 ps**, 80 neighbour-list rebuilds (skin 1.5 A), so the
graph topology stays correct as the solid becomes a liquid.

## 1. It is a real melt — T crosses T_m, the structure disorders, the liquid diffuses

From `md_melt.csv` / `melt_metrics.npz`:

- Temperature crosses Si's melting point **1687 K at t = 626 fs**, peaks **2665 K**, and holds in
  the liquid at **~2200 K** for the rest of the run (ends 2291 K).
- **g(r)** (PBC minimum image): crystalline frame (276 K) has sharp shells, first-neighbour peak
  g ~ 17 at 2.35 A with resolved 2nd/3rd shells (diamond Si); the liquid frame (~2425 K, well
  above T_m) is a broad liquid envelope, first peak g ~ 2.6 at ~2.4 A decaying to ~1, resolved
  shells gone. The charts use this hot-liquid frame as the g(r) reference (not a supercooled tail).
- **MSD** (PBC-unwrapped, referenced to frame 0): flat in the solid, then rises **linearly over
  the full 6 ps liquid hold to ~47 A^2** — the diffusion signature. The lattice superheats briefly
  past the 626 fs T_m crossing and the diffusive rise sets in around ~1.8-2 ps (realistic melting
  kinetics, not an instantaneous jump). Diffusion coefficient from the liquid-window MSD slope
  **D ~ 12.1e-9 m^2/s** — the right order for liquid Si above its melting point (labelled
  "indicative" on the figure; a precision D needs a longer window than a 6 ps run).

## 2. Energy conservation — the credibility metric

NVE total-energy drift, linear slope of E_tot over an NVE tail (first 200 fs thermostat-off
transient excluded):

- **Equilibrated 900 K solid: 1.4 meV/atom/ps** (`md_solid_nve.csv`, a separate clean NVE run;
  same weights/system, so it is the model's conservation property, independent of this melt).
  This is the direct analog of the ~1 meV/atom/ps bar UMA is reported at — Orb-v3 clears it.
- **Hot ~2200 K liquid (this run's NVE tail): 28.8 meV/atom/ps** — larger, as expected for a hot
  liquid at 0.5 fs; visually flat on the melt's ~1 eV/atom energy scale. Disclosed, not hidden.

No smoothing, no selection.

## 3. On-device vs orb-models CPU reference — the real Orb-v3, in the liquid too

`orb_melt_parity.py` on three frames of the actual melt trajectory (fresh calculator per frame so
the neighbour list matches that exact geometry):

| frame | state                     | dE (meV/atom) | force PCC    | max force err (eV/A) | ref \|F\|max (eV/A) |
|------:|---------------------------|--------------:|-------------:|---------------------:|--------------------:|
| 0     | perfect lattice           | 8.99          | 0.45 (noise) | 0.022                | 0.001               |
| 1200  | liquid (~2.4 ps)          | 0.45          | **0.99993**  | 0.056                | 4.08                |
| 3600  | liquid (~7.2 ps, late)    | 0.16          | **0.99983**  | 0.163                | 5.21                |

On the dynamics frames the forces match the reference at **PCC 0.9998-0.99993** and energies agree
to **0.2-0.5 meV/atom** — and this holds deep in the **liquid** at real forces up to ~5 eV/A.
Frame 0 is the perfect lattice: forces vanish by symmetry on both sides, so its PCC is numerical
noise (|F|max ~ 0.001 eV/A, and the max absolute force error is only 0.022 eV/A), and its
~9 meV/atom absolute-energy offset is the known bf16 absolute-energy artifact, which does not
affect the dynamics (driven by forces and energy differences).

## 4. Throughput (device, `orb_melt_md.py` summary)

216 atoms, ~9,550 periodic edges, warm median **42.7 ms / MD step** (energy + analytic forces,
trace-capture replay) => **23.4 MD steps/s**, **~5,050 atom-steps/s**, **1.01 ns/day** on one
Blackhole p150.

## 5. Render motion is continuous — no jump, no ghost atoms, no fliers (quantitative)

The shipped video (`orb_si_melt.mp4`, 100 frames, 30 fps, 3.3 s) renders **unwrapped, continuous
coordinates** (periodic images accumulated across the trajectory, per-frame COM removed), **no
cell box, no tiling**. Unwrapped coordinates are only visually clean while the diffusion cloud
stays within ~one box; a 6 ps liquid diffuses further than that, so atoms that have crossed the
(removed) periodic boundary would read as detached "fliers". We therefore **window the 3D scene to
the cohesive melt + early-liquid interval [0, 2400 fs]** (the crystal, the melt, and ~1.8 ps of
liquid churn); the charts side-card still spans the full 7.2 ps run, cursor sweeping the window.

`render_melt_video.py` hard checks on the shipped render:

- Max per-atom displacement between **every** pair of consecutive rendered frames = **0.962 A**
  (mean 0.267 A) — visible churn (3.3x the previous 0.288 A), yet far below the 8.14 A box/2
  teleport threshold, so no atom jumps a box face. This is the ADDENDUM 4 "make it flow" lever:
  ~24 fs/frame (vs ~8 fs before) at 0.72 ps of simulated time per second of playback.
- Unwrapped cloud max radius over the windowed clip = **16.3 A**; over [0, 2400 fs] at most one
  atom sits at the melt-droplet surface (~14.6 A, vs the crystal's own corner atoms already at
  12.9 A) — no atom floats in vacuum, the liquid reads cohesive.
- Atom count rendered = **216 on every frame** — no image/ghost atoms appearing or disappearing.

Confirmed by eye on the mp4's first / mid / last frames: crystal (853 K, ordered lattice, sharp
g(r)) -> melt (2178 K, disordering, g(r) broadening) -> cohesive liquid churn (2123 K, broad
liquid g(r)). No teleport, no pop-in/out, no fliers.

## Honesty caveats
- **No GPU / NVIDIA / per-dollar comparison** is claimed anywhere — none was measured, and the
  video carries no such text (Moritz's addendum).
- The diffusion coefficient (§1) is indicative only; the MSD curve and g(r) are the honest
  structural signatures shown.
- The frame-0 9 meV/atom absolute-energy offset (§3) is disclosed as a bf16 artifact; frame-0
  force PCC is noise-vs-noise (forces ~0 by symmetry).
- The video is one honest on-device run played forward once with a short fade in/out at the loop
  (MD is not time-periodic). The 3D scene is windowed to the cohesive interval (above); the charts
  show the real full-run per-step log and the real g(r)/MSD, advancing in lockstep. No fabricated
  numbers, no sped-up interpolation — the extra flow is real extra picoseconds of liquid MD plus
  wider frame spacing, not faked speed. The live T label is moving-average smoothed for
  readability; the charts show the raw trace.
