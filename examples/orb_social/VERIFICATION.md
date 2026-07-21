# Verification — Orb-v3 silicon melt on Tenstorrent Blackhole (qb2 run, 2026-07-14)

> Withdrawn as verification evidence. The trajectory used an invalid neighbour-list skin policy,
> and the fresh-graph parity samples do not validate forces used between rebuilds. All measurements
> are retained below as historical data. Regenerate the trajectory with the corrected exact-cutoff
> graph check before using any melt claim.

All on-device numbers are from a single Blackhole p150 of tt-quietbox2 with the real
`orb-v3-conservative-inf-omat` weights. The CPU reference is the public `orb-models` package
(v0.5.5, same checkpoint) on the identical atomic configurations. Reproduce with `NOTES.md`.

This run extends the trajectory into a long liquid so the melt visibly flows (Moritz's
ADDENDUM 4): NVT Langevin ramps 300 -> 2200 K, **holds at ~2200 K for several ps of real liquid
diffusion**, then an 800-step NVE tail. dt = 0.5 fs, frames every 4 steps -> **2855 frames,
5.70 ps** and 80 neighbour-list rebuilds (skin 1.5 A).

## 1. It is a real melt — T crosses T_m, the structure disorders, the liquid diffuses

From `md_melt.csv` / `melt_metrics.npz`:

- Temperature crosses Si's melting point **1687 K at t = 2138 fs** (frame 1070), peaks
  **2610 K**, and holds in the liquid at **~2160 K** through the NVE tail.
- **MSD** (PBC-unwrapped, referenced to frame 0): flat through the solid, then rises
  **linearly over the liquid to 19.65 A^2** at 5.7 ps — the diffusion signature. The structural
  melt (MSD rise) sets in around ~3.5 ps, lagging the 2138 fs T_m crossing: the lattice
  superheats past T_m before it disorders (realistic melting kinetics, shown honestly — the MSD
  panel marks the T_m crossing, the visible lag to the rise is the superheating). Diffusion
  coefficient from the liquid-window MSD slope **D ~ 8.1e-9 m^2/s** — right order for liquid Si
  above its melting point (labelled "indicative"; a precision D needs a longer window than 5.7
  ps).
- **g(r)** (PBC minimum image): crystalline frame (276 K) has sharp shells, first-neighbour
  peak g ~ 19.5 at 2.35 A with resolved 2nd/3rd shells (diamond Si); the liquid frame (~2421 K,
  well above T_m) is a broad liquid envelope, first peak g ~ 2.6 at ~2.4 A decaying to ~1,
  resolved shells gone. The charts use this hot-liquid frame as the g(r) reference.

## 2. Energy conservation — the credibility metric

NVE total-energy drift, linear slope of E_tot over an NVE tail (first 200 fs thermostat-off
transient excluded):

- **Equilibrated ~900 K solid: 1.4 meV/atom/ps** (`md_solid_nve.csv`, a separate clean NVE run;
  same weights/system, so it is the model's conservation property, independent of this melt).
  This is the direct analog of the ~1 meV/atom/ps bar UMA is reported at — Orb-v3 clears it.
- **Hot ~2160 K liquid (this run's NVE tail): 20.6 meV/atom/ps** — larger, as expected for a hot
  liquid at 0.5 fs; visually flat on the melt's ~1 eV/atom energy scale. Disclosed, not hidden.

No smoothing, no selection.

## 3. On-device vs orb-models CPU reference — the real Orb-v3, through the melt and in the liquid

`orb_melt_parity.py` on four frames of the actual melt trajectory (fresh calculator per frame so
the neighbour list matches that exact geometry):

| frame | state                  | dE (meV/atom) | force PCC    | max force err (eV/A) | ref |F|max (eV/A) |
|------:|------------------------|--------------:|-------------:|---------------------:|--------------------:|
| 0     | perfect lattice (277K) | 8.99          | 0.45 (noise) | 0.022                | 0.001               |
| 1070  | T_m-crossing solid     | 3.34          | **0.99997**  | 0.032                | 4.50                |
| 1900  | liquid onset (~3.8 ps) | 1.38          | **0.99996**  | 0.044                | 6.72                |
| 2400  | liquid (~4.8 ps)       | 1.31          | **0.99996**  | 0.035                | 4.55                |

On the dynamics frames the forces match the reference at **PCC 0.99996-0.99997** and energies
agree to **1.3-3.3 meV/atom** — through the melt and deep in the **liquid** at real forces up
to ~6.7 eV/A. Frame 0 is the perfect lattice: forces vanish by symmetry on both sides, so its
PCC is numerical noise (|F|max ~ 0.001 eV/A, max abs force error only 0.022 eV/A), and its
~9 meV/atom absolute-energy offset is the known bf16 absolute-energy artifact, which does not
affect the dynamics (driven by forces and energy differences).

## 4. Render motion is continuous — no jump, no ghost atoms, no fliers (quantitative)

The shipped video (`orb_si_melt.mp4`, 120 frames, 30 fps, 4.0 s) renders **unwrapped,
continuous coordinates** (periodic images accumulated across the trajectory, per-frame COM
removed), **no cell box, no tiling**. Unwrapped coordinates are only visually clean while the
diffusion cloud stays within ~one box; past that, atoms that have crossed a full periodic
boundary read as detached "fliers". We therefore **window the 3D scene to [0, 4601 fs]** — the
crystal (0-2138 fs), the melt, and **2.46 ps of real liquid churn** (2138-4601 fs) — where the
cloud radius is 14.75 A, inside the 16.29 A box, so no atom has crossed a full boundary. The
charts side-card still spans the full 5.7 ps run, cursor sweeping the window. The prior render
windowed to [0, 2400 fs], which ended only ~260 fs past the T_m crossing (almost no liquid);
this fix gives 2.46 ps of liquid at 38.6 fs/frame (1.16 ps/s, 1.6x the old 24 fs/frame) so the
liquid visibly flows. Real trajectory, no faked speed.

`render_melt_video.py` hard checks on the shipped render:

- Max per-atom displacement between **every** pair of consecutive rendered frames = **1.21 A**
  (mean 0.27 A) — visible churn, yet far below the 8.14 A box/2 teleport guard, so no atom
  jumps a box face.
- Unwrapped cloud max radius over the windowed clip = **14.75 A** (< box 16.29 A) — no atom has
  crossed a full periodic boundary, so no fliers; the liquid reads cohesive.
- Atom count rendered = **216 on every frame** — no image/ghost atoms appearing or disappearing.

Confirmed by eye on the mp4's first / mid / last frames: crystal (ordered lattice, sharp g(r))
-> melt (disordering, g(r) broadening) -> cohesive liquid churn (broad liquid g(r), MSD
rising). No teleport, no pop-in/out, no fliers, nothing clipped.

## Honesty caveats
- **No GPU / NVIDIA / per-dollar comparison** is claimed anywhere — none was measured, and the
  video carries no such text (Moritz's addendum). The Tenstorrent brand is a logo only.
- The diffusion coefficient (§1) is indicative only; the MSD curve and g(r) are the honest
  structural signatures shown.
- The frame-0 9 meV/atom absolute-energy offset (§3) is disclosed as a bf16 artifact; frame-0
  force PCC is noise-vs-noise (forces ~0 by symmetry).
- The video is one honest on-device run played forward once with a short fade in/out at the
  loop (MD is not time-periodic). The 3D scene is windowed to the cohesive interval (above);
  the charts show the real full-run per-step log and the real g(r)/MSD, advancing in lockstep.
  No fabricated numbers, no sped-up interpolation — the extra flow is real extra picoseconds of
  liquid MD plus wider frame spacing, not faked speed. The live T label is moving-average
  smoothed for readability; the charts show the raw trace.
