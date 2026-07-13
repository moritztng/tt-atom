# Verification — Orb-v3 silicon melt on Tenstorrent Blackhole (qb2 run, 2026-07-13)

All on-device numbers are from physical card 1 of tt-quietbox2 (single Blackhole p150) with the
real `orb-v3-conservative-inf-omat` weights. The CPU reference is the public `orb-models` package
(v0.5.5, same checkpoint) on the identical atomic configurations. Reproduce with `NOTES.md`.

## 1. It is a real melt — T crosses T_m, the structure disorders

NVT Langevin ramps 300 -> 2800 K over 1800 steps (900 fs at dt = 0.5 fs), then 1000 NVE steps.
From `md_melt.csv` / `melt_metrics.npz`:

- Temperature crosses Si's melting point **1687 K at t = 554 fs**, peaks **2644 K**, ends the run
  in the liquid at **1589 K**.
- **g(r)** (PBC minimum image, from the trajectory):
  - crystalline frame (276 K): sharp shells, first-neighbour peak g ~ 17 at 2.35 A and resolved
    2nd/3rd shells (diamond Si).
  - liquid frame (1589 K): broad single-peak liquid envelope, g ~ 2.5 at ~2.4 A; the long-range
    crystalline order is gone.
- **MSD** (PBC-unwrapped, referenced to frame 0): flat in the solid, rising to ~2.6 A^2 once
  atoms leave their lattice sites — the diffusion onset. A diffusion coefficient is computed
  (corrected units: D ~ 8.3e-9 m^2/s, the right order for liquid Si) but treated as **indicative
  only** — the ~500 fs liquid window is too short for the long-time linear regime an honest D
  needs, and the MSD is still super-linear (the system is melting through the window, not an
  equilibrated liquid). The MSD curve and g(r) are the primary structural evidence.

The neighbour list rebuilt **9 times** over the melt (skin 1.5 A), so the graph topology stayed
correct as the solid became a liquid.

## 2. Energy conservation — the credibility metric

NVE total-energy drift, linear slope of E_tot over the NVE tail (first 200 fs thermostat-off
transient excluded):

- **Equilibrated 900 K solid: 1.4 meV/atom/ps** (`md_solid_nve.csv`, 400 NVT settle + 1200 NVE).
  This is the direct analog of the ~1 meV/atom/ps bar UMA is reported at — Orb-v3 on this port
  clears it. (The on-device MD summary's own window reported 0.8 meV/atom/ps; the 1.4 figure is
  the transient-excluded `analyze_melt.py` slope used in the charts — both are real slopes of the
  same E_tot.)
- **Hot liquid (the melt run's NVE tail): 14.6 meV/atom/ps** — larger, as expected for a ~1800 K
  liquid at 0.5 fs; visually flat on the melt's 0.8 eV/atom scale. Disclosed, not hidden.

No smoothing, no selection.

## 3. On-device vs orb-models CPU reference — the real Orb-v3, in the liquid too

`orb_melt_parity.py` on three frames of the actual melt trajectory (fresh calculator per frame so
the neighbour list matches that exact geometry):

| frame | state                    | dE (meV/atom) | force PCC   | max force err (eV/A) | ref \|F\|max (eV/A) |
|------:|--------------------------|--------------:|------------:|---------------------:|--------------------:|
| 0     | perfect lattice          | 8.99          | 0.45 (noise)| 0.022                | 0.001               |
| 350   | thermal solid (~175 fs)  | 1.39          | **0.99998** | 0.035                | 5.26                |
| 700   | liquid (~1400 fs)        | 1.24          | **0.99995** | 0.030                | 4.10                |

On the dynamics frames the forces match the reference at **PCC 0.9999**, max error ~0.7 % of the
peak force, and energies agree to ~1.2–1.4 meV/atom — and this holds in the **liquid** (frame
700). Frame 0 is the perfect lattice: forces vanish by symmetry on both sides, so its PCC is
numerical noise, and its ~9 meV/atom absolute-energy offset is the known bf16 absolute-energy
artifact, which does not affect the dynamics (driven by forces and energy differences).

## 4. Throughput (device, `orb_melt_md.py` summary)

216 atoms, ~9,400 periodic edges, warm median **42.7 ms / MD step** (energy + analytic forces,
trace-capture replay) => **23.4 MD steps/s**, **~5,050 atom-steps/s**, **1.01 ns/day** on one
Blackhole p150.

## Honesty caveats
- **No GPU / NVIDIA / per-dollar comparison** is claimed anywhere — none was measured, and per
  Moritz the video carries no such claim.
- The diffusion coefficient (§1) is indicative only; the MSD curve and g(r) are the honest
  structural signatures shown.
- The frame-0 9 meV/atom absolute-energy offset (§3) is disclosed as a bf16 artifact.
- The video is one honest on-device run. The loop is a boomerang purely so it seams cleanly; MD
  is not time-periodic. The periodic-image shell in the render is exactly the periodic system the
  MD integrated (§ render decision 4 in NOTES).
