# Verification — Orb-v3 silicon melt on Tenstorrent Blackhole

Every claim in `DRAFT_POST.md` is backed here. All on-device numbers are from physical device 0
(single Blackhole p150) with the real `orb-v3-conservative-inf-omat` weights; the CPU reference is
the public `orb-models` package (v0.5.5, `orb-v3-conservative-inf-omat`, `float32-high`) on the
same atomic configurations. Reproduce with `NOTES.md`.

## 1. It is a real melt — temperature crosses T_m, the structure disorders

NVT Langevin ramps the thermostat target 300 → 2800 K over 1800 steps (900 fs at dt = 0.5 fs), then
runs 1000 steps (500 fs) of NVE. From `md_melt.csv` / `melt_metrics.npz`:

- Temperature crosses Si's melting point **1687 K at t ≈ 554 fs**, peaks ~2640 K, and the run ends
  in the liquid at ~1590 K.
- **Radial distribution g(r)** (computed from the saved trajectory with PBC minimum image):
  - first-thermal frame (t ≈ 2 fs, 276 K solid): sharp crystalline peaks, first-shell g_max ≈ 19.
  - liquid frame (t ≈ 1400 fs): broad single-peak liquid envelope, g_max ≈ 2.5 at r ≈ 2.4 Å — the
    long-range crystalline structure is gone.
- **MSD** (PBC-unwraped, referenced to the t = 0 lattice): rises **0 → 2.6 Å²** as atoms leave
  their lattice sites — the onset of diffusion. (A diffusion *coefficient* is not quoted: over a
  ~400 fs liquid window the MSD slope is dominated by vibration, not the long-time linear regime
  needed for an honest D, so quoting one would be fabricated. The MSD curve and g(r) are the
  honest structural signatures; both evolve live in the video.)

The neighbour list rebuilds 9 times over the melt (whenever any atom moves > 1.5 Å from its last
build), so the graph topology stays correct as the solid becomes a liquid — the frozen-topology
trace replay that's exact for a solid is wrong once atoms diffuse, so the melt calculator rebuilds
it (and re-captures the trace) on a skin margin.

## 2. Energy conservation — the credibility metric

NVE total-energy drift, measured as the linear slope of E_tot over the NVE tail (excluding the
first 200 fs thermostat-off transient):

- **Equilibrated 900 K solid: 1.4 meV/atom/ps** (`md_solid_nve.csv`, 400 NVT settle + 1200 NVE
  steps at dt = 0.5 fs). This is the direct analog of the ~1 meV/atom/ps bar UMA is reported at —
  Orb-v3 on this port clears it.
- **Hot liquid (the melt run's NVE tail, ~1800 K mean): 14.6 meV/atom/ps.** Larger, as expected
  for a 2000 K liquid at 0.5 fs; the video's energy panel shows this tail, and it is visually flat
  on the 0.8 eV/atom scale of the melt (the drift is 0.007 eV/atom over 500 fs).

Both numbers are real slopes of the measured E_tot; no smoothing, no selection. The solid-state
number is the headline; the liquid-state number is disclosed.

## 3. On-device vs orb-models CPU reference — the real Orb-v3, in the liquid too

`orb_melt_parity.py` on three frames of the actual melt trajectory (CPU reference
`orb-v3-conservative-inf-omat` `float32-high` vs on-device, fresh calculator per frame so the
neighbour list matches that exact geometry):

| frame | state | ΔE (meV/atom) | force PCC | max force err (eV/Å) | ref \|F\|max (eV/Å) |
|------:|-------|--------------:|----------:|---------------------:|--------------------:|
| 0     | perfect lattice | 9.0 | — (forces ≈ 0) | 0.023 | 0.001 |
| 350   | thermal solid (~175 fs) | 1.39 | **0.99998** | 0.035 | 5.26 |
| 700   | liquid (~1400 fs)       | 1.24 | **0.99995** | 0.030 | 4.10 |

On the dynamics frames the forces match the reference at **PCC 0.9999** with max error ~0.7 % of
the peak force, and energies agree to ~1.2–1.4 meV/atom — and this holds in the **liquid** (frame
700), not just the solid. Frame 0 is the perfect lattice (forces vanish by symmetry on both sides,
so its PCC is numerical noise); its 9 meV/atom absolute-energy offset is the known bf16
absolute-energy artifact, which does not affect the dynamics (driven by forces and energy
*differences*).

## 4. Throughput (device, `orb_melt_md.py` summary)

- 216 atoms, ~9,900 periodic edges, warm median **48 ms / MD step** (energy + analytic forces,
  trace-capture replay) ⇒ **20.8 MD steps/s**, **~4,490 atom-steps/s**, **0.90 ns/day** on one
  Blackhole p150. The 9 neighbour-list rebuilds over the melt each re-capture the trace; the
  reported step time is the warm *replay* median (the steady-state rate between rebuilds).

## Honesty caveats
- No GPU or per-dollar comparison is claimed — none was measured (would need a like-for-like GPU MD
  run of the identical 216-atom Si melt). The draft leaves it an explicit TBD.
- A diffusion coefficient is not quoted (§1): the liquid window is too short for the long-time
  linear MSD regime an honest D requires. The MSD curve and g(r) are shown instead.
- The 9 meV/atom frame-0 absolute-energy offset (§3) is disclosed; it is an absolute-energy bf16
  artifact, not a dynamics error.
- The video is one honest on-device run; the HUD is composited from the real per-step log and the
  real g(r)/MSD of that trajectory. The loop is a boomerang (forward then reverse) purely so it
  seams cleanly; MD is not time-periodic.
