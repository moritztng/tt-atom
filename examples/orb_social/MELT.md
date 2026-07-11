# Orb-v3 silicon-melt social demo — artifacts + reproduction

A silicon diamond crystal heated on **one Tenstorrent Blackhole card** with real Orb-v3
conservative forces until it melts into a flowing liquid. Propose-only (not posted).

## Deliverables (this folder)
- `orb_si_melt.mp4` — 720², 30 fps, 2.5 MB  (primary, for the post)
- `orb_si_melt.gif` — 420 px, 16 fps, 6.3 MB (preview)
- `melt_verify.png` — T-ramp + E/MSD + solid-vs-liquid g(r) (the melt proof)
- `melt.extxyz` — the on-device MD trajectory (252 frames); `melt.csv` — per-step log
- `parity.json`, `parity_ref.npz` — on-device vs orb-models CPU-reference parity
- `melt_run.log` — full MD stdout (throughput summary at the end)

## System / run
- **Si diamond 3×3×3 cubic + 10-atom spherical void → 206 atoms.** The void is a realistic
  nucleation site: homogeneous melting under 3-D PBC superheats past the true T_m, so seeding a
  vacancy cluster lets melting initiate at the defect. Presented honestly as "heated until it
  melts" — a small, fast-ramped PBC cell is not a calibrated melting-point measurement.
- **Ensemble:** NVT Langevin, 1 fs step, friction 0.02 fs⁻¹, temperature ramp **300 → 3000 K over
  2000 steps (2.0 ps)**, no hold.
- **Neighbour list:** Verlet, 1.5 Å skin, rebuilt from geometry when an atom drifts > skin/2 (42
  rebuilds over the run); a diffusing liquid needs a live list, unlike the frozen-list solid demo.

## Throughput (device 0, single Blackhole card)
- **89.3 ms / MD step** median (untraced fwd + reverse-VJP; warm, fixed-list between rebuilds)
  ⇒ **≈11.2 MD steps/s**. 17 218 periodic edges (≤ r_max+skin). Full run wall **338 s (5.6 min)**
  incl. device open + 42 list-rebuild recompiles. No trace: Orb's force path is host-autograd/
  dispatch-bound, so trace replay gives no speedup for the dynamic-topology liquid.

## Melt evidence (melt_verify.png / plot_melt.py)
- **MSD:** flat ≈0 through the solid, then lifts off sharply at ~1.5 ps to **2.45 Å²** final-segment
  (> 1.5 ⇒ diffusive/liquid). MSD lift-off = the melt.
- **g(r):** first-peak height collapses **20.15 → 1.82**; the sharp diamond-lattice peaks (and the
  split second shell) wash out into the broad single-peak envelope of a disordered liquid.
- **E_pot:** rises smoothly with T, no NaN/blow-up; **T tracks the setpoint ramp** throughout.

## Parity — the trajectory is the real Orb-v3 (parity.json)
On-device vs `orb-models` CPU reference (`orb-v3-conservative-inf-omat`, float32-high), same frames:

| frame | state | ΔE (meV/atom) | force PCC | max |ΔF| (eV/Å) | ref |F|max (eV/Å) |
|------:|-------|--------------:|----------:|-----------------:|-----------------:|
| 0   | perfect lattice | 0.68 | 0.9964* | 0.025 | 0.38 |
| 160 | thermal solid   | 0.18 | **0.999973** | 0.046 | 4.57 |
| 251 | **liquid**      | 1.60 | **0.999947** | 0.051 | 5.92 |

\*Frame 0 forces are near-zero (ref |F|max 0.38), so its PCC is dominated by noise. On the dynamics
frames — including the **fully liquid** one — forces match at **PCC 0.99995**, max error ≈0.05 eV/Å
≈ 0.9 % of the ~6 eV/Å peak, energy to ≤1.6 meV/atom. This is within the port's established
conservative-force bar. The melt is the real model, not a degraded approximation.

## Reproduce
Environments (py3.10 on qb1/tt-quietbox): device `~/.ttatom_run/env` (ttnn+tt_atom);
refenv `~/.ttatom_run/refenv` (orb-models 0.5.5 + ovito). Weight bundle
`~/.ttatom_run/goldens_real/si_supercell_orb.npz` (see the solid demo's NOTES.md §0). With
`WT=<tt-atom checkout>`:

    # 1. melt MD  (writes melt.extxyz + melt.csv)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python orb_melt_device.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz --nx 3 --ny 3 --nz 3 \
        --void-r 3.3 --steps 2000 --dt 1.0 --t0 300 --t1 3000 --skin 1.5 --save-every 8 \
        --seed 1 --out melt.extxyz --log-csv melt.csv
    # 2. melt verification figure
    ~/.ttatom_run/env/bin/python plot_melt.py --csv melt.csv --traj melt.extxyz --out melt_verify.png
    # 3. parity (CPU ref, then device)
    ~/.ttatom_run/refenv/bin/python orb_parity.py ref --traj melt.extxyz --frames 0 160 251 --out parity_ref.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python orb_parity.py device \
        --traj melt.extxyz --frames 0 160 251 --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --ref parity_ref.npz --out parity.json --md-dir <dir with orb_md_device.py>
    # 4. render (liquid --wrap mode: fold into cell, subtle box, centering shift kills face-flicker)
    ~/.ttatom_run/refenv/bin/python render_ovito.py --traj melt.extxyz --out orb_si_melt --wrap \
        --px 720 --fps 30 --nframes 130 --spin 120 --gif-px 420 --gif-fps 16
