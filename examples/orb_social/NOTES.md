# Orb-v3 silicon melt on Tenstorrent — reproduction (qb2, self-contained)

A 216-atom diamond-cubic Si supercell heated through its melting point and **held in the liquid
for 6 ps** so it visibly diffuses/flows, run on-device with the **real Orb-v3
(`conservative-inf-omat`) weights** on a single Blackhole p150 of tt-quietbox2, rendered to a
clean MP4/GIF, and verified against the orb-models CPU reference.

## Environments (qb2, self-contained — qb1's render env is offline)

- **device** : `~/.ttatom_qb2_fanout/env` (source-built ttnn + `tt_atom` from this worktree via
  `PYTHONPATH`). qb2's firmware presents each p150 as a 1x1 P300 mesh, so device open needs the
  fabric descriptor. Every device command is:

      E=~/.ttatom_qb2_fanout
      WT=<this worktree>
      MG=$E/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
      TT_VISIBLE_DEVICES=0 TT_METAL_HOME=$E/tt-metal TT_MESH_GRAPH_DESC_PATH=$MG \
          PYTHONPATH=$WT $E/env/bin/python ...

- **refenv** : `~/.coworker/orb-refenv` (orb-models 0.5.5, ase, matplotlib, ovito 3.15; numpy>=2).
  Used for the weight export, the CPU parity reference, the analysis, and the render (no device).

- **weights** : golden bundle `~/orb_goldens/si_supercell_orb.npz`
  (`tests/gen_golden_orb.py --ckpt conservative-inf-omat --system supercell`). Carries the
  weights plus the host-side `node_feat` row the MD calculator tiles to N (monatomic Si).

- **scratch / artifacts** : device/render scratch in `~/orb_melt_tmp` (NOT under a dotdir —
  ffmpeg is snap-confined). Finals copied to `~/.coworker/artifacts/orb-social-melt/`.

## 1. On-device melt — trajectory + per-step energy/temperature log

NVT Langevin ramps 300 -> 2200 K (the bath feeds the latent heat so the lattice reliably
disorders instead of refreezing), then **holds at a constant 2200 K for 6 ps** — the liquid
diffuses over this window, which is what makes the render actually flow (Moritz's ADDENDUM 4) —
then a short NVE tail for the energy-conservation signature. `--hold-steps` / `--t-hold` add the
constant-T liquid phase between the ramp and the NVE tail. The neighbour list rebuilds whenever any
atom moves > `--skin` (1.5 A), so the graph topology stays correct as the solid becomes a liquid.

    RUN --weights $W --ramp-steps 1600 --hold-steps 12000 --nve-steps 800 \
        --t-start 300 --t-end 2200 --t-hold 2200 \
        --dt 0.5 --save-every 4 --out $TMP/si_melt.extxyz --log-csv $TMP/md_melt.csv

-> 3604 frames (7.2 ps), 80 neighbour-list rebuilds, 23.4 MD steps/s warm (42.7 ms/step),
1.01 ns/day.

## 2. Clean equilibrated-solid NVE drift (the credibility number)

    RUN --weights $W --ramp-steps 400 --nve-steps 1200 --t-start 900 --t-end 900 \
        --dt 0.5 --save-every 20 --out $TMP/si_solid_nve.extxyz --log-csv $TMP/md_solid_nve.csv

-> NVE drift **1.4 meV/atom/ps** (900 K equilibrated solid) — clears the ~1 meV/atom/ps bar.

## 3. Analysis — T/E traces, g(r) sweep, MSD

    <refenv> analyze_melt.py --csv $TMP/md_melt.csv --traj $TMP/si_melt.extxyz \
        --solid-csv $TMP/md_solid_nve.csv --save-every 4 --dt 0.5 \
        --out $TMP/melt_metrics.npz --summary $TMP/melt_summary.json

Writes the per-step T/E, the g(r) series, the MSD, and the drift numbers. Over the 6 ps liquid the
MSD rises linearly to ~47 A^2, giving **D ~ 12.1e-9 m^2/s** (right order for liquid Si above T_m,
labelled indicative). The analysis spans the full 7.2 ps run.

## 4. Parity vs orb-models CPU reference (on real melt frames)

    <refenv> orb_melt_parity.py ref --traj $TMP/si_melt.extxyz --frames 0 1200 3600 --out $TMP/parity_ref.npz
    RUN orb_melt_parity.py device --traj $TMP/si_melt.extxyz --frames 0 1200 3600 \
        --weights $W --ref $TMP/parity_ref.npz --out $TMP/parity.json --md-dir <src>

-> liquid frame 1200 (~2.4 ps): force PCC 0.99993, dE 0.45 meV/atom; late-liquid frame 3600
(~7.2 ps): PCC 0.99983, dE 0.16 meV/atom (real forces up to ~5 eV/A). The on-device Orb-v3
provably matches the CPU model deep in the liquid, not just the solid. Frame 0 (perfect lattice)
is noise-vs-noise (forces ~0 by symmetry) with a ~9 meV/atom bf16 absolute-energy offset.

## 5. Charts figure — the physics, for scientific scrutiny (separate from the video)

    <refenv> plot_melt_charts.py --metrics $TMP/melt_metrics.npz --out $TMP/melt_charts.png

4 panels: T-ramp crossing T_m, E_tot conservation (rises under NVT, flat under NVE), MSD (flat
solid -> rising diffusion), g(r) (sharp crystal -> broad liquid). Verified by eye. The g(r) liquid
reference is the hottest frame (~2311 K, clearly above T_m), not the cooled NVE tail.

## 6. Render — clean premium video with a synced physics side-card

    <refenv> render_melt_video.py --traj $TMP/si_melt.extxyz --metrics $TMP/melt_metrics.npz \
        --out orb_si_melt --workdir $TMP --nframes 100 --fps 30 --tmax-fs 2400 --radius 1.0

1920x1080 MP4 (100 frames, 30 fps, 3.3 s) + 720px GIF: a square 3D scene on the left, a synced
physics side-card on the right (T-ramp, MSD, g(r)) whose cursors/curves advance in lockstep with
the melt. The decisions:

4. **The jump — unwrapped continuous coordinates, no box, no tiling (ADDENDUM 3), windowed to the
   cohesive interval (ADDENDUM 4).** ADDENDUM 2 tiled the cell 3x3x3; that draws image atoms
   *outside* the cell that pop in/out as atoms cross faces — a flicker. So tiling is abandoned.
   Instead we unwrap: for each atom we accumulate periodic images across the trajectory
   (minimum-image on each frame-to-frame step), never re-wrapping and never tiling, then remove the
   per-frame centre of mass so the cloud stays centred. There is no PBC teleport and no ghost
   atoms. **Tradeoff (the ADDENDUM 4 subtlety):** unwrapped coordinates only read cleanly while the
   diffusion cloud is smaller than ~one box. A 6 ps liquid at 2200 K diffuses further than that
   (cloud radius grows past the cell half-diagonal 14.1 A), so atoms that have crossed the removed
   periodic boundary would appear as detached "fliers" in vacuum. We therefore **window the 3D
   scene to `--tmax-fs 2400`** — the crystal, the melt, and ~1.8 ps of liquid churn — where the
   cloud stays cohesive (max radius 16.3 A, at most one atom at the droplet surface ~14.6 A vs the
   crystal's own corner atoms at 12.9 A, none in vacuum). The charts side-card still spans the full
   7.2 ps run (cursor sweeps the window). The cell box is dropped entirely — nothing to clip
   against, no "atoms outside the box".
   *Hard verification (`render_melt_video.py`):* max per-atom displacement between every pair of
   consecutive rendered frames = **0.962 A** (mean 0.267) — visible churn (3.3x the previous
   0.288 A, the ADDENDUM 4 "make it flow" lever: ~24 fs/frame, 0.72 ps of sim-time per playback
   second) yet far below the 8.14 A box/2 teleport threshold, so no atom jumps a box face. Atom
   count = **216 every frame** (no ghost/image atoms). Confirmed by eye on the mp4 first/mid/last:
   crystal -> melt -> cohesive liquid churn, no teleport, no pop-in/out, no fliers.
1. **Framing** — camera distance is derived from the unwrapped cloud's max extent over the
   *windowed* clip (+ margin), constant at every timestep and every turntable angle. Verified by
   eye on first/mid/last frames: nothing clipped.
2. **Minimal text** — one line only: `Orb-v3 . 216-atom Si . T = <live> K`, plus a small
   sub-line (state). No stats paragraph. The numbers live in the side-card and the docs. The live
   T is edge-corrected moving-average smoothed (raw instantaneous T of 216 atoms swings ±150 K
   frame-to-frame and reads as unstable).
3. **Atom colour** — a premium cool "silicon" blue with Tachyon ambient occlusion + shadows on a
   near-black canvas.

Charts included in the video (Moritz's ADDENDUM 2): T-ramp (heating through T_m), MSD (diffusion
onset), g(r) (crystal -> liquid). Energy conservation was dropped from the video panel (kept in
the standalone 4-panel figure) — the two structural signatures are the compelling ones. The loop
plays forward once with a short fade in/out so the restart is not a hard snap (MD is not
time-periodic; a boomerang would rewind the ramp, which reads as cooling — misleading).

**No GPU / NVIDIA / per-dollar comparison anywhere in the video** (per Moritz).

## Artifacts (~/.coworker/artifacts/orb-social-melt/)
- `orb_si_melt.mp4` / `orb_si_melt.gif` — the video
- `melt_charts.png` — the 4-panel physics figure
- `si_melt.extxyz`, `md_melt.csv`, `md_solid_nve.csv` — trajectory + per-step logs
- `melt_metrics.npz`, `melt_summary.json` — traces, g(r), MSD, drift, D
- `parity.json`, `parity_ref.npz` — on-device vs orb-models parity
- `NOTES.md`, `VERIFICATION.md`
