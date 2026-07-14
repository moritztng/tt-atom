# Orb-v3 silicon melt on Tenstorrent — reproduction (qb2, self-contained)

A 216-atom diamond-cubic Si supercell heated through its melting point into the liquid, run
on-device with the **real Orb-v3 (`conservative-inf-omat`) weights** on a single Blackhole p150
(physical card 1 of tt-quietbox2), rendered to a clean MP4/GIF, and verified against the
orb-models CPU reference.

## Environments (qb2, self-contained — qb1's render env is offline)

- **device** : `~/.ttatom_qb2_fanout/env` (source-built ttnn + `tt_atom` from this worktree via
  `PYTHONPATH`). qb2's firmware presents each p150 as a 1x1 P300 mesh, so device open needs the
  fabric descriptor. Every device command is:

      E=~/.ttatom_qb2_fanout
      WT=<this worktree>
      MG=$E/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
      TT_VISIBLE_DEVICES=1 TT_METAL_HOME=$E/tt-metal TT_MESH_GRAPH_DESC_PATH=$MG \
          PYTHONPATH=$WT $E/env/bin/python ...

- **refenv** : `~/.coworker/orb-refenv` (orb-models 0.5.5, ase, matplotlib, ovito 3.15; numpy>=2).
  Used for the weight export, the CPU parity reference, the analysis, and the render (no device).

- **weights** : golden bundle `~/orb_goldens/si_supercell_orb.npz`
  (`tests/gen_golden_orb.py --ckpt conservative-inf-omat --system supercell`). Carries the
  weights plus the host-side `node_feat` row the MD calculator tiles to N (monatomic Si).

- **scratch / artifacts** : device/render scratch in `~/orb_melt_tmp` (NOT under a dotdir —
  ffmpeg is snap-confined). Finals copied to `~/.coworker/artifacts/orb-social-melt/`.

## 1. On-device melt — trajectory + per-step energy/temperature log

NVT Langevin ramps 300 -> 2800 K (the bath feeds the latent heat so the lattice reliably
disorders instead of refreezing), then an NVE tail for the energy-conservation signature. The
neighbour list rebuilds whenever any atom moves > `--skin` (1.5 A), so the graph topology stays
correct as the solid becomes a liquid.

    RUN --weights $W --ramp-steps 1800 --nve-steps 1000 --t-start 300 --t-end 2800 \
        --dt 0.5 --save-every 4 --out $TMP/si_melt.extxyz --log-csv $TMP/md_melt.csv

-> 703 frames, 9 neighbour-list rebuilds, 23.4 MD steps/s warm (42.7 ms/step), 1.01 ns/day.

## 2. Clean equilibrated-solid NVE drift (the credibility number)

    RUN --weights $W --ramp-steps 400 --nve-steps 1200 --t-start 900 --t-end 900 \
        --dt 0.5 --save-every 20 --out $TMP/si_solid_nve.extxyz --log-csv $TMP/md_solid_nve.csv

-> NVE drift **1.4 meV/atom/ps** (900 K equilibrated solid) — clears the ~1 meV/atom/ps bar.

## 3. Analysis — T/E traces, g(r) sweep, MSD

    <refenv> analyze_melt.py --csv $TMP/md_melt.csv --traj $TMP/si_melt.extxyz \
        --solid-csv $TMP/md_solid_nve.csv --save-every 4 --dt 0.5 \
        --out $TMP/melt_metrics.npz --summary $TMP/melt_summary.json

Writes the per-step T/E, the g(r) series, the MSD, and the drift numbers. (Fixed here: the MSD
diffusion-coefficient unit conversion was `1e-10` instead of `1e-5` for A^2/fs -> m^2/s, which
made D read ~1e-13 m^2/s; corrected D ~ 8.3e-9 m^2/s is the right order for liquid Si. Also made
per-frame temperature an interpolation so the last frame's label is not `nan`.)

## 4. Parity vs orb-models CPU reference (on real melt frames)

    <refenv> orb_melt_parity.py ref --traj $TMP/si_melt.extxyz --frames 0 350 700 --out $TMP/parity_ref.npz
    RUN orb_melt_parity.py device --traj $TMP/si_melt.extxyz --frames 0 350 700 \
        --weights $W --ref $TMP/parity_ref.npz --out $TMP/parity.json --md-dir <src>

-> thermal-solid frame 350: force PCC 0.99998, dE 1.39 meV/atom; liquid frame 700: PCC 0.99995,
dE 1.24 meV/atom. The on-device Orb-v3 provably matches the CPU model in the liquid, not just the
solid.

## 5. Charts figure — the physics, for scientific scrutiny (separate from the video)

    <refenv> plot_melt_charts.py --metrics $TMP/melt_metrics.npz --out $TMP/melt_charts.png

4 panels: T-ramp crossing T_m, E_tot conservation (rises under NVT, flat under NVE), MSD (flat
solid -> rising diffusion), g(r) (sharp crystal -> broad liquid). Verified by eye. The g(r) liquid
reference is the hottest frame (~2311 K, clearly above T_m), not the cooled NVE tail.

## 6. Render — clean premium video with a synced physics side-card

    <refenv> render_melt_video.py --traj $TMP/si_melt.extxyz --metrics $TMP/melt_metrics.npz \
        --out orb_si_melt --workdir $TMP

1920x1080 MP4 + 720px GIF: a square 3D scene on the left, a synced physics side-card on the right
(T-ramp, MSD, g(r)) whose cursors/curves advance in lockstep with the melt. The decisions:

4. **The jump — final approach (Moritz's ADDENDUM 3): unwrapped continuous coordinates, no box,
   no tiling.** ADDENDUM 2 tiled the cell 3x3x3; that draws image atoms *outside* the cell that
   pop in/out as atoms cross faces — still a flicker. So tiling is abandoned. Instead we unwrap:
   for each atom we accumulate periodic images across the trajectory (minimum-image on each
   frame-to-frame step), never re-wrapping and never tiling, then remove the per-frame centre of
   mass so the cloud stays centred. Over the ~1.4 ps run each atom moves in small continuous steps
   and the cloud stays compact (max radius 13.6 A, inside the cell half-diagonal 14.1 A), so it
   neither teleports nor flies apart. The cell box is dropped entirely — with continuous
   coordinates there is nothing to clip against and no "atoms outside the box".
   *Hard verification (quantitative, `--verify-only`):* max per-atom displacement between every
   pair of consecutive rendered frames = **0.288 A** (mean 0.080), well under 1 A and far below a
   box length (16.29 A) — no jump. Atom count = **216 every frame** (no ghost atoms). Confirmed by
   eye on consecutive hot-liquid frames: near-identical, no teleport, no pop-in/out.
1. **Framing** — camera distance is derived from the unwrapped cloud's max extent over the whole
   clip (+ margin), constant at every timestep and every turntable angle. Verified by eye on
   first/mid/last frames: nothing clipped.
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
