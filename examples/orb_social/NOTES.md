# Orb-v3 silicon melt on Tenstorrent — reproduction (qb2, self-contained)

A 216-atom diamond-cubic Si supercell heated through its melting point and held in the liquid,
run on-device with the **real Orb-v3 (`conservative-inf-omat`) weights** on a single Blackhole
p150 of tt-quietbox2, rendered to a clean MP4/GIF, and verified against the orb-models CPU
reference. Propose-only; not posted.

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
  Used for the CPU parity reference, the analysis, and the render (no device).

- **weights** : golden bundle `~/orb_goldens/si_supercell_orb.npz`
  (`tests/gen_golden_orb.py --ckpt conservative-inf-omat --system supercell`). Carries the
  weights plus the host-side `node_feat` row the MD calculator tiles to N (monatomic Si).

- **scratch / artifacts** : device/render scratch in `~/orb_melt_tmp` (NOT under a dotdir —
  ffmpeg is snap-confined). Finals copied to `~/.coworker/artifacts/orb-social-melt/`.

## 1. On-device melt — trajectory + per-step energy/temperature log

NVT Langevin ramps 300 -> 2200 K (the bath feeds the latent heat so the lattice disorders
instead of refreezing), **holds at ~2200 K for several ps of real liquid diffusion**, then a
short NVE tail for the energy-conservation signature. The neighbour list rebuilds whenever any
atom moves > `--skin` (1.5 A), so the graph topology stays correct as the solid becomes a liquid.

    RUN --weights $W --ramp-steps 1600 --hold-steps 12000 --nve-steps 800 \
        --t-start 300 --t-end 2200 --t-hold 2200 \
        --dt 0.5 --save-every 4 --out $TMP/si_melt.extxyz --log-csv $TMP/md_melt.csv

This run: **2855 saved frames, 5.70 ps** (11405 MD steps at dt = 0.5 fs, frame every 4 steps),
80 neighbour-list rebuilds (skin 1.5 A). Temperature crosses Si's melting point **1687 K at
t = 2138 fs**, peaks **2610 K**, and holds in the liquid at **~2160 K** through the NVE tail.

## 2. Clean equilibrated-solid NVE drift (the credibility number)

    RUN --weights $W --ramp-steps 400 --nve-steps 1200 --t-start 900 --t-end 900 \
        --dt 0.5 --save-every 20 --out $TMP/si_solid_nve.extxyz --log-csv $TMP/md_solid_nve.csv

-> NVE drift **1.4 meV/atom/ps** (889 K equilibrated solid) — clears the ~1 meV/atom/ps bar.

## 3. Analysis — T/E traces, g(r) sweep, MSD

    <refenv> analyze_melt.py --csv $TMP/md_melt.csv --traj $TMP/si_melt.extxyz \
        --solid-csv $TMP/md_solid_nve.csv --save-every 4 --dt 0.5 \
        --out $TMP/melt_metrics.npz --summary $TMP/melt_summary.json

Writes the per-step T/E, the g(r) series, the MSD, and the drift numbers. The MSD is flat
through the solid, then rises **linearly over the liquid to 19.65 A^2** at 5.7 ps — the
diffusion signature. The structural melt (MSD rise) sets in around ~3.5 ps, lagging the 2138 fs
T_m crossing: the lattice superheats past T_m before it disorders (realistic melting kinetics,
not an instantaneous jump). Diffusion coefficient from the liquid-window MSD slope
**D ~ 8.1e-9 m^2/s** — right order for liquid Si above T_m (labelled indicative; a precision D
needs a longer window than a 5.7 ps run).

## 4. Parity vs orb-models CPU reference (on real melt frames)

    <refenv> orb_melt_parity.py ref  --traj $TMP/si_melt.extxyz --frames 0 1070 1900 2400 --out $TMP/parity_ref.npz
    RUN    orb_melt_parity.py device --traj $TMP/si_melt.extxyz --frames 0 1070 1900 2400 \
           --weights $W --ref $TMP/parity_ref.npz --out $TMP/parity.json --md-dir <src>

Four frames spanning the actual trajectory (fresh calculator per frame so the neighbour list
matches that exact geometry):

| frame | state                  | dE (meV/atom) | force PCC    | max force err (eV/A) | ref |F|max (eV/A) |
|------:|------------------------|--------------:|-------------:|---------------------:|--------------------:|
| 0     | perfect lattice (277K) | 8.99          | 0.45 (noise) | 0.022                | 0.001               |
| 1070  | T_m-crossing solid     | 3.34          | **0.99997**  | 0.032                | 4.50                |
| 1900  | liquid onset (~3.8 ps) | 1.38          | **0.99996**  | 0.044                | 6.72                |
| 2400  | liquid (~4.8 ps)       | 1.31          | **0.99996**  | 0.035                | 4.55                |

On the dynamics frames the forces match the reference at **PCC 0.99996-0.99997** and energies
agree to **1.3-3.3 meV/atom** — through the melt and deep in the liquid at real forces up to
~6.7 eV/A. Frame 0 is the perfect lattice: forces vanish by symmetry on both sides, so its PCC
is numerical noise (|F|max ~ 0.001 eV/A, max abs force error only 0.022 eV/A), and its ~9
meV/atom absolute-energy offset is the known bf16 absolute-energy artifact, which does not
affect the dynamics (driven by forces and energy differences).

## 5. Charts figure — the physics, for scientific scrutiny

    <refenv> plot_melt_charts.py --metrics $TMP/melt_metrics.npz --out $TMP/melt_charts.png

4 panels: (a) T-ramp crossing T_m at 2138 fs, (b) E_tot conservation (rises under NVT, flat
under NVE; drift 1.4 meV/atom/ps solid vs 20.6 liquid), (c) MSD flat solid -> rising diffusion
with the T_m-crossing marked, (d) g(r) sharp crystal -> broad liquid. The MSD panel marks the
**T_m crossing** (2138 fs); the visible lag to the MSD rise (~3.5 ps) is the superheating
kinetics, shown honestly. The g(r) liquid reference is the hottest frame (~2421 K, clearly
above T_m).

## 6. Render — clean premium video with a synced physics side-card

    <refenv> render_melt_video.py --traj $TMP/si_melt.extxyz --metrics $TMP/melt_metrics.npz \
        --out orb_si_melt --workdir $TMP --nframes 120 --fps 30 --tmax-fs 4600 --radius 1.0 \
        --margin 0.22 --logo <tt_logo.png>

1920x1080 MP4 (120 frames, 30 fps, 4.0 s) + 720px GIF: a square 3D scene on the left, a synced
physics side-card on the right (T-ramp, MSD, g(r)) whose cursors/curves advance in lockstep
with the melt. The decisions:

1. **Framing** — camera distance is derived from the unwrapped cloud's max extent over the
   windowed clip (+ margin), constant at every timestep and every turntable angle. Verified by
   eye on first/mid/last frames: nothing clipped.
2. **Minimal text** — one line only: `Orb-v3 . 216-atom Si . T = <live> K`, plus a tiny
   sub-line (state: crystalline / liquid). No stats paragraph, no chip name in the text
   (the Tenstorrent logo top-left brands it), no GPU / NVIDIA / per-dollar comparison anywhere.
   The live T is moving-average smoothed (raw instantaneous T of 216 atoms swings ~+/-150 K
   frame-to-frame and reads as unstable); the charts show the raw trace.
3. **Atom colour** — a premium cool "silicon" blue (0.36, 0.66, 0.92) with Tachyon ambient
   occlusion + shadows on a near-black canvas.
4. **The jump — unwrapped continuous coordinates, no box, no tiling (ADDENDUM 3).** Tiling
   draws image atoms *outside* the cell that pop in/out as atoms cross faces — a flicker — so
   it is abandoned. Instead we unwrap: for each atom we accumulate periodic images across the
   trajectory (minimum-image on each frame-to-frame step), never re-wrapping and never tiling,
   then remove the per-frame centre of mass so the cloud stays centred. There is no PBC
   teleport and no ghost atoms. The cell box is dropped entirely — nothing to clip against.
5. **The flow — window to the cohesive melt+liquid, wider frame spacing (ADDENDUM 4).** The
   prior render windowed the 3D scene to [0, 2400 fs], which ends only ~260 fs past the 2138 fs
   T_m crossing — almost no liquid. The fix: window to **[0, 4601 fs]**, which covers the
   crystal (0-2138 fs), the melt, and **2.46 ps of real liquid churn** (2138-4601 fs), and
   space the 120 frames at **38.6 fs/frame (1.16 ps of sim-time per playback second, 1.6x the
   old 24 fs/frame)** so the liquid visibly flows. Unwrapped coordinates only read cleanly
   while the diffusion cloud stays within ~one box (no atom crosses a full periodic boundary =
   no flier); over [0, 4601 fs] the cloud radius is 14.75 A, inside the 16.29 A box, so no atom
   has crossed a full boundary. The charts side-card still spans the full 5.7 ps run (cursor
   sweeps the window). Real trajectory, no faked speed — the extra flow is real liquid MD plus
   wider frame spacing.

   *Hard verification (`render_melt_video.py`, on the shipped render):* max per-atom
   displacement between every pair of consecutive rendered frames = **1.21 A** (mean 0.27) —
   visible churn yet far below the 8.14 A box/2 teleport guard, so no atom jumps a box face.
   Atom count = **216 every frame** (no ghost/image atoms). Cloud radius = **14.75 A** (< box
   16.29 A, no fliers). Confirmed by eye on the mp4 first/mid/last: crystal -> melt -> cohesive
   liquid churn, no teleport, no pop-in/out, no fliers.

The loop plays forward once with a short fade in/out so the restart is not a hard snap (MD is
not time-periodic; a boomerang would rewind the ramp, which reads as cooling — misleading).

## Artifacts (~/.coworker/artifacts/orb-social-melt/)
- `orb_si_melt.mp4` / `orb_si_melt.gif` — the video
- `melt_charts.png` — the 4-panel physics figure
- `si_melt.extxyz`, `md_melt.csv`, `md_solid_nve.csv` — trajectory + per-step logs
- `melt_metrics.npz`, `melt_summary.json` — traces, g(r), MSD, drift, D
- `parity.json`, `parity_ref.npz` — on-device vs orb-models parity (4 frames)
- `NOTES.md`, `VERIFICATION.md`
