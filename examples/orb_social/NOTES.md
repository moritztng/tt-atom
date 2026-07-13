# Orb-v3 silicon melt on Tenstorrent — reproduction

Everything runs on **physical device 0** (single Blackhole p150 card) with real Orb-v3 weights.
Two environments, both on pc/moritz (the whole Orb stack is pc/moritz, not qb1/ttuser):

- device : `~/.ttatom_run/env` (ttnn + tt_atom) — the MD, the on-device parity
- refenv : `~/.ttatom_run/refenv` (orb_models 0.5.5 + ovito + matplotlib) — CPU reference, render

The weight bundle (`~/.ttatom_run/goldens_real/si_supercell_orb.npz`, from
`tests/gen_golden_orb.py --ckpt conservative-inf-omat`) is system-independent; for a monatomic
crystal the atomic-number node feature is one row tiled to any supercell size — no new golden
needed for the melt.

    WT=/home/moritz/.coworker/wt/tt-atom-orb-social-video-metrics
    SRC=$WT/examples/orb_social
    W=~/.ttatom_run/goldens_real/si_supercell_orb.npz
    TMP=~/orb_melt_tmp        # NOT under a dotdir — ffmpeg is snap-confined

## 1. On-device melt — trajectory + per-step energy/temperature log

NVT Langevin ramps T 300 → 2800 K (the bath keeps feeding energy as the lattice absorbs the latent
heat of fusion, so the crystal reliably disorders instead of refreezing), then switches to NVE for
an energy-conservation tail. The neighbour list rebuilds whenever any atom has moved > 1.5 Å
(`--skin`), so the topology stays correct as the solid becomes a liquid — the frozen-topology
trick that's exact for a solid is wrong once atoms diffuse, so the melt calculator rebuilds it.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python $SRC/orb_melt_md.py \
        --weights $W --ramp-steps 1800 --nve-steps 1000 --t-start 300 --t-end 2800 \
        --dt 0.5 --save-every 4 --out $TMP/si_melt.extxyz --log-csv $TMP/md_melt.csv

Prints the throughput summary (steps/s, ns/day, atom-steps/s) and the NVE-tail energy drift.

## 2. Clean solid NVE drift (the headline credibility number)

The melt's NVE tail is a 2000 K liquid (drift ~15 meV/atom/ps). The UMA-analog credibility number
is a clean equilibrated-solid NVE: 400 steps of NVT at 900 K to settle, then 1200 steps of NVE.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python $SRC/orb_melt_md.py \
        --weights $W --ramp-steps 400 --nve-steps 1200 --t-start 900 --t-end 900 \
        --dt 0.5 --save-every 20 --out $TMP/si_solid_nve.extxyz --log-csv $TMP/md_solid_nve.csv

→ NVE drift **1.4 meV/atom/ps** (900 K equilibrated solid).

## 3. Analysis — T/E traces, g(r) sweep, MSD

    ~/.ttatom_run/env/bin/python $SRC/analyze_melt.py \
        --csv $TMP/md_melt.csv --traj $TMP/si_melt.extxyz --solid-csv $TMP/md_solid_nve.csv \
        --save-every 4 --dt 0.5 --out $TMP/melt_metrics.npz --summary $TMP/melt_summary.json

Writes `melt_metrics.npz` (per-step T/Epot/Ekin/Etot, the g(r) series, the MSD, the drift numbers)
and `melt_summary.json` (the headlines).

## 4. Parity vs orb-models CPU reference (on real melt frames)

    ~/.ttatom_run/refenv/bin/python $SRC/orb_melt_parity.py ref \
        --traj $TMP/si_melt.extxyz --frames 0 350 700 --out $TMP/parity_ref.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python $SRC/orb_melt_parity.py device \
        --traj $TMP/si_melt.extxyz --frames 0 350 700 --weights $W \
        --ref $TMP/parity_ref.npz --out $TMP/parity.json --md-dir $SRC

→ forces PCC 0.99995–0.99998, ΔE 1.2–1.4 meV/atom on the thermal-solid and liquid frames.

## 5. Render — OVITO 3D melt + live, frame-synced scientist HUD

    ~/.ttatom_run/refenv/bin/python $SRC/render_melt_hud.py \
        --traj $TMP/si_melt.extxyz --metrics $TMP/melt_metrics.npz \
        --steps-s 20.8 --ns-day 0.90 --atoms 216 --perf-dollar 40 --gpu-speedup 1.74 \
        --drift "1.4 meV/atom·ps (NVE, 900 K)" --parity "F PCC 0.9999, ΔE 1.4 meV/atom" \
        --out $TMP/orb_si_melt

Left: shaded Jmol-Si spheres, the diamond bond network, **the periodic-cell wireframe
(restored)**, AO + shadows, gentle turntable. Right: a live T + E_tot trace (T rising through
1687 K, E_tot flat under NVE) and the g(r) of the current frame morphing crystal → liquid, both
advancing in sync with the atoms; a compact caption carries throughput, conservation, parity.
Boomerang loop. MP4 1280×720 (post), GIF 540 px (preview). The script renders into `$TMP`
(non-dotdir) because ffmpeg is snap-confined; copy the finals to `~/.coworker/artifacts/orb-social/`.

## Artifacts (in ~/.coworker/artifacts/orb-social/)
- `orb_si_melt.mp4` — 1280×720, 30 fps, ~6.1 MB (primary, for the post)
- `orb_si_melt.gif` — 540 px, 18 fps, ~5.8 MB (README/preview)
- `si_melt.extxyz` — the on-device melt trajectory (703 frames)
- `md_melt.csv`, `md_solid_nve.csv` — per-step energy/temperature logs
- `melt_metrics.npz`, `melt_summary.json` — T/E traces, g(r) series, MSD, drift numbers
- `parity.json`, `parity_ref.npz` — on-device vs orb-models parity on melt frames
- `DRAFT_POST.md`, `VERIFICATION.md` — post text + full evidence
- `prev_solid_demo/` — the earlier 900 K solid-vibration demo, archived
