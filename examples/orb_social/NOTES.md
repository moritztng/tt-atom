# Orb-v3 materials-MD social demo — reproduction

Everything runs on **physical device 0** (single Blackhole card) with real Orb-v3 weights. No
reference env is needed at MD time: `orb-v3-conservative-inf-omat`'s weight bundle is
system-independent, and for a monatomic crystal the node feature is identical per atom, so it's
tiled to any supercell size (see `orb_md_device.py` docstring). The reference env
(`~/.ttatom_run/refenv`, `orb-models` 3.29) is only used for the CPU-reference parity and rendering.

Two environments:
- device : `~/.ttatom_run/env` (py3.10, ttnn + tt_atom) — MD, on-device parity
- refenv : `~/.ttatom_run/refenv` (py3.11, orb_models + ovito) — CPU reference, lattice, render

    export TT_METAL_HOME=/home/moritz/tt-metal
    WT=/home/moritz/.coworker/wt/tt-atom-orb-social-post-polish     # tt-atom checkout (Orb port)

## 1. On-device MD — writes the trajectory + per-step energy/temperature log

    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python orb_md_device.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --nx 3 --ny 3 --nz 3 --steps 1500 --temp 900 --dt 1.0 --friction 0.02 --save-every 6 \
        --seed 1 --out si216_final.extxyz --log-csv md_series.csv

Prints the throughput summary (atoms/edges, warm ms/step, steps/s, eV/atom). Forces are Orb-v3
conservative F = −dE/dpos on-device; the neighbour list is frozen at t=0 (valid for a solid) and
the forward+backward is trace-captured once and replayed. Scale knobs: `--nx/ny/nz` (512 atoms =
4 4 4), `--temp`, `--element`/`--a`.

## 2. Verification (correctness — see VERIFICATION.md)

    # (a) MD stability plot: temperature + potential energy vs time
    ~/.ttatom_run/env/bin/python plot_stability.py --csv md_series.csv --out energy_temp.png \
        --target 900 --equil-fs 200

    # (b) parity vs orb-models CPU reference on real MD frames
    ~/.ttatom_run/refenv/bin/python orb_parity.py ref \
        --traj si216_final.extxyz --frames 0 125 251 --out parity_ref.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=$WT ~/.ttatom_run/env/bin/python orb_parity.py device \
        --traj si216_final.extxyz --frames 0 125 251 \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz --ref parity_ref.npz --out parity.json

    # (c) lattice-constant / energy sanity (CPU reference)
    ~/.ttatom_run/refenv/bin/python lattice_check.py | tee lattice_check.txt

## 3. Render the trajectory — professional OVITO (Tachyon) MP4 + GIF

    ~/.ttatom_run/refenv/bin/python render_ovito.py --traj si216_final.extxyz --out orb_si_md \
        --px 720 --fps 30 --nframes 130 --spin 150 --gif-px 340 --gif-fps 12

Shaded spheres (Jmol Si colour), diamond bond network, periodic cell box, ambient occlusion +
shadows; coordinates wrapped into the cell (PBC), bonds not drawn across periodic faces; gentle
turntable, boomerang loop, composited caption. MP4 720² (posting), GIF downscaled (preview).
The GIF may need a final size pass from the MP4 (dark AO background inflates the palette):

    ffmpeg -y -i orb_si_md.mp4 -vf "fps=12,scale=340:-1:flags=lanczos,palettegen=max_colors=72:stats_mode=diff" /tmp/pal.png
    ffmpeg -y -i orb_si_md.mp4 -i /tmp/pal.png -lavfi "fps=12,scale=340:-1:flags=lanczos [x];[x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" orb_si_md.gif

## Artifacts
- `orb_si_md.mp4` — 720², 30 fps, 7.4 MB   (primary, for the post)
- `orb_si_md.gif` — 340px, 12 fps, 6 MB    (README/preview)
- `si216_final.extxyz` — the on-device MD trajectory (252 frames)
- `md_series.csv` — per-step energy/temperature; `energy_temp.png` — stability plot
- `parity.json`, `parity_ref.npz` — on-device vs orb-models CPU-reference parity
- `lattice_check.txt` — Orb-v3 Si lattice-constant / energy sanity (CPU reference)
- `orb_md_device.py`, `orb_parity.py`, `plot_stability.py`, `lattice_check.py`, `render_ovito.py`
- `DRAFT_POST.md` — post text + measured numbers; `VERIFICATION.md` — full evidence + caveats
