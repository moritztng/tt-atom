"""Render the on-device Si melt to a looping MP4 + GIF with a live, frame-synced scientist HUD.

The 3D render is OVITO (Tachyon): shaded Jmol-Si spheres, the diamond bond network, ambient
occlusion + shadows, and — deliberately restored — the periodic simulation-cell wireframe (a
periodic MD system is shown with its cell box; it reads the boundary and the scale). A gentle
turntable plays the melt; coordinates are wrapped into the cell so nothing streaks across a face.

Layered alongside the render, evolving in lock-step with each frame:

* a live temperature + total-energy trace (T rising through Si's 1687 K melting point, the total
  energy rising as the thermostat heats the crystal then going flat the instant the run switches
  to NVE — the energy-conservation signature), with a "now" cursor advancing frame by frame;
* the radial distribution function g(r) of the current frame, morphing from sharp crystalline
  peaks to the broad liquid envelope as the lattice disorders.

A compact caption carries the static headline numbers: the on-device throughput (steps/s, ns/day,
atom count, Blackhole p150), the NVE energy-conservation drift, and the force/energy parity vs
orb-models. The video is a boomerang loop so it seams cleanly (MD is not time-periodic).

    ~/.ttatom_run/refenv/bin/python examples/orb_social/render_melt_hud.py \
        --traj si_melt.extxyz --metrics melt_metrics.npz \
        --steps-s 20.8 --ns-day 0.90 --atoms 216 --perf-dollar 40 --gpu-speedup 1.74 \
        --drift "1.4 meV/atom·ps (NVE, 900 K)" --parity "F PCC 0.9999, ΔE 1.4 meV/atom" \
        --out orb_si_melt
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import WrapPeriodicImagesModifier, CreateBondsModifier

JMOL_SI = (0.941, 0.784, 0.627)
BG = (12, 15, 21)
PANEL_BG = "#0d1117"
GRID = "#243040"
T_COLOR = "#e07b5a"
E_COLOR = "#6fa8dc"
RDF_COLOR = "#8fd0a8"
NOW_COLOR = "#f0f3f8"
TEXT = "#e6edf3"
DIM = "#8b9aae"


def _font(sz, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    try:
        return ImageFont.truetype(p, sz)
    except OSError:
        return ImageFont.load_default()


def _interp_rdf(metrics, t_now):
    """Linear interpolation of the g(r) series to the current frame time."""
    g = metrics["g_series"]
    gt = np.asarray(metrics["g_time"], float)
    if t_now <= gt[0]:
        return g[0]
    if t_now >= gt[-1]:
        return g[-1]
    j = int(np.searchsorted(gt, t_now) - 1)
    a = (t_now - gt[j]) / max(gt[j + 1] - gt[j], 1e-9)
    return (1 - a) * g[j] + a * g[j + 1]


def _trace_panel(metrics, t_now, t_end, w, h, t_melt=1687.0):
    """Temperature (left axis) + total energy per atom (right axis) vs time, drawn up to t_now."""
    t = np.asarray(metrics["time_fs"], float)
    T = np.asarray(metrics["temp_K"], float)
    et = np.asarray(metrics["etot_ev_atom"], float)
    reg = metrics["regime"]
    nve0 = float(metrics["nve"]) if "nve" in metrics else None

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    m = t <= t_now
    ax.plot(t[m], T[m], color=T_COLOR, lw=1.8)
    ax.axhline(t_melt, color="#5a4a3a", ls="--", lw=1.0)
    ax.text(t_end * 0.99, t_melt + 40, "Si $T_m$ 1687 K", color="#a98a6a", fontsize=8, ha="right")
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, max(3000, T.max() * 1.05))
    ax.set_ylabel("temperature (K)", color=T_COLOR, fontsize=9)
    ax.tick_params(axis="y", labelcolor=T_COLOR, labelsize=8, colors=DIM)
    ax.tick_params(axis="x", labelsize=8, colors=DIM)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(alpha=0.18, color=GRID)

    ax2 = ax.twinx()
    ax2.plot(t[m], et[m], color=E_COLOR, lw=1.6)
    ax2.set_xlim(0, t_end)
    et_lo, et_hi = float(et.min()) - 0.05, float(et.max()) + 0.05
    ax2.set_ylim(et_lo, et_hi)
    ax2.set_ylabel("$E_{tot}$ (eV/atom)", color=E_COLOR, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=E_COLOR, labelsize=8, colors=DIM)
    for s in ax2.spines.values():
        s.set_color(GRID)
    if nve0 is not None and nve0 < t_end:
        ax.axvline(nve0, color="#3a4a5a", ls=":", lw=1.0)
        ax.text(nve0 + 6, ax.get_ylim()[1] * 0.94, "NVE", color="#7a8aa0", fontsize=7)
    if t_now > 0:
        ax.axvline(t_now, color=NOW_COLOR, lw=1.0, alpha=0.8)
    ax.set_xlabel("time (fs)", color=DIM, fontsize=9)
    ax.set_title("heating through the melt — T rises, $E_{tot}$ flat under NVE",
                 color=TEXT, fontsize=10, pad=6)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, [1, 2, 3, 0]]            # argb -> rgba
    plt.close(fig)
    return img


def _rdf_panel(metrics, t_now, w, h):
    """g(r) for the current frame, interpolated from the analysed sweep."""
    r = np.asarray(metrics["r"], float)
    g = _interp_rdf(metrics, t_now)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(r, g, color=RDF_COLOR, lw=1.8)
    ax.fill_between(r, 0, g, color=RDF_COLOR, alpha=0.12)
    ax.axhline(1.0, color=GRID, lw=0.8)
    ax.set_xlim(0, float(r[-1]))
    ax.set_ylim(0, max(8.0, float(np.asarray(metrics["g_series"][0]).max()) * 0.35))
    ax.set_xlabel("r (Å)", color=DIM, fontsize=9)
    ax.set_ylabel("g(r)", color=DIM, fontsize=9)
    ax.tick_params(labelsize=8, colors=DIM)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(alpha=0.18, color=GRID)
    state = "crystal" if t_now < float(metrics["t_melt_cross"]) else "liquid"
    ax.set_title(f"radial distribution g(r) — {state}", color=TEXT, fontsize=10, pad=6)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, [1, 2, 3, 0]]
    plt.close(fig)
    return img


def _caption_bar(w, h, title, lines):
    """Compact lower caption strip with the static headline numbers."""
    img = Image.new("RGBA", (w, h), (13, 17, 23, 255))
    d = ImageDraw.Draw(img)
    fb, fr = _font(int(h * 0.22), bold=True), _font(int(h * 0.165))
    d.text((12, 3), title, font=fb, fill=TEXT)
    y = int(h * 0.34)
    for ln in lines:
        d.text((12, y), ln, font=fr, fill=DIM)
        y += int(h * 0.175)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--metrics", required=True, help="melt_metrics.npz from analyze_melt.py")
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--canvas-w", type=int, default=1280)
    ap.add_argument("--canvas-h", type=int, default=720)
    ap.add_argument("--render-px", type=int, default=600, help="square 3D render size")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=110, help="rendered MD frames (subsampled)")
    ap.add_argument("--spin", type=float, default=120.0, help="turntable degrees over the play")
    ap.add_argument("--radius", type=float, default=0.72)
    ap.add_argument("--bond-cutoff", type=float, default=2.55)
    ap.add_argument("--gif-px", type=int, default=540)
    ap.add_argument("--gif-fps", type=int, default=18)
    ap.add_argument("--gif-colors", type=int, default=160)
    # headline caption numbers (from the real runs):
    ap.add_argument("--steps-s", type=float, default=20.8)
    ap.add_argument("--ns-day", type=float, default=0.90)
    ap.add_argument("--atoms", type=int, default=216)
    ap.add_argument("--drift", default="1.4 meV/atom/ps (NVE, 900 K)")
    ap.add_argument("--parity", default="F PCC 0.99996 vs orb-models")
    ap.add_argument("--perf-dollar", type=float, default=40.0, help="perf-per-dollar multiple vs H200-class GPU")
    ap.add_argument("--gpu-speedup", type=float, default=1.74, help="raw throughput multiple vs H200 at this system size")
    ap.add_argument("--workdir", default=os.path.expanduser("~/orb_melt_tmp"),
                    help="scratch dir for frames (must NOT be under a dotdir — ffmpeg is snap-confined)")
    args = ap.parse_args()

    metrics = np.load(args.metrics, allow_pickle=True)
    ftime = np.asarray(metrics["msd_time"], float)
    t_end = float(ftime[-1])

    # ---- OVITO scene: render, periodic cell box restored ----
    pl = import_file(args.traj)
    nsrc = pl.source.num_frames
    pl.modifiers.append(WrapPeriodicImagesModifier())

    def _drop_pbc(frame, data):
        data.cell_.pbc = (False, False, False)      # don't draw bonds across periodic faces
    pl.modifiers.append(_drop_pbc)

    bonds = CreateBondsModifier(cutoff=args.bond_cutoff)
    bonds.vis.width = 0.26
    bonds.vis.color = (0.42, 0.47, 0.54)
    pl.modifiers.append(bonds)
    pl.add_to_scene()

    st = pl.source.data.particles_.particle_types_.type_by_id_(1)
    st.radius = args.radius
    st.color = JMOL_SI
    cell_vis = pl.source.data.cell_.vis
    cell_vis.enabled = True
    cell_vis.line_width = 0.05                       # restored: thin, subtle periodic-cell wireframe
    cell_vis.rendering_color = (0.62, 0.68, 0.78)

    cell = np.array(pl.compute(0).cell[:3, :3])
    L = float(np.abs(cell).sum(axis=0).max())
    center = cell.sum(axis=0) * 0.5
    dist = L * 2.35
    tilt = math.radians(18.0)

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.fov = math.radians(30.0)
    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True,
                               ambient_occlusion_samples=16, antialiasing_samples=5)

    cw, ch = args.canvas_w, args.canvas_h
    rp = args.render_px
    panel_w = cw - rp
    trace_h = int(ch * 0.42)
    rdf_h = int(ch * 0.40)
    cap_h = ch - trace_h - rdf_h
    # per-frame time for each rendered frame (subsample the trajectory evenly)
    idx = np.linspace(0, nsrc - 1, args.nframes).round().astype(int)
    frame_times = np.asarray(metrics["msd_time"], float)[idx] if len(ftime) == nsrc else \
        np.linspace(0, t_end, args.nframes)

    os.makedirs(args.workdir, exist_ok=True)
    title = "Silicon melt — Orb-v3 MD"
    cap_lines = [
        f"{args.atoms}-atom Si  ·  {args.steps_s:.1f} steps/s  ·  {args.ns_day:.2f} ns/day  ·  Blackhole p150",
        f"~{args.perf_dollar}x perf/$ vs H200-class GPU  ({args.gpu_speedup}x faster, ~1/23 the cost)",
        f"energy conservation: {args.drift}",
        f"parity vs orb-models: {args.parity}",
    ]

    imgs = []
    with tempfile.TemporaryDirectory(dir=args.workdir) as td:
        for k, f in enumerate(idx):
            az = math.radians(-25.0 + args.spin * k / max(1, len(idx) - 1))
            eye = center + dist * np.array([math.cos(tilt) * math.sin(az),
                                            -math.cos(tilt) * math.cos(az),
                                            math.sin(tilt)])
            vp.camera_pos = tuple(eye)
            vp.camera_dir = tuple(center - eye)
            raw = f"{td}/r{k:04d}.png"
            vp.render_image(size=(rp, rp), filename=raw, frame=int(f),
                            renderer=renderer, alpha=True)
            render_img = Image.open(raw).convert("RGBA")
            bg = Image.new("RGBA", (rp, rp), BG + (255,))
            render_img = Image.alpha_composite(bg, render_img).convert("RGB")

            t_now = float(frame_times[k])
            tr = _trace_panel(metrics, t_now, t_end, panel_w, trace_h)
            rdf = _rdf_panel(metrics, t_now, panel_w, rdf_h)
            cap = _caption_bar(panel_w, cap_h, title, cap_lines)

            canvas = Image.new("RGB", (cw, ch), BG)
            canvas.paste(render_img, (0, (ch - rp) // 2))                  # square render, centred
            canvas.paste(Image.fromarray(tr[..., :3]), (rp, 0))            # matplotlib panels are opaque
            canvas.paste(Image.fromarray(rdf[..., :3]), (rp, trace_h))
            canvas.paste(cap, (rp, trace_h + rdf_h))
            imgs.append(canvas)
            if k % 15 == 0:
                print(f"  rendered {k+1}/{len(idx)}  t={t_now:.0f} fs", flush=True)

        loop = imgs + imgs[-2:0:-1]                  # boomerang
        for j, im in enumerate(loop):
            im.save(f"{td}/f{j:04d}.png")
        out = os.path.join(args.workdir, os.path.basename(args.out))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", f"{td}/f%04d.png",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", out + ".mp4"], check=True, capture_output=True)
        print(f"wrote {out}.mp4  ({len(loop)} frames, {args.fps} fps)")
        subprocess.run(
            ["ffmpeg", "-y", "-i", f"{td}/f%04d.png", "-vf",
             f"palettegen=max_colors={args.gif_colors}", f"{td}/pal.png"],
            check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", f"{td}/f%04d.png",
             "-i", f"{td}/pal.png", "-lavfi",
             f"fps={args.gif_fps},scale={args.gif_px}:-1:flags=lanczos [x]; "
             f"[x][1:v] paletteuse=dither=bayer:bayer_scale=4", out + ".gif"],
            check=True, capture_output=True)
        print(f"wrote {out}.gif  ({os.path.getsize(out + '.gif')/1e6:.1f} MB, {args.gif_px}px)")


if __name__ == "__main__":
    main()
