"""Render the on-device Si melt to a clean, premium MP4 + GIF: a pure 3D scene with one minimal
label. No stats paragraph, no side charts (those live in ``plot_melt_charts.py``).

Four decisions behind the look:

* **Framing is rotation-invariant.** The camera distance is derived from the bounding sphere of
  the simulation cell, so the whole cell (with a constant ~20% margin) stays fully in frame at
  every timestep and every turntable angle -- nothing is ever clipped.
* **No PBC "teleport".** A melt diffuses, and wrapping atoms back into the box makes them jump
  across a face between frames -- physically correct but visually jarring. Instead we render the
  cell surrounded by a shell of its periodic images: an atom leaving one face is matched by its
  image entering the opposite face, so the liquid reads continuous while remaining exactly the
  periodic system the MD integrated. The wireframe marks the primitive cell.
* **Premium shading.** A single cool "silicon" tone with Tachyon ambient occlusion + shadows.
* **Minimal text.** One line, bottom-left: model, system, live temperature. Nothing else.

    <refenv>/bin/python render_melt_video.py --traj si_melt.extxyz --metrics melt_metrics.npz \
        --out orb_si_melt
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import (WrapPeriodicImagesModifier, ReplicateModifier,
                             CreateBondsModifier, PythonScriptModifier)

# premium cool-silicon tone on a near-black canvas
SI_COLOR = (0.36, 0.66, 0.92)
BOND_COLOR = (0.34, 0.40, 0.50)
CELL_COLOR = (0.55, 0.62, 0.74)
BG = (10, 13, 18)


def _font(sz, bold=True):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    try:
        return ImageFont.truetype(p, sz)
    except OSError:
        return ImageFont.load_default()


def _label(canvas, text, sub):
    """One clean label line, bottom-left, on a soft rounded plate."""
    d = ImageDraw.Draw(canvas, "RGBA")
    W, H = canvas.size
    f = _font(int(H * 0.040)); fs = _font(int(H * 0.026), bold=False)
    pad = int(H * 0.028)
    tb = d.textbbox((0, 0), text, font=f)
    sb = d.textbbox((0, 0), sub, font=fs)
    tw = max(tb[2] - tb[0], sb[2] - sb[0])
    th = (tb[3] - tb[1]) + (sb[3] - sb[1]) + int(H * 0.018)
    x0, y0 = pad, H - pad - th - int(H * 0.03)
    d.rounded_rectangle([x0 - int(pad * 0.6), y0 - int(pad * 0.5),
                         x0 + tw + int(pad * 0.6), y0 + th + int(pad * 0.5)],
                        radius=int(H * 0.02), fill=(10, 13, 18, 150))
    d.text((x0, y0), text, font=f, fill=(233, 240, 247, 255))
    d.text((x0, y0 + (tb[3] - tb[1]) + int(H * 0.018)), sub, font=fs, fill=(150, 167, 184, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--w", type=int, default=1080)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=120)
    ap.add_argument("--spin", type=float, default=80.0, help="turntable degrees over the play")
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--bond-cutoff", type=float, default=2.7)
    ap.add_argument("--margin", type=float, default=0.16, help="fraction of empty frame around the cell")
    ap.add_argument("--shell", type=float, default=3.2, help="periodic-image shell thickness (A)")
    ap.add_argument("--fov-deg", type=float, default=24.0)
    ap.add_argument("--tilt-deg", type=float, default=16.0)
    ap.add_argument("--gif-px", type=int, default=560)
    ap.add_argument("--gif-fps", type=int, default=20)
    ap.add_argument("--gif-colors", type=int, default=180)
    ap.add_argument("--model", default="Orb-v3")
    ap.add_argument("--atoms", type=int, default=216)
    ap.add_argument("--element", default="Si")
    ap.add_argument("--preview", type=int, default=0, help="render only N probe frames (first/mid/last) as PNGs")
    ap.add_argument("--workdir", default=os.path.expanduser("~/orb_melt_tmp"))
    args = ap.parse_args()

    metrics = np.load(args.metrics, allow_pickle=True)
    ftime = np.asarray(metrics["msd_time"], float)
    temp = np.asarray(metrics["temp_K"], float)
    ttime = np.asarray(metrics["time_fs"], float)
    tcross = float(metrics["t_melt_cross"])
    t_end = float(ftime[-1])

    pl = import_file(args.traj)
    nsrc = pl.source.num_frames

    cell0 = np.array(pl.compute(0).cell[:3, :3])
    L = np.abs(np.diag(cell0))                     # orthorhombic Si supercell -> box lengths
    center = 1.5 * L                               # centre of the central cell after a 3x3x3 tiling
    halfwin = 0.5 * L + args.shell

    # 1) wrap into the primitive cell, 2) tile 3x3x3 so the central cell is fully surrounded,
    # 3) keep the central cell plus a thin periodic-image shell, and reset the drawn cell to the
    #    primitive box -- this is what removes the wrap "teleport" while staying periodic-exact.
    pl.modifiers.append(WrapPeriodicImagesModifier())
    pl.modifiers.append(ReplicateModifier(num_x=3, num_y=3, num_z=3))

    cx, cy, cz = center
    hx, hy, hz = halfwin
    Lx, Ly, Lz = L

    def shell_crop(frame, data):
        pos = np.asarray(data.particles.positions)
        drop = ((np.abs(pos[:, 0] - cx) > hx) | (np.abs(pos[:, 1] - cy) > hy) |
                (np.abs(pos[:, 2] - cz) > hz))
        data.particles_.delete_elements(drop)
        m = np.array([[Lx, 0, 0, cx - 0.5 * Lx],
                      [0, Ly, 0, cy - 0.5 * Ly],
                      [0, 0, Lz, cz - 0.5 * Lz]], dtype=float)
        data.cell_[...] = m
        data.cell_.pbc = (False, False, False)

    pl.modifiers.append(PythonScriptModifier(function=shell_crop))

    bonds = CreateBondsModifier(cutoff=args.bond_cutoff)
    bonds.vis.width = 0.24
    bonds.vis.color = BOND_COLOR
    pl.modifiers.append(bonds)
    pl.add_to_scene()

    st = pl.source.data.particles_.particle_types_.type_by_id_(1)
    st.radius = args.radius
    st.color = SI_COLOR
    cv = pl.source.data.cell_.vis
    cv.enabled = True
    cv.line_width = 0.045
    cv.rendering_color = CELL_COLOR

    # rotation-invariant framing: fit the cell's bounding sphere (+ atom radius) with a margin,
    # constant at every turntable angle.
    R = 0.5 * float(np.linalg.norm(L)) + args.radius
    fov = math.radians(args.fov_deg)
    dist = R * (1.0 + args.margin) / math.sin(fov / 2.0)
    tilt = math.radians(args.tilt_deg)

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.fov = fov
    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True,
                               ambient_occlusion_samples=20, antialiasing_samples=6)

    if args.preview > 0:
        idx = np.linspace(0, nsrc - 1, args.preview).round().astype(int)
    else:
        idx = np.linspace(0, nsrc - 1, args.nframes).round().astype(int)
    frame_times = ftime[idx] if len(ftime) == nsrc else np.linspace(0, t_end, len(idx))

    os.makedirs(args.workdir, exist_ok=True)
    W, H = args.w, args.h
    out = os.path.join(args.workdir, os.path.basename(args.out))

    def render_one(f, az_deg, t_now):
        az = math.radians(az_deg)
        eye = center + dist * np.array([math.cos(tilt) * math.sin(az),
                                        -math.cos(tilt) * math.cos(az),
                                        math.sin(tilt)])
        vp.camera_pos = tuple(eye)
        vp.camera_dir = tuple(center - eye)
        raw = os.path.join(args.workdir, "_raw.png")
        vp.render_image(size=(W, H), filename=raw, frame=int(f), renderer=renderer, alpha=True)
        rimg = Image.open(raw).convert("RGBA")
        canvas = Image.new("RGBA", (W, H), BG + (255,))
        canvas = Image.alpha_composite(canvas, rimg)
        Tk = float(np.interp(t_now, ttime, temp))
        state = "crystalline" if t_now < tcross else "liquid"
        _label(canvas, "%s   .   %d-atom %s   .   T = %.0f K" % (args.model, args.atoms, args.element, Tk),
               "molecular dynamics on Tenstorrent Blackhole   .   %s" % state)
        return canvas.convert("RGB")

    if args.preview > 0:
        for k, f in enumerate(idx):
            im = render_one(f, -25.0 + args.spin * k / max(1, len(idx) - 1), float(frame_times[k]))
            p = "%s_preview_%02d.png" % (out, k)
            im.save(p)
            print("wrote", p, "  t=%.0f fs" % frame_times[k], flush=True)
        return

    imgs = []
    with tempfile.TemporaryDirectory(dir=args.workdir) as td:
        for k, f in enumerate(idx):
            im = render_one(f, -25.0 + args.spin * k / max(1, len(idx) - 1), float(frame_times[k]))
            imgs.append(im)
            if k % 15 == 0:
                print("  rendered %d/%d  t=%.0f fs" % (k + 1, len(idx), frame_times[k]), flush=True)
        loop = imgs + imgs[-2:0:-1]                 # boomerang so it seams (MD is not time-periodic)
        for j, im in enumerate(loop):
            im.save("%s/f%04d.png" % (td, j))
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", "%s/f%%04d.png" % td,
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", out + ".mp4"], check=True, capture_output=True)
        print("wrote %s.mp4  (%d frames, %d fps)" % (out, len(loop), args.fps))
        subprocess.run(
            ["ffmpeg", "-y", "-i", "%s/f%%04d.png" % td, "-vf",
             "fps=%d,scale=%d:-1:flags=lanczos,palettegen=max_colors=%d" % (args.gif_fps, args.gif_px, args.gif_colors),
             "%s/pal.png" % td], check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", "%s/f%%04d.png" % td,
             "-i", "%s/pal.png" % td, "-lavfi",
             "fps=%d,scale=%d:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4"
             % (args.gif_fps, args.gif_px), out + ".gif"], check=True, capture_output=True)
        print("wrote %s.gif  (%.1f MB, %dpx)" % (out, os.path.getsize(out + ".gif") / 1e6, args.gif_px))


if __name__ == "__main__":
    main()
