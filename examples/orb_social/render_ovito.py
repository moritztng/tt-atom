"""Render an on-device Orb-v3 MD trajectory to a professional looping MP4 + GIF using OVITO
(Tachyon software ray-tracer) — shaded spheres, ambient occlusion, the periodic simulation cell,
and the diamond bond network. Coordinates are *unwrapped* (continuous per-atom images accumulated
across the trajectory, not re-folded into the cell every frame), so an atom whose lattice site
sits near a periodic face vibrates smoothly instead of teleporting to the opposite side of the box
each time its wrapped image flips. This is valid here specifically because the crystal is solid at
900 K over 1.5 ps (no diffusion): no atom's unwrapped displacement grows beyond a small fraction of
the cell, so nothing drifts out of frame. Bonds use the same unwrapped (non-periodic) positions, so
none are drawn across periodic faces either.

A gentle turntable rotates the camera while the trajectory plays; the loop is a boomerang
(forward then reverse) so it seams cleanly even though MD is not time-periodic. A small, clean
caption is composited in.

    ~/.ttatom_run/refenv/bin/python render_ovito.py --traj si216_final.extxyz --out orb_si_md
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
from ovito.vis import Viewport, TachyonRenderer, BondsVis
from ovito.modifiers import UnwrapTrajectoriesModifier, CreateBondsModifier

JMOL_SI = (0.941, 0.784, 0.627)      # CPK/Jmol silicon colour (the materials-viz standard)
BG = (14, 17, 23)                    # solid dark slate


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def caption(img, px, lines_bold, lines_reg):
    """Composite a clean lower-left caption block (with a translucent panel for legibility)."""
    fb, fr = _font(int(px * 0.033)), _font(int(px * 0.027))
    pad = int(px * 0.028)
    x0 = int(px * 0.030)
    lh_b, lh_r = int(px * 0.046), int(px * 0.040)
    h = pad * 2 + lh_b * len(lines_bold) + int(px * 0.008) + lh_r * len(lines_reg)
    widths = ([fb.getbbox(t)[2] for t in lines_bold] + [fr.getbbox(t)[2] for t in lines_reg])
    w = pad * 2 + max(widths)
    y0 = px - int(px * 0.030) - h
    panel = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle([x0, y0, x0 + w, y0 + h],
                                            radius=int(px * 0.018), fill=(9, 12, 17, 170))
    img = Image.alpha_composite(img.convert("RGBA"), panel).convert("RGB")
    d = ImageDraw.Draw(img)
    x, y = x0 + pad, y0 + pad
    for t in lines_bold:
        d.text((x, y), t, font=fb, fill=(238, 242, 248))
        y += lh_b
    y += int(px * 0.008)
    for t in lines_reg:
        d.text((x, y), t, font=fr, fill=(150, 178, 208))
        y += lh_r
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=130, help="rendered MD frames (subsampled)")
    ap.add_argument("--spin", type=float, default=150.0, help="turntable degrees over the play")
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--bond-cutoff", type=float, default=2.55)
    ap.add_argument("--gif-px", type=int, default=440)
    ap.add_argument("--gif-fps", type=int, default=18)
    ap.add_argument("--gif-colors", type=int, default=128)
    args = ap.parse_args()

    pl = import_file(args.traj)
    nsrc = pl.source.num_frames
    pl.modifiers.append(UnwrapTrajectoriesModifier())    # continuous per-atom images, no teleport

    def _drop_pbc(frame, data):
        # unwrapped positions are absolute, not periodic images -- treat the rendered cell as
        # finite so bonds use plain (non-minimum-image) distances and none are drawn across what
        # used to be the periodic faces.
        data.cell_.pbc = (False, False, False)
    pl.modifiers.append(_drop_pbc)

    bonds = CreateBondsModifier(cutoff=args.bond_cutoff)
    bonds.vis.width = 0.30
    bonds.vis.color = (0.40, 0.45, 0.52)
    pl.modifiers.append(bonds)
    pl.add_to_scene()

    st = pl.source.data.particles_.particle_types_.type_by_id_(1)
    st.radius = args.radius
    st.color = JMOL_SI
    pl.source.data.cell_.vis.line_width = 0.045
    pl.source.data.cell_.vis.rendering_color = (0.55, 0.60, 0.68)

    # cell centre + a camera distance that frames the whole box with margin
    cell = np.array(pl.compute(0).cell[:3, :3])
    L = float(np.abs(cell).sum(axis=0).max())
    center = cell.sum(axis=0) * 0.5
    dist = L * 2.4
    tilt = math.radians(18.0)                     # look slightly down onto the cell

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.fov = math.radians(30.0)
    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True,
                               ambient_occlusion_samples=24, antialiasing_samples=6)

    idx = np.linspace(0, nsrc - 1, args.nframes).round().astype(int)
    imgs = []
    # the system ffmpeg is snap-confined: readable/writable only under $HOME, and not inside a
    # dotdir (snap's home interface denies hidden paths), so use a plain visible subdirectory
    home_tmp = os.path.expanduser("~/orb_render_tmp")
    os.makedirs(home_tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home_tmp) as td:
        for k, f in enumerate(idx):
            az = math.radians(-25.0 + args.spin * k / max(1, len(idx) - 1))
            eye = center + dist * np.array([math.cos(tilt) * math.sin(az),
                                            -math.cos(tilt) * math.cos(az),
                                            math.sin(tilt)])
            vp.camera_pos = tuple(eye)
            vp.camera_dir = tuple(center - eye)
            raw = f"{td}/r{k:04d}.png"
            vp.render_image(size=(args.px, args.px), filename=raw, frame=int(f),
                            renderer=renderer, alpha=True)
            # composite over the solid dark background, then caption
            fg = Image.open(raw).convert("RGBA")
            bg = Image.new("RGBA", fg.size, BG + (255,))
            comp = Image.alpha_composite(bg, fg).convert("RGB")
            comp = caption(comp, args.px,
                           ["Orb-v3  ·  216-atom silicon crystal"],
                           ["NVT molecular dynamics, 900 K",
                            "real conservative forces on Tenstorrent Blackhole"])
            imgs.append(comp)
            if k % 20 == 0:
                print(f"  rendered {k+1}/{len(idx)}", flush=True)

        loop = imgs + imgs[-2:0:-1]               # boomerang
        for j, im in enumerate(loop):
            im.save(f"{td}/f{j:04d}.png")

        def _ffmpeg(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{r.stderr[-4000:]}")

        # ffmpeg (snap-confined) can't write into a path with a dotdir component (e.g. a
        # ~/.coworker/... worktree) -- render into the plain tempdir, then plain-copy out.
        mp4_tmp, gif_tmp = f"{td}/out.mp4", f"{td}/out.gif"
        _ffmpeg(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", f"{td}/f%04d.png",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", mp4_tmp])

        _ffmpeg(["ffmpeg", "-y", "-i", f"{td}/f%04d.png", "-vf",
                 f"palettegen=max_colors={args.gif_colors}", f"{td}/pal.png"])
        _ffmpeg(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", f"{td}/f%04d.png",
                 "-i", f"{td}/pal.png", "-lavfi",
                 f"fps={args.gif_fps},scale={args.gif_px}:-1:flags=lanczos [x]; "
                 f"[x][1:v] paletteuse=dither=bayer:bayer_scale=4", gif_tmp])

        import shutil
        shutil.copyfile(mp4_tmp, args.out + ".mp4")
        shutil.copyfile(gif_tmp, args.out + ".gif")
    print(f"wrote {args.out}.mp4  ({len(loop)} frames, {args.fps} fps)")
    print(f"wrote {args.out}.gif  ({os.path.getsize(args.out + '.gif')/1e6:.1f} MB, {args.gif_px}px)")


if __name__ == "__main__":
    main()
