"""Render an on-device Orb-v3 MD trajectory to a professional looping MP4 + GIF using OVITO
(Tachyon software ray-tracer) — shaded spheres, ambient occlusion, soft shadows, and the bond
network.

Two coordinate modes, matched to the phase being shown:

* Solid (default): coordinates are *unwrapped* (continuous per-atom images accumulated across the
  trajectory, not re-folded into the cell every frame), so an atom whose lattice site sits near a
  periodic face vibrates smoothly instead of teleporting when its wrapped image flips. Valid only
  because a crystal below melting has no diffusion, so no atom drifts out of frame. No cell
  wireframe is drawn (unwrapped atoms sit outside the original box, so an outline would cage it).

* Liquid (``--wrap``): a melt *diffuses*, so unwrapping would fly atoms out of frame. Coordinates
  are instead folded back into the periodic cell every frame (the standard, correct liquid view),
  with a subtle thin cell outline that frames the wrapped fluid and explains why an atom leaving
  one face reappears at the opposite one.

In both modes the rendered cell is treated as finite (pbc off) before bonds are built, so no bond
is ever drawn across a periodic face as a long line spanning the box.

Framing: the camera distance is set from the bounding sphere of *every atom position across every
frame* (not just frame 0's cell box), so the whole crystal stays fully in view with margin at every
turntable angle and every timestep -- nothing clipped at the frame edge.

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
from ovito.modifiers import (UnwrapTrajectoriesModifier, WrapPeriodicImagesModifier,
                             CreateBondsModifier)

SI_COLOR = (0.22, 0.38, 0.58)        # premium metallic blue-steel (richer, more saturated)
BOND_COLOR = (0.14, 0.22, 0.32)      # muted, darker echo of the atom colour (not a stark cage)
BG = (8, 10, 15)                     # solid near-black slate, complements the cool atom tone


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


def _bounding_sphere(pl, nsrc):
    """Bounding sphere (center, radius) over every atom position in every frame, so the camera
    distance keeps the whole crystal in view at every timestep and turntable angle."""
    all_pos = [np.array(pl.compute(f).particles.positions) for f in range(nsrc)]
    stacked = np.concatenate(all_pos, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    center = (lo + hi) / 2.0
    radius = max(np.linalg.norm(pos - center, axis=1).max() for pos in all_pos)
    return center, float(radius)


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
    ap.add_argument("--margin", type=float, default=1.18, help="framing headroom over the bounding sphere")
    ap.add_argument("--gif-px", type=int, default=440)
    ap.add_argument("--gif-fps", type=int, default=18)
    ap.add_argument("--gif-colors", type=int, default=128)
    ap.add_argument("--wrap", action="store_true",
                    help="liquid mode: fold atoms into the cell (a melt diffuses, so unwrapping "
                         "would fly atoms out of frame), draw a subtle box, non-periodic bonds")
    ap.add_argument("--cap-title", default="Orb-v3  ·  216-atom silicon crystal")
    ap.add_argument("--cap-line1", default="NVT molecular dynamics, 900 K")
    ap.add_argument("--cap-line2", default="real conservative forces on Tenstorrent Blackhole")
    args = ap.parse_args()

    pl = import_file(args.traj)
    nsrc = pl.source.num_frames

    if args.wrap:
        # a melt diffuses -- fold every atom back into the periodic cell each frame (the standard,
        # correct way to view a liquid), so nothing drifts out of frame.
        pl.modifiers.append(WrapPeriodicImagesModifier())
    else:
        # a solid only vibrates -- unwrap to continuous images so atoms near a face don't teleport.
        pl.modifiers.append(UnwrapTrajectoriesModifier())

    def _drop_pbc(frame, data):
        # After wrapping/unwrapping, render the cell as finite so bonds use plain (non-minimum-
        # image) distances: no bond is drawn across a periodic face (a long line spanning the box).
        # For the liquid this is the "clean, no misleading cross-face bonds" view; for the solid
        # the unwrapped images are absolute anyway.
        data.cell_.pbc = (False, False, False)
    pl.modifiers.append(_drop_pbc)

    bonds = CreateBondsModifier(cutoff=args.bond_cutoff)
    bonds.vis.width = 0.16
    bonds.vis.color = BOND_COLOR
    pl.modifiers.append(bonds)
    pl.add_to_scene()

    st = pl.source.data.particles_.particle_types_.type_by_id_(1)
    st.radius = args.radius
    st.color = SI_COLOR
    if args.wrap:
        # a subtle, thin cell outline: it frames the wrapped liquid and explains why an atom that
        # diffuses out one face reappears at the opposite one -- without caging the view.
        cv = pl.source.data.cell_.vis
        cv.enabled = True
        cv.line_width = 0.06
        cv.rendering_color = (0.18, 0.24, 0.32)
    else:
        pl.source.data.cell_.vis.enabled = False   # no wireframe -- unwrapped atoms sit outside it

    # bounding sphere over EVERY frame's actual atom positions (not just frame 0's cell box), so
    # nothing clips at the frame edge as atoms vibrate and the camera turntables
    center, bound_r = _bounding_sphere(pl, nsrc)
    tilt = math.radians(18.0)                     # look slightly down onto the cell
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.fov = math.radians(30.0)
    dist = (bound_r * args.margin) / math.tan(vp.fov / 2.0)
    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True,
                               ambient_occlusion_samples=24, ambient_occlusion_brightness=0.65,
                               direct_light_intensity=1.05, antialiasing_samples=8)

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
            comp = caption(comp, args.px, [args.cap_title],
                           [args.cap_line1, args.cap_line2])
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
    print(f"framing: bounding-sphere radius={bound_r:.2f} A, camera dist={dist:.2f} A, margin={args.margin}")


if __name__ == "__main__":
    main()
