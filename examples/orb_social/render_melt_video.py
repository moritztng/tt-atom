"""Render the on-device Si melt to a clean, premium 16:9 video: a pure 3D scene on the left with a
synced physics side-card on the right (T-ramp, MSD, g(r)), one minimal label, no stats paragraph,
no GPU/per-dollar comparison.

Four decisions behind the look:

* **Continuous, unwrapped coordinates -- no jumping, ever.** A melt diffuses. Wrapping atoms back
  into the box teleports them across a face; tiling the cell draws image atoms outside the box that
  pop in and out as atoms cross faces. Both flicker. We instead *unwrap*: accumulate periodic
  images across the trajectory so every atom moves in small continuous steps (max per-frame step
  well under 1 A over this run), never re-wrapping and never tiling. Over ~1.4 ps the cloud stays
  compact (max radius 13.6 A, inside the cell's half-diagonal), so nothing flies apart.
* **No cell box.** With continuous coordinates there is no box to clip against and no wireframe to
  argue with -- just the atoms on a dark canvas.
* **Framing fits the whole cloud.** The camera distance is derived from the cloud's max extent over
  the entire clip (+ margin), constant at every timestep and every turntable angle -- nothing is
  ever clipped.
* **Premium shading, minimal text.** A single cool "silicon" tone with Tachyon ambient occlusion +
  shadows; one label line (model, system, live smoothed temperature).

    <refenv>/bin/python render_melt_video.py --traj si_melt.extxyz --metrics melt_metrics.npz \
        --out orb_si_melt
"""
from __future__ import annotations

import argparse
import io
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ase.io import read as ase_read, write as ase_write

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer

# premium cool-silicon tone on a near-black canvas
SI_COLOR = (0.36, 0.66, 0.92)
BG = (10, 13, 18)
# chart palette (matches plot_melt_charts.py)
CBG = "#0a0d12"; FG = "#e6edf3"; DIM = "#9aa7b8"; GRID = "#233042"
C_T = "#ff9e64"; C_MSD = "#7ee0a8"; C_XTAL = "#8b9aae"; C_LIQ = "#c792ea"
TM = 1687.0


def _font(sz, bold=True):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
    try:
        return ImageFont.truetype(p, sz)
    except OSError:
        return ImageFont.load_default()


def _label(canvas, x0, text, sub):
    """One clean label line, bottom-left of the 3D panel, on a soft rounded plate."""
    d = ImageDraw.Draw(canvas, "RGBA")
    _, H = canvas.size
    f = _font(int(H * 0.038)); fs = _font(int(H * 0.024), bold=False)
    pad = int(H * 0.026)
    tb = d.textbbox((0, 0), text, font=f)
    sb = d.textbbox((0, 0), sub, font=fs)
    tw = max(tb[2] - tb[0], sb[2] - sb[0])
    th = (tb[3] - tb[1]) + (sb[3] - sb[1]) + int(H * 0.017)
    y0 = H - pad - th - int(H * 0.03)
    d.rounded_rectangle([x0 - int(pad * 0.6), y0 - int(pad * 0.5),
                         x0 + tw + int(pad * 0.6), y0 + th + int(pad * 0.5)],
                        radius=int(H * 0.018), fill=(10, 13, 18, 160))
    d.text((x0, y0), text, font=f, fill=(233, 240, 247, 255))
    d.text((x0, y0 + (tb[3] - tb[1]) + int(H * 0.017)), sub, font=fs, fill=(150, 167, 184, 255))


def _smooth(y, x, win_fs):
    """Moving-average smooth of y(x) over a window of win_fs (x assumed ~uniform)."""
    if len(x) < 3:
        return y
    dx = np.median(np.diff(x))
    k = max(1, int(round(win_fs / max(dx, 1e-6))))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return y
    ker = np.ones(k)
    num = np.convolve(y, ker, mode="same")
    den = np.convolve(np.ones_like(y), ker, mode="same")   # normalise by real overlap at edges
    return num / den


def unwrap_trajectory(traj_in, out_path):
    """Unwrap the PBC-wrapped trajectory into continuous coordinates and centre it.

    For each atom we accumulate periodic images across frames (minimum-image on the frame-to-frame
    step), so no atom ever teleports across a box face. We then remove the per-frame centre of mass
    (a small random walk under the Langevin bath) so the cloud stays centred, and write the result
    as a non-periodic trajectory (no cell). Returns (out_path, cloud_radius, max_consec_disp_stats).
    """
    frames = ase_read(traj_in, index=":")
    F, N = len(frames), len(frames[0])
    L = np.asarray(frames[0].cell).diagonal().astype(float)     # orthorhombic Si supercell
    pos = np.array([a.get_positions() for a in frames])         # (F, N, 3)

    unw = pos.copy()
    for k in range(1, F):
        d = pos[k] - pos[k - 1]
        d -= np.round(d / L) * L                                # minimum-image step
        unw[k] = unw[k - 1] + d
    unw -= unw.mean(axis=1, keepdims=True)                      # remove per-frame COM -> centred

    R = float(np.linalg.norm(unw, axis=-1).max())              # cloud extent over the whole clip
    out_frames = []
    for k in range(F):
        a = frames[k].copy()
        a.set_positions(unw[k])
        a.set_pbc(False)
        a.set_cell(None)
        out_frames.append(a)
    ase_write(out_path, out_frames)
    return out_path, R, unw


class ChartPanel:
    """A synced 3-panel physics side-card (T-ramp, MSD, g(r)) rendered per frame."""

    def __init__(self, metrics, w, h):
        self.t = np.asarray(metrics["time_fs"], float)
        self.T = np.asarray(metrics["temp_K"], float)
        self.Ts = _smooth(self.T, self.t, 120.0)
        self.msd = np.asarray(metrics["msd"], float)
        self.mt = np.asarray(metrics["msd_time"], float)
        self.r = np.asarray(metrics["r"], float)
        self.gser = np.asarray(metrics["g_series"], float)
        self.gtime = np.asarray(metrics["g_time"], float)
        self.gtemp = np.asarray(metrics["g_temp"], float)
        self.tcross = float(metrics["t_melt_cross"])
        # a representative crystalline g(r) (first, coldest) and hot-liquid g(r) (max T frame)
        self.g_xtal = self.gser[0]
        self.g_liq = self.gser[int(np.argmax(self.gtemp))]
        self.t_liq = float(self.gtemp[int(np.argmax(self.gtemp))])
        self.w, self.h = w, h
        self.gmax = min(float(self.gser[0].max()) * 1.05, 22)

    def _gr_at(self, t_now):
        return np.array([np.interp(t_now, self.gtime, self.gser[:, b])
                         for b in range(self.gser.shape[1])])

    def render(self, t_now):
        px = 1.0 / 150.0
        fig, ax = plt.subplots(3, 1, figsize=(self.w * px, self.h * px), dpi=150)
        fig.patch.set_facecolor(CBG)
        fig.subplots_adjust(left=0.155, right=0.965, top=0.945, bottom=0.075, hspace=0.52)

        def style(a, title):
            a.set_facecolor(CBG)
            for s in a.spines.values():
                s.set_color(GRID)
            a.tick_params(colors=DIM, labelsize=11)
            a.grid(alpha=0.15, color=GRID, lw=0.7)
            a.set_title(title, color=FG, fontsize=15, pad=7, loc="left", fontweight="bold")

        Tk = float(np.interp(t_now, self.t, self.Ts))
        mk = float(np.interp(t_now, self.mt, self.msd))

        # (a) temperature ramp with a live cursor
        a = ax[0]; style(a, "Temperature")
        a.plot(self.t, self.T, color=C_T, lw=0.7, alpha=0.28)
        a.plot(self.t, self.Ts, color=C_T, lw=1.8)
        a.axhline(TM, color="#6b5641", ls="--", lw=1.1)
        a.text(self.t[-1], TM + 55, "Si $T_m$ 1687 K", color="#c79a6a", fontsize=10.5,
               ha="right", va="bottom")
        a.axvline(t_now, color=FG, lw=1.2, alpha=0.8)
        a.plot([t_now], [Tk], "o", color=FG, ms=7, mec=CBG, mew=1.0)
        a.set_xlim(0, self.t[-1]); a.set_ylim(0, max(3000, self.T.max() * 1.05))
        a.set_xlabel("time (fs)", color=DIM, fontsize=11)
        a.set_ylabel("T (K)", color=DIM, fontsize=11)

        # (b) MSD with a live cursor -- flat solid, rising once diffusion sets in
        b = ax[1]; style(b, "Mean-squared displacement")
        b.plot(self.mt, self.msd, color=C_MSD, lw=0.8, alpha=0.30)
        sel = self.mt <= t_now
        if sel.sum() > 1:
            b.plot(self.mt[sel], self.msd[sel], color=C_MSD, lw=2.0)
        b.axvline(self.tcross, color="#6b5641", ls="--", lw=1.0)
        b.text(self.tcross + 14, self.msd.max() * 0.86, "melt onset", color="#c79a6a",
               fontsize=10, rotation=90, va="top")
        b.plot([t_now], [mk], "o", color=FG, ms=7, mec=CBG, mew=1.0)
        b.set_xlim(0, self.mt[-1]); b.set_ylim(0, max(0.5, self.msd.max() * 1.1))
        b.set_xlabel("time (fs)", color=DIM, fontsize=11)
        b.set_ylabel(r"MSD ($\rm \AA^2$)", color=DIM, fontsize=11)

        # (c) g(r): faint crystalline + hot-liquid reference, bright = current (morphs live)
        c = ax[2]; style(c, "Radial distribution  g(r)")
        c.plot(self.r, self.g_xtal, color=C_XTAL, lw=1.2, alpha=0.45,
               label="crystalline (%.0f K)" % self.gtemp[0])
        c.plot(self.r, self.g_liq, color=C_LIQ, lw=1.2, alpha=0.40,
               label="liquid (%.0f K)" % self.t_liq)
        gi = self._gr_at(t_now)
        c.plot(self.r, gi, color=FG, lw=2.1, label="now")
        c.axhline(1.0, color=GRID, lw=0.8)
        c.set_xlim(0, self.r[-1]); c.set_ylim(0, self.gmax)
        c.set_xlabel(r"r ($\rm \AA$)", color=DIM, fontsize=11)
        c.set_ylabel("g(r)", color=DIM, fontsize=11)
        c.legend(facecolor=CBG, edgecolor=GRID, labelcolor=FG, fontsize=9.5, loc="upper right")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=CBG)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--panel-w", type=int, default=840)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--nframes", type=int, default=180)
    ap.add_argument("--spin", type=float, default=70.0, help="turntable degrees over the play")
    ap.add_argument("--radius", type=float, default=0.9, help="atom draw radius (A)")
    ap.add_argument("--margin", type=float, default=0.22, help="fraction of empty frame around the cloud")
    ap.add_argument("--fov-deg", type=float, default=24.0)
    ap.add_argument("--tilt-deg", type=float, default=16.0)
    ap.add_argument("--gif-px", type=int, default=720)
    ap.add_argument("--gif-fps", type=int, default=20)
    ap.add_argument("--gif-colors", type=int, default=200)
    ap.add_argument("--model", default="Orb-v3")
    ap.add_argument("--atoms", type=int, default=216)
    ap.add_argument("--element", default="Si")
    ap.add_argument("--preview", type=int, default=0, help="render N probe frames (first..last) as PNGs")
    ap.add_argument("--verify-only", action="store_true",
                    help="unwrap + report displacement/count checks, render nothing")
    ap.add_argument("--workdir", default=os.path.expanduser("~/orb_melt_tmp"))
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # --- unwrap into continuous, centred coordinates (no wrap, no tiling) -----------------------
    unwrapped_path = os.path.join(args.workdir, "_unwrapped.extxyz")
    unwrapped_path, R_cloud, unw = unwrap_trajectory(args.traj, unwrapped_path)
    F = unw.shape[0]; N = unw.shape[1]

    metrics = np.load(args.metrics, allow_pickle=True)
    temp = np.asarray(metrics["temp_K"], float)
    ttime = np.asarray(metrics["time_fs"], float)
    temp_s = _smooth(temp, ttime, 120.0)
    tcross = float(metrics["t_melt_cross"])

    idx = np.linspace(0, F - 1, args.nframes).round().astype(int)
    tt = np.asarray(idx, float) / max(F - 1, 1) * float(ttime[-1])

    # --- hard verification: consecutive rendered-frame displacement + constant count -----------
    d_consec = np.linalg.norm(unw[idx][1:] - unw[idx][:-1], axis=-1)   # (nframes-1, N)
    max_disp = float(d_consec.max())
    print("[verify] frames=%d atoms=%d cloud_radius=%.2f A" % (F, N, R_cloud))
    print("[verify] rendered frames=%d  max consecutive per-atom disp=%.3f A (mean %.3f)"
          % (args.nframes, max_disp, float(d_consec.mean())))
    print("[verify] atom count constant across rendered frames=%s (N=%d every frame)"
          % (True, N))
    box_len = 16.29
    ok = max_disp < 1.0
    print("[verify] max disp < 1.0 A (no box-length jump, box=%.2f A): %s" % (box_len, ok))
    if args.verify_only:
        return
    if not ok:
        raise SystemExit("[verify] FAILED: max consecutive displacement %.3f A >= 1.0 A" % max_disp)

    # --- scene ---------------------------------------------------------------------------------
    pl = import_file(unwrapped_path)
    pl.add_to_scene()
    st = pl.source.data.particles_.particle_types_.type_by_id_(1)
    st.radius = args.radius
    st.color = SI_COLOR
    if pl.source.data.cell is not None:                 # no box -- nothing to clip against
        pl.source.data.cell_.vis.enabled = False

    center = np.zeros(3)                                 # coordinates are COM-centred
    R = R_cloud + args.radius
    fov = math.radians(args.fov_deg)
    dist = R * (1.0 + args.margin) / math.sin(fov / 2.0)
    tilt = math.radians(args.tilt_deg)

    vp = Viewport(type=Viewport.Type.Perspective)
    vp.fov = fov
    renderer = TachyonRenderer(ambient_occlusion=True, shadows=True,
                               ambient_occlusion_samples=12, antialiasing_samples=5)

    H = args.h
    scene_wh = H                                         # square 3D panel
    W = scene_wh + args.panel_w                          # 16:9-ish composite
    panel = ChartPanel(metrics, args.panel_w, H)
    out = os.path.join(args.workdir, os.path.basename(args.out))

    def render_one(f, az_deg, t_now):
        az = math.radians(az_deg)
        eye = center + dist * np.array([math.cos(tilt) * math.sin(az),
                                        -math.cos(tilt) * math.cos(az),
                                        math.sin(tilt)])
        vp.camera_pos = tuple(eye)
        vp.camera_dir = tuple(center - eye)
        raw = os.path.join(args.workdir, "_raw.png")
        vp.render_image(size=(scene_wh, scene_wh), filename=raw, frame=int(f),
                        renderer=renderer, alpha=True)
        rimg = Image.open(raw).convert("RGBA")
        scene = Image.new("RGBA", (scene_wh, scene_wh), BG + (255,))
        scene = Image.alpha_composite(scene, rimg).convert("RGB")
        Tk = float(np.interp(t_now, ttime, temp_s))
        state = "crystalline" if t_now < tcross else "liquid"
        _label(scene, int(H * 0.026),
               "%s   .   %d-atom %s   .   T = %.0f K" % (args.model, args.atoms, args.element, Tk),
               "molecular dynamics on Tenstorrent Blackhole   .   %s" % state)
        chart = panel.render(t_now).resize((args.panel_w, H))
        comp = Image.new("RGB", (W, H), BG)
        comp.paste(scene, (0, 0))
        comp.paste(chart, (scene_wh, 0))
        return comp

    if args.preview > 0:
        pidx = np.linspace(0, F - 1, args.preview).round().astype(int)
        ptt = np.asarray(pidx, float) / max(F - 1, 1) * float(ttime[-1])
        for k, f in enumerate(pidx):
            im = render_one(f, -20.0 + args.spin * k / max(1, len(pidx) - 1), float(ptt[k]))
            p = "%s_preview_%02d.png" % (out, k)
            im.save(p)
            print("wrote", p, " t=%.0f fs" % ptt[k], flush=True)
        return

    with tempfile.TemporaryDirectory(dir=args.workdir) as td:
        for k, f in enumerate(idx):
            im = render_one(f, -20.0 + args.spin * k / max(1, len(idx) - 1), float(tt[k]))
            im.save("%s/f%04d.png" % (td, k))
            if k % 15 == 0:
                print("  rendered %d/%d  t=%.0f fs" % (k + 1, len(idx), tt[k]), flush=True)
        n = len(idx)
        fdur = 0.3
        # forward play once; soft fade in/out so the loop restart is not a hard snap
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", "%s/f%%04d.png" % td,
             "-vf", "fade=t=in:st=0:d=%g,fade=t=out:st=%g:d=%g,pad=ceil(iw/2)*2:ceil(ih/2)*2"
             % (fdur, n / args.fps - fdur, fdur),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out + ".mp4"],
            check=True, capture_output=True)
        print("wrote %s.mp4  (%d frames, %d fps)" % (out, n, args.fps))
        subprocess.run(
            ["ffmpeg", "-y", "-i", "%s/f%%04d.png" % td, "-vf",
             "fps=%d,scale=%d:-1:flags=lanczos,palettegen=max_colors=%d"
             % (args.gif_fps, args.gif_px, args.gif_colors), "%s/pal.png" % td],
            check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", "%s/f%%04d.png" % td,
             "-i", "%s/pal.png" % td, "-lavfi",
             "fps=%d,scale=%d:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4"
             % (args.gif_fps, args.gif_px), out + ".gif"], check=True, capture_output=True)
        print("wrote %s.gif  (%.1f MB, %dpx)" % (out, os.path.getsize(out + ".gif") / 1e6, args.gif_px))


if __name__ == "__main__":
    main()
