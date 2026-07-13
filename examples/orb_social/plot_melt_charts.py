"""Publication-quality 4-panel figure of the on-device Si-melt physics, for scientific scrutiny.

Reads ``melt_metrics.npz`` (from ``analyze_melt.py``) and draws the four signatures a materials
scientist checks a melt against:

  (a) temperature ramp crossing Si's melting point,
  (b) total-energy conservation (thermostat heats the ramp, E_tot flat under NVE),
  (c) mean-squared displacement -- flat in the solid, rising once diffusion sets in,
  (d) radial distribution g(r) -- sharp crystalline shells dissolving into the liquid envelope.

    <refenv>/bin/python plot_melt_charts.py --metrics melt_metrics.npz --out melt_charts.png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0d1117"; FG = "#e6edf3"; DIM = "#9aa7b8"; GRID = "#233042"
C_T = "#ff9e64"; C_E = "#6cb6ff"; C_MSD = "#7ee0a8"; C_XTAL = "#8b9aae"; C_LIQ = "#c792ea"
TM = 1687.0


def _style(ax, title):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=9)
    ax.grid(alpha=0.16, color=GRID, lw=0.7)
    ax.set_title(title, color=FG, fontsize=12, pad=8, loc="left", fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = np.load(args.metrics, allow_pickle=True)

    t = np.asarray(m["time_fs"], float); T = np.asarray(m["temp_K"], float)
    etot = np.asarray(m["etot_ev_atom"], float); epot = np.asarray(m["epot_ev_atom"], float)
    nve0 = float(m["nve"])
    r = np.asarray(m["r"], float); gser = np.asarray(m["g_series"], float)
    gtemp = np.asarray(m["g_temp"], float)
    msd = np.asarray(m["msd"], float); mt = np.asarray(m["msd_time"], float)
    tcross = float(m["t_melt_cross"]); drift_s = float(m["drift_solid"]); drift_l = float(m["drift_liquid"])
    D = float(m["D_m2s"])
    # for the liquid g(r) reference use the hottest frame (clearly above T_m), not the cooled tail
    iliq = int(np.argmax(gtemp))

    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.2), dpi=150)
    fig.patch.set_facecolor(BG)

    # (a) temperature ramp
    a = ax[0, 0]; _style(a, "a   Temperature ramp")
    a.plot(t, T, color=C_T, lw=1.6)
    a.axhline(TM, color="#6b5641", ls="--", lw=1.1)
    a.text(t[-1], TM + 60, "Si  $T_m$ = 1687 K", color="#c79a6a", fontsize=9, ha="right", va="bottom")
    a.axvline(nve0, color="#41546b", ls=":", lw=1.1)
    a.text(nve0 + 8, T.max() * 0.96, "NVT | NVE", color=DIM, fontsize=8, va="top")
    a.plot([tcross], [TM], "o", color=FG, ms=5)
    a.annotate("crosses $T_m$ at %.0f fs" % tcross, (tcross, TM), (tcross + 60, TM - 520),
               color=FG, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8))
    a.set_xlabel("time (fs)", color=DIM, fontsize=10); a.set_ylabel("temperature (K)", color=DIM, fontsize=10)
    a.set_xlim(0, t[-1]); a.set_ylim(0, max(3000, T.max() * 1.05))

    # (b) energy conservation
    b = ax[0, 1]; _style(b, "b   Total-energy conservation")
    b.plot(t, etot, color=C_E, lw=1.6, label="$E_{tot}$")
    b.plot(t, epot, color="#4a6a8a", lw=1.0, alpha=0.8, label="$E_{pot}$")
    b.axvline(nve0, color="#41546b", ls=":", lw=1.1)
    b.text(nve0 + 8, etot.max(), "NVT | NVE", color=DIM, fontsize=8, va="top")
    b.set_xlabel("time (fs)", color=DIM, fontsize=10); b.set_ylabel("energy (eV/atom)", color=DIM, fontsize=10)
    b.set_xlim(0, t[-1])
    b.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8.5, loc="lower right")
    b.text(0.03, 0.06,
           "NVE drift: %.1f meV/atom/ps (900 K solid)\n                %.1f meV/atom/ps (this liquid tail)"
           % (abs(drift_s), abs(drift_l)),
           transform=b.transAxes, color=DIM, fontsize=8.2, va="bottom", family="monospace")

    # (c) MSD
    c = ax[1, 0]; _style(c, "c   Mean-squared displacement")
    c.plot(mt, msd, color=C_MSD, lw=1.8)
    c.axvline(tcross, color="#6b5641", ls="--", lw=1.1)
    c.text(tcross + 12, msd.max() * 0.2, "melt onset", color="#c79a6a", fontsize=8.5, rotation=90, va="bottom")
    c.set_xlabel("time (fs)", color=DIM, fontsize=10); c.set_ylabel(r"MSD (${\rm \AA}^2$)", color=DIM, fontsize=10)
    c.set_xlim(0, mt[-1]); c.set_ylim(0, max(0.5, msd.max() * 1.1))
    c.text(0.03, 0.94,
           "flat (solid) $\\rightarrow$ rising (diffusion)\n$D \\approx %.1f\\times10^{-9}$ m$^2$/s  (indicative, short window)"
           % (D * 1e9),
           transform=c.transAxes, color=DIM, fontsize=8.2, va="top")

    # (d) g(r)
    d = ax[1, 1]; _style(d, "d   Radial distribution  g(r)")
    d.plot(r, gser[0], color=C_XTAL, lw=1.7, label="crystalline (%.0f K)" % gtemp[0])
    d.plot(r, gser[iliq], color=C_LIQ, lw=1.9, label="liquid (%.0f K)" % gtemp[iliq])
    d.fill_between(r, 0, gser[iliq], color=C_LIQ, alpha=0.10)
    d.axhline(1.0, color=GRID, lw=0.8)
    d.set_xlabel(r"r (${\rm \AA}$)", color=DIM, fontsize=10); d.set_ylabel("g(r)", color=DIM, fontsize=10)
    d.set_xlim(0, r[-1]); d.set_ylim(0, min(float(gser[0].max()) * 1.05, 22))
    d.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper right")

    fig.suptitle("Silicon melt on Tenstorrent Blackhole  .  Orb-v3 (conservative-inf-omat)  .  216 atoms",
                 color=FG, fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, facecolor=BG)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
