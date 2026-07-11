"""MD stability evidence: temperature and energy vs time from the per-step CSV.

NVT Langevin does not conserve total energy (the thermostat exchanges energy with a bath),
so the correct stability checks are: (1) temperature fluctuates around the 900 K target with
no drift, (2) potential energy equilibrates to a stable plateau (no blow-up, no monotonic
drift). Both are shown; the equilibration transient (first ~100 fs, atoms leaving the perfect
lattice) is shaded and excluded from the reported plateau averages.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, default=900.0)
    ap.add_argument("--equil-fs", type=float, default=100.0, help="equilibration window to exclude")
    args = ap.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = d["time_fs"]
    T = d["temp_K"]
    epot = d["epot_ev_atom"]
    plateau = t >= args.equil_fs
    T_mean, T_std = T[plateau].mean(), T[plateau].std()
    e_mean, e_std = epot[plateau].mean(), epot[plateau].std()
    # linear drift of the plateau potential energy (meV/atom per ps)
    slope = np.polyfit(t[plateau], epot[plateau], 1)[0] * 1e3 * 1e3  # eV/A/fs -> meV/atom/ps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for ax in (ax1, ax2):
        ax.axvspan(0, args.equil_fs, color="0.9", zorder=0)
        ax.grid(alpha=0.3)

    ax1.plot(t, T, lw=1.0, color="#c44")
    ax1.axhline(args.target, ls="--", color="0.4", lw=1.0)
    ax1.axhline(T_mean, color="#c44", lw=1.2, alpha=0.6)
    ax1.set_ylabel("temperature (K)")
    ax1.set_title(f"216-atom Si, Orb-v3 on Tenstorrent Blackhole — NVT Langevin @ {args.target:.0f} K")
    ax1.text(0.98, 0.06, f"plateau ⟨T⟩ = {T_mean:.0f} ± {T_std:.0f} K",
             transform=ax1.transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    ax2.plot(t, epot, lw=1.0, color="#446")
    ax2.axhline(e_mean, color="#446", lw=1.2, alpha=0.6)
    ax2.set_ylabel("potential energy (eV/atom)")
    ax2.set_xlabel("time (fs)")
    ax2.text(0.98, 0.10, f"plateau ⟨E⟩ = {e_mean:.4f} ± {e_std:.4f} eV/atom\n"
                         f"drift = {slope:+.2f} meV/atom·ps",
             transform=ax2.transAxes, ha="right", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    print(f"plateau T = {T_mean:.1f} +/- {T_std:.1f} K (target {args.target:.0f})")
    print(f"plateau Epot = {e_mean:.4f} +/- {e_std:.4f} eV/atom")
    print(f"Epot drift = {slope:+.2f} meV/atom/ps")


if __name__ == "__main__":
    main()
