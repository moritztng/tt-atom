"""Melt verification: one figure proving the run really melts (not just vibrates hot).

Left   : NVT setpoint vs realised temperature over the run (thermostat tracks the ramp).
Middle : potential energy / atom + mean-squared displacement -- a solid's MSD plateaus at the
         vibration amplitude; a liquid's grows without bound (self-diffusion). The MSD lift-off is
         the melt.
Right  : radial distribution function g(r), first (solid) vs last (liquid) frame -- the sharp
         diamond-lattice peaks (and the split second-neighbour shell) wash out into the broad
         single-peak envelope of a disordered liquid: loss of long-range order.

    ~/.ttatom_run/env/bin/python plot_melt.py --csv melt.csv --traj melt.extxyz --out melt_verify.png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read


def rdf(atoms, rmax=8.0, nbins=120):
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    N = len(atoms)
    inv = np.linalg.inv(cell)
    d = pos[:, None, :] - pos[None, :, :]
    frac = d @ inv
    frac -= np.round(frac)                       # minimum image
    d = frac @ cell
    dist = np.linalg.norm(d, axis=-1)
    dist = dist[~np.eye(N, dtype=bool)]
    hist, edges = np.histogram(dist, bins=nbins, range=(0.1, rmax))
    r = 0.5 * (edges[1:] + edges[:-1])
    vol = np.linalg.det(cell)
    rho = N / vol
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (N * rho * shell)
    return r, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t_ps = d["time_fs"] / 1000.0

    frames = read(args.traj, index=":")
    r0, g0 = rdf(frames[0])
    r1, g1 = rdf(frames[-1])

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax[0].plot(t_ps, d["temp_set_K"], "--", color="#c0392b", label="setpoint")
    ax[0].plot(t_ps, d["temp_K"], color="#2c3e50", lw=1.0, label="realised")
    ax[0].set(xlabel="time (ps)", ylabel="temperature (K)", title="NVT temperature ramp")
    ax[0].legend(frameon=False)

    ax1b = ax[1].twinx()
    l1, = ax[1].plot(t_ps, d["epot_ev_atom"], color="#2980b9", label="E$_{pot}$/atom")
    l2, = ax1b.plot(t_ps, d["msd_A2"], color="#e67e22", label="MSD")
    ax[1].set(xlabel="time (ps)", ylabel="E$_{pot}$ (eV/atom)", title="energy & diffusion")
    ax1b.set_ylabel(r"MSD ($\rm\AA^2$)")
    ax[1].legend(handles=[l1, l2], frameon=False, loc="upper left")

    ax[2].plot(r0, g0, color="#7f8c8d", label="frame 0 (solid)")
    ax[2].plot(r1, g1, color="#c0392b", label="final (liquid)")
    ax[2].set(xlabel=r"r ($\rm\AA$)", ylabel="g(r)", title="radial distribution")
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    msd_end = float(np.mean(d["msd_A2"][-min(50, len(d)):]))
    print(f"wrote {args.out}")
    print(f"final MSD           : {msd_end:.2f} A^2  ({'LIQUID' if msd_end > 1.5 else 'not clearly liquid'})")
    print(f"g(r) first-peak height: solid {g0.max():.2f} -> liquid {g1.max():.2f} "
          f"(order parameter: sharper solid peak collapses)")


if __name__ == "__main__":
    main()
