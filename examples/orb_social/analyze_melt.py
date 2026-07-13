"""Melt analysis — the scientist-readable signatures from the on-device Si melt trajectory.

Reads the per-step CSV (energy / temperature) and the saved extxyz frames and produces:

* the temperature trace and total-energy trace (with the NVE-tail energy-conservation drift),
* a radial distribution function g(r) evaluated at a sweep of frames across the melt — the
  crystalline peak structure dissolving into the broad liquid envelope is the structural signature,
* a mean-squared-displacement curve (PBC-unwraped) showing the onset of diffusion in the liquid.

Everything is written to a single ``melt_metrics.npz`` (for the renderer to draw live) and a
``melt_summary.json`` of the headline numbers for the post / caption.

    ~/.ttatom_run/env/bin/python examples/orb_social/analyze_melt.py \
        --csv md_melt.csv --traj si_melt.extxyz --solid-csv md_solid_nve.csv \
        --out melt_metrics.npz --summary melt_summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from ase.io import read


def _wrap(pos, cell):
    """Fold positions into the cell along each lattice vector (fractional wrap)."""
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    frac -= np.floor(frac)
    return frac @ cell


def _rdf(pos, cell, r_max, nbins):
    """Radial distribution function g(r) with PBC minimum image, monatomic N atoms."""
    n = len(pos)
    inv = np.linalg.inv(cell)
    wrapped = pos @ inv
    dr = wrapped[:, None, :] - wrapped[None, :, :]
    dr -= np.round(dr)                      # minimum image (fractional)
    dcart = dr @ cell
    d2 = (dcart ** 2).sum(-1)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(d2[iu])
    hist, edges = np.histogram(d, bins=nbins, range=(0, r_max))
    r = 0.5 * (edges[1:] + edges[:-1])
    vol = abs(np.linalg.det(cell))
    rho = n / vol
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (shell * rho * n / 2.0)      # N/2 pair count normalisation
    return r, g


def _unwrap_series(frames):
    """Reconstruct unwrapped positions from a trajectory whose frames may have crossed a
    periodic face between saves: subtract any integer-cell jump that exceeds half a cell
    length along that lattice vector. Returns positions shaped [F, N, 3] in the frame-0
    reference image, and the frame times (indices)."""
    xs = np.array([f.get_positions() for f in frames])
    cell = np.array(frames[0].get_cell())
    inv = np.linalg.inv(cell)
    frac = xs @ inv
    # jump = round(frac[t] - frac[t-1]) when |delta| > 0.5 -> that many cells crossed
    dfrac = np.diff(frac, axis=0)
    jumps = np.round(dfrac)
    unwrapped_frac = np.empty_like(frac)
    unwrapped_frac[0] = frac[0]
    for k in range(1, len(frac)):
        unwrapped_frac[k] = unwrapped_frac[k - 1] + (dfrac[k - 1] - jumps[k - 1])
    return unwrapped_frac @ cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="melt run per-step CSV")
    ap.add_argument("--traj", required=True, help="melt run extxyz trajectory")
    ap.add_argument("--solid-csv", default=None, help="clean equilibrated-solid NVE CSV for the headline drift")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--r-max", type=float, default=6.0)
    ap.add_argument("--nbins", type=int, default=90)
    ap.add_argument("--n-rdf", type=int, default=14, help="number of g(r) frames across the melt")
    ap.add_argument("--save-every", type=int, default=4, help="frame save interval (MD steps)")
    ap.add_argument("--dt", type=float, default=0.5, help="MD timestep (fs)")
    ap.add_argument("--nve-skip-fs", type=float, default=200.0, help="exclude the NVE thermostat-off transient")
    args = ap.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    step = d["step"]; t = d["time_fs"]; epot = d["epot_ev_atom"]; ekin = d["ekin_ev_atom"]
    etot = d["etot_ev_atom"]; T = d["temp_K"]; reg = d["regime"]
    nve = reg == "nve"
    t_nve0 = float(t[nve].min())
    nve_settle = nve & (t >= t_nve0 + args.nve_skip_fs)
    drift_liquid = float(np.polyfit(t[nve_settle], etot[nve_settle], 1)[0]) * 1e3 * 1e3

    # headline: clean equilibrated-solid NVE drift (the UMA-analog credibility number)
    drift_solid = float("nan")
    if args.solid_csv:
        s = np.genfromtxt(args.solid_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
        sr = s["regime"] == "nve"
        st = s["time_fs"]; setot = s["etot_ev_atom"]
        st0 = float(st[sr].min())
        sset = sr & (st >= st0 + args.nve_skip_fs)
        drift_solid = float(np.polyfit(st[sset], setot[sset], 1)[0]) * 1e3 * 1e3

    # T at which the ramp crosses Si Tmelt (interpolate in time)
    t_melt_cross = float(np.interp(1687.0, T[reg == "rmp"], t[reg == "rmp"]))

    frames = read(args.traj, index=":")
    nf = len(frames)
    # frame k was saved at MD step k*save_every -> time k*save_every*dt (frame 0 is the t=0 snap)
    ftime = np.arange(nf, dtype=np.float64) * args.save_every * args.dt
    # per-frame temperature: build a step -> T lookup from the CSV (dedupe the doubled step 0)
    step_to_T = {}
    for stv, tv in zip(step, T):
        step_to_T.setdefault(int(stv), float(tv))
    ftemp = np.array([step_to_T.get(int(k * args.save_every), float("nan")) for k in range(nf)])
    # subsample frames evenly for the g(r) sweep (skip the very first perfect-lattice frame)
    rdf_idx = np.linspace(1, nf - 1, args.n_rdf).round().astype(int)
    rdf_idx = np.unique(rdf_idx)
    r_grid, _ = _rdf(frames[0].get_positions(), np.array(frames[0].get_cell()), args.r_max, args.nbins)
    g_series = []
    for fi in rdf_idx:
        f = frames[fi]
        cell = np.array(f.get_cell())
        pos = _wrap(f.get_positions(), cell)
        _, g = _rdf(pos, cell, args.r_max, args.nbins)
        g_series.append(g)
    g_series = np.array(g_series)
    g_time = ftime[rdf_idx]
    g_temp = ftemp[rdf_idx]

    # MSD over the full frame series (unwrapped), referenced to frame 0
    unw = _unwrap_series(frames)
    msd = ((unw - unw[0]) ** 2).sum(-1).mean(-1)        # [F] Angstrom^2
    # diffusion coefficient from the LATE liquid (NVE tail, well past the melt transient):
    # there the MSD slope is self-diffusion in the equilibrating liquid, not the disordering.
    liq = ftime >= max(t_melt_cross + 200.0, 1000.0)
    D = float("nan")
    if liq.sum() > 5:
        slope = float(np.polyfit(ftime[liq], msd[liq], 1)[0])      # A^2/fs
        if slope > 0:
            D = slope / 6.0 * 1e-10                                 # A^2/fs -> m^2/s

    np.savez(args.out,
             time_fs=t, temp_K=T, epot_ev_atom=epot, ekin_ev_atom=ekin, etot_ev_atom=etot,
             regime=reg, nve=t_nve0,
             r=r_grid, g_series=g_series, g_time=np.array(g_time), g_temp=np.array(g_temp),
             msd=msd, msd_time=ftime, t_melt_cross=t_melt_cross,
             drift_liquid=drift_liquid, drift_solid=drift_solid, D_m2s=D)
    print(f"wrote {args.out}")

    summary = {
        "system": "Si diamond 3x3x3, 216 atoms",
        "T_melt_K": 1687.0,
        "t_melt_cross_fs": t_melt_cross,
        "T_max_K": float(T.max()),
        "T_end_liquid_K": float(T[nve][-1]),
        "NVE_drift_solid_900K_meV_atom_ps": drift_solid,
        "NVE_drift_liquid_meV_atom_ps": drift_liquid,
        "diffusion_D_m2_s": D,
        "n_frames": int(nf),
        "n_rdf_frames": int(len(rdf_idx)),
    }
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
