"""Reusable ASE relax / MD loops — the single source both the single-card CLI path and the
multi-card fanout workers call.

Factored out of ``tt_atom.cli`` so ``cmd_run`` / ``cmd_relax`` / ``cmd_md`` (one card, one
structure) and ``tt_atom.batch.MultiCardSim`` (N cards, a batch of independent structures)
run the identical FIRE / Langevin loop — per-structure parity is bit-exact by construction,
not by coincidence. Each function attaches nothing to the caller's environment beyond what
ASE itself does; the caller is responsible for the calculator lifecycle (``atoms.calc``).
"""
from __future__ import annotations

import numpy as np


def relax_atoms(atoms, *, fmax=0.05, steps=200, logfile=None):
    """Run a FIRE geometry relaxation in place; return the final-state dict.

    ``logfile`` is ASE's optimizer log sink (``"-"`` for stdout, ``None`` to silence, a
    file path / handle otherwise). The returned dict carries the final energy, forces,
    achieved fmax, step count, and convergence flag — everything the single-card CLI prints
    and everything a multi-card worker needs to ship back over its IPC queue."""
    from ase.optimize import FIRE

    opt = FIRE(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)
    E = float(atoms.get_potential_energy())
    F = np.asarray(atoms.get_forces(), dtype=np.float64)
    fmax_actual = float((F ** 2).sum(1).max() ** 0.5)
    return dict(energy=E, forces=F, fmax=fmax_actual, nsteps=int(opt.nsteps),
                converged=bool(fmax_actual <= fmax))


def md_atoms(atoms, *, steps=100, dt=1.0, temp=300.0, logfile=None, seed=None):
    """Run Langevin MD in place; return the final-state dict.

    Velocities are drawn from a Maxwell-Boltzmann distribution at ``temp`` before the run
    (ASE's convention). ``seed`` (optional) makes the velocity draw reproducible across
    single-card and multi-card runs of the same structure — without it, MD trajectories
    diverge by RNG draw, not by numerics, so a bit-exact parity gate would be meaningless.
    The CLI leaves it ``None`` (one run); the multi-card parity harness pins it so every
    worker draws the same velocities the single-card reference did."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    if seed is not None:
        # Pin BOTH the explicit default_rng (for the Maxwell-Boltzmann velocity draw) and the
        # global numpy RNG (for ASE Langevin's per-step stochastic kicks, which draw from
        # np.random, not from a passed generator). Without the global seed, two runs of the
        # same structure in separate processes diverge after step 1 — not a numerics bug, an
        # ASE API gap (Langevin exposes no rng= kwarg on this version). The CLI leaves seed
        # None (real MD is stochastic); the parity harness / reproducible runs pin it.
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp, rng=rng)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=temp, friction=0.01 / units.fs,
                  logfile=logfile)
    if logfile is not None:
        def _log():
            ekin = atoms.get_kinetic_energy()
            print(f"  step {dyn.nsteps:4d}  E={atoms.get_potential_energy():.5f}  "
                  f"T={ekin / (1.5 * units.kB * len(atoms)):.1f} K")

        dyn.attach(_log, interval=max(1, steps // 10))
    dyn.run(steps)
    E = float(atoms.get_potential_energy())
    F = np.asarray(atoms.get_forces(), dtype=np.float64)
    return dict(energy=E, forces=F, nsteps=int(dyn.nsteps), temp=float(temp),
                dt=float(dt))
