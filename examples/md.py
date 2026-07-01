"""Molecular dynamics on Tenstorrent via the TT-Atom ASE calculator.

Runs Langevin MD with conservative analytic forces from the device. ``--trace`` captures the
device forward+backward once and replays it each step (fixed topology) for ~2x fewer host
dispatches — the forces are bit-for-bit identical to the eager path. The shipped demo bundle is
random-weight (arbitrary surface); point ``--weights`` at a bundle exported from a real UMA
checkpoint (``tools/export_weights.py`` / ``tt-atom convert-checkpoint``) for real dynamics.

    ~/.ttatom_run/env/bin/python examples/md.py --trace
"""
from __future__ import annotations

import argparse
import pathlib

from ase import units
from ase.build import molecule
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from tt_atom.calculator import TTAtomCalculator

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "model_tiny_demo.npz"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.5, help="timestep (fs)")
    ap.add_argument("--temp", type=float, default=300.0, help="temperature (K)")
    ap.add_argument("--trace", action="store_true", help="trace-captured device loop (~2x)")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    atoms = molecule("CH3CH2OH")
    atoms.info.update(charge=0, spin=0)
    calc = TTAtomCalculator(args.weights, device_id=args.device_id, trace=args.trace)
    atoms.calc = calc
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)
        dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.temp,
                       friction=0.01 / units.fs)

        def _log():
            ekin = atoms.get_kinetic_energy()
            print(f"step {dyn.nsteps:4d}  E={atoms.get_potential_energy():.5f} eV  "
                  f"T={ekin / (1.5 * units.kB * len(atoms)):.1f} K")

        dyn.attach(_log, interval=max(1, args.steps // 10))
        e0 = atoms.get_potential_energy()
        dyn.run(args.steps)
        print(f"\nMD: {args.steps} steps x {args.dt} fs at {args.temp} K; "
              f"E {e0:.5f} -> {atoms.get_potential_energy():.5f} eV  (trace={args.trace})")
    finally:
        calc.close()


if __name__ == "__main__":
    main()
