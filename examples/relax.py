"""Geometry relaxation on Tenstorrent via the TT-Atom ASE calculator.

Runs a real FIRE optimization to convergence on device. The shipped demo bundle holds
*random* weights (the eSCN-MD architecture, no UMA checkpoint), so the energy surface is
arbitrary — but the forces are the exact analytic gradient of that energy, so the relaxation
genuinely converges. Point ``--weights`` at a bundle exported from a fairchem checkpoint
(``tools/export_weights.py``) to relax on the real potential.

    ~/.ttatom_run/env/bin/python examples/relax.py
"""
from __future__ import annotations

import argparse
import pathlib

from ase.build import molecule
from ase.optimize import FIRE

from tt_atom.calculator import TTAtomCalculator

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "model_tiny_demo.npz"))
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    atoms = molecule("CH3CH2OH")          # ethanol
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    atoms.rattle(stdev=0.05, seed=0)      # perturb so there is something to relax

    calc = TTAtomCalculator(args.weights, device_id=args.device_id)
    atoms.calc = calc
    try:
        e0 = atoms.get_potential_energy()
        opt = FIRE(atoms, logfile="-")
        opt.run(fmax=args.fmax, steps=args.steps)
        e1 = atoms.get_potential_energy()
        fmax = float((atoms.get_forces() ** 2).sum(1).max() ** 0.5)
        print(f"\nrelaxation: E {e0:.6f} -> {e1:.6f} eV over {opt.nsteps} steps; "
              f"fmax={fmax:.4f} (target {args.fmax}); converged={fmax <= args.fmax}")
    finally:
        calc.close()


if __name__ == "__main__":
    main()
