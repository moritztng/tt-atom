"""Periodic materials (PBC) on Tenstorrent via the TT-Atom ASE calculator.

Energy + forces (and an optional cell-fixed relaxation) for a bulk crystal. TT-Atom's cell-aware
neighbour list (minimum-image, matching fairchem ``radius_graph_pbc``) makes materials tasks
work; validated against the fairchem uma-s-1 oracle on bulk Si (omat) to energy rel < 1e-3 and
force PCC > 0.99. Use a bundle exported with ``--task omat`` (or oc20/odac/omc) for real numbers;
the shipped demo bundle is random-weight but still exercises the full periodic path.

    ~/.ttatom_run/env/bin/python examples/periodic.py --weights si_omat.npz
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from ase.build import bulk
from ase.optimize import FIRE

from tt_atom.calculator import TTAtomCalculator

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE / "model_tiny_demo.npz"))
    ap.add_argument("--relax", action="store_true", help="run a FIRE relaxation (atoms only)")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    atoms = bulk("Si", "diamond", a=5.43) * (2, 1, 1)     # 4-atom periodic cell
    atoms.rattle(stdev=0.1, seed=1)
    atoms.info.update(charge=0, spin=1)
    print(f"periodic system: {atoms.get_chemical_formula()}, pbc={atoms.get_pbc().tolist()}, "
          f"cell diag={np.round(atoms.cell.lengths(), 3).tolist()}")

    calc = TTAtomCalculator(args.weights, device_id=args.device_id)
    atoms.calc = calc
    try:
        E = atoms.get_potential_energy()
        fmax = float((atoms.get_forces() ** 2).sum(1).max() ** 0.5)
        print(f"energy = {E:.6f} eV   |F|max = {fmax:.4f} eV/A")
        if args.relax:
            FIRE(atoms, logfile="-").run(fmax=0.05, steps=100)
            print(f"relaxed energy = {atoms.get_potential_energy():.6f} eV")
    finally:
        calc.close()


if __name__ == "__main__":
    main()
