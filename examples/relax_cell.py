"""Variable-cell relaxation on Tenstorrent — the end-to-end proof of the stress tensor.

An ``ase.filters.FrechetCellFilter`` lets FIRE relax the atomic positions *and* the unit cell
together; the cell degrees of freedom are driven by the stress the TT-Atom calculator now
exposes (virial = symmetrized ``dE/dstrain``, divided by volume — fairchem's convention). This
is UMA's flagship materials use case (NPT / variable-cell geometry optimization).

Point ``--weights`` at a bundle exported from the real uma-s-1 ``omat`` checkpoint to relax on
the physical potential (default: the periodic golden bundle if present, else the random-weight
demo bundle — which still converges, on its own arbitrary surface).

    ~/.ttatom_run/env/bin/python examples/relax_cell.py
"""
from __future__ import annotations

import argparse
import pathlib

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from tt_atom.calculator import TTAtomCalculator

HERE = pathlib.Path(__file__).parent
_OMAT = pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(_OMAT if _OMAT.exists()
                                             else HERE / "model_tiny_demo.npz"))
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    atoms = bulk("Si", "diamond", a=5.35) * (2, 1, 1)   # slightly compressed -> nonzero stress
    atoms.info.update(charge=0, spin=1)

    calc = TTAtomCalculator(args.weights, device_id=args.device_id)
    atoms.calc = calc
    try:
        e0 = atoms.get_potential_energy()
        v0 = atoms.get_volume()
        s0 = float((atoms.get_stress() ** 2).sum() ** 0.5)
        opt = FIRE(FrechetCellFilter(atoms), logfile="-")
        opt.run(fmax=args.fmax, steps=args.steps)
        e1 = atoms.get_potential_energy()
        v1 = atoms.get_volume()
        fmax = float((atoms.get_forces() ** 2).sum(1).max() ** 0.5)
        smax = float((atoms.get_stress() ** 2).sum() ** 0.5)
        print(f"\nvariable-cell relax: E {e0:.5f} -> {e1:.5f} eV, V {v0:.3f} -> {v1:.3f} A^3 "
              f"over {opt.nsteps} steps")
        print(f"  |stress| {s0:.4f} -> {smax:.4f} eV/A^3; fmax={fmax:.4f} "
              f"(target {args.fmax}); converged={fmax <= args.fmax}")
    finally:
        calc.close()


if __name__ == "__main__":
    main()
