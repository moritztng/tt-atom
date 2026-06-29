"""The ASE calculator runs, gives finite energy/forces, and a relaxation converges on device."""
import pathlib

import numpy as np
from ase.build import molecule
from ase.optimize import FIRE

from tt_atom.calculator import TTAtomCalculator
from tt_atom.weights import WeightBundle

BUNDLE = pathlib.Path(__file__).parent.parent / "examples" / "model_tiny_demo.npz"


def test_coverage():
    ok, missing, n = WeightBundle.load(str(BUNDLE)).verify_coverage()
    assert ok, f"missing weight keys: {missing}"
    assert n > 0


def test_calculator_relaxation(device):
    calc = TTAtomCalculator(str(BUNDLE), device=device)
    atoms = molecule("CH3CH2OH")
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    atoms.rattle(stdev=0.05, seed=0)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    assert np.isfinite(e0)
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=200)
    fmax = float((atoms.get_forces() ** 2).sum(1).max() ** 0.5)
    e1 = atoms.get_potential_energy()
    assert fmax <= 0.05, f"did not converge, fmax={fmax}"
    assert e1 <= e0 + 1e-3, f"energy increased: {e0} -> {e1}"
