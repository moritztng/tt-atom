"""TensorNetCalculator on one chip against the float64 backend, from a saved bundle (no matgl): python
tensornet_calc_check.py BUNDLE.pt [cells]. Prints signed dE (meV/atom) and force MAE (meV/A) on rattled diamond Si."""
import json
import sys

import numpy as np
import torch
from ase.build import bulk

from tt_atom.tensornet import potential as P
from tt_atom.tensornet_calculator import TensorNetCalculator


def main():
    bundle, n = torch.load(sys.argv[1], weights_only=True), int(sys.argv[2]) if len(sys.argv) > 2 else 3
    torch.set_num_threads(1)
    calc, out = TensorNetCalculator(bundle), {}
    for rattle in (0.01, 0.1):
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (n, n, n)
        atoms.positions += np.random.default_rng(0).normal(0, rattle, atoms.positions.shape)
        e0, f0 = P.evaluate(bundle, atoms)
        atoms.calc = calc
        e1, f1 = atoms.get_potential_energy(), atoms.get_forces()
        out[f"rattle_{rattle}"] = {"atoms": len(atoms), "dE_meV_per_atom": 1e3 * (e1 - e0) / len(atoms),
                                   "force_mae_meV_A": 1e3 * float(np.abs(f1 - f0).mean()), "mean_abs_force_eV_A": float(np.abs(f0).mean())}
    calc.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
