"""End-to-end ASE calculator test for the PET-MAD port — step 5's gate.

Drives the public ``Calculator(atoms, "pet-mad-s-v1.5.0")`` front door (``tt_atom.auto``)
on the 16-atom rattled Si golden and checks the reported energy + forces against the
pass-1 bit-exact golden. The energy comes from the device backbone (bf16, ~0.026 eV from
the host reference — asserted < 0.05 eV, same gate as ``tests/test_pet_device.py``); the
forces come from the host autograd finish (float32, PCC 1.0 / max abs ~1.7e-6 vs golden —
asserted PCC >= 0.999).

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest tests/test_pet_calculator.py -q
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

WEIGHTS = os.environ.get(
    "TTATOM_PET_WEIGHTS",
    str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz"),
)
GOLDEN = "tests/data/pet_mad_s_si_golden.npz"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(WEIGHTS).exists() or not pathlib.Path(GOLDEN).exists(),
    reason=f"PET weights ({WEIGHTS}) or golden ({GOLDEN}) not found",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def test_calculator_energy_and_forces(device):
    from ase import Atoms

    from tt_atom import Calculator

    fx = np.load(GOLDEN)
    atoms = Atoms(numbers=fx["numbers"], positions=fx["positions"],
                  cell=fx["cell"], pbc=fx["pbc"])
    atoms.calc = Calculator(atoms, "pet-mad-s-v1.5.0", device=device)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    ref_E = float(fx["energy"][0])
    ref_F = fx["forces"]
    dE = abs(E - ref_E)
    pcc = _pcc(F, ref_F)
    maxabs = float(np.abs(F - ref_F).max())
    print(f"\n[pet-calc] E={E:.6f} eV (ref {ref_E:.6f}, diff {dE:.6f})")
    print(f"[pet-calc] forces PCC={pcc:.8f} max abs={maxabs:.3e} (ref max abs {np.abs(ref_F).max():.3f})")
    assert dE < 0.05, dE
    assert pcc > 0.999, pcc
