"""Trace path parity: the device-resident, trace-captured engine must return exactly the eager
energy+forces (it only removes host dispatch, never changes the math). Uses the committed
random-weight demo bundle so it runs without a UMA checkpoint.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from ase.build import molecule

from tt_atom.calculator import TTAtomCalculator

DEMO = str(pathlib.Path(__file__).parent.parent / "examples" / "model_tiny_demo.npz")


def _pcc(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def test_traced_matches_eager(device):
    """Traced calculator == eager calculator on energy and forces, over several geometries
    (including a re-capture triggered by moving atoms)."""
    eager = TTAtomCalculator(DEMO, device=device)
    traced = TTAtomCalculator(DEMO, device=device, trace=True)
    try:
        rng = np.random.default_rng(0)
        for k in range(3):
            atoms = molecule("CH3CH2OH")
            atoms.info.update(charge=0, spin=0)
            atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
            atoms.calc = eager
            Ee, Fe = atoms.get_potential_energy(), atoms.get_forces()
            atoms.calc = traced
            Et, Ft = atoms.get_potential_energy(), atoms.get_forces()
            assert abs(Et - Ee) < 1e-4, f"step {k}: energy {Et} vs {Ee}"
            assert _pcc(Ft, Fe) > 0.9999, f"step {k}: force PCC {_pcc(Ft, Fe)}"
            assert np.abs(Ft - Fe).max() < 1e-3, f"step {k}: max force diff {np.abs(Ft - Fe).max()}"
    finally:
        traced.close()
