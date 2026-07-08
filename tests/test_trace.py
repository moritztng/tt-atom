"""Trace path parity: the device-resident, trace-captured engine must return exactly the eager
energy+forces (it only removes host dispatch, never changes the math). Uses the committed
random-weight demo bundle so it runs without a UMA checkpoint.
"""
from __future__ import annotations

import pathlib

import numpy as np
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


def test_traced_recaptures_on_cell_shift_change(device):
    """Regression: the traced engine bakes in ``edge_cell_shift``, so a changing cell at *fixed*
    edge topology (the NPT / variable-cell step that doesn't cross the cutoff) must trigger a
    re-capture — else the replay reuses the stale shift and returns wrong forces. Two cells that
    yield the SAME edge_index but different shifts must both match eager."""
    import torch
    from ase import Atoms

    from tt_atom.geometry import radius_graph

    eager = TTAtomCalculator(DEMO, device=device)
    traced = TTAtomCalculator(DEMO, device=device, trace=True)
    cut = eager.cfg["cutoff"]

    def _atoms(cx):
        a = Atoms(numbers=[6, 8], positions=[[0, 0, 0], [2.0, 0, 0]],
                  cell=[cx, 40.0, 40.0], pbc=[True, False, False])
        a.info.update(charge=0, spin=0)
        return a

    try:
        # the two cells must share edge topology but differ in cell shift (else the test would be
        # trivially satisfied by the edge-index re-capture that already existed)
        pbc = [True, False, False]
        ei1, sh1 = radius_graph(torch.tensor(_atoms(5.1).get_positions(), dtype=torch.float32), cut,
                                cell=torch.tensor(np.asarray(_atoms(5.1).get_cell()), dtype=torch.float32),
                                pbc=pbc)
        ei2, sh2 = radius_graph(torch.tensor(_atoms(5.2).get_positions(), dtype=torch.float32), cut,
                                cell=torch.tensor(np.asarray(_atoms(5.2).get_cell()), dtype=torch.float32),
                                pbc=pbc)
        assert torch.equal(ei1, ei2) and not torch.equal(sh1, sh2), "setup: need same edges, diff shift"

        for cx in (5.1, 5.2):                          # capture at 5.1, then shift-only change to 5.2
            a = _atoms(cx)
            a.calc = eager
            Ee, Fe = a.get_potential_energy(), a.get_forces()
            a.calc = traced
            Et, Ft = a.get_potential_energy(), a.get_forces()
            assert abs(Et - Ee) < 1e-4, f"cx={cx}: energy {Et} vs {Ee}"
            assert _pcc(Ft, Fe) > 0.9999, f"cx={cx}: force PCC {_pcc(Ft, Fe)}"
            assert np.abs(Ft - Fe).max() < 1e-3, f"cx={cx}: max force diff {np.abs(Ft - Fe).max()}"
    finally:
        traced.close()


def test_traced_stress_falls_back_to_eager(device):
    """trace=True must still deliver stress (via the eager fallback) instead of silently dropping
    it — else an ASE variable-cell relaxation with trace=True would hit PropertyNotImplementedError."""
    from ase import Atoms

    a = Atoms(numbers=[6, 8], positions=[[0, 0, 0], [2.0, 0, 0]],
              cell=[5.1, 40.0, 40.0], pbc=[True, False, False])
    a.info.update(charge=0, spin=0)
    eager = TTAtomCalculator(DEMO, device=device)
    traced = TTAtomCalculator(DEMO, device=device, trace=True)
    try:
        a.calc = eager
        s_e, F_e = a.get_stress(), a.get_forces()
        a.calc = traced
        s_t, F_t = a.get_stress(), a.get_forces()
        assert s_t is not None and s_t.shape == (6,)
        assert np.abs(s_t - s_e).max() < 1e-4, f"stress {s_t} vs {s_e}"
        assert _pcc(F_t, F_e) > 0.9999, f"force PCC {_pcc(F_t, F_e)}"
    finally:
        traced.close()
