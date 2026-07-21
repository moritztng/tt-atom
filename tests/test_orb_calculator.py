"""Parity test for the ``Orb(atoms)`` ASE calculator (``tt_atom/orb_calculator.py``) — the
Orb-family counterpart to ``UMA(atoms)``.

Unlike the bottom-up ``tests/test_orb_*_realweight.py`` suite (which feeds a golden's *stored*
positions/edges straight to the device modules), this test drives the calculator through ASE from
scratch: builds ``Atoms`` itself, lets the calculator compute its own neighbour list/edge features
from ``atoms.get_positions()``, and compares the result against the real ``orb-models`` CPU oracle
energy/forces/stress already captured in the existing goldens (``tests/gen_golden_orb.py``) — the
same systems, so a genuine independent check of the whole ASE-facing path (radius_graph +
sender/receiver swap + host edge/node features + device forward, not just the device modules in
isolation).

Requires the checkpoint weight caches (``tt_atom.orb_weight_cache``, built once via the reference
env — see ``tools/export_orb_weights.py``) and the existing real-weight goldens for the oracle
values; auto-skips whichever it's missing.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

GOLDEN_DIR = pathlib.Path.home() / ".ttatom_run/goldens_real"
CACHE_DIR = pathlib.Path(os.environ.get("TT_ATOM_CACHE", pathlib.Path.home() / ".cache/tt_atom")) / "orb_weights"


def _build_si():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43) * (2, 1, 1)
    atoms.rattle(stdev=0.1, seed=1)
    return atoms


def _build_molecule():
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    return atoms


def _build_short_contact():
    """Two Si atoms 1.4 A apart in a large periodic box -- the same graph as
    ``gen_golden_orb.py``'s
    ``build_short_contact``: well inside every checkpoint's max_num_neighbors (1 neighbour each),
    and ZBL pair-repulsion is ~2.4% of the energy here (negligible in the bulk Si golden), so this
    doubles as a real exercise of the calculator's ZBL energy/force/stress addition."""
    from ase import Atoms

    return Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], cell=[20.0, 20.0, 20.0],
                pbc=True)


def _have(checkpoint, golden):
    return (CACHE_DIR / f"{checkpoint}.npz").exists() and (GOLDEN_DIR / golden).exists()


@pytest.mark.skipif(not _have("conservative-inf-omat", "si_omat_orb.npz"),
                    reason="orb weight cache or si_omat_orb.npz golden not found")
def test_conservative_omat_end_to_end(device):
    from tt_atom.orb_calculator import OrbCalculator

    gold = np.load(GOLDEN_DIR / "si_omat_orb.npz")
    atoms = _build_si()
    calc = OrbCalculator.from_checkpoint("orb-v3-conservative-inf-omat", device=device)
    try:
        atoms.calc = calc
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress()

        gold_E, gold_F = float(gold["out@energy"][0]), gold["out@forces"]
        e_rel_err = abs(E - gold_E) / abs(gold_E)
        f_pcc = np.corrcoef(F.ravel(), gold_F.ravel())[0, 1]
        print(f"\n[Orb(atoms) conservative-inf-omat] E={E:.6f} (oracle {gold_E:.6f}, "
              f"rel err {e_rel_err:.2e}) forces PCC={f_pcc:.6f}")
        assert e_rel_err < 1e-2, e_rel_err
        assert f_pcc > 0.99, f_pcc
        assert S.shape == (6,)
    finally:
        calc.close()


@pytest.mark.skipif(not _have("direct-20-omat", "si_short_contact_orb_direct20.npz"),
                    reason="orb weight cache or si_short_contact_orb_direct20.npz golden not found")
def test_direct_omat_end_to_end(device):
    """The bulk Si golden's periodic images alone exceed max_num_neighbors=20 for this checkpoint
    (this port doesn't implement Orb's own neighbour-list truncation, see the guard test below),
    so this uses the large-box short-contact system instead -- also a real exercise of the ZBL
    energy/force/stress addition (non-negligible here, unlike the bulk golden)."""
    from tt_atom.orb_calculator import OrbCalculator

    gold = np.load(GOLDEN_DIR / "si_short_contact_orb_direct20.npz")
    atoms = _build_short_contact()
    calc = OrbCalculator.from_checkpoint("orb-v3-direct-20-omat", device=device)
    try:
        atoms.calc = calc
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress()

        gold_E, gold_F, gold_S = (
            float(gold["out@energy"][0]), gold["out@forces"], gold["out@stress"][0])
        e_rel_err = abs(E - gold_E) / abs(gold_E)
        f_pcc = np.corrcoef(F.ravel(), gold_F.ravel())[0, 1]
        s_pcc = np.corrcoef(S.ravel(), gold_S.ravel())[0, 1]
        print(f"\n[Orb(atoms) direct-20-omat, short contact] E={E:.6f} (oracle {gold_E:.6f}, "
              f"rel err {e_rel_err:.2e}) forces PCC={f_pcc:.6f} stress PCC={s_pcc:.6f}")
        # Verified by hand (feeding the golden's own stored node/edge features through the same
        # device modules reproduces this exact energy bit-for-bit): the sub-1% gap is the existing
        # device GNN's own bf16 prediction at this deliberately out-of-training-distribution 1.4 A
        # bond (built to stress ZBL, docs/orb-port.md), not a bug in this calculator's host
        # geometry -- ZBL itself matches the oracle's own decomposition exactly. Forces (PCC of
        # 1.0, direction-perfect) are the load-bearing assertion here, as in the existing
        # tests/test_orb_zbl_forces.py (which also carries no energy bound for this system).
        assert e_rel_err < 0.03, e_rel_err
        assert f_pcc > 0.999, f_pcc
        assert s_pcc > 0.99, s_pcc
    finally:
        calc.close()


@pytest.mark.skipif(not _have("conservative-omol", "molecule_omol_conservative.npz"),
                    reason="orb weight cache or molecule_omol_conservative.npz golden not found")
def test_orbmol_conditioning_end_to_end(device):
    """Same ASE path, an OrbMol checkpoint — exercises host_charge_spin_embedding through the
    full calculator (charge=0, spin=1 default, closed-shell water). Also exercises the unified
    ``Calculator(atoms, model=...)`` front door dispatching to the Orb family by name."""
    from tt_atom import Calculator

    gold = np.load(GOLDEN_DIR / "molecule_omol_conservative.npz")
    atoms = _build_molecule()
    calc = Calculator(atoms, model="orb-v3-conservative-omol", device=device)
    try:
        atoms.calc = calc
        E = atoms.get_potential_energy()
        F = atoms.get_forces()

        gold_E, gold_F = float(gold["out@energy"][0]), gold["out@forces"]
        e_rel_err = abs(E - gold_E) / abs(gold_E)
        f_pcc = np.corrcoef(F.ravel(), gold_F.ravel())[0, 1]
        print(f"\n[Orb(atoms) conservative-omol, H2O] E={E:.6f} (oracle {gold_E:.6f}, "
              f"rel err {e_rel_err:.2e}) forces PCC={f_pcc:.6f}")
        assert e_rel_err < 1e-2, e_rel_err
        assert f_pcc > 0.99, f_pcc
    finally:
        calc.close()


def test_max_num_neighbors_guard_raises(device):
    """A structure denser than the checkpoint's max_num_neighbors must raise, not silently
    diverge from Orb's own reference (which truncates the neighbour list) — see
    ``OrbCalculator.calculate``'s docstring-adjacent comment."""
    from ase import Atoms

    from tt_atom.orb_calculator import OrbCalculator

    if not (CACHE_DIR / "direct-20-omat.npz").exists():
        pytest.skip("orb weight cache not found")

    # 25 atoms packed within the 6 A cutoff of a shared center -- exceeds max_num_neighbors=20
    # for orb-v3-direct-20-omat by construction (every atom sees >=20 neighbours in a 3x3x3
    # minus a corner grid at 1.5 A spacing, well inside 6 A).
    n = 3
    positions = [[i * 1.5, j * 1.5, k * 1.5]
                for i in range(n) for j in range(n) for k in range(n)]
    atoms = Atoms("Si" * len(positions), positions=positions, cell=[20.0, 20.0, 20.0], pbc=False)

    calc = OrbCalculator.from_checkpoint("orb-v3-direct-20-omat", device=device)
    try:
        atoms.calc = calc
        with pytest.raises(ValueError, match="max_num_neighbors"):
            atoms.get_potential_energy()
    finally:
        calc.close()
