"""Parity test for ``OrbCalculator.evaluate_batch`` -- the disjoint-union batched calculator method
(``tt_atom/orb_calculator.py``), Orb-family counterpart to ``tests/test_batch.py``.

The load-bearing claim is the same as UMA batching's: evaluating K systems as one block-diagonal
graph gives the *same* per-system energies and forces as evaluating each system separately. The
Orb backbone (Encoder + AttentionInteractionLayer) was already verified batch-transparent
(``test_orb_disjoint_batch.py``, bit-exact row-independence); this test closes the loop through the
ASE-facing calculator -- assembly + per-system denormalize + ZBL + the two force paths
(conservative VJP and direct ForceHead with per-system net-force removal) -- against the
single-system ``calculate`` loop on the real checkpoints.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_evaluate_batch.py -q -s

Auto-skips whichever checkpoint cache / golden is missing.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

GOLDEN_DIR = pathlib.Path.home() / ".ttatom_run/goldens_real"
CACHE_DIR = pathlib.Path(os.environ.get("TT_ATOM_CACHE", pathlib.Path.home() / ".cache" / "tt_atom")) / "orb_weights"


def _have(checkpoint, golden):
    return (CACHE_DIR / f"{checkpoint}.npz").exists() and (GOLDEN_DIR / golden).exists()


def _si_systems(k, jitter):
    from ase.build import bulk

    a0 = bulk("Si", "diamond", a=5.43) * (2, 1, 1)
    a0.rattle(stdev=0.1, seed=1)
    out = []
    for i in range(k):
        a = a0.copy()
        if i:
            a.rattle(stdev=jitter, seed=100 + i)
        out.append(a)
    return out


def _short_contact_systems(k, jitter):
    from ase import Atoms

    base = [0.0, 0.0, 0.0, 1.4, 0.0, 0.0]
    out = []
    for i in range(k):
        pos = np.array(base, dtype=float).reshape(2, 3)
        if i:
            rng = np.random.default_rng(100 + i)
            pos = pos + jitter * rng.standard_normal(pos.shape)
        out.append(Atoms("Si2", positions=pos, cell=[20.0, 20.0, 20.0], pbc=False))
    return out


def _water_systems(k, jitter):
    from ase.build import molecule

    out = []
    for i in range(k):
        a = molecule("H2O")
        a.info.update(charge=0, spin=1)
        if i:
            a.rattle(stdev=jitter, seed=100 + i)
        out.append(a)
    return out


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _check_evaluate_batch_vs_loop(calc, systems):
    """evaluate_batch([s0..sK-1]) == looping calculate(s_k) per-system: energies and forces."""
    out = calc.evaluate_batch(systems)
    E_batch = out["energy"]
    F_batch = out["forces"]
    assert len(E_batch) == len(systems)
    assert len(F_batch) == len(systems)

    E_loop, F_loop = [], []
    for s in systems:
        s.calc = calc
        E_loop.append(s.get_potential_energy())
        F_loop.append(s.get_forces())
    E_loop = np.array(E_loop)
    F_concat = np.concatenate(F_loop, axis=0)
    F_batch_concat = np.concatenate(F_batch, axis=0)

    e_max = np.abs(E_batch - E_loop).max()
    e_rel = e_max / (np.abs(E_loop).max() + 1e-9)
    f_pcc = _pcc(F_batch_concat, F_concat)
    f_max = np.abs(F_batch_concat - F_concat).max()
    f_rel = f_max / (np.abs(F_concat).max() + 1e-9)
    print(f"\n[orb evaluate_batch] K={len(systems)} E maxdiff={e_max:.3e} rel={e_rel:.2e} "
          f"F PCC={f_pcc:.6f} maxdiff={f_max:.3e} rel={f_rel:.2e}")
    # Same bf16-reduction-order tolerance as test_orb_disjoint_batch / test_batch: block-diagonal
    # => within-system only, but the batched forward/backward accumulates over a larger Etot so the
    # last bf16 bit differs. PCC is the load-bearing correctness gate (forces are the same vector,
    # bf16-rounded); the relative maxdiff bar bounds the backward's accumulated rounding to a few
    # ULPs -- ~1 ULP at the force magnitude (e.g. 0.5 / 60 eV/A on the ZBL-dominated 1.4 A
    # short-contact), verified by identical-copy runs giving exactly that 1-ULP gap.
    assert e_rel < 1e-2, f"energy rel err {e_rel:.2e} (E_batch={E_batch[:3]} vs E_loop={E_loop[:3]})"
    assert f_pcc > 0.999, f"force PCC {f_pcc:.6f}"
    assert f_rel < 5e-2, f"force rel maxdiff {f_rel:.2e} (abs {f_max:.3e} on |F|={np.abs(F_concat).max():.3e})"


@pytest.mark.skipif(not _have("conservative-inf-omat", "si_omat_orb.npz"),
                    reason="orb weight cache or si_omat_orb.npz golden not found")
def test_evaluate_batch_conservative_omat(device):
    from tt_atom.orb_calculator import OrbCalculator

    calc = OrbCalculator.from_checkpoint("orb-v3-conservative-inf-omat", device=device)
    try:
        _check_evaluate_batch_vs_loop(calc, _si_systems(4, jitter=0.05))
    finally:
        calc.close()


@pytest.mark.skipif(not (CACHE_DIR / "direct-20-omat.npz").exists(),
                    reason="orb weight cache not found")
def test_evaluate_batch_direct_omat(device):
    """direct-20-omat: bulk Si is too dense for max_num_neighbors=20, so use the aperiodic
    short-contact system. Exercises the ForceHead.batch per-system net-force removal path."""
    from tt_atom.orb_calculator import OrbCalculator

    calc = OrbCalculator.from_checkpoint("orb-v3-direct-20-omat", device=device)
    try:
        _check_evaluate_batch_vs_loop(calc, _short_contact_systems(4, jitter=0.03))
    finally:
        calc.close()


@pytest.mark.skipif(not _have("conservative-omol", "molecule_omol_conservative.npz"),
                    reason="orb weight cache or molecule_omol_conservative.npz golden not found")
def test_evaluate_batch_conservative_omol(device):
    """OrbMol conditioning through the batched path: per-system charge/spin embedding (here all
    charge=0/spin=1, the water default) concatenated and uploaded once."""
    from tt_atom.orb_calculator import OrbCalculator

    calc = OrbCalculator.from_checkpoint("orb-v3-conservative-omol", device=device)
    try:
        _check_evaluate_batch_vs_loop(calc, _water_systems(4, jitter=0.03))
    finally:
        calc.close()


@pytest.mark.skipif(not (CACHE_DIR / "conservative-inf-omat.npz").exists(),
                    reason="orb weight cache not found")
def test_evaluate_batch_mixes_compositions(device):
    """Orb has no per-composition MoLE routing (unlike UMA), so a batch that mixes compositions is
    valid -- energies/forces come out for each system's own composition. Sanity: run a mixed batch
    (Si2 short-contact + bulk Si) and check the per-system energies are finite and distinct."""
    from tt_atom.orb_calculator import OrbCalculator

    calc = OrbCalculator.from_checkpoint("orb-v3-conservative-inf-omat", device=device)
    try:
        systems = _short_contact_systems(2, jitter=0.0) + _si_systems(2, jitter=0.0)
        out = calc.evaluate_batch(systems, properties=("energy",))
        assert np.all(np.isfinite(out["energy"]))
        assert out["forces"] is None
        # the two system families are genuinely different geometries -> different energies
        assert len(set(np.round(out["energy"], 4).tolist())) > 1
    finally:
        calc.close()


def test_evaluate_batch_max_num_neighbors_guard(device):
    """A denser system inside a batch must raise the same clear error as the single-system path,
    not silently exceed the cap or get truncated."""
    from ase import Atoms

    from tt_atom.orb_calculator import OrbCalculator

    if not (CACHE_DIR / "direct-20-omat.npz").exists():
        pytest.skip("orb weight cache not found")

    n = 3
    dense = Atoms("Si" * 27, positions=[[i * 1.5, j * 1.5, k * 1.5]
                                        for i in range(n) for j in range(n) for k in range(n)],
                 cell=[20.0, 20.0, 20.0], pbc=False)
    sparse = _short_contact_systems(1, jitter=0.0)[0]
    calc = OrbCalculator.from_checkpoint("orb-v3-direct-20-omat", device=device)
    try:
        with pytest.raises(ValueError, match="max_num_neighbors"):
            calc.evaluate_batch([sparse, dense])
    finally:
        calc.close()
