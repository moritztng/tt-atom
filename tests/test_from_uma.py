"""Tests for the ``from_uma`` auto-bundle factory + composition cache (tt_atom.bundle_cache).

Split into two tiers:
  * pure cache logic (composition hashing, cache-path shape, refenv resolution/error) — always run,
    no device / no fairchem;
  * device tiers gated on availability: the cached fast path + factory-vs-direct parity use a real
    uma-s-1 golden bundle (skip cleanly if absent), and the auto-build-vs-manual-export parity uses
    the reference (fairchem) env + the gated UMA checkpoint (skip cleanly if either is absent).

Nothing here commits or requires new weights; it reuses the out-of-repo real goldens.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from tt_atom import bundle_cache as BC

REAL_GOLDEN = pathlib.Path(
    os.environ.get("TTATOM_REAL_GOLDEN", pathlib.Path.home() / ".ttatom_run/goldens_real/ethanol_omol.npz")
)
HF_CKPT = next(iter(glob.glob(str(pathlib.Path.home() / ".cache/huggingface/**/uma-s-1.pt"),
                              recursive=True)), None)


def _default_refenv():
    p = pathlib.Path.home() / ".ttatom_run/refenv/bin/python"
    return str(p) if p.exists() else None


# ------------------------------------------------------------------ pure cache logic (no device)

def test_reduced_composition_is_scale_invariant():
    from ase import Atoms

    h2o = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    h4o2 = Atoms("H4O2", positions=[[i, 0, 0] for i in range(6)])
    assert BC.reduced_composition(h2o.numbers) == ((1, 2), (8, 1))
    assert BC.reduced_composition(h2o.numbers) == BC.reduced_composition(h4o2.numbers)
    assert BC.composition_hash(h2o.numbers) == BC.composition_hash(h4o2.numbers)
    assert BC.formula(h2o.numbers) == "H2O"


def test_composition_hash_is_order_independent():
    a = [6, 1, 1, 1, 1]       # CH4
    b = [1, 1, 6, 1, 1]       # same atoms, permuted
    assert BC.composition_hash(a) == BC.composition_hash(b)
    # a genuinely different composition hashes differently
    assert BC.composition_hash([6, 1, 1, 1, 1]) != BC.composition_hash([6, 1, 1, 1])


def test_bundle_path_shape(tmp_path):
    p = BC.bundle_path("uma-s-1", "omol", [1, 1, 8], charge=0, spin=1, cache_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name.startswith("uma-s-1_omol_") and p.name.endswith("_c0_s1.npz")
    # charge/spin land in the name so distinct charge states never collide
    p2 = BC.bundle_path("uma-s-1", "omol", [1, 1, 8], charge=-1, spin=2, cache_dir=tmp_path)
    assert p2.name.endswith("_c-1_s2.npz") and p2 != p


def test_resolve_refenv_positive_when_default_present():
    if _default_refenv() is None and not os.environ.get("TT_ATOM_REFENV"):
        pytest.skip("no reference env installed on this machine")
    assert pathlib.Path(BC.resolve_refenv()).exists()


def test_resolve_refenv_errors_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_ATOM_REFENV", raising=False)
    monkeypatch.setattr(BC.pathlib.Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(RuntimeError) as ei:
        BC.resolve_refenv()
    msg = str(ei.value)
    assert "TT_ATOM_REFENV" in msg and "refenv" in msg  # actionable, names the knobs


def test_from_uma_requires_atoms():
    from tt_atom import TTAtomCalculator

    with pytest.raises(ValueError, match="needs `atoms`"):
        TTAtomCalculator.from_uma(atoms=None)


def test_infer_task_from_periodicity():
    from ase import Atoms
    from ase.build import molecule

    assert BC.infer_task(molecule("H2O")) == "omol"                 # aperiodic -> molecules
    bulk = Atoms("Si2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[5.4] * 3, pbc=True)
    assert BC.infer_task(bulk) == "omat"                            # fully periodic -> materials


def test_uma_is_exported_and_delegates():
    import tt_atom

    assert "UMA" in tt_atom.__all__
    assert callable(tt_atom.UMA)


# ------------------------------------------------------------------ device: cached fast path + parity

real_golden = pytest.mark.skipif(
    not REAL_GOLDEN.exists(),
    reason=f"real uma-s-1 golden not found at {REAL_GOLDEN} (UMA checkpoint not available)",
)


def _atoms_from_golden(d):
    from ase import Atoms

    numbers = d["in@atomic_numbers"]
    pos = d["in@pos"]
    atoms = Atoms(numbers=numbers, positions=pos)
    atoms.info["charge"] = float(d["in@charge"][0])
    atoms.info["spin"] = float(d["in@spin"][0])
    return atoms


@real_golden
def test_cached_fast_path_needs_no_refenv_and_matches_direct(tmp_path, device, monkeypatch):
    """A cache hit loads without fairchem and yields exactly the same energy as constructing the
    calculator directly from the same bundle — the factory-vs-direct parity claim."""
    from tt_atom import TTAtomCalculator

    d = np.load(REAL_GOLDEN)
    task = json.loads(bytes(d["config"]).decode())["task"]
    atoms = _atoms_from_golden(d)
    charge, spin = atoms.info["charge"], atoms.info["spin"]

    # seed the cache: copy the golden to exactly the path from_uma will compute for this system
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    target = BC.bundle_path("uma-s-1", task, atoms.numbers, charge, spin, cache_dir=cache_dir)
    shutil.copyfile(REAL_GOLDEN, target)

    # point the refenv at nothing: if the fast path tried to build, this would raise
    monkeypatch.setenv("TT_ATOM_REFENV", "/nonexistent/python")

    calc = TTAtomCalculator.from_uma(task_name=task, atoms=atoms, charge=charge, spin=spin,
                                     refenv="/nonexistent/python", cache_dir=str(cache_dir),
                                     device=device)
    atoms.calc = calc
    e_factory = atoms.get_potential_energy()

    direct = TTAtomCalculator(str(target), device=device)
    a2 = _atoms_from_golden(d)
    a2.calc = direct
    e_direct = a2.get_potential_energy()

    assert e_factory == pytest.approx(e_direct, abs=1e-6), f"{e_factory} vs {e_direct}"

    # the zero-config UMA(atoms) face must reach the identical result (task inferred = omol here)
    if task == "omol":
        from tt_atom import UMA

        a3 = _atoms_from_golden(d)
        a3.calc = UMA(a3, refenv="/nonexistent/python", cache_dir=str(cache_dir), device=device)
        assert a3.get_potential_energy() == pytest.approx(e_direct, abs=1e-6)


refenv_and_ckpt = pytest.mark.skipif(
    _default_refenv() is None or HF_CKPT is None,
    reason="reference (fairchem) env and/or UMA checkpoint not available for a live merge",
)


@refenv_and_ckpt
def test_autobuild_matches_manual_export(tmp_path):
    """from_uma's transparent subprocess build produces a bundle numerically identical to a
    hand-run tools/export_weights.py merge on the same structure (no device needed)."""
    from ase.build import molecule
    from ase.io import write

    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)

    # (a) auto path: the factory's own build machinery on a fresh cache
    cache_dir = tmp_path / "cache"
    auto = BC.get_or_build(atoms, model="uma-s-1", task="omol", charge=0, spin=1,
                           cache_dir=cache_dir, log=False)

    # (b) manual path: invoke the exporter exactly as a user would in the reference env
    xyz = tmp_path / "h2o.xyz"
    write(str(xyz), atoms)
    manual = tmp_path / "manual.npz"
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    tools = pathlib.Path(BC.__file__).resolve().parent.parent / "tools" / "export_weights.py"
    subprocess.run([_default_refenv(), str(tools), "--uma-s-1", "--xyz", str(xyz),
                    "--task", "omol", "--charge", "0", "--spin", "1", "--out", str(manual)],
                   check=True, env=env)

    da, dm = np.load(auto), np.load(manual)
    assert json.loads(bytes(da["config"]).decode()) == json.loads(bytes(dm["config"]).decode())
    payload = [k for k in da.files if k.startswith(("w@", "scale@", "host@"))]
    assert payload, "no weight/scale/buffer arrays in the built bundle"
    for k in payload:
        assert k in dm.files, f"auto bundle has {k}, manual does not"
        assert np.allclose(da[k], dm[k], atol=0, rtol=0), f"array {k} differs auto vs manual"
