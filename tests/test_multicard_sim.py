"""No-device unit tests for the multi-card relax/MD fan-out plumbing.

Covers the pieces that don't need a card: the system-dict normalization, the --devices
parser, the MultiCardSim sim_params validation, and the cmd_run single-card-vs-fanout
branching decision. The on-device bit-exact parity is exercised by the sibling
``_multicard_sim_parity.py`` harness (Orb, stock ttnn); these tests guard the plumbing so a
regression in the dispatch logic is caught without a card.
"""
import argparse

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from tt_atom.batch import MultiCardSim, _to_system_dict
from tt_atom.cli import _parse_devices


def test_parse_devices():
    assert _parse_devices("0") == (0,)
    assert _parse_devices("0,1,2,3") == (0, 1, 2, 3)
    assert _parse_devices(" 1 , 3 ") == (1, 3)
    assert _parse_devices("") == ()
    with pytest.raises(SystemExit):
        _parse_devices("0,a,2")
    with pytest.raises(SystemExit):
        _parse_devices("0,0")          # two workers pinned to one card would contend for it


def test_to_system_dict_from_atoms():
    a = molecule("H2O")
    a.info.update(charge=-1, spin=2)
    idx, d = _to_system_dict(a, 0)
    assert idx == 0
    assert d["pos"].shape == (3, 3)
    assert sorted(d["Z"].tolist()) == [1, 1, 8]   # order is ASE's, not assumed
    assert d["charge"] == -1.0
    assert d["spin"] == 2.0
    assert d["cell"] is None and d["pbc"] is False


def test_to_system_dict_from_tuple():
    pos = np.zeros((4, 3), dtype=np.float32)
    Z = np.array([1, 1, 1, 1], dtype=np.int64)
    idx, d = _to_system_dict((pos, Z), 7)
    assert idx == 7
    assert d["pos"].shape == (4, 3)
    assert d["Z"].tolist() == [1, 1, 1, 1]
    assert "charge" not in d or d.get("charge") is None or d.get("charge", 0.0) == 0.0


def test_to_system_dict_passes_dict_through():
    src = dict(pos=np.zeros((2, 3)), Z=np.array([1, 1]), charge=1.0, spin=0.0)
    idx, d = _to_system_dict(src, 3)
    assert idx == 3
    assert d is src


def test_to_system_dict_periodic():
    a = Atoms("Si2", positions=np.zeros((2, 3)), cell=np.eye(3) * 5.0, pbc=True)
    idx, d = _to_system_dict(a, 0)
    assert d["pbc"] is True
    assert d["cell"] is not None and d["cell"].shape == (3, 3)


def test_multicardsim_rejects_bad_sim_params():
    with pytest.raises(ValueError):
        MultiCardSim("orb-v3-conservative-omol", sim_params=None)
    with pytest.raises(ValueError):
        MultiCardSim("orb-v3-conservative-omol", sim_params={})
    with pytest.raises(ValueError):
        MultiCardSim("orb-v3-conservative-omol", sim_params=dict(mode="singlepoint"))


def test_cmd_run_branches_to_multicard(monkeypatch):
    """cmd_run must take the multi-card fan-out path iff --devices is set and >1 structure
    or >1 device is in play; otherwise the original single-card path. We mock both paths and
    the structure-file read so no device is opened and no file is needed."""
    import tt_atom.cli as cli
    from ase.build import molecule

    calls = {"single": 0, "multi": 0}

    def fake_single(args, structures):
        calls["single"] += 1
        return 0

    def fake_multi(args, structures, devices):
        calls["multi"] += 1
        return 0

    monkeypatch.setattr(cli, "_run_single_card", fake_single)
    monkeypatch.setattr(cli, "_run_multicard", fake_multi)
    # stub the structure read so fake filenames resolve to real Atoms (no file needed)
    monkeypatch.setattr(cli, "_read_structures", lambda paths: [molecule("H2O") for _ in paths])

    def mk(structures, devices=None, device_id=0, relax=False, md=False):
        return argparse.Namespace(structures=structures, devices=devices, device_id=device_id,
                                  task=None, charge=0.0, spin=1.0, refenv=None, fast=False,
                                  trace=False, relax=relax, md=md, fmax=0.05, steps=200,
                                  dt=1.0, temp=300.0, seed=None, out=None)

    # one structure, no --devices -> single-card path (no regression)
    cli.cmd_run(mk(["a.xyz"]))
    assert calls["single"] == 1 and calls["multi"] == 0

    # one structure, --devices 0 -> single card (one device) stays single-card
    cli.cmd_run(mk(["a.xyz"], devices="0"))
    assert calls["single"] == 2 and calls["multi"] == 0

    # multiple structures, no --devices -> single-card sequential (no fanout requested)
    cli.cmd_run(mk(["a.xyz", "b.xyz"]))
    assert calls["single"] == 3 and calls["multi"] == 0

    # multiple structures WITH --devices -> multi-card fan-out
    cli.cmd_run(mk(["a.xyz", "b.xyz"], devices="0,1"))
    assert calls["multi"] == 1

    # one structure, --devices 0,1 -> multi-card (explicit multi-device)
    cli.cmd_run(mk(["a.xyz"], devices="0,1"))
    assert calls["multi"] == 2


def test_run_single_card_processes_every_structure(monkeypatch, capsys):
    """The sequential single-card path must run ALL given structures, not just the first."""
    import tt_atom.cli as cli

    built = []

    class FakeCalc:
        @classmethod
        def from_uma(cls, model, task_name, atoms, **kw):
            built.append((task_name, len(atoms)))
            return cls()

        def close(self):
            pass

    monkeypatch.setattr("tt_atom.calculator.TTAtomCalculator", FakeCalc)
    monkeypatch.setattr("tt_atom.bundle_cache.infer_task", lambda atoms: "omol")
    import ase
    monkeypatch.setattr(ase.Atoms, "get_potential_energy", lambda self: 1.5)

    args = argparse.Namespace(structures=["a.xyz", "b.xyz", "c.xyz"], devices=None, device_id=0,
                              task=None, charge=0.0, spin=1.0, refenv=None, fast=False,
                              trace=False, relax=False, md=False, fmax=0.05, steps=200,
                              dt=1.0, temp=300.0, seed=None, out=None)
    structures = [molecule("H2O"), molecule("CO2"), molecule("NH3")]
    for a in structures:
        a.info.update(charge=0.0, spin=1.0)
    assert cli._run_single_card(args, structures) == 0
    assert built == [("omol", 3), ("omol", 3), ("omol", 4)]
    assert capsys.readouterr().out.count("energy:") == 3
