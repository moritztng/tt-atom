"""Unified ``Calculator(atoms, model=...)`` front door: family dispatch by name + the
family-inapplicable-argument errors. Host-only — every case here raises (or resolves a family)
before any device is opened, so it needs no hardware and no weight cache."""
import pytest
from ase import Atoms

from tt_atom import Calculator
from tt_atom.auto import _family


def _water():
    return Atoms("OH2", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])


def test_family_dispatch_by_name():
    assert _family("uma-s-1") == "uma"
    assert _family("uma-m-1p1") == "uma"
    assert _family("orb-v3-conservative-inf-omat") == "orb"
    assert _family("orb-v3-direct-omol") == "orb"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        _family("gemnet")
    with pytest.raises(ValueError, match="unknown model"):
        Calculator(_water(), model="mace-mp-0")


def test_uma_needs_atoms():
    with pytest.raises(ValueError, match="needs `atoms`"):
        Calculator(None, model="uma-s-1")


def test_task_rejected_for_orb():
    with pytest.raises(ValueError, match="task"):
        Calculator(_water(), model="orb-v3-conservative-omol", task="omol")


def test_checkpoint_override_rejected_for_orb():
    with pytest.raises(ValueError, match="checkpoint"):
        Calculator(_water(), model="orb-v3-conservative-omol", checkpoint="some.npz")


def test_trace_rejected_for_orb():
    with pytest.raises(ValueError, match="trace"):
        Calculator(_water(), model="orb-v3-conservative-inf-omat", trace=True)
