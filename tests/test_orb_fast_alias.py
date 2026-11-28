"""Orb has one precision path. ``fast`` stays accepted at the public entry points and reaches
nothing below them. Device-free."""
import inspect

from tt_atom import orb_model
from tt_atom.orb_calculator import OrbCalculator


def test_public_entry_points_still_accept_fast():
    assert "fast" in inspect.signature(OrbCalculator.__init__).parameters
    assert "fast" in inspect.signature(OrbCalculator.from_checkpoint).parameters


def test_model_has_no_fast_path():
    for name in ("MLPNorm", "Encoder", "AttentionInteractionLayer", "EnergyHead", "ForceHead",
                 "StressHead"):
        assert "fast" not in inspect.signature(getattr(orb_model, name).__init__).parameters, name
    assert "bfloat8" not in inspect.getsource(orb_model)
