"""Hand-written MACE reverse pass (tt_atom/mace/backward.py) against autograd and the golden, float64."""
import torch

from tt_atom.mace import backward
from test_mace_reference import GOLDEN, g  # noqa: F401  (fixture)
import pytest

pytestmark = pytest.mark.skipif(not __import__("os").path.exists(GOLDEN), reason="MACE golden not generated")


def test_forces_without_autograd(g):  # noqa: F811
    t = g["t"]
    e, f = backward.energy_forces(g["w"], g["Z"], t["in@pos"].double(), g["ei"], t["in@shifts"].double())
    want_e, want_f = t["out@energy"].double().sum(), t["out@forces"].double()
    assert abs(e.item() - want_e.item()) < 1e-9 * abs(want_e.item())
    err = ((f - want_f).abs().max() / want_f.abs().max()).item()
    assert err < 1e-9, err
