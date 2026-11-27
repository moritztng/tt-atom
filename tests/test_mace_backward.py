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


@pytest.mark.parametrize("fn,deriv", [("radial_embedding", "radial_deriv"), ("zbl", "zbl_deriv")])
def test_edge_derivatives_closed_form(g, fn, deriv):  # noqa: F811
    """Closed-form r-derivatives against autograd, over 0.5-6 A so ZBL is live (Si-Si bonds sit past its cutoff)."""
    from tt_atom.mace import reference as ref
    w, Z, ei = g["w"], g["Z"], g["ei"]
    r = torch.linspace(0.5, 5.9, ei.shape[1], dtype=torch.float64)[:, None].requires_grad_(True)
    if fn == "zbl":  # the node sum's gradient wrt each edge length is that edge's own derivative
        (want,) = torch.autograd.grad(ref.zbl(r, Z, ei, w, int(ei.max()) + 1).sum(), r)
        got = getattr(backward, deriv)(r.detach(), Z, ei, w)
        assert got.abs().max() > 0
    else:
        out = ref.radial_embedding(r, Z, ei, w)
        want = torch.stack([torch.autograd.grad(out[:, k].sum(), r, retain_graph=True)[0][:, 0] for k in range(out.shape[1])], 1)
        got = getattr(backward, deriv)(r.detach(), Z, ei, w)
    assert (got - want).abs().max() < 1e-9 * want.abs().max()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equivariance_fixed_edges(g, seed):  # noqa: F811
    """Gate leg 1 (tt_atom.equivariant.gate) on the hand-written forces: rotate and translate the cell at a fixed
    edge set (periodic shifts rotate as vectors). Energy is invariant, forces rotate. Seed 0 is the identity
    control and must be exact."""
    from tt_atom.equivariant import gate
    t = g["t"]
    pos, shifts = t["in@pos"].double(), t["in@shifts"].double()
    R, tr = gate.rigid_transform(seed)
    e0, f0 = backward.energy_forces(g["w"], g["Z"], pos, g["ei"], shifts)
    e1, f1 = backward.energy_forces(g["w"], g["Z"], gate.apply_transform(pos, R, tr), g["ei"], gate.apply_transform(shifts, R))
    _, de = gate.residual(e1.reshape(1), e0.reshape(1))
    _, df = gate.residual(f1, gate.apply_transform(f0, R))
    if seed == 0:
        assert de == 0 and df == 0
    assert de < 1e-12 and df < 1e-10, (de, df)
