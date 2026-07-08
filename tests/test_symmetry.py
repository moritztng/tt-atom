"""Regression for the exact-symmetry wrong-force bug (host-only, no device).

The legacy ZYZ-Euler edge frame (``geometry._euler_angles``) has a coordinate singularity on the
+-Y axis: ``alpha = atan2(x, z)`` with ``x, z -> 0`` there. ``_Safeatan2``'s ``clamp(min=EPS)``
backward then *annihilates* the azimuth's position-gradient (denominator clamped, numerator ->0),
so ``d(alpha)/dpos -> 0`` — a degenerate frame derivative. At an exactly-symmetric geometry every
edge sits on that set, so the analytic force (which needs ``d(wigner)/dpos``) was wrong while the
energy (roll-gauge invariant) stayed fine. Fix 1 swaps the frame for fairchem's smooth two-chart
quaternion (``quaternion.wigner_from_edge``), which is finite, orthogonal and non-degenerate on the
axes.

These are host-only (torch + the coefficient asset — no card, no fairchem, no weight bundle), so the
symmetry gap that the ethanol-only goldens miss is covered in fast CI. Force-level correctness at
symmetry (vs fairchem) is validated end-to-end by the A/B harness (its ``symF`` metric evaluates the
exact-equilibrium geometry) and by the passing device parity suite.
"""
import torch

from tt_atom import quaternion
from tt_atom.geometry import _euler_angles

AXES = torch.tensor([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                    dtype=torch.float64)


def test_quaternion_wigner_finite_orthogonal_at_axes():
    """Quaternion Wigner-D and its gradient are finite, and D is orthogonal (a valid rotation), for
    edges exactly on the coordinate axes — the singular set of the old Euler frame. lmax 2 and 4."""
    for lmax in (2, 4):
        kern = quaternion.WignerKernels(lmax)
        e = AXES.clone().requires_grad_(True)
        W = quaternion.wigner_from_edge(e, lmax, kern, gamma=0.0)
        assert torch.isfinite(W).all(), f"non-finite wigner at axes (lmax={lmax})"
        g, = torch.autograd.grad(W.sum(), e)
        assert torch.isfinite(g).all(), f"non-finite d(wigner)/d(edge) at axes (lmax={lmax})"
        eye = torch.eye((lmax + 1) ** 2, dtype=torch.float64)
        assert (W @ W.transpose(1, 2) - eye).abs().max() < 1e-10, "wigner not orthogonal on axes"


def test_euler_frame_degenerate_at_pole():
    """Guard the ROOT CAUSE: on the +-Y axis the old Euler azimuth gradient is ANNIHILATED
    (``clamp`` denominator, vanishing numerator) -> ``d(alpha)/dpos ~= 0``. A degenerate frame
    derivative is what corrupted forces at exact symmetry; the quaternion frame (above) is
    non-degenerate there instead."""
    pole = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    _, _, alpha = _euler_angles(pole, 0.0)
    g, = torch.autograd.grad(alpha.sum(), pole)
    assert g.abs().max() < 1e-9, "expected the Euler azimuth gradient to be annihilated at the pole"
