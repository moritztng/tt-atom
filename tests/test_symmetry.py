"""Regression for the exact-symmetry wrong-force bug (host-only, no device).

The edge->+Y frame feeds ``d(wigner)/dpos``, so a degenerate frame derivative corrupts the analytic
force while leaving the energy (roll-gauge invariant) correct. The ZYZ-Euler frame this port
originally used had exactly that defect: its azimuth ``atan2(x, z)`` is singular on the +-Y axis, and
at an exactly-symmetric geometry every edge sits on that singular set. ``quaternion.wigner_from_edge``
(fairchem's smooth two-chart quaternion, now the only frame) is finite, orthogonal and
non-degenerate everywhere, which is what this pins.

Host-only (torch + the vendored coefficient asset: no card, no fairchem, no weight bundle), so the
symmetry gap that the off-axis real-weight goldens all miss is covered in fast CI.
"""
import torch

from tt_atom import quaternion

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


def test_quaternion_wigner_jacobian_nondegenerate_at_pole():
    """The +-Y pole is where the Euler azimuth gradient was ANNIHILATED (clamped denominator,
    vanishing numerator) — a degenerate ``d(wigner)/dpos`` is what corrupted forces at exact
    symmetry. The quaternion frame's Jacobian there must stay the same order of magnitude as at a
    generic direction (measured: 1.73 at the pole vs 1.49 off-axis, lmax=2)."""
    kern = quaternion.WignerKernels(2)

    def frame(e):
        return quaternion.wigner_from_edge(e, 2, kern, gamma=0.0)

    poles = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float64)
    generic = torch.nn.functional.normalize(torch.tensor([[0.3, 0.5, 0.81]], dtype=torch.float64))
    J_pole = torch.autograd.functional.jacobian(frame, poles)
    J_generic = torch.autograd.functional.jacobian(frame, generic)
    assert J_pole.abs().max() > 0.5 * J_generic.abs().max(), \
        "quaternion frame Jacobian is degenerate at the +-Y pole"
