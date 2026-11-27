"""The shared equivariance-gate helpers grade what they claim, including their own controls."""
import pytest
import torch

from tt_atom.equivariant import apply_transform, flips_off_cutoff, residual, rigid_transform, wigner_3j


@pytest.mark.parametrize("seed", range(1, 6))
def test_rotation_is_proper_and_deterministic(seed):
    R, t = rigid_transform(seed)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-14)
    assert abs(torch.det(R).item() - 1) < 1e-14
    assert t.abs().max() > 0
    R2, t2 = rigid_transform(seed)
    assert torch.equal(R, R2) and torch.equal(t, t2)


def test_identity_control_gives_exactly_zero():
    pos = torch.randn(20, 3, dtype=torch.float32)
    R, t = rigid_transform(0)
    assert residual(apply_transform(pos, R, t), pos) == (0.0, 0.0)


def test_residual_separates_an_equivariant_map_from_a_broken_one():
    # A vector-valued pairwise map: sum of displacement vectors weighted by an invariant.
    def forces(p):
        d = p[:, None] - p[None]
        return (d * torch.exp(-(d ** 2).sum(-1, keepdim=True))).sum(1)

    def broken(p):  # leaks an absolute coordinate into the weight
        d = p[:, None] - p[None]
        return (d * torch.exp(-(d ** 2).sum(-1, keepdim=True) - p[:, None, :1])).sum(1)

    pos = torch.randn(16, 3, dtype=torch.float64)
    R, t = rigid_transform(3)
    _, good = residual(forces(apply_transform(pos, R, t)), apply_transform(forces(pos), R))
    _, bad = residual(broken(apply_transform(pos, R, t)), apply_transform(broken(pos), R))
    assert good < 1e-13 and bad > 1e-2


def test_rotation_matches_l1_wigner_coupling():
    # The gate's R and the w3j table agree on what an l=1 rotation is: C(1,1,0) contracts two
    # rotated vectors to the same scalar.
    R, _ = rigid_transform(2)
    a, b = torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
    C = wigner_3j(1, 1, 0)[..., 0]
    assert torch.allclose(a @ C @ b, (R @ a) @ C @ (R @ b), atol=1e-14)


def test_flips_count_only_off_the_cutoff():
    cutoff = 5.0
    dist_a = torch.tensor([1.0, 4.99995, 5.00004, 7.0, 4.0], dtype=torch.float64)
    dist_b = torch.tensor([1.0, 5.00002, 4.99999, 7.0, 6.0], dtype=torch.float64)
    # bits 1 and 2 flip on the cutoff (legal), bit 4 flips 1 A inside it (a broken builder)
    assert flips_off_cutoff(dist_a, dist_b, cutoff) == 1
    assert flips_off_cutoff(dist_a, dist_a, cutoff) == 0
