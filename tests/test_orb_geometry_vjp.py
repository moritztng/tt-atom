import torch

from tt_atom.orb_geometry import host_edge_features, host_edge_features_vjp


def test_host_edge_features_vjp_matches_autograd():
    generator = torch.Generator().manual_seed(17)
    num_nodes, num_edges = 19, 173
    pos = torch.randn(num_nodes, 3, dtype=torch.float64, generator=generator)
    senders = torch.randint(num_nodes, (num_edges,), generator=generator)
    receivers = (senders + torch.randint(1, num_nodes, (num_edges,), generator=generator)) % num_nodes
    shift = 0.1 * torch.randn(num_edges, 3, dtype=torch.float64, generator=generator)
    # Keep every synthetic edge inside the physical cutoff and away from r=0.
    vectors = pos[receivers] - pos[senders] + shift
    shift = shift * torch.clamp(4.0 / vectors.norm(dim=-1), max=1.0)[:, None]

    pos_ref = pos.clone().requires_grad_(True)
    edge_feat, cutoff, vectors_ref = host_edge_features(
        pos_ref, senders, receivers, shift
    )
    g_edge = torch.randn(edge_feat.shape, dtype=torch.float64, generator=generator)
    g_cutoff = torch.randn(cutoff.shape, dtype=torch.float64, generator=generator)
    expected = -torch.autograd.grad(
        [edge_feat, cutoff], pos_ref, grad_outputs=[g_edge, g_cutoff]
    )[0]
    actual = host_edge_features_vjp(
        vectors_ref.detach(), senders, receivers, num_nodes, g_edge, g_cutoff
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
