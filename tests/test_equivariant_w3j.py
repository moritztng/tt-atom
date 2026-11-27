"""wigner_3j against closed forms everywhere, and against the installed e3nn when there is one."""
import itertools

import pytest
import torch

from tt_atom.equivariant import E3NN_04_FLIPS, wigner_3j

TRIPLES = [l for l in itertools.product(range(5), repeat=3) if abs(l[0] - l[1]) <= l[2] <= l[0] + l[1]]


def test_closed_forms():
    assert torch.allclose(wigner_3j(1, 1, 0), torch.eye(3, dtype=torch.float64).unsqueeze(-1) / 3 ** 0.5, atol=1e-15)
    eps = torch.zeros(3, 3, 3, dtype=torch.float64)
    for (i, j, k), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
        eps[i, j, k] = s
    assert torch.allclose(wigner_3j(1, 1, 1).abs(), eps.abs() / 6 ** 0.5, atol=1e-15)


@pytest.mark.parametrize("l", TRIPLES)
def test_unit_norm_and_symmetry(l):
    C = wigner_3j(*l)
    assert abs(C.norm().item() - 1) < 1e-14
    # swapping the first two legs multiplies by (-1)^(l1+l2+l3)
    assert torch.allclose(wigner_3j(l[1], l[0], l[2]), C.transpose(0, 1) * (-1) ** sum(l), atol=1e-14)


def test_e3nn_conventions():
    e3nn = pytest.importorskip("e3nn")
    from e3nn import o3
    conv = "e3nn-0.4" if e3nn.__version__.startswith("0.4") else "e3nn-0.5"
    for l in TRIPLES:
        ref = o3.wigner_3j(*l).double()
        assert (wigner_3j(*l, convention=conv) - ref).abs().max() < 1e-6, (l, e3nn.__version__)
    assert all(sum(l) % 2 for l in E3NN_04_FLIPS)
