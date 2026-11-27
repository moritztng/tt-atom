"""Real Wigner 3j symbols in e3nn's basis, computed in float64 with no e3nn dependency.

The Racah formula gives the complex Clebsch-Gordan coefficients; e3nn's real/complex change
of basis turns them into the real tensor, normalised to unit Frobenius norm. This covers every
triple, including odd l1+l2+l3 (the cross-product paths such as 1x1->1), which a Gaunt integral
of spherical harmonics cannot produce because it vanishes there.

The sign of a 3j tensor is a convention, and checkpoints are fitted against one. The default
matches e3nn 0.5.x. e3nn 0.4.x (what mace-torch pins) differs on some odd-sum triples, listed in
``E3NN_04_FLIPS`` for l <= 4; pass ``convention="e3nn-0.4"`` for a model trained there. Every
even-sum triple agrees across both versions.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial, sqrt

import torch

# Odd-sum triples whose e3nn 0.4.4 tensor is the negative of e3nn 0.5.x's, measured for l <= 4.
E3NN_04_FLIPS = frozenset({(1, 2, 2), (2, 1, 2), (2, 2, 1), (1, 3, 3), (3, 1, 3), (3, 3, 1),
                           (1, 4, 4), (4, 1, 4), (4, 4, 1)})


def _su2_cg(j1: int, j2: int, j3: int) -> torch.Tensor:
    C = torch.zeros(2 * j1 + 1, 2 * j2 + 1, 2 * j3 + 1, dtype=torch.float64)
    f = factorial
    pre = sqrt((2 * j3 + 1) * f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1))
    for m1 in range(-j1, j1 + 1):
        for m2 in range(-j2, j2 + 1):
            m3 = m1 + m2
            if abs(m3) > j3:
                continue
            s = 0.0
            for k in range(j1 + j2 - j3 + 1):
                d = (k, j1 + j2 - j3 - k, j1 - m1 - k, j2 + m2 - k, j3 - j2 + m1 + k, j3 - j1 - m2 + k)
                if min(d) < 0:
                    continue
                den = 1
                for x in d:
                    den *= f(x)
                s += (-1) ** k / den
            norm = sqrt(f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3))
            C[j1 + m1, j2 + m2, j3 + m3] = pre * norm * s
    return C


def _real_to_complex(l: int) -> torch.Tensor:
    """e3nn's change of basis, real SH order m = -l..l, including its (-i)^l phase."""
    q = torch.zeros(2 * l + 1, 2 * l + 1, dtype=torch.complex128)
    r = 2 ** -0.5
    for m in range(-l, 0):
        q[l + m, l - m] = r
        q[l + m, l + m] = -1j * r
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + m] = (-1) ** m * r
        q[l + m, l - m] = 1j * (-1) ** m * r
    return (-1j) ** l * q


@lru_cache(maxsize=None)
def _w3j64(l1: int, l2: int, l3: int, flip: bool) -> torch.Tensor:
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(f"({l1}, {l2}, {l3}) violates the triangle rule")
    C = _su2_cg(l1, l2, l3).to(torch.complex128)
    C = torch.einsum("ij,kl,mn,ikn->jlm", _real_to_complex(l1), _real_to_complex(l2),
                     _real_to_complex(l3).conj().T, C)
    if C.imag.abs().max() > 1e-12:
        raise AssertionError(f"({l1}, {l2}, {l3}) did not come out real")
    C = C.real / C.real.norm()
    return -C if flip else C


def wigner_3j(l1: int, l2: int, l3: int, *, convention: str = "e3nn-0.5",
              dtype=torch.float64, device=None) -> torch.Tensor:
    """Unit-norm real 3j tensor of shape [2l1+1, 2l2+1, 2l3+1]. Kept in float64 until cast here."""
    if convention not in ("e3nn-0.5", "e3nn-0.4"):
        raise ValueError(f"unknown convention {convention!r}")
    flip = convention == "e3nn-0.4" and (l1, l2, l3) in E3NN_04_FLIPS
    if convention == "e3nn-0.4" and max(l1, l2, l3) > 4 and (l1 + l2 + l3) % 2:
        raise ValueError("e3nn 0.4 signs are only measured for l <= 4 on odd-sum triples")
    return _w3j64(l1, l2, l3, flip).to(dtype=dtype, device=device)
