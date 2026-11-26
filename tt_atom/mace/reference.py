"""Float64 CPU reference for MACE-MP-0b3 medium, written from the mace-torch / e3nn equations.

No e3nn or mace imports: each function takes plain tensors and the weights dict a golden carries
(``w@<module path>``), so a parity miss is pinned to one block. Graded against the mace-torch
golden by ``tests/test_mace_reference.py``.
"""
from __future__ import annotations

import math

import torch


def edge_vectors(pos, edge_index, shifts):
    """MACE convention: vector from sender to receiver, plus the periodic shift."""
    sender, receiver = edge_index
    vec = pos[receiver] - pos[sender] + shifts
    return vec, vec.norm(dim=-1, keepdim=True)


def node_attrs(atomic_numbers, z_table):
    """One-hot over the model's element table."""
    idx = (atomic_numbers[:, None] == z_table[None, :]).long().argmax(dim=1)
    return torch.nn.functional.one_hot(idx, len(z_table)).to(torch.float64)


def node_embedding(attrs, w):
    """e3nn Linear ``n x 0e -> c x 0e``: element-normalised weight, no bias."""
    n = attrs.shape[1]
    W = w["node_embedding.linear.weight"].reshape(n, -1)
    return attrs @ W / math.sqrt(n)


def spherical_harmonics(vec, lmax=3):
    """e3nn real spherical harmonics, normalize=True, component normalisation, e3nn's (x, y, z) order."""
    u = torch.nn.functional.normalize(vec, dim=-1)
    x, y, z = u.unbind(-1)
    s3 = math.sqrt(3.0)
    out = [torch.ones_like(x), x, y, z]
    if lmax >= 2:
        y2, x2z2 = y * y, x * x + z * z
        s20, s24 = s3 * x * z, s3 / 2.0 * (z * z - x * x)
        out += [s20, s3 * x * y, y2 - 0.5 * x2z2, s3 * y * z, s24]
    if lmax >= 3:
        out += [math.sqrt(5.0 / 6.0) * (s20 * z + s24 * x), math.sqrt(5.0) * s20 * y,
                math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x, 0.5 * y * (2.0 * y2 - 3.0 * x2z2),
                math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2), math.sqrt(5.0) * s24 * y,
                math.sqrt(5.0 / 6.0) * (s24 * z - s20 * x)]
    sh = torch.stack(out, dim=-1)
    norm = torch.cat([torch.full((2 * l + 1,), math.sqrt(2 * l + 1), dtype=sh.dtype) for l in range(lmax + 1)])
    return sh * norm


def polynomial_cutoff(r, r_max, p):
    t = r / r_max
    env = 1.0 - (p + 1.0) * (p + 2.0) / 2.0 * t ** p + p * (p + 2.0) * t ** (p + 1) - p * (p + 1.0) / 2.0 * t ** (p + 2)
    return env * (r < r_max)


def _pair_z(atomic_numbers, edge_index):
    z = atomic_numbers.long()[:, None]
    return z[edge_index[0]], z[edge_index[1]]


def radial_embedding(r, atomic_numbers, edge_index, w):
    """Agnesi distance transform -> Bessel basis, times the polynomial cutoff of the raw length."""
    pre = "radial_embedding."
    cut = polynomial_cutoff(r, w[pre + "cutoff_fn.r_max"], w[pre + "cutoff_fn.p"])
    zu, zv = _pair_z(atomic_numbers, edge_index)
    cov = w[pre + "distance_transform.covalent_radii"]
    a, p, q = (w[pre + "distance_transform." + k] for k in "apq")
    t = r / (0.5 * (cov[zu] + cov[zv]))
    x = 1.0 / (1.0 + a * t ** q / (1.0 + t ** (q - p)))
    radial = w[pre + "bessel_fn.prefactor"] * torch.sin(w[pre + "bessel_fn.bessel_weights"] * x) / x
    return radial * cut


def zbl(r, atomic_numbers, edge_index, w, n_nodes):
    """Per-node ZBL pair repulsion (half of each edge's energy lands on its receiver)."""
    pre = "pair_repulsion_fn."
    zu, zv = _pair_z(atomic_numbers, edge_index)
    zu, zv = zu.to(r.dtype), zv.to(r.dtype)
    a_exp = w.get(pre + "a_exp", torch.tensor(0.300, dtype=r.dtype))
    a_pre = w.get(pre + "a_prefactor", torch.tensor(0.4543, dtype=r.dtype))
    a = a_pre * 0.529 / (zu ** a_exp + zv ** a_exp)
    c = w[pre + "c"]
    ra = r / a
    phi = c[0] * torch.exp(-3.2 * ra) + c[1] * torch.exp(-0.9423 * ra) + c[2] * torch.exp(-0.4028 * ra) \
        + c[3] * torch.exp(-0.2016 * ra)
    cov = w[pre + "covalent_radii"]
    zi_u, zi_v = _pair_z(atomic_numbers, edge_index)
    env = polynomial_cutoff(r, cov[zi_u] + cov[zi_v], w[pre + "p"])
    v = 0.5 * (14.3996 * zu * zv) / r * phi * env
    return torch.zeros(n_nodes, 1, dtype=r.dtype).index_add_(0, edge_index[1], v).squeeze(-1)
