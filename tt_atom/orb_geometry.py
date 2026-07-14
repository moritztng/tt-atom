"""Host geometry: the differentiable ``pos -> edge_feat`` map for Orb-v3 (torch, host).

Orb's edge embedding (``MoleculeGNS.featurize_edges``, ``outer_product_with_cutoff=True``) is a
fixed (no learned parameters) function of the edge vectors: a Bessel radial basis outer-producted
with a real spherical-harmonic angular descriptor (lmax=3, ``normalize=True``,
``normalization="component"``), gated by the same polynomial cutoff envelope used for attention
(``orb_model.host_cutoff``). Both closed forms are vendored here (MIT, matching Orb's own
``rbf.BesselBasis``/``angular.SphericalHarmonics``, themselves cribbed from e3nn 0.4.4) so this
module needs no dependency on ``orb_models``/``fairchem`` -- it must coexist with ttnn
(numpy<2) exactly like ``tt_atom/geometry.py`` does for UMA.

The geometry stays on the host. The eager path lets ``torch.autograd`` finish
``d(edge_feat, cutoff)/dpos``; the fixed-topology trace path uses the equivalent
closed-form VJP below to avoid rebuilding a large host autograd graph every step.

Sign convention: Orb's own ``vectors = pos[receivers] - pos[senders] + cell_shift``
(``featurization_utilities.compute_supercell_neighbors``) -- the opposite of fairchem/UMA's
src/tgt convention in ``tt_atom/geometry.py``. ``cell_shift`` is the constant (pos-independent)
per-edge periodic-image cartesian offset, zero for aperiodic edges.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bessel_basis(r: torch.Tensor, r_max: float = 6.0, num_bases: int = 8) -> torch.Tensor:
    """``orb_models.forcefield.rbf.BesselBasis`` -- closed-form, no learned parameters."""
    n = torch.arange(1, num_bases + 1, dtype=r.dtype, device=r.device)
    w = math.pi / r_max * n
    pre = math.sqrt(2.0 / r_max)
    return pre * torch.sin(w * r[:, None]) / r[:, None]


def spherical_harmonics_l3(unit_vec: torch.Tensor) -> torch.Tensor:
    """``orb_models.forcefield.angular.SphericalHarmonics(lmax=3, normalize=True,
    normalization="component")`` -- real solid harmonics l=0..3 (16 components) of a unit
    vector. Closed-form (e3nn 0.4.4 auto-generated code, MIT), no learned parameters."""
    x, y, z = unit_vec[..., 0], unit_vec[..., 1], unit_vec[..., 2]
    sh_0_0 = torch.ones_like(x)
    sh_1_0, sh_1_1, sh_1_2 = x, y, z
    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    sh = torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
                      sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6], dim=-1)
    mult = torch.cat([torch.full((2 * l + 1,), math.sqrt(2 * l + 1), dtype=sh.dtype, device=sh.device)
                      for l in range(4)])
    return sh * mult


def host_edge_features(pos: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                       cell_shift: torch.Tensor | None, *, r_max: float = 6.0, num_bases: int = 8,
                       strain: torch.Tensor | None = None):
    """Differentiable ``pos -> (edge_feat, cutoff, vectors)``.

    ``edge_feat`` [E, num_bases*16] is the ``outer_product_with_cutoff`` encoder input;
    ``cutoff`` [E, 1] is the identical polynomial envelope also used to gate attention
    (``orb_model.AttentionInteractionLayer``) -- both quantities depend on ``pos`` only through
    the same edge vectors, so a single ``torch.autograd.grad`` call finishes both adjoints.

    ``strain`` (optional, symmetric 3x3, see ``tt_atom/forces.py``'s ``_forward``) scales the edge
    vectors as ``r' = r (I + sym(strain))`` -- identical convention to both UMA's ``geometry.py``
    and Orb's own ``base.create_and_apply_stress_displacement`` (both fold the same fairchem-style
    displacement trick into positions *and* cell before differencing, which is algebraically
    equivalent to scaling the already-formed, cell-shift-inclusive vector this way). Since ALL
    pos/cell dependence of the energy flows through the edge vectors, ``dE/dstrain`` is exactly
    the (unsymmetrized) virial the caller divides by the cell volume for the stress tensor."""
    from .orb_model import host_cutoff

    vectors = pos[receivers] - pos[senders]
    if cell_shift is not None:
        vectors = vectors + cell_shift
    if strain is not None:
        sym = 0.5 * (strain + strain.transpose(0, 1))
        vectors = vectors + vectors @ sym
    lengths = vectors.norm(dim=-1)
    rbf = bessel_basis(lengths, r_max=r_max, num_bases=num_bases)
    unit = F.normalize(vectors, dim=-1)
    ang = spherical_harmonics_l3(unit)
    cutoff = host_cutoff(lengths, r_max=r_max)
    outer = rbf[:, :, None] * ang[:, None, :]
    edge_feat = cutoff * outer.reshape(vectors.shape[0], -1)
    return edge_feat, cutoff, vectors


def host_edge_features_vjp(vectors: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor,
                           num_nodes: int, g_edge_feat: torch.Tensor, g_cutoff: torch.Tensor, *,
                           r_max: float = 6.0, num_bases: int = 8) -> torch.Tensor:
    """Analytic VJP of :func:`host_edge_features`, returning ``-dE/dpos``.

    The traced MD path receives the two upstream adjoints from the device.  Replaying
    PyTorch autograd through the 128-component RBF × spherical-harmonic descriptor was
    a substantial host-side cost at large edge counts.  This closed-form VJP evaluates
    the same derivatives directly and avoids constructing an autograd graph each step.
    """
    dtype, device = vectors.dtype, vectors.device
    r = vectors.norm(dim=-1)
    unit = vectors / r[:, None]
    rbf = bessel_basis(r, r_max=r_max, num_bases=num_bases)
    ang = spherical_harmonics_l3(unit)
    cutoff = _cutoff_value(r, r_max)
    grad = g_edge_feat.to(dtype=dtype, device=device).reshape(-1, num_bases, 16)
    g_cut = g_cutoff.to(dtype=dtype, device=device).reshape(-1)

    # edge_feat[e,n,m] = cutoff[e] * rbf[e,n] * ang[e,m]
    g_rbf = cutoff[:, None] * torch.einsum("enm,em->en", grad, ang)
    g_ang = cutoff[:, None] * torch.einsum("enm,en->em", grad, rbf)
    g_cut = g_cut + torch.einsum("enm,en,em->e", grad, rbf, ang)

    n = torch.arange(1, num_bases + 1, dtype=dtype, device=device)
    w = math.pi / r_max * n
    wr = r[:, None] * w
    pre = math.sqrt(2.0 / r_max)
    drbf = pre * (w * torch.cos(wr) * r[:, None] - torch.sin(wr)) / r[:, None].square()
    g_r = (g_rbf * drbf).sum(dim=-1) + g_cut * _cutoff_derivative(r, r_max)

    g_unit = _spherical_harmonics_l3_vjp(unit, g_ang)
    radial_unit = (g_unit * unit).sum(dim=-1, keepdim=True)
    g_vectors = unit * g_r[:, None] + (g_unit - unit * radial_unit) / r[:, None]

    # vectors = pos[receivers] - pos[senders] + constant shift; return forces=-grad(pos).
    forces = torch.zeros((num_nodes, 3), dtype=dtype, device=device)
    forces.index_add_(0, senders, g_vectors)
    forces.index_add_(0, receivers, -g_vectors)
    return forces


def _cutoff_value(r: torch.Tensor, r_max: float) -> torch.Tensor:
    p = 4
    q = r / r_max
    value = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * q.pow(p)
        + p * (p + 2.0) * q.pow(p + 1)
        - (p * (p + 1.0) / 2.0) * q.pow(p + 2)
    )
    return value * (r < r_max)


def _cutoff_derivative(r: torch.Tensor, r_max: float) -> torch.Tensor:
    p = 4
    q = r / r_max
    derivative = (
        -((p + 1.0) * (p + 2.0) / 2.0) * p * q.pow(p - 1)
        + p * (p + 2.0) * (p + 1) * q.pow(p)
        - (p * (p + 1.0) / 2.0) * (p + 2) * q.pow(p + 1)
    ) / r_max
    return derivative * (r < r_max)


def _spherical_harmonics_l3_vjp(unit: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """VJP of :func:`spherical_harmonics_l3` with respect to its unit-vector input."""
    x, y, z = unit.unbind(dim=-1)
    mult = torch.cat([
        torch.full((2 * l + 1,), math.sqrt(2 * l + 1), dtype=unit.dtype, device=unit.device)
        for l in range(4)
    ])
    q = grad * mult
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(y)
    gz = torch.zeros_like(z)
    sqrt3 = math.sqrt(3.0)
    sqrt15 = math.sqrt(15.0)
    c = math.sqrt(3.0 / 8.0)
    b = math.sqrt(5.0 / 6.0) * sqrt3 / 2.0

    gx += q[:, 1]
    gy += q[:, 2]
    gz += q[:, 3]

    gx += q[:, 4] * sqrt3 * z
    gz += q[:, 4] * sqrt3 * x
    gx += q[:, 5] * sqrt3 * y
    gy += q[:, 5] * sqrt3 * x
    gx -= q[:, 6] * x
    gy += q[:, 6] * 2.0 * y
    gz -= q[:, 6] * z
    gy += q[:, 7] * sqrt3 * z
    gz += q[:, 7] * sqrt3 * y
    gx -= q[:, 8] * sqrt3 * x
    gz += q[:, 8] * sqrt3 * z

    gx += q[:, 9] * b * (3.0 * z.square() - 3.0 * x.square())
    gz += q[:, 9] * b * (6.0 * x * z)
    gx += q[:, 10] * sqrt15 * y * z
    gy += q[:, 10] * sqrt15 * x * z
    gz += q[:, 10] * sqrt15 * x * y
    gx += q[:, 11] * c * (4.0 * y.square() - 3.0 * x.square() - z.square())
    gy += q[:, 11] * c * (8.0 * x * y)
    gz -= q[:, 11] * c * (2.0 * x * z)
    gx -= q[:, 12] * 3.0 * x * y
    gy += q[:, 12] * (3.0 * y.square() - 1.5 * (x.square() + z.square()))
    gz -= q[:, 12] * 3.0 * y * z
    gx -= q[:, 13] * c * (2.0 * x * z)
    gy += q[:, 13] * c * (8.0 * y * z)
    gz += q[:, 13] * c * (4.0 * y.square() - x.square() - 3.0 * z.square())
    gx -= q[:, 14] * sqrt15 * x * y
    gy += q[:, 14] * (sqrt15 / 2.0) * (z.square() - x.square())
    gz += q[:, 14] * sqrt15 * z * y
    gx -= q[:, 15] * b * (6.0 * x * z)
    gz += q[:, 15] * b * (3.0 * z.square() - 3.0 * x.square())
    return torch.stack([gx, gy, gz], dim=-1)
