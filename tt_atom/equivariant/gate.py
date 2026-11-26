"""Grading a model for SE(3) equivariance, as two separate properties.

Lifted from tt-bio's DiffDock gate (tests/test_diffdock_equivariance.py, main 77a1f8ca2) so MACE and
DiffDock grade the same way. The method point it encodes: a graph network's residual under a rigid
transform has two causes that differ by orders of magnitude, so they get two gates.

* Arithmetic, at fixed topology: build the edge set once, transform the geometry, compare outputs.
  This is fp32 round-off (DiffDock upstream: 1.6e-7 to 1.3e-6 relative) and gets a tight bar.
* Graph construction, as an exact count: does the transformed input give the same edge mask? A
  single edge flipped at the cutoff cost DiffDock upstream 1.5e-2, four orders above the arithmetic,
  so one blended tolerance would hide real kernel bugs. A flip is legal only inside a band around the
  cutoff about as wide as fp32 noise on a coordinate; :func:`flips_off_cutoff` counts the rest.

Seed 0 of :func:`rigid_transform` is the identity control: a residual that is not exactly zero there
is the test's own artefact, and a test that silently rounded its transform away would otherwise read
as perfect.
"""
from __future__ import annotations

import torch


def rigid_transform(seed: int, translation_scale: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic float64 ``(R, t)``: Haar-uniform proper rotation, translation ~``translation_scale``.

    ``seed == 0`` is the identity rotation and zero translation.
    """
    if seed == 0:
        return torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    g = torch.Generator().manual_seed(1000 + seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q, torch.randn(3, generator=g, dtype=torch.float64) * translation_scale


def apply_transform(pos: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor | None = None) -> torch.Tensor:
    """``pos @ R.T + t`` computed in float64, returned at ``pos``'s dtype. Pass no ``t`` for vectors."""
    out = pos.to(torch.float64) @ rot.T
    if trans is not None:
        out = out + trans
    return out.to(pos.dtype)


def residual(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """``(max abs error, max abs error / max |want|)`` at float64; relative is 0 when ``want`` is 0."""
    got, want = got.to(torch.float64), want.to(torch.float64)
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    return err, (err / scale if scale > 0 else 0.0)


def flips_off_cutoff(dist_a: torch.Tensor, dist_b: torch.Tensor, cutoff: float, band: float = 1e-4) -> int:
    """Edge-mask bits (``dist < cutoff``) that differ between two poses away from the cutoff.

    ``dist_a`` and ``dist_b`` are the same edges' lengths before and after a rigid transform. A bit
    may flip only when the edge sits within ``band`` of the cutoff in either pose; every other flip
    is a broken graph builder, and the count of those is returned. Zero is the pass.
    """
    flipped = (dist_a < cutoff) != (dist_b < cutoff)
    near = ((dist_a - cutoff).abs() <= band) | ((dist_b - cutoff).abs() <= band)
    return int((flipped & ~near).sum())
