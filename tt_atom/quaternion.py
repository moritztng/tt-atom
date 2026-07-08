"""Smooth two-chart quaternion edge-frame rotation (vendored from fairchem UMA, MIT).

Replaces the singular ZYZ-Euler edge->+Y frame (``geometry._euler_angles``) with fairchem's DEFAULT
quaternion frame (``use_quaternion_wigner=True``), which is C-infinity across the whole sphere. The
Euler frame has a coordinate singularity on the +-Y axis: at axis-aligned/exactly-symmetric
geometries its ``d(wigner)/dpos`` is non-differentiable, so the analytic force comes out wrong
(the energy, being roll-gauge invariant, is fine). The quaternion frame removes that pole, so forces
are correct everywhere. It also removes the exact-zero Wigner entries that made the value-thresholded
``rotation.pack`` change width across a trajectory (the trace-buffer shape crash).

Pure torch, no fairchem import (same stance as ``geometry.py``). Only the l<=4 kernels are vendored
(uma-s lmax=2, uma-m lmax=4); the l>=5 Ra/Rb machinery is never reached. Verbatim from
``fairchem/core/models/uma/common/quaternion/{quaternion_utils,wigner_d_custom_kernels,
wigner_d_hybrid}.py``; the ~29 KB coefficient table lives in ``assets/wigner_d_coefficients.pt``.
"""
from __future__ import annotations

from pathlib import Path

import torch

_ASSET = Path(__file__).parent / "assets" / "wigner_d_coefficients.pt"

# Blend region for the two-chart quaternion: ey in [-0.9, 0.9]
BLEND_START = -0.9
BLEND_WIDTH = 1.8


# ------------------------------------------------------------------ quaternion helpers


def _smooth_step_cinf(t: torch.Tensor) -> torch.Tensor:
    """C-infinity smooth step (all derivatives 0 at t=0,1). step(t)=sigmoid((2t-1)/(t(1-t)))."""
    t_clamped = t.clamp(0, 1)
    eps = torch.finfo(t.dtype).eps
    numerator = 2.0 * t_clamped - 1.0
    denom_safe = (t_clamped * (1.0 - t_clamped)).clamp(min=eps)
    result = torch.sigmoid(numerator / denom_safe)
    result = torch.where(t_clamped < eps, torch.zeros_like(result), result)
    result = torch.where(t_clamped > 1 - eps, torch.ones_like(result), result)
    return result


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1*q2, (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_y_rotation(gamma: torch.Tensor) -> torch.Tensor:
    """Quaternion for a rotation about +Y by angle gamma, shape (N,) -> (N,4)."""
    half = gamma / 2
    return torch.stack([torch.cos(half), torch.zeros_like(gamma),
                        torch.sin(half), torch.zeros_like(gamma)], dim=-1)


def quaternion_nlerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Normalized linear interpolation normalize((1-t)q1 + t q2), sign-aligned."""
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q1_aligned = torch.where(dot < 0, -q1, q1)
    t_exp = t.unsqueeze(-1) if t.dim() < q1.dim() else t
    return torch.nn.functional.normalize((1.0 - t_exp) * q1_aligned + t_exp * q2, dim=-1)


def _quaternion_chart1_standard(ex, ey, ez):
    """edge->+Y directly (half-vector); singular at edge=-Y (unused there, clamp detaches grad)."""
    q = torch.stack([1.0 + ey, -ez, torch.zeros_like(ex), ex], dim=-1)
    eps = torch.finfo(ex.dtype).eps
    return q / torch.sqrt(torch.clamp(torch.sum(q ** 2, dim=-1, keepdim=True), min=eps))


def _quaternion_chart2_via_minus_y(ex, ey, ez):
    """edge->+Y via -Y; singular at edge=+Y (unused there, clamp detaches grad)."""
    q = torch.stack([-ez, 1.0 - ey, ex, torch.zeros_like(ex)], dim=-1)
    eps = torch.finfo(ex.dtype).eps
    return q / torch.sqrt(torch.clamp(torch.sum(q ** 2, dim=-1, keepdim=True), min=eps))


def quaternion_edge_to_y_stable(edge_vec: torch.Tensor) -> torch.Tensor:
    """Two-chart edge->+Y quaternion with C-infinity NLERP blend (chart2 near -Y, chart1 near +Y).
    ``edge_vec`` assumed normalized, shape (N,3) -> (N,4)."""
    ex, ey, ez = edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
    q1 = _quaternion_chart1_standard(ex, ey, ez)
    q2 = _quaternion_chart2_via_minus_y(ex, ey, ez)
    t_smooth = _smooth_step_cinf((ey - BLEND_START) / BLEND_WIDTH)
    return quaternion_nlerp(q2, q1, t_smooth)


# ------------------------------------------------------------------ quaternion -> Wigner-D (l<=4)


def _generate_monomials(n_vars: int, total_degree: int):
    monomials = []

    def gen(rv, rd, cur):
        if rv == 1:
            monomials.append(tuple(cur + [rd])); return
        for i in range(rd + 1):
            gen(rv - 1, rd - i, cur + [i])

    gen(n_vars, total_degree, [])
    return monomials


def _precompute_powers(w, x, y, z, max_power):
    def pv(var):
        p = {0: torch.ones_like(var), 1: var}
        for i in range(2, max_power + 1):
            p[i] = p[i // 2] * p[(i + 1) // 2]
        return p

    return {0: pv(w), 1: pv(x), 2: pv(y), 3: pv(z)}


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """l=1 Wigner-D: quaternion (N,4) -> 3x3 rotation (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
    ], dim=-2)


def quaternion_to_wigner_d_l2_einsum(q: torch.Tensor, C_l2: torch.Tensor) -> torch.Tensor:
    """l=2 Wigner-D via degree-4 polynomial einsum. C_l2: (5,5,4,4,4,4) -> (N,5,5)."""
    C = C_l2.to(dtype=q.dtype, device=q.device)
    q2 = q.unsqueeze(-1) * q.unsqueeze(-2)                        # (N,4,4)
    q4 = q2.unsqueeze(-1).unsqueeze(-1) * q2.unsqueeze(-3).unsqueeze(-3)   # (N,4,4,4,4)
    return torch.einsum("nabcd,ijabcd->nij", q4, C)


def quaternion_to_wigner_d_matmul(q, ell, C, monomials):
    """l=3 or l=4 standalone: D = M @ C^T. Returns (N,2ell+1,2ell+1)."""
    C_cast = C.to(dtype=q.dtype, device=q.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    powers = _precompute_powers(w, x, y, z, 2 * ell)
    M = torch.stack([powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
                     for a, b, c, d in monomials], dim=1)
    size = 2 * ell + 1
    return (M @ C_cast.T).view(q.shape[0], size, size)


def quaternion_to_wigner_d_l3l4_batched(q, C_combined, monomials_l4):
    """l=3 and l=4 in one degree-8 matmul. C_combined (130,165) -> (D_l3 (N,7,7), D_l4 (N,9,9))."""
    C_cast = C_combined.to(dtype=q.dtype, device=q.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    powers = _precompute_powers(w, x, y, z, 8)
    M = torch.stack([powers[0][a] * powers[1][b] * powers[2][c] * powers[3][d]
                     for a, b, c, d in monomials_l4], dim=1)
    D_flat = M @ C_cast.T
    N = q.shape[0]
    return D_flat[:, :49].reshape(N, 7, 7), D_flat[:, 49:].reshape(N, 9, 9)


class WignerKernels:
    """Loads the palette-compressed l=2,3,4 coefficient tables once (from the vendored asset).
    Held on the ``HostGeometry`` and reused every step; device-independent (cast per call)."""

    def __init__(self, lmax: int, asset_path: Path = _ASSET):
        raw = torch.load(asset_path, map_location="cpu", weights_only=True)

        def dec(ell):
            k = f"C_l{ell}"
            return raw[f"{k}_palette"][raw[f"{k}_indices"].long()].reshape(tuple(raw[f"{k}_shape"].tolist()))

        # the quaternion->Wigner kernels are provided up to l=4 (uma-s lmax=2, uma-m lmax=4); a
        # higher-lmax checkpoint would be silently zero-filled for l>=5, so fail loudly instead.
        if lmax > 4:
            raise ValueError(f"quaternion edge frame supports lmax<=4, got lmax={lmax}; "
                             "use the Euler path (use_quaternion=False) for higher lmax")
        self.lmax = lmax
        self.C_l2 = dec(2)
        if lmax >= 3:
            self.C_l3 = dec(3)
            self.monomials_l3 = _generate_monomials(4, 6)
        if lmax >= 4:
            self.C_l4 = dec(4)
            self.monomials_l4 = _generate_monomials(4, 8)
            self.C_combined_l3l4 = self._build_combined_l3l4()

    def _build_combined_l3l4(self):
        """Lift l=3 (deg-6) to deg-8 by |q|^2=1 and stack with l=4 -> (130,165)."""
        idx = {m: i for i, m in enumerate(self.monomials_l4)}
        lifted = torch.zeros(self.C_l3.shape[0], len(self.monomials_l4),
                             dtype=self.C_l3.dtype, device=self.C_l3.device)
        for j, (a, b, c, d) in enumerate(self.monomials_l3):
            for m8 in [(a + 2, b, c, d), (a, b + 2, c, d), (a, b, c + 2, d), (a, b, c, d + 2)]:
                lifted[:, idx[m8]] += self.C_l3[:, j]
        return torch.cat([lifted, self.C_l4], dim=0)


def wigner_from_edge(edge_vec: torch.Tensor, lmax: int, kernels: WignerKernels,
                     gamma: float = 0.0) -> torch.Tensor:
    """``edge_vec`` [E,3] -> block-diagonal Wigner-D [E,(lmax+1)^2,(lmax+1)^2] for the edge->+Y frame,
    fully differentiable in ``edge_vec``. ``gamma`` is the deterministic roll (0 by default) that
    keeps forces conservative (fairchem randomizes it for training augmentation; we fix it)."""
    en = torch.nn.functional.normalize(edge_vec, dim=-1)
    q = quaternion_edge_to_y_stable(en)
    if gamma != 0.0:
        q = quaternion_multiply(quaternion_y_rotation(en.new_full((q.shape[0],), gamma)), q)
    size = (lmax + 1) ** 2
    D = q.new_zeros(q.shape[0], size, size)
    D[:, 0, 0] = 1.0
    if lmax >= 1:
        D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
    if lmax >= 2:
        D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q, kernels.C_l2)
    if lmax >= 4:
        D3, D4 = quaternion_to_wigner_d_l3l4_batched(q, kernels.C_combined_l3l4, kernels.monomials_l4)
        D[:, 9:16, 9:16] = D3
        D[:, 16:25, 16:25] = D4
    elif lmax >= 3:
        D[:, 9:16, 9:16] = quaternion_to_wigner_d_matmul(q, 3, kernels.C_l3, kernels.monomials_l3)
    return D
