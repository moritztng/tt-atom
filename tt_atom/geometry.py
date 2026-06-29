"""Host geometry: the differentiable ``pos -> {geometric device inputs}`` map (torch, host).

These are the per-edge geometric terms (Wigner rotation, radial edge embedding, envelope, the
edge-degree node init) that the device backbone consumes as fixed inputs. They are <1% of the
compute, so we keep them on host where ``torch.autograd`` supplies the cheap geometric Jacobian
``d(terms)/dpos`` for the analytic force. Nothing here imports fairchem (it must coexist with
ttnn / numpy<2); the pure-torch rotation helpers are vendored from fairchem (MIT) with the
e3nn 0.4.0 Wigner-D construction they themselves borrow.

The roll angle ``gamma`` is a gauge the architecture is invariant to; we fix it (default 0) so
the geometry — and therefore the forward and the force — is deterministic.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-7


# ----------------------------------------------------------- rotation (vendored, MIT/e3nn 0.4)


class _Safeacos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x.clamp(-1 + EPS, 1 - EPS))
        return torch.acos(x)

    @staticmethod
    def backward(ctx, g):
        (xc,) = ctx.saved_tensors
        return -g / torch.sqrt(1 - xc.pow(2)).clamp(min=EPS)


class _Safeatan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, g):
        y, x = ctx.saved_tensors
        denom = (x.pow(2) + y.pow(2)).clamp(min=EPS)
        return g * x / denom, -g * y / denom


def _euler_angles(edge_vec, gamma_val):
    xyz = F.normalize(edge_vec).clamp(-1.0, 1.0)
    x, y, z = torch.split(xyz, 1, dim=1)
    beta = _Safeacos.apply(y.squeeze(-1))
    alpha = _Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = torch.full_like(alpha, gamma_val)
    return -gamma, -beta, -alpha          # intrinsic -> extrinsic


def _z_rot_mat(angle, lv):
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))
    inds = list(range(2 * lv + 1))
    rinds = list(range(2 * lv, -1, -1))
    freqs = list(range(lv, -lv - 1, -1))
    for i in range(len(freqs)):
        M[..., inds[i], rinds[i]] = torch.sin(freqs[i] * angle)
        M[..., inds[i], inds[i]] = torch.cos(freqs[i] * angle)
    return M


def _wigner_D(lv, alpha, beta, gamma, Jd):
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = Jd[lv]
    return _z_rot_mat(alpha, lv) @ J @ _z_rot_mat(beta, lv) @ J @ _z_rot_mat(gamma, lv)


def _eulers_to_wigner(eulers, lmax, Jd):
    alpha, beta, gamma = eulers
    size = (lmax + 1) ** 2
    wigner = torch.zeros(len(alpha), size, size, dtype=alpha.dtype)
    start = 0
    for lv in range(lmax + 1):
        blk = _wigner_D(lv, alpha, beta, gamma, Jd)
        end = start + blk.shape[1]
        wigner[:, start:end, start:end] = blk
        start = end
    return wigner


# --------------------------------------------------------------------------- radial MLP (host)


def radial_mlp(x, w, p):
    x = F.linear(x, w[f"{p}.net.0.weight"], w[f"{p}.net.0.bias"])
    x = F.layer_norm(x, (x.shape[-1],), w[f"{p}.net.1.weight"], w[f"{p}.net.1.bias"], 1e-5)
    x = F.silu(x)
    x = F.linear(x, w[f"{p}.net.3.weight"], w[f"{p}.net.3.bias"])
    x = F.layer_norm(x, (x.shape[-1],), w[f"{p}.net.4.weight"], w[f"{p}.net.4.bias"], 1e-5)
    x = F.silu(x)
    return F.linear(x, w[f"{p}.net.6.weight"], w[f"{p}.net.6.bias"])


# --------------------------------------------------------------------------- the geometry


class HostGeometry:
    def __init__(self, weights, cfg, to_m, gauss_offset, gauss_coeff, *, gamma=0.0):
        self.w = weights
        self.cfg = cfg
        self.lmax = cfg["lmax"]
        self.C = cfg["sphere_channels"]
        self.cutoff = cfg["cutoff"]
        self.gamma = gamma
        self.Jd = [weights[f"Jd_{l}"] for l in range(self.lmax + 1)]
        self.to_m = to_m
        self.offset = gauss_offset
        self.coeff = float(gauss_coeff.reshape(-1)[0])
        p = float(5)                                       # PolynomialEnvelope exponent
        self.env_a = -(p + 1) * (p + 2) / 2
        self.env_b = p * (p + 2)
        self.env_c = -p * (p + 1) / 2
        self.env_p = p
        self.rescale = 5.0                                 # edge_degree_embedding.rescale_factor

    def _wigner(self, edge_vec):
        wig = _eulers_to_wigner(_euler_angles(edge_vec, self.gamma), self.lmax, self.Jd)
        wig_inv = torch.transpose(wig, 1, 2).contiguous()
        # mmax == lmax for our config -> no coefficient subselection
        wig_M = torch.einsum("mk,nkj->nmj", self.to_m, wig)
        wig_M_inv = torch.einsum("njk,mk->njm", wig_inv, self.to_m)
        return wig_M, wig_M_inv

    def __call__(self, pos, atomic_numbers, edge_index, sys_node_embedding, edge_vec=None):
        """Returns a dict of differentiable geometric device-inputs as functions of ``pos``."""
        w, C, lmax = self.w, self.C, self.lmax
        src, tgt = edge_index[0], edge_index[1]
        if edge_vec is None:
            edge_vec = pos[src] - pos[tgt]          # fairchem edge_distance_vec convention
        dist = torch.linalg.norm(edge_vec, dim=1)

        wig_M, wig_M_inv = self._wigner(edge_vec)

        # x_edge = [gaussian(dist) | source_emb[Z] | target_emb[Z]]
        gauss = torch.exp(self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1)) ** 2)
        se = F.embedding(atomic_numbers[src], w["source_embedding.weight"])
        te = F.embedding(atomic_numbers[tgt], w["target_embedding.weight"])
        x_edge = torch.cat([gauss, se, te], dim=1)

        # envelope
        ds = dist / self.cutoff
        env_val = 1 + (ds ** self.env_p) * (self.env_a + ds * (self.env_b + self.env_c * ds))
        envelope = torch.where(ds < 1, env_val, torch.zeros_like(env_val)).reshape(-1, 1, 1)

        # edge-degree node init
        N = atomic_numbers.shape[0]
        nsph = (lmax + 1) ** 2
        m0 = self.cfg["mmax_m0_coeffs"] if "mmax_m0_coeffs" in self.cfg else (lmax + 1)
        edm = radial_mlp(x_edge, w, "edge_degree_embedding.rad_func").reshape(-1, m0, C)
        edm = F.pad(edm, (0, 0, 0, nsph - m0))             # [E,9,C]
        edm = torch.bmm(wig_M_inv, edm) * envelope
        node = torch.zeros(N, nsph, C, dtype=pos.dtype)
        node = node.index_add(0, tgt, edm / self.rescale)
        # pos-independent l=0 init
        l0 = F.embedding(atomic_numbers, w["sphere_embedding.weight"]) + sys_node_embedding
        l0 = F.pad(l0.unsqueeze(1), (0, 0, 0, nsph - 1))   # [N,9,C] with only l0 set
        x_init = node + l0

        return dict(x_init=x_init, wigner=wig_M, wigner_inv=wig_M_inv,
                    x_edge=x_edge, edge_envelope=envelope, edge_distance=dist,
                    edge_distance_vec=edge_vec)
