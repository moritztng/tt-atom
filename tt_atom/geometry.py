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


def csd_embedding(w, charge, spin, sphere_channels, dataset="omat"):
    """System (charge/spin/dataset) embedding -> [nsys, C]. Mirrors fairchem ``csd_embedding``.

    Supports both charge/spin encodings: ``pos_emb`` (sin/cos of a random projection, the
    self-chosen random-weight config) and ``rand_emb`` (a learned lookup table indexed by
    charge+100 / spin, which uma-s-1 uses). The dataset token is the per-dataset embedding for
    the active task (``omol``/``omat``/``oc20``/...). Dispatch is by which keys the bundle carries."""
    if "charge_embedding.rand_emb.weight" in w:                 # rand_emb (uma-s-1)
        chg = F.embedding((charge.long() + 100), w["charge_embedding.rand_emb.weight"])
        sp = F.embedding(spin.long(), w["spin_embedding.rand_emb.weight"])
    else:                                                        # pos_emb (sin/cos)
        def _cs(x, W, is_spin):
            xp = x[:, None] * W[None, :] * 2 * math.pi
            emb = torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)
            if is_spin:
                emb = emb.clone()
                emb[torch.where(x == 0)[0]] = 0
            return emb

        chg = _cs(charge, w["charge_embedding.W"], False)
        sp = _cs(spin, w["spin_embedding.W"], True)
    ds = w[f"dataset_embedding.dataset_emb_dict.{dataset}.weight"][0].expand(charge.shape[0], -1)
    return F.silu(F.linear(torch.cat([chg, sp, ds], dim=1), w["mix_csd.weight"], w["mix_csd.bias"]))


def radius_graph(pos, cutoff, cell=None, pbc=None):
    """Brute-force O(N^2) neighbour list -> ``(edge_index[2, E], cell_shift[E, 3])``.

    ``cell_shift`` is the cartesian periodic image offset for each edge (zeros when aperiodic),
    so the caller forms ``edge_vec = pos[src] - pos[tgt] + cell_shift``. Convention matches
    fairchem ``radius_graph_pbc`` + ``get_pbc_distances`` exactly: an edge (src = imaged j,
    tgt = i) has ``distance_vec = pos[j] - pos[i] + n·cell``. The graph is host-side and a
    negligible fraction of the compute, so the O(N^2 · n_cells) brute force is fine for a cell.

    ``cell`` rows are the lattice vectors (ASE convention). ``pbc`` is a bool or length-3 mask;
    an aperiodic graph results when ``cell``/``pbc`` are absent or ``pbc`` is all-False."""
    periodic = cell is not None and pbc is not None and bool(torch.as_tensor(pbc).any())
    if not periodic:
        d = torch.linalg.norm(pos[:, None, :] - pos[None, :, :], dim=-1)
        mask = (d < cutoff) & (d > 0)
        src, tgt = torch.where(mask)
        return torch.stack([src, tgt], dim=0), pos.new_zeros(int(mask.sum()), 3)

    cell = torch.as_tensor(cell, dtype=pos.dtype)                # rows = lattice vectors
    pbc = torch.as_tensor(pbc, dtype=torch.bool).reshape(-1).expand(3)
    # perpendicular plane spacing along a_k is 1/||b_k|| (b = reciprocal rows); the number of
    # image cells needed each way is ceil(cutoff · ||b_k||) — matches fairchem's rep_a{1,2,3}.
    recip = torch.linalg.inv(cell).transpose(0, 1)               # rows = reciprocal vectors
    reps = [int(math.ceil(float(cutoff * torch.linalg.norm(recip[k])))) if bool(pbc[k]) else 0
            for k in range(3)]
    ranges = [torch.arange(-r, r + 1, dtype=pos.dtype) for r in reps]
    cells = torch.cartesian_prod(*ranges).reshape(-1, 3)         # [n_cells, 3] integer offsets
    shifts = cells @ cell                                        # [n_cells, 3] cartesian
    # disp[i, j, c] = pos[i] - (pos[j] + shift[c]); mask magnitude, keep (i, j, c) within cutoff
    disp = pos[:, None, None, :] - (pos[None, :, None, :] + shifts[None, None, :, :])
    d2 = (disp ** 2).sum(-1)                                     # [N, N, n_cells]
    mask = (d2 <= cutoff * cutoff) & (d2 > 1e-8)
    i, j, c = torch.where(mask)                                  # i = receiver, j = imaged source
    return torch.stack([j, i], dim=0), shifts[c]


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


_ZROT_FREQS: dict = {}


def _z_rot_mat(angle, lv):
    """Wigner z-rotation block: cos on the diagonal, sin on the anti-diagonal (frequency order
    ``l .. -l``). Built functionally (``diag_embed`` + column-flip) instead of a Python loop of
    per-element in-place writes — the matrix is identical (diagonal and anti-diagonal overlap only
    at the centre, where ``cos(0) + sin(0) = 1``) but the autograd graph has far fewer nodes, so
    the analytic-force VJP through the Wigner build is ~2x cheaper. Forward is bit-exact vs the
    loop; the gradient differs only by float reduction order (~1e-6)."""
    freqs = _ZROT_FREQS.get((lv, angle.dtype))
    if freqs is None:
        freqs = torch.arange(lv, -lv - 1, -1, dtype=angle.dtype)
        _ZROT_FREQS[(lv, angle.dtype)] = freqs
    fa = freqs * angle[..., None]
    return torch.diag_embed(torch.cos(fa)) + torch.diag_embed(torch.sin(fa)).flip(-1)


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
    def __init__(self, weights, cfg, to_m, gauss_offset, gauss_coeff, *, gamma=0.0,
                 coefficient_index=None):
        self.w = weights
        self.cfg = cfg
        self.lmax = cfg["lmax"]
        self.mmax = cfg.get("mmax", self.lmax)
        self.C = cfg["sphere_channels"]
        self.cutoff = cfg["cutoff"]
        self.gamma = gamma
        self.Jd = [weights[f"Jd_{l}"] for l in range(self.lmax + 1)]
        self.to_m = to_m
        # to_m is a permutation matrix (one 1 per row): the m-mapping einsums in _wigner are just a
        # coefficient reorder. Precompute the permutation so we can index_select instead of a dense
        # [E,nred,nsph] einsum -- bit-exact, and its autograd is a cheap index_add rather than a
        # matmul backprop (a large chunk of the per-step host geometric-Jacobian cost at scale).
        tm = torch.as_tensor(to_m)
        self._is_perm = bool(((tm == 0) | (tm.abs() == 1)).all()) and bool((tm != 0).sum(1).max() == 1)
        if self._is_perm:
            self._to_m_perm = tm.abs().argmax(dim=1).long()   # perm[m] = source coeff index
        # coefficient subselection for mmax<lmax (uma-m): the full (lmax+1)^2 spherical-harmonic
        # coefficients are reduced to the |m|<=mmax m-space. ``to_m`` maps that reduced space
        # (nred = to_m.shape[0]) onto the m-primed layout. None/identity when mmax==lmax (uma-s).
        self.coefficient_index = (coefficient_index.long() if coefficient_index is not None
                                  else None)
        self.nred = to_m.shape[0]                          # reduced m-space size (== nsph when mmax==lmax)
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
        # mmax<lmax: keep only the |m|<=mmax coefficient rows/cols before the m-mapping (exactly
        # fairchem's prepare_wigner). wig_M: [E, nred, nsph]; wig_M_inv: [E, nsph, nred].
        if self.coefficient_index is not None:
            wig = wig.index_select(1, self.coefficient_index)      # [E, nred, nsph]
            wig_inv = wig_inv.index_select(2, self.coefficient_index)  # [E, nsph, nred]
        if self._is_perm:
            # permutation m-mapping: wig_M[n,m,j] = wig[n, perm[m], j]; wig_M_inv[n,j,m] = wig_inv[n,j,perm[m]]
            wig_M = wig.index_select(1, self._to_m_perm)
            wig_M_inv = wig_inv.index_select(2, self._to_m_perm)
        else:
            wig_M = torch.einsum("mk,nkj->nmj", self.to_m, wig)
            wig_M_inv = torch.einsum("njk,mk->njm", wig_inv, self.to_m)
        return wig_M, wig_M_inv

    def __call__(self, pos, atomic_numbers, edge_index, sys_node_embedding, edge_vec=None,
                 edge_cell_shift=None):
        """Returns a dict of differentiable geometric device-inputs as functions of ``pos``.

        ``edge_cell_shift`` [E, 3] is the cartesian periodic image offset per edge (from
        ``radius_graph``); it is constant wrt ``pos`` so the analytic force is unaffected."""
        w, C, lmax = self.w, self.C, self.lmax
        src, tgt = edge_index[0], edge_index[1]
        if edge_vec is None:
            edge_vec = pos[src] - pos[tgt]          # fairchem edge_distance_vec convention
            if edge_cell_shift is not None:
                edge_vec = edge_vec + edge_cell_shift
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
        # the radial output is the m=0 block; place it at the front of the reduced m-space
        # (nred = nsph when mmax==lmax) and rotate back to node SH via wig_M_inv [E, nsph, nred]
        edm = F.pad(edm, (0, 0, 0, self.nred - m0))        # [E, nred, C]
        edm = torch.bmm(wig_M_inv, edm) * envelope         # [E, nsph, C]
        node = torch.zeros(N, nsph, C, dtype=pos.dtype)
        node = node.index_add(0, tgt, edm / self.rescale)
        # pos-independent l=0 init
        l0 = F.embedding(atomic_numbers, w["sphere_embedding.weight"]) + sys_node_embedding
        l0 = F.pad(l0.unsqueeze(1), (0, 0, 0, nsph - 1))   # [N,9,C] with only l0 set
        x_init = node + l0

        return dict(x_init=x_init, wigner=wig_M, wigner_inv=wig_M_inv,
                    x_edge=x_edge, edge_envelope=envelope, edge_distance=dist,
                    edge_distance_vec=edge_vec)
