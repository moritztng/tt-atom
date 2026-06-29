"""A faithful PyTorch transcription of the TT-Atom device forward.

This is the differentiable *oracle* for the analytic-force work: it computes exactly the same
mathematical function as the ttnn backbone (same m-primed SO(2) split, same one-hot scatter,
same RMS-norm-SH), so ``torch.autograd`` on it yields the ground-truth adjoints that the
hand-written on-device VJP (``tt_atom/forces.py``) must reproduce. It also serves as the host
forward that composes with ``tt_atom/geometry.py`` for the full ``-dE/dpos`` force.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tt_atom.activation import _expand_index_m_prime


def _l_of_coeff(lmax):
    return [l for l in range(lmax + 1) for _ in range(2 * l + 1)]


def radial_mlp(x_edge, w, p):
    x = F.linear(x_edge, w[f"{p}.net.0.weight"], w[f"{p}.net.0.bias"])
    x = F.layer_norm(x, (x.shape[-1],), w[f"{p}.net.1.weight"], w[f"{p}.net.1.bias"], 1e-5)
    x = F.silu(x)
    x = F.linear(x, w[f"{p}.net.3.weight"], w[f"{p}.net.3.bias"])
    x = F.layer_norm(x, (x.shape[-1],), w[f"{p}.net.4.weight"], w[f"{p}.net.4.bias"], 1e-5)
    x = F.silu(x)
    return F.linear(x, w[f"{p}.net.6.weight"], w[f"{p}.net.6.bias"])


def so2(x, x_edge, w, p, lmax, mmax, Cin, H, extra):
    E, nsph = x.shape[0], (lmax + 1) ** 2
    xf = x.reshape(E, nsph * Cin)
    num_coef = [lmax - m + 1 for m in range(mmax + 1)]
    off = [0, (lmax + 1) * Cin]
    for m in range(1, mmax + 1):
        off.append(off[-1] + 2 * num_coef[m] * Cin)
    if f"{p}.rad_func.net.0.weight" in w:
        rad = radial_mlp(x_edge, w, f"{p}.rad_func")
        sizes = [num_coef[m] * Cin for m in range(mmax + 1)]
        o, rms = 0, []
        for m in range(mmax + 1):
            rms.append(rad[:, o:o + sizes[m]]); o += sizes[m]
        mult = torch.cat([rms[0]] + sum(([rms[m], rms[m]] for m in range(1, mmax + 1)), []), 1)
        xf = xf * mult
    blocks = []
    x0 = F.linear(xf[:, off[0]:off[1]], w[f"{p}.fc_m0.weight"], w[f"{p}.fc_m0.bias"])
    extra_t = None
    if extra:
        extra_t, x0 = x0[:, :extra], x0[:, extra:]
    blocks.append(x0)
    for m in range(1, mmax + 1):
        nc = num_coef[m]
        blk = xf[:, off[m]:off[m + 1]].reshape(E, 2, nc * Cin)
        blk = blk @ w[f"{p}.so2_m_conv.{m-1}.fc.weight"].T
        Hh = blk.shape[-1] // 2
        blk = blk.reshape(E, 4 * Hh)
        a, b, c, d = blk[:, :Hh], blk[:, Hh:2 * Hh], blk[:, 2 * Hh:3 * Hh], blk[:, 3 * Hh:]
        blocks.append(a - d); blocks.append(c + b)
    out = torch.cat(blocks, 1).reshape(E, nsph, H)
    return (out, extra_t) if extra else out


def rms_norm_sh(x, w, p, lmax, C, eps=1e-5):
    nsph = (lmax + 1) ** 2
    lc = _l_of_coeff(lmax)
    bdw = torch.tensor([1.0 / (2 * l + 1) / (lmax + 1) for l in lc]).view(1, nsph, 1)
    l0 = x[:, 0:1, :]
    x = torch.cat([l0 - l0.mean(2, keepdim=True), x[:, 1:, :]], 1)
    fn = (x * x * bdw).sum(1, keepdim=True).mean(2, keepdim=True)
    fn = torch.rsqrt(fn + eps)
    aw = w[f"{p}.affine_weight"][torch.tensor(lc)].view(1, nsph, C)
    out = x * (fn * aw)
    ab = w[f"{p}.affine_bias"].view(1, 1, C)
    return torch.cat([out[:, 0:1, :] + ab, out[:, 1:, :]], 1)


def gate(gating, x, lmax, mmax, H):
    N = x.shape[0]
    ei = _expand_index_m_prime(lmax, mmax)
    g = torch.sigmoid(gating).view(N, lmax, H)[:, ei, :]
    return torch.cat([F.silu(x[:, 0:1, :]), x[:, 1:, :] * g], 1)


def grid_atomwise(x, w, p, to_grid, from_grid):
    b, a, nsph = to_grid.shape
    tg, fg = to_grid.reshape(b * a, nsph), from_grid.reshape(b * a, nsph)
    g = torch.einsum("pi,nic->npc", tg, x)
    g = F.silu(g @ w[f"{p}.grid_mlp.0.weight"].T)
    g = F.silu(g @ w[f"{p}.grid_mlp.2.weight"].T)
    g = g @ w[f"{p}.grid_mlp.4.weight"].T
    return torch.einsum("pi,npc->nic", fg, g)


def edgewise(x, w, p, cfg, wigner, winv, x_edge, envelope, edge_index, N):
    C, H = cfg["sphere_channels"], cfg["hidden_channels"]
    lmax, mmax = cfg["lmax"], cfg["mmax"]
    nsph = (lmax + 1) ** 2
    src, tgt = edge_index[0], edge_index[1]
    m = torch.cat([x[src], x[tgt]], dim=2)
    m = torch.bmm(wigner, m)
    m, gating = so2(m, x_edge, w, f"{p}.so2_conv_1", lmax, mmax, 2 * C, H, lmax * H)
    m = gate(gating, m, lmax, mmax, H)
    m = so2(m, x_edge, w, f"{p}.so2_conv_2", lmax, mmax, H, C, 0)
    m = m * envelope
    m = torch.bmm(winv, m)
    out = torch.zeros(N, nsph, C, dtype=m.dtype)
    out.index_add_(0, tgt, m)
    return out


def backbone(w, cfg, x_init, wigner, winv, x_edge, envelope, sys_emb, edge_index,
             to_grid, from_grid):
    C, lmax = cfg["sphere_channels"], cfg["lmax"]
    N = x_init.shape[0]
    x = x_init
    for i in range(cfg["num_layers"]):
        p = f"blocks.{i}"
        x_res = x
        n = rms_norm_sh(x, w, f"{p}.norm_1", lmax, C)
        n = torch.cat([n[:, 0:1, :] + sys_emb.view(N, 1, C), n[:, 1:, :]], 1)
        x = edgewise(n, w, f"{p}.edge_wise", cfg, wigner, winv, x_edge, envelope, edge_index, N) + x_res
        x_res = x
        n = rms_norm_sh(x, w, f"{p}.norm_2", lmax, C)
        x = grid_atomwise(n, w, f"{p}.atom_wise", to_grid, from_grid) + x_res
    return rms_norm_sh(x, w, "norm", lmax, C)


def energy(node_emb, w):
    h = node_emb[:, 0, :]
    h = F.silu(F.linear(h, w["energy_block.0.weight"], w["energy_block.0.bias"]))
    h = F.silu(F.linear(h, w["energy_block.2.weight"], w["energy_block.2.bias"]))
    h = F.linear(h, w["energy_block.4.weight"], w["energy_block.4.bias"])
    return h.sum()
