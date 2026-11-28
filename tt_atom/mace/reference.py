"""Float64 CPU reference for MACE-MP-0b3 medium, written from the mace-torch / e3nn equations.

No e3nn or mace imports: each function takes plain tensors and the weights dict a golden carries
(``w@<module path>``), so a parity miss is pinned to one block. Graded against the mace-torch
golden by ``tests/test_mace_reference.py``.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from tt_atom.equivariant import wigner_3j


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


def spherical_harmonics(vec):
    """e3nn real spherical harmonics to l = 3, normalize=True, component normalisation, e3nn's (x, y, z) order:
    the same basis Orb uses, so one implementation (and one closed-form VJP) serves both."""
    from tt_atom.orb_geometry import spherical_harmonics_l3
    return spherical_harmonics_l3(torch.nn.functional.normalize(vec, dim=-1))


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


# ---- e3nn building blocks. Irreps are lists of (mul, l) with the natural parity (-1)^l, which is all
# MACE-MP uses; features are e3nn's layout, each block [mul, 2l+1] flattened mul-major.

def split(x, irreps):
    out, i = [], 0
    for mul, l in irreps:
        d = 2 * l + 1
        out.append(x[:, i:i + mul * d].reshape(-1, mul, d))
        i += mul * d
    return out


def merge(blocks):
    return torch.cat([b.reshape(b.shape[0], -1) for b in blocks], dim=1)


def linear(x, irreps_in, irreps_out, weight):
    """e3nn o3.Linear, element path normalisation: each output sums every same-l input / sqrt(fan-in)."""
    xs, outs, k = split(x, irreps_in), [], 0
    out = [torch.zeros(x.shape[0], m, 2 * l + 1, dtype=x.dtype) for m, l in irreps_out]
    fan = [sum(mi for mi, li in irreps_in if li == l) for _, l in irreps_out]
    for i, (mi, li) in enumerate(irreps_in):
        for o, (mo, lo) in enumerate(irreps_out):
            if li == lo:
                W = weight[k:k + mi * mo].reshape(mi, mo)
                k += mi * mo
                out[o] = out[o] + torch.einsum("nui,uv->nvi", xs[i], W) / math.sqrt(fan[o])
    assert k == weight.numel()
    return merge(out)


def fc_net(x, weights, act=True):
    """e3nn nn.FullyConnectedNet: x @ W / sqrt(fan-in), normalised SiLU between layers, none after the last."""
    for i, W in enumerate(weights):
        x = x @ W / math.sqrt(W.shape[0])
        if act and i < len(weights) - 1:
            x = torch.nn.functional.silu(x) * _silu_2mom()
    return x


@lru_cache(maxsize=None)
def _silu_2mom():
    """e3nn normalize2mom constant for SiLU: its own seed-0 Monte Carlo estimate, reproduced exactly."""
    z = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    return torch.nn.functional.silu(z).pow(2).mean().pow(-0.5).item()


def conv_tp_irreps(irreps1, sh_ls, target_ls):
    """MACE tp_out_irreps_with_instructions: every allowed (i, j, l_out), outputs sorted by l (stable)."""
    ins = [(i, j, lo) for i, (_, l1) in enumerate(irreps1) for j, l2 in enumerate(sh_ls)
           for lo in range(abs(l1 - l2), l1 + l2 + 1) if lo in target_ls and (l1 + l2 + lo) % 2 == 0]
    ins = sorted(ins, key=lambda t: t[2])
    return [(irreps1[i][0], lo) for i, _, lo in ins], ins


def conv_tp(x1, irreps1, sh, sh_ls, w, target_ls):
    """'uvu' tensor product, per-edge weights: one output block per instruction, component normalised."""
    irreps_mid, ins = conv_tp_irreps(irreps1, sh_ls, target_ls)
    xs, ys, out, k = split(x1, irreps1), split(sh, [(1, l) for l in sh_ls]), [], 0
    for i, j, lo in ins:
        mul, l1 = irreps1[i]
        C = wigner_3j(l1, sh_ls[j], lo) * math.sqrt(2 * lo + 1)
        out.append(torch.einsum("eui,ej,ijk,eu->euk", xs[i], ys[j][:, 0], C, w[:, k:k + mul]))
        k += mul
    return merge(out), irreps_mid


def fc_tp(x1, irreps1, attrs, irreps_out, weight):
    """e3nn FullyConnectedTensorProduct with scalar node attributes (the element-selector skip)."""
    n_el, xs, k = attrs.shape[1], split(x1, irreps1), 0
    out = [torch.zeros(x1.shape[0], m, 2 * l + 1, dtype=x1.dtype) for m, l in irreps_out]
    fan = [sum(mi * n_el for mi, li in irreps1 if li == l) for _, l in irreps_out]
    for i, (mi, li) in enumerate(irreps1):
        for o, (mo, lo) in enumerate(irreps_out):
            if li == lo:
                W = weight[k:k + mi * n_el * mo].reshape(mi, n_el, mo)
                k += mi * n_el * mo
                out[o] = out[o] + torch.einsum("nui,nv,uvw->nwi", xs[i], attrs, W) / math.sqrt(fan[o])
    assert k == weight.numel()
    return merge(out)


def reshape_irreps(x, irreps):
    """[n, sum mul*(2l+1)] -> [n, mul, sum(2l+1)] (MACE reshape_irreps, equal muls)."""
    return torch.cat(split(x, irreps), dim=-1)


def density_interaction(w, layer, attrs, feats, irreps_in, sh, edge_feats, edge_index, irreps_out,
                        sc_irreps=None, sh_ls=(0, 1, 2, 3)):
    """RealAgnosticDensity[Residual]InteractionBlock. With ``sc_irreps`` (the residual variant, MP-0b3 layer 1)
    the element-selector TP runs on the input and is returned as the skip; otherwise it runs on the message.
    Returns (node features [n, mul, (lmax+1)^2], skip or None)."""
    p = f"interactions.{layer}."
    sender, receiver = edge_index
    n = feats.shape[0]
    sc = fc_tp(feats, irreps_in, attrs, sc_irreps, w[p + "skip_tp.weight"]) if sc_irreps else None
    up = linear(feats, irreps_in, irreps_in, w[p + "linear_up.weight"])  # MACE keeps the irreps here
    tp_w = fc_net(edge_feats, _layers(w, p + "conv_tp_weights"))
    dens = torch.tanh(fc_net(edge_feats, _layers(w, p + "density_fn")) ** 2)
    density = torch.zeros(n, 1, dtype=feats.dtype).index_add_(0, receiver, dens)
    mji, irreps_mid = conv_tp(up[sender], irreps_in, sh, list(sh_ls), tp_w, [l for _, l in irreps_out])
    msg = torch.zeros(n, mji.shape[1], dtype=feats.dtype).index_add_(0, receiver, mji)
    msg = linear(msg, irreps_mid, irreps_out, w[p + "linear.weight"]) / (density + 1)
    if sc is None:
        msg = fc_tp(msg, irreps_out, attrs, irreps_out, w[p + "skip_tp.weight"])
    return reshape_irreps(msg, irreps_out), sc


def _layers(w, prefix):
    out, i = [], 0
    while f"{prefix}.layer{i}.weight" in w:
        out.append(w[f"{prefix}.layer{i}.weight"])
        i += 1
    return out


def symmetric_contraction(x, attrs, U, W):
    """One MACE Contraction (one output irrep), correlation = len(U).

    x [n, c, (lmax+1)^2]; U[nu-1] = U_matrix_nu, shape [(2L+1,)] + [ell]*nu + [K_nu]; W[nu-1] = the
    [n_elements, K_nu, c] weights. Highest order first: contract the element-mixed weights into U, add the
    running sum, then contract one ell index with x; repeat down to nu = 1. Returns [n, c * (2L+1)].
    """
    out = None
    for nu in range(len(U), 0, -1):
        wc = torch.einsum("ekc,be->bkc", W[nu - 1], attrs)
        t = torch.einsum("...k,bkc->bc...", U[nu - 1], wc)
        out = torch.einsum("bc...i,bci->bc...", t if out is None else t + out, x)
    return out.reshape(out.shape[0], -1)


def product_block(w, layer, x, attrs, irreps_out, sc=None):
    """EquivariantProductBasisBlock: symmetric contraction per output irrep, linear, plus the skip."""
    p = f"products.{layer}."
    outs = []
    for k in range(len(irreps_out)):
        q = f"{p}symmetric_contractions.contractions.{k}."
        corr = sum(1 for key in w if key.startswith(q + "U_matrix_"))
        U = [w[f"{q}U_matrix_{nu}"] for nu in range(1, corr + 1)]
        W = [w[f"{q}weights.{corr - 1 - nu}"] for nu in range(1, corr)] + [w[q + "weights_max"]]
        outs.append(symmetric_contraction(x, attrs, U, W))
    out = linear(torch.cat(outs, dim=1), irreps_out, irreps_out, w[p + "linear.weight"])
    return out if sc is None else out + sc


def readout(w, layer, x, irreps_in):
    """Layer 0: linear to one scalar. Last layer: linear -> normalised SiLU -> linear (NonLinearReadoutBlock)."""
    p = f"readouts.{layer}."
    if p + "linear.weight" in w:
        return linear(x, irreps_in, [(1, 0)], w[p + "linear.weight"])
    hidden = w[p + "linear_2.weight"].numel()
    h = linear(x, irreps_in, [(hidden, 0)], w[p + "linear_1.weight"])
    return linear(torch.nn.functional.silu(h) * _silu_2mom(), [(hidden, 0)], [(1, 0)], w[p + "linear_2.weight"])


HIDDEN = [(128, l) for l in range(4)]  # MP-0b3 medium: 128 channels, messages up to l = 3


def energy(w, atomic_numbers, pos, edge_index, shifts):
    """Total energy of ScaleShiftMACE (MP-0b3 medium): E0 + scale * (ZBL + readouts) + shift, summed over atoms."""
    attrs = node_attrs(atomic_numbers, w["atomic_numbers"].long())
    vec, r = edge_vectors(pos, edge_index, shifts)
    sh = spherical_harmonics(vec)
    ef = radial_embedding(r, atomic_numbers, edge_index, w)
    feats, irreps, sc, node_e = node_embedding(attrs, w), [(128, 0)], None, zbl(r, atomic_numbers, edge_index, w, len(pos))
    for layer, (prod_irreps, sc_irreps) in enumerate([([(128, 0), (128, 1)], None), ([(128, 0)], [(128, 0)])]):
        x, sc = density_interaction(w, layer, attrs, feats, irreps, sh, ef, edge_index, HIDDEN, sc_irreps)
        feats, irreps = product_block(w, layer, x, attrs, prod_irreps, sc), prod_irreps
        node_e = node_e + readout(w, layer, feats, irreps).squeeze(-1)
    node_e = w["scale_shift.scale"] * node_e + w["scale_shift.shift"]
    return (attrs @ w["atomic_energies_fn.atomic_energies"]).sum() + node_e.sum()


def energy_forces(w, atomic_numbers, pos, edge_index, shifts):
    pos = pos.detach().clone().requires_grad_(True)
    e = energy(w, atomic_numbers, pos, edge_index, shifts)
    (g,) = torch.autograd.grad(e, pos)
    return e.detach(), -g
