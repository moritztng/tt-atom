"""MACE forces without autograd: the cm-layout forward with a hand-written reverse pass, in float64.

The card has no autograd, so this is the op list it runs: every backward step is the transpose of a forward
GEMM, a gather where the forward scattered (and the reverse), or an elementwise derivative. The network part
yields dE/d(sh) and dE/d(radial features) per edge; the per-edge geometry (spherical harmonics, Bessel radial,
ZBL) is differentiated in closed form per edge by ``geometry_vjp``, then edges scatter to atoms. No autograd.
Graded against autograd on ``dense.energy_cm`` in tests/test_mace_backward.py.
"""
from __future__ import annotations

import torch

from . import dense
from . import reference as ref

HIDDEN = [(128, l) for l in range(4)]
LAYERS = (([(128, 0)], [(128, 0), (128, 1)], False), ([(128, 0), (128, 1)], [(128, 0)], True))


def linear_t(g, plan, a_in, mi):
    """Transpose of dense.linear_cm."""
    out = torch.zeros(g.shape[0], a_in, mi, dtype=g.dtype)
    for ai, ao, d, W in plan:
        out[:, ai:ai + d] += g[:, ao:ao + d] @ W.T
    return out


def skip_t(g, element, plans, a_in, mi):
    out = torch.zeros(g.shape[0], a_in, mi, dtype=g.dtype)
    for e, plan in enumerate(plans):
        rows = element == e
        out[rows] = linear_t(g[rows], plan, a_in, mi)
    return out


def fc_fwd(x, Ws):
    """FullyConnectedNet keeping the pre-activations for the reverse pass."""
    pre, c = [], ref._silu_2mom()
    for i, W in enumerate(Ws):
        x = x @ W / W.shape[0] ** 0.5
        if i < len(Ws) - 1:
            pre.append(x)
            x = torch.nn.functional.silu(x) * c
    return x, pre


def fc_bwd(g, Ws, pre):
    c = ref._silu_2mom()
    for i in range(len(Ws) - 1, -1, -1):
        g = g @ Ws[i].T / Ws[i].shape[0] ** 0.5
        if i > 0:
            s = torch.sigmoid(pre[i - 1])
            g = g * c * (s * (1 + pre[i - 1] * (1 - s)))
    return g


def _interaction_fwd(w, layer, feats, element, elements, sh, ef, edge_index, irreps_in, residual):
    p, (snd, rcv), n = f"interactions.{layer}.", edge_index, feats.shape[0]
    a_in = sum(2 * l + 1 for _, l in irreps_in)
    sk_in, sk_out = (irreps_in, [(128, 0)]) if residual else (HIDDEN, HIDDEN)
    sk = dense.skip_plan(sk_in, len(w["atomic_numbers"]), sk_out, w[p + "skip_tp.weight"], elements)
    C, ins_of_k, mid = dense.conv_tp_plan(irreps_in, [0, 1, 2, 3], [0, 1, 2, 3])
    K, A = C.shape[2], C.shape[0]
    up_plan = dense.linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"])
    lin_plan = dense.linear_plan(mid, HIDDEN, w[p + "linear.weight"])
    Wt, Wd = ref._layers(w, p + "conv_tp_weights"), ref._layers(w, p + "density_fn")
    up = dense.linear_cm(feats, up_plan, a_in, 128)
    x1 = up[snd]
    tw, tw_pre = fc_fwd(ef, Wt)
    tw = tw.reshape(len(snd), -1, 128)
    u, u_pre = fc_fwd(ef, Wd)
    dens = torch.tanh(u ** 2)
    density = torch.zeros(n, 1, dtype=feats.dtype).index_add_(0, rcv, dens)
    Cf = C.permute(1, 2, 0).reshape(C.shape[1], -1)  # [B, K * A]
    MT = (sh @ Cf).reshape(len(snd), K, A)
    Z = MT @ x1  # [E, K, 128]
    m = Z * tw[:, ins_of_k, :]
    raw = torch.zeros(n, K, 128, dtype=feats.dtype).index_add_(0, rcv, m)
    lin = dense.linear_cm(raw, lin_plan, 16, 128)
    msg = lin / (density + 1)[:, :, None]
    out = msg if residual else dense.skip_cm(msg, element, sk, 16, 128)
    sc = dense.skip_cm(feats, element, sk, 1, 128) if residual else None
    tape = dict(sk=sk, a_in=a_in, ins_of_k=ins_of_k, up_plan=up_plan, lin_plan=lin_plan, Wt=Wt, Wd=Wd, x1=x1,
                tw=tw, tw_pre=tw_pre, u=u, u_pre=u_pre, dens=dens, density=density, Cf=Cf, MT=MT, Z=Z, lin=lin,
                K=K, A=A, residual=residual)
    return out, sc, tape


def _interaction_bwd(g_out, g_sc, t, element, edge_index):
    """-> (g_feats [n, a_in, 128], g_sh [E, 16], g_ef [E, 8])."""
    snd, rcv = edge_index
    n = g_out.shape[0]
    g_msg = g_out if t["residual"] else skip_t(g_out, element, t["sk"], 16, 128)
    d1 = (t["density"] + 1)[:, :, None]
    g_lin = g_msg / d1
    g_density = -(g_msg * t["lin"]).sum(dim=(1, 2))[:, None] / (t["density"] + 1) ** 2  # [n, 1]
    g_raw = linear_t(g_lin, t["lin_plan"], t["K"], 128)
    g_m = g_raw[rcv]  # scatter^T = gather
    wk = t["tw"][:, t["ins_of_k"], :]
    g_Z = g_m * wk
    g_wk = g_m * t["Z"]
    g_tw = torch.zeros_like(t["tw"]).index_add_(1, t["ins_of_k"], g_wk)
    g_MT = g_Z @ t["x1"].transpose(1, 2)  # [E, K, A]
    g_x1 = t["MT"].transpose(1, 2) @ g_Z  # [E, A, 128]
    g_sh = g_MT.reshape(len(snd), -1) @ t["Cf"].T
    g_ef = fc_bwd(g_tw.reshape(len(snd), -1), t["Wt"], t["tw_pre"])
    g_dens = g_density[rcv]  # [E, 1]
    g_u = g_dens * (1 - t["dens"] ** 2) * 2 * t["u"]
    g_ef = g_ef + fc_bwd(g_u, t["Wd"], t["u_pre"])
    g_up = torch.zeros(n, t["a_in"], 128, dtype=g_out.dtype).index_add_(0, snd, g_x1)  # gather^T = scatter
    g_feats = linear_t(g_up, t["up_plan"], t["a_in"], 128)
    if g_sc is not None:
        g_feats = g_feats + skip_t(g_sc, element, t["sk"], t["a_in"], 128)
    return g_feats, g_sh, g_ef


def _product_fwd(w, layer, x, element, elements, irreps_out, sc):
    a = sum(2 * l + 1 for _, l in irreps_out)
    tables = dense.product_tables(w, layer, len(irreps_out), elements=elements)
    _, parent, last = tables[0]
    T = torch.cat([tt for tt, _, _ in tables], dim=-1)  # [E, c, F, a]
    sels = dense.monomial_selectors(parent, last, 16)
    n = x.shape[0]
    x2 = x.transpose(1, 2).reshape(n * 128, 16)  # rows (node, channel)
    feats, xls, prev = [], [], None
    for Sp, Sl in sels:
        xl = x2 @ Sl
        xls.append(xl)
        prev = xl if Sp is None else (prev @ Sp) * xl
        feats.append(prev)
    P = torch.cat(feats, dim=1).reshape(n, 128, -1)
    s = torch.zeros(n, 128, a, dtype=x.dtype)
    for e in range(T.shape[0]):
        rows = element == e
        s[rows] = torch.einsum("ncf,cfa->nca", P[rows], T[e])
    s = s.transpose(1, 2)  # [n, a, 128]
    plan = dense.linear_plan(irreps_out, irreps_out, w[f"products.{layer}.linear.weight"])
    y = dense.linear_cm(s, plan, a, 128)
    if sc is not None:
        y = y + sc
    r = f"readouts.{layer}."
    y0 = y[:, 0, :]  # every readout reads the l = 0 channels only
    if r + "linear.weight" in w:
        W1, h, W2 = w[r + "linear.weight"].reshape(128, 1) / 128 ** 0.5, None, None
        e = (y0 @ W1)[:, 0]
    else:
        hid = w[r + "linear_2.weight"].numel()
        W1 = w[r + "linear_1.weight"].reshape(128, hid) / 128 ** 0.5
        W2 = w[r + "linear_2.weight"].reshape(hid, 1) / hid ** 0.5
        h = y0 @ W1
        e = (torch.nn.functional.silu(h) * ref._silu_2mom() @ W2)[:, 0]
    tape = dict(T=T, sels=sels, feats=feats, xls=xls, plan=plan, a=a, W1=W1, W2=W2, h=h, n=n)
    return y, e, tape


def _product_bwd(g_e, g_y, t, element):
    """g_e [n] (dE/d node energy), g_y [n, a, 128] or None (from the next layer) -> (g_x [n, 16, 128], g_y_total)."""
    n, a = t["n"], t["a"]
    g_y = torch.zeros(n, a, 128, dtype=g_e.dtype) if g_y is None else g_y.clone()
    if t["W2"] is None:
        g_y[:, 0, :] += g_e[:, None] * t["W1"][:, 0]
    else:
        sg = torch.sigmoid(t["h"])
        g_h = g_e[:, None] * t["W2"][:, 0] * ref._silu_2mom() * (sg * (1 + t["h"] * (1 - sg)))
        g_y[:, 0, :] += g_h @ t["W1"].T
    g_s = linear_t(g_y, t["plan"], a, 128).transpose(1, 2)  # [n, 128, a]
    T = t["T"]
    g_P = torch.zeros(n, 128, T.shape[2], dtype=g_e.dtype)
    for e in range(T.shape[0]):
        rows = element == e
        g_P[rows] = torch.einsum("nca,cfa->ncf", g_s[rows], T[e])
    g_P = g_P.reshape(n * 128, -1)
    widths = [f.shape[1] for f in t["feats"]]
    offs = [sum(widths[:i]) for i in range(len(widths))]
    g_x2 = torch.zeros(n * 128, 16, dtype=g_e.dtype)
    g_prev = None
    for k in range(len(t["sels"]) - 1, -1, -1):  # degree nu = (deg nu-1 @ Sp) * (x @ Sl)
        Sp, Sl = t["sels"][k]
        g_f = g_P[:, offs[k]:offs[k] + widths[k]] + (0 if g_prev is None else g_prev)
        if Sp is None:
            g_x2 += g_f @ Sl.T
        else:
            g_x2 += (g_f * (t["feats"][k - 1] @ Sp)) @ Sl.T
            g_prev = (g_f * t["xls"][k]) @ Sp.T
    return g_x2.reshape(n, 128, 16).transpose(1, 2), g_y


def energy_forces(w, atomic_numbers, pos, edge_index, shifts):
    el = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long()).argmax(1)
    elements = el.unique()
    element = torch.searchsorted(elements, el)
    attrs = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long())
    vec, r = ref.edge_vectors(pos, edge_index, shifts)
    sh, ef = ref.spherical_harmonics(vec), ref.radial_embedding(r, atomic_numbers, edge_index, w)
    zbl = ref.zbl(r, atomic_numbers, edge_index, w, len(pos))
    feats, irreps, node_e, tapes = ref.node_embedding(attrs, w)[:, None, :], [(128, 0)], zbl, []
    for layer, (irr_in, irr_out, res) in enumerate(LAYERS):
        x, sc, ti = _interaction_fwd(w, layer, feats, element, elements, sh, ef, edge_index, irr_in, res)
        feats, e, tp = _product_fwd(w, layer, x, element, elements, irr_out, sc)
        node_e, tapes = node_e + e, tapes + [(ti, tp)]
    scale = w["scale_shift.scale"]
    energy = (attrs @ w["atomic_energies_fn.atomic_energies"]).sum() + (scale * node_e + w["scale_shift.shift"]).sum()
    # reverse pass
    g_e = torch.full((len(pos),), float(scale), dtype=pos.dtype)
    g_sh, g_ef, g_feats = torch.zeros_like(sh), torch.zeros_like(ef), None
    for layer in (1, 0):
        ti, tp = tapes[layer]
        g_x, g_y = _product_bwd(g_e, g_feats, tp, element)
        g_feats, gs, ge = _interaction_bwd(g_x, g_y if ti["residual"] else None, ti, element, edge_index)
        g_sh, g_ef = g_sh + gs, g_ef + ge
    return energy, -geometry_vjp(w, atomic_numbers, pos, edge_index, shifts, g_sh, g_ef, float(scale))


def _cutoff_and_deriv(r, r_max, p):
    t = r / r_max
    env = 1.0 - (p + 1.0) * (p + 2.0) / 2.0 * t ** p + p * (p + 2.0) * t ** (p + 1) - p * (p + 1.0) / 2.0 * t ** (p + 2)
    d = (-(p + 1.0) * (p + 2.0) / 2.0 * p * t ** (p - 1) + p * (p + 2.0) * (p + 1) * t ** p
         - p * (p + 1.0) * (p + 2.0) / 2.0 * t ** (p + 1)) / r_max
    inside = r < r_max
    return env * inside, d * inside


def radial_deriv(r, atomic_numbers, edge_index, w):
    """d(reference.radial_embedding)/dr, closed form: Agnesi transform -> Bessel basis, times the cutoff."""
    pre = "radial_embedding."
    cut, dcut = _cutoff_and_deriv(r, w[pre + "cutoff_fn.r_max"], w[pre + "cutoff_fn.p"])
    zu, zv = ref._pair_z(atomic_numbers, edge_index)
    cov = w[pre + "distance_transform.covalent_radii"]
    a, p, q = (w[pre + "distance_transform." + k] for k in "apq")
    r0 = 0.5 * (cov[zu] + cov[zv])
    t = r / r0
    den = 1.0 + t ** (q - p)
    D = a * t ** q / den
    dD = a * (q * t ** (q - 1) * den - t ** q * (q - p) * t ** (q - p - 1)) / den ** 2
    x = 1.0 / (1.0 + D)
    dx = -dD / (1.0 + D) ** 2 / r0
    bw, pf = w[pre + "bessel_fn.bessel_weights"], w[pre + "bessel_fn.prefactor"]
    b = pf * torch.sin(bw * x) / x
    db = pf * (bw * torch.cos(bw * x) * x - torch.sin(bw * x)) / x ** 2
    return db * dx * cut + b * dcut


def zbl_deriv(r, atomic_numbers, edge_index, w):
    """d(per-edge ZBL energy)/dr, closed form (reference.zbl sums these onto receivers)."""
    pre = "pair_repulsion_fn."
    zu, zv = ref._pair_z(atomic_numbers, edge_index)
    zu, zv = zu.to(r.dtype), zv.to(r.dtype)
    a_exp = w.get(pre + "a_exp", torch.tensor(0.300, dtype=r.dtype))
    a_pre = w.get(pre + "a_prefactor", torch.tensor(0.4543, dtype=r.dtype))
    a = a_pre * 0.529 / (zu ** a_exp + zv ** a_exp)
    c, k = w[pre + "c"], (3.2, 0.9423, 0.4028, 0.2016)
    phi = sum(c[i] * torch.exp(-k[i] * r / a) for i in range(4))
    dphi = sum(-c[i] * k[i] / a * torch.exp(-k[i] * r / a) for i in range(4))
    cov = w[pre + "covalent_radii"]
    zi_u, zi_v = ref._pair_z(atomic_numbers, edge_index)
    env, denv = _cutoff_and_deriv(r, cov[zi_u] + cov[zi_v], w[pre + "p"])
    K = 0.5 * 14.3996 * zu * zv
    return K * (-phi * env / r ** 2 + dphi * env / r + phi * denv / r)


def geometry_vjp(w, atomic_numbers, pos, edge_index, shifts, g_sh, g_ef, scale):
    """dE/dpos from the per-edge feature gradients plus ZBL, closed form per edge (the spherical-harmonic VJP is
    Orb's, the same e3nn l <= 3 basis), then a scatter to both atoms of each edge."""
    from tt_atom.orb_geometry import _spherical_harmonics_l3_vjp
    vec, r = ref.edge_vectors(pos, edge_index, shifts)
    unit = vec / r
    g_unit = _spherical_harmonics_l3_vjp(unit, g_sh)
    g_r = (g_ef * radial_deriv(r, atomic_numbers, edge_index, w)).sum(-1, keepdim=True) \
        + scale * zbl_deriv(r, atomic_numbers, edge_index, w)
    g_vec = (g_unit - unit * (unit * g_unit).sum(-1, keepdim=True)) / r + g_r * unit
    g = torch.zeros_like(pos)
    g.index_add_(0, edge_index[1], g_vec)
    g.index_add_(0, edge_index[0], -g_vec)
    return g
