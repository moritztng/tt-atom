"""Analytic forces — reverse-mode VJP through the device backbone (``F = -dE/dpos``).

This is the production force path (not finite differences). The heavy ``dE/dfeature`` terms are
exactly the transposes of the forward GEMMs and run on device; the cheap geometric Jacobian
``d(geometric terms)/dpos`` is finished on host with torch autograd (it is <1% of the compute).

The device VJP produces adjoints at the pos-dependent device inputs:
  * ``g_x_init``       [N, nsph, C]
  * ``g_wigner``       [E, nsph, nsph]   (from the edge-frame rotation)
  * ``g_wigner_inv``   [E, nsph, nsph]
  * ``g_envelope``     [E, 1, 1]
  * ``g_rad`` per radial conv (adjoint at the radial-MLP *output*) -> host autograd finishes
    ``g_x_edge`` through the radial MLP, whose LayerNorms we deliberately keep off-device.

Each VJP mirrors a forward module in ``tt_atom/`` and is unit-tested against ``tests/mirror.py``
(a bit-exact torch transcription of the device forward) in ``tests/test_forces.py``.
"""
from __future__ import annotations

import torch

from .device import compute_kernel_config


def _mm(ttnn, g, W, kcfg):
    """grad wrt x of ``y = x @ W`` (W stored [in,out]): ``g @ W^T``."""
    return ttnn.matmul(g, ttnn.transpose(W, -2, -1), compute_kernel_config=kcfg)


# --------------------------------------------------------------------------- elementwise


def silu_bw(ttnn, g, x):
    return ttnn.silu_bw(g, x)[0]


# --------------------------------------------------------------------------- RMS norm SH


def rmsnorm_bw(norm, g_out):
    """VJP of ``RMSNormSH``. ``norm`` is the forward module (holds bdw, aw, ab, eps)."""
    ttnn = norm.ttnn
    x = norm._cache_x                                   # input saved on forward
    N, nsph, C = x.shape
    # recompute forward internals
    l0 = ttnn.slice(x, [0, 0, 0], [N, 1, C])
    l0c = ttnn.subtract(l0, ttnn.mean(l0, dim=2, keepdim=True))
    rest = ttnn.slice(x, [0, 1, 0], [N, nsph, C])
    xc = ttnn.concat([l0c, rest], dim=1)
    fn2 = ttnn.sum(ttnn.multiply(ttnn.multiply(xc, xc), norm.bdw), dim=1, keepdim=True)
    ms = ttnn.mean(fn2, dim=2, keepdim=True)            # [N,1,1]
    inv = ttnn.rsqrt(ttnn.add(ms, norm.eps))            # [N,1,1]

    # drop the affine bias on l0 (additive -> identity for grad)
    s = ttnn.multiply(inv, norm.aw)                     # [N,nsph,C] via broadcast
    g_xc = ttnn.multiply(g_out, s)                      # direct path
    # g_inv = sum_{coeff,C}(g_out * xc * aw)
    g_inv = ttnn.sum(ttnn.multiply(ttnn.multiply(g_out, xc), norm.aw), dim=1, keepdim=True)
    g_inv = ttnn.sum(g_inv, dim=2, keepdim=True)        # [N,1,1]
    # inv = (ms+eps)^-1/2 -> d inv/d ms = -1/2 inv^3
    g_ms = ttnn.multiply(g_inv, ttnn.multiply(ttnn.multiply(inv, inv), inv))
    g_ms = ttnn.multiply(g_ms, -0.5)
    # ms = mean_C(sum_coeff(xc^2 bdw)) -> d ms/d xc = (1/C) 2 xc bdw
    g_xc = ttnn.add(g_xc, ttnn.multiply(ttnn.multiply(ttnn.multiply(xc, norm.bdw), g_ms), 2.0 / C))
    # centering backward on l0: g_x_l0 = g_xc_l0 - mean_C(g_xc_l0)
    g_l0 = ttnn.slice(g_xc, [0, 0, 0], [N, 1, C])
    g_l0 = ttnn.subtract(g_l0, ttnn.mean(g_l0, dim=2, keepdim=True))
    g_rest = ttnn.slice(g_xc, [0, 1, 0], [N, nsph, C])
    return ttnn.concat([g_l0, g_rest], dim=1)


# --------------------------------------------------------------------------- gate


def gate_bw(gate, g_out):
    """VJP of ``GateActivation``. Returns (g_gating[N,lmax*H], g_x[N,nsph,H])."""
    ttnn = gate.ttnn
    gating = gate._cache_gating                         # pre-sigmoid [N, lmax*H]
    x = gate._cache_x                                   # pre-gate input [N,nsph,H]
    N, nsph, H = x.shape
    lmax = gate.lmax
    ei = gate.expand_index

    sig = ttnn.sigmoid(gating)
    sig_v = ttnn.reshape(sig, (N, lmax, H))
    gate_rows = [ttnn.slice(sig_v, [0, l, 0], [N, l + 1, H]) for l in range(lmax)]
    gate_exp = ttnn.concat([gate_rows[i] for i in ei], dim=1)   # [N, nsph-1, H]

    g_scalar = ttnn.slice(g_out, [0, 0, 0], [N, 1, H])
    g_vec = ttnn.slice(g_out, [0, 1, 0], [N, nsph, H])
    x_scalar = ttnn.slice(x, [0, 0, 0], [N, 1, H])
    x_vec = ttnn.slice(x, [0, 1, 0], [N, nsph, H])

    g_x_scalar = silu_bw(ttnn, g_scalar, x_scalar)
    g_x_vec = ttnn.multiply(g_vec, gate_exp)
    g_x = ttnn.concat([g_x_scalar, g_x_vec], dim=1)

    g_gate_exp = ttnn.multiply(g_vec, x_vec)            # [N,nsph-1,H]
    # segment-sum back to [N,lmax,H] over expand_index
    rows = []
    for l in range(lmax):
        pos = [k for k, e in enumerate(ei) if e == l]
        acc = ttnn.slice(g_gate_exp, [0, pos[0], 0], [N, pos[0] + 1, H])
        for k in pos[1:]:
            acc = ttnn.add(acc, ttnn.slice(g_gate_exp, [0, k, 0], [N, k + 1, H]))
        rows.append(acc)
    g_sig = ttnn.concat(rows, dim=1)                    # [N,lmax,H]
    g_sig = ttnn.reshape(g_sig, (N, lmax * H))
    g_gating = ttnn.sigmoid_bw(g_sig, gating)[0]
    return g_gating, g_x


# --------------------------------------------------------------------------- SO(2) conv


def so2_bw(conv, g_out, g_extra=None):
    """VJP of ``SO2Convolution``. Returns (g_x[E,nsph,Cin], g_rad or None).

    ``g_rad`` is the adjoint at the radial-MLP *output* (the per-m multiplier), to be finished
    on host. Matmul backward = transpose-matmul on device."""
    ttnn = conv.ttnn
    kcfg = conv.kcfg
    Cin, H = conv.Cin, conv.H
    lmax, mmax = conv.lmax, conv.mmax
    nsph = (lmax + 1) ** 2
    E = g_out.shape[0]

    gf = ttnn.reshape(g_out, (E, nsph * H))
    # split g into out-blocks: m0 (lmax+1 coeffs), then per m>0 (real Hh, imag Hh)
    coeff_w = []                                        # column width per out-block (in H units)
    coeff_w.append((lmax + 1) * H)
    for m in range(1, mmax + 1):
        Hh = conv.w_m[m - 1].shape[1] // 2
        coeff_w.append(Hh)                              # real
        coeff_w.append(Hh)                              # imag
    seg, off = [], 0
    for wd in coeff_w:
        seg.append(ttnn.slice(gf, [0, off], [E, off + wd])); off += wd

    g_blocks = []                                       # adjoint per input m-block (flattened)
    # m = 0
    g_lin = seg[0]
    if conv.extra:
        g_lin = ttnn.concat([g_extra, g_lin], dim=1)    # [E, extra + H*(lmax+1)]
    g_blocks.append(_mm(ttnn, g_lin, conv.w_m0, kcfg))  # [E, w0]
    # m > 0
    si = 1
    for m in range(1, mmax + 1):
        nc = conv.num_coef[m]
        Hh = conv.w_m[m - 1].shape[1] // 2
        g_real = seg[si]; g_imag = seg[si + 1]; si += 2
        # real=a-d, imag=c+b  ->  g_a=g_real, g_d=-g_real, g_c=g_imag, g_b=g_imag
        g_a = g_real
        g_b = g_imag
        g_c = g_imag
        g_d = ttnn.multiply(g_real, -1.0)
        g_blk4 = ttnn.concat([g_a, g_b, g_c, g_d], dim=1)   # [E,4Hh]
        g_blk = ttnn.reshape(g_blk4, (E, 2, 2 * Hh))
        g_in = _mm(ttnn, g_blk, conv.w_m[m - 1], kcfg)      # [E,2,nc*Cin]
        g_blocks.append(ttnn.reshape(g_in, (E, 2 * nc * Cin)))

    g_xf = ttnn.concat(g_blocks, dim=1)                 # [E, nsph*Cin]

    g_rad = None
    if conv.has_radial:
        xin = conv._cache_xin                           # [E, nsph*Cin] pre-multiply
        mult = conv._cache_mult                         # [E, nsph*Cin]
        g_mult = ttnn.multiply(g_xf, xin)
        g_xf = ttnn.multiply(g_xf, mult)
        # collapse duplicated real/imag halves back to per-m radial channels
        parts, o = [], 0
        widths = [conv.rad_sizes[0]]
        for m in range(1, mmax + 1):
            widths += [conv.rad_sizes[m], conv.rad_sizes[m]]
        segs = []
        for wd in widths:
            segs.append(ttnn.slice(g_mult, [0, o], [E, o + wd])); o += wd
        g_rad_parts = [segs[0]]
        i = 1
        for m in range(1, mmax + 1):
            g_rad_parts.append(ttnn.add(segs[i], segs[i + 1])); i += 2
        g_rad = ttnn.concat(g_rad_parts, dim=1)         # [E, sum rad_sizes]

    g_x = ttnn.reshape(g_xf, (E, nsph, Cin))
    return g_x, g_rad


# --------------------------------------------------------------------------- grid atomwise


def grid_bw(grid, g_out):
    ttnn = grid.ttnn
    kcfg = grid.kcfg
    N = g_out.shape[0]
    a1, a2 = grid._cache_a1, grid._cache_a2             # pre-silu activations [N,npts,H]
    # from_grid backward: o = transpose(gt @ fg); gt = transpose(g_mlp_out)
    # forward: gt=transpose(mlp,1,2); o=gt@fg; out=transpose(o,1,2)
    g_o = ttnn.transpose(g_out, 1, 2)                   # [N,C,nsph]
    g_gt = _mm(ttnn, g_o, grid.fg, kcfg)                # [N,C,npts]
    g_mlp = ttnn.transpose(g_gt, 1, 2)                  # [N,npts,C]
    # mlp backward (no bias): a3 = s2@W4 ; s2=silu(a2); a2=s1@W2; s1=silu(a1); a1=g0@W0
    g_s2 = _mm(ttnn, g_mlp, grid.w4, kcfg)
    g_a2 = silu_bw(ttnn, g_s2, a2)
    g_s1 = _mm(ttnn, g_a2, grid.w2, kcfg)
    g_a1 = silu_bw(ttnn, g_s1, a1)
    g_g0 = _mm(ttnn, g_a1, grid.w0, kcfg)               # [N,npts,C]
    # to_grid backward: g0 = transpose(xt @ tg_T); xt=transpose(x,1,2)
    g_g0t = ttnn.transpose(g_g0, 1, 2)                  # [N,C,npts]
    g_xt = _mm(ttnn, g_g0t, grid.tg_T, kcfg)            # [N,C,nsph]
    return ttnn.transpose(g_xt, 1, 2)                   # [N,nsph,C]


# --------------------------------------------------------------------------- bmm / scatter


def bmm_bw(ttnn, W, x, g_out, kcfg):
    """y = bmm(W, x). Returns (g_W, g_x)."""
    g_W = ttnn.matmul(g_out, ttnn.transpose(x, 1, 2), compute_kernel_config=kcfg)
    g_x = ttnn.matmul(ttnn.transpose(W, 1, 2), g_out, compute_kernel_config=kcfg)
    return g_W, g_x


# --------------------------------------------------------------------------- edgewise


def edgewise_bw(ew, graph, g_out, acc):
    """VJP of ``Edgewise``. Returns g wrt node features [N,nsph,C]; accumulates the geometric
    adjoints (g_wigner, g_wigner_inv, g_envelope) and the radial adjoint g_rad into ``acc``."""
    ttnn = ew.ttnn
    kcfg = ew.kcfg
    C = ew.C
    N, nsph = g_out.shape[0], g_out.shape[1]
    E = graph.E

    # scatter backward: g_m[e] = g_out[tgt[e]]  (gather by target)
    gof = ttnn.to_layout(ttnn.reshape(g_out, (N, nsph * C)), ttnn.ROW_MAJOR_LAYOUT)
    g_m = ttnn.reshape(ttnn.embedding(graph.tgt_idx, gof), (E, nsph, C))
    g_m = ttnn.to_layout(g_m, ttnn.TILE_LAYOUT)

    # bmm(wigner_inv): m = bmm(winv, m_env)
    g_winv, g_menv = bmm_bw(ttnn, graph.wigner_inv, ew._cache_menv, g_m, kcfg)
    # envelope: m_env = m_so2 * envelope
    g_mso2 = ttnn.multiply(g_menv, graph.edge_envelope)
    g_env = ttnn.sum(ttnn.sum(ttnn.multiply(g_menv, ew._cache_mso2), dim=1, keepdim=True),
                     dim=2, keepdim=True)               # [E,1,1]
    # so2_2 -> gate -> so2_1
    g_mgate, _ = so2_bw(ew.so2_2, g_mso2)
    g_gating, g_mso1 = gate_bw(ew.gate, g_mgate)
    g_mrot, g_rad = so2_bw(ew.so2_1, g_mso1, g_gating)
    # bmm(wigner): m_rot = bmm(wigner, m_cat)
    g_wigner, g_mcat = bmm_bw(ttnn, graph.wigner, ew._cache_mcat, g_mrot, kcfg)
    # concat([xs,xt],dim2) backward
    g_xs = ttnn.slice(g_mcat, [0, 0, 0], [E, nsph, C])
    g_xt = ttnn.slice(g_mcat, [0, 0, C], [E, nsph, 2 * C])
    # gather backward (embedding) as one-hot matmuls: g_nodes = S_src @ g_xs + S_tgt @ g_xt
    g_xs_f = ttnn.reshape(g_xs, (E, nsph * C))
    g_xt_f = ttnn.reshape(g_xt, (E, nsph * C))
    g_nodes = ttnn.add(ttnn.matmul(graph.scatter_src, g_xs_f, compute_kernel_config=kcfg),
                       ttnn.matmul(graph.scatter, g_xt_f, compute_kernel_config=kcfg))
    g_nodes = ttnn.reshape(g_nodes, (N, nsph, C))

    acc["wigner"] = g_wigner if acc["wigner"] is None else ttnn.add(acc["wigner"], g_wigner)
    acc["wigner_inv"] = g_winv if acc["wigner_inv"] is None else ttnn.add(acc["wigner_inv"], g_winv)
    acc["envelope"] = g_env if acc["envelope"] is None else ttnn.add(acc["envelope"], g_env)
    acc["g_rad"].append((ew.so2_1, g_rad))
    return g_nodes


# --------------------------------------------------------------------------- block / backbone


def block_bw(blk, graph, g_out, acc):
    """VJP of ``_Block``. ``blk`` is the forward block module."""
    ttnn = blk.norm_1.ttnn
    # x = atom_wise(n2) + x_res2
    g_n2 = grid_bw(blk.atom_wise, g_out)
    g_after_edge = ttnn.add(rmsnorm_bw(blk.norm_2, g_n2), g_out)
    # x = edge_wise(s) + x_res
    g_s = edgewise_bw(blk.edge_wise, graph, g_after_edge, acc)
    # s = n1 with l0 += sys_emb (identity wrt n1); add residual
    g_x_in = ttnn.add(rmsnorm_bw(blk.norm_1, g_s), g_after_edge)
    return g_x_in


def energy_bw(bb, node_emb):
    """VJP of the energy head; returns g wrt node_emb [N,nsph,C] (only l=0 nonzero)."""
    ttnn = bb.ttnn
    kcfg = bb.kcfg
    N, nsph, C = node_emb.shape
    h = ttnn.reshape(ttnn.slice(node_emb, [0, 0, 0], [N, 1, C]), (N, C))
    a1 = ttnn.linear(h, bb.eh_w[0], bias=bb.eh_b[0], compute_kernel_config=kcfg)
    s1 = ttnn.silu(a1)
    a2 = ttnn.linear(s1, bb.eh_w[1], bias=bb.eh_b[1], compute_kernel_config=kcfg)
    g_a3 = ttnn.ones((N, 1), dtype=a1.dtype, layout=ttnn.TILE_LAYOUT, device=bb.device)
    g_s2 = _mm(ttnn, g_a3, bb.eh_w[2], kcfg)
    g_a2 = silu_bw(ttnn, g_s2, a2)
    g_s1 = _mm(ttnn, g_a2, bb.eh_w[1], kcfg)
    g_a1 = silu_bw(ttnn, g_s1, a1)
    g_h = _mm(ttnn, g_a1, bb.eh_w[0], kcfg)             # [N,C]
    g_h = ttnn.reshape(g_h, (N, 1, C))
    zeros = ttnn.zeros((N, nsph - 1, C), dtype=g_h.dtype, layout=ttnn.TILE_LAYOUT, device=bb.device)
    return ttnn.concat([g_h, zeros], dim=1)


def backbone_bw(bb, graph, node_emb):
    """Full reverse pass of the backbone+energy head. Returns a dict of device adjoints
    (g_x_init, g_wigner, g_wigner_inv, g_envelope) plus per-conv radial adjoints g_rad
    (list of (conv, g_rad)) for the host radial finish that yields g_x_edge."""
    ttnn = bb.ttnn
    acc = {"wigner": None, "wigner_inv": None, "envelope": None, "g_rad": []}
    g = energy_bw(bb, node_emb)
    g = rmsnorm_bw(bb.final_norm, g)
    for blk in reversed(bb.blocks):
        g = block_bw(blk, graph, g, acc)
    acc["x_init"] = g
    return acc


# --------------------------------------------------------------------------- full energy+force


def energy_and_forces(bb, geo, pos, atomic_numbers, edge_index, sys_node_embedding):
    """Conservative energy + analytic forces ``F = -dE/dpos`` for one system.

    Device-resident forward + reverse VJP gives ``dE/d{geometric inputs}``; ``torch.autograd``
    through the host geometry supplies the cheap ``d(geometric)/dpos`` to finish the force.
    Returns ``(energy: float, forces: torch.Tensor[N,3])``.
    """
    import ttnn

    from .model import GraphContext

    device = bb.device
    N, C = atomic_numbers.shape[0], geo.C
    pos = pos.detach().clone().requires_grad_(True)
    t = geo(pos, atomic_numbers, edge_index, sys_node_embedding)

    # the analytic-force backward keeps bf16 geometric operands (bf8 wigner would mix dtypes in
    # the transpose-matmul adjoints); ``fast`` (bf8) is for the energy-throughput path.
    graph = GraphContext(device, edge_index=edge_index, wigner=t["wigner"].detach(),
                         wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
                         edge_envelope=t["edge_envelope"].detach(), num_nodes=N)
    se3 = ttnn.from_torch(sys_node_embedding.reshape(N, 1, C), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    x_init = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
    node_emb, energy = bb(x_init, graph, se3)
    E = float(ttnn.to_torch(energy).reshape(-1)[0])

    acc = backbone_bw(bb, graph, node_emb)
    g_xi = ttnn.to_torch(acc["x_init"]).float()
    g_wig = ttnn.to_torch(acc["wigner"]).float()
    g_winv = ttnn.to_torch(acc["wigner_inv"]).float()
    g_env = ttnn.to_torch(acc["envelope"]).float()
    # host radial finish: radial-output adjoints -> g_x_edge
    from .geometry import radial_mlp
    xe = t["x_edge"].detach().clone().requires_grad_(True)
    for conv, grad in acc["g_rad"]:
        radial_mlp(xe, geo.w, conv.rad_prefix).backward(ttnn.to_torch(grad).float())
    g_xe = xe.grad

    g_pos = torch.autograd.grad(
        [t["x_init"], t["wigner"], t["wigner_inv"], t["x_edge"], t["edge_envelope"]],
        pos, grad_outputs=[g_xi, g_wig, g_winv, g_xe, g_env])[0]
    return E, -g_pos
