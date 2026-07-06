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


def _mm(ttnn, g, W, kcfg, memory_config=None):
    """grad wrt x of ``y = x @ W`` (W stored [in,out]): ``g @ W^T``. ``transpose_b`` folds the
    transpose into the matmul (bit-identical), dropping an explicit transpose op per call — the
    backward makes ~40 of these on constant weights, all in the captured trace.

    ``memory_config`` lets the BW-bound grid_bw chain keep its transpose-matmul outputs
    L1-resident instead of round-tripping DRAM (same residency win as the forward grid module)."""
    if memory_config is not None:
        return ttnn.matmul(g, W, transpose_b=True, compute_kernel_config=kcfg,
                           memory_config=memory_config)
    return ttnn.matmul(g, W, transpose_b=True, compute_kernel_config=kcfg)


# --------------------------------------------------------------------------- elementwise


def silu_bw(ttnn, g, x):
    return ttnn.silu_bw(g, x)[0]


# --------------------------------------------------------------------------- RMS norm SH


def rmsnorm_bw(norm, g_out):
    """VJP of ``RMSNormSH``. ``norm`` is the forward module (holds bdw, aw, ab, eps).
    Reuses the centered input ``xc`` and rsqrt scale ``inv`` cached on the forward."""
    ttnn = norm.ttnn
    xc = norm._cache_xc                                 # centered input saved on forward
    inv = norm._cache_inv                               # [N,1,1] rsqrt scale saved on forward
    N, nsph, C = xc.shape

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
    """VJP of ``GateActivation`` (flat). Returns (g_gating[E,lmax*H], g_x flat[E,nsph*H])."""
    ttnn = gate.ttnn
    gating = gate._cache_gating                         # pre-sigmoid [E, lmax*H]
    x = gate._cache_x                                   # pre-gate input flat [E, nsph*H]
    gate_exp = gate._cache_gate                         # expanded sigmoid gate [E,(nsph-1)*H] (cached fwd)
    E, H, lmax, ei = x.shape[0], gate.H, gate.lmax, gate.expand_index

    g_scalar = ttnn.slice(g_out, [0, 0], [E, H])
    g_vec = ttnn.slice(g_out, [0, H], [E, x.shape[1]])
    x_scalar = ttnn.slice(x, [0, 0], [E, H])
    x_vec = ttnn.slice(x, [0, H], [E, x.shape[1]])

    g_x = ttnn.concat([silu_bw(ttnn, g_scalar, x_scalar), ttnn.multiply(g_vec, gate_exp)], dim=1)

    g_gate_exp = ttnn.multiply(g_vec, x_vec)            # [E, (nsph-1)*H]
    # segment-sum the H-blocks back to [E, lmax*H] over expand_index
    rows = []
    for l in range(lmax):
        pos = [k for k, e in enumerate(ei) if e == l]
        acc = ttnn.slice(g_gate_exp, [0, pos[0] * H], [E, (pos[0] + 1) * H])
        for k in pos[1:]:
            acc = ttnn.add(acc, ttnn.slice(g_gate_exp, [0, k * H], [E, (k + 1) * H]))
        rows.append(acc)
    g_sig = ttnn.concat(rows, dim=1)                    # [E, lmax*H]
    g_gating = ttnn.sigmoid_bw(g_sig, gating)[0]
    return g_gating, g_x


# --------------------------------------------------------------------------- SO(2) conv


def so2_bw(conv, g_out, g_extra=None):
    """VJP of ``SO2Convolution``. Returns (g_x flat ``[E, nsph*Cin]``, g_rad or None).

    ``g_rad`` is the adjoint at the radial-MLP *output* (the per-m multiplier), to be finished
    on host. Matmul backward = transpose-matmul on device."""
    ttnn = conv.ttnn
    kcfg = conv.kcfg
    H = conv.H
    lmax, mmax = conv.lmax, conv.mmax
    nsph = (lmax + 1) ** 2
    E = g_out.shape[0]

    gf = g_out if len(g_out.shape) == 2 else ttnn.reshape(g_out, (E, nsph * H))
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
        g_real = seg[si]; g_imag = seg[si + 1]; si += 2   # adjoints of out_real, out_imag
        # fwd: out_real = r0 - i1, out_imag = i0 + r1, with [r0|r1]=real@W, [i0|i1]=imag@W.
        # => g_fr = [g_real, g_imag], g_fi = [g_imag, -g_real]; g_{real,imag} = g_f @ W^T.
        g_fr = ttnn.concat([g_real, g_imag], dim=1)        # [E,2Hh]
        g_fi = ttnn.concat([g_imag, ttnn.multiply(g_real, -1.0)], dim=1)
        g_in_real = _mm(ttnn, g_fr, conv.w_m[m - 1], kcfg)  # [E, nc*Cin]
        g_in_imag = _mm(ttnn, g_fi, conv.w_m[m - 1], kcfg)
        g_blocks.append(ttnn.concat([g_in_real, g_in_imag], dim=1))   # [E, 2*nc*Cin]

    g_xf = ttnn.concat(g_blocks, dim=1)                 # [E, nsph*Cin]

    g_rad = None
    if conv.has_radial:
        xin = conv._cache_xin                           # [E, nsph*Cin] pre-multiply
        mult = conv._cache_mult                         # [E, nsph*Cin]
        g_mult = ttnn.multiply(g_xf, xin)
        g_xf = ttnn.multiply(g_xf, mult)
        # collapse duplicated real/imag halves back to per-m radial channels
        o = 0
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

    return g_xf, g_rad                                  # g_xf flat [E, nsph*Cin]


# --------------------------------------------------------------------------- grid atomwise


def grid_bw(grid, g_out):
    ttnn = grid.ttnn
    kcfg = grid.kcfg
    N = g_out.shape[0]
    from .device import l1_if_fits, L1_NODE_BUDGET   # BW-bound [N,npts,C] chain -> L1 while it fits
    _npts_pad = ((grid.npts + 31) // 32) * 32         # tile-padded point dim (3D tensor)
    L1 = l1_if_fits(ttnn, N, _npts_pad * g_out.shape[2], budget=L1_NODE_BUDGET)
    a1, a2 = grid._cache_a1, grid._cache_a2             # pre-silu activations [N,npts,H]
    # from_grid backward: o = transpose(gt @ fg); gt = transpose(g_mlp_out)
    # forward: gt=transpose(mlp,1,2); o=gt@fg; out=transpose(o,1,2)
    g_o = ttnn.transpose(g_out, 1, 2, memory_config=L1)  # [N,C,nsph]
    g_gt = _mm(ttnn, g_o, grid.fg, kcfg, memory_config=L1)  # [N,C,npts]
    g_mlp = ttnn.transpose(g_gt, 1, 2, memory_config=L1)  # [N,npts,C]
    # mlp backward (no bias): a3 = s2@W4 ; s2=silu(a2); a2=s1@W2; s1=silu(a1); a1=g0@W0
    g_s2 = _mm(ttnn, g_mlp, grid.w4, kcfg, memory_config=L1)
    g_a2 = silu_bw(ttnn, g_s2, a2)
    g_s1 = _mm(ttnn, g_a2, grid.w2, kcfg, memory_config=L1)
    g_a1 = silu_bw(ttnn, g_s1, a1)
    g_g0 = _mm(ttnn, g_a1, grid.w0, kcfg, memory_config=L1)  # [N,npts,C]
    # to_grid backward: g0 = transpose(xt @ tg_T); xt=transpose(x,1,2)
    g_g0t = ttnn.transpose(g_g0, 1, 2, memory_config=L1)  # [N,C,npts]
    g_xt = _mm(ttnn, g_g0t, grid.tg_T, kcfg)            # [N,C,nsph] -> DRAM (feeds residual add)
    return ttnn.transpose(g_xt, 1, 2)                   # [N,nsph,C]


# --------------------------------------------------------------------------- spectral atomwise


def _so3_linear_bw(sp, g_out, w_blocks):
    """VJP of one ``SO3_Linear`` wrt its input. ``g_out`` [N,nsph,cout] -> g_x [N,nsph,cin].
    Per degree the GEMM is shared, so the input adjoint is ``g_out_block @ W_block^T`` (bias on
    l=0 is additive -> identity wrt the input)."""
    ttnn = sp.ttnn
    N = g_out.shape[0]
    outs, start = [], 0
    for l in range(sp.lmax + 1):
        n = 2 * l + 1
        gb = ttnn.slice(g_out, [0, start, 0], [N, start + n, g_out.shape[2]])
        outs.append(_mm(ttnn, gb, w_blocks[l], sp.kcfg))
        start += n
    return ttnn.concat(outs, dim=1)


def spectral_bw(sp, g_out):
    """VJP of ``SpectralAtomwise``. ``g_out`` [N,nsph,C] -> g wrt input x [N,nsph,C]."""
    ttnn = sp.ttnn
    N, H, C = g_out.shape[0], sp.H, sp.C
    a_scalar = sp._cache_a_scalar
    gating, h = sp._cache_gating, sp._cache_h

    # so3_linear_2 backward
    g_g = _so3_linear_bw(sp, g_out, sp.l2_w)                  # [N, nsph, H]

    # gate backward: l0 SiLU; l>=1 multiply by sigmoid(gating) per degree
    sg = ttnn.sigmoid(gating)                                 # [N, lmax*H]
    g_h_parts = [silu_bw(ttnn, ttnn.slice(g_g, [0, 0, 0], [N, 1, H]),
                         ttnn.slice(h, [0, 0, 0], [N, 1, H]))]
    g_sg_rows, start = [], 1
    for l in range(1, sp.lmax + 1):
        n = 2 * l + 1
        g_gb = ttnn.slice(g_g, [0, start, 0], [N, start + n, H])
        h_b = ttnn.slice(h, [0, start, 0], [N, start + n, H])
        gl = ttnn.reshape(ttnn.slice(sg, [0, (l - 1) * H], [N, l * H]), (N, 1, H))
        g_h_parts.append(ttnn.multiply(g_gb, gl))             # g wrt h block
        # g wrt the (broadcast) gate = sum over the n coeffs of g_gb * h_block
        g_sg_rows.append(ttnn.sum(ttnn.multiply(g_gb, h_b), dim=1))   # [N, H]
        start += n
    g_h = ttnn.concat(g_h_parts, dim=1)                       # [N, nsph, H]
    g_sg = ttnn.concat(g_sg_rows, dim=1)                      # [N, lmax*H]
    g_gating = ttnn.sigmoid_bw(g_sg, gating)[0]               # through sigmoid

    # so3_linear_1 backward -> g wrt x (l>=... all degrees)
    g_x = _so3_linear_bw(sp, g_h, sp.l1_w)                    # [N, nsph, C]

    # scalar_mlp backward: gating = SiLU(scalar @ W + b); add g_scalar onto x's l=0 channel
    g_a = silu_bw(ttnn, g_gating, a_scalar)                   # [N, lmax*H]
    g_scalar = _mm(ttnn, g_a, sp.smlp_w, sp.kcfg)             # [N, C]
    g_scalar = ttnn.reshape(g_scalar, (N, 1, C))
    g_x_l0 = ttnn.add(ttnn.slice(g_x, [0, 0, 0], [N, 1, C]), g_scalar)
    g_x_rest = ttnn.slice(g_x, [0, 1, 0], [N, sp.nsph, C])
    return ttnn.concat([g_x_l0, g_x_rest], dim=1)


# --------------------------------------------------------------------------- edgewise


def edgewise_bw(ew, graph, g_out, acc):
    """VJP of ``Edgewise`` (flat MAC rotations). Returns g wrt node features [N,nsph,C];
    accumulates the geometric coefficient adjoints (g rot_fwd / rot_inv, g_envelope) and the
    radial adjoint g_rad into ``acc``."""
    from . import rotation
    ttnn = ew.ttnn
    kcfg = ew.kcfg
    C = ew.C
    N, nsph = g_out.shape[0], g_out.shape[1]
    E = graph.E
    dev = ew.device

    # scatter backward: g_m_back[e] = g_out[tgt[e]]  (gather by target), flat [E, nsph*C]
    gof = ttnn.to_layout(ttnn.reshape(g_out, (N, nsph * C)), ttnn.ROW_MAJOR_LAYOUT)
    g_mback = ttnn.to_layout(ttnn.embedding(graph.tgt_idx, gof), ttnn.TILE_LAYOUT)
    # inverse rotation backward: forward mapped reduced m-space (nred) -> node SH (nsph)
    g_menv, g_rinv = rotation.rotate_bw(ttnn, ew._cache_menv, g_mback, graph.rot_inv_ij,
                                        graph.rot_inv_coef, graph.nred, C, dev, n_out=nsph)
    # envelope: m_env = m_so2 * envelope  (flat [E,9C] * [E,1])
    g_mso2 = ttnn.multiply(g_menv, graph.edge_envelope_f)
    g_env = ttnn.sum(ttnn.multiply(g_menv, ew._cache_mso2), dim=1, keepdim=True)   # [E,1]
    # so2_2 -> gate -> so2_1 (all flat)
    g_mgate, _ = so2_bw(ew.so2_2, g_mso2)
    g_gating, g_mso1 = gate_bw(ew.gate, g_mgate)
    g_mrot, g_rad = so2_bw(ew.so2_1, g_mso1, g_gating)         # g_mrot flat [E, 9*2C]
    # forward rotation backward: forward mapped node SH (nsph) -> reduced m-space (nred)
    g_mcat, g_rfwd = rotation.rotate_bw(ttnn, ew._cache_mcat, g_mrot, graph.rot_fwd_ij,
                                        graph.rot_fwd_coef, nsph, 2 * C, dev, n_out=graph.nred)
    # m_cat per coord = [xs_i | xt_i]; split channels back out. Do the 3D<->flat coeff-dim reshapes
    # + the channel split in ROW_MAJOR (contiguous, no 9->32 tile-pad repack) with a single TILE
    # round-trip -- the direct TILE reshapes here cost ~18 ms each at E~46k. Bit-exact.
    g_mcat_rm = ttnn.to_layout(g_mcat, ttnn.ROW_MAJOR_LAYOUT)
    g_mcat3 = ttnn.reshape(g_mcat_rm, (E, nsph, 2 * C))
    g_xs_f = ttnn.to_layout(ttnn.reshape(ttnn.slice(g_mcat3, [0, 0, 0], [E, nsph, C]), (E, nsph * C)),
                            ttnn.TILE_LAYOUT)
    g_xt_f = ttnn.to_layout(ttnn.reshape(ttnn.slice(g_mcat3, [0, 0, C], [E, nsph, 2 * C]), (E, nsph * C)),
                            ttnn.TILE_LAYOUT)
    # gather backward: g_nodes = scatter_src(g_xs) + scatter_tgt(g_xt). Dense one-hot matmuls
    # (small N) or the linear O(E) gather+reduce (large N) — mirrors the forward scatter.
    if graph.linear_scatter:
        from . import scatter
        W = nsph * C
        g_nodes = ttnn.add(scatter.segment_sum(ttnn, g_xs_f, graph.src_gather, graph.Dmax_s, N, W),
                           scatter.segment_sum(ttnn, g_xt_f, graph.tgt_gather, graph.Dmax_t, N, W))
    else:
        g_nodes = ttnn.add(ttnn.matmul(graph.scatter_src, g_xs_f, compute_kernel_config=kcfg),
                           ttnn.matmul(graph.scatter, g_xt_f, compute_kernel_config=kcfg))
    g_nodes = ttnn.reshape(g_nodes, (N, nsph, C))

    acc["rot_fwd"] = g_rfwd if acc["rot_fwd"] is None else ttnn.add(acc["rot_fwd"], g_rfwd)
    acc["rot_inv"] = g_rinv if acc["rot_inv"] is None else ttnn.add(acc["rot_inv"], g_rinv)
    acc["envelope"] = g_env if acc["envelope"] is None else ttnn.add(acc["envelope"], g_env)
    acc["g_rad"].append((ew.so2_1, g_rad))
    return g_nodes


# --------------------------------------------------------------------------- block / backbone


def block_bw(blk, graph, g_out, acc):
    """VJP of ``_Block``. ``blk`` is the forward block module."""
    ttnn = blk.norm_1.ttnn
    # x = atom_wise(n2) + x_res2
    g_n2 = (spectral_bw(blk.atom_wise, g_out) if getattr(blk, "ff_type", "grid") == "spectral"
            else grid_bw(blk.atom_wise, g_out))
    g_after_edge = ttnn.add(rmsnorm_bw(blk.norm_2, g_n2), g_out)
    # x = edge_wise(s) + x_res
    g_s = edgewise_bw(blk.edge_wise, graph, g_after_edge, acc)
    # s = n1 with l0 += sys_emb (identity wrt n1); add residual
    g_x_in = ttnn.add(rmsnorm_bw(blk.norm_1, g_s), g_after_edge)
    return g_x_in


def energy_bw(bb, node_emb):
    """VJP of the energy head; returns g wrt node_emb [N,nsph,C] (only l=0 nonzero).

    The two constants (the ``dE/dE = 1`` seed and the l>=1 zero padding) are created once and
    cached on ``bb`` so that no device buffer is allocated inside a captured trace region — the
    ttnn trace machinery forbids allocations during capture (it hangs)."""
    ttnn = bb.ttnn
    kcfg = bb.kcfg
    N, nsph, C = node_emb.shape
    if getattr(bb, "_bw_seed", None) is None or tuple(bb._bw_seed.shape) != (N, 1):
        bb._bw_seed = ttnn.ones((N, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bb.device)
        bb._bw_zeros = ttnn.zeros((N, nsph - 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=bb.device)
    h = ttnn.reshape(ttnn.slice(node_emb, [0, 0, 0], [N, 1, C]), (N, C))
    a1 = ttnn.linear(h, bb.eh_w[0], bias=bb.eh_b[0], compute_kernel_config=kcfg)
    s1 = ttnn.silu(a1)
    a2 = ttnn.linear(s1, bb.eh_w[1], bias=bb.eh_b[1], compute_kernel_config=kcfg)
    g_s2 = _mm(ttnn, bb._bw_seed, bb.eh_w[2], kcfg)
    g_a2 = silu_bw(ttnn, g_s2, a2)
    g_s1 = _mm(ttnn, g_a2, bb.eh_w[1], kcfg)
    g_a1 = silu_bw(ttnn, g_s1, a1)
    g_h = _mm(ttnn, g_a1, bb.eh_w[0], kcfg)             # [N,C]
    g_h = ttnn.reshape(g_h, (N, 1, C))
    return ttnn.concat([g_h, bb._bw_zeros], dim=1)


def backbone_bw(bb, graph, node_emb):
    """Full reverse pass of the backbone+energy head. Returns a dict of device adjoints
    (g_x_init, g_wigner, g_wigner_inv, g_envelope, g_x_edge). ``g_x_edge`` is finished on device
    (radial-MLP backward), so the host only reads back adjoints and drives the geometric autograd."""
    ttnn = bb.ttnn
    acc = {"rot_fwd": None, "rot_inv": None, "envelope": None, "g_rad": []}
    g = energy_bw(bb, node_emb)
    g = rmsnorm_bw(bb.final_norm, g)
    for blk in reversed(bb.blocks):
        g = block_bw(blk, graph, g, acc)
    acc["x_init"] = g
    # radial-MLP backward on device: each conv's radial adjoint (at the radial output) is finished
    # to g wrt the shared invariant edge embedding x_edge on device (hand-written LN/SiLU VJP),
    # summed across convs. This replaces the host torch.autograd radial finish (~100 ms at N=128)
    # with a captured device pass -- only a single [E, x_edge] readback remains on host.
    g_xe = None
    for conv, g_rad in acc["g_rad"]:
        gc = conv.rad.bw(g_rad)
        g_xe = gc if g_xe is None else ttnn.add(g_xe, gc)
    acc["x_edge"] = g_xe
    return acc


# --------------------------------------------------------------------------- full energy+force


def _forward(bb, geo, pos, atomic_numbers, edge_index, sys_node_embedding, edge_cell_shift,
             requires_grad, compute_stress=False):
    """Shared forward: host geometry -> device-resident GraphContext + backbone node embedding.
    Returns ``(node_emb, graph, t, pos_leaf, strain_leaf)``; ``pos_leaf`` tracks grad when
    ``requires_grad``. ``strain_leaf`` is a zero symmetric 3x3 leaf (else None) that is applied to
    the edge vectors as ``r' = r(I + sym(strain))`` — since ALL pos/cell dependence of the energy
    flows through the edge vectors, ``dE/dstrain`` is exactly fairchem's symmetrized virial (the
    combined position + cell contribution), so stress = dE/dstrain / volume."""
    import ttnn

    from .model import GraphContext

    device = bb.device
    N, C = atomic_numbers.shape[0], geo.C
    pos = pos.detach().clone().requires_grad_(requires_grad)
    strain, edge_vec = None, None
    if compute_stress:
        src, tgt = edge_index[0], edge_index[1]
        strain = torch.zeros(3, 3, dtype=pos.dtype, requires_grad=True)
        ev = pos[src] - pos[tgt]
        if edge_cell_shift is not None:
            ev = ev + edge_cell_shift
        sym = 0.5 * (strain + strain.transpose(0, 1))
        edge_vec = ev + ev @ sym                        # r' = r (I + sym(strain))
    t = geo(pos, atomic_numbers, edge_index, sys_node_embedding, edge_vec=edge_vec,
            edge_cell_shift=edge_cell_shift)

    # the analytic-force backward keeps bf16 geometric operands (bf8 wigner would mix dtypes in
    # the transpose-matmul adjoints); ``fast`` (bf8) is for the energy-throughput path.
    graph = GraphContext(device, edge_index=edge_index, wigner=t["wigner"].detach(),
                         wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
                         edge_envelope=t["edge_envelope"].detach(), num_nodes=N)
    se3 = ttnn.from_torch(sys_node_embedding.reshape(N, 1, C), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    x_init = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
    node_emb = bb.node_embedding(x_init, graph, se3)
    return node_emb, graph, t, pos, strain


def _forces(bb, geo, graph, node_emb, t, pos, strain=None):
    """Reverse pass: device VJP ``dE/d{geometric inputs}`` finished by host autograd to ``-dE/dpos``.

    The energy seed is ``dE/dh = 1`` per node — the gradient of the *summed* energy. For a
    disjoint-union batch that sum is ``sum_k E_k`` and block-diagonality makes each atom's
    gradient ``-dE_(its system)/dx``, so the batched forces are the concatenation (no change).

    When ``strain`` is given (a zero symmetric 3x3 leaf that scaled the edge vectors in the
    forward), the same host autograd also yields the virial ``dE/dstrain`` and this returns
    ``(forces, virial)``; the caller divides the virial by the volume for the stress tensor."""
    import ttnn

    from . import rotation

    acc = backbone_bw(bb, graph, node_emb)
    nsph, nred = graph.nsph, graph.nred
    g_xi = ttnn.to_torch(acc["x_init"]).float()
    # scatter packed rotation-coefficient adjoints back to dense for the host dW/dpos autograd.
    # wig_M is [E, nred, nsph] (fwd), wig_M_inv is [E, nsph, nred] (inv) — rectangular for uma-m.
    g_wig = rotation.scatter_coef(ttnn.to_torch(acc["rot_fwd"]).float(), graph.rot_fwd_ij, nred, nsph)
    g_winv = rotation.scatter_coef(ttnn.to_torch(acc["rot_inv"]).float(), graph.rot_inv_ij, nsph, nred)
    g_env = ttnn.to_torch(acc["envelope"]).float().reshape(-1, 1, 1)   # [E,1]->[E,1,1]
    # radial finish is done on device (see backbone_bw); read back the single g_x_edge adjoint.
    # x_edge = [gaussian(dist) | src_emb | tgt_emb]; only the gaussian block depends on pos, so the
    # force VJP needs only its adjoint. Cast just that block bf16->f32 (the cast dominates readback);
    # the pos-independent embedding columns contribute zero to dE/dpos.
    ng = geo.offset.shape[0]
    gx = ttnn.to_torch(acc["x_edge"])
    g_xe = torch.zeros(tuple(gx.shape), dtype=torch.float32)
    g_xe[:, :ng] = gx[:, :ng].float()

    outs = [t["x_init"], t["wigner"], t["wigner_inv"], t["x_edge"], t["edge_envelope"]]
    gouts = [g_xi, g_wig, g_winv, g_xe, g_env]
    inputs = [pos] if strain is None else [pos, strain]
    grads = torch.autograd.grad(outs, inputs, grad_outputs=gouts)
    if strain is None:
        return -grads[0]
    return -grads[0], grads[1]                          # (forces, virial = dE/dstrain)


def energy_and_forces(bb, geo, pos, atomic_numbers, edge_index, sys_node_embedding,
                      edge_cell_shift=None, compute_stress=False):
    """Conservative energy + analytic forces ``F = -dE/dpos`` for one system.

    Device-resident forward + reverse VJP gives ``dE/d{geometric inputs}``; ``torch.autograd``
    through the host geometry supplies the cheap ``d(geometric)/dpos`` to finish the force.
    ``edge_cell_shift`` [E, 3] carries the periodic image offsets (None for aperiodic systems).
    Returns ``(energy: float, forces: torch.Tensor[N,3])``, or when ``compute_stress`` is set
    ``(energy, forces, virial[3,3])`` where ``virial = dE/dstrain`` (the caller divides by the
    cell volume for the stress tensor).
    """
    import ttnn

    node_emb, graph, t, pos, strain = _forward(
        bb, geo, pos, atomic_numbers, edge_index, sys_node_embedding, edge_cell_shift,
        requires_grad=True, compute_stress=compute_stress)
    E = float(ttnn.to_torch(bb.energy(node_emb)).reshape(-1)[0])
    if compute_stress:
        F, virial = _forces(bb, geo, graph, node_emb, t, pos, strain=strain)
        return E, F, virial
    F = _forces(bb, geo, graph, node_emb, t, pos)
    return E, F


def energy_and_forces_batch(bb, geo, bg, *, compute_forces=True):
    """Disjoint-union batched energy (+ optional forces) for a ``disjoint.BatchedGraph`` ``bg``.

    One device forward over the concatenated block-diagonal graph; per-system energies come from
    the segment-sum readout (``Backbone.energy_batch``) and forces — when requested — from the
    single shared reverse pass (block-diagonal => per-system correct). Returns
    ``(E_raw: torch[K], F: torch[Ntot, 3] or None)`` where ``E_raw`` is unnormalized (the caller
    applies the per-system energy normalizer)."""
    import ttnn

    node_emb, graph, t, pos, _ = _forward(bb, geo, bg.pos, bg.Z, bg.edge_index, bg.sys_emb,
                                          bg.cell_shift, requires_grad=compute_forces)
    seg = ttnn.from_torch(bg.segment_matrix(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=bb.device)
    E = ttnn.to_torch(bb.energy_batch(node_emb, seg)).float().reshape(-1)[:bg.K]
    F = _forces(bb, geo, graph, node_emb, t, pos) if compute_forces else None
    return E, F
