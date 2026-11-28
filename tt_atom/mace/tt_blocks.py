"""The two MACE hot blocks on the card, built from dense.py's device forms with stock ttnn ops.

Host-side setup (``*_setup``) turns the folded float64 constants into device tensors once; the per-step
functions take and return device tensors. Graded against the float64 reference in tests/test_mace_device.py.

Forces: the card has no autograd, so each block has a hand-written reverse (``*_bwd``, the device form of
backward.py). A forward call given a ``tape`` dict records the activations its reverse needs; the reverse runs
the transposed GEMMs (``plan["t"]``), gathers where the forward scattered and scatters where it gathered.
"""
from __future__ import annotations

import torch

from tt_atom import device as D
from . import dense


def const(t, dev, dtype):
    """A constant matmul operand on the card. At float32 it is split on the host into its bfloat16 hi + lo
    parts (see ``mm``), once, instead of on the card at every call."""
    import ttnn

    t = t.contiguous()
    if dtype != ttnn.float32:
        return D.to_dev(t, dev, dtype)
    hi = t.to(torch.bfloat16)
    lo = (t - hi.to(t.dtype)).to(torch.bfloat16)
    return tuple(D.to_dev(part.float(), dev, ttnn.bfloat16) for part in (hi, lo))


def mm(a, b, cfg):
    """matmul that keeps float32 operands at float32 accuracy.

    The matrix engine reads float32 operands through 19-bit source registers, i.e. truncated to a 10-bit
    mantissa, and truncation is biased: MACE's per-atom energies came out ~9 meV/atom high from that alone.
    So a float32 operand is split into bfloat16 hi + lo parts and the significant products are summed (a
    bfloat16 x bfloat16 product is exact in the float32 accumulator), leaving ~2^-16 relative. A constant from
    ``const`` arrives already split. A bfloat16 operand is taken as exact (0/1 selectors are uploaded that way)
    and is not split: two matmuls instead of three. Two bfloat16 operands take the plain matmul.
    """
    import ttnn

    def parts(x):
        if isinstance(x, tuple) or x.dtype != ttnn.float32:
            return x if isinstance(x, tuple) else (x,)
        hi = ttnn.typecast(x, ttnn.bfloat16)
        return hi, ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)

    a, b = parts(a), parts(b)
    if len(a) == len(b) == 1:
        return ttnn.matmul(a[0], b[0], compute_kernel_config=cfg)
    out = None
    for i, x in enumerate(a):
        for y in b[:2 - i]:  # lo x lo is below float32
            term = ttnn.matmul(x, y, compute_kernel_config=cfg, dtype=ttnn.float32)
            out = term if out is None else ttnn.add(out, term)
    return out


def perm(x, dims):
    """permute, float32-exact (see scatter.linear32)."""
    import ttnn

    from tt_atom import scatter
    return scatter.linear32(ttnn, lambda t: ttnn.permute(t, dims), x)


def rows(idx, x, shape):
    """Rows ``idx`` of a tile tensor, reshaped to ``shape`` (tile)."""
    import ttnn

    from tt_atom import scatter
    got = scatter.embed_rows(ttnn, idx, ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT))
    return ttnn.to_layout(ttnn.reshape(got, shape), ttnn.TILE_LAYOUT)


def silu_grad(pre, act_scale):
    """d(act_scale * silu(pre))/d pre."""
    import ttnn

    s = ttnn.sigmoid(pre)
    return ttnn.multiply(ttnn.multiply(s, ttnn.add(ttnn.multiply(pre, ttnn.add(ttnn.neg(s), 1.0)), 1.0)), act_scale)


def symmetric_contraction_setup(dev, tables, dtype):
    """``tables`` = dense.product_tables(..., elements=present) for one layer: [(T, parent, last)] per output
    irrep, T [n_present, c, F, 2L+1]. Atoms are sorted by species (graph_setup), so each species is one
    contiguous row range and costs one channel-batched matmul: all output irreps of the layer share the
    monomials, so their tables go side by side on the last axis.
    """
    _, parent, last = tables[0]
    ell = int(last.max()) + 1
    sels = dense.monomial_selectors(parent, last, ell)
    T = torch.cat([t for t, _, _ in tables], dim=-1)  # [E, c, F, sum(2L+1)]
    import ttnn
    sel = lambda S: D.to_dev(S.contiguous(), dev, ttnn.bfloat16)  # noqa: E731  (0/1: exact, see mm)
    return {
        "sels": [(None if Sp is None else sel(Sp), sel(Sl)) for Sp, Sl in sels],
        "T": [const(T[e], dev, dtype) for e in range(T.shape[0])],
        "t": {"sels": [(None if Sp is None else sel(Sp.T), sel(Sl.T)) for Sp, Sl in sels],
              "T": [const(T[e].transpose(1, 2), dev, dtype) for e in range(T.shape[0])]},
        "widths": [t.shape[-1] for t, _, _ in tables],
        "cfg": D.compute_kernel_config(),
    }


def symmetric_contraction(x, plan, ranges, n, c, tape=None):
    """x: device [n * c, ell] (rows = (node, channel)), ``ranges`` = graph_setup's species row ranges
    -> device [c, n, sum(2L+1)]."""
    import ttnn

    cfg, feats, prev, steps = plan["cfg"], [], None, []
    for Sp, Sl in plan["sels"]:
        xl = mm(x, Sl, cfg)
        up = None if Sp is None else mm(prev, Sp, cfg)
        prev = xl if Sp is None else ttnn.multiply(up, xl)
        feats.append(prev)
        steps.append((up, xl))
    if tape is not None:
        tape["steps"] = steps
    P = ttnn.concat(feats, dim=1)  # [n * c, F]
    P = perm(ttnn.reshape(P, (n, c, P.shape[-1])), (1, 0, 2))  # [c, n, F]
    outs = []
    for (lo, hi), T in zip(ranges, plan["T"]):
        if hi > lo:
            outs.append(mm(P[:, lo:hi, :], T, cfg))  # [c, n_e, D]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)


def symmetric_contraction_bwd(g, plan, ranges, n, c, tape):
    """g [c, n, sum(2L+1)] -> dE/dx [n * c, ell]. Degree nu = (degree nu-1 @ Sp) * (x @ Sl), walked from the top
    degree down: each degree's gradient is its table column block plus what the degree above passed down."""
    import ttnn

    cfg, t = plan["cfg"], plan["t"]
    outs = [mm(g[:, lo:hi, :], T, cfg) for (lo, hi), T in zip(ranges, t["T"]) if hi > lo]  # [c, n_e, F]
    g_P = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)
    hi = g_P.shape[-1]
    g_P = ttnn.reshape(perm(g_P, (1, 0, 2)), (n * c, hi))
    g_x = down = None
    for (Sp, Sl), (up, xl) in zip(reversed(t["sels"]), reversed(tape["steps"])):
        lo = hi - xl.shape[-1]
        g_f = g_P[:, lo:hi] if down is None else ttnn.add(g_P[:, lo:hi], down)
        hi = lo
        if Sp is not None:
            down = mm(ttnn.multiply(g_f, xl), Sp, cfg)
            g_f = ttnn.multiply(g_f, up)
        part = mm(g_f, Sl, cfg)
        g_x = part if g_x is None else ttnn.add(g_x, part)
    return g_x


def symmetric_contraction_to_ref(out, widths):
    """[c, n, sum(2L+1)] torch -> reference layout [n, sum c*(2L+1)]."""
    out, blocks, d = out.permute(1, 0, 2), [], 0
    for wd in widths:
        blocks.append(out[:, :, d:d + wd].reshape(out.shape[0], -1))
        d += wd
    return torch.cat(blocks, dim=1)


def fc_net(x, Ws, act_scale, pre=None):
    """e3nn FullyConnectedNet on device; ``Ws`` already divided by sqrt(fan-in). ``pre`` (a list) collects the
    pre-activations for fc_net_bwd."""
    import ttnn

    cfg = D.compute_kernel_config()
    for i, W in enumerate(Ws):
        x = mm(x, W, cfg)
        if i < len(Ws) - 1:
            if pre is not None:
                pre.append(x)
            x = ttnn.multiply(ttnn.silu(x), act_scale)
    return x


def fc_net_bwd(g, Wts, pre, act_scale):
    """``Wts`` = the forward's weights, transposed."""
    import ttnn

    cfg = D.compute_kernel_config()
    for i in range(len(Wts) - 1, -1, -1):
        g = mm(g, Wts[i], cfg)
        if i > 0:
            g = ttnn.multiply(g, silu_grad(pre[i - 1], act_scale))
    return g


def species_ranges(element, n_present):
    """Row range of each present species in a species-sorted atom order."""
    return [(int((element < e).sum()), int((element <= e).sum())) for e in range(n_present)]


def graph_setup(dev, element, n_present, edge_index):
    """Per-structure tables, shared by every layer: species row ranges and the edge layout. Built once per
    neighbour list; the weights (interaction_setup, product_setup) never see the graph, so an MD rebuild or a new
    structure in a screen re-uploads only these. ``element`` [n] indexes the present species and must be sorted
    (atoms ordered by species at setup).

    Edges live on the card in the padded receiver layout: slot (i, d) of [n, D, ...] is the d-th edge into atom
    i, D the largest degree. A pad slot carries zero sh and radial features (``pad_edges``) and reads sender 0, so
    it adds exactly zero forward and reverse, and summing a node's edges is a matmul over the D axis."""
    import ttnn

    from tt_atom import scatter

    assert bool((element[1:] >= element[:-1]).all()), "sort atoms by species at setup"
    n, E = len(element), edge_index.shape[1]
    slot_edge, dmax = scatter.build_gather(edge_index[1], n, E)
    slot_edge = torch.from_numpy(slot_edge).long()  # [n * D] edge of each slot, E = pad
    real = slot_edge < E
    slot_of_edge = torch.empty(E, dtype=torch.long)
    slot_of_edge[slot_edge[real]] = torch.arange(n * dmax)[real]
    slot_sender = torch.zeros(n * dmax, dtype=torch.int32)
    slot_sender[real] = edge_index[0][slot_edge[real]].to(torch.int32)
    # the reverse pass scatters slots back to their senders
    send, dmax_send = scatter.build_gather(edge_index[0], n, E, sentinel=n * dmax)
    send = torch.from_numpy(send).long()
    send[send < n * dmax] = slot_of_edge[send[send < n * dmax]]
    rdev = lambda a: D.to_dev(a.to(torch.int32), dev, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    return {"n": n, "E": E, "D": dmax, "dmax_send": dmax_send, "ranges": species_ranges(element, n_present),
            "slot_edge": slot_edge, "slot_of_edge": slot_of_edge,
            "slot_sender": rdev(slot_sender), "send": rdev(send)}


def pad_edges(graph, x):
    """Per-edge host tensor [E, w] -> the padded receiver layout [n * D, w], pad slots zero."""
    return torch.cat([x, x.new_zeros(1, x.shape[1])])[graph["slot_edge"]]


def edge_sh(dev, graph, sh, dtype):
    """Per-edge spherical harmonics [E, 16] (host) -> device ([n, D, 16], its transpose [n, 16, D]): the edge
    sum multiplies from the left by one and the reverse pass by the other, and the host transposes for free."""
    sh = pad_edges(graph, sh).reshape(graph["n"], graph["D"], -1)
    return D.to_dev(sh, dev, dtype), D.to_dev(sh.transpose(1, 2).contiguous(), dev, dtype)


def _per_node(plan, key, n):
    """The constant ``plan[key]`` [p, q] (host) as a device [n, p, q] matmul operand, built once per atom count:
    matmul has no shared left operand, and a left multiply keeps the layer in the cm layout without permutes."""
    cache = plan["per_node"]
    if (key, n) not in cache:
        cache[key, n] = const(plan[key].expand(n, -1, -1), plan["dev"], plan["dtype"])
    return cache[key, n]


def interaction_setup(dev, w, layer, elements, irreps_in, residual, dtype, sh_ls=(0, 1, 2, 3)):
    """Device weights for one interaction block in the flattened cm layout [rows, A * 128], for the species
    ``elements`` present. Graph-free: pair with graph_setup."""
    import ttnn

    from . import reference as ref

    p = f"interactions.{layer}."
    hidden = [(128, l) for l in range(4)]
    a_in = sum(2 * l + 1 for _, l in irreps_in)
    sk_in, sk_out = (irreps_in, [(128, 0)]) if residual else (hidden, hidden)
    sk = dense.skip_plan(sk_in, len(w["atomic_numbers"]), sk_out, w[p + "skip_tp.weight"], elements)
    a_sk = (sum(2 * l + 1 for _, l in sk_in), sum(2 * l + 1 for _, l in sk_out))
    C, ins_of_k, mid = dense.conv_tp_plan(irreps_in, list(sh_ls), [l for _, l in hidden])
    K = C.shape[2]
    tdev = lambda a: const(a, dev, dtype)  # noqa: E731
    both = lambda a: (tdev(a), tdev(a.T))  # noqa: E731  (forward, reverse)
    fcw = lambda Ws: zip(*(both(W / W.shape[0] ** 0.5) for W in Ws))  # noqa: E731
    # The density MLP ends in one column. Widen it to 32 zero-padded columns (tanh(0^2) = 0): a width-1 row-major
    # tensor through to_layout/typecast hung two Galaxy chips (11-27), so nothing here is narrower than a tile.
    dens = ref._layers(w, p + "density_fn")
    dens[-1] = torch.nn.functional.pad(dens[-1], (0, 31))
    # conv_tp with the edge sum ahead of the Clebsch-Gordan contraction (dense.conv_tp_node_plan): the 0/1
    # selections onto its R rows fold into linear_up and into the radial MLP's last weight.
    a_of_r, ins_of_r, Cn = dense.conv_tp_node_plan(C, ins_of_k)
    R = len(a_of_r)
    tp_w = ref._layers(w, p + "conv_tp_weights")
    tp_w[-1] = tp_w[-1].reshape(tp_w[-1].shape[0], -1, 128)[:, ins_of_r, :].reshape(-1, R * 128)
    (dens, dens_t), (tp_w, tp_w_t) = fcw(dens), fcw(tp_w)
    up = dense.plan_matrix(dense.linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"]), a_in, a_in)
    up, up_t = both(up.reshape(-1, a_in, 128)[:, a_of_r, :].reshape(-1, R * 128))
    lin, lin_t = both(dense.plan_matrix(dense.linear_plan(mid, hidden, w[p + "linear.weight"]), K, 16))
    skip, skip_t = zip(*(both(dense.plan_matrix(s, *a_sk)) for s in sk))
    ones = torch.zeros(16 * 128, 32, dtype=torch.float64)  # row sum as a GEMM, into column 0 of a tile
    ones[:, 0] = 1.0
    return {
        "R": R, "K": K, "residual": residual,
        "up": up, "lin": lin, "skip": skip, "tp_w": tp_w, "dens": dens, "silu": ref._silu_2mom(),
        "cn": Cn, "cn_t": Cn.T, "per_node": {}, "dev": dev, "dtype": dtype,  # see _per_node
        "t": {"up": up_t, "lin": lin_t, "skip": skip_t, "tp_w": tp_w_t, "dens": dens_t,
              "ones": D.to_dev(ones, dev, ttnn.bfloat16)},
        "cfg": D.compute_kernel_config(),
    }


def _per_element(x, ranges, mats, cfg):
    import ttnn

    outs = [mm(x[lo:hi], M, cfg) for (lo, hi), M in zip(ranges, mats)
            if hi > lo]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)


def interaction(feats, sh, ef, plan, graph, tape=None):
    """feats [n, A * 128], sh = edge_sh(...), ef [n * D, n_radial] (device, tile, padded receiver layout)
    -> (x [n, 16 * 128] cm, skip [n, 128] or None)."""
    import ttnn

    from tt_atom import scatter

    cfg, n, Dn, R, K = plan["cfg"], graph["n"], graph["D"], plan["R"], plan["K"]
    t = {} if tape is None else tape
    t.update(tw_pre=[], u_pre=[])
    sc = _per_element(feats, graph["ranges"], plan["skip"], cfg) if plan["residual"] else None
    x1 = rows(graph["slot_sender"], mm(feats, plan["up"], cfg), (n * Dn, R * 128))
    tw = fc_net(ef, plan["tp_w"], plan["silu"], t["tw_pre"])  # [n * D, R * 128]
    Q = ttnn.reshape(ttnn.multiply(x1, tw), (n, Dn, R * 128))
    B = mm(sh[1], Q, cfg)  # [n, 16, R * 128]: the sum over each node's edges
    msg = mm(_per_node(plan, "cn_t", n), ttnn.reshape(B, (n, 16 * R, 128)), cfg)  # [n, K, 128]
    msg = ttnn.reshape(msg, (n, K * 128))
    u = fc_net(ef, plan["dens"], plan["silu"], t["u_pre"])
    dens = ttnn.tanh(ttnn.multiply(u, u))  # [n * D, 32], columns 1.. are zero
    density = scatter.linear32(ttnn, lambda p: ttnn.sum(ttnn.reshape(p, (n, Dn, 32)), dim=1), dens)[:, :1]
    lin, inv = mm(msg, plan["lin"], cfg), ttnn.reciprocal(ttnn.add(density, 1.0))
    t.update(x1=x1, tw=tw, Q=Q, sh=sh[0], u=u, dens=dens, lin=lin, inv=inv)
    msg = ttnn.multiply(lin, inv)
    if not plan["residual"]:
        msg = _per_element(msg, graph["ranges"], plan["skip"], cfg)
    return msg, sc


def interaction_bwd(g_out, g_sc, plan, graph, tape):
    """dE/d(x) [n, 16 * 128] and dE/d(skip) [n, 128] or None -> (dE/d feats [n, A * 128], dE/d sh [n * D, 16],
    dE/d ef [n * D, n_radial])."""
    import ttnn

    from tt_atom import scatter

    cfg, n, Dn, R, K = plan["cfg"], graph["n"], graph["D"], plan["R"], plan["K"]
    t, w, ranges = tape, plan["t"], graph["ranges"]
    g_msg = g_out if plan["residual"] else _per_element(g_out, ranges, w["skip"], cfg)
    # msg = lin / (density + 1): the density gradient lands in column 0 of a [n, 32] tile, like the forward's
    g_density = ttnn.neg(ttnn.multiply(mm(ttnn.multiply(g_msg, t["lin"]), w["ones"], cfg),
                                       ttnn.multiply(t["inv"], t["inv"])))
    g_raw = mm(ttnn.multiply(g_msg, t["inv"]), w["lin"], cfg)  # [n, K * 128]
    g_B = mm(_per_node(plan, "cn", n), ttnn.reshape(g_raw, (n, K, 128)), cfg)  # [n, 16 * R, 128]
    g_B = ttnn.reshape(g_B, (n, 16, R * 128))
    g_sh = ttnn.reshape(mm(t["Q"], perm(g_B, (0, 2, 1)), cfg), (n * Dn, 16))
    g_Q = ttnn.reshape(mm(t["sh"], g_B, cfg), (n * Dn, R * 128))
    g_u = ttnn.multiply(ttnn.reshape(g_density, (n, 1, 32)), ttnn.reshape(
        ttnn.multiply(ttnn.multiply(t["u"], 2.0), ttnn.add(ttnn.neg(ttnn.multiply(t["dens"], t["dens"])), 1.0)),
        (n, Dn, 32)))
    g_ef = ttnn.add(fc_net_bwd(ttnn.multiply(g_Q, t["x1"]), w["tp_w"], t["tw_pre"], plan["silu"]),
                    fc_net_bwd(ttnn.reshape(g_u, (n * Dn, 32)), w["dens"], t["u_pre"], plan["silu"]))
    g_up = scatter.segment_sum(ttnn, ttnn.multiply(g_Q, t["tw"]), graph["send"], graph["dmax_send"], n, R * 128)
    g_feats = mm(g_up, w["up"], cfg)
    if g_sc is not None:
        g_feats = ttnn.add(g_feats, _per_element(g_sc, ranges, w["skip"], cfg))
    return g_feats, g_sh, g_ef


def product_setup(dev, w, layer, elements, irreps_out, dtype):
    """Symmetric contraction + linear of ``products.<layer>`` and the layer's readout, flattened cm layout."""
    from . import reference as ref

    p, a = f"products.{layer}.", sum(2 * l + 1 for _, l in irreps_out)
    tables = dense.product_tables(w, layer, len(irreps_out), elements=elements)
    tdev = lambda t: const(t, dev, dtype)  # noqa: E731
    r = f"readouts.{layer}."
    if r + "linear.weight" in w:
        ro = [dense.plan_matrix(dense.linear_plan(irreps_out, [(1, 0)], w[r + "linear.weight"]), a, 1)]
    else:
        hid = w[r + "linear_2.weight"].numel()
        ro = [dense.plan_matrix(dense.linear_plan(irreps_out, [(hid, 0)], w[r + "linear_1.weight"]), a, 1),
              dense.plan_matrix(dense.linear_plan([(hid, 0)], [(1, 0)], w[r + "linear_2.weight"]), 1, 1)]
    lin = dense.plan_matrix(dense.linear_plan(irreps_out, irreps_out, w[p + "linear.weight"]), a, a)
    # The reverse pass starts at the readouts: dE/d(node energy) is the model's energy scale for every atom,
    # folded into the last readout weight, which then is one constant row.
    ro_t = [W.T for W in ro]
    ro_t[-1] = ro_t[-1] * float(w["scale_shift.scale"])
    return {"sc": symmetric_contraction_setup(dev, tables, dtype), "a": a, "lin": tdev(lin),
            "readout": [tdev(W) for W in ro], "silu": ref._silu_2mom(), "cfg": D.compute_kernel_config(),
            # the last reverse readout weight is a broadcast row, not a matmul operand
            "t": {"lin": tdev(lin.T),
                  "readout": [tdev(W) for W in ro_t[:-1]] + [D.to_dev(ro_t[-1].contiguous(), dev, dtype)]}}


def product(x, sc, plan, graph, tape=None):
    """x [n, 16 * 128] cm (interaction output), sc [n, a * 128] or None -> (features [n, a * 128], node energy [n, 1])."""
    import ttnn

    cfg, n = plan["cfg"], graph["n"]
    xr = ttnn.reshape(perm(ttnn.reshape(x, (n, 16, 128)), (0, 2, 1)), (n * 128, 16))
    y = symmetric_contraction(xr, plan["sc"], graph["ranges"], n, 128, tape)  # [128, n, a]
    y = ttnn.reshape(perm(y, (1, 2, 0)), (n, plan["a"] * 128))
    y = mm(y, plan["lin"], cfg)
    if sc is not None:
        y = ttnn.add(y, sc)
    ro = plan["readout"]
    e = mm(y, ro[0], cfg)
    if len(ro) > 1:
        if tape is not None:
            tape["h"] = e
        e = mm(ttnn.multiply(ttnn.silu(e), plan["silu"]), ro[1], cfg)
    return y, e


def product_bwd(g_y, plan, graph, tape):
    """dE/d(features) [n, a * 128] from the layer above (None for the last layer) -> (dE/dx [n, 16 * 128],
    dE/d(features) including this layer's readout, which is also the gradient of a residual skip)."""
    import ttnn

    cfg, n, a, t = plan["cfg"], graph["n"], plan["a"], plan["t"]
    g_ro = t["readout"][-1]  # [1, width]: the same for every atom
    if len(t["readout"]) > 1:
        g_ro = mm(ttnn.multiply(silu_grad(tape["h"], plan["silu"]), g_ro), t["readout"][0], cfg)
    g_y = g_ro if g_y is None else ttnn.add(g_y, g_ro)
    g_s = perm(ttnn.reshape(mm(g_y, t["lin"], cfg), (n, a, 128)), (2, 0, 1))  # [128, n, a]
    g_x = symmetric_contraction_bwd(g_s, plan["sc"], graph["ranges"], n, 128, tape)  # [n * 128, 16]
    return ttnn.reshape(perm(ttnn.reshape(g_x, (n, 128, 16)), (0, 2, 1)), (n, 16 * 128)), g_y
