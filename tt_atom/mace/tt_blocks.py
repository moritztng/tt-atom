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


def mm(a, b, cfg):
    """matmul that keeps float32 operands at float32 accuracy.

    The matrix engine reads float32 operands through 19-bit source registers, i.e. truncated to a 10-bit
    mantissa, and truncation is biased: MACE's per-atom energies came out ~9 meV/atom high from that alone.
    For float32 inputs each operand is split into bfloat16 hi + lo parts and the three significant products
    are summed (a bfloat16 x bfloat16 product is exact in the float32 accumulator), leaving ~2^-16 relative.
    bfloat16 inputs take the plain matmul.
    """
    import ttnn

    if a.dtype != ttnn.float32 and b.dtype != ttnn.float32:
        return ttnn.matmul(a, b, compute_kernel_config=cfg)

    def split(x):
        hi = ttnn.typecast(x, ttnn.bfloat16)
        return hi, ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)

    (ah, al), (bh, bl) = split(a), split(b)
    f = lambda x, y: ttnn.matmul(x, y, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(f(ah, bh), f(ah, bl)), f(al, bh))


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
    return {
        "sels": [(None if Sp is None else D.to_dev(Sp, dev, dtype), D.to_dev(Sl, dev, dtype)) for Sp, Sl in sels],
        "T": [D.to_dev(T[e], dev, dtype) for e in range(T.shape[0])],
        "t": {"sels": [(None if Sp is None else D.to_dev(Sp.T.contiguous(), dev, dtype),
                        D.to_dev(Sl.T.contiguous(), dev, dtype)) for Sp, Sl in sels],
              "T": [D.to_dev(T[e].transpose(1, 2).contiguous(), dev, dtype) for e in range(T.shape[0])]},
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


def conv_tp_setup(dev, C, ins_of_k, dtype):
    """C [A, B, K] -> device [B, A * K]; G [n_ins, K] 0/1 maps each instruction's weight to its columns."""
    A, B, K = C.shape
    G = torch.zeros(int(ins_of_k.max()) + 1, K, dtype=torch.float64)
    G[ins_of_k, torch.arange(K)] = 1.0
    Cf = C.permute(1, 0, 2).reshape(B, A * K)
    return {"C": D.to_dev(Cf, dev, dtype), "G": D.to_dev(G, dev, dtype),
            "t": {"C": D.to_dev(Cf.T.contiguous(), dev, dtype), "G": D.to_dev(G.T.contiguous(), dev, dtype)},
            "A": A, "K": K, "cfg": D.compute_kernel_config()}


def conv_tp(x1, sh, w, plan, tape=None):
    """x1 device [E, mul, A], sh [E, B], w [E, n_ins, mul] -> device [E, mul, K] (dense.conv_tp before its
    block reorder)."""
    import ttnn

    cfg, E = plan["cfg"], x1.shape[0]
    M = ttnn.reshape(mm(sh, plan["C"], cfg), (E, plan["A"], plan["K"]))
    wk = mm(perm(w, (0, 2, 1)), plan["G"], cfg)  # [E, mul, K]
    Z = mm(x1, M, cfg)
    if tape is not None:
        tape.update(M=M, wk=wk, Z=Z)
    return ttnn.multiply(Z, wk)


def conv_tp_bwd(g, x1t, plan, tape):
    """g [E, mul, K], x1t [E, A, mul] (the forward's x1, transposed) -> (g_x1 [E, mul, A], g_sh [E, B],
    g_w [E, n_ins * mul])."""
    import ttnn

    cfg, E, t = plan["cfg"], g.shape[0], plan["t"]
    g_Z = ttnn.multiply(g, tape["wk"])
    g_w = perm(mm(ttnn.multiply(g, tape["Z"]), t["G"], cfg), (0, 2, 1))  # [E, n_ins, mul]
    g_sh = mm(ttnn.reshape(mm(x1t, g_Z, cfg), (E, plan["A"] * plan["K"])), t["C"], cfg)
    return mm(g_Z, perm(tape["M"], (0, 2, 1)), cfg), g_sh, ttnn.reshape(g_w, (E, g_w.shape[1] * g_w.shape[2]))


def conv_tp_to_ref(out, ins_of_k):
    """[E, mul, K] torch -> reference layout (instruction blocks, each [E, mul * (2l+1)])."""
    blocks, k = [], 0
    for n in range(int(ins_of_k.max()) + 1):
        d = int((ins_of_k == n).sum())
        blocks.append(out[:, :, k:k + d].reshape(out.shape[0], -1))
        k += d
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
    """Per-structure device tables, shared by every layer: species row ranges and the edge gather/scatter
    tables. Built once per neighbour list; the weights (interaction_setup, product_setup) never see the graph,
    so an MD rebuild or a new structure in a screen re-uploads only these. ``element`` [n] indexes the present
    species and must be sorted (atoms ordered by species at setup)."""
    import ttnn

    from tt_atom import scatter

    assert bool((element[1:] >= element[:-1]).all()), "sort atoms by species at setup"
    n, E = len(element), edge_index.shape[1]
    g_recv, dmax = scatter.build_gather(edge_index[1], n, E)
    g_send, dmax_send = scatter.build_gather(edge_index[0], n, E)
    rdev = lambda a: D.to_dev(a, dev, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    return {"n": n, "E": E, "dmax": dmax, "dmax_send": dmax_send,
            "ranges": species_ranges(element, n_present),
            "sender": rdev(edge_index[0].to(torch.int32)), "recv": rdev(torch.from_numpy(g_recv)),
            # the reverse pass gathers by receiver and scatters to senders
            "receiver": rdev(edge_index[1].to(torch.int32)), "send": rdev(torch.from_numpy(g_send))}


def interaction_setup(dev, w, layer, elements, irreps_in, residual, dtype, sh_ls=(0, 1, 2, 3)):
    """Device weights for one interaction block in the flattened cm layout [rows, A * 128], for the species
    ``elements`` present. Graph-free: pair with graph_setup."""
    from . import reference as ref

    p = f"interactions.{layer}."
    hidden = [(128, l) for l in range(4)]
    a_in = sum(2 * l + 1 for _, l in irreps_in)
    sk_in, sk_out = (irreps_in, [(128, 0)]) if residual else (hidden, hidden)
    sk = dense.skip_plan(sk_in, len(w["atomic_numbers"]), sk_out, w[p + "skip_tp.weight"], elements)
    a_sk = (sum(2 * l + 1 for _, l in sk_in), sum(2 * l + 1 for _, l in sk_out))
    C, ins_of_k, mid = dense.conv_tp_plan(irreps_in, list(sh_ls), [l for _, l in hidden])
    K = C.shape[2]
    tdev = lambda a: D.to_dev(a.contiguous(), dev, dtype)  # noqa: E731
    both = lambda a: (tdev(a), tdev(a.T))  # noqa: E731  (forward, reverse)
    fcw = lambda Ws: zip(*(both(W / W.shape[0] ** 0.5) for W in Ws))  # noqa: E731
    # The density MLP ends in one column. Widen it to 32 zero-padded columns (tanh(0^2) = 0): a width-1 row-major
    # tensor through to_layout/typecast hung two Galaxy chips (11-27), so nothing here is narrower than a tile.
    dens = ref._layers(w, p + "density_fn")
    dens[-1] = torch.nn.functional.pad(dens[-1], (0, 31))
    (dens, dens_t), (tp_w, tp_w_t) = fcw(dens), fcw(ref._layers(w, p + "conv_tp_weights"))
    up, up_t = both(dense.plan_matrix(dense.linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"]), a_in, a_in))
    lin, lin_t = both(dense.plan_matrix(dense.linear_plan(mid, hidden, w[p + "linear.weight"]), K, 16))
    skip, skip_t = zip(*(both(dense.plan_matrix(s, *a_sk)) for s in sk))
    ones = torch.zeros(16 * 128, 32, dtype=torch.float64)  # row sum as a GEMM, into column 0 of a tile
    ones[:, 0] = 1.0
    return {
        "a_in": a_in, "K": K, "residual": residual,
        "up": up, "lin": lin, "skip": skip, "tp_w": tp_w, "dens": dens, "silu": ref._silu_2mom(),
        "t": {"up": up_t, "lin": lin_t, "skip": skip_t, "tp_w": tp_w_t, "dens": dens_t, "ones": tdev(ones)},
        "tp": conv_tp_setup(dev, C, ins_of_k, dtype),
        "cfg": D.compute_kernel_config(),
    }


def _per_element(x, ranges, mats, cfg):
    import ttnn

    outs = [mm(x[lo:hi], M, cfg) for (lo, hi), M in zip(ranges, mats)
            if hi > lo]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)


def interaction(feats, sh, ef, plan, graph, tape=None):
    """feats [n, A * 128], sh [E, 16], ef [E, n_radial] (device, tile) -> (x [n, 16 * 128] cm, skip [n, 128] or None)."""
    import ttnn

    from tt_atom import scatter

    cfg, n, E, A, K = plan["cfg"], graph["n"], graph["E"], plan["a_in"], plan["K"]
    t = {"tw_pre": [], "u_pre": []} if tape is None else tape
    t.update(tw_pre=[], u_pre=[])
    sc = _per_element(feats, graph["ranges"], plan["skip"], cfg) if plan["residual"] else None
    up = mm(feats, plan["up"], cfg)  # [n, A * 128]
    x1t = rows(graph["sender"], up, (E, A, 128))
    tw = ttnn.reshape(fc_net(ef, plan["tp_w"], plan["silu"], t["tw_pre"]), (E, -1, 128))  # [E, n_ins, 128]
    m = conv_tp(perm(x1t, (0, 2, 1)), sh, tw, plan["tp"], t)  # [E, 128, K]
    m = ttnn.reshape(perm(m, (0, 2, 1)), (E, K * 128))
    msg = scatter.segment_sum(ttnn, m, graph["recv"], graph["dmax"], n, K * 128)
    u = fc_net(ef, plan["dens"], plan["silu"], t["u_pre"])
    dens = ttnn.tanh(ttnn.multiply(u, u))  # [E, 32], columns 1.. are zero
    density = scatter.segment_sum(ttnn, dens, graph["recv"], graph["dmax"], n, 32)[:, :1]
    lin, inv = mm(msg, plan["lin"], cfg), ttnn.reciprocal(ttnn.add(density, 1.0))
    t.update(x1t=x1t, u=u, dens=dens, lin=lin, inv=inv)
    msg = ttnn.multiply(lin, inv)
    if not plan["residual"]:
        msg = _per_element(msg, graph["ranges"], plan["skip"], cfg)
    return msg, sc


def interaction_bwd(g_out, g_sc, plan, graph, tape):
    """dE/d(x) [n, 16 * 128] and dE/d(skip) [n, 128] or None -> (dE/d feats [n, A * 128], dE/d sh [E, 16],
    dE/d ef [E, n_radial])."""
    import ttnn

    from tt_atom import scatter

    cfg, n, E, A, K = plan["cfg"], graph["n"], graph["E"], plan["a_in"], plan["K"]
    t, w, ranges = tape, plan["t"], graph["ranges"]
    g_msg = g_out if plan["residual"] else _per_element(g_out, ranges, w["skip"], cfg)
    # msg = lin / (density + 1): the density gradient lands in column 0 of a [n, 32] tile, like the forward's
    g_density = ttnn.neg(ttnn.multiply(mm(ttnn.multiply(g_msg, t["lin"]), w["ones"], cfg),
                                       ttnn.multiply(t["inv"], t["inv"])))
    g_raw = mm(ttnn.multiply(g_msg, t["inv"]), w["lin"], cfg)  # [n, K * 128]
    g_m = perm(rows(graph["receiver"], g_raw, (E, K, 128)), (0, 2, 1))  # [E, 128, K]
    g_x1, g_sh, g_tw = conv_tp_bwd(g_m, t["x1t"], plan["tp"], t)
    g_u = ttnn.multiply(rows(graph["receiver"], g_density, (E, 32)),
                        ttnn.multiply(ttnn.multiply(t["u"], 2.0), ttnn.add(ttnn.neg(ttnn.multiply(t["dens"], t["dens"])), 1.0)))
    g_ef = ttnn.add(fc_net_bwd(g_tw, w["tp_w"], t["tw_pre"], plan["silu"]),
                    fc_net_bwd(g_u, w["dens"], t["u_pre"], plan["silu"]))
    g_up = scatter.segment_sum(ttnn, ttnn.reshape(perm(g_x1, (0, 2, 1)), (E, A * 128)), graph["send"],
                               graph["dmax_send"], n, A * 128)
    g_feats = mm(g_up, w["up"], cfg)
    if g_sc is not None:
        g_feats = ttnn.add(g_feats, _per_element(g_sc, ranges, w["skip"], cfg))
    return g_feats, g_sh, g_ef


def product_setup(dev, w, layer, elements, irreps_out, dtype):
    """Symmetric contraction + linear of ``products.<layer>`` and the layer's readout, flattened cm layout."""
    from . import reference as ref

    p, a = f"products.{layer}.", sum(2 * l + 1 for _, l in irreps_out)
    tables = dense.product_tables(w, layer, len(irreps_out), elements=elements)
    tdev = lambda t: D.to_dev(t.contiguous(), dev, dtype)  # noqa: E731
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
            "t": {"lin": tdev(lin.T), "readout": [tdev(W) for W in ro_t]}}


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
