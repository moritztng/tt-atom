"""The two MACE hot blocks on the card, built from dense.py's device forms with stock ttnn ops.

Host-side setup (``*_setup``) turns the folded float64 constants into device tensors once; the per-step
functions take and return device tensors. Graded against the float64 reference in tests/test_mace_device.py.
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
        "widths": [t.shape[-1] for t, _, _ in tables],
        "cfg": D.compute_kernel_config(),
    }


def symmetric_contraction(x, plan, ranges, n, c):
    """x: device [n * c, ell] (rows = (node, channel)), ``ranges`` = graph_setup's species row ranges
    -> device [c, n, sum(2L+1)]."""
    import ttnn

    cfg, feats, prev = plan["cfg"], [], None
    for Sp, Sl in plan["sels"]:
        xl = mm(x, Sl, cfg)
        prev = xl if Sp is None else ttnn.multiply(mm(prev, Sp, cfg), xl)
        feats.append(prev)
    P = ttnn.concat(feats, dim=1)  # [n * c, F]
    P = perm(ttnn.reshape(P, (n, c, P.shape[-1])), (1, 0, 2))  # [c, n, F]
    outs = []
    for (lo, hi), T in zip(ranges, plan["T"]):
        if hi > lo:
            outs.append(mm(P[:, lo:hi, :], T, cfg))  # [c, n_e, D]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)


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
    return {"C": D.to_dev(C.permute(1, 0, 2).reshape(B, A * K), dev, dtype), "G": D.to_dev(G, dev, dtype),
            "A": A, "K": K, "cfg": D.compute_kernel_config()}


def conv_tp(x1, sh, w, plan):
    """x1 device [E, mul, A], sh [E, B], w [E, n_ins, mul] -> device [E, mul, K] (dense.conv_tp before its
    block reorder)."""
    import ttnn

    cfg, E = plan["cfg"], x1.shape[0]
    M = ttnn.reshape(mm(sh, plan["C"], cfg), (E, plan["A"], plan["K"]))
    wk = mm(perm(w, (0, 2, 1)), plan["G"], cfg)  # [E, mul, K]
    return ttnn.multiply(mm(x1, M, cfg), wk)


def conv_tp_to_ref(out, ins_of_k):
    """[E, mul, K] torch -> reference layout (instruction blocks, each [E, mul * (2l+1)])."""
    blocks, k = [], 0
    for n in range(int(ins_of_k.max()) + 1):
        d = int((ins_of_k == n).sum())
        blocks.append(out[:, :, k:k + d].reshape(out.shape[0], -1))
        k += d
    return torch.cat(blocks, dim=1)


def fc_net(x, Ws, act_scale):
    """e3nn FullyConnectedNet on device; ``Ws`` already divided by sqrt(fan-in)."""
    import ttnn

    cfg = D.compute_kernel_config()
    for i, W in enumerate(Ws):
        x = mm(x, W, cfg)
        if i < len(Ws) - 1:
            x = ttnn.multiply(ttnn.silu(x), act_scale)
    return x


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
    rdev = lambda a: D.to_dev(a, dev, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    return {"n": n, "E": E, "dmax": dmax,
            "ranges": species_ranges(element, n_present),
            "sender": rdev(edge_index[0].to(torch.int32)), "recv": rdev(torch.from_numpy(g_recv))}


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
    tdev = lambda a: D.to_dev(a, dev, dtype)  # noqa: E731
    fcw = lambda name: [tdev(W / W.shape[0] ** 0.5) for W in ref._layers(w, p + name)]  # noqa: E731
    # The density MLP ends in one column. Widen it to 32 zero-padded columns (tanh(0^2) = 0): a width-1 row-major
    # tensor through to_layout/typecast hung two Galaxy chips (11-27), so nothing here is narrower than a tile.
    dens = ref._layers(w, p + "density_fn")
    dens[-1] = torch.nn.functional.pad(dens[-1], (0, 31))
    dens = [tdev(W / W.shape[0] ** 0.5) for W in dens]
    return {
        "a_in": a_in, "K": K, "residual": residual,
        "up": tdev(dense.plan_matrix(dense.linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"]), a_in, a_in)),
        "lin": tdev(dense.plan_matrix(dense.linear_plan(mid, hidden, w[p + "linear.weight"]), K, 16)),
        "skip": [tdev(dense.plan_matrix(s, *a_sk)) for s in sk],
        "tp_w": fcw("conv_tp_weights"), "dens": dens, "silu": ref._silu_2mom(),
        "tp": conv_tp_setup(dev, C, ins_of_k, dtype),
        "cfg": D.compute_kernel_config(),
    }


def _per_element(x, ranges, mats, cfg):
    import ttnn

    outs = [mm(x[lo:hi], M, cfg) for (lo, hi), M in zip(ranges, mats)
            if hi > lo]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)


def interaction(feats, sh, ef, plan, graph):
    """feats [n, A * 128], sh [E, 16], ef [E, 8] (device, tile) -> (x [n, 16 * 128] cm, skip [n, 128] or None)."""
    import ttnn

    from tt_atom import scatter

    cfg, n, E, A, K = plan["cfg"], graph["n"], graph["E"], plan["a_in"], plan["K"]
    sc = _per_element(feats, graph["ranges"], plan["skip"], cfg) if plan["residual"] else None
    up = mm(feats, plan["up"], cfg)  # [n, A * 128]
    x1 = scatter.embed_rows(ttnn, graph["sender"], ttnn.to_layout(up, ttnn.ROW_MAJOR_LAYOUT))  # [E, A * 128]
    x1 = perm(ttnn.to_layout(ttnn.reshape(x1, (E, A, 128)), ttnn.TILE_LAYOUT), (0, 2, 1))  # [E, 128, A]
    tw = ttnn.reshape(fc_net(ef, plan["tp_w"], plan["silu"]), (E, -1, 128))  # [E, n_ins, 128]
    m = conv_tp(x1, sh, tw, plan["tp"])  # [E, 128, K]
    m = ttnn.reshape(perm(m, (0, 2, 1)), (E, K * 128))
    msg = scatter.segment_sum(ttnn, m, graph["recv"], graph["dmax"], n, K * 128)
    dens = fc_net(ef, plan["dens"], plan["silu"])
    dens = ttnn.tanh(ttnn.multiply(dens, dens))  # [E, 32], columns 1.. are zero
    density = scatter.segment_sum(ttnn, dens, graph["recv"], graph["dmax"], n, 32)[:, :1]
    msg = ttnn.multiply(mm(msg, plan["lin"], cfg),
                        ttnn.reciprocal(ttnn.add(density, 1.0)))
    if not plan["residual"]:
        msg = _per_element(msg, graph["ranges"], plan["skip"], cfg)
    return msg, sc


def product_setup(dev, w, layer, elements, irreps_out, dtype):
    """Symmetric contraction + linear of ``products.<layer>`` and the layer's readout, flattened cm layout."""
    from . import reference as ref

    p, a = f"products.{layer}.", sum(2 * l + 1 for _, l in irreps_out)
    tables = dense.product_tables(w, layer, len(irreps_out), elements=elements)
    tdev = lambda t: D.to_dev(t, dev, dtype)  # noqa: E731
    r = f"readouts.{layer}."
    if r + "linear.weight" in w:
        ro = [tdev(dense.plan_matrix(dense.linear_plan(irreps_out, [(1, 0)], w[r + "linear.weight"]), a, 1))]
    else:
        hid = w[r + "linear_2.weight"].numel()
        ro = [tdev(dense.plan_matrix(dense.linear_plan(irreps_out, [(hid, 0)], w[r + "linear_1.weight"]), a, 1)),
              tdev(dense.plan_matrix(dense.linear_plan([(hid, 0)], [(1, 0)], w[r + "linear_2.weight"]), 1, 1))]
    return {"sc": symmetric_contraction_setup(dev, tables, dtype), "a": a,
            "lin": tdev(dense.plan_matrix(dense.linear_plan(irreps_out, irreps_out, w[p + "linear.weight"]), a, a)),
            "readout": ro, "silu": ref._silu_2mom(), "cfg": D.compute_kernel_config()}


def product(x, sc, plan, graph):
    """x [n, 16 * 128] cm (interaction output), sc [n, a * 128] or None -> (features [n, a * 128], node energy [n, 1])."""
    import ttnn

    cfg, n = plan["cfg"], graph["n"]
    rows = ttnn.reshape(perm(ttnn.reshape(x, (n, 16, 128)), (0, 2, 1)), (n * 128, 16))
    y = symmetric_contraction(rows, plan["sc"], graph["ranges"], n, 128)  # [128, n, a]
    y = ttnn.reshape(perm(y, (1, 2, 0)), (n, plan["a"] * 128))
    y = mm(y, plan["lin"], cfg)
    if sc is not None:
        y = ttnn.add(y, sc)
    ro = plan["readout"]
    e = mm(y, ro[0], cfg)
    if len(ro) > 1:
        e = mm(ttnn.multiply(ttnn.silu(e), plan["silu"]), ro[1], cfg)
    return y, e
