"""The two MACE hot blocks on the card, built from dense.py's device forms with stock ttnn ops.

Host-side setup (``*_setup``) turns the folded float64 constants into device tensors once; the per-step
functions take and return device tensors. Graded against the float64 reference in tests/test_mace_device.py.
"""
from __future__ import annotations

import torch

from tt_atom import device as D
from . import dense


def symmetric_contraction_setup(dev, tables, element, dtype):
    """``tables`` = dense.product_tables(..., elements=present) for one layer: [(T, parent, last)] per output
    irrep, T [n_present, c, F, 2L+1]. ``element`` [n] indexes T's first axis and must be sorted (callers
    order atoms by species once at setup), so each species is one contiguous row range.

    All output irreps of the layer share the monomials, so their tables go side by side on the last axis
    and each species costs one channel-batched matmul.
    """
    import ttnn

    assert bool((element[1:] >= element[:-1]).all()), "sort atoms by species at setup"
    _, parent, last = tables[0]
    ell = int(last.max()) + 1
    sels = dense.monomial_selectors(parent, last, ell)
    T = torch.cat([t for t, _, _ in tables], dim=-1)  # [E, c, F, sum(2L+1)]
    ranges = [(int((element < e).sum()), int((element <= e).sum())) for e in range(T.shape[0])]
    return {
        "sels": [(None if Sp is None else D.to_dev(Sp, dev, dtype), D.to_dev(Sl, dev, dtype)) for Sp, Sl in sels],
        "T": [D.to_dev(T[e], dev, dtype) for e in range(T.shape[0])],
        "ranges": ranges,
        "widths": [t.shape[-1] for t, _, _ in tables],
        "cfg": D.compute_kernel_config(),
    }


def symmetric_contraction(x, plan, n, c):
    """x: device [n * c, ell] (rows = (node, channel)) -> device [c, n, sum(2L+1)]."""
    import ttnn

    cfg, feats, prev = plan["cfg"], [], None
    for Sp, Sl in plan["sels"]:
        xl = ttnn.matmul(x, Sl, compute_kernel_config=cfg)
        prev = xl if Sp is None else ttnn.multiply(ttnn.matmul(prev, Sp, compute_kernel_config=cfg), xl)
        feats.append(prev)
    P = ttnn.concat(feats, dim=1)  # [n * c, F]
    P = ttnn.permute(ttnn.reshape(P, (n, c, P.shape[-1])), (1, 0, 2))  # [c, n, F]
    outs = []
    for (lo, hi), T in zip(plan["ranges"], plan["T"]):
        if hi > lo:
            outs.append(ttnn.matmul(P[:, lo:hi, :], T, compute_kernel_config=cfg))  # [c, n_e, D]
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
    M = ttnn.reshape(ttnn.matmul(sh, plan["C"], compute_kernel_config=cfg), (E, plan["A"], plan["K"]))
    wk = ttnn.matmul(ttnn.permute(w, (0, 2, 1)), plan["G"], compute_kernel_config=cfg)  # [E, mul, K]
    return ttnn.multiply(ttnn.matmul(x1, M, compute_kernel_config=cfg), wk)


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
        x = ttnn.matmul(x, W, compute_kernel_config=cfg)
        if i < len(Ws) - 1:
            x = ttnn.multiply(ttnn.silu(x), act_scale)
    return x


def interaction_setup(dev, w, layer, element, elements, edge_index, irreps_in, residual, dtype, sh_ls=(0, 1, 2, 3)):
    """Device constants and gather tables for one interaction block in the flattened cm layout [rows, A * 128].
    ``element`` [n] indexes ``elements`` and must be sorted (atoms ordered by species at setup)."""
    import ttnn

    from tt_atom import scatter
    from . import reference as ref

    assert bool((element[1:] >= element[:-1]).all()), "sort atoms by species at setup"
    p, n, E = f"interactions.{layer}.", len(element), edge_index.shape[1]
    hidden = [(128, l) for l in range(4)]
    a_in = sum(2 * l + 1 for _, l in irreps_in)
    sk_in, sk_out = (irreps_in, [(128, 0)]) if residual else (hidden, hidden)
    sk = dense.skip_plan(sk_in, len(w["atomic_numbers"]), sk_out, w[p + "skip_tp.weight"], elements)
    a_sk = (sum(2 * l + 1 for _, l in sk_in), sum(2 * l + 1 for _, l in sk_out))
    C, ins_of_k, mid = dense.conv_tp_plan(irreps_in, list(sh_ls), [l for _, l in hidden])
    K = C.shape[2]
    tdev = lambda a: D.to_dev(a, dev, dtype)  # noqa: E731
    rdev = lambda a: D.to_dev(a, dev, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    g_recv, dmax = scatter.build_gather(edge_index[1], n, E)
    fcw = lambda name: [tdev(W / W.shape[0] ** 0.5) for W in ref._layers(w, p + name)]  # noqa: E731
    return {
        "n": n, "E": E, "a_in": a_in, "K": K, "residual": residual,
        "up": tdev(dense.plan_matrix(dense.linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"]), a_in, a_in)),
        "lin": tdev(dense.plan_matrix(dense.linear_plan(mid, hidden, w[p + "linear.weight"]), K, 16)),
        "skip": [tdev(dense.plan_matrix(s, *a_sk)) for s in sk],
        "ranges": [(int((element < e).sum()), int((element <= e).sum())) for e in range(len(elements))],
        "tp_w": fcw("conv_tp_weights"), "dens": fcw("density_fn"), "silu": ref._silu_2mom(),
        "tp": conv_tp_setup(dev, C, ins_of_k, dtype),
        "sender": rdev(edge_index[0].to(torch.int32)),
        "recv": rdev(torch.from_numpy(g_recv)), "dmax": dmax,
        "cfg": D.compute_kernel_config(),
    }


def _per_element(x, plan, mats):
    import ttnn

    outs = [ttnn.matmul(x[lo:hi], M, compute_kernel_config=plan["cfg"]) for (lo, hi), M in zip(plan["ranges"], mats)
            if hi > lo]
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)


def interaction(feats, sh, ef, plan):
    """feats [n, A * 128], sh [E, 16], ef [E, 8] (device, tile) -> (x [n, 16 * 128] cm, skip [n, 128] or None)."""
    import ttnn

    from tt_atom import scatter

    cfg, n, E, A, K = plan["cfg"], plan["n"], plan["E"], plan["a_in"], plan["K"]
    sc = _per_element(feats, plan, plan["skip"]) if plan["residual"] else None
    up = ttnn.matmul(feats, plan["up"], compute_kernel_config=cfg)  # [n, A * 128]
    x1 = scatter.embed_rows(ttnn, plan["sender"], ttnn.to_layout(up, ttnn.ROW_MAJOR_LAYOUT))  # [E, A * 128]
    x1 = ttnn.permute(ttnn.to_layout(ttnn.reshape(x1, (E, A, 128)), ttnn.TILE_LAYOUT), (0, 2, 1))  # [E, 128, A]
    tw = ttnn.reshape(fc_net(ef, plan["tp_w"], plan["silu"]), (E, -1, 128))  # [E, n_ins, 128]
    m = conv_tp(x1, sh, tw, plan["tp"])  # [E, 128, K]
    m = ttnn.reshape(ttnn.permute(m, (0, 2, 1)), (E, K * 128))
    msg = scatter.segment_sum(ttnn, m, plan["recv"], plan["dmax"], n, K * 128)
    dens = fc_net(ef, plan["dens"], plan["silu"])
    dens = ttnn.tanh(ttnn.multiply(dens, dens))  # [E, 1]
    density = scatter.segment_sum(ttnn, dens, plan["recv"], plan["dmax"], n, 1)
    msg = ttnn.multiply(ttnn.matmul(msg, plan["lin"], compute_kernel_config=cfg),
                        ttnn.reciprocal(ttnn.add(density, 1.0)))
    if not plan["residual"]:
        msg = _per_element(msg, plan, plan["skip"])
    return msg, sc
