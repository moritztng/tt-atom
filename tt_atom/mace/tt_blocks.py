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
