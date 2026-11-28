"""MACE blocks recast as the dense ops a device runs: constant gathers, elementwise products, batched matmuls.

Same maths as ``reference.py``, a different association; graded against it in float64.

Symmetric contraction. MACE contracts per node ``sum_nu U_nu . (W_nu[element] x^(x)nu)`` with U carrying nu
equal ell indices. Because x^(x)nu is symmetric, only its distinct monomials matter (16 + 136 + 816 = 968
for ell = 16, nu <= 3, against 16 + 256 + 4096 full products), and ``U_nu @ W_nu[element]`` does not depend
on the node. So at load time each element folds into one [channels, 968, 2L+1] table, and at run time the
block is: build the monomials (one gather + one multiply per order), then one channel-batched matmul per
element. No per-node U contraction is left.
"""
from __future__ import annotations

from itertools import combinations_with_replacement, product

import torch

from tt_atom.equivariant import wigner_3j


def monomials(ell, corr):
    """Sorted index tuples of every distinct monomial of degree 1..corr, plus the gather plan that builds
    degree nu from degree nu-1: feature f = features[parent[f]] * x[last[f]] (parent -1 = degree 1)."""
    tuples, parent, last, where = [], [], [], {}
    for nu in range(1, corr + 1):
        for t in combinations_with_replacement(range(ell), nu):
            where[t] = len(tuples)
            tuples.append(t)
            parent.append(where[t[:-1]] if nu > 1 else -1)
            last.append(t[-1])
    return tuples, torch.tensor(parent), torch.tensor(last)


def fold_contraction(U, W):
    """Fold one MACE Contraction into per-element tables T [n_elements, c, F, 2L+1] over the monomials.

    U[nu-1]: [(2L+1)] + [ell]*nu + [K_nu] (no leading axis for L = 0); W[nu-1]: [n_elements, K_nu, c].
    A monomial's entry sums U over every index order that sorts to it, so U need not be symmetric.
    """
    corr, ell = len(U), U[0].shape[-2]
    tuples, _, _ = monomials(ell, corr)
    D = U[0][..., 0, 0].numel()
    blocks = []
    for nu in range(1, corr + 1):
        u = U[nu - 1].reshape(D, ell ** nu, -1)
        mono = [t for t in tuples if len(t) == nu]
        col = {t: m for m, t in enumerate(mono)}
        S = torch.zeros(ell ** nu, len(mono), dtype=u.dtype)
        for f, idx in enumerate(product(range(ell), repeat=nu)):
            S[f, col[tuple(sorted(idx))]] = 1.0
        blocks.append(torch.einsum("dfk,fm,ekc->ecmd", u, S, W[nu - 1]))
    return torch.cat(blocks, dim=2)


def monomial_features(x, parent, last):
    """x [n, c, ell] -> [n, c, F]: each feature is one earlier feature times one component of x."""
    feats = []
    for p, l in zip(parent.tolist(), last.tolist()):
        xl = x[..., l]
        feats.append(xl if p < 0 else feats[p] * xl)
    return torch.stack(feats, dim=-1)


def symmetric_contraction(x, element, T, parent, last):
    """x [n, c, ell], element [n] (index into T's first axis) -> [n, c * (2L+1)], reference.py's layout."""
    P = monomial_features(x, parent, last)
    out = torch.zeros(x.shape[0], x.shape[1], T.shape[-1], dtype=x.dtype)
    for e in element.unique().tolist():
        rows = element == e
        out[rows] = torch.bmm(P[rows].transpose(0, 1), T[e]).transpose(0, 1)  # batched over channels
    return out.reshape(x.shape[0], -1)


def monomial_selectors(parent, last, ell, dtype=torch.float64):
    """0/1 matrices that build the monomials with GEMMs instead of a gather, one (Sp, Sl) per degree.

    Degree nu = (degree nu-1 @ Sp) * (x @ Sl), rows = (node, channel). A 0/1 GEMM only selects, so it is
    exact at any precision; this is the form a tile engine runs with stock matmul and multiply ops.
    """
    parent, last = parent.tolist(), last.tolist()
    deg = [0] * len(parent)
    for f, q in enumerate(parent):
        deg[f] = 1 if q < 0 else deg[q] + 1
    starts = [deg.index(nu) for nu in range(1, max(deg) + 1)] + [len(deg)]
    sels = []
    for nu in range(1, max(deg) + 1):
        lo, hi = starts[nu - 1], starts[nu]
        Sl = torch.zeros(ell, hi - lo, dtype=dtype)
        Sl[torch.tensor(last[lo:hi]), torch.arange(hi - lo)] = 1.0
        Sp = None
        if nu > 1:
            plo = starts[nu - 2]
            Sp = torch.zeros(lo - plo, hi - lo, dtype=dtype)
            Sp[torch.tensor(parent[lo:hi]) - plo, torch.arange(hi - lo)] = 1.0
        sels.append((Sp, Sl))
    return sels


def symmetric_contraction_gemm(x, element, T, sels):
    """``symmetric_contraction`` in stock tile ops only: selection GEMMs for the monomials, then per element
    a broadcast multiply by its table and a sum over monomials (2L+1 is 1 or 3, too narrow for a matmul).

    ``element`` must be constant over a run (it is: atoms keep their species), so the per-element row
    groups are fixed at setup and cost nothing per step.
    """
    n, c, ell = x.shape
    x2 = x.reshape(n * c, ell)
    feats, prev = [], None
    for Sp, Sl in sels:
        xl = x2 @ Sl
        prev = xl if Sp is None else (prev @ Sp) * xl
        feats.append(prev)
    P = torch.cat(feats, dim=1).reshape(n, c, -1)  # [n, c, F]
    out = torch.zeros(n, c, T.shape[-1], dtype=x.dtype)
    for e in element.unique().tolist():
        rows = element == e
        Pe = P[rows].unsqueeze(-1)  # [n_e, c, F, 1]
        out[rows] = (Pe * T[e].unsqueeze(0)).sum(dim=2)  # [n_e, c, 2L+1]
    return out.reshape(n, -1)


def product_tables(w, layer, n_out, elements=None):
    """Fold every output irrep of ``products.<layer>``: [(T, parent, last)] in output order.

    ``elements`` (indices into the model's element table) folds only those, in that order: a table is
    [c, 968, 2L+1] per element, so folding all 89 of MP-0b3 costs ~0.6 GB in float64 for nothing.
    """
    out = []
    for k in range(n_out):
        q = f"products.{layer}.symmetric_contractions.contractions.{k}."
        corr = sum(1 for key in w if key.startswith(q + "U_matrix_"))
        U = [w[f"{q}U_matrix_{nu}"] for nu in range(1, corr + 1)]
        W = [w[f"{q}weights.{corr - 1 - nu}"] for nu in range(1, corr)] + [w[q + "weights_max"]]
        W = W if elements is None else [v[elements] for v in W]
        _, parent, last = monomials(U[0].shape[-2], corr)
        out.append((fold_contraction(U, W), parent, last))
    return out


def conv_tp_plan(irreps1, sh_ls, target_ls):
    """Constants for the 'uvu' edge tensor product when every input block has the same multiplicity.

    C [A, B, K]: A = input components (all blocks stacked), B = SH components, K = output components of
    every instruction in reference order, each instruction's CG block placed at its (input, SH, output)
    slice and scaled by sqrt(2 l_out + 1). ins_of_k [K] names the instruction (weight column block) of each k.
    """
    from . import reference as ref
    assert len({m for m, _ in irreps1}) == 1, "uvu recast needs equal multiplicities"
    irreps_mid, ins = ref.conv_tp_irreps(irreps1, sh_ls, target_ls)
    a0 = [sum(2 * l + 1 for _, l in irreps1[:i]) for i in range(len(irreps1))]
    b0 = [sum(2 * l + 1 for l in sh_ls[:j]) for j in range(len(sh_ls))]
    A, B = a0[-1] + 2 * irreps1[-1][1] + 1, b0[-1] + 2 * sh_ls[-1] + 1
    K = sum(2 * lo + 1 for _, _, lo in ins)
    C, ins_of_k, k = torch.zeros(A, B, K, dtype=torch.float64), torch.empty(K, dtype=torch.long), 0
    for n, (i, j, lo) in enumerate(ins):
        l1, l2 = irreps1[i][1], sh_ls[j]
        C[a0[i]:a0[i] + 2 * l1 + 1, b0[j]:b0[j] + 2 * l2 + 1, k:k + 2 * lo + 1] = \
            wigner_3j(l1, l2, lo) * (2 * lo + 1) ** 0.5
        ins_of_k[k:k + 2 * lo + 1] = n
        k += 2 * lo + 1
    return C, ins_of_k, irreps_mid


def conv_tp(x1, sh, w, C, ins_of_k):
    """x1 [E, mul, A] (input blocks stacked on the last axis), sh [E, B], w [E, n_ins * mul] -> reference layout.

    Per edge the SH fold into one coupling matrix (sh @ C), the channels share it (one batched matmul),
    and the per-edge weights scale each instruction's columns.
    """
    E, mul, A = x1.shape
    M = (sh @ C.permute(1, 0, 2).reshape(C.shape[1], -1).to(sh.dtype)).reshape(E, A, -1)  # [E, A, K]
    out = torch.bmm(x1, M) * w.reshape(E, -1, mul).transpose(1, 2)[:, :, ins_of_k]  # [E, mul, K]
    blocks, k = [], 0
    for n in range(int(ins_of_k.max()) + 1):
        d = int((ins_of_k == n).sum())
        blocks.append(out[:, :, k:k + d].reshape(E, -1))
        k += d
    return torch.cat(blocks, dim=1)


# ---- Component-major ("cm") layout for the whole layer: features [rows, A, mul], A = every irrep component
# stacked (sum 2l+1), channels last. Channels (128) fill the tile lane axis, and an e3nn linear on any l is
# one matmul over the channel axis of a contiguous component slice, so no layer needs a permute except the
# symmetric contraction's (node, channel) rows.

def to_cm(x, irreps):
    """Reference layout [n, sum mul*(2l+1)] -> [n, A, mul] (equal multiplicities)."""
    from . import reference as ref
    return torch.cat([b.transpose(1, 2) for b in ref.split(x, irreps)], dim=1)


def from_cm(x, irreps):
    out, a = [], 0
    for _, l in irreps:
        d = 2 * l + 1
        out.append(x[:, a:a + d, :].transpose(1, 2).reshape(x.shape[0], -1))
        a += d
    return torch.cat(out, dim=1)


def _offsets(irreps):
    a, out = 0, []
    for _, l in irreps:
        out.append(a)
        a += 2 * l + 1
    return out


def linear_plan(irreps_in, irreps_out, weight):
    """e3nn Linear as [(in component offset, out component offset, 2l+1, W [mi, mo] / sqrt(fan))], one per path."""
    fan = [sum(mi for mi, li in irreps_in if li == l) for _, l in irreps_out]
    ai, ao, plan, k = _offsets(irreps_in), _offsets(irreps_out), [], 0
    for i, (mi, li) in enumerate(irreps_in):
        for o, (mo, lo) in enumerate(irreps_out):
            if li == lo:
                plan.append((ai[i], ao[o], 2 * li + 1, weight[k:k + mi * mo].reshape(mi, mo) / fan[o] ** 0.5))
                k += mi * mo
    assert k == weight.numel()
    return plan


def linear_cm(x, plan, a_out, mo):
    out = torch.zeros(x.shape[0], a_out, mo, dtype=x.dtype)
    for ai, ao, d, W in plan:
        out[:, ao:ao + d] += x[:, ai:ai + d] @ W
    return out


def skip_plan(irreps_in, n_el, irreps_out, weight, elements):
    """Element-selector FullyConnectedTensorProduct (one-hot attrs) as a linear_plan per present element."""
    fan = [sum(mi * n_el for mi, li in irreps_in if li == l) for _, l in irreps_out]
    ai, ao, plans, k = _offsets(irreps_in), _offsets(irreps_out), [[] for _ in elements], 0
    for i, (mi, li) in enumerate(irreps_in):
        for o, (mo, lo) in enumerate(irreps_out):
            if li == lo:
                W = weight[k:k + mi * n_el * mo].reshape(mi, n_el, mo) / fan[o] ** 0.5
                k += mi * n_el * mo
                for p, e in zip(plans, elements.tolist()):
                    p.append((ai[i], ao[o], 2 * li + 1, W[:, e]))
    assert k == weight.numel()
    return plans


def skip_cm(x, element, plans, a_out, mo):
    out = torch.zeros(x.shape[0], a_out, mo, dtype=x.dtype)
    for e, plan in enumerate(plans):
        rows = element == e
        out[rows] = linear_cm(x[rows], plan, a_out, mo)
    return out


def conv_tp_cm(x1, sh, w, C, ins_of_k):
    """x1 [E, A, mul], sh [E, B], w [E, n_ins, mul] -> [E, K, mul]: per-edge coupling (sh @ C)^T times x1,
    each output component scaled by its instruction's weight."""
    E = x1.shape[0]
    MT = (sh @ C.permute(1, 2, 0).reshape(C.shape[1], -1)).reshape(E, C.shape[2], C.shape[0])  # [E, K, A]
    return (MT @ x1) * w[:, ins_of_k, :]


def conv_tp_node_plan(C, ins_of_k):
    """Constants that move the conv_tp sum over a node's edges in front of the Clebsch-Gordan contraction.

    The per-edge product [E, K, mul] is the block's largest tensor and a device pays for it in tiny per-edge
    matmuls. With rows r = (instruction, input component of that instruction), R of them:
        Q[e, r, c]    = x1[e, a_of_r[r], c] * w[e, ins_of_r[r], c]      per edge, elementwise
        B[n, b, r, c] = sum over edges into n of sh[e, b] * Q[e, r, c]  one matmul over the neighbour axis
        msg[n, k, c]  = sum_(b, r) Cn[b * R + r, k] * B[n, b, r, c]     node sized, constant matrix
    equals summing ``conv_tp_cm`` onto the receivers. Returns (a_of_r [R], ins_of_r [R], Cn [B * R, K])."""
    A, Bn, K = C.shape
    a_of_r, ins_of_r = [], []
    for n in range(int(ins_of_k.max()) + 1):
        rows = C[:, :, ins_of_k == n].abs().sum(dim=(1, 2)).nonzero()[:, 0].tolist()
        a_of_r += rows
        ins_of_r += [n] * len(rows)
    a_of_r, ins_of_r = torch.tensor(a_of_r), torch.tensor(ins_of_r)
    Cn = C[a_of_r].permute(1, 0, 2) * (ins_of_r[:, None] == ins_of_k[None, :])  # [B, R, K]
    return a_of_r, ins_of_r, Cn.reshape(Bn * len(a_of_r), K)


def conv_tp_nodes(x1, sh, w, receiver, n, plan):
    """x1 [E, A, mul], sh [E, B], w [E, n_ins, mul] -> messages summed onto receivers [n, K, mul]."""
    a_of_r, ins_of_r, Cn = plan
    Q = x1[:, a_of_r, :] * w[:, ins_of_r, :]  # [E, R, mul]
    B = torch.zeros(n, sh.shape[1], *Q.shape[1:], dtype=Q.dtype).index_add_(0, receiver, sh[:, :, None, None] * Q[:, None])
    return torch.einsum("pk,npc->nkc", Cn, B.reshape(n, -1, Q.shape[2]))


def interaction_cm(w, layer, feats, element, elements, sh, ef, edge_index, irreps_in, residual, sh_ls=(0, 1, 2, 3)):
    """reference.density_interaction in cm layout. feats [n, A, mul]; element [n] indexes ``elements``
    (the present species). Returns (x [n, 16, mul] for the product block, skip [n, 1, mul] or None)."""
    from . import reference as ref
    p, (sender, receiver), n = f"interactions.{layer}.", edge_index, feats.shape[0]
    hidden = [(128, l) for l in range(4)]
    n_el = len(w["atomic_numbers"])
    sc_irreps = [(128, 0)] if residual else hidden
    sk = skip_plan(irreps_in if residual else hidden, n_el, sc_irreps, w[p + "skip_tp.weight"], elements)
    a_in = sum(2 * l + 1 for _, l in irreps_in)
    sc = skip_cm(feats, element, sk, 1, 128) if residual else None
    up = linear_cm(feats, linear_plan(irreps_in, irreps_in, w[p + "linear_up.weight"]), a_in, 128)
    tp_w = ref.fc_net(ef, ref._layers(w, p + "conv_tp_weights")).reshape(ef.shape[0], -1, 128)
    dens = torch.tanh(ref.fc_net(ef, ref._layers(w, p + "density_fn")) ** 2)
    density = torch.zeros(n, 1, dtype=feats.dtype).index_add_(0, receiver, dens)
    C, ins_of_k, mid = conv_tp_plan(irreps_in, list(sh_ls), [l for _, l in hidden])
    m = conv_tp_cm(up[sender], sh, tp_w, C, ins_of_k)
    msg = torch.zeros(n, *m.shape[1:], dtype=m.dtype).index_add_(0, receiver, m)
    msg = linear_cm(msg, linear_plan(mid, hidden, w[p + "linear.weight"]), 16, 128) / (density + 1)[:, :, None]
    if not residual:
        msg = skip_cm(msg, element, sk, 16, 128)
    return msg, sc


def plan_matrix(plan, a_in, a_out):
    """A linear_plan as one [a_in * mi, a_out * mo] matrix on the flattened cm layout (column a * mul + u):
    each path is the block kron(I_{2l+1}, W). Block-sparse, so it spends flops for one plain GEMM."""
    mi, mo = plan[0][3].shape
    M = torch.zeros(a_in * mi, a_out * mo, dtype=torch.float64)
    for ai, ao, d, W in plan:
        for c in range(d):
            M[(ai + c) * mi:(ai + c + 1) * mi, (ao + c) * mo:(ao + c + 1) * mo] += W
    return M


# ---- Whole model in the cm layout, and its hand-written backward (the device has no autograd).

def product_cm(w, layer, x, element, elements, irreps_out, sc=None):
    """products.<layer> + readouts.<layer> in cm: x [n, 16, 128] -> (features [n, a, 128], node energy [n])."""
    from . import reference as ref
    a = sum(2 * l + 1 for _, l in irreps_out)
    tables = product_tables(w, layer, len(irreps_out), elements=elements)
    _, parent, last = tables[0]
    T = torch.cat([t for t, _, _ in tables], dim=-1)  # [E, c, F, a]
    y = symmetric_contraction_gemm(x.transpose(1, 2), element, T, monomial_selectors(parent, last, x.shape[1]))
    y = y.reshape(x.shape[0], 128, a).transpose(1, 2)  # [n, a, 128]
    y = linear_cm(y, linear_plan(irreps_out, irreps_out, w[f"products.{layer}.linear.weight"]), a, 128)
    if sc is not None:
        y = y + sc
    return y, ref.readout(w, layer, from_cm(y, irreps_out), irreps_out).squeeze(-1)


def energy_cm(w, atomic_numbers, pos, edge_index, shifts):
    """reference.energy through the cm-layout blocks (what the card runs), atoms in caller order."""
    from . import reference as ref
    el = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long()).argmax(1)
    elements = el.unique()
    local = torch.searchsorted(elements, el)
    attrs = ref.node_attrs(atomic_numbers, w["atomic_numbers"].long())
    vec, r = ref.edge_vectors(pos, edge_index, shifts)
    sh, ef = ref.spherical_harmonics(vec), ref.radial_embedding(r, atomic_numbers, edge_index, w)
    node_e = ref.zbl(r, atomic_numbers, edge_index, w, len(pos))
    feats, irreps = ref.node_embedding(attrs, w)[:, None, :], [(128, 0)]
    for layer, (out_irreps, residual) in enumerate([([(128, 0), (128, 1)], False), ([(128, 0)], True)]):
        x, sc = interaction_cm(w, layer, feats, local, elements, sh, ef, edge_index, irreps, residual)
        feats, e = product_cm(w, layer, x, local, elements, out_irreps, sc)
        node_e, irreps = node_e + e, out_irreps
    node_e = w["scale_shift.scale"] * node_e + w["scale_shift.shift"]
    return (attrs @ w["atomic_energies_fn.atomic_energies"]).sum() + node_e.sum()
