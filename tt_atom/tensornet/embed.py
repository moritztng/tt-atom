"""The TensorNet embedding block, forward and reverse, over the ops of tt_atom.tensornet.layer.

Edges sit in the receiver's padded slots like in the interaction layer. What depends on the atoms' species only
(the pair embedding z) and the geometry of each edge (27 scalars: the components of the identity, of the skew
tensor and of the traceless tensor of its unit vector) come from the host; the block returns the gradient of
those 27 scalars, of the radial basis and of the cutoff, and the host finishes the chain to positions.
"""
import torch

from . import layer as L


def weights(emb, dtype=torch.float64):
    """matgl TensorEmbedding -> transposed weights; the last scalar linear reordered to one block of U per part."""
    u = emb.units
    order = torch.arange(3 * u).reshape(u, 3).T.reshape(-1)
    lin = lambda l: (l.weight.detach().to(dtype).T.contiguous(), l.bias.detach().to(dtype))
    proj = [lin(l) for l in (emb.distance_proj1, emb.distance_proj2, emb.distance_proj3)]
    sc = [lin(l) for l in emb.linears_scalar]
    sc[1] = (sc[1][0][:, order].contiguous(), sc[1][1][order].contiguous())
    node = [l.weight.detach().to(dtype).T.contiguous() for l in emb.linears_tensor]
    return {"proj": proj, "proj_t": [wt.T.contiguous() for wt, _ in proj], "scalar": sc,
            "scalar_t": [wt.T.contiguous() for wt, _ in sc], "node": node, "node_t": [wt.T.contiguous() for wt in node],
            "ln": (emb.init_norm.weight.detach().to(dtype), emb.init_norm.bias.detach().to(dtype))}


def pair(emb, z, edge_index, dtype=torch.float64):
    """Species part of every edge's message, [E, U]. Constant in the positions."""
    x = emb.emb.weight.detach().to(dtype)[z]
    return torch.cat([x[edge_index[0]], x[edge_index[1]]], -1) @ emb.emb2.weight.detach().to(dtype).T + emb.emb2.bias.detach().to(dtype)


def geometry(vec):
    """Edge vectors [E, 3] -> [E, 27]: identity, skew and traceless components of the unit vector, row-major."""
    v = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-6)
    eye = torch.eye(3, dtype=vec.dtype)
    zero = torch.zeros_like(v[:, 0])
    skew = torch.stack([zero, -v[:, 2], v[:, 1], v[:, 2], zero, -v[:, 0], -v[:, 1], v[:, 0], zero], 1)
    sym = (v[:, :, None] * v[:, None, :] - (v * v).sum(1)[:, None, None] / 3 * eye).reshape(-1, 9)
    return torch.cat([eye.reshape(1, 9).expand(len(v), 9), skew, sym], 1)


def forward(o, w, z, rbf, cut, geo, n):
    """z [S, U], rbf [S, R], cut [S, 1], geo 27 columns [S, 1]. Returns (X [3, 3, N, U], tape)."""
    lin = [o.linear(rbf, wt, b) for wt, b in w["proj"]]
    zw = [o.mul(o.mul(l, cut), z) for l in lin]                               # [S, U] per part
    edge = o.cat([o.mul(zw[p], geo[9 * p + c]) for p in range(3) for c in range(9)])
    parts = o.unrows(o.slot_sum(edge, n), 3)
    x0 = o.add(o.add(parts[0], parts[1]), parts[2])

    h, ln = L.norm(o, o.csum(o.mul(x0, x0)), *w["ln"])
    h, pre = L.mlp(o, h, w["scalar"])
    h = o.silu(h)
    f = o.split(h, 3)                                                         # [1, 1, N, U] per part
    lp = [o.linear(p, wt) for p, wt in zip(parts, w["node"])]
    out = o.add(o.add(o.mul(lp[0], f[0]), o.mul(lp[1], f[1])), o.mul(lp[2], f[2]))
    return out, dict(lin=lin, zw=zw, z=z, cut=cut, geo=geo, x0=x0, ln=ln, pre=pre, f=f, lp=lp)


def backward(o, w, t, g_out, d):
    """Gradient of X -> gradients of (rbf [S, R], cut [S, 1], geo [S, 27])."""
    g_h = o.cat([o.csum(o.mul(g_out, p)) for p in t["lp"]])
    g_h = L.mlp_t(o, o.mul(g_h, o.silu_grad(t["pre"][-1])), w["scalar_t"], t["pre"])
    g_x0 = o.mul(o.scale(t["x0"], 2.0), L.norm_t(o, g_h, w["ln"][0], t["ln"]))
    g_parts = [o.add(o.linear(o.mul(g_out, f), wt_t), g_x0) for f, wt_t in zip(t["f"], w["node_t"])]
    blocks = o.split(o.slot_spread(o.rows(g_parts), d), 27)                   # [S, U] per part and component

    g_geo, g_rbf, g_cut = [], None, None
    for p in range(3):
        g_zw = None
        for c in range(9):
            blk = blocks[9 * p + c]
            g_geo.append(o.usum(o.mul(blk, t["zw"][p])))
            term = o.mul(blk, t["geo"][9 * p + c])
            g_zw = term if g_zw is None else o.add(g_zw, term)
        g_w = o.mul(g_zw, t["z"])
        gc, gr = o.usum(o.mul(g_w, t["lin"][p])), o.linear(o.mul(g_w, t["cut"]), w["proj_t"][p])
        g_cut, g_rbf = (gc, gr) if g_cut is None else (o.add(g_cut, gc), o.add(g_rbf, gr))
    return g_rbf, g_cut, o.cat(g_geo)
