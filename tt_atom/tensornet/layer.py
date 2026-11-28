"""One TensorNet interaction layer, forward and reverse, written once over a small set of tensor ops.

Node tensors are held as [3, 3, N, U]: the 3x3 Cartesian components lead, so every op works on whole [N, U]
planes. Edges live in the receiver's padded slots, S = N * D rows, slot (i, k) the k-th edge into node i; pad
slots carry cutoff 0, which zeroes their message forward and their gradient in reverse.

`TorchOps` is the float64 reference of the ops; the device backend implements the same names with ttnn.
"""
import numpy as np
import torch

from ..scatter import build_gather


class TorchOps:
    add, sub, mul = staticmethod(torch.add), staticmethod(torch.sub), staticmethod(torch.mul)
    silu = staticmethod(torch.nn.functional.silu)

    @staticmethod
    def scale(x, s):
        return x * s

    @staticmethod
    def recip(x):
        return 1.0 / x

    @staticmethod
    def silu_grad(x):
        s = torch.sigmoid(x)
        return s * (1 + x * (1 - s))

    sigmoid = staticmethod(torch.sigmoid)

    @staticmethod
    def cast(x):                                   # to this backend's dtype
        return x

    @staticmethod
    def rsqrt(x):
        return x.rsqrt()

    @staticmethod
    def cat(xs):                                   # along the last dim
        return torch.cat(xs, -1)

    @staticmethod
    def split(x, k):                               # last dim into k equal blocks
        return list(x.chunk(k, -1))

    @staticmethod
    def usum(x):                                   # sum over the last dim, kept
        return x.sum(-1, keepdim=True)

    @staticmethod
    def linear(x, w, b=None):                      # w [in, out]
        y = x @ w
        return y if b is None else y + b

    @staticmethod
    def t(x):                                      # transpose of the 3x3 components
        return x.transpose(0, 1)

    @staticmethod
    def csum(x):                                   # sum over the 3x3 components, kept as [1, 1, N, U]
        return x.sum((0, 1), keepdim=True)

    @staticmethod
    def mat(a, b):                                 # 3x3 matrix product per node and unit
        return torch.einsum("ijnu,jknu->iknu", a, b)

    @staticmethod
    def rows(parts):                               # three [3, 3, N, U] -> [N, 27 U]
        return torch.cat([p.permute(2, 0, 1, 3).reshape(p.shape[2], -1) for p in parts], -1)

    @staticmethod
    def unrows(r, k):                              # [N, k 9 U] -> k tensors [3, 3, N, U]
        n = r.shape[0]
        return list(r.reshape(n, k, 3, 3, -1).permute(1, 2, 3, 0, 4))

    @staticmethod
    def gather(idx, r):
        return r[idx]

    @staticmethod
    def expand(f):                                 # [S, 3 U] -> [S, 27 U], each part's U block under its 9 components
        s = f.shape[0]
        return f.reshape(s, 3, 1, -1).expand(s, 3, 9, f.shape[1] // 3).reshape(s, -1)

    @staticmethod
    def fold9(x):                                  # [S, 27 U] -> [S, 3 U], sum over the 9 components (expand reversed)
        s = x.shape[0]
        return x.reshape(s, 3, 9, -1).sum(2).reshape(s, -1)

    @staticmethod
    def tile3(m):                                  # [S, 9 U] -> [S, 27 U], one copy per part
        return torch.cat([m, m, m], -1)

    @staticmethod
    def fold3(x):                                  # [S, 27 U] -> [S, 9 U], sum over the three parts (tile3 reversed)
        s = x.shape[0]
        return x.reshape(s, 3, -1).sum(1)

    @staticmethod
    def slot_sum(x, n):                            # [N D, W] -> [N, W]
        return x.reshape(n, -1, x.shape[-1]).sum(1)

    @staticmethod
    def slot_spread(x, d):                         # [N, W] -> [N D, W]
        return x.repeat_interleave(d, 0)

    @staticmethod
    def scatter(x, table, n):                      # rows of x summed onto nodes; table [N D'] of row ids, len(x) = none
        pad = torch.cat([x, torch.zeros_like(x[:1])])
        return pad[table].reshape(n, -1, x.shape[-1]).sum(1)


def weights(layer, dtype=torch.float64):
    """matgl TensorNetInteraction -> transposed weights. The last edge linear is reordered from matgl's
    interleaved (unit, part) columns to three blocks of U, one per part (scalar, skew, traceless)."""
    u = layer.units
    order = torch.arange(3 * u).reshape(u, 3).T.reshape(-1)
    ws = [(l.weight.detach().to(dtype).T.contiguous(), l.bias.detach().to(dtype)) for l in layer.linears_scalar]
    ws[2] = (ws[2][0][:, order].contiguous(), ws[2][1][order].contiguous())
    node = [l.weight.detach().to(dtype).T.contiguous() for l in layer.linears_tensor]
    return {"edge": ws, "node": node, "edge_t": [wt.T.contiguous() for wt, _ in ws],
            "node_t": [wt.T.contiguous() for wt in node]}


def slots(edge_index, n, min_width=0, own=False):
    """Edge list (receiver row 0, sender row 1) -> (edge [S] edge id of each receiver slot, E in pad slots;
    send [S] sender node, 0 in pad slots; table [N D'] slot ids grouped by sender, S = none; D).
    own=True is the message of weights trained before matgl 3 (v2025.1): row 1 receives, and the features an
    edge carries are the receiver's own. The loader sets it."""
    edge_index, e = edge_index.long(), edge_index.shape[1]
    recv = sender = edge_index[1]
    if not own:
        recv, sender = edge_index
    edge, d = build_gather(recv, n, e, min_width=min_width)
    edge = torch.from_numpy(edge.astype(np.int64))
    real = (edge < e).nonzero()[:, 0]
    send = torch.zeros(n * d, dtype=torch.long)
    send[real] = sender[edge[real]]
    table, _ = build_gather(send[real], n, len(real), sentinel=-1, min_width=min_width)
    table = torch.from_numpy(table.astype(np.int64))
    table = torch.where(table < 0, torch.tensor(n * d), real[table.clamp(min=0)])
    return edge, send, table, d


EPS = 1e-5  # torch.nn.LayerNorm default, which matgl uses


def norm(o, x, g, b):
    """Layer norm over the last dim. Returns (y, tape)."""
    u = x.shape[-1]
    dev = o.sub(x, o.scale(o.usum(x), 1 / u))
    r = o.rsqrt(o.add(o.scale(o.usum(o.mul(dev, dev)), 1 / u), EPS))
    return o.add(o.mul(o.mul(dev, r), g), b), (dev, r, u)


def norm_t(o, g_y, g, tape):
    dev, r, u = tape
    g_y = o.mul(g_y, g)
    g_dev = o.sub(o.mul(g_y, r), o.mul(dev, o.scale(o.mul(o.mul(r, o.mul(r, r)), o.usum(o.mul(g_y, dev))), 1 / u)))
    return o.sub(g_dev, o.scale(o.usum(g_dev), 1 / u))


def mlp(o, h, ws):
    """Linears with SiLU between them, none after the last. Returns (out, pre-activations)."""
    pre = []
    for k, (wt, b) in enumerate(ws):
        pre.append(o.linear(h if k == 0 else o.silu(pre[-1]), wt, b))
    return pre[-1], pre


def mlp_t(o, g, ws_t, pre):
    for k in range(len(ws_t) - 1, -1, -1):
        g = o.linear(g if k == len(ws_t) - 1 else o.mul(g, o.silu_grad(pre[k])), ws_t[k])
    return g


def decompose(o, x, eye):
    """x -> scalar, skew and symmetric traceless parts."""
    xt = o.t(x)
    i = o.mul(o.scale(o.csum(o.mul(x, eye)), 1 / 3), eye)
    return i, o.scale(o.sub(x, xt), 0.5), o.sub(o.scale(o.add(x, xt), 0.5), i)


def decompose_t(o, gi, ga, gs, eye):
    """Reverse of decompose: gradients of the three parts -> gradient of x."""
    tr = o.mul(o.scale(o.csum(o.mul(o.sub(gi, gs), eye)), 1 / 3), eye)
    return o.add(o.add(o.scale(o.sub(ga, o.t(ga)), 0.5), o.scale(o.add(gs, o.t(gs)), 0.5)), tr)


def forward(o, w, x, rbf, cut, send, n, eye, one):
    """x [3, 3, N, U], rbf [S, R], cut [S, 1], send [S] sender node of each slot. Returns (out, tape)."""
    t = {}
    h, t["pre"] = rbf, []
    for wt, b in w["edge"]:
        t["pre"].append(o.linear(h, wt, b))
        h = o.silu(t["pre"][-1])
    t["h"], t["cut"] = h, cut
    f = o.expand(o.mul(h, cut))                                          # [S, 27 U]

    t["x"], t["rn"] = x, o.recip(o.add(o.csum(o.mul(x, x)), one))
    xn = o.mul(x, t["rn"])
    parts = [o.linear(p, wt) for p, wt in zip(decompose(o, xn, eye), w["node"][:3])]
    y = o.add(o.add(parts[0], parts[1]), parts[2])
    g = o.gather(send, o.rows(parts))                                    # [S, 27 U]
    msg = o.unrows(o.slot_sum(o.fold3(o.mul(g, f)), n), 1)[0]
    c = o.add(o.mat(msg, y), o.mat(y, msg))
    rc = o.recip(o.add(o.csum(o.mul(c, c)), one))
    cn = o.mul(c, rc)
    dparts = [o.linear(p, wt) for p, wt in zip(decompose(o, cn, eye), w["node"][3:])]
    dx = o.add(o.add(dparts[0], dparts[1]), dparts[2])
    t.update(xn=xn, y=y, g=g, f=f, msg=msg, c=c, rc=rc, cn=cn, dx=dx)
    return o.add(o.add(xn, dx), o.mat(dx, dx)), t


def _unnorm(o, g, v, vn, r):
    """Reverse of vn = v * r, r = 1 / (sum v^2 + 1): gradient of vn -> gradient of v."""
    return o.sub(o.mul(g, r), o.mul(v, o.scale(o.mul(o.csum(o.mul(g, vn)), r), 2.0)))


def backward(o, w, t, g_out, send_table, n, d, eye):
    """Gradient of the output -> gradients of (x, rbf, cut); the last as [S, 3 U], to be summed over its row."""
    dx, y, msg = t["dx"], t["y"], t["msg"]
    g_dx = o.add(g_out, o.add(o.mat(g_out, o.t(dx)), o.mat(o.t(dx), g_out)))
    g_cn = decompose_t(o, *[o.linear(g_dx, wt_t) for wt_t in w["node_t"][3:]], eye)
    g_c = _unnorm(o, g_cn, t["c"], t["cn"], t["rc"])
    g_msg = o.add(o.mat(g_c, o.t(y)), o.mat(o.t(y), g_c))
    g_y = o.add(o.mat(o.t(msg), g_c), o.mat(g_c, o.t(msg)))

    g_m = o.tile3(o.slot_spread(o.rows([g_msg]), d))                     # [S, 27 U]
    g_f = o.fold9(o.mul(t["g"], g_m))
    g_parts = o.unrows(o.scatter(o.mul(t["f"], g_m), send_table, n), 3)
    g_xn = decompose_t(o, *[o.linear(o.add(gp, g_y), wt_t) for gp, wt_t in zip(g_parts, w["node_t"][:3])], eye)
    g_x = _unnorm(o, o.add(g_xn, g_out), t["x"], t["xn"], t["rn"])

    g_cut = o.mul(g_f, t["h"])
    g_h = o.mul(g_f, t["cut"])
    for (wt_t, pre) in zip(w["edge_t"][::-1], t["pre"][::-1]):
        g_h = o.linear(o.mul(g_h, o.silu_grad(pre)), wt_t)
    return g_x, g_h, g_cut
