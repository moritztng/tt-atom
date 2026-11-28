"""The ops of tt_atom.tensornet.layer on one chip with stock ttnn ops. Node tensors [3, 3, N, U], edge tensors
[S, W], all tile layout. Component mixing is done with slices and concats over whole [N, U] planes, so nothing
smaller than a tile ever sits in the tile plane."""
import torch

from .. import device as D
from .. import scatter as SC


class TtOps:
    def __init__(self, dev, n, d, units, dtype=None, ttnn=None):
        if ttnn is None:
            import ttnn

        self.tt, self.dev, self.n, self.d, self.u = ttnn, dev, n, d, units
        self.dtype = dtype or ttnn.bfloat16
        self.cfg = D.compute_kernel_config() if ttnn.__name__ == "ttnn" else None
        self.add, self.sub, self.mul = ttnn.add, ttnn.subtract, ttnn.multiply
        self.silu, self.recip, self.rsqrt, self.sigmoid = ttnn.silu, ttnn.reciprocal, ttnn.rsqrt, ttnn.sigmoid

    def index(self, idx):
        return self.tt.from_torch(idx.to(torch.int32), dtype=self.tt.uint32, layout=self.tt.ROW_MAJOR_LAYOUT, device=self.dev)

    def up(self, t):
        return self.tt.from_torch(t.float().contiguous(), dtype=self.dtype, layout=self.tt.TILE_LAYOUT, device=self.dev)

    def weight(self, t):
        """A matmul operand. At float32 it goes up in pieces that `linear` multiplies one by one and adds, because only
        the card's float32 multiplies and adds are exact: a matmul's accumulation over more than one tile of the inner
        dimension truncates (1e-4, one sign, an energy offset), see benchmarks/probe_mm_bits.py. A matrix with a few
        input rows (the radial projections, which feed neighbour sums that cancel ~100x on a lattice) goes up row by
        row; a wide one in 32-row blocks, each pre-split for scatter.mm."""
        if not self.cfg or self.dtype != self.tt.float32:
            return self.up(t)
        k = t.shape[-2]
        if k <= 8 and t.dim() == 2:
            return [self.up(row[None]) for row in t]
        step = 32 if k % 32 == 0 else k
        return [SC.const(t[..., i:i + step, :].float(), self.dev, self.dtype) for i in range(0, k, step)]

    def cast(self, x):
        return x if x.dtype == self.dtype else self.tt.typecast(x, self.dtype)

    def cat(self, xs):
        return self.tt.concat(xs, dim=-1)

    def split(self, x, k):
        shape, w = list(x.shape), x.shape[-1] // k
        return [self.tt.slice(x, [0] * (len(shape) - 1) + [c * w], shape[:-1] + [(c + 1) * w]) for c in range(k)]

    def usum(self, x):
        return SC.linear32(self.tt, lambda p: self.tt.sum(p, dim=-1, keepdim=True), x)

    def scale(self, x, s):
        return self.tt.multiply(x, float(s))

    def silu_grad(self, x):
        s = self.tt.sigmoid(x)
        return self.tt.multiply(s, self.tt.add(self.tt.multiply(x, self.tt.add(self.tt.neg(s), 1.0)), 1.0))

    def linear(self, x, w, b=None):
        if isinstance(w, list):
            y = self._sum([SC.mm(c, p, self.cfg, self.tt) if isinstance(p, tuple) else self.tt.multiply(c, p)
                           for c, p in zip(self.split(x, len(w)), w)])
        else:
            y = SC.mm(x, w, self.cfg, self.tt)
        return y if b is None else self.tt.add(y, b)

    def comp(self, x, i, j):                       # [3, 3, N, U] -> component (i, j) as [1, 1, N, U]
        return self.tt.slice(x, [i, j, 0, 0], [i + 1, j + 1, self.n, self.u])

    def stack(self, planes):                       # nine [1, 1, N, U], row-major over (i, j) -> [3, 3, N, U]
        return self.tt.concat([self.tt.concat(planes[3 * i:3 * i + 3], dim=1) for i in range(3)], dim=0)

    def t(self, x):
        return self.stack([self.comp(x, j, i) for i in range(3) for j in range(3)])

    def csum(self, x):
        out = None
        for i in range(3):
            for j in range(3):
                c = self.comp(x, i, j)
                out = c if out is None else self.tt.add(out, c)
        return out

    def mat(self, a, b):
        rows = []
        for i in range(3):
            acc = None
            for j in range(3):
                term = self.tt.multiply(self.tt.slice(b, [j, 0, 0, 0], [j + 1, 3, self.n, self.u]), self.comp(a, i, j))
                acc = term if acc is None else self.tt.add(acc, term)
            rows.append(acc)
        return self.tt.concat(rows, dim=0)

    def rows(self, parts):
        flat = [self.tt.reshape(self.comp(p, i, j), (self.n, self.u)) for p in parts for i in range(3) for j in range(3)]
        return self.tt.concat(flat, dim=1)

    def unrows(self, r, k):
        u, n = self.u, self.n
        plane = lambda c: self.tt.reshape(self.tt.slice(r, [0, c * u], [n, (c + 1) * u]), (1, 1, n, u))
        return [self.stack([plane(9 * p + c) for c in range(9)]) for p in range(k)]

    def gather(self, idx, r):
        tt = self.tt
        return tt.to_layout(SC.embed_rows(tt, idx, tt.to_layout(r, tt.ROW_MAJOR_LAYOUT)), tt.TILE_LAYOUT)

    def _cols(self, x, k, width):
        return [self.tt.slice(x, [0, c * width], [x.shape[0], (c + 1) * width]) for c in range(k)]

    def _sum(self, xs):
        out = xs[0]
        for x in xs[1:]:
            out = self.tt.add(out, x)
        return out

    def expand(self, f):
        return self.tt.concat([p for p in self._cols(f, 3, self.u) for _ in range(9)], dim=1)

    def fold9(self, x):
        return self.tt.concat([self._sum(self._cols(p, 9, self.u)) for p in self._cols(x, 3, 9 * self.u)], dim=1)

    def tile3(self, m):
        return self.tt.concat([m, m, m], dim=1)

    def fold3(self, x):
        return self._sum(self._cols(x, 3, 9 * self.u))

    def slot_sum(self, x, n):
        """Sum of each node's slots: the D row blocks of N added in float32. On a near-symmetric lattice the vector
        and tensor parts cancel ~100x over the neighbours; ttnn.sum is good to ~1e-4 of the summands (bfloat16: 4e-3),
        which is percent of such a result, while float32 adds are exact to rounding."""
        tt, w, acc = self.tt, x.shape[-1], None
        for k in range(self.d):
            s = tt.slice(x, [k * n, 0], [(k + 1) * n, w])
            s = s if s.dtype == tt.float32 else tt.typecast(s, tt.float32)
            acc = s if acc is None else tt.add(acc, s)
        return self.cast(acc)

    def slot_spread(self, x, d):
        return self.tt.concat([x] * d, dim=0)

    def scatter(self, x, table, n):
        width = x.shape[-1]
        out = SC.segment_sum(self.tt, x, table, table.shape[-1] // n, n, width)
        return self.tt.to_layout(self.tt.reshape(out, (n, width)), self.tt.TILE_LAYOUT)


def to_device(o, w):
    """Weights from layer.weights -> device tensors (node weights as [1, 1, U, U], biases as [1, out])."""
    node = lambda m: o.weight(m[None, None])
    return {"edge": [(o.weight(wt), o.up(b[None])) for wt, b in w["edge"]], "edge_t": [o.weight(m) for m in w["edge_t"]],
            "node": [node(m) for m in w["node"]], "node_t": [node(m) for m in w["node_t"]]}


def model_to_device(o, ol, w):
    """Weights from model.weights -> device tensors: embedding and readout through `o`, layers through `ol`.
    Matrices that meet [.., .., N, U] tensors go up as [1, 1, in, out], their biases as [1, 1, 1, out]."""
    m4, b4, t = (lambda m: o.weight(m[None, None])), (lambda b: o.up(b[None, None, None])), o.weight
    e, r = w["embed"], w["readout"]
    chain = lambda ws: [(m4(wt), b4(b)) for wt, b in ws]
    return {"embed": {"proj": [(t(wt), o.up(b[None])) for wt, b in e["proj"]], "proj_t": [t(m) for m in e["proj_t"]],
                      "scalar": chain(e["scalar"]), "scalar_t": [m4(m) for m in e["scalar_t"]],
                      "node": [m4(m) for m in e["node"]], "node_t": [m4(m) for m in e["node_t"]], "ln": tuple(b4(v) for v in e["ln"])},
            "layers": [to_device(ol, wl) for wl in w["layers"]],
            "readout": {"ln": tuple(b4(v) for v in r["ln"]), "value": chain(r["value"]), "gate": chain(r["gate"]),
                        "value_t": [m4(m) for m in r["value_t"]], "gate_t": [m4(m) for m in r["gate_t"]]}}


def runner(dev, w, units, layer_dtype=None, ttnn=None):
    """The network on one chip as potential.energy_forces's `run`: embedding and readout float32, the interaction
    layers at `layer_dtype` (float32 unless given). Weights go up once; the ops of a graph shape (N, D) are kept."""
    if ttnn is None:
        import ttnn
    from . import model as M

    shapes, wd = {}, []
    down = lambda t: ttnn.to_torch(t).double()

    def run(z, rbf, cut, geo, send, table, n, d):
        if (n, d) not in shapes:
            o = TtOps(dev, n, d, units, dtype=ttnn.float32, ttnn=ttnn)
            ol = o if layer_dtype in (None, ttnn.float32) else TtOps(dev, n, d, units, dtype=layer_dtype, ttnn=ttnn)
            eye = torch.eye(3).reshape(3, 3, 1, 1).expand(3, 3, n, units)
            shapes[n, d] = o, ol, o.up(eye), ol.up(eye), o.up(torch.ones(1, 1, n, 1))
            wd or wd.append(model_to_device(o, ol, w))
        o, ol, eye, eye_l, ones = shapes[n, d]
        edge = o.up(rbf), o.up(cut)
        # the 27 geometry scalars go up as one tensor and are cut into columns on the card: a [S, 1] upload is padded
        # to a full tile of columns, so 27 of them move 32x the data
        e, tape = M.forward(o, wd[0], o.up(z), edge, o._cols(o.up(geo), geo.shape[-1], 1), o.index(send), n, eye,
                            ol=ol, edge_l=edge if ol is o else (ol.up(rbf), ol.up(cut)), eye_l=eye_l)
        grbf, gcut, ggeo = M.backward(o, wd[0], tape, ones, o.index(table), n, d, eye, ol=ol, eye_l=eye_l)
        return down(e).reshape(-1), sum(down(g) for g in grbf), sum(down(g) for g in gcut), down(ggeo)
    return run
