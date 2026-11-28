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
        self.silu, self.recip = ttnn.silu, ttnn.reciprocal
        self.recv = self.index(torch.arange(n).repeat_interleave(d))

    def index(self, idx):
        return self.tt.from_torch(idx.to(torch.int32), dtype=self.tt.uint32, layout=self.tt.ROW_MAJOR_LAYOUT, device=self.dev)

    def up(self, t):
        return self.tt.from_torch(t.float().contiguous(), dtype=self.dtype, layout=self.tt.TILE_LAYOUT, device=self.dev)

    def scale(self, x, s):
        return self.tt.multiply(x, float(s))

    def silu_grad(self, x):
        s = self.tt.sigmoid(x)
        return self.tt.multiply(s, self.tt.add(self.tt.multiply(x, self.tt.add(self.tt.neg(s), 1.0)), 1.0))

    def linear(self, x, w, b=None):
        y = self.tt.matmul(x, w, compute_kernel_config=self.cfg)
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
        tt = self.tt
        g = tt.reshape(tt.to_layout(x, tt.ROW_MAJOR_LAYOUT), (n, self.d, x.shape[-1]))
        return tt.to_layout(tt.reshape(tt.sum(g, dim=1), (n, x.shape[-1])), tt.TILE_LAYOUT)

    def slot_spread(self, x, d):
        return self.gather(self.recv, x)

    def scatter(self, x, table, n):
        width = x.shape[-1]
        out = SC.segment_sum(self.tt, x, table, table.shape[-1] // n, n, width)
        return self.tt.to_layout(self.tt.reshape(out, (n, width)), self.tt.TILE_LAYOUT)


def to_device(o, w):
    """Weights from layer.weights -> device tensors (node weights as [1, 1, U, U], biases as [1, out])."""
    node = lambda m: o.up(m[None, None])
    return {"edge": [(o.up(wt), o.up(b[None])) for wt, b in w["edge"]], "edge_t": [o.up(m) for m in w["edge_t"]],
            "node": [node(m) for m in w["node"]], "node_t": [node(m) for m in w["node_t"]]}
