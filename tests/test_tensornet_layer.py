"""The TensorNet interaction layer of tt_atom.tensornet.layer against matgl in float64: output and reverse pass."""
import numpy as np
import pytest
import torch

matgl = pytest.importorskip("matgl")
from tt_atom.tensornet import layer as L  # noqa: E402
from tt_atom.tensornet.tt_ops import TtOps, to_device  # noqa: E402

MODEL = "materialyze/TensorNet-PES-MatPES-PBE-2025.2"


def captured(n_cells=2):
    """Inputs of every interaction layer of the float64 matgl model on jittered diamond Si."""
    from ase.build import bulk
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(MODEL).double()
    pot.calc_stresses = False
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (n_cells, n_cells, n_cells)
    atoms.positions += np.random.default_rng(0).normal(0, 0.05, atoms.positions.shape)
    calls = []
    hooks = [m.register_forward_hook(lambda mod, args, kw, out: calls.append((mod, args, kw)), with_kwargs=True)
             for m in pot.model.layers]
    atoms.calc = PESCalculator(pot)
    atoms.get_potential_energy()
    for h in hooks:
        h.remove()
    return calls


def layer_args(args, kw):
    names = ("edge_index", "edge_weight", "edge_attr", "X")
    a = dict(zip(names, args), **kw)
    return [a[k].detach() for k in names]


class FakeTtnn:
    """The ttnn calls TtOps makes, on torch float64: checks the slicing and concat bookkeeping without a card."""
    __name__ = "fake"
    bfloat16, float32, uint32, TILE_LAYOUT, ROW_MAJOR_LAYOUT = torch.float64, torch.float32, None, None, None
    add, subtract, multiply, matmul = torch.add, torch.sub, torch.mul, lambda a, b, **k: a @ b
    silu, sigmoid, reciprocal, neg, reshape = torch.nn.functional.silu, torch.sigmoid, torch.reciprocal, torch.neg, torch.reshape
    concat, sum = (lambda xs, dim: torch.cat(xs, dim)), (lambda x, dim: x.sum(dim))
    to_layout = staticmethod(lambda x, layout: x)
    embedding = staticmethod(lambda idx, table, **k: table[idx])
    from_torch = staticmethod(lambda t, **k: t.long() if not t.is_floating_point() else t.double())
    slice = staticmethod(lambda x, b, e: x[tuple(slice(i, j) for i, j in zip(b, e))])


@pytest.mark.parametrize("backend", ["torch", "tt"])
@pytest.mark.parametrize("which", [0, 1])
def test_layer_matches_matgl(which, backend):
    mod, args, kw = captured()[which]
    ei, r, rbf, x = layer_args(args, kw)
    x = x.clone().requires_grad_(True)
    rbf = rbf.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    want = mod(ei, r, rbf, x)                                           # [N, U, 3, 3]
    g_out = torch.randn(want.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    wx, wrbf, wr = torch.autograd.grad((want * g_out).sum(), (x, rbf, r))

    n = x.shape[0]
    edge, send, table, d = L.slots(ei, n, min_width=32)
    o = L.TorchOps if backend == "torch" else TtOps(None, n, d, mod.units, ttnn=FakeTtnn)
    pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge]
    cut = torch.where(r <= mod.cutoff, 0.5 * (torch.cos(torch.pi * r / mod.cutoff) + 1), 0.0)
    eye, one = torch.eye(3, dtype=torch.float64).reshape(3, 3, 1, 1).expand(3, 3, n, x.shape[1]), 1.0
    w = L.weights(mod) if backend == "torch" else to_device(o, L.weights(mod))
    to_c = lambda t: t.detach().permute(2, 3, 0, 1)                     # [N, U, 3, 3] -> [3, 3, N, U]
    got, tape = L.forward(o, w, to_c(x), pad(rbf.detach()), pad(cut.detach()[:, None]), send, n, eye, one)
    assert torch.allclose(got.permute(2, 3, 0, 1), want, rtol=1e-10, atol=1e-12)

    gx, grbf, gcut = L.backward(o, w, tape, to_c(g_out), table, n, d, eye)
    real = edge < ei.shape[1]
    unpad = lambda t: torch.zeros(ei.shape[1], *t.shape[1:], dtype=t.dtype).index_copy(0, edge[real], t[real])
    assert torch.allclose(gx.permute(2, 3, 0, 1), wx, rtol=1e-9, atol=1e-12)
    assert torch.allclose(unpad(grbf), wrbf, rtol=1e-9, atol=1e-12)
    dcut = torch.autograd.grad(cut.sum(), r)[0]
    assert torch.allclose(unpad(gcut).sum(-1) * dcut, wr, rtol=1e-9, atol=1e-12)
