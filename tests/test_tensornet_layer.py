"""The TensorNet interaction layer of tt_atom.tensornet.layer against matgl in float64: output and reverse pass."""
from pathlib import Path

import numpy as np
import pytest
import torch

matgl = pytest.importorskip("matgl")
from tt_atom.tensornet import layer as L  # noqa: E402
from tt_atom.tensornet.tt_ops import TtOps, to_device  # noqa: E402

MODEL = "materialyze/TensorNet-PES-MatPES-PBE-2025.2"
V2025_1 = Path.home() / ".ttatom_run/tensornet_v2025_1"   # TensorNet-MatPES-PBE-v2025.1-PES of matgl tag v1.3.0
OLD = int(matgl.__version__.split(".")[0]) < 3


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


def check(mod, want_fn, ei, r, rbf, x, backend, own):
    x = x.clone().requires_grad_(True)
    rbf = rbf.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    want = want_fn(r, rbf, x)                                           # [N, U, 3, 3]
    g_out = torch.randn(want.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    wx, wrbf, wr = torch.autograd.grad((want * g_out).sum(), (x, rbf, r))

    n = x.shape[0]
    edge, send, table, d = L.slots(ei, n, min_width=32, own=own)
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


@pytest.mark.skipif(OLD, reason="needs matgl >= 3")
@pytest.mark.parametrize("backend", ["torch", "tt"])
@pytest.mark.parametrize("which", [0, 1])
def test_layer_matches_matgl(which, backend):
    mod, args, kw = captured()[which]
    ei, r, rbf, x = layer_args(args, kw)
    check(mod, lambda r, rbf, x: mod(ei, r, rbf, x), ei, r, rbf, x, backend, own=False)


@pytest.mark.skipif(not (OLD and V2025_1.exists()), reason="needs matgl 1.x with DGL and the v2025.1 files")
@pytest.mark.parametrize("backend", ["torch", "tt"])
@pytest.mark.parametrize("which", [0, 1])
def test_layer_matches_matgl1(which, backend):
    """The BSD-3 v2025.1 weights against the matgl 1.3 layer they were trained with."""
    import dgl
    from ase.build import bulk
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(str(V2025_1))
    pot.calc_stresses = False
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    atoms.positions += np.random.default_rng(0).normal(0, 0.05, atoms.positions.shape)
    calls = []
    hooks = [m.register_forward_hook(lambda mod, args, out: calls.append((mod, args))) for m in pot.model.layers]
    atoms.calc = PESCalculator(pot)
    atoms.get_potential_energy()
    for h in hooks:
        h.remove()
    mod, (g, x) = calls[which]
    mod, ei = mod.double(), torch.stack(g.edges()).long()
    r, rbf = g.edata["bond_dist"].detach().double(), g.edata["edge_attr"].detach().double()

    def want(r, rbf, x):
        g2 = dgl.graph((ei[0], ei[1]), num_nodes=x.shape[0])
        g2.edata["bond_dist"], g2.edata["edge_attr"] = r, rbf
        return mod(g2, x)

    check(mod, want, ei, r, rbf, x.detach().double(), backend, own=True)
