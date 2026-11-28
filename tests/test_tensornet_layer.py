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
    concat, sum = (lambda xs, dim: torch.cat(xs, dim)), (lambda x, dim, keepdim=False: x.sum(dim, keepdim=keepdim))
    rsqrt = torch.rsqrt
    typecast = staticmethod(lambda t, dtype: t)
    to_torch = staticmethod(lambda t: t)
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


@pytest.mark.skipif(OLD, reason="needs matgl >= 3")
@pytest.mark.parametrize("backend", ["torch", "tt"])
def test_embedding_matches_matgl(backend):
    """Embedding block: output and the gradients of the radial basis, the distance and the edge vector."""
    from tt_atom.tensornet import embed as E

    from ase.build import bulk
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(MODEL).double()
    pot.calc_stresses = False
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    atoms.positions += np.random.default_rng(0).normal(0, 0.05, atoms.positions.shape)
    emb, calls = pot.model.tensor_embedding, []
    h = emb.register_forward_hook(lambda mod, args, kw, out: calls.append((args, kw)), with_kwargs=True)
    atoms.calc = PESCalculator(pot)
    atoms.get_potential_energy()
    h.remove()
    a = dict(zip(("z", "edge_index", "edge_attr", "edge_weight", "edge_vec"), calls[0][0]), **calls[0][1])
    z, ei = a["z"], a["edge_index"]
    rbf, r, vec = [a[k].detach().clone().requires_grad_(True) for k in ("edge_attr", "edge_weight", "edge_vec")]
    want = emb(z, ei, rbf, r, vec)[0]
    g_out = torch.randn(want.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    wrbf, wr, wvec = torch.autograd.grad((want * g_out).sum(), (rbf, r, vec))

    n, e = len(z), ei.shape[1]
    edge, _, _, d = L.slots(ei, n, min_width=32)
    o = L.TorchOps if backend == "torch" else TtOps(None, n, d, emb.units, ttnn=FakeTtnn)
    pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge]
    cut = torch.where(r <= emb.cutoff, 0.5 * (torch.cos(torch.pi * r / emb.cutoff) + 1), 0.0)[:, None]
    geo = E.geometry(vec)
    w = E.weights(emb)
    if backend == "tt":
        up2 = lambda pair: (o.up(pair[0]), o.up(pair[1][None]))
        w = {"proj": [up2(p) for p in w["proj"]], "scalar": [up2(p) for p in w["scalar"]], "ln": w["ln"],
             "proj_t": w["proj_t"], "scalar_t": w["scalar_t"], "node": [m[None, None] for m in w["node"]],
             "node_t": [m[None, None] for m in w["node_t"]]}
    got, tape = E.forward(o, w, pad(E.pair(emb, z, ei)), pad(rbf.detach()), pad(cut.detach()),
                          list(pad(geo.detach()).split(1, -1)), n)
    assert torch.allclose(got.permute(2, 3, 0, 1), want, rtol=1e-10, atol=1e-12)

    grbf, gcut, ggeo = E.backward(o, w, tape, g_out.permute(2, 3, 0, 1), d)
    real = edge < e
    unpad = lambda t: torch.zeros(e, *t.shape[1:], dtype=t.dtype).index_copy(0, edge[real], t[real])
    assert torch.allclose(unpad(grbf), wrbf, rtol=1e-9, atol=1e-12)
    gr, gvec = torch.autograd.grad((cut * unpad(gcut)).sum() + (geo * unpad(ggeo)).sum(), (r, vec))
    assert torch.allclose(gr, wr, rtol=1e-9, atol=1e-12)
    assert torch.allclose(gvec, wvec, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(OLD, reason="needs matgl >= 3")
def test_model_matches_matgl():
    """Embedding, both layers and readout chained: summed atomic energies and their gradient to every edge input."""
    from ase.build import bulk
    from matgl.ext.ase import PESCalculator
    from matgl.utils.maths import decompose_tensor, tensor_norm

    from tt_atom.tensornet import embed as E
    from tt_atom.tensornet import model as M

    pot = matgl.load_model(MODEL).double()
    pot.calc_stresses = False
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    atoms.positions += np.random.default_rng(0).normal(0, 0.05, atoms.positions.shape)
    m, calls = pot.model, []
    h = m.tensor_embedding.register_forward_hook(lambda mod, args, kw, out: calls.append((args, kw)), with_kwargs=True)
    atoms.calc = PESCalculator(pot)
    atoms.get_potential_energy()
    h.remove()
    a = dict(zip(("z", "edge_index", "edge_attr", "edge_weight", "edge_vec"), calls[0][0]), **calls[0][1])
    z, ei = a["z"], a["edge_index"]
    rbf, r, vec = [a[k].detach().clone().requires_grad_(True) for k in ("edge_attr", "edge_weight", "edge_vec")]
    x = m.tensor_embedding(z, ei, rbf, r, vec)[0]
    for layer in m.layers:
        x = layer(ei, r, rbf, x)
    want = m.final_layer(m.linear(m.out_norm(torch.cat([tensor_norm(p) for p in decompose_tensor(x)], -1)))).reshape(-1)
    wrbf, wr, wvec = torch.autograd.grad(want.sum(), (rbf, r, vec))

    n, e, o = len(z), ei.shape[1], L.TorchOps
    edge, send, table, d = L.slots(ei, n, min_width=32)
    pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge]
    cut = torch.where(r <= m.cutoff, 0.5 * (torch.cos(torch.pi * r / m.cutoff) + 1), 0.0)[:, None]
    geo = E.geometry(vec)
    eye = torch.eye(3, dtype=torch.float64).reshape(3, 3, 1, 1).expand(3, 3, n, m.units)
    w = M.weights(m, torch.float64)
    en, tape = M.forward(o, w, pad(E.pair(m.tensor_embedding, z, ei)), (pad(rbf.detach()), pad(cut.detach())),
                         list(pad(geo.detach()).split(1, -1)), send, n, eye)
    assert torch.allclose(en.reshape(-1), want, rtol=1e-10, atol=1e-12)

    grbf, gcut, ggeo = M.backward(o, w, tape, torch.ones_like(en), table, n, d, eye)
    grbf, gcut = sum(grbf), sum(gcut)
    real = edge < e
    unpad = lambda t: torch.zeros(e, *t.shape[1:], dtype=t.dtype).index_copy(0, edge[real], t[real])
    assert torch.allclose(unpad(grbf), wrbf, rtol=1e-8, atol=1e-11)
    gr, gvec = torch.autograd.grad((cut * unpad(gcut)).sum() + (geo * unpad(ggeo)).sum(), (r, vec))
    assert torch.allclose(gr, wr, rtol=1e-8, atol=1e-11)
    assert torch.allclose(gvec, wvec, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(OLD, reason="needs matgl >= 3")
def test_radial_basis_matches_matgl():
    from tt_atom.tensornet import embed as E

    m = matgl.load_model(MODEL).double().model
    r = torch.linspace(0.8, 5.0, 400, dtype=torch.float64).requires_grad_(True)
    want, got = m.bond_expansion(r), E.radial(r, m.cutoff, m.bond_expansion.max_n)
    assert got.shape == want.shape and torch.allclose(got, want, rtol=1e-11, atol=1e-13)
    for k in range(got.shape[1]):
        gw, gg = torch.autograd.grad(want[:, k].sum(), r, retain_graph=True)[0], torch.autograd.grad(got[:, k].sum(), r, retain_graph=True)[0]
        assert torch.allclose(gg, gw, rtol=1e-9, atol=1e-11)


def test_potential_without_matgl_graph():
    """Neighbour list, radial basis, cutoff and geometry built here from an ase.Atoms: energy and forces of a
    rattled, sheared cell against matgl's own calculator in float64."""
    from ase.build import bulk
    from matgl.ext.ase import PESCalculator

    from tt_atom.tensornet import embed as E
    from tt_atom.tensornet import model as M
    from tt_atom.tensornet import potential as P

    pot = matgl.load_model(MODEL).double()
    pot.calc_stresses = False
    m = pot.model
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    atoms.set_cell(atoms.cell.array @ np.array([[1, 0.03, 0], [0, 1, 0.02], [0, 0, 0.98]]), scale_atoms=True)
    atoms.positions += np.random.default_rng(0).normal(0, 0.08, atoms.positions.shape)
    ref = atoms.copy()
    ref.calc = PESCalculator(pot)
    z = torch.tensor([pot.model.element_types.index(s) for s in atoms.get_chemical_symbols()])
    shift = float(pot.data_mean) + float(pot.element_refs.property_offset.double()[z].sum())
    from tt_atom.tensornet.tt_ops import runner

    w, args = M.weights(m, torch.float64), (lambda ei: E.pair(m.tensor_embedding, z, ei), atoms, float(m.cutoff), m.bond_expansion.max_n)
    e, f = P.energy_forces(P.torch_run(w, m.units), *args, scale=float(pot.data_std), shift=shift)
    e_tt, f_tt = P.energy_forces(runner(None, w, m.units, ttnn=FakeTtnn), *args, scale=float(pot.data_std), shift=shift)
    # the card's op bookkeeping on the stand-in; inputs go up as float32, hence 1e-7
    assert abs(e_tt - e) < 1e-7 * len(atoms) and np.abs(f_tt - f).max() < 1e-6
    assert abs(e - ref.get_potential_energy()) < 1e-7 * len(atoms)   # matgl's own offsets pass through float32
    # matgl's calculator agrees with its own float64 autograd only to ~4e-6 relative (same level with matgl's graph)
    assert np.abs(f - ref.get_forces()).max() < 1e-5 * np.abs(f).max()
