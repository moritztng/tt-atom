"""MACE hot blocks on the card (tt_atom/mace/tt_blocks.py) against the mace-torch golden's own activations.

Inputs and expected outputs are the golden's recorded module activations, so this grades the device block
in place, not against another approximation. Reported error is max |device - golden| / max |golden|.
"""
import os

import pytest
import torch

from tt_atom.mace import dense, tt_blocks
from tt_atom.mace import reference as ref
from test_mace_reference import GOLDEN, g  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not os.path.exists(GOLDEN), reason="MACE golden not generated")
# Transform check, relative to the block's max |output|: a wrong transform is O(1) off. Precision is judged end to
# end (energy, forces). Measured on a BH Galaxy chip, 11-27: float32 2.1-4.5e-3 (matmul operands enter the 19-bit
# source registers, so fp32 GEMMs carry TF32-like rounding), bfloat16 0.6-1.6e-2.
BAR = {"float32": 1e-2, "bfloat16": 3e-2}


def _dtype(name):
    import ttnn
    return getattr(ttnn, name)


def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
@pytest.mark.parametrize("layer,n_out", [(0, 2), (1, 1)])
def test_symmetric_contraction_device(g, device, layer, n_out, dt):  # noqa: F811
    import ttnn

    w, t = g["w"], g["t"]
    el = ref.node_attrs(g["Z"], w["atomic_numbers"].long()).argmax(1)
    present = el.unique()
    local = torch.searchsorted(present, el)
    order = torch.argsort(local, stable=True)
    tables = dense.product_tables(w, layer, n_out, elements=present)
    plan = tt_blocks.symmetric_contraction_setup(device, tables, _dtype(dt))
    ranges = tt_blocks.species_ranges(local[order], len(present))
    x = t[f"a@interactions.{layer}.out0"].double()[order]  # [n, c, ell]
    n, c, ell = x.shape
    out = tt_blocks.symmetric_contraction(ttnn.from_torch(x.reshape(n * c, ell), dtype=_dtype(dt),
                                                          layout=ttnn.TILE_LAYOUT, device=device), plan, ranges, n, c)
    got = tt_blocks.symmetric_contraction_to_ref(ttnn.to_torch(out), plan["widths"])
    want = t[f"a@products.{layer}.symmetric_contractions.out0"].double()[order]
    err = rel(got, want)
    print(f"symmetric_contraction layer {layer} {dt}: rel err {err:.2e}")
    assert err < BAR[dt], err


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
@pytest.mark.parametrize("layer", [0, 1])
def test_interaction_device(g, device, layer, dt):  # noqa: F811
    """The whole interaction block on the card (gather, radial MLPs, conv_tp, scatter, density norm, linears,
    element skip), atoms ordered by species and edges renumbered to match."""
    import ttnn

    w, t = g["w"], g["t"]
    el = ref.node_attrs(g["Z"], w["atomic_numbers"].long()).argmax(1)
    present = el.unique()
    local = torch.searchsorted(present, el)
    order = torch.argsort(local, stable=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(len(order))
    ei = inv[g["ei"]]
    irreps = [(128, 0)] if layer == 0 else [(128, 0), (128, 1)]
    plan = tt_blocks.interaction_setup(device, w, layer, present, irreps, layer == 1, _dtype(dt))
    graph = tt_blocks.graph_setup(device, local[order], len(present), ei)
    feats = dense.to_cm(t["a@node_embedding.out0" if layer == 0 else "a@products.0.out0"].double(), irreps)[order]

    def dev(a):
        return ttnn.from_torch(a, dtype=_dtype(dt), layout=ttnn.TILE_LAYOUT, device=device)

    def edges(a):
        return dev(tt_blocks.pad_edges(graph, a.double()))
    x, sc = tt_blocks.interaction(dev(feats.reshape(len(order), -1)), edges(t["a@spherical_harmonics.out0"]),
                                  edges(t["a@radial_embedding.out0"]), plan, graph)
    got = ttnn.to_torch(x).double().reshape(len(order), 16, 128).transpose(1, 2)
    err = rel(got, t[f"a@interactions.{layer}.out0"].double()[order])
    print(f"interaction layer {layer} {dt}: rel err {err:.2e}")
    assert err < BAR[dt], err
    if layer == 1:
        err = rel(ttnn.to_torch(sc).double(), t["a@interactions.1.out1"].double()[order])
        print(f"interaction skip layer 1 {dt}: rel err {err:.2e}")
        assert err < BAR[dt], err


@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
@pytest.mark.parametrize("layer", [0, 1])
def test_product_readout_device(g, device, layer, dt):  # noqa: F811
    """Product block (symmetric contraction, linear, skip) and the layer's readout on the card."""
    import ttnn

    w, t = g["w"], g["t"]
    el = ref.node_attrs(g["Z"], w["atomic_numbers"].long()).argmax(1)
    present = el.unique()
    local = torch.searchsorted(present, el)
    order = torch.argsort(local, stable=True)
    irreps = [(128, 0), (128, 1)] if layer == 0 else [(128, 0)]
    plan = tt_blocks.product_setup(device, w, layer, present, irreps, _dtype(dt))
    graph = {"n": len(order), "ranges": tt_blocks.species_ranges(local[order], len(present))}
    n = len(order)

    def dev(a):
        return ttnn.from_torch(a.reshape(n, -1), dtype=_dtype(dt), layout=ttnn.TILE_LAYOUT, device=device)
    x = t[f"a@interactions.{layer}.out0"].double()[order].transpose(1, 2)  # cm [n, 16, 128]
    sc = dev(t["a@interactions.1.out1"].double()[order]) if layer == 1 else None
    y, e = tt_blocks.product(dev(x), sc, plan, graph)
    got = dense.from_cm(ttnn.to_torch(y).double().reshape(n, -1, 128), irreps)
    err = rel(got, t[f"a@products.{layer}.out0"].double()[order])
    err_e = rel(ttnn.to_torch(e).double(), t[f"a@readouts.{layer}.out0"].double()[order])
    de = (ttnn.to_torch(e).double().sum() - t[f"a@readouts.{layer}.out0"].double().sum()).item()
    print(f"product layer {layer} {dt}: rel err {err:.2e}, readout {err_e:.2e}, readout sum off {de:+.4f} eV")
    assert err < BAR[dt] and err_e < BAR[dt], (err, err_e)


# bfloat16 blocks miss the 5 meV/atom bar on card (34 meV/atom, 6.6 with float32 products; qb2 11-27), so float32
# is the working precision. The two cases stay as strict xfails: a pass means the bar moved and this list is stale.
_BF16 = pytest.mark.xfail(strict=True, reason="bfloat16 blocks exceed 5 meV/atom")


@pytest.mark.parametrize("dt,pdt", [pytest.param("bfloat16", "bfloat16", marks=_BF16),
                                    pytest.param("bfloat16", "float32", marks=_BF16), ("float32", "float32")])
def test_energy_device(g, device, dt, pdt):  # noqa: F811
    """Total energy with both layers on the card; geometry features (sh, radial, ZBL) from the float64 reference."""
    from tt_atom.mace.tt_model import Mace

    w, t = g["w"], g["t"]
    m = Mace(device, w, _dtype(dt), _dtype(pdt))
    zbl = ref.zbl(g["r"], g["Z"], g["ei"], w, len(g["Z"]))
    e = m.energy(g["Z"], g["ei"], t["a@spherical_harmonics.out0"].double(), t["a@radial_embedding.out0"].double(), zbl)
    want = float(t["out@energy"].double().sum())
    n = len(g["Z"])
    print(f"energy {dt}/{pdt}: {e:.6f} vs {want:.6f} eV, dE/atom {abs(e - want) / n:.2e} eV")
    assert abs(e - want) / n < 5e-3


def test_energy_forces_device(g, device):  # noqa: F811
    """Energy and forces with both layers and their reverse passes on the card at float32; host float64 geometry.
    The force bar is Orb's shipped bfloat16 error against float64 (6-7 meV/A MAE)."""
    import ttnn
    from tt_atom.mace.tt_model import Mace

    t = g["t"]
    e, f = Mace(device, g["w"], ttnn.float32).energy_forces(g["Z"], t["in@pos"].double(), g["ei"], t["in@shifts"].double())
    want_e, want_f = float(t["out@energy"].double().sum()), t["out@forces"].double()
    d = (f - want_f).abs()
    print(f"energy+forces float32: dE/atom {abs(e - want_e) / len(g['Z']):.2e} eV, force MAE {d.mean():.2e} "
          f"max {d.max():.2e} eV/A (max |F| {want_f.abs().max():.2e})")
    assert abs(e - want_e) / len(g["Z"]) < 5e-3 and d.mean() < 7e-3
