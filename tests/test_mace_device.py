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
@pytest.mark.parametrize("layer,irreps", [(0, [(128, 0)]), (1, [(128, 0), (128, 1)])])
def test_conv_tp_device(g, device, layer, irreps, dt):  # noqa: F811
    import ttnn

    t, ls = g["t"], [0, 1, 2, 3]
    C, ins_of_k, _ = dense.conv_tp_plan(irreps, ls, ls)
    plan = tt_blocks.conv_tp_setup(device, C, ins_of_k, _dtype(dt))
    p = f"a@interactions.{layer}."
    x1 = ref.reshape_irreps(t[p + "linear_up.out0"].double()[g["ei"][0]], irreps)  # [E, mul, A]
    sh = t["a@spherical_harmonics.out0"].double()
    w = t[p + "conv_tp_weights.out0"].double()
    E, mul = x1.shape[:2]

    def dev(a):
        return ttnn.from_torch(a, dtype=_dtype(dt), layout=ttnn.TILE_LAYOUT, device=device)
    out = tt_blocks.conv_tp(dev(x1), dev(sh), dev(w.reshape(E, -1, mul)), plan)
    got = tt_blocks.conv_tp_to_ref(ttnn.to_torch(out), ins_of_k)
    err = rel(got, t[p + "conv_tp.out0"])
    print(f"conv_tp layer {layer} {dt}: rel err {err:.2e}")
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
    x, sc = tt_blocks.interaction(dev(feats.reshape(len(order), -1)), dev(t["a@spherical_harmonics.out0"].double()),
                                  dev(t["a@radial_embedding.out0"].double()), plan, graph)
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


@pytest.mark.parametrize("dt,pdt", [("bfloat16", "bfloat16"), ("bfloat16", "float32"), ("float32", "float32")])
def test_energy_device(g, device, dt, pdt):  # noqa: F811
    """Total energy with both layers on the card; geometry features (sh, radial, ZBL) from the float64 reference."""
    from tt_atom.mace.tt_model import MaceEnergy

    w, t = g["w"], g["t"]
    m = MaceEnergy(device, w, _dtype(dt), _dtype(pdt))
    zbl = ref.zbl(g["r"], g["Z"], g["ei"], w, len(g["Z"]))
    e = m.energy(g["Z"], g["ei"], t["a@spherical_harmonics.out0"].double(), t["a@radial_embedding.out0"].double(), zbl)
    want = float(t["out@energy"].double().sum())
    n = len(g["Z"])
    print(f"energy {dt}/{pdt}: {e:.6f} vs {want:.6f} eV, dE/atom {abs(e - want) / n:.2e} eV")
    assert abs(e - want) / n < 5e-3
