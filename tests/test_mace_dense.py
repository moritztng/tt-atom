"""The device-shaped MACE blocks (tt_atom/mace/dense.py) against the float64 reference, on the golden's weights."""
import pytest
import torch

from tt_atom.mace import dense
from tt_atom.mace import reference as ref
from test_mace_reference import GOLDEN, g  # noqa: F401  (fixture)

needs_golden = pytest.mark.skipif(not __import__("os").path.exists(GOLDEN), reason="MACE golden not generated")


@needs_golden
@pytest.mark.parametrize("layer,n_out", [(0, 2), (1, 1)])
def test_symmetric_contraction_folded(g, layer, n_out):  # noqa: F811
    w = g["w"]
    attrs = ref.node_attrs(g["Z"], w["atomic_numbers"].long())
    x = torch.randn(len(g["Z"]), 128, 16, generator=torch.Generator().manual_seed(layer), dtype=torch.float64)
    for k, (T, parent, last) in enumerate(dense.product_tables(w, layer, n_out)):
        q = f"products.{layer}.symmetric_contractions.contractions.{k}."
        corr = sum(1 for key in w if key.startswith(q + "U_matrix_"))
        U = [w[f"{q}U_matrix_{nu}"] for nu in range(1, corr + 1)]
        W = [w[f"{q}weights.{corr - 1 - nu}"] for nu in range(1, corr)] + [w[q + "weights_max"]]
        a = ref.symmetric_contraction(x, attrs, U, W)
        b = dense.symmetric_contraction(x, attrs.argmax(1), T, parent, last)
        assert T.shape[2] == 16 + 136 + 816
        assert (a - b).abs().max() < 1e-12 * a.abs().max()
        c = dense.symmetric_contraction_gemm(x, attrs.argmax(1), T, dense.monomial_selectors(parent, last, 16))
        assert (a - c).abs().max() < 1e-12 * a.abs().max()


@needs_golden
def test_fold_only_present_elements(g):  # noqa: F811
    w = g["w"]
    el = ref.node_attrs(g["Z"], w["atomic_numbers"].long()).argmax(1)
    present = el.unique()
    x = torch.randn(len(g["Z"]), 128, 16, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    full = dense.product_tables(w, 0, 2)
    sub = dense.product_tables(w, 0, 2, elements=present)
    local = torch.searchsorted(present, el)
    for (Tf, p, l), (Ts, _, _) in zip(full, sub):
        assert Ts.shape[0] == len(present)
        assert torch.equal(dense.symmetric_contraction(x, el, Tf, p, l), dense.symmetric_contraction(x, local, Ts, p, l))


@pytest.mark.parametrize("irreps", [[(128, 0)], [(128, 0), (128, 1)]])  # MP-0b3 layer 0 and layer 1 inputs
def test_conv_tp_coupling_matrix(irreps):
    gen = torch.Generator().manual_seed(len(irreps))
    ls = [0, 1, 2, 3]
    sh = ref.spherical_harmonics(torch.randn(64, 3, generator=gen, dtype=torch.float64))
    C, ins_of_k, mid = dense.conv_tp_plan(irreps, ls, ls)
    x = torch.randn(64, sum(m * (2 * l + 1) for m, l in irreps), generator=gen, dtype=torch.float64)
    w = torch.randn(64, (int(ins_of_k.max()) + 1) * 128, generator=gen, dtype=torch.float64)
    a, mid_ref = ref.conv_tp(x, irreps, sh, ls, w, ls)
    b = dense.conv_tp(ref.reshape_irreps(x, irreps), sh, w, C, ins_of_k)
    assert mid == mid_ref
    assert (a - b).abs().max() < 1e-12 * a.abs().max()


@needs_golden
@pytest.mark.parametrize("layer", [0, 1])
def test_interaction_component_major(g, layer):  # noqa: F811
    """The whole interaction block in the device layout ([rows, components, channels]) against the golden."""
    w, t = g["w"], g["t"]
    el = ref.node_attrs(g["Z"], w["atomic_numbers"].long()).argmax(1)
    present = el.unique()
    irreps = [(128, 0)] if layer == 0 else [(128, 0), (128, 1)]
    feats = t["a@node_embedding.out0" if layer == 0 else "a@products.0.out0"].double()
    x, sc = dense.interaction_cm(w, layer, dense.to_cm(feats, irreps), torch.searchsorted(present, el), present,
                                 t["a@spherical_harmonics.out0"].double(), t["a@radial_embedding.out0"].double(),
                                 g["ei"], irreps, residual=layer == 1)
    want = t[f"a@interactions.{layer}.out0"].double()  # [n, mul, 16]
    assert (x.transpose(1, 2) - want).abs().max() < 1e-10 * want.abs().max()
    if layer == 1:
        want_sc = t["a@interactions.1.out1"].double()
        assert (dense.from_cm(sc, [(128, 0)]) - want_sc).abs().max() < 1e-10 * want_sc.abs().max()
