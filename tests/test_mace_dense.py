"""The device-shaped MACE blocks (tt_atom/mace/dense.py) against the float64 reference, on the golden's weights."""
import pytest
import torch

from tt_atom.mace import dense
from tt_atom.mace import reference as ref
from test_mace_reference import GOLDEN, g  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not __import__("os").path.exists(GOLDEN), reason="MACE golden not generated")


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
