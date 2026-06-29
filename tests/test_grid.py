"""Per-module parity: GridAtomwise on TT vs the fairchem golden (PCC >= 0.98)."""
import ttnn

from tt_atom.grid import GridAtomwise
from util import pcc


def test_grid_atomwise(golden, device):
    w = golden.w()
    aw = GridAtomwise(
        w, "blocks.0.atom_wise", device,
        golden.host("to_grid_mat"), golden.host("from_grid_mat"),
    )
    x = ttnn.from_torch(golden.act("block0.atomwise.in0"), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(aw(x)).float()
    p = pcc(o, golden.act("block0.atomwise.out0"))
    assert p >= 0.98, f"grid atomwise PCC {p}"
