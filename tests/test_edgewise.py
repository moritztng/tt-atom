"""Per-module parity: Edgewise message block on TT vs the fairchem golden (PCC >= 0.98)."""
import ttnn

from tt_atom.edgewise import Edgewise
from tt_atom.model import GraphContext
from util import pcc


def _graph(golden, device):
    return GraphContext(
        device,
        edge_index=golden.inp("edge_index"),
        wigner=golden.host("wigner"), wigner_inv=golden.host("wigner_inv"),
        x_edge=golden.host("x_edge"), edge_envelope=golden.host("edge_envelope"),
        num_nodes=golden.act("block0.edgewise.in0").shape[0],
    )


def test_edgewise(golden, device):
    cfg = golden.config
    ew = Edgewise(golden.w(), "blocks.0.edge_wise", device,
                  sphere_channels=cfg["sphere_channels"], hidden_channels=cfg["hidden_channels"],
                  lmax=cfg["lmax"], mmax=cfg["mmax"])
    graph = _graph(golden, device)
    x = ttnn.from_torch(golden.act("block0.edgewise.in0"), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(ew(x, graph)).float()
    p = pcc(o, golden.act("block0.edgewise.out0"))
    assert p >= 0.98, f"edgewise PCC {p}"
