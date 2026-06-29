"""End-to-end backbone parity: node embedding + energy on TT vs the fairchem golden."""
import ttnn

from tt_atom.model import Backbone, GraphContext
from util import pcc


def _build(golden, device):
    cfg = golden.config
    bb = Backbone(golden.w(), device, cfg,
                  golden.host("to_grid_mat"), golden.host("from_grid_mat"))
    graph = GraphContext(
        device,
        edge_index=golden.inp("edge_index"),
        wigner=golden.host("wigner"), wigner_inv=golden.host("wigner_inv"),
        x_edge=golden.host("x_edge"), edge_envelope=golden.host("edge_envelope"),
        num_nodes=golden.host("x_message_init").shape[0],
    )
    x_init = ttnn.from_torch(golden.host("x_message_init"), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
    sys_emb = golden.host("sys_node_embedding")
    sys_emb = ttnn.from_torch(sys_emb.reshape(sys_emb.shape[0], 1, sys_emb.shape[1]),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return bb, graph, x_init, sys_emb


def test_node_embedding(golden, device):
    bb, graph, x_init, sys_emb = _build(golden, device)
    node_emb = bb.node_embedding(x_init, graph, sys_emb)
    p = pcc(ttnn.to_torch(node_emb).float(), golden.out("node_embedding"))
    assert p >= 0.98, f"node_embedding PCC {p}"


def test_energy(golden, device):
    bb, graph, x_init, sys_emb = _build(golden, device)
    _, energy = bb(x_init, graph, sys_emb)
    e = ttnn.to_torch(energy).float().reshape(-1)
    ref = golden.out("energy").reshape(-1)
    rel = abs(float(e[0]) - float(ref[0])) / (abs(float(ref[0])) + 1e-6)
    assert rel < 0.05, f"energy {float(e[0])} vs {float(ref[0])} rel {rel}"
