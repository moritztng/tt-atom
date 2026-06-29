"""Analytic-force VJP parity: the on-device reverse pass must reproduce, to PCC >= 0.98, the
adjoints that ``torch.autograd`` produces on the bit-exact PyTorch mirror of the forward.

This isolates the *device backward* (the hard, hand-written part). The remaining host
geometric Jacobian ``d(geometric)/dpos`` is exercised by the end-to-end force test once
``tt_atom/geometry.py`` lands.
"""
import ttnn

import mirror
from tt_atom.model import Backbone, GraphContext
from tt_atom import forces
from util import pcc


def _leaves(golden):
    xi = golden.host("x_message_init")
    return dict(xi=xi, wig=golden.host("wigner"), winv=golden.host("wigner_inv"),
                xe=golden.host("x_edge"), env=golden.host("edge_envelope"))


def test_backbone_vjp(golden, device):
    cfg = golden.config
    w = golden.w()
    se = golden.host("sys_node_embedding")
    ei = golden.inp("edge_index").long()
    tg, fg = golden.host("to_grid_mat"), golden.host("from_grid_mat")
    N = se.shape[0]

    # oracle: autograd through the PyTorch mirror
    lv = {k: v.clone().requires_grad_() for k, v in _leaves(golden).items()}
    ne = mirror.backbone(w, cfg, lv["xi"], lv["wig"], lv["winv"], lv["xe"], lv["env"], se, ei, tg, fg)
    mirror.energy(ne, w).backward()

    # device forward + reverse VJP
    bb = Backbone(w, device, cfg, tg, fg)
    graph = GraphContext(device, edge_index=golden.inp("edge_index"), wigner=lv["wig"].detach(),
                         wigner_inv=lv["winv"].detach(), x_edge=lv["xe"].detach(),
                         edge_envelope=lv["env"].detach(), num_nodes=N)
    se3 = ttnn.from_torch(se.reshape(N, 1, se.shape[1]), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    xi = ttnn.from_torch(lv["xi"].detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    node_emb = bb.node_embedding(xi, graph, se3)
    acc = forces.backbone_bw(bb, graph, node_emb)

    assert pcc(ttnn.to_torch(acc["x_init"]).float(), lv["xi"].grad) >= 0.98
    assert pcc(ttnn.to_torch(acc["wigner"]).float(), lv["wig"].grad) >= 0.98
    assert pcc(ttnn.to_torch(acc["wigner_inv"]).float(), lv["winv"].grad) >= 0.98
    assert pcc(ttnn.to_torch(acc["envelope"]).float(), lv["env"].grad) >= 0.98

    # host radial finish: g_rad (radial-MLP output adjoint) -> g_x_edge
    xel = _leaves(golden)["xe"].clone().requires_grad_()
    for conv, grad in acc["g_rad"]:
        mirror.radial_mlp(xel, w, conv.rad_prefix).backward(ttnn.to_torch(grad).float())
    assert pcc(xel.grad, lv["xe"].grad) >= 0.98
