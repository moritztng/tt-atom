"""Trace-capture parity for Orb-v3 (``tt_atom/orb_trace.py``'s ``OrbTracedEngine``) -- the item
flagged open in ``docs/orb-port.md``'s "Profiling re-measurement at production scale" section: a
quick exploratory attempt wired raw ``begin_trace_capture``/``execute_trace`` with no refresh
logic and measured a real 1.28x replay speedup, but the replayed output did NOT match eager
(missing the UMA ``TracedEngine`` pattern of explicit captured-tensor handles + in-place
``copy_host_to_device_tensor`` refreshes). This test is the correctness gate for the proper port:
the trace-replayed forward(+backward) must be BIT-EXACT vs the eager forward(+backward) -- a
trace only removes host dispatch, it is the same device op stream -- for both checkpoints
(conservative-inf-omat: analytic-VJP forces; direct-20-omat: forward-only ForceHead) at both the
toy 4-atom golden and the 24-atom/1064-edge production-scale supercell golden.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_trace.py -q -s

Any individual golden absent -> that case auto-skips (module-level goldens are independent).
"""
from __future__ import annotations

import os
import pathlib

import pytest
import torch

GOLDENS = pathlib.Path.home() / ".ttatom_run" / "goldens_real"
CONSERVATIVE = {
    "toy": GOLDENS / "si_omat_orb.npz",
    "supercell": GOLDENS / "si_supercell_orb.npz",
}
DIRECT = {
    "toy": GOLDENS / "si_omat_orb_direct20.npz",
    "supercell": GOLDENS / "si_supercell_orb_direct20.npz",
}


@pytest.fixture(scope="module")
def device():
    from tt_atom.device import open_device
    import ttnn

    dev = open_device(int(os.environ.get("TT_VISIBLE_DEVICES", "0")), trace_region_size=200_000_000)
    yield dev
    ttnn.close_device(dev)


def _build_modules(gw, device):
    from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead, ForceHead

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=cfg["latent_dim"], hidden_dim=1024)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    fhead = ForceHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024) \
        if gw.has("w@forces_head.mlp.NN-0.weight") else None
    return encoder, layers, ehead, fhead


@pytest.mark.parametrize("system", ["toy", "supercell"])
def test_conservative_trace_matches_eager(system, device):
    from tt_atom.orb_forces import energy_and_forces
    from tt_atom.orb_trace import OrbTracedEngine
    from tt_atom.orb_weights import OrbWeights

    path = CONSERVATIVE[system]
    if not path.exists():
        pytest.skip(f"golden not found at {path}")
    gw = OrbWeights.load(path)
    encoder, layers, ehead, _ = _build_modules(gw, device)

    pos = gw.inp("pos").float()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").float()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")

    raw_e_eager, forces_eager = energy_and_forces(
        encoder, layers, ehead, device, pos=pos, senders=senders, receivers=receivers,
        atomic_numbers=atomic_numbers, node_feat=node_feat, cell_shift=cell_shift)

    eng = OrbTracedEngine(encoder, layers, device, senders=senders, receivers=receivers,
                          atomic_numbers=atomic_numbers, node_feat=node_feat, ehead=ehead,
                          cell_shift=cell_shift)
    raw_e_cap, forces_cap = eng(pos)             # capture step (records + replays)
    raw_e_replay, forces_replay = eng(pos)       # pure replay
    eng.close()

    e_err = abs(raw_e_cap - raw_e_eager)
    f_err = (forces_cap - forces_eager).abs().max().item()
    print(f"\n[orb-trace] conservative/{system}: E eager={raw_e_eager:.6f} cap={raw_e_cap:.6f} "
          f"(err {e_err:.2e}); F max abs diff={f_err:.2e}")
    assert e_err == 0, f"traced E != eager: {raw_e_cap} vs {raw_e_eager}"
    assert f_err == 0, f"traced forces != eager (max abs diff {f_err:.2e})"
    assert raw_e_cap == raw_e_replay and torch.equal(forces_cap, forces_replay), \
        "replay not deterministic"


@pytest.mark.parametrize("system", ["toy", "supercell"])
def test_direct_trace_matches_eager(system, device):
    from tt_atom.orb_model import OrbGraphContext, _to_dev
    from tt_atom.orb_geometry import host_edge_features
    from tt_atom.orb_trace import OrbTracedEngine
    from tt_atom.orb_weights import OrbWeights
    import ttnn

    path = DIRECT[system]
    if not path.exists():
        pytest.skip(f"golden not found at {path}")
    gw = OrbWeights.load(path)
    encoder, layers, ehead, fhead = _build_modules(gw, device)
    assert fhead is not None, "direct-20 golden must carry a forces_head"

    pos = gw.inp("pos").float()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").float()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")
    N = atomic_numbers.shape[0]

    # eager forward: same op stream OrbTracedEngine's direct mode captures
    edge_feat, cutoff, _ = host_edge_features(pos, senders, receivers, cell_shift)
    node_dev = _to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev = _to_dev(edge_feat.detach().float(), device, ttnn.bfloat16)
    graph = OrbGraphContext(device, senders=senders, receivers=receivers,
                            cutoff=cutoff.detach().float(), num_nodes=N)
    nodes, edges = encoder(node_dev, edge_dev)
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)
    raw_e_eager = ttnn.to_torch(ehead(nodes)).double().view(())
    raw_f_eager = ttnn.to_torch(fhead(nodes)).double()

    eng = OrbTracedEngine(encoder, layers, device, senders=senders, receivers=receivers,
                          atomic_numbers=atomic_numbers, node_feat=node_feat, ehead=ehead,
                          fhead=fhead, cell_shift=cell_shift)
    raw_e_cap, raw_f_cap = eng(pos)
    raw_e_replay, raw_f_replay = eng(pos)
    eng.close()

    e_err = abs(raw_e_cap - raw_e_eager).item()
    f_err = (raw_f_cap - raw_f_eager).abs().max().item()
    print(f"\n[orb-trace] direct/{system}: E eager={raw_e_eager:.6f} cap={raw_e_cap:.6f} "
          f"(err {e_err:.2e}); F max abs diff={f_err:.2e}")
    assert e_err == 0, f"traced E != eager (err {e_err:.2e})"
    assert f_err == 0, f"traced forces != eager (max abs diff {f_err:.2e})"
    assert torch.equal(raw_e_cap, raw_e_replay) and torch.equal(raw_f_cap, raw_f_replay), \
        "replay not deterministic"
