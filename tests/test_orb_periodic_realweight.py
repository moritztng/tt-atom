"""Periodic-image edge construction at production scale.

Checks UMA's ``tt_atom/geometry.py`` ``radius_graph`` (architecture-agnostic host neighbor-list,
proven for UMA) against Orb's own neighbor-list convention, on a 24-atom / 1064-edge periodic Si
supercell (``tests/gen_golden_orb.py --system supercell``) -- big enough to genuinely exercise
periodic wraparound (unlike the tiny 4-atom golden used for the rest of the port, whose periodic
images were only handled implicitly via a captured ``cell_shift``, see ``test_orb_forces_realweight.py``).

Orb's sign convention (``vectors = pos[receivers] - pos[senders] + shift``,
``featurization_utilities.compute_supercell_neighbors``) is the OPPOSITE of fairchem/UMA's
``edge_vec = pos[src] - pos[tgt] + shift`` (``radius_graph``'s own docstring) -- so reuse is just
``orb_senders, orb_receivers = tgt, src`` (swap), not a code change to ``radius_graph`` itself.

    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt conservative-inf-omat \
        --system supercell --out ~/.ttatom_run/goldens_real/si_supercell_orb.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_periodic_realweight.py -q -s

Absent the golden bundle the whole module auto-skips.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

REAL_GOLDEN = os.environ.get(
    "TTATOM_ORB_SUPERCELL_GOLDEN",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_supercell_orb.npz"),
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"Orb supercell golden bundle not found at {REAL_GOLDEN}",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _edge_fingerprint(senders, receivers, vectors, decimals=4):
    v = np.round(vectors.detach().numpy(), decimals)
    s, r = senders.numpy(), receivers.numpy()
    return {(int(s[i]), int(r[i]), tuple(v[i])) for i in range(len(s))}


@pytest.fixture(scope="module")
def gw():
    from tt_atom.orb_weights import OrbWeights

    return OrbWeights.load(REAL_GOLDEN)


def test_radius_graph_matches_orb_neighbor_list(gw):
    """The from-scratch reconstruction (UMA's ``radius_graph`` + Orb's sender/receiver swap)
    must reproduce the exact edge SET real ``orb-models`` built (order-independent -- both graphs
    list the same 1064 (sender, receiver, vector) edges, just not necessarily in the same order)."""
    from tt_atom.geometry import radius_graph

    pos = gw.inp("pos").double()
    cell = gw.inp("cell").double().squeeze(0)
    senders_gold = gw.inp("senders").long()
    receivers_gold = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").double()

    edge_index, shift = radius_graph(pos, 6.0, cell=cell, pbc=[True, True, True])
    src, tgt = edge_index[0], edge_index[1]
    orb_senders, orb_receivers = tgt, src               # Orb's convention is the opposite of UMA's
    orb_vectors = pos[orb_receivers] - pos[orb_senders] + shift

    fp_gold = _edge_fingerprint(senders_gold, receivers_gold, vectors_gold)
    fp_mine = _edge_fingerprint(orb_senders, orb_receivers, orb_vectors)
    print(f"\n[orb] periodic reconstruction: gold {len(fp_gold)} edges, reconstructed "
          f"{len(fp_mine)} edges, symmetric diff {len(fp_gold ^ fp_mine)}")
    assert orb_senders.shape[0] == senders_gold.shape[0]
    assert fp_mine == fp_gold


def test_backbone_on_reconstructed_graph(gw, device):
    """End-to-end: feed the device backbone with OUR reconstructed topology/geometry (not the
    golden's captured one) and confirm it still reproduces the real oracle's final node
    embedding -- segment_sum is order-invariant, so a different (but set-equal) edge order must
    give the same result to within the usual bf16 PCC bar."""
    from tt_atom.geometry import radius_graph
    from tt_atom.orb_geometry import host_edge_features
    from tt_atom.orb_model import Encoder, AttentionInteractionLayer, OrbGraphContext, _to_dev
    import ttnn

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    pos = gw.inp("pos").double()
    cell = gw.inp("cell").double().squeeze(0)
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")
    N = atomic_numbers.shape[0]

    edge_index, shift = radius_graph(pos, 6.0, cell=cell, pbc=[True, True, True])
    src, tgt = edge_index[0], edge_index[1]
    senders, receivers = tgt, src
    edge_feat, cutoff, _ = host_edge_features(pos, senders, receivers, shift)

    enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                 latent_dim=cfg["latent_dim"], hidden_dim=1024)
    node_dev = _to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev = _to_dev(edge_feat.float(), device, ttnn.bfloat16)
    nodes, edges = enc(node_dev, edge_dev)

    graph = OrbGraphContext(device, senders=senders, receivers=receivers,
                            cutoff=cutoff.float(), num_nodes=N)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)

    gold_nodes = gw.activation(f"gnn{L-1}.out0")
    pcc = _pcc(ttnn.to_torch(nodes).float(), gold_nodes)
    print(f"\n[orb] backbone on reconstructed periodic graph: final node PCC={pcc:.6f}")
    assert pcc > 0.99, pcc
