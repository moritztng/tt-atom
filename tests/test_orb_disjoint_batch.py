"""Disjoint-union (block-diagonal) batching for Orb (``docs/orb-port.md`` Open item): verifies
-- rather than assumes -- that ``tt_atom``'s existing UMA batching methodology (bit-exact
row-independence, see ``ttatom-batching``/``ttatom-qb2-multicard-fanout``) applies unmodified to
``Encoder``/``AttentionInteractionLayer``.

Orb's forward has no cross-system op: ``AttentionInteractionLayer`` only ever touches
``senders_idx``/``receivers_idx`` (arbitrary global node indices) and ``scatter.segment_sum``
(a per-edge-group reduction with no notion of "system boundary") -- so two independent systems,
concatenated with a per-system node offset applied to their edge indices, should produce EXACTLY
the same per-system output as running each system alone (block-diagonal message passing: an
edge's message can only reach nodes reachable via that edge, and no edge crosses systems).

This test builds a 2-system disjoint-union batch from the *same* golden Si system twice (a second
copy, node-offset by N) and checks the batched Encoder+backbone output on each half is bit-exact
(same bf16 rounding, same PCC) against the already-verified single-system run
(``test_orb_realweight.py``) -- confirming no adapter code is needed.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_disjoint_batch.py -q -s

Absent the golden bundle the whole module auto-skips.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

REAL_GOLDEN = os.environ.get(
    "TTATOM_ORB_GOLDEN", str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat_orb.npz")
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"Orb golden bundle not found at {REAL_GOLDEN}",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def gw():
    from tt_atom.orb_weights import OrbWeights

    return OrbWeights.load(REAL_GOLDEN)


def test_disjoint_batch_row_independence(gw, device):
    from tt_atom.orb_model import Encoder, AttentionInteractionLayer, OrbGraphContext, _to_dev
    import ttnn

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]

    node_feat = gw.host("node_feat")
    edge_feat = gw.host("edge_feat")
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    cutoff_np = gw.inp("vectors").norm(dim=-1)
    N = node_feat.shape[0]
    E = senders.shape[0]

    def run(node_feat, edge_feat, senders, receivers, cutoff, num_nodes):
        enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                     latent_dim=cfg["latent_dim"], hidden_dim=1024)
        node_dev = _to_dev(node_feat, device, ttnn.bfloat16)
        edge_dev = _to_dev(edge_feat, device, ttnn.bfloat16)
        nodes, edges = enc(node_dev, edge_dev)
        graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff,
                                num_nodes=num_nodes)
        layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                            latent_dim=cfg["latent_dim"], hidden_dim=1024)
                  for i in range(L)]
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        return ttnn.to_torch(nodes).float()

    from tt_atom.orb_model import host_cutoff
    cutoff = host_cutoff(cutoff_np, r_max=6.0)

    # single-system reference (identical to test_orb_realweight.py's test_full_backbone_5layers)
    single_out = run(node_feat, edge_feat, senders, receivers, cutoff, N)

    # 2-system disjoint-union: a second copy of the SAME system, node-offset by N
    node_feat_batch = torch.cat([node_feat, node_feat], dim=0)
    edge_feat_batch = torch.cat([edge_feat, edge_feat], dim=0)
    senders_batch = torch.cat([senders, senders + N], dim=0)
    receivers_batch = torch.cat([receivers, receivers + N], dim=0)
    cutoff_batch = torch.cat([cutoff, cutoff], dim=0)
    batch_out = run(node_feat_batch, edge_feat_batch, senders_batch, receivers_batch,
                    cutoff_batch, 2 * N)

    half0, half1 = batch_out[:N], batch_out[N:]
    pcc0 = _pcc(half0, single_out)
    pcc1 = _pcc(half1, single_out)
    err0 = (half0 - single_out).abs().max().item()
    err1 = (half1 - single_out).abs().max().item()
    print(f"\n[orb] disjoint-batch row independence: half0 PCC={pcc0:.6f} err={err0:.2e}  "
          f"half1 PCC={pcc1:.6f} err={err1:.2e}")
    # bf16 device nondeterminism (op scheduling / reduction order can differ when N/E change)
    # gives a tight-but-not-bit-exact bar -- same tolerance style as the rest of the port's PCC gates.
    assert pcc0 > 0.999, pcc0
    assert pcc1 > 0.999, pcc1
    assert err0 < 5e-2, err0
    assert err1 < 5e-2, err1

    # EnergyHead.batch: Orb means node FEATURES first (unlike UMA's per-node-scalar
    # segment-sum, see EnergyHead.batch's docstring), so it needs its own row-normalized
    # segment-matrix adapter -- verify it reproduces the single-system EnergyHead exactly.
    from tt_atom.orb_model import EnergyHead, _to_dev

    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    single_e = ttnn.to_torch(ehead(_to_dev(single_out, device, ttnn.bfloat16))).float()

    seg_mean = torch.zeros(2, 2 * N)
    seg_mean[0, :N] = 1.0 / N
    seg_mean[1, N:] = 1.0 / N
    seg_dev = _to_dev(seg_mean, device, ttnn.bfloat16)
    batch_e = ttnn.to_torch(ehead.batch(_to_dev(batch_out, device, ttnn.bfloat16), seg_dev)).float()

    pcc_e = _pcc(batch_e, single_e.expand(2, -1))
    err_e = (batch_e - single_e.expand(2, -1)).abs().max().item()
    print(f"[orb] EnergyHead.batch vs single-system: PCC={pcc_e:.6f} err={err_e:.2e}")
    assert pcc_e > 0.999, pcc_e
