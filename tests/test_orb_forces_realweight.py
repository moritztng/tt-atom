"""Real-weight parity for ``orb-v3-conservative-inf-omat`` analytic forces.

Conservative forces come from backprop through the energy, unlike direct-20's per-node ForceHead
MLP (already ported, see ``test_orb_direct_realweight.py``).

Verifies against the REAL oracle: ``orb-models``' own ``torch.autograd`` forces, captured
straight into the golden bundle by ``tests/gen_golden_orb.py`` (``out@forces``) -- not a
hand-rolled reference.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_forces_realweight.py -q -s

Absent the golden bundle the whole module auto-skips.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

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



def test_edge_geometry_matches_golden(gw):
    """Sanity gate before trusting any gradient: the from-scratch differentiable
    ``host_edge_features`` (pure torch, no orb-models dependency) must reproduce the golden's
    captured ``host@edge_feat`` (real ``featurize_edges`` output) at the golden's own ``pos``."""
    from tt_atom.orb_geometry import host_edge_features

    pos = gw.inp("pos").double()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").double()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()

    edge_feat, cutoff, vectors = host_edge_features(pos, senders, receivers, cell_shift)
    err_vec = (vectors - vectors_gold).abs().max().item()
    gold_edge_feat = gw.host("edge_feat").double()
    pcc = _pcc(edge_feat.detach(), gold_edge_feat)
    rel_err = (edge_feat.detach() - gold_edge_feat).abs().max() / gold_edge_feat.abs().max()
    print(f"\n[orb] host_edge_features: vector err={err_vec:.2e} edge_feat PCC={pcc:.8f} "
          f"rel_err={rel_err:.2e}")
    assert err_vec < 1e-5
    assert pcc > 0.999999
    assert rel_err < 1e-4


def test_conservative_forces(gw, device):
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, EnergyHead,
                                   host_conservative_force_denormalize)
    from tt_atom.orb_forces import energy_and_forces

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=cfg["latent_dim"], hidden_dim=1024)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)

    pos = gw.inp("pos").float()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").float()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")

    raw_pred, raw_forces = energy_and_forces(
        encoder, layers, ehead, device, pos=pos, senders=senders, receivers=receivers,
        atomic_numbers=atomic_numbers, node_feat=node_feat, cell_shift=cell_shift)
    N = atomic_numbers.shape[0]
    forces = host_conservative_force_denormalize(
        raw_forces, N, running_var=w["energy_head.normalizer.bn.running_var"])

    gold_forces = gw.out("forces").double()   # real orb-models torch.autograd oracle (incl. ZBL)
    pcc = _pcc(forces.detach(), gold_forces)
    mae = (forces.detach() - gold_forces).abs().mean().item()
    fmax = gold_forces.abs().max().item()
    print(f"\n[orb] conservative device forces (real units): PCC={pcc:.6f} MAE={mae:.4f} eV/A "
          f"(oracle |F|max {fmax:.4f})")
    # This module isolates the learned conservative path and does not add the host ZBL force.
    # The oracle includes it, but its energy is only 9.5e-8 eV for this Si golden. The dedicated
    # short-contact ZBL parity is covered in test_orb_zbl_forces.py.
    assert pcc > 0.999, pcc
