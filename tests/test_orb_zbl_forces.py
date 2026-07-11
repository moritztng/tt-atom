"""Real-weight parity test for ZBL pair-repulsion forces (``dV_ZBL/dr``, ``tt_atom/orb_model.py``
``host_zbl_forces``) -- the item flagged open in ``docs/orb-port.md``: the existing bulk-Si
golden's nearest-neighbor distance sits just outside the ZBL envelope cutoff (measured negligible,
~1e-7 eV), so it never exercises this term. This test uses a dedicated short-contact golden
(``tests/gen_golden_orb.py --system short_contact``: two Si atoms 1.4 A apart, well inside the
~2.2 A covalent-radii-sum cutoff) where ZBL is ~1.3% of the total energy and dominates the force
magnitude (oracle |F|max ~51 eV/A vs ~2.5 eV/A for the bulk golden).

Verifies ``orb-v3-direct-20-omat``'s *total* force = device ``ForceHead`` MLP prediction (no ZBL
contribution baked in) + ``host_zbl_forces`` (host ``torch.autograd`` on the closed-form ZBL
energy) against the real orb-models oracle total force.

    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat \
        --system short_contact \
        --out ~/.ttatom_run/goldens_real/si_short_contact_orb_direct20.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_zbl_forces.py -q -s

Absent the golden bundle the whole module auto-skips.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

REAL_GOLDEN = os.environ.get(
    "TTATOM_ORB_SHORT_CONTACT_GOLDEN",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_short_contact_orb_direct20.npz"),
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"Orb short-contact golden bundle not found at {REAL_GOLDEN}",
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



def test_zbl_energy_is_non_negligible_here(gw):
    """Sanity gate: confirm this golden actually exercises the ZBL term (unlike the bulk-Si
    golden's ~1e-7 eV), else the force test below wouldn't be testing anything."""
    from tt_atom.orb_model import host_zbl_energy

    senders, receivers = gw.inp("senders").long(), gw.inp("receivers").long()
    atomic_numbers = gw.inp("atomic_numbers").long()
    vectors = gw.inp("vectors")
    zbl_e = float(host_zbl_energy(atomic_numbers, senders, receivers, vectors))
    total_e = float(gw.out("energy")[0])
    print(f"\n[orb] short-contact ZBL energy={zbl_e:.4f} eV ({100*zbl_e/total_e:.1f}% of total "
          f"{total_e:.4f} eV) -- vs ~1e-7 eV for the bulk-Si golden")
    assert abs(zbl_e) > 1e-3


def test_zbl_forces_match_finite_difference(gw):
    """Independent ground truth (no orb-models/device dependency): dV_ZBL/dr via central finite
    differences on the same closed-form ``host_zbl_energy``."""
    from tt_atom.orb_model import host_zbl_energy, host_zbl_forces

    senders, receivers = gw.inp("senders").long(), gw.inp("receivers").long()
    atomic_numbers = gw.inp("atomic_numbers").long()
    pos = gw.inp("pos").double()
    vectors_gold = gw.inp("vectors").double()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()

    forces = host_zbl_forces(atomic_numbers, senders, receivers, pos, cell_shift)

    eps = 1e-5
    fd = torch.zeros_like(pos)
    for n in range(pos.shape[0]):
        for d in range(3):
            for sign, out in ((1.0, "p"), (-1.0, "m")):
                pp = pos.clone()
                pp[n, d] += sign * eps
                vp = pp[receivers] - pp[senders] + cell_shift
                e = host_zbl_energy(atomic_numbers, senders, receivers, vp)
                if out == "p":
                    ep = e
                else:
                    em = e
            fd[n, d] = -(ep - em) / (2 * eps)

    err = (forces - fd).abs().max().item()
    print(f"\n[orb] host_zbl_forces vs finite-difference: max abs err={err:.2e}")
    assert err < 1e-6


def test_direct20_total_force_with_zbl(gw, device):
    """Device ``ForceHead`` (no ZBL) + host ``host_zbl_forces`` (ZBL) vs the real oracle total
    force -- the previously-skipped term, now exercised on a golden where it's non-negligible."""
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, OrbGraphContext,
                                   ForceHead, host_cutoff, host_force_denormalize,
                                   host_zbl_forces, _to_dev)
    import ttnn

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                 latent_dim=cfg["latent_dim"], hidden_dim=1024)
    node_dev = _to_dev(gw.host("node_feat"), device, ttnn.bfloat16)
    edge_dev = _to_dev(gw.host("edge_feat"), device, ttnn.bfloat16)
    nodes, edges = enc(node_dev, edge_dev)

    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors = gw.inp("vectors")
    lengths = vectors.norm(dim=-1)
    cutoff = host_cutoff(lengths, r_max=6.0)
    atomic_numbers = gw.inp("atomic_numbers").long()
    N = atomic_numbers.shape[0]
    graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff, num_nodes=N)

    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)

    fhead = ForceHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    raw_f = ttnn.to_torch(fhead(nodes)).double()
    gnn_forces = host_force_denormalize(
        raw_f,
        running_mean=w["forces_head.normalizer.bn.running_mean"],
        running_var=w["forces_head.normalizer.bn.running_var"],
    )

    pos = gw.inp("pos").double()
    cell_shift = (vectors.double() - (pos[receivers] - pos[senders])).detach()
    zbl_forces = host_zbl_forces(atomic_numbers, senders, receivers, pos, cell_shift)

    total_forces = (gnn_forces + zbl_forces).numpy()
    gnn_only = gnn_forces.numpy()
    gold_forces = gw.out("forces").numpy()

    pcc_with = _pcc(total_forces, gold_forces)
    pcc_without = _pcc(gnn_only, gold_forces)
    mae_with = float(np.abs(total_forces - gold_forces).mean())
    mae_without = float(np.abs(gnn_only - gold_forces).mean())
    print(f"\n[orb] short-contact total force: PCC(GNN+ZBL)={pcc_with:.6f} MAE={mae_with:.4f} "
          f"eV/A  vs  PCC(GNN only)={pcc_without:.6f} MAE={mae_without:.4f} eV/A "
          f"(oracle |F|max={np.abs(gold_forces).max():.4f})")
    # adding the ZBL force must not make things worse than omitting it, and should clear a
    # reasonable bar given ForceHead itself is only approximate at this out-of-distribution
    # (deliberately unphysical) short contact.
    assert mae_with <= mae_without
    assert pcc_with > 0.9, pcc_with
