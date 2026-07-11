"""Real-weight parity test for orb-v3-direct-20-omat -- the fast, direct-force checkpoint (no
energy-autograd VJP; forces come straight from a per-node ForceHead MLP). Reuses the exact same
``Encoder``/``AttentionInteractionLayer`` device modules as the conservative checkpoint
unmodified (confirms the port's bottom-up pieces are checkpoint-agnostic, not conservative-
specific), and adds the direct-only ``ForceHead`` device path.

    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat \
        --out ~/.ttatom_run/goldens_real/si_omat_orb_direct20.npz
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest tests/test_orb_direct_realweight.py -q -s

Absent the golden bundle the whole module auto-skips.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

REAL_GOLDEN = os.environ.get(
    "TTATOM_ORB_DIRECT_GOLDEN",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat_orb_direct20.npz"),
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"Orb direct-20 golden bundle not found at {REAL_GOLDEN}",
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


def test_direct_end_to_end(gw, device):
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, OrbGraphContext,
                                   EnergyHead, ForceHead, host_cutoff, host_zbl_energy,
                                   host_energy_denormalize, host_force_denormalize, _to_dev)
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
    for i, layer in enumerate(layers):
        nodes, edges = layer(nodes, edges, graph)
        pcc_n = _pcc(ttnn.to_torch(nodes).float(), gw.activation(f"gnn{i}.out0"))
        print(f"[orb-direct20] backbone layer {i}/{L-1} node PCC={pcc_n:.6f}")

    # architecture reuse check: same Encoder/AttentionInteractionLayer classes, different (direct)
    # checkpoint weights and a smaller (max_num_neighbors=20) graph -- no code changes needed.
    final_pcc = _pcc(ttnn.to_torch(nodes).float(), gw.activation(f"gnn{L-1}.out0"))
    print(f"[orb-direct20] final node PCC={final_pcc:.6f}")
    assert final_pcc > 0.99, final_pcc

    # energy (same EnergyHead device path as the conservative checkpoint)
    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    raw_e = ttnn.to_torch(ehead(nodes)).double().view(())
    gnn_energy = host_energy_denormalize(
        raw_e, atomic_numbers, N,
        running_mean=w["energy_head.normalizer.bn.running_mean"],
        running_var=w["energy_head.normalizer.bn.running_var"],
        ref_weight=w["energy_head.reference.linear.weight"].view(-1),
    )
    zbl_energy = host_zbl_energy(atomic_numbers, senders, receivers, vectors)
    total_energy = float(gnn_energy + zbl_energy)
    gold_energy = float(gw.out("energy")[0])
    e_rel_err = abs(total_energy - gold_energy) / abs(gold_energy)
    print(f"[orb-direct20] energy: {total_energy:.6f} eV (oracle {gold_energy:.6f}, "
          f"rel err {e_rel_err:.2e}); ZBL={float(zbl_energy):.2e} (negligible at these bond "
          f"lengths -- envelope cutoff not reached, verified separately)")
    assert e_rel_err < 1e-2, e_rel_err

    # forces: direct prediction, no autograd -- the whole point of this checkpoint
    fhead = ForceHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    raw_f = ttnn.to_torch(fhead(nodes)).double()
    forces = host_force_denormalize(
        raw_f,
        running_mean=w["forces_head.normalizer.bn.running_mean"],
        running_var=w["forces_head.normalizer.bn.running_var"],
    ).numpy()
    # ZBL force contribution is skipped (verified negligible: ZBL energy ~1e-7 eV for this
    # system -- Si-Si bond lengths sit outside the ZBL envelope cutoff, see docs/orb-port.md)
    gold_forces = gw.out("forces").numpy()
    pcc_f = _pcc(forces, gold_forces)
    mae_f = float(np.abs(forces - gold_forces).mean())
    print(f"[orb-direct20] forces PCC={pcc_f:.6f} MAE={mae_f:.4f} eV/A "
          f"(oracle |F|max={np.abs(gold_forces).max():.4f})")
    assert pcc_f > 0.99, pcc_f
