"""Orb-v3 ``--fast`` accuracy check for both checkpoints.

Weight-only bf8 was a measured dead end.  The accelerated mode also stores the two 1024-wide
hidden MLP activations in bf8 while keeping the 256-wide residual stream bf16 and matmul
accumulation fp32.  This test is the real-weight release gate for that numerics policy.

``tt_atom/device.py`` keeps HiFi4 + fp32 destination accumulation.  Only weights and hidden
MLP activations use bf8; outputs entering residual connections stay bf16.

Bar: UMA's own real-weight ``--fast`` bar (commit 836af75: "forward+forces PCC 0.99997, no
accuracy loss") -- same energy/force/stress thresholds as the existing bf16 real-weight tests
(``test_orb_forces_realweight.py``, ``test_orb_direct_realweight.py``, ``test_orb_stress_realweight.py``).

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_bf8_fast.py -q -s

Absent the golden bundle(s) each test auto-skips independently.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

CONSERVATIVE_GOLDEN = os.environ.get(
    "TTATOM_ORB_GOLDEN", str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat_orb.npz")
)
DIRECT_GOLDEN = os.environ.get(
    "TTATOM_ORB_DIRECT_GOLDEN",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat_orb_direct20.npz"),
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])




@pytest.mark.skipif(not pathlib.Path(CONSERVATIVE_GOLDEN).exists(),
                    reason=f"Orb golden bundle not found at {CONSERVATIVE_GOLDEN}")
def test_conservative_fast_energy_forces_stress(device):
    from tt_atom.orb_weights import OrbWeights
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, EnergyHead,
                                   host_energy_denormalize, host_zbl_energy,
                                   host_conservative_force_denormalize, host_conservative_stress)
    from tt_atom.orb_forces import energy_and_forces

    gw = OrbWeights.load(CONSERVATIVE_GOLDEN)
    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
              for i in range(L)]
    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)

    pos = gw.inp("pos").float()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").float()
    cell_shift = (vectors_gold - (pos[receivers] - pos[senders])).detach()
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")
    cell = gw.inp("cell").squeeze(0)
    N = atomic_numbers.shape[0]

    raw_pred, raw_forces, virial_raw = energy_and_forces(
        encoder, layers, ehead, device, pos=pos, senders=senders, receivers=receivers,
        atomic_numbers=atomic_numbers, node_feat=node_feat, cell_shift=cell_shift,
        compute_stress=True)

    import torch as _torch
    gnn_energy = host_energy_denormalize(
        _torch.tensor(raw_pred), atomic_numbers, N,
        running_mean=w["energy_head.normalizer.bn.running_mean"],
        running_var=w["energy_head.normalizer.bn.running_var"],
        ref_weight=w["energy_head.reference.linear.weight"].view(-1))
    zbl_energy = host_zbl_energy(atomic_numbers, senders, receivers, vectors_gold)
    total_energy = float(gnn_energy + zbl_energy)
    gold_energy = float(gw.out("energy")[0])
    e_rel_err = abs(total_energy - gold_energy) / abs(gold_energy)
    print(f"\n[orb-fast] conservative energy: {total_energy:.6f} eV (oracle {gold_energy:.6f}, "
          f"rel err {e_rel_err:.2e})")
    assert e_rel_err < 1e-2, e_rel_err

    forces = host_conservative_force_denormalize(
        raw_forces, N, running_var=w["energy_head.normalizer.bn.running_var"])
    gold_forces = gw.out("forces").double()
    f_pcc = _pcc(forces.detach(), gold_forces)
    f_mae = (forces.detach() - gold_forces).abs().mean().item()
    print(f"[orb-fast] conservative forces PCC={f_pcc:.6f} MAE={f_mae:.4f} eV/A "
          f"(oracle |F|max {gold_forces.abs().max().item():.4f})")
    assert f_pcc > 0.999, f_pcc

    stress = host_conservative_stress(
        virial_raw, N, cell, running_var=w["energy_head.normalizer.bn.running_var"])
    gold_stress = gw.out("stress")[0].double()
    s_pcc = _pcc(stress, gold_stress)
    s_err = (stress - gold_stress).abs().max().item()
    print(f"[orb-fast] conservative stress PCC={s_pcc:.6f} max abs err={s_err:.2e}")
    assert s_pcc > 0.999, s_pcc
    assert s_err < 5e-3, s_err


@pytest.mark.skipif(not pathlib.Path(DIRECT_GOLDEN).exists(),
                    reason=f"Orb direct-20 golden bundle not found at {DIRECT_GOLDEN}")
def test_direct_fast_energy_forces(device):
    from tt_atom.orb_weights import OrbWeights
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, OrbGraphContext,
                                   EnergyHead, ForceHead, host_cutoff, host_zbl_energy,
                                   host_energy_denormalize, host_force_denormalize, _to_dev)
    import ttnn

    gw = OrbWeights.load(DIRECT_GOLDEN)
    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                 latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
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
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
              for i in range(L)]
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)

    final_pcc = _pcc(ttnn.to_torch(nodes).float(), gw.activation(f"gnn{L-1}.out0"))
    print(f"\n[orb-direct20-fast] final node PCC={final_pcc:.6f}")
    assert final_pcc > 0.99, final_pcc

    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
    raw_e = ttnn.to_torch(ehead(nodes)).double().view(())
    gnn_energy = host_energy_denormalize(
        raw_e, atomic_numbers, N,
        running_mean=w["energy_head.normalizer.bn.running_mean"],
        running_var=w["energy_head.normalizer.bn.running_var"],
        ref_weight=w["energy_head.reference.linear.weight"].view(-1))
    zbl_energy = host_zbl_energy(atomic_numbers, senders, receivers, vectors)
    total_energy = float(gnn_energy + zbl_energy)
    gold_energy = float(gw.out("energy")[0])
    e_rel_err = abs(total_energy - gold_energy) / abs(gold_energy)
    print(f"[orb-direct20-fast] energy: {total_energy:.6f} eV (oracle {gold_energy:.6f}, "
          f"rel err {e_rel_err:.2e})")
    assert e_rel_err < 1e-2, e_rel_err

    fhead = ForceHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=True)
    raw_f = ttnn.to_torch(fhead(nodes)).double()
    forces = host_force_denormalize(
        raw_f,
        running_mean=w["forces_head.normalizer.bn.running_mean"],
        running_var=w["forces_head.normalizer.bn.running_var"]).numpy()
    gold_forces = gw.out("forces").numpy()
    pcc_f = _pcc(forces, gold_forces)
    mae_f = float(np.abs(forces - gold_forces).mean())
    print(f"[orb-direct20-fast] forces PCC={pcc_f:.6f} MAE={mae_f:.4f} eV/A "
          f"(oracle |F|max={np.abs(gold_forces).max():.4f})")
    assert pcc_f > 0.99, pcc_f
