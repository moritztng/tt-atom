"""Noise-floor diagnostic for the OrbMol real-weight parity tests.

For each (system, ckpt) golden, run the device port forward (the exact code path the
parity tests use) to get the device forces D, then vs the reference forces R (the golden)
compute:

  - PCC(D, R), MAE, RMSE               -- the measured parity
  - sigma_R = std(R)                    -- the signal magnitude
  - predicted PCC = sigma_R / sqrt(sigma_R^2 + RMSE^2)   -- the PCC you'd get if the
    device error were pure additive zero-mean noise at the measured RMSE (the bf16 floor)
  - corr(R, D - R)                      -- structural-bug indicator: ~0 means the error is
    independent of the signal (noise floor); large |corr| means the error scales with the
    signal (a real bug -- dropped/scaled/misconditioned component)

If predicted PCC tracks measured PCC across all systems and corr(R, D-R) ~ 0, the PCC is
fully explained by the device's own absolute-error floor -- i.e. the gap is precision-bound
(SNR-limited), not a port bug. A genuine open-shell/charge/spin bug would show up as
corr(R, D-R) far from zero AND measured PCC well below the noise-floor prediction.

Run on card 0 with the ttnn env:
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python scripts/orb_omol_noise_floor.py
"""
from __future__ import annotations

import pathlib
import numpy as np
import torch


GOLDEN_DIR = pathlib.Path.home() / ".ttatom_run/goldens_real"
SYSTEMS = ["molecule", "molecule_charged", "molecule_openshell"]
CKPT_TAGS = ["conservative", "direct"]


def _golden_path(system, tag):
    return GOLDEN_DIR / f"{system}_omol_{tag}.npz"


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _load(system, tag):
    from tt_atom.orb_weights import OrbWeights
    return OrbWeights.load(_golden_path(system, tag))


def device_forces(system, tag):
    """Replicate the test's device forward and return (D_forces, R_forces, e_rel_err)."""
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, OrbGraphContext,
                                   EnergyHead, ForceHead, host_cutoff, host_zbl_energy,
                                   host_zbl_forces, host_energy_denormalize,
                                   host_force_denormalize, host_conservative_force_denormalize,
                                   host_charge_spin_embedding, _to_dev)
    from tt_atom import orb_forces
    import ttnn

    gw = _load(system, tag)
    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    latent_dim = cfg["latent_dim"]
    atomic_numbers = gw.inp("atomic_numbers").long()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors = gw.inp("vectors")
    pos = gw.inp("pos").double()
    N = atomic_numbers.shape[0]
    charge, spin = float(gw.inp("charge")[0]), float(gw.inp("spin")[0])
    cond_nodes = host_charge_spin_embedding(w, charge, spin, N, latent_dim)

    device = ttnn.open_device(device_id=0)
    try:
        enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                     latent_dim=latent_dim, hidden_dim=1024)
        node_dev = _to_dev(gw.host("node_feat"), device, ttnn.bfloat16)
        edge_dev = _to_dev(gw.host("edge_feat"), device, ttnn.bfloat16)
        cutoff = host_cutoff(vectors.norm(dim=-1), r_max=6.0)
        graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff,
                                num_nodes=N, cond_nodes=cond_nodes)
        layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device, latent_dim=latent_dim,
                                            hidden_dim=1024) for i in range(L)]
        nodes, edges = enc(node_dev, edge_dev)
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        ehead = EnergyHead(w, device, latent_dim=latent_dim, hidden_dim=1024)
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

        if tag == "direct":
            fhead = ForceHead(w, device, latent_dim=latent_dim, hidden_dim=1024)
            raw_f = ttnn.to_torch(fhead(nodes)).double()
            gnn_forces = host_force_denormalize(
                raw_f, running_mean=w["forces_head.normalizer.bn.running_mean"],
                running_var=w["forces_head.normalizer.bn.running_var"],
            )
            zbl_forces = host_zbl_forces(atomic_numbers, senders, receivers, pos)
            D = (gnn_forces + zbl_forces).numpy()
        else:
            raw_e_f, forces_raw = orb_forces.energy_and_forces(
                enc, layers, ehead, device, pos=pos, senders=senders, receivers=receivers,
                atomic_numbers=atomic_numbers, node_feat=gw.host("node_feat"), r_max=6.0,
                num_bases=cfg["num_bases"], cond_nodes=cond_nodes,
            )
            forces = host_conservative_force_denormalize(
                forces_raw, N, running_var=w["energy_head.normalizer.bn.running_var"],
            )
            zbl_forces = host_zbl_forces(atomic_numbers, senders, receivers, pos)
            D = (forces + zbl_forces).numpy()
        R = gw.out("forces").numpy()
        return D, R, e_rel_err
    finally:
        ttnn.close_device(device)


def main():
    print(f"{'system':<20} {'tag':<13} {'|F|max':>7} {'PCC(D,R)':>9} {'MAE':>8} {'RMSE':>8} "
          f"{'sig_R':>8} {'PCCpred':>8} {'corr(R,e)':>10} {'verdict':>10}")
    print("-" * 110)
    for system in SYSTEMS:
        for tag in CKPT_TAGS:
            D, R, e_rel = device_forces(system, tag)
            fmax = float(np.abs(R).max())
            mae = float(np.abs(D - R).mean())
            rmse = float(np.sqrt(((D - R) ** 2).mean()))
            pcc_dr = _pcc(D, R)
            sig_R = float(R.std())
            pred = sig_R / np.sqrt(sig_R ** 2 + rmse ** 2) if (sig_R + rmse) > 0 else 1.0
            corr_re = _pcc(R, D - R)
            # verdict: noise-floor if |corr(R,e)| small AND pcc_dr >= 0.95*pred
            floor_ok = abs(corr_re) < 0.20 and pcc_dr >= 0.90 * pred
            verdict = "noise-floor" if floor_ok else "STRUCTURAL?"
            print(f"{system:<20} {tag:<13} {fmax:>7.4f} {pcc_dr:>9.4f} {mae:>8.4f} {rmse:>8.4f} "
                  f"{sig_R:>8.4f} {pred:>8.4f} {corr_re:>10.4f} {verdict:>10}   Erel={e_rel:.2e}")


if __name__ == "__main__":
    main()
