"""Real-weight parity test for stress (``docs/orb-port.md`` Open item), both checkpoints.

Conservative (``orb-v3-conservative-inf-omat``): same displacement-gradient pattern as forces
(``tt_atom/forces.py``'s ``strain`` leaf) -- ``tt_atom/orb_geometry.py``'s ``host_edge_features``
takes an optional symmetric ``strain`` that scales the edge vectors (``r' = r(I + sym(strain))``,
Orb's own ``base.create_and_apply_stress_displacement`` uses the identical convention), and
``tt_atom/orb_forces.energy_and_forces(..., compute_stress=True)`` returns the extra virial
``dE/dstrain`` alongside forces from the same backward pass -- no new device VJP.

Direct (``orb-v3-direct-20-omat``): a dedicated ``StressHead`` device MLP (same 2-Linear shape as
``EnergyHead``, mean-aggregated node features -> Voigt-6, with two SEPARATE diag/off-diag
``ScalarNormalizer``s) -- needs the golden regenerated to capture ``stress_head`` weights (the
original ``gen_golden_orb.py`` only saved ``energy_head``/``forces_head``/``pair_repulsion``).

    ~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat \
        --out ~/.ttatom_run/goldens_real/si_omat_orb_direct20.npz   # regenerate: now has stress_head
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
        tests/test_orb_stress_realweight.py -q -s

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


@pytest.fixture(scope="module")
def device():
    from tt_atom.device import open_device
    import ttnn

    dev = open_device(int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    yield dev
    ttnn.close_device(dev)


@pytest.mark.skipif(not pathlib.Path(CONSERVATIVE_GOLDEN).exists(),
                    reason=f"Orb golden bundle not found at {CONSERVATIVE_GOLDEN}")
def test_conservative_stress(device):
    from tt_atom.orb_weights import OrbWeights
    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, EnergyHead,
                                   host_conservative_stress)
    from tt_atom.orb_forces import energy_and_forces

    gw = OrbWeights.load(CONSERVATIVE_GOLDEN)
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
    cell = gw.inp("cell").squeeze(0)
    N = atomic_numbers.shape[0]

    _, _, virial_raw = energy_and_forces(
        encoder, layers, ehead, device, pos=pos, senders=senders, receivers=receivers,
        atomic_numbers=atomic_numbers, node_feat=node_feat, cell_shift=cell_shift,
        compute_stress=True)

    stress = host_conservative_stress(
        virial_raw, N, cell, running_var=w["energy_head.normalizer.bn.running_var"])
    gold_stress = gw.out("stress")[0].double()

    pcc = _pcc(stress, gold_stress)
    err = (stress - gold_stress).abs().max().item()
    print(f"\n[orb] conservative stress (Voigt-6): PCC={pcc:.6f} max abs err={err:.2e}")
    print(f"  device: {stress.tolist()}")
    print(f"  oracle: {gold_stress.tolist()}")
    assert pcc > 0.999, pcc
    assert err < 5e-3, err


@pytest.mark.skipif(not pathlib.Path(DIRECT_GOLDEN).exists(),
                    reason=f"Orb direct-20 golden bundle not found at {DIRECT_GOLDEN}")
def test_direct_stress_head(device):
    from tt_atom.orb_weights import OrbWeights

    gw = OrbWeights.load(DIRECT_GOLDEN)
    if not gw.has("w@stress_head.mlp.NN-0.weight"):
        pytest.skip("golden predates stress_head weight capture -- regenerate with "
                    "gen_golden_orb.py (see module docstring)")

    from tt_atom.orb_model import (Encoder, AttentionInteractionLayer, OrbGraphContext,
                                   StressHead, host_cutoff, host_stress_denormalize, _to_dev)
    import ttnn

    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]
    encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=cfg["latent_dim"], hidden_dim=1024)
    node_dev = _to_dev(gw.host("node_feat"), device, ttnn.bfloat16)
    edge_dev = _to_dev(gw.host("edge_feat"), device, ttnn.bfloat16)
    nodes, edges = encoder(node_dev, edge_dev)

    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors = gw.inp("vectors")
    lengths = vectors.norm(dim=-1)
    cutoff = host_cutoff(lengths, r_max=6.0)
    N = gw.inp("atomic_numbers").shape[0]
    graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff, num_nodes=N)

    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    for layer in layers:
        nodes, edges = layer(nodes, edges, graph)

    shead = StressHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
    raw = ttnn.to_torch(shead(nodes)).double()
    stress = host_stress_denormalize(
        raw,
        diag_mean=w["stress_head.diag_normalizer.bn.running_mean"],
        diag_var=w["stress_head.diag_normalizer.bn.running_var"],
        offdiag_mean=w["stress_head.offdiag_normalizer.bn.running_mean"],
        offdiag_var=w["stress_head.offdiag_normalizer.bn.running_var"])

    gold_stress = gw.out("stress")[0].double()
    pcc = _pcc(stress, gold_stress)
    err = (stress - gold_stress).abs().max().item()
    print(f"\n[orb-direct20] StressHead (Voigt-6): PCC={pcc:.6f} max abs err={err:.2e}")
    print(f"  device: {stress.tolist()}")
    print(f"  oracle: {gold_stress.tolist()}")
    assert pcc > 0.99, pcc
