"""Trace vs eager benchmark for Orb-v3's MD / relaxation loop (single card), at production scale.

Extends ``benchmarks/bench_orb_profile.py``'s finding (dispatch-bound: edges scaled 6.2x, latency
1.03x) with the applicable lever it identified: ``tt_atom/orb_trace.py``'s ``OrbTracedEngine``.
Each simulated step jitters ``pos`` slightly (same fixed topology, different geometry) to exercise
the real per-step refresh path, not a degenerate repeated-identical-input replay.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python benchmarks/bench_orb_trace.py

Requires the goldens from tests/gen_golden_orb.py (--system bulk and --system supercell,
conservative-inf-omat). Falls back gracefully (skips) if a golden is missing.
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import torch

GOLDENS = pathlib.Path.home() / ".ttatom_run" / "goldens_real"


def _median_ms(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def _device_only_ms(device, encoder, layers, ehead, graph, node_dev, edge_dev):
    """Pure device forward+backward, no host geometry / autograd -- the same slice the broken
    quick attempt measured (1.28x, toy scale only), so the traced number here is comparable to
    both that number and UMA's own ~2.6x forward-only trace win."""
    import ttnn
    from tt_atom.orb_forces import backbone_bw

    def fwd_bw():
        nodes, edges = encoder(node_dev, edge_dev)
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        ehead(nodes)
        backbone_bw(encoder, layers, ehead, graph)
        ttnn.synchronize_device(device)

    return _median_ms(fwd_bw)


def _bench_one(label, path, device):
    from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead, OrbGraphContext, _to_dev
    from tt_atom.orb_forces import energy_and_forces
    from tt_atom.orb_geometry import host_edge_features
    from tt_atom.orb_trace import OrbTracedEngine
    from tt_atom.orb_weights import OrbWeights
    import ttnn

    gw = OrbWeights.load(path)
    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]

    encoder = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=cfg["latent_dim"], hidden_dim=1024)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024)
              for i in range(L)]
    ehead = EnergyHead(w, device, latent_dim=cfg["latent_dim"], hidden_dim=1024)

    pos0 = gw.inp("pos").float()
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    vectors_gold = gw.inp("vectors").float()
    cell_shift = (vectors_gold - (pos0[receivers] - pos0[senders])).detach()
    atomic_numbers = gw.inp("atomic_numbers").long()
    node_feat = gw.host("node_feat")
    N, E = atomic_numbers.shape[0], senders.shape[0]

    g = torch.Generator().manual_seed(0)
    jitters = [pos0 + 0.01 * torch.randn(pos0.shape, generator=g) for _ in range(25)]
    it = iter(jitters * 2)

    def eager_step():
        pos = next(it)
        return energy_and_forces(encoder, layers, ehead, device, pos=pos, senders=senders,
                                 receivers=receivers, atomic_numbers=atomic_numbers,
                                 node_feat=node_feat, cell_shift=cell_shift)

    it = iter(jitters * 2)
    eager_ms = _median_ms(eager_step)
    E_e, F_e = eager_step()

    eng = OrbTracedEngine(encoder, layers, device, senders=senders, receivers=receivers,
                          atomic_numbers=atomic_numbers, node_feat=node_feat, ehead=ehead,
                          cell_shift=cell_shift)
    eng(pos0)  # capture step (not timed)
    it2 = iter(jitters * 2)

    def traced_step():
        return eng(next(it2))

    traced_ms = _median_ms(traced_step)
    E_t, F_t = traced_step()

    # device-only slice: fixed edge_feat/cutoff (no per-step host geometry/refresh/autograd) --
    # eager runs the same forward+backward fresh each call; traced just replays the captured tid.
    edge_feat0, cutoff0, _ = host_edge_features(pos0, senders, receivers, cell_shift)
    node_dev0 = _to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev0 = _to_dev(edge_feat0.detach().float(), device, ttnn.bfloat16)
    graph0 = OrbGraphContext(device, senders=senders, receivers=receivers,
                             cutoff=cutoff0.detach().float(), num_nodes=N)
    eager_dev_ms = _device_only_ms(device, encoder, layers, ehead, graph0, node_dev0, edge_dev0)
    replay_ms = _median_ms(lambda: ttnn.execute_trace(device, eng.tid, cq_id=0, blocking=True))
    eng.close()

    f_err = (F_t - F_e).abs().max().item()
    print(f"{label}: N={N} E={E} eager={eager_ms:.3f}ms traced={traced_ms:.3f}ms "
          f"speedup={eager_ms / traced_ms:.2f}x  (energy diff {abs(E_t - E_e):.2e} eV, "
          f"force maxdiff {f_err:.2e} eV/A -- different jittered pos than the capture step's "
          f"bit-exactness test, so a small refresh-path diff is expected here, not a bug)")
    print(f"{label}: device-only fwd+bw eager={eager_dev_ms:.3f}ms replay={replay_ms:.3f}ms "
          f"speedup={eager_dev_ms / replay_ms:.2f}x  (comparable to the broken quick attempt's "
          f"1.28x fwd-only number and UMA's own ~2.6x fwd-only trace win)")
    return N, E, eager_ms, traced_ms, eager_dev_ms, replay_ms


def main():
    from tt_atom.device import open_device
    import ttnn

    device = open_device(int(os.environ.get("TT_VISIBLE_DEVICES", "0")), trace_region_size=200_000_000)
    try:
        cases = [
            ("toy (4-atom bulk)", GOLDENS / "si_omat_orb.npz"),
            ("production (24-atom supercell)", GOLDENS / "si_supercell_orb.npz"),
        ]
        results = []
        for label, path in cases:
            if not path.exists():
                print(f"skip {label}: {path} not found")
                continue
            results.append(_bench_one(label, path, device))

        if len(results) == 2:
            (_, _, e0, t0, ed0, r0), (_, _, e1, t1, ed1, r1) = results
            print(f"\nfull-step speedup: toy {e0/t0:.2f}x -> production {e1/t1:.2f}x")
            print(f"device-only speedup: toy {ed0/r0:.2f}x -> production {ed1/r1:.2f}x")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
