"""Warm eager Orb-v3 backbone latency on a toy 4-atom golden and a production-scale periodic
supercell.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/bench_orb_profile.py

Requires the goldens from tests/gen_golden_orb.py (--system bulk and --system supercell,
conservative-inf-omat). Falls back gracefully (skips) if a golden is missing.
"""
from __future__ import annotations

from _harness import golden_dir, median_ms

GOLDENS = golden_dir()
SAMPLES, WARMUP = 30, 5


def _build_and_time(golden_path, device, *, fast=False):
    from tt_atom.device import to_dev
    from tt_atom.orb_model import Encoder, AttentionInteractionLayer, OrbGraphContext, host_cutoff
    from tt_atom.orb_weights import OrbWeights
    import ttnn

    gw = OrbWeights.load(golden_path)
    cfg = gw.config
    w = gw.weights
    L = cfg["num_message_passing_steps"]

    node_feat = gw.host("node_feat")
    edge_feat = gw.host("edge_feat")
    senders = gw.inp("senders").long()
    receivers = gw.inp("receivers").long()
    lengths = gw.inp("vectors").norm(dim=-1)
    cutoff = host_cutoff(lengths, r_max=6.0)
    N = node_feat.shape[0]
    E = senders.shape[0]

    enc = Encoder(w, device, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                 latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=fast)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", device,
                                        latent_dim=cfg["latent_dim"], hidden_dim=1024, fast=fast)
              for i in range(L)]
    graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff,
                            num_nodes=N)
    node_dev = to_dev(node_feat, device, ttnn.bfloat16)
    edge_dev = to_dev(edge_feat, device, ttnn.bfloat16)

    def fwd():
        nodes, edges = enc(node_dev, edge_dev)
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        ttnn.synchronize_device(device)
        return nodes

    ms = median_ms(fwd, SAMPLES, WARMUP)
    return N, E, ms


def main():
    from tt_atom.device import open_device
    import ttnn

    # TT_VISIBLE_DEVICES selects which physical cards are visible; the visible set is then
    # re-indexed from zero, so the logical id is always 0 (see tt_atom/batch.py:29 and
    # scripts/release_gate.py:211). Pick the card in the environment, not here.
    device = open_device(0)
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
            N, E, ms = _build_and_time(path, device)
            results.append((label, N, E, ms))
            print(f"{label}: N={N} E={E} warm forward={ms:.3f} ms/call "
                  f"({1000*ms/max(E,1):.4f} us/edge)")

        if len(results) == 2:
            (_, n0, e0, ms0), (_, n1, e1, ms1) = results
            scale_E = e1 / e0
            scale_ms = ms1 / ms0
            print(f"\nedges scaled {scale_E:.1f}x ({e0}->{e1}), latency scaled {scale_ms:.2f}x "
                 f"({ms0:.3f}->{ms1:.3f} ms)")
            print("sub-linear-in-edges latency growth => still dispatch-bound (op count is "
                  "fixed at ~9 ops/layer regardless of N/E, so latency shouldn't scale with "
                  "graph size at all if purely dispatch-bound; a compute-bound regime would "
                  "instead scale roughly linearly with edges)."
                  if scale_ms < scale_E * 0.5 else
                  "latency growth tracks edge count => compute-bound territory reached, "
                  "revisit the custom-kernel question at this scale.")

        print("\n-- bf16 vs bf8 (--fast) weights --")
        bf8_cases = [
            ("conservative toy (4-atom)", GOLDENS / "si_omat_orb.npz"),
            ("conservative production (24-atom)", GOLDENS / "si_supercell_orb.npz"),
            ("direct-20 toy (4-atom)", GOLDENS / "si_omat_orb_direct20.npz"),
        ]
        for label, path in bf8_cases:
            if not path.exists():
                print(f"skip {label}: {path} not found")
                continue
            _, _, ms16 = _build_and_time(path, device, fast=False)
            _, _, ms8 = _build_and_time(path, device, fast=True)
            print(f"{label}: bf16={ms16:.3f} ms bf8={ms8:.3f} ms ({ms16/ms8:.2f}x)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
