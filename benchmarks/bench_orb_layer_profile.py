"""Per-op profile of one Orb-v3 AttentionInteractionLayer (forward + backward VJP) at production
scale, to attribute device time and materialized-DRAM traffic to aggregation / norm / node-update
/ edge-MLP / attention-gate stages -- the "where is the pole now that edge-MLP SiLU BW is fused"
instrument the prior edge-MLP fusion task did not have.

Methodology mirrors benchmarks/bench_orb_edge_mlp.py: real first-layer weights, real 2016-atom
graph (radius_graph on a 6x6x7 Si diamond supercell), bounded-random activations, and an explicit
ttnn.synchronize_device barrier after every op so each op's device latency is measured in
isolation. Barriered timings over-attribute relative to a pipelined trace, but the *share* of time
per stage is the stable signal for finding the next pole. Logical DRAM traffic is computed per
stage from the exact tensor shapes in tt_atom/orb_model.py and tt_atom/orb_forces.py.

    source .source_env.sh
    TT_VISIBLE_DEVICES=0 PYTHONPATH=.:$PYTHONPATH $PYREFENV benchmarks/bench_orb_layer_profile.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --out benchmarks/orb_layer_profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk


def _med(fn, ttnn, device, *, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    xs = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        xs.append((time.perf_counter() - t) * 1e3)
    return float(statistics.median(xs))


def _build_graph(device, *, nx, ny, nz, latent_dim, ttnn):
    from tt_atom.geometry import radius_graph
    from tt_atom.orb_model import OrbGraphContext, host_cutoff

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz)
    N = len(atoms)
    pos0 = torch.tensor(atoms.get_positions(), dtype=torch.float64)
    cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
    edge_index, shift = radius_graph(pos0, 6.0, cell=cell, pbc=[True, True, True])
    src, tgt = edge_index[0], edge_index[1]
    senders, receivers = tgt, src
    lengths = (pos0[receivers] - pos0[senders] + (shift if shift is not None else 0)).norm(dim=-1)
    cutoff = host_cutoff(lengths, r_max=6.0)
    graph = OrbGraphContext(device, senders=senders, receivers=receivers, cutoff=cutoff.detach().float(),
                            num_nodes=N)
    return graph, N, int(senders.shape[0])


def _logical_bytes(shapes, dtype_bytes=2):
    """Sum of tile-padded byte sizes for a list of (rows, width) read/write transfers."""
    total = 0
    for rows, width in shapes:
        rp = ((rows + 31) // 32) * 32
        wp = ((width + 31) // 32) * 32
        total += rp * wp * dtype_bytes
    return total


def _profile_forward(ttnn, device, layer, graph, nodes, edges, *, iters):
    """Replicate AttentionInteractionLayer.__call__ op-by-op with sync barriers."""
    C, N = layer.C, graph.N
    kcfg = layer.kcfg

    def attn_linears():
        return (ttnn.linear(edges, layer.receive_attn_w, bias=layer.receive_attn_b,
                            compute_kernel_config=kcfg),
                ttnn.linear(edges, layer.send_attn_w, bias=layer.send_attn_b,
                            compute_kernel_config=kcfg))

    out = {}
    out["attn_linears"] = _med(attn_linears, ttnn, device, iters=iters)
    ra_lin, sa_lin = attn_linears()

    def sigmoids():
        return ttnn.sigmoid(ra_lin), ttnn.sigmoid(sa_lin)
    out["sigmoids"] = _med(sigmoids, ttnn, device, iters=iters)
    ra_sig, sa_sig = sigmoids()

    def cutoff_mul():
        return (ttnn.multiply(ra_sig, graph.cutoff), ttnn.multiply(sa_sig, graph.cutoff))
    out["cutoff_mul"] = _med(cutoff_mul, ttnn, device, iters=iters)
    receive_attn, send_attn = cutoff_mul()

    nodes_rm = ttnn.to_layout(nodes, ttnn.ROW_MAJOR_LAYOUT)

    def gather_nodes():
        sa = ttnn.to_layout(ttnn.embedding(graph.senders_idx, nodes_rm), ttnn.TILE_LAYOUT)
        ra = ttnn.to_layout(ttnn.embedding(graph.receivers_idx, nodes_rm), ttnn.TILE_LAYOUT)
        return sa, ra
    out["gather_nodes"] = _med(gather_nodes, ttnn, device, iters=iters)
    sent_attrs, recv_attrs = gather_nodes()

    def concat_edge_in():
        return ttnn.concat([edges, sent_attrs, recv_attrs], dim=1)
    out["concat_edge_in"] = _med(concat_edge_in, ttnn, device, iters=iters)
    edge_in = concat_edge_in()

    out["edge_mlp"] = _med(lambda: layer.edge_mlp(edge_in), ttnn, device, iters=iters)
    updated_edges = layer.edge_mlp(edge_in)

    def msg_mul():
        return (ttnn.multiply(updated_edges, send_attn), ttnn.multiply(updated_edges, receive_attn))
    out["msg_mul"] = _med(msg_mul, ttnn, device, iters=iters)
    sent_msg, recv_msg = msg_mul()

    from tt_atom import scatter as _sc
    def segment_sum():
        return (_sc.segment_sum(ttnn, sent_msg, graph.src_gather, graph.Dmax_s, N, C),
                _sc.segment_sum(ttnn, recv_msg, graph.tgt_gather, graph.Dmax_t, N, C))
    out["segment_sum_agg"] = _med(segment_sum, ttnn, device, iters=iters)
    sent_agg, recv_agg = segment_sum()

    def concat_node_in():
        return ttnn.concat([nodes, recv_agg, sent_agg], dim=1)
    out["concat_node_in"] = _med(concat_node_in, ttnn, device, iters=iters)
    node_in = concat_node_in()

    out["node_mlp"] = _med(lambda: layer.node_mlp(node_in), ttnn, device, iters=iters)
    updated_nodes = layer.node_mlp(node_in)

    def residual_adds():
        return ttnn.add(nodes, updated_nodes), ttnn.add(edges, updated_edges)
    out["residual_adds"] = _med(residual_adds, ttnn, device, iters=iters)

    # restore the layer's backward caches (the real __call__ populates these)
    layer._cache = dict(ra_lin=ra_lin, sa_lin=sa_lin, ra_sig=ra_sig, sa_sig=sa_sig,
                        receive_attn=receive_attn, send_attn=send_attn,
                        updated_edges=updated_edges)
    return out


def _profile_backward(ttnn, device, layer, graph, g_nodes_out, g_edges_out, *, iters):
    """Replicate orb_forces.attn_layer_bw op-by-op with sync barriers."""
    from tt_atom import scatter as _sc
    from tt_atom.orb_forces import _mm, mlpnorm_bw

    C, N, E = layer.C, graph.N, graph.E
    c = layer._cache
    out = {}

    out["node_mlp_bw"] = _med(lambda: mlpnorm_bw(layer.node_mlp, g_nodes_out), ttnn, device, iters=iters)
    g_node_in = mlpnorm_bw(layer.node_mlp, g_nodes_out)
    g_nodes_direct = ttnn.slice(g_node_in, [0, 0], [N, C])
    g_recv_agg = ttnn.slice(g_node_in, [0, C], [N, 2 * C])
    g_sent_agg = ttnn.slice(g_node_in, [0, 2 * C], [N, 3 * C])

    def edge_gather_adj():
        gr = ttnn.to_layout(
            ttnn.embedding(graph.receivers_idx, ttnn.to_layout(g_recv_agg, ttnn.ROW_MAJOR_LAYOUT)),
            ttnn.TILE_LAYOUT)
        gs = ttnn.to_layout(
            ttnn.embedding(graph.senders_idx, ttnn.to_layout(g_sent_agg, ttnn.ROW_MAJOR_LAYOUT)),
            ttnn.TILE_LAYOUT)
        return gr, gs
    out["edge_gather_adj"] = _med(edge_gather_adj, ttnn, device, iters=iters)
    g_recv_msg, g_sent_msg = edge_gather_adj()

    def msg_mul_bw():
        g_ue = ttnn.add(g_edges_out, ttnn.add(
            ttnn.multiply(g_recv_msg, c["receive_attn"]), ttnn.multiply(g_sent_msg, c["send_attn"])))
        g_ra = ttnn.sum(ttnn.multiply(g_recv_msg, c["updated_edges"]), dim=1, keepdim=True)
        g_sa = ttnn.sum(ttnn.multiply(g_sent_msg, c["updated_edges"]), dim=1, keepdim=True)
        return g_ue, g_ra, g_sa
    out["msg_mul_bw"] = _med(msg_mul_bw, ttnn, device, iters=iters)
    g_updated_edges, g_receive_attn, g_send_attn = msg_mul_bw()

    out["edge_mlp_bw"] = _med(lambda: mlpnorm_bw(layer.edge_mlp, g_updated_edges), ttnn, device, iters=iters)
    g_edge_in = mlpnorm_bw(layer.edge_mlp, g_updated_edges)
    g_edges_direct = ttnn.slice(g_edge_in, [0, 0], [E, C])
    g_sent_attrs = ttnn.slice(g_edge_in, [0, C], [E, 2 * C])
    g_recv_attrs = ttnn.slice(g_edge_in, [0, 2 * C], [E, 3 * C])

    def node_scatter_adj():
        return (_sc.segment_sum(ttnn, g_sent_attrs, graph.src_gather, graph.Dmax_s, N, C),
                _sc.segment_sum(ttnn, g_recv_attrs, graph.tgt_gather, graph.Dmax_t, N, C))
    out["node_scatter_adj"] = _med(node_scatter_adj, ttnn, device, iters=iters)
    g_nodes_from_sent, g_nodes_from_recv = node_scatter_adj()

    def attn_sigmoid_bw():
        g_ra_sig = ttnn.multiply(g_receive_attn, graph.cutoff)
        g_sa_sig = ttnn.multiply(g_send_attn, graph.cutoff)
        g_cutoff = ttnn.add(ttnn.multiply(g_receive_attn, c["ra_sig"]),
                            ttnn.multiply(g_send_attn, c["sa_sig"]))
        g_ra_lin = ttnn.sigmoid_bw(g_ra_sig, c["ra_lin"])[0]
        g_sa_lin = ttnn.sigmoid_bw(g_sa_sig, c["sa_lin"])[0]
        g_ea = ttnn.add(_mm(ttnn, g_ra_lin, layer.receive_attn_w, layer.kcfg),
                        _mm(ttnn, g_sa_lin, layer.send_attn_w, layer.kcfg))
        return g_cutoff, g_ea
    out["attn_sigmoid_bw"] = _med(attn_sigmoid_bw, ttnn, device, iters=iters)
    g_cutoff, g_edges_from_attn = attn_sigmoid_bw()

    # Only the residual add chain itself (the two 4-way / 3-way sums) -- sub-stages above are
    # already timed separately, so this must NOT re-run them (would double-count).
    def residual_adds_bw():
        g_nodes_in = ttnn.add(g_nodes_out, ttnn.add(g_nodes_direct,
                                                    ttnn.add(g_nodes_from_sent, g_nodes_from_recv)))
        g_edges_in = ttnn.add(g_edges_out, ttnn.add(g_edges_direct, g_edges_from_attn))
        return g_nodes_in, g_edges_in
    out["residual_adds_bw"] = _med(residual_adds_bw, ttnn, device, iters=iters)
    return out


def _traffic_forward(N, E, C, H, Dmax_s, Dmax_t, dtype_bytes=2):
    """Logical materialized activation traffic per stage (reads + writes of every op's
    inputs/outputs; weights counted once per matmul). Mirrors the exact forward op sequence."""
    b = dtype_bytes
    tw = {
        "attn_linears": b * (E * C + C * 1 + E * 1) * 2,          # two [E,C]@[C,1]->[E,1]
        "sigmoids": b * (E * 1 * 2) * 2,                          # two sigmoid [E,1] read+write
        "cutoff_mul": b * (E * 1 * 2) * 2,                        # two mul [E,1]
        "gather_nodes": b * (E * C) * 2 * 2,                      # two embedding gathers [E,C]
        "concat_edge_in": b * (E * 3 * C) * 3,                    # 3 inputs read + 1 out write
        # edge_mlp: linear0 [E,3C]@[3C,H]->[E,H]; silu; linear1 [E,H]@[H,H]->[E,H]; silu;
        # linear2 [E,H]@[H,C]->[E,C]; rmsnorm (sq,mean,add,rsqrt,scale,affine ~ 8 passes over [E,C])
        "edge_mlp": b * ((E * 3 * C + 3 * C * H + E * H)
                         + (2 * E * H)
                         + (E * H + H * H + E * H)
                         + (2 * E * H)
                         + (E * H + H * C + E * C)
                         + (8 * E * C)),
        "msg_mul": b * (E * C * 3) * 2,                           # two mul: out,updated,attn read + out
        "segment_sum_agg": b * ((E * C + N * Dmax_s * C + N * C)   # sent: msg read, gather [N*Dmax,C], sum out
                                 + (E * C + N * Dmax_t * C + N * C)),
        "concat_node_in": b * (N * 3 * C) * 3,
        "node_mlp": b * ((N * 3 * C + 3 * C * H + N * H)
                         + (2 * N * H)
                         + (N * H + H * H + N * H)
                         + (2 * N * H)
                         + (N * H + H * C + N * C)
                         + (8 * N * C)),
        "residual_adds": b * (N * C + E * C) * 2 * 2,             # two adds
    }
    return tw


def _traffic_backward(N, E, C, H, Dmax_s, Dmax_t, dtype_bytes=2):
    b = dtype_bytes
    tw = {
        # mlpnorm_bw: rmsnorm_bw(~9 passes [N,C]) + 3 transpose-matmuls + 2 silu_bw
        "node_mlp_bw": b * ((9 * N * C)
                            + (N * C + H * C + N * H)
                            + (4 * N * H)            # silu_bw read g + read x + write out (~4)
                            + (N * H + H * H + N * H)
                            + (4 * N * H)
                            + (N * H + 3 * C * H + N * 3 * C)),
        "edge_gather_adj": b * (E * C) * 2 * 2,                  # two embedding gathers [E,C]
        "msg_mul_bw": b * (E * C * 4) * 2,                       # adds+muls over [E,C]
        "edge_mlp_bw": b * ((9 * E * C)
                            + (E * C + H * C + E * H)
                            + (4 * E * H)
                            + (E * H + H * H + E * H)
                            + (4 * E * H)
                            + (E * H + 3 * C * H + E * 3 * C)),
        "node_scatter_adj": b * ((E * C + N * Dmax_s * C + N * C)
                                 + (E * C + N * Dmax_t * C + N * C)),
        "attn_sigmoid_bw": b * ((E * 1 * 4) * 2 + (E * C) + (E * 1 + C * 1 + E * 1) * 2),
        "residual_adds_bw": b * (N * C + E * C) * 2 * 3,
    }
    return tw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--nz", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", default="benchmarks/orb_layer_profile.json")
    args = ap.parse_args()

    import ttnn
    from tt_atom.device import open_device, orb_fused_silu_bw
    from tt_atom.orb_model import AttentionInteractionLayer, _to_dev
    from tt_atom.orb_weights import OrbWeights

    torch.manual_seed(17)
    device = open_device(0, trace_region_size=400_000_000)
    rec = {}
    try:
        gw = OrbWeights.load(args.weights)
        cfg, w = gw.config, gw.weights
        C, H = cfg["latent_dim"], 1024
        L = cfg["num_message_passing_steps"]
        layer = AttentionInteractionLayer(w, "gnn_stacks.0", device, latent_dim=C, hidden_dim=H)
        graph, N, E = _build_graph(device, nx=args.nx, ny=args.ny, nz=args.nz,
                                   latent_dim=C, ttnn=ttnn)
        Dmax_s, Dmax_t = int(graph.Dmax_s), int(graph.Dmax_t)

        # Bounded-random activations representative of normalized residual features.
        nodes = _to_dev(torch.randn(N, C, dtype=torch.bfloat16) * 0.25, device, ttnn.bfloat16)
        edges = _to_dev(torch.randn(E, C, dtype=torch.bfloat16) * 0.25, device, ttnn.bfloat16)

        # Run the real forward once to warm caches and populate layer._cache exactly.
        layer(nodes, edges, graph)
        ttnn.synchronize_device(device)

        fwd = _profile_forward(ttnn, device, layer, graph, nodes, edges, iters=args.iters)
        # Re-populate cache (profile_forward ran sub-ops, not the real __call__).
        layer(nodes, edges, graph)
        g_nodes = _to_dev(torch.randn(N, C, dtype=torch.bfloat16) * 0.125, device, ttnn.bfloat16)
        g_edges = _to_dev(torch.randn(E, C, dtype=torch.bfloat16) * 0.125, device, ttnn.bfloat16)
        bwd = _profile_backward(ttnn, device, layer, graph, g_nodes, g_edges, iters=args.iters)

        tf = _traffic_forward(N, E, C, H, Dmax_s, Dmax_t)
        tb = _traffic_backward(N, E, C, H, Dmax_s, Dmax_t)

        def _pack(stage_ms, traffic):
            tot_ms = sum(stage_ms.values())
            return {
                "ms": {k: round(v, 4) for k, v in stage_ms.items()},
                "share_pct": {k: round(100 * v / tot_ms, 1) for k, v in stage_ms.items()},
                "total_ms": round(tot_ms, 4),
                "logical_bytes": {k: int(v) for k, v in traffic.items()},
                "logical_bytes_total": int(sum(traffic.values())),
            }

        rec = {
            "N": N, "E": E, "C": C, "H": H, "L": L,
            "Dmax_s": Dmax_s, "Dmax_t": Dmax_t,
            "orb_fused_silu_bw": orb_fused_silu_bw(),
            "forward": _pack(fwd, tf),
            "backward": _pack(bwd, tb),
            "forward_plus_backward_ms": round(sum(fwd.values()) + sum(bwd.values()), 4),
        }
        print(f"\n=== layer 0 @ N={N} E={E} Dmax={Dmax_s}/{Dmax_t} "
              f"fused_silu_bw={rec['orb_fused_silu_bw']} ===", flush=True)
        print(f"forward  total {rec['forward']['total_ms']:.3f} ms:", flush=True)
        for k, v in sorted(fwd.items(), key=lambda kv: -kv[1]):
            print(f"  {k:18s} {v:8.3f} ms  {100*v/rec['forward']['total_ms']:5.1f}%  "
                  f"{tf[k]/1e6:7.1f} MB logical", flush=True)
        print(f"backward total {rec['backward']['total_ms']:.3f} ms:", flush=True)
        for k, v in sorted(bwd.items(), key=lambda kv: -kv[1]):
            print(f"  {k:18s} {v:8.3f} ms  {100*v/rec['backward']['total_ms']:5.1f}%  "
                  f"{tb[k]/1e6:7.1f} MB logical", flush=True)
    finally:
        ttnn.close_device(device)

    payload = {
        "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
        "model": "orb-v3-conservative-inf-omat",
        "scope": "first interaction layer (gnn_stacks.0) forward + backward VJP, per-op with sync barriers",
        "method": "eager per-op with ttnn.synchronize_device after every op; share-of-total is the signal",
        "precision": "bf16",
        "orb_fused_silu_bw": rec.get("orb_fused_silu_bw"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "record": rec,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
