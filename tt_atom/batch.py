"""Multi-card throughput: fan independent systems across all cards, one process per card.

The eSEN/eSCN-MD (UMA) or Orb-v3/OrbMol evaluation of one system is independent of every other, so
throughput scales by running one worker process per Tenstorrent card (each pinned with
``TT_VISIBLE_DEVICES`` so it owns exactly one device) with the model + weights resident on that
card. The parent streams systems to a shared queue and the workers pull, evaluate, and return
energies — embarrassingly parallel, so aggregate throughput is the sum across cards.

``ttnn`` is imported only *inside* the worker (after the device is pinned); the parent never
touches a device, which is what keeps the fan-out deadlock-free.

The worker dispatches on the loaded weight bundle's family — UMA bundles carry an eSCN-MD
``config`` (``sphere_channels``), Orb bundles an MPNN ``config`` (``num_message_passing_steps``) —
the same family split ``tt_atom.auto`` exposes by name. Pointing ``MultiCard`` at an Orb weights
file builds the Orb backbone (``OrbWeights`` + ``Encoder``/``AttentionInteractionLayer``/
``EnergyHead``); pointing it at a UMA bundle builds the eSCN-MD ``Backbone`` exactly as before.
"""
from __future__ import annotations

import multiprocessing as mp


def _worker(device_id, weights_path, fast, in_q, out_q):
    import json
    import os
    import pathlib

    os.environ["TT_VISIBLE_DEVICES"] = str(device_id)          # pin one card -> it is device 0
    # one host thread per worker: the host geometry (torch) otherwise grabs every core, so N
    # workers oversubscribe the CPU and throttle each other (4-card went *slower* than 1).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    import numpy as np

    npz = np.load(pathlib.Path(weights_path))
    cfg = json.loads(bytes(npz["config"]).decode())
    if "sphere_channels" in cfg:                               # UMA eSCN-MD bundle
        from .weights import WeightBundle

        _run_uma(WeightBundle(npz), cfg, fast, device_id, in_q, out_q)
    else:                                                      # Orb MPNN bundle
        from .orb_weights import OrbWeights

        _run_orb(OrbWeights(npz), cfg, fast, device_id, in_q, out_q)


def _run_uma(b, cfg, fast, device_id, in_q, out_q):
    import torch

    from .device import open_device
    from .geometry import HostGeometry, csd_embedding, radius_graph
    from .model import Backbone, GraphContext
    import ttnn

    w = b.weights
    C = cfg["sphere_channels"]
    dev = open_device(0)
    bb = Backbone(w, dev, cfg, b.to_grid_mat, b.from_grid_mat, fast=fast)
    geo = HostGeometry(w, cfg, b.to_m, b.gauss_offset, b.gauss_coeff, gamma=0.0)
    out_q.put(("ready", device_id))

    while True:
        job = in_q.get()
        if job is None:
            break
        idx, pos_np, Z_np = job
        pos = torch.tensor(pos_np, dtype=torch.float32)
        Z = torch.tensor(Z_np)
        ei, _ = radius_graph(pos, cfg["cutoff"])
        N, E = Z.shape[0], ei.shape[1]
        se = csd_embedding(w, torch.tensor([0.0]), torch.tensor([0.0]), C)[torch.zeros(N, dtype=torch.long)]
        t = geo(pos, Z, ei, se)
        graph = GraphContext(dev, edge_index=ei, wigner=t["wigner"].detach(),
                             wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
                             edge_envelope=t["edge_envelope"].detach(), num_nodes=N, fast=fast)
        se3 = ttnn.from_torch(se.reshape(N, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        xi = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        _, energy = bb(xi, graph, se3)
        out_q.put((idx, float(ttnn.to_torch(energy).reshape(-1)[0]), E))

    ttnn.close_device(dev)


def _run_orb(b, cfg, fast, device_id, in_q, out_q):
    """Energy-only Orb forward, mirroring ``OrbCalculator.calculate``'s direct/conservative
    forward path (the two share the same encoder/layers/``EnergyHead``; only forces differ, and
    ``MultiCard`` returns energies). The systems come in as ``(pos, Z)`` tuples with no
    charge/spin/cell, so this is the aperiodic, neutral path — ``cond_nodes`` is built from
    charge=0/spin=0 for the OrbMol checkpoints (deterministic; both sharded and sequential use the
    same default, so parity is unaffected) and is ``None`` for the omat checkpoints that carry no
    conditioning weights."""
    import torch

    from .device import open_device
    from .geometry import radius_graph
    from .orb_geometry import host_edge_features
    from .orb_model import (AttentionInteractionLayer, Encoder, EnergyHead, OrbGraphContext,
                            _to_dev, host_charge_spin_embedding, host_energy_denormalize,
                            host_node_features, host_zbl_energy)
    import ttnn

    w = b.weights
    r_max = cfg["cutoff"]
    num_bases = cfg["num_bases"]
    max_num_neighbors = cfg["max_num_neighbors"]
    L = cfg["num_message_passing_steps"]
    latent_dim, hidden_dim = cfg["latent_dim"], 1024
    has_cond = "conditioner.charge_embedding.W" in w
    dev = open_device(0)
    encoder = Encoder(w, dev, node_in=cfg["node_embed_size"], edge_in=cfg["edge_embed_size"],
                      latent_dim=latent_dim, hidden_dim=hidden_dim, fast=fast)
    layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", dev, latent_dim=latent_dim,
                                         hidden_dim=hidden_dim, fast=fast) for i in range(L)]
    ehead = EnergyHead(w, dev, latent_dim=latent_dim, hidden_dim=hidden_dim, fast=fast)
    out_q.put(("ready", device_id))

    while True:
        job = in_q.get()
        if job is None:
            break
        idx, pos_np, Z_np = job
        pos = torch.tensor(pos_np, dtype=torch.float32)
        Z = torch.tensor(Z_np)
        edge_index, cell_shift = radius_graph(pos, r_max)
        E = edge_index.shape[1]
        if E == 0:
            raise ValueError("no edges within cutoff — system too sparse for this model")
        src, tgt = edge_index
        senders, receivers = tgt, src          # Orb's edge convention is the opposite of UMA's
        N = Z.shape[0]
        max_deg = max(int(torch.bincount(senders, minlength=N).max()),
                     int(torch.bincount(receivers, minlength=N).max()))
        if max_deg > max_num_neighbors:
            raise ValueError(
                f"an atom has {max_deg} neighbours within the {r_max} A cutoff, exceeding this "
                f"checkpoint's max_num_neighbors={max_num_neighbors}. Orb's own reference truncates "
                "to the closest max_num_neighbors per atom; this port does not implement that "
                "truncation, so it refuses rather than silently return a different graph.")
        node_feat = host_node_features(w, Z)
        cond_nodes = (host_charge_spin_embedding(w, 0.0, 0.0, N, latent_dim)
                      if has_cond else None)
        edge_feat, cutoff, vectors = host_edge_features(pos, senders, receivers, cell_shift,
                                                        r_max=r_max, num_bases=num_bases)
        graph = OrbGraphContext(dev, senders=senders, receivers=receivers,
                                cutoff=cutoff.detach().float(), num_nodes=N, cond_nodes=cond_nodes)
        node_dev = _to_dev(node_feat, dev, ttnn.bfloat16)
        edge_dev = _to_dev(edge_feat.detach().float(), dev, ttnn.bfloat16)
        nodes, edges = encoder(node_dev, edge_dev)
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        raw_e = ttnn.to_torch(ehead(nodes)).double().view(())
        E_gnn = host_energy_denormalize(
            raw_e, Z, N, running_mean=w["energy_head.normalizer.bn.running_mean"],
            running_var=w["energy_head.normalizer.bn.running_var"],
            ref_weight=w["energy_head.reference.linear.weight"].view(-1))
        E_tot = float(E_gnn + host_zbl_energy(Z, senders, receivers, vectors))
        out_q.put((idx, E_tot, E))

    ttnn.close_device(dev)


class MultiCard:
    """A persistent pool of one worker per device. Use as a context manager.

    ``weights_path`` is a UMA bundle (``WeightBundle``) or an Orb weights file (``OrbWeights``);
    the worker detects the family from the bundle's ``config`` and builds the matching backbone.
    """

    def __init__(self, weights_path, device_ids=(0, 1, 2, 3), *, fast=False):
        self.ctx = mp.get_context("spawn")
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.procs = [self.ctx.Process(target=_worker, args=(d, weights_path, fast, self.in_q, self.out_q),
                                       daemon=True) for d in device_ids]
        for p in self.procs:
            p.start()
        for _ in self.procs:                                   # wait until every card is ready
            self.out_q.get()

    def energies(self, systems):
        """``systems``: list of (positions[N,3], atomic_numbers[N]) numpy arrays.
        Returns (energies list in input order, total edges processed)."""
        for i, (pos, Z) in enumerate(systems):
            self.in_q.put((i, pos, Z))
        out = [None] * len(systems)
        total_edges = 0
        for _ in systems:
            idx, en, E = self.out_q.get()
            out[idx] = en
            total_edges += E
        return out, total_edges

    def close(self):
        for _ in self.procs:
            self.in_q.put(None)
        for p in self.procs:
            p.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
