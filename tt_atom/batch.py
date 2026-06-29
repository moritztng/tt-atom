"""Multi-card throughput: fan independent systems across all cards, one process per card.

The eSEN/eSCN-MD evaluation of one system is independent of every other, so throughput scales
by running one worker process per Tenstorrent card (each pinned with ``TT_VISIBLE_DEVICES`` so it
owns exactly one device) with the model + weights resident on that card. The parent streams
systems to a shared queue and the workers pull, evaluate, and return energies — embarrassingly
parallel, so aggregate throughput is the sum across cards.

``ttnn`` is imported only *inside* the worker (after the device is pinned); the parent never
touches a device, which is what keeps the fan-out deadlock-free.
"""
from __future__ import annotations

import multiprocessing as mp


def _worker(device_id, weights_path, fast, in_q, out_q):
    import os

    os.environ["TT_VISIBLE_DEVICES"] = str(device_id)          # pin one card -> it is device 0
    # one host thread per worker: the host geometry (torch) otherwise grabs every core, so N
    # workers oversubscribe the CPU and throttle each other (4-card went *slower* than 1).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    from .device import open_device
    from .model import Backbone, GraphContext
    from .geometry import HostGeometry, csd_embedding, radius_graph
    from .weights import WeightBundle
    import ttnn

    b = WeightBundle.load(weights_path)
    cfg, w = b.config, b.weights
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
        ei = radius_graph(pos, cfg["cutoff"])
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


class MultiCard:
    """A persistent pool of one worker per device. Use as a context manager."""

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
