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


def _pin_worker(device_id):
    """Pin this worker process to one card (it becomes device 0) and one host thread: the host
    geometry (torch) otherwise grabs every core, so N workers oversubscribe the CPU and throttle
    each other (4-card went *slower* than 1)."""
    import os

    os.environ["TT_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)


class _WorkerPool:
    """One spawn-context worker process per device, pulling jobs off a shared queue.

    The parent never imports ``ttnn`` or touches a device — that is what keeps the fan-out
    deadlock-free. Subclasses pass their worker ``target`` plus per-device leading args (the
    shared queues are appended), then implement their own submit/collect on top of the pool."""

    def __init__(self, target, args_per_device):
        self.ctx = mp.get_context("spawn")
        self.in_q = self.ctx.Queue()
        self.out_q = self.ctx.Queue()
        self.procs = [self.ctx.Process(target=target, args=(*a, self.in_q, self.out_q),
                                       daemon=True) for a in args_per_device]
        for p in self.procs:
            p.start()
        for _ in self.procs:                                   # wait until every card is ready
            self.out_q.get()

    def close(self):
        for _ in self.procs:
            self.in_q.put(None)
        for p in self.procs:
            p.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker(device_id, weights_path, fast, bucketing, in_q, out_q):
    import json
    import pathlib

    _pin_worker(device_id)

    import numpy as np

    npz = np.load(pathlib.Path(weights_path))
    cfg = json.loads(bytes(npz["config"]).decode())
    if "sphere_channels" in cfg:                               # UMA eSCN-MD bundle
        from .weights import WeightBundle

        _run_uma(WeightBundle(npz), cfg, fast, device_id, in_q, out_q)
    else:                                                      # Orb MPNN bundle
        from .orb_weights import OrbWeights

        _run_orb(OrbWeights(npz), cfg, fast, device_id, in_q, out_q, bucketing=bucketing)


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


def _run_orb(b, cfg, fast, device_id, in_q, out_q, bucketing=False):
    """Energy-only Orb forward, mirroring ``OrbCalculator.calculate``'s direct/conservative
    forward path (the two share the same encoder/layers/``EnergyHead``; only forces differ, and
    ``MultiCard`` returns energies). The systems come in as ``(pos, Z)`` tuples with no
    charge/spin/cell, so this is the aperiodic, neutral path — ``cond_nodes`` is built from
    charge=0/spin=0 for the OrbMol checkpoints (deterministic; both sharded and sequential use the
    same default, so parity is unaffected) and is ``None`` for the omat checkpoints that carry no
    conditioning weights.

    ``bucketing=True`` pads each system's edges to the ``tt_atom.bucketing`` ladder (same
    post-encoder zero-pad construction as ``OrbCalculator(bucketing=True)``) so a screening
    stream of differently-sized systems reuses compiled kernel shapes instead of compiling
    fresh per size."""
    import torch

    from .bucketing import pad_device_rows, pad_graph
    from .device import open_device
    from .geometry import radius_graph
    from .orb_geometry import check_max_neighbors, host_edge_features
    from .orb_model import (MLP_HIDDEN_DIM, AttentionInteractionLayer, Encoder, EnergyHead,
                            OrbGraphContext, _to_dev, host_charge_spin_embedding,
                            host_energy_denormalize, host_node_features, host_zbl_energy)
    import ttnn

    w = b.weights
    r_max = cfg["cutoff"]
    num_bases = cfg["num_bases"]
    max_num_neighbors = cfg["max_num_neighbors"]
    L = cfg["num_message_passing_steps"]
    latent_dim, hidden_dim = cfg["latent_dim"], MLP_HIDDEN_DIM
    has_cond = "conditioner.charge_embedding.W" in w
    zbl_aggregation = "sum" if "forces_head.mlp.NN-0.weight" in w else "mean"
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
        check_max_neighbors(senders, receivers, N, max_num_neighbors=max_num_neighbors,
                            r_max=r_max)
        node_feat = host_node_features(w, Z)
        cond_nodes = (host_charge_spin_embedding(w, 0.0, 0.0, N, latent_dim)
                      if has_cond else None)
        edge_feat, cutoff, _vec = host_edge_features(pos, senders, receivers, cell_shift,
                                                     r_max=r_max, num_bases=num_bases)
        vectors = pos[receivers] - pos[senders] + cell_shift       # true edges, for ZBL
        dev_senders, dev_receivers, gkw, e_bucket = senders, receivers, {}, 0
        if bucketing:
            dev_senders, dev_receivers, cutoff, gkw, e_bucket = pad_graph(
                senders, receivers, cutoff, max_num_neighbors=max_num_neighbors)
        graph = OrbGraphContext(dev, senders=dev_senders, receivers=dev_receivers,
                                cutoff=cutoff.detach().float(), num_nodes=N, cond_nodes=cond_nodes,
                                **gkw)
        node_dev = _to_dev(node_feat, dev, ttnn.bfloat16)
        edge_dev = _to_dev(edge_feat.detach().float(), dev, ttnn.bfloat16)
        nodes, edges = encoder(node_dev, edge_dev)
        if e_bucket:
            edges = pad_device_rows(ttnn, edges, e_bucket)
        for layer in layers:
            nodes, edges = layer(nodes, edges, graph)
        raw_e = ttnn.to_torch(ehead(nodes)).double().view(())
        E_gnn = host_energy_denormalize(
            raw_e, Z, N, running_mean=w["energy_head.normalizer.bn.running_mean"],
            running_var=w["energy_head.normalizer.bn.running_var"],
            ref_weight=w["energy_head.reference.linear.weight"].view(-1))
        E_tot = float(E_gnn + host_zbl_energy(
            Z, senders, receivers, vectors, node_aggregation=zbl_aggregation))
        out_q.put((idx, E_tot, E))

    ttnn.close_device(dev)


class MultiCard(_WorkerPool):
    """A persistent pool of one worker per device. Use as a context manager.

    ``weights_path`` is a UMA bundle (``WeightBundle``) or an Orb weights file (``OrbWeights``);
    the worker detects the family from the bundle's ``config`` and builds the matching backbone.

    ``bucketing=True`` (Orb bundles only) pads each system's edges to the fixed ladder in
    ``tt_atom.bucketing`` so differently-sized systems reuse compiled kernel shapes — the
    screening-stream win (energies bit-exact vs unpadded; see ``tests/test_bucketing.py``).
    """

    def __init__(self, weights_path, device_ids=(0, 1, 2, 3), *, fast=False, bucketing=False):
        super().__init__(_worker, [(d, weights_path, fast, bucketing) for d in device_ids])

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


# ── multi-card relax / MD fan-out ───────────────────────────────────────────
# Each worker builds a real ASE Calculator on its pinned card and runs the FULL relax/MD
# loop (FIRE / Langevin) for every structure it pulls — the same code path as the single-card
# CLI (``cmd_run`` / ``cmd_relax`` / ``cmd_md`` via ``tt_atom.simulate``). Per-structure parity
# is bit-exact BY CONSTRUCTION: each structure runs in exactly one worker's Calculator +
# optimizer, identical code to the single-card path; there is no cross-system batching or
# regrouping (the same argument ``test_multicard.py`` makes for energies). The worker caches
# one Calculator per (family, reduced-composition, charge, spin): Orb reuses one Calculator
# for all structures (composition-independent); UMA reuses one per composition group (MoLE
# bakes composition; the bundle cache makes repeat compositions a plain load).

def _worker_sim(device_id, model, task, refenv, cache_dir, fast, trace, sim_params, in_q, out_q):
    _pin_worker(device_id)

    import numpy as np

    from . import device as D
    from .auto import Calculator, _family
    from .simulate import relax_atoms, md_atoms

    mode = sim_params["mode"]
    calcs = {}        # (reduced_composition, charge, spin, task) -> Calculator (UMA); None for Orb
    dev = None        # opened once on first use, reused across every Calculator this worker builds

    def get_calc(atoms):
        nonlocal dev
        if _family(model) == "orb":
            key = None
        else:
            from .bundle_cache import infer_task, reduced_composition
            key = (reduced_composition(atoms.get_atomic_numbers()),
                   float(atoms.info.get("charge", 0.0)), float(atoms.info.get("spin", 0.0)),
                   task or infer_task(atoms))
        if key not in calcs:
            if dev is None:
                # Open the device ONCE per worker and reuse it. UMA builds one Calculator per
                # reduced composition, and a second open_device() in the same process is a hard
                # TT_FATAL ("No MetalContext instance for context_id N"); two different-composition
                # structures in one worker used to hit that. Reusing one device across every
                # bundle this worker owns is the fix — the Calculator's close() won't close a
                # device it didn't open (device= passed, not device_id=). trace_region_size
                # matches TTAtomCalculator's default so trace=True still gets a capture region.
                dev = D.open_device(0, trace_region_size=400_000_000 if trace else 0)
            calcs[key] = Calculator(atoms, model, task=task, refenv=refenv, cache_dir=cache_dir,
                                    device=dev, fast=fast, trace=trace)
        return calcs[key]

    out_q.put(("ready", device_id))

    while True:
        job = in_q.get()
        if job is None:
            break
        idx, sys_dict = job
        try:
            from ase import Atoms
            atoms = Atoms(numbers=sys_dict["Z"], positions=sys_dict["pos"],
                          cell=sys_dict.get("cell"), pbc=sys_dict.get("pbc", False))
            atoms.info.update(charge=sys_dict.get("charge", 0.0), spin=sys_dict.get("spin", 0.0))
            calc = get_calc(atoms)
            atoms.calc = calc
            if mode == "relax":
                res = relax_atoms(atoms, fmax=sim_params["fmax"], steps=sim_params["steps"],
                                  logfile=None)
            elif mode == "md":
                res = md_atoms(atoms, steps=sim_params["steps"], dt=sim_params["dt"],
                               temp=sim_params["temp"], logfile=None,
                               seed=sim_params.get("seed"))
            else:        # energy: a single point through the same cached Calculator
                res = dict(energy=float(atoms.get_potential_energy()),
                           forces=np.asarray(atoms.get_forces(), dtype=np.float64),
                           fmax=None, nsteps=0, converged=None)
            out_q.put((idx, dict(ok=True, pos=atoms.get_positions(), Z=sys_dict["Z"],
                                 energy=res["energy"], forces=res["forces"],
                                 fmax=res.get("fmax"), nsteps=res.get("nsteps"),
                                 converged=res.get("converged"))))
        except Exception as e:        # noqa: BLE001 - ship the error back, don't kill the worker
            import traceback
            out_q.put((idx, dict(ok=False, error=f"{type(e).__name__}: {e}",
                                 tb=traceback.format_exc())))

    for c in calcs.values():
        try:
            c.close()        # device= was passed, so this never closes the worker's shared device
        except Exception:
            pass
    if dev is not None:
        import ttnn

        ttnn.close_device(dev)


class MultiCardSim(_WorkerPool):
    """A persistent pool of one worker per device that runs full relax/MD loops or single
    points, not just a forward energy pass. Use as a context manager.

    The counterpart to :class:`MultiCard` for the headline materials-screening use case
    (high-throughput virtual screening): fan a batch of *independent* structures across N
    local cards, each card owning a full ASE ``Calculator`` + FIRE/Langevin loop for its
    assigned structures. Per-structure results (final geometry, energy, forces) come back in
    input order, bit-exact vs running each structure on one card through ``tt-atom run``
    (``--relax``/``--md`` or the single-point default) — each structure runs the identical
    single-card code path inside its worker, so sharding changes nothing about a structure's
    own numerics.

    ``model`` selects the family by name (``"uma-s-1"`` or an Orb checkpoint), exactly like
    ``tt_atom.auto.Calculator``. ``sim_params`` is a dict with ``mode`` (``"relax"``/``"md"``/
    ``"energy"``) and the optimizer/integrator knobs (``fmax``/``steps`` for relax;
    ``steps``/``dt``/``temp`` for md; an optional ``seed`` pins the MD velocity draw for
    reproducible parity; ``"energy"`` is a single point and ignores the knobs).
    """

    def __init__(self, model, device_ids=(0, 1, 2, 3), *, task=None, refenv=None,
                 cache_dir=None, fast=False, trace=False, sim_params=None):
        if sim_params is None or sim_params.get("mode") not in ("relax", "md", "energy"):
            raise ValueError("sim_params must be a dict with mode 'relax', 'md', or 'energy'")
        self.sim_params = sim_params
        super().__init__(_worker_sim,
                         [(d, model, task, refenv, cache_dir, fast, trace, sim_params)
                          for d in device_ids])

    def run(self, systems):
        """``systems``: list of dicts (or ASE ``Atoms``) with ``pos``, ``Z``, optional
        ``charge``/``spin``/``cell``/``pbc``. Returns one result dict per system in input
        order: ``{ok, pos, Z, energy, forces, fmax, nsteps, converged}`` (relax; ``fmax``/
        ``converged`` are ``None`` for md and energy, ``nsteps`` is 0 for energy);
        ``{ok: False, error}`` on a per-system failure (the pool keeps the other systems'
        results)."""
        for i, d in (_to_system_dict(s, i) for i, s in enumerate(systems)):
            self.in_q.put((i, d))
        out = [None] * len(systems)
        for _ in systems:
            idx, res = self.out_q.get()
            out[idx] = res
        return out


def _to_system_dict(system, idx):
    """Normalize an ASE ``Atoms`` / ``(pos, Z)`` tuple / dict to the worker message."""
    import numpy as np

    if isinstance(system, dict) and "pos" in system and "Z" in system:
        return idx, system
    if isinstance(system, tuple) and len(system) == 2:        # (positions, atomic_numbers)
        return idx, dict(pos=np.asarray(system[0], dtype=np.float32),
                         Z=np.asarray(system[1], dtype=np.int64))
    # ASE Atoms (duck-typed)
    cell = system.get_cell()
    pbc = system.get_pbc()
    return idx, dict(
        pos=np.asarray(system.get_positions(), dtype=np.float32),
        Z=np.asarray(system.get_atomic_numbers(), dtype=np.int64),
        charge=float(system.info.get("charge", 0.0)),
        spin=float(system.info.get("spin", 0.0)),
        cell=np.asarray(cell) if np.asarray(pbc).any() else None,
        pbc=bool(np.asarray(pbc).any()),
    )
