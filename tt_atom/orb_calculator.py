"""``OrbCalculator`` — an ASE calculator backed by the device-resident Orb-v3/OrbMol backbone.

The Orb-family counterpart to ``tt_atom.calculator.TTAtomCalculator``: it shares the ASE device
lifecycle + results packing (:class:`ase_base.DeviceCalculator`) and is reachable through the same
unified ``Calculator(atoms, model=...)`` front door (:mod:`tt_atom.auto`), differing only in the
backbone it drives. The one genuine architectural difference (see ``docs/orb-port.md``):
Orb has no
MoLE (or any) expert routing baked in at merge time, so its weights are valid for *any*
composition/charge/spin — there is no per-system bundle to build or cache, only a per-*checkpoint*
weight export (``tt_atom.orb_weight_cache``), built once ever and reused across every structure.
The device modules (``Encoder``/``AttentionInteractionLayer``/heads) are likewise built once at
construction and reused, unlike UMA's per-composition ``Backbone``.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .ase_base import DeviceCalculator


class OrbCalculator(DeviceCalculator):
    def __init__(self, weights, device=None, device_id=0, fast=False, **kwargs):
        """``weights`` is an ``OrbWeights`` (or a path to one, see ``tools/export_orb_weights.py``
        / ``tt_atom.orb_weight_cache``): the raw checkpoint's config + state dict, no system-
        specific data. Builds the device-resident encoder/backbone/heads once; every subsequent
        ``calculate()`` call reuses them for whatever structure ASE hands it — no per-composition
        rebuild, since Orb bakes no routing into the weights."""
        super().__init__(device=device, device_id=device_id, fast=fast, **kwargs)
        if isinstance(weights, (str, pathlib.Path)):
            from .orb_weights import OrbWeights

            weights = OrbWeights.load(weights)
        self.cfg = weights.config
        w = weights.weights
        self._w = w
        self.r_max = self.cfg["cutoff"]
        self.num_bases = self.cfg["num_bases"]
        self.max_num_neighbors = self.cfg["max_num_neighbors"]
        self.task = self.task_name = self.cfg["task"]

        from .orb_model import AttentionInteractionLayer, Encoder, EnergyHead, ForceHead, StressHead

        L = self.cfg["num_message_passing_steps"]
        latent_dim, hidden_dim = self.cfg["latent_dim"], 1024
        self.encoder = Encoder(w, self.device, node_in=self.cfg["node_embed_size"],
                               edge_in=self.cfg["edge_embed_size"], latent_dim=latent_dim,
                               hidden_dim=hidden_dim, fast=fast)
        self.layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", self.device,
                                                 latent_dim=latent_dim, hidden_dim=hidden_dim,
                                                 fast=fast) for i in range(L)]
        self.ehead = EnergyHead(w, self.device, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                fast=fast)
        # ForceHead present => this is a "direct" checkpoint (forces are a device MLP head, no
        # autograd); absent => "conservative" (forces via tt_atom.orb_forces's analytic VJP).
        self.is_direct = "forces_head.mlp.NN-0.weight" in w
        self.fhead = (ForceHead(w, self.device, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                fast=fast) if self.is_direct else None)
        self.shead = (StressHead(w, self.device, latent_dim=latent_dim, hidden_dim=hidden_dim,
                                 fast=fast) if "stress_head.mlp.NN-0.weight" in w else None)
        self.has_cond = "conditioner.charge_embedding.W" in w

    @classmethod
    def from_checkpoint(cls, checkpoint="orb-v3-conservative-inf-omat", refenv=None, cache_dir=None,
                        device=None, device_id=0, fast=False, **kwargs):
        """Export (or load from cache) ``checkpoint``'s weights via the reference env and return a
        ready calculator. Resolution order for the reference python: ``refenv`` arg >
        ``$TT_ATOM_REFENV`` > ``~/.ttatom_run/refenv/bin/python`` (same as the UMA path's). A cache hit
        needs no reference env at all."""
        from . import orb_weight_cache as OWC
        from .orb_weights import OrbWeights

        path = OWC.get_or_build(checkpoint, refenv=refenv, cache_dir=cache_dir)
        return cls(OrbWeights.load(str(path)), device=device, device_id=device_id, fast=fast,
                  **kwargs)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import ttnn

        from .disjoint import _as_atoms_fields
        from .geometry import radius_graph
        from .orb_forces import energy_and_forces
        from .orb_geometry import host_edge_features
        from .orb_model import (OrbGraphContext, _to_dev, host_charge_spin_embedding,
                                host_conservative_force_denormalize, host_conservative_stress,
                                host_energy_denormalize, host_force_denormalize,
                                host_node_features, host_stress_denormalize, host_zbl_energy,
                                host_zbl_forces)

        pos, Z, charge, spin, cell, pbc = _as_atoms_fields(atoms)
        N = Z.shape[0]
        edge_index, cell_shift = radius_graph(pos, self.r_max, cell=cell, pbc=pbc)
        if edge_index.shape[1] == 0:
            raise ValueError("no edges within cutoff — system too sparse for this model")
        src, tgt = edge_index
        senders, receivers = tgt, src  # Orb's edge convention is the opposite of UMA/fairchem's

        # This port reuses UMA's brute-force radius_graph (no per-atom neighbour cap); Orb's own
        # reference truncates to the closest max_num_neighbors per atom. Rather than silently
        # diverge from the reference on a denser structure, refuse with a clear error (same
        # philosophy as the uma-m shape-mismatch error in tt_atom/model.py).
        max_deg = max(int(torch.bincount(senders, minlength=N).max()),
                     int(torch.bincount(receivers, minlength=N).max()))
        if max_deg > self.max_num_neighbors:
            raise ValueError(
                f"an atom has {max_deg} neighbours within the {self.r_max} A cutoff, exceeding "
                f"this checkpoint's max_num_neighbors={self.max_num_neighbors}. Orb's own "
                "reference truncates to the closest max_num_neighbors per atom; this port does "
                "not implement that truncation (unverified against the reference), so it refuses "
                "rather than silently return a different graph than Orb's own inference would use."
            )

        node_feat = host_node_features(self._w, Z)
        cond_nodes = None
        if self.has_cond:
            cond_nodes = host_charge_spin_embedding(self._w, float(charge), float(spin), N,
                                                    self.cfg["latent_dim"])

        explicit_stress = "stress" in properties
        want_stress = cell is not None and (explicit_stress or bool(pbc.all()))
        if explicit_stress and self.is_direct and self.shead is None:
            raise ValueError(
                "stress requested but this direct checkpoint carries no stress_head weights."
            )
        if want_stress and self.is_direct and self.shead is None:
            want_stress = False  # implicit-only (a periodic system that didn't ask for stress)

        if self.is_direct:
            edge_feat, cutoff, vectors = host_edge_features(pos, senders, receivers, cell_shift,
                                                            r_max=self.r_max,
                                                            num_bases=self.num_bases)
            graph = OrbGraphContext(self.device, senders=senders, receivers=receivers,
                                    cutoff=cutoff.detach().float(), num_nodes=N,
                                    cond_nodes=cond_nodes)
            node_dev = _to_dev(node_feat, self.device, ttnn.bfloat16)
            edge_dev = _to_dev(edge_feat.detach().float(), self.device, ttnn.bfloat16)
            nodes, edges = self.encoder(node_dev, edge_dev)
            for layer in self.layers:
                nodes, edges = layer(nodes, edges, graph)
            raw_e = ttnn.to_torch(self.ehead(nodes)).double().view(())
            E = host_energy_denormalize(
                raw_e, Z, N, running_mean=self._w["energy_head.normalizer.bn.running_mean"],
                running_var=self._w["energy_head.normalizer.bn.running_var"],
                ref_weight=self._w["energy_head.reference.linear.weight"].view(-1))
            raw_f = ttnn.to_torch(self.fhead(nodes)).double()
            F = host_force_denormalize(
                raw_f, running_mean=self._w["forces_head.normalizer.bn.running_mean"],
                running_var=self._w["forces_head.normalizer.bn.running_var"])
            F = F + host_zbl_forces(Z, senders, receivers, pos, cell_shift)
            stress = None
            if want_stress:
                raw_s = ttnn.to_torch(self.shead(nodes)).double()
                stress = host_stress_denormalize(
                    raw_s,
                    diag_mean=self._w["stress_head.diag_normalizer.bn.running_mean"],
                    diag_var=self._w["stress_head.diag_normalizer.bn.running_var"],
                    offdiag_mean=self._w["stress_head.offdiag_normalizer.bn.running_mean"],
                    offdiag_var=self._w["stress_head.offdiag_normalizer.bn.running_var"],
                ).view(6)
        else:
            raw_e, F_raw, *rest = energy_and_forces(
                self.encoder, self.layers, self.ehead, self.device, pos=pos, senders=senders,
                receivers=receivers, atomic_numbers=Z, node_feat=node_feat,
                cell_shift=cell_shift, r_max=self.r_max, num_bases=self.num_bases,
                compute_stress=want_stress, cond_nodes=cond_nodes)
            E = host_energy_denormalize(
                torch.tensor(raw_e, dtype=torch.float64), Z, N,
                running_mean=self._w["energy_head.normalizer.bn.running_mean"],
                running_var=self._w["energy_head.normalizer.bn.running_var"],
                ref_weight=self._w["energy_head.reference.linear.weight"].view(-1))
            F = host_conservative_force_denormalize(
                F_raw, N, running_var=self._w["energy_head.normalizer.bn.running_var"])
            vectors = pos[receivers] - pos[senders] + cell_shift
            F = F + host_zbl_forces(Z, senders, receivers, pos, cell_shift)
            stress = None
            if want_stress:
                stress = host_conservative_stress(
                    rest[0], N, cell, running_var=self._w["energy_head.normalizer.bn.running_var"])

        E = float(E + host_zbl_energy(Z, senders, receivers, vectors))
        self._store_results(atoms, E, F, stress=stress)

    def evaluate_batch(self, systems, properties=("energy", "forces")):
        """Disjoint-union batched evaluation -- K systems in ONE device forward (Orb-family
        counterpart to :meth:`tt_atom.calculator.TTAtomCalculator.evaluate_batch`).

        ``systems`` is a list of ASE ``Atoms`` (or ``(positions, atomic_numbers)`` / dicts). The
        systems are concatenated into one block-diagonal graph, evaluated in a single device call,
        and the per-system energies recovered by ``EnergyHead.batch`` (a per-system mean of the
        node features feeding the same 2-layer MLP); forces come from one shared backward
        (conservative checkpoints: the analytic ``-dE/dpos`` VJP, block-diagonal so each atom's
        own-system force falls out automatically; direct checkpoints: the ``ForceHead`` MLP with
        per-system net-force removal). This is the throughput path for many small systems.

        Returns ``dict(energy=np.ndarray[K], forces=list[np.ndarray[N_k, 3]] | None)``, mirroring
        ``TTAtomCalculator.evaluate_batch``'s shape so the two are drop-in parallels from a user's
        perspective.

        Orb has no per-composition MoLE routing (unlike UMA), so the batch may mix compositions,
        charges, and spins freely -- the only constraint is :meth:`assemble_orb`'s per-atom
        ``max_num_neighbors`` cap, enforced per-system just like ``calculate`` does for one
        structure (a denser system inside a batch still raises the same clear error)."""
        import ttnn

        from .disjoint import assemble_orb
        from .orb_forces import energy_and_forces_batch
        from .orb_geometry import host_edge_features
        from .orb_model import (OrbGraphContext, _to_dev, host_charge_spin_embedding,
                                host_conservative_force_denormalize, host_energy_denormalize,
                                host_force_denormalize, host_node_features, host_zbl_energy,
                                host_zbl_forces)

        want_forces = "forces" in properties
        bg = assemble_orb(systems, self.r_max, self.max_num_neighbors)
        Ntot = bg.Z.shape[0]
        seg, seg_mean = bg.segment_matrices()
        seg_mean_T = seg_mean.t().contiguous()

        node_feat = host_node_features(self._w, bg.Z)
        cond_nodes = None
        if self.has_cond:
            cond_nodes = torch.cat([
                host_charge_spin_embedding(self._w, float(bg.charges[k]), float(bg.spins[k]),
                                          bg.natoms[k], self.cfg["latent_dim"])
                for k in range(bg.K)], dim=0)

        if self.is_direct or not want_forces:
            # Forward-only: encoder + layers, then EnergyHead.batch (and ForceHead.batch for direct
            # with forces). Conservative-without-forces reuses this path (no backward needed).
            edge_feat, cutoff, _vectors = host_edge_features(
                bg.pos, bg.senders, bg.receivers, bg.cell_shift,
                r_max=self.r_max, num_bases=self.num_bases)
            graph = OrbGraphContext(self.device, senders=bg.senders, receivers=bg.receivers,
                                   cutoff=cutoff.detach().float(), num_nodes=Ntot,
                                   cond_nodes=cond_nodes)
            seg_dev = _to_dev(seg.float(), self.device, ttnn.bfloat16)
            seg_mean_dev = _to_dev(seg_mean.float(), self.device, ttnn.bfloat16)
            node_dev = _to_dev(node_feat, self.device, ttnn.bfloat16)
            edge_dev = _to_dev(edge_feat.detach().float(), self.device, ttnn.bfloat16)
            nodes, edges = self.encoder(node_dev, edge_dev)
            for layer in self.layers:
                nodes, edges = layer(nodes, edges, graph)
            E_raw = ttnn.to_torch(self.ehead.batch(nodes, seg_mean_dev)).double().view(-1)
            if self.is_direct and want_forces:
                F_raw = ttnn.to_torch(self.fhead.batch(nodes, seg_dev, seg_mean_dev)).double()
            else:
                F_raw = None
        else:
            # Conservative with forces: one device forward + one batched VJP + host autograd finish.
            E_raw, F_raw = energy_and_forces_batch(
                self.encoder, self.layers, self.ehead, self.device,
                pos=bg.pos, senders=bg.senders, receivers=bg.receivers, atomic_numbers=bg.Z,
                node_feat=node_feat, cell_shift=bg.cell_shift, seg_mean=seg_mean,
                seg_mean_T=seg_mean_T, r_max=self.r_max, num_bases=self.num_bases,
                cond_nodes=cond_nodes)
        # Per-system denormalize + ZBL. Orb's normalizer scales forces by sigma * N_k per system
        # (conservative) or sigma (direct), and ZBL pair-repulsion is added per-system. ZBL forces
        # are block-diagonal (edges never cross systems), so one host autograd over the union gives
        # correct per-atom ZBL forces; ZBL energy is split per-system (mean over each system's atoms).
        vectors = bg.pos[bg.receivers] - bg.pos[bg.senders] + bg.cell_shift
        F_zbl = (host_zbl_forces(bg.Z, bg.senders, bg.receivers, bg.pos, bg.cell_shift)
                if want_forces else None)
        energies, forces_out, off = [], [], 0
        for k, n in enumerate(bg.natoms):
            Z_k = bg.Z[off:off + n]
            E_k = host_energy_denormalize(
                E_raw[k], Z_k, n,
                running_mean=self._w["energy_head.normalizer.bn.running_mean"],
                running_var=self._w["energy_head.normalizer.bn.running_var"],
                ref_weight=self._w["energy_head.reference.linear.weight"].view(-1))
            # ZBL energy per system: slice this system's edges (block-diagonal => batch[senders]==k).
            m = bg.batch[bg.senders] == k
            E_zbl = host_zbl_energy(Z_k, bg.senders[m] - off, bg.receivers[m] - off, vectors[m])
            energies.append(float(E_k + E_zbl))
            if want_forces:
                F_k = F_raw[off:off + n]
                if self.is_direct:
                    F_k = host_force_denormalize(
                        F_k, running_mean=self._w["forces_head.normalizer.bn.running_mean"],
                        running_var=self._w["forces_head.normalizer.bn.running_var"])
                else:
                    F_k = host_conservative_force_denormalize(
                        F_k, n, running_var=self._w["energy_head.normalizer.bn.running_var"])
                F_k = F_k + F_zbl[off:off + n]
                forces_out.append(F_k.detach().numpy().astype(np.float64))
            off += n
        return dict(energy=np.array(energies), forces=forces_out if want_forces else None)
