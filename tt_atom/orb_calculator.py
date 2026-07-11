"""``OrbCalculator`` — an ASE calculator backed by the device-resident Orb-v3/OrbMol backbone.

The Orb-family counterpart to ``tt_atom.calculator.TTAtomCalculator``/``UMA``: same ASE-calculator
shape, same ``Model(atoms, charge=, spin=)`` one-liner, same device-open/close lifecycle. The one
genuine architectural difference (see ``docs/orb-port.md``'s "Architecture verdict"): Orb has no
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
from ase.calculators.calculator import Calculator, all_changes

from . import device as D


def Orb(atoms=None, checkpoint="orb-v3-conservative-inf-omat", charge=0, spin=1, refenv=None,
       cache_dir=None, device=None, device_id=0, fast=False, **kwargs):
    """Zero-config entry point — the Orb-family counterpart to :func:`tt_atom.calculator.UMA`.

        from tt_atom import Orb
        atoms.calc = Orb(atoms)            # orb-v3-conservative-inf-omat, energy + forces

    Unlike ``UMA(atoms)``, Orb has no MoLE routing to bake per composition, so the checkpoint
    export is cached once per *checkpoint name* (not per structure) and ``atoms`` is only used to
    stamp ``charge``/``spin`` defaults, mirroring ``UMA``'s convention (``Orb(atoms, charge=-1,
    spin=2)``) — pass ``atoms=None`` to build a calculator with no structure in hand yet. Charge/
    spin conditioning only exists on the OrbMol checkpoints (``checkpoint="orb-v3-conservative-
    omol"`` / ``"orb-v3-direct-omol"``); the omat checkpoints ignore both (no conditioning
    weights) since they were never trained with them."""
    if atoms is not None:
        atoms.info.setdefault("charge", charge)
        atoms.info.setdefault("spin", spin)
    return OrbCalculator.from_checkpoint(checkpoint=checkpoint, refenv=refenv, cache_dir=cache_dir,
                                         device=device, device_id=device_id, fast=fast, **kwargs)


class OrbCalculator(Calculator):
    implemented_properties = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(self, weights, device=None, device_id=0, fast=False, **kwargs):
        """``weights`` is an ``OrbWeights`` (or a path to one, see ``tools/export_orb_weights.py``
        / ``tt_atom.orb_weight_cache``): the raw checkpoint's config + state dict, no system-
        specific data. Builds the device-resident encoder/backbone/heads once; every subsequent
        ``calculate()`` call reuses them for whatever structure ASE hands it — no per-composition
        rebuild, since Orb bakes no routing into the weights."""
        super().__init__(**kwargs)
        if isinstance(weights, (str, pathlib.Path)):
            from .orb_weights import OrbWeights

            weights = OrbWeights.load(weights)
        self.cfg = weights.config
        w = weights.weights
        self._w = w
        self.fast = fast
        self.r_max = self.cfg["cutoff"]
        self.num_bases = self.cfg["num_bases"]
        self.max_num_neighbors = self.cfg["max_num_neighbors"]
        self.task = self.task_name = self.cfg["task"]

        self._owns_device = device is None
        self.device = device if device is not None else D.open_device(device_id)

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
        ``$TT_ATOM_REFENV`` > ``~/.ttatom_run/refenv/bin/python`` (same as ``UMA``'s). A cache hit
        needs no reference env at all."""
        from . import orb_weight_cache as OWC
        from .orb_weights import OrbWeights

        path = OWC.get_or_build(checkpoint, refenv=refenv, cache_dir=cache_dir)
        return cls(OrbWeights.load(str(path)), device=device, device_id=device_id, fast=fast,
                  **kwargs)

    def close(self):
        if self._owns_device and self.device is not None:
            import ttnn

            ttnn.close_device(self.device)
            self.device = None

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
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["energies"] = np.full(N, E / N, dtype=np.float64)
        self.results["forces"] = F.detach().numpy().astype(np.float64)
        if stress is not None:
            self.results["stress"] = stress.detach().numpy().astype(np.float64)
