"""Disjoint-union (block-diagonal) graph batching — the fairchem/PyG way.

Concatenate K independent systems into one big block-diagonal graph so the device backbone
evaluates all K in a *single* forward. This is a throughput win in the dispatch-bound regime
(many small systems), where per-call host overhead — not device compute — dominates.

Every eSCN-MD backbone op is per-node or per-edge: the SO(2) convs and rotations act edgewise,
the norms and grid/spectral feed-forwards act nodewise, and the scatter-add is the one-hot
matmul ``S[N, E] @ messages`` (see edgewise.py). Give each system's edges a per-system node
offset and that scatter matrix is automatically block-diagonal — a message on an edge lands only
on its own system's nodes. So the *entire* forward is batch-transparent as-is; the only change is
the energy readout, which becomes a segment-sum per system (``Backbone.energy_batch``). Forces
need no change either: the batch energy is the sum of per-system energies, and block-diagonality
makes ``dE_total/dx_n = dE_(system of n)/dx_n``, so the existing summed-energy backward yields
each atom's own-system force (the batched forces are just the concatenation).

The single change that assembly must respect: build each system's neighbour list *separately*
(never on the concatenated positions), or a global radius graph would wire atoms across systems.

Composition constraint: a merged uma-s-1 WeightBundle bakes the MoLE expert routing for one
reduced composition (fairchem's ``merge_MOLE_model`` asserts the same), so a batch that wants
*correct* energies shares that composition — e.g. conformers / an MD ensemble of one molecule.
The assembly itself is composition-agnostic (block-diagonal is block-diagonal); only the routing
baked into the weights is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import csd_embedding, radius_graph


def _as_atoms_fields(system):
    """Accept an ASE ``Atoms`` or a ``(positions, atomic_numbers)`` / dict and return the fields
    disjoint-union assembly needs: positions, atomic numbers, charge, spin, cell, pbc."""
    if hasattr(system, "get_positions"):                       # ASE Atoms
        pos = torch.tensor(np.asarray(system.get_positions()), dtype=torch.float32)
        Z = torch.tensor(np.asarray(system.get_atomic_numbers()), dtype=torch.long)
        charge = float(system.info.get("charge", 0.0))
        spin = float(system.info.get("spin", 0.0))
        pbc = np.asarray(system.get_pbc())
        cell = (torch.tensor(np.asarray(system.get_cell()), dtype=torch.float32)
                if pbc.any() else None)
        return pos, Z, charge, spin, cell, pbc
    if isinstance(system, dict):
        pos = torch.as_tensor(system["pos"], dtype=torch.float32)
        Z = torch.as_tensor(system["Z"], dtype=torch.long)
        return pos, Z, float(system.get("charge", 0.0)), float(system.get("spin", 0.0)), \
            system.get("cell"), system.get("pbc")
    pos, Z = system                                            # (positions, atomic_numbers)
    return (torch.as_tensor(pos, dtype=torch.float32), torch.as_tensor(Z, dtype=torch.long),
            0.0, 0.0, None, None)


@dataclass
class BatchedGraph:
    """A disjoint-union of K systems ready for one device forward.

    ``edge_index`` carries per-system node offsets (block-diagonal scatter). ``batch`` maps each
    atom to its system id; ``natoms`` are the per-system atom counts (to split concatenated
    outputs back). ``sys_emb`` [Ntot, C] is the per-node system (charge/spin/dataset) embedding.
    """
    pos: torch.Tensor           # [Ntot, 3]
    Z: torch.Tensor             # [Ntot]
    edge_index: torch.Tensor    # [2, Etot], node offsets applied
    cell_shift: torch.Tensor    # [Etot, 3]
    batch: torch.Tensor         # [Ntot] atom -> system id
    natoms: list                # per-system atom counts (len K)
    sys_emb: torch.Tensor       # [Ntot, C]

    @property
    def K(self):
        return len(self.natoms)

    def segment_matrix(self):
        """One-hot segment matrix ``seg[K, Ntot]`` (``seg[k, n] = 1`` iff atom n in system k) for
        the segment-sum energy readout."""
        seg = torch.zeros(self.K, self.pos.shape[0])
        seg[self.batch, torch.arange(self.pos.shape[0])] = 1.0
        return seg


@dataclass
class OrbBatchedGraph:
    """A disjoint-union of K systems ready for one Orb device forward -- the Orb-family counterpart
    to ``BatchedGraph``, without the eSCN ``sys_emb`` (Orb has no per-system dataset/MoLE routing:
    charge/spin conditioning is a per-system additive node shift built separately by the caller via
    ``orb_model.host_charge_spin_embedding``, optional and only for the OrbMol checkpoints). Edge
    indices follow Orb's own convention (``senders = tgt``, ``receivers = src``, the opposite of
    fairchem/UMA -- see ``orb_calculator.OrbCalculator.calculate``), with per-system node offsets
    applied so the union stays block-diagonal."""
    pos: torch.Tensor           # [Ntot, 3]
    Z: torch.Tensor             # [Ntot]
    senders: torch.Tensor       # [Etot], Orb convention (tgt), node offsets applied
    receivers: torch.Tensor     # [Etot], Orb convention (src), node offsets applied
    cell_shift: torch.Tensor    # [Etot, 3]
    batch: torch.Tensor         # [Ntot] atom -> system id
    natoms: list                # per-system atom counts (len K)
    charges: list               # per-system total charge (len K) -- OrbMol conditioning only
    spins: list                 # per-system total spin (len K) -- OrbMol conditioning only

    @property
    def K(self):
        return len(self.natoms)

    def segment_matrices(self):
        """The two segment matrices Orb's batched readout/backward need (UMA's single ``segment_matrix``
        splits into two because Orb's ``EnergyHead`` *means* node features rather than segment-*summing*
        a per-node scalar): ``seg`` [K, Ntot] plain indicator (``seg[k, n] = 1`` iff atom n in system k)
        and ``seg_mean`` [K, Ntot] row-normalized (``seg_mean[k, n] = 1/N_k`` iff atom n in system k).
        ``seg_mean`` feeds ``EnergyHead.batch``/``ForceHead.batch``'s per-system mean; ``seg``^T broadcasts
        the per-system result back to each node; ``seg_mean``^T seeds ``orb_forces.energy_bw_batch``."""
        K, Ntot = self.K, self.pos.shape[0]
        seg = torch.zeros(K, Ntot)
        seg[self.batch, torch.arange(Ntot)] = 1.0
        seg_mean = seg / torch.tensor(self.natoms, dtype=seg.dtype).unsqueeze(1)
        return seg, seg_mean


def assemble(systems, cutoff, weights, sphere_channels, task="omol"):
    """Build the block-diagonal graph for ``systems`` (list of ASE ``Atoms`` / dicts / tuples).

    Each system's neighbour list is built independently (never on concatenated positions), then
    edges are offset by the running node count so the union stays block-diagonal. Returns a
    ``BatchedGraph``. Raises if any system has no edges within ``cutoff`` (too sparse for the model).
    """
    if len(systems) == 0:
        raise ValueError("empty batch")
    pos_all, Z_all, ei_all, shift_all, batch_all, sys_all = [], [], [], [], [], []
    natoms = []
    node_off = 0
    for k, system in enumerate(systems):
        pos, Z, charge, spin, cell, pbc = _as_atoms_fields(system)
        n = Z.shape[0]
        ei, shift = radius_graph(pos, cutoff, cell=cell, pbc=pbc)
        if ei.shape[1] == 0:
            raise ValueError(f"system {k} has no edges within cutoff — too sparse for this model")
        pos_all.append(pos)
        Z_all.append(Z)
        ei_all.append(ei + node_off)                           # per-system node offset
        shift_all.append(shift)
        batch_all.append(torch.full((n,), k, dtype=torch.long))
        se = csd_embedding(weights, torch.tensor([charge]), torch.tensor([spin]),
                           sphere_channels, dataset=task)       # [1, C]
        sys_all.append(se.expand(n, -1))
        natoms.append(n)
        node_off += n
    return BatchedGraph(
        pos=torch.cat(pos_all, dim=0),
        Z=torch.cat(Z_all, dim=0),
        edge_index=torch.cat(ei_all, dim=1),
        cell_shift=torch.cat(shift_all, dim=0),
        batch=torch.cat(batch_all, dim=0),
        natoms=natoms,
        sys_emb=torch.cat(sys_all, dim=0),
    )


def assemble_orb(systems, r_max, max_num_neighbors):
    """Build the block-diagonal graph for ``systems`` under Orb's edge convention -- the Orb-family
    counterpart to ``assemble``, sharing the same disjoint-union discipline (each system's neighbour
    list is built independently, never on the concatenated positions, then edges are offset by the
    running node count so the union stays block-diagonal) but producing Orb's fields: senders/
    receivers in Orb's ``tgt/src`` convention and no eSCN ``sys_emb``. Returns an
    :class:`OrbBatchedGraph`.

    Enforces Orb's own per-atom neighbour cap (``max_num_neighbors``, 20 for ``-20`` checkpoints /
    120 otherwise) per-system, exactly as ``OrbCalculator.calculate`` does for a single structure:
    a denser system inside a batch still raises the same clear error rather than silently exceeding
    the cap or being truncated. Raises if any system has no edges within ``r_max`` (too sparse).
    """
    if len(systems) == 0:
        raise ValueError("empty batch")
    pos_all, Z_all, send_all, recv_all, shift_all, batch_all = [], [], [], [], [], []
    natoms, charges, spins = [], [], []
    node_off = 0
    for k, system in enumerate(systems):
        pos, Z, charge, spin, cell, pbc = _as_atoms_fields(system)
        n = Z.shape[0]
        ei, shift = radius_graph(pos, r_max, cell=cell, pbc=pbc)
        if ei.shape[1] == 0:
            raise ValueError(f"system {k} has no edges within cutoff — too sparse for this model")
        src, tgt = ei
        senders, receivers = tgt, src           # Orb's edge convention (opposite of fairchem/UMA)
        max_deg = max(int(torch.bincount(senders, minlength=n).max()),
                     int(torch.bincount(receivers, minlength=n).max()))
        if max_deg > max_num_neighbors:
            raise ValueError(
                f"system {k} has an atom with {max_deg} neighbours within the {r_max} A cutoff, "
                f"exceeding this checkpoint's max_num_neighbors={max_num_neighbors}. Orb's own "
                "reference truncates to the closest max_num_neighbors per atom; this port does "
                "not implement that truncation (unverified against the reference), so it refuses "
                "rather than silently return a different graph than Orb's own inference would use."
            )
        pos_all.append(pos)
        Z_all.append(Z)
        send_all.append(senders + node_off)
        recv_all.append(receivers + node_off)
        shift_all.append(shift)
        batch_all.append(torch.full((n,), k, dtype=torch.long))
        natoms.append(n)
        charges.append(charge)
        spins.append(spin)
        node_off += n
    return OrbBatchedGraph(
        pos=torch.cat(pos_all, dim=0),
        Z=torch.cat(Z_all, dim=0),
        senders=torch.cat(send_all, dim=0),
        receivers=torch.cat(recv_all, dim=0),
        cell_shift=torch.cat(shift_all, dim=0),
        batch=torch.cat(batch_all, dim=0),
        natoms=natoms,
        charges=charges,
        spins=spins,
    )
