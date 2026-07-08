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
    charge: float = 0.0         # shared system charge (a batch is one composition => one charge)

    @property
    def K(self):
        return len(self.natoms)

    def segment_matrix(self):
        """One-hot segment matrix ``seg[K, Ntot]`` (``seg[k, n] = 1`` iff atom n in system k) for
        the segment-sum energy readout."""
        seg = torch.zeros(self.K, self.pos.shape[0])
        seg[self.batch, torch.arange(self.pos.shape[0])] = 1.0
        return seg


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
    charges = []
    node_off = 0
    for k, system in enumerate(systems):
        pos, Z, charge, spin, cell, pbc = _as_atoms_fields(system)
        charges.append(charge)
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
    # charge_balanced_channels needs a per-system charge target; a merged bundle is one composition,
    # so the batch shares one charge.
    if len(set(charges)) != 1:
        raise ValueError(f"evaluate_batch needs one shared charge; got {sorted(set(charges))}")
    charge = charges[0]
    # A charged batch's per-system target is charge/natoms — a per-system quantity the batched balance
    # currently expresses as one scalar (charge/natoms[0]). That is only exact when all systems share
    # an atom count, so require it (the reduced-composition guard permits e.g. CH-reducible systems of
    # different sizes). Neutral batches (target 0) are unaffected.
    if charge != 0.0 and len(set(natoms)) > 1:
        raise ValueError(
            "charged uma-s-1.2 batched forces need equal atom counts per system (the per-system "
            f"charge/natoms target differs across sizes {sorted(set(natoms))}); batch equal-size "
            "systems, or evaluate the charged systems one at a time.")
    return BatchedGraph(
        pos=torch.cat(pos_all, dim=0),
        Z=torch.cat(Z_all, dim=0),
        edge_index=torch.cat(ei_all, dim=1),
        cell_shift=torch.cat(shift_all, dim=0),
        batch=torch.cat(batch_all, dim=0),
        natoms=natoms,
        sys_emb=torch.cat(sys_all, dim=0),
        charge=charge,
    )
