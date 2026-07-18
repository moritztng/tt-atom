"""Host geometry: the differentiable ``pos -> {NEF edge tensors}`` map for PET-MAD (torch, host).

Mirrors ``metatrain.pet.modules.structures.systems_to_batch`` (the PET-MAD v1.5.0
``adaptive_cutoff_method="grid"`` path) plus the NEF / reverse-neighbor-list build from
``metatrain.pet.modules.nef``. Everything is pure torch (no metatomic / metatensor) so it
runs in the ttnn env exactly like ``tt_atom/orb_geometry.py`` does for Orb.

Edge ordering. The reference uses ``vesin``'s cell-list neighbour list, whose traversal
order is not reproducible in pure torch. PET's forward is invariant to the within-center
edge order given a self-consistent ``reverse_neighbor_index`` (the ij<->ji map is built
from the same order), so this module sorts edges canonically by
``(center, neighbor, cell_shift)`` and builds the reverse-neighbor list from that order.
The forward energy/forces are order-independent and match the reference bit-exactly;
the per-GNN-layer slot-ordered internals are verified against fixtures re-captured in
this same canonical order (see ``tests/gen_golden_pet.py``).

Sign convention. PET: ``edge_vec = pos[neighbor] - pos[center] + cell_contrib`` (the
``i -> j`` edge with ``center = i``). ``tt_atom.geometry.radius_graph`` returns
``edge_vec = pos[src] - pos[tgt] + shift`` with ``src`` the imaged source, so here
``src = neighbor``, ``tgt = center`` and the cartesian ``shift`` is the cell
contribution (``cell_shift_int @ cell``).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# --- vendored from metatrain.pet.modules.utilities (MIT, lab-cosmo) ---


def cutoff_func_bump(values: torch.Tensor, cutoff: torch.Tensor, width: float) -> torch.Tensor:
    scaled = (values - (cutoff - width)) / width
    clamped = scaled.clamp(1e-6, 1.0 - 1e-6)
    return 0.5 * (1.0 + torch.tanh(1.0 / torch.tan(math.pi * clamped)))


# --- vendored from metatrain.pet.modules.nef (MIT, lab-cosmo) ---


def get_nef_indices(centers: torch.Tensor, num_neighbors: torch.Tensor, n_edges_per_node: int):
    n_nodes = num_neighbors.shape[0]
    arange = torch.arange(n_edges_per_node, device=centers.device)
    nef_mask = arange.view(1, -1).expand(n_nodes, -1) < num_neighbors.view(-1, 1)
    argsort = torch.argsort(centers, stable=True)
    n_edges = centers.shape[0]
    sorted_centers = centers.index_select(0, argsort)
    starts = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    position_within = torch.arange(n_edges, device=centers.device) - starts.index_select(0, sorted_centers)
    nef_indices = torch.zeros(n_nodes * n_edges_per_node, dtype=torch.long, device=centers.device)
    flat_target = sorted_centers * n_edges_per_node + position_within
    nef_indices[flat_target] = argsort
    nef_indices = nef_indices.view(n_nodes, n_edges_per_node)
    nef_to_edges_neighbor = torch.empty_like(centers, dtype=torch.long)
    nef_to_edges_neighbor[argsort] = position_within
    return nef_indices, nef_to_edges_neighbor, nef_mask


def get_corresponding_edges(centers: torch.Tensor, neighbors: torch.Tensor, cell_shifts: torch.Tensor) -> torch.Tensor:
    if centers.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=centers.device)
    centers = centers.to(torch.int64); neighbors = neighbors.to(torch.int64); cell_shifts = cell_shifts.to(torch.int64)
    min_per_axis = cell_shifts.amin(dim=0)
    cs_norm = cell_shifts - min_per_axis
    neg_cs_norm = -cell_shifts - min_per_axis
    max_per_axis = cs_norm.amax(dim=0) + 1
    max_centers_neighbors = centers.amax() + 1
    size_z = max_per_axis[2]; size_yz = max_per_axis[1] * size_z; size_xyz = max_per_axis[0] * size_yz
    size_total = max_centers_neighbors * size_xyz
    unique_id = (centers * size_total + neighbors * size_xyz
                 + cs_norm[:, 0] * size_yz + cs_norm[:, 1] * size_z + cs_norm[:, 2])
    unique_id_inverse = (neighbors * size_total + centers * size_xyz
                         + neg_cs_norm[:, 0] * size_yz + neg_cs_norm[:, 1] * size_z + neg_cs_norm[:, 2])
    corresponding_edges = torch.empty_like(centers)
    corresponding_edges[unique_id.argsort()] = unique_id_inverse.argsort()
    return corresponding_edges


def edge_array_to_nef(edge_array, nef_indices, mask=None, fill_value=0.0):
    if mask is None:
        return edge_array[nef_indices]
    shape = mask.shape + (1,) * (len(edge_array.shape) - 1)
    return torch.where(mask.reshape(shape), edge_array[nef_indices], fill_value)


def compute_reversed_neighbor_list(nef_indices, corresponding_edges, nef_to_edges_neighbor, nef_mask):
    reverse_edge_idx = corresponding_edges[nef_indices]
    rnl = nef_to_edges_neighbor[reverse_edge_idx]
    return rnl.masked_fill(~nef_mask, 0)


# --- the geometry ---


def _radius_graph_int_shift(pos, cutoff, cell, pbc):
    """``radius_graph`` returning INTEGER cell shifts (a, b, c) instead of cartesian.

    Reuses ``tt_atom.geometry.radius_graph`` (which returns cartesian ``shift =
    a*cell[0] + b*cell[1] + c*cell[2]``) and inverts to integers via ``shift @ inv(cell).T``.
    Round-trip is bit-exact for the lattice vectors PET-MAD ships. Convention:
    ``edge_index[0] = neighbor`` (src), ``edge_index[1] = center`` (tgt), so
    ``edge_vec = pos[neighbor] - pos[center] + ints @ cell`` matches PET's
    ``positions[neighbors] - positions[centers] + cell_contrib``."""
    from .geometry import radius_graph

    edge_index, shift = radius_graph(pos, cutoff, cell=cell, pbc=pbc)
    src, tgt = edge_index  # src=neighbor, tgt=center
    if cell is not None and bool(torch.as_tensor(pbc).any()):
        inv = torch.linalg.inv(cell.to(pos.dtype))
        ints = torch.round(shift.to(pos.dtype) @ inv.transpose(0, 1)).long()
    else:
        ints = torch.zeros((int(edge_index.shape[1]), 3), dtype=torch.long, device=pos.device)
    return src.long(), tgt.long(), ints


def _canonical_sort(centers, neighbors, ints):
    """Lexicographic sort by (center, neighbor, a, b, c) — deterministic, so the
    reverse-neighbor-list build is self-consistent and the forward is reproducible
    across runs (vesin's own order is not pure-torch reproducible; see module docstring)."""
    keys = (centers * (1 << 25) + neighbors * (1 << 18)
            + (ints[:, 0] + 16) * (1 << 12) + (ints[:, 1] + 16) * (1 << 6) + (ints[:, 2] + 16))
    return torch.argsort(keys)


def host_pet_geometry(pos: torch.Tensor, atomic_numbers: torch.Tensor, *,
                       cell: Optional[torch.Tensor] = None, pbc: Optional[torch.Tensor] = None,
                       cfg: dict) -> dict:
    """Differentiable ``pos -> {NEF edge tensors + NEF index tables}`` for one system.

    Mirrors ``metatrain.pet.modules.structures.systems_to_batch`` for the
    ``adaptive_cutoff_method="grid"`` path (PET-MAD v1.5.0). Returns a dict with:

    - ``element_indices_nodes`` [N] (pos-independent species index)
    - ``element_indices_neighbors`` [N, Dmax] (NEF, padded)
    - ``edge_vectors`` [N, Dmax, 3], ``edge_distances`` [N, Dmax], ``cutoff_factors`` [N, Dmax] (NEF)
    - ``padding_mask`` [N, Dmax] (bool), ``reverse_neighbor_index`` [N, Dmax] (int64)
    - ``centers`` [E], ``neighbors`` [E], ``cell_shifts`` [E, 3] (post-filter, flat)
    - ``nef_to_edges_neighbor`` [E], ``atomic_cutoffs_stats`` [N] (detached per-atom cutoff)
    - ``num_nodes`` N, ``max_edges_per_node`` Dmax

    ``pos`` must have ``requires_grad=True`` for the conservative-force VJP; the
    pos-dependent outputs (``edge_vectors``, ``edge_distances``, ``cutoff_factors``)
    carry grad, the index tables and ``padding_mask`` are detached.
    """
    cutoff = float(cfg["cutoff"])
    cutoff_width = float(cfg["cutoff_width"])
    cutoff_width_adaptive = float(cfg["cutoff_width_adaptive"])
    num_neighbors_adaptive = float(cfg["num_neighbors_adaptive"])
    atomic_types = cfg["atomic_types"]  # list[int], 1..102

    # species -> species index (1..102 -> 0..101)
    species_to_index = torch.full((max(atomic_types) + 1,), -1, dtype=torch.long, device=pos.device)
    for i, z in enumerate(atomic_types):
        species_to_index[z] = i

    neighbors_raw, centers_raw, ints_raw = _radius_graph_int_shift(pos, cutoff, cell, pbc)
    # canonical order (deterministic; vesin's order is not pure-torch reproducible)
    order = _canonical_sort(centers_raw, neighbors_raw, ints_raw)
    centers = centers_raw.index_select(0, order)
    neighbors = neighbors_raw.index_select(0, order)
    cell_shifts = ints_raw.index_select(0, order)

    # edge vectors / distances (differentiable wrt pos)
    cell_contrib = cell_shifts.to(pos.dtype) @ cell.to(pos.dtype) if cell is not None else torch.zeros_like(cell_shifts, dtype=pos.dtype)
    edge_vectors = pos[neighbors] - pos[centers] + cell_contrib
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15
    num_nodes = atomic_numbers.shape[0]

    # adaptive cutoff (grid) -> per-atom cutoffs (differentiable wrt edge_distances)
    from .pet_adaptive_cutoff import get_adaptive_cutoffs_grid

    atomic_cutoffs = get_adaptive_cutoffs_grid(
        centers, edge_distances, num_neighbors_adaptive, num_nodes, cutoff, cutoff_width_adaptive)
    atomic_cutoffs_stats = atomic_cutoffs.detach()
    pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
    keep = torch.nonzero(edge_distances <= pair_cutoffs).squeeze(-1)
    centers = centers.index_select(0, keep)
    neighbors = neighbors.index_select(0, keep)
    cell_shifts = cell_shifts.index_select(0, keep)
    edge_vectors = edge_vectors.index_select(0, keep)
    edge_distances = edge_distances.index_select(0, keep)
    pair_cutoffs = pair_cutoffs.index_select(0, keep)

    # cutoff factors (Bump, differentiable)
    cutoff_factors = cutoff_func_bump(edge_distances, pair_cutoffs, cutoff_width)

    # NEF layout
    num_neighbors = torch.bincount(centers, minlength=num_nodes)
    max_edges_per_node = int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(centers, num_neighbors, max_edges_per_node)

    element_indices_nodes = species_to_index[atomic_numbers.long()]
    element_indices_neighbors_flat = element_indices_nodes[neighbors]
    edge_vectors_nef = edge_array_to_nef(edge_vectors, nef_indices)
    edge_distances_nef = torch.sqrt(torch.sum(edge_vectors_nef ** 2, dim=2) + 1e-15)
    element_indices_neighbors = edge_array_to_nef(element_indices_neighbors_flat, nef_indices)
    cutoff_factors_nef = edge_array_to_nef(cutoff_factors, nef_indices, nef_mask, 0.0)

    corresponding_edges = get_corresponding_edges(centers, neighbors, cell_shifts)
    reversed_neighbor_list = compute_reversed_neighbor_list(nef_indices, corresponding_edges, nef_to_edges_neighbor, nef_mask)
    neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)
    reverse_neighbor_index = neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
    # replace padded indices with unique values (matches the reference; avoids the
    # torch.index_select backward slowdown from many duplicated indices)
    num_padded = reverse_neighbor_index.numel() - centers.shape[0]
    if num_padded > 0:
        reverse_neighbor_index = reverse_neighbor_index.clone()
        reverse_neighbor_index[~nef_mask] = torch.arange(num_padded, device=pos.device)

    return dict(
        element_indices_nodes=element_indices_nodes,
        element_indices_neighbors=element_indices_neighbors,
        edge_vectors=edge_vectors_nef,
        edge_distances=edge_distances_nef,
        cutoff_factors=cutoff_factors_nef,
        padding_mask=nef_mask,
        reverse_neighbor_index=reverse_neighbor_index,
        centers=centers, neighbors=neighbors, cell_shifts=cell_shifts,
        nef_to_edges_neighbor=nef_to_edges_neighbor,
        atomic_cutoffs_stats=atomic_cutoffs_stats,
        num_nodes=num_nodes, max_edges_per_node=max_edges_per_node,
    )

