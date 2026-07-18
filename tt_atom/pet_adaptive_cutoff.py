"""Adaptive per-atom cutoff (grid method) — vendored from
``metatrain.pet.modules.adaptive_cutoff.get_adaptive_cutoffs_grid`` (MIT, lab-cosmo).

PET-MAD v1.5.0 uses ``adaptive_cutoff_method="grid"`` (``cutoff_width_adaptive=0.5``):
a discrete probe-cutoff grid + Gaussian-weighted average of the probes whose smoothed
neighbour counts are closest to ``num_neighbors_adaptive``. Pure torch, differentiable
wrt ``edge_distances`` (so the conservative-force VJP can flow through it)."""
from __future__ import annotations

from typing import Optional

import torch

from .pet_geometry import cutoff_func_bump

DEFAULT_MIN_PROBE_CUTOFF = 0.5


def _get_effective_num_neighbors(edge_distances, probe_cutoffs, centers, num_nodes, width):
    weights = cutoff_func_bump(edge_distances.unsqueeze(0), probe_cutoffs.unsqueeze(1), width)
    probe_num_neighbors = torch.zeros((len(probe_cutoffs), num_nodes), dtype=edge_distances.dtype, device=edge_distances.device)
    probe_num_neighbors.index_add_(1, centers, weights)
    return probe_num_neighbors.T


def _get_gaussian_cutoff_weights(effective_num_neighbors, num_neighbors_adaptive, width=None):
    if effective_num_neighbors.numel() == 0:
        return torch.empty_like(effective_num_neighbors)
    diff = effective_num_neighbors - num_neighbors_adaptive
    x = torch.linspace(0, 1, effective_num_neighbors.shape[1], device=effective_num_neighbors.device, dtype=effective_num_neighbors.dtype)
    baseline = num_neighbors_adaptive * x ** 3
    diff = diff + baseline.unsqueeze(0)
    if width is None:
        eps = 1e-12
        if diff.shape[-1] == 1:
            width_t = diff.abs() * 0.5 + eps
        else:
            (width_t,) = torch.gradient(diff, dim=-1)
            width_t = width_t.abs().clamp_min(eps)
    else:
        width_t = torch.ones_like(diff) * width
    logw = -0.5 * (diff / width_t) ** 2
    weights = torch.exp(logw - logw.max())
    return weights / weights.sum(dim=1, keepdim=True)


def get_adaptive_cutoffs_grid(centers, edge_distances, num_neighbors_adaptive, num_nodes, max_cutoff, cutoff_width,
                              min_cutoff=DEFAULT_MIN_PROBE_CUTOFF, probe_spacing=None, weight_width=None):
    if probe_spacing is None:
        probe_spacing = cutoff_width / 4.0
    probe_cutoffs = torch.arange(min_cutoff, max_cutoff, probe_spacing, device=edge_distances.device, dtype=edge_distances.dtype)
    enn = _get_effective_num_neighbors(edge_distances, probe_cutoffs, centers, num_nodes, width=cutoff_width)
    weights = _get_gaussian_cutoff_weights(enn, num_neighbors_adaptive, width=weight_width)
    return probe_cutoffs @ weights.T
