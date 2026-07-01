"""Periodic (PBC) parity tests against the released uma-s-1 checkpoint on a bulk crystal.

Materials are UMA's flagship domain, so this is the parity anchor for the periodic path: the
host cell-aware neighbour list (``geometry.radius_graph`` with a cell + pbc) plus the shared
device backbone must reproduce the fairchem oracle on a periodic system. Like ``test_realweight``
the module auto-skips when the (gated, uncommitted) golden bundle is absent.

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system bulk --task omat --out ~/.ttatom_run/goldens_real/si_omat.npz
    PYTHONPATH=~/TT-Atom ~/.ttatom_run/env/bin/python -m pytest tests/test_periodic.py -q

What is checked (bulk Si diamond, omat, p150):
  * the periodic neighbour list reproduces fairchem's ``edge_index`` + ``edge_distance_vec``
    exactly (same edge set, matching image offsets) — the graph-construction anchor;
  * end-to-end device energy + analytic forces match the fairchem oracle
    (energy rel err < 1e-3, force PCC > 0.99).
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest
import torch

GOLDEN = os.environ.get(
    "TTATOM_PERIODIC_GOLDEN", str(pathlib.Path.home() / ".ttatom_run/goldens_real/si_omat.npz")
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(GOLDEN).exists(),
    reason=f"periodic golden bundle not found at {GOLDEN} (UMA checkpoint not available)",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def rg():
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def rcfg(rg):
    return json.loads(bytes(rg["config"]).decode())


def test_config_is_periodic_omat(rcfg):
    assert rcfg["task"] == "omat"
    assert rcfg["ff_type"] == "spectral" and rcfg["num_layers"] == 4


def test_neighbour_list_matches_fairchem(rg):
    """The host cell-aware graph reproduces fairchem's edge set + image offsets exactly."""
    from tt_atom.geometry import radius_graph

    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    cell = torch.from_numpy(rg["in@cell"].reshape(3, 3).copy()).float()
    cutoff = json.loads(bytes(rg["config"]).decode())["cutoff"]
    ei, shift = radius_graph(pos, cutoff, cell=cell, pbc=[True, True, True])
    edge_vec = (pos[ei[0]] - pos[ei[1]] + shift).numpy()

    ei_fc = rg["in@edge_index"]
    vec_fc = rg["host@edge_distance_vec"]
    assert ei.shape[1] == ei_fc.shape[1], f"edge count {ei.shape[1]} vs fairchem {ei_fc.shape[1]}"

    def keyset(ei_, vec_):
        return {(int(ei_[0][k]), int(ei_[1][k]), tuple(np.round(vec_[k], 3)))
                for k in range(ei_.shape[1])}

    assert keyset(ei.numpy(), edge_vec) == keyset(ei_fc, vec_fc)


def test_end_to_end_periodic_energy_forces(rg, rcfg, device):
    """Full periodic path — our own neighbour list + device forward + analytic forces — vs the
    fairchem oracle (real uma-s-1, bulk Si)."""
    from tt_atom import forces as Fmod
    from tt_atom.geometry import HostGeometry, csd_embedding, radius_graph
    from tt_atom.model import Backbone
    from tt_atom.weights import WeightBundle

    b = WeightBundle.load(GOLDEN)
    w = b.weights
    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    Z = torch.from_numpy(rg["in@atomic_numbers"].copy()).long()
    cell = torch.from_numpy(rg["in@cell"].reshape(3, 3).copy()).float()
    charge = torch.from_numpy(rg["in@charge"].copy()).float()
    spin = torch.from_numpy(rg["in@spin"].copy()).float()

    edge_index, edge_cell_shift = radius_graph(pos, rcfg["cutoff"], cell=cell, pbc=[True, True, True])
    geo = HostGeometry(w, rcfg, b.to_m, b.gauss_offset, b.gauss_coeff)
    sys_emb = csd_embedding(w, charge, spin, rcfg["sphere_channels"],
                            dataset=b.task)[torch.zeros(Z.shape[0], dtype=torch.long)]
    bb = Backbone(w, device, rcfg, b.to_grid_mat, b.from_grid_mat)

    E_raw, F_raw = Fmod.energy_and_forces(bb, geo, pos, Z, edge_index, sys_emb,
                                          edge_cell_shift=edge_cell_shift)
    E = b.scale_rmsd * E_raw + b.scale_mean + float(b.elem_refs[Z].sum())
    F = b.scale_rmsd * F_raw

    E_oracle = float(rg["out@energy_oracle"][0])
    F_oracle = torch.from_numpy(rg["out@forces_oracle"].copy()).float()
    rel = abs(E - E_oracle) / abs(E_oracle)
    fpcc = _pcc(F, F_oracle)
    assert rel < 1e-3, f"energy rel err {rel} (E={E}, oracle={E_oracle})"
    assert fpcc > 0.99, f"force PCC {fpcc}"
