"""Periodic (PBC) parity tests against the released uma-s-1 checkpoint on periodic systems.

Materials are UMA's flagship domain, so this is the parity anchor for the periodic path: the
host cell-aware neighbour list (``geometry.radius_graph`` with a cell + pbc) plus the shared
device backbone must reproduce the fairchem oracle. Two periodic tasks are covered when their
(gated, uncommitted) golden bundles are present, else each auto-skips:

  * omat  — bulk Si diamond, fully periodic pbc=[T,T,T];
  * oc20  — Cu(100) slab + H adsorbate, mixed pbc=[T,T,F] (catalysis).

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system bulk --task omat --out ~/.ttatom_run/goldens_real/si_omat.npz
    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system slab --task oc20 --out ~/.ttatom_run/goldens_real/cuh_oc20.npz
    PYTHONPATH=~/TT-Atom ~/.ttatom_run/env/bin/python -m pytest tests/test_periodic.py -q

What is checked per task:
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

GOLDEN_DIR = pathlib.Path(os.environ.get(
    "TTATOM_GOLDEN_DIR", str(pathlib.Path.home() / ".ttatom_run/goldens_real")))

# (task label, bundle filename) — parametrized; each case skips if its bundle is absent.
PERIODIC_CASES = [("omat", "si_omat.npz"), ("oc20", "cuh_oc20.npz")]


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _load(fname):
    path = GOLDEN_DIR / fname
    if not path.exists():
        pytest.skip(f"periodic golden {path} not found (UMA checkpoint not available)")
    return np.load(path), str(path)


def _pbc(rg):
    return rg["in@pbc"].tolist() if "in@pbc" in rg.files else [True, True, True]


@pytest.mark.parametrize("task,fname", PERIODIC_CASES)
def test_neighbour_list_matches_fairchem(task, fname):
    """The host cell-aware graph reproduces fairchem's edge set + image offsets exactly."""
    from tt_atom.geometry import radius_graph

    rg, _ = _load(fname)
    cfg = json.loads(bytes(rg["config"]).decode())
    assert cfg["task"] == task
    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    cell = torch.from_numpy(rg["in@cell"].reshape(3, 3).copy()).float()
    ei, shift = radius_graph(pos, cfg["cutoff"], cell=cell, pbc=_pbc(rg))
    edge_vec = (pos[ei[0]] - pos[ei[1]] + shift).numpy()

    ei_fc, vec_fc = rg["in@edge_index"], rg["host@edge_distance_vec"]
    assert ei.shape[1] == ei_fc.shape[1], f"edge count {ei.shape[1]} vs fairchem {ei_fc.shape[1]}"

    def keyset(ei_, vec_):
        return {(int(ei_[0][k]), int(ei_[1][k]), tuple(np.round(vec_[k], 3)))
                for k in range(ei_.shape[1])}

    assert keyset(ei.numpy(), edge_vec) == keyset(ei_fc, vec_fc)


@pytest.mark.parametrize("task,fname", PERIODIC_CASES)
def test_end_to_end_periodic_energy_forces(task, fname, device):
    """Full periodic path — our own neighbour list + device forward + analytic forces — vs the
    fairchem oracle (real uma-s-1)."""
    from tt_atom import forces as Fmod
    from tt_atom.geometry import HostGeometry, csd_embedding, radius_graph
    from tt_atom.model import Backbone
    from tt_atom.weights import WeightBundle

    rg, path = _load(fname)
    rcfg = json.loads(bytes(rg["config"]).decode())
    b = WeightBundle.load(path)
    w = b.weights
    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    Z = torch.from_numpy(rg["in@atomic_numbers"].copy()).long()
    cell = torch.from_numpy(rg["in@cell"].reshape(3, 3).copy()).float()
    charge = torch.from_numpy(rg["in@charge"].copy()).float()
    spin = torch.from_numpy(rg["in@spin"].copy()).float()

    edge_index, edge_cell_shift = radius_graph(pos, rcfg["cutoff"], cell=cell, pbc=_pbc(rg))
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
    assert rel < 1e-3, f"[{task}] energy rel err {rel} (E={E}, oracle={E_oracle})"
    assert fpcc > 0.99, f"[{task}] force PCC {fpcc}"


@pytest.mark.parametrize("task,fname", PERIODIC_CASES)
def test_stress_matches_fairchem(task, fname, device):
    """Stress (virial = symmetrized dE/dstrain, / volume) vs the fairchem oracle on a fully
    periodic cell — the anchor for variable-cell relaxation / NPT. Runs through the ASE
    ``TTAtomCalculator`` so the Voigt output + normalizer scaling + volume are all exercised.
    Skips a mixed-pbc case (stress ill-defined; oracle stored zeros)."""
    from ase import Atoms

    from tt_atom.calculator import TTAtomCalculator
    from tt_atom.weights import WeightBundle

    rg, path = _load(fname)
    if "out@stress_oracle" not in rg.files or not np.any(rg["out@stress_oracle"]):
        pytest.skip(f"[{task}] no oracle stress (mixed-pbc or pre-stress golden)")
    pbc = _pbc(rg)
    if not all(pbc):
        pytest.skip(f"[{task}] stress only validated for a fully periodic cell")

    atoms = Atoms(
        numbers=rg["in@atomic_numbers"].copy(),
        positions=rg["in@pos"].copy(),
        cell=rg["in@cell"].reshape(3, 3).copy(),
        pbc=pbc,
    )
    atoms.info.update(charge=int(rg["in@charge"][0]), spin=int(rg["in@spin"][0]))
    calc = TTAtomCalculator(WeightBundle.load(path), device=device)
    atoms.calc = calc
    stress = atoms.get_stress()                              # ASE Voigt-6
    S_oracle = rg["out@stress_oracle"]
    spcc = _pcc(stress, S_oracle)
    maxrel = float(np.max(np.abs(stress - S_oracle) / (np.abs(S_oracle) + 1e-9)))
    # stress is noisier than forces (bf16 device); accept fairchem parity by PCC or rel err
    assert spcc > 0.99 or maxrel < 1e-2, (
        f"[{task}] stress PCC {spcc}, maxrel {maxrel}\n mine={stress}\n oracle={S_oracle}")
