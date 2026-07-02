"""Parity for the larger uma-m-1p1 checkpoint (lmax=4, mmax=2, 10 layers).

uma-m uses spherical-harmonic coefficient subselection: the 25 SH coefficients of the node
representation are reduced to a 19-dim ``|m|<=mmax`` m-space inside the edgewise SO(2) block
(fairchem's ``prepare_wigner`` — an ``index_select`` by ``coefficient_index`` + the ``to_m``
map). The Wigner rotation is therefore rectangular (node SH 25 <-> reduced m-space 19), unlike
uma-s (square 9<->9). This test is the anchor that TT-Atom's rectangular reduced-m-space path
reproduces the released uma-m inference. uma-s-1 remains the default; uma-m is validated here.

Golden (gated, uncommitted; the checkpoint is 11 GB so the generator loads a single merged unit):

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system molecule --task omol --ckpt uma-m-1p1 --merged-only \
        --out ~/.ttatom_run/goldens_real/ethanol_omol_umam.npz
    PYTHONPATH=~/TT-Atom ~/.ttatom_run/env/bin/python -m pytest tests/test_umam.py -q
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
GOLDEN = GOLDEN_DIR / "ethanol_omol_umam.npz"


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_umam_energy_forces(device):
    """End-to-end uma-m-1p1 (lmax=4/mmax=2/10 layers) energy + analytic forces vs the fairchem
    oracle on ethanol — exercises the rectangular reduced-m-space Wigner rotation."""
    if not GOLDEN.exists():
        pytest.skip(f"uma-m golden {GOLDEN} not found (checkpoint not available)")
    from tt_atom import forces as Fmod
    from tt_atom.geometry import HostGeometry, csd_embedding, radius_graph
    from tt_atom.model import Backbone
    from tt_atom.weights import WeightBundle

    rg = np.load(GOLDEN)
    rcfg = json.loads(bytes(rg["config"]).decode())
    assert rcfg["lmax"] == 4 and rcfg["mmax"] == 2, "expected the uma-m lmax=4/mmax=2 config"
    b = WeightBundle.load(str(GOLDEN))
    assert b.coefficient_index is not None, "uma-m golden must carry coefficient_index"
    w = b.weights
    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    Z = torch.from_numpy(rg["in@atomic_numbers"].copy()).long()
    charge = torch.from_numpy(rg["in@charge"].copy()).float()
    spin = torch.from_numpy(rg["in@spin"].copy()).float()

    edge_index, edge_cell_shift = radius_graph(pos, rcfg["cutoff"])
    geo = HostGeometry(w, rcfg, b.to_m, b.gauss_offset, b.gauss_coeff,
                       coefficient_index=b.coefficient_index)
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
    assert rel < 1e-3, f"uma-m energy rel err {rel} (E={E}, oracle={E_oracle})"
    assert fpcc > 0.99, f"uma-m force PCC {fpcc}"
