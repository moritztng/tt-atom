"""uma-m is UNSUPPORTED in this build: it must raise a clear error, not silently fall back.

tt-atom is the custom-kernel-only, highest-performance build for uma-s. uma-m uses spherical-
harmonic coefficient subselection: the 25 SH coefficients of the node representation are reduced
to a 19-dim ``|m|<=mmax`` m-space inside the edgewise SO(2) block, so its Wigner rotation is
RECTANGULAR (node SH 25 <-> reduced m-space 19, W=256), unlike uma-s (square 9<->9). That shape
overflows the fused_rotate kernel's L1 CB budget, and this build has no slow MAC fallback -- so
the rotation raises a clear ``RuntimeError`` naming the unsupported shape. This test anchors that
contract (uma-s is the validated target; uma-m is explicitly out of scope here).

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


def test_umam_unsupported_raises(device):
    """uma-m-1p1 (lmax=4/mmax=2) must raise a clear RuntimeError: its rectangular reduced-m Wigner
    rotation (25<->19, W=256) overflows the fused kernel's L1 budget and this build has no fallback."""
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

    with pytest.raises(RuntimeError, match="unsupported in this build"):
        Fmod.energy_and_forces(bb, geo, pos, Z, edge_index, sys_emb,
                               edge_cell_shift=edge_cell_shift)
