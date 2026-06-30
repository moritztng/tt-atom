"""Real-weight parity tests against the released uma-s-1 checkpoint.

These run only when a real-weight golden bundle is present (generated in the fairchem refenv by
``tests/gen_golden_real.py`` and stored OUTSIDE the repo, since the UMA checkpoint is gated and
must not be committed). Absent the bundle the whole module auto-skips, so the suite stays green
for anyone without UMA access.

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system molecule --task omol --out ~/.ttatom_run/goldens_real/ethanol_omol.npz
    PYTHONPATH=~/TT-Atom ~/.ttatom_run/env/bin/python -m pytest tests/test_realweight.py -q

Point ``TTATOM_REAL_GOLDEN`` at a different bundle to override the default path.

What is checked (all numbers measured on the p150, real uma-s-1, ethanol/omol):
  * MoLE host-merge anchor: the merged plain backbone reproduces the unmerged-MoE fairchem
    oracle E+F (the golden records both) to PCC>0.999;
  * the device spectral atomwise matches the golden per-module (PCC>=0.98);
  * WeightBundle.verify_coverage passes on the real merged bundle;
  * end-to-end device energy + analytic forces match the fairchem oracle
    (energy rel err < 1e-2, force PCC > 0.99).
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest
import torch

REAL_GOLDEN = os.environ.get(
    "TTATOM_REAL_GOLDEN", str(pathlib.Path.home() / ".ttatom_run/goldens_real/ethanol_omol.npz")
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"real-weight golden bundle not found at {REAL_GOLDEN} (UMA checkpoint not available)",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def rg():
    return np.load(REAL_GOLDEN)


@pytest.fixture(scope="module")
def rcfg(rg):
    return json.loads(bytes(rg["config"]).decode())


def _w(rg):
    return {k[2:]: torch.from_numpy(rg[k].copy()).float() for k in rg.files if k.startswith("w@")}


def _act(rg, name):
    return torch.from_numpy(rg[f"a@{name}"].copy()).float()


def test_config_is_real_uma_s(rcfg):
    assert rcfg["ff_type"] == "spectral"
    assert rcfg["num_layers"] == 4
    assert rcfg["lmax"] == 2 and rcfg["mmax"] == 2
    assert rcfg["chg_spin_emb_type"] == "rand_emb"
    assert rcfg["task"] == "omol"


def test_merge_anchor(rg):
    """Host MoLE merge reproduces the unmerged-MoE fairchem oracle to PCC>0.999."""
    Eo = float(rg["out@energy_oracle"][0])
    Em = float(rg["out@energy_merged_oracle"][0])
    Fo = rg["out@forces_oracle"]
    Fm = rg["out@forces_merged_oracle"]
    assert abs(Em - Eo) / abs(Eo) < 1e-6, f"merge energy rel err {abs(Em - Eo) / abs(Eo)}"
    assert _pcc(Fm, Fo) > 0.999, f"merge force PCC {_pcc(Fm, Fo)}"


def test_verify_coverage(rg):
    from tt_atom.weights import WeightBundle

    b = WeightBundle.load(REAL_GOLDEN)
    ok, missing, present = b.verify_coverage()
    assert ok, f"missing weight keys: {missing}"
    assert b.scale_rmsd > 0 and b.elem_refs is not None and b.task == "omol"


def test_spectral_atomwise_module(rg, rcfg, device):
    import ttnn

    from tt_atom.spectral import SpectralAtomwise

    w = _w(rg)
    sp = SpectralAtomwise(w, "blocks.0.atom_wise", device,
                          sphere_channels=rcfg["sphere_channels"], hidden_channels=rcfg["hidden_channels"],
                          lmax=rcfg["lmax"], mmax=rcfg["mmax"])
    x = ttnn.from_torch(_act(rg, "block0.atomwise.in0"), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(sp(x)).float()
    assert _pcc(o, _act(rg, "block0.atomwise.out0")) >= 0.98


def test_end_to_end_energy_forces(rg, rcfg, device):
    """Full device forward + analytic forces vs the fairchem oracle (real uma-s-1)."""
    from tt_atom import forces as Fmod
    from tt_atom.geometry import HostGeometry, csd_embedding
    from tt_atom.model import Backbone
    from tt_atom.weights import WeightBundle

    b = WeightBundle.load(REAL_GOLDEN)
    w = b.weights
    pos = torch.from_numpy(rg["in@pos"].copy()).float()
    Z = torch.from_numpy(rg["in@atomic_numbers"].copy()).long()
    edge_index = torch.from_numpy(rg["in@edge_index"].copy()).long()
    charge = torch.from_numpy(rg["in@charge"].copy()).float()
    spin = torch.from_numpy(rg["in@spin"].copy()).float()

    geo = HostGeometry(w, rcfg, b.to_m, b.gauss_offset, b.gauss_coeff)
    sys_emb = csd_embedding(w, charge, spin, rcfg["sphere_channels"],
                            dataset=b.task)[torch.zeros(Z.shape[0], dtype=torch.long)]
    bb = Backbone(w, device, rcfg, b.to_grid_mat, b.from_grid_mat)

    E_raw, F_raw = Fmod.energy_and_forces(bb, geo, pos, Z, edge_index, sys_emb)
    E = b.scale_rmsd * E_raw + b.scale_mean + float(b.elem_refs[Z].sum())
    F = b.scale_rmsd * F_raw

    E_oracle = float(rg["out@energy_oracle"][0])
    F_oracle = torch.from_numpy(rg["out@forces_oracle"].copy()).float()
    rel = abs(E - E_oracle) / abs(E_oracle)
    fpcc = _pcc(F, F_oracle)
    assert rel < 1e-2, f"energy rel err {rel} (E={E}, oracle={E_oracle})"
    assert fpcc > 0.99, f"force PCC {fpcc}"
