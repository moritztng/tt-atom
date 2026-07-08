"""Real-weight parity tests for uma-s-1.2 (charge-balanced channels) against fairchem.

uma-s-1.2 differs from uma-s-1 by fairchem's ``charge_balanced_channels``: the l=0 charge
channels are re-balanced (a self-adjoint per-system mean-subtraction, plus the charge/natoms
target) after every block. Without it the force PCC on the released checkpoint collapses to
~0.83; with it, parity is restored (>0.99). This module is the on-device regression for that
path, and complements the 757-system CPU-vs-TT screen written up in
``docs/uma-s-1p2-validation.md``.

Like ``test_realweight.py`` it runs only when a real-weight golden is present (the UMA checkpoint
is gated and must not be committed), so it auto-skips for anyone without access. Generate the
golden in the fairchem refenv, then run it on a card:

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_real.py \
        --system molecule --task omol --ckpt uma-s-1p2 \
        --out ~/.ttatom_run/goldens_real/ethanol_omol_uma_s_1p2.npz
    # (or add --ckpt-path /path/to/uma-s-1p2.pt to use a local checkpoint file)
    TT_VISIBLE_DEVICES=0 ~/.ttatom_run/venv/bin/python -m pytest tests/test_realweight_uma_s_1p2.py -q

Point ``TTATOM_REAL_GOLDEN_S1P2`` at a different bundle to override the default path.

Neutral ethanol/omol still exercises the balancing: the per-system l=0 mean-subtraction runs on
every block (the charge/natoms target is 0 for a neutral system). Charged-system parity (e.g.
[Cu(EDTA)]2-) is covered by the A/B screen in the validation doc.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest
import torch

REAL_GOLDEN = os.environ.get(
    "TTATOM_REAL_GOLDEN_S1P2",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/ethanol_omol_uma_s_1p2.npz"),
)

pytestmark = pytest.mark.skipif(
    not pathlib.Path(REAL_GOLDEN).exists(),
    reason=f"uma-s-1.2 real-weight golden not found at {REAL_GOLDEN} (UMA checkpoint not available)",
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


def test_config_enables_charge_balancing(rcfg):
    """The golden must be uma-s-1.2: an s-size spectral model with charge-balanced l=0 channels."""
    assert rcfg["ff_type"] == "spectral"
    assert rcfg["num_layers"] == 4
    assert rcfg["lmax"] == 2 and rcfg["mmax"] == 2
    assert rcfg["task"] == "omol"
    cs = int(rcfg.get("charge_channel_start", 0))
    ce = int(rcfg.get("charge_channel_end", 0))
    assert cs < ce, f"charge balancing inactive (cs={cs}, ce={ce}); this is not a uma-s-1.2 golden"


def test_verify_coverage(rg):
    from tt_atom.weights import WeightBundle

    b = WeightBundle.load(REAL_GOLDEN)
    ok, missing, present = b.verify_coverage()
    assert ok, f"missing weight keys: {missing}"
    assert b.scale_rmsd > 0 and b.elem_refs is not None and b.task == "omol"


def test_end_to_end_energy_forces(rg, rcfg, device):
    """Full device forward + analytic forces vs the fairchem oracle (real uma-s-1.2).

    Exercises the charge-balanced channels end to end: with balancing the force PCC is >0.99;
    without it (the pre-1.2 port) it collapses to ~0.83, so this is a genuine regression guard.
    """
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

    # pass the golden's charge so the additive charge/natoms target branch of channel balancing is
    # exercised (the fairchem oracle balanced to this charge); 0 for the neutral ethanol golden.
    E_raw, F_raw = Fmod.energy_and_forces(bb, geo, pos, Z, edge_index, sys_emb,
                                          charge=float(charge.reshape(-1)[0]))
    E = b.scale_rmsd * E_raw + b.scale_mean + float(b.elem_refs[Z].sum())
    F = b.scale_rmsd * F_raw

    E_oracle = float(rg["out@energy_oracle"][0])
    F_oracle = torch.from_numpy(rg["out@forces_oracle"].copy()).float()
    rel = abs(E - E_oracle) / abs(E_oracle)
    fpcc = _pcc(F, F_oracle)
    assert rel < 1e-2, f"energy rel err {rel} (E={E}, oracle={E_oracle})"
    assert fpcc > 0.99, f"force PCC {fpcc}"
