"""Device (ttnn) parity tests for the PET-MAD port — the per-layer PCC gate for step 3's
device half (``tt_atom/pet_model.py``), against the canonical-order fixture captured from
the real PET under forward hooks (``tests/data/pet_mad_s_si_canon_internals.npz``).

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest tests/test_pet_device.py -q

The fixture (16-atom rattled Si golden) holds the per-layer ``gnn{0,1,2}_node_out`` /
``gnn{0,1,2}_edge_out`` (the PRE-combination transformer outputs — verified in pass 2 by
``tt_atom/pet_model_host.py`` at PCC 1.0 / max abs ~2e-5 vs the real PET) plus
``raw_energy`` / ``energy_fwd``. This test gates the device backbone + energy head
against those, on card 0.

Hard gate: per-layer node/edge PCC >= 0.999 (cleared at 0.9998-0.9999). The energy is
bf16-looser than the host's 1.15e-5 eV float32 noise floor (anticipated in the pass-3
scope); the real device number is ~0.026 eV (asserted < 0.05 eV, with the measured value
printed). See ``~/.coworker/notes/tt-atom-pet-mad-port-p3.md``.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

WEIGHTS = os.environ.get(
    "TTATOM_PET_WEIGHTS",
    str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz"),
)
FIXTURE = "tests/data/pet_mad_s_si_canon_internals.npz"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(WEIGHTS).exists() or not pathlib.Path(FIXTURE).exists(),
    reason=f"PET weights ({WEIGHTS}) or fixture ({FIXTURE}) not found",
)


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def gw():
    from tt_atom.pet_weights import PetWeights
    return PetWeights.load(WEIGHTS)


@pytest.fixture(scope="module")
def bd(gw):
    from tt_atom.pet_geometry import host_pet_geometry
    fx = np.load(FIXTURE)
    pos = torch.tensor(fx["positions"], dtype=torch.float64).requires_grad_(True)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)
    return host_pet_geometry(pos, numbers, cell=cell, pbc=pbc, cfg=gw.config)


@pytest.fixture(scope="module")
def model_out(gw, bd, device):
    from tt_atom.pet_model import PetModel, build_device_inputs
    import ttnn
    bd_dev = build_device_inputs(bd, gw.config, device)
    model = PetModel(gw.weights, device, cfg=gw.config)
    raw, node_outs, edge_outs = model.forward(bd_dev, return_layers=True)
    raw = float(ttnn.to_torch(raw).float().view(-1)[0])
    return raw, node_outs, edge_outs


def test_per_layer_parity(model_out):
    fx = np.load(FIXTURE)
    _, node_outs, edge_outs = model_out
    for i in range(3):
        gn = fx[f"gnn{i}_node_out"]
        ge = fx[f"gnn{i}_edge_out"]
        pn = _pcc(node_outs[i], gn)
        pe = _pcc(edge_outs[i], ge)
        mn = float((node_outs[i].float() - torch.tensor(gn)).abs().max())
        me = float((edge_outs[i].float() - torch.tensor(ge)).abs().max())
        print(f"\n[pet-device] layer {i}: node PCC={pn:.6f} maxabs={mn:.4f} | "
              f"edge PCC={pe:.6f} maxabs={me:.4f}")
        assert pn > 0.999, (i, pn)
        assert pe > 0.999, (i, pe)


def test_energy(model_out, gw):
    fx = np.load(FIXTURE)
    raw, _, _ = model_out
    comp = gw.composition_energy_by_z()
    scale = gw.energy_scale()
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    comp_sum = float(comp[numbers].sum())
    E = raw * scale + comp_sum
    ref = float(fx["energy_ref"][0])
    diff = abs(E - ref)
    print(f"\n[pet-device] device E = {E:.6f} eV  (ref {ref:.6f}, diff {diff:.6f} eV, "
          f"raw {raw:.6f} vs fixture {float(fx['raw_energy'][0]):.6f})")
    # Hard gate is per-layer PCC (above). Energy is bf16-looser than the host's 1.15e-5 eV
    # float32 floor; the real device number is ~0.026 eV. Assert < 0.05 eV (headroom for
    # nondeterministic tile-reduction order) and print the measured value.
    assert diff < 0.05, diff
