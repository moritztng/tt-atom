"""Conservative forces for the PET-MAD (UPET) port -- ``F = -dE/dpos``.

PET-MAD's default ASE force path is conservative: ``UPETCalculator`` returns
``-dE/dpos`` via autograd through the energy, NOT the checkpoint's
``non_conservative_force`` head (see the ``pet-mad-default-forces-are-conservative``
memory). The port mirrors that: the force is the gradient of the *energy* the
calculator reports, so an ASE MD run sees a self-consistent ``(E, F)`` pair (no
energy drift from a force/energy graph mismatch).

Route (this pass): **device forward for the energy + host autograd finish for
the forces**. The host reference backbone (``tt_atom/pet_model_host.py``) is a
pure-torch reimplementation of metatrain.pet whose autograd through
``tt_atom/pet_geometry.py``'s differentiable edge featurization already
reproduces the golden conservative forces at PCC 1.0 / max abs 9.5e-7 (pass 2).
This module runs that verified host backward for the forces and the device
backbone (``tt_atom/pet_model.py``) for the energy, so the user-facing
calculator gets the device's fast energy and the host's bit-exact forces.

This is the bounded-turn recommendation from the pass-3 plan over a full device
VJP: PET's attention is a manual ``softmax(QK^T/sqrt(d) + log(cutoff_mask))``
whose backward (the log-additive mask's contribution to ``d scores``) plus the
``LayerNorm``/``SwiGLU``/NEF-scatter adjoints are a large, bug-prone VJP surface
that the SACRED-correctness bar does not want half-built in one pass. The
trade-off is real: the host backward re-runs the whole transformer on CPU, so
the force path is *strictly slower* than host-only (the device forward is
free-ridden on top). See ``profile_forces`` for the measured cost; the full
device VJP is the known perf gap a future pass closes.

    The host backward is float32 (the same path that hits PCC 1.0 vs the golden
    forces); the device energy is bf16 (0.026 eV from the host reference, see
    ``tests/test_pet_device.py``). The two come from different graphs, so the
    reported forces are conservative w.r.t. the *host* energy (1.15e-5 eV from
    golden), not the device energy (0.026 eV from host). For MD that mismatch is
    ~0.026 eV of drift-driving inconsistency -- documented here, gated on the
    device VJP pass for a true device-conservative pair.
"""
from __future__ import annotations

import torch


def host_energy_and_forces(pos, atomic_numbers, weights, *, cfg, cell=None, pbc=None):
    """Conservative energy + forces for one system, host autograd through the
    verified reference backbone.

    ``pos`` [N, 3] (any dtype; cast to float32 to match the verified pass-2
    reference path whose weights are float32), ``atomic_numbers``
    [N] long, ``weights`` a ``PetWeights.weights`` dict, ``cfg`` its ``config``.
    Returns ``(energy_raw: float, forces: torch.Tensor[N, 3])`` where ``energy_raw``
    is the pre-scaler/pre-composition scalar (the caller applies
    ``E = raw * scale + sum_i comp[Z_i]``) and ``forces`` is ``-dE_raw/dpos`` in the
    same raw/normalized space (the caller multiplies by ``scale`` to denormalize,
    mirroring Orb's ``host_conservative_force_denormalize``).

    The energy here is the *host reference* energy (float32, 1.15e-5 eV from the
    golden), NOT the device bf16 energy -- the force is the gradient of THIS
    energy, so the pair is self-consistent. The calculator may instead report
    the device energy for the ``energy`` property (faster, 0.026 eV looser); see
    the module docstring for the trade-off.
    """
    from .pet_geometry import host_pet_geometry
    from .pet_model_host import forward_energy

    pos = pos.detach().to(torch.float32).clone().requires_grad_(True)
    bd = host_pet_geometry(pos, atomic_numbers, cell=cell, pbc=pbc, cfg=cfg)
    raw = forward_energy(bd, weights, cfg=cfg)
    g = torch.autograd.grad(raw, pos)[0]
    return float(raw.detach()), (-g)


def profile_forces(pos, atomic_numbers, weights, *, cfg, cell=None, pbc=None,
                   device=None, repeat=5):
    """Measure the device-forward (energy) and host-backward (forces) wall time
    so the route-(b) trade-off is quantified, not guessed. Returns a dict with
    ``device_forward_ms`` (the ``PetModel.forward`` energy call, averaged over
    ``repeat`` runs after a warmup) and ``host_force_ms`` (the
    ``host_energy_and_forces`` call, averaged the same way).

    ``device`` is an already-open TT device; if ``None``, the device forward is
    skipped (only the host force path is timed). The host backward dominates by
    construction (it re-runs the whole transformer on CPU); this just measures
    by how much, which is the number a future device-VJP pass would erase.
    """
    import time

    out = {}
    if device is not None:
        import ttnn

        from .pet_geometry import host_pet_geometry
        from .pet_model import PetModel, build_device_inputs

        bd = host_pet_geometry(pos.detach().to(torch.float64), atomic_numbers,
                               cell=cell, pbc=pbc, cfg=cfg)
        bd_dev = build_device_inputs(bd, cfg, device)
        model = PetModel(weights, device, cfg=cfg)
        # warmup
        for _ in range(2):
            raw = model.forward(bd_dev)
            ttnn.to_torch(raw)
        t0 = time.perf_counter()
        for _ in range(repeat):
            raw = model.forward(bd_dev)
            ttnn.to_torch(raw)
        out["device_forward_ms"] = (time.perf_counter() - t0) / repeat * 1e3

    # warmup the host path too (first call builds autograd graph caches in torch)
    for _ in range(2):
        host_energy_and_forces(pos, atomic_numbers, weights, cfg=cfg, cell=cell, pbc=pbc)
    t0 = time.perf_counter()
    for _ in range(repeat):
        host_energy_and_forces(pos, atomic_numbers, weights, cfg=cfg, cell=cell, pbc=pbc)
    out["host_force_ms"] = (time.perf_counter() - t0) / repeat * 1e3
    return out
