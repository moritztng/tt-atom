"""``PETCalculator`` — an ASE calculator backed by the device-resident PET-MAD backbone.

The PET-MAD (UPET, ``pet-mad-s`` v1.5.0) counterpart to
:meth:`tt_atom.orb_calculator.OrbCalculator`: it shares the ASE device lifecycle + results
packing (:class:`ase_base.DeviceCalculator`) and is reachable through the same unified
``Calculator(atoms, model=...)`` front door (:mod:`tt_atom.auto`), differing only in the
backbone it drives. Like Orb, PET-MAD bakes no per-composition routing into its weights, so
there is no per-system bundle to build or cache — only a per-checkpoint weight export
(:mod:`tt_atom.pet_weight_cache`), built once ever and reused across every structure.

Energy + forces. The energy comes from the device backbone (:class:`tt_atom.pet_model.PetModel`,
bf16, ~0.026 eV from the host reference — see ``tests/test_pet_device.py``). The forces are
conservative ``F = -dE/dpos`` via the **device VJP** (:mod:`tt_atom.pet_vjp`,
:func:`tt_atom.pet_forces.device_energy_and_forces`): one device forward + one device reverse pass
produce the adjoint at the pos-dependent uploaded inputs, and a short host ``torch.autograd.grad``
finish through :mod:`tt_atom.pet_geometry`'s differentiable edge featurization turns it into
per-atom forces. The force is the gradient of the *same* device bf16 energy reported above, so the
``(E, F)`` pair is self-consistent (no ~0.026 eV host/device drift — the gap the pass-4 host-backward
route left open). PCC vs the golden forces is ~0.99 (bf16 backward noise; the float32 host route
hits 1.0 but is ~1.6x slower). The device VJP is also faster than the pass-4 host backward
(~26 ms vs ~41 ms forward+backward on the 16-atom golden, see
``pet_forces.profile_device_forces``).

Out of scope (documented, not implemented): stress (PET-MAD has no stress head and the
conservative stress would need a strain adjoint through the geometry — a later pass), the
non-conservative force head (the default ASE path does not use it, see the
``pet-mad-default-forces-are-conservative`` memory), the LLPR uncertainty ensemble, and
disjoint-union batched evaluation (single-system ``calculate`` only for now; a batched path
mirroring ``OrbCalculator.evaluate_batch`` is a later perf pass).
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from ase.calculators.calculator import all_changes

from .ase_base import DeviceCalculator


class PETCalculator(DeviceCalculator):
    def __init__(self, weights, device=None, device_id=0, fast=False, **kwargs):
        """``weights`` is a ``PetWeights`` (or a path to one, see
        ``tools/export_pet_weights.py`` / ``tt_atom.pet_weight_cache``): the upgraded
        checkpoint's config + base-PET state dict + the composition-energy table and
        energy scaler, no system-specific data. Builds the device-resident backbone once;
        every subsequent ``calculate()`` reuses it for whatever structure ASE hands it."""
        super().__init__(device=device, device_id=device_id, fast=fast, **kwargs)
        if isinstance(weights, (str, pathlib.Path)):
            from .pet_weights import PetWeights

            weights = PetWeights.load(weights)
        self.cfg = weights.config
        self._w = weights.weights
        self._weights_obj = weights
        self.scale = weights.energy_scale()
        self.comp = weights.composition_energy_by_z()

        from .pet_model import PetModel

        self.model = PetModel(self._w, self.device, cfg=self.cfg)

    @classmethod
    def from_checkpoint(cls, checkpoint="pet-mad-s-v1.5.0", refenv=None, cache_dir=None,
                        device=None, device_id=0, fast=False, **kwargs):
        """Export (or load from cache) ``checkpoint``'s weights via the upet reference env
        and return a ready calculator. Resolution order for the reference python:
        ``refenv`` arg > ``$TT_ATOM_UPETENV`` > ``~/.ttatom_run/upetenv/bin/python`` (see
        ``pet_weight_cache._resolve_upetenv``). A cache hit needs no reference env at all."""
        from . import pet_weight_cache as PWC
        from .pet_weights import PetWeights

        path = PWC.get_or_build(checkpoint, refenv=refenv, cache_dir=cache_dir)
        return cls(PetWeights.load(str(path)), device=device, device_id=device_id, fast=fast,
                   **kwargs)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import ttnn

        from .disjoint import _as_atoms_fields
        from .pet_forces import device_energy_and_forces
        from .pet_geometry import host_pet_geometry
        from .pet_model import build_device_inputs

        pos, Z, _charge, _spin, cell, pbc = _as_atoms_fields(atoms)
        want_forces = "forces" in properties

        comp_sum = float(self.comp[Z.long()].sum())

        if want_forces:
            # device VJP: one forward + one reverse pass -> (E, F) self-consistent
            # (F is the gradient of the SAME device bf16 energy reported below).
            raw_dev, F_raw = device_energy_and_forces(
                pos, Z, self._w, cfg=self.cfg,
                cell=cell.double() if cell is not None else None, pbc=pbc,
                device=self.device, model=self.model)
            E = raw_dev * self.scale + comp_sum
            F = (F_raw * self.scale).double()
            self._store_results(atoms, E, F, stress=None)
        else:
            # energy-only: device backbone (bf16), no backward pass needed.
            bd = host_pet_geometry(pos.double().requires_grad_(False), Z,
                                   cell=cell.double() if cell is not None else None,
                                   pbc=pbc, cfg=self.cfg)
            bd_dev = build_device_inputs(bd, self.cfg, self.device)
            raw_dev = self.model.forward(bd_dev)
            raw_e = float(ttnn.to_torch(raw_dev).float().view(-1)[0])
            E = raw_e * self.scale + comp_sum
            # do NOT populate results["forces"] (ASE caches results, so a later
            # get_forces() must re-enter calculate to compute them for real).
            self.results["energy"] = E
            self.results["free_energy"] = E
            self.results["energies"] = np.full(Z.shape[0], E / Z.shape[0], dtype=np.float64)
