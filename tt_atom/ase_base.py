"""Shared ASE-calculator scaffolding for the device-resident engines.

``calculator.TTAtomCalculator`` (UMA) and ``orb_calculator.OrbCalculator`` (Orb) wrap
architecturally different backbones, but the ASE glue around them is identical: open (and
eventually close) one TT device, and pack the computed energy/forces/stress into ASE's ``results``
dict in ASE's units/shapes. That glue lives here so both share one implementation; each subclass
keeps its own ``calculate`` (the graph build + device forward, which is genuinely family-specific)
and, if it holds trace engines, a ``_release_engines`` hook run on close."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator

from . import device as D


class DeviceCalculator(Calculator):
    """ASE calculator backed by a TT device: owns the device lifecycle and results packing."""

    implemented_properties = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(self, *, device=None, device_id=0, fast=False, trace_region_size=0, **kwargs):
        super().__init__(**kwargs)
        self.fast = fast
        self._owns_device = device is None
        self.device = device if device is not None else D.open_device(
            device_id, trace_region_size=trace_region_size)

    def _release_engines(self):
        """Hook: subclasses release any captured trace engines here. Default: nothing to release."""

    def close(self):
        self._release_engines()
        if self._owns_device and self.device is not None:
            import ttnn

            ttnn.close_device(self.device)
            self.device = None

    def _store_results(self, atoms, energy, forces, stress=None):
        """Pack ASE's ``results`` dict: total + per-atom energy, forces, and optional stress.
        ``forces``/``stress`` may be torch tensors or numpy arrays; ``stress`` is already in ASE's
        Voigt-6 (UMA) / per-checkpoint (Orb) form."""
        E = float(energy)
        n = len(atoms)
        F = forces.detach().numpy() if torch.is_tensor(forces) else np.asarray(forces)
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["energies"] = np.full(n, E / n, dtype=np.float64)
        self.results["forces"] = F.astype(np.float64)
        if stress is not None:
            S = stress.detach().numpy() if torch.is_tensor(stress) else np.asarray(stress)
            self.results["stress"] = S.astype(np.float64)
