"""ASE calculator for TensorNet on one TT device (energy and forces)."""
from __future__ import annotations

import torch
from ase.calculators.calculator import all_changes

from .ase_base import DeviceCalculator


class TensorNetCalculator(DeviceCalculator):
    implemented_properties = ["energy", "energies", "free_energy", "forces"]

    def __init__(self, bundle, device=None, device_id=0, layer_dtype=None, **kwargs):
        """``bundle``: what ``tensornet.potential.export`` returns, or the path of one saved with torch.save. The host
        builds the neighbour list and the geometry in float64 on every call; the network runs on the card in float32
        (``layer_dtype`` sets the interaction layers' own dtype)."""
        from .tensornet.tt_ops import runner

        super().__init__(device=device, device_id=device_id, **kwargs)
        self.bundle = bundle if isinstance(bundle, dict) else torch.load(bundle, weights_only=True)
        self.run = runner(self.device, self.bundle["w"], self.bundle["units"], layer_dtype=layer_dtype)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        from .tensornet.potential import evaluate

        super().calculate(atoms, properties, system_changes)
        self._store_results(self.atoms, *evaluate(self.bundle, self.atoms, self.run))
