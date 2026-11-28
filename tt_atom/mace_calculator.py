"""ASE calculator for MACE on one TT device (energy and forces; see docs/mace-port.md)."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.neighborlist import neighbor_list

from .ase_base import DeviceCalculator


def load_weights(path):
    """The float64 state dict stored in a MACE golden npz (tests/gen_golden_mace.py), keys ``w@<name>``."""
    z = np.load(path)
    w = {k[2:]: torch.from_numpy(np.asarray(z[k])) for k in z.files if k.startswith("w@")}
    return {k: v.double() if v.is_floating_point() else v for k, v in w.items()}


class MaceCalculator(DeviceCalculator):
    implemented_properties = ["energy", "energies", "free_energy", "forces"]

    def __init__(self, weights, device=None, device_id=0, trace=True, trace_region_size=400_000_000, **kwargs):
        """``weights``: a MACE state dict or the path of an npz holding one (MIT-licensed checkpoints only).
        The neighbour list is rebuilt on every call; while it comes out the same (an MD or relaxation step that
        moves no atom across the cutoff) the device tables and, with ``trace``, the captured step are reused."""
        import ttnn

        from .mace.tt_model import Mace

        super().__init__(device=device, device_id=device_id, trace_region_size=trace_region_size if trace else 0,
                         **kwargs)
        w = load_weights(weights) if isinstance(weights, (str, bytes)) or hasattr(weights, "__fspath__") else weights
        self.r_max = float(w["radial_embedding.cutoff_fn.r_max"])
        self.model = Mace(self.device, w, ttnn.float32, trace=trace)

    def _release_engines(self):
        g = self.model._graph
        if g is not None and "trace" in g and self.device is not None:
            import ttnn

            ttnn.release_trace(self.device, g.pop("trace")[0])

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        i, j, S = neighbor_list("ijS", atoms, self.r_max)
        energy, forces = self.model.energy_forces(
            torch.from_numpy(atoms.get_atomic_numbers()).long(), torch.from_numpy(atoms.get_positions()).double(),
            torch.from_numpy(np.stack([i, j])).long(), torch.from_numpy(S @ atoms.cell.array).double())
        self._store_results(atoms, energy, forces)
