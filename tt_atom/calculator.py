"""``TTAtomCalculator`` — an ASE calculator backed by the device-resident eSCN-MD engine.

Wraps the host geometry + device backbone + analytic-force VJP behind ASE's interface so the
model is usable for real geometry relaxations and MD. Energy and conservative forces come from
``tt_atom.forces.energy_and_forces`` (forces are ``-dE/dpos`` via the on-device reverse pass,
not finite differences)."""
from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from . import device as D
from . import forces as Fmod
from .geometry import HostGeometry, csd_embedding, radius_graph
from .model import Backbone
from .weights import WeightBundle


class TTAtomCalculator(Calculator):
    implemented_properties = ["energy", "energies", "free_energy", "forces"]

    def __init__(self, bundle, device=None, device_id=0, gamma=0.0, fast=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(bundle, str):
            bundle = WeightBundle.load(bundle)
        self.bundle = bundle
        self.cfg = bundle.config
        self.C = self.cfg["sphere_channels"]
        self.fast = fast
        self._owns_device = device is None
        self.device = device if device is not None else D.open_device(device_id)
        w = bundle.weights
        self.backbone = Backbone(w, self.device, self.cfg, bundle.to_grid_mat,
                                 bundle.from_grid_mat, fast=fast)
        self.geo = HostGeometry(w, self.cfg, bundle.to_m, bundle.gauss_offset,
                                bundle.gauss_coeff, gamma=gamma)
        self._w = w
        # energy normalizer (real checkpoints: E = rmsd*E_raw + mean + sum_i refs[Z_i],
        # F = rmsd*F_raw); identity for the random-weight bundles (rmsd=1, mean=0, refs=None)
        self.scale_rmsd = bundle.scale_rmsd
        self.scale_mean = bundle.scale_mean
        self.elem_refs = bundle.elem_refs
        self.task = bundle.task

    def close(self):
        if self._owns_device and self.device is not None:
            import ttnn

            ttnn.close_device(self.device)
            self.device = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = torch.tensor(np.asarray(atoms.get_positions()), dtype=torch.float32)
        Z = torch.tensor(np.asarray(atoms.get_atomic_numbers()), dtype=torch.long)
        charge = torch.tensor([float(atoms.info.get("charge", 0.0))])
        spin = torch.tensor([float(atoms.info.get("spin", 0.0))])

        edge_index = radius_graph(pos, self.cfg["cutoff"])
        if edge_index.shape[1] == 0:
            raise ValueError("no edges within cutoff — system too sparse for this model")
        sys_emb = csd_embedding(self._w, charge, spin, self.C,
                                dataset=self.task)[torch.zeros(Z.shape[0], dtype=torch.long)]

        E, F = Fmod.energy_and_forces(self.backbone, self.geo, pos, Z, edge_index, sys_emb)
        # apply the per-task energy normalizer + element references (forces scale by rmsd)
        E = self.scale_rmsd * E + self.scale_mean
        if self.elem_refs is not None:
            E += float(self.elem_refs[Z].sum())
        F = self.scale_rmsd * F
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["energies"] = np.full(len(atoms), E / len(atoms), dtype=np.float64)
        self.results["forces"] = F.detach().numpy().astype(np.float64)
