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
    implemented_properties = ["energy", "energies", "free_energy", "forces", "stress"]

    def __init__(self, bundle, task_name=None, device=None, device_id=0, gamma=0.0,
                 fast=False, trace=False, trace_region_size=400_000_000, **kwargs):
        """``bundle`` is a TT-Atom weight bundle (path or ``WeightBundle``) exported for a fixed
        (composition, charge, spin, task): UMA's MoLE routing consumes the dataset token, so the
        task is baked in at merge time and cannot be switched at runtime. ``task_name`` mirrors
        ``fairchem.core.FAIRChemCalculator(task_name=...)``; when given it must match the bundle's
        task (a mismatch raises, rather than silently using the wrong normalizer).

        ``trace=True`` captures the device forward+backward once and replays it each step for a
        fixed topology (MD / relaxation): ~2x fewer host dispatches, bit-for-bit the same forces.
        The neighbour list is rechecked every step and the trace is re-captured automatically if
        an atom crosses the cutoff, so results are always correct. When passing your own
        ``device`` with ``trace=True``, open it with a non-zero ``trace_region_size``."""
        super().__init__(**kwargs)
        if isinstance(bundle, str):
            bundle = WeightBundle.load(bundle)
        self.bundle = bundle
        self.cfg = bundle.config
        self.C = self.cfg["sphere_channels"]
        self.fast = fast
        self.trace = trace
        if task_name is not None and task_name != bundle.task:
            raise ValueError(
                f"task_name={task_name!r} does not match this bundle's task {bundle.task!r}. "
                f"UMA's MoLE routing bakes the task into the merged bundle; export a bundle for "
                f"{task_name!r} (tools/export_weights.py --task {task_name}) to use that task.")
        self._owns_device = device is None
        self.device = device if device is not None else D.open_device(
            device_id, trace_region_size=trace_region_size if trace else 0)
        self._engine = None
        self._engine_edges = None
        w = bundle.weights
        self.backbone = Backbone(w, self.device, self.cfg, bundle.to_grid_mat,
                                 bundle.from_grid_mat, fast=fast)
        self.geo = HostGeometry(w, self.cfg, bundle.to_m, bundle.gauss_offset,
                                bundle.gauss_coeff, gamma=gamma,
                                coefficient_index=bundle.coefficient_index)
        self._w = w
        # energy normalizer (real checkpoints: E = rmsd*E_raw + mean + sum_i refs[Z_i],
        # F = rmsd*F_raw); identity for the random-weight bundles (rmsd=1, mean=0, refs=None)
        self.scale_rmsd = bundle.scale_rmsd
        self.scale_mean = bundle.scale_mean
        self.elem_refs = bundle.elem_refs
        self.task = self.task_name = bundle.task

    @classmethod
    def from_uma(cls, model="uma-s-1", task_name="omol", atoms=None, charge=0, spin=1,
                 refenv=None, checkpoint=None, cache_dir=None, device=None, device_id=0,
                 fast=False, trace=False, **kwargs):
        """fairchem-parallel entry point: return a ready calculator for ``atoms``, auto-building
        and caching the composition-specific merged bundle on first use.

        Mirrors ``FAIRChemCalculator`` in spirit — you hand it a structure + task and get a
        calculator back — but hides the two frictions inherent to running UMA on ttnn:

          * MoLE routing bakes one merged bundle per *(reduced composition, charge, spin, task)*,
            so we hash those into a cache key under ``~/.cache/tt_atom/bundles`` (override with
            ``$TT_ATOM_CACHE`` or ``cache_dir``).
          * ttnn (numpy<2) and fairchem (numpy>=2) cannot share a process, so the *build* runs the
            reference env as a subprocess. Resolution order: ``refenv`` arg > ``$TT_ATOM_REFENV`` >
            ``~/.ttatom_run/refenv/bin/python``. A **cache hit needs no fairchem/refenv at all** —
            it is a plain load, which is the common path.

        ``atoms`` is required (it determines the composition). When it carries ``info['charge']`` /
        ``info['spin']`` those win over the args, so the bundle is merged with the exact charge/spin
        the runtime will read back. First use per composition logs an honest one-time build notice.
        """
        from . import bundle_cache as BC

        if atoms is None:
            raise ValueError(
                "from_uma needs `atoms` to determine the composition — MoLE bakes one bundle per "
                "reduced composition/charge/spin/task, so there is no way to pick (or build) a "
                "bundle without the structure. Pass the Atoms you want to run."
            )
        # an explicit charge/spin on the atoms wins: it is what `calculate` reads back at runtime,
        # so the merge must use the same value for a consistent result.
        charge = atoms.info.get("charge", charge)
        spin = atoms.info.get("spin", spin)
        path = BC.get_or_build(atoms, model=model, task=task_name, charge=charge, spin=spin,
                               refenv=refenv, checkpoint=checkpoint, cache_dir=cache_dir)
        return cls(str(path), task_name=task_name, device=device, device_id=device_id,
                   fast=fast, trace=trace, **kwargs)

    def close(self):
        if self._engine is not None:
            self._engine.close()
            self._engine = None
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

        pbc = np.asarray(atoms.get_pbc())
        cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=torch.float32) if pbc.any() else None
        edge_index, edge_cell_shift = radius_graph(pos, self.cfg["cutoff"], cell=cell, pbc=pbc)
        if edge_index.shape[1] == 0:
            raise ValueError("no edges within cutoff — system too sparse for this model")
        sys_emb = csd_embedding(self._w, charge, spin, self.C,
                                dataset=self.task)[torch.zeros(Z.shape[0], dtype=torch.long)]

        # stress is autograd of energy wrt a symmetric strain — only meaningful for a periodic
        # cell (variable-cell relaxation / NPT); request it when ASE asks or a cell is present.
        want_stress = cell is not None and ("stress" in properties or pbc.all())
        virial = None
        if self.trace:
            E, F = self._traced(pos, Z, edge_index, edge_cell_shift, sys_emb)
        elif want_stress:
            E, F, virial = Fmod.energy_and_forces(self.backbone, self.geo, pos, Z, edge_index,
                                                  sys_emb, edge_cell_shift=edge_cell_shift,
                                                  compute_stress=True)
        else:
            E, F = Fmod.energy_and_forces(self.backbone, self.geo, pos, Z, edge_index, sys_emb,
                                          edge_cell_shift=edge_cell_shift)
        # apply the per-task energy normalizer + element references (forces/virial scale by rmsd)
        E = self.scale_rmsd * E + self.scale_mean
        if self.elem_refs is not None:
            E += float(self.elem_refs[Z].sum())
        F = self.scale_rmsd * F
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["energies"] = np.full(len(atoms), E / len(atoms), dtype=np.float64)
        self.results["forces"] = F.detach().numpy().astype(np.float64)
        if virial is not None:
            from ase.stress import full_3x3_to_voigt_6_stress

            # stress = (rmsd * dE_raw/dstrain) / V; fairchem's convention (uma/outputs.py)
            stress = self.scale_rmsd * virial.detach().numpy().astype(np.float64) / atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

    def evaluate_batch(self, systems, properties=("energy", "forces")):
        """Disjoint-union batched evaluation — K systems in ONE device forward (fairchem/PyG style).

        ``systems`` is a list of ASE ``Atoms`` (or ``(positions, atomic_numbers)`` / dicts). The
        systems are concatenated into one block-diagonal graph, evaluated in a single device call,
        and the per-system energies recovered by segment-sum; forces (when requested) come from the
        one shared analytic backward (block-diagonal => each atom's own-system force). This is the
        throughput path for the dispatch-bound regime of *many small systems*.

        Returns ``dict(energy=np.ndarray[K], forces=list[np.ndarray[N_k, 3]] | None)`` with the
        per-system energy normalizer applied, mirroring the single-system ``calculate`` results.

        All systems must share this bundle's reduced composition: a merged uma-s-1 bundle bakes the
        MoLE expert routing for one composition (fairchem's merged batched inference requires the
        same), so the batch is e.g. conformers / an MD ensemble of one molecule."""
        from . import disjoint

        bg = disjoint.assemble(systems, self.cfg["cutoff"], self._w, self.C, task=self.task)
        want_forces = "forces" in properties
        E_raw, F = Fmod.energy_and_forces_batch(self.backbone, self.geo, bg,
                                                 compute_forces=want_forces)
        energies, forces_out, off = [], [], 0
        for k, n in enumerate(bg.natoms):
            Ek = self.scale_rmsd * float(E_raw[k]) + self.scale_mean
            if self.elem_refs is not None:
                Ek += float(self.elem_refs[bg.Z[off:off + n]].sum())
            energies.append(Ek)
            if want_forces:
                Fk = (self.scale_rmsd * F[off:off + n]).detach().numpy().astype(np.float64)
                forces_out.append(Fk)
            off += n
        return dict(energy=np.array(energies), forces=forces_out if want_forces else None)

    def _traced(self, pos, Z, edge_index, edge_cell_shift, sys_emb):
        """Trace-replayed energy+forces; (re)captures when the neighbour list changes."""
        from .trace import TracedEngine

        changed = (self._engine_edges is None or self._engine_edges.shape != edge_index.shape
                   or not torch.equal(self._engine_edges, edge_index))
        if changed:
            if self._engine is not None:
                self._engine.close()
            self._engine = TracedEngine(self.backbone, self.geo, Z, edge_index, sys_emb,
                                        edge_cell_shift=edge_cell_shift)
            self._engine_edges = edge_index.clone()
        return self._engine(pos)
