"""Molecular dynamics of a periodic crystal with Orb-v3 forces on Tenstorrent.

Orb-v3 (``orb-v3-conservative-inf-omat``, OMat24) is a *materials* potential, so this is the
periodic-crystal counterpart to ``examples/md.py`` (UMA, single molecule): real conservative
forces ``F = -dE/dpos`` come off the device every timestep and drive ASE's Langevin thermostat.

Two things make it fast and simple:

* The neighbour list is frozen at the initial geometry and the device forward+backward is
  trace-captured once, then replayed (``tt_atom.orb_trace.OrbTracedEngine``). This is exact for a
  solid crystal (atoms vibrate about their lattice sites and never cross the cutoff, so the
  topology is genuinely constant) and keeps the tensor shapes fixed so the program cache hits --
  rebuilding the graph each step instead changes shapes and recompiles kernels every step.
* Orb's node feature is atomic-number-only, so for a monatomic crystal every row is identical and
  one row (from any golden bundle of the same checkpoint) tiles to any supercell size -- no
  reference-env export needed for the MD system itself.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python examples/orb_md.py \
        --weights si_supercell_orb.npz --nx 3 --ny 3 --nz 3 --steps 300 --temp 900 --out traj.extxyz

The weight bundle is produced by ``tests/gen_golden_orb.py --ckpt conservative-inf-omat`` in the
``orb-models`` reference env; its weights are system-independent (see ``docs/orb-port.md``).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes


class OrbDeviceCalculator(Calculator):
    """ASE calculator: energy + conservative forces from Orb-v3 on a Tenstorrent card.

    Single-element crystal only (tiles one node-feature row to the whole supercell). The
    neighbour list is fixed at the first geometry and the device graph is trace-captured, so use
    it for solid-state MD / relaxation where the topology does not change."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, weights_path, device_id=0, r_max=6.0, *, fast=False):
        super().__init__()
        from tt_atom.device import open_device
        from tt_atom.orb_weights import OrbWeights
        from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead

        self.r_max = r_max
        self.device = open_device(device_id, trace_region_size=400_000_000)
        gw = OrbWeights.load(weights_path)
        cfg, w = gw.config, gw.weights
        self.w = w
        L = cfg["num_message_passing_steps"]
        self.enc = Encoder(w, self.device, node_in=cfg["node_embed_size"],
                           edge_in=cfg["edge_embed_size"], latent_dim=cfg["latent_dim"],
                           hidden_dim=1024, fast=fast)
        self.layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", self.device,
                                                 latent_dim=cfg["latent_dim"], hidden_dim=1024,
                                                 fast=fast)
                       for i in range(L)]
        self.ehead = EnergyHead(w, self.device, latent_dim=cfg["latent_dim"], hidden_dim=1024,
                                fast=fast)
        self._node_row = gw.host("node_feat")[0:1]     # atomic-number-only => identical per atom
        self.engine = None
        self.step_ms = []
        self.n_edges = None

    def _build_engine(self, atoms):
        from tt_atom.geometry import radius_graph
        from tt_atom.orb_trace import OrbTracedEngine

        pos0 = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        self._Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        N = len(self._Z)
        edge_index, shift = radius_graph(pos0, self.r_max, cell=cell, pbc=[True, True, True])
        src, tgt = edge_index[0], edge_index[1]
        senders, receivers = tgt, src                  # Orb convention is the opposite of UMA's
        self.n_edges = int(senders.shape[0])
        self.engine = OrbTracedEngine(
            self.enc, self.layers, self.device, senders=senders, receivers=receivers,
            atomic_numbers=self._Z, node_feat=self._node_row.repeat(N, 1), ehead=self.ehead,
            cell_shift=shift, r_max=self.r_max)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from tt_atom.orb_model import host_conservative_force_denormalize, host_energy_denormalize

        if self.engine is None:
            self._build_engine(atoms)
        Z, N = self._Z, len(self._Z)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)

        t0 = time.perf_counter()
        raw_e, raw_f = self.engine(pos)                # captures on first call, replays afterwards
        self.step_ms.append((time.perf_counter() - t0) * 1e3)

        forces = host_conservative_force_denormalize(
            raw_f, N, running_var=self.w["energy_head.normalizer.bn.running_var"])
        energy = host_energy_denormalize(
            torch.as_tensor(raw_e, dtype=torch.float64), Z, N,
            running_mean=self.w["energy_head.normalizer.bn.running_mean"],
            running_var=self.w["energy_head.normalizer.bn.running_var"],
            ref_weight=self.w["energy_head.reference.linear.weight"].view(-1))
        self.results["energy"] = float(energy)
        self.results["forces"] = forces.detach().numpy().astype(np.float64)

    def close(self):
        import ttnn
        if self.engine is not None:
            self.engine.close()
        ttnn.close_device(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Orb-v3 conservative bundle (gen_golden_orb.py)")
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43, help="lattice constant (A)")
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=1.0, help="timestep (fs)")
    ap.add_argument("--temp", type=float, default=900.0, help="temperature (K)")
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction (1/fs)")
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--out", help="write the trajectory (extxyz) here")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fast", action="store_true",
                    help="use bf8 weights and hidden MLP activations (release-gated)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (args.nx, args.ny, args.nz)
    N = len(atoms)
    calc = OrbDeviceCalculator(args.weights, device_id=args.device_id, fast=args.fast)
    atoms.calc = calc

    frames = []
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp,
                                     rng=np.random.default_rng(args.seed))
        dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.temp,
                       friction=args.friction / units.fs)
        if args.out:
            dyn.attach(lambda: frames.append(atoms.copy()), interval=args.save_every)
        dyn.attach(lambda: print(f"step {dyn.nsteps:4d}  E={atoms.get_potential_energy():.4f} eV  "
                                 f"T={atoms.get_kinetic_energy() / (1.5 * units.kB * N):.1f} K",
                                 flush=True), interval=max(1, args.steps // 10))

        if args.out:
            frames.append(atoms.copy())
        e0 = atoms.get_potential_energy()
        t0 = time.perf_counter()
        dyn.run(args.steps)
        wall = time.perf_counter() - t0
        e1 = atoms.get_potential_energy()
        if args.out:
            write(args.out, frames)

        warm = sorted(calc.step_ms[1:]) or calc.step_ms      # drop the first (trace capture)
        warm_ms = warm[len(warm) // 2]
        print("\n" + "=" * 68)
        print(f"system            : {args.element} diamond ({args.nx}x{args.ny}x{args.nz} cubic cells)")
        print(f"atoms / edges     : {N} / {calc.n_edges}")
        print(f"MD                : {args.steps} steps x {args.dt} fs @ {args.temp} K (NVT Langevin)")
        print(f"energy            : {e0:.4f} -> {e1:.4f} eV  ({e1 / N:.4f} eV/atom)")
        print(f"device MD step    : {warm_ms:.1f} ms warm median (energy + analytic forces, trace/replay)")
        print(f"                  : => {1000.0 / warm_ms:.1f} MD steps/s on one Blackhole card")
        if args.out:
            print(f"trajectory        : {len(frames)} frames -> {args.out}")
        print(f"full loop wall    : {wall:.2f} s")
        print("=" * 68)
    finally:
        calc.close()


if __name__ == "__main__":
    main()
