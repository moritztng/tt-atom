"""Periodic NVT molecular dynamics of a crystal supercell, forces from Orb-v3 on a Tenstorrent
Blackhole card.

Orb-v3 (`orb-v3-conservative-inf-omat`, OMat24-trained) is a materials universal potential, so the
demo is a bulk crystal, not a single organic molecule. Real conservative forces (F = -dE/dpos) are
computed on-device every step via `tt_atom.orb_forces.energy_and_forces`; the host rebuilds the
periodic neighbour list each step (topology-safe) and integrates Langevin dynamics with ASE.

The `orb-v3-conservative-inf-omat` weight bundle carries the full MoleculeGNS parameters (system
independent) plus `node_feat` for the golden's atoms. For a monatomic crystal every node_feat row
is identical (atomic-number-only featurisation), so we tile one row to the whole supercell -- no
new golden / reference env needed, and the supercell can be any size.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=<tt-atom> ~/.ttatom_run/env/bin/python orb_md_device.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz --nx 3 --ny 3 --nz 3 \
        --steps 400 --temp 900 --out traj.extxyz
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
    """ASE calculator: energy + conservative forces from Orb-v3 on a Tenstorrent card."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, weights_path, device_id=0, r_max=6.0):
        super().__init__()
        from tt_atom.device import open_device
        from tt_atom.orb_weights import OrbWeights
        from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead

        self.r_max = r_max
        # trace-capture needs a reserved DRAM trace region on the device.
        self.device = open_device(device_id, trace_region_size=400_000_000)
        gw = OrbWeights.load(weights_path)
        cfg, w = gw.config, gw.weights
        self.w = w
        self.cfg = cfg
        L = cfg["num_message_passing_steps"]
        self.enc = Encoder(w, self.device, node_in=cfg["node_embed_size"],
                           edge_in=cfg["edge_embed_size"], latent_dim=cfg["latent_dim"],
                           hidden_dim=1024)
        self.layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", self.device,
                                                 latent_dim=cfg["latent_dim"], hidden_dim=1024)
                       for i in range(L)]
        self.ehead = EnergyHead(w, self.device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
        # node_feat is atomic-number-only; monatomic crystal => one row tiled to N atoms.
        self._node_row = gw.host("node_feat")[0:1]
        self.engine = None
        self.nsteps = 0
        self._step_ms = []
        self._n_edges = None

    def _build_engine(self, atoms):
        """Freeze the periodic neighbour list at the initial geometry and capture the device
        forward+backward once. Valid for a solid crystal below melting: atoms vibrate about their
        lattice sites (<~0.3 A here) and never cross the 6 A cutoff, so the topology is genuinely
        constant -- the assumption OrbTracedEngine is built on (docs/orb-port.md)."""
        from tt_atom.geometry import radius_graph
        from tt_atom.orb_trace import OrbTracedEngine

        pos0 = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        N = len(Z)
        edge_index, shift = radius_graph(pos0, self.r_max, cell=cell, pbc=[True, True, True])
        src, tgt = edge_index[0], edge_index[1]
        senders, receivers = tgt, src               # Orb convention is the opposite of UMA's
        self._n_edges = int(senders.shape[0])
        self._Z = Z
        node_feat = self._node_row.repeat(N, 1)
        self.engine = OrbTracedEngine(
            self.enc, self.layers, self.device, senders=senders, receivers=receivers,
            atomic_numbers=Z, node_feat=node_feat, ehead=self.ehead, cell_shift=shift,
            r_max=self.r_max)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from tt_atom.orb_model import (host_conservative_force_denormalize,
                                        host_energy_denormalize)

        if self.engine is None:
            self._build_engine(atoms)
        Z = self._Z
        N = len(Z)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)

        t0 = time.perf_counter()
        raw_e, raw_f = self.engine(pos)             # captures on first call, replays afterwards
        self._step_ms.append((time.perf_counter() - t0) * 1e3)

        forces = host_conservative_force_denormalize(
            raw_f, N, running_var=self.w["energy_head.normalizer.bn.running_var"])
        energy = host_energy_denormalize(
            torch.as_tensor(raw_e, dtype=torch.float64), Z, N,
            running_mean=self.w["energy_head.normalizer.bn.running_mean"],
            running_var=self.w["energy_head.normalizer.bn.running_var"],
            ref_weight=self.w["energy_head.reference.linear.weight"].view(-1))
        self.results["energy"] = float(energy)
        self.results["forces"] = forces.detach().numpy().astype(np.float64)
        self.nsteps += 1

    def close(self):
        import ttnn
        if self.engine is not None:
            self.engine.close()
        ttnn.close_device(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43, help="lattice constant (A)")
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=1.0, help="timestep (fs)")
    ap.add_argument("--temp", type=float, default=900.0, help="temperature (K)")
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction (1/fs)")
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-csv", default=None, help="per-step energy/temperature CSV for stability plots")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (args.nx, args.ny, args.nz)
    N = len(atoms)
    calc = OrbDeviceCalculator(args.weights, device_id=args.device_id)
    atoms.calc = calc

    frames = []
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=np.random.default_rng(args.seed))
        dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.temp,
                       friction=args.friction / units.fs)

        def _snap():
            frames.append(atoms.copy())

        series = []   # (step, t_fs, epot_per_atom, ekin_per_atom, etot_per_atom, T)

        def _record():
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            T = ekin / (1.5 * units.kB * N)
            series.append((dyn.nsteps, dyn.nsteps * args.dt, epot / N, ekin / N,
                           (epot + ekin) / N, T))

        def _log():
            s = series[-1]
            print(f"step {s[0]:4d}  Epot={s[2]:.4f} eV/atom  T={s[5]:.1f} K", flush=True)

        dyn.attach(_snap, interval=args.save_every)
        dyn.attach(_record, interval=1)
        dyn.attach(_log, interval=max(1, args.steps // 10))

        _snap()
        _record()                       # t=0 state
        e0 = atoms.get_potential_energy()
        t0 = time.perf_counter()
        dyn.run(args.steps)
        wall = time.perf_counter() - t0
        e1 = atoms.get_potential_energy()

        write(args.out, frames)
        if args.log_csv:
            with open(args.log_csv, "w") as fh:
                fh.write("step,time_fs,epot_ev_atom,ekin_ev_atom,etot_ev_atom,temp_K\n")
                for s in series:
                    fh.write(f"{s[0]},{s[1]:.1f},{s[2]:.6f},{s[3]:.6f},{s[4]:.6f},{s[5]:.3f}\n")
            print(f"wrote {args.log_csv}  ({len(series)} steps)")
        warm = sorted(calc._step_ms[1:]) or calc._step_ms   # drop the first (trace capture)
        warm_ms = warm[len(warm) // 2]
        warm_sps = 1000.0 / warm_ms
        print("\n" + "=" * 68)
        print(f"system              : {args.element} diamond ({args.nx}x{args.ny}x{args.nz} cubic cells)")
        print(f"atoms / edges       : {N} / {calc._n_edges}")
        print(f"MD                  : {args.steps} steps x {args.dt} fs @ {args.temp} K (NVT Langevin)")
        print(f"energy              : {e0:.4f} -> {e1:.4f} eV  ({e1 / N:.4f} eV/atom)")
        print(f"frames saved        : {len(frames)} -> {args.out}")
        print(f"full loop wall      : {wall:.2f} s ({args.steps} steps, incl. trace capture + host)")
        print(f"device MD step      : {warm_ms:.1f} ms warm median (energy + analytic forces, trace/replay)")
        print(f"                    : => {warm_sps:.1f} MD steps/s on one Blackhole card")
        print(f"atom-steps/s (warm) : {N * warm_sps:,.0f}")
        print("=" * 68)
    finally:
        calc.close()


if __name__ == "__main__":
    main()
