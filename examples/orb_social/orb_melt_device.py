"""Melt a silicon crystal on a Tenstorrent Blackhole card: an ordered diamond lattice heated
through melting into a flowing liquid, with real Orb-v3 conservative forces every step.

This is the melt companion to ``orb_md_device.py`` (the solid-crystal vibration demo). The
physics differs in one essential way: a liquid *diffuses*, so the periodic neighbour list can no
longer be frozen at ``t=0`` (the assumption behind ``orb_md_device.py``'s trace-captured engine).
Here the host rebuilds the neighbour list from the current geometry every step and runs the plain
(untraced) device forward + reverse VJP -- and because Orb's force path is dominated by the host
autograd finish over the differentiable edge geometry, not device dispatch, the untraced step is
*the same speed* as a trace replay (measured ~58 ms vs ~55 ms at 216 atoms on one Blackhole), so
there is nothing to gain from a trace here and topology stays exactly correct as atoms flow.

Nucleation: homogeneous melting under 3-D PBC superheats well past the true melting point, so we
seed a small spherical void (a vacancy cluster) as a realistic nucleation site -- melting then
initiates at the defect rather than requiring the whole superheated lattice to collapse at once.
We still present the run honestly as "heated until it melts" over a stated NVT temperature ramp,
without asserting an exact T_m (a small fast-ramped PBC cell is not a calibrated melting-point
measurement).

    TT_VISIBLE_DEVICES=0 PYTHONPATH=<tt-atom> ~/.ttatom_run/env/bin/python orb_melt_device.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz --nx 3 --ny 3 --nz 3 \
        --void-r 3.3 --steps 3000 --t0 300 --t1 3000 --hold 400 --out melt.extxyz --log-csv melt.csv
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from ase import units
from ase.build import bulk
from ase.io import write
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


from ase.calculators.calculator import Calculator, all_changes


class DynamicOrbCalculator(Calculator):
    """ASE calculator: Orb-v3 energy + conservative forces on a Tenstorrent card, rebuilding the
    periodic neighbour list from the current geometry every step (topology-safe for a diffusing
    liquid). Reuses the on-device modules; no trace, no frozen graph. ASE's ``Calculator`` base
    caches results and only re-invokes ``calculate`` when the atoms actually change, so energy and
    forces for one step trigger a single device pass."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, weights_path, device_id=0, r_max=6.0, skin=1.0):
        super().__init__()
        from tt_atom.device import open_device
        from tt_atom.orb_weights import OrbWeights
        from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead

        self.r_max = r_max
        # Verlet skin: the neighbour list is built out to r_max + skin and held fixed until an
        # atom drifts more than skin/2 from its position at the last build (so no pair can enter
        # r_max unlisted in between). Holding the edge set fixed between rebuilds keeps the device
        # tensor shapes constant -> ttnn reuses compiled kernels (warm ~60 ms/step); rebuilding
        # every step instead varies the edge count and recompiles each time (~1.4 s/step). Edges
        # in (r_max, r_max+skin] carry zero weight (the cutoff envelope is flat-zero there) so they
        # add no energy or force -- they are pure look-ahead.
        self.skin = skin
        self.r_build = r_max + skin
        self._senders = self._receivers = self._shift = None
        self._ref_pos = None
        self.n_rebuilds = 0
        self.device = open_device(device_id)
        gw = OrbWeights.load(weights_path)
        cfg, w = gw.config, gw.weights
        self.w, self.cfg = w, cfg
        L = cfg["num_message_passing_steps"]
        self.enc = Encoder(w, self.device, node_in=cfg["node_embed_size"],
                           edge_in=cfg["edge_embed_size"], latent_dim=cfg["latent_dim"],
                           hidden_dim=1024)
        self.layers = [AttentionInteractionLayer(w, f"gnn_stacks.{i}", self.device,
                                                 latent_dim=cfg["latent_dim"], hidden_dim=1024)
                       for i in range(L)]
        self.ehead = EnergyHead(w, self.device, latent_dim=cfg["latent_dim"], hidden_dim=1024)
        self._node_row = gw.host("node_feat")[0:1]   # atomic-number-only -> one row tiled to N
        self._step_ms = []
        self._n_edges = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from tt_atom.geometry import radius_graph
        from tt_atom.orb_forces import energy_and_forces
        from tt_atom.orb_model import (host_conservative_force_denormalize,
                                        host_energy_denormalize)

        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        N = len(Z)
        pbc = list(atoms.get_pbc())

        drift = 0.0 if self._ref_pos is None else float(
            np.linalg.norm(pos.numpy() - self._ref_pos, axis=1).max())
        if self._senders is None or drift > 0.5 * self.skin:
            edge_index, shift = radius_graph(pos, self.r_build, cell=cell, pbc=pbc)
            src, tgt = edge_index[0], edge_index[1]
            self._senders, self._receivers = tgt, src   # Orb convention (opposite of UMA's)
            self._shift = shift
            self._ref_pos = pos.numpy().copy()
            self.n_rebuilds += 1
        senders, receivers, shift = self._senders, self._receivers, self._shift
        node_feat = self._node_row.repeat(N, 1)

        t0 = time.perf_counter()
        raw_e, raw_f = energy_and_forces(
            self.enc, self.layers, self.ehead, self.device, pos=pos, senders=senders,
            receivers=receivers, atomic_numbers=Z, node_feat=node_feat, cell_shift=shift,
            r_max=self.r_max)
        self._step_ms.append((time.perf_counter() - t0) * 1e3)
        self._n_edges = int(senders.shape[0])

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
        ttnn.close_device(self.device)


def seed_void(atoms, radius):
    """Remove atoms within ``radius`` of the cell centre -> a spherical vacancy cluster that
    nucleates melting. Returns the number of atoms removed."""
    if radius <= 0:
        return 0
    center = atoms.get_cell().sum(axis=0) / 2.0
    d = np.linalg.norm(atoms.get_positions() - center, axis=1)
    keep = d > radius
    n_removed = int((~keep).sum())
    del atoms[[i for i in range(len(atoms)) if not keep[i]]]
    return n_removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--void-r", type=float, default=3.3, help="vacancy-cluster radius (A); 0=none")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dt", type=float, default=1.0, help="timestep (fs)")
    ap.add_argument("--t0", type=float, default=300.0, help="ramp start temperature (K)")
    ap.add_argument("--t1", type=float, default=3000.0, help="ramp end temperature (K)")
    ap.add_argument("--hold", type=int, default=0, help="steps held at t1 after the ramp")
    ap.add_argument("--tdamp", type=float, default=25.0, help="Nose-Hoover damping time (fs)")
    ap.add_argument("--thermostat", choices=["nose-hoover", "langevin"], default="langevin")
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction (1/fs)")
    ap.add_argument("--skin", type=float, default=1.0, help="Verlet neighbour-list skin (A)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (args.nx, args.ny, args.nz)
    n_full = len(atoms)
    n_removed = seed_void(atoms, args.void_r)
    N = len(atoms)
    calc = DynamicOrbCalculator(args.weights, device_id=args.device_id, skin=args.skin)
    atoms.calc = calc

    ramp = max(1, args.steps - args.hold)

    def temp_at(step):
        frac = min(1.0, step / ramp)
        return args.t0 + (args.t1 - args.t0) * frac

    frames, series, pos0 = [], [], None
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.t0,
                                     rng=np.random.default_rng(args.seed))
        Stationary(atoms)
        pos0 = atoms.get_positions().copy()

        if args.thermostat == "nose-hoover":
            dyn = NoseHooverChainNVT(atoms, timestep=args.dt * units.fs, temperature_K=args.t0,
                                     tdamp=args.tdamp * units.fs)

            def set_T(step):
                # NoseHooverChainNVT exposes no public temperature setter; retune the thermostat
                # target kT and its chain masses (Q ~ kT * tdamp^2) so the damping time is kept
                # fixed as the setpoint ramps.
                th = dyn._thermostat
                th._kT = units.kB * temp_at(step)
                th._Q[0] = 3 * th._num_atoms_global * th._kT * th._tdamp ** 2
                th._Q[1:] = th._kT * th._tdamp ** 2
        else:
            dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.t0,
                           friction=args.friction / units.fs)

            def set_T(step):
                dyn.set_temperature(temperature_K=temp_at(step))

        def _ramp():
            set_T(dyn.nsteps)

        def _snap():
            frames.append(atoms.copy())

        def _record():
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            T = ekin / (1.5 * units.kB * N)
            msd = float(((atoms.get_positions() - pos0) ** 2).sum(axis=1).mean())
            series.append((dyn.nsteps, dyn.nsteps * args.dt, epot / N, ekin / N, T,
                           temp_at(dyn.nsteps), msd))

        def _log():
            s = series[-1]
            print(f"step {s[0]:4d}  T_set={s[5]:6.0f}K  T={s[4]:6.0f}K  "
                  f"Epot={s[2]:.3f} eV/atom  MSD={s[6]:.2f} A^2", flush=True)

        dyn.attach(_ramp, interval=1)
        dyn.attach(_snap, interval=args.save_every)
        dyn.attach(_record, interval=1)
        dyn.attach(_log, interval=max(1, args.steps // 20))

        _snap()
        _record()
        t0 = time.perf_counter()
        dyn.run(args.steps)
        wall = time.perf_counter() - t0

        write(args.out, frames)
        if args.log_csv:
            with open(args.log_csv, "w") as fh:
                fh.write("step,time_fs,epot_ev_atom,ekin_ev_atom,temp_K,temp_set_K,msd_A2\n")
                for s in series:
                    fh.write(f"{s[0]},{s[1]:.1f},{s[2]:.6f},{s[3]:.6f},{s[4]:.3f},"
                             f"{s[5]:.1f},{s[6]:.4f}\n")
            print(f"wrote {args.log_csv}  ({len(series)} steps)")
        warm = sorted(calc._step_ms[1:]) or calc._step_ms
        warm_ms = warm[len(warm) // 2]
        msd_end = np.mean([s[6] for s in series[-min(50, len(series)):]])
        print("\n" + "=" * 70)
        print(f"system              : {args.element} diamond {args.nx}x{args.ny}x{args.nz} "
              f"cubic, {n_removed}-atom void -> {N} atoms (of {n_full})")
        print(f"ensemble            : NVT ({args.thermostat}), {args.dt} fs, "
              f"ramp {args.t0:.0f} -> {args.t1:.0f} K over {ramp} steps"
              + (f" + {args.hold} hold" if args.hold else ""))
        print(f"MD                  : {args.steps} steps ({args.steps * args.dt / 1000:.1f} ps)")
        print(f"neighbour list      : Verlet, skin {args.skin} A, {calc.n_rebuilds} rebuilds, "
              f"{calc._n_edges} edges (<=r_max+skin)")
        print(f"frames saved        : {len(frames)} -> {args.out}")
        print(f"final-segment MSD   : {msd_end:.2f} A^2  (>~1.5 => diffusive/liquid)")
        print(f"full loop wall      : {wall:.1f} s")
        print(f"device MD step      : {warm_ms:.1f} ms median (untraced fwd+VJP, warm/fixed-list)")
        print(f"                    : => {1000.0 / warm_ms:.1f} MD steps/s on one Blackhole card")
        print("=" * 70)
    finally:
        calc.close()


if __name__ == "__main__":
    main()
