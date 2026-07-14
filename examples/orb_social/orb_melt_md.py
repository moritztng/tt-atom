"""Silicon melt on a Tenstorrent Blackhole card — Orb-v3 conservative forces on device.

Heats a 216-atom diamond-cubic Si supercell through its melting point and into the liquid, then
switches off the thermostat for an NVE tail that measures energy conservation. Every force comes
off the device at every step (Orb-v3 conservative ``F = -dE/dpos``); the host integrates.

Two things make this a *melt* rather than the solid-vibration demo in ``examples/orb_md.py``:

* **Rebuildable neighbour list.** The traced engine in ``orb_md.py`` freezes the graph at the
  initial geometry — exact for a solid, wrong the moment atoms diffuse. Here the engine is rebuilt
  (new ``radius_graph`` + fresh trace capture) whenever any atom has moved more than a ``--skin``
  margin since the last build, so the topology stays correct as the lattice disorders and the
  liquid forms. Rebuild cost is amortised: most steps replay the captured trace at full speed, a
  rebuild only fires when the structure has genuinely shifted.
* **NVT ramp -> NVE tail.** A Langevin thermostat ramps the target temperature from ``--t-start``
  to ``--t-end`` over ``--ramp-steps`` (robust melting — the bath keeps feeding energy as the
  lattice absorbs the latent heat of fusion, so the crystal reliably disorders instead of
  refreezing). The thermostat is then removed for ``--nve-steps`` of velocity-Verlet NVE, where
  total energy is conserved: the linear drift of ``E_tot`` over that tail (meV/atom/ps) is the
  credibility metric for the potential. One trajectory, both the melt and the conservation proof.

Per-step energy / temperature are logged to a CSV; frames are written to an extxyz trajectory for
rendering and structural analysis (RDF, MSD).

    TT_VISIBLE_DEVICES=0 PYTHONPATH=<tt-atom> ~/.ttatom_run/env/bin/python examples/orb_social/orb_melt_md.py \
        --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --ramp-steps 1800 --nve-steps 500 --t-start 300 --t-end 2800 --dt 0.5 \
        --save-every 4 --out si_melt.extxyz --log-csv md_melt.csv
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
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.calculators.calculator import Calculator, all_changes


class OrbMeltCalculator(Calculator):
    """ASE calculator: Orb-v3 energy + conservative forces on device, with a neighbour list that
    rebuilds as the structure disorders. The device modules (encoder / message-passing layers /
    energy head) are constructed once and reused across rebuilds; only the traced graph wrapper is
    discarded and rebuilt when the topology moves past the skin margin."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, weights_path, device_id=0, r_max=6.0, skin=1.5, *, fast=False):
        super().__init__()
        from tt_atom.device import open_device
        from tt_atom.orb_weights import OrbWeights
        from tt_atom.orb_model import Encoder, AttentionInteractionLayer, EnergyHead

        self.r_max = r_max
        self.skin = skin
        self.device = open_device(device_id, trace_region_size=400_000_000)
        gw = OrbWeights.load(weights_path)
        cfg, w = gw.config, gw.weights
        self.w = w
        self.cfg = cfg
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
        self._node_row = gw.host("node_feat")[0:1]      # monatomic Si => one row tiled to N
        self.engine = None
        self._build_pos = None
        self._Z = None
        self.n_edges = None
        self.replay_ms = []
        self.rebuild_ms = []
        self.n_rebuilds = 0

    def _build_engine(self, atoms):
        from tt_atom.geometry import radius_graph
        from tt_atom.orb_trace import OrbTracedEngine

        if self.engine is not None:
            self.engine.close()
            self.engine = None
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        cell = torch.tensor(np.array(atoms.get_cell()), dtype=torch.float64)
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        N = len(Z)
        edge_index, shift = radius_graph(pos, self.r_max, cell=cell, pbc=[True, True, True])
        src, tgt = edge_index[0], edge_index[1]
        senders, receivers = tgt, src            # Orb convention is the opposite of UMA's
        self._Z = Z
        self.n_edges = int(senders.shape[0])
        node_feat = self._node_row.repeat(N, 1)
        self.engine = OrbTracedEngine(
            self.enc, self.layers, self.device, senders=senders, receivers=receivers,
            atomic_numbers=Z, node_feat=node_feat, ehead=self.ehead, cell_shift=shift,
            r_max=self.r_max)
        self._build_pos = pos.clone()
        self.n_rebuilds += 1

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from tt_atom.orb_model import (host_conservative_force_denormalize,
                                        host_energy_denormalize)

        pos_np = atoms.get_positions()
        if self.engine is None:
            t0 = time.perf_counter()
            self._build_engine(atoms)
            self.rebuild_ms.append((time.perf_counter() - t0) * 1e3)
        else:
            disp = np.linalg.norm(pos_np - self._build_pos.numpy(), axis=1).max()
            if disp > self.skin:
                t0 = time.perf_counter()
                self._build_engine(atoms)
                self.rebuild_ms.append((time.perf_counter() - t0) * 1e3)

        Z = self._Z
        N = len(Z)
        pos = torch.tensor(pos_np, dtype=torch.float64)

        t0 = time.perf_counter()
        raw_e, raw_f = self.engine(pos)
        self.replay_ms.append((time.perf_counter() - t0) * 1e3)

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


def _ramp_target(step, ramp_steps, t_start, t_end):
    if step >= ramp_steps:
        return t_end
    return t_start + (t_end - t_start) * (step / ramp_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43, help="lattice constant (A)")
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--ramp-steps", type=int, default=1800)
    ap.add_argument("--nve-steps", type=int, default=500)
    ap.add_argument("--hold-steps", type=int, default=0,
                    help="constant-T NVT liquid hold between the ramp and the NVE tail — the "
                         "many-picosecond window over which the melt actually diffuses/flows")
    ap.add_argument("--t-hold", type=float, default=None,
                    help="liquid-hold temperature in K (default: --t-end)")
    ap.add_argument("--prehold-steps", type=int, default=0,
                    help="constant-T NVT crystalline hold BEFORE the ramp — the lattice sits and "
                         "vibrates at --t-prehold (stays ordered/crystalline) for this many steps "
                         "so the render dwells on the crystal before it melts")
    ap.add_argument("--t-prehold", type=float, default=None,
                    help="crystalline pre-hold temperature in K (default: --t-start)")
    ap.add_argument("--dt", type=float, default=0.5, help="timestep (fs)")
    ap.add_argument("--t-start", type=float, default=300.0)
    ap.add_argument("--t-end", type=float, default=2800.0)
    ap.add_argument("--friction", type=float, default=0.02, help="Langevin friction (1/fs)")
    ap.add_argument("--skin", type=float, default=1.5, help="neighbour-list rebuild margin (A)")
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="use bf8 weights and hidden MLP activations")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (args.nx, args.ny, args.nz)
    N = len(atoms)
    calc = OrbMeltCalculator(args.weights, device_id=args.device_id, skin=args.skin, fast=args.fast)
    atoms.calc = calc

    frames = []
    series = []           # (step, t_fs, epot/N, ekin/N, etot/N, T, regime)
    state = {"step": 0}

    def _snap():
        frames.append(atoms.copy())

    def _record(regime):
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        T = ekin / (1.5 * units.kB * N)
        s = state["step"]
        series.append((s, s * args.dt, epot / N, ekin / N, (epot + ekin) / N, T, regime))

    def _log():
        s = series[-1]
        print(f"step {s[0]:5d}  regime={s[6]:3s}  Epot={s[2]:.4f}  Etot={s[4]:.4f} eV/atom  "
              f"T={s[5]:.0f} K", flush=True)

    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.t_start,
                                     rng=np.random.default_rng(args.seed))
        Stationary(atoms)                       # zero COM momentum
        atoms.calc = calc

        _snap()
        _record("xtl" if args.prehold_steps > 0 else "rmp")
        total = args.prehold_steps + args.ramp_steps + args.hold_steps + args.nve_steps
        log_every = max(1, total // 12)

        # ---- phase 0: constant-T NVT crystalline pre-hold (before any heating) ----
        # The bare ramp starts melting almost immediately, so a render barely shows the ordered
        # crystal before it goes. Holding at a low temperature first lets the diamond lattice sit
        # and vibrate in place — it stays crystalline (atoms rattle about their sites, no
        # diffusion) — so the video dwells on the solid before the melt. The device calculator /
        # neighbour-list state carries straight into the ramp.
        prehold_wall = 0.0
        if args.prehold_steps > 0:
            t_pre = args.t_prehold if args.t_prehold is not None else args.t_start
            pdyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=t_pre,
                            friction=args.friction / units.fs)
            pdyn.attach(_snap, interval=args.save_every)

            def _pre_record():
                _record("xtl")
                state["step"] += 1
            pdyn.attach(_pre_record, interval=1)
            pdyn.attach(_log, interval=log_every)

            tp = time.perf_counter()
            pdyn.run(args.prehold_steps)
            prehold_wall = time.perf_counter() - tp
            print(f"[prehold done] {args.prehold_steps} steps at {t_pre:.0f} K in "
                  f"{prehold_wall:.1f}s (rebuilds={calc.n_rebuilds})", flush=True)

        # ---- phase 1: NVT Langevin temperature ramp ----
        # state["step"] is advanced inside _ramp_record (an interval=1 observer), so the
        # temperature target for the *next* integration step tracks the ramp schedule. The ramp
        # progress is measured from the ramp's own start (state["step"] carries any pre-hold steps).
        ramp_start = state["step"]
        dyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=args.t_start,
                       friction=args.friction / units.fs)
        dyn.attach(_snap, interval=args.save_every)

        def _ramp_record():
            _record("rmp")
            state["step"] += 1
            dyn.set_temperature(temperature_K=_ramp_target(state["step"] - ramp_start,
                                                           args.ramp_steps,
                                                           args.t_start, args.t_end))
        dyn.attach(_ramp_record, interval=1)
        dyn.attach(_log, interval=log_every)

        t0 = time.perf_counter()
        dyn.run(args.ramp_steps)
        ramp_wall = time.perf_counter() - t0
        print(f"[ramp done] {args.ramp_steps} steps in {ramp_wall:.1f}s "
              f"(rebuilds={calc.n_rebuilds})", flush=True)

        # ---- phase 1b: constant-T NVT liquid hold — real diffusion / visible flow ----
        # The ~1 ps ramp only just melts the lattice; over that window each atom barely moves, so a
        # render of it reads as sluggish. Holding at a fixed high temperature for many picoseconds
        # lets the liquid actually diffuse — atoms slide past neighbours and the cloud churns — so
        # the render shows real flow (not sped-up interpolation of a tiny run). The device
        # calculator persists, so neighbour-list / trace state stays continuous across the switch.
        hold_wall = 0.0
        if args.hold_steps > 0:
            t_hold = args.t_hold if args.t_hold is not None else args.t_end
            dyn.observers = []
            hdyn = Langevin(atoms, timestep=args.dt * units.fs, temperature_K=t_hold,
                            friction=args.friction / units.fs)
            hdyn.attach(_snap, interval=args.save_every)

            def _hold_record():
                _record("liq")
                state["step"] += 1
            hdyn.attach(_hold_record, interval=1)
            hdyn.attach(_log, interval=log_every)

            th = time.perf_counter()
            hdyn.run(args.hold_steps)
            hold_wall = time.perf_counter() - th
            print(f"[hold done] {args.hold_steps} steps at {t_hold:.0f} K in {hold_wall:.1f}s "
                  f"(rebuilds={calc.n_rebuilds})", flush=True)

        # ---- phase 2: NVE (no thermostat) — energy conservation tail ----
        # Drop the ramp observers (snap/record/log) by clearing the observer list, then build a
        # fresh velocity-Verlet integrator with its own callbacks. The device calculator persists
        # across the switch, so the neighbour-list / trace state is continuous.
        dyn.observers = []
        vdyn = VelocityVerlet(atoms, timestep=args.dt * units.fs)
        vdyn.attach(_snap, interval=args.save_every)

        def _nve_record():
            _record("nve")
            state["step"] += 1
        vdyn.attach(_nve_record, interval=1)
        vdyn.attach(_log, interval=log_every)

        t1 = time.perf_counter()
        vdyn.run(args.nve_steps)
        nve_wall = time.perf_counter() - t1
        print(f"[nve done] {args.nve_steps} steps in {nve_wall:.1f}s", flush=True)

        write(args.out, frames)
        if args.log_csv:
            with open(args.log_csv, "w") as fh:
                fh.write("step,time_fs,epot_ev_atom,ekin_ev_atom,etot_ev_atom,temp_K,regime\n")
                for s in series:
                    fh.write(f"{s[0]},{s[1]:.2f},{s[2]:.6f},{s[3]:.6f},{s[4]:.6f},{s[5]:.3f},{s[6]}\n")
            print(f"wrote {args.log_csv}  ({len(series)} steps)")

        # ---- metrics: NVE energy-conservation drift + throughput ----
        arr = np.array([(s[1], s[4]) for s in series if s[6] == "nve"], dtype=np.float64)
        drift = float(np.polyfit(arr[:, 0], arr[:, 1], 1)[0]) * 1e3 * 1e3 if len(arr) > 4 else float("nan")
        replay = sorted(calc.replay_ms)
        warm_ms = replay[len(replay) // 2] if replay else float("nan")
        sps = 1000.0 / warm_ms if warm_ms and warm_ms > 0 else float("nan")
        nsday = sps * args.dt * 0.0864

        print("\n" + "=" * 70)
        print(f"system          : {args.element} diamond ({args.nx}x{args.ny}x{args.nz} = {N} atoms)")
        print(f"edges (last)    : {calc.n_edges}   (rebuilt {calc.n_rebuilds} times, skin {args.skin} A)")
        t_hold_disp = args.t_hold if args.t_hold is not None else args.t_end
        t_pre_disp = args.t_prehold if args.t_prehold is not None else args.t_start
        print(f"MD              : prehold {args.prehold_steps} @ {t_pre_disp:.0f} K NVT (crystalline) "
              f"+ ramp {args.ramp_steps} ({args.t_start:.0f}->{args.t_end:.0f} K NVT) "
              f"+ hold {args.hold_steps} @ {t_hold_disp:.0f} K NVT + NVE {args.nve_steps}, x {args.dt} fs")
        print(f"NVE E drift     : {drift:+.3f} meV/atom/ps   (conservation credibility metric)")
        print(f"device MD step  : {warm_ms:.2f} ms warm median (replay, energy + analytic forces)")
        print(f"throughput      : {sps:.1f} MD steps/s  |  {N*sps:,.0f} atom-steps/s  |  {nsday:.3f} ns/day")
        print(f"frames saved    : {len(frames)} -> {args.out}")
        print("=" * 70)
    finally:
        calc.close()


if __name__ == "__main__":
    main()
