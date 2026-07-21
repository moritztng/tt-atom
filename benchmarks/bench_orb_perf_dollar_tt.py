"""Tenstorrent leg of the fair Orb-v3 perf-per-dollar comparison (one Blackhole p150).

Production path: ``OrbTracedEngine`` trace/replay, neighbour list FROZEN at the first
geometry (the honest production mode for a solid crystal -- atoms vibrate about their
lattice sites and never cross the cutoff, so the topology is genuinely constant and
trace replay is bit-exact). Computes energy + conservative analytic forces (F = -dE/dpos),
no stress -- the same quantity the GPU leg computes. Load + first-call trace capture are
excluded; positions are jittered each step so the per-step refresh path is exercised like
a real MD loop, not a degenerate identical-input replay.

Sweeps supercell sizes so the compute-bound crossover is visible, not a single point, and
dumps a JSON record with raw per-step timings + edges + git SHA for committed evidence.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python \
        benchmarks/bench_orb_perf_dollar_tt.py --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
        --out benchmarks/orb_perf_dollar_tt.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk


SIZES = [
    ("3x3x3", 3, 3, 3),   #   216 atoms
    ("4x4x4", 4, 4, 4),   #   512 atoms
    ("5x5x5", 5, 5, 5),   #  1000 atoms
    ("6x6x7", 6, 6, 7),   #  2016 atoms (~2000)
]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="Orb-v3 conservative bundle (gen_golden_orb.py --system supercell)")
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fast", action="store_true",
                    help="use bf8 weights and hidden MLP activations (release-gated)")
    ap.add_argument("--out", default="benchmarks/orb_perf_dollar_tt.json")
    ap.add_argument("--sizes", default=",".join(s[0] for s in SIZES),
                    help="comma list of size tags from: " + ",".join(s[0] for s in SIZES))
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from examples.orb_md import OrbDeviceCalculator  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wanted = [t for t in args.sizes.split(",") if t.strip()]
    plan = [s for s in SIZES if s[0] in wanted]

    def _dump(records):
        out = {
            "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
            "model": "orb-v3-conservative-inf-omat",
            "system": "periodic Si diamond supercell, monatomic (node feature tiled, system-independent weights)",
            "quantity": "one MD step = energy + conservative analytic forces (F = -dE/dpos), no stress",
            "neighbour_policy": "frozen at first geometry (solid crystal; topology constant)",
            "execution_model": "trace/replay (OrbTracedEngine) -- zero per-step host dispatch, bit-exact vs eager",
            "load_and_first_compile_excluded": True,
            "positions_jittered_each_step": True,
            "orb_fused_silu_bw": os.environ.get("TT_ATOM_ORB_FUSED_SILU_BW", "auto"),
            "orb_minimal_matmul": os.environ.get("TT_ATOM_ORB_MINIMAL_MATMUL", "auto"),
            "git_sha": _git_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "env_python": sys.executable,
            "records": records,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    records = []
    for tag, nx, ny, nz in plan:
        print(f"\n=== {tag}  ({nx}x{ny}x{nz} cubic cells) ===", flush=True)
        try:
            atoms0 = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
            N = len(atoms0)
            calc = OrbDeviceCalculator(args.weights, device_id=int(
                os.environ.get("TT_VISIBLE_DEVICES", "0")), fast=args.fast)
            atoms = atoms0.copy()
            atoms.calc = calc

            # Warmup: first call captures the trace (excluded), rest prime caches.
            rng = np.random.default_rng(args.seed)
            atoms.set_positions(atoms.get_positions())
            _ = atoms.get_potential_energy(); _ = atoms.get_forces()
            for _ in range(max(0, args.warmup - 1)):
                p = atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape)
                atoms.set_positions(p)
                _ = atoms.get_potential_energy(); _ = atoms.get_forces()

            calc.step_ms.clear()
            e0 = None
            for i in range(args.steps):
                p = atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape)
                atoms.set_positions(p)
                e = atoms.get_potential_energy()
                _ = atoms.get_forces()
                if i == 0:
                    e0 = float(e)

            raw = list(calc.step_ms)
            med = float(np.median(raw))
            rec = {
                "tag": tag, "nx": nx, "ny": ny, "nz": nz, "N": N,
                "edges": int(calc.n_edges),
                "warmup": args.warmup, "timed_steps": args.steps,
                "step_ms_raw": [round(x, 4) for x in raw],
                "step_ms_median": med,
                "step_ms_min": float(min(raw)),
                "step_ms_max": float(max(raw)),
                "steps_per_s": 1000.0 / med,
                "energy_sample_eV": e0,
                "energy_sample_eV_per_atom": (e0 / N) if e0 is not None else None,
                "precision": ("bf8 weights/hidden MLP activations, bf16 residual stream, "
                              "fp32-accumulate matmul" if args.fast else
                              "bf16 weights/activations, fp32-accumulate matmul"),
                "path": "OrbTracedEngine trace/replay, neighbour list frozen, energy+conservative forces (no stress)",
            }
            records.append(rec)
            print(f"N={N} edges={calc.n_edges} median={med:.3f} ms  => {1000.0/med:.2f} steps/s"
                  f"  (e0={e0:.3f} eV, {e0/N:.4f} eV/atom)", flush=True)
            calc.close()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR on {tag}: {type(exc).__name__}: {exc}", flush=True)
            records.append({"tag": tag, "nx": nx, "ny": ny, "nz": nz,
                            "error": f"{type(exc).__name__}: {exc}"})
            try:
                calc.close()
            except Exception:
                pass
        _dump(records)

    print(f"\nwrote {args.out}: {len(records)} sizes", flush=True)


if __name__ == "__main__":
    main()
