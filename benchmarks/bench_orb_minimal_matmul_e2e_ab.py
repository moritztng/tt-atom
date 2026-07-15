"""Controlled whole-MD A/B for Orb edge-MLP minimal-matmul.

Both paths run in one process with identical displaced frames.  Besides repeated
trace/replay timings, the first measured frame is compared for energy and force parity.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import numpy as np
from ase.build import bulk


SIZES = [
    ("3x3x3", 3, 3, 3),
    ("4x4x4", 4, 4, 4),
    ("5x5x5", 5, 5, 5),
    ("6x6x7", 6, 6, 7),
]


def _measure(weights, device_id, nx, ny, nz, *, warmup, steps, seed):
    from examples.orb_md import OrbDeviceCalculator

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz)
    calc = OrbDeviceCalculator(weights, device_id=device_id)
    atoms.calc = calc
    rng = np.random.default_rng(seed)
    atoms.set_positions(atoms.get_positions())
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
    for _ in range(warmup - 1):
        atoms.set_positions(
            atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape))
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()

    calc.step_ms.clear()
    sample_energy = None
    sample_forces = None
    for step in range(steps):
        atoms.set_positions(
            atoms.get_positions() + rng.normal(0.0, 0.01, atoms.positions.shape))
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        if step == 0:
            sample_energy = float(energy)
            sample_forces = forces.copy()
    result = {
        "median_ms": float(np.median(calc.step_ms)),
        "raw_ms": [round(value, 4) for value in calc.step_ms],
        "energy": sample_energy,
        "forces": sample_forces,
        "edges": int(calc.n_edges),
    }
    calc.close()
    return result


def _pcc(a, b):
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="benchmarks/orb_minimal_matmul_e2e_ab.json")
    args = parser.parse_args()

    records = []
    for tag, nx, ny, nz in SIZES:
        old_runs = []
        new_runs = []
        parity = None
        edges = None
        for rep in range(args.reps):
            os.environ["TT_ATOM_ORB_MINIMAL_MATMUL"] = "0"
            old = _measure(
                args.weights, args.device_id, nx, ny, nz,
                warmup=args.warmup, steps=args.steps, seed=args.seed + rep)
            os.environ["TT_ATOM_ORB_MINIMAL_MATMUL"] = "1"
            new = _measure(
                args.weights, args.device_id, nx, ny, nz,
                warmup=args.warmup, steps=args.steps, seed=args.seed + rep)
            old_runs.append(old["median_ms"])
            new_runs.append(new["median_ms"])
            edges = old["edges"]
            if parity is None:
                diff = np.abs(old["forces"] - new["forces"])
                parity = {
                    "energy_abs_delta_eV": abs(old["energy"] - new["energy"]),
                    "force_pcc": _pcc(old["forces"], new["forces"]),
                    "force_mae_eV_per_A": float(diff.mean()),
                    "force_max_abs_eV_per_A": float(diff.max()),
                    "forces_bit_exact": bool(np.array_equal(old["forces"], new["forces"])),
                }

        old_ms = float(statistics.median(old_runs))
        new_ms = float(statistics.median(new_runs))
        record = {
            "tag": tag,
            "atoms": len(bulk("Si", "diamond", a=5.43, cubic=True) * (nx, ny, nz)),
            "edges": edges,
            "baseline_ms": old_ms,
            "minimal_matmul_ms": new_ms,
            "speedup": old_ms / new_ms,
            "baseline_runs_ms": old_runs,
            "minimal_matmul_runs_ms": new_runs,
            "parity": parity,
        }
        records.append(record)
        print(
            f"{tag}: {old_ms:.3f} -> {new_ms:.3f} ms ({record['speedup']:.3f}x), "
            f"PCC={parity['force_pcc']:.9f}, max|dF|={parity['force_max_abs_eV_per_A']:.3e}, "
            f"dE={parity['energy_abs_delta_eV']:.3e}",
            flush=True,
        )

    with open(args.out, "w") as handle:
        json.dump({
            "platform": "Tenstorrent Blackhole p150 (one card, device 0)",
            "warmup": args.warmup,
            "steps": args.steps,
            "reps": args.reps,
            "records": records,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
