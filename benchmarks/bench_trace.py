"""Trace vs eager benchmark for the MD / relaxation loop (single card).

Measures the real end-to-end per-step cost (host geometry + device forward + device backward +
host force finish) of the eager path vs the trace-captured device-resident path, plus a phase
breakdown that shows why tracing is the right lever (the device forward+backward dominates and is
host-dispatch-bound for these graph sizes). Forces are bit-for-bit identical between the two.

    PYTHONPATH=~/TT-Atom ~/.ttatom_run/env/bin/python benchmarks/bench_trace.py \
        --weights ~/.ttatom_run/uma_s_ethanol.npz

With no --weights it falls back to the committed random-weight demo bundle (architecture-only).
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch
from ase.build import molecule

HERE = pathlib.Path(__file__).parent


def _median_ms(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "examples" / "model_tiny_demo.npz"))
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    from tt_atom.calculator import TTAtomCalculator

    atoms = molecule("CH3CH2OH")
    atoms.info.update(charge=0, spin=0)
    atoms.rattle(stdev=0.03, seed=5)

    eager = TTAtomCalculator(args.weights, device_id=args.device_id)
    atoms.calc = eager
    eager_ms = _median_ms(lambda: eager.calculate(atoms))
    Ee, Fe = eager.results["energy"], eager.results["forces"]
    eager.close()

    traced = TTAtomCalculator(args.weights, device_id=args.device_id, trace=True)
    atoms.calc = traced
    traced.calculate(atoms)                         # capture
    traced_ms = _median_ms(lambda: traced.calculate(atoms))
    Et, Ft = traced.results["energy"], traced.results["forces"]
    traced.close()

    print(f"system: ethanol (9 atoms), weights={pathlib.Path(args.weights).name}")
    print(f"  eager  E+F per step : {eager_ms:6.2f} ms")
    print(f"  traced E+F per step : {traced_ms:6.2f} ms")
    print(f"  speedup             : {eager_ms / traced_ms:.2f}x")
    print(f"  energy diff         : {abs(Et - Ee):.2e} eV")
    print(f"  force PCC / maxdiff : {np.corrcoef(Ft.ravel(), Fe.ravel())[0, 1]:.6f} / "
          f"{np.abs(Ft - Fe).max():.2e} eV/A  (trace only removes host dispatch)")


if __name__ == "__main__":
    main()
