"""orb-models reference benchmark on an NVIDIA GPU: warm steady-state Orb-v3 MD step.

Mirror of the Tenstorrent measurement in ``examples/orb_md.py``: same checkpoint
(``orb-v3-conservative-inf-omat``), same system (Si diamond supercell), same quantity
(one energy + conservative-force evaluation per step). Load + first-call warmup are
excluded; positions are jittered each step so the torch graph is exercised like a real
MD loop, not a degenerate identical-input replay. Reports median ms/step and steps/s.

Precision is the ``orb_models`` default (fp32); the Tenstorrent side runs bf16. See
``docs/orb-port.md`` "Performance per dollar" for the apples-to-apples methodology.

Run on a CUDA box with ``pip install orb-models ase`` (and a CUDA-matched torch), then::

    python benchmarks/orb_gpu_bench.py --ckpt orb-v3-conservative-inf-omat \
        --nx 3 --ny 3 --nz 3 --warmup 15 --steps 80
"""
from __future__ import annotations
import argparse, time, statistics
import numpy as np
import torch
from ase.build import bulk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="orb-v3-conservative-inf-omat",
                    choices=["orb-v3-conservative-inf-omat", "orb-v3-direct-20-omat"])
    ap.add_argument("--nx", type=int, default=3)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--bf16", action="store_true", help="run model in bf16 (else fp32)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    if args.ckpt == "orb-v3-conservative-inf-omat":
        model = pretrained.orb_v3_conservative_inf_omat(device=device)
    else:
        model = pretrained.orb_v3_direct_20_omat(device=device)
    if args.bf16:
        model = model.to(torch.bfloat16)

    atoms0 = bulk("Si", "diamond", a=5.43, cubic=True) * (args.nx, args.ny, args.nz)
    N = len(atoms0)
    calc = ORBCalculator(model, device=device)
    atoms0.calc = calc

    # warmup (excluded): triggers any lazy graph / neighbour-list build
    atoms = atoms0.copy()
    atoms.calc = calc
    _ = atoms.get_potential_energy(); _ = atoms.get_forces()
    for _ in range(max(0, args.warmup - 1)):
        p = atoms.get_positions() + np.random.normal(0.0, 0.01, atoms.positions.shape)
        atoms.set_positions(p)
        _ = atoms.get_potential_energy(); _ = atoms.get_forces()
    torch.cuda.synchronize()

    ms, es = [], []
    atoms = atoms0.copy()
    atoms.calc = calc
    for i in range(args.steps):
        p = atoms.get_positions() + np.random.normal(0.0, 0.01, atoms.positions.shape)
        atoms.set_positions(p)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        ms.append((t1 - t0) * 1e3)
        es.append(float(e))

    med = statistics.median(ms)
    print("=" * 68)
    print(f"checkpoint        : {args.ckpt}")
    print(f"system            : Si diamond ({args.nx}x{args.ny}x{args.nz} cubic cells), N={N}")
    print(f"precision         : {'bf16' if args.bf16 else 'fp32'} (orb_models default)")
    print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"steps             : {args.steps} (warmup {args.warmup} excluded)")
    print(f"MD step (E+F)     : {med:.3f} ms median  => {1000.0/med:.2f} MD steps/s")
    print(f"  min/max ms      : {min(ms):.3f} / {max(ms):.3f}")
    print(f"  E sample        : {es[0]:.4f} eV  ({es[0]/N:.4f} eV/atom)")
    print(f"  |F|max sample   : {float(np.abs(f).max()):.4f} eV/A")
    print("=" * 68)


if __name__ == "__main__":
    main()
