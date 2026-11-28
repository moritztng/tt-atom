"""Does TensorNet-MatPES keep its forces when every module output is held in bfloat16? CPU emulation, no device.

Every module's output is rounded to bfloat16 on the way forward and its gradient on the way back, which is what
bfloat16 activations on a card would do at module boundaries (ops inside a module stay float32, so this is a
lower bound on the device error). Graded against the unmodified float32 model.

    python benchmarks/tensornet_bf16_emulation.py --n 3
"""
import argparse
import json

import numpy as np
import torch

from census_tensornet import build


def bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return bf16(x)

    @staticmethod
    def backward(ctx, g):
        return bf16(g)


def hook(mod, inp, out):
    if torch.is_tensor(out) and out.is_floating_point():
        return Round.apply(out)
    if isinstance(out, tuple) and all(torch.is_tensor(o) and o.is_floating_point() for o in out):
        return tuple(Round.apply(o) for o in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="si")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="materialyze/TensorNet-PES-MatPES-PBE-2025.2")
    ap.add_argument("--skip", default="", help="comma list of module-name substrings left in float32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(args.model)
    pot.calc_stresses = False
    atoms = build(args.cell, args.n)
    atoms.calc = PESCalculator(pot)
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces().copy()
    skip = [s for s in args.skip.split(",") if s]
    hooked = [(n, m) for n, m in pot.model.named_modules() if n and not any(s in n for s in skip)]
    hs = [m.register_forward_hook(hook) for _, m in hooked]
    atoms.calc = PESCalculator(pot)
    e1, f1 = atoms.get_potential_energy(), atoms.get_forces().copy()
    for h in hs:
        h.remove()
    res = {"atoms": len(atoms), "modules_rounded": len(hooked), "skip": skip,
           "dE_meV_per_atom": 1e3 * abs(e1 - e0) / len(atoms), "force_mae_meV_A": 1e3 * float(np.abs(f1 - f0).mean()),
           "force_max_meV_A": 1e3 * float(np.abs(f1 - f0).max()), "mean_abs_force_eV_A": float(np.abs(f0).mean())}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
