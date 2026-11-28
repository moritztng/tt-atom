"""Op census of TensorNet-MatPES (matgl) for one energy+forces step on CPU.

Counts what a device port would have to run: GEMM FLOPs per Linear shape and the bytes
every other op writes, split by whether the tensor is per edge or per node. No device.

    python benchmarks/census_tensornet.py --cell si --n 8      # 4096 atoms
"""
import argparse
import collections
import json
import time

import numpy as np
import torch


VIEWS = {"as_strided", "view", "_unsafe_view", "resolve_conj", "resolve_neg", "result_type", "expand", "reshape",
         "permute", "transpose", "select", "slice", "unsqueeze", "squeeze", "alias", "detach", "t", "to", "resize_",
         "unbind", "lift_fresh", "detach_", "item", "_local_scalar_dense", "empty", "empty_like", "numpy_T"}


def build(cell, n):
    from ase.build import bulk

    a = {"si": bulk("Si", "diamond", a=5.43, cubic=True), "cu": bulk("Cu", "fcc", a=3.61, cubic=True)}[cell]
    a = a * (n, n, n)
    a.positions += np.random.default_rng(0).normal(0, 0.01, a.positions.shape)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="si", choices=["si", "cu"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default="materialyze/TensorNet-PES-MatPES-PBE-2025.2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)

    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(args.model)
    pot.calc_stresses = False
    atoms = build(args.cell, args.n)
    atoms.calc = PESCalculator(pot)

    gemm = collections.Counter()  # (rows, in, out) -> calls

    def hook(mod, inp, out):
        x = inp[0]
        gemm[(x.numel() // x.shape[-1], mod.in_features, mod.out_features)] += 1

    hooks = [m.register_forward_hook(hook) for m in pot.modules() if isinstance(m, torch.nn.Linear)]
    t0 = time.perf_counter()
    with torch.profiler.profile(record_shapes=True) as prof:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
    wall = time.perf_counter() - t0
    for h in hooks:
        h.remove()

    N = len(atoms)
    # edges: the largest leading dim any Linear saw is the edge count
    E = max(r for (r, i, o) in gemm if i <= 4) if any(i <= 4 for (_, i, _) in gemm) else 0
    fwd = sum(2 * r * i * o * c for (r, i, o), c in gemm.items())

    # bytes written by every non-GEMM op, by leading dim (edge / node / other), forward and backward
    traffic = collections.Counter()
    ops = collections.Counter()
    for ev in prof.events():
        if not ev.name.startswith("aten::") and "Backward" not in ev.name:
            continue
        if ev.cpu_children or ev.name.split("::")[-1] in VIEWS:  # leaves that move data
            continue
        shapes = [s for s in ev.input_shapes if s]
        if not shapes:
            continue
        big = max(shapes, key=lambda s: int(np.prod(s)))
        nbytes = 4 * int(np.prod(big))
        lead = big[0]
        kind = "edge" if lead == E else "node" if lead == N else "other"
        traffic[kind] += nbytes
        ops[(ev.name, kind)] += 1

    res = {
        "model": args.model, "matgl": matgl.__version__, "cell": args.cell, "atoms": N, "edges": E,
        "edges_per_atom": E / N, "energy_eV": float(e), "max_abs_force": float(np.abs(f).max()),
        "cpu_wall_s_with_profiler": wall,
        "gemm_forward": [{"rows": r, "in": i, "out": o, "calls": c, "gflop": 2e-9 * r * i * o * c}
                         for (r, i, o), c in sorted(gemm.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[0][2] * kv[1])],
        "gemm_forward_gflop": fwd * 1e-9,
        "gemm_step_gflop_est": 3 * fwd * 1e-9,  # reverse pass: two GEMMs per forward GEMM
        "leaf_op_largest_input_GB": {k: v / 1e9 for k, v in traffic.items()},
        "leaf_ops": {f"{n}|{k}": c for (n, k), c in sorted(ops.items(), key=lambda kv: -kv[1])[:40]},
    }
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
