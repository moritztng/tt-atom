"""Tenstorrent leg of the MACE-MP-0b3 perf-per-dollar comparison (one Blackhole chip), the protocol of the Orb
leg and of the NVIDIA leg (mace_gpu_bench_fair.py on wk/vrt-atom-mace-gpu): periodic Si diamond supercells,
energy + forces, no stress, neighbour list built once and frozen, positions jittered and uploaded every step,
median step time, AICLK sampled during. Each size is graded against float64: the NVIDIA leg's anchors
(--anchors) when given, else this repo's float64 reference at the same positions.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python benchmarks/bench_mace_tt.py \
        --weights ~/.ttatom_run/goldens_real/si_mace_mp0b3.npz --out benchmarks/mace_perf_dollar_tt.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk
from ase.neighborlist import neighbor_list

from _harness import git_sha
from bench_orb_perf_dollar_tt import SIZES as ORB_SIZES, AiclkSampler, grade, load_anchors

SIZES = [("2x2x2", 2, 2, 2), *ORB_SIZES]  # 64, 216, 512, 1000, 2016 atoms


def profile(dev):
    """Wrap the block functions with device-synced timers -> {name: [seconds, calls]}, inclusive of callees.
    Counts restart when the caller clears the dict's values."""
    import ttnn

    from tt_atom import scatter
    from tt_atom.mace import tt_blocks as B

    acc = {}

    def wrap(mod, name):
        fn = getattr(mod, name)

        def timed(*a, **k):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn(*a, **k)
            ttnn.synchronize_device(dev)
            rec = acc.setdefault(name, [0.0, 0])
            rec[0] += time.perf_counter() - t0
            rec[1] += 1
            return out
        setattr(mod, name, timed)

    for name in ("mm", "perm", "rows", "fc_net", "fc_net_bwd", "symmetric_contraction",
                 "symmetric_contraction_bwd", "interaction", "interaction_bwd", "product", "product_bwd"):
        wrap(B, name)
    wrap(scatter, "segment_sum")
    from tt_atom import device as D  # the host side of a step: geometry, its VJP, transfers
    from tt_atom.mace import backward, reference
    for mod, names in ((reference, ("edge_vectors", "spherical_harmonics", "radial_embedding", "zbl")),
                       (backward, ("geometry_vjp",)), (D, ("to_dev",)), (ttnn, ("to_torch",)), (B, ("pad_edges",))):
        for name in names:
            wrap(mod, name)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="MACE golden npz (tests/gen_golden_mace.py), carries the weights")
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.01, help="per-step position jitter, A")
    ap.add_argument("--sizes", default=",".join(s[0] for s in SIZES))
    ap.add_argument("--anchors", default=None, help="float64 anchors npz from mace_gpu_bench_fair.py --anchors-out")
    ap.add_argument("--platform", default="Tenstorrent Blackhole, one chip")
    ap.add_argument("--out", default="benchmarks/mace_perf_dollar_tt.json")
    ap.add_argument("--profile", action="store_true",
                    help="device-synced inclusive time per block function (slows the step; not a timing run)")
    args = ap.parse_args()

    import ttnn

    from tt_atom import device as D
    from tt_atom.mace import backward
    from tt_atom.mace.tt_model import Mace

    # The host side of a step is a few elementwise ops over the edges (2 % of the step). Left to torch's thread
    # pool on a busy 64-core host it took 3x longer than the card (1.4 s of a 1.9 s step at 216 atoms).
    torch.set_num_threads(1)
    z = np.load(args.weights)
    w = {k[2:]: torch.from_numpy(np.asarray(z[k])) for k in z.files if k.startswith("w@")}
    w = {k: v.double() if v.is_floating_point() else v for k, v in w.items()}
    r_max = float(w["radial_embedding.cutoff_fn.r_max"])
    anchors = load_anchors(args.anchors) if args.anchors else {}
    rng = np.random.default_rng(args.seed)
    records = []

    def dump():
        with open(args.out, "w") as f:
            json.dump({"model": "MACE-MP-0b3 medium", "precision": "float32", "platform": args.platform,
                       "git_sha": git_sha(), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "protocol": {"jitter_A": args.jitter, "warmup": args.warmup, "steps": args.steps,
                                    "neighbour_list": "frozen", "quantity": "energy + forces"},
                       "results": records}, f, indent=1)

    with D.device(0) as dev:
        model = Mace(dev, w, ttnn.float32)
        prof = profile(dev) if args.profile else None
        # the /dev node this process opened: the AICLK that counts (node number != TT_VISIBLE_DEVICES index)
        fds = (os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd"))
        nodes = sorted({p.rsplit("/", 1)[1] for p in fds if p.startswith("/dev/tenstorrent/")})
        for tag, nx, ny, nz in (s for s in SIZES if s[0] in args.sizes.split(",")):
            atoms = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
            pos0 = atoms.get_positions()
            i, j, S = neighbor_list("ijS", atoms, r_max)  # list of the unjittered lattice, frozen
            Z = torch.from_numpy(atoms.get_atomic_numbers()).long()
            ei = torch.from_numpy(np.stack([i, j])).long()
            shifts = torch.from_numpy(S @ atoms.cell.array).double()
            if tag in anchors:
                p_ref, e_ref, f_ref = anchors[tag]
                ref_src = "mace-torch float64 (NVIDIA leg anchors)"
            else:
                p_ref = pos0 + rng.normal(0.0, args.jitter, pos0.shape)
                e, f = backward.energy_forces(w, Z, torch.from_numpy(p_ref), ei, shifts)
                e_ref, f_ref, ref_src = float(e), f.numpy(), "tt_atom.mace float64 reference"
            t0 = time.perf_counter()
            e, f = model.energy_forces(Z, torch.from_numpy(p_ref), ei, shifts)
            first = time.perf_counter() - t0
            acc = grade(e, f.numpy(), e_ref, f_ref)
            for _ in range(args.warmup):
                model.energy_forces(Z, torch.from_numpy(pos0 + rng.normal(0.0, args.jitter, pos0.shape)), ei, shifts)
            ms = []
            if prof is not None:
                prof.clear()
            with AiclkSampler() as clk:
                for _ in range(args.steps):
                    p = torch.from_numpy(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                    t0 = time.perf_counter()
                    model.energy_forces(Z, p, ei, shifts)
                    ms.append((time.perf_counter() - t0) * 1e3)
            if prof is not None:
                print(f"{tag} inclusive ms/step: " + ", ".join(
                    f"{k} {v[0] * 1e3 / args.steps:.1f} ({v[1] // args.steps}x)" for k, v in
                    sorted(prof.items(), key=lambda kv: -kv[1][0])), flush=True)
            records.append({"size": tag, "atoms": len(atoms), "edges": int(ei.shape[1]), "first_call_s": first,
                            "ms_per_step_median": float(np.median(ms)), "ms_per_step_min": min(ms), "ms_raw": ms,
                            "reference": ref_src, **acc, "opened_nodes": nodes,
                            "host_load1": os.getloadavg()[0], "aiclk_mhz_during_by_node": clk.summary()})
            dump()
            mine = {k: v for k, v in clk.summary().items() if k in nodes} or clk.summary()
            busy = min(mine.items(), key=lambda kv: kv[1]["min"])
            print(f"{tag} {len(atoms)} atoms {ei.shape[1]} edges: {np.median(ms):.1f} ms/step (min {min(ms):.1f}), "
                  f"dE/atom {acc['dE_per_atom_eV']:.2e} eV, force MAE {acc['mae_dF_eV_per_A']:.2e} eV/A, "
                  f"AICLK node {busy[0]} median {busy[1]['median']} min {busy[1]['min']}, host load {os.getloadavg()[0]:.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
