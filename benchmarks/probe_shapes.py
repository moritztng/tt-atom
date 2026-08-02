"""JOB 1 probe: with EDGE bucketing on, do NODE-dim shape changes still trigger kernel compiles?

Edge bucketing pins every edge-dim tensor to a ladder rung, but node-dim tensors ([N, C] node
features, [N*Dmax] gather tables) still vary per system. If changing N at a fixed edge bucket
recompiles kernels (~11s / ~40s signatures from bench_compile_pain.py JOB 0), node bucketing is
needed too; if it costs ~0 (tile-padded node ops reuse the edge-compiled programs), edge-only
bucketing suffices. This probe answers that with a cold-cache single leg, bucketing ON:

  A (2,2,4) N=32  -> E bucket 2464, N_tiles=1   cold: compiles everything
  B (2,2,5) N=40  -> E bucket 2464, N_tiles=2   N-tile crossing 1->2 at fixed edge bucket
  C (2,3,4) N=48  -> E bucket 2464, N_tiles=2   exact-N change, same tiles
  D (3,3,3) N=54  -> E bucket 2464, N_tiles=2   exact-N change again
  E (2,4,4) N=64  -> E bucket 3808, N_tiles=2   new edge bucket, same N_tiles (per-bucket cost)
  F (2,4,5) N=80  -> E bucket 3808, N_tiles=3   N-tile crossing 2->3 at fixed edge bucket
  G (2,2,5) N=40 seed 1 -> warm control: same shapes as B, new geometry -> ~0.1-0.3s expected

Same fleet discipline as bench_compile_pain.py: sandboxed HOME controls the tt-metal kernel
cache, the child holds the device-lease flock, and the parent waits for a quiet host window.

Run (qb1):  .venv/bin/python benchmarks/probe_shapes.py --card 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

LEASES = "/home/ttuser/.coworker/state/leases"

SYSTEMS = [("A", (2, 2, 4), 0), ("B", (2, 2, 5), 0), ("C", (2, 3, 4), 0), ("D", (3, 3, 3), 0),
           ("E", (2, 4, 4), 0), ("F", (2, 4, 5), 0), ("G", (2, 2, 5), 1)]


def run_child(weights, card):
    import fcntl
    import socket

    import torch
    from ase.build import bulk

    lease_path = pathlib.Path(LEASES) / f"{socket.gethostname()}-card{card}.json"
    lease_fd = os.open(lease_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(lease_fd, fcntl.LOCK_EX)
    with open(lease_path, "w") as f:
        json.dump({"host": socket.gethostname(), "card": str(card),
                   "holder": "worker:tt-atom-edge-bucketing", "pid": os.getpid(),
                   "acquired": time.time(), "released": None}, f)

    from tt_atom.bucketing import bucket_size
    from tt_atom.geometry import radius_graph
    from tt_atom.orb_calculator import OrbCalculator
    from tt_atom.orb_weights import OrbWeights

    t0 = time.perf_counter()
    calc = OrbCalculator(OrbWeights.load(weights), device_id=0, bucketing=True)
    print(json.dumps(dict(event="setup", seconds=round(time.perf_counter() - t0, 3))), flush=True)

    for name, cells, seed in SYSTEMS:
        atoms = bulk("Si", "diamond", a=5.43) * cells
        atoms.rattle(stdev=0.1, seed=seed)
        atoms.pbc = True
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        ei, _ = radius_graph(pos, calc.r_max,
                             cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
                             pbc=torch.tensor([True, True, True]))
        n, e = len(atoms), ei.shape[1]
        t0 = time.perf_counter()
        calc.calculate(atoms)
        dt = time.perf_counter() - t0
        print(json.dumps(dict(event="eval", name=name, N=n, E=e, E_bucket=bucket_size(e),
                              N_tiles=(n + 31) // 32, seconds=round(dt, 4),
                              energy=calc.results["energy"])), flush=True)
    calc.close()
    sys.stdout.flush()
    os._exit(0)          # skip the C++ atexit teardown abort (see bench_compile_pain.py)


def cache_stats(home):
    root = pathlib.Path(home) / ".cache"
    n = 0
    for p in root.rglob("*"):
        if p.is_file():
            n += 1
    return n


def host_quiet():
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "tt_bio" in line or "tt-bio-dev/env" in line:
            return False
        if "mcscale" in line and ".sh" in line:
            return False
    return True


def wait_for_quiet(poll_s=15, settle_s=10, max_wait_s=2400):
    t0 = time.time()
    quiet_since = None
    while time.time() - t0 < max_wait_s:
        if host_quiet():
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= settle_s:
                return True
        else:
            quiet_since = None
        time.sleep(poll_s)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",
                    default="/home/ttuser/.cache/tt_atom/orb_weights/conservative-inf-omat.npz")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--workdir", default="/tmp/ttatom_probe_shapes")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()

    if args.child:
        run_child(args.weights, args.card)
        return

    if not args.no_wait and not wait_for_quiet():
        print("no quiet window within budget; rerun later", flush=True)
        sys.exit(2)
    home = pathlib.Path(args.workdir) / "home_cold"
    import shutil
    shutil.rmtree(pathlib.Path(args.workdir), ignore_errors=True)
    home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["TT_VISIBLE_DEVICES"] = str(args.card)
    env.setdefault("OMP_NUM_THREADS", "4")
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, __file__, "--child", "--weights", args.weights,
                           "--card", str(args.card)], env=env, capture_output=True, text=True,
                          timeout=3600)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"probe child failed (rc={proc.returncode})")
    print(proc.stdout, end="")
    print(f"wall={wall:.1f}s cache_files={cache_stats(home)}")


if __name__ == "__main__":
    main()
