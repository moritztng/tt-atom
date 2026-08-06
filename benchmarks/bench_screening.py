"""JOB 3: screening-stream benchmark — bucketed vs unbucketed Orb on differently-sized systems.

The production case from the workstream: a virtual-screening stream feeds K differently-sized
systems through one calculator; every distinct (N, E) shape compiles fresh kernels, and the
shape distribution is effectively continuous so the disk cache never saves you. This benchmark
streams the SAME K rattled Si systems (sizes log-spread across the full edge ladder) through
OrbCalculator with bucketing OFF vs ON, in cold vs warm sandbox-HOME cache states, and reports
per-leg wall-clock, kernel-cache growth (compile proxy), per-eval seconds, and steady-state
Medges/s. The verdict pair is ``unbucketed_cold`` vs ``bucketed_cold`` (the production first-
run experience) with the ``*_warm`` legs as the warm-cache-fairness control.

Legs interleave cold/warm per mode so thermal drift hits both modes symmetrically; each cold
leg gets a fresh sandbox HOME (its cache then serves as that mode's warm HOME). Fleet
discipline identical to bench_compile_pain.py: child holds the device-lease flock, parent
waits for a quiet host window unless --no-wait.

Run (qb1):  .venv/bin/python benchmarks/bench_screening.py --card 0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pwd
import subprocess
import sys
import time


def _real_home() -> pathlib.Path:
    """The invoking user's home from the passwd database, NOT ``$HOME``: the sandbox-HOME legs
    below override ``$HOME`` to control the kernel cache, and the fleet lease must still land in
    the real ``~/.coworker/state/leases`` so we serialize with sibling fleet jobs."""
    return pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)


LEASES = _real_home() / ".coworker" / "state" / "leases"
HOLDER = os.environ.get("TT_BIO_LEASE_HOLDER", "tt-atom-benchmark")
DEFAULT_WEIGHTS = _real_home() / ".cache/tt_atom/orb_weights/conservative-inf-omat.npz"


def run_child(weights, systems, tag, card, bucketing):
    import fcntl
    import socket

    import torch
    from ase.build import bulk

    LEASES.mkdir(parents=True, exist_ok=True)
    lease_path = LEASES / f"{socket.gethostname()}-card{card}.json"
    lease_fd = os.open(lease_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(lease_fd, fcntl.LOCK_EX)
    with open(lease_path, "w") as f:
        json.dump({"host": socket.gethostname(), "card": str(card),
                   "holder": HOLDER, "pid": os.getpid(),
                   "acquired": time.time(), "released": None}, f)

    from tt_atom.bucketing import bucket_size
    from tt_atom.geometry import radius_graph
    from tt_atom.orb_calculator import OrbCalculator
    from tt_atom.orb_weights import OrbWeights

    t0 = time.perf_counter()
    calc = OrbCalculator(OrbWeights.load(weights), device_id=0, bucketing=bucketing)
    print(json.dumps(dict(tag=tag, event="setup", seconds=round(time.perf_counter() - t0, 3))),
          flush=True)

    for i, (cells, seed) in enumerate(systems):
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
        print(json.dumps(dict(tag=tag, event="eval", i=i, N=n, E=e, E_bucket=bucket_size(e),
                              seconds=round(dt, 4))), flush=True)
    calc.close()
    sys.stdout.flush()
    os._exit(0)          # skip the C++ atexit teardown abort (see bench_compile_pain.py)


def cache_stats(home):
    root = pathlib.Path(home) / ".cache"
    n, b = 0, 0
    for p in root.rglob("*"):
        if p.is_file():
            n += 1
            b += p.stat().st_size
    return n, b


def run_leg(weights, systems, home, tag, card, bucketing):
    env = dict(os.environ)
    env["HOME"] = home
    env["XDG_CACHE_HOME"] = str(pathlib.Path(home) / ".cache")
    env["TT_VISIBLE_DEVICES"] = str(card)
    env.setdefault("OMP_NUM_THREADS", "4")
    pathlib.Path(home).mkdir(parents=True, exist_ok=True)
    files0, _ = cache_stats(home)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, __file__, "--child", "--weights", weights, "--tag", tag,
         "--card", str(card), "--bucketing", str(int(bucketing)),
         "--systems", json.dumps(systems)],
        env=env, capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - t0
    files1, bytes1 = cache_stats(home)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"child leg {tag} failed (rc={proc.returncode})")
    evals = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    return dict(tag=tag, bucketing=bucketing, wall_s=round(wall, 2),
                cache_files_before=files0, cache_files_after=files1,
                cache_mb=round(bytes1 / 1e6, 1), events=evals)


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


def screening_stream():
    """K=20 rattled Si systems, sizes log-spread N=16..512 -> E ~0.7k..21k (the full ladder).

    Supercell shapes (a, b, c) with 2*a*b*c atoms; each system is its own (N, E) shape, so the
    unbucketed stream compiles per system — the production distribution the workstream cites.
    """
    cells_list = [(2, 2, 2), (2, 2, 3), (2, 3, 3), (2, 2, 5), (3, 3, 3), (2, 4, 4), (3, 3, 4),
                  (2, 4, 5), (3, 4, 4), (2, 5, 5), (3, 4, 5), (4, 4, 4), (3, 4, 6), (2, 5, 7),
                  (3, 5, 5), (4, 4, 5), (3, 5, 6), (4, 4, 6), (3, 6, 6), (4, 4, 8)]
    return [(c, s) for s, c in enumerate(cells_list)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--workdir", default="/tmp/ttatom_screening")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--tag", default="leg")
    ap.add_argument("--bucketing", type=int, default=0)
    args = ap.parse_args()

    if args.child:
        run_child(args.weights, [tuple(x) for x in json.loads(args.systems)], args.tag,
                  args.card, bool(args.bucketing))
        return

    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    jsonl = pathlib.Path(args.out or str(workdir / "screening.jsonl"))
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["tag"])

    systems = screening_stream()
    plan = [("unbucketed_cold", False, str(workdir / "home_unbucketed")),
            ("bucketed_cold", True, str(workdir / "home_bucketed")),
            ("unbucketed_warm", False, str(workdir / "home_unbucketed")),
            ("bucketed_warm", True, str(workdir / "home_bucketed"))]
    for tag, bucketing, home in plan:
        if tag in done:
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        if not args.no_wait and not wait_for_quiet():
            print(f"[{tag}] no quiet window within budget; stopping (resumable)", flush=True)
            break
        leg = run_leg(args.weights, systems, home, tag, args.card, bucketing)
        with open(jsonl, "a") as f:
            f.write(json.dumps(leg) + "\n")
        ev = [e for e in leg["events"] if e.get("event") == "eval"]
        medges = sum(e["E"] for e in ev) / 1e6 / max(sum(e["seconds"] for e in ev), 1e-9)
        print(f"[{tag}] wall={leg['wall_s']}s new_cache_files="
              f"{leg['cache_files_after'] - leg['cache_files_before']} "
              f"eval_s_mean={sum(e['seconds'] for e in ev) / max(len(ev), 1):.3f} "
              f"stream_Medges_s={medges:.3f}", flush=True)
    print(f"results -> {jsonl}")


if __name__ == "__main__":
    main()
