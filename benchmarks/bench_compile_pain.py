"""JOB 0: quantify the kernel-compile pain of differently-sized systems on one card.

Each leg is a subprocess whose HOME (and XDG_CACHE_HOME) points at a per-leg directory, so the
tt-metal persistent kernel cache under $HOME/.cache/tt-metal-cache is exactly controlled:
a fresh dir = cold (every new shape compiles), a populated dir = warm. The child evaluates
rattled periodic Si supercells (energy+forces, orb-v3-conservative-inf-omat) and prints one
JSON line per system with N / E / Dmax / tile shapes / eval seconds; the parent diffs cache
file counts and wall-clock per leg.

Fleet discipline is ``benchmarks/_harness.py``'s (lease flock, quiet-host wait, sandbox env),
shared with the other subprocess benchmarks. Legs are kept short (<= ~2 systems) so a sibling's
120 s lease timeout never trips.

Run (qb1):  .venv/bin/python benchmarks/bench_compile_pain.py --card 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from _harness import cache_stats, orb_weights, sandbox_env, take_lease, wait_for_quiet


def run_child(weights, systems, tag, card):
    """Child mode: one process, one device, evaluate the systems sequentially, JSON-lines out."""
    import torch
    from ase.build import bulk

    wait_s = take_lease(card)

    from tt_atom.geometry import radius_graph
    from tt_atom.orb_weights import OrbWeights

    t_setup0 = time.perf_counter()
    from tt_atom.orb_calculator import OrbCalculator

    calc = OrbCalculator(OrbWeights.load(weights), device_id=0)
    setup_s = time.perf_counter() - t_setup0
    print(json.dumps(dict(tag=tag, event="setup", seconds=round(setup_s, 3),
                          lease_wait_s=round(wait_s, 3))), flush=True)

    for i, (cells, seed) in enumerate(systems):
        atoms = bulk("Si", "diamond", a=5.43) * (cells, cells, cells)
        atoms.rattle(stdev=0.1, seed=seed)
        atoms.pbc = True
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        ei, _ = radius_graph(pos, calc.r_max,
                             cell=torch.tensor(atoms.cell.array, dtype=torch.float32),
                             pbc=torch.tensor([True, True, True]))
        src, tgt = ei
        n = len(atoms)
        dmax = max(int(torch.bincount(tgt, minlength=n).max()),
                   int(torch.bincount(src, minlength=n).max()))
        t0 = time.perf_counter()
        calc.calculate(atoms)
        dt = time.perf_counter() - t0
        e = ei.shape[1]
        print(json.dumps(dict(tag=tag, event="eval", i=i, cells=cells, seed=seed, N=n, E=e,
                              Dmax=dmax, N_tiles=(n + 31) // 32, E_tiles=(e + 31) // 32,
                              seconds=round(dt, 4),
                              energy=calc.results["energy"])), flush=True)
    calc.close()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip the C++ atexit MetalContext teardown: it aborts on this build after close_device,
    # and the abort can leave the card needing a reset. The flock is released by the kernel.
    os._exit(0)


def run_leg(weights, systems, home, tag, card):
    env = sandbox_env(home, card)
    pathlib.Path(home).mkdir(parents=True, exist_ok=True)
    files0, _ = cache_stats(home)
    proc = None
    for attempt in (1, 2):          # device opens are occasionally transient; retry once
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, "--child", "--weights", weights, "--tag", tag,
             "--card", str(card), "--systems", json.dumps(systems)],
            env=env, capture_output=True, text=True, timeout=3600)
        wall = time.perf_counter() - t0
        if proc.returncode == 0:
            break
        print(f"[{tag}] attempt {attempt} failed rc={proc.returncode}: "
              f"{proc.stderr[-800:]}", file=sys.stderr, flush=True)
        if tag.startswith("cold"):
            # A crashed cold attempt leaves a partially populated cache; rerunning on it
            # would measure a warm-ish first eval. Wipe so the retry is a true cold leg.
            shutil.rmtree(home, ignore_errors=True)
            pathlib.Path(home).mkdir(parents=True, exist_ok=True)
            files0, _ = cache_stats(home)
        time.sleep(5)
    files1, bytes1 = cache_stats(home)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"child leg {tag} failed (rc={proc.returncode})")
    evals = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    return dict(tag=tag, wall_s=round(wall, 2), cache_files_before=files0,
                cache_files_after=files1, cache_mb=round(bytes1 / 1e6, 1), events=evals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(orb_weights()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--cells", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--workdir", default="/tmp/ttatom_compile_pain")
    ap.add_argument("--no-wait", action="store_true", help="don't wait for quiet windows")
    ap.add_argument("--deadline-s", type=float, default=3300,
                    help="stop starting new legs after this many seconds")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--tag", default="leg")
    args = ap.parse_args()

    if args.child:
        run_child(args.weights, [tuple(x) for x in json.loads(args.systems)], args.tag, args.card)
        return

    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    jsonl = pathlib.Path(args.out or str(workdir / "compile_pain.jsonl"))
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["tag"])

    # Per-size legs: cold1 populates the size's warm home, then warm1/cold2/warm2 interleave
    # so thermal drift hits cold and warm legs of a size symmetrically.
    plan = []
    for cells in args.cells:
        systems = [(cells, s) for s in args.seeds]
        warm_home = str(workdir / f"home_warm_{cells}")
        plan += [(f"cold1_c{cells}", systems, warm_home),
                 (f"warm1_c{cells}", systems, warm_home),
                 (f"cold2_c{cells}", systems, str(workdir / f"home_cold2_{cells}")),
                 (f"warm2_c{cells}", systems, warm_home)]
    t_start = time.time()
    for tag, systems, home in plan:
        if tag in done:
            print(f"[{tag}] already done, skipping", flush=True)
            continue
        if time.time() - t_start > args.deadline_s:
            print(f"[{tag}] deadline reached; stopping (resumable)", flush=True)
            break
        if not args.no_wait and not wait_for_quiet():
            print(f"[{tag}] no quiet window within budget; stopping (resumable)", flush=True)
            break
        leg = run_leg(args.weights, systems, home, tag, args.card)
        with open(jsonl, "a") as f:
            f.write(json.dumps(leg) + "\n")
        ev = [e for e in leg["events"] if e.get("event") == "eval"]
        new_files = leg["cache_files_after"] - leg["cache_files_before"]
        print(f"[{leg['tag']}] wall={leg['wall_s']}s new_cache_files={new_files} "
              f"eval_s={[e['seconds'] for e in ev]}", flush=True)
    print(f"results -> {jsonl}")


if __name__ == "__main__":
    main()
