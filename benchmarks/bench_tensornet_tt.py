"""TensorNet on one TT chip, timed: python bench_tensornet_tt.py BUNDLE.pt --out OUT.json [--sizes 216,512,1000].

A step is what an MD step costs through the bundle: neighbour list, float64 geometry, the network on the card, the
force closure on the host. Positions are re-jittered every step. Reports the median step and the share spent inside
the network call, with AICLK sampled during the timed steps, and grades each size against the float64 backend at two
jitters (signed dE per atom at each, so the energy difference between them can be read off, and force MAE).
--cpu times the float64 backend instead, to exercise the script without a card."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk

from tt_atom.tensornet import potential as P

CELLS = {64: 2, 216: 3, 512: 4, 1000: 5, 1728: 6}


def profile(dev):
    """Wrap every method of the device backend with device-synced timers -> {name: [seconds, calls]}, inclusive of
    callees. Counts restart when the caller clears the dict."""
    import ttnn
    from tt_atom.tensornet.tt_ops import TtOps

    acc = {}

    def wrap(name, fn):
        def timed(*a, **k):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn(*a, **k)
            ttnn.synchronize_device(dev)
            rec = acc.setdefault(name, [0.0, 0])
            rec[0] += time.perf_counter() - t0
            rec[1] += 1
            return out
        return timed

    for name, fn in list(vars(TtOps).items()):
        if callable(fn) and not name.startswith("_") and not isinstance(fn, (staticmethod, classmethod, property)):
            setattr(TtOps, name, wrap(name, fn))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="216,512,1000")
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--layer-dtype", default=None)
    ap.add_argument("--platform", default="")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--profile", action="store_true", help="device-synced inclusive time per backend method (slows the step)")
    args = ap.parse_args()
    bundle, rng = torch.load(args.bundle, weights_only=True), np.random.default_rng(0)
    torch.set_num_threads(1)
    calc, nodes, sampler, prof = None, [], None, None
    if args.cpu:
        inner = P.torch_run(bundle["w"], bundle["units"])
    else:
        import ttnn
        from bench_orb_perf_dollar_tt import AiclkSampler as sampler
        from tt_atom.tensornet_calculator import TensorNetCalculator
        calc = TensorNetCalculator(bundle, layer_dtype=getattr(ttnn, args.layer_dtype) if args.layer_dtype else None)
        inner = calc.run
        if args.profile:
            prof = profile(calc.device)
        # the /dev node this process opened: the AICLK that counts (node number != TT_VISIBLE_DEVICES index)
        fds = (os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd"))
        nodes = sorted({p.rsplit("/", 1)[1] for p in fds if p.startswith("/dev/tenstorrent/")})
    net = []

    def run(*a):
        t0 = time.perf_counter()
        out = inner(*a)
        net.append((time.perf_counter() - t0) * 1e3)
        return out

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    records = []
    for n in (int(s) for s in args.sizes.split(",")):
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * ((CELLS[n],) * 3)
        pos0, acc = atoms.get_positions(), {}
        first = None
        for jit in (0.01, 0.1):
            atoms.positions = pos0 + np.random.default_rng(1).normal(0, jit, pos0.shape)
            torch.set_num_threads(8)   # the float64 reference is not timed
            e0, f0 = P.evaluate(bundle, atoms)
            torch.set_num_threads(1)
            t0 = time.perf_counter()
            e1, f1 = P.evaluate(bundle, atoms, run)
            first = first or time.perf_counter() - t0
            acc[f"jitter_{jit}"] = {"dE_meV_per_atom": 1e3 * (e1 - e0) / n, "force_mae_meV_A": 1e3 * float(np.abs(f1 - f0).mean()),
                                    "mean_abs_force_eV_A": float(np.abs(f0).mean())}

        def step():
            atoms.positions = pos0 + rng.normal(0, args.jitter, pos0.shape)
            t0 = time.perf_counter()
            P.evaluate(bundle, atoms, run)
            return (time.perf_counter() - t0) * 1e3

        for _ in range(args.warmup):
            step()
        net.clear()
        if prof is not None:
            prof.clear()
        if sampler is None:
            ms, clk = [step() for _ in range(args.steps)], None
        else:
            with sampler() as s:
                ms = [step() for _ in range(args.steps)]
            clk = s.summary()
        rec = {"atoms": n, "first_call_s": first, "ms_per_step_median": float(np.median(ms)), "ms_per_step_min": min(ms),
               "network_ms_per_step_median": float(np.median(net)), "atoms_per_s": 1e3 * n / float(np.median(ms)), "ms_raw": ms,
               "accuracy_vs_float64": acc, "opened_nodes": nodes, "host_load1": os.getloadavg()[0], "aiclk_mhz_during_by_node": clk}
        records.append(rec)
        with open(args.out, "w") as f:
            json.dump({"model": "TensorNet (bundle %s)" % os.path.basename(args.bundle), "backend": "float64 cpu" if args.cpu else "tt",
                       "layer_dtype": args.layer_dtype or "float32", "platform": args.platform, "git_sha": sha,
                       "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "protocol": {"jitter_A": args.jitter, "warmup": args.warmup, "steps": args.steps, "mode": "eager",
                                    "neighbour_list": "rebuilt every step", "quantity": "energy + forces"},
                       "results": records}, f, indent=1)
        if prof is not None:
            print(f"{n} inclusive ms/step: " + ", ".join(f"{k} {v[0] * 1e3 / args.steps:.1f} ({v[1] // args.steps}x)" for k, v in
                                                          sorted(prof.items(), key=lambda kv: -kv[1][0])), flush=True)
        mine = {k: v for k, v in (clk or {}).items() if k in nodes} or clk or {}
        print(f"{n} atoms: {rec['ms_per_step_median']:.1f} ms/step (network {rec['network_ms_per_step_median']:.1f}), "
              f"{rec['atoms_per_s']:.0f} atoms/s, dE " + " / ".join(f"{v['dE_meV_per_atom']:+.2f}" for v in acc.values())
              + " meV/atom, force MAE " + " / ".join(f"{v['force_mae_meV_A']:.2f}" for v in acc.values()) + " meV/A"
              + (f", AICLK {({k: (v['median'], v['min']) for k, v in mine.items()})}" if mine else ""), flush=True)
    if calc is not None:
        calc.close()


if __name__ == "__main__":
    main()
