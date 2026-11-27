"""NVIDIA leg of the Orb-v3 per-dollar comparison, with the GPU at its best.

One protocol on both sides (the TT leg is `bench_orb_perf_dollar_tt.py`):
  * same checkpoint (`orb-v3-conservative-inf-omat`), same periodic Si diamond supercells,
    and the edge count is recorded so a graph mismatch is visible;
  * same quantity: energy + conservative forces (F = -dE/dpos), stress OFF
    (`disable_stress()`; orb's rotational-gradient term stays, a little extra GPU work);
  * same neighbour treatment: the list is built once at the first geometry and frozen,
    positions jittered each step and copied host -> device, as the TT leg does;
  * load, compile and capture excluded; median of the timed steps; the SM clock is sampled
    during the timed loop and recorded.

What "at its best" means here was checked on 2026-11-26 against what ships:
  * orb-models 0.7.0 (latest on PyPI): precision "float32-high" (TF32 matmuls, the package
    default), "float32-highest", "float64"; no bf16 path. `compile` defaults to
    torch.compile(mode="default", dynamic=True). kNN via NVIDIA ALCHEMI Toolkit-Ops.
  * NVIDIA ALCHEMI toolkit 0.3.0 (nvalchemi-toolkit): model wrappers for MACE, UMA,
    AIMNet2, no Orb wrapper, so there is no vendor-accelerated Orb path to compare to.

Variants (each parity-checked against the stock calculator at the same positions):
  stock            stock ORBCalculator, neighbours rebuilt per call, stress on. The plain
                   user path; reported for context, never the headline.
  frozen_compiled  package-compiled model, TF32, frozen graph, stress off.
  frozen_cudagraph uncompiled model captured whole (forward + autograd forces) into one CUDA
                   graph and replayed; the GPU counterpart of the TT leg's trace/replay.
The headline GPU number is the fastest variant that passes parity.

    python benchmarks/orb_gpu_bench_fair.py --out orb_perf_dollar_gpu_<gpu>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk

from _harness import git_sha
from bench_orb_perf_dollar_tt import SIZES

FROZEN = ("frozen_compiled", "frozen_cudagraph")


class ClockSampler:
    """Samples the SM clock (MHz) via nvidia-smi every 0.2 s while active."""

    def __init__(self):
        self.mhz, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.mhz.append(int(out.split()[0]))
            except Exception:  # noqa: BLE001 -- a missing sample must not kill the timing
                pass
            self._stop.wait(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def rotation_capture_safe(generator):
    """orb's `rotation_from_generator` without `torch.matrix_exp`, which syncs to the host and
    so cannot be captured in a CUDA graph. orb always evaluates it at generator = 0, where
    exp(S) and I + S + S@S/2 agree in value (I) and in first derivative, so energy, forces and
    the rotational gradient are unchanged. Parity against the stock calculator is still checked."""
    s = generator - generator.transpose(-2, -1)
    eye = torch.eye(3, dtype=s.dtype, device=s.device).expand_as(s)
    return eye + s + 0.5 * (s @ s)


def time_steps(step, warm, n):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    ts = []
    with ClockSampler() as clk:
        for _ in range(n):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return {"step_ms_raw": [round(x, 4) for x in ts], "step_ms_median": float(np.median(ts)),
            "steps_per_s": 1e3 / float(np.median(ts)),
            "sm_clock_mhz_during": (int(np.median(clk.mhz)) if clk.mhz else None),
            "sm_clock_mhz_min_max": ([min(clk.mhz), max(clk.mhz)] if clk.mhz else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.01, help="per-step position jitter, A")
    ap.add_argument("--sizes", default=",".join(s[0] for s in SIZES))
    ap.add_argument("--variants", default="stock,frozen_compiled,frozen_cudagraph")
    ap.add_argument("--precision", default="float32-high",
                    choices=["float32-high", "float32-highest"])
    ap.add_argument("--dev", action="store_true", help="dev run on a cheap card, not a claim")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import orb_models
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    dev = "cuda"
    load = pretrained.orb_v3_conservative_inf_omat
    m_stock, adapter = load(device=dev, precision=args.precision)
    calc = ORBCalculator(m_stock.eval(), adapter, device=dev)
    m_comp, _ = load(device=dev, precision=args.precision)               # package compile
    m_eager, _ = load(device=dev, precision=args.precision, compile=False)
    for m in (m_comp, m_eager):
        m.eval().disable_stress()
    models = {"frozen_compiled": m_comp, "frozen_cudagraph": m_eager}
    variants = [v for v in args.variants.split(",") if v]
    plan = [s for s in SIZES if s[0] in args.sizes.split(",")]
    meta = {"gpu_name": torch.cuda.get_device_name(0), "orb_models": orb_models.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "precision": args.precision,
            "dev_run": args.dev}
    print(meta, flush=True)

    rng = np.random.default_rng(args.seed)
    records = []
    for tag, nx, ny, nz in plan:
        atoms0 = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
        N = len(atoms0)
        pos0 = atoms0.get_positions()
        rec = {"tag": tag, "N": N, "variants": {}}
        print(f"\n=== {tag}  N={N} ===", flush=True)

        # parity anchor: stock calculator at one jittered geometry
        ref_atoms = atoms0.copy(); ref_atoms.calc = calc
        p_ref = pos0 + rng.normal(0.0, args.jitter, pos0.shape)
        ref_atoms.set_positions(p_ref)
        e_ref = float(ref_atoms.get_potential_energy())
        f_ref = np.asarray(ref_atoms.get_forces())

        if "stock" in variants:
            st = atoms0.copy(); st.calc = calc

            def stock_step():
                st.set_positions(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                st.get_potential_energy(); st.get_forces()
            rec["variants"]["stock"] = time_steps(stock_step, args.warmup, args.steps)

        graph = adapter.from_ase_atoms(atoms0).to(dev)
        rec["edges"] = int(graph.n_edge.item())
        for v in [v for v in variants if v in FROZEN]:
            m = models[v]
            batch = adapter.from_ase_atoms(atoms0).to(dev)
            pos = batch.node_features["positions"]
            host = torch.empty(N, 3, dtype=pos.dtype).pin_memory()

            def upload(p):  # predict marks positions requires_grad in place
                host.copy_(torch.as_tensor(p, dtype=pos.dtype))
                with torch.no_grad():
                    pos.copy_(host, non_blocking=True)

            def run():
                return m.predict(batch)
            try:
                if v == "frozen_cudagraph":
                    feat = orb_models.common.atoms.featurization
                    exp_rot = feat.rotation_from_generator
                    feat.rotation_from_generator = rotation_capture_safe
                    try:
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            for _ in range(3):
                                run()
                        torch.cuda.current_stream().wait_stream(s)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            static_out = run()
                    finally:
                        feat.rotation_from_generator = exp_rot

                    def run():  # noqa: F811 -- replay replaces the eager call
                        g.replay()
                        return static_out
                upload(p_ref)
                out = run()
                e = float(out[m.energy_name].double().sum())
                f = out[m.grad_forces_name].float().cpu().numpy()
                dE = abs(e - e_ref) / N
                fmax = float(np.abs(f - f_ref).max())
                ok = dE < 1e-4 and fmax < 5e-3
                print(f"  {v} parity: dE/atom={dE:.2e} eV  max|dF|={fmax:.2e} eV/A  ok={ok}", flush=True)

                def step():
                    upload(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                    run()
                r = time_steps(step, args.warmup, args.steps)
                r["parity"] = {"dE_per_atom_eV": dE, "max_abs_dF_eV_per_A": fmax, "ok": ok}
                rec["variants"][v] = r
            except Exception as exc:  # noqa: BLE001 -- a variant that cannot run is a recorded result
                import traceback  # the failing package line, so a capture break is fixable from the log
                where = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.line}"
                         for f in traceback.extract_tb(exc.__traceback__)[-4:]]
                rec["variants"][v] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}", "where": where}
                print("    " + "\n    ".join(where), flush=True)
        for v, r in rec["variants"].items():
            if "step_ms_median" in r:
                print(f"  {v:17s} {r['step_ms_median']:8.3f} ms  sm={r['sm_clock_mhz_during']} MHz", flush=True)
            else:
                print(f"  {v:17s} {r['error']}", flush=True)
        ok = {v: r for v, r in rec["variants"].items() if v in FROZEN and r.get("parity", {}).get("ok")}
        rec["headline"] = min(ok, key=lambda v: ok[v]["step_ms_median"]) if ok else None
        records.append(rec)
        dump(args, meta, records)


def dump(args, meta, records):
    with open(args.out, "w") as fh:
        json.dump({**meta, "git_sha": git_sha(),
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol": __doc__, "warmup": args.warmup, "steps": args.steps,
                   "records": records}, fh, indent=1)


if __name__ == "__main__":
    main()
