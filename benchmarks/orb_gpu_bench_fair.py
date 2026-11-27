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

Every variant, stock included, is graded at the same jittered positions against the float64 model
through the stock calculator (dE/atom, MAE and max |dF|); the anchors are saved (--anchors-out) and
the TT leg is graded against the very same file. No bar is applied here: the headline is the
fastest frozen variant, at either precision, that is no less accurate than the TT leg at that size.
Run once per precision.

Variants:
  stock            stock ORBCalculator, neighbours rebuilt per call, stress on. The plain
                   user path; reported for context, never the headline.
  frozen_compiled  package-compiled model, TF32, frozen graph, stress off.
  frozen_cudagraph uncompiled model captured whole (forward + autograd forces) into one CUDA
                   graph and replayed; the GPU counterpart of the TT leg's trace/replay.
  frozen_compiled_cudagraph  both: torch.compile (static shapes, the graph is frozen) for fused
                   kernels, then the whole step captured into one CUDA graph.

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
from bench_orb_perf_dollar_tt import SIZES, grade

FROZEN = ("frozen_compiled", "frozen_cudagraph", "frozen_compiled_cudagraph")
CAPTURED = ("frozen_cudagraph", "frozen_compiled_cudagraph")


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


def sized_repeat(rep):
    """`repeat_interleave` with a tensor of repeats reads its output size back to the host, which a
    CUDA graph cannot capture. The graph here is frozen, so the forward makes the same calls with the
    same sizes every time: the first eager run records each call's size in call order and later runs
    (capture included) replay them. Call `begin()` before each forward. A call whose shape signature
    differs from the recording falls back to computing its size, so a mismatch shows up in the sync
    probe instead of silently reusing a wrong size."""
    sizes, i = [], [0]

    def sized(x, repeats=None, dim=None, *, output_size=None):
        if output_size is None and isinstance(repeats, torch.Tensor):
            n = x.numel() if dim is None else x.shape[dim]
            sig = (tuple(repeats.shape), dim, n)
            k = i[0]; i[0] += 1
            if k < len(sizes) and sizes[k][0] == sig:
                output_size = sizes[k][1]
            else:  # one repeat broadcasts over the axis, several are summed
                output_size = int(repeats.sum()) if repeats.numel() > 1 else n * int(repeats)
                if k == len(sizes):
                    sizes.append((sig, output_size))
        if repeats is None:
            return rep(x, output_size=output_size)
        return rep(x, repeats, dim, output_size=output_size)

    sized.begin = lambda: i.__setitem__(0, 0)
    return sized


def sync_probe(run):
    """Log every host-syncing site of one eager call; any of them breaks a CUDA-graph capture."""
    import traceback
    import warnings
    sites = []

    def record(*_a, **_k):
        frames = [f for f in traceback.extract_stack()[:-1] if "orb_models" in f.filename]
        site = " <- ".join(f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in frames[-3:][::-1])
        if site not in sites:
            sites.append(site)
    torch.cuda.set_sync_debug_mode("warn")
    old, warnings.showwarning = warnings.showwarning, record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = record
            run()
    finally:
        warnings.showwarning = old
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    print(f"    sync probe: {len(sites)} host-syncing site(s)" + "".join("\n      " + x for x in sites), flush=True)


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
    ap.add_argument("--variants", default="stock,frozen_compiled,frozen_cudagraph,frozen_compiled_cudagraph")
    ap.add_argument("--anchors-out", default=None, help="save the float64 anchors (npz) for the TT leg")
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
    plan = [s for s in SIZES if s[0] in args.sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    # Parity anchor: the float64 model through the stock calculator, one jittered geometry per size.
    # Done first and alone: orb's precision (default dtype, matmul precision) is process-global, and its
    # float32-high default puts the edge geometry (unit_shifts @ cell) through TF32. On an H200 that made
    # stock and frozen float32-high disagree by 0.1-0.2 eV/A (11-27), so neither may serve as the anchor.
    m64, adapter = load(device=dev, precision="float64", compile=False)
    calc64 = ORBCalculator(m64.eval(), adapter, device=dev)
    anchors = {}
    for tag, nx, ny, nz in plan:
        a = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
        a.set_positions(a.get_positions() + rng.normal(0.0, args.jitter, (len(a), 3)))
        a.calc = calc64
        anchors[tag] = (a.get_positions().copy(), float(a.get_potential_energy()),
                        np.array(a.get_forces(), dtype=np.float64))
    if args.anchors_out:
        np.savez(args.anchors_out, **{f"{t}{k}": v for t, (p, e, f) in anchors.items()
                                      for k, v in (("_pos", p), ("_e", np.float64(e)), ("_f", f))})
    del calc64, m64
    torch.cuda.empty_cache()
    m_stock, adapter = load(device=dev, precision=args.precision)
    calc = ORBCalculator(m_stock.eval(), adapter, device=dev)
    m_comp, _ = load(device=dev, precision=args.precision)               # package compile
    m_eager, _ = load(device=dev, precision=args.precision, compile=False)
    m_cc, _ = load(device=dev, precision=args.precision, compile=False)
    m_cc.compile(mode="default", dynamic=False)  # frozen graph: one static shape per size
    for m in (m_comp, m_eager, m_cc):
        m.eval().disable_stress()
    models = {"frozen_compiled": m_comp, "frozen_cudagraph": m_eager, "frozen_compiled_cudagraph": m_cc}
    # One process walks four sizes and three compiled call paths; dynamo's default recompile cap (8 per
    # frame, 256 overall) was hit on H200 (11-27) and silently dropped frames to eager.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    variants = [v for v in args.variants.split(",") if v]
    meta = {"gpu_name": torch.cuda.get_device_name(0), "orb_models": orb_models.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "precision": args.precision,
            "dev_run": args.dev, "parity_anchor": "orb float64, stock calculator, fresh graph"}
    print(meta, flush=True)

    records = []
    for tag, nx, ny, nz in plan:
        atoms0 = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
        N = len(atoms0)
        pos0 = atoms0.get_positions()
        rec = {"tag": tag, "N": N, "variants": {}}
        print(f"\n=== {tag}  N={N} ===", flush=True)

        p_ref, e_ref, f_ref = anchors[tag]

        def parity(e, f):
            return grade(e, f, e_ref, f_ref)

        def show(v, par):
            print(f"  {v} vs float64: dE/atom={par['dE_per_atom_eV']:.2e} eV  MAE|dF|={par['mae_dF_eV_per_A']:.2e}"
                  f"  max|dF|={par['max_abs_dF_eV_per_A']:.2e} eV/A", flush=True)

        if "stock" in variants:
            st = atoms0.copy(); st.calc = calc
            st.set_positions(p_ref)
            par = parity(float(st.get_potential_energy()), np.asarray(st.get_forces()))
            show("stock", par)

            def stock_step():
                st.set_positions(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                st.get_potential_energy(); st.get_forces()
            rec["variants"]["stock"] = {**time_steps(stock_step, args.warmup, args.steps), "parity": par}

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
                if v in CAPTURED:
                    feat = orb_models.common.atoms.featurization
                    exp_rot = feat.rotation_from_generator
                    feat.rotation_from_generator = rotation_capture_safe
                    rep_t, rep_m = torch.repeat_interleave, torch.Tensor.repeat_interleave
                    sized = sized_repeat(rep_t)
                    if v == "frozen_compiled_cudagraph":  # the size cache is host state: keep it out of the trace
                        inner = sized
                        sized = torch.compiler.disable(inner)
                        sized.begin = inner.begin
                    torch.repeat_interleave = torch.Tensor.repeat_interleave = sized
                    eager = run

                    def run():  # noqa: F811 -- every forward replays the recorded sizes in order
                        sized.begin()
                        return eager()
                    try:
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            for _ in range(3):
                                run()
                        torch.cuda.current_stream().wait_stream(s)
                        sync_probe(run)
                        if v == "frozen_compiled_cudagraph":
                            # Dynamo may not compile inside a capture: log why a frame recompiles and keep
                            # calling (the probe ran on the default stream) until a call compiles nothing.
                            from torch._dynamo.utils import counters
                            torch._logging.set_logs(recompiles=True)
                            for _ in range(8):
                                before = counters["frames"]["total"]
                                with torch.cuda.stream(s):
                                    run()
                                torch.cuda.synchronize()
                                if counters["frames"]["total"] == before:
                                    break
                            torch._logging.set_logs(recompiles=False)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, capture_error_mode="thread_local"):
                            static_out = run()
                    finally:
                        feat.rotation_from_generator = exp_rot
                        torch.repeat_interleave, torch.Tensor.repeat_interleave = rep_t, rep_m

                    def run():  # noqa: F811 -- replay replaces the eager call
                        g.replay()
                        return static_out
                upload(p_ref)
                out = run()
                e = float(out[m.energy_name].double().sum())
                f = out[m.grad_forces_name].float().cpu().numpy()
                par = parity(e, f)
                show(v, par)

                def step():
                    upload(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                    run()
                r = time_steps(step, args.warmup, args.steps)
                r["parity"] = par
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
