"""NVIDIA leg of the MACE per-dollar comparison, with the GPU at its best.

Same protocol as the Orb leg (`orb_gpu_bench_fair.py`, shared machinery in `_gpu_fair.py`):
  * checkpoint MACE-MP-0b3 medium (MIT), periodic Si diamond supercells, edge count recorded;
  * one step = energy + conservative forces (F = -dE/dpos), stress off;
  * the neighbour list is built once and frozen, positions are jittered every step and copied
    host -> device;
  * load, compile and capture excluded; median of the timed steps; SM clock sampled during them.

What "at its best" means was checked on 2026-11-28 against what ships in mace-torch 0.3.17 (latest
on PyPI). The package has four tensor-product backends and its own compile path, and every one is
a variant here, named <backend>_<mode>:
  backend  e3nn    stock
           cueq    NVIDIA cuEquivariance (`enable_cueq`)
           oeq     OpenEquivariance (`enable_oeq`)
           hybrid  cueq for the symmetric contraction and linears, oeq for the convolution
  mode     eager            the model as the calculator calls it
           compiled         the calculator's `compile_mode="default"`
           reduce_overhead  the calculator's `compile_mode="reduce-overhead"` (PyTorch CUDA graphs)
           cudagraph        the eager step, forces included, captured whole into one CUDA graph
plus `stock`: the plain MACECalculator call (neighbours rebuilt, stress on), for context only.
Precision is per process: float32-high (TF32 matmuls), float32-highest, float64.

Packing: MACE batches structures, so a GPU that is not full at one cell is timed again with B
copies of the cell in one batch (--pack). atoms/s there is B x N per step, and the best of them
is what a per-dollar comparison at that cell size has to use.

Every variant is graded at the same jittered positions against stock mace-torch in float64
(dE/atom, MAE and max |dF|); the anchors are saved (--anchors-out) for the TT leg.

    python benchmarks/mace_gpu_bench_fair.py --out mace_gpu_<gpu>_<precision>.json
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np
import torch
from ase.build import bulk

from _gpu_fair import capture, dump, time_steps
from bench_orb_perf_dollar_tt import SIZES as ORB_SIZES, grade

SIZES = [("2x2x2", 2, 2, 2), *ORB_SIZES]  # 64, 216, 512, 1000, 2016 atoms
BACKENDS = {"e3nn": {}, "cueq": {"enable_cueq": True}, "oeq": {"enable_oeq": True},
            "hybrid": {"enable_cueq": True, "enable_oeq": True}}
MODES = {"eager": None, "compiled": "default", "reduce_overhead": "reduce-overhead", "cudagraph": None}
ALL = [f"{b}_{m}" for b in BACKENDS for m in MODES]


def failure(exc):
    """A variant that cannot load, capture or fit is a recorded result, with the failing line."""
    where = [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.line}"
             for f in traceback.extract_tb(exc.__traceback__)[-4:]]
    print("    " + "\n    ".join(where), flush=True)
    return {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "where": where}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium-0b3")
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jitter", type=float, default=0.01, help="per-step position jitter, A")
    ap.add_argument("--sizes", default=",".join(s[0] for s in SIZES))
    ap.add_argument("--variants", default=",".join(["stock"] + ALL))
    ap.add_argument("--pack", default="", help="comma list of copies per batch, e.g. 4,16,64")
    ap.add_argument("--pack-variants", default="", help="variants timed packed (default: all but stock)")
    ap.add_argument("--pack-max-atoms", type=int, default=40000)
    ap.add_argument("--anchors-out", default=None, help="save the float64 anchors (npz) for the TT leg")
    ap.add_argument("--precision", default="float32-high",
                    choices=["float32-high", "float32-highest", "float64"])
    ap.add_argument("--device", default="cuda", help="cpu only smoke-tests the script")
    ap.add_argument("--dev", action="store_true", help="dev run on a cheap card, not a claim")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mace
    from mace import data as mace_data
    from mace.calculators import mace_mp
    from mace.tools import torch_geometric, torch_tools

    dev = args.device
    dtype = "float64" if args.precision == "float64" else "float32"
    torch.set_float32_matmul_precision("high" if args.precision == "float32-high" else "highest")
    torch.backends.cudnn.allow_tf32 = args.precision == "float32-high"
    plan = [s for s in SIZES if s[0] in args.sizes.split(",")]
    variants = [v for v in args.variants.split(",") if v]
    packs = [int(b) for b in args.pack.split(",") if b]
    pack_variants = [v for v in (args.pack_variants.split(",") if args.pack_variants else variants) if v != "stock"]
    rng = np.random.default_rng(args.seed)

    def cell(nx, ny, nz):
        return bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)

    # Parity anchor first and alone: stock mace-torch, float64, fresh neighbour list.
    calc64 = mace_mp(model=args.model, device=dev, default_dtype="float64")
    anchors = {}
    for tag, nx, ny, nz in plan:
        a = cell(nx, ny, nz)
        a.set_positions(a.get_positions() + rng.normal(0.0, args.jitter, (len(a), 3)))
        a.calc = calc64
        anchors[tag] = (a.get_positions().copy(), float(a.get_potential_energy()),
                        np.array(a.get_forces(), dtype=np.float64))
    if args.anchors_out:
        np.savez(args.anchors_out, **{f"{t}{k}": v for t, (p, e, f) in anchors.items()
                                      for k, v in (("_pos", p), ("_e", np.float64(e)), ("_f", f))})
    del calc64

    def versions():
        from importlib.metadata import PackageNotFoundError, version
        out = {}
        for pkg in ("mace-torch", "e3nn", "cuequivariance-torch", "cuequivariance-ops-torch-cu12", "openequivariance"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = None
        return out
    meta = {"gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "model": f"mace-mp {args.model}",
            "mace": mace.__version__, "packages": versions(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "precision": args.precision, "dev_run": args.dev,
            "parity_anchor": "mace-torch float64, stock calculator, fresh graph"}
    print(meta, flush=True)
    torch._dynamo.config.cache_size_limit = 64  # one static shape per size and pack, per compiled model
    torch._dynamo.config.accumulated_cache_size_limit = 4096

    calcs = {}

    def calculator(v):
        """The package's own constructor for this backend and compile mode, built once."""
        b, m = v.split("_", 1)
        key = (b, MODES[m])
        if key not in calcs:
            try:
                calcs[key] = mace_mp(model=args.model, device=dev, default_dtype=dtype, compute_stress=False,
                                     compile_mode=MODES[m], **BACKENDS[b])
            except Exception as exc:  # noqa: BLE001 -- e.g. a backend that is not installed
                calcs[key] = exc
        if isinstance(calcs[key], Exception):
            raise calcs[key]
        return calcs[key]

    def frozen_step(v, atoms0, copies):
        """(upload(positions), run() -> (energy of cell 0, forces of cell 0)) on a frozen graph of
        `copies` identical cells. The model is called exactly as MACECalculator.calculate calls it."""
        calc = calculator(v)
        model, N = calc.models[0], len(atoms0)
        with torch_tools.default_dtype(dtype):
            config = mace_data.config_from_atoms(atoms0, head_name=calc.head)
            one = mace_data.AtomicData.from_config(config, z_table=calc.z_table, cutoff=calc.r_max,
                                                   heads=calc.available_heads)
            loader = torch_geometric.dataloader.DataLoader([one] * copies, batch_size=copies, shuffle=False)
            batch = calc._clone_batch(next(iter(loader)).to(dev)).to_dict()  # noqa: SLF001 -- marks grads for compile
        pos = batch["positions"]
        host = torch.empty(N, 3, dtype=pos.dtype)
        host = host.pin_memory() if dev == "cuda" else host
        kwargs = {"compute_stress": False, "training": calc.use_compile and not calc._enable_oeq}  # noqa: SLF001

        def upload(p):
            host.copy_(torch.as_tensor(p, dtype=pos.dtype))
            with torch.no_grad():
                pos.view(copies, N, 3).copy_(host.to(dev, non_blocking=True))

        def run():
            with torch_tools.default_dtype(dtype):
                return model(batch, **kwargs)
        if v.endswith("_cudagraph"):
            run = capture(run, "mace")

        def result():
            out = run()
            return float(out["energy"][0].detach().double()), out["forces"][:N].detach().double().cpu().numpy()
        return upload, run, result, int(batch["edge_index"].shape[1]) // copies

    records = []
    for tag, nx, ny, nz in plan:
        atoms0 = cell(nx, ny, nz)
        N, pos0 = len(atoms0), atoms0.get_positions()
        p_ref, e_ref, f_ref = anchors[tag]
        rec = {"tag": tag, "N": N, "variants": {}, "packed": {}}
        print(f"\n=== {tag}  N={N} ===", flush=True)

        def show(name, r):
            if "error" in r:
                print(f"  {name:26s} {r['error']}", flush=True)
                return
            par = r["parity"]
            print(f"  {name:26s} {r['step_ms_median']:9.3f} ms  {r['atoms_per_s']:10.0f} atoms/s  "
                  f"sm={r['sm_clock_mhz_during']} MHz  dE/atom={par['dE_per_atom_eV']:.1e} eV  "
                  f"MAE|dF|={par['mae_dF_eV_per_A']:.1e}  max|dF|={par['max_abs_dF_eV_per_A']:.1e} eV/A", flush=True)

        def timed(step, par, atoms, warm, n):
            r = time_steps(step, warm, n)
            return {**r, "atoms_per_s": atoms * r["steps_per_s"], "parity": par}

        if "stock" in variants:
            try:
                st = atoms0.copy()
                st.calc = calcs.setdefault("stock", mace_mp(model=args.model, device=dev, default_dtype=dtype))
                st.set_positions(p_ref)
                par = grade(float(st.get_potential_energy()), np.asarray(st.get_forces()), e_ref, f_ref)

                def stock_step():
                    st.set_positions(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                    st.get_potential_energy(); st.get_forces()
                rec["variants"]["stock"] = timed(stock_step, par, N, args.warmup, args.steps)
            except Exception as exc:  # noqa: BLE001
                rec["variants"]["stock"] = failure(exc)
            show("stock", rec["variants"]["stock"])

        for copies in [1] + [b for b in packs if b * N <= args.pack_max_atoms]:
            for v in [v for v in (variants if copies == 1 else pack_variants) if v != "stock"]:
                name = v if copies == 1 else f"{v} x{copies}"
                try:
                    upload, run, result, edges = frozen_step(v, atoms0, copies)
                    rec["edges"] = edges
                    upload(p_ref)
                    par = grade(*result(), e_ref, f_ref)

                    def step():
                        upload(pos0 + rng.normal(0.0, args.jitter, pos0.shape))
                        run()
                    full = copies == 1
                    r = timed(step, par, copies * N, args.warmup if full else 5, args.steps if full else 30)
                except Exception as exc:  # noqa: BLE001
                    r = failure(exc)
                    upload = run = result = step = None
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                if copies == 1:
                    rec["variants"][v] = r
                else:
                    rec["packed"].setdefault(v, {})[str(copies)] = r
                show(name, r)
        records.append(rec)
        dump(args.out, meta, records, __doc__, args.warmup, args.steps)


if __name__ == "__main__":
    main()
