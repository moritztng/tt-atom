"""NVIDIA leg of the fair Orb-v3 perf-per-dollar comparison (one H100/H200-class GPU).

Governing principle (Moritz, 2026-07-14): compare the best *easy, near-out-of-box* path
a normal user would run on each side -- NOT a hand-tuned lab setup. On the GPU that means
stock `orb_models.forcefield.calculator.ORBCalculator`, the package a user gets from
`pip install orb-models`. A survey of the orb-models repo/PYPI (2026-07-14) found that the
genuine current user path already bakes in the two things the earlier comparison accused it
of lacking:

  * **compiled by default** -- Orb-v3 models call `model.compile()` on the GNS backbone
    (~1.7x at 10k atoms) since orb-models v0.5.0.
  * **fast GPU neighbour search by default** -- `knn_alchemi` (NVIDIA ALCHEMI Toolkit-Ops,
    GPU-accelerated kNN) is the default `edge_method` since v0.5.6.

So `naive_rebuild` below -- stock `ORBCalculator` on the latest `pip install orb-models`,
neighbour list rebuilt every call (the package default for a changing geometry), fp32
(orb_models default; bf16 is NOT a documented flag -- see orb-models issue #71, it needs
manual Triton rewrites) -- IS the fair, best-easy, out-of-box GPU number. No CUDA graphs,
no custom kernels, no re-architected inference loop on the comparison point.

Two extra variants are run *only for transparency* and are clearly labelled as NOT the
user path -- hand-tuned upper bounds, never the headline:

  frozen_eager      -- neighbour list frozen + `regressor.predict` called directly (bypasses
                        ORBCalculator): isolates the per-step neighbour-rebuild cost.
  frozen_cudagraph  -- above + `torch.compile(mode="reduce-overhead")` (CUDA graphs): the
                        GPU's hand-tuned ceiling, what you'd get only by writing custom code.

Same model (`orb-v3-conservative-inf-omat`), same periodic Si diamond supercell, same
quantity (one energy + conservative-force eval per step), same size sweep as the TT leg
(`benchmarks/bench_orb_perf_dollar_tt.py`). Load + first-call warmup excluded; positions
jittered each step so the path is exercised like a real MD loop. The stock ORBCalculator
path is its own correctness reference; the frozen path is parity-checked against it before
any frozen timing is trusted.

    python benchmarks/orb_gpu_bench_fair.py --out orb_perf_dollar_gpu.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from ase.build import bulk


SIZES = [
    ("3x3x3", 3, 3, 3),   #   216 atoms
    ("4x4x4", 4, 4, 4),   #   512 atoms
    ("5x5x5", 5, 5, 5),   #  1000 atoms
    ("6x6x7", 6, 6, 7),   #  2016 atoms (~2000)
]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _median_step_ms(step_fn, warm, n, sync=torch.cuda.synchronize):
    for _ in range(warm):
        step_fn()
    sync()
    ts = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        step_fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts, float(np.median(ts))


def _model_is_compiled(model):
    """True if the GNS backbone has a compiled call impl (orb_models compile default)."""
    backbone = getattr(model, "model", model)
    return any(hasattr(backbone, a) for a in ("_compiled_call_impl", "_orig_forward"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="orb-v3-conservative-inf-omat",
                    choices=["orb-v3-conservative-inf-omat", "orb-v3-direct-20-omat"])
    ap.add_argument("--element", default="Si")
    ap.add_argument("--a", type=float, default=5.43)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="orb_perf_dollar_gpu.json")
    ap.add_argument("--sizes", default=",".join(s[0] for s in SIZES))
    ap.add_argument("--variants", default="naive_rebuild,frozen_eager,frozen_cudagraph")
    ap.add_argument("--bf16", action="store_true", help="run model in bf16 (else fp32 default)")
    ap.add_argument("--parity-tol", type=float, default=2e-3,
                    help="max |dE|/atom and rel force error vs ORBCalculator for the frozen path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    import orb_models
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    if args.ckpt == "orb-v3-conservative-inf-omat":
        model = pretrained.orb_v3_conservative_inf_omat(device=device)
    else:
        model = pretrained.orb_v3_direct_20_omat(device=device)
    model = model.eval()
    if args.bf16:
        model = model.to(torch.bfloat16)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    calc = ORBCalculator(model, device=device)
    regressor = calc.model
    edge_method_used = getattr(calc, "edge_method", None) or "<orb_models default>"
    compiled_default = _model_is_compiled(regressor)

    wanted = [t for t in args.sizes.split(",") if t.strip()]
    variants = [v for v in args.variants.split(",") if v.strip()]
    plan = [s for s in SIZES if s[0] in wanted]

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  orb_models={orb_models.__version__}  torch={torch.__version__}  "
          f"cuda={torch.version.cuda}  dtype={dtype}  edge_method={edge_method_used}  "
          f"compiled_default={compiled_default}", flush=True)

    records = []
    for tag, nx, ny, nz in plan:
        print(f"\n=== {tag}  ({nx}x{ny}x{nz}) ===", flush=True)
        atoms0 = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
        N = len(atoms0)
        rng = np.random.default_rng(args.seed)

        rec = {"tag": tag, "nx": nx, "ny": ny, "nz": nz, "N": N,
               "precision": "bf16" if args.bf16 else "fp32 (orb_models default)",
               "variants": {}}

        # ---- variant: naive_rebuild (stock ORBCalculator, rebuild every call) ----
        # This is the headline / fair out-of-box GPU number. Run first, unconditionally.
        if "naive_rebuild" in variants:
            na_atoms = atoms0.copy(); na_atoms.calc = calc
            _ = na_atoms.get_potential_energy(); _ = na_atoms.get_forces()  # prime + first compile
            n_edge = None
            def na_step():
                p = na_atoms.get_positions() + rng.normal(0.0, 0.01, na_atoms.positions.shape)
                na_atoms.set_positions(p)
                _ = na_atoms.get_potential_energy()
                _ = na_atoms.get_forces()
            try:
                ts, med = _median_step_ms(na_step, args.warmup, args.steps)
                # edge count from one AtomGraphs build (for disclosure, matches TT leg's edges)
                try:
                    from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
                    ag = ase_atoms_to_atom_graphs(
                        atoms0, system_config=calc.system_config,
                        max_num_neighbors=calc.max_num_neighbors,
                        edge_method=calc.edge_method, half_supercell=calc.half_supercell,
                        device=calc.device)
                    n_edge = int(ag.n_edge.item())
                except Exception:
                    n_edge = None
                rec["edges"] = n_edge
                rec["variants"]["naive_rebuild"] = {
                    "path": "stock ORBCalculator (pip install orb-models), neighbour list "
                            "rebuilt every call (package default), compiled backbone + "
                            f"edge_method={edge_method_used}",
                    "is_user_out_of_box_path": True,
                    "step_ms_raw": [round(x, 4) for x in ts],
                    "step_ms_median": med, "steps_per_s": 1000.0 / med,
                }
                print(f"  naive_rebuild     : median={med:.3f} ms  => {1000.0/med:.2f} steps/s"
                      f"  (edges={n_edge})", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["variants"]["naive_rebuild"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  naive_rebuild     : ERROR {exc}", flush=True)

        # ---- frozen variants (transparency only; NOT the user path) ----
        # Parity-checked against the stock ORBCalculator on a near-equilibrium frame.
        if any(v in variants for v in ("frozen_eager", "frozen_cudagraph")):
            try:
                from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
                ref_atoms = atoms0.copy(); ref_atoms.calc = calc
                ref_atoms.set_positions(ref_atoms.get_positions()
                                        + rng.normal(0.0, 0.01, ref_atoms.positions.shape))
                e_ref = float(ref_atoms.get_potential_energy())
                f_ref = torch.as_tensor(ref_atoms.get_forces(), device=device, dtype=torch.float32)

                frozen_batch = ase_atoms_to_atom_graphs(
                    atoms0, system_config=calc.system_config,
                    max_num_neighbors=calc.max_num_neighbors, edge_method=calc.edge_method,
                    half_supercell=calc.half_supercell, device=calc.device)
                frozen_batch = frozen_batch.to(calc.device)
                if n_edge is None:
                    n_edge = int(frozen_batch.n_edge.item())
                    rec["edges"] = n_edge
                pos_buf = torch.zeros(N, 3, device=device, dtype=torch.float32)
                pos_buf.copy_(torch.as_tensor(ref_atoms.get_positions(), device=device,
                                              dtype=torch.float32))
                frozen_batch.node_features["positions"] = pos_buf

                def _frozen_step():
                    pos_buf.requires_grad_(True)
                    frozen_batch.node_features["positions"] = pos_buf
                    return regressor.predict(frozen_batch)

                with torch.enable_grad():
                    out0 = _frozen_step()
                e_frz = float(out0[regressor.energy_name].item())
                f_frz = out0[regressor.grad_forces_name].to(torch.float32)
                dE_per_atom = abs(e_frz - e_ref) / N
                f_pcc = float(torch.corrcoef(torch.stack([f_ref.flatten(), f_frz.flatten()]))[0, 1])
                f_mae = float((f_ref - f_frz).abs().max())
                parity_ok = (dE_per_atom < args.parity_tol) and (f_pcc > 0.9999)
                rec["parity"] = {
                    "energy_ref_eV": e_ref, "energy_frozen_eV": e_frz,
                    "abs_dE_per_atom": dE_per_atom,
                    "force_pcc_vs_ref": f_pcc, "force_max_abs_err_eV_per_A": f_mae,
                    "ok": parity_ok,
                }
                print(f"  frozen parity: dE/atom={dE_per_atom:.2e}  PCC={f_pcc:.6f}  "
                      f"max|dF|={f_mae:.2e}  -> {'OK' if parity_ok else 'FAIL'}", flush=True)
                if not parity_ok:
                    raise RuntimeError("frozen path parity check failed vs stock ORBCalculator")

                if "frozen_eager" in variants:
                    def fe_step():
                        p = pos_buf + torch.from_numpy(
                            rng.normal(0.0, 0.01, pos_buf.shape).astype(np.float32)).to(device)
                        pos_buf.copy_(p)
                        return _frozen_step()
                    try:
                        with torch.enable_grad():
                            ts, med = _median_step_ms(fe_step, args.warmup, args.steps)
                        rec["variants"]["frozen_eager"] = {
                            "path": "TRANSPARENCY ONLY (not user path): frozen neighbours, "
                                    "regressor.predict called directly (bypasses ORBCalculator), "
                                    "eager (no graph)",
                            "is_user_out_of_box_path": False,
                            "step_ms_raw": [round(x, 4) for x in ts],
                            "step_ms_median": med, "steps_per_s": 1000.0 / med,
                        }
                        print(f"  frozen_eager      : median={med:.3f} ms  => "
                              f"{1000.0/med:.2f} steps/s", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        rec["variants"]["frozen_eager"] = {"error": f"{type(exc).__name__}: {exc}"}
                        print(f"  frozen_eager      : ERROR {exc}", flush=True)

                if "frozen_cudagraph" in variants:
                    try:
                        compiled = torch.compile(_frozen_step, mode="reduce-overhead")
                        def fc_step():
                            p = pos_buf + torch.from_numpy(
                                rng.normal(0.0, 0.01, pos_buf.shape).astype(np.float32)).to(device)
                            pos_buf.copy_(p)
                            with torch.enable_grad():
                                return compiled()
                        with torch.enable_grad():
                            ts, med = _median_step_ms(fc_step, args.warmup, args.steps)
                        rec["variants"]["frozen_cudagraph"] = {
                            "path": "TRANSPARENCY ONLY (not user path): frozen neighbours + "
                                    "regressor.predict + torch.compile(reduce-overhead)=CUDA graphs",
                            "is_user_out_of_box_path": False,
                            "step_ms_raw": [round(x, 4) for x in ts],
                            "step_ms_median": med, "steps_per_s": 1000.0 / med,
                        }
                        print(f"  frozen_cudagraph  : median={med:.3f} ms  => "
                              f"{1000.0/med:.2f} steps/s", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        rec["variants"]["frozen_cudagraph"] = {"error": f"{type(exc).__name__}: {exc}"}
                        print(f"  frozen_cudagraph  : ERROR {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["frozen_block_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  frozen variants   : BLOCKED ({exc})", flush=True)

        records.append(rec)
        _dump(records, args, gpu_name, orb_models.__version__, edge_method_used, compiled_default)

    _dump(records, args, gpu_name, orb_models.__version__, edge_method_used, compiled_default)
    print(f"\nwrote {args.out}: {len(records)} sizes", flush=True)


def _dump(records, args, gpu_name, orb_models_version, edge_method_used, compiled_default):
    out = {
        "platform": f"NVIDIA {gpu_name} (one GPU, rented on vast.ai)",
        "model": args.ckpt,
        "system": "periodic Si diamond supercell, monatomic",
        "quantity": "one MD step = energy + conservative forces (F=-dE/dpos via torch.autograd). "
                    "Note: orb_models conservative predict also computes stress+rotation, which "
                    "the TT leg does not -- the GPU does slightly more work per step, understating "
                    "its relative speed.",
        "fairness_framing": "naive_rebuild = stock ORBCalculator on `pip install orb-models` "
                            "(compiled-by-default backbone + knn_alchemi default GPU kNN, neighbour "
                            "rebuilt every call, fp32): the genuine out-of-box user path and the "
                            "headline GPU number. frozen_eager/frozen_cudagraph are hand-tuned "
                            "transparency upper bounds (bypass ORBCalculator, freeze neighbours, "
                            "add CUDA graphs) -- NOT the user path, never the headline.",
        "neighbour_policy": "rebuilt every call for naive_rebuild (orb_models default); frozen for "
                            "the transparency variants",
        "execution_model": "compiled backbone (orb_models default) for naive_rebuild; "
                           "torch.compile(reduce-overhead)=CUDA graphs for frozen_cudagraph; "
                           "eager for frozen_eager",
        "load_and_first_compile_excluded": True,
        "positions_jittered_each_step": True,
        "precision": "bf16" if args.bf16 else "fp32 (orb_models default)",
        "orb_models_version": orb_models_version,
        "edge_method_used": edge_method_used,
        "compiled_by_default": compiled_default,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
