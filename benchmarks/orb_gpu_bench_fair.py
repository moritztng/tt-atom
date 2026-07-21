"""NVIDIA leg of the fair Orb-v3 perf-per-dollar comparison (one H100/H200-class GPU).

Governing principle (Moritz, 2026-07-14): compare the best *easy, near-out-of-box* path
a normal user would run on each side -- NOT a hand-tuned lab setup. On the GPU that means
stock `orb_models` via `ORBCalculator`, the package a user gets from `pip install
orb-models`. A survey of the orb-models repo/PYPI (2026-07-14) found the genuine current
user path already bakes in the two things the earlier comparison accused it of lacking:

  * **compiled by default** -- Orb-v3 models compile the GNS backbone (~1.7x at 10k
    atoms) since orb-models v0.5.0.
  * **fast GPU neighbour search by default** -- `knn_alchemi` (NVIDIA ALCHEMI Toolkit-Ops,
    GPU-accelerated kNN) is the default `edge_method` since v0.5.6.

So `naive_rebuild` below -- stock `ORBCalculator` on the latest `pip install orb-models`
(v0.7.0 here), neighbour list rebuilt every call (the package default for a changing
geometry), fp32 (orb_models default; bf16 is NOT a documented flag -- orb-models issue
#71 needs manual Triton rewrites) -- IS the fair, best-easy, out-of-box GPU number. No
CUDA graphs, no custom kernels, no re-architected inference loop on the comparison point.

One extra variant runs *only for transparency* and is clearly labelled NOT the user path:

  frozen_eager      -- neighbour list frozen + `regressor.predict` called directly (bypasses
                        ORBCalculator): isolates the per-step neighbour-rebuild cost, i.e.
                        "what would the GPU do if orb_models reused an unchanged graph like the
                        p150 path". A hand-tuned upper bound, never the headline.

`frozen_cudagraph` (torch.compile reduce-overhead = CUDA graphs) is intentionally NOT run
here: on orb_models' conservative model it recompiles every step (~539 ms/step measured on
v0.5.5, a recompile pathology), exactly the behaviour orb-models commit f3d7837 documents
("compiling the full conservative regressor pulls the energy autograd backward into the
traced graph and dynamo fragments badly"). There is no easy, officially-supported CUDA-
graph path for this model -- which is itself the point: the stock number is the user
experience, and matching the p150's trace/replay would require hand-tuning the GPU.

Same model (`orb-v3-conservative-inf-omat`), same periodic Si diamond supercell, same
quantity (one energy + conservative-force eval per step), same size sweep as the TT leg
(`benchmarks/bench_orb_perf_dollar_tt.py`). Load + first-call warmup excluded; positions
jittered each step so the path is exercised like a real MD loop. Stock ORBCalculator is
its own correctness reference; the frozen path is parity-checked against it before any
frozen timing is trusted.

Tested against orb-models v0.7.0 (ORBCalculator now lives in
`orb_models.forcefield.inference.calculator` and takes an `atoms_adapter`).

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
    ap.add_argument("--variants", default="naive_rebuild,frozen_eager")
    ap.add_argument("--bf16", action="store_true", help="run model in bf16 (else fp32 default)")
    ap.add_argument("--parity-tol", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    import orb_models
    from orb_models.forcefield import pretrained
    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator  # v0.7.0+
    except ImportError:  # older layout (v0.5.x)
        from orb_models.forcefield.calculator import ORBCalculator

    loader = (pretrained.orb_v3_conservative_inf_omat if args.ckpt
              == "orb-v3-conservative-inf-omat" else pretrained.orb_v3_direct_20_omat)
    res = loader(device=device)
    if isinstance(res, tuple):  # v0.7.0+ returns (model, atoms_adapter)
        model, adapter = res
    else:  # v0.5.x returns just the model
        model, adapter = res, None
    model = model.eval()
    if args.bf16:
        model = model.to(torch.bfloat16)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if adapter is not None:
        calc = ORBCalculator(model, adapter, device=device)
    else:
        calc = ORBCalculator(model, device=device)
    regressor = calc.model
    edge_method_used = getattr(calc, "edge_method", None) or "knn_alchemi (orb_models default)"
    compiled_default = _model_is_compiled(regressor)

    wanted = [t for t in args.sizes.split(",") if t.strip()]
    variants = [v for v in args.variants.split(",") if v.strip()]
    plan = [s for s in SIZES if s[0] in wanted]

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}  orb_models={orb_models.__version__}  torch={torch.__version__}  "
          f"cuda={torch.version.cuda}  dtype={dtype}  edge_method={edge_method_used}  "
          f"compiled_default={compiled_default}", flush=True)

    def _ase_to_graphs(atoms):
        if adapter is not None:
            return adapter.from_ase_atoms(atoms)
        from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
        return ase_atoms_to_atom_graphs(
            atoms, system_config=calc.system_config, max_num_neighbors=calc.max_num_neighbors,
            edge_method=calc.edge_method, half_supercell=calc.half_supercell, device=calc.device)

    records = []
    for tag, nx, ny, nz in plan:
        print(f"\n=== {tag}  ({nx}x{ny}x{nz}) ===", flush=True)
        atoms0 = bulk(args.element, "diamond", a=args.a, cubic=True) * (nx, ny, nz)
        N = len(atoms0)
        rng = np.random.default_rng(args.seed)
        rec = {"tag": tag, "nx": nx, "ny": ny, "nz": nz, "N": N,
               "precision": "bf16" if args.bf16 else "fp32 (orb_models default)",
               "variants": {}}

        # ---- naive_rebuild: stock ORBCalculator (headline, runs first/unconditional) ----
        if "naive_rebuild" in variants:
            na_atoms = atoms0.copy(); na_atoms.calc = calc
            _ = na_atoms.get_potential_energy(); _ = na_atoms.get_forces()  # prime + first compile
            def na_step():
                p = na_atoms.get_positions() + rng.normal(0.0, 0.01, na_atoms.positions.shape)
                na_atoms.set_positions(p)
                _ = na_atoms.get_potential_energy()
                _ = na_atoms.get_forces()
            try:
                ts, med = _median_step_ms(na_step, args.warmup, args.steps)
                n_edge = None
                try:
                    n_edge = int(_ase_to_graphs(atoms0).n_edge.item())
                except Exception:
                    pass
                rec["edges"] = n_edge
                rec["variants"]["naive_rebuild"] = {
                    "path": "stock ORBCalculator (pip install orb-models), neighbour list "
                            "rebuilt every call (package default), compiled backbone + "
                            f"edge_method={edge_method_used}",
                    "is_user_out_of_box_path": True,
                    "step_ms_raw": [round(x, 4) for x in ts],
                    "step_ms_median": med, "steps_per_s": 1000.0 / med,
                }
                print(f"  naive_rebuild : median={med:.3f} ms  => {1000.0/med:.2f} steps/s"
                      f"  (edges={n_edge})", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["variants"]["naive_rebuild"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  naive_rebuild : ERROR {exc}", flush=True)

        # ---- frozen_eager: transparency only, parity-checked vs stock ----
        if "frozen_eager" in variants:
            try:
                ref_atoms = atoms0.copy(); ref_atoms.calc = calc
                ref_atoms.set_positions(ref_atoms.get_positions()
                                        + rng.normal(0.0, 0.01, ref_atoms.positions.shape))
                e_ref = float(ref_atoms.get_potential_energy())
                f_ref = torch.as_tensor(ref_atoms.get_forces(), device=device, dtype=torch.float32)

                frozen_batch = _ase_to_graphs(atoms0).to(device)
                n_edge = int(frozen_batch.n_edge.item())
                rec.setdefault("edges", n_edge)
                pos_buf = torch.zeros(N, 3, device=device, dtype=torch.float32)
                pos_buf.copy_(torch.as_tensor(ref_atoms.get_positions(), device=device,
                                              dtype=torch.float32))
                pos_buf.requires_grad_(True)
                frozen_batch.node_features["positions"] = pos_buf

                def _frozen_step():
                    frozen_batch.node_features["positions"] = pos_buf
                    return regressor.predict(frozen_batch)

                with torch.enable_grad():
                    out0 = _frozen_step()
                e_frz = float(out0[regressor.energy_name].item())
                f_frz = out0[regressor.grad_forces_name].to(torch.float32)
                dE = abs(e_frz - e_ref) / N
                f_pcc = float(torch.corrcoef(torch.stack([f_ref.flatten(), f_frz.flatten()]))[0, 1])
                f_mae = float((f_ref - f_frz).abs().max())
                ok = (dE < args.parity_tol) and (f_pcc > 0.9999)
                rec["parity"] = {"energy_ref_eV": e_ref, "energy_frozen_eV": e_frz,
                                 "abs_dE_per_atom": dE, "force_pcc_vs_ref": f_pcc,
                                 "force_max_abs_err_eV_per_A": f_mae, "ok": ok}
                print(f"  frozen parity: dE/atom={dE:.2e}  PCC={f_pcc:.6f}  "
                      f"max|dF|={f_mae:.2e}  -> {'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    raise RuntimeError("frozen path parity failed vs stock ORBCalculator")

                def fe_step():
                    p = pos_buf + torch.from_numpy(
                        rng.normal(0.0, 0.01, pos_buf.shape).astype(np.float32)).to(device)
                    with torch.no_grad():
                        pos_buf.copy_(p)
                    return _frozen_step()
                with torch.enable_grad():
                    ts, med = _median_step_ms(fe_step, args.warmup, args.steps)
                rec["variants"]["frozen_eager"] = {
                    "path": "TRANSPARENCY ONLY (not user path): frozen neighbours + "
                            "regressor.predict called directly (bypasses ORBCalculator), eager",
                    "is_user_out_of_box_path": False,
                    "step_ms_raw": [round(x, 4) for x in ts],
                    "step_ms_median": med, "steps_per_s": 1000.0 / med,
                }
                print(f"  frozen_eager  : median={med:.3f} ms  => {1000.0/med:.2f} steps/s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                rec["variants"]["frozen_eager"] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  frozen_eager  : ERROR {exc}", flush=True)

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
                            "headline GPU number. frozen_eager is a hand-tuned transparency upper "
                            "bound (bypasses ORBCalculator, freezes neighbours) -- NOT the user path. "
                            "frozen_cudagraph is intentionally omitted: it recompiles every step on "
                            "the conservative model (orb-models commit f3d7837), i.e. no easy CUDA-"
                            "graph path exists, so the stock number is the user experience.",
        "neighbour_policy": "rebuilt every call for naive_rebuild (orb_models default); frozen for "
                            "the transparency variant",
        "execution_model": "compiled backbone (orb_models default) for naive_rebuild; eager for "
                           "frozen_eager",
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
