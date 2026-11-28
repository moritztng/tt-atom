"""Fair Orb-v3 comparison table from the two legs' JSON (one protocol, see orb_gpu_bench_fair.py).

For each size and each TT precision mode, the GPU headline is the fastest frozen variant, at any
precision run, whose force MAE and max |dF| against the shared float64 anchor are no larger than the
TT leg's. Per-dollar uses the purchase price of the hardware that ran each leg.

    python benchmarks/orb_fair_compare.py --tt tt_default.json \
        --gpu h200_high.json h200_highest.json --tt-price 1399 --gpu-price 51250
"""
from __future__ import annotations

import argparse
import json

FROZEN = ("frozen_compiled", "frozen_cudagraph", "frozen_compiled_cudagraph", "frozen_reduce_overhead")


def no_worse(g, t):
    return (g["mae_dF_eV_per_A"] <= t["mae_dF_eV_per_A"]
            and g["max_abs_dF_eV_per_A"] <= t["max_abs_dF_eV_per_A"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt", nargs="+", required=True)
    ap.add_argument("--gpu", nargs="+", required=True)
    ap.add_argument("--tt-price", type=float, required=True, help="USD for the TT hardware that ran")
    ap.add_argument("--gpu-price", type=float, required=True, help="USD for one GPU (system share)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpu = []  # (tag, label, record)
    for path in args.gpu:
        d = json.load(open(path))
        for rec in d["records"]:
            for v, r in rec["variants"].items():
                if v in FROZEN and "step_ms_median" in r:
                    gpu.append((rec["tag"], f"{d['gpu_name']} {d['precision']} {v}", r))

    rows = []
    for path in args.tt:
        d = json.load(open(path))
        for t in d["records"]:
            acc = t.get("accuracy_vs_float64")
            if "step_ms_median" not in t or acc is None:
                continue
            ok = [(lab, r) for tag, lab, r in gpu if tag == t["tag"] and no_worse(r["parity"], acc)]
            if not ok:  # a size where no GPU variant is as accurate is a result, not a gap
                print(f"{t['N']:5d} atoms  TT {t['step_ms_median']:8.2f} ms: no GPU variant as accurate")
                rows.append({"tag": t["tag"], "N": t["N"], "tt_ms": t["step_ms_median"], "tt_acc": acc, "gpu": None})
                continue
            lab, g = min(ok, key=lambda x: x[1]["step_ms_median"])
            speed = g["step_ms_median"] / t["step_ms_median"]  # TT throughput relative to the GPU
            rows.append({
                "tag": t["tag"], "N": t["N"], "tt_precision": t["precision"],
                "tt_ms": t["step_ms_median"], "tt_aiclk": t["aiclk_mhz_during_by_node"],
                "tt_acc": acc, "gpu": lab, "gpu_ms": g["step_ms_median"],
                "gpu_sm_mhz": g["sm_clock_mhz_during"], "gpu_acc": g["parity"],
                "tt_over_gpu_throughput": speed,
                "tt_over_gpu_per_dollar": speed * args.gpu_price / args.tt_price,
            })
    for r in [r for r in rows if r["gpu"]]:
        print(f"{r['N']:5d} atoms  TT {r['tt_ms']:8.2f} ms (MAE {r['tt_acc']['mae_dF_eV_per_A']:.1e})  "
              f"GPU {r['gpu_ms']:7.2f} ms {r['gpu']} (MAE {r['gpu_acc']['mae_dF_eV_per_A']:.1e})  "
              f"throughput x{r['tt_over_gpu_throughput']:.3f}  per-$ x{r['tt_over_gpu_per_dollar']:.2f}")
    if args.out:
        json.dump({"tt_price_usd": args.tt_price, "gpu_price_usd": args.gpu_price, "rows": rows},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
