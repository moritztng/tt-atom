"""The per-dollar bar the TT leg of the MACE comparison has to clear, from the GPU leg's results.

Merges the per-GPU, per-precision files written by `mace_gpu_bench_fair.py` into one JSON. For each
GPU, precision and cell size it keeps every variant, names the fastest single-cell variant and the
best packed one, and states the bar: the atoms/s one TT chip needs for the same throughput per
dollar, GPU atoms/s x (TT chip price / price of one GPU in its 8-GPU system). A GPU given without
a price (a dev card: `--gpu RTX3090=:glob`) gets its rates and no bar.

    python benchmarks/mace_gpu_bar.py --gpu H200=51250:h200_*.json --gpu B200=68750:b200_*.json \
        --tt-price 1399 --out benchmarks/mace_perf_dollar_gpu.json
"""
from __future__ import annotations

import argparse
import glob
import json


def timed(runs):
    return {k: r for k, r in runs.items() if "step_ms_median" in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="append", required=True, help="NAME=PRICE_USD:GLOB, once per GPU")
    ap.add_argument("--tt-price", type=float, default=1399.0, help="USD for one TT chip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {"tt_chip_price_usd": args.tt_price, "gpus": {}}
    for spec in args.gpu:
        name, rest = spec.split("=", 1)
        price, pattern = rest.split(":", 1)
        share = args.tt_price / float(price) if price else None
        g = out["gpus"][name] = {"price_usd": float(price) if price else None, "runs": [], "sizes": {}}
        for path in sorted(glob.glob(pattern)):
            d = json.load(open(path))
            g["runs"].append({k: d[k] for k in d if k not in ("records", "protocol")})
            for rec in d["records"]:
                runs = dict(rec["variants"])
                runs.update({f"{v} x{b}": r for v, by in rec["packed"].items() for b, r in by.items()})
                single = timed({k: r for k, r in rec["variants"].items() if k != "stock"})
                best = max(timed({k: r for k, r in runs.items() if k != "stock"}).items(),
                           key=lambda kv: kv[1]["atoms_per_s"], default=None)
                fast = min(single.items(), key=lambda kv: kv[1]["step_ms_median"], default=None)
                row = {"N": rec["N"], "edges": rec.get("edges"),
                       "variants": {k: ({x: r[x] for x in ("step_ms_median", "atoms_per_s", "sm_clock_mhz_during",
                                                            "sm_clock_mhz_min_max", "peak_mem_gb", "parity") if x in r}
                                        if "step_ms_median" in r else {"error": r["error"]})
                                    for k, r in runs.items()}}
                if fast:
                    row["fastest_single"] = {"variant": fast[0], "step_ms": fast[1]["step_ms_median"],
                                             "atoms_per_s": fast[1]["atoms_per_s"]}
                    row["best_throughput"] = {"variant": best[0], "atoms_per_s": best[1]["atoms_per_s"]}
                    bar = ""
                    if share:
                        row["tt_chip_bar_atoms_per_s_for_1x_per_dollar"] = best[1]["atoms_per_s"] * share
                        row["tt_chip_bar_step_ms_for_1x_per_dollar"] = 1e3 * rec["N"] / (best[1]["atoms_per_s"] * share)
                        bar = f"  TT 1x/$ bar {row['tt_chip_bar_step_ms_for_1x_per_dollar']:8.1f} ms/step"
                    print(f"{name} {d['precision']:15s} N={rec['N']:5d}  single {fast[1]['step_ms_median']:8.3f} ms "
                          f"({fast[0]})  best {best[1]['atoms_per_s']:9.0f} atoms/s ({best[0]}){bar}", flush=True)
                g["sizes"].setdefault(rec["tag"], {})[d["precision"]] = row
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
