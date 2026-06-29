"""Generate the release perf charts from measured benchmark JSON (no hardcoded numbers).

Run after the benchmarks:
  ~/.ttatom_run/env/bin/python benchmarks/plot.py
Reads benchmarks/results/{throughput,multicard}.json -> assets/*.png
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "assets"
TT = "#7C3AED"
CPU = "#9CA3AF"
GRN = "#10B981"


def plot_device_vs_cpu():
    d = json.loads((RESULTS / "throughput.json").read_text())
    rows = d["rows"]
    N = [r["natoms"] for r in rows]
    dev = [r["dev_ms"] for r in rows]
    cpu = [r["cpu_ms"] for r in rows]
    spd = [r["speedup_dev"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(N, cpu, "o-", color=CPU, label="PyTorch CPU", lw=2)
    ax1.plot(N, dev, "o-", color=TT, label="TT-Atom (1x Blackhole p150)", lw=2)
    ax1.set_xlabel("atoms"); ax1.set_ylabel("ms / energy eval (warm)")
    ax1.set_title("Device compute vs CPU"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(N, spd, "o-", color=GRN, lw=2)
    ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax2.set_xlabel("atoms"); ax2.set_ylabel("device speedup vs CPU (x)")
    ax2.set_title("Speedup grows with system size")
    for x, y in zip(N, spd):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    ax2.grid(alpha=0.3)
    fig.suptitle("TT-Atom — eSEN / eSCN-MD inference on Tenstorrent (full config, random weights)",
                 fontweight="bold")
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "device_vs_cpu.png", dpi=130)
    print("wrote", ASSETS / "device_vs_cpu.png")


def plot_multicard():
    p = RESULTS / "multicard.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    rows = d["rows"]
    cards = [r["cards"] for r in rows]
    medges = [r["medges_per_s"] for r in rows]
    ideal = [medges[0] * c for c in cards]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(cards, ideal, "--", color=CPU, label="ideal linear", lw=1.5)
    ax.plot(cards, medges, "o-", color=TT, label="measured", lw=2)
    for c, m in zip(cards, medges):
        ax.annotate(f"{m:.3f}", (c, m), textcoords="offset points", xytext=(6, -4))
    ax.set_xlabel("cards"); ax.set_ylabel("aggregate Medges/s")
    ax.set_title(f"Multi-card throughput ({d['natoms']}-atom systems)\n"
                 f"{rows[-1]['scaling_vs_1card']:.2f}x on {cards[-1]} cards")
    ax.set_xticks(cards); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "multicard_scaling.png", dpi=130)
    print("wrote", ASSETS / "multicard_scaling.png")


if __name__ == "__main__":
    plot_device_vs_cpu()
    plot_multicard()
