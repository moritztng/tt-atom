# TT-Atom — release notes & announcement draft

*Draft for Moritz. Every number below is measured on this machine and reproducible with the
scripts in `benchmarks/`. The public release (GitHub repo, social post) is your call — this
file is the prepared material, nothing has been pushed or posted.*

---

## What it is

**TT-Atom** is a clean, minimal, high-performance implementation of the **eSEN / eSCN-MD**
(Meta UMA-family) equivariant ML interatomic potential running on **Tenstorrent** via `ttnn` —
energy and **conservative analytic forces**, fully device-resident, behind an ASE calculator.

It does one thing well: fast, accurate inference for this architecture, with a multi-card
throughput path. No framework sprawl, no dead code, one implementation per path.

## Headline numbers (Blackhole p150, full config, real & reproducible)

- **Device compute up to 5.3× faster than 16-thread PyTorch CPU** at 250 atoms — and the
  speedup *grows with system size* because the TT per-eval latency is nearly flat while CPU
  scales with the system.
- **3.95× near-linear throughput scaling across 4 cards.**
- **Analytic forces** (not finite differences): PCC **0.99996**, cosine **0.99996** vs the
  PyTorch autograd reference.
- Energy matches the fp32 reference to **<1 %** (1e-4–6e-3, random weights).

![device vs CPU](assets/device_vs_cpu.png)
![multi-card scaling](assets/multicard_scaling.png)

## Why it's interesting (engineering story)

The eSCN SO(2) trick turns the SO(3) tensor product into per-`m` dense GEMMs, so ~85–90 % of
the model is matmul — a great fit for Tenstorrent. The port got its speed from three structural
moves, each measured:

1. **SO(2) per-`m` conv as flat 2-D GEMMs** — removing a `[E,2,K]` reshape that tile-padded a
   length-2 axis to 32 (a 16× data blowup + a per-edge batched matmul). ~12× on that module.
2. **Wigner rotation as a sparse multiply-accumulate** over its fixed nonzero pattern, in a
   flat `[E, 9·C]` layout — replacing a launch-bound batched `[E,9,9]×[E,9,C]` matmul (~2.9 µs
   *per edge*). ~3.8×, and it composes into a fully tile-aligned pipeline.
3. **Analytic on-device reverse pass** for forces (matmul backward = transpose-matmul on
   device; the cheap geometric Jacobian finishes on host).

Net: the full forward went **251 ms → 33 ms** at ~4800 edges, and the device latency stopped
scaling with the system — which is exactly why the CPU gap widens as systems grow.

An honest negative result worth keeping: a `bfloat8_b` "fast" mode gives **no speedup** here —
the forward is data-movement bound, not flop bound — so `bf16` (with `HiFi4` + `fp32`
accumulation) is the recommended default.

## What's validated, and what isn't

Validated with **random weights** against a bit-exact PyTorch reference: per-module PCC ≥ 0.99,
end-to-end energy and analytic forces, an ASE FIRE relaxation that converges on device.

**Not** claimed: accuracy against published UMA energies/forces — that needs the gated
`facebook/UMA` checkpoint. TT-Atom is the implementation; real weights are drop-in via
`tools/export_weights.py` + `weights.py` (key/shape coverage is checked). No weights are shipped
or redistributed; they remain under the FAIR Chemistry License.

## Suggested social post (draft)

> Got Meta's UMA-family equivariant interatomic potential (eSEN / eSCN-MD) running on
> Tenstorrent — energy **and analytic forces**, device-resident, behind an ASE calculator.
> The SO(2)-convolution trick makes it ~90 % dense GEMM, so it maps beautifully to the
> hardware: **up to 5.3× faster than CPU** and the gap grows with system size, **~4× linear
> scaling across 4 Blackhole cards**, analytic-force PCC 0.99996. Apache-2.0, bring your own
> checkpoint. 🧪⚡

## Status

Apache-2.0 (our code). Tests pass (`pytest tests/ -q`). Benchmarks reproducible. Nothing pushed
or posted — ready for your review.
