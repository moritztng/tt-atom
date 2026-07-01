# TT-Atom — release notes & announcement draft

*Draft for Moritz. Every number below is measured on this machine and reproducible with the
scripts in `benchmarks/`. The public release (GitHub repo, social post) is your call — this
file is the prepared material, nothing has been pushed or posted.*

---

## What it is

**TT-Atom** is a clean, minimal, high-performance port of Meta's **UMA** (eSEN / eSCN-MD)
equivariant ML interatomic potential to **Tenstorrent** via `ttnn` — energy and **conservative
analytic forces**, fully device-resident, behind an ASE calculator that **mirrors fairchem's**.
Moving off fairchem is a one-line change. Molecules **and** periodic materials, validated against
the released `uma-s-1` checkpoint.

It does one thing well: fast, accurate inference for this architecture, with a device-resident
trace loop for MD/relaxation and a multi-card throughput path. No framework sprawl, no dead code.

## Headline numbers (Blackhole p150, real & reproducible)

- **Drop-in for fairchem:** `FAIRChemCalculator(...)` → `TTAtomCalculator(bundle, task_name=...)`,
  same ASE surface. One-command checkpoint converter + `tt-atom verify` device roundtrip.
- **Validated vs released uma-s-1 across three tasks / all graph regimes** (energy rel < 1e-3,
  force PCC > 0.99): omol ethanol (1.8e-7 / 0.99958), omat bulk Si (3.0e-4 / 0.99999), oc20
  Cu(100)+H slab (8.6e-5 / 1.00000). The periodic neighbour list reproduces fairchem's
  `radius_graph_pbc` edge-for-edge.
- **Trace-captured MD/relaxation loop: 2.33× wall-clock** (FIRE, real uma-s-1) — identical
  trajectory, bit-for-bit the same analytic forces.
- **Device compute up to 5.3× faster than 16-thread PyTorch CPU** at 250 atoms, growing with
  system size (device latency is nearly flat).
- **3.95× near-linear throughput scaling across 4 cards** (validated on qb1).

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

Validated with **real `facebook/UMA` uma-s-1 weights** against the official fairchem reference on
a single p150 (reproducible via `tests/test_realweight.py` + `tests/test_periodic.py`):

| task | system | graph | energy rel err | force PCC | MAE (eV/Å) |
|---|---|---|---:|---:|---:|
| omol | ethanol | aperiodic | 1.8e-7 | 0.99958 | 3.4e-3 |
| omat | bulk Si | periodic [T,T,T] | 3.0e-4 | 0.99999 | 6.5e-3 |
| oc20 | Cu(100)+H | mixed [T,T,F] | 8.6e-5 | 1.00000 | 9.7e-4 |

- MoLE experts host-merged to a plain `eSCNMDBackbone` (fairchem's own `merge_mole` path):
  merged vs unmerged-MoE oracle **E rel 1.3e-12, force PCC 1.0** — the merge is exact.
- Periodic neighbour list reproduces fairchem's `radius_graph_pbc` edge-for-edge (edges + image
  offsets), so materials tasks work. `odac`/`omc` use the identical data-driven path (export with
  `--task`).
- Real-weight ASE FIRE relaxation of ethanol **converges on device** (fmax 9.16 → 0.049 eV/Å).
- **Trace path** (`trace=True`): device fwd+bwd is ~96% of a step and dispatch-bound, so capturing
  and replaying the op-stream gives **2.14× per step / 2.33× a full FIRE relaxation**, with
  bit-for-bit identical forces.

Also validated with **random weights** against a bit-exact PyTorch reference (the always-available,
ungated CI path): per-module PCC ≥ 0.99, end-to-end energy/forces, module VJPs. 22 tests pass.

**Model coverage / honest ceiling.** `uma-s-1` (lmax=mmax=2) is the validated default. `uma-m-1p1`
exports cleanly but is **not supported**: it uses lmax=4/mmax=2 spherical-harmonic coefficient
subselection, a code path TT-Atom does not implement (the calculator raises a clear error), and its
oracle+merge parity harness OOMs on a 30 GB host. **Single-card only on pc**; the 4-card 3.95× was
validated on qb1 and is not re-measured here.

No weights are shipped or redistributed; the `facebook/UMA` checkpoint is gated under the FAIR
Chemistry License and the real-weight tests auto-skip when absent. Conversion is one command
(`tt-atom convert-checkpoint` / `tools/export_weights.py`), and `tt-atom verify` closes the
device roundtrip against a fairchem reference embedded in the bundle at convert time.

## Suggested social post (draft)

> Got Meta's UMA interatomic potential running on Tenstorrent as a **drop-in for fairchem** —
> swap one calculator class and your ASE relaxations/MD run on the card. Energy **and analytic
> forces**, device-resident. Matches the **released uma-s-1** across molecules *and* periodic
> materials (omol / omat / oc20): energy to ≤3e-4, force PCC ≥ 0.9996 vs the fairchem oracle.
> The SO(2)-convolution trick makes it ~90 % dense GEMM, so it maps beautifully: a trace-captured
> MD loop runs **2.3× faster**, device compute up to **5.3× over CPU** (gap grows with size), and
> **~4× linear scaling across 4 Blackhole cards**. Apache-2.0, bring your own checkpoint. 🧪⚡

## Status

Apache-2.0 (our code). Tests pass (`pytest tests/ -q`). Benchmarks reproducible. Nothing pushed
or posted — ready for your review.
