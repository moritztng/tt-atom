# Orb-v3 and OrbMol

TT-Atom runs all four public Orb-v3/OrbMol checkpoints with one device implementation and the
same ASE calculator interface as UMA.

## Supported checkpoints

| checkpoint | domain | forces | stress | charge/spin |
|---|---|---|---|---|
| `orb-v3-conservative-inf-omat` | periodic materials | analytic `-dE/dpos` | analytic virial | no |
| `orb-v3-direct-20-omat` | periodic materials | direct force head | direct stress head | no |
| `orb-v3-conservative-omol` | molecules | analytic `-dE/dpos` | no | yes |
| `orb-v3-direct-omol` | molecules | direct force head | no | yes |

The weights are public and independent of composition. TT-Atom exports each checkpoint once and
caches it under `~/.cache/tt_atom/orb_weights`.

## Architecture

Orb is a scalar attention message-passing network, not an equivariant tensor network. Its
spherical harmonics are fixed edge features; they are not rotated or carried as hidden tensor
representations. UMA's custom Wigner rotation therefore does not apply and Orb runs on stock
`ttnn` operations; the one custom kernel it does reuse when the source build supplies it is
`fused_gate`, for the edge-MLP SiLU VJP (`custom_kernels/README.md`).

The device path implements the encoder, five interaction layers, energy head, direct force/stress
heads, and the analytic reverse pass used by conservative checkpoints. Geometry, cutoff features,
charge/spin embeddings, output normalization, and the fixed ZBL pair potential are computed from
the same definitions as `orb-models`.

Orb's reference truncates each atom to the checkpoint's nearest-neighbour limit. TT-Atom does not
silently approximate that rule: it raises when a graph exceeds the limit. Use the `-inf` or
`omol` checkpoints, which allow up to 120 neighbours, or a smaller cell for dense systems.

## Device parity

The release suite compares real checkpoint outputs with `orb-models==0.5.5` goldens on the same
structures.

| system | checkpoint | energy relative error | force PCC |
|---|---|---:|---:|
| bulk Si | conservative-inf-omat | 1.19e-4 | 0.999975 |
| bulk Si | direct-20-omat | 5.79e-4 | 0.999966 |
| H2O | conservative-omol | 1.59e-6 | 0.999732 |
| H2O | direct-omol | 5.30e-7 | 0.998059 |
| NH4+ | conservative-omol | 4.55e-6 | 0.993305 |
| NH4+ | direct-omol | 1.83e-5 | 0.994663 |
| CH3 radical | conservative-omol | 9.23e-6 | 0.978500 |
| CH3 radical | direct-omol | 1.25e-5 | 0.892594 |

The open-shell radical is a disclosed bf16 noise-floor case. Its reference force magnitude is
about an order of magnitude below the closed-shell systems, while its absolute force error remains
in the same 0.006 to 0.010 eV/Å range. The direct checkpoint is gated at PCC 0.85 and MAE
0.02 eV/Å; the conservative path for the same structure reaches PCC 0.9785. The reference is
bit-identical across reruns. The analysis is reproducible with
`scripts/orb_omol_noise_floor.py` and `scripts/orb_omol_ref_self_consistency.py`.

Additional release rows cover periodic graph reconstruction, MgO mixed-element chemistry,
conservative and direct stress, and short-contact ZBL energy, forces, and stress. See
[`materials-benchmark.md`](materials-benchmark.md) for the complete table.

## Throughput

`OrbCalculator.evaluate_batch` concatenates independent systems into one block-diagonal graph.
At 128 nine-atom molecules, the measured speedup over a Python loop is about 19x for
`conservative-omol` and 12x for `direct-omol`.

`OrbTracedEngine` captures a fixed-topology forward and analytic reverse pass for MD or
relaxation. Replay is bit-exact with eager execution for energy and forces on the tested toy and
production graphs. The measured whole-step speedup is 1.30x to 1.51x.

The bf16 conservative-force curve for periodic Si measured on a p150a in July 2026 (AICLK not
recorded; the clocked p300c numbers are under [GPU comparison](#gpu-comparison)) is:

| atoms | warm traced step |
|---:|---:|
| 216 | 39.70 ms |
| 512 | 84.61 ms |
| 1000 | 164.36 ms |
| 2016 | 341.03 ms |

These values use the source-ttnn edge-MLP path. Stock `ttnn` keeps the same numerics and uses the
ordinary matmul and activation path.

Orb has one precision path: bf16 weights and activations with fp32 accumulation. `--fast` is still
accepted and does nothing. The bf8 mode it used to select was removed: at 1000 atoms it was 4.5%
faster (203.2 against 212.8 ms per step at 1350 MHz) for about three times the force error, it
gained nothing at 216 and 512 atoms, and it crashed at 2016.

`bucketing=True` on `OrbCalculator` (and `MultiCard`) pads a system graph to the next edge count
in a fixed ladder (256 to 22016, about 1.55x rung spacing) so differently-sized systems share
compiled kernels. Production screening feeds each (atoms, edges) pair once, so the compile cache
never helps an unbucketed stream: a cold first eval at a new size costs 40 to 47 s versus 1.3 to
1.7 s warm (five Si supercell sizes, 16 to 432 atoms, warm-repeat noise under 2%), and even a
second geometry at the same atom count costs another 11+ s when its edge count crosses a tile
boundary (`benchmarks/compile_pain_orb_si_p150a.jsonl`). Padded edges carry exact zeros; energies,
forces and stress are bit-exact against unpadded runs within the device instance-noise floor (the
parity gate is `tests/test_bucketing.py`, 6 sizes x all 4 Orb checkpoints). Measured on a 20-system
Si stream (16 to 256 atoms, 670 to 10934 edges, p150a):

| stream | cold wall clock | distinct edge shapes | kernel files compiled |
|---|---:|---:|---:|
| unbucketed | 235.1 / 236.5 s | 20 / 20 | 20831 |
| bucketed | 210.9 / 207.6 s (1.11x / 1.14x) | 7 buckets | 18481 (-11.3%) |

Two draws on the pinned tt-metal source build (2026-09-17, `bench_screening.py --card 0`). Warm,
the same stream runs 11.6 to 12.0 s unbucketed versus 7.0 to 11.2 s bucketed; the warm legs are
short enough that their spread swamps the difference.

What bucketing saves is compiles, and that part is exact and reproducible: 20 distinct edge shapes
collapse to 7 buckets and 2350 fewer kernel files get built, identical to the file across both
draws. How much wall clock that saves depends entirely on what one compile costs. An earlier
measurement of this same stream recorded 463.1 s versus 334.4 s (1.38x) with the same -11% file
count, so compiles were about twice as expensive in whatever environment produced it — which that
log does not record, the reason `bench_screening.py` now stamps results. Read the 1.1x as the
figure for the environment named above, not as a property of bucketing.

Raw logs: `benchmarks/screening_orb_si_p150a.jsonl` (the earlier, unstamped run).

## GPU comparison

Both sides run the same `orb-v3-conservative-inf-omat` checkpoint on the same periodic Si cells,
compute energy plus conservative forces with stress off, keep the neighbour list frozen and move
positions every step. The GPU runs `orb-models` 0.7.0 at its fastest setting: TF32 matmuls (the
package default, `float32-high`) with the whole step captured in one CUDA graph. That beat both
`torch.compile` and the stock calculator on every size. TT runs its traced bf16 path on one
Blackhole chip of a p300c at 1350 MHz AICLK. Times are medians of 80 to 100 steps.

| atoms | TT chip | H200 | B200 | TT atoms/s | TT per $ vs H200 / B200 |
|---:|---:|---:|---:|---:|---:|
| 216 | 41.9 ms | 7.24 ms | 6.29 ms | 5,155 | 4.5x / 5.0x |
| 512 | 93.6 ms | 13.98 ms | 11.63 ms | 5,470 | 4.7x / 5.3x |
| 1000 | 212.8 ms | 25.04 ms | 20.90 ms | 4,699 | 4.1x / 4.6x |
| 2016 | 392.0 ms | 47.58 ms | 39.95 ms | 5,143 | 4.4x / 5.0x |

A TT chip holds about 5,000 atoms/s at every size. A GPU does not: the H200 runs 29,800 atoms/s at
216 atoms and 42,400 at 2016 (B200: 34,300 and 50,500), because one small structure leaves it
partly idle. Someone screening small structures would batch them until the GPU is full, so the
per-dollar column credits each GPU with its best measured rate, the 2016-atom one, at every size.
On that basis one GPU does 8 to 11 times the work of one TT chip, and the TT chip delivers 4.1 to 4.7 times the throughput per dollar of an H200 and 4.6
to 5.3 times that of a B200, priced at $1,399 per chip against one eighth of an 8-GPU system
($51,250 for H200, $68,750 for B200). Batching several structures into one GPU step
has only been measured on an RTX 3090, where it raised the best rate by 10% (9,470 atoms/s for 16
cells of 216 atoms against 8,640 for one 2016-atom cell, same TF32 CUDA-graph setting). The rate
was still rising at 16 cells, where the sweep stopped, so 10% is a lower bound. The same gain on
the larger GPUs would bring the ranges to 3.7 to 4.3 times for H200 and 4.2 to 4.9 times for
B200. That arm has not been run on those GPUs.

Every leg is graded against the same float64 forces. TT bf16 has a force MAE of 6.0 to 6.7 meV/Å.
B200 in TF32 sits at 0.4 meV/Å, so the GPU numbers above are also the more accurate ones. H200 in
TF32 graded 1.7 to 10.5 meV/Å, with a larger worst-case force error than TT at three of the four
sizes; the table still uses that faster setting rather than charging the H200 for full fp32. Full fp32 (`float32-highest`) is below 0.01 meV/Å on both GPUs and costs 2.4 to
2.9 times the TF32 step.

The protocol is in `benchmarks/orb_gpu_bench_fair.py`, the comparison in
`benchmarks/orb_fair_compare.py`, and the raw results, clocks and float64 anchors in
`benchmarks/orb_fair/` (the RTX 3090 batching run is `gpu_rtx3090_float32-high_packed.json`). An earlier version of this section compared TT's traced path to the stock
H200 calculator and quoted 4.4x to 9.1x per dollar; that was not one protocol on both sides. A
later one compared size by size and quoted up to 7.4x, which charged the GPU for running a single
small structure.

## Reproduce

Generate real-weight goldens in the reference environment:

```bash
refenv/bin/python tests/gen_golden_orb.py \
  --ckpt conservative-inf-omat \
  --system supercell \
  --out ~/.ttatom_run/goldens_real/si_supercell_orb.npz
refenv/bin/python tests/gen_golden_orb.py \
  --ckpt conservative-inf-omat \
  --out ~/.ttatom_run/goldens_real/si_omat_orb.npz
refenv/bin/python tests/gen_golden_orb.py \
  --ckpt direct-20-omat \
  --out ~/.ttatom_run/goldens_real/si_omat_orb_direct20.npz
```

Run the full accuracy leg:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py --leg accuracy
```

Run the periodic Orb MD example:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 examples/orb_md.py \
  --weights ~/.ttatom_run/goldens_real/si_supercell_orb.npz \
  --nx 3 --ny 3 --nz 3 --steps 300 --temp 900
```
