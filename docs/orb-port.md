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
representations. UMA's custom Wigner rotation and SO(3) kernels therefore do not apply. Orb runs
on stock `ttnn` operations.

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

The current bf16 conservative-force curve for periodic Si is:

| atoms | warm traced step |
|---:|---:|
| 216 | 39.70 ms |
| 512 | 84.61 ms |
| 1000 | 164.36 ms |
| 2016 | 341.03 ms |

These values use the source-ttnn edge-MLP path. Stock `ttnn` keeps the same numerics and uses the
ordinary matmul and activation path.

`examples/orb_md.py --fast` stores hidden edge activations in bf8 while keeping the residual
stream in bf16 and accumulation in fp32. It is opt-in because the measured conservative-force MAE
is 0.0490 eV/Å versus 0.0089 eV/Å for bf16.

## H200 comparison

The committed comparison uses the same `orb-v3-conservative-inf-omat` checkpoint and periodic Si
cells. The TT side uses its normal traced MD path; the H200 side uses the stock
`orb-models==0.7.0` calculator.

| atoms | p150 bf16 traced | H200 fp32 stock | H200 speedup |
|---:|---:|---:|---:|
| 216 | 42.68 ms | 16.85 ms | 2.5x |
| 512 | 91.80 ms | 19.43 ms | 4.7x |
| 1000 | 188.27 ms | 44.47 ms | 4.2x |
| 2016 | 372.13 ms | 70.51 ms | 5.3x |

The H200 is faster at every tested size. At the measured card prices, the p150 still provides
about 4.4x to 9.1x more throughput per dollar. Raw timings and environment metadata are in
`benchmarks/orb_perf_dollar_gpu_v0.7.0.json` and
`benchmarks/orb_perf_edge_mlp_fused.json`.

## Reproduce

Generate real-weight goldens in the reference environment:

```bash
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py \
  --ckpt conservative-inf-omat \
  --system supercell \
  --out ~/.ttatom_run/goldens_real/si_supercell_orb.npz
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py \
  --ckpt conservative-inf-omat \
  --out ~/.ttatom_run/goldens_real/si_omat_orb.npz
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py \
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
