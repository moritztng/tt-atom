# MACE

TT-Atom runs MACE-MP-0b3 medium, energy and forces, on one Blackhole chip. This is an accuracy
port: it matches the float64 reference inside the bars below and it is not a performance win
yet. The measured step times and the reason for them are under [Throughput](#throughput).

## Supported checkpoints

| checkpoint | license | energy | forces | stress |
|---|---|---|---|---|
| `MACE-MP-0b3` medium | MIT | yes | analytic `-dE/dpos` | no |

Only MIT-licensed MACE weights are supported. Goldens carry the weights, so a checkpoint under
the Academic Software License (OMAT-0, MATPES, OMOL-0, MH, MDP, MACE-OFF23) must never be written
into one.

## Architecture

MACE-MP-0b3 medium has two interaction layers with 128 channels, spherical harmonics up to l = 3,
a 6 A cutoff and correlation order 3. The device path covers both interactions, both product
blocks (symmetric contraction), the readouts and the whole reverse pass that gives the forces.
The edge geometry (vectors, spherical harmonics, radial basis, ZBL pair repulsion) and its
vector-Jacobian product run on the host in float64.

Three choices shape the device code (`tt_atom/mace/tt_blocks.py`):

- **Dense real coupling.** The e3nn tensor product and the symmetric contraction are folded at
  load time into constant matrices, so a layer is matmuls and elementwise products of whole
  tensors and nothing is indexed per path.
- **Messages are summed before the coupling.** Edges are stored in a padded per-receiver layout
  and the sum over a node's neighbours is itself a matmul against the spherical harmonics. The
  Clebsch-Gordan coupling then runs once per atom, not once per edge.
- **Hand-written reverse pass.** There is no autograd on the card. Each forward block records
  what its backward needs and the backward uses transposed constants.

The matrix engine reads a float32 operand through 19-bit registers, which truncates it to a
10-bit mantissa and biased MACE's energy by about 9 meV/atom. Every float32 matmul operand is
therefore split into two bfloat16 parts and the significant products are summed, which keeps
about 16 bits. That costs two or three matmuls where one would do.

## Device parity

Graded against stock mace-torch in float64 on a rattled 64-atom Si supercell. Bars: 5 meV/atom
on the energy, 7 meV/A mean absolute error on the forces.

| path | energy error | force MAE | force max error |
|---|---:|---:|---:|
| float32 (shipped) | 1.17 meV/atom | 1.72 meV/A | 18 meV/A |
| bfloat16 interactions, float32 products | 14 to 15 meV/atom | 20 to 21 meV/A | |

Eager execution and trace replay give identical results. The bfloat16 path is faster (see below)
and fails both bars, so it is not offered.

## Throughput

One chip (p150, AICLK 1350 MHz sampled during every run), periodic Si, energy and forces, frozen
neighbour list, positions jittered by 0.01 A and uploaded every step, median of 20 steps, trace
replay. Errors are against float64 on the benchmark structure itself.

| atoms | edges | step | of which on the card | energy error | force MAE |
|---:|---:|---:|---:|---:|---:|
| 64 | 2944 | 82.5 ms | 73.9 ms | 0.77 meV/atom | 0.50 meV/A |
| 216 | 9936 | 229.9 ms | 208.2 ms | 0.71 meV/atom | 0.52 meV/A |
| 512 | 23552 | 453.3 ms | 419.0 ms | 0.50 meV/atom | 0.50 meV/A |
| 1000 | 46000 | 912.5 ms | 833.8 ms | 0.33 meV/atom | 0.49 meV/A |

About 0.9 ms per atom, more than 90 % of it on the card. This is far slower than MACE on a GPU
(a consumer RTX 3090 runs the 64-atom step in 2.1 ms with cuEquivariance, see
[GPU comparison](#gpu-comparison)), so there is no per-dollar claim for MACE.

The step is not limited by arithmetic. At 1000 atoms it issues 206 matmuls totalling 715 GFLOP
(`bench_mace_tt.py --eager --census`), which the card executes at 0.9 TFLOP/s. The largest of those matmuls timed alone
(`bench_mace_gemm_shapes.py`) take 200 ms per step, running anywhere from 0.2 TFLOP/s for the
batched per-node shapes to 66 TFLOP/s for the wide node layers. Tracing barely helps at this
size (an eager step takes 949 ms), so the host is not the limit either.

## GPU comparison

The GPU side was measured on one RTX 3090 (24 GB) with mace-torch 0.3.17, same checkpoint, same Si
cells and same protocol as the table above: energy and forces, stress off, frozen neighbour list,
positions jittered and uploaded every step, median of 100 steps. Every backend the package ships
(e3nn, cuEquivariance, OpenEquivariance, the cueq+oeq hybrid) was timed eager, compiled, with
`compile_mode="reduce-overhead"` and with the whole step captured in one CUDA graph. cuEquivariance
with TF32 matmuls won at every size. The table shows it for one cell, for several copies of the
cell batched until the card is full, and in float64.

| atoms | one cell, TF32 | atoms/s | batched, TF32 | one cell, float64 | TF32 energy error | TF32 force MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2.05 ms | 31,200 | 113,200 atoms/s (64 cells) | 14.7 ms | 0.01 meV/atom | 0.42 meV/A |
| 216 | 3.51 ms | 61,500 | 126,300 atoms/s (64 cells) | not run | 0.23 meV/atom | 0.43 meV/A |
| 512 | 5.92 ms | 86,500 | 127,800 atoms/s (16 cells) | 79.1 ms | 0.23 meV/atom | 0.43 meV/A |
| 1000 | 9.53 ms | 105,000 | 132,100 atoms/s (16 cells) | 156.4 ms | 0.39 meV/atom | 0.46 meV/A |
| 2016 | 17.34 ms | 116,300 | 125,400 atoms/s (4 cells) | 311.4 ms | 0.36 meV/atom | 0.45 meV/A |

Errors are against stock mace-torch in float64 on the same positions, the reference the TT numbers
use. The worst single force component is off by at most 2.3 meV/A. At 64 and 216 atoms the fastest
single-cell mode is the whole-step CUDA graph, from 512 atoms up it is the package's own
`reduce-overhead` compile. The stock calculator takes 72 ms at 512 atoms and 239 ms at 2016, and
float64 is 7 to 18 times slower than TF32.

So one RTX 3090 saturates near 130,000 atoms/s at the same force accuracy as the TT fp32 path, and
one TT chip does about 1,100 atoms/s. The gap is roughly 100 times against a consumer card, which
is why no H200 or B200 run was rented and no per-dollar ratio is quoted.

The protocol is `benchmarks/mace_gpu_bench_fair.py`, which shares its timing loop, clock sampler
and CUDA-graph capture with the Orb GPU bench through `benchmarks/_gpu_fair.py`. Every variant with
its time, GPU clock during the run, memory and error is in `benchmarks/mace_perf_dollar_gpu.json`
(built by `benchmarks/mace_gpu_bar.py`), and the raw files and float64 anchors are in
`benchmarks/mace_fair/`.

## Reproduce

Generate the golden in the reference environment (stock mace-torch, CPU, float64):

```bash
~/.ttatom_run/maceref/bin/python tests/gen_golden_mace.py \
  --model medium-0b3 --out ~/.ttatom_run/goldens_real/si_mace_mp0b3.npz
```

Host tests (reference, dense form, reverse pass) and device tests:

```bash
python -m pytest tests/test_mace_reference.py tests/test_mace_dense.py tests/test_mace_backward.py
TT_VISIBLE_DEVICES=0 python -m pytest tests/test_mace_device.py
```

Benchmark:

```bash
TT_VISIBLE_DEVICES=0 PYTHONPATH=.:benchmarks python benchmarks/bench_mace_tt.py \
  --weights ~/.ttatom_run/goldens_real/si_mace_mp0b3.npz --sizes 2x2x2,3x3x3,4x4x4,5x5x5
```

Use it from ASE:

```python
from ase.build import bulk
from tt_atom import MaceCalculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
atoms.calc = MaceCalculator("si_mace_mp0b3.npz")  # the state dict stored in the golden
print(atoms.get_potential_energy(), atoms.get_forces())
atoms.calc.close()
```

The calculator rebuilds the neighbour list every call. While the list is unchanged the device
tables and the captured trace are reused; when it changes they are rebuilt, which costs a compile
for a new edge count.
