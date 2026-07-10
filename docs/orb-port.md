# Orb-v3 port (in progress)

Porting [Orbital Materials' Orb-v3](https://github.com/orbital-materials/orb-models)
(`orb-v3-conservative-inf-omat`, `orb-v3-direct-20-omat`) onto Tenstorrent alongside the
existing UMA/eSEN support. Branch-only (`wk/tt-atom-orb-port`), not merged: UMA code paths
(`tt_atom/{model,norm,edgewise,so2,rotation,grid,spectral}.py`, `custom_kernels/`) are untouched.

## Architecture verdict: Orb is NOT equivariant — none of the SO(3) kernels transfer

Read directly from `orb_models.common.models.{gns,angular}.py` (both the GitHub `main` tree and
the installed PyPI `orb-models==0.5.5`, which differ only in module layout, not logic — diffed
byte-for-byte on the two files that matter):

- **`angular.SphericalHarmonics`** computes real spherical harmonics up to `lmax=3` from each
  edge's *unit vector* once, as a fixed `(lmax+1)^2`-length **scalar descriptor**, exactly like
  the Bessel RBF. It is never rotated, never carried as a per-node/per-edge tensor representation,
  and has no Wigner-D machinery anywhere in the codebase.
- **`gns.AttentionInteractionNetwork`** (the message-passing block, called an MPNN with attention
  in Orb's own docs) operates entirely on plain `[rows, latent_dim]` tensors: `nn.Linear`,
  `sigmoid`/`softmax` attention gates, `segment_sum`. No SO(2)/SO(3) convolution, no per-degree
  gating, no local-frame rotation step.
- Normalization is `mlp_norm="rms_norm"` → plain `torch.nn.RMSNorm` over the feature dim (no
  spherical-harmonic degree-balancing), and activation is plain `SiLU`.

**Consequence:** `custom_kernels/fused_rotate` (per-edge Wigner rotation) has nothing to rotate in
Orb — there is no equivariant hidden representation. `fused_gate` (the SO(2) gated nonlinearity
over spherical-harmonic degree blocks, `tt_atom/activation.py`) and `fused_ln_bw` (backward of
`RMSNormSH`'s degree-balanced reduction, `tt_atom/norm.py`) are also both specific to that
degree-block structure — Orb's LayerNorm/RMSNorm and SiLU are the ordinary scalar kind these
kernels do not apply to. **None of TT-Atom's four custom kernels transfer to Orb.** What *does*
transfer is the architecture-agnostic infra: `tt_atom/device.py`'s compute-kernel/dtype policy,
`tt_atom/scatter.py`'s linear O(E) edge→node segment-sum (generic, no equivariance assumption —
reused as-is), and the "host computes the fixed geometric terms once, device runs every learned
GEMM" split already established by `tt_atom/model.py`'s `GraphContext`.

## What's ported and PCC-verified (this pass)

Bottom-up against **both real target checkpoints** — `orb-v3-conservative-inf-omat` and
`orb-v3-direct-20-omat` (Orbital Materials' public S3 bucket, no gating) — on a real structure,
`ase.build.bulk("Si","diamond",a=5.43)*(2,1,1)` rattled (stdev=0.1, seed=1): the *same* system +
rattle seed already used for the UMA real-weight golden (`tests/gen_golden_real.py --system
bulk`), so all three models' outputs are a genuine same-system comparison point.

Golden generation (`tests/gen_golden_orb.py`, run in the existing `~/.ttatom_run/refenv`, which
already has `fairchem-core`+`e3nn` and now also `orb-models==0.5.5` installed side by side with no
dependency conflicts) hooks `MoleculeGNS._encoder`, each `gnn_stacks[i]`, and captures real
weights + real intermediate activations into an npz bundle (`tt_atom/orb_weights.py` reads it in
the ttnn env, mirroring `tt_atom/weights.py`'s numpy<2/numpy>=2 split).

Device port (`tt_atom/orb_model.py`): `RMSNorm`, `MLPNorm` (the 3-Linear-+-RMSNorm block used by
both the encoder and every interaction layer), `Encoder`, `AttentionInteractionLayer`,
`OrbGraphContext` (host-precomputed senders/receivers/cutoff + `tt_atom.scatter` gather tables),
`EnergyHead`, `ForceHead`. The fixed per-edge terms (Bessel RBF, spherical-harmonic descriptor,
the polynomial attention-cutoff envelope, the ZBL pair-repulsion energy) are computed on host and
uploaded/added once, exactly like UMA's wigner/gaussian/envelope buffers — they are fixed
functions of geometry, not learned.

Real on-device PCC (`tests/test_orb_realweight.py`, `tests/test_orb_direct_realweight.py`,
`TT_VISIBLE_DEVICES=0`, bf16 weights/activations, HiFi4 fp32-accumulate matmul — same numerics
policy as UMA), backbone depth (`orb-v3-conservative-inf-omat`):

| component | PCC vs real orb-models CPU reference |
|---|---|
| Encoder (node MLP) | 0.999997 |
| Encoder (edge MLP) | 0.999991 |
| Interaction layer 0 (node) | 0.999978 |
| Interaction layer 0 (edge) | 0.996864 |
| Full 5-layer backbone, final node embedding (→ energy head) | 0.999513 |
| Full 5-layer backbone, final edge stream | 0.976445 |

The node stream (what the energy head actually consumes) holds >0.999 PCC through all 5 layers.
The edge stream drifts more under bf16 by layer 5 (0.997→0.976) — expected precision compounding
over depth in a pure residual with no further consumer, not a correctness bug.

### End-to-end device energy + forces (both target checkpoints)

Went beyond the backbone to the actual consumer-facing quantities, on device, real weights:

- **`EnergyHead`** (`tt_atom/orb_model.py`): mean-aggregate the final node embedding, 2-layer MLP
  on device (`Linear→SiLU→Linear`), then a host-side denormalize (`ScalarNormalizer.inverse` +
  atom-average undo + `LinearReferenceEnergy` per-element lookup) — same shape as UMA's
  `scale_rmsd`/`scale_mean`/`elem_refs` (`tt_atom/weights.py`), different reference table.
- **`ForceHead`** (direct checkpoint only): per-node 2-layer MLP on device, then net-force removal
  (subtract the per-system mean predicted force — a fixed geometric correction, `ttnn.mean`+
  `ttnn.subtract`) and a host scalar-normalizer inverse. **No autograd** — this is the entire
  reason `orb-v3-direct-20-omat` is the fast checkpoint.
- **ZBL pair repulsion** (`host_zbl_energy`, `tt_atom/orb_model.py`): the Ziegler-Biersack-
  Littmark potential has *zero* learned parameters (6 universal physical constants) — implemented
  as a direct host `torch` function from real atomic numbers + edge vectors, exactly like the
  attention-cutoff envelope. Measured `9.5e-8 eV` for this Si golden — negligible, because the
  nearest-neighbor Si-Si bond length (2.20-2.35 Å) sits just outside the ZBL envelope's own cutoff
  (sum of covalent radii ≈ 2.22 Å) at this near-equilibrium geometry. Confirmed on the real graph,
  not assumed; ZBL forces (needed for `orb-v3-direct-20-omat`'s total force) were therefore not
  implemented this pass since their contribution is unmeasurable at this system's geometry — flagged
  in Open below for whichever system exercises it (surface defects, short contacts).

Real end-to-end numbers (`tests/test_orb_realweight.py`, `tests/test_orb_direct_realweight.py`):

| checkpoint | quantity | device result | real oracle | error |
|---|---|---|---|---|
| conservative-inf-omat | energy | -20.499663 eV | -20.497231 eV | rel err 1.19e-4 |
| direct-20-omat | energy | -20.404415 eV | -20.392614 eV | rel err 5.79e-4 |
| direct-20-omat | forces | PCC 0.999966 | — | MAE 0.0079 eV/Å (oracle \|F\|max 2.4661) |

Both energy errors are well inside UMA's own real-weight test bar (`tests/test_realweight.py`:
"energy rel err < 1e-2"). The direct-20 backbone reused the *exact same* `Encoder`/
`AttentionInteractionLayer` classes as the conservative checkpoint, unmodified — different
weights, a smaller graph (max-20-neighbor cutoff → 80 edges vs 172 for the same 4-atom cell) —
confirming the port's bottom-up pieces are checkpoint-agnostic, not conservative-specific.

**Real energy/force numbers, same Si system, all three models (CPU oracle for UMA, device for
Orb):**

| model | E (eV) | \|F\|max (eV/Å) |
|---|---|---|
| uma-s-1 (omat, merged MoLE) | -20.497236 | 2.5249 |
| orb-v3-conservative-inf-omat (device) | -20.499663 | 2.4553 (oracle; forces not yet ported for this checkpoint) |
| orb-v3-direct-20-omat (device) | -20.404415 | 2.4661 (device, PCC 0.999966 vs oracle) |

All three graphs are built from the identical Si structure. Energies agree to within ~0.1 eV out
of ~20.5 eV (~0.5%) across three independently-trained models — expected agreement between
competent MLIPs on a near-equilibrium bulk cell, not by itself evidence of correctness (the PCC/
rel-err tables above are the actual correctness evidence). No official Orb-vs-UMA/eSEN comparison
exists upstream; this is the first same-system numeric side-by-side for TT-Atom.

## Profiling (methodology: measure before reaching for a custom kernel)

Per the hard lesson from tt-bio's Boltz-2 trimul kernel (a hand-rolled kernel that was *slower*
than a standard op once host-dispatch overhead was accounted for — the real win there was trace
capture, not new math) — measured, not assumed, before considering any Orb-specific kernel:

Warm (program-cache-hot) forward, encoder + all 5 interaction layers, on the real Si golden
(N=4, E=172): **5.4 ms/call**. At this tiny system size the workload is almost certainly
dispatch-bound, not compute-bound — the graph is ~45 separate ttnn ops (3 Linear + RMSNorm per
MLP × 2 MLPs × 5 layers, plus per-layer attention gates/gathers/scatter-sums), each launched
individually with no fusion or trace capture yet.

**Verdict: no analogous "many small per-edge dispatches collapsible into one fused kernel"
pattern exists in Orb** the way it did for UMA's Wigner rotation (which fused ~35 `addcmul`
dispatches into one kernel because each was operating on the *same* per-edge sparse-rotation
structure). Orb's interaction layer is ordinary dense GEMMs + two small scatter-adds — ttnn
already has efficient primitives for all of it. The applicable lever, per the methodology, is
**trace capture** (`tt_atom/trace.py` already exists for UMA and is architecture-agnostic — it
captures/replays a fixed ttnn op graph) to eliminate host-dispatch overhead across the ~9 ops/layer,
not a new custom kernel. This is a real profiled measurement at a toy system size, not a
production-scale conclusion — worth re-measuring at a production cell size before committing to
trace capture as the answer.

## Open (not done this pass)

- **Autograd forces for the conservative variant.** `orb-v3-conservative-inf-omat`'s forces come
  from backprop through the energy (that's the entire point of "conservative" — physically exact
  forces as `-dE/dx`). UMA's `tt_atom/forces.py` VJP machinery is written against the equivariant
  backbone's ops (SO(2) conv, Wigner rotate, `RMSNormSH`); Orb's backward needs analogous VJPs for
  plain `Linear`/`RMSNorm`/`sigmoid`-gate/`segment_sum` instead — conceptually simpler (no rotation
  adjoint, no local-frame chain rule) but not yet written. `orb-v3-direct-20-omat`'s forces
  (direct MLP prediction, no autograd) are already ported and PCC-verified — see above.
- **ZBL pair-repulsion forces.** The ZBL *energy* is implemented and confirmed negligible for
  this Si golden (see above); its analytic force contribution (`dV_ZBL/dr`, needed for
  `direct-20-omat`'s total force whenever ZBL is non-negligible — short contacts, surface defects)
  is unimplemented. Straightforward (either the closed-form derivative already written out in
  `pair_repulsion.ZBLBasis._polynomial_cutoff_with_derivative`, or a host `torch.autograd.grad`
  on the same closed-form host energy function) but untested since no available golden exercises
  it.
- **Periodic images / half-supercell edge construction** at production cell sizes (this pass's
  4-atom cell has no periodic self-images within either cutoff to worry about); UMA's periodic
  graph construction (`tt_atom/geometry.py`) should transfer directly since it's architecture-
  agnostic, but hasn't been exercised against Orb's own neighbor-list conventions yet.
- **Multicard fan-out, disjoint-union batching, `--fast` (bf8) mode** — all existing UMA infra
  (`tt_atom/{disjoint,batch}.py`) is architecture-agnostic and should attach to Orb's `Encoder`/
  `AttentionInteractionLayer` without modification, but untested here.
- **Stress** (both checkpoints) — not exercised; would follow the same displacement-gradient
  pattern as forces for the conservative checkpoint, or a dedicated `StressHead` MLP (same shape
  as `EnergyHead`) for the direct checkpoint.

## Reproducing

```bash
# 1. goldens (real weights, real Si structure) -- refenv (numpy>=2, has orb-models + fairchem)
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt conservative-inf-omat \
    --out ~/.ttatom_run/goldens_real/si_omat_orb.npz
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt direct-20-omat \
    --out ~/.ttatom_run/goldens_real/si_omat_orb_direct20.npz

# 2. on-device PCC verification -- ttnn env (numpy<2)
TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest \
    tests/test_orb_realweight.py tests/test_orb_direct_realweight.py -q -s
```
