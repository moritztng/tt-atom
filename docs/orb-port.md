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

Bottom-up against the **real** `orb-v3-conservative-inf-omat` checkpoint (Orbital Materials'
public S3 bucket, no gating), on a real structure — `ase.build.bulk("Si","diamond",a=5.43)*(2,1,1)`
rattled (stdev=0.1, seed=1), the *same* system + rattle seed already used for the UMA real-weight
golden (`tests/gen_golden_real.py --system bulk`), so the two backbones' outputs are a genuine
same-system comparison point.

Golden generation (`tests/gen_golden_orb.py`, run in the existing `~/.ttatom_run/refenv`, which
already has `fairchem-core`+`e3nn` and now also `orb-models==0.5.5` installed side by side with no
dependency conflicts) hooks `MoleculeGNS._encoder`, each `gnn_stacks[i]`, and captures real
weights + real intermediate activations into an npz bundle (`tt_atom/orb_weights.py` reads it in
the ttnn env, mirroring `tt_atom/weights.py`'s numpy<2/numpy>=2 split).

Device port (`tt_atom/orb_model.py`): `RMSNorm`, `MLPNorm` (the 3-Linear-+-RMSNorm block used by
both the encoder and every interaction layer), `Encoder`, `AttentionInteractionLayer`,
`OrbGraphContext` (host-precomputed senders/receivers/cutoff + `tt_atom.scatter` gather tables).
The fixed per-edge terms (Bessel RBF, spherical-harmonic descriptor, the polynomial attention-cutoff
envelope) are computed on host and uploaded once, exactly like UMA's wigner/gaussian/envelope
buffers — they are fixed functions of geometry, not learned.

Real on-device PCC (`tests/test_orb_realweight.py`, `TT_VISIBLE_DEVICES=0`, bf16 weights/activations,
HiFi4 fp32-accumulate matmul — same numerics policy as UMA):

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

**Real energy/force numbers, same Si system, both real checkpoints (CPU oracle, not device — no
device energy head yet, see Open below):**

| model | E (eV) | \|F\|max (eV/Å) |
|---|---|---|
| uma-s-1 (omat, merged MoLE) | -20.497236 | 2.5249 |
| orb-v3-conservative-inf-omat | -20.497231 | 2.4553 |

Both graphs have identical edge count (172 edges, 4 atoms, 6 Å cutoff) since it's the same
structure. The energies agree to ~5 μeV and forces to the same order of magnitude — expected
agreement between two competent independently-trained MLIPs on a near-equilibrium bulk cell, not
by itself evidence that either port is correct (the per-component PCC table above is the actual
correctness evidence). No official Orb-vs-UMA/eSEN comparison exists upstream; this is the first
same-system numeric side-by-side for TT-Atom.

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

- **Energy head + pair-repulsion (ZBL) on device.** `EnergyHead` mean-aggregates the l=0 node
  channel through a 2-layer MLP, applies a `ScalarNormalizer` (learned mean/std) and adds a
  per-element reference-energy table (`REFERENCE_ENERGIES["vasp-shifted"]`) — same shape as
  UMA's `scale_rmsd`/`scale_mean`/`elem_refs` normalizer already in `tt_atom/weights.py`, just a
  different reference table. `pair_repulsion=True` adds an analytic ZBL term computed directly
  from atomic numbers/distances (no learned weights beyond a handful of scalars) outside the GNN.
  Neither is architecturally hard; both were left open to keep this pass to a verified,
  bottom-up chunk rather than a half-verified full pipeline.
- **Autograd forces for the conservative variant.** UMA's `tt_atom/forces.py` VJP machinery is
  written against the equivariant backbone's ops (SO(2) conv, Wigner rotate, `RMSNormSH`); Orb's
  backward needs analogous VJPs for plain `Linear`/`RMSNorm`/`sigmoid`-gate/`segment_sum` instead
  — conceptually simpler (no rotation adjoint) but not yet written.
- **`orb-v3-direct-20-omat`** not started. Per Orb's own docs it predicts forces directly (no
  energy-autograd path) — likely *simpler* to port than the conservative variant (skips the whole
  force-VJP problem above) and is the better perf target; next candidate for a follow-up pass.
- **Periodic images / half-supercell edge construction** at production cell sizes (this pass's
  4-atom cell has no periodic self-images within the 6 Å cutoff to worry about); UMA's periodic
  graph construction (`tt_atom/geometry.py`) should transfer directly since it's architecture-
  agnostic, but hasn't been exercised against Orb's own neighbor-list conventions yet.
- **Multicard fan-out, disjoint-union batching, `--fast` (bf8) mode** — all existing UMA infra
  (`tt_atom/{disjoint,batch}.py`) is architecture-agnostic and should attach to Orb's `Encoder`/
  `AttentionInteractionLayer` without modification, but untested here.

## Reproducing

```bash
# 1. golden (real weights, real Si structure) -- refenv (numpy>=2, has orb-models + fairchem)
~/.ttatom_run/refenv/bin/python tests/gen_golden_orb.py --ckpt conservative-inf-omat \
    --out ~/.ttatom_run/goldens_real/si_omat_orb.npz

# 2. on-device PCC verification -- ttnn env (numpy<2)
TT_VISIBLE_DEVICES=0 PYTHONPATH=. ~/.ttatom_run/env/bin/python -m pytest tests/test_orb_realweight.py -q -s
```
