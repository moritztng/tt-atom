# PET-MAD (UPET) port

Porting [lab-cosmo's PET-MAD](https://github.com/lab-cosmo/upet) (`pet-mad-s` v1.5.0, the r2SCAN refresh
of early 2026) into TT-Atom as a third model family alongside UMA (eSEN/eSCN-MD) and Orb-v3/OrbMol.
Nothing is merged to `master` until Moritz's explicit go-ahead (model-merge-approval-gate).

## Current state (pass 4)

The port is reachable end-to-end as `Calculator(atoms, "pet-mad-s-v1.5.0")` (`tt_atom.pet_calculator`,
wired through `tt_atom.auto`), with energy + conservative forces verified on the 16-atom rattled Si
golden against the upstream `UPETCalculator` reference:

| quantity | device/host source | vs upstream reference |
| --- | --- | --- |
| per-GNN-layer node/edge features | device backbone (`tt_atom.pet_model`, bf16) | PCC 0.9998–0.9999 (gate ≥ 0.999) |
| energy | device backbone (bf16) | 0.026 eV (host float32 floor 1.15e-5 eV; gap is Blackhole bf16 matmul) |
| forces (`F = -dE/dpos`) | host autograd through `tt_atom.pet_model_host` (float32) | PCC 1.0, max abs 1.7e-6 eV/Å |

The energy comes from the device backbone; the forces come from host autograd through the verified
pure-torch reference backbone (the device VJP through PET's manual-attention log-mask is the
non-trivial new surface, deferred — see the conservative-forces section below). The reported forces
are the gradient of the *host* energy, so the `(energy, forces)` pair is consistent to ~0.026 eV
rather than bit-conservative; a full device VJP makes the force the gradient of the *device* energy
and is the known gap a later pass closes. On the 16-atom golden the device forward is ~12 ms and the
host backward ~32 ms (`tt_atom.pet_forces.profile_forces`), so the host backward is the cost path the
device VJP would erase.

Not yet implemented (documented gaps): conservative stress (needs a strain adjoint through the
geometry), the non-conservative force/stress heads (not the ASE default), the LLPR uncertainty
ensemble, disjoint-union batched evaluation, and trace capture.

## License (verified, not assumed)

Read directly from the upstream `LICENSE` file and the GitHub repository metadata, not the scout
summary:

- `LICENSE` (raw file, `lab-cosmo/upet@main`): "BSD 3-Clause License", "Copyright (c) 2023,
  Laboratory of Computational Science and Modeling", with the standard three clauses intact.
- GitHub API `repos/lab-cosmo/upet .license`: `name = "BSD 3-Clause \"New\" or \"Revised\"
  License"`, `spdx_id = "BSD-3-Clause"`.

Both agree: **BSD-3-Clause**, ungated, weights on HuggingFace `lab-cosmo/upet` (no gating form).
Compatible with TT-Atom's MIT license and the existing Orb-v3 / UMA weight redistribution model.

## Architecture verdict: PET is an unconstrained scalar transformer — same TT-fit family as Orb

Read from the installed `metatrain.pet` package (`metatrain==2026.3.1`, the version `upet==0.2.6`
pulls) and confirmed against the real `pet-mad-s-v1.5.0.ckpt` weights:

- **No e3nn irreps, no Wigner-D, no tensor products.** PET (Point Edge Transformer) is an
  unconstrained transformer over scalar tokens. Each GNN layer is a `CartesianTransformer`: it
  embeds each edge from its raw Cartesian vector + distance (`edge_embedder = Linear(4, d_pet)`),
  concatenates the edge embedding with the neighbor's species embedding and the inbound message,
  runs a small `Transformer` (multi-head attention + SwiGLU feedforward, PreLN, RMSNorm) over
  the per-atom token sequence `[center, neighbors...]`, and returns updated node + edge
  embeddings. Equivariance is by training-time augmentation, not by representation — exactly the
  same architectural class as Orb-v3's `AttentionInteractionNetwork`, and the same class the
  ICLR 2026 MLIP blogpost flags as needing **no** local-frame rotation.
- **No custom kernel transfers.** `custom_kernels/fused_rotate` (per-edge Wigner rotation),
  `fused_gate` (SO(2) degree-block nonlinearity), `fused_ln_bw` (degree-balanced RMSNorm backward)
  are all specific to the equivariant degree-block structure that PET does not have. None apply.
  What *does* transfer is the architecture-agnostic infra: `tt_atom/device.py`'s dtype/kernel
  policy, `tt_atom/scatter.py`'s linear O(E) segment-sum, `tt_atom/geometry.py`'s `radius_graph`,
  and the "host computes fixed geometric terms once, device runs every learned GEMM" split.

### Real `pet-mad-s` v1.5.0 hypers (from the checkpoint, not the docs defaults)

| hyper | value |
| --- | --- |
| `cutoff` | 8.0 Å (Bump switching, `cutoff_width=0.5`) |
| `num_neighbors_adaptive` | 16 (adaptive cutoff **on**, `"grid"` method) |
| `d_pet` (edge) | 256 |
| `d_node` | 1024 |
| `d_head` | 256 |
| `d_feedforward` | 512 |
| `num_heads` | 8 |
| `num_attention_layers` (per GNN layer) | 1 |
| `num_gnn_layers` | 3 |
| `normalization` | RMSNorm |
| `activation` | SwiGLU |
| `transformer_type` | PreLN |
| `featurizer_type` | feedforward |
| `zbl` | False |
| `long_range.enable` | False |
| `system_conditioning` | False (pet-mad-s v1.5.0; the head exists for the OMol-conditioned variants) |
| atomic species | 102 (elements 1..102) |

### Real param count and inference cost band (from the checkpoint)

Base PET (no LLPR uncertainty head): **25,924,122 params (~25.9M)**, 133 parameter tensors.
The deployed checkpoint wraps this in an `LLPRUncertaintyModel` (128-member ensemble for energy
UQ, +65,536 params) — that wrapper is **not** needed for plain energy/forces and is out of scope
for the port.

Param distribution (top-level submodules of the base PET):

| submodule | params | share |
| --- | --- | --- |
| `gnn_layers` (3 × CartesianTransformer) | 23,222,528 | 89% |
| `combination_mlps` (3 × 2-layer MLP) | 1,181,952 | 4.6% |
| `node_heads` (energy + 2 non-conservative) | 984,576 | 3.8% |
| `edge_heads` | 394,752 | 1.5% |
| `node_embedders` (Embedding(102, 1024)) | 104,448 | 0.4% |
| `edge_embedder` (Embedding(102, 256)) | 26,112 | 0.1% |
| last layers + combination norms | ~12k | <0.1% |

Within each GNN layer (~7.74M params), the dominant block is the **center MLP**
(`center_mlp.w_in` 1024→2048 SwiGLU = 4.19M, `center_mlp.w_out` 2048→1024 = 2.10M; 6.29M of the
7.74M, 81% of the layer). This is a **fat node-level GEMM** (1024-wide inner dim, runs on N atoms),
well above Blackhole's bf16 matmul ridge (K≈1024). Unlike UMA's skinny C=128 GEMMs that sit ~8× below
the ridge and keep the card dispatch-bound forever (see `tt-atom-mlip-perf`), PET's center MLP is
squarely in compute-bound territory. The attention is short-sequence (target 16 neighbors → 17
tokens/atom), batched over atoms — cheap. **PET-MAD is a better architectural fit for Blackhole
than UMA**: the param mass is where the card is strong.

## Force path verdict: PET-MAD's default ASE forces are CONSERVATIVE (autograd), not the direct head

This is the single most important pass-1 finding for the port, and it inverts the first-glance
reading of the checkpoint.

The checkpoint ships three heads: `energy`, `non_conservative_forces`, `non_conservative_stress`.
Seeing a `non_conservative_forces` head suggests (wrongly) that the port mirrors **Orb-v3-direct**
(a per-node force MLP, no device backward). It does not. The default `UPETCalculator` (no
`non_conservative=True`) returns **conservative forces = -dE/dpos via autograd through the energy
head**, exactly like **Orb-v3-conservative**. Verified bit-exactly on the 16-atom rattled Si
diamond golden (the same system + rattle seed used by the UMA/Orb goldens):

| quantity | device-side source | vs `UPETCalculator` reference |
| --- | --- | --- |
| energy | energy head (mean-pool + 2-layer MLP + composition reference) | bit-exact: -91.815010 == -91.815010 eV |
| forces (default ASE path) | **-dE/dpos autograd** through encoder + 3 GNN layers + energy head + host geometry | bit-exact: PCC 1.0, max abs 1.2e-6 eV/Å, \|F\|max 3.941864 == 3.941864 |
| forces (`non_conservative_force` head) | per-node MLP head | PCC 0.9989, rel 4.6% — a *different quantity*, not the ASE default |

**Consequence for the port:** the device path must implement the analytic-force VJP (hand-written
backward through the transformer + the host edge-featurization), mirroring `tt_atom/orb_forces.py`
(Orb-v3-conservative), **not** the direct `ForceHead` path. The non-conservative head exists in
the weights but is not what ASE users get from `UPETCalculator`; it is out of scope for parity
unless a caller explicitly opts into `non_conservative=True` (defer).

Pass 4 ships the conservative forces via the **host-finish route** (device forward for the energy +
host autograd through `pet_model_host` for the forces, PCC 1.0 vs the golden), not the full device
VJP — the manual-attention log-mask backward is the one non-trivial new VJP vs Orb, and the
bounded-turn SACRED-correctness bar prefers the already-verified host autograd path. The device VJP
is the known perf-and-consistency gap a later pass closes (see *Current state* above).

Stress (`non_conservative_stress`) is a non-conservative head only — there is no conservative
stress path in the checkpoint, so variable-cell stress is a documented gap (same gap class as
OrbMol, and the same one the scout flagged: "stress is not in the public PET-MAD head set" — more
precisely, a non-conservative stress head exists but is not the autograd-of-energy stress that
variable-cell relaxation needs).

## Reference parity captured this pass (for pass 2)

All captured on the 16-atom rattled Si diamond `(2,1,1)` cubic cell, `a=5.43`, rattle
`stdev=0.1, seed=1` — the same system the UMA and Orb real-weight goldens use, so the three
families share a genuine same-system comparison point. Committed under `tests/data/`:

- `pet_mad_s_si_golden.npz` — `UPETCalculator(model="pet-mad-s", version="1.5.0")` reference:
  `energy = -91.815010 eV`, `forces` (16,3) with `\|F\|max = 3.941864 eV/Å`, `sum(F) ≈ 0`
  (translation invariance of the conservative path).
- `pet_mad_s_si_internals.npz` — host geometry (`batch_data`: `edge_vectors`, `edge_distances`,
  `cutoff_factors`, `padding_mask`, `reverse_neighbor_index`, `nef_to_edges_neighbor`,
  `centers`, `neighbors`, `cell_shifts`, `element_indices_*`, `atomic_cutoffs_stats`) **after**
  the adaptive-cutoff edge filter (264 edges for 16 atoms, degree 16–19), plus the per-GNN-layer
  node `(16,1024)` and edge `(16,19,256)` outputs captured by forward hooks, plus the raw
  `non_conservative_force` head output. These are the component-by-component PCC targets for
  pass 2.

`docs/pet-mad-port/pet_mad_s_statedict_shapes.json` and `pet_mad_s_base_params.txt` carry the
full 148-tensor state-dict shape map and per-tensor param counts, so pass 2 can build the device
weight loader without re-deriving the layout.

## Pass-2 scoping (the next bounded chunk)

1. **Weight export** (`tools/export_pet_weights.py` + `tt_atom/pet_weights.py`, mirroring
   `tools/export_orb_weights.py` + `tt_atom/orb_weights.py`): load the `.ckpt` in the reference
   env (the new `~/.ttatom_run/upetenv`, which has `upet==0.2.6` + `metatrain==2026.3.1`
   installed alongside the existing `fairchem`/`orb-models` refenv via a `.pth` reuse — no
   numpy-2 conflict), flatten the 148-tensor state dict into the layout `pet_model.py` will
   consume, and capture the golden intermediates. Skip the LLPR ensemble (UQ-only).
2. **Host geometry** (`tt_atom/pet_geometry.py`, mirroring `tt_atom/orb_geometry.py`):
   reimplement `systems_to_batch`'s pos-dependent pieces — edge vectors/distances, the adaptive
   cutoff (target-16 "grid" method, per-atom cutoff then symmetrized pair cutoff, edge filter),
   the Bump cutoff factor, and the NEF padding + `reverse_neighbor_index` — as a differentiable
   host `torch` function (its `d/dpos` feeds the conservative-force VJP). PCC-gate against the
   captured `bd_*` tensors.
3. **Device backbone** (`tt_atom/pet_model.py`, mirroring `tt_atom/orb_model.py`): `Encoder`
   (species embedding → `d_node`), `CartesianTransformerLayer` (edge embedder `Linear(4,256)`,
   compress MLP, the `Transformer` attention + SwiGLU feedforward + RMSNorm + PreLN, the center
   contraction/expansion + center MLP on `d_node=1024`), `EnergyHead`. Reuse `tt_atom/scatter.py`
   for the NEF edge→node reductions. PCC-gate each layer's node/edge output against the captured
   `gnn{i}_node_out` / `gnn{i}_edge_out`.
4. **Conservative forces** (`tt_atom/pet_forces.py`, mirroring `tt_atom/orb_forces.py`):
   hand-written device VJPs for `Linear`, `RMSNorm`, `SiLU`/`SwiGLU`, and the NEF scatter-sum, plus
   a host autograd finish for the edge-featurization VJP (the `orb_geometry` pattern). Gate at
   PCC > 0.999 / MAE bar matching the Orb-conservative force test.
5. **Calculator wiring** (`tt_atom/pet_calculator.py` + `tt_atom/auto.py`): add `pet-mad-s` (and
   `pet-mad-xs`) to the `Calculator(atoms, model=...)` dispatch — `pet-*` routes to
   `PETCalculator.from_checkpoint`, mirroring `OrbCalculator.from_checkpoint` (the model name is
   the checkpoint; no per-composition bundle, since PET bakes no MoLE routing). Expose
   energy + conservative forces; document the non-conservative-stress gap.
6. **Trace capture** (later pass): port `tt_atom/orb_trace.py`'s `OrbTracedEngine` pattern once
   the eager force path is parity-clean — the same dispatch-bound-at-small-N, compute-bound-at-the-
   center-MLP profile that makes trace capture the right lever (not a custom kernel).

The port is **BSD-3-Clause upstream + ungated weights**, architecturally a clean scalar-transformer
fit for stock ttnn, and its default force path is the conservative one TT-Atom already hand-writes
for Orb-v3. No new custom kernel, no new dependency major-bump, no accuracy/OOM risk — the port
stays on `experimental/pet-mad-port` and is held out of `master` until Moritz's explicit
go-ahead (model-merge-approval-gate).
