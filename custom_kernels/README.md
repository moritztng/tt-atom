# Custom tt-metal kernels for TT-Atom

`tt_atom/rotation.py` routes UMA's per-edge Wigner rotation through custom tt-metal compute
kernels that the pip `ttnn` wheel does not carry, so UMA needs a **source tt-metal build** that
includes this op. Orb-v3 and OrbMol use stock `ttnn`. The op is pre-integrated on the
[`moritztng/tt-atom`](https://github.com/tenstorrent/tt-metal/tree/moritztng/tt-atom)
branch of tt-metal, so the normal install just clones and builds that branch (see the top-level
README). This directory is the authoritative backup of the op source and the recipe for
re-integrating it onto a newer tt-metal commit.

## What the op provides
One tt-metal experimental op library (`TTNN::Ops::Experimental::FusedRotate`) exposing four kernels
under `ttnn._ttnn.operations.experimental`:

- **`fused_rotate`** — the per-edge sparse Wigner rotation. Replaces the ~35 `ttnn.addcmul`
  dispatches of `rotation.rotate` with ONE compute-kernel launch: reads x once, keeps all `nnz`
  multiply-accumulates in the dest registers (fp32 accumulate, HiFi4), writes out once. **Measured
  4.3x faster** than the split+addcmul MAC it replaces on the uma-s forward-rotate shape (E=46016,
  n=9, W=256, nnz=35): 7.01 ms -> 1.62 ms, PCC=0.999995. This is the ALWAYS-ON path for uma-s.
- **`fused_rotate_gc`** — the coefficient-adjoint (`dE/dcoef`) mul-reduce used in the backward at
  large graphs (products L1-resident, one accumulating matmul reduces + places into `gc[E,nnz]`).
- `fused_gate`, `fused_ln_bw` — the SO(3) gate activation and the LayerNorm-backward reduction
  (optional; used by other modules).

### `fused_rotate` contract
`ttnn._ttnn.operations.experimental.fused_rotate(x_flat, coef_exp, n_in, n_out, W, deg, ks, js)`
- `x_flat`   `[E, n_in*W]` bf16/bf8_b TILE
- `coef_exp` `[E, nnz*32]` — each nonzero k's coefficient broadcast across a 32-col tile
             (`ttnn.repeat_interleave(coef[E,nnz], 32, dim=1)`)
- `deg[i]`   nonzeros feeding output block i; `ks`/`js` (len nnz) the coef-tile index and input
             block for each nonzero, grouped by output block i in order.
- returns `[E, n_out*W]`.

The kernel fans-in all `d` per-block products into `dst[0..d-1]` and sums them there, so `max(deg)`
must fit the fp32 DST register file (`dst_full_sync_en` -> 8 slots) and the per-core CBs must fit
L1 (~1.5 MB). uma-s (square 9x9, W=128/256) fits; uma-m (rectangular 19x25, W=256) overflows and is
unsupported — `rotation.rotate` raises rather than falling back.

## Performance flags

Every boolean knob in TT-Atom reads the same way: `=1` on, `=0` off, unset or anything else the
documented default (`tt_atom/device.py:flag`).

These five sit on top of the kernels above, so they need the source `ttnn` build. They no-op
safely on stock `ttnn`, where the capability probe drops each back to its ordinary path. The
first four are UMA-only (Orb has no equivariant representation; see `docs/orb-port.md`).

- **`fused_lnbw`** — fuses the radial-LayerNorm backward into one kernel, defaults ON. The kernel is bf16-only while the unfused default runs the radial backward in fp32, so this one *does* trade a little force accuracy on out-of-distribution compressed cells for the fuse; `TT_ATOM_FUSED_LNBW=0` takes the fp32 path.
- **`TT_ATOM_DEVICE_EDE=1`** — on-device edge-degree computation (off the host dispatch path).
- **`TT_ATOM_BF8_EDGE=1`** — the edge-activation dataflow through `fused_rotate`/`fused_gate` in bf8. This is where UMA's real bf8 bandwidth win comes from, not weight dtype: bf8 weights alone measure 1.00x (the forward is dispatch-bound, not DRAM-bandwidth-bound), so `fast=` is threaded through for reproducibility only.
- **`orb_fused_silu_bw`** — Orb's edge-MLP SiLU VJP through `fused_gate`, replacing six DRAM-backed elementwise programs with one fused device program. Defaults ON; `TT_ATOM_ORB_FUSED_SILU_BW=0` overrides.
- **`TT_ATOM_FUSED_GATE=1`** — UMA's gate fwd/bw column-split glue (slice+silu+slice+multiply+concat) through `fused_gate`, one launch instead of five ops. Default OFF: it is not bit-identical to the ordinary path (force PCC 0.99996 when it landed).

Five more A/B knobs gate device paths that need no custom kernel, so they work on stock `ttnn` too.
All five default ON and exist so the pre-optimization path stays reachable for comparison.

- **`TT_ATOM_SO2_FUSED=0`** — restores the per-m SO(2) convolution (~27 slice/matmul/combine ops) instead of the one fused `ttnn.linear` per m. UMA only.
- **`TT_ATOM_SPECTRAL_FUSED=0`** — restores the per-degree batched matmul in `SpectralAtomwise`, whose tiny coefficient dim tile-pads to 32. UMA only.
- **`TT_ATOM_NORM_FLAT=0`** — restores the 3D `[N, nsph, C]` `RMSNormSH` instead of the flat `[N, nsph*C]` reformulation. UMA only.
- **`TT_ATOM_ORB_SCATTER_RM=0`** — restores the tile-layout concat in `segment_sum` instead of row-major. Bit-exact either way (same reduction order). Orb only.
- **`TT_ATOM_ORB_MINIMAL_MATMUL=0`** — routes Orb's large edge MLPs through `ttnn.linear` instead of `ttnn.experimental.minimal_matmul`. Orb only.

`device_ede`/`bf8_edge` are manual opt-ins: ~2x on a traced MD step at large systems (512 atoms: 389 -> 194 ms; 216 atoms: 158 -> 85 ms, force PCC 0.9997), but they regress small molecules (~0.85x at 9 atoms), so they are not global defaults.

## Re-integrating onto a newer tt-metal commit

The `moritztng/tt-atom` branch already carries this op at the validated commit
**`8d759240fdd763a38e3abdc8344076f584dc4f4d`**, so use the steps below only when moving to a
different tt-metal commit.

1. Copy this op into the tt-metal tree:
   ```
   cp -r custom_kernels/fused_rotate \
       $TT_METAL_HOME/ttnn/cpp/ttnn/operations/experimental/fused_rotate
   ```
2. Apply the 3 registration edits (the op's own `CMakeLists.txt`/`sources.cmake` build all four
   kernels into one library; these hook that library into ttnn):
   - `ttnn/CMakeLists.txt`: add `add_subdirectory(cpp/ttnn/operations/experimental/fused_rotate)`
     and add `TTNN::Ops::Experimental::FusedRotate` to the `target_link_libraries(ttnn ...)` list.
   - `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp`: add
     `#include "ttnn/operations/experimental/fused_rotate/fused_rotate_nanobind.hpp"` and, inside
     `py_module`, `fr_detail::bind_fused_rotate(mod);` (binds all four kernels).
   - `ttnn/sources.cmake`: add
     `cpp/ttnn/operations/experimental/fused_rotate/fused_rotate_nanobind.cpp` to `TTNN_SRC_PYBIND`
     (omitting this links but leaves the Python symbols undefined — the classic trap).
3. Rebuild and re-stage the shared object (the `install` target both compiles and copies
   `_ttnn.so` into `ttnn/ttnn/` — this is what `build_metal.sh` itself runs):
   ```
   cmake --build $TT_METAL_HOME/build_Release --target install
   ```
   `pip install -e .` (top-level README) only needs to run once per venv — it's a packaging step,
   not a build step, so re-running it after this rebuild is unnecessary.
4. Verify:
   ```
   TT_METAL_HOME=$TT_METAL_HOME python3 -c \
     "import ttnn; e=ttnn._ttnn.operations.experimental; \
      print(hasattr(e,'fused_rotate'), hasattr(e,'fused_rotate_gc'))"   # -> True True
   ```

Run TT-Atom (both `ttnn` and `tt-atom` are `pip install -e`'d into the active venv):
`TT_METAL_HOME=$TT_METAL_HOME python ...`
