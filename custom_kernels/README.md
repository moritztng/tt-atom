# Custom tt-metal kernels for TT-Atom

`tt_atom/rotation.py` routes the per-edge Wigner rotation through custom tt-metal compute kernels
that the pip `ttnn` wheel does not carry, so TT-Atom needs a **source tt-metal build** that
includes this op. The op is pre-integrated on the
[`moritztng/tt-atom-kernels`](https://github.com/tenstorrent/tt-metal/tree/moritztng/tt-atom-kernels)
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

## Re-integrating onto a newer tt-metal commit

The `moritztng/tt-atom-kernels` branch already carries this op on top of validated commit
**`b5522097b39`** (`Migrate experimental/ssm leftovers to ProgramDescriptor`, #44403), so you only
need the steps below to rebase the op onto a *different* tt-metal commit. They are exactly how that
branch was produced.

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
3. Build and stage the shared object:
   ```
   ninja -C $TT_METAL_HOME/build ttnn
   cp $TT_METAL_HOME/build/lib/_ttnn.so $TT_METAL_HOME/ttnn/ttnn/_ttnn.so
   ```
4. Verify:
   ```
   TT_METAL_HOME=$TT_METAL_HOME PYTHONPATH=$TT_METAL_HOME/ttnn python3 -c \
     "import ttnn; e=ttnn._ttnn.operations.experimental; \
      print(hasattr(e,'fused_rotate'), hasattr(e,'fused_rotate_gc'))"   # -> True True
   ```

Run TT-Atom with the source ttnn on the path:
`TT_METAL_HOME=$TT_METAL_HOME PYTHONPATH=$TT_METAL_HOME/ttnn:/path/to/TT-Atom python ...`
