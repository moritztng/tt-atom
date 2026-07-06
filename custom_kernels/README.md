# Custom tt-metal kernels for TT-Atom (feat/perf-radical)

## fused_rotate
Fused per-edge sparse Wigner rotation. Replaces the ~35 `ttnn.addcmul` dispatches of
`tt_atom/rotation.py::rotate` with ONE compute-kernel launch: reads x once, keeps all `nnz`
multiply-accumulates in the dest registers, writes out once. Measured **13.9x faster** in
isolation on the uma-s forward-rotate shape (E=46016, n=9, W=256, nnz=35): 20.78ms -> 1.50ms,
PCC=0.999995 vs the MAC form.

### Contract
`ttnn._ttnn.operations.experimental.fused_rotate(x_flat, coef_exp, n_in, n_out, W, deg, ks, js)`
- `x_flat`   [E, n_in*W] bf16 TILE
- `coef_exp` [E, nnz*32] bf16 TILE — each nonzero k's coefficient broadcast across a 32-col tile
              (host: `np.repeat(coef[E,nnz], 32, axis=1)`)
- `deg[i]`   nonzeros feeding output block i; `ks`/`js` (len nnz) the coef-tile index and input
              block for each nonzero, grouped by output block i in order.
- returns [E, n_out*W].

Compute: per tile-row (32 edges), per output block i, per w-tile: `mul_tiles` each nonzero's
coef-tile * x-block into dst[0..d-1], then SFPU `add_binary_tile` sums into dst[0]. Needs
`dst_full_sync_en` (8 fp32 dst slots; d<=5 for lmax=2). HiFi4, fp32 dest accumulate.

### Build (source ttnn at ~/tt-metal, HEAD b5522097)
Files live at `~/tt-metal/ttnn/cpp/ttnn/operations/experimental/fused_rotate/` (this dir is the
backup copy). Registration edits (4 points):
  1. ttnn/CMakeLists.txt: `add_subdirectory(.../fused_rotate)` + link `TTNN::Ops::Experimental::FusedRotate`
  2. ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp: include hpp + `fr_detail::bind_fused_rotate(mod)`
  3. ttnn/sources.cmake: add `.../fused_rotate/fused_rotate_nanobind.cpp` (undefined-symbol trap)
Build: `ninja -C ~/tt-metal/build ttnn && cp ~/tt-metal/build_Release/ttnn/_ttnn.so ~/tt-metal/ttnn/ttnn/_ttnn.so`
Run:   `TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal/ttnn:~/TT-Atom python ...`
