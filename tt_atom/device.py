"""Device + kernel-configuration helpers for TT-Atom.

Everything that touches a Tenstorrent card goes through here so the numerics policy lives
in one place. The policy (validated on Blackhole p150):

  * matmul accumulation in fp32 (``fp32_dest_acc_en=True``) at ``HiFi4`` fidelity, with
    ``packer_l1_acc=True`` -- this is what gives matmul PCC ~1.0 vs torch.
  * weights default to ``bfloat16``; ``--fast`` mode may store weights as ``bfloat8_b`` but
    keeps the fp32/HiFi4 accumulation above (only the operands get cheaper, not the math).

``import ttnn`` is done lazily inside functions so that ``import tt_atom`` never opens or
probes a device.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

# Three measured wins landed on the source-ttnn build (branch ``moritztng/tt-atom``, carrying
# UMA's custom ``fused_rotate``/``fused_rotate_gc``/``fused_gate``/``fused_ln_bw`` kernels):
#   * ``fused_lnbw``  -- one-kernel radial-LayerNorm backward vs ~15 ttnn ops. A pure kernel fuse,
#                       win at every system size -> default ON when the kernel is present.
#   * ``device_ede`` -- moves the radial-MLP fwd+bw (the largest per-step host cost) onto the device
#                       inside the captured trace. Size-dependent: a ~2x traced-step win at large
#                       systems (512 atoms -> 2.00x, 216 -> 1.87x) but a regression on small ones
#                       (9 atoms / E=72 -> 0.85x: the device path's overhead exceeds the tiny host
#                       radial cost it saves). Default OFF; opt in with TT_ATOM_DEVICE_EDE=1 for
#                       bulk/large MD. A per-graph size gate is the right eventual default but needs
#                       the EdgeDegreeEmbedding constructed lazily per graph (not at Backbone build).
#   * ``bf8_edge``   -- bf8 edge-activation dataflow (DRAM-bandwidth win on the fat [E,nsph*C] mats).
#                       Same size-dependence: win at large E, typecast overhead dominates at small E.
#                       Default OFF; opt in with TT_ATOM_BF8_EDGE=1 for large systems.
# Re-measured 2026-07-12 on current master (real uma-s bundle, Blackhole p150, card 0).
#
# The ``=1`` opt-in is honored only when the capability is present, so setting it on a stock ttnn
# wheel (e.g. 0.68, which lacks the custom kernels) downgrades to the safe path instead of crashing
# -- the ESMFold2 ``fuse_swiglu`` build-probe pattern. ``=0`` always forces off. The *default* is
# the capability probe for fused_lnbw and OFF for the two size-dependent wins.
_CAP_CACHE: dict = {}


def _experimental_ops():
    """The ttnn experimental-ops module, or ``None`` if ttnn / the submodule isn't importable.
    Imported lazily so ``import tt_atom`` never pulls ttnn or opens a device."""
    try:
        import ttnn._ttnn.operations.experimental as e  # noqa: PLC0415
        return e
    except Exception:
        return None


def _cap(*ops: str) -> bool:
    """Cached probe: are all ``ops`` present on the installed ttnn's experimental module? False on
    stock ttnn wheels that lack UMA's custom kernels (the fallback trigger)."""
    key = "cap:" + ",".join(ops)
    return _CAP_CACHE.setdefault(key, bool(
        (e := _experimental_ops()) is not None and all(hasattr(e, o) for o in ops)))


def _flag(env: str, *, default_on: bool, cap: bool) -> bool:
    """``=1`` forces on (but only where the capability ``cap`` holds -- silently no-op on a stock
    ttnn that lacks the kernel, instead of crashing); ``=0`` forces off; unset -> ``cap`` when
    ``default_on`` else OFF."""
    v = os.environ.get(env)
    if v == "1":
        return cap
    if v == "0":
        return False
    return cap if default_on else False


def device_ede() -> bool:
    """Whether the edge-degree embedding (node init) is computed on device inside the trace.

    Moves the largest per-step host cost (the radial-MLP fwd+bw over E edges) onto the device inside
    the captured trace (see tt_atom/edge_degree.py). Size-dependent: ~2x traced-step win at large
    systems, a small regression under ~100 edges (the device path overhead exceeds the host radial
    cost it saves). Default OFF; ``TT_ATOM_DEVICE_EDE=1`` opts in (honored only on the source-ttnn
    build carrying ``fused_rotate``)."""
    return _flag("TT_ATOM_DEVICE_EDE", default_on=False, cap=_cap("fused_rotate"))


def bf8_edge() -> bool:
    """Whether the edgewise message dataflow runs in bfloat8_b (the E-sized [E,nsph*C] activations
    that dominate the bandwidth-bound device replay). The device replay is DRAM-bandwidth bound on
    these activations (bf8 halves the traffic -> ~2x on the fat matmuls at large E; measured), NOT
    compute-bound (HiFi4==LoFi) nor weight-bound (bf8 weights alone = 1.00x). The bf16<->bf8
    boundary sits at the SMALL N-sized node features (gather input / scatter output), so there is no
    per-edge typecast overhead. Node residual stream + norms stay bf16. Size-dependent (typecast
    overhead dominates at small E). Default OFF; ``TT_ATOM_BF8_EDGE=1`` opts in (honored only on the
    source-ttnn build whose ``fused_rotate``/``fused_rotate_gc``/``fused_gate`` kernels accept bf8)."""
    return _flag("TT_ATOM_BF8_EDGE", default_on=False,
                 cap=_cap("fused_rotate", "fused_rotate_gc", "fused_gate"))


def fused_lnbw() -> bool:
    """Whether the radial-MLP LayerNorm backward routes through the custom ``fused_ln_bw`` kernel
    (one launch: mean/rstd + dx with W L1-resident, vs ~15 ttnn ops). A pure kernel fuse, win at
    every system size. Default ON when ``fused_ln_bw`` is present (source-ttnn build); OFF on stock
    ttnn. ``TT_ATOM_FUSED_LNBW=0/1`` overrides."""
    return _flag("TT_ATOM_FUSED_LNBW", default_on=True, cap=_cap("fused_ln_bw"))


def orb_fused_silu_bw() -> bool:
    """Use ``fused_gate`` for Orb's edge-MLP SiLU VJP when available.

    This replaces six DRAM-backed elementwise programs with one fused device program. It is
    profitable for the edge-sized tensors and defaults on in the source-ttnn build; stock wheels
    keep the ordinary ttnn path. ``TT_ATOM_ORB_FUSED_SILU_BW=0/1`` overrides.
    """
    return _flag("TT_ATOM_ORB_FUSED_SILU_BW", default_on=True, cap=_cap("fused_gate"))


def edge_dtype(ttnn):
    """bfloat8_b when bf8_edge() else bfloat16 — the working dtype of the edgewise E-sized flow."""
    return ttnn.bfloat8_b if bf8_edge() else ttnn.bfloat16


# Budgets (bytes) for a single L1-resident intermediate. The L1-residency perf wins (grid, so2,
# norm chains) keep intermediates on-chip, but L1 is small (~1.4 MB/bank x 130) and op circular
# buffers grow with problem size, so at large N/E they consume nearly all L1 and an L1-resident
# activation OOMs (or fails trace capture with "writes not supported"). Residency is gated
# per-tensor: L1 only when the estimated tile-padded byte size fits the relevant budget, else
# DRAM-interleaved (the pre-optimization path -- correct at any size, just no L1 speedup).
#
# Two budgets because the crossovers differ (empirically verified on Blackhole p150):
#   * SO2 edge tensors [E, nsph*Cin]: L1 fits at E<=2234 (N=128), fails at E=4834 (N=250).
#     12 MB -> edge cap ~2600 (see SO2Convolution.l1_max_edges).
#   * GRID/NORM node tensors: pass the TRUE tile-padded feature width (the 3D [N,nsph,C]/[N,npts,C]
#     tensors pad the coeff dim up to a tile: nsph 9->32, npts 42->64 -- a 1.5-3.5x blowup a naive
#     nsph*C estimate misses). Verified real-byte crossovers: grid L1 fits at N=432 (~7MB), fails
#     at N=686 (~11MB); norm fits at N=686 (~5.6MB), fails at N=1024 (~8.4MB). 8 MB covers both.
L1_RESIDENCY_BUDGET = 12_000_000        # SO(2) edge tensors [E, nsph*Cin] (flat, no coeff padding)
L1_NODE_BUDGET = 8_000_000              # grid / norm node tensors (pass tile-padded feature width)


def coeff_reshape(ttnn, t, shape):
    """Reshape that collapses/expands the spherical-harmonic coefficient dim (nsph, e.g. 9),
    which is NOT tile-aligned. A direct ``ttnn.reshape`` on a TILE tensor physically repacks the
    tile padding (9 -> 32, a 3.5x data reorg -> ~18 ms at E~46k); routing through ROW_MAJOR
    (contiguous, no coeff padding) is ~4x faster and bit-exact (a lossless layout round-trip, no
    dtype change). For an already-ROW_MAJOR tensor a plain reshape is cheap, so pass through.

    Only use for reshapes that move the coefficient dim across the flat/3D boundary; a reshape
    that only touches the batch (outer) dim never repacks and should stay a direct reshape."""
    try:
        is_tile = t.layout == ttnn.TILE_LAYOUT
    except Exception:
        is_tile = True
    if is_tile:
        r = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        r = ttnn.reshape(r, shape)
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)
    return ttnn.reshape(t, shape)


def l1_if_fits(ttnn, rows, width, *, dtype_bytes=2, budget=L1_RESIDENCY_BUDGET):
    """Return an L1 memory config if a tile-padded ``[rows, width]`` tensor of ``dtype_bytes``
    fits the per-tensor L1 residency budget, else DRAM-interleaved. Guards the residency wins so
    large systems degrade gracefully to DRAM instead of OOMing the trace."""
    rp = ((rows + 31) // 32) * 32
    wp = ((width + 31) // 32) * 32
    return ttnn.L1_MEMORY_CONFIG if rp * wp * dtype_bytes <= budget else ttnn.DRAM_MEMORY_CONFIG


def compute_kernel_config(fast: bool = False):
    """The TT-Atom matmul/compute numerics policy.

    ``fast`` does not change the accumulation math here (operand dtype is chosen at weight
    load time); HiFi4 + fp32 dest accumulation is kept in both modes for accuracy.
    """
    import ttnn

    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def open_device(device_id: int = 0, *, l1_small_size: int = 0, trace_region_size: int = 0):
    """Open a single Tenstorrent device with the program cache enabled.

    The program cache is what makes warm calls cheap (kernels are compiled once); device
    residency + trace capture build on top of it.
    """
    import ttnn

    dev = ttnn.open_device(
        device_id=device_id,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
    )
    dev.enable_program_cache()
    return dev


@contextmanager
def device(device_id: int = 0, **kwargs):
    """Context manager that opens a device and guarantees it is closed."""
    import ttnn

    dev = open_device(device_id, **kwargs)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)
