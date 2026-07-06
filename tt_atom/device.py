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


def device_ede() -> bool:
    """Whether the edge-degree embedding (node init) is computed on device inside the trace.

    Default OFF (host torch, the original path). When ``TT_ATOM_DEVICE_EDE=1`` the radial MLP ->
    rotate-back -> envelope -> scatter -> +l0 chain runs on device (see tt_atom/edge_degree.py),
    removing the largest per-step host cost (the radial-MLP fwd+bw over E edges). Read at call
    time so tests / benches can toggle it per run."""
    return os.environ.get("TT_ATOM_DEVICE_EDE") == "1"


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


def open_mesh(device_ids, *, l1_small_size: int = 0, trace_region_size: int = 0):
    """Open a row mesh over ``device_ids`` for multi-card throughput.

    Requires ``TT_MESH_GRAPH_DESC_PATH`` to point at the matching fabric descriptor
    (e.g. ``p150_x4_mesh_graph_descriptor.textproto`` for a 4-card QuietBox).
    """
    import ttnn

    ids = list(device_ids)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(ids)),
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        device_ids=ids,
    )
    mesh.enable_program_cache()
    return mesh


@contextmanager
def device(device_id: int = 0, **kwargs):
    """Context manager that opens a device and guarantees it is closed."""
    import ttnn

    dev = open_device(device_id, **kwargs)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)
