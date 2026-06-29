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

from contextlib import contextmanager


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
