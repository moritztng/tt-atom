"""Shared parity-test fixtures: a single TT device for the session and the golden loader."""
import os

import pytest

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

from util import Golden  # noqa: E402  (pytest puts tests/ on sys.path)


@pytest.fixture(scope="session")
def golden():
    return Golden("golden_tiny.npz")


@pytest.fixture(scope="module")
def gw(request):
    """The requesting module's Orb weights, loaded from its own ``REAL_GOLDEN``.

    Seven parity modules each score a different Orb golden and each held a byte-identical copy of
    this fixture. The golden is what differs between them, and it is already a module constant.
    """
    from tt_atom.orb_weights import OrbWeights

    return OrbWeights.load(request.module.REAL_GOLDEN)


@pytest.fixture(scope="session")
def device():
    from tt_atom import device as D

    # reserve a trace region so the trace-path test can capture on this shared device; harmless
    # (only reserves DRAM) for the eager parity tests.
    dev = D.open_device(0, trace_region_size=400_000_000)
    yield dev
    import ttnn

    ttnn.close_device(dev)
