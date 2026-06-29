"""Shared parity-test fixtures: a single TT device for the session and the golden loader."""
import os

import pytest

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

from util import Golden  # noqa: E402  (pytest puts tests/ on sys.path)


@pytest.fixture(scope="session")
def golden():
    return Golden("golden_tiny.npz")


@pytest.fixture(scope="session")
def device():
    from tt_atom import device as D

    dev = D.open_device(0)
    yield dev
    import ttnn

    ttnn.close_device(dev)
