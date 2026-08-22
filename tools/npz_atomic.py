"""Atomic ``.npz`` write: sidecar in the destination directory, then ``os.replace``.

An interrupted export must never leave a truncated file at the final name, because every consumer
gates on presence (``scripts/release_gate.py`` checks ``.exists()``) rather than on integrity.
Mirrors ``tt_atom/bundle_cache.py:run_export``, which does the same for subprocess exports.

This lives in ``tools/`` rather than ``tt_atom/`` because all its callers run in the *reference*
environment (fairchem / orb-models, numpy>=2), where ``tt_atom`` is not installed.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np


def savez_atomic(out, **arrays):
    """``np.savez`` into a sidecar, then rename onto ``out``. Returns the final path."""
    out = pathlib.Path(out)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")        # np.savez appends .npz; keep that behavior
    out.parent.mkdir(parents=True, exist_ok=True)
    # The suffix stays .npz so np.savez does not append a second extension.
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=f".{out.name}.", suffix=".npz", delete=False
    ) as sidecar:
        tmp = pathlib.Path(sidecar.name)
    try:
        np.savez(tmp, **arrays)
    except BaseException:                    # KeyboardInterrupt is the case this exists for
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return out
