"""Atomic ``.npz`` materialization: sidecar in the destination directory, then ``os.replace``.

An interrupted export must never leave a truncated file at the final name, because every consumer
gates on presence (``scripts/release_gate.py`` checks ``.exists()``) rather than on integrity.

Two producers share it: the seven in-process exporters call ``savez_atomic``, and
``tt_atom.bundle_cache.run_export`` drives a reference-env subprocess into the same sidecar. Each
export gets its own sidecar, so concurrent first-use processes cannot overwrite each other.

This lives in ``tools/`` rather than ``tt_atom/`` because most of its callers run in the
*reference* environment (fairchem / orb-models, numpy>=2), where ``tt_atom`` is not installed.
``tools`` ships in the wheel, so the ttnn-side caller can import it too.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import tempfile

import numpy as np


def npz_path(out):
    """``out`` with the ``.npz`` suffix applied, matching ``np.savez``'s own behavior."""
    out = pathlib.Path(out)
    return out if out.suffix == ".npz" else out.with_suffix(".npz")


@contextlib.contextmanager
def atomic_npz(out):
    """Yield a sidecar path to write; on a clean exit rename it onto ``out``, else delete it.

    The sidecar keeps the ``.npz`` suffix so ``np.savez`` does not append a second extension.
    """
    out = npz_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=f".{out.name}.", suffix=".npz", delete=False
    ) as sidecar:
        tmp = pathlib.Path(sidecar.name)
    try:
        yield tmp
    except BaseException:                    # KeyboardInterrupt is the case this exists for
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)


def savez_atomic(out, **arrays):
    """``np.savez`` into a sidecar, then rename onto ``out``. Returns the final path."""
    with atomic_npz(out) as tmp:
        np.savez(tmp, **arrays)
    return npz_path(out)
