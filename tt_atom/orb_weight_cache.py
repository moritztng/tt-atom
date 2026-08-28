"""Per-checkpoint weight cache — the machinery behind ``Calculator(atoms, "orb-...")``.

Orb has no MoLE or other expert routing baked at merge time (see ``docs/orb-port.md``): the raw
checkpoint weights are valid for *any* composition/charge/spin, so unlike
``tt_atom.bundle_cache`` (one merged bundle per *(composition, charge, spin, task)*, a subprocess
rebuild per system) this only ever needs one export per *checkpoint name*, ever. A cache hit is a
plain ``np.load``, exactly like ``bundle_cache``'s. The cache root, the refenv resolution and the
atomic subprocess-export mechanics all come from ``bundle_cache`` (``CACHE_ROOT`` /
``resolve_refenv`` / ``run_export``); only the export command is Orb-specific.
"""
from __future__ import annotations

import pathlib
import sys

from .bundle_cache import CACHE_ROOT, exporter_path, resolve_refenv, run_export

CACHE_DIR = CACHE_ROOT / "orb_weights"

CHECKPOINTS = ("orb-v3-conservative-inf-omat", "orb-v3-direct-20-omat",
              "orb-v3-conservative-omol", "orb-v3-direct-omol")


def _short_name(checkpoint):
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown Orb checkpoint {checkpoint!r}; choose from {CHECKPOINTS}")
    return checkpoint.removeprefix("orb-v3-")


def weights_path(checkpoint, cache_dir=None):
    return pathlib.Path(cache_dir or CACHE_DIR) / f"{_short_name(checkpoint)}.npz"


def get_or_build(checkpoint, *, refenv=None, cache_dir=None, log=True):
    """Return the cache path for ``checkpoint``'s weights, exporting on a miss. Pure I/O +
    subprocess — no ttnn, no device."""
    path = weights_path(checkpoint, cache_dir=cache_dir)
    if path.exists():
        return path
    if log:
        print(f"[tt-atom] exporting {checkpoint} weights — one-time, ~10s via the reference "
             f"env...", file=sys.stderr, flush=True)
    py = resolve_refenv(refenv)
    tools = exporter_path("export_orb_weights.py")
    run_export(path, lambda tmp_out: [py, str(tools), "--ckpt", _short_name(checkpoint),
                                      "--out", str(tmp_out)])
    if log:
        print(f"[tt-atom] cached -> {path}", file=sys.stderr, flush=True)
    return path
