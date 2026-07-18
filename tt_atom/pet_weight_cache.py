"""Per-checkpoint weight cache for the PET-MAD (UPET) port — the machinery behind
``Calculator(atoms, "pet-mad-s-v1.5.0")``.

PET-MAD bakes no MoLE / per-composition routing into its weights (the raw checkpoint
is valid for any composition, like Orb), so this is a plain one-time per-checkpoint
export, read back by ``tt_atom.pet_weights.PetWeights`` in the ttnn env. Mirrors
``tt_atom.orb_weight_cache``; only the export command (``tools/export_pet_weights.py``,
run in the upet reference env ``~/.ttatom_run/upetenv``) is PET-specific.
"""
from __future__ import annotations

import os
import pathlib
import sys

from .bundle_cache import run_export

CACHE_DIR = pathlib.Path(
    os.environ.get("TT_ATOM_CACHE", pathlib.Path.home() / ".cache" / "tt_atom")
) / "pet_weights"

CHECKPOINTS = ("pet-mad-s-v1.5.0",)


def _short_name(checkpoint):
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown PET-MAD checkpoint {checkpoint!r}; choose from {CHECKPOINTS}")
    return checkpoint


def weights_path(checkpoint, cache_dir=None):
    return pathlib.Path(cache_dir or CACHE_DIR) / f"{_short_name(checkpoint)}.npz"


def _resolve_upetenv(refenv=None):
    """Locate the upet reference python: explicit arg > ``$TT_ATOM_UPETENV`` >
    ``~/.ttatom_run/upetenv/bin/python``. Only called on a cache *miss* — a hit needs no
    refenv. (Distinct from ``bundle_cache.resolve_refenv``'s fairchem env: PET-MAD's
    export needs ``upet`` + ``metatrain``, not ``fairchem-core``.)"""
    candidates = [
        refenv,
        os.environ.get("TT_ATOM_UPETENV"),
        str(pathlib.Path.home() / ".ttatom_run" / "upetenv" / "bin" / "python"),
    ]
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return c
    raise RuntimeError(
        "No upet reference environment found to export PET-MAD weights.\n"
        "The one-time export needs upet + metatrain (with the shared refenv torch/numpy\n"
        "reused via a .pth, see tools/export_pet_weights.py). Create it once, or point\n"
        "tt-atom at an existing one via TT_ATOM_UPETENV=/path/to/bin/python (or the\n"
        "refenv= argument). A cached weight file needs no refenv — this is only the\n"
        "first-use export per checkpoint."
    )


def get_or_build(checkpoint, *, refenv=None, cache_dir=None, log=True):
    """Return the cache path for ``checkpoint``'s weights, exporting on a miss. Pure I/O +
    subprocess — no ttnn, no device. The export runs in the upet reference env
    (``~/.ttatom_run/upetenv``) which has ``upet`` + ``metatrain`` alongside the shared
    refenv torch/numpy (see ``tools/export_pet_weights.py``)."""
    path = weights_path(checkpoint, cache_dir=cache_dir)
    if path.exists():
        return path
    if log:
        print(f"[tt-atom] exporting {checkpoint} weights — one-time, via the upet reference "
              f"env...", file=sys.stderr, flush=True)
    py = _resolve_upetenv(refenv)
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools" / "export_pet_weights.py"
    run_export(path, lambda tmp_out: [py, str(tools), "--ckpt", _short_name(checkpoint),
                                      "--out", str(tmp_out)])
    if log:
        print(f"[tt-atom] cached -> {path}", file=sys.stderr, flush=True)
    return path
