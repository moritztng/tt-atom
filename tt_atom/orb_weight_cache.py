"""Per-checkpoint weight cache — the machinery behind ``Orb(atoms)``.

Orb has no MoLE (or any) expert routing baked at merge time (see ``docs/orb-port.md``'s
architecture verdict): the raw checkpoint weights are valid for *any* composition/charge/spin, so
unlike ``tt_atom.bundle_cache`` (one merged bundle per *(composition, charge, spin, task)*, a
subprocess rebuild per system) this only ever needs one export per *checkpoint name*, ever. A
cache hit is a plain ``np.load``, exactly like ``bundle_cache``'s.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

from .bundle_cache import resolve_refenv  # same refenv resolution UMA uses, no Orb-specific logic

CACHE_DIR = pathlib.Path(
    os.environ.get("TT_ATOM_CACHE", pathlib.Path.home() / ".cache" / "tt_atom")
) / "orb_weights"

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
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools" / "export_orb_weights.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = path.with_name(path.name + ".building.npz")
    cmd = [py, str(tools), "--ckpt", _short_name(checkpoint), "--out", str(tmp_out)]
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(
            f"reference-env Orb weight export failed (exit {e.returncode}). Command:\n  "
            + " ".join(cmd)
        ) from e
    os.replace(tmp_out, path)
    if log:
        print(f"[tt-atom] cached -> {path}", file=sys.stderr, flush=True)
    return path
