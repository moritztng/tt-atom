"""Composition-cached uma bundle factory — the machinery behind ``TTAtomCalculator.from_uma``.

UMA's MoLE routing is baked at merge time, so one merged bundle is valid for exactly one
*(reduced composition, charge, spin, task)*. This module turns those into a stable cache key and,
on a miss, transparently builds the bundle by invoking the reference (fairchem, numpy>=2)
environment as a subprocess — the ttnn-side user never has to touch the two-env split for the
common path. A cache *hit* needs no fairchem at all: it is a plain ``np.load``.

Two inherent frictions this hides (see README): the per-composition merge, and the fact that ttnn
(numpy<2) and fairchem (numpy>=2) cannot live in one interpreter. Both are automated + cached here
so a scientist sees them at most once per composition.
"""
from __future__ import annotations

import hashlib
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter
from functools import reduce

CACHE_DIR = pathlib.Path(
    os.environ.get("TT_ATOM_CACHE", pathlib.Path.home() / ".cache" / "tt_atom" / "bundles")
)


def infer_task(atoms):
    """Zero-config task default: a fully periodic cell -> ``'omat'`` (bulk materials), otherwise
    ``'omol'`` (molecules). Slabs / MOFs / molecular crystals (oc20/odac/omc) should pass the task
    explicitly — this only picks the right *common* default so the bare entry point Just Works."""
    import numpy as np

    return "omat" if np.asarray(atoms.get_pbc()).all() else "omol"


def reduced_composition(numbers):
    """Return sorted ``((Z, reduced_count), ...)``.

    MoLE routes on the *fractional* composition, so H2O and H4O2 share a bundle — we divide the
    counts by their GCD to make the key scale-invariant."""
    counts = Counter(int(z) for z in numbers)
    g = reduce(math.gcd, counts.values())
    return tuple(sorted((z, c // g) for z, c in counts.items()))


def formula(numbers):
    """Human-readable reduced formula (e.g. ``C2H6O``) for logging."""
    from ase.data import chemical_symbols

    return "".join(f"{chemical_symbols[z]}{c if c > 1 else ''}"
                   for z, c in reduced_composition(numbers))


def composition_hash(numbers):
    comp = reduced_composition(numbers)
    s = ";".join(f"{z}:{c}" for z, c in comp)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def bundle_path(model, task, numbers, charge, spin, cache_dir=None):
    """Deterministic cache path for a merged bundle. Same composition/charge/spin/task/model ->
    same file, regardless of atom ordering or an integer scaling of the counts."""
    h = composition_hash(numbers)
    name = f"{model}_{task}_{h}_c{int(charge)}_s{int(spin)}.npz"
    return pathlib.Path(cache_dir or CACHE_DIR) / name


def resolve_refenv(refenv=None):
    """Locate the reference (fairchem) python: explicit arg > ``$TT_ATOM_REFENV`` > default
    ``~/.ttatom_run/refenv/bin/python``. Raises a clear, actionable error if none is found —
    only ever called on a cache *miss* (a hit needs no refenv)."""
    candidates = [
        refenv,
        os.environ.get("TT_ATOM_REFENV"),
        str(pathlib.Path.home() / ".ttatom_run" / "refenv" / "bin" / "python"),
    ]
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return c
    raise RuntimeError(
        "No reference (fairchem) environment found to build the uma bundle.\n"
        "The one-time MoLE merge needs fairchem (numpy>=2), which cannot share the ttnn env\n"
        "(numpy<2). Create it once with this one command:\n\n"
        "  python -m venv ~/.ttatom_run/refenv && "
        "~/.ttatom_run/refenv/bin/pip install 'fairchem-core>=2.10'\n\n"
        "or point TT-Atom at an existing fairchem env via TT_ATOM_REFENV=/path/to/bin/python\n"
        "(or the refenv= argument). A cached bundle needs no refenv — this is only the\n"
        "first-use build per composition."
    )


def build_bundle(atoms, out_path, *, model="uma-s-1", task="omol", charge=0, spin=1,
                 refenv=None, checkpoint=None):
    """Merge + export a bundle for ``atoms`` by running ``tools/export_weights.py`` in the
    reference env. Writes atomically (build to a sidecar, then ``os.replace``) so an interrupted
    build can never leave a half-written file that later looks like a cache hit."""
    if model != "uma-s-1":
        raise ValueError(
            f"auto-build supports model='uma-s-1'; got {model!r}. Export other checkpoints "
            "manually with tools/export_weights.py and load the .npz directly."
        )
    py = resolve_refenv(refenv)
    tools = pathlib.Path(__file__).resolve().parent.parent / "tools" / "export_weights.py"
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # sidecar ends in .npz so np.savez does not append a second extension
    tmp_out = out_path.with_name(out_path.name + ".building.npz")
    with tempfile.TemporaryDirectory() as td:
        from ase.io import write

        xyz = pathlib.Path(td) / "structure.xyz"
        write(str(xyz), atoms)
        cmd = [py, str(tools), "--uma-s-1", "--xyz", str(xyz), "--task", task,
               "--charge", str(int(charge)), "--spin", str(int(spin)), "--out", str(tmp_out)]
        if checkpoint:
            cmd += ["--checkpoint", str(checkpoint)]
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(
                f"reference-env bundle build failed (exit {e.returncode}). Command:\n  "
                + " ".join(cmd)
                + "\n\nIf this is a checkpoint/access error: the UMA weights are gated — accept the "
                "license at\n  https://huggingface.co/facebook/UMA\nand log in once with "
                "`huggingface-cli login` (or set HF_TOKEN) in the reference env."
            ) from e
    os.replace(tmp_out, out_path)
    return out_path


def get_or_build(atoms, *, model="uma-s-1", task="omol", charge=0, spin=1, refenv=None,
                 checkpoint=None, cache_dir=None, log=True):
    """Return the cache path for this system's bundle, building it on a miss. Pure I/O + subprocess
    — no ttnn, no device. ``TTAtomCalculator.from_uma`` wraps this and returns a calculator."""
    numbers = atoms.get_atomic_numbers()
    path = bundle_path(model, task, numbers, charge, spin, cache_dir=cache_dir)
    if path.exists():
        return path
    if log:
        print(
            f"[tt-atom] building {model} bundle for composition {formula(numbers)} "
            f"(task={task}, charge={int(charge)}, spin={int(spin)}) — one-time per composition, "
            f"~30s via the reference env...",
            file=sys.stderr, flush=True,
        )
    build_bundle(atoms, path, model=model, task=task, charge=charge, spin=spin,
                 refenv=refenv, checkpoint=checkpoint)
    if log:
        print(f"[tt-atom] cached bundle -> {path}", file=sys.stderr, flush=True)
    return path
