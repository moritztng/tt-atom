"""Shared harness for the hand-run benchmarks — timing, fixture location, fleet discipline.

``benchmarks/`` is the one place that both times a device and has to coexist with sibling fleet
jobs, and until this module existed each script carried its own copy of the same mechanisms.
Everything here is either the repo's existing implementation (``tests.util.GOLDEN_DIR``,
``tt_atom.orb_weight_cache``) reached from ``benchmarks/``, or the mechanism the subprocess
benchmarks already shared by copy.

Scripts run from the repo root (``python benchmarks/<name>.py``), so ``import _harness`` resolves
through the script directory.
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import pwd
import socket
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --- timing and provenance ------------------------------------------------------------------------------------

def median_ms(fn, n, warm, sync=None):
    """Median wall-clock of ``fn`` in ms over ``n`` calls after ``warm`` untimed ones.

    ``sync`` is called after the warmup and after every timed call, for the benchmarks that time
    a fire-and-forget device submission rather than a whole calculator call: without it the last
    op's cost lands in the next sample.
    """
    import numpy as np

    for _ in range(warm):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def mean_s(fn, iters):
    """Mean wall-clock of ``fn`` in SECONDS over ``iters`` calls, after one untimed call to fill
    the program cache for this shape. The unit is in the name because the callers report both a
    per-step time and a systems-per-second rate from it."""
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def git_sha():
    """HEAD of the checkout being measured, recorded in every result file, or ``None`` when that
    cannot be determined (no git, not a checkout)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


# --- fixtures ----------------------------------------------------------------------------------

def real_home() -> pathlib.Path:
    """The invoking user's home from the passwd database, NOT ``$HOME``: the sandbox-HOME legs
    override ``$HOME`` to control the kernel cache, and the fleet lease and the weight caches must
    still land in the real home."""
    return pathlib.Path(pwd.getpwuid(os.getuid()).pw_dir)


def golden_dir() -> pathlib.Path:
    """The real-weight golden directory — ``tests.util``'s, so ``TTATOM_GOLDEN_DIR`` relocates the
    benchmarks' fixtures exactly as it relocates the tests' and the release gate's."""
    from tests.util import GOLDEN_DIR

    return GOLDEN_DIR


def orb_weights(checkpoint: str = "orb-v3-conservative-inf-omat") -> pathlib.Path:
    """Path to an exported Orb checkpoint, via ``tt_atom.orb_weight_cache`` (so ``TT_ATOM_CACHE``
    relocates it, and an unknown checkpoint name raises here rather than as a missing file).

    The default root is re-rooted under :func:`real_home`: a child leg runs with ``$HOME`` pointed
    at its sandbox cache and would otherwise resolve the weights inside it.
    """
    from tt_atom import orb_weight_cache as OWC

    cache_dir = None if os.environ.get("TT_ATOM_CACHE") else \
        real_home() / ".cache" / "tt_atom" / "orb_weights"
    return OWC.weights_path(checkpoint, cache_dir=cache_dir)


def conformers(k, mol, seed0=10):
    """``k`` rattled copies of an ASE molecule, the batch benchmarks' fixture: same composition
    (so a UMA batch is legal) and different geometry (so no result is reused)."""
    from ase.build import molecule as ase_molecule

    out = []
    for i in range(k):
        a = ase_molecule(mol)
        a.rattle(stdev=0.08, seed=seed0 + i)
        a.info.update(charge=0, spin=1)
        out.append(a)
    return out


# --- fleet discipline --------------------------------------------------------------------------

LEASES = real_home() / ".coworker" / "state" / "leases"
HOLDER = os.environ.get("TT_BIO_LEASE_HOLDER", "tt-atom-benchmark")


_HELD_LEASES = []


def take_lease(card):
    """Take the SAME exclusive flock tt_bio's device_lease uses, before opening the card, so a
    sibling fleet job serializes with us instead of colliding on the PCI device. Returns the
    seconds spent waiting for it.

    The fd is kept for the life of the process here, not handed back: the lock must outlive the
    device, and the kernel releases it at exit — which is what makes the ``os._exit`` teardown
    skip in the child legs safe.
    """
    LEASES.mkdir(parents=True, exist_ok=True)
    lease_path = LEASES / f"{socket.gethostname()}-card{card}.json"
    fd = os.open(lease_path, os.O_RDWR | os.O_CREAT)
    _HELD_LEASES.append(fd)
    t0 = time.perf_counter()
    fcntl.flock(fd, fcntl.LOCK_EX)
    waited = time.perf_counter() - t0
    with open(lease_path, "w") as f:
        json.dump({"host": socket.gethostname(), "card": str(card),
                   "holder": HOLDER, "pid": os.getpid(),
                   "acquired": time.time(), "released": None}, f)
    return waited


def sandbox_env(home, card):
    """Child environment for one leg: ``$HOME`` (and ``XDG_CACHE_HOME``) pointed at ``home``, so
    the tt-metal persistent kernel cache under it is exactly controlled — a fresh dir is cold,
    a populated one warm — and the card pinned."""
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(pathlib.Path(home) / ".cache")
    env["TT_VISIBLE_DEVICES"] = str(card)
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def cache_stats(home):
    """(file count, total bytes) under the tt-metal kernel cache of a sandbox HOME."""
    root = pathlib.Path(home) / ".cache"
    n, b = 0, 0
    for p in root.rglob("*"):
        if p.is_file():
            n += 1
            b += p.stat().st_size
    return n, b


def host_quiet():
    """True when no sibling fleet device job is running on this host. The sibling audit's legs
    announce as ``tt_bio.main`` / ``chain*.sh`` processes (its embed fanout does NOT take the
    lease flock, so process liveness is the only reliable signal). sampler.py is a harmless
    1 Hz CPU monitor and is ignored."""
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "tt_bio" in line or "tt-bio-dev/env" in line:
            return False                                    # sibling device job or shard worker
        if "mcscale" in line and ".sh" in line:
            return False                                    # sibling campaign script
    return True


def wait_for_quiet(poll_s=15, settle_s=10, max_wait_s=2400):
    """Block until the host has been continuously quiet for ``settle_s`` seconds."""
    t0 = time.time()
    quiet_since = None
    while time.time() - t0 < max_wait_s:
        if host_quiet():
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= settle_s:
                return True
        else:
            quiet_since = None
        time.sleep(poll_s)
    return False
