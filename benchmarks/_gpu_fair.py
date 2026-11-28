"""Model-independent half of the fair GPU legs (`orb_gpu_bench_fair.py`, `mace_gpu_bench_fair.py`):
SM-clock sampling during the timed loop, the timed loop itself, the host-sync probe, whole-step
CUDA-graph capture and the result file. Everything a leg adds on top is its package's own API."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone

import numpy as np
import torch

from _harness import git_sha


class ClockSampler:
    """Samples the SM clock (MHz) via nvidia-smi every 0.2 s while active."""

    def __init__(self):
        self.mhz, self._stop = [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                self.mhz.append(int(out.split()[0]))
            except Exception:  # noqa: BLE001 -- a missing sample must not kill the timing
                pass
            self._stop.wait(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def sync_probe(run, pkg):
    """Log every host-syncing site of one eager call; any of them breaks a CUDA-graph capture."""
    import traceback
    import warnings
    sites = []

    def record(*_a, **_k):
        frames = [f for f in traceback.extract_stack()[:-1] if pkg in f.filename]
        site = " <- ".join(f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in frames[-3:][::-1])
        if site not in sites:
            sites.append(site)
    torch.cuda.set_sync_debug_mode("warn")
    old, warnings.showwarning = warnings.showwarning, record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = record
            run()
    finally:
        warnings.showwarning = old
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    print(f"    sync probe: {len(sites)} host-syncing site(s)" + "".join("\n      " + x for x in sites), flush=True)


def sync():
    """Wait for the GPU; a no-op on a CPU smoke test."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_steps(step, warm, n):
    for _ in range(warm):
        step()
    sync()
    ts = []
    with ClockSampler() as clk:
        for _ in range(n):
            t0 = time.perf_counter()
            step()
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
    return {"step_ms_raw": [round(x, 4) for x in ts], "step_ms_median": float(np.median(ts)),
            "steps_per_s": 1e3 / float(np.median(ts)),
            "sm_clock_mhz_during": (int(np.median(clk.mhz)) if clk.mhz else None),
            "sm_clock_mhz_min_max": ([min(clk.mhz), max(clk.mhz)] if clk.mhz else None)}


def capture(run, pkg, settle=None):
    """Capture one whole step (forward + autograd forces) into a CUDA graph and return its replay.
    Warm on a side stream first, log the host-syncing sites of `pkg` (any of them breaks the
    capture), and let `settle(stream)` finish whatever must not happen inside it (compilation)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    sync_probe(run, pkg)
    if settle:
        settle(s)
    torch.cuda.empty_cache()  # the capture allocates from its own pool: give it the warm-up's memory back
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        static_out = run()

    def replay():
        g.replay()
        return static_out
    return replay


def dump(path, meta, records, protocol, warmup, steps):
    with open(path, "w") as fh:
        json.dump({**meta, "git_sha": git_sha(),
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol": protocol, "warmup": warmup, "steps": steps,
                   "records": records}, fh, indent=1)
