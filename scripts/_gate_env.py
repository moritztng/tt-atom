"""What the gate scripts share: the checkout's root, and the environment a child runs in.

``release_gate.py``, ``ux_regression.py`` and ``_multicard_sim_parity.py`` all drive TT-Atom in
subprocesses, and all three need the same three things in that child: this checkout importable
ahead of any editable or wheel install pointing elsewhere, one visible card, and tt-metal quiet.
Each had its own copy, and they had drifted — ``ux_regression``'s docstring claimed it "matches
the release_gate invocation convention" while omitting the card, and ``_multicard_sim_parity``
hard-set the logger level so a caller could not turn it back up for debugging.

``benchmarks/_harness.py`` is the same idea for ``benchmarks/``; that one owns device leases and
timing, which no gate needs, and its ``sandbox_env`` deliberately redirects ``$HOME`` to control
the kernel cache, which no gate wants.

Scripts run as ``python3 scripts/<name>.py``, so ``import _gate_env`` resolves through the script
directory.
"""
from __future__ import annotations

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def child_env(extra: dict | None = None) -> dict:
    """Environment for a gate subprocess: this checkout importable, one visible card, quiet
    tt-metal. ``extra`` wins over all of it.

    Every value except ``PYTHONPATH`` is a ``setdefault``, so an operator can pin a different
    card or raise the log level from the parent's environment without editing a gate.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    if extra:
        env.update(extra)
    return env
