#!/usr/bin/env python3
"""UX-regression release gate — the user-experience leg of RELEASING.md.

Complements ``scripts/release_gate.py`` (accuracy / no-OOM / perf). This leg does NOT
measure accuracy or speed — it asserts the user-facing *plumbing* every release ships
with still works, headlessly and fast, on a tiny input:

  1. CLI BEHAVES — ``tt-atom run --help`` (and ``relax``/``md``/top-level ``--help``) exit 0
     and list the core flags (``--relax``, ``--md``, ``--steps``, ``--temp``, ``--trace``,
     ``--out``). Pure argparse — no card, no env gap — so this leg runs in GitHub CI too.
  2. OUTPUT FILES PARSE — a real single-point + relax + MD(--steps 5) on a tiny molecule
     (``ase.build.molecule("H2O")``) exit 0, write the ``--out`` geometry, and it parses
     under the strict standard reader (``ase.io.read``) with finite energy and forces
     (not NaN / not empty). Catches the malformed-output class.
  3. LIVE PROGRESS ADVANCES — the per-step print stream the CLI's relax/MD drivers emit
     (``cli.py``'s ``cmd_md``/``cmd_run`` ``_log``: ``  step {n:4d}  E=...  T=... K``;
     ``cmd_relax``/``cmd_run`` ``--relax``: ASE ``FIRE``'s ``Step N`` log via ``logfile="-"``)
     advances through every real step for a multi-step run, not stuck/skipped — the direct
     analogue of tt-bio's "0 -> diffusion" live-progress-bug class.

Orb-only by constraint: the literal ``tt-atom run``/``relax``/``md`` CLI subcommands are
UMA-only (``WeightBundle`` + ``TTAtomCalculator``) and blocked on a host whose ``ttnn``
build lacks the ALWAYS-ON ``fused_rotate`` kernel (memory
``pc-ttatom-env-missing-fused-rotate``). Legs 2-3 therefore drive the *identical* ASE
``FIRE``/``Langevin`` + ``_log`` print pattern the CLI uses, via the Orb ``Calculator`` API
(``OrbCalculator.from_checkpoint("orb-v3-conservative-omol")``) — the stock-``ttnn`` path
with no ``fused_rotate`` dependency. Leg 1 tests the literal CLI ``--help`` (argparse, no
device), so the CLI surface itself is gated regardless of env. When the ``fused_rotate``
env is rebuilt on the release host, a follow-up can swap legs 2-3 to invoke the literal
``tt-atom run``/``md`` subcommands; the assertions below are already the ones that matter.

Fast + deterministic: H2O (3 atoms), MD ``--steps 5``, relax ``steps=20`` on a rattled
geometry. This checks UX plumbing, not accuracy — it does not need a real relaxation.
Exit 0 iff every requested leg PASSES; 1 otherwise. Runs on one card (one device context).

    # gate every surface on card 0 (run with the project venv, like release_gate)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. /path/to/env/bin/python scripts/ux_regression.py
    # CLI-behaviour leg only (no card needed — usable in GitHub CI)
    /path/to/env/bin/python scripts/ux_regression.py --cli-only
"""

from __future__ import annotations

import argparse
import io
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# H2O (3 atoms) is the canonical tiny target — small enough that the Orb load + 5 MD
# steps + a 20-step FIRE relax clear in well under a minute on card. UX plumbing only.
MOL = "H2O"
ORB_CHECKPOINT = "orb-v3-conservative-omol"   # stock ttnn, no fused_rotate dependency
MD_STEPS = 5
RELAX_STEPS = 20
RELAX_FMAX = 0.05
RELAX_RATTLE = 0.10
SEED = 42
PER_LEG_TIMEOUT_S = 300

# Core `tt-atom run` flags a user reaches for. Leg 1 asserts each appears in --help.
RUN_FLAGS = ("--relax", "--md", "--steps", "--temp", "--trace", "--out")


def _subprocess_env(extra: dict | None = None) -> dict:
    """Environment for invoking ``tt_atom.cli`` so it resolves to THIS worktree's tt_atom
    (PYTHONPATH=REPO_ROOT) regardless of any editable install pointing at another
    checkout. Matches the release_gate invocation convention."""
    env = dict(os.environ)
    pp = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        pp = pp + os.pathsep + existing
    env["PYTHONPATH"] = pp
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], *, env: dict | None = None, timeout: int | None = None,
         cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout,
                          capture_output=True, text=True)


# -- leg 1: CLI behaves -------------------------------------------------------

def _check_cli() -> list[str]:
    """Assert `tt-atom run --help` (and relax/md/top-level --help) exit 0 and list the
    core flags. Returns problem strings (empty == pass). No card needed."""
    problems: list[str] = []

    def _help(args: list[str], label: str, required_flags: tuple[str, ...] = ()) -> None:
        try:
            r = _run([sys.executable, "-m", "tt_atom.cli", *args, "--help"],
                     env=_subprocess_env(), timeout=60)
        except Exception as e:
            problems.append(f"{label} --help failed to run: {e}")
            return
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            problems.append(f"{label} --help exited {r.returncode}: "
                            f"{tail[-1] if tail else ''}")
            return
        for flag in required_flags:
            if flag not in r.stdout:
                problems.append(f"{label} --help missing flag {flag}")

    _help(["run"], "tt-atom run", RUN_FLAGS)
    _help(["relax"], "tt-atom relax", ("--steps", "--fmax", "--trace", "--out"))
    _help(["md"], "tt-atom md", ("--steps", "--dt", "--temp", "--trace", "--out"))
    _help([], "tt-atom")  # top-level --help just needs to exit 0
    return problems


# -- leg 2: output files parse -----------------------------------------------

def _check_geometry(path: Path, expected_natoms: int, label: str) -> list[str]:
    """Strict ase.io.read parse of the --out geometry — catches writer/format regressions
    and NaN/empty output. Returns problem strings (empty == pass)."""
    if not path.exists():
        return [f"{label}: CLI wrote no output file at {path}"]
    try:
        from ase.io import read
        atoms = read(str(path))
    except Exception as e:
        return [f"{label}: ase.io.read failed: {type(e).__name__}: {e}"]
    if len(atoms) != expected_natoms:
        return [f"{label}: parsed {len(atoms)} atoms, expected {expected_natoms}"]
    try:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
    except Exception as e:
        return [f"{label}: energy/forces read failed: {type(e).__name__}: {e}"]
    problems: list[str] = []
    if not np.isfinite(e):
        problems.append(f"{label}: energy not finite (={e})")
    if f.shape != (expected_natoms, 3):
        problems.append(f"{label}: forces shape {f.shape}, expected ({expected_natoms},3)")
    if not np.all(np.isfinite(f)):
        problems.append(f"{label}: forces contain NaN/inf")
    if np.allclose(f, 0.0):
        problems.append(f"{label}: forces all-zero (calculator did not populate them)")
    return problems


def _build_calc(device_id: int):
    """OrbCalculator on the stock-ttnn path (no fused_rotate). One device context."""
    from tt_atom.orb_calculator import OrbCalculator
    return OrbCalculator.from_checkpoint(ORB_CHECKPOINT, device_id=device_id)


def _single_point(atoms, calc) -> tuple[float | None, list[str]]:
    """Run one energy+forces evaluation; return (energy, problems)."""
    problems: list[str] = []
    try:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
    except Exception as ex:
        return None, [f"single-point raised: {type(ex).__name__}: {str(ex)[:200]}"]
    if not np.isfinite(e):
        problems.append(f"single-point energy not finite (={e})")
    if not np.all(np.isfinite(f)):
        problems.append("single-point forces contain NaN/inf")
    if np.allclose(f, 0.0):
        problems.append("single-point forces all-zero")
    return e, problems


# -- leg 3: live progress advances -------------------------------------------
# The CLI's relax/MD drivers (cli.py cmd_relax / cmd_md / cmd_run) emit a per-step print
# stream — the direct analogue of tt-bio's "0 -> diffusion" live-progress surface:
#   MD:     "  step {n:4d}  E=...  T=... K"   (cmd_md/cmd_run _log, every max(1,steps//10))
#   relax:  "FIRE:   N  time  Energy  fmax"  (ASE FIRE logfile="-", cmd_relax/cmd_run)
# The headline bug class: the stream emits only the initial step (no advancement), or
# jumps straight to the summary with no per-step ticks, or ticks out of order. These
# checks catch exactly that.

_MD_STEP_RE = re.compile(r"^\s*step\s+(\d+)\s+E=", re.IGNORECASE)
_FIRE_STEP_RE = re.compile(r"^FIRE:\s+(\d+)\b", re.IGNORECASE)
_MD_SUMMARY_RE = re.compile(r"^md:\s+(\d+)\s+steps\b", re.IGNORECASE)


def _check_md_progress(captured: str, steps: int) -> list[str]:
    """Assert the MD step stream advances through every real step (0..steps) and the
    `md: N steps` summary appears. Returns problem strings (empty == pass)."""
    problems: list[str] = []
    md_step_ns: list[int] = []
    summaries: list[int] = []
    for line in captured.splitlines():
        s = line.strip()
        m = _MD_STEP_RE.match(s)
        if m:
            md_step_ns.append(int(m.group(1)))
            continue
        ms = _MD_SUMMARY_RE.match(s)
        if ms:
            summaries.append(int(ms.group(1)))
    if not md_step_ns:
        problems.append("no `step N  E=...` MD progress lines captured — the CLI's MD "
                        "_log print did not fire (live progress not wired?)")
        return problems
    if len(md_step_ns) < 2:
        problems.append(f"MD progress emitted only {len(md_step_ns)} step line(s) "
                        f"{md_step_ns} — the run did not advance past the initial step "
                        f"(the 0->summary jump class)")
    if md_step_ns != sorted(md_step_ns):
        problems.append(f"MD step numbers not monotonic non-decreasing: {md_step_ns}")
    if md_step_ns[-1] != steps:
        problems.append(f"MD final step {md_step_ns[-1]} != requested {steps} — the run "
                        f"did not complete every step")
    if steps >= 1 and md_step_ns[0] != 0:
        problems.append(f"MD first step line is step {md_step_ns[0]}, not 0 "
                        f"(initial tick missing): {md_step_ns[:3]}")
    if not summaries:
        problems.append("no `md: N steps ...` summary line — MD driver did not report "
                        "completion")
    elif summaries[-1] != steps:
        problems.append(f"MD summary says {summaries[-1]} steps, requested {steps}")
    return problems


def _check_relax_progress(captured: str, min_steps: int) -> list[str]:
    """Assert the FIRE log advances (Step 0 then >=1 more, monotonic). Returns problem
    strings (empty == pass)."""
    problems: list[str] = []
    fire_ns: list[int] = []
    for line in captured.splitlines():
        m = _FIRE_STEP_RE.match(line.strip())
        if m:
            fire_ns.append(int(m.group(1)))
    if not fire_ns:
        problems.append("no `FIRE: N ...` relax log lines captured — ASE FIRE logfile "
                        "not wired to stdout (cmd_relax/cmd_run --relax regression?)")
        return problems
    if len(fire_ns) < 2:
        problems.append(f"relax FIRE log emitted only {len(fire_ns)} line(s) {fire_ns} — "
                        f"the optimizer did not advance past Step 0")
    if fire_ns != sorted(fire_ns):
        problems.append(f"FIRE step numbers not monotonic non-decreasing: {fire_ns}")
    if fire_ns[-1] < min_steps - 1 and fire_ns[-1] < 1:
        problems.append(f"relax FIRE log stopped at Step {fire_ns[-1]} — no advancement")
    return problems


# -- per-surface runner ------------------------------------------------------

def run_orb_ux(base: Path) -> dict:
    """Run single-point + relax + MD(--steps 5) on H2O via the Orb Calculator API,
    capturing the per-step print stream, and gate legs 2 (output parses) + 3 (progress
    advances). Returns a result row. Leg 1 (CLI --help) is checked separately in main()."""
    from ase.build import molecule
    from ase import units
    from ase.optimize import FIRE
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.io import write

    dev_id = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    row = {"surface": "orb-md/relax/single-point", "seconds": None, "parse": False,
           "progress": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        calc = _build_calc(dev_id)
    except Exception as ex:
        row["error"] = f"OrbCalculator.from_checkpoint failed: {type(ex).__name__}: {str(ex)[:200]}"
        row["seconds"] = time.monotonic() - t0
        return row

    out_single = base / "single.xyz"
    out_relax = base / "relaxed.xyz"
    out_md = base / "md.xyz"
    parse_problems: list[str] = []
    prog_problems: list[str] = []
    md_captured = io.StringIO()
    relax_captured = io.StringIO()
    try:
        # --- single point (leg 2) ---
        sp = molecule(MOL)
        sp.info.update(charge=0, spin=1)
        sp.calc = calc
        e_sp, sp_problems = _single_point(sp, calc)
        parse_problems += sp_problems
        if e_sp is not None:
            write(str(out_single), sp)
            parse_problems += _check_geometry(out_single, len(sp), "single-point")
        sp_ok = not [p for p in sp_problems if "single" in p]
        e_str = f"{e_sp:.5f}" if e_sp is not None else "n/a"
        row["checks"].append(f"single-point: E={e_str} parse={'OK' if sp_ok else 'FAIL'}")

        # --- relax (legs 2 + 3) ---
        ra = molecule(MOL)
        ra.rattle(stdev=RELAX_RATTLE, seed=SEED)
        ra.info.update(charge=0, spin=1)
        ra.calc = calc
        with contextlib.redirect_stdout(relax_captured):
            FIRE(ra, logfile="-").run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
        write(str(out_relax), ra)
        parse_problems += _check_geometry(out_relax, len(ra), "relax")
        prog_problems += _check_relax_progress(relax_captured.getvalue(), RELAX_STEPS)

        # --- MD (legs 2 + 3) ---
        md = molecule(MOL)
        md.info.update(charge=0, spin=1)
        md.calc = calc
        MaxwellBoltzmannDistribution(md, temperature_K=300.0)
        dyn = Langevin(md, timestep=1.0 * units.fs, temperature_K=300.0,
                       friction=0.01 / units.fs)
        e_md0 = md.get_potential_energy()

        def _log():
            ekin = md.get_kinetic_energy()
            print(f"  step {dyn.nsteps:4d}  E={md.get_potential_energy():.5f}  "
                  f"T={ekin / (1.5 * units.kB * len(md)):.1f} K", flush=True)

        dyn.attach(_log, interval=max(1, MD_STEPS // 10))
        with contextlib.redirect_stdout(md_captured):
            dyn.run(MD_STEPS)
            print(f"md: {MD_STEPS} steps (1.0 fs) at 300.0 K; "
                  f"E {e_md0:.5f} -> {md.get_potential_energy():.5f} eV", flush=True)
        write(str(out_md), md)
        parse_problems += _check_geometry(out_md, len(md), "md")
        prog_problems += _check_md_progress(md_captured.getvalue(), MD_STEPS)
    except Exception as ex:
        if not row["error"]:
            row["error"] = f"orb-ux run raised: {type(ex).__name__}: {str(ex)[:200]}"
    finally:
        try:
            calc.close()
        except Exception:
            pass

    row["seconds"] = time.monotonic() - t0
    row["parse"] = not parse_problems
    row["progress"] = not prog_problems
    row["gate"] = row["parse"] and row["progress"] and not row["error"]
    row["checks"].append(f"parse: {'OK' if not parse_problems else 'FAIL'}")
    for p in parse_problems:
        row["checks"].append(f"  * {p}")
    row["checks"].append(f"progress: {'OK' if not prog_problems else 'FAIL'}")
    for p in prog_problems:
        row["checks"].append(f"  * {p}")
    if not row["error"]:
        errs = parse_problems + prog_problems
        if errs:
            row["error"] = "; ".join(errs)
    return row


# -- driver -------------------------------------------------------------------

def _print_row(r: dict) -> None:
    wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
    verdict = "PASS" if r["gate"] else (f"FAIL ({r['error']})" if r["error"] else "FAIL")
    print(f"{r['surface']:<32}{'parse':>8}{'progress':>10}{wall:>9}  {verdict}")
    print(f"  parse={r['parse']} progress={r['progress']}")
    for c in r["checks"]:
        print(f"  {c}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli-only", action="store_true",
                    help="Run ONLY the CLI-behaviour leg (tt-atom run/relax/md --help). No "
                         "card needed — usable in GitHub CI. Skips the on-device legs.")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the per-run output dir under the tmp dir for inspection.")
    args = ap.parse_args()

    # The guard drives the real `tt_atom.cli` via sys.executable, so it must be launched
    # with a Python that has tt-atom's deps installed (numpy / ttnn / ase) — i.e. the
    # project venv, exactly like scripts/release_gate.py:
    #     /home/moritz/.ttatom_run/env/bin/python scripts/ux_regression.py
    probe = _run([sys.executable, "-c", "import tt_atom, ase"],
                 env=_subprocess_env(), timeout=60)
    if probe.returncode != 0:
        sys.exit(
            f"this Python ({sys.executable}) cannot import tt_atom/ase with "
            f"PYTHONPATH={REPO_ROOT}:\n{(probe.stderr or probe.stdout).strip()}\n"
            f"Run the guard with the project venv, e.g. "
            f"/home/moritz/.ttatom_run/env/bin/python scripts/ux_regression.py")

    all_pass = True

    # Leg 1 (CLI behaves) runs always — it needs no card.
    print(f"\n{'#' * 78}\nUX GATE — leg 1: CLI behaves (tt-atom run/relax/md --help)\n{'#' * 78}")
    cli_problems = _check_cli()
    if cli_problems:
        all_pass = False
        for p in cli_problems:
            print(f"  X {p}")
    else:
        print("  OK tt-atom run --help / relax --help / md --help / --help all exit 0 "
              "and list the core flags")
    print(f"{'#' * 78}")

    if args.cli_only:
        return 0 if all_pass else 1

    # Legs 2 + 3 (output parses + live progress advances) — on-device, Orb path.
    print(f"\n{'#' * 78}\nUX GATE — legs 2+3: output parses + live progress advances "
          f"(Orb {ORB_CHECKPOINT}, {MOL}, MD --steps {MD_STEPS})\n{'#' * 78}")
    base = Path(tempfile.mkdtemp(prefix="ux_gate_", dir=str(REPO_ROOT)))
    try:
        r = run_orb_ux(base)
        _print_row(r)
        all_pass &= r["gate"]
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)

    print(f"\n{'#' * 78}\nUX GATE SUMMARY\n{'#' * 78}")
    print("GATE PASS — CLI behaves, output files parse, live progress advances through "
          "every step" if all_pass
          else "GATE FAIL — a UX leg missed (see above). A UX regression blocks a tag, "
               "same standing as an accuracy regression.")
    print(f"{'#' * 78}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
