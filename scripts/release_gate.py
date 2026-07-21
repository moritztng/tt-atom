#!/usr/bin/env python3
"""TT-Atom release gate with machine-readable PASS/FAIL/GAP results.

The runnable equivalent of ``RELEASING.md``'s manual checklist, and the live harness behind the
parity table in ``docs/materials-benchmark.md`` (run ``--leg accuracy`` to reproduce those R/D/X
numbers on your card). Four regular legs run before each tag:

  1. ACCURACY — numerical parity vs each shipped model family's own reference, within tolerance
     (energy rel-error, force/stress PCC), across every supported task and graph regime for which
     a real-weight golden fixture exists. Reuses the existing ``tests/test_*realweight*.py``
     parity modules verbatim (they already encode the bars and the real oracles — fairchem
     ``uma-s-1`` for UMA, the real ``orb-models`` CPU oracle for Orb-v3 / OrbMol) by invoking
     pytest as a subprocess with JUnit XML, so the gate never re-derives a parity bar or oracle.
     A module whose golden is absent auto-skips; the gate reports that leg as GAP (missing
     fixture), never as a silent PASS.
  2. NO-OOM — runs disjoint-union batch sweeps for Orb and UMA over K=1..128 small systems.
     Each family runs in a fresh process, and every configured size must complete.
  3. PERF — warm steady-state throughput on a fixed small input vs a committed per-card,
     per-model baseline (``docs/perf_baselines.json``), FAILs beyond a configurable noise
     margin. One entry per shipped family's throughput path (OrbMol ``conservative-omol``,
     Orb-v3 bulk ``conservative-inf-omat``, UMA ``uma-s-1``), mirroring tt-bio's
     ``scripts/perf_regression.py``: a ``--model`` flag iterates a SPECS-style dict, one
     baseline entry per model per card. Card-type-aware (a P300c baseline is never judged
     against a P150a run), fails loudly on NO BASELINE, and updates only via
     ``--update-baseline --note "<why>"``. Seeds the baseline the first time a card type is
     run for a model. UMA's batched forward needs the ALWAYS-ON ``fused_rotate`` kernel
     absent from stock ttnn builds, so on such a host the UMA row reports GAP (env), not FAIL.
  4. UX — the user-facing plumbing still works headlessly on a tiny input (H2O): CLI --help
     behaves and lists the core flags, a real single-point + relax + MD(--steps 5) write an
     --out geometry that parses under ase.io.read with finite energy/forces, and the CLI's
     per-step MD/relax progress stream advances through every real step (the "0 -> diffusion"
     bug-class analogue). Mirrors tt-bio's scripts/ux_regression.py in methodology; lives in
     the sibling scripts/ux_regression.py (also runnable standalone, --cli-only needs no card).

The opt-in INSTALL leg runs once per release. It builds the pinned tt-metal source in a clean
environment, installs the exact committed TT-Atom source from origin, and smokes UMA and Orb.

Honest reporting: every leg prints PASS / FAIL / GAP with the real numbers (or the real skip
reason). Nothing fabricated. Release mode exits 0 only when every selected row passes;
``--allow-gaps`` is an explicit diagnostics mode.

Usage::

    # full gate on card 0 (one device context per leg)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/release_gate.py

    # one leg
    python3 scripts/release_gate.py --leg accuracy
    python3 scripts/release_gate.py --leg oom
    python3 scripts/release_gate.py --leg perf
    python3 scripts/release_gate.py --leg perf --model orb-conservative-inf-omat-batch
    python3 scripts/release_gate.py --leg ux
    python3 scripts/release_gate.py --leg ux --cli-only   # no card — GitHub CI
    python3 scripts/release_gate.py --leg install          # clean source build, per release

    # seed / refresh one model's perf baseline from the current warm run (explicit, needs a note)
    python3 scripts/release_gate.py --leg perf --model orb-conservative-inf-omat-batch \
        --update-baseline --note "seed p150a bulk baseline"

    # quick subset (fewer accuracy modules, smaller OOM sweep, fewer perf iters) for a fast smoke
    python3 scripts/release_gate.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import date

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN_DIR = pathlib.Path.home() / ".ttatom_run" / "goldens_real"
BASELINE_FILE = REPO_ROOT / "docs" / "perf_baselines.json"
RESULTS_DIR = pathlib.Path(tempfile.gettempdir()) / "tt-atom-release-gate"

# ── leg 1: accuracy parity ─────────────────────────────────────────────────
# Each spec is one real-weight parity module + the golden(s) it needs. A module auto-skips if
# its golden is absent, so the gate reports GAP (missing fixture) rather than a silent PASS. The
# bars and oracles live in the test modules themselves — the gate only runs them and reads the
# JUnit XML, so it can never drift from the parity definition.
ACCURACY_SPECS = [
    dict(family="uma", checkpoint="uma-s-1", regime="molecular / omol",
         module="tests/test_realweight.py", golden="ethanol_omol.npz"),
    dict(family="uma", checkpoint="uma-s-1",
         regime="periodic / omat+oc20+odac+omc",
         module="tests/test_periodic.py",
         golden="si_omat.npz;cuh_oc20.npz;mgo_odac.npz;co2_omc.npz"),
    dict(family="orb", checkpoint="conservative-inf-omat", regime="bulk / omat (toy)",
         module="tests/test_orb_realweight.py", golden="si_omat_orb.npz"),
    dict(family="orb", checkpoint="conservative-inf-omat", regime="analytic forces",
         module="tests/test_orb_forces_realweight.py", golden="si_omat_orb.npz"),
    dict(family="orb", checkpoint="direct-20-omat", regime="bulk / omat (direct)",
         module="tests/test_orb_direct_realweight.py", golden="si_omat_orb_direct20.npz"),
    dict(family="orb", checkpoint="conservative-inf-omat", regime="periodic supercell",
         module="tests/test_orb_periodic_realweight.py", golden="si_supercell_orb.npz"),
    dict(family="orb", checkpoint="conservative-inf-omat", regime="bulk / omat (MgO oxide)",
         module="tests/test_orb_mgo_realweight.py", golden="mgo_omat_orb.npz"),
    dict(family="orb", checkpoint="conservative-inf-omat", regime="stress (conservative)",
         module="tests/test_orb_stress_realweight.py", golden="si_omat_orb.npz"),
    dict(family="orb", checkpoint="direct-20-omat", regime="ZBL short-contact forces",
         module="tests/test_orb_zbl_forces.py", golden="si_short_contact_orb_direct20.npz"),
    dict(family="orb", checkpoint="conservative + direct",
         regime="bf8 fast-mode accuracy",
         module="tests/test_orb_bf8_fast.py",
         golden="si_omat_orb.npz;si_omat_orb_direct20.npz"),
    dict(family="orbmol", checkpoint="conservative + direct",
         regime="molecule / charged / openshell",
         module="tests/test_orb_omol_realweight.py",
         golden="molecule_omol_conservative.npz;molecule_charged_omol_conservative.npz;"
                "molecule_openshell_omol_conservative.npz;molecule_omol_direct.npz;"
                "molecule_charged_omol_direct.npz;molecule_openshell_omol_direct.npz"),
]
QUICK_ACCURACY = [
    "tests/test_orb_realweight.py",
    "tests/test_orb_direct_realweight.py",
    "tests/test_orb_forces_realweight.py",
]

# ── leg 2: no-OOM sweep ─────────────────────────────────────────────────────
# K small systems in one device forward. Each family runs in its own fully-exiting child so
# process-global Metal state cannot leak between implementations.
OOM_SPECS = {
    "orb": dict(checkpoint="orb-v3-conservative-omol"),
    "uma": dict(checkpoint="uma-s-1"),
}
OOM_MOL = "CH3CH2OH"
OOM_KS_DEFAULT = [1, 2, 4, 8, 16, 32, 64, 128]
OOM_KS_QUICK = [1, 2, 4, 8]

# ── leg 3: perf regression ───────────────────────────────────────────────────
# Warm steady-state throughput on a fixed small batch, vs docs/perf_baselines.json. Card-type
# aware (per-card baseline key), fails loudly on NO BASELINE, updates only via --update-baseline.
#
# Per-model SPECS dict (mirrors tt-bio's scripts/perf_regression.py): one entry per shipped
# family's throughput path, keyed by the same baseline key used in docs/perf_baselines.json.
# ``kind`` dispatches the calculator/protocol; ``fixture`` picks the small-system generator
# (molecule conformers vs periodic Si bulk). The pre-existing ``orb-conservative-omol-batch``
# entry is unchanged — same checkpoint, mol, K, warmup/repeat as before the generalization.
#
# UMA's batched forward goes through the same ALWAYS-ON ``fused_rotate`` kernel as its accuracy
# leg, absent from stock ttnn builds, so the UMA row measures as GAP (env), not FAIL. The gate
# reports it loudly rather than silently skipping the family. With a source build carrying the op,
# the UMA row measures and gates against its seeded baseline like any other model.
PERF_SPECS: dict[str, dict] = {
    "orb-conservative-omol-batch": dict(
        family="orbmol", checkpoint="orb-v3-conservative-omol",
        kind="orb-batch", fixture="molecule", mol="CH3CH2OH", k=8,
        unit="sys/s", direction="higher",
        regime="molecule / charged / openshell (OrbMol batch)"),
    "orb-conservative-inf-omat-batch": dict(
        family="orb", checkpoint="orb-v3-conservative-inf-omat",
        kind="orb-batch", fixture="bulk-si", mol="Si", k=8,
        unit="sys/s", direction="higher",
        regime="bulk / omat (Si toy, periodic)"),
    "uma-s-1-omol-batch": dict(
        family="uma", checkpoint="uma-s-1",
        kind="uma-batch", fixture="molecule", mol="CH3CH2OH", k=8,
        unit="sys/s", direction="higher",
        regime="molecular / omol (UMA batch)"),
}
DEFAULT_PERF_MODELS = list(PERF_SPECS)
PERF_WARMUP = 2
PERF_REPEAT = 5
PERF_WARMUP_QUICK = 1
PERF_REPEAT_QUICK = 3
DEFAULT_THRESHOLD = 15.0  # % regression allowed before FAIL
# A normal measurement (warmup+repeat evaluate_batch calls) finishes in seconds; this only
# guards against a wedged device hanging the whole gate forever.
PERF_MEASURE_TIMEOUT_S = 240
OOM_MEASURE_TIMEOUT_S = 600
# Per-release clean-env install leg: a full from-zero tt-metal build + tt-atom install + smoke.
# Deliberately long (the build alone is tens of minutes) and opt-in (`--leg install`), not part of
# the default per-tick gate. See RELEASING.md leg 5.
INSTALL_TIMEOUT_S = 3600
# The tt-metal commit (branch moritztng/tt-atom) this release was validated against. A customer
# following the README pins this commit for a reproducible build; the install leg clones exactly
# it, so the gate catches any drift between the pin and what actually builds + runs.
PINNED_TT_METAL_COMMIT = "8d759240fdd763a38e3abdc8344076f584dc4f4d"
PINNED_TT_METAL_BRANCH = "moritztng/tt-atom"
LOGICAL_DEVICE_ID = 0

# ── card-type detection (mirrors tt-bio's perf_regression.py) ────────────────
_P300_SUBSYSTEMS = {"0x0044", "0x0045", "0x0046"}


def _resolve_tt_smi():
    found = shutil.which("tt-smi")
    if found:
        return found
    for c in (pathlib.Path.home() / ".local" / "bin" / "tt-smi",
              pathlib.Path("/usr/local/bin/tt-smi"), pathlib.Path("/usr/bin/tt-smi")):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def _sysfs_subsystem_device(device_id):
    for entry in pathlib.Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            did = entry.name.rsplit("!", 1)[1]
        except Exception:
            continue
        if did != device_id:
            continue
        try:
            return (entry / "device" / "subsystem_device").read_text().strip().lower()
        except Exception:
            return None
    return None


def detect_card_type():
    """Canonical board-type key ('p150a', 'p300c', ...). No device opened; safe in the parent."""
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    tt_smi = _resolve_tt_smi()
    if tt_smi is not None:
        try:
            out = subprocess.run([tt_smi, "-s"], capture_output=True, text=True,
                                 timeout=20, check=False)
            info = json.loads(out.stdout).get("device_info", [])
            if info:
                idx = min(int(visible), len(info) - 1) if visible.isdigit() else 0
                bt = info[idx].get("board_info", {}).get("board_type")
                if bt:
                    return str(bt).lower()
        except Exception:
            pass
    else:
        print(f"{sys.argv[0]}: WARNING: tt-smi not found; card detection falling back to sysfs "
              f"(may report 'unknown' -> NO BASELINE). Add tt-smi to PATH and re-run.",
              file=sys.stderr)
    sub = _sysfs_subsystem_device(visible)
    if sub in _P300_SUBSYSTEMS:
        return "p300c"
    return f"unknown:{sub}" if sub else "unknown"


def _version():
    import re
    txt = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M)
    return m.group(1) if m else "unknown"


def _golden_present(spec):
    return all((GOLDEN_DIR / g).exists() for g in spec["golden"].split(";"))


# ── leg 1 implementation ─────────────────────────────────────────────────────

def run_accuracy(specs, quick):
    """Run the real-weight parity modules via pytest + JUnit XML. Returns (rows, all_pass)."""
    if quick:
        selected = [s for s in specs if s["module"] in QUICK_ACCURACY]
    else:
        selected = list(specs)
    rows = []
    any_fail = False
    for s in selected:
        if not _golden_present(s):
            rows.append({**s, "verdict": "GAP", "passed": 0, "skipped": 0, "failed": 0,
                        "note": "missing golden bundle", "wall_s": 0.0})
            continue
        rows.append(_run_pytest_module(s))
        if rows[-1]["verdict"] == "FAIL":
            any_fail = True
    return rows, not any_fail


def _run_pytest_module(spec):
    mod = spec["module"]
    xml_dir = pathlib.Path(tempfile.mkdtemp(prefix="gate-acc-"))
    xml_path = xml_dir / "junit.xml"
    cmd = [sys.executable, "-m", "pytest", mod, "-q", "-p", "no:cacheprovider",
           f"--junit-xml={xml_path}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    print(f"\n[accuracy] pytest {mod}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    wall = time.monotonic() - t0
    passed = skipped = failed = 0
    fail_names = []
    env_gap_only = True
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            for tc in tree.iter("testcase"):
                skipped_el = tc.find("skipped")
                if skipped_el is not None:
                    skipped += 1
                elif len(tc.findall("failure")) > 0 or len(tc.findall("error")) > 0:
                    failed += 1
                    fail_names.append(tc.get("name", "?"))
                    fail_text = "".join(el.get("message", "") + (el.text or "")
                                        for el in tc.findall("failure") + tc.findall("error"))
                    if "fused_rotate" not in fail_text:
                        env_gap_only = False
                else:
                    passed += 1
        except ET.ParseError:
            failed = -1
            env_gap_only = False
    if proc.returncode != 0 and failed == 0:
        failed = 1
        env_gap_only = False
    # A missing fused_rotate op crashes mid-test rather than auto-skipping, so it otherwise
    # reads as FAIL here while the OOM/perf legs already classify the identical cause as GAP.
    if failed > 0 and env_gap_only:
        verdict = "GAP"
    else:
        verdict = "PASS" if (failed == 0 and skipped == 0) else (
            "GAP" if (failed == 0 and skipped > 0) else "FAIL")
    note = ""
    if skipped > 0:
        note = f"{skipped} skipped (missing fixture or condition)"
    if fail_names:
        tag = "env gap (missing fused_rotate): " if (
            failed > 0 and env_gap_only) else "failed: "
        note = (note + "; " if note else "") + tag + ",".join(fail_names[:3])
    shutil.rmtree(xml_dir, ignore_errors=True)
    return {**spec, "verdict": verdict, "passed": passed, "skipped": skipped,
            "failed": failed, "wall_s": round(wall, 1), "note": note}


def _print_accuracy(rows, all_pass):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 1: ACCURACY parity (real-weight goldens, real oracles)\n{'#' * 78}")
    print(f"{'family':<8}{'checkpoint':<26}{'regime':<34}{'pass/skip/fail':>16}{'verdict':>9}")
    print("-" * 93)
    for r in rows:
        psf = f"{r['passed']}/{r['skipped']}/{r['failed']}"
        print(f"{r['family']:<8}{r['checkpoint']:<26}{r['regime']:<34}{psf:>16}{r['verdict']:>9}")
        if r.get("note"):
            print(f"        -> {r['note']}")
    print("-" * 93)
    gaps = [r for r in rows if r["verdict"] == "GAP"]
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    if fails:
        msg = f"GATE FAIL — {len(fails)} accuracy module(s) FAILED (see above)"
    elif gaps:
        msg = f"GATE GAP — {len(gaps)} module(s) not fully verified " \
              f"(missing fixture or documented env gap); the rest passed. See notes above."
    else:
        msg = "GATE PASS — every accuracy module passed parity vs its real oracle"
    print(f"{'#' * 78}\n{msg}")


# ── leg 2 implementation ─────────────────────────────────────────────────────

def measure_oom(family, out_path, quick):
    """Run one family's OOM sweep and persist its result before process teardown."""
    out_path.write_text(json.dumps(run_oom(family, quick)))


def run_oom(family, quick):
    """Find the largest configured disjoint batch that clears for one model family."""
    spec = OOM_SPECS[family]
    ks = OOM_KS_QUICK if quick else OOM_KS_DEFAULT
    print(f"\n[oom] {family} disjoint-union batch sweep: checkpoint={spec['checkpoint']} "
          f"mol={OOM_MOL} K={ks}", flush=True)
    try:
        from ase.build import molecule
        if family == "orb":
            from tt_atom.orb_calculator import OrbCalculator
        else:
            from tt_atom.calculator import TTAtomCalculator
    except Exception as e:
        return dict(family=family, checkpoint=spec["checkpoint"], verdict="GAP",
                    ceiling=None, failed_at=None, note=f"calculator import failed: {e}", rows=[])
    try:
        if family == "orb":
            calc = OrbCalculator.from_checkpoint(
                spec["checkpoint"], device_id=LOGICAL_DEVICE_ID)
        else:
            seed = molecule(OOM_MOL)
            seed.rattle(stdev=0.08, seed=10)
            seed.info.update(charge=0, spin=1)
            calc = TTAtomCalculator.from_uma(
                model=spec["checkpoint"], task_name="omol", atoms=seed,
                device_id=LOGICAL_DEVICE_ID)
    except Exception as e:
        return dict(family=family, checkpoint=spec["checkpoint"], verdict="GAP",
                    ceiling=None, failed_at=None,
                    note=f"calculator setup failed (weights cached? refenv?): {e}", rows=[])
    natoms = len(molecule(OOM_MOL))
    rows = []
    ceiling = None
    failed_at = None
    env_gap = False
    for k in ks:
        systems = []
        for i in range(k):
            a = molecule(OOM_MOL)
            a.rattle(stdev=0.08, seed=10 + i)
            a.info.update(charge=0, spin=1)
            systems.append(a)
        try:
            calc.evaluate_batch(systems)
            ceiling = k
            rows.append(dict(K=k, natoms_total=int(natoms * k), ok=True))
            print(f"  K={k:4d}  Ntot={natoms*k:5d}  OK")
        except RuntimeError as e:
            msg = str(e).splitlines()[0][:90]
            env_gap = "fused_rotate" in msg
            rows.append(dict(K=k, natoms_total=int(natoms * k), ok=False, err=msg,
                             env_gap=env_gap))
            failed_at = None if env_gap else k
            label = "GAP" if env_gap else "OOM/err"
            print(f"  K={k:4d}  Ntot={natoms*k:5d}  {label}: {msg}")
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            env_gap = "fused_rotate" in msg
            rows.append(dict(K=k, natoms_total=int(natoms * k), ok=False, err=msg,
                             env_gap=env_gap))
            failed_at = None if env_gap else k
            label = "GAP" if env_gap else "err"
            print(f"  K={k:4d}  Ntot={natoms*k:5d}  {label}: {msg}")
            break
    try:
        calc.close()
    except Exception:
        pass
    verdict = ("PASS" if ceiling == ks[-1] else
               "GAP" if env_gap else
               "FAIL" if failed_at is not None else "GAP")
    note = (f"batch ceiling = {ceiling} systems "
            f"({None if ceiling is None else natoms * ceiling} atoms) in one device forward")
    return dict(family=family, checkpoint=spec["checkpoint"], verdict=verdict, ceiling=ceiling,
                failed_at=failed_at, note=note, rows=rows)


def _run_oom_family(family, quick):
    """Spawn one family in a fresh process so its MetalContext dies before the next run.

    ``ttnn.close_device`` closes the logical device but the process-global MetalContext and
    its UMD mappings live until process exit.
    """
    td = tempfile.mkdtemp(prefix=f"gate-oom-{family}-")
    out = pathlib.Path(td) / "result.json"
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
           "--measure-oom", family, "--out", str(out)]
    if quick:
        cmd.append("--quick")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=OOM_MEASURE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[oom] TIMED OUT after {OOM_MEASURE_TIMEOUT_S}s (device hang) — "
              f"stopping child and resetting the card", file=sys.stderr)
        reset_error = _stop_child_and_reset(proc)
        shutil.rmtree(td, ignore_errors=True)
        note = f"{family} OOM sweep timed out after {OOM_MEASURE_TIMEOUT_S}s (device hang)"
        if reset_error:
            note += f"; card reset failed: {reset_error}"
        return dict(family=family, checkpoint=OOM_SPECS[family]["checkpoint"], verdict="FAIL",
                    ceiling=None, failed_at=None, note=note, rows=[])
    if returncode != 0:
        shutil.rmtree(td, ignore_errors=True)
        return dict(family=family, checkpoint=OOM_SPECS[family]["checkpoint"], verdict="FAIL",
                    ceiling=None, failed_at=None,
                    note=f"OOM subprocess exited {returncode} during device teardown", rows=[])
    if not out.exists():
        shutil.rmtree(td, ignore_errors=True)
        return dict(family=family, checkpoint=OOM_SPECS[family]["checkpoint"], verdict="FAIL",
                    ceiling=None, failed_at=None,
                    note=f"OOM subprocess exited {returncode} (no result json)", rows=[])
    try:
        return json.loads(out.read_text())
    except Exception as e:
        return dict(family=family, checkpoint=OOM_SPECS[family]["checkpoint"], verdict="FAIL",
                    ceiling=None, failed_at=None,
                    note=f"OOM result parse failed: {e}", rows=[])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _run_oom(quick):
    results = [_run_oom_family(family, quick) for family in OOM_SPECS]
    verdicts = {result["verdict"] for result in results}
    verdict = "FAIL" if "FAIL" in verdicts else ("GAP" if "GAP" in verdicts else "PASS")
    return dict(verdict=verdict, results=results)


def _print_oom(res):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 2: NO-OOM sweeps (single card)\n{'#' * 78}")
    for result in res["results"]:
        print(f"{result['family']} ({result['checkpoint']}):")
        print(f"  {'K':<8}{'Ntot':>8}  result")
        for row in result["rows"]:
            ok = ("OK" if row["ok"] else
                  f"GAP: {row.get('err', '')}" if row.get("env_gap") else
                  f"OOM/err: {row.get('err', '')}")
            print(f"  {row['K']:<8}{row['natoms_total']:>8}  {ok}")
        print(f"  ceiling: {result['ceiling']} | failed_at: {result['failed_at']} "
              f"-> {result['verdict']}")
        print(f"  note: {result['note']}")
    print("-" * 78)
    print(f"leg verdict: {res['verdict']}")
    print(f"{'#' * 78}")


# ── leg 3 implementation ───────────────────────────────────────────────────
# Per-model: spawn one measurement subprocess per model (one device context each), then the
# parent compares each against the per-card baseline and prints a per-model table. Mirrors
# tt-bio's scripts/perf_regression.py parent/child split so model weights are released cleanly
# between models and we never take a cross-model device-reopen path in one process.

def _perf_systems(spec, k):
    """Build K small systems for the perf leg's disjoint-union batch, per spec."""
    if spec["fixture"] == "molecule":
        from ase.build import molecule
        out = []
        for i in range(k):
            a = molecule(spec["mol"])
            a.rattle(stdev=0.08, seed=10 + i)
            a.info.update(charge=0, spin=1)
            out.append(a)
        return out
    if spec["fixture"] == "bulk-si":
        from ase.build import bulk
        out = []
        for i in range(k):
            a = bulk("Si", "diamond", a=5.43) * (2, 1, 1)
            a.rattle(stdev=0.08, seed=10 + i)
            out.append(a)
        return out
    raise ValueError(f"unknown fixture {spec['fixture']!r}")


def measure_perf(model, out_path, quick):
    """In-process warm-throughput measurement for one perf model; writes a JSON result.

    Runs in its own subprocess (see ``_run_measure_perf``) so each model gets a fresh device
    context. Warm steady-state throughput: K small systems per ``evaluate_batch`` call, median
    of ``repeat`` timed calls after ``warmup`` warm calls. The gated metric is systems/s
    (higher is better). On a measurement failure the child writes a ``failed`` result so the
    parent can render it as GAP (env) or FAIL honestly instead of a silent skip."""
    from importlib.metadata import version

    spec = PERF_SPECS[model]
    warmup = PERF_WARMUP_QUICK if quick else PERF_WARMUP
    repeat = PERF_REPEAT_QUICK if quick else PERF_REPEAT
    try:
        if spec["kind"] == "orb-batch":
            from tt_atom.orb_calculator import OrbCalculator
            calc = OrbCalculator.from_checkpoint(
                spec["checkpoint"], device_id=LOGICAL_DEVICE_ID)
        elif spec["kind"] == "uma-batch":
            from tt_atom.calculator import TTAtomCalculator
            seed = _perf_systems(spec, 1)[0]
            calc = TTAtomCalculator.from_uma(model=spec["checkpoint"], task_name="omol",
                                             atoms=seed, device_id=LOGICAL_DEVICE_ID)
        else:
            raise ValueError(f"unknown kind {spec['kind']!r}")
    except Exception as e:
        _write_perf_error(out_path, model, spec, e)
        return
    try:
        systems = _perf_systems(spec, spec["k"])
        natoms = len(systems[0])

        def one():
            calc.evaluate_batch(systems)

        for _ in range(warmup):
            one()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            one()
            times.append(time.perf_counter() - t0)
    except Exception as e:
        _write_perf_error(out_path, model, spec, e)
        try:
            calc.close()
        except Exception:
            pass
        return
    try:
        calc.close()
    except Exception:
        pass
    times.sort()
    median = times[len(times) // 2]
    result = dict(
        model=model, family=spec["family"], checkpoint=spec["checkpoint"],
        regime=spec["regime"], unit=spec["unit"], direction=spec["direction"],
        kind=spec["kind"], fixture=spec["fixture"], mol=spec["mol"],
        throughput=spec["k"] / median, latency_ms=median * 1000.0 / spec["k"],
        median_s=median, k=spec["k"], natoms_per_system=natoms,
        warmup=warmup, repeat=repeat,
        times_s=[round(t, 4) for t in times],
        card_type=detect_card_type(), tt_atom_version=_version(),
        ttnn_version=version("ttnn"),
        date=date.today().isoformat(),
        input=f"{spec['mol']} ({spec['fixture']}, {natoms} atoms/system, K={spec['k']})",
        failed=False,
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] {result['throughput']:.4g} {spec['unit']}  "
          f"({result['latency_ms']:.2f} ms/batch)", file=sys.stderr)


def _write_perf_error(out_path, model, spec, e):
    import traceback
    traceback.print_exc()
    msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    env_gap = "fused_rotate" in str(e)
    result = dict(
        model=model, family=spec["family"], checkpoint=spec["checkpoint"],
        regime=spec["regime"], unit=spec["unit"], direction=spec["direction"],
        failed=True, error=msg, env_gap=env_gap,
        card_type=detect_card_type(), tt_atom_version=_version(),
        date=date.today().isoformat(),
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] MEASURE FAILED: {msg}", file=sys.stderr)


def _stop_child_and_reset(proc):
    """Stop a timed-out device child, then reset only after no process holds the card."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    tt_smi = _resolve_tt_smi()
    if tt_smi is None:
        return "tt-smi not found"
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    try:
        reset = subprocess.run([tt_smi, "-r", visible], timeout=120,
                               capture_output=True, text=True, check=False)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    if reset.returncode != 0:
        detail = (reset.stderr or reset.stdout or "").strip().splitlines()
        return detail[-1] if detail else f"tt-smi exited {reset.returncode}"
    return None


def _run_measure_perf(model, quick):
    """Spawn the per-model measurement in a fresh subprocess (one device context)."""
    td = tempfile.mkdtemp(prefix="gate-perf-")
    out = pathlib.Path(td) / "result.json"
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
           "--measure-perf", model, "--out", str(out)]
    if quick:
        cmd.append("--quick")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=PERF_MEASURE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[{model}] MEASURE TIMED OUT after {PERF_MEASURE_TIMEOUT_S}s "
              f"(device hang) — stopping child and resetting the card", file=sys.stderr)
        reset_error = _stop_child_and_reset(proc)
        shutil.rmtree(td, ignore_errors=True)
        error = f"measurement timed out after {PERF_MEASURE_TIMEOUT_S}s (device hang"
        error += ", card was reset)" if not reset_error else f"; card reset failed: {reset_error})"
        return dict(model=model, failed=True,
                    error=error)
    if returncode != 0:
        shutil.rmtree(td, ignore_errors=True)
        return dict(model=model, failed=True,
                    error=f"subprocess exited {returncode} during device teardown")
    if not out.exists():
        shutil.rmtree(td, ignore_errors=True)
        return dict(model=model, failed=True,
                    error=f"subprocess exited {returncode} (no result json)")
    try:
        return json.loads(out.read_text())
    except Exception as e:
        return dict(model=model, failed=True, error=f"result parse failed: {e}")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _load_baselines():
    if not BASELINE_FILE.exists():
        return {"cards": {}}
    return json.loads(BASELINE_FILE.read_text())


def _save_baselines(data):
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _card_baselines(data, card_type):
    cards = data.get("cards")
    if not cards and data.get("models"):
        return data["models"]
    entry = cards.get(card_type) if cards else None
    return entry.get("models", {}) if entry else None


def run_perf(quick, models, update_baseline, note, threshold):
    if quick and update_baseline:
        sys.exit("--quick cannot update performance baselines")
    card = detect_card_type()
    print(f"\n[perf] warm throughput: models={','.join(models)} card={card} "
          f"(warmup+repeat per model, K per spec)", flush=True)
    rows = [_run_measure_perf(m, quick) for m in models]
    if update_baseline:
        return _update_perf_baselines(rows, card, note, threshold)
    return _compare_perf(rows, card, threshold)


def _compare_perf(rows, card, threshold):
    bm = _card_baselines(_load_baselines(), card) or {}
    overall_pass = True
    any_gap = False
    for r in rows:
        if r.get("failed"):
            if r.get("env_gap"):
                r["verdict"] = "GAP"
                any_gap = True
                r["baseline"] = None
                r["delta"] = "n/a"
                r["note"] = (f"env gap: {r.get('error','')} — UMA's batched forward needs the "
                             f"custom fused_rotate ttnn op, absent from stock ttnn builds; "
                             f"not a perf regression")
            else:
                r["verdict"] = "FAIL"
                r["baseline"] = None
                r["delta"] = "n/a"
                r["note"] = f"measurement failed: {r.get('error','')}"
                overall_pass = False
            continue
        key = r["model"]
        if key not in bm:
            r["verdict"] = "GAP"
            any_gap = True
            r["baseline"] = None
            r["delta"] = "n/a"
            r["note"] = (f"NO BASELINE for card '{card}' / model '{key}' in "
                     f"{BASELINE_FILE.relative_to(REPO_ROOT)}. Seed it with: "
                     f"python3 scripts/release_gate.py --leg perf --model {key} "
                     f"--update-baseline --note \"seed {card} baseline\"")
            continue
        baseline = bm[key]
        protocol_fields = ("checkpoint", "k", "natoms_per_system", "ttnn_version")
        mismatches = [
            f"{field}={baseline.get(field)!r} (run {r.get(field)!r})"
            for field in protocol_fields if baseline.get(field) != r.get(field)
        ]
        if mismatches:
            r["verdict"] = "FAIL"
            r["baseline"] = baseline.get("value")
            r["delta"] = "n/a"
            r["note"] = "baseline protocol mismatch: " + ", ".join(mismatches)
            overall_pass = False
            continue
        base = float(baseline["value"])
        direction = baseline.get("direction", r["direction"])
        raw_pct = (r["throughput"] - base) / base * 100.0
        pct = raw_pct if direction == "higher" else -raw_pct
        r["baseline"] = base
        r["delta"] = f"{'+' if raw_pct >= 0 else ''}{raw_pct:.1f}%"
        r["verdict"] = "PASS" if pct >= -threshold else "FAIL"
        r["note"] = f"vs baseline {base:.4g} {r['unit']} (threshold -{threshold:.0f}%)"
        if r["verdict"] == "FAIL":
            overall_pass = False
    verdict = "FAIL" if not overall_pass else ("GAP" if any_gap else "PASS")
    return dict(verdict=verdict, rows=rows, card=card)


def _update_perf_baselines(rows, card, note, threshold):
    if not note:
        sys.exit("--update-baseline requires --note \"<why this perf change is intended>\"")
    data = _load_baselines()
    cards = data.setdefault("cards", {})
    entry = cards.setdefault(card, {})
    models = entry.setdefault("models", {})
    any_ok = False
    for r in rows:
        if r.get("failed"):
            tag = "env gap" if r.get("env_gap") else "FAILED"
            print(f"[{r['model']}] {tag} — not updating its baseline "
                  f"({r.get('error','')})", file=sys.stderr)
            continue
        any_ok = True
        key = r["model"]
        models[key] = dict(unit=r["unit"], direction=r["direction"],
                           value=r["throughput"], latency_ms=r["latency_ms"],
                           checkpoint=r["checkpoint"], k=r["k"],
                           warmup=r["warmup"], repeat=r["repeat"],
                           natoms_per_system=r["natoms_per_system"],
                           family=r["family"], fixture=r["fixture"], mol=r["mol"],
                           tt_atom_version=r["tt_atom_version"], ttnn_version=r["ttnn_version"],
                           date=r["date"], note=note)
        entry.update(date=r["date"], tt_atom_version=r["tt_atom_version"], note=note)
    data.pop("models", None)
    _save_baselines(data)
    seeded = [r["model"] for r in rows if not r.get("failed")]
    print(f"\nWrote {BASELINE_FILE.relative_to(REPO_ROOT)}  "
          f"(card {card}: {len(models)} model(s); seeded: {', '.join(seeded) or 'none'})")
    print("Review the diff, then commit it with the change that justifies the new numbers.")
    rows_out = []
    for r in rows:
        if r.get("failed"):
            rows_out.append({**r, "verdict": "GAP" if r.get("env_gap") else "FAIL",
                             "baseline": None, "delta": "n/a",
                             "note": f"not seeded: {r.get('error','')}"})
        else:
            rows_out.append({**r, "verdict": "PASS (baseline updated)",
                             "baseline": r["throughput"], "delta": "+0.0% (seeded)",
                             "note": f"seeded {card} baseline for {r['model']}"})
    all_ok = any_ok and all(not r.get("failed") for r in rows)
    return dict(verdict="PASS" if all_ok else "FAIL", rows=rows_out, card=card)


def _print_perf(res):
    rows = res["rows"]
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 3: PERF regression (card {res['card']}, "
          f"{len(rows)} model(s))\n{'#' * 78}")
    print(f"{'model':<34}{'baseline':>12}{'current':>12}{'delta':>10}{'verdict':>10}")
    print("-" * 78)
    for r in rows:
        key = r["model"]
        if r.get("failed") or r.get("throughput") is None:
            cur = "FAILED" if not r.get("env_gap") and r.get("verdict") == "FAIL" else "n/a"
        else:
            cur = f"{r['throughput']:.4g}"
        base = f"{r['baseline']:.4g}" if r.get("baseline") is not None else "(none)"
        delta = r.get("delta", "n/a")
        verdict = r.get("verdict", "?")
        print(f"{key:<34}{base:>12}{cur:>12}{delta:>10}{verdict:>10}")
        if r.get("note"):
            print(f"    -> {r['note']}")
    print("-" * 78)
    fails = [r for r in rows if r.get("verdict") == "FAIL"]
    gaps = [r for r in rows if r.get("verdict") == "GAP"]
    if fails:
        msg = f"GATE FAIL — {len(fails)} model(s) regressed beyond the threshold (see above)"
    elif gaps:
        msg = (f"GATE GAP — {len(gaps)} model(s) skipped (missing baseline or env); "
               f"the rest passed. Seed the missing baselines / close the env gap to clear it.")
    else:
        msg = "GATE PASS — every perf model is within threshold of its baseline"
    print(f"{'#' * 78}\n{msg}")


# ── leg 4: UX regression (sibling script) ───────────────────────────────────
# The user-experience leg: CLI --help behaves, a real tiny run's --out geometry parses
# under ase.io.read with finite energy/forces, and the CLI's per-step MD/relax progress
# stream advances through every real step (the "0 -> diffusion" bug-class analogue).
# Mirrors tt-bio's scripts/ux_regression.py in methodology; lives in a sibling script so
# it can also run standalone (`--cli-only` runs in GitHub CI with no card). See
# scripts/ux_regression.py for the leg-by-leg assertions and the Orb-vs-UMA env note.

UX_SCRIPT = REPO_ROOT / "scripts" / "ux_regression.py"


def run_ux(quick, cli_only):
    """Shell out to scripts/ux_regression.py and report its verdict. Returns a result dict.
    `quick` is accepted for API symmetry but the UX gate is already tiny (H2O, 5 MD steps)."""
    if not UX_SCRIPT.exists():
        return dict(verdict="GAP", note=f"missing {UX_SCRIPT.relative_to(REPO_ROOT)}",
                    rows=[])
    cmd = [sys.executable, str(UX_SCRIPT)]
    if cli_only:
        cmd.append("--cli-only")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    print(f"\n[ux] {' '.join(cmd[1:])}", flush=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(verdict="FAIL", note="ux_regression timed out after 600s",
                    wall_s=time.monotonic() - t0)
    wall = time.monotonic() - t0
    verdict = "PASS" if proc.returncode == 0 else "FAIL"
    note = "see scripts/ux_regression.py output above for per-leg detail"
    return dict(verdict=verdict, note=note, wall_s=round(wall, 1))


def _print_ux(res):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 4: UX regression (CLI + output parse + live progress)\n{'#' * 78}")
    print(f"verdict: {res['verdict']}  |  wall: {res.get('wall_s','-')}s")
    print(f"note: {res['note']}")
    print(f"{'#' * 78}")


# ── leg 5: clean-env install (per-release) ──────────────────────────────────
# The from-zero install a customer does, in an isolated scratch dir with its own venv and
# TT_METAL_HOME, on the pinned tt-metal commit. Proves the README "Install" works end-to-end
# (build + op load + UMA forward through fused_rotate + Orb stock-ttnn forward), not just on a
# box where everything is already built. Opt-in (`--leg install`), per-release: the build takes
# tens of minutes, so it is NOT wired into the default per-tick gate invocation.

def _run_cap(cmd, env=None, cwd=None, timeout=None, *, check=True):
    proc = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = "\n".join(detail[-10:])
        raise RuntimeError(f"{' '.join(str(c) for c in cmd)} exited {proc.returncode}"
                           + (f":\n{tail}" if tail else ""))
    return proc


def measure_install(out_path):
    """Full from-zero install + smoke in a fresh scratch dir. Writes a JSON result.

    Runs in its own subprocess (see ``_run_install``) so its device context dies before any later
    leg. Assumes the host build deps are already installed (the README's one-time
    ``sudo ./install_dependencies.sh``); the gate does not run sudo."""
    import tempfile, textwrap
    t_start = time.monotonic()
    result = dict(verdict="FAIL", pinned_commit=PINNED_TT_METAL_COMMIT,
                  branch=PINNED_TT_METAL_BRANCH, tt_atom_version=_version(),
                  card_type=detect_card_type(), date=date.today().isoformat())
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="gate-install-"))
    venv = scratch / "venv"
    venv_py = str(venv / "bin" / "python")
    venv_pip = str(venv / "bin" / "pip")
    ttm = scratch / "tt-metal"
    tta = scratch / "tt-atom"
    log = []
    clean_python = (
        getattr(sys, "_base_executable", None) or shutil.which("python3") or sys.executable
    )

    def note(msg):
        log.append(msg)
        print(msg, flush=True)

    try:
        note(f"[install] scratch={scratch}")
        source_status = _run_cap(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, timeout=30).stdout.strip()
        if source_status:
            raise RuntimeError(
                "tt-atom worktree is dirty; commit the exact source before running the install gate")
        _run_cap([clean_python, "-m", "venv", str(venv)], timeout=120)
        _run_cap([venv_pip, "install", "--upgrade", "pip", "build"], timeout=180)
        _run_cap(["git", "clone", "--recursive", "-b", PINNED_TT_METAL_BRANCH,
                  "https://github.com/tenstorrent/tt-metal.git", str(ttm)],
                 cwd=scratch, timeout=1800)
        _run_cap(["git", "checkout", PINNED_TT_METAL_COMMIT], cwd=ttm, timeout=120)
        _run_cap(["git", "submodule", "update", "--recursive", "--init"], cwd=ttm, timeout=1800)
        head = _run_cap(["git", "rev-parse", "HEAD"], cwd=ttm, timeout=30).stdout.strip()
        if head != PINNED_TT_METAL_COMMIT:
            raise RuntimeError(f"tt-metal checkout mismatch: {head} != {PINNED_TT_METAL_COMMIT}")
        result["built_commit"] = head
        note(f"[install] tt-metal HEAD={head} (pinned {PINNED_TT_METAL_COMMIT[:12]})")
        env = dict(os.environ)
        env["TT_METAL_HOME"] = str(ttm)
        t_b = time.monotonic()
        bproc = subprocess.run(["./build_metal.sh", "--build-type", "Release"],
                               cwd=ttm, env=env, timeout=INSTALL_TIMEOUT_S - 600)
        build_s = time.monotonic() - t_b
        result["build_time_s"] = round(build_s, 1)
        result["build_rc"] = bproc.returncode
        note(f"[install] build rc={bproc.returncode} time={build_s:.0f}s")
        if bproc.returncode != 0:
            result["note"] = f"build_metal.sh exited {bproc.returncode}"
            out_path.write_text(json.dumps(result))
            return
        _run_cap([venv_pip, "install", "-e", "."], cwd=ttm, env=env, timeout=600)
        source_commit = _run_cap(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, timeout=30).stdout.strip()
        source_url = _run_cap(
            ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT, timeout=30).stdout.strip()
        result["tt_atom_source_commit"] = source_commit
        note(f"[install] tt-atom source={source_commit} from {source_url}")
        _run_cap(["git", "clone", source_url, str(tta)], cwd=scratch, timeout=600)
        _run_cap(["git", "checkout", source_commit], cwd=tta, timeout=120)
        wheel_dir = scratch / "dist"
        _run_cap([venv_py, "-m", "build", "--wheel", "--outdir", str(wheel_dir)],
                 cwd=tta, env=env, timeout=600)
        wheels = list(wheel_dir.glob("tt_atom-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one tt-atom wheel, found {wheels}")
        # Force the candidate wheel and its runtime dependencies into the isolated venv.
        # Hide the gate driver's source tree so its egg-info cannot make pip mistake the
        # checkout for an installed candidate.
        install_env = dict(env)
        install_env.pop("PYTHONPATH", None)
        _run_cap(
            [venv_py, "-m", "pip", "install", "--force-reinstall", str(wheels[0])],
            cwd=scratch, env=install_env, timeout=600)
        # reference env for the one-time Orb weight export (numpy>=2, has orb-models). Pinned to
        # 0.5.5 — the export tool's target; newer orb-models changed the pretrained API. UMA's
        # real-weight bundle would additionally need fairchem-core here (see README), but this
        # gate's UMA smoke uses the committed random-weight golden, so orb-models alone suffices.
        refenv = scratch / "refenv"
        refenv_py = str(refenv / "bin" / "python")
        refenv_pip = str(refenv / "bin" / "pip")
        _run_cap([clean_python, "-m", "venv", str(refenv)], timeout=120)
        _run_cap([refenv_pip, "install", "--upgrade", "pip"], timeout=180)
        _run_cap([refenv_pip, "install", "orb-models==0.5.5"], timeout=600)
        sm_env = dict(env)
        sm_env.pop("PYTHONPATH", None)
        sm_env["TT_VISIBLE_DEVICES"] = os.environ.get("TT_VISIBLE_DEVICES", "0")
        sm_env["TT_METAL_LOGGER_LEVEL"] = "FATAL"
        sm_env["TT_ATOM_REFENV"] = refenv_py
        sm_env["TT_ATOM_CACHE"] = str(scratch / "cache")
        # one-time orb checkpoint download (ungated); off would block the first-use export
        sm_env["HF_HUB_OFFLINE"] = "0"
        # this board's UMD misreads the board ID without an explicit mesh graph descriptor
        mesh_desc = ttm / "tt_metal" / "fabric" / "mesh_graph_descriptors" / "p150_mesh_graph_descriptor.textproto"
        if mesh_desc.exists():
            sm_env["TT_MESH_GRAPH_DESC_PATH"] = str(mesh_desc)
        opc = _run_cap([venv_py, "-c",
                        "import ttnn; e=ttnn._ttnn.operations.experimental; "
                        "print(hasattr(e,'fused_rotate'), hasattr(e,'fused_rotate_gc'))"],
                       env=sm_env, cwd=scratch, timeout=300, check=False)
        op_out = opc.stdout.strip()
        result["fused_rotate"] = op_out
        op_ok = opc.returncode == 0 and op_out == "True True"
        note(f"[install] op check: {op_out!r} rc={opc.returncode}")
        pkg = _run_cap(
            [venv_py, "-c",
             "import importlib.resources as r, tt_atom; "
             "print(tt_atom.__file__); "
             "print((r.files('tools')/'export_weights.py').is_file(), "
             "(r.files('tools')/'export_orb_weights.py').is_file())"],
            env=sm_env, cwd=scratch, timeout=120, check=False)
        pkg_lines = pkg.stdout.strip().splitlines()
        result["tt_atom_import"] = pkg_lines[0] if pkg_lines else ""
        result["exporters"] = pkg_lines[-1] if pkg_lines else ""
        wheel_ok = (
            pkg.returncode == 0
            and "site-packages" in result["tt_atom_import"]
            and result["exporters"] == "True True"
        )
        result["wheel_ok"] = wheel_ok
        note(f"[install] wheel import={result['tt_atom_import']!r}; "
             f"exporters={result['exporters']!r}")
        uma_script = tta / "scripts" / "_install_gate_uma_smoke.py"
        uma_script.write_text(textwrap.dedent('''
            import math, pathlib, sys
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
            from util import Golden, pcc
            import torch
            from tt_atom.model import Backbone
            from tt_atom.geometry import HostGeometry
            from tt_atom import forces, device as D
            g = Golden("golden_tiny.npz")
            cfg = dict(g.config); w = g.w()
            dev = D.open_device(0)
            try:
                bb = Backbone(w, dev, cfg, g.host("to_grid_mat"), g.host("from_grid_mat"))
                geo = HostGeometry(w, cfg, g.host("to_m"), g.host("gauss_offset"), g.host("gauss_coeff"), gamma=0.0)
                E, F = forces.energy_and_forces(bb, geo, g.inp("pos").float(), g.inp("atomic_numbers").long(),
                                                g.inp("edge_index").long(), g.host("sys_node_embedding"))
            finally:
                import ttnn; ttnn.close_device(dev)
            Eref = float(g.out("energy").reshape(-1)[0]); Fref = g.out("forces")
            rel = abs(E - Eref) / (abs(Eref) + 1e-6); p = pcc(F, Fref)
            finite = math.isfinite(E) and bool((F == F).all()) and math.isfinite(p)
            print("UMA_SMOKE_JSON", __import__("json").dumps(
                {"energy": E, "eref": Eref, "rel": rel, "force_pcc": p, "finite": finite}))
            assert finite and rel < 0.05 and p >= 0.98
        '''))
        uma = _run_cap([venv_py, str(uma_script)], env=sm_env, cwd=scratch,
                       timeout=900, check=False)
        result["uma_smoke_rc"] = uma.returncode
        result["uma_smoke_tail"] = (uma.stdout + uma.stderr)[-1500:]
        uma_ok = False
        try:
            line = [ln for ln in uma.stdout.splitlines() if ln.startswith("UMA_SMOKE_JSON")][-1]
            ud = json.loads(line.split(" ", 1)[1])
            result["uma_energy"] = ud.get("energy")
            result["uma_energy_ref"] = ud.get("eref")
            result["uma_energy_rel"] = ud.get("rel")
            result["uma_force_pcc"] = ud.get("force_pcc")
            uma_ok = uma.returncode == 0 and bool(ud.get("finite"))
        except Exception as e:
            result["uma_parse_err"] = f"{type(e).__name__}: {e}"
        note(f"[install] UMA end2end smoke rc={uma.returncode} E={result.get('uma_energy')} pcc={result.get('uma_force_pcc')}")
        cli_out = scratch / "cli_relaxed.xyz"
        cli = _run_cap(
            [venv_py, "-m", "tt_atom.cli", "relax",
             str(tta / "examples" / "model_tiny_demo.npz"),
             "--molecule", "H2O", "--steps", "1", "--out", str(cli_out)],
            env=sm_env, cwd=scratch, timeout=900, check=False)
        cli_parse = _run_cap(
            [venv_py, "-c",
             "from ase.io import read; import sys; "
             "a=read(sys.argv[1]); assert len(a)==3 and a.positions.shape==(3,3)",
             str(cli_out)],
            env=sm_env, cwd=scratch, timeout=120, check=False) if cli_out.exists() else None
        cli_ok = cli.returncode == 0 and cli_parse is not None and cli_parse.returncode == 0
        result["cli_smoke_rc"] = cli.returncode
        result["cli_parse_rc"] = None if cli_parse is None else cli_parse.returncode
        result["cli_smoke_tail"] = (cli.stdout + cli.stderr)[-1500:]
        result["cli_ok"] = cli_ok
        note(f"[install] CLI relax+parse smoke rc={cli.returncode} "
             f"parse_rc={result['cli_parse_rc']}")
        orb_script = tta / "scripts" / "_install_gate_orb_smoke.py"
        orb_script.write_text(textwrap.dedent('''
            from ase.build import molecule
            from tt_atom import Calculator
            import math, json
            a = molecule("H2O"); a.calc = Calculator(a, "orb-v3-conservative-omol")
            E = float(a.get_potential_energy()); F = a.get_forces()
            fnorm = float((F * F).sum() ** 0.5)
            finite = math.isfinite(E) and bool((F == F).all()) and math.isfinite(fnorm)
            print("ORB_SMOKE_JSON", json.dumps({"energy": E, "force_norm": fnorm, "finite": finite}))
            assert finite and abs(E) > 0
        '''))
        orb = _run_cap([venv_py, str(orb_script)], env=sm_env, cwd=scratch,
                       timeout=900, check=False)
        result["orb_smoke_rc"] = orb.returncode
        result["orb_smoke_tail"] = (orb.stdout + orb.stderr)[-1500:]
        orb_ok = orb.returncode == 0
        try:
            line = [ln for ln in orb.stdout.splitlines() if ln.startswith("ORB_SMOKE_JSON")][-1]
            od = json.loads(line.split(" ", 1)[1])
            result["orb_energy"] = od.get("energy")
            result["orb_force_norm"] = od.get("force_norm")
            orb_ok = orb_ok and bool(od.get("finite"))
        except Exception as e:
            result["orb_parse_err"] = f"{type(e).__name__}: {e}"
            orb_ok = False
        note(f"[install] Orb smoke rc={orb.returncode} E={result.get('orb_energy')} |F|={result.get('orb_force_norm')}")
        result["op_ok"], result["uma_ok"], result["orb_ok"] = op_ok, uma_ok, orb_ok
        if wheel_ok and op_ok and uma_ok and orb_ok and cli_ok:
            result["verdict"] = "PASS"
            result["note"] = (f"built-wheel install + smoke passed on tt-metal {head[:12]} "
                              f"(build {build_s:.0f}s); fused_rotate={op_out}, UMA end2end parity PASSED, "
                              f"CLI relax+parse PASSED, Orb E={result.get('orb_energy')}")
        else:
            result["verdict"] = "FAIL"
            result["note"] = (f"wheel_ok={wheel_ok} op_ok={op_ok} uma_ok={uma_ok} cli_ok={cli_ok} "
                              f"orb_ok={orb_ok}")
    except subprocess.TimeoutExpired as e:
        result["verdict"] = "FAIL"
        result["note"] = f"timeout: {e}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["verdict"] = "FAIL"
        result["note"] = f"{type(e).__name__}: {e}"
    finally:
        result["wall_s"] = round(time.monotonic() - t_start, 1)
        result["log_tail"] = "\n".join(log[-30:])
        out_path.write_text(json.dumps(result))
        if result.get("verdict") == "PASS":
            shutil.rmtree(scratch, ignore_errors=True)


def _run_install():
    """Spawn the clean-env install leg in a fresh process (one device context for the smoke)."""
    td = tempfile.mkdtemp(prefix="gate-install-parent-")
    out = pathlib.Path(td) / "result.json"
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
           "--measure-install", "--out", str(out)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[install] TIMED OUT after {INSTALL_TIMEOUT_S}s — stopping child and resetting card",
              file=sys.stderr)
        reset_error = _stop_child_and_reset(proc)
        shutil.rmtree(td, ignore_errors=True)
        note = f"install leg timed out after {INSTALL_TIMEOUT_S}s"
        if reset_error:
            note += f"; card reset failed: {reset_error}"
        return dict(verdict="FAIL", note=note)
    if returncode != 0:
        shutil.rmtree(td, ignore_errors=True)
        return dict(verdict="FAIL", note=f"install subprocess exited {returncode}")
    if not out.exists():
        shutil.rmtree(td, ignore_errors=True)
        return dict(verdict="FAIL", note=f"install subprocess exited {returncode} (no result json)")
    try:
        return json.loads(out.read_text())
    except Exception as e:
        return dict(verdict="FAIL", note=f"install result parse failed: {e}")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _print_install(res):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 5: CLEAN-ENV INSTALL (per-release, from-zero on pinned commit)\n{'#' * 78}")
    print(f"tt-atom source commit:  {res.get('tt_atom_source_commit', '?')}")
    print(f"pinned tt-metal commit: {res.get('pinned_commit', '?')}  (branch {res.get('branch', '?')})")
    print(f"built commit:           {res.get('built_commit', '?')}")
    print(f"build time:             {res.get('build_time_s', '?')}s  (rc={res.get('build_rc', '?')})")
    print(f"installed package:      {res.get('tt_atom_import', '?')}  "
          f"exporters={res.get('exporters', '?')}  -> {'PASS' if res.get('wheel_ok') else 'FAIL'}")
    print(f"fused_rotate / gc:      {res.get('fused_rotate', '?')}")
    print(f"UMA smoke (end2end):    rc={res.get('uma_smoke_rc', '?')}  E={res.get('uma_energy', '?')} (rel={res.get('uma_energy_rel', '?')}, pcc={res.get('uma_force_pcc', '?')})  -> {'PASS' if res.get('uma_ok') else 'FAIL'}")
    print(f"CLI relax + parse:      rc={res.get('cli_smoke_rc', '?')}/{res.get('cli_parse_rc', '?')}  -> {'PASS' if res.get('cli_ok') else 'FAIL'}")
    print(f"Orb smoke (stock ttnn): rc={res.get('orb_smoke_rc', '?')}  E={res.get('orb_energy', '?')}  |F|={res.get('orb_force_norm', '?')}  -> {'PASS' if res.get('orb_ok') else 'FAIL'}")
    print(f"wall: {res.get('wall_s', '?')}s")
    print(f"verdict: {res.get('verdict', '?')}")
    if res.get("note"):
        print(f"note: {res['note']}")
    for name in ("uma", "cli", "orb"):
        tail = res.get(f"{name}_smoke_tail")
        if tail and not res.get(f"{name}_ok"):
            print(f"{name} failure tail:\n{tail}")
    print(f"{'#' * 78}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["accuracy", "oom", "perf", "ux", "install"], action="append",
                    help="Run only this leg (repeatable). Default: the four per-tick legs "
                         "(accuracy/oom/perf/ux). `install` is the per-release clean-env install "
                         "leg (tens of minutes, opt-in — see RELEASING.md leg 5); it is NOT in the "
                         "default set. `ux` shells out to scripts/ux_regression.py "
                         "(CLI --help + output parses + live progress advances on a tiny H2O, Orb path).")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke: fewer accuracy modules, smaller OOM sweep, fewer perf iters.")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="Diagnostics only: return success when required rows are GAP. "
                         "The release default requires PASS for every selected row.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Perf regression %% allowed before FAIL (default {DEFAULT_THRESHOLD:g}).")
    ap.add_argument("--model", action="append", choices=list(PERF_SPECS),
                    help="Perf leg only: gate this model (repeatable). Default: every shipped "
                         "perf model. Choose from: " + ", ".join(PERF_SPECS) + ".")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Perf leg only: seed docs/perf_baselines.json from this warm run "
                         "instead of gating. Writes only the model(s) selected via --model (or "
                         "all, if none selected) that measured successfully. Requires --note.")
    ap.add_argument("--note", default=None,
                    help="Required with --update-baseline: why this perf change is intended.")
    ap.add_argument("--cli-only", action="store_true",
                    help="UX leg only: run scripts/ux_regression.py --cli-only (no card). "
                         "Convenience passthrough; ignored for other legs.")
    # internal: one in-process family OOM sweep subprocess
    ap.add_argument("--measure-oom", choices=list(OOM_SPECS), help=argparse.SUPPRESS)
    # internal: the per-model in-process perf measurement subprocess
    ap.add_argument("--measure-perf", metavar="MODEL", help=argparse.SUPPRESS)
    # internal: the clean-env install leg subprocess
    ap.add_argument("--measure-install", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.measure_oom is not None:
        if args.out is None:
            sys.exit("--out is required with --measure-oom")
        measure_oom(args.measure_oom, args.out, args.quick)
        return 0

    if args.measure_perf is not None:
        if args.measure_perf not in PERF_SPECS:
            sys.exit(f"unknown perf model {args.measure_perf!r}")
        if args.out is None:
            sys.exit("--out is required with --measure-perf")
        measure_perf(args.measure_perf, args.out, args.quick)
        return 0

    if args.measure_install:
        if args.out is None:
            sys.exit("--out is required with --measure-install")
        measure_install(args.out)
        return 0

    legs = args.leg or ["accuracy", "oom", "perf", "ux"]
    overall_pass = True
    ran_any = False

    if "accuracy" in legs:
        rows, acc_pass = run_accuracy(ACCURACY_SPECS, args.quick)
        _print_accuracy(rows, acc_pass)
        has_gap = any(r["verdict"] == "GAP" for r in rows)
        overall_pass &= acc_pass and (args.allow_gaps or not has_gap)
        ran_any = True

    if "oom" in legs:
        res = _run_oom(args.quick)
        _print_oom(res)
        if res["verdict"] != "PASS" and not (
                args.allow_gaps and res["verdict"] == "GAP"):
            overall_pass = False
        ran_any = True

    if "perf" in legs:
        models = args.model or DEFAULT_PERF_MODELS
        res = run_perf(args.quick, models, args.update_baseline, args.note, args.threshold)
        _print_perf(res)
        has_gap = any(r.get("verdict") == "GAP" for r in res.get("rows", []))
        if res["verdict"] == "FAIL" or (has_gap and not args.allow_gaps):
            overall_pass = False
        ran_any = True

    if "ux" in legs:
        res = run_ux(args.quick, args.cli_only)
        _print_ux(res)
        if res["verdict"] != "PASS" and not (
                args.allow_gaps and res["verdict"] == "GAP"):
            overall_pass = False
        ran_any = True

    if "install" in legs:
        res = _run_install()
        _print_install(res)
        if res.get("verdict") != "PASS" and not (
                args.allow_gaps and res.get("verdict") == "GAP"):
            overall_pass = False
        ran_any = True

    print(f"\n{'#' * 78}\nRELEASE GATE SUMMARY — legs: {', '.join(legs)}\n{'#' * 78}")
    if not ran_any:
        print("no legs selected")
    gap_policy = "allowed for diagnostics" if args.allow_gaps else "release-blocking"
    print("OVERALL: " + ("PASS" if overall_pass else "FAIL") +
          f"  (GAP rows are {gap_policy})")
    print(f"{'#' * 78}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
