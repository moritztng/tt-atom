#!/usr/bin/env python3
"""TT-Atom release gate — one command, four legs, machine-readable PASS/FAIL/GAP per leg.

The runnable equivalent of ``RELEASING.md``'s manual checklist, and the live harness behind the
parity table in ``docs/materials-benchmark.md`` (run ``--leg accuracy`` to reproduce those R/D/X
numbers on your card). Four legs, matching the four things a tagged release must clear on real
hardware before it ships:

  1. ACCURACY — numerical parity vs each shipped model family's own reference, within tolerance
     (energy rel-error, force/stress PCC), across every supported task and graph regime for which
     a real-weight golden fixture exists. Reuses the existing ``tests/test_*realweight*.py``
     parity modules verbatim (they already encode the bars and the real oracles — fairchem
     ``uma-s-1`` for UMA, the real ``orb-models`` CPU oracle for Orb-v3 / OrbMol) by invoking
     pytest as a subprocess with JUnit XML, so the gate never re-derives a parity bar or oracle.
     A module whose golden is absent auto-skips; the gate reports that leg as GAP (missing
     fixture), never as a silent PASS.
  2. NO-OOM — runs the supported size range on the card to completion and reports the largest
     size that cleared. Orb family: a disjoint-union batch sweep (``OrbCalculator.evaluate_batch``
     over K=1..K_max small systems in one device forward), reusing
     ``benchmarks/bench_orb_evaluate_batch.py``'s exact protocol — the batch ceiling is the OOM
     frontier. UMA OOM sweep is a documented GAP: UMA's batched forward goes through the same
     ALWAYS-ON ``fused_rotate`` kernel as its accuracy leg, absent from this host's ttnn build
     (memory pc-ttatom-env-missing-fused-rotate); the per-composition bundle is not the blocker.
  3. PERF — warm steady-state throughput on a fixed small input vs a committed per-card,
     per-model baseline (``docs/perf_baselines.json``), FAILs beyond a configurable noise
     margin. One entry per shipped family's throughput path (OrbMol ``conservative-omol``,
     Orb-v3 bulk ``conservative-inf-omat``, UMA ``uma-s-1``), mirroring tt-bio's
     ``scripts/perf_regression.py``: a ``--model`` flag iterates a SPECS-style dict, one
     baseline entry per model per card. Card-type-aware (a P300c baseline is never judged
     against a P150a run), fails loudly on NO BASELINE, and updates only via
     ``--update-baseline --note "<why>"``. Seeds the baseline the first time a card type is
     run for a model. UMA's batched forward needs the ALWAYS-ON ``fused_rotate`` kernel
     absent from this host's ttnn build (memory pc-ttatom-env-missing-fused-rotate), so on
     such a host the UMA row reports GAP (env), not FAIL — reported loudly, not skipped.
  4. UX — the user-facing plumbing still works headlessly on a tiny input (H2O): CLI --help
     behaves and lists the core flags, a real single-point + relax + MD(--steps 5) write an
     --out geometry that parses under ase.io.read with finite energy/forces, and the CLI's
     per-step MD/relax progress stream advances through every real step (the "0 -> diffusion"
     bug-class analogue). Mirrors tt-bio's scripts/ux_regression.py in methodology; lives in
     the sibling scripts/ux_regression.py (also runnable standalone, --cli-only needs no card).

Honest reporting: every leg prints PASS / FAIL / GAP with the real numbers (or the real skip
reason). Nothing fabricated. Exit 0 iff every leg that ran PASSES (GAP does not fail the gate —
it is reported, not hidden — but a leg that ran and FAILED does).

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
    dict(family="orbmol", checkpoint="conservative-omol", regime="molecule / charged / openshell",
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
# Orb family disjoint-union batch sweep: K small systems in one device forward. The largest K
# that clears is the batch ceiling (the OOM frontier). Same protocol as
# benchmarks/bench_orb_evaluate_batch.py. UMA's OOM sweep is a GAP: UMA's batched forward
# (TTAtomCalculator.evaluate_batch -> energy_and_forces_batch -> edgewise -> rotation.rotate)
# goes through the same ALWAYS-ON fused_rotate kernel as its accuracy leg's end-to-end test,
# which is absent from this host's ttnn build (see memory pc-ttatom-env-missing-fused-rotate /
# ttatom-uma-fused-rotate-env-downgrade-reverted). The per-composition bundle itself is not the
# blocker — the bundle cache works and evaluate_batch enforces same-composition batching; the
# blocker is the env. Reported as GAP, not forced to a number; closes automatically once the
# fused_rotate env is rebuilt on the release host.
OOM_CHECKPOINT = "orb-v3-conservative-omol"
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
# leg, absent from this host's ttnn build (memory pc-ttatom-env-missing-fused-rotate), so on such
# a host the UMA row measures as GAP (env), not FAIL — the gate reports it loudly rather than
# silently skipping the family. Once the fused_rotate env is rebuilt on the release host, the
# UMA row measures and gates against its seeded baseline like any other model.
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
                else:
                    passed += 1
        except ET.ParseError:
            failed = -1
    if proc.returncode != 0 and failed == 0:
        failed = 1
    verdict = "PASS" if (failed == 0 and skipped == 0) else (
        "GAP" if (failed == 0 and skipped > 0) else "FAIL")
    note = ""
    if skipped > 0:
        note = f"{skipped} skipped (missing fixture or condition)"
    if fail_names:
        note = (note + "; " if note else "") + "failed: " + ",".join(fail_names[:3])
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
        msg = f"GATE PASS with GAP — {len(gaps)} module(s) skipped (missing fixture); " \
              f"the rest passed. Generate the missing goldens to close the gap."
    else:
        msg = "GATE PASS — every accuracy module passed parity vs its real oracle"
    print(f"{'#' * 78}\n{msg}")


# ── leg 2 implementation ─────────────────────────────────────────────────────

def run_oom(quick):
    """Orb disjoint-union batch sweep — find the largest K (small systems in one device forward)
    that clears without OOM. Returns a result dict. UMA OOM sweep is a documented GAP."""
    ks = OOM_KS_QUICK if quick else OOM_KS_DEFAULT
    print(f"\n[oom] Orb disjoint-union batch sweep: checkpoint={OOM_CHECKPOINT} "
          f"mol={OOM_MOL} K={ks}", flush=True)
    try:
        from ase.build import molecule
        from tt_atom.orb_calculator import OrbCalculator
    except Exception as e:
        return dict(verdict="GAP", ceiling=None, failed_at=None,
                    note=f"OrbCalculator import failed: {e}", rows=[])
    dev_id = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    try:
        calc = OrbCalculator.from_checkpoint(OOM_CHECKPOINT, device_id=dev_id)
    except Exception as e:
        return dict(verdict="GAP", ceiling=None, failed_at=None,
                    note=f"from_checkpoint failed (weights cached? refenv?): {e}", rows=[])
    natoms = len(molecule(OOM_MOL))
    rows = []
    ceiling = None
    failed_at = None
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
            rows.append(dict(K=k, natoms_total=int(natoms * k), ok=False, err=msg))
            failed_at = k
            print(f"  K={k:4d}  Ntot={natoms*k:5d}  OOM/err: {msg}")
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            rows.append(dict(K=k, natoms_total=int(natoms * k), ok=False, err=msg))
            failed_at = k
            print(f"  K={k:4d}  Ntot={natoms*k:5d}  err: {msg}")
            break
    try:
        calc.close()
    except Exception:
        pass
    verdict = "PASS" if ceiling == ks[-1] else ("FAIL" if failed_at is not None else "GAP")
    note = (f"batch ceiling = {ceiling} systems ({None if ceiling is None else natoms*ceiling} atoms) "
            f"in one device forward; UMA OOM sweep is a GAP — UMA's batched forward goes through the "
            f"same ALWAYS-ON fused_rotate kernel as its accuracy leg, absent from this host's ttnn "
            f"build (memory pc-ttatom-env-missing-fused-rotate), not a per-composition-bundle issue")
    return dict(verdict=verdict, ceiling=ceiling, failed_at=failed_at, note=note, rows=rows)


def _print_oom(res):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 2: NO-OOM sweep (single card, Orb disjoint-union batch)\n{'#' * 78}")
    print(f"{'K':<8}{'Ntot':>8}  result")
    for r in res["rows"]:
        ok = "OK" if r["ok"] else f"OOM/err: {r.get('err','')}"
        print(f"{r['K']:<8}{r['natoms_total']:>8}  {ok}")
    print("-" * 78)
    print(f"ceiling: {res['ceiling']}  |  failed_at: {res['failed_at']}  ->  {res['verdict']}")
    print(f"note: {res['note']}")
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
    spec = PERF_SPECS[model]
    warmup = PERF_WARMUP_QUICK if quick else PERF_WARMUP
    repeat = PERF_REPEAT_QUICK if quick else PERF_REPEAT
    dev_id = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    try:
        if spec["kind"] == "orb-batch":
            from tt_atom.orb_calculator import OrbCalculator
            calc = OrbCalculator.from_checkpoint(spec["checkpoint"], device_id=dev_id)
        elif spec["kind"] == "uma-batch":
            from tt_atom.calculator import TTAtomCalculator
            seed = _perf_systems(spec, 1)[0]
            calc = TTAtomCalculator.from_uma(model=spec["checkpoint"], task_name="omol",
                                             atoms=seed, device_id=dev_id)
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
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if not out.exists():
        shutil.rmtree(td, ignore_errors=True)
        return dict(model=model, failed=True,
                    error=f"subprocess exited {proc.returncode} (no result json)")
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
    for r in rows:
        if r.get("failed"):
            if r.get("env_gap"):
                r["verdict"] = "GAP"
                r["baseline"] = None
                r["delta"] = "n/a"
                r["note"] = (f"env gap: {r.get('error','')} — UMA's batched forward needs the "
                             f"custom fused_rotate ttnn op, absent from this host's ttnn build "
                             f"(memory pc-ttatom-env-missing-fused-rotate); not a perf regression")
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
            r["baseline"] = None
            r["delta"] = "n/a"
            r["note"] = (f"NO BASELINE for card '{card}' / model '{key}' in "
                     f"{BASELINE_FILE.relative_to(REPO_ROOT)}. Seed it with: "
                     f"python3 scripts/release_gate.py --leg perf --model {key} "
                     f"--update-baseline --note \"seed {card} baseline\"")
            continue
        base = float(bm[key]["value"])
        pct = (r["throughput"] - base) / base * 100.0
        r["baseline"] = base
        r["delta"] = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
        r["verdict"] = "PASS" if pct >= -threshold else "FAIL"
        r["note"] = f"vs baseline {base:.4g} {r['unit']} (threshold -{threshold:.0f}%)"
        if r["verdict"] == "FAIL":
            overall_pass = False
    return dict(verdict="PASS" if overall_pass else "FAIL", rows=rows, card=card)


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
                           tt_atom_version=r["tt_atom_version"], date=r["date"], note=note)
        entry.setdefault("date", r["date"])
        entry.setdefault("tt_atom_version", r["tt_atom_version"])
        entry.setdefault("note", note)
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
    return dict(verdict="PASS" if any_ok else "FAIL", rows=rows_out, card=card)


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
        msg = (f"GATE PASS with GAP — {len(gaps)} model(s) skipped (missing baseline or env); "
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
        return dict(verdict="FAIL", note=f"ux_regression timed out after 600s",
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["accuracy", "oom", "perf", "ux"], action="append",
                    help="Run only this leg (repeatable). Default: all four. `ux` shells "
                         "out to scripts/ux_regression.py (CLI --help + output parses + "
                         "live progress advances on a tiny H2O, Orb path).")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke: fewer accuracy modules, smaller OOM sweep, fewer perf iters.")
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
    # internal: the per-model in-process perf measurement subprocess
    ap.add_argument("--measure-perf", metavar="MODEL", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=pathlib.Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.measure_perf is not None:
        if args.measure_perf not in PERF_SPECS:
            sys.exit(f"unknown perf model {args.measure_perf!r}")
        if args.out is None:
            sys.exit("--out is required with --measure-perf")
        measure_perf(args.measure_perf, args.out, args.quick)
        return 0

    legs = args.leg or ["accuracy", "oom", "perf", "ux"]
    overall_pass = True
    ran_any = False

    if "accuracy" in legs:
        rows, acc_pass = run_accuracy(ACCURACY_SPECS, args.quick)
        _print_accuracy(rows, acc_pass)
        overall_pass &= acc_pass
        ran_any = True

    if "oom" in legs:
        res = run_oom(args.quick)
        _print_oom(res)
        if res["verdict"] == "FAIL":
            overall_pass = False
        ran_any = True

    if "perf" in legs:
        models = args.model or DEFAULT_PERF_MODELS
        res = run_perf(args.quick, models, args.update_baseline, args.note, args.threshold)
        _print_perf(res)
        if res["verdict"] == "FAIL":
            overall_pass = False
        ran_any = True

    if "ux" in legs:
        res = run_ux(args.quick, args.cli_only)
        _print_ux(res)
        if res["verdict"] == "FAIL":
            overall_pass = False
        ran_any = True

    print(f"\n{'#' * 78}\nRELEASE GATE SUMMARY — legs: {', '.join(legs)}\n{'#' * 78}")
    if not ran_any:
        print("no legs selected")
    print("OVERALL: " + ("PASS" if overall_pass else "FAIL") +
          "  (GAP legs are reported, not counted as failures)")
    print(f"{'#' * 78}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
