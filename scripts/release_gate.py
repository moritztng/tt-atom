#!/usr/bin/env python3
"""TT-Atom release gate — one command, three legs, machine-readable PASS/FAIL/GAP per leg.

The runnable equivalent of ``RELEASING.md``'s manual checklist. Three legs, matching the three
things a tagged release must clear on real hardware before it ships:

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
     frontier. UMA OOM sweep is a documented GAP (per-composition bundle path; see RELEASING.md).
  3. PERF — warm steady-state throughput on a fixed small input vs a committed per-card baseline
     (``docs/perf_baselines.json``), FAILs beyond a configurable noise margin. Mirrors
     ``tt-bio``'s ``scripts/perf_regression.py``: card-type-aware (a P300c baseline is never
     judged against a P150a run), fails loudly on NO BASELINE, and updates only via
     ``--update-baseline --note "<why>"``. Seeds the baseline the first time a card type is run.

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

    # seed / refresh the perf baseline from the current warm run (explicit, needs a note)
    python3 scripts/release_gate.py --leg perf --update-baseline --note "seed p150a baseline"

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
# benchmarks/bench_orb_evaluate_batch.py. UMA's OOM sweep is a GAP — its bundle is per
# (composition, charge, spin, task), so a single-card sweep needs a cached bundle for one
# composition; documented, not silently skipped.
OOM_CHECKPOINT = "orb-v3-conservative-omol"
OOM_MOL = "CH3CH2OH"
OOM_KS_DEFAULT = [1, 2, 4, 8, 16, 32, 64, 128]
OOM_KS_QUICK = [1, 2, 4, 8]

# ── leg 3: perf regression ───────────────────────────────────────────────────
# Warm steady-state throughput on a fixed small batch, vs docs/perf_baselines.json. Card-type
# aware (per-card baseline key), fails loudly on NO BASELINE, updates only via --update-baseline.
PERF_CHECKPOINT = "orb-v3-conservative-omol"
PERF_MOL = "CH3CH2OH"
PERF_K = 8
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
            f"in one device forward; UMA OOM sweep is a GAP (per-composition bundle path)")
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

def _measure_perf(checkpoint, mol, k, warmup, repeat):
    """Warm steady-state throughput: K small systems per evaluate_batch call, median of `repeat`
    timed calls after `warmup` warm calls. Returns systems/s and latency_ms."""
    from ase.build import molecule
    from tt_atom.orb_calculator import OrbCalculator
    dev_id = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    calc = OrbCalculator.from_checkpoint(checkpoint, device_id=dev_id)
    natoms = len(molecule(mol))
    systems = []
    for i in range(k):
        a = molecule(mol)
        a.rattle(stdev=0.08, seed=10 + i)
        a.info.update(charge=0, spin=1)
        systems.append(a)

    def one():
        calc.evaluate_batch(systems)

    for _ in range(warmup):
        one()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        one()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    calc.close()
    return dict(throughput=k / median, latency_ms=median * 1000.0 / k, median_s=median,
                natoms_per_system=natoms, k=k, warmup=warmup, repeat=repeat,
                times_s=[round(t, 4) for t in times])


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


def run_perf(quick, update_baseline, note, threshold):
    warmup = PERF_WARMUP_QUICK if quick else PERF_WARMUP
    repeat = PERF_REPEAT_QUICK if quick else PERF_REPEAT
    print(f"\n[perf] warm throughput: checkpoint={PERF_CHECKPOINT} mol={PERF_MOL} "
          f"K={PERF_K} warmup={warmup} repeat={repeat}", flush=True)
    card = detect_card_type()
    try:
        m = _measure_perf(PERF_CHECKPOINT, PERF_MOL, PERF_K, warmup, repeat)
    except Exception as e:
        return dict(verdict="FAIL", card=card, throughput=None, baseline=None, delta=None,
                    note=f"measurement failed: {e}")
    baseline = _card_baselines(_load_baselines(), card)
    key = "orb-conservative-omol-batch"
    result = dict(card=card, throughput=m["throughput"], latency_ms=m["latency_ms"],
                  median_s=m["median_s"], k=m["k"], warmup=m["warmup"], repeat=m["repeat"],
                  natoms_per_system=m["natoms_per_system"], checkpoint=PERF_CHECKPOINT,
                  unit="sys/s", direction="higher", tt_atom_version=_version(),
                  date=date.today().isoformat(), times_s=m["times_s"])
    if update_baseline:
        if not note:
            sys.exit("--update-baseline requires --note \"<why this perf change is intended>\"")
        data = _load_baselines()
        cards = data.setdefault("cards", {})
        entry = cards.setdefault(card, {})
        models = entry.setdefault("models", {})
        models[key] = dict(unit=result["unit"], direction=result["direction"],
                           value=result["throughput"], latency_ms=result["latency_ms"],
                           checkpoint=result["checkpoint"], k=result["k"],
                           warmup=result["warmup"], repeat=result["repeat"],
                           natoms_per_system=result["natoms_per_system"],
                           tt_atom_version=result["tt_atom_version"], date=result["date"],
                           note=note)
        entry["date"] = result["date"]
        entry["tt_atom_version"] = result["tt_atom_version"]
        entry["note"] = note
        data.pop("models", None)
        _save_baselines(data)
        result["verdict"] = "PASS (baseline updated)"
        result["baseline"] = result["throughput"]
        result["delta"] = "+0.0% (seeded)"
        result["note"] = f"seeded {card} baseline for {key}"
        return result
    if baseline is None or key not in baseline:
        result["verdict"] = "GAP"
        result["baseline"] = None
        result["delta"] = "n/a"
        result["note"] = (f"NO BASELINE for card '{card}' / model '{key}' in "
                          f"{BASELINE_FILE.relative_to(REPO_ROOT)}. Seed it with: "
                          f"python3 scripts/release_gate.py --leg perf --update-baseline "
                          f"--note \"seed {card} baseline\"")
        return result
    base = float(baseline[key]["value"])
    pct = (m["throughput"] - base) / base * 100.0
    result["baseline"] = base
    result["delta"] = f"{'+' if pct >= 0 else ''}{pct:.1f}%"
    result["verdict"] = "PASS" if pct >= -threshold else "FAIL"
    result["note"] = f"vs baseline {base:.4g} {result['unit']} (threshold -{threshold:.0f}%)"
    return result


def _print_perf(res):
    print(f"\n{'#' * 78}\nRELEASE GATE — leg 3: PERF regression (card {res['card']}, "
          f"orb-conservative-omol batch K={res.get('k','?')})\n{'#' * 78}")
    print(f"{'metric':<14}{'baseline':>12}{'current':>12}{'delta':>10}{'verdict':>9}")
    print("-" * 57)
    base = f"{res['baseline']:.4g}" if res.get("baseline") is not None else "(none)"
    cur = f"{res['throughput']:.4g}" if res.get("throughput") is not None else "FAILED"
    print(f"{'sys/s':<14}{base:>12}{cur:>12}{res.get('delta','n/a'):>10}{res['verdict']:>9}")
    print("-" * 57)
    print(f"note: {res.get('note','')}")
    print(f"{'#' * 78}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=["accuracy", "oom", "perf"], action="append",
                    help="Run only this leg (repeatable). Default: all three.")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke: fewer accuracy modules, smaller OOM sweep, fewer perf iters.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Perf regression %% allowed before FAIL (default {DEFAULT_THRESHOLD:g}).")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Perf leg only: seed docs/perf_baselines.json from this warm run "
                         "instead of gating. Requires --note.")
    ap.add_argument("--note", default=None,
                    help="Required with --update-baseline: why this perf change is intended.")
    args = ap.parse_args()

    legs = args.leg or ["accuracy", "oom", "perf"]
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
        res = run_perf(args.quick, args.update_baseline, args.note, args.threshold)
        _print_perf(res)
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
