"""Bit-exact parity: MultiCardSim relax/MD fan-out vs the single-card path.

Runs the single-card reference and the MultiCardSim pool in SEPARATE subprocesses so the
UMD ``CHIP_IN_USE`` device lock fully releases between phases (close_device closes the
logical device but the process-global MetalContext / UMD mappings live until process exit,
so a second open of the same card in the same process blocks on the lock — this is a
property of the runtime, not of the fan-out code; the CLI ``cmd_run`` is single-card XOR
multicard, never both in one process).

Each structure runs the identical Calculator + relax_atoms / md_atoms loop in both paths
(one Calculator reused for Orb, composition-independent — exactly what the MultiCardSim
worker does), so bit-exact is by construction; this confirms the spawn/queue/gather
plumbing on real device. Uses orb-v3-direct-omol (direct ForceHead, no autograd) + stock
ttnn (no fused_rotate needed).
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from ase.build import molecule

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
MODEL = "orb-v3-direct-omol"
DEVICES = (0,)
RELAX_STEPS = 3
MD_STEPS = 3
NAMES = ["H2O", "CH3CH2OH", "C6H6"]   # 3, 9, 12 atoms — three different sizes
SEED = 42

# Structures are generated deterministically and pickled to a temp file so both subprocesses
# load the EXACT same inputs (rattle seed pinned).
def _build_systems():
    out = []
    for i, name in enumerate(NAMES):
        a = molecule(name)
        a.info.update(charge=0, spin=0)
        a.rattle(stdev=0.05, seed=i)
        out.append(dict(pos=a.get_positions().tolist(), Z=a.get_atomic_numbers().tolist(),
                         name=name))
    return out


# Each phase is a standalone script run in its own process (one device context). All parameters
# come from argv/env so the script body has no .format() braces to escape.
_PHASE_SCRIPT = '''
import json, os, sys
sys.path.insert(0, os.environ["PARITY_REPO"])
import numpy as np
from ase import Atoms
from tt_atom import Calculator, MultiCardSim, relax_atoms, md_atoms

phase, mode, systems_path, out_path = sys.argv[1:5]
model = os.environ["PARITY_MODEL"]
devices = tuple(int(d) for d in os.environ["PARITY_DEVICES"].split(","))
relax_steps = int(os.environ["PARITY_RELAX_STEPS"])
md_steps = int(os.environ["PARITY_MD_STEPS"])
seed = int(os.environ["PARITY_SEED"])

systems_data = json.load(open(systems_path))
systems = [Atoms(numbers=d["Z"], positions=d["pos"]) for d in systems_data]
for a in systems:
    a.info.update(charge=0, spin=0)

def single():
    calc = Calculator(systems[0], model)
    results = []
    try:
        for a in systems:
            a.calc = calc
            if mode == "relax":
                r = relax_atoms(a, fmax=0.05, steps=relax_steps, logfile=None)
            elif mode == "md":
                r = md_atoms(a, steps=md_steps, dt=1.0, temp=300.0, logfile=None, seed=seed)
            else:
                r = dict(energy=float(a.get_potential_energy()),
                         forces=a.get_forces(), nsteps=0)
            results.append(dict(pos=a.get_positions().tolist(), energy=float(r["energy"]),
                                 forces=np.asarray(r["forces"]).tolist(), nsteps=int(r["nsteps"])))
    finally:
        calc.close()
    return results

def multi():
    if mode == "relax":
        sim = dict(mode="relax", fmax=0.05, steps=relax_steps)
    elif mode == "md":
        sim = dict(mode="md", steps=md_steps, dt=1.0, temp=300.0, seed=seed)
    else:
        sim = dict(mode="energy")
    with MultiCardSim(model, device_ids=devices, sim_params=sim) as pool:
        return pool.run([dict(pos=a.get_positions().tolist(), Z=a.get_atomic_numbers().tolist())
                         for a in systems])

import numpy as _np
def _ser(o):
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_ser(x) for x in o]
    return o

results = single() if phase == "single" else multi()
json.dump(_ser(results), open(out_path, "w"))
'''


def _run_phase(phase, mode, systems_path, env):
    out = tempfile.mktemp(suffix=".json")
    cmd = [sys.executable, "-c", _PHASE_SCRIPT, phase, mode, systems_path, out]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=400)
    if proc.returncode != 0:
        print(f"[{phase}/{mode}] FAILED rc={proc.returncode}", file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    return json.load(open(out))


def _cmp(label, ref, mc):
    problems = []
    for i, (r, m) in enumerate(zip(ref, mc)):
        if not m.get("ok", True):
            problems.append(f"[{label}][{i}] multicard FAILED: {m.get('error')}")
            continue
        e_diff = abs(m["energy"] - r["energy"])
        pos_diff = float(np.abs(np.array(m["pos"]) - np.array(r["pos"])).max())
        f_diff = float(np.abs(np.array(m["forces"]) - np.array(r["forces"])).max())
        print(f"  [{label}][{i}] E_ref={r['energy']:.8f} E_mc={m['energy']:.8f} "
              f"dE={e_diff:.3e} dpos={pos_diff:.3e} dF={f_diff:.3e}")
        if e_diff > 0.0 or pos_diff > 0.0 or f_diff > 0.0:
            problems.append(f"[{label}][{i}] NOT bit-exact: dE={e_diff} dpos={pos_diff} dF={f_diff}")
    return problems


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env["TT_METAL_LOGGER_LEVEL"] = "FATAL"
    env["PARITY_REPO"] = str(REPO)
    env["PARITY_MODEL"] = MODEL
    env["PARITY_DEVICES"] = ",".join(map(str, DEVICES))
    env["PARITY_RELAX_STEPS"] = str(RELAX_STEPS)
    env["PARITY_MD_STEPS"] = str(MD_STEPS)
    env["PARITY_SEED"] = str(SEED)

    systems_path = tempfile.mktemp(suffix=".json")
    json.dump(_build_systems(), open(systems_path, "w"))
    print(f"parity: {len(NAMES)} structures {NAMES}, model={MODEL}, devices={DEVICES}")

    print("\n-- relax (FIRE, deterministic) --")
    ref_r = _run_phase("single", "relax", systems_path, env)
    mc_r = _run_phase("multi", "relax", systems_path, env)
    probs_r = _cmp("relax", ref_r, mc_r)

    print("\n-- md (Langevin, seed-pinned velocities) --")
    ref_m = _run_phase("single", "md", systems_path, env)
    mc_m = _run_phase("multi", "md", systems_path, env)
    probs_m = _cmp("md", ref_m, mc_m)

    print("\n-- energy (single point) --")
    ref_e = _run_phase("single", "energy", systems_path, env)
    mc_e = _run_phase("multi", "energy", systems_path, env)
    probs_e = _cmp("energy", ref_e, mc_e)

    all_probs = probs_r + probs_m + probs_e
    print("\n" + "=" * 60)
    if all_probs:
        print("PARITY FAIL:")
        for p in all_probs:
            print("  " + p)
        return 1
    print("PARITY PASS: bit-exact per-structure relax, md AND energy, single-card vs MultiCardSim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
