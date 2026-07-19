"""Probe: does UMA's batched forward (evaluate_batch) run on this card, or is it
blocked by the same fused_rotate env gap as the accuracy leg's end-to-end test?

Reuses the ethanol/omol golden bundle (the accuracy leg's fixture) repeated K=2,
the smallest possible batch. If this raises the fused_rotate AttributeError, the
OOM sweep is blocked by the env, not a separate issue.
"""
import os
import pathlib
import sys
import traceback

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ase.build import molecule

from tt_atom.calculator import TTAtomCalculator

GOLDEN = str(pathlib.Path.home() / ".ttatom_run/goldens_real/ethanol_omol.npz")
assert pathlib.Path(GOLDEN).exists(), f"golden missing: {GOLDEN}"

calc = TTAtomCalculator(GOLDEN, task_name="omol", device_id=0)
systems = []
for i in range(2):
    a = molecule("CH3CH2OH")
    a.rattle(stdev=0.08, seed=10 + i)
    a.info.update(charge=0, spin=1)
    systems.append(a)

try:
    out = calc.evaluate_batch(systems, properties=("energy", "forces"))
    print("PROBE_RESULT: evaluate_batch RAN — energies:", out["energy"])
    print("PROBE_VERDICT: SWEEP_POSSIBLE")
except Exception as e:
    print("PROBE_RESULT: evaluate_batch RAISED:",
          f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    traceback.print_exc()
    print("PROBE_VERDICT: SWEEP_BLOCKED")
finally:
    try:
        calc.close()
    except Exception:
        pass
