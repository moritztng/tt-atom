"""Sanity check (orb-models CPU reference): Orb-v3's energy-minimum lattice constant for
diamond-cubic Si should sit near the experimental 5.43 A, and the energy/atom in the physically
expected range for this OMat24-trained potential. Confirms the MD system is a correct Si crystal,
not an artifact. Runs in refenv (orb_models)."""
from __future__ import annotations

import numpy as np
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

MODEL = "orb-v3-conservative-inf-omat"

orbff = pretrained.ORB_PRETRAINED_MODELS[MODEL](device="cpu", precision="float32-high")
calc = ORBCalculator(orbff, device="cpu")

a_grid = np.arange(5.30, 5.56, 0.02)
es = []
for a in a_grid:
    at = bulk("Si", "diamond", a=float(a), cubic=True)   # 8-atom conventional cell
    at.calc = calc
    es.append(at.get_potential_energy() / len(at))
es = np.array(es)
# parabolic fit near the minimum -> equilibrium a
c = np.polyfit(a_grid, es, 2)
a_min = -c[1] / (2 * c[0])
e_min = np.polyval(c, a_min)
print(f"model              : {MODEL} (CPU reference)")
print(f"a grid             : {a_grid[0]:.2f} .. {a_grid[-1]:.2f} A")
for a, e in zip(a_grid, es):
    mark = "  <- min" if abs(a - a_grid[es.argmin()]) < 1e-9 else ""
    print(f"  a={a:.2f} A   E={e:.4f} eV/atom{mark}")
print(f"equilibrium a (fit): {a_min:.3f} A   (experimental Si: 5.431 A)")
print(f"E at equilibrium   : {e_min:.4f} eV/atom")
print(f"E at a=5.43        : {es[np.argmin(np.abs(a_grid-5.43))]:.4f} eV/atom  (MD lattice constant)")
