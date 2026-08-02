"""Edge-bucketing parity gate: padded vs unpadded evaluation must be BIT-EXACT.

``OrbCalculator(bucketing=True)`` pads each system's edge set to the ``tt_atom.bucketing``
ladder with zero-contributing sentinel edges (self-loops on node 0 at exactly r_max, so the
attention-cutoff envelope is exactly 0.0) and builds the scatter gather tables from the true
edges only (same per-node reduction order, sentinel slots gather the zero pad row). The claim
under test: energies AND forces (and stress, where computed) are bitwise identical to the
unpadded path — maxdiff exactly 0.0, not a tolerance — across system sizes that span several
ladder rungs, including a small molecule padded UP to the bottom rung and, for the checkpoints
present, the conservative (analytic-VJP), direct (ForceHead), and charge/spin-conditioned
paths.

Run (qb1):  TT_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m pytest tests/test_bucketing.py -q
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

WEIGHTS_DIR = pathlib.Path.home() / ".cache" / "tt_atom" / "orb_weights"
CHECKPOINTS = ["conservative-inf-omat", "direct-20-omat", "conservative-omol"]

pytestmark = pytest.mark.skipif(not WEIGHTS_DIR.is_dir(), reason="no cached Orb weights")


def _systems():
    """(name, ASE Atoms) covering >=2 ladder rungs + the pad-up case, >=5 distinct sizes."""
    from ase.build import bulk

    systems = []
    for cells, seed in [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)]:
        atoms = bulk("Si", "diamond", a=5.43) * (cells, cells, cells)
        atoms.rattle(stdev=0.1, seed=seed)
        atoms.pbc = True
        systems.append((f"si{cells}", atoms))
    clus = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
    clus.rattle(stdev=0.1, seed=6)
    clus.pbc = False                             # aperiodic, few edges -> pads UP to 1024
    systems.append(("si2_ap", clus))
    return systems


def test_helpers():
    import torch

    from tt_atom.bucketing import EDGE_BUCKETS, bucket_size, pad_edge_index, pad_graph

    assert bucket_size(1) == EDGE_BUCKETS[0]
    assert bucket_size(EDGE_BUCKETS[0]) == EDGE_BUCKETS[0]
    assert bucket_size(EDGE_BUCKETS[0] + 1) == EDGE_BUCKETS[1]
    assert bucket_size(EDGE_BUCKETS[-1]) == EDGE_BUCKETS[-1]
    n = EDGE_BUCKETS[-1] + 7
    assert bucket_size(n) == n                       # above the ladder: unpadded

    s = torch.tensor([0, 1, 2])
    r = torch.tensor([1, 2, 0])
    cs = torch.zeros(3, 3)
    s2, r2, cs2 = pad_edge_index(s, r, cs, 5, 6.0)
    assert s2.tolist() == [0, 1, 2, 0, 0]            # sentinel self-loops on node 0
    assert r2.tolist() == [1, 2, 0, 0, 0]
    assert cs2[3].tolist() == [6.0, 0.0, 0.0]        # displacement exactly r_max along x
    s3, r3, cs3, gkw = pad_graph(s, r, cs, r_max=6.0, max_num_neighbors=64)
    assert s3.shape[0] == EDGE_BUCKETS[0]
    assert gkw == dict(gather_edge_count=3, gather_width=64)


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_bucketing_bitexact(device, checkpoint):
    path = WEIGHTS_DIR / f"{checkpoint}.npz"
    if not path.exists():
        pytest.skip(f"weights not cached: {path}")

    from tt_atom.orb_calculator import OrbCalculator

    plain = OrbCalculator(str(path), device=device)
    try:
        ref = {}
        for name, atoms in _systems():
            plain.calculate(atoms)
            ref[name] = (plain.results["energy"], np.asarray(plain.results["forces"]),
                         plain.results.get("stress"))
    finally:
        plain.close()

    bucketed = OrbCalculator(str(path), device=device, bucketing=True)
    try:
        for name, atoms in _systems():
            bucketed.calculate(atoms)
            e0, f0, s0 = ref[name]
            e1, f1, s1 = (bucketed.results["energy"], np.asarray(bucketed.results["forces"]),
                          bucketed.results.get("stress"))
            fdiff = float(np.abs(f0 - f1).max())
            print(f"\n[{checkpoint}:{name}] dE={abs(e0 - e1)} dF_max={fdiff}")
            assert e0 == e1, f"{name}: energy {e0!r} != {e1!r}"
            assert np.array_equal(f0, f1), f"{name}: forces maxdiff {fdiff}"
            assert (s0 is None) == (s1 is None)
            if s0 is not None:
                assert np.array_equal(np.asarray(s0), np.asarray(s1)), \
                    f"{name}: stress maxdiff {np.abs(np.asarray(s0) - np.asarray(s1)).max()}"
    finally:
        bucketed.close()
