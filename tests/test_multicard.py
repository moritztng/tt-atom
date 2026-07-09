"""Multi-card fan-out parity: MultiCard sharding independent systems across N cards must give the
SAME per-system energies, in the SAME order, as running every system sequentially on one card.

Each system is evaluated by exactly one worker in exactly one code path (the ``_worker`` loop in
``tt_atom/batch.py``) regardless of which card it lands on or how many cards are in the pool, so
sharding is bit-exact by construction: there is no cross-system batching/regrouping (unlike
disjoint-union batching's bf16 accumulation-order sensitivity, see test_batch.py) that could make
the result depend on the shard layout. This test exercises the real queue/dispatch/gather path
(``MultiCard.energies``), not just the per-system compute.
"""
import glob
import pathlib

import numpy as np
import pytest
from ase.build import molecule

from tt_atom.batch import MultiCard

HERE = pathlib.Path(__file__).parent
WEIGHTS = HERE.parent / "examples" / "model_tiny_demo.npz"


def _num_devices():
    return len(glob.glob("/dev/tenstorrent/[0-9]*"))


pytestmark = [
    pytest.mark.skipif(not WEIGHTS.exists(), reason="examples/model_tiny_demo.npz not present"),
    pytest.mark.skipif(_num_devices() < 2, reason="needs >=2 Tenstorrent cards"),
]


def _systems(n, seed=0):
    base = molecule("CH3CH2OH")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pos = base.get_positions() + rng.normal(scale=0.05, size=base.get_positions().shape)
        out.append((pos.astype(np.float32), base.get_atomic_numbers()))
    return out


@pytest.mark.parametrize("n_systems", [8, 10])  # 10 is uneven across e.g. 3 or 4 cards
def test_sharded_matches_sequential(n_systems):
    n_dev = min(4, _num_devices())
    systems = _systems(n_systems)

    with MultiCard(str(WEIGHTS), device_ids=(0,)) as pool:
        e_ref, edges_ref = pool.energies(systems)

    with MultiCard(str(WEIGHTS), device_ids=tuple(range(n_dev))) as pool:
        e_sharded, edges_sharded = pool.energies(systems)

    assert edges_sharded == edges_ref
    assert e_sharded == e_ref, f"sharded != sequential: {e_sharded} vs {e_ref}"
