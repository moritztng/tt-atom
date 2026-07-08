"""Disjoint-union (block-diagonal) batching parity.

The whole correctness claim of batching is: evaluating K systems as one concatenated
block-diagonal graph gives the *same* per-system energies and forces as evaluating each system
separately. Block-diagonality means every backbone op stays within-system, so the only new piece
is the segment-sum energy readout (and forces need no change). The mechanism is pinned on the
tiny random-weight golden (device, no fairchem needed); ``test_batched_vs_fairchem`` closes the
loop against fairchem's OWN batched merged inference on the real uma-s-1 checkpoint.
"""
import os
import pathlib

import numpy as np
import pytest
import torch

from tt_atom.model import Backbone
from tt_atom.geometry import HostGeometry, radius_graph
from tt_atom import forces, disjoint
from util import pcc

# real-weight batched parity (skipped unless the merged bundle + fairchem batched golden exist)
BUNDLE = os.environ.get("TTATOM_BUNDLE", str(pathlib.Path.home() / ".ttatom_run/uma_s_ethanol.npz"))
BATCH_GOLDEN = os.environ.get(
    "TTATOM_BATCH_GOLDEN",
    str(pathlib.Path.home() / ".ttatom_run/goldens_real/batch_ethanol_omol.npz"))


def _build(golden, device):
    cfg = dict(golden.config)
    w = golden.w()
    bb = Backbone(w, device, cfg, golden.host("to_grid_mat"), golden.host("from_grid_mat"))
    geo = HostGeometry(w, cfg, golden.host("to_m"), golden.host("gauss_offset"),
                       golden.host("gauss_coeff"), gamma=0.0)
    return bb, geo, w, cfg


def _systems(golden, k, jitter):
    """K variants of the tiny system: same atoms, deterministically jittered positions."""
    pos0 = golden.inp("pos").float()
    Z = golden.inp("atomic_numbers").long()
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(k):
        dp = (0.0 if i == 0 else jitter) * torch.randn(pos0.shape, generator=g)
        out.append(dict(pos=pos0 + dp, Z=Z, charge=0.0, spin=0.0))
    return out


def test_assemble_block_diagonal(golden):
    """Assembly concatenates atoms/edges with per-system offsets and a correct batch index."""
    w = golden.w()
    cfg = dict(golden.config)
    systems = _systems(golden, 3, jitter=0.02)
    bg = disjoint.assemble(systems, cfg["cutoff"], w, cfg["sphere_channels"], task=cfg.get("task", "omat"))
    n = systems[0]["Z"].shape[0]
    assert bg.natoms == [n, n, n]
    assert bg.pos.shape[0] == 3 * n
    assert torch.equal(bg.batch, torch.arange(3).repeat_interleave(n))
    # edges of system k must reference only system k's node block
    for k in range(3):
        m = bg.batch[bg.edge_index[0]] == k
        assert torch.equal(bg.batch[bg.edge_index[0][m]], bg.batch[bg.edge_index[1][m]])
    seg = bg.segment_matrix()
    assert torch.equal(seg.sum(0), torch.ones(3 * n))          # each atom in exactly one system
    assert torch.equal(seg.sum(1), torch.tensor([float(n)] * 3))


def test_batched_equals_separate(golden, device):
    """Batched per-system energies and forces == evaluating each system separately."""
    bb, geo, w, cfg = _build(golden, device)
    cutoff, C = cfg["cutoff"], cfg["sphere_channels"]
    task = cfg.get("task", "omat")
    systems = _systems(golden, 4, jitter=0.03)
    bg = disjoint.assemble(systems, cutoff, w, C, task=task)

    E_batch, F_batch = forces.energy_and_forces_batch(bb, geo, bg)

    # separate baseline: identical inputs, one system at a time
    off = 0
    E_sep, F_sep = [], []
    for k, s in enumerate(systems):
        pos, Z = s["pos"], s["Z"]
        n = Z.shape[0]
        ei, shift = radius_graph(pos, cutoff)
        se = bg.sys_emb[off:off + n]
        E, F = forces.energy_and_forces(bb, geo, pos, Z, ei, se, edge_cell_shift=shift)
        E_sep.append(E)
        F_sep.append(F)
        off += n
    E_sep = torch.tensor(E_sep)
    F_sep = torch.cat(F_sep, dim=0)

    # Block-diagonal batching leaves every backbone op within-system, so batched == separate up
    # to bf16 rounding only: the batched forward uses the linear O(E) gather+reduce scatter (a
    # K>1 batch is block-diagonal, so the dense one-hot S[N,E] would be O(K^2) off-diagonal zeros)
    # while the separate baseline keeps the dense matmul, and the energy readout accumulates over a
    # larger E — so the tile/accumulation order, and thus the last bf16 bit, differs. On the tiny
    # random-weight model energies are O(1), where
    # one bf16 ULP is ~8e-3; agreement at that level is exact-to-precision. (PCC>0.999 / rel<1e-3
    # is asserted on the real, large-magnitude energies vs fairchem in test_realweight.py.)
    assert (E_batch - E_sep).abs().max() < 2e-2, f"energy maxdiff {(E_batch - E_sep).abs().max()}"
    assert pcc(F_batch, F_sep) > 0.99, f"force PCC {pcc(F_batch, F_sep)}"
    f_err = (F_batch - F_sep).abs().max()
    assert f_err < 2e-2, f"force max abs diff {f_err}"


def test_batch_forces_linear_scatter(golden, device):
    """A K>1 disjoint-union batch must use the linear O(E) scatter regardless of node count (the
    block-diagonal dense one-hot is O(K^2) off-diagonal zeros); a 1-system batch keeps the
    single-system node-count threshold. Guards the batched-throughput optimization."""
    from tt_atom.model import GraphContext, SCATTER_LINEAR_THRESHOLD

    bb, geo, w, cfg = _build(golden, device)
    cutoff, C = cfg["cutoff"], cfg["sphere_channels"]
    task = cfg.get("task", "omat")

    captured = {}
    orig_init = GraphContext.__init__

    def spy(self, *a, **kw):
        orig_init(self, *a, **kw)
        captured["linear"] = self.linear_scatter
        captured["N"] = self.N

    GraphContext.__init__ = spy
    try:
        bg2 = disjoint.assemble(_systems(golden, 2, jitter=0.03), cutoff, w, C, task=task)
        forces.energy_and_forces_batch(bb, geo, bg2, compute_forces=False)
        assert captured["N"] <= SCATTER_LINEAR_THRESHOLD, "test premise: batch below threshold"
        assert captured["linear"] is True, "K>1 batch must force the linear scatter"

        bg1 = disjoint.assemble(_systems(golden, 1, jitter=0.0), cutoff, w, C, task=task)
        forces.energy_and_forces_batch(bb, geo, bg1, compute_forces=False)
        assert captured["linear"] is False, "K=1 batch (below threshold) keeps the dense scatter"
    finally:
        GraphContext.__init__ = orig_init


def test_batched_identical_copies(golden, device):
    """A batch of identical copies must yield identical per-system energies (segment-sum sanity)."""
    bb, geo, w, cfg = _build(golden, device)
    systems = _systems(golden, 5, jitter=0.0)                  # all copies identical
    bg = disjoint.assemble(systems, cfg["cutoff"], w, cfg["sphere_channels"],
                           task=cfg.get("task", "omat"))
    E_batch, _ = forces.energy_and_forces_batch(bb, geo, bg, compute_forces=False)
    assert (E_batch - E_batch[0]).abs().max() < 1e-3, f"copies disagree: {E_batch}"


def test_traced_batch_matches_eager(golden, device):
    """The trace-replayed batched forward (``evaluate_batch(trace=True)`` engine) must be BIT-EXACT
    vs the eager batched forward: a trace only removes host dispatch, it is the same device op
    stream (and both use the K>1 linear scatter + segment-sum readout). Also exercises replay: a
    second call on the same topology must match too. Guards the batched-MD throughput path."""
    from tt_atom.trace import TracedEngine

    bb, geo, w, cfg = _build(golden, device)
    systems = _systems(golden, 4, jitter=0.03)
    bg = disjoint.assemble(systems, cfg["cutoff"], w, cfg["sphere_channels"],
                           task=cfg.get("task", "omat"))
    E_eager, F_eager = forces.energy_and_forces_batch(bb, geo, bg)

    eng = TracedEngine(bb, geo, bg.Z, bg.edge_index, bg.sys_emb, edge_cell_shift=bg.cell_shift,
                       seg=bg.segment_matrix(), linear_scatter=True)
    E_cap, F_cap = eng(bg.pos)          # capture step (records + replays)
    E_replay, F_replay = eng(bg.pos)    # pure replay
    eng.close()

    assert (E_cap - E_eager).abs().max() == 0, f"traced E != eager: {E_cap} vs {E_eager}"
    assert (F_cap - F_eager).abs().max() == 0, "traced forces != eager"
    assert torch.equal(E_cap, E_replay) and torch.equal(F_cap, F_replay), "replay not deterministic"


@pytest.mark.skipif(not (pathlib.Path(BUNDLE).exists() and pathlib.Path(BATCH_GOLDEN).exists()),
                    reason="real merged bundle or fairchem batched golden not present")
def test_batched_vs_fairchem(device):
    """Real uma-s-1: TT-Atom disjoint-union batched E+F vs fairchem's OWN batched merged inference
    (Batch.from_data_list) on a same-composition conformer batch — E rel<1e-3, F PCC>0.99."""
    from ase import Atoms
    from tt_atom import TTAtomCalculator

    d = np.load(BATCH_GOLDEN)
    charge, spin = float(d["charge"][0]), float(d["spin"][0])
    natoms = d["natoms"].tolist()
    Z, pos = d["Z"], d["pos"]
    E_ref, F_ref = d["energy"].astype(np.float64), d["forces"].astype(np.float64)

    systems, off = [], 0
    for n in natoms:
        a = Atoms(numbers=Z[off:off + n], positions=pos[off:off + n])
        a.info.update(charge=charge, spin=spin)
        systems.append(a)
        off += n

    calc = TTAtomCalculator(BUNDLE, device=device)
    res = calc.evaluate_batch(systems)
    E = res["energy"]
    F = np.concatenate(res["forces"], axis=0)

    e_rel = np.abs(E - E_ref).max() / (np.abs(E_ref).max() + 1e-6)
    fp = pcc(F, F_ref)
    assert e_rel < 1e-3, f"batched energy rel err {e_rel:.2e} (E={E[:3]} vs {E_ref[:3]})"
    assert fp > 0.99, f"batched force PCC {fp:.4f}"


real_bundle = pytest.mark.skipif(
    not pathlib.Path(BUNDLE).exists(),
    reason=f"real uma-s-1 bundle not found at {BUNDLE}")


@real_bundle
def test_evaluate_batch_rejects_composition_charge_spin_mismatch(device):
    """A merged bundle bakes the MoLE routing for one (composition, charge, spin); evaluate_batch
    must reject a batch that mixes them rather than silently returning wrong energies. The bundle
    is ethanol (C2H6O), merged at charge=0, spin=1."""
    from ase.build import molecule

    from tt_atom import TTAtomCalculator

    calc = TTAtomCalculator(BUNDLE, device=device)
    good = molecule("CH3CH2OH"); good.info.update(charge=0, spin=1)   # matches the bundle
    water = molecule("H2O"); water.info.update(charge=0, spin=1)      # wrong composition
    with pytest.raises(ValueError, match="reduced composition"):
        calc.evaluate_batch([good, water])
    bad_cs = molecule("CH3CH2OH"); bad_cs.info.update(charge=0, spin=0)  # wrong spin
    with pytest.raises(ValueError, match="merged for"):
        calc.evaluate_batch([good, bad_cs])
