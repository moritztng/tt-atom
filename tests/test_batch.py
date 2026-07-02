"""Disjoint-union (block-diagonal) batching parity.

The whole correctness claim of batching is: evaluating K systems as one concatenated
block-diagonal graph gives the *same* per-system energies and forces as evaluating each system
separately. Block-diagonality means every backbone op stays within-system, so the only new piece
is the segment-sum energy readout (and forces need no change). These tests pin that on the tiny
random-weight golden (device, no fairchem needed); real-weight vs-fairchem batched parity lives
in test_realweight.py.
"""
import torch

from tt_atom.model import Backbone
from tt_atom.geometry import HostGeometry, radius_graph
from tt_atom import forces, disjoint
from util import pcc


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
    # to bf16 rounding only: the concatenated graph makes the dense scatter matmul (S[N,E] @ m)
    # and the energy readout accumulate over a larger E, so the tile/accumulation order — and
    # thus the last bf16 bit — differs. On the tiny random-weight model energies are O(1), where
    # one bf16 ULP is ~8e-3; agreement at that level is exact-to-precision. (PCC>0.999 / rel<1e-3
    # is asserted on the real, large-magnitude energies vs fairchem in test_realweight.py.)
    assert (E_batch - E_sep).abs().max() < 2e-2, f"energy maxdiff {(E_batch - E_sep).abs().max()}"
    assert pcc(F_batch, F_sep) > 0.99, f"force PCC {pcc(F_batch, F_sep)}"
    f_err = (F_batch - F_sep).abs().max()
    assert f_err < 2e-2, f"force max abs diff {f_err}"


def test_batched_identical_copies(golden, device):
    """A batch of identical copies must yield identical per-system energies (segment-sum sanity)."""
    bb, geo, w, cfg = _build(golden, device)
    systems = _systems(golden, 5, jitter=0.0)                  # all copies identical
    bg = disjoint.assemble(systems, cfg["cutoff"], w, cfg["sphere_channels"],
                           task=cfg.get("task", "omat"))
    E_batch, _ = forces.energy_and_forces_batch(bb, geo, bg, compute_forces=False)
    assert (E_batch - E_batch[0]).abs().max() < 1e-3, f"copies disagree: {E_batch}"
