"""End-to-end parity from raw positions: host geometry + device forward + analytic forces.

Validates the full production path (``tt_atom.forces.energy_and_forces``) against the fairchem
golden — energy and the conservative analytic force ``F = -dE/dpos`` (NOT finite differences)."""
import torch

from tt_atom.model import Backbone
from tt_atom.geometry import HostGeometry
from tt_atom import forces
from util import pcc


def _build(golden, device):
    cfg = dict(golden.config)
    w = golden.w()
    bb = Backbone(w, device, cfg, golden.host("to_grid_mat"), golden.host("from_grid_mat"))
    geo = HostGeometry(w, cfg, golden.host("to_m"), golden.host("gauss_offset"),
                       golden.host("gauss_coeff"), gamma=0.0)
    return bb, geo


def test_geometry_forward(golden, device):
    # gamma-independent geometric terms must match the fairchem golden exactly
    _, geo = _build(golden, device)
    t = geo(golden.inp("pos").float(), golden.inp("atomic_numbers").long(),
            golden.inp("edge_index").long(), golden.host("sys_node_embedding"))
    assert pcc(t["edge_distance"], golden.host("edge_distance")) >= 0.999
    assert pcc(t["x_edge"], golden.host("x_edge")) >= 0.999
    assert pcc(t["edge_envelope"], golden.host("edge_envelope")) >= 0.999


def test_energy_and_forces(golden, device):
    bb, geo = _build(golden, device)
    E, F = forces.energy_and_forces(
        bb, geo, golden.inp("pos").float(), golden.inp("atomic_numbers").long(),
        golden.inp("edge_index").long(), golden.host("sys_node_embedding"))
    Eref = float(golden.out("energy").reshape(-1)[0])
    Fref = golden.out("forces")
    assert abs(E - Eref) / (abs(Eref) + 1e-6) < 0.05, f"energy {E} vs {Eref}"
    p = pcc(F, Fref)
    cos = float(torch.nn.functional.cosine_similarity(F.reshape(1, -1), Fref.reshape(1, -1)))
    assert p >= 0.98, f"force PCC {p}"
    assert cos >= 0.98, f"force cosine {cos}"
