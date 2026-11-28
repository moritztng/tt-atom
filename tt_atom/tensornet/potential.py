"""TensorNet energy and forces from an ase.Atoms without matgl at run time. The host builds the neighbour list and the
per-edge geometry in float64 and closes the reverse pass to the positions; the network between them runs on whatever
backend `run` wraps (the float64 reference here, a card through tt_ops)."""
import numpy as np
import torch
from ase.neighborlist import neighbor_list

from . import embed as E
from . import layer as L
from . import model as M


def edges(atoms, cutoff):
    """Full neighbour list: edge_index [2, E] and each edge's periodic shift [E, 3] in A, both ends of every pair."""
    i, j, s = neighbor_list("ijS", atoms, cutoff)
    return torch.from_numpy(np.stack([i, j])).long(), torch.from_numpy(s @ atoms.cell.array).double()


def torch_run(w, units, dtype=torch.float64):
    """The network on the float64 reference backend: slot inputs -> (atomic energies [N], edge gradients)."""
    def run(z, rbf, cut, geo, send, table, n, d):
        eye = torch.eye(3, dtype=dtype).reshape(3, 3, 1, 1).expand(3, 3, n, units)
        en, tape = M.forward(L.TorchOps, w, z, (rbf, cut), list(geo.split(1, -1)), send, n, eye)
        grbf, gcut, ggeo = M.backward(L.TorchOps, w, tape, torch.ones_like(en), table, n, d, eye)
        return en.reshape(-1), sum(grbf), sum(gcut), ggeo
    return run


def energy_forces(run, pair, atoms, cutoff, max_n, scale=1.0, shift=0.0, min_width=32, own=False):
    """(energy in eV, forces [N, 3] in eV/A). `pair(edge_index)` is the species part of each edge's message [E, U];
    scale and shift are the model's energy normalisation, shift already summed over the atoms."""
    n, (ei, off) = len(atoms), edges(atoms, cutoff)
    pos = torch.from_numpy(atoms.positions).double().requires_grad_(True)
    vec = pos[ei[1]] + off - pos[ei[0]]
    r = vec.norm(dim=1)
    rbf, cut, geo = E.radial(r, cutoff, max_n), E.cosine_cutoff(r, cutoff), E.geometry(vec)
    edge, send, table, d = L.slots(ei, n, min_width=min_width, own=own)
    pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge].detach()
    real = edge < ei.shape[1]
    unpad = lambda t: torch.zeros(ei.shape[1], *t.shape[1:], dtype=torch.float64).index_copy(0, edge[real], t[real].double())
    en, grbf, gcut, ggeo = run(pad(pair(ei)), pad(rbf), pad(cut), pad(geo), send, table, n, d)
    grad = torch.autograd.grad((rbf * unpad(grbf)).sum() + (cut * unpad(gcut)).sum() + (geo * unpad(ggeo)).sum(), pos)[0]
    return scale * float(en.double().sum()) + shift, -(scale * grad).numpy()
