"""Grade the float64 CPU MACE reference block by block against the mace-torch golden.

The golden is written by tests/gen_golden_mace.py (MIT weights only); the test skips without it.
"""
import os

import numpy as np
import pytest
import torch

from tt_atom.mace import reference as ref

GOLDEN = os.path.expanduser(os.environ.get("TT_ATOM_MACE_GOLDEN", "~/.ttatom_run/goldens_real/si_mace_mp0b3.npz"))
pytestmark = pytest.mark.skipif(not os.path.exists(GOLDEN), reason="MACE golden not generated")
TOL = 1e-10  # float64 vs float64: only summation order differs


@pytest.fixture(scope="module")
def g():
    z = np.load(GOLDEN)
    t = {k: torch.from_numpy(np.asarray(z[k])) for k in z.files if k != "config"}
    w = {k[2:]: v.to(torch.float64) if v.is_floating_point() else v for k, v in t.items() if k.startswith("w@")}
    pos, ei, sh = t["in@pos"], t["in@edge_index"].long(), t["in@shifts"]
    vec, r = ref.edge_vectors(pos, ei, sh)
    return {"t": t, "w": w, "ei": ei, "vec": vec, "r": r, "Z": t["in@atomic_numbers"].long()}


def close(a, b):
    err = (a - b).abs().max().item()
    assert err < TOL * max(1.0, b.abs().max().item()), err


def test_node_embedding(g):
    attrs = ref.node_attrs(g["Z"], g["w"]["atomic_numbers"].long())
    close(ref.node_embedding(attrs, g["w"]), g["t"]["a@node_embedding.out0"])


def test_spherical_harmonics(g):
    close(ref.spherical_harmonics(g["vec"]), g["t"]["a@spherical_harmonics.out0"])


def test_radial_embedding(g):
    close(ref.radial_embedding(g["r"], g["Z"], g["ei"], g["w"]), g["t"]["a@radial_embedding.out0"])


def test_zbl(g):
    close(ref.zbl(g["r"], g["Z"], g["ei"], g["w"], len(g["Z"])), g["t"]["a@pair_repulsion_fn.out0"])


H = [(128, l) for l in range(4)]  # 128x0e+128x1o+128x2e+128x3o


def test_interaction_0(g):
    w, t = g["w"], g["t"]
    attrs = ref.node_attrs(g["Z"], w["atomic_numbers"].long())
    out, sc = ref.density_interaction(w, 0, attrs, t["a@node_embedding.out0"], [(128, 0)],
                                      t["a@spherical_harmonics.out0"], t["a@radial_embedding.out0"], g["ei"], H)
    assert sc is None
    close(out, t["a@interactions.0.out0"])


def test_product_0(g):
    w, t = g["w"], g["t"]
    attrs = ref.node_attrs(g["Z"], w["atomic_numbers"].long())
    close(ref.product_block(w, 0, t["a@interactions.0.out0"], attrs, [(128, 0), (128, 1)]), t["a@products.0.out0"])


def test_interaction_1(g):
    w, t = g["w"], g["t"]
    attrs = ref.node_attrs(g["Z"], w["atomic_numbers"].long())
    out, sc = ref.density_interaction(w, 1, attrs, t["a@products.0.out0"], [(128, 0), (128, 1)],
                                      t["a@spherical_harmonics.out0"], t["a@radial_embedding.out0"], g["ei"], H,
                                      sc_irreps=[(128, 0)])
    close(sc, t["a@interactions.1.out1"])
    close(out, t["a@interactions.1.out0"])


def test_product_1(g):
    w, t = g["w"], g["t"]
    attrs = ref.node_attrs(g["Z"], w["atomic_numbers"].long())
    close(ref.product_block(w, 1, t["a@interactions.1.out0"], attrs, [(128, 0)], sc=t["a@interactions.1.out1"]),
          t["a@products.1.out0"])
