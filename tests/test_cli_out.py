"""``tt-atom run --out`` file naming (host-only: no card, no calculator, no weights).

One structure writes ``--out`` verbatim. Several structures share one ``--out``, so each needs its
own file — the single-card path used to write every result to the same path, silently leaving only
the last structure's geometry. Both the single-card and the multi-card path now go through
``_batch_out``/``_batch_out_path``, so they cannot drift apart.

The release gate's UX leg drives ``tt-atom relax``/``md`` on a bundle, not ``run`` on structure
files, so this naming contract has no on-device gate row; it is pinned here instead.
"""
import argparse

import numpy as np
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from tt_atom import cli


def _args(out, structures):
    return argparse.Namespace(out=out, structures=structures, relax=False, md=False)


def _atoms(name):
    a = molecule(name)
    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((len(a), 3)))
    return a


def test_single_structure_writes_out_verbatim(tmp_path):
    args = _args(str(tmp_path / "single.xyz"), ["h2o.xyz"])
    assert cli._batch_out(args, per_structure=False) is None
    cli._write_output(args, _atoms("H2O"))
    assert (tmp_path / "single.xyz").exists()


def test_several_structures_do_not_overwrite_each_other(tmp_path):
    args = _args(str(tmp_path / "out.xyz"), ["h2o.xyz", "ch3oh.xyz"])
    out_dir = cli._batch_out(args, per_structure=True)
    assert out_dir == tmp_path                       # a file pattern degrades to its parent
    for i, name in enumerate(("H2O", "CH3OH")):
        cli._write_output(args, _atoms(name), cli._batch_out_path(args, out_dir, i))
    assert sorted(p.name for p in tmp_path.glob("*.xyz")) == ["ch3oh_1.xyz", "h2o_0.xyz"]


def test_out_directory_is_created(tmp_path):
    args = _args(str(tmp_path / "relaxed"), ["h2o.xyz", "ch3oh.xyz"])
    out_dir = cli._batch_out(args, per_structure=True)
    assert out_dir == tmp_path / "relaxed" and out_dir.is_dir()
    assert cli._batch_out_path(args, out_dir, 1).name == "ch3oh_1.xyz"   # default .xyz


def test_no_out_writes_nothing(tmp_path):
    args = _args(None, ["h2o.xyz"])
    assert cli._batch_out(args, per_structure=True) is None
    cli._write_output(args, _atoms("H2O"))
    assert list(tmp_path.iterdir()) == []
