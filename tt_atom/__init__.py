"""TT-Atom — high-performance Tenstorrent inference for ML interatomic potentials: Meta's UMA
(eSEN / eSCN-MD, equivariant) and Orbital Materials' Orb-v3 / OrbMol (non-equivariant).

Every public name resolves lazily through :data:`_EXPORTS`, so ``import tt_atom`` stays cheap and
never imports ttnn or torch (and therefore never opens or probes a device).
"""

import os
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("tt-atom")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0+unknown"



def _drop_numpy_hugepage_hint():
    """numpy asks for huge pages behind every array of 4 MB or more. On a host that has been up for
    days with no free 2 MB block, each such page fault stalls in kernel memory compaction: four
    127 MB arrays took 1.2 to 4.5 s to fill on a Galaxy host 15 days up, 0.35 s without the hint.
    numpy reads the variable at import; if it is already loaded the switch is flipped. A user's
    own NUMPY_MADVISE_HUGEPAGE wins."""
    if os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0") != "0":
        return
    for name in ("numpy._core.multiarray", "numpy.core.multiarray"):  # numpy 2, numpy 1
        flip = getattr(sys.modules.get(name), "_set_madvise_hugepage", None)
        if flip is not None:
            flip(False)
            return


_drop_numpy_hugepage_hint()

_EXPORTS = {
    "Calculator": "auto",
    "TTAtomCalculator": "calculator",
    "OrbCalculator": "orb_calculator",
    "MaceCalculator": "mace_calculator",
    "WeightBundle": "weights",
    "Backbone": "model",
    "HostGeometry": "geometry",
    "MultiCard": "batch",
    "MultiCardSim": "batch",
    "relax_atoms": "simulate",
    "md_atoms": "simulate",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted([*__all__, "__version__"])
