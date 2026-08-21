"""TT-Atom — high-performance Tenstorrent inference for ML interatomic potentials: Meta's UMA
(eSEN / eSCN-MD, equivariant) and Orbital Materials' Orb-v3 / OrbMol (non-equivariant).

Every public name resolves lazily through :data:`_EXPORTS`, so ``import tt_atom`` stays cheap and
never imports ttnn or torch (and therefore never opens or probes a device).
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("tt-atom")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0+unknown"

_EXPORTS = {
    "Calculator": "auto",
    "TTAtomCalculator": "calculator",
    "OrbCalculator": "orb_calculator",
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
