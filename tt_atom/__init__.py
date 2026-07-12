"""TT-Atom — high-performance Tenstorrent inference for ML interatomic potentials: Meta's UMA
(eSEN / eSCN-MD, equivariant) and Orbital Materials' Orb-v3 / OrbMol (non-equivariant).

Public API is populated as modules land (model, calculator, orb_model, orb_calculator, ...).
Submodules import ttnn lazily so that ``import tt_atom`` is cheap and never opens a device.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("tt-atom")
except PackageNotFoundError:  # running from a source tree, not an installed dist
    __version__ = "0+unknown"

__all__ = ["Calculator", "TTAtomCalculator", "OrbCalculator", "WeightBundle", "Backbone",
          "HostGeometry", "MultiCard"]


def __getattr__(name):
    # lazy so that ``import tt_atom`` stays cheap and never imports ttnn/torch eagerly
    if name == "Calculator":
        from .auto import Calculator

        return Calculator
    if name == "TTAtomCalculator":
        from .calculator import TTAtomCalculator

        return TTAtomCalculator
    if name == "OrbCalculator":
        from .orb_calculator import OrbCalculator

        return OrbCalculator
    if name == "WeightBundle":
        from .weights import WeightBundle

        return WeightBundle
    if name == "Backbone":
        from .model import Backbone

        return Backbone
    if name == "HostGeometry":
        from .geometry import HostGeometry

        return HostGeometry
    if name == "MultiCard":
        from .batch import MultiCard

        return MultiCard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
