"""TT-Atom — high-performance Tenstorrent inference for eSEN / eSCN-MD (UMA-family)
equivariant ML interatomic potentials.

Public API is populated as modules land (model, calculator, weights, ...). Submodules
import ttnn lazily so that ``import tt_atom`` is cheap and never opens a device.
"""

__version__ = "0.1.0"

__all__ = ["TTAtomCalculator", "WeightBundle", "Backbone", "HostGeometry"]


def __getattr__(name):
    # lazy so that ``import tt_atom`` stays cheap and never imports ttnn/torch eagerly
    if name == "TTAtomCalculator":
        from .calculator import TTAtomCalculator

        return TTAtomCalculator
    if name == "WeightBundle":
        from .weights import WeightBundle

        return WeightBundle
    if name == "Backbone":
        from .model import Backbone

        return Backbone
    if name == "HostGeometry":
        from .geometry import HostGeometry

        return HostGeometry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
