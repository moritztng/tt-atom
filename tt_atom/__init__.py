"""TT-Atom — high-performance Tenstorrent inference for eSEN / eSCN-MD (UMA-family)
equivariant ML interatomic potentials.

Public API is populated as modules land (model, calculator, weights, ...). Submodules
import ttnn lazily so that ``import tt_atom`` is cheap and never opens a device.
"""

__version__ = "0.1.0"

__all__ = ["UMA", "TTAtomCalculator", "WeightBundle", "Backbone", "HostGeometry", "MultiCard"]


def __getattr__(name):
    # lazy so that ``import tt_atom`` stays cheap and never imports ttnn/torch eagerly
    if name == "UMA":
        from .calculator import UMA

        return UMA
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
    if name == "MultiCard":
        from .batch import MultiCard

        return MultiCard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
