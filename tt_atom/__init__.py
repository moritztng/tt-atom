"""TT-Atom — high-performance Tenstorrent inference for eSEN / eSCN-MD (UMA-family)
equivariant ML interatomic potentials.

Public API is populated as modules land (model, calculator, weights, ...). Submodules
import ttnn lazily so that ``import tt_atom`` is cheap and never opens a device.
"""

__version__ = "0.1.0"
