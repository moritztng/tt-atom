"""Equivariant primitives shared by tt-atom (MACE) and tt-bio (DiffDock). Torch only, no repo imports."""
from tt_atom.equivariant.w3j import E3NN_04_FLIPS, wigner_3j

__all__ = ["wigner_3j", "E3NN_04_FLIPS"]
