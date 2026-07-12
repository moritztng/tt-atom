"""Weight loading for the Orb-v3 port: a portable ``.npz`` bundle of MoleculeGNS parameters +
real activations/inputs, generated in the orb-models reference env (numpy>=2) by
``tests/gen_golden_orb.py`` and read here in the ttnn env (numpy<2).

Shares the ``.npz`` container mechanics (config parse, ``w@`` weights, lazy tensor copy) with
``tt_atom/weights.py`` via :class:`bundle.NpzBundle`, but the extra accessors are Orb-specific:
the weight namespace, config keys, and activation names come straight from
``orb_models.forcefield.gns.MoleculeGNS`` / ``AttentionInteractionNetwork``, which share nothing
with the eSCN-MD/UMA bundle."""
from __future__ import annotations

from .bundle import NpzBundle


class OrbWeights(NpzBundle):
    def activation(self, key):
        return self._t(f"a@{key}").float()

    def host(self, key):
        return self._t(f"host@{key}").float()

    def inp(self, key):
        return self._t(f"in@{key}")

    def out(self, key):
        return self._t(f"out@{key}")
