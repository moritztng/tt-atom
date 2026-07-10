"""Weight loading for the Orb-v3 port: a portable ``.npz`` bundle of MoleculeGNS parameters +
real activations/inputs, generated in the orb-models reference env (numpy>=2) by
``tests/gen_golden_orb.py`` and read here in the ttnn env (numpy<2).

Mirrors ``tt_atom/weights.py``'s bundle format/rationale (ttnn and orb-models/fairchem cannot
share a numpy major version in one process) but is Orb-specific: the weight namespace, config
keys, and activation names come straight from ``orb_models.forcefield.gns.MoleculeGNS`` /
``AttentionInteractionNetwork``, which share nothing with the eSCN-MD/UMA bundle.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch


class OrbWeights:
    def __init__(self, npz):
        self._d = npz
        self.config = json.loads(bytes(npz["config"]).decode())

    @classmethod
    def load(cls, path):
        return cls(np.load(pathlib.Path(path)))

    def _t(self, key):
        return torch.from_numpy(self._d[key].copy())

    @property
    def weights(self):
        return {k[2:]: self._t(k).float() for k in self._d.files if k.startswith("w@")}

    def has(self, key):
        return key in self._d.files

    def activation(self, key):
        return self._t(f"a@{key}").float()

    def host(self, key):
        return self._t(f"host@{key}").float()

    def inp(self, key):
        return self._t(f"in@{key}")

    def out(self, key):
        return self._t(f"out@{key}")
