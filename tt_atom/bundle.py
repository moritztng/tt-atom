"""Shared base for the portable ``.npz`` weight bundles.

Both model families export their reference weights (in the fairchem / orb-models refenv, numpy>=2)
into a numpy ``.npz`` that the ttnn env (numpy<2) reads back — the two cannot share a process. The
container mechanics are identical across families: parse the embedded JSON ``config``, lazily copy
arrays into torch tensors, and pull the ``w@``-prefixed learned weights. Only the *extra*
per-family accessors differ, so this holds the shared mechanics and ``weights.WeightBundle`` (UMA)
/ ``orb_weights.OrbWeights`` (Orb) subclass it."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch


class NpzBundle:
    """A loaded ``.npz`` carrying an embedded JSON ``config`` and ``w@``-prefixed weight arrays."""

    def __init__(self, npz):
        self._d = npz
        self.config = json.loads(bytes(npz["config"]).decode())

    @classmethod
    def load(cls, path):
        return cls(np.load(pathlib.Path(path)))

    def _t(self, key):
        return torch.from_numpy(self._d[key].copy())

    def has(self, key):
        return key in self._d.files

    @property
    def weights(self):
        return {k[2:]: self._t(k).float() for k in self._d.files if k.startswith("w@")}
