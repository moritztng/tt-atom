"""Golden-fixture helpers shared by the parity tests."""
import json
import pathlib

import numpy as np
import torch

DATA = pathlib.Path(__file__).parent / "data"


def pcc(a, b):
    """Pearson correlation of two tensors/arrays, flattened to fp64."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


class Golden:
    """Accessor over a golden npz: weights (``w@``), activations (``a@``),
    host terms (``host@``), inputs (``in@``), outputs (``out@``)."""

    def __init__(self, name):
        self.d = np.load(DATA / name)
        self.config = json.loads(bytes(self.d["config"]).decode())

    def _t(self, key):
        return torch.from_numpy(self.d[key].copy())

    def w(self):
        return {k[2:]: self._t(k).float() for k in self.d.files if k.startswith("w@")}

    def act(self, name):
        return self._t(f"a@{name}").float()

    def host(self, name):
        return self._t(f"host@{name}").float()

    def inp(self, name):
        return self._t(f"in@{name}")

    def out(self, name):
        return self._t(f"out@{name}").float()
