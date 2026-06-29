"""Gated nonlinearity (``GateActivation``) used between the two SO(2) convolutions.

The l=0 (scalar) coefficient gets a plain SiLU; every higher-degree (vector) coefficient is
multiplied by a sigmoid gate broadcast from a per-degree scalar. We operate in the m-primed
coefficient order produced by the host Wigner map, so the gate-expansion is a fixed gather
that we realise as a slice + concat (only ``lmax`` distinct gate rows exist).

Reference: ``fairchem ... nn/activation.py:GateActivation`` (``m_prime=True``).
"""
from __future__ import annotations

import torch


def _expand_index_m_prime(lmax, mmax):
    """The per-vector-coefficient gate index in m-primed order (see reference)."""
    idx = []
    idx += list(range(lmax))                       # m=0 block: l1m0, l2m0, ...
    for mval in range(1, mmax + 1):
        r = list(range(mval - 1, lmax))
        idx += r + r                               # real + imag halves
    return idx


class GateActivation:
    def __init__(self, device, *, lmax, mmax, num_channels):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.lmax = lmax
        self.H = num_channels
        self.expand_index = _expand_index_m_prime(lmax, mmax)

    def __call__(self, gating_scalars, x):
        """gating_scalars: ttnn ``[N, lmax*H]``; x: ttnn ``[N, nsph, H]`` (m-primed)."""
        ttnn = self.ttnn
        N = x.shape[0]
        g = ttnn.sigmoid(gating_scalars)
        g = ttnn.reshape(g, (N, self.lmax, self.H))
        # gather gate rows per vector coefficient (lmax distinct rows -> slice+concat)
        rows = [ttnn.slice(g, [0, l, 0], [N, l + 1, self.H]) for l in range(self.lmax)]
        gate = ttnn.concat([rows[i] for i in self.expand_index], dim=1)   # [N, nsph-1, H]

        scalar = ttnn.silu(ttnn.slice(x, [0, 0, 0], [N, 1, self.H]))
        vector = ttnn.slice(x, [0, 1, 0], [N, x.shape[1], self.H])
        vector = ttnn.multiply(vector, gate)
        return ttnn.concat([scalar, vector], dim=1)
