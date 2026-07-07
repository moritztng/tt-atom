"""Gated nonlinearity (``GateActivation``) used between the two SO(2) convolutions.

The l=0 (scalar) coefficient gets a plain SiLU; every higher-degree (vector) coefficient is
multiplied by a sigmoid gate broadcast from a per-degree scalar. We operate in the m-primed
coefficient order produced by the host Wigner map, so the gate-expansion is a fixed gather
that we realise as a slice + concat (only ``lmax`` distinct gate rows exist).

Reference: ``fairchem ... nn/activation.py:GateActivation`` (``m_prime=True``).
"""
from __future__ import annotations


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
        import torch

        self.ttnn = ttnn
        self.device = device
        self.lmax = lmax
        self.H = num_channels
        self.expand_index = _expand_index_m_prime(lmax, mmax)
        # gate expand as ONE matmul: gate_exp[E,(nsph-1)*H] = sigmoid(gating)[E,lmax*H] @ Expand.
        # Expand [lmax*H, (nsph-1)*H] is a 0/1 selector (column-block c = I_H at row-block
        # expand_index[c]) -> replaces the lmax*(nsph-1) slice+concat gather (fwd) and its
        # segment-sum transpose (bw) with a single (transpose-)matmul. Bit-identical (0/1, fp32 acc).
        from .device import compute_kernel_config
        self.kcfg = compute_kernel_config()
        H = num_channels
        ncol = len(self.expand_index)
        Ex = torch.zeros(lmax * H, ncol * H)
        for c, row in enumerate(self.expand_index):
            Ex[row * H:(row + 1) * H, c * H:(c + 1) * H] = torch.eye(H)
        self.expand_w = ttnn.from_torch(Ex.contiguous(), dtype=ttnn.bfloat16,
                                        layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, gating_scalars, x):
        """gating_scalars: ttnn ``[E, lmax*H]``; x: flat ttnn ``[E, nsph*H]`` (m-primed).
        Returns flat ``[E, nsph*H]``."""
        ttnn = self.ttnn
        E, H = x.shape[0], self.H
        self._cache_gating, self._cache_x = gating_scalars, x   # for the analytic-force VJP
        g = ttnn.sigmoid(gating_scalars)                         # [E, lmax*H], H-block per degree
        # expand the gate rows per vector coefficient as ONE matmul (0/1 selector); see __init__
        gate = ttnn.matmul(g, self.expand_w, compute_kernel_config=self.kcfg)  # [E, (nsph-1)*H]
        self._cache_gate = gate                                  # expanded gate for the VJP (fewer bw ops)
        scalar = ttnn.silu(ttnn.slice(x, [0, 0], [E, H]))        # l=0 coeff
        vector = ttnn.multiply(ttnn.slice(x, [0, H], [E, x.shape[1]]), gate)
        return ttnn.concat([scalar, vector], dim=1)
