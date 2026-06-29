"""eSEN / eSCN-MD backbone forward, device-resident on Tenstorrent.

Assembles the ported modules (RMS-norm-SH, edgewise SO(2) message passing, grid feed-forward)
into the full backbone and an energy head. The geometric, per-edge terms (Wigner matrices,
radial edge embedding, envelope, the graph itself) are computed on host once — they are <1% of
the compute — and uploaded as the device-resident ``GraphContext``. Everything else (the dense
GEMM bulk) stays on device across the whole forward.

Reference: ``fairchem ... escn_md.py:eSCNMDBackbone.forward`` + ``escn_md_block.py``.
"""
from __future__ import annotations

import torch

from .device import compute_kernel_config
from .norm import RMSNormSH
from .edgewise import Edgewise
from .grid import GridAtomwise


def _to_dev(t, device, dtype, layout=None):
    import ttnn

    layout = layout or ttnn.TILE_LAYOUT
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


class GraphContext:
    """Host-precomputed, device-resident geometric terms for one fixed topology."""

    def __init__(self, device, *, edge_index, wigner, wigner_inv, x_edge, edge_envelope,
                 num_nodes, fast=False):
        import ttnn

        wdtype = ttnn.bfloat16
        E = edge_index.shape[1]
        self.E = E
        self.N = num_nodes
        src = edge_index[0].to(torch.int32)
        tgt = edge_index[1].to(torch.int32)
        self.src_idx = _to_dev(src, device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tgt_idx = _to_dev(tgt, device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # scatter-add matrix S[N, E]: messages on edge e land on node tgt[e]
        S = torch.zeros(num_nodes, E)
        S[tgt.long(), torch.arange(E)] = 1.0
        self.scatter = _to_dev(S, device, wdtype)
        self.wigner = _to_dev(wigner, device, wdtype)
        self.wigner_inv = _to_dev(wigner_inv, device, wdtype)
        self.x_edge = _to_dev(x_edge, device, wdtype)
        self.edge_envelope = _to_dev(edge_envelope, device, wdtype)


class _Block:
    def __init__(self, weights, prefix, device, cfg, to_grid, from_grid, fast=False):
        self.norm_1 = RMSNormSH(weights, f"{prefix}.norm_1", device,
                                lmax=cfg["lmax"], num_channels=cfg["sphere_channels"])
        self.edge_wise = Edgewise(weights, f"{prefix}.edge_wise", device,
                                  sphere_channels=cfg["sphere_channels"],
                                  hidden_channels=cfg["hidden_channels"],
                                  lmax=cfg["lmax"], mmax=cfg["mmax"], fast=fast)
        self.norm_2 = RMSNormSH(weights, f"{prefix}.norm_2", device,
                                lmax=cfg["lmax"], num_channels=cfg["sphere_channels"])
        self.atom_wise = GridAtomwise(weights, f"{prefix}.atom_wise", device,
                                      to_grid, from_grid, fast=fast)

    def __call__(self, x, graph, sys_node_embedding):
        import ttnn

        C = sys_node_embedding.shape[-1]
        N = x.shape[0]
        x_res = x
        x = self.norm_1(x)
        # add system embedding at l=0 only
        l0 = ttnn.add(ttnn.slice(x, [0, 0, 0], [N, 1, C]), sys_node_embedding)
        x = ttnn.concat([l0, ttnn.slice(x, [0, 1, 0], [N, x.shape[1], C])], dim=1)
        x = ttnn.add(self.edge_wise(x, graph), x_res)
        x_res = x
        x = self.norm_2(x)
        x = ttnn.add(self.atom_wise(x), x_res)
        return x


class Backbone:
    """The eSCN-MD backbone forward + energy head, fully device-resident."""

    def __init__(self, weights, device, cfg, to_grid_mat, from_grid_mat, *, fast=False):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.cfg = cfg
        self.C = cfg["sphere_channels"]
        self.kcfg = compute_kernel_config()
        wdtype = ttnn.bfloat16

        self.blocks = [
            _Block(weights, f"blocks.{i}", device, cfg, to_grid_mat, from_grid_mat, fast=fast)
            for i in range(cfg["num_layers"])
        ]
        self.final_norm = RMSNormSH(weights, "norm", device,
                                    lmax=cfg["lmax"], num_channels=self.C)
        # energy head: Linear-SiLU-Linear-SiLU-Linear on the l=0 channel
        self.eh_w = [_to_dev(weights[f"energy_block.{i}.weight"].T.contiguous(), device, wdtype)
                     for i in (0, 2, 4)]
        self.eh_b = [_to_dev(weights[f"energy_block.{i}.bias"], device, wdtype)
                     for i in (0, 2, 4)]

    def node_embedding(self, x_init, graph, sys_node_embedding):
        """Run the backbone; returns device node embedding ``[N, nsph, C]``."""
        x = x_init
        for blk in self.blocks:
            x = blk(x, graph, sys_node_embedding)
        return self.final_norm(x)

    def energy(self, node_emb):
        """Sum of per-node energy MLP on the l=0 channel (single system)."""
        ttnn = self.ttnn
        N = node_emb.shape[0]
        h = ttnn.slice(node_emb, [0, 0, 0], [N, 1, self.C])
        h = ttnn.reshape(h, (N, self.C))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[0], bias=self.eh_b[0], compute_kernel_config=self.kcfg))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[1], bias=self.eh_b[1], compute_kernel_config=self.kcfg))
        h = ttnn.linear(h, self.eh_w[2], bias=self.eh_b[2], compute_kernel_config=self.kcfg)  # [N,1]
        return ttnn.sum(h, dim=0)

    def __call__(self, x_init, graph, sys_node_embedding):
        node_emb = self.node_embedding(x_init, graph, sys_node_embedding)
        return node_emb, self.energy(node_emb)
