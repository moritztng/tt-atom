"""eSEN / eSCN-MD backbone forward, device-resident on Tenstorrent.

Assembles the ported modules (RMS-norm-SH, edgewise SO(2) message passing, grid feed-forward)
into the full backbone and an energy head. The geometric, per-edge terms (Wigner matrices,
radial edge embedding, envelope, the graph itself) are computed on host once — they are <1% of
the compute — and uploaded as the device-resident ``GraphContext``. Everything else (the dense
GEMM bulk) stays on device across the whole forward.

Reference: ``fairchem ... escn_md.py:eSCNMDBackbone.forward`` + ``escn_md_block.py``.
"""
from __future__ import annotations

import os

import torch

from .device import compute_kernel_config
from .norm import RMSNormSH
from .edgewise import Edgewise
from .grid import GridAtomwise
from .spectral import SpectralAtomwise

# Above this node count the dense one-hot scatter matmul S[N,E]@m (O(N^2)) is replaced by the
# linear O(E) gather+reduce scatter (tt_atom/scatter.py). Small systems keep the dense path
# (one fat matmul, bit-identical to the golden mirror tests). Override with $TT_ATOM_SCATTER_THRESHOLD
# (set to 0 to force the linear path everywhere — used by the scatter parity test).
# The dense one-hot matmul scatter S[N,E]@m is the golden (bit-identical) path AND measured ~5x
# faster than the linear gather+reduce at N<=~1728 (segment_sum's RM-pad + gather dominate; the
# matmul is one fat bf16 op). The O(N*E) one-hot (92 MB @N=1000, 546 MB @N=1728) fits DRAM easily
# at MD sizes, so use it up to ~2048 nodes; only truly-large scale runs fall back to the linear
# O(E) path (scatter.py) to bound the O(N^2) memory. Was 384 (linear kicked in far too early).
SCATTER_LINEAR_THRESHOLD = int(os.environ.get("TT_ATOM_SCATTER_THRESHOLD", "2048"))


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
        # edge->node scatter-add (``out[n] = sum_{e:tgt[e]==n} m[e]``, and the src transpose used by
        # the force VJP). Small systems: dense one-hot matmul S[N,E]@m — one fat op, bit-identical to
        # the golden mirror tests. Large systems: linear O(E) gather+reduce (scatter.py) — the dense
        # matmul is O(N^2) compute+memory (S alone is 92 MB at N=1000) and is why large-N scaling blew
        # up; fairchem/PyG scatter_add is linear. See SCATTER_LINEAR_THRESHOLD.
        self.linear_scatter = num_nodes > SCATTER_LINEAR_THRESHOLD
        if self.linear_scatter:
            from . import scatter as _sc

            tgt_g, self.Dmax_t = _sc.build_gather(tgt, num_nodes, E)
            src_g, self.Dmax_s = _sc.build_gather(src, num_nodes, E)
            self.tgt_gather = _to_dev(torch.from_numpy(tgt_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.src_gather = _to_dev(torch.from_numpy(src_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        else:
            S = torch.zeros(num_nodes, E)
            S[tgt.long(), torch.arange(E)] = 1.0
            self.scatter = _to_dev(S, device, wdtype)
            Ssrc = torch.zeros(num_nodes, E)
            Ssrc[src.long(), torch.arange(E)] = 1.0
            self.scatter_src = _to_dev(Ssrc, device, wdtype)
        # Wigner rotation as a flat sparse multiply-accumulate (see rotation.py): pack the dense
        # per-edge matrices to their structural nonzeros. bf8 coefficients run faster and stay
        # PCC-safe (the rotation is an orthogonal basis change) -> use in --fast.
        from . import rotation
        from .device import bf8_edge
        # wigner (wig_M) is [E, nred, nsph], its inverse [E, nsph, nred]. nred is the reduced
        # m-space (|m|<=mmax); nred == nsph when mmax==lmax (uma-s), nred < nsph for uma-m.
        self.nred, self.nsph = wigner.shape[1], wigner.shape[2]
        # bf8-edge: coef stays bf16 ROW_MAJOR here (bf8 can't be RM; RM is needed for the cheap
        # per-step refresh). rotation.rotate casts the on-device TILE-expanded coef to bf8 to match
        # its bf8 x input. bf8 rotation coef is parity-safe (orthogonal basis change, O(1) coefs).
        _b8 = bf8_edge()
        wig_dtype = ttnn.bfloat8_b if fast else wdtype
        self.rot_fwd_ij, cf = rotation.pack(wigner)        # node SH (nsph) -> reduced m-space (nred)
        self.rot_inv_ij, ci = rotation.pack(wigner_inv)    # reduced m-space (nred) -> node SH (nsph)
        # coef stored ROW_MAJOR: the per-step refresh's from_torch of a [E, nnz] TILE tensor pays a
        # tile-pad host tilize (~1.7-3.9 ms each; nnz pads to 32) vs ~0.04 ms RM. Consumers
        # (rotation._coef_exp for the fused kernel; the non-fused MAC split) to_layout to TILE on
        # device. Only affects the pos-dependent refresh cost -- topology buffers are unchanged.
        self.rot_fwd_coef = _to_dev(cf, device, wig_dtype, ttnn.ROW_MAJOR_LAYOUT)
        self.rot_inv_coef = _to_dev(ci, device, wig_dtype, ttnn.ROW_MAJOR_LAYOUT)
        # x_edge is stored ROW_MAJOR: the per-step trace refresh's from_torch of a wide [E,320] TILE
        # tensor does a slow host tilize (~24 ms vs ~1.4 ms RM); RadialMLP to_layouts it to TILE on
        # device (~0.16 ms, inside the trace) instead. Only consumer is RadialMLP (so2 rad + edge_degree).
        self.x_edge = _to_dev(x_edge, device, wdtype, ttnn.ROW_MAJOR_LAYOUT)
        # only the flat [E,1] envelope is consumed on device (edgewise / edge-degree broadcast); the
        # 3D [E,1,1] form tile-pads to [E,32,32] (a ~64 ms/step re-tilize on the trace refresh) and
        # is read by nothing, so it is not materialised.
        # envelope multiplies the bf8 so2 output in bf8-edge mode -> store bf8 so dtypes match
        env_dtype = ttnn.bfloat8_b if _b8 else wdtype
        self.edge_envelope_f = _to_dev(edge_envelope.reshape(E, 1), device, env_dtype)  # flat bcast


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
        self.ff_type = cfg.get("ff_type", "grid")
        if self.ff_type == "spectral":
            self.atom_wise = SpectralAtomwise(weights, f"{prefix}.atom_wise", device,
                                              sphere_channels=cfg["sphere_channels"],
                                              hidden_channels=cfg["hidden_channels"],
                                              lmax=cfg["lmax"], mmax=cfg["mmax"], fast=fast)
        else:
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
        # optional on-device edge-degree embedding (node init) — moves the largest per-step host
        # cost (radial-MLP fwd+bw over E edges) onto the device inside the trace. When enabled the
        # ``x_init`` operand passed to node_embedding is instead the CONSTANT l0 init and the full
        # node init is computed on device from the graph's geometric terms. See tt_atom/edge_degree.
        from .device import device_ede
        if device_ede():
            from .edge_degree import EdgeDegreeEmbedding
            self.edge_degree = EdgeDegreeEmbedding(weights, device, cfg,
                                                   rescale=cfg.get("edge_degree_rescale", 5.0))
        else:
            self.edge_degree = None
        # energy head: Linear-SiLU-Linear-SiLU-Linear on the l=0 channel
        self.eh_w = [_to_dev(weights[f"energy_block.{i}.weight"].T.contiguous(), device, wdtype)
                     for i in (0, 2, 4)]
        self.eh_b = [_to_dev(weights[f"energy_block.{i}.bias"], device, wdtype)
                     for i in (0, 2, 4)]

    def node_embedding(self, x_init, graph, sys_node_embedding):
        """Run the backbone; returns device node embedding ``[N, nsph, C]``.

        When the device edge-degree embedding is active, ``x_init`` is the constant l0 node init
        and the full node init is built on device from the graph's geometric terms."""
        if self.edge_degree is not None:
            x_init = self.edge_degree(graph, x_init)
        x = x_init
        for blk in self.blocks:
            x = blk(x, graph, sys_node_embedding)
        return self.final_norm(x)

    def node_energy(self, node_emb):
        """Per-node energy MLP (Linear-SiLU-Linear-SiLU-Linear) on the l=0 channel -> ``[N, 1]``."""
        ttnn = self.ttnn
        N = node_emb.shape[0]
        h = ttnn.slice(node_emb, [0, 0, 0], [N, 1, self.C])
        h = ttnn.reshape(h, (N, self.C))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[0], bias=self.eh_b[0], compute_kernel_config=self.kcfg))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[1], bias=self.eh_b[1], compute_kernel_config=self.kcfg))
        return ttnn.linear(h, self.eh_w[2], bias=self.eh_b[2], compute_kernel_config=self.kcfg)  # [N,1]

    def energy(self, node_emb):
        """Total energy of a single system: sum of the per-node energy."""
        return self.ttnn.sum(self.node_energy(node_emb), dim=0)

    def energy_batch(self, node_emb, seg):
        """Per-system energies of a disjoint-union batch: segment-sum of the per-node energy by
        the one-hot segment matrix ``seg`` [K, N] (``seg[k, n] = 1`` iff atom n is in system k),
        expressed as the tile-friendly matmul ``seg @ node_energy`` -> ``[K, 1]``. Block-diagonal
        batching leaves every backbone op within-system, so this reduction is the only change."""
        return self.ttnn.matmul(seg, self.node_energy(node_emb),
                                compute_kernel_config=self.kcfg)

    def __call__(self, x_init, graph, sys_node_embedding):
        node_emb = self.node_embedding(x_init, graph, sys_node_embedding)
        return node_emb, self.energy(node_emb)
