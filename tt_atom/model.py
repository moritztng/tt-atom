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


def balance_l0(ttnn, x, mean_op, cs, ce, add_scalar, kcfg):
    """Charge/spin channel balancing (fairchem ``eSCNMDBackbone.balance_channels``): shift the
    l=0 scalar part of channels ``[cs:ce]`` so each system's per-channel sum equals its target
    (charge). ``mean_op`` is the [N,N] per-system mean operator (block-diagonal ``1/natoms``);
    ``add_scalar = target/natoms`` (uniform per system — a bundle is one composition/charge, and
    a same-composition batch shares natoms). The map is a projection ``I - mean`` on those
    channels, hence self-adjoint: the identical call with ``add_scalar=0`` is its own VJP."""
    N, nsph, C = x.shape
    l0 = ttnn.reshape(ttnn.slice(x, [0, 0, 0], [N, 1, C]), (N, C))
    ch = ttnn.slice(l0, [0, cs], [N, ce])                                   # [N, nch]
    ch = ttnn.subtract(ch, ttnn.matmul(mean_op, ch, compute_kernel_config=kcfg))
    if add_scalar != 0.0:
        ch = ttnn.add(ch, add_scalar)
    parts = []
    if cs > 0:
        parts.append(ttnn.slice(l0, [0, 0], [N, cs]))
    parts.append(ch)
    if ce < C:
        parts.append(ttnn.slice(l0, [0, ce], [N, C]))
    l0n = ttnn.reshape(parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1), (N, 1, C))
    rest = ttnn.slice(x, [0, 1, 0], [N, nsph, C])
    return ttnn.concat([l0n, rest], dim=1)


class GraphContext:
    """Host-precomputed, device-resident geometric terms for one fixed topology."""

    def __init__(self, device, *, edge_index, wigner, wigner_inv, x_edge, edge_envelope,
                 num_nodes, fast=False, linear_scatter=None, system_natoms=None,
                 build_mean_op=False):
        import ttnn

        wdtype = ttnn.bfloat16
        E = edge_index.shape[1]
        self.E = E
        self.N = num_nodes
        # per-system mean operator for charge/spin channel balancing: M[i,j] = 1/natoms(sys i)
        # iff atoms i,j share a system, else 0. One system -> (1/N) ones[N,N]; a disjoint-union
        # batch -> block-diagonal. Built on host per topology (so it is captured once in a trace),
        # and only when a bundle actually balances channels (uma-s-1.2); None otherwise.
        self.node_meanM = None
        if build_mean_op:
            if system_natoms is None:
                M = torch.full((num_nodes, num_nodes), 1.0 / num_nodes)
            else:
                M = torch.zeros(num_nodes, num_nodes)
                off = 0
                for n in system_natoms:
                    M[off:off + n, off:off + n] = 1.0 / int(n)
                    off += int(n)
            self.node_meanM = _to_dev(M, device, wdtype)
        src = edge_index[0].to(torch.int32)
        tgt = edge_index[1].to(torch.int32)
        self.src_idx = _to_dev(src, device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tgt_idx = _to_dev(tgt, device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # edge->node scatter-add (``out[n] = sum_{e:tgt[e]==n} m[e]``, and the src transpose used by
        # the force VJP). Small systems: dense one-hot matmul S[N,E]@m — one fat op, bit-identical to
        # the golden mirror tests. Large systems: linear O(E) gather+reduce (scatter.py) — the dense
        # matmul is O(N^2) compute+memory (S alone is 92 MB at N=1000) and is why large-N scaling blew
        # up; fairchem/PyG scatter_add is linear. See SCATTER_LINEAR_THRESHOLD.
        # ``linear_scatter`` override: a disjoint-union BATCH is block-diagonal, so the dense
        # one-hot S[Ntot,Etot] is mostly off-diagonal zeros — its cost is O(Ntot*Etot) ~ O(K^2)
        # in the batch size while the linear gather+reduce stays O(Etot). So batches force the
        # linear path (see energy_and_forces_batch); single systems keep the node-count threshold.
        self.linear_scatter = (num_nodes > SCATTER_LINEAR_THRESHOLD if linear_scatter is None
                               else linear_scatter)
        if self.linear_scatter:
            from . import scatter as _sc

            tgt_g, self.Dmax_t = _sc.build_gather(tgt, num_nodes, E)
            src_g, self.Dmax_s = _sc.build_gather(src, num_nodes, E)
            self.tgt_gather = _to_dev(torch.from_numpy(tgt_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.src_gather = _to_dev(torch.from_numpy(src_g), device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        else:
            # scatter one-hot stays bf16: bf8_b is block-float (shared per-tile exponent), so the
            # 0/1 one-hot is NOT bit-exact in bf8 (measured Fpcc 0.98, no speed gain) — keep bf16.
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
        # (rotation._coef_exp for the fused kernel) to_layout to TILE on device. Only affects the
        # pos-dependent refresh cost -- topology buffers are unchanged.
        self.rot_fwd_coef = _to_dev(cf, device, wig_dtype, ttnn.ROW_MAJOR_LAYOUT)
        self.rot_inv_coef = _to_dev(ci, device, wig_dtype, ttnn.ROW_MAJOR_LAYOUT)
        # x_edge is stored ROW_MAJOR: the per-step trace refresh's from_torch of a wide [E,320] TILE
        # tensor does a slow host tilize (~24 ms vs ~1.4 ms RM); RadialMLP to_layouts it to TILE on
        # device (~0.16 ms, inside the trace) instead. Only consumer is RadialMLP (so2 rad + edge_degree).
        self.x_edge = _to_dev(x_edge, device, wdtype, ttnn.ROW_MAJOR_LAYOUT)
        # only the flat [E,1] envelope is consumed on device (edgewise / edge-degree broadcast); the
        # 3D [E,1,1] form tile-pads to [E,32,32] (a ~64 ms/step re-tilize on the trace refresh) and
        # is read by nothing, so it is not materialised.
        # Store ROW_MAJOR bf16 (like x_edge / rot coefs): a TILE (esp. bf8) host from_torch of the
        # [E,1] envelope on the per-step refresh does a pathological host tilize + bf8 shared-exp
        # pack (~7.8 ms/step for bf8, the single largest refresh cost). RM upload is ~0.1 ms; the
        # forward tilizes (and, in bf8-edge mode, casts to bf8) ON DEVICE inside the trace via
        # ``materialize_envelope`` — moving the whole cost to a tiny device op on the [E,1] tensor.
        self._env_dtype = ttnn.bfloat8_b if _b8 else wdtype
        self.edge_envelope_rm = _to_dev(edge_envelope.reshape(E, 1), device, wdtype,
                                        ttnn.ROW_MAJOR_LAYOUT)
        # materialize once here so the eager / per-module test path (which calls edge_wise without
        # going through node_embedding) has a valid buffer; the traced forward re-materializes at
        # its start so the tilize op reads the per-step-refreshed RM buffer (see node_embedding).
        self.materialize_envelope()

    def materialize_envelope(self):
        """Tilize (and, in bf8-edge mode, cast to bf8) the RM envelope on device. Called once at
        the start of the backbone forward; the resulting device tensor is reused by every edgewise
        block, the edge-degree init, and the backward. Captured in the trace so the per-step
        refresh only writes the cheap RM buffer."""
        import ttnn
        ev = ttnn.to_layout(self.edge_envelope_rm, ttnn.TILE_LAYOUT)
        if self._env_dtype == ttnn.bfloat8_b:
            ev = ttnn.typecast(ev, ttnn.bfloat8_b)
        self.edge_envelope_f = ev
        return ev


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
        # charge-balanced channels (fairchem charge_balanced_channels): l=0 scalar channels [cs:ce]
        # are shifted after every block so their per-system sum equals the charge. cs==ce disables
        # it (uma-s-1 / random-weight bundles); uma-s-1.2 uses [0:3].
        self.cs = int(cfg.get("charge_channel_start", 0))
        self.ce = int(cfg.get("charge_channel_end", 0))
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

    def node_embedding(self, x_init, graph, sys_node_embedding, balance_add=0.0):
        """Run the backbone; returns device node embedding ``[N, nsph, C]``.

        When the device edge-degree embedding is active, ``x_init`` is the constant l0 node init
        and the full node init is built on device from the graph's geometric terms.

        ``balance_add`` is the per-atom charge target ``charge/natoms`` (0 for neutral or when
        balancing is disabled); when ``cs<ce`` the l=0 charge channels are re-balanced after every
        block to mirror fairchem's ``eSCNMDBackbone``."""
        graph.materialize_envelope()   # tilize (+bf8 cast) the RM envelope on device, once per fwd
        if self.edge_degree is not None:
            x_init = self.edge_degree(graph, x_init)
        x = x_init
        do_bal = self.ce > self.cs
        for blk in self.blocks:
            x = blk(x, graph, sys_node_embedding)
            if do_bal:
                x = balance_l0(self.ttnn, x, graph.node_meanM, self.cs, self.ce, balance_add, self.kcfg)
        return self.final_norm(x)

    def node_energy(self, node_emb):
        """Per-node energy MLP (Linear-SiLU-Linear-SiLU-Linear) on the l=0 channel -> ``[N, 1]``."""
        ttnn = self.ttnn
        N = node_emb.shape[0]
        h = ttnn.slice(node_emb, [0, 0, 0], [N, 1, self.C])
        h = ttnn.reshape(h, (N, self.C))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[0], bias=self.eh_b[0], compute_kernel_config=self.kcfg))
        h = ttnn.silu(ttnn.linear(h, self.eh_w[1], bias=self.eh_b[1], compute_kernel_config=self.kcfg))
        # fp32 output: the per-node energy (~1-2 eV once element references are subtracted) would
        # otherwise be re-quantized to bf16 (~2^-8 rel), which is the dominant device energy error
        # for large-|raw| systems (MgO, radicals). Does NOT affect forces — their VJP (energy_bw)
        # seeds from the head weights, not this value.
        return ttnn.linear(h, self.eh_w[2], bias=self.eh_b[2], compute_kernel_config=self.kcfg,
                           dtype=ttnn.float32)  # [N,1] fp32

    def energy(self, node_emb):
        """Total energy of a single system: sum of the per-node energy (fp32)."""
        return self.ttnn.sum(self.node_energy(node_emb), dim=0)

    def energy_batch(self, node_emb, seg):
        """Per-system energies of a disjoint-union batch: segment-sum of the per-node energy by
        the one-hot segment matrix ``seg`` [K, N] (``seg[k, n] = 1`` iff atom n is in system k),
        expressed as the tile-friendly matmul ``seg @ node_energy`` -> ``[K, 1]``. Block-diagonal
        batching leaves every backbone op within-system, so this reduction is the only change."""
        ttnn = self.ttnn
        ne = self.node_energy(node_emb)                              # [N,1] fp32
        return ttnn.matmul(ttnn.typecast(seg, ttnn.float32), ne, compute_kernel_config=self.kcfg)

    def __call__(self, x_init, graph, sys_node_embedding, balance_add=0.0):
        node_emb = self.node_embedding(x_init, graph, sys_node_embedding, balance_add)
        return node_emb, self.energy(node_emb)
