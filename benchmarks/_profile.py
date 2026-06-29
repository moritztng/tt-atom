"""Throwaway profiling: Medges/s vs size, trace replay speedup, bmm-vs-so2 split."""
import sys, pathlib, time
import numpy as np, torch
from ase.build import bulk
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tests"))
from tt_atom import device as D
from tt_atom.model import Backbone, GraphContext
from tt_atom.geometry import HostGeometry, csd_embedding, radius_graph
from tt_atom.weights import WeightBundle
import ttnn

b = WeightBundle.load("/tmp/tt_full.npz")
cfg, w = b.config, b.weights
C = cfg["sphere_channels"]
dev = D.open_device(0, l1_small_size=32768, trace_region_size=200_000_000)
bb = Backbone(w, dev, cfg, b.to_grid_mat, b.from_grid_mat)
geo = HostGeometry(w, cfg, b.to_m, b.gauss_offset, b.gauss_coeff, gamma=0.0)


def build(nc):
    a = bulk("Si", "diamond", a=5.43) * (nc, nc, nc)
    a.rattle(stdev=0.1, seed=1)
    pos = torch.tensor(a.get_positions(), dtype=torch.float32)
    Z = torch.tensor(a.get_atomic_numbers())
    ei = radius_graph(pos, cfg["cutoff"])
    N, E = Z.shape[0], ei.shape[1]
    se = csd_embedding(w, torch.tensor([0.0]), torch.tensor([0.0]), C)[torch.zeros(N, dtype=torch.long)]
    t = geo(pos, Z, ei, se)
    graph = GraphContext(dev, edge_index=ei, wigner=t["wigner"].detach(),
                         wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
                         edge_envelope=t["edge_envelope"].detach(), num_nodes=N)
    se3 = ttnn.from_torch(se.reshape(N, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xi = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return N, E, graph, se3, xi


def timeit(fn, iters=20):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e3


for nc in [3, 4, 5]:
    N, E, graph, se3, xi = build(nc)

    def fwd():
        bb(xi, graph, se3)
    ms = timeit(fwd)
    print(f"[warm] N={N:4d} E={E:5d}  {ms:7.2f}ms  {E/(ms*1e-3)/1e6:5.3f} Medges/s")

    # micro: just the per-edge wigner bmm (forward only), repeated
    mcat = ttnn.from_torch(torch.randn(E, 9, 2 * C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    kcfg = D.compute_kernel_config()
    def bmm():
        ttnn.matmul(graph.wigner, mcat, compute_kernel_config=kcfg)
    bms = timeit(bmm, 30)
    print(f"        bmm[E,9,9]x[E,9,{2*C}] = {bms:6.2f}ms")

    # trace capture of full forward
    try:
        fwd(); ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        bb(xi, graph, se3)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        def tr():
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        tms = timeit(tr)
        print(f"        TRACE replay = {tms:6.2f}ms  ({ms/tms:.2f}x vs warm)  {E/(tms*1e-3)/1e6:5.3f} Medges/s")
        ttnn.release_trace(dev, tid)
    except Exception as e:
        print("        trace failed:", repr(e)[:200])

ttnn.close_device(dev)
