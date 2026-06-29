"""Per-module device-time breakdown at large E."""
import sys, pathlib, time
import torch
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
dev = D.open_device(0, l1_small_size=32768)
bb = Backbone(w, dev, cfg, b.to_grid_mat, b.from_grid_mat)
geo = HostGeometry(w, cfg, b.to_m, b.gauss_offset, b.gauss_coeff, gamma=0.0)

a = bulk("Si", "diamond", a=5.43) * (5, 5, 5)
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
print(f"N={N} E={E}")

def timeit(fn, iters=20):
    fn(); ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e3

blk = bb.blocks[0]
x = xi
xn = blk.norm_1(x)

# inputs for sub-modules
ew = blk.edge_wise
def gather_only():
    xf = ttnn.reshape(xn, (N, 9 * C)); xf = ttnn.to_layout(xf, ttnn.ROW_MAJOR_LAYOUT)
    xs = ttnn.reshape(ttnn.embedding(graph.src_idx, xf), (E, 9, C))
    xt = ttnn.reshape(ttnn.embedding(graph.tgt_idx, xf), (E, 9, C))
    xs = ttnn.to_layout(xs, ttnn.TILE_LAYOUT); xt = ttnn.to_layout(xt, ttnn.TILE_LAYOUT)
    m_cat = ttnn.concat([xs, xt], dim=2)
    return m_cat
m_cat = gather_only()
print(f"  norm_1            {timeit(lambda: blk.norm_1(x)):7.2f}ms")
print(f"  gather+concat     {timeit(gather_only):7.2f}ms")
kcfg = D.compute_kernel_config()
print(f"  wigner bmm        {timeit(lambda: ttnn.matmul(graph.wigner, m_cat, compute_kernel_config=kcfg)):7.2f}ms")
m_rot = ttnn.matmul(graph.wigner, m_cat, compute_kernel_config=kcfg)
print(f"  so2_1             {timeit(lambda: ew.so2_1(m_rot, graph.x_edge)):7.2f}ms")
m1, gating = ew.so2_1(m_rot, graph.x_edge)
print(f"  gate              {timeit(lambda: ew.gate(gating, m1)):7.2f}ms")
mg = ew.gate(gating, m1)
print(f"  so2_2             {timeit(lambda: ew.so2_2(mg, graph.x_edge)):7.2f}ms")
m2 = ew.so2_2(mg, graph.x_edge)
def envrot():
    m = ttnn.multiply(m2, graph.edge_envelope)
    return ttnn.matmul(graph.wigner_inv, m, compute_kernel_config=kcfg)
print(f"  env+wigner_inv    {timeit(envrot):7.2f}ms")
m_back = envrot()
def scatter():
    mf = ttnn.reshape(m_back, (E, 9 * C))
    return ttnn.matmul(graph.scatter, mf, compute_kernel_config=kcfg)
print(f"  scatter matmul    {timeit(scatter):7.2f}ms")
print(f"  --- full edgewise {timeit(lambda: ew(xn, graph)):7.2f}ms")
print(f"  norm_2            {timeit(lambda: blk.norm_2(x)):7.2f}ms")
print(f"  atom_wise(grid)   {timeit(lambda: blk.atom_wise(x)):7.2f}ms")
print(f"  === full block    {timeit(lambda: blk(x, graph, se3)):7.2f}ms")
ttnn.close_device(dev)
