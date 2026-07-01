"""Single-card throughput + end-to-end CPU-vs-TT benchmark for TT-Atom.

Measures, over a system-size sweep, on real hardware:
  * warm device-resident forward (energy) ms/eval and Medges/s   -- pure device compute
  * end-to-end ms/eval (host geometry + upload + forward + readback) -- what a user pays
  * a PyTorch-CPU reference ms/eval (tests/mirror.py, a bit-exact transcription of the same
    function) for an honest CPU-vs-TT speedup

All numbers are real and measured here; nothing is hardcoded. Results -> benchmarks/results/.
Run:  ~/.ttatom_run/env/bin/python benchmarks/bench_throughput.py --weights /tmp/tt_full.npz
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from ase.build import bulk

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tests"))
import mirror  # noqa: E402

from tt_atom import device as D  # noqa: E402
from tt_atom.model import Backbone, GraphContext  # noqa: E402
from tt_atom.geometry import HostGeometry, csd_embedding, radius_graph  # noqa: E402
from tt_atom.weights import WeightBundle  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"


def make_system(n_cells):
    a = bulk("Si", "diamond", a=5.43) * (n_cells, n_cells, n_cells)
    a.rattle(stdev=0.1, seed=1)
    return a


def time_it(fn, iters):
    fn()  # warmup (compile / program-cache fill)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cells", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="bf8 weights + rotation coefficients")
    args = ap.parse_args()

    b = WeightBundle.load(args.weights)
    cfg, w = b.config, b.weights
    C = cfg["sphere_channels"]
    dev = D.open_device(args.device_id)
    bb = Backbone(w, dev, cfg, b.to_grid_mat, b.from_grid_mat, fast=args.fast)
    geo = HostGeometry(w, cfg, b.to_m, b.gauss_offset, b.gauss_coeff, gamma=0.0)
    import ttnn

    rows = []
    for nc in args.cells:
        atoms = make_system(nc)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        Z = torch.tensor(atoms.get_atomic_numbers())
        ei, _ = radius_graph(pos, cfg["cutoff"])
        N, E = Z.shape[0], ei.shape[1]
        se = csd_embedding(w, torch.tensor([0.0]), torch.tensor([0.0]), C)[torch.zeros(N, dtype=torch.long)]

        def host_geom():
            return geo(pos, Z, ei, se)

        t = host_geom()

        def upload(t):
            graph = GraphContext(dev, edge_index=ei, wigner=t["wigner"].detach(),
                                 wigner_inv=t["wigner_inv"].detach(), x_edge=t["x_edge"].detach(),
                                 edge_envelope=t["edge_envelope"].detach(), num_nodes=N, fast=args.fast)
            se3 = ttnn.from_torch(se.reshape(N, 1, C), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            xi = ttnn.from_torch(t["x_init"].detach(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            return graph, se3, xi

        graph, se3, xi = upload(t)

        # warm device-resident forward (energy)
        def dev_fwd():
            bb(xi, graph, se3)
            ttnn.synchronize_device(dev)

        dev_ms = time_it(dev_fwd, args.iters) * 1e3

        # end-to-end (host geom + upload + forward + readback)
        def e2e():
            tt = host_geom()
            g, s, x = upload(tt)
            _, en = bb(x, g, s)
            float(ttnn.to_torch(en).reshape(-1)[0])

        e2e_ms = time_it(e2e, max(3, args.iters // 4)) * 1e3

        # CPU reference (bit-exact mirror)
        def cpu_fwd():
            ne = mirror.backbone(w, cfg, t["x_init"], t["wigner"], t["wigner_inv"],
                                 t["x_edge"], t["edge_envelope"], se, ei, b.to_grid_mat, b.from_grid_mat)
            float(mirror.energy(ne, w))

        with torch.no_grad():
            cpu_ms = time_it(cpu_fwd, max(3, args.iters // 4)) * 1e3

        # honest accuracy: TT energy vs the CPU reference (same fp32 mirror)
        _, en_tt = bb(xi, graph, se3)
        e_tt = float(ttnn.to_torch(en_tt).reshape(-1)[0])
        with torch.no_grad():
            e_cpu = float(mirror.energy(mirror.backbone(
                w, cfg, t["x_init"], t["wigner"], t["wigner_inv"], t["x_edge"],
                t["edge_envelope"], se, ei, b.to_grid_mat, b.from_grid_mat), w))
        rel_err = abs(e_tt - e_cpu) / (abs(e_cpu) + 1e-6)

        medges = E / (dev_ms * 1e-3) / 1e6
        rows.append(dict(cells=nc, natoms=N, nedges=E, dev_ms=dev_ms, e2e_ms=e2e_ms,
                         cpu_ms=cpu_ms, medges_per_s=medges, speedup_dev=cpu_ms / dev_ms,
                         speedup_e2e=cpu_ms / e2e_ms, energy_rel_err=rel_err))
        print(f"N={N:4d} E={E:5d}  dev={dev_ms:7.2f}ms  e2e={e2e_ms:7.2f}ms  cpu={cpu_ms:7.2f}ms  "
              f"{medges:5.3f} Medges/s  dev x{cpu_ms/dev_ms:.2f}  e2e x{cpu_ms/e2e_ms:.2f}  "
              f"Erel={rel_err:.1e}")

    ttnn.close_device(dev)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("throughput_fast.json" if args.fast else "throughput.json")
    out.write_text(json.dumps(dict(config=cfg, fast=args.fast, rows=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
