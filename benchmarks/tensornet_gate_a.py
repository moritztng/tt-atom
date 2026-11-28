"""Forces of TensorNet-MatPES with interaction layers run on a chip, graded against the float64 model.

matgl runs here in float64; the chosen interaction layers (forward and reverse pass) run on the device behind
tensornet_layer_server.py, reached through the command in --server (a local python or an ssh to the card's host).

    python benchmarks/tensornet_gate_a.py --layers 0 --dtype bfloat16 --server "ssh host run_server.sh 11" --out a.json
"""
import argparse
import io
import json
import shlex
import struct
import subprocess

import numpy as np
import torch

from census_tensornet import build
from tt_atom.tensornet import layer as L


class Remote:
    def __init__(self, cmd):
        self.p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def call(self, **msg):
        buf = io.BytesIO()
        torch.save(msg, buf)
        self.p.stdin.write(struct.pack("<Q", buf.getbuffer().nbytes) + buf.getvalue())
        self.p.stdin.flush()
        head = self.p.stdout.read(8)
        if len(head) < 8:
            raise RuntimeError(f"server ended during {msg['op']} (rc {self.p.wait()})")
        return torch.load(io.BytesIO(self.p.stdout.read(struct.unpack("<Q", head)[0])), weights_only=True)


class DeviceLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rbf, cut, remote, key, edge):
        ctx.remote, ctx.key, ctx.edge, ctx.e = remote, key, edge, rbf.shape[0]
        pad = lambda t: torch.cat([t, torch.zeros_like(t[:1])])[edge]
        out = remote.call(op="fwd", key=key, x=x.permute(2, 3, 0, 1), rbf=pad(rbf), cut=pad(cut))["out"]
        return out.permute(2, 3, 0, 1).to(x.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        r = ctx.remote.call(op="bwd", key=ctx.key, g=g.permute(2, 3, 0, 1))
        real = ctx.edge < ctx.e
        unpad = lambda t: torch.zeros(ctx.e, *t.shape[1:], dtype=g.dtype).index_copy(0, ctx.edge[real], t[real].to(g.dtype))
        return r["gx"].permute(2, 3, 0, 1).to(g.dtype), unpad(r["grbf"]), unpad(r["gcut"]).sum(-1, keepdim=True), None, None, None


def on_device(mod, remote, key, dtype, min_width):
    """Replace a matgl TensorNetInteraction's forward with the device layer."""
    state = {}

    def forward(*args, **kw):
        a = dict(zip(("edge_index", "edge_weight", "edge_attr", "X"), args), **kw)
        ei, r, rbf, x = a["edge_index"], a["edge_weight"].reshape(-1), a["edge_attr"], a["X"]
        if "edge" not in state:
            state["edge"], send, table, d = L.slots(ei, x.shape[0], min_width=min_width)
            remote.call(op="setup", key=key, w=L.weights(mod), n=x.shape[0], d=d, units=mod.units, send=send, table=table,
                        dtype=dtype)
        cut = torch.where(r <= mod.cutoff, 0.5 * (torch.cos(torch.pi * r / mod.cutoff) + 1), 0.0)[:, None]
        return DeviceLayer.apply(x, rbf, cut, remote, key, state["edge"])

    mod.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="diamond Si supercell, 8 n^3 atoms")
    ap.add_argument("--model", default="materialyze/TensorNet-PES-MatPES-PBE-2025.2")
    ap.add_argument("--layers", default="0", help="comma list of interaction layers to run on the device")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--min-width", type=int, default=32, help="floor of the padded degree")
    ap.add_argument("--server", default="python benchmarks/tensornet_layer_server.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    import matgl
    from matgl.ext.ase import PESCalculator

    pot = matgl.load_model(args.model).double()
    pot.calc_stresses = False
    atoms = build("si", args.n)
    atoms.calc = PESCalculator(pot)
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces().copy()

    remote = Remote(args.server)
    for k in [int(s) for s in args.layers.split(",")]:
        on_device(pot.model.layers[k], remote, k, args.dtype, args.min_width)
    atoms.calc = PESCalculator(pot)
    e1, f1 = atoms.get_potential_energy(), atoms.get_forces().copy()
    res = {"model": args.model, "atoms": len(atoms), "layers_on_device": args.layers, "dtype": args.dtype,
           "dE_meV_per_atom": 1e3 * abs(e1 - e0) / len(atoms), "force_mae_meV_A": 1e3 * float(np.abs(f1 - f0).mean()),
           "force_max_meV_A": 1e3 * float(np.abs(f1 - f0).max()), "mean_abs_force_eV_A": float(np.abs(f0).mean()),
           **remote.call(op="close")}
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
