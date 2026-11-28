"""One TensorNet interaction layer on a chip, served over stdin/stdout to tensornet_gate_a.py.

Messages are length-prefixed torch.save blobs. `setup` builds a layer, `fwd` runs it and keeps the tape, `bwd`
runs the reverse pass on that tape, `close` returns the AICLK seen while the device was open.
"""
import io
import os
import struct
import sys
import time

import torch

PIPE = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)  # device logs go to stderr, the pipe carries messages only


def recv():
    head = sys.stdin.buffer.read(8)
    if len(head) < 8:
        return None
    return torch.load(io.BytesIO(sys.stdin.buffer.read(struct.unpack("<Q", head)[0])), weights_only=True)


def send(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    PIPE.write(struct.pack("<Q", buf.getbuffer().nbytes) + buf.getvalue())
    PIPE.flush()


def backend(cpu):
    """(ttnn, device context, clock sampler). --cpu swaps ttnn for the float64 stand-in of the tests: a dry run of
    the bridge and of the op bookkeeping, no card."""
    if not cpu:
        import ttnn

        from bench_orb_perf_dollar_tt import AiclkSampler
        from tt_atom import device as D

        return ttnn, D.device(0), AiclkSampler()
    import contextlib
    import types

    from tests.test_tensornet_layer import FakeTtnn

    FakeTtnn.float32, FakeTtnn.to_torch, FakeTtnn.typecast = torch.float64, staticmethod(lambda t: t), staticmethod(lambda t, dtype: t)
    return FakeTtnn, contextlib.nullcontext(), contextlib.nullcontext(types.SimpleNamespace(summary=dict))


def main():
    from tt_atom.tensornet import layer as L
    from tt_atom.tensornet import model as M
    from tt_atom.tensornet.tt_ops import TtOps as Ops, model_to_device, to_device

    ttnn, device, sampler = backend("--cpu" in sys.argv)
    TtOps = lambda *a, **k: Ops(*a, ttnn=ttnn, **k)
    torch.set_num_threads(1)
    layers = {}
    down = lambda t: ttnn.to_torch(t).double()
    flat = lambda tape: {f"{k}{i}": down(t) for k, v in tape.items() for i, t in enumerate(v if isinstance(v, (list, tuple)) else [v])
                         if not isinstance(t, (int, float, dict))}
    with device as dev, sampler as clk:
        nodes = sorted({os.readlink(f"/proc/self/fd/{f}").rsplit("/", 1)[1] for f in os.listdir("/proc/self/fd")
                        if os.path.exists(f"/proc/self/fd/{f}") and os.readlink(f"/proc/self/fd/{f}").startswith("/dev/tenstorrent/")})
        while (m := recv()) is not None:
            if m["op"] == "setup":
                n, d, u = m["n"], m["d"], m["units"]
                o = TtOps(dev, n, d, u, dtype=getattr(ttnn, m["dtype"]))
                layers[m["key"]] = {"o": o, "w": to_device(o, m["w"]), "send": o.index(m["send"]), "table": o.index(m["table"]),
                                    "eye": o.up(torch.eye(3).reshape(3, 3, 1, 1).expand(3, 3, n, u)), "n": n, "d": d}
                send({"ok": True})
            elif m["op"] == "fwd":
                s = layers[m["key"]]
                o = s["o"]
                out, s["tape"] = L.forward(o, s["w"], o.up(m["x"]), o.up(m["rbf"]), o.up(m["cut"]), s["send"], s["n"], s["eye"], 1.0)
                send({"out": down(out)})
            elif m["op"] == "bwd":
                s = layers[m["key"]]
                o = s["o"]
                gx, grbf, gcut = L.backward(o, s["w"], s["tape"], o.up(m["g"]), s["table"], s["n"], s["d"], s["eye"])
                send({"gx": down(gx), "grbf": down(grbf), "gcut": down(gcut)})
            elif m["op"] == "model":
                n, d, u = m["n"], m["d"], m["units"]
                o, ol = TtOps(dev, n, d, u, dtype=ttnn.float32), TtOps(dev, n, d, u, dtype=getattr(ttnn, m["dtype"]))
                eye = torch.eye(3).reshape(3, 3, 1, 1).expand(3, 3, n, u)
                layers["model"] = {"o": o, "ol": ol, "w": model_to_device(o, ol, m["w"]), "send": o.index(m["send"]),
                                   "table": o.index(m["table"]), "eye": o.up(eye), "eye_l": ol.up(eye), "n": n, "d": d}
                send({"ok": True})
            elif m["op"] == "energy":
                s = layers["model"]
                o, ol, n = s["o"], s["ol"], s["n"]
                t0, stages = time.perf_counter(), [] if m.get("stages") else None
                geo = [o.up(c) for c in m["geo"].split(1, -1)]
                e, tape = M.forward(o, s["w"], o.up(m["z"]), (o.up(m["rbf"]), o.up(m["cut"])), geo, s["send"], n, s["eye"],
                                    ol=ol, edge_l=(ol.up(m["rbf"]), ol.up(m["cut"])), eye_l=s["eye_l"], stages=stages)
                grbf, gcut, ggeo = M.backward(o, s["w"], tape, o.up(torch.ones(1, 1, n, 1)), s["table"], n, s["d"], s["eye"],
                                              ol=ol, eye_l=s["eye_l"])
                out = {"e": down(e).reshape(-1), "grbf": sum(down(g) for g in grbf), "gcut": sum(down(g) for g in gcut),
                       "ggeo": down(ggeo), **({"stages": [down(x) for x in stages], "tape": flat(tape[0])} if stages else {})}
                send({**out, "seconds": time.perf_counter() - t0})
            elif m["op"] == "close":
                send({"aiclk_mhz_during": {k: v for k, v in clk.summary().items() if k in nodes}, "opened_nodes": nodes,
                      "host_load1": os.getloadavg()[0]})
                break


if __name__ == "__main__":
    main()
