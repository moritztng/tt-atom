"""Every op of tensornet.tt_ops.TtOps on one chip against the float64 backend, at bfloat16 and float32: relative
rms error per op. A float32 op that reads its operand through the matrix engine shows ~5e-4 instead of ~1e-7."""
import json

import torch

from tt_atom import device as D
from tt_atom.tensornet.layer import TorchOps as T
from tt_atom.tensornet.tt_ops import TtOps


def main():
    import ttnn

    n, d, u = 64, 32, 128
    g = torch.Generator().manual_seed(0)
    r = lambda *shape: torch.randn(*shape, generator=g).double()
    x, y, e27, e9, e3, w, rw = r(3, 3, n, u), r(3, 3, n, u), r(n * d, 27 * u), r(n * d, 9 * u), r(n * d, 3 * u), r(u, u), r(n, 27 * u)
    idx = torch.randint(0, n, (n * d,), generator=g)
    cases = {  # name -> (args; a tensor goes up, anything else is passed to both as is)
        "add": (x, y), "sub": (x, y), "mul": (x, y), "silu": (x,), "silu_grad": (x,), "sigmoid": (x,), "t": (x,), "csum": (x,),
        "mat": (x, y), "usum": (x,), "rows": ([x, y, x],), "unrows": (rw, 3), "expand": (e3,), "fold9": (e27,), "tile3": (e9,),
        "fold3": (e27,), "slot_sum": (e9, n), "slot_spread": (rw, d), "cat": ([e3, e9],), "split": (e9, 3), "gather": (idx, rw),
        "linear": (x, w), "scale": (x, 0.37)}
    out = {}
    with D.device(0) as dev:
        for dt in ("float32", "bfloat16"):
            o = TtOps(dev, n, d, u, dtype=getattr(ttnn, dt))
            lift = lambda a: ([lift(v) for v in a] if isinstance(a, list) else o.index(a) if torch.is_tensor(a) and not a.is_floating_point()
                              else o.up(a) if torch.is_tensor(a) else a)
            seen = lambda a: ([seen(v) for v in a] if isinstance(a, list) else ttnn.to_torch(o.up(a)).double() if torch.is_tensor(a) and a.is_floating_point() else a)
            for name, args in cases.items():
                try:
                    if name == "linear":
                        got, want = o.linear(o.up(x), o.weight(w[None, None])), T.linear(seen(x), w.float().double())
                    else:
                        got, want = getattr(o, name)(*[lift(a) for a in args]), getattr(T, name)(*[seen(a) for a in args])
                    got, want = (got if isinstance(got, list) else [got]), (want if isinstance(want, list) else [want])
                    out[f"{name}_{dt}"] = max(float((ttnn.to_torch(a).double().reshape(b.shape) - b).norm() / b.norm()) for a, b in zip(got, want))
                except Exception as err:
                    out[f"{name}_{dt}"] = f"{type(err).__name__}: {str(err)[:120]}"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
