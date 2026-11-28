"""Is a bfloat16 x bfloat16 matmul with a float32 result exact on the card, and is a broadcast float32 add? Operands are
bfloat16-exact, positive (no cancellation) and signed; reference in float64."""
import json

import torch

from tt_atom import device as D


def main():
    import ttnn

    g, out, cfg = torch.Generator().manual_seed(0), {}, D.compute_kernel_config()
    bf = lambda t: t.to(torch.bfloat16).double()
    with D.device(0) as dev:
        up = lambda t, dt: ttnn.from_torch(t.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        rel = lambda got, want: float((ttnn.to_torch(got).double() - want).norm() / want.norm())
        for k in (3, 32, 128):
            for sign in ("pos", "signed"):
                x, w = torch.rand(6912, k, generator=g), torch.rand(k, 128, generator=g)
                x, w = (bf(x), bf(w)) if sign == "pos" else (bf(x - 0.5), bf(w - 0.5))
                y = ttnn.matmul(up(x, ttnn.bfloat16), up(w, ttnn.bfloat16), compute_kernel_config=cfg, dtype=ttnn.float32)
                out[f"mm_k{k}_{sign}"] = rel(y, x @ w)
        a, b = torch.randn(6912, 128, generator=g).float().double(), torch.randn(1, 128, generator=g).float().double()
        out["add_broadcast_f32"] = rel(ttnn.add(up(a, ttnn.float32), up(b, ttnn.float32)), a + b)
        out["mul_broadcast_f32"] = rel(ttnn.multiply(up(a, ttnn.float32), up(a[:, :1], ttnn.float32)), a * a[:, :1])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
