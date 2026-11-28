"""Signed error of the unary ttnn ops TensorNet uses (silu, sigmoid, rsqrt, reciprocal) against torch float64, at
bfloat16 and float32, on one chip. A biased activation shows up as a constant energy offset."""
import json

import torch

from tt_atom import device as D


def main():
    import ttnn

    g = torch.Generator().manual_seed(0)
    wide, pos = 12 * torch.rand(64, 128, generator=g, dtype=torch.float64) - 6, 10 ** (4 * torch.rand(64, 128, generator=g, dtype=torch.float64) - 3)
    cases = {"silu": (wide, torch.nn.functional.silu), "sigmoid": (wide, torch.sigmoid), "rsqrt": (pos, torch.rsqrt), "reciprocal": (pos, torch.reciprocal)}
    out = {}
    with D.device(0) as dev:
        for dt in ("bfloat16", "float32"):
            for name, (x, ref) in cases.items():
                xt = ttnn.from_torch(x.float(), dtype=getattr(ttnn, dt), layout=ttnn.TILE_LAYOUT, device=dev)
                xin = ttnn.to_torch(xt).double()                      # what the card was given
                err = ttnn.to_torch(getattr(ttnn, name)(xt)).double() - ref(xin)
                rel = err / ref(xin).abs().clamp_min(1e-3)
                out[f"{name}_{dt}"] = {"mean_signed": float(err.mean()), "max_abs": float(err.abs().max()), "mean_rel_signed": float(rel.mean()), "max_rel": float(rel.abs().max())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
