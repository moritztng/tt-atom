"""How many mantissa bits survive the inner-product accumulation of ttnn.matmul (policy config, float32 result)?
Operands carry m mantissa bits, either all in [1, 2) (one exponent) or spread over 2^-4..2^4; reference in float64."""
import json

import torch

from tt_atom import device as D


def main():
    import ttnn

    g, out, cfg = torch.Generator().manual_seed(0), {}, D.compute_kernel_config()

    def quant(shape, m, spread):
        v = 1 + torch.randint(0, 2 ** (m - 1), shape, generator=g).double() / 2 ** (m - 1)      # m bits, in [1, 2)
        return v * 2.0 ** torch.randint(-4, 5, shape, generator=g) if spread else v

    with D.device(0) as dev:
        up = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for k in (2, 3, 32, 128):
            for m in (1, 2, 4, 6, 8):
                for spread in (False, True):
                    x, w = quant((2048, k), m, spread), quant((k, 128), m, spread)
                    err = ttnn.to_torch(ttnn.matmul(up(x), up(w), compute_kernel_config=cfg, dtype=ttnn.float32)).double() - x @ w
                    out[f"k{k}_m{m}_{'spread' if spread else 'flat'}"] = [float(err.norm() / (x @ w).norm()), float((err / (x @ w)).mean())]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
