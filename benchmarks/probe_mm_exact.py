"""Is a bfloat16 x bfloat16 matmul with a float32 result exact on the card? Operands are bfloat16-exact; reference in
float64. K = 1 is a single product per output (no accumulation). Reports relative rms and the mean signed relative
error (a bias) per kernel config, plus broadcast float32 add and mul."""
import json

import torch

from tt_atom import device as D


def main():
    import ttnn

    g, out = torch.Generator().manual_seed(0), {}
    bf = lambda t: t.to(torch.bfloat16).double()
    F = ttnn.MathFidelity
    cfgs = {"policy": D.compute_kernel_config(),
            "hifi4_noacc": ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False),
            "hifi4_acc_nol1": ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False),
            "hifi3_acc": ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True),
            "hifi2_acc": ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)}
    with D.device(0) as dev:
        up = lambda t, dt: ttnn.from_torch(t.float(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

        def grade(got, want):
            err = ttnn.to_torch(got).double() - want
            return {"rel_rms": float(err.norm() / want.norm()), "bias": float((err / want.abs().clamp_min(1e-30)).mean())}

        for k in (1, 3, 128):
            x, w = bf(torch.rand(6912, k, generator=g) + 0.5), bf(torch.rand(k, 128, generator=g) + 0.5)
            for name, cfg in cfgs.items():
                for dt in ("bfloat16", "float32"):   # operand dtype on the card; values are bfloat16-exact either way
                    try:
                        y = ttnn.matmul(up(x, getattr(ttnn, dt)), up(w, getattr(ttnn, dt)), compute_kernel_config=cfg, dtype=ttnn.float32)
                        out[f"mm_k{k}_{name}_{dt}"] = grade(y, x @ w)
                    except Exception as e:
                        out[f"mm_k{k}_{name}_{dt}"] = str(e)[:80]
        x, w = bf(torch.rand(6912, 128, generator=g) + 0.5), bf(torch.rand(1, 128, generator=g) + 0.5)
        out["mul_bf16_values_f32"] = grade(ttnn.multiply(up(x, ttnn.float32), up(w, ttnn.float32)), x * w)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
