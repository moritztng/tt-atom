"""Isolated attention backward FD check: perturb attention input x (normed tokens), FD the attention
output, compare to _attention_bw's g_x. Isolates the manual-attention log-mask backward."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
import pathlib
import numpy as np
import torch

WEIGHTS = str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz")
GOLDEN = "tests/data/pet_mad_s_si_golden.npz"


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import ttnn
    from tt_atom.device import open_device, compute_kernel_config
    from tt_atom.pet_weights import PetWeights
    from tt_atom.pet_geometry import host_pet_geometry
    from tt_atom.pet_model import PetModel, build_device_inputs
    from tt_atom.pet_vjp import _attention_bw

    gw = PetWeights.load(WEIGHTS)
    fx = np.load(GOLDEN)
    pos = torch.tensor(fx["positions"], dtype=torch.float64)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)

    dev = open_device(0)
    try:
        model = PetModel(gw.weights, dev, cfg=gw.config)
        bd = host_pet_geometry(pos.detach().to(torch.float64), numbers, cell=cell, pbc=pbc, cfg=gw.config)
        bd_dev = build_device_inputs(bd, gw.config, dev)
        # run forward to populate caches
        model.forward(bd_dev)
        # use layer 0's attention
        layer0 = model.layers[0]
        attn = layer0.attention
        log_mask = bd_dev["log_mask"]
        N = bd_dev["N"]
        Dmax = bd_dev["Dmax"]
        S = 1 + Dmax
        d_pet = gw.config["d_pet"]
        # build a normed-tokens input (the attention input) -- reuse the cached one:
        # tokens = concat([input_node_tok, edge_tokens]); normed = rmsnorm(tokens)
        # We can reconstruct normed by re-running layer0 up to the normed point, but
        # simpler: just call attention with a fresh x of the right shape [N, S, d_pet].
        x0_np = np.random.default_rng(2).normal(0, 0.5, (N, S, d_pet)).astype(np.float32)

        def attn_energy(x_np):
            x_t = ttnn.from_torch(torch.tensor(x_np, dtype=torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            out = attn(x_t, log_mask)
            return float(ttnn.to_torch(out).float().sum())

        # adjoint via _attention_bw with g_new_tokens = ones
        g_new = ttnn.ones((N, S, d_pet), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        g_x_dev, g_lm_dev = _attention_bw(attn, g_new, log_mask)
        g_x_np = ttnn.to_torch(g_x_dev).float().numpy()
        g_lm_np = ttnn.to_torch(g_lm_dev).float().numpy()

        eps = 0.5
        # FD g_x (sample 100 entries)
        fd_x = np.zeros_like(x0_np)
        rng = np.random.default_rng(3)
        idxs = rng.choice(N * S * d_pet, size=min(150, N * S * d_pet), replace=False)
        for s in idxs:
            n, k = divmod(s, S * d_pet)
            i, k = divmod(k, d_pet)
            pp = x0_np.copy(); pp[n, i, k] += eps
            pm = x0_np.copy(); pm[n, i, k] -= eps
            fd_x[n, i, k] = (attn_energy(pp) - attn_energy(pm)) / (2 * eps)
        mask = np.zeros_like(fd_x, dtype=bool)
        for s in idxs:
            n, k = divmod(s, S * d_pet); i, k = divmod(k, d_pet); mask[n, i, k] = True
        print(f"[FD attn g_x] PCC={_pcc(g_x_np[mask], fd_x[mask]):.4f} "
              f"max|g|={np.abs(g_x_np[mask]).max():.3f} max|fd|={np.abs(fd_x[mask]).max():.3f}")

        # FD g_lm (sample 150 entries of [N*heads, S, S])
        lm0 = ttnn.to_torch(log_mask).float().numpy()
        Nh, S2, S3 = g_lm_np.shape
        idxs = rng.choice(Nh * S2 * S3, size=min(150, Nh * S2 * S3), replace=False)
        fd_lm = np.zeros_like(g_lm_np)
        x_t = ttnn.from_torch(torch.tensor(x0_np, dtype=torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for s in idxs:
            h, i, j = np.unravel_index(s, (Nh, S2, S3))
            pp = lm0.copy(); pp[h, i, j] += eps
            pm = lm0.copy(); pm[h, i, j] -= eps
            lm_p = ttnn.from_torch(torch.tensor(pp, dtype=torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            lm_m = ttnn.from_torch(torch.tensor(pm, dtype=torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            out_p = attn(x_t, lm_p); out_m = attn(x_t, lm_m)
            fd_lm[h, i, j] = (float(ttnn.to_torch(out_p).float().sum()) - float(ttnn.to_torch(out_m).float().sum())) / (2 * eps)
        mask = np.zeros_like(fd_lm, dtype=bool)
        for s in idxs:
            h, i, j = np.unravel_index(s, (Nh, S2, S3)); mask[h, i, j] = True
        print(f"[FD attn g_lm] PCC={_pcc(g_lm_np[mask], fd_lm[mask]):.4f} "
              f"max|g|={np.abs(g_lm_np[mask]).max():.3f} max|fd|={np.abs(fd_lm[mask]).max():.3f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
