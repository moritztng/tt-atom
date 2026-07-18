"""Per-input FD gradient check: is g_edge_vec_cat / g_cutoff / g_log_mask each the true grad of
the device forward energy wrt that input? Isolates which adjoint is wrong."""
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
    from tt_atom.device import open_device
    from tt_atom.pet_weights import PetWeights
    from tt_atom.pet_geometry import host_pet_geometry
    from tt_atom.pet_model import PetModel, build_device_inputs
    from tt_atom.pet_forces import device_energy_and_forces

    gw = PetWeights.load(WEIGHTS)
    fx = np.load(GOLDEN)
    pos = torch.tensor(fx["positions"], dtype=torch.float64)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)
    scale = gw.energy_scale()
    comp = gw.composition_energy_by_z()

    dev = open_device(0)
    try:
        model = PetModel(gw.weights, dev, cfg=gw.config)
        bd = host_pet_geometry(pos.detach().to(torch.float64), numbers, cell=cell, pbc=pbc, cfg=gw.config)
        bd_dev = build_device_inputs(bd, gw.config, dev)
        # run forward + backward to get adjoints
        raw = model.forward(bd_dev)
        g_evc_dev, g_cut_dev, g_lm_dev = model.backward(bd_dev, g_raw=1.0)
        g_evc = ttnn.to_torch(g_evc_dev).float().numpy()
        g_cut = ttnn.to_torch(g_cut_dev).float().numpy()
        g_lm = ttnn.to_torch(g_lm_dev).float().numpy()
        E0 = float(ttnn.to_torch(raw).float().view(-1)[0]) * scale + float(comp[numbers].sum())
        print(f"E_dev0 = {E0:.6f}")
        print(f"g_evc shape {g_evc.shape} sum {g_evc.sum():.4f}")
        print(f"g_cut shape {g_cut.shape} sum {g_cut.sum():.4f}")
        print(f"g_lm shape {g_lm.shape} sum {g_lm.sum():.4f}")

        # FD against each device input independently by perturbing the DEVICE tensor.
        def dev_energy_with(evc=None, cut=None, lm=None):
            bd2 = dict(bd_dev)
            if evc is not None:
                bd2 = dict(bd_dev)
                bd2["edge_vec_cat"] = ttnn.from_torch(
                    torch.tensor(evc, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev)
            if cut is not None:
                bd2 = dict(bd2)
                bd2["cutoff_factors"] = ttnn.from_torch(
                    torch.tensor(cut, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev)
            if lm is not None:
                bd2 = dict(bd2)
                bd2["log_mask"] = ttnn.from_torch(
                    torch.tensor(lm, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=dev)
            r = model.forward(bd2)
            return float(ttnn.to_torch(r).float().view(-1)[0]) * scale + float(comp[numbers].sum())

        evc0 = ttnn.to_torch(bd_dev["edge_vec_cat"]).float().numpy()
        cut0 = ttnn.to_torch(bd_dev["cutoff_factors"]).float().numpy()
        lm0 = ttnn.to_torch(bd_dev["log_mask"]).float().numpy()

        eps = 2.0
        # FD for edge_vec_cat (full N*Dmax*4 = small; sample a few entries)
        N, Dmax, _ = evc0.shape
        fd_evc = np.zeros_like(evc0)
        rng = np.random.default_rng(0)
        # check ALL entries (N*Dmax*4 small)
        for n in range(N):
            for d in range(Dmax):
                for k in range(4):
                    pp = evc0.copy(); pp[n,d,k] += eps
                    pm = evc0.copy(); pm[n,d,k] -= eps
                    fd_evc[n,d,k] = (dev_energy_with(evc=pp) - dev_energy_with(evc=pm)) / (2*eps)
        pcc_evc = _pcc(g_evc, fd_evc)
        print(f"[FD edge_vec_cat] PCC={pcc_evc:.4f} max|g|={np.abs(g_evc).max():.3f} max|fd|={np.abs(fd_evc).max():.3f}")

        # FD for cutoff_factors [N, Dmax, 1]
        fd_cut = np.zeros_like(cut0)
        for n in range(N):
            for d in range(Dmax):
                pp = cut0.copy(); pp[n,d,0] += eps
                pm = cut0.copy(); pm[n,d,0] -= eps
                fd_cut[n,d,0] = (dev_energy_with(cut=pp) - dev_energy_with(cut=pm)) / (2*eps)
        pcc_cut = _pcc(g_cut, fd_cut)
        print(f"[FD cutoff_factors] PCC={pcc_cut:.4f} max|g|={np.abs(g_cut).max():.3f} max|fd|={np.abs(fd_cut).max():.3f}")

        # FD for log_mask [N*heads, S, S] -- sample a few entries (S*S large)
        Nh, S, S2 = lm0.shape
        fd_lm = np.zeros_like(lm0)
        # sample 200 entries
        idxs = [(h, i, j) for h in range(Nh) for i in range(S) for j in range(S2)]
        sample = rng.choice(len(idxs), size=min(200, len(idxs)), replace=False)
        for s in sample:
            h, i, j = idxs[s]
            pp = lm0.copy(); pp[h,i,j] += eps
            pm = lm0.copy(); pm[h,i,j] -= eps
            fd_lm[h,i,j] = (dev_energy_with(lm=pp) - dev_energy_with(lm=pm)) / (2*eps)
        # compare only sampled
        mask = np.zeros_like(fd_lm, dtype=bool)
        for s in sample:
            h, i, j = idxs[s]; mask[h,i,j] = True
        pcc_lm = _pcc(g_lm[mask], fd_lm[mask])
        print(f"[FD log_mask] PCC={pcc_lm:.4f} (sampled {len(sample)}) max|g|={np.abs(g_lm[mask]).max():.3f} max|fd|={np.abs(fd_lm[mask]).max():.3f}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
