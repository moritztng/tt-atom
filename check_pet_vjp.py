"""Standalone check for the pass-5 device VJP: PCC vs golden forces + finite-difference
self-consistency (the device VJP must be the gradient of the device forward energy)."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
import pathlib
import numpy as np
import torch

WEIGHTS = str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz")
GOLDEN = "tests/data/pet_mad_s_si_golden.npz"
FIXTURE = "tests/data/pet_mad_s_si_canon_internals.npz"


def _pcc(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 and b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    from tt_atom.device import open_device
    from tt_atom.pet_weights import PetWeights
    from tt_atom.pet_geometry import host_pet_geometry
    from tt_atom.pet_model import PetModel, build_device_inputs
    from tt_atom.pet_forces import device_energy_and_forces, host_energy_and_forces

    gw = PetWeights.load(WEIGHTS)
    fx = np.load(GOLDEN)
    pos = torch.tensor(fx["positions"], dtype=torch.float64)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)
    ref_f = fx["forces"]
    scale = gw.energy_scale()
    comp = gw.composition_energy_by_z()

    dev = open_device(0)
    try:
        model = PetModel(gw.weights, dev, cfg=gw.config)

        # --- device VJP forces ---
        raw_dev, F_dev = device_energy_and_forces(
            pos, numbers, gw.weights, cfg=gw.config, cell=cell, pbc=pbc,
            device=dev, model=model)
        F_dev_full = (F_dev * scale).double()
        pcc_dev = _pcc(F_dev_full, ref_f)
        maxabs_dev = float((F_dev_full - torch.tensor(ref_f)).abs().max())
        E_dev = raw_dev * scale + float(comp[numbers].sum())
        print(f"[device-vjp] E_dev={E_dev:.6f} eV (ref {float(fx['energy'][0]):.6f})")
        print(f"[device-vjp] forces PCC={pcc_dev:.8f} max abs={maxabs_dev:.3e} "
              f"(ref max abs {np.abs(ref_f).max():.3f})")

        # --- host route (pass 4) for comparison ---
        raw_host, F_host = host_energy_and_forces(
            pos, numbers, gw.weights, cfg=gw.config, cell=cell, pbc=pbc)
        F_host_full = (F_host * scale).double()
        pcc_host = _pcc(F_host_full, ref_f)
        maxabs_host = float((F_host_full - torch.tensor(ref_f)).abs().max())
        print(f"[host-route] forces PCC={pcc_host:.8f} max abs={maxabs_host:.3e}")

        # --- self-consistency: finite-difference gradient of the DEVICE energy ---
        # F = -dE_dev/dpos. Check via central FD on the device forward energy.
        pos32 = pos.double().clone()
        eps = 2.5e-1  # bohr; large enough to dominate bf16 energy quantization (~0.4 eV
                      # near -100 eV). Self-consistency PCC rises with eps (0.93/0.97/0.99/0.99
                      # at eps=0.05/0.1/0.25/0.5), confirming the VJP is the device-energy
                      # gradient; the residual is FD nonlinearity + bf16 quantization.
        fd = torch.zeros_like(pos32)
        # use the device forward (energy only) at perturbed positions
        def dev_energy(p):
            bd = host_pet_geometry(p.detach().to(torch.float64), numbers,
                                    cell=cell, pbc=pbc, cfg=gw.config)
            bd_dev = build_device_inputs(bd, gw.config, dev)
            r = model.forward(bd_dev)
            import ttnn
            return float(ttnn.to_torch(r).float().view(-1)[0]) * scale + float(comp[numbers].sum())
        # central difference on a few atoms/dims (full 16x3 = 48 evals is fine)
        for i in range(pos32.shape[0]):
            for j in range(3):
                pp = pos32.clone(); pp[i, j] += eps
                pm = pos32.clone(); pm[i, j] -= eps
                fd[i, j] = -(dev_energy(pp) - dev_energy(pm)) / (2 * eps)
        pcc_fd = _pcc(F_dev_full, fd)
        maxabs_fd = float((F_dev_full - fd).abs().max())
        print(f"[device-vjp self-consistency] vs FD(-dE_dev/dpos): "
              f"PCC={pcc_fd:.6f} max abs={maxabs_fd:.3e} (eps={eps})")

        # gate
        ok = pcc_dev > 0.999
        print(f"\nGATE pcc_dev>0.999: {ok}")
        return ok
    finally:
        import ttnn
        ttnn.close_device(dev)


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
