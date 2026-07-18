"""Definitive pass-6 check -- NO interleaving (avoids ttnn-trace-interleaved-eager-
corruption). Block 1: run eager device_energy_and_forces for pos0 + 6 tiny rattles, store
(E, F). Block 2: build PetTracedEngine, capture on pos0, replay all 7 positions, store
(E, F). Compare offline: max|dE|, max|dF|, traced PCC vs golden. Also report warm full-
path timing for both (eager block timed, then traced block timed)."""
import os
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
import pathlib, time, numpy as np, torch

WEIGHTS = str(pathlib.Path.home() / ".cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz")
GOLDEN = "tests/data/pet_mad_s_si_golden.npz"
RATTLE_AMP = 0.001

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
    from tt_atom.pet_model import PetModel
    from tt_atom.pet_forces import device_energy_and_forces
    from tt_atom.pet_trace import PetTracedEngine

    gw = PetWeights.load(WEIGHTS)
    fx = np.load(GOLDEN)
    pos0 = torch.tensor(fx["positions"], dtype=torch.float64)
    numbers = torch.tensor(fx["numbers"], dtype=torch.long)
    cell = torch.tensor(fx["cell"], dtype=torch.float64)
    pbc = torch.tensor(fx["pbc"], dtype=torch.bool)
    ref_f = fx["forces"]
    scale = gw.energy_scale()
    comp = gw.composition_energy_by_z()
    gen = torch.Generator().manual_seed(1234)
    rattles = [pos0 + RATTLE_AMP * torch.randn(pos0.shape, generator=gen) for _ in range(6)]
    eval_pos = [pos0] + rattles

    eager_out = []
    dev = open_device(0, trace_region_size=400_000_000)
    try:
        em = PetModel(gw.weights, dev, cfg=gw.config)
        # ---- BLOCK 1: eager (no trace engine alive) ----
        for _ in range(5):
            device_energy_and_forces(pos0, numbers, gw.weights, cfg=gw.config, cell=cell, pbc=pbc, device=dev, model=em)
        n = 20
        t0 = time.perf_counter()
        for _ in range(n):
            device_energy_and_forces(pos0, numbers, gw.weights, cfg=gw.config, cell=cell, pbc=pbc, device=dev, model=em)
        eager_full_ms = (time.perf_counter() - t0) / n * 1e3
        for p in eval_pos:
            raw, F = device_energy_and_forces(p, numbers, gw.weights, cfg=gw.config, cell=cell, pbc=pbc, device=dev, model=em)
            eager_out.append((raw, F.clone()))

        # ---- BLOCK 2: traced (no further eager ops interleave) ----
        eng = PetTracedEngine(gw.weights, dev, cfg=gw.config, atomic_numbers=numbers, cell=cell, pbc=pbc)
        eng(pos0)  # capture + first replay
        for _ in range(5):
            eng(pos0)
        t0 = time.perf_counter()
        for _ in range(n):
            eng(pos0)
        traced_full_ms = (time.perf_counter() - t0) / n * 1e3
        traced_out = []
        for p in eval_pos:
            raw, F = eng(p)
            traced_out.append((raw, F.clone()))
        eng.close()
    finally:
        ttnn.close_device(dev)

    # ---- offline compare ----
    max_abs_E = 0.0
    max_abs_F = 0.0
    for (re, Fe), (rt, Ft) in zip(eager_out, traced_out):
        max_abs_E = max(max_abs_E, abs(rt - re))
        max_abs_F = max(max_abs_F, float((Ft - Fe).abs().max()))
    bit_exact = (max_abs_E == 0.0 and max_abs_F == 0.0)
    # traced PCC vs golden at pos0
    F0 = (traced_out[0][1] * scale).double()
    pcc_t = _pcc(F0.numpy(), ref_f)
    E_t = traced_out[0][0] * scale + float(comp[numbers].sum())
    print(f"[bit-exact vs eager] max |dE|={max_abs_E:.3e}  max |dF|={max_abs_F:.3e}  -> {bit_exact}")
    print(f"[traced] E={E_t:.6f} eV (ref {float(fx['energy'][0]):.6f})")
    print(f"[traced] forces PCC vs golden={pcc_t:.8f} max abs={float((F0-torch.tensor(ref_f)).abs().max()):.3e}")
    print(f"[timing] eager full={eager_full_ms:.3f} ms  traced full={traced_full_ms:.3f} ms  speedup={eager_full_ms/traced_full_ms:.3f}x")
    print(f"\nGATE bit-exact vs eager: {bit_exact}")
    print(f"GATE traced PCC vs golden ~0.98990: {abs(pcc_t - 0.98990027) < 1e-4}")
    print(f"GATE traced not slower than eager: {traced_full_ms <= eager_full_ms * 1.05}")
    return bit_exact and abs(pcc_t - 0.98990027) < 1e-4

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
