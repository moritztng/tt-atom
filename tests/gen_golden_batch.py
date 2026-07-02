"""Generate a REAL-weight BATCHED golden for disjoint-union batching parity (refenv only).

Run in the reference environment (fairchem-core, numpy>=2), NOT the ttnn env:

    HF_HUB_OFFLINE=1 ~/.ttatom_run/refenv/bin/python tests/gen_golden_batch.py \
        --k 8 --task omol --out ~/.ttatom_run/goldens_real/batch_ethanol_omol.npz

Builds K same-reduced-composition conformers (ethanol, rattled) — the regime a merged uma-s-1
bundle is valid for (fairchem's ``merge_MOLE_model`` asserts one reduced composition per batch).
It then runs fairchem's OWN batched merged inference (``data_list_collater`` of K ``AtomicData``
-> one ``predict`` call, per-system energies split by ``data.batch``) and cross-checks that
against the per-conformer ``FAIRChemCalculator`` — proving fairchem itself batches block-
diagonally — before dumping the per-system energies + forces as the golden.

The ttnn env (test_realweight.py) then loads this, assembles the same conformers into one
block-diagonal graph, runs ``energy_and_forces_batch``, and asserts E rel<1e-3, F PCC>0.99.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from ase.build import molecule
from huggingface_hub import hf_hub_download
from fairchem.core import FAIRChemCalculator
from fairchem.core.datasets import data_list_collater
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def conformers(k, seed0=10):
    """K rattled ethanol conformers — identical composition, different geometry."""
    out = []
    for i in range(k):
        a = molecule("CH3CH2OH")
        a.rattle(stdev=0.08, seed=seed0 + i)
        a.info.update(charge=0, spin=1)
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--task", default="omol")
    ap.add_argument("--ckpt", default="uma-s-1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = hf_hub_download("facebook/UMA", f"checkpoints/{args.ckpt}.pt")
    settings = InferenceSettings(
        tf32=False, activation_checkpointing=True, merge_mole=True,
        compile=False, external_graph_gen=False, internal_graph_gen_version=2,
    )
    pu = load_predict_unit(ckpt, inference_settings=settings, device="cpu")
    calc = FAIRChemCalculator(pu, task_name=args.task)

    systems = conformers(args.k)

    # ---- per-conformer reference (fairchem's validated single-system inference) -----------
    E_single, F_single = [], []
    for a in systems:
        a.calc = calc
        E_single.append(float(a.get_potential_energy()))
        F_single.append(a.get_forces().astype(np.float32))
    E_single = np.array(E_single, dtype=np.float64)

    # ---- fairchem's OWN batched merged inference (Batch.from_data_list equivalent) --------
    # build each AtomicData exactly as the calculator does (task_name + r_data_keys carry the
    # charge/spin the merged model checks), then collate K of them into one batch.
    data = []
    for a in systems:
        calc.predictor.validate_atoms_data(a, args.task)
        data.append(calc.a2g(a))
    batch = data_list_collater(data, otf_graph=True)
    pred = pu.predict(batch)
    Ebt = pred["energy"].detach().cpu().numpy().astype(np.float64).reshape(-1)
    Fbt_all = pred["forces"].detach().cpu().numpy().astype(np.float32)
    bidx = batch.batch.detach().cpu().numpy()
    F_batched = [Fbt_all[bidx == i] for i in range(args.k)]

    # cross-check: fairchem batched == fairchem per-system (block-diagonal, no cross terms)
    e_rel = np.abs(Ebt - E_single).max() / (np.abs(E_single).max() + 1e-6)
    f_max = max(np.abs(F_batched[i] - F_single[i]).max() for i in range(args.k))
    print(f"fairchem batched vs single: E rel={e_rel:.2e}  F maxdiff={f_max:.2e}")
    assert e_rel < 1e-4 and f_max < 1e-3, "fairchem batched != fairchem single — investigate"

    # ---- dump the golden (positions + fairchem batched E/F per system) --------------------
    saved = {
        "k": np.array([args.k]),
        "task": np.frombuffer(args.task.encode(), dtype=np.uint8),
        "charge": np.array([0.0]),
        "spin": np.array([1.0]),
        "natoms": np.array([len(a) for a in systems], dtype=np.int64),
        "Z": np.concatenate([a.get_atomic_numbers() for a in systems]).astype(np.int64),
        "pos": np.concatenate([a.get_positions() for a in systems]).astype(np.float64),
        "energy": Ebt,                                          # [K] fairchem batched energies
        "forces": np.concatenate(F_batched).astype(np.float32),  # [Ntot,3]
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out}: K={args.k} |E|={np.abs(Ebt).mean():.2f} eV  "
          f"|F|max={np.abs(Fbt_all).max():.3f}")


if __name__ == "__main__":
    main()
