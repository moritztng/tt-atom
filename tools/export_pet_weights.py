"""Export a TT-Atom PET-MAD (UPET) weight bundle from a real checkpoint (refenv only).

PET-MAD (``lab-cosmo/upet``, ``pet-mad-s`` v1.5.0) bakes no MoLE / per-composition
routing into its weights — the raw checkpoint is valid for *any* composition — so this
is a plain one-time per-checkpoint export (no structure, no forward pass), read back by
``tt_atom.pet_weights.PetWeights`` in the ttnn env. Same shape of tool as
``tools/export_orb_weights_cache.py``'s exporter; the only PET-specific pieces are the
checkpoint-upgrade step (the public v1.5.0 ckpt is model-version 11 and must be upgraded
to the installed metatrain's version) and the extraction of the two post-processing
artefacts the device path needs: the per-element composition reference energy and the
single energy scaler.

The LLPR uncertainty wrapper (128-member ensemble) is skipped — it is UQ-only and not
needed for plain energy/forces.

Run with the reference env (``~/.ttatom_run/upetenv``, which has ``upet==0.2.6`` +
``metatrain==2026.3.1`` installed alongside the shared refenv torch/numpy via a ``.pth``
reuse — no numpy-2 conflict)::

    ~/.ttatom_run/upetenv/bin/python tools/export_pet_weights.py \
        --ckpt pet-mad-s-v1.5.0 --out ~/.cache/tt_atom/pet_weights/pet-mad-s-v1.5.0.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

# these import the metatomic C++ extensions so torch.load on the ckpt works
import metatomic.torch  # noqa: F401
import metatensor.torch  # noqa: F401
from huggingface_hub import hf_hub_download

from metatrain.pet import PET

# repo id + filename for each public PET-MAD checkpoint we support. The model name IS
# the checkpoint (no per-composition bundle), mirroring the Orb family.
CHECKPOINTS = {
    "pet-mad-s-v1.5.0": ("lab-cosmo/upet", "models/pet-mad-s-v1.5.0.ckpt"),
}


def npy(t):
    return t.detach().to(torch.float64).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CHECKPOINTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_id, filename = CHECKPOINTS[args.ckpt]
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # The public ckpt wraps the base PET in an LLPR uncertainty model. The base PET is
    # in `wrapped_model_checkpoint`; the outer `model_data.hypers` only carries the
    # LLPR ensemble size, which we don't need.
    wmc = ckpt["wrapped_model_checkpoint"]
    # The wrapped ckpt is model-version 11; the installed metatrain expects v14. Run
    # the registered upgrade chain before load.
    wmc = PET.upgrade_checkpoint(wmc)
    hypers = wmc["model_data"]["model_hypers"]
    dataset_info = wmc["model_data"]["dataset_info"]

    model = PET.load_checkpoint(wmc, "export")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 25_924_122, f"unexpected base-PET param count {n_params}"

    cfg = dict(
        checkpoint=args.ckpt,
        cutoff=float(hypers["cutoff"]),
        cutoff_width=float(hypers["cutoff_width"]),
        cutoff_function=str(hypers["cutoff_function"]),
        cutoff_width_adaptive=float(hypers["cutoff_width_adaptive"]),
        adaptive_cutoff_method=str(hypers["adaptive_cutoff_method"]),
        num_neighbors_adaptive=float(hypers["num_neighbors_adaptive"]),
        d_pet=int(hypers["d_pet"]),
        d_node=int(hypers["d_node"]),
        d_head=int(hypers["d_head"]),
        d_feedforward=int(hypers["d_feedforward"]),
        num_heads=int(hypers["num_heads"]),
        num_attention_layers=int(hypers["num_attention_layers"]),
        num_gnn_layers=int(hypers["num_gnn_layers"]),
        normalization=str(hypers["normalization"]),
        activation=str(hypers["activation"]),
        transformer_type=str(hypers["transformer_type"]),
        featurizer_type=str(hypers["featurizer_type"]),
        attention_temperature=float(hypers["attention_temperature"]),
        zbl=bool(hypers["zbl"]),
        long_range_enable=bool(hypers["long_range"]["enable"]),
        n_atomic_species=int(len(dataset_info.atomic_types)),
        atomic_types=[int(z) for z in dataset_info.atomic_types],
        length_unit=str(dataset_info.length_unit),
        n_params=int(n_params),
    )

    saved: dict[str, np.ndarray] = {"config": np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)}

    # Base PET state dict. Drop `species_to_species_index` (recomputed from
    # `atomic_types` on load) and the additive/scaler TensorMap buffers (extracted
    # below as plain arrays instead).
    sd = model.state_dict()
    skip_prefixes = ("species_to_species_index", "additive_models.", "scaler.")
    n_w = 0
    for k, v in sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        saved[f"w@{k}"] = npy(v)
        n_w += 1

    # Composition reference energy: per-element weights, indexed by atomic number.
    # `weights["energy"]` is a single-block TensorMap keyed on `_`, samples
    # `center_type` (one row per atomic species in 1..102), one property `energy`.
    cm = model.additive_models[0]
    cm.sync_tensor_maps()
    comp = cm.model.weights["energy"]
    blk = comp.block(comp.keys[0])
    comp_vals = blk.values.reshape(-1).to(torch.float64)
    # samples are sorted by center_type; build a 103-row lookup indexed by Z (1..102),
    # so the device path can gather with `comp[Z]` directly (matches Orb's
    # `ref_weight`/UMA's `elem_refs` convention).
    comp_by_z = np.zeros(103, dtype=np.float64)
    for i, z in enumerate(blk.samples.values[:, 0].tolist()):
        comp_by_z[int(z)] = float(comp_vals[i])
    saved["composition_energy_by_z"] = comp_by_z  # [103], index by atomic number

    # Energy scaler: a single per-structure scale (PET-MAD energy is a single-property
    # per-structure target, so `scales["energy"]` carries one scalar). The device path
    # applies it as `E_real = raw * scale + sum_i comp[Z_i]` (scaler `apply` multiplies,
    # composition adds — see metatrain Scaler/BaseCompositionModel.forward).
    sc = model.scaler
    sc.sync_tensor_maps()
    scales = sc.model.scales["energy"]
    sblk = scales.block(scales.keys[0])
    saved["energy_scale"] = sblk.values.reshape(-1).to(torch.float64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **saved)
    print(f"wrote {args.out}  (n weight tensors={n_w}, n_params={n_params})")
    print(f"  energy_scale={float(saved['energy_scale'][0]):.10f}")
    print(f"  composition_energy_by_z[14] (Si)={comp_by_z[14]:.6f}")


if __name__ == "__main__":
    main()
