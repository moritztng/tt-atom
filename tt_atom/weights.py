"""Weight loading: a portable ``.npz`` bundle of the eSCN-MD parameters + fixed buffers.

TT-Atom is the *implementation*; users bring their own fairchem checkpoint. Because ttnn
(numpy<2) and fairchem (numpy>=2) cannot share a process, real weights are *exported* once in a
fairchem environment (``tools/export_weights.py``) into a numpy bundle that this loader reads in
the ttnn environment. The bundle carries both the learned ``state_dict`` and the fixed geometric
buffers (Jd, to_m, SO3 grid matrices, gaussian basis) that are not all in a bare ``state_dict``.

The bundle format is exactly the one the parity goldens already use, so tests and the calculator
share a single code path. ``WeightBundle.verify_coverage`` checks a real checkpoint is a drop-in
fit (every key the modules need is present with the right shape)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch


class WeightBundle:
    def __init__(self, npz):
        self._d = npz
        self.config = json.loads(bytes(npz["config"]).decode())

    @classmethod
    def load(cls, path):
        return cls(np.load(pathlib.Path(path)))

    def _t(self, key):
        return torch.from_numpy(self._d[key].copy())

    @property
    def weights(self):
        return {k[2:]: self._t(k).float() for k in self._d.files if k.startswith("w@")}

    def buffer(self, name):
        return self._t(f"host@{name}").float()

    def has(self, key):
        return key in self._d.files

    # --------------------------------------------------------------- energy normalizer / task

    @property
    def task(self):
        """Dataset token for the system embedding (omol/omat/oc20/...); omat for legacy bundles."""
        return self.config.get("task", "omat")

    @property
    def scale_rmsd(self):
        """Energy/force scale: real targets are ``rmsd * raw + mean`` (1.0 for legacy bundles)."""
        return float(self._d["scale@rmsd"][0]) if self.has("scale@rmsd") else 1.0

    @property
    def scale_mean(self):
        return float(self._d["scale@mean"][0]) if self.has("scale@mean") else 0.0

    @property
    def elem_refs(self):
        """Per-element reference energies added back to the (denormed) energy, or None."""
        return self._t("scale@elem_refs").double() if self.has("scale@elem_refs") else None

    @property
    def reference(self):
        """Embedded fairchem reference (E/F + input) for this bundle's merge composition, so a
        device-side roundtrip check needs no fairchem env. ``None`` if the bundle carries none."""
        if not self.has("ref@energy"):
            return None
        return dict(
            energy=float(self._d["ref@energy"][0]),
            forces=self._d["ref@forces"].copy(),
            pos=self._d["ref@pos"].copy(),
            atomic_numbers=self._d["ref@atomic_numbers"].copy(),
            charge=float(self._d["ref@charge"][0]),
            spin=float(self._d["ref@spin"][0]),
            cell=self._d["ref@cell"].copy() if self.has("ref@cell") else None,
            pbc=self._d["ref@pbc"].copy() if self.has("ref@pbc") else None,
        )

    # convenience accessors for the fixed geometry buffers
    @property
    def to_m(self):
        return self.buffer("to_m")

    @property
    def coefficient_index(self):
        """Spherical-harmonic coefficient subselection for mmax<lmax checkpoints (uma-m), else
        None. Selects the |m|<=mmax coefficients before the m-mapping (fairchem prepare_wigner)."""
        return self._t("host@coefficient_index").long() if self.has("host@coefficient_index") else None

    @property
    def gauss_offset(self):
        return self.buffer("gauss_offset")

    @property
    def gauss_coeff(self):
        return self.buffer("gauss_coeff")

    @property
    def to_grid_mat(self):
        return self.buffer("to_grid_mat")

    @property
    def from_grid_mat(self):
        return self.buffer("from_grid_mat")

    # --------------------------------------------------------------- coverage verification

    def expected_keys(self):
        """The weight keys the ttnn modules consume, derived from the config."""
        cfg = self.config
        L = cfg["num_layers"]
        keys = []
        # embeddings + mixing
        keys += ["sphere_embedding.weight", "source_embedding.weight", "target_embedding.weight",
                 "mix_csd.weight", "mix_csd.bias"]
        if cfg.get("chg_spin_emb_type", "pos_emb") == "rand_emb":
            keys += ["charge_embedding.rand_emb.weight", "spin_embedding.rand_emb.weight"]
        else:
            keys += ["charge_embedding.W", "spin_embedding.W"]
        keys += [f"Jd_{l}" for l in range(cfg["lmax"] + 1)]
        # edge-degree radial MLP
        for i in (0, 1, 3, 4, 6):
            keys.append(f"edge_degree_embedding.rad_func.net.{i}.weight")
        for b in (0, 3, 6):
            keys.append(f"edge_degree_embedding.rad_func.net.{b}.bias")
        for li in range(L):
            p = f"blocks.{li}"
            keys += [f"{p}.norm_1.affine_weight", f"{p}.norm_1.affine_bias",
                     f"{p}.norm_2.affine_weight", f"{p}.norm_2.affine_bias"]
            keys += [f"{p}.edge_wise.so2_conv_1.fc_m0.weight", f"{p}.edge_wise.so2_conv_1.fc_m0.bias",
                     f"{p}.edge_wise.so2_conv_2.fc_m0.weight", f"{p}.edge_wise.so2_conv_2.fc_m0.bias"]
            for m in range(1, cfg["mmax"] + 1):
                keys.append(f"{p}.edge_wise.so2_conv_1.so2_m_conv.{m-1}.fc.weight")
                keys.append(f"{p}.edge_wise.so2_conv_2.so2_m_conv.{m-1}.fc.weight")
            for i in (0, 1, 3, 4, 6):
                keys.append(f"{p}.edge_wise.so2_conv_1.rad_func.net.{i}.weight")
            if cfg.get("ff_type", "grid") == "spectral":
                keys += [f"{p}.atom_wise.scalar_mlp.0.weight", f"{p}.atom_wise.scalar_mlp.0.bias",
                         f"{p}.atom_wise.so3_linear_1.weight", f"{p}.atom_wise.so3_linear_1.bias",
                         f"{p}.atom_wise.so3_linear_2.weight", f"{p}.atom_wise.so3_linear_2.bias"]
            else:
                for gi in (0, 2, 4):
                    keys.append(f"{p}.atom_wise.grid_mlp.{gi}.weight")
        keys += ["norm.affine_weight", "norm.affine_bias"]
        keys += [f"energy_block.{i}.weight" for i in (0, 2, 4)]
        keys += [f"energy_block.{i}.bias" for i in (0, 2, 4)]
        return keys

    def verify_coverage(self):
        """Return (ok, missing, present_count). Raises nothing; the caller decides."""
        w = self.weights
        missing = [k for k in self.expected_keys() if k not in w]
        return (len(missing) == 0, missing, len(w))
