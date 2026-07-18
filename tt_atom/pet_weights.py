"""Weight loading for the PET-MAD (UPET) port: a portable ``.npz`` bundle of the base
PET state dict + the two post-processing artefacts the device path needs (the per-element
composition reference energy and the single energy scaler), generated in the upet
reference env (``~/.ttatom_run/upetenv``) by ``tools/export_pet_weights.py`` and read
here in the ttnn env.

Shares the ``.npz`` container mechanics (config parse, ``w@`` weights, lazy tensor copy)
with ``tt_atom/weights.py`` / ``tt_atom/orb_weights.py`` via :class:`bundle.NpzBundle`.
The extra accessors are PET-specific: the composition reference lookup and the energy
scale, which the calculator applies as ``E = raw * scale + sum_i comp[Z_i]`` (mirroring
metatrain's Scaler `apply` + BaseCompositionModel `forward`, see
``tools/export_pet_weights.py``)."""
from __future__ import annotations

from .bundle import NpzBundle


class PetWeights(NpzBundle):
    def composition_energy_by_z(self):
        """Per-atomic-number composition reference energy, ``[103]`` indexed by Z
        (entries 0 and >102 are zero — PET-MAD covers species 1..102). Sum over the
        atoms' entries and add to the scaled raw energy, exactly like Orb's
        ``ref_weight`` / UMA's ``elem_refs``."""
        return self._t("composition_energy_by_z").double()

    def energy_scale(self):
        """Single per-structure energy scaler (``E_real = raw * scale``)."""
        return float(self._d["energy_scale"].reshape(-1)[0])
