# Orb-v3 release post — draft for Moritz

DRAFT for Moritz to review and post himself. Nothing here is posted anywhere and no
social account is touched. He'll attach the Si-melt video (`orb_si_melt.mp4`) himself;
the text is deliberately short because the video does the showing.

Companion copy in `~/.coworker/state/orb-release-post.md` (same content plus the
Facts + sources appendix). Video + full verification: `VERIFICATION.md`, `NOTES.md`.

---

## (a) LinkedIn

Orb-v3, one of the strongest open models for simulating materials, now runs on Tenstorrent.

Orb is a machine-learning interatomic potential. It predicts the energy and forces on a set
of atoms, so you can run molecular dynamics at close to the accuracy of a full
quantum-mechanical calculation but at a fraction of the cost. From Orbital Materials, trained
on the OMat24 dataset, Apache-2.0. The kind of model people use to screen new crystals,
catalysts, and battery materials without paying for a quantum calculation at every step.

The clip is a real 216-atom silicon crystal heated through its melting point into a flowing
liquid. Every force comes off a single Blackhole card at every timestep. Checked against the
reference orb-models package on the same frames: forces match at a correlation of 0.9999,
energy conserved to ~1.4 meV/atom/ps. The real Orb-v3, verified, not a degraded port.

Open source in tt-atom, alongside the bio models in tt-bio.

---

## (b) X / Twitter (fits a tweet, ~273 chars)

Orb-v3, one of the strongest open models for simulating materials, now runs on Tenstorrent.
The clip is a real 216-atom silicon crystal melted on a single Blackhole card. Forces match
the reference at 0.9999 correlation, energy conserved to ~1.4 meV/atom/ps. Open source in tt-atom.
