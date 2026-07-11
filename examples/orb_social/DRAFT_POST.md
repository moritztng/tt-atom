# Social post draft — Orb-v3 materials MD on Tenstorrent (goal #7 Amplify)

Status: DRAFT, awaiting Moritz to review/post. Nothing sent anywhere.
Video + full verification in this folder (`VERIFICATION.md`, `energy_temp.png`, `parity.json`).
Mirrors the UMA caffeine-MD post: prose, first person, one concrete number, the GPU-shift angle,
forward-looking closer.

Attach: `orb_si_md.mp4` (post) — or `orb_si_md.gif` (README/preview).

---

## Post (LinkedIn / X)

Orb, one of the strongest materials models out there, now runs on Tenstorrent.

Orb-v3 from Orbital Materials is a universal interatomic potential trained on OMat24 — one of the
top models on Matbench Discovery, and built for the thing that used to demand a cluster: simulating
real materials, periodic crystals and all, at scale. We ported it into TT-Atom and ran it end to
end on a single Blackhole card.

The clip is a live molecular-dynamics run of a silicon crystal — a 216-atom supercell, held at
900 K, with the real conservative forces (F = −dE/dpos) coming off the device at every single
timestep, about 20 steps a second on one card. What you're watching is genuine thermal vibration of
the lattice, computed by a neural-network potential on hardware that isn't a GPU.

That last part is the point. Materials simulation has run on GPUs by default for a decade. It
doesn't have to. The forces on device reproduce the reference Orb-v3 to a correlation of 0.99996,
the whole backbone runs on stock ops with no custom kernels, and the temperature holds its target
to 0.1% over the run. If you work in materials discovery, it's worth a look — the model you already
trust, running somewhere new and a lot cheaper. All open source in TT-Atom.

---

## Optional first line if he wants the number harder up top

Molecular dynamics of a silicon crystal, ~20 steps a second, on a single Tenstorrent card and not a
GPU in sight — running Orbital Materials' Orb-v3.

## Optional throughput-per-dollar sentence (needs a number Moritz can stand behind)

Add if wanted, to sharpen the GPU-shift angle:
"...roughly [N]× the simulation-per-dollar of an [GPU] on the same run [NUMBER TBD — not measured;
would need a like-for-like GPU MD run of the identical system]."
I did NOT fabricate a GPU comparison. Everything quoted in the post above is measured on-device.

---

## Numbers behind the post (all measured this task; see `VERIFICATION.md` for the receipts)

- **System:** silicon diamond, 3×3×3 cubic supercell = **216 atoms, 9,936 periodic edges**, NVT
  Langevin @ 900 K, 1 fs steps, single Blackhole card (physical device 0), stock `ttnn`.
- **Model:** `orb-v3-conservative-inf-omat` (OMat24), analytic conservative forces F = −dE/dpos.
- **Throughput:** **48–51 ms / MD step** warm median (energy + forces, trace-capture replay) ⇒
  **≈20 MD steps/s**; ~4,300–4,500 atom-steps/s. Scales on the same card: 512-atom cell ~9 steps/s,
  no OOM.
- **Accuracy vs orb-models CPU reference** (same MD frames): forces **PCC 0.99996**, max error
  ~0.03 eV/Å (≈1% of the ~3.3 eV/Å peak), energy within ~3–5 meV/atom — matching the port's
  established parity bar. It's the real Orb-v3, not a degraded port.
- **Stability:** temperature holds **899 ± 48 K** (0.1% from target; the ±5.4% fluctuation is the
  expected thermodynamic magnitude for 216 atoms), potential energy on a flat plateau
  (−5.288 eV/atom, drift +1.7 meV/atom·ps). Crystal stays solid (900 K ≪ Si melt ~1687 K).
- **Structure sanity:** Orb-v3's energy-minimum lattice constant for diamond Si is 5.462 Å (vs
  experimental 5.431 Å, within 0.6%); the ≈−5.4 eV/atom is the potential's reference-energy per
  atom (not the cohesive energy, which is a different ~4.6 eV/atom quantity).
- On-device forces are **bit-exact between the eager and trace-captured paths** (the optimisation
  the speed relies on); Orb-v3 is non-equivariant, so the whole backbone is stock `ttnn` with none
  of UMA's custom kernels.

## Why silicon / why this fits Orb (not caffeine)

Orb-v3 is a *materials* potential (OMat24, periodic, large-scale) — the opposite of a single small
organic molecule. So the demo is a periodic crystal supercell under thermal MD, squarely Orb's
wheelhouse: bulk inorganic material, periodic boundary conditions, forces at scale. Silicon diamond
is the iconic semiconductor lattice and reads instantly as "a crystal"; monatomic Si also lets one
node feature tile to any supercell size, so the on-device run needed no new reference-env export.
(A two-colour oxide like MgO would look richer but needs a fresh Orb golden for that composition — a
good v2 if Moritz wants it.)

## Notes for Moritz
- The video is rendered in OVITO (Tachyon ray-tracer): shaded spheres in a metallic blue-steel,
  the diamond bond network (thin, muted), ambient occlusion + soft shadows, no simulation-cell
  wireframe (with unwrapped coordinates a cell outline would clip/cage the view rather than
  describe it). Coordinates are *unwrapped* (continuous per-atom images, not re-folded into the
  cell every frame) so an atom whose lattice site sits near a periodic face vibrates smoothly
  instead of jumping to the opposite side of the box — valid because the crystal is solid at 900 K
  over 1.5 ps (no diffusion, nothing drifts out of frame). Bonds use the same unwrapped positions,
  so none are drawn across periodic faces either — no "exploding crystal" or teleporting-atom
  artifact. The camera frames the bounding sphere of every atom position across every frame, so
  the whole crystal stays fully in view with margin at every turntable angle.
- One honest on-device run; no compositing, no sped-up trickery. Loop is a boomerang (forward then
  reverse) so it seams cleanly — MD isn't time-periodic.
- MP4 720² for the post (7.4 MB); GIF 340px for README/preview (6 MB).
- Reproduction: `NOTES.md`. Full verification + honesty caveats: `VERIFICATION.md`.
- If you like it as a README asset (like `assets/caffeine_md.gif`), say the word and I'll add
  `orb_si_md.gif` + wire the reproduction scripts into TT-Atom on the branch — I left the repo
  otherwise untouched.
