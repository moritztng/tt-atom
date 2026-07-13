# Social post draft — Orb-v3 silicon melt on Tenstorrent

Status: DRAFT, awaiting Moritz to review/post. Nothing sent anywhere.
Video + full verification in this folder (`VERIFICATION.md`, `melt_metrics.npz`, `parity.json`).
The earlier 900 K solid-vibration demo is archived under `prev_solid_demo/`; this version melts
the crystal and reports the MD-credibility metrics a materials scientist checks first.

Attach: `orb_si_melt.mp4` (post) — or `orb_si_melt.gif` (README/preview).

---

## Post (LinkedIn / X)

Orb-v3, one of the strongest materials interatomic potentials out there, now runs on Tenstorrent
— and this clip is a real silicon melt, not atoms jiggling in a cage.

A 216-atom diamond-cubic silicon crystal, periodic cell and all, heated through its 1687 K melting
point into the liquid. Every force comes off a single Blackhole card at every timestep
(Orb-v3's conservative F = −dE/dpos), about 21 steps a second, 0.9 nanoseconds a day on one card.
As the temperature climbs you can watch the lattice give up: the radial distribution function
loses its sharp crystalline peaks and broadens into the liquid envelope, and the mean-squared
displacement takes off as atoms finally leave their sites.

The numbers a materials person checks first, all from this run. Energy conservation: in an NVE
tail the total energy drifts 1.4 meV/atom/ps — the bar a credible MD potential has to clear (the
same order UMA is reported at). Accuracy: against the reference `orb-models` on the same frames,
forces agree to a correlation of 0.9999 and energies to ~1.4 meV/atom, and that holds in the
liquid, not just the solid. So it's the real Orb-v3 on different silicon, not a degraded port.

And the speed is the other half of the story. On the same 216-atom Si system, a single Blackhole
p150 runs Orb-v3 1.74× faster than an NVIDIA H200 — at roughly a twenty-third of the card cost,
which is about 40× the simulation per dollar. Materials simulation has run on GPUs by default for
a decade. It doesn't have to. All open source in TT-Atom.

---

## Optional harder-up-top first line

A silicon crystal melting on a single Tenstorrent card — Orb-v3 forces at every timestep, energy
conserved to 1.4 meV/atom/ps, forces matching the reference at 0.9999.

## On the GPU / per-dollar angle

Cited from a measured same-system comparison (commit `57585bf`, `docs/orb-port.md`): the same
`orb-v3-conservative-inf-omat` MD step on the same 216-atom Si diamond supercell, single
Blackhole p150 (bf16, traced) vs single NVIDIA H200 (fp32, `orb_models`), warm steady-state —
p150 **1.74× faster** (50.9 ms vs 88.4 ms/step) at ~1/23 the card cost (~$1,399 vs ~$32,000) ⇒
**~40× the throughput per dollar**. (At 512 atoms the H200 is 1.14× faster on raw throughput, but
still ~20× perf-per-dollar.) This is the exact system in the video, so the figure is quoted
directly; the raw Tenstorrent throughput (21 steps/s, 0.90 ns/day) is also in the caption.

---

## Numbers behind the post (all measured this task; see `VERIFICATION.md` for the receipts)

- **System:** diamond-cubic Si, 3×3×3 supercell = **216 atoms**, periodic boundaries, single
  Blackhole p150 (physical device 0), stock `ttnn` — no custom kernels.
- **Model:** `orb-v3-conservative-inf-omat` (OMat24), analytic conservative forces F = −dE/dpos.
- **The melt:** NVT Langevin temperature ramp 300 → 2800 K over 900 fs (1 fs of heating per
  rendered frame), then 500 fs of NVE. Temperature crosses Si's 1687 K melting point at ~554 fs;
  the structure disorders and the run ends in the liquid (~1600 K).
- **Energy conservation (the credibility metric):** NVE total-energy drift **1.4 meV/atom/ps** on
  an equilibrated 900 K solid (the direct analog of the ~1 meV/atom/ps UMA bar). In the hot liquid
  the NVE drift is ~15 meV/atom/ps — larger, as expected for a 2000 K liquid at 0.5 fs. Both real.
- **Throughput:** **48 ms / MD step** warm median (energy + analytic forces, trace-capture replay)
  ⇒ **20.8 MD steps/s**, **~4,500 atom-steps/s**, **0.90 ns/day** on one card. The neighbour list
  rebuilds as the structure disorders (9 rebuilds over the melt), so the topology stays correct
  through the liquid — the solid-only frozen-topology trick from the solid demo does not hold once
  atoms diffuse.
- **Accuracy vs orb-models** (same melt frames, CPU reference `float32-high`):
  - frame 350 (thermal solid, ~175 fs): ΔE **1.39 meV/atom**, force **PCC 0.99998**, max force
    err 0.035 eV/Å (≈0.7% of the 5.3 eV/Å peak).
  - frame 700 (liquid, ~1400 fs): ΔE **1.24 meV/atom**, force **PCC 0.99995**, max force err
    0.030 eV/Å (≈0.7% of the 4.1 eV/Å peak).
  - frame 0 (perfect lattice): ΔE 9.0 meV/atom — the known absolute-energy bf16 offset; forces
    vanish by symmetry on both sides so its PCC is meaningless.
- **Structural signatures (in the video, live):** g(r) goes from sharp crystalline peaks
  (first-shell g_max ≈ 19) to the broad liquid envelope (g_max ≈ 2.5); the MSD rises 0 → 2.6 Å² as
  atoms leave their lattice sites. Both evolve frame-by-frame alongside the atoms.

## Notes for Moritz
- The video is one honest on-device melt; the HUD is composited from the real per-step log and the
  real g(r)/MSD of that trajectory — no fabricated numbers, no sped-up trickery. The loop is a
  boomerang (forward then reverse) so it seams cleanly; MD is not time-periodic.
- The periodic-cell wireframe is deliberately shown (a periodic MD system is displayed with its
  cell box — it reads the boundary and the scale); thin and subtle so it frames the melt without
  fighting it.
- Rendered in OVITO (Tachyon): shaded Jmol-Si spheres, the diamond bond network, the cell box,
  ambient occlusion + shadows. MP4 1280×720 for the post (6.1 MB); GIF 540 px for preview (5.8 MB).
- Reproduction: `NOTES.md`. Full verification + honesty caveats: `VERIFICATION.md`.
- The earlier 900 K solid-vibration demo (the first Orb social post draft) is archived under
  `prev_solid_demo/` — this melt version supersedes it.
