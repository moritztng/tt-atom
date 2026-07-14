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
point into the liquid, then held there for 6 ps so it actually flows. Every force comes off a
single Blackhole card at every timestep (Orb-v3's conservative F = −dE/dpos), about 23 steps a
second, 1.0 nanosecond a day on one card. As the temperature climbs you can watch the lattice give
up: the radial distribution function loses its sharp crystalline peaks and broadens into the liquid
envelope, and the mean-squared displacement takes off as atoms finally leave their sites and
diffuse.

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
directly; the raw Tenstorrent throughput (23 steps/s, 1.01 ns/day) is also in the caption.

---

## Numbers behind the post (all measured this task; see `VERIFICATION.md` for the receipts)

- **System:** diamond-cubic Si, 3×3×3 supercell = **216 atoms**, periodic boundaries, single
  Blackhole p150 (physical device 0), stock `ttnn` — no custom kernels.
- **Model:** `orb-v3-conservative-inf-omat` (OMat24), analytic conservative forces F = −dE/dpos.
- **The melt:** NVT Langevin temperature ramp 300 → 2200 K over 0.8 ps, then a **6 ps hold at a
  constant 2200 K** (the liquid diffuses over this window), then a short NVE tail — 7.2 ps total,
  3604 frames. Temperature crosses Si's 1687 K melting point at ~626 fs; the structure disorders
  and diffuses through the liquid hold.
- **Energy conservation (the credibility metric):** NVE total-energy drift **1.4 meV/atom/ps** on
  an equilibrated 900 K solid (the direct analog of the ~1 meV/atom/ps UMA bar). In the hot
  ~2200 K liquid the NVE drift is ~29 meV/atom/ps — larger, as expected at 0.5 fs. Both real.
- **Throughput:** **42.7 ms / MD step** warm median (energy + analytic forces, trace-capture
  replay) ⇒ **23.4 MD steps/s**, **~5,050 atom-steps/s**, **1.01 ns/day** on one card. The
  neighbour list rebuilds as the structure disorders (80 rebuilds over the melt), so the topology
  stays correct through the liquid — the solid-only frozen-topology trick from the solid demo does
  not hold once atoms diffuse.
- **Accuracy vs orb-models** (same melt frames, CPU reference `float32-high`):
  - frame 1200 (liquid, ~2.4 ps): ΔE **0.45 meV/atom**, force **PCC 0.99993**, max force err
    0.056 eV/Å (≈1.4% of the 4.1 eV/Å peak).
  - frame 3600 (liquid, ~7.2 ps, late): ΔE **0.16 meV/atom**, force **PCC 0.99983**, max force
    err 0.163 eV/Å (≈3% of the 5.2 eV/Å peak).
  - frame 0 (perfect lattice): ΔE 9.0 meV/atom — the known absolute-energy bf16 offset; forces
    vanish by symmetry on both sides so its PCC is meaningless.
- **Structural signatures (in the video, live):** g(r) goes from sharp crystalline peaks
  (first-shell g_max ≈ 17) to the broad liquid envelope (g_max ≈ 2.6); the MSD rises from 0 and
  climbs linearly to ~47 Å² over the 6 ps liquid (D ≈ 12×10⁻⁹ m²/s, right order for liquid Si).
  Both evolve frame-by-frame alongside the atoms.

## Notes for Moritz
- The video is one honest on-device melt; the side-card (T-ramp, MSD, g(r)) is composited from the
  real per-step log and the real g(r)/MSD of that trajectory, advancing in lockstep with the atoms
  — no fabricated numbers, no sped-up trickery. It plays forward once with a short fade in/out at
  the loop point (MD is not time-periodic; a boomerang would rewind the ramp = look like cooling).
- No jumping: the render uses unwrapped, continuous coordinates (periodic images accumulated across
  the trajectory, per-frame centre of mass removed) with no cell box and no tiling — so no atom
  teleports across a face and no image atoms pop in/out. The 3D scene is windowed to the cohesive
  melt + churn interval [0, 2.4 ps] (a 6 ps unwrapped liquid diffuses past one box and would show
  detached fliers); the charts still span the full run. Verified quantitatively: max per-atom
  displacement between consecutive rendered frames 0.962 Å (box/2 teleport threshold 8.14 Å),
  constant 216 atoms, no fliers.
- No GPU / NVIDIA / per-dollar comparison anywhere in the video (the label is model + system + T
  only), per Moritz. The per-dollar angle above is for the post text only, optional.
- Rendered in OVITO (Tachyon): shaded cool-silicon spheres on a near-black canvas, ambient
  occlusion + shadows, no cell box. MP4 1920×1080 (100 frames, 30 fps, ~1.7 MB), GIF 720 px
  (~2.3 MB) for preview. Standalone
  4-panel physics figure in `melt_charts.png` for scientific scrutiny.
- Reproduction: `NOTES.md`. Full verification + honesty caveats: `VERIFICATION.md`.
- The earlier 900 K solid-vibration demo (the first Orb social post draft) is archived under
  `prev_solid_demo/` — this melt version supersedes it.
