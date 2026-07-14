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

And the speed is the other half of the story: about 21 Orb-v3 MD steps a second, 0.9
nanoseconds of simulation per day, off one card — forces at every timestep, no GPU in the
loop. Materials simulation has run on GPUs by default for a decade. It doesn't have to.
All open source in TT-Atom.

> *(2026-07-14) An earlier draft of this post claimed the p150 was "1.74× faster than an
> NVIDIA H200 / ~40× per dollar". That claim has been withdrawn — a fair, evidenced
> redo (`docs/orb-port.md` "Performance per dollar") found the H200 is in fact faster
> than the p150 on raw throughput at every size tested. The p150 still wins on
> throughput-per-dollar (~3-8×, because it is ~23× cheaper), but the "faster than H200"
> framing was wrong and is not in the post. This post now makes no GPU comparison.)*

---

## Optional harder-up-top first line

A silicon crystal melting on a single Tenstorrent card — Orb-v3 forces at every timestep, energy
conserved to 1.4 meV/atom/ps, forces matching the reference at 0.9999.

## On the GPU / per-dollar angle

**Withdrawn (2026-07-14).** An earlier draft quoted "p150 1.74× faster than an H200 /
~40× per dollar" from `docs/orb-port.md` (commit `57585bf`). That comparison paired
Tenstorrent's optimized trace/replay path against the GPU's stock `orb_models` path
(neighbour list rebuilt every step, no CUDA graph) and its H200 timings had no committed
raw evidence, so it was not apples-to-apples and not verifiable. A fair, evidenced redo
(branch `wk/tt-atom-orb-gpu-fair-comparison`, see `docs/orb-port.md` "Performance per
dollar") comparing the out-of-box path on each side (TT traced; stock `pip install
orb-models` v0.7.0 `ORBCalculator`), with a size sweep and committed raw timings, found
the opposite: the H200 is faster than the p150 on raw throughput at every size tested
(3.0× / 5.5× / 5.3× / 6.7× at 216 / 512 / 1000 / 2016 atoms). The p150 still wins on
throughput-per-dollar because it is ~23× cheaper (~7.6× at 216 atoms, falling to ~3.5×
near 2000 atoms), but the "faster than H200" claim was wrong. **No GPU comparison is made
in this post**; the post speaks only to the on-device melt, accuracy, and energy
conservation. (Per Moritz, the video itself carries no GPU comparison either.)

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
- The video is one honest on-device melt; the side-card (T-ramp, MSD, g(r)) is composited from the
  real per-step log and the real g(r)/MSD of that trajectory, advancing in lockstep with the atoms
  — no fabricated numbers, no sped-up trickery. It plays forward once with a short fade in/out at
  the loop point (MD is not time-periodic; a boomerang would rewind the ramp = look like cooling).
- No jumping: the render uses unwrapped, continuous coordinates (periodic images accumulated across
  the trajectory, per-frame centre of mass removed) with no cell box and no tiling — so no atom
  teleports across a face and no image atoms pop in/out. Verified quantitatively: max per-atom
  displacement between consecutive rendered frames 0.288 Å (box is 16.29 Å), constant 216 atoms.
- No GPU / NVIDIA / per-dollar comparison anywhere in the video (the label is model + system + T
  only), per Moritz. The post text also makes no GPU comparison now: the earlier "1.74× faster
  than H200 / ~40× per dollar" claim was withdrawn after a fair, evidenced redo
  (`docs/orb-port.md` "Performance per dollar") found the H200 faster on raw throughput at every
  size; the p150's case is price/perf, not raw speed, and that nuance is left to the doc rather
  than the post.
- Rendered in OVITO (Tachyon): shaded cool-silicon spheres on a near-black canvas, ambient
  occlusion + shadows, no cell box. MP4 1920×1080 (~3 MB), GIF 720 px for preview. Standalone
  4-panel physics figure in `melt_charts.png` for scientific scrutiny.
- Reproduction: `NOTES.md`. Full verification + honesty caveats: `VERIFICATION.md`.
- The earlier 900 K solid-vibration demo (the first Orb social post draft) is archived under
  `prev_solid_demo/` — this melt version supersedes it.
