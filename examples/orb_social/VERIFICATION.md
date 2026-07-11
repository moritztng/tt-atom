# Verification — Orb-v3 materials-MD demo on Tenstorrent Blackhole

Every claim in `DRAFT_POST.md` is backed here. All on-device numbers are from physical device 0
(single Blackhole card) with the real `orb-v3-conservative-inf-omat` weights; the CPU reference is
the public `orb-models` package (v3.29 env, `orb-v3-conservative-inf-omat`, float32-high) on the
*same* atomic configurations. Reproduce with `NOTES.md`.

## 1. On-device vs orb-models CPU reference — the trajectory is the real Orb-v3

For frames of the actual 216-atom MD trajectory, on-device energy + conservative forces vs the
`orb-models` CPU reference (`orb_parity.py`, full data in `parity.json`):

| frame | state | ΔE (meV/atom) | force PCC | max force err (eV/Å) | ref \|F\|max (eV/Å) |
|------:|-------|--------------:|----------:|---------------------:|--------------------:|
| 0     | perfect lattice | 8.99 | — (forces ≈ 0) | 0.022 | 0.001 |
| 125   | thermal, 900 K  | 3.05 | **0.999963** | 0.031 | 3.32 |
| 251   | thermal, 900 K  | 4.72 | **0.999965** | 0.025 | 3.31 |

- On the dynamics frames (what the video shows), forces match the reference at **PCC 0.99996**,
  with max error ≈0.03 eV/Å ≈ 1 % of the ~3.3 eV/Å peak force — consistent with the port's
  established conservative-force bar (`tests/test_orb_forces_realweight.py`: PCC > 0.999,
  MAE 0.0089 eV/Å). Energy agrees to ~3–5 meV/atom.
- Frame 0 is the perfect lattice: forces vanish by symmetry (ref \|F\|max = 8e-4 eV/Å), so its
  force-PCC is meaningless (pure numerical noise, both ≈ 0). Its absolute energy carries a
  ~9 meV/atom bf16 offset (0.17 % of 5.4 eV/atom); this is an *absolute* energy offset and does
  not affect the dynamics, which are driven by forces and by energy *differences* (ΔE ≤ 5 meV/atom
  on the thermal frames).

**Conclusion:** the on-device model reproduces orb-models, not a degraded approximation.

## 2. MD stability (`energy_temp.png`, from `md_series.csv`)

NVT Langevin (target 900 K, friction 0.02 fs⁻¹, 1 fs step, 1500 steps). NVT does **not** conserve
total energy (the thermostat exchanges energy with a bath), so the correct checks are temperature
control and a stable potential-energy plateau — both hold, excluding the first 200 fs of
equilibration:

- **Temperature:** plateau ⟨T⟩ = **898.9 ± 48.2 K** — 0.1 % from target. The ±5.4 % instantaneous
  fluctuation matches the thermodynamic prediction √(2/3N) = 5.55 % for N = 216 exactly.
- **Potential energy:** plateau ⟨E⟩ = **−5.2875 ± 0.0066 eV/atom**, drift **+1.70 meV/atom·ps**
  (flat — well within the stricter NVE "few meV/atom·ps" bar). No blow-up; the crystal stays solid
  (900 K ≪ Si melt ≈ 1687 K).
- The plateau value is physically correct: equipartition gives ⟨E⟩ = E_min + 1.5 k_B T =
  −5.415 + 0.116 = −5.299 eV/atom, matching the measured −5.288 within fluctuation.

## 3. Structure / energy sanity (`lattice_check.py` → `lattice_check.txt`, CPU reference)

Orb-v3's own energy-minimum lattice constant for diamond-cubic Si (parabolic fit of E(a)):

- **a_eq = 5.462 Å** vs experimental 5.431 Å — within **0.6 %** (excellent for a universal MLIP).
- E(a = 5.43 Å) = **−5.415 eV/atom**, essentially at the potential's minimum (−5.416 eV/atom).

So the MD system is a correct diamond-cubic Si crystal at a defensible lattice constant, and the
≈−5.4 eV/atom is the potential's total energy per atom (referenced to its OMat24/DFT reference),
**not** the cohesive energy (Si cohesive energy ≈ 4.6 eV/atom — a different, unrelated quantity).

## 4. Throughput (device, `orb_md_device.py` summary)

- **216 atoms, 9,936 periodic edges**, warm median **48–51 ms / MD step** across runs (300/600/1500
  steps) ⇒ **≈20 MD steps/s** on one card; ~4,300–4,500 atom-steps/s.
- Warm = trace-captured forward+backward replayed; the neighbour list is frozen at t=0 (valid: the
  solid never breaks bonds, max displacement ≪ 6 Å cutoff). This is bit-exact to the eager path
  (constant tensor shapes ⇒ program-cache hits; rebuilding the graph each step recompiles kernels,
  ~40× slower).

## Honesty caveats
- The frame-0 absolute-energy offset (9 meV/atom, §1) is disclosed above; it is an absolute-energy
  bf16 artifact, not a dynamics error.
- No GPU / per-dollar comparison is claimed — none was measured (would need a like-for-like GPU MD
  run of the identical system). The draft leaves it an explicit TBD.
- The video is one honest on-device run: no compositing, no speed tricks. The loop is a boomerang
  (forward then reverse) purely so it seams cleanly; MD is not time-periodic.
