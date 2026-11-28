# g6me roadmap — temporary working record (2026-11-28)

Status: **WAVE 1 IN FLIGHT.** This file is the execution record for the review and the
HyprEQ audit pasted on 2026-11-28. It is a working document, not a claim of completion.
Checkpoints are committed and pushed after each wave.

## Doctrine (applies to every item)

- Production CPU and GPU implement the same feature natively on different hardware.
- Bit-exactness by specification: same algorithm and operation order, strict IEEE, zero FMA
  (runtime `nvcc --fmad=false`). Integer stages bit-identical; float stages gated at the
  smallest measured difference.
- No "missing CUDA stage" and no silent CPU fallback counted as GPU success. Every GPU-only
  feature gets a bit-exact CPU counterpart; every CPU production feature gets a GPU copy.
- A paper's name appears only on an implementation of that paper; intentional approximations
  are renamed.
- Every fix/feature lands with a sabotage test that fails on the unfixed/wrong implementation.
- Primary sources only: full text fetched via scrapling with proxy `127.0.0.1:2080`; no
  formulas from memory.
- No new EQ features. HyprEQ primitives are ported only where a paper's algorithm needs them.

## Wave 1 — bugs (in flight)

| # | System / files | Items | Status |
|---|---|---|---|
| 1 | `room.rs`, `checks.rs` | E2E CUDA stage gate must fail on absent/fallback stage; per-band `e_rel` floor; double ISO 9613-1 air; `band_value_at` 125 Hz step -> PCHIP; test index labels; CPU<->CUDA parity | DONE (fix-room-gate) |
| 2 | `zbc.rs` | tail padding ramp; decimator 31-sample phase; PLA/chirp gains vs BEZ2018a C source; SPL calibration; integer step counter; 3 `excessive_precision` lints | DONE (fix-zbc); 9/9 zbc tests, C reference pinned |
| 3 | `iso3382/{metrics,filterbank}.rs` | Eq. A.14 LF denominator; B-format W/Y; IEC 61260 class-1; zero-phase energy loss; docstring drift; Neumaier | DONE (fix-iso3382); 46/46 tests, clippy clean |
| 4 | `externalization.rs` | invented ITSD -> paper-literal 7 cues; pinna template root cause; Eq. 7 clamp; true Butterworth; 2021 citation | DONE (fix-extcues); now on room Track 1 |
| 5 | `dng.rs`, `bem.rs` remesh | GainMap plane offset; float denom + HDR clip; remesh budget + exact `target_met`; clippy already fixed at HEAD; stale strings reported | DONE (fix-dng-scan); report now wired through `scan.rs` + CLI `--remesh-max-faces` (orchestrator) |
| 6 | `scattering.rs` | `r_hi` cutoff; scaled Miller `J_m`; mpmath verification | DONE (fix-scattering); 12/12 tests, 60-digit references |
| 7 | `plus.rs`, CUDA UPOLA | rollback-before-commit; single inverse + single D2H | DONE (fix-upola); 24/24 plus tests; now on CUDA optimal-Q |
| 8 | parity | full CPU/GPU stage + kernel + feature matrix | AUDIT DONE (/home/sanjxz/.cache/g6me-parity/audit.md); gate fix in flight (parity-audit) |
| 9 | docs drift (P2) | README/gui/`ACADEMIC_NUKE_ACOUSTICS.md`/`SCAN_FMM_BEM_NTT.md`: PCG32, "native CPU tracer pending", 2.2e-9, gammatone, counts; `bem.rs:509`, `checks.rs:2695/3120/3242/3255` strings | Wave 4 |
| 10 | head-mesh / room integration | timing misalignment, snowman room decode, BEM ear point / interpolation / remesh budget (review 2 below) | IN FLIGHT (fix-headmesh) |
| 11 | parity P0 (parity-audit) | `bem-head-mesh` GPU success never recorded; e2e gate requires 3 of 9 stages; `checks.rs:4713` synthetic fixture; `ntt.rs:403-454` silent CPU fallback | IN FLIGHT (parity-audit) |

**Wave-1 verification (2026-11-28).** Focused gate on the finished systems: iso3382 46/46,
zbc 9/9, remesh 5/5, dng 15/15. The first iso3382 run reported 42/46 during a concurrent
rebuild by a sibling agent; an immediate rerun was green (46/46), so it was a transient build
artifact, not a test regression. The full workspace gate is deferred to wave 4: wave-1.5
agents are editing `bem.rs`, `plus.rs`, `room.rs`, `checks.rs`, `ntt.rs`, `blackwell.rs`,
`g6me-cuda`, `scan.rs`, and new modules concurrently, so a whole-tree gate would measure
in-flight work. Checkpoint commits are safety snapshots and may not compile by policy.

## Review 2 — head-mesh / room integration (recorded 2026-11-28 mid-turn)

All anchors below re-verified in-tree after the review arrived.

**P0 timing — direct sound 42-84 ms late, room reflections arrive before it.**
`plus.rs:824` aligns the whole renderer and the room to `PRE_DELAY_S = 0.001` (48 samples at
48 kHz). `bem.rs:2243` instead uses `pre_delay_s = (n_fft as f64 / 8.0) / fs`; `plus.rs:941`
passes `n_ir` = 16_384 for Control Room and 32_768 for Theater, so the direct path lands at
sample 2048 (42.7 ms) / 4096 (85.3 ms). Even without a room (`n_ir = 2048`) the direct path
sits at sample 256 while the LFE path sits at 48. `head_mesh_fmm_bem_and_exact_ntt_end_to_end`
missed it because it runs Pure only, with no room and no LFE.
Decision: one shared causal alignment constant (`PRE_DELAY_S`, already in `plus.rs`) used by
`bem.rs`; BEM direct IR delayed by exactly that constant; test with `--room` and LFE asserting
direct/LFE/room all share the same reference; no `n_fft`-derived delay anywhere.

**P0 room decode uses the snowman, not the user's head.**
`MagLsDecoder::build` (`plus.rs:868`) is always built from `snowman_geom` (two smooth spheres,
no pinnae) and decodes every ISM early reflection and every HOA-traced ray. `--head-mesh`
(`bem_direct`, `plus.rs:952`) replaces only the 11-column direct path. The room around the
listener is therefore the wrong head.
Decision: room early/late decoding must use the same BEM head geometry (HRIR set or on-the-fly
BEM HRTF interpolation). Until that lands, `--head-mesh` + `--room` must refuse loudly rather
than silently decode the room with a snowman.

**P1 BEM-HRTF correctness (`bem.rs`)**
- `find_ear_element` (`bem.rs:2287-2299`) maximizes `y*ear_sign - 0.5|x| - 0.5|z|`: selects the
  most lateral helix-rim element, not the recessed ear-canal entrance. Pinna diffraction is
  measured at the wrong point. Replace with an anatomical concha/canal-entrance criterion
  (inward-facing element near the concha bowl, validated against a known HRTF).
- `interp_complex_spectrum` (`bem.rs:2302+`) interpolates Cartesian Re/Im between 28 log-spaced
  anchors although the docstring promises log-magnitude + unwrapped phase. At 8-16 kHz anchor
  spacing is ~1.5-2.5 kHz; with contralateral ITD ~650 us the phase advances ~2*pi between
  anchors and the chord creates phantom HF notches. Interpolate log-magnitude linearly and phase
  unwrapped, then recombine.
- Remesh ceiling 16_384 faces (`bem.rs:800`) + TSDF `clamp(10, 48)` (`scan.rs:1105`) + 16^3
  auto-repair (`plus.rs:891`) erase the pinna before BEM ever runs (~22 cm head at 48^3 is
  ~5.5-6 mm/voxel; concha spans ~3x5x8 voxels). Head meshes need a larger budget and/or local
  refinement, and `target_met = false` must abort loudly instead of shipping a smoothed potato.

**Scan realism (moves to Wave 5, not a wave-1 bug fix)**
- `burst3d.rs` is not wired into `scan.rs` (only `mat3_mul`/`so3_exp` are used);
  `dense_depth_hint` is thinned to 16x16 (256 points, `scan.rs:2867`).
- SfM takes <=120 Harris corners/frame; ~50-100 surviving points become 18 mm surfels
  (`scan.rs:2897`) with centroid-sphere normals and 4 coordinate-descent steps.
- `detect_rect_edge_points` thresholds globally at `(p20+p80)/2`; works on a synthetic white
  card in a black void, almost certainly fails on a real selfie.
- Consequence recorded honestly: live-phone capture will not produce a millimetre pinna until
  the capture path is fixed; correct near-term input is a proper scan or Mesh2HRTF/SOFA (Wave 3).

**Test hygiene**
- `zbc.rs:1160/1181` write to `/home/sanjxz/.cache/g6me-papers/zbc/` from `#[test]`; must use a
  portable temp dir (sent to fix-zbc).
- `pipeline.rs:2573` hardcodes `/mnt/aosp/xlam/Atmos/AgesOfPain.mka`; Wave 4 portability item.

## Review 3 — real capture and product plan (recorded 2026-11-28)

Camera side is proven: the `fold5-camera2-research` project already produces 50 MP RAW DNG +
OIS data, and `dng.rs` decodes it. The gap is reconstruction quality, personalisation data,
SOFA I/O, and the real-time/product shell. Dispositions, in execution order:

**Now (wave 1.5): junction bugs + gate floor + import/meshing foundations**
- 4 junction bugs: pre-delay sync, phase-unwrapped interpolation, ear-canal point, room decoded
  through the BEM head (Review 2). Owner: fix-headmesh.
- SOFA AES69-2022 reader/writer + PLY/OBJ mesh import/export (OBJ already exists:
  `TriangleMesh::from_obj_str` / `load_obj` at `bem.rs:100/194`; SOFA and PLY do not exist).
  Owner: new `sofa.rs` + `mesh_io.rs` modules.
- Scan realism phase A: adaptive card rectangle detection (LSD/Canny + RANSAC, not global
  `(p20+p80)/2`), wire `burst3d.rs` dense depth into `scan.rs`, raise the 16x16 depth thinning,
  adaptive/sparse TSDF above `clamp(10, 48)`. Owner: scan agent once a slot frees.
- GPU perf: CUDA_PERF_PLAN sections 2-5 + ETH optimal-Q Ozaki-II (arXiv:2609.37693).
  Owner: fix-upola.

**Wave 3 (papers):**
- SOFA-data personalisation: retarget `PersonalizationSession` BALD (`cochlea.rs:193`, currently
  3 snowman scalars) onto real measured heads (HUTUBS / SONICOM / SADIE II, ~100 ears; 6-8 PCA
  components or nearest-neighbour), 15 A/B clicks, no scanner. Needs the SOFA reader.
- MagLS phase continuation (Hold 2023), BiMagLS (Engel 2021/2022), diffuse-field EQ (JAES 2024).
- Room: Lundeby 1995 decay estimation, Tier-A shoebox presets wired to `--room`, OpenAIR /
  dEchorate validation gates, colorless FDN (Dal Santo 2023/2024), GFDN per-band absorption
  (Das et al. arXiv:2508.06686), passive FDTD boundaries (Kowalczyk-van Walstijn; Bilbao-Hamilton).
- BEM validation against KEMAR/KU100 + HUTUBS at <= 1-1.5 dB to 16 kHz.

**Wave 5 (real scans / capture):**
- Two-pass protocol: head 15-20 photos @ ~60 cm; macro ear orbit 30-40 per ear @ 15-20 cm,
  +-60 deg azimuth x +-30 deg elevation, diffuse light; AR coverage ring from the 500 Hz IMU,
  blur/glare indicators, GOST ID-1 overlay. Wide-baseline orbit is what buys sub-mm depth;
  hand-tremor micro-baseline cannot (At ~25 cm, f~6000 px: 0.5 mm depth = 0.144 px for a 3 mm
  tremor baseline vs 4.80 px for a 10 cm orbit).
- Reconstructor: dense features + PatchMatch MVS on CUDA (or a real differentiable 2DGS/PGSR
  rasteriser); sparse-octree TSDF (~0.3-0.4 mm in the ear cube, ~4 mm on the skull); gradient
  acoustic remesh (~0.8-1.0 mm pinna, 6-10 mm skull/torso, 25-35k triangles). Replace the
  isotropic 16_384 ceiling for head meshes; keep `RemeshBudgetReport` honest.
- Resolution ceilings recorded from the review (re-derive at implementation, Anomaly-First):
  SfM 120 Harris corners/frame (`scan.rs:2566`); depth thinned to 16x16 (`scan.rs:2867`);
  18 mm surfels (`scan.rs:2897`); TSDF `clamp(10,48)` -> ~7.0 mm voxel on a 22 cm head
  (`scan.rs:1105`); winding repair 16^3 in `plus.rs:891` -> ~21 mm voxel; 16_384-face remesh ->
  ~4.1 mm minimum edge on a 0.12 m^2 head (`bem.rs:800`).
- Ear-canal point: auto-detect deepest concha point, plus a 3D-preview click as ground truth.
- BEM at ~120 frequencies -> `.sofa` export of the user's own ears.
- One-click import of `.obj/.ply` from Polycam / Kiri / Apple Object Capture / Heges.

**Wave 6 (real-time product):**
- NUPOLA (Gardner 1995): 128/256-sample direct/early blocks, 1024-4096 tail; rotation-to-sound
  latency < 10-15 ms.
- Head tracking: OpenTrack UDP 4242 (phone webcam 6-DoF on-device, < 8 ms, or IMU);
  order-7 HOA rotated by a block-diagonal Wigner-D matrix (64x64), no BRIR recompute.
- Native 7.1.4 (12-channel, heights not nulled); PipeWire virtual sink; VST3/CLAP via nih-plug;
  LUFS-matched A/B bypass.
- HpTF: AutoEQ / Oratory1990 / HyprEQ preset load + 2-min swept-sinus canal-peak tuner
  (2-14 kHz, Griesinger/Earful style) + optional $20 capsule measurement (Moller 2007).
- HUD telemetry (Klippel coil, BEZ2018a spikes/s, Baumgartner-Majdak externalisation, Cramer
  atmosphere, SASS 0-FMA, Lean status) + live 3D viewport.
- README: BEM-vs-KEMAR/HUTUBS graph, room-vs-OpenAIR graph, ear .obj-vs-photo, 30 s binaural
  demo. Lean certificate (FDTD passivity / Goldilocks reduction) optional, not shipping-blocking.

Scope decision recorded: import + high-res meshing + SOFA first (fastest path to real ears),
then the real-time stack, then HUD/demo polish. No new EQ features.

## Wave 2 — HyprEQ ports (paper-required only)

| Item | Decision |
|---|---|
| Hilbert-cepstral minimum-phase FIR (`src/ir.rs`: `minimum_phase_spectrum`, `synthesize_minimum_phase_ir`) | PORT as shared primitive for HRTF/MagLS; do not replace the existing half-Kaiser path in `plus.rs` |
| PCHIP (`src/pchip.rs`, Fritsch-Carlson + SciPy endpoints) | PORT (room in flight; reuse for `bem.rs`/`hoa.rs` grid interpolation) |
| Phase-matched stereo split (Møller et al. 2007, `synthesize_stereo_phase_matched_ir`) | PORT for per-ear HPEQ/binaural; gate: < 35 us dichotic group delay, 50-2000 Hz |
| Complex-IR mixed-phase inversion (`src/complex_eq.rs`: Hatziantoniou-Mourjopoulos, Radlovic, Kirkeby-Nelson, Gomez-Bolanos) | PORT only for the DFE/HRTF-inversion papers |
| IEC 61260-1 third-octave filters + `--verify-with` diagnostics | PORT (iso3382, in flight) |
| Neumaier compensated summation (`accum.rs`) | PORT (iso3382, in flight) |
| Static scalar FIR ceiling scaling | CONDITIONAL: only if BRIR/HpTF export lands |
| ITU-R BS.1770-5 x32 bank | CONDITIONAL: audit limiter; add only if x4/x16 is all we ship |
| ISO 226:2023 equal-loudness | SKIP: EQ feature, no paper requires it |
| Vicanek PEQ / parallel-IIR export | SKIP: EQ export |
| Lean 4 proofs | SKIP for shipping: not paper-required; do not add Mathlib |
| Tier 3 (limiter, dither, plotting, `--experimental-phase`) | DO NOT TOUCH |

## Wave 3 — papers

**Track 1: room acoustics and validation**
- Lundeby, Vigran, Bietz & Vorlaender (1995), Acustica 81:344-355: iterative cross-point,
  Chu noise subtraction, truncation correction; IEC 61260 class-1 time-reversed Butterworth.
- Tier-A shoebox presets wired to `--room` (CLI/GUI); OpenAIR and dEchorate validation gates.
- Passive frequency-dependent FDTD boundaries: Kowalczyk & van Walstijn (JAES 2008; IEEE
  TASLP 2011); Bilbao & Hamilton (2016/2017). Foster/passive RLC admittance per boundary voxel.
- Colorless FDN: Dal Santo, Prawda, Schlecht & Valimaki (DAFx 2023; EURASIP JASMP 2024).
- GFDN per-band absorption: Das, Dal Santo, Schlecht, Valimaki & Cvetkovic
  (arXiv:2508.06686).

**Track 2: binaural, HRTF, perception**
- MagLS phase continuation: Hold, Meyer-Kahlen & Pulkki (2023).
- BiMagLS: Engel, Goodman & Picinali (2021/2022); diffuse-field equalisation (JAES 2024).
- Reverberant externalisation: Li, Baumgartner & Peissig (2020), Acta Acustica 4, 21,
  DOI 10.1051/aacus/2020020 (Eqs. 1-9).
- Full-sphere localisation: Barumerli et al. (2023), Acta Acustica 7, 12; 2026 extension,
  Acta Acustica 10, 89.
- SOFA AES69-2022 I/O; Mesh2HRTF/NumCalc interoperability; Pinna2HRTF.
- BEZ2018a Dec 2023 update: Bruce, Buller & Zilany, Acoustics 2023.

**Track 3: compute**
- ETH Zuerich optimal-Q Ozaki-II: arXiv:2609.37693 (FP4 59 GEMMs, FP8 30 GEMMs, INT8
  moduli <= 239, reduce-then-combine epilogue).
- Uchino, Ozaki & Imamura: arXiv:2504.08009, 2602.02549, 2603.10634.
- `CUDA_PERF_PLAN.md` sections 2-5; FCC FDTD (Hamilton & Bilbao).

**Track 4: scan completeness**
- Wire the remesh budget report through `scan.rs:1210` and the CLI.
- Tiled micro-depth above 262,144 pixels; optional AprilTag3 (tag36h11) payload decoding.

## Wave 4 — docs and final gate

- Reconcile README, GUI text, `docs/ACADEMIC_NUKE_ACOUSTICS.md`,
  `docs/G6ME_SCAN_FMM_BEM_NTT.md` with the landed code and counts.
- Gate: `cargo fmt --check`; `cargo clippy --workspace --all-targets --features cuda -- -D
  warnings`; workspace tests; `g6me-verify --require-all`; `lean/check_axioms.sh`.
- Parity matrix from `parity-audit` must show every production stage with CPU + GPU +
  parity test; every remaining one-sided item listed with an owner.

## Sources

`/mnt/aosp/Projects/a16re/analysis/05-g6me-gap-checklist.md` section 9 holds 42 verified
URLs for the paper map. Fetch full texts via scrapling proxy `127.0.0.1:2080`; store under
`/home/sanjxz/.cache/g6me-papers/<topic>/`.

## Checkpoint policy

Commit and push after each wave (user instruction, 2026-11-28). WIP commits may not compile;
they are safety snapshots.
