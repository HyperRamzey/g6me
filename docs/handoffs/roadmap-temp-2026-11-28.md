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
| 1 | `room.rs`, `checks.rs` | E2E CUDA stage gate must fail on absent/fallback stage; per-band `e_rel` floor; double ISO 9613-1 air; `band_value_at` 125 Hz step -> PCHIP; test index labels; CPU<->CUDA parity | IN FLIGHT (fix-room-gate) |
| 2 | `zbc.rs` | tail padding ramp; decimator 31-sample phase; PLA/chirp gains vs BEZ2018a C source; SPL calibration; integer step counter; 3 `excessive_precision` lints | IN FLIGHT (fix-zbc) |
| 3 | `iso3382/{metrics,filterbank}.rs` | Eq. A.14 LF denominator; B-format W/Y; IEC 61260 class-1; zero-phase energy loss; docstring drift; Neumaier | IN FLIGHT (fix-iso3382) |
| 4 | `externalization.rs` | invented ITSD -> paper-literal 7 cues; pinna template root cause; Eq. 7 clamp; true Butterworth; 2021 citation | IN FLIGHT (fix-extcues) |
| 5 | `dng.rs`, `bem.rs` remesh | GainMap plane offset; float denom + HDR clip; remesh budget + exact `target_met`; clippy already fixed at HEAD; stale strings reported | DONE (fix-dng-scan); residual: wire report through `scan.rs:1210` |
| 6 | `scattering.rs` | `r_hi` cutoff; scaled Miller `J_m`; mpmath verification | IN FLIGHT (fix-scattering) |
| 7 | `plus.rs`, CUDA UPOLA | rollback-before-commit; single inverse + single D2H | IN FLIGHT (fix-upola) |
| 8 | parity | full CPU/GPU stage + kernel + feature matrix | IN FLIGHT (parity-audit) |
| 9 | docs drift (P2) | README/gui/`ACADEMIC_NUKE_ACOUSTICS.md`/`SCAN_FMM_BEM_NTT.md`: PCG32, "native CPU tracer pending", 2.2e-9, gammatone, counts; `bem.rs:509`, `checks.rs:2695/3120/3242/3255` strings | Wave 4 |

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
