# CUDA Parity Fix — Handoff (`cuda-parity-audit`)

Date: 2026-11-28. Base HEAD: `e94322a`. No commits, no pushes.
Standing standard applied: paper-literal, no silent CPU fallback behind a CUDA banner.

## Seven mandate items — final status

1. **P0 `--cuda` inert in default `reference` mode — DONE (warning path, not auto-switch).**
   The reference renderer has no GPU-capable stage, so there is nothing to select: switching
   to `--mode plus` automatically would change the DSP renderer, not just the device. Instead
   `cuda_idle_reason()` (`crates/g6me-cli/src/main.rs:1248`) emits
   `warning: --cuda requested but this mode has no GPU work — … GPU stages run: 0.` at startup,
   and the end-of-run report prints `[cuda] GPU stages run: 0 (this configuration has no GPU work)`.
   Tests: `cuda_with_the_default_reference_mode_says_it_has_no_gpu_work`,
   `cuda_with_plus_mode_has_gpu_work` (both pass).
2. **P0 README/docs claims — DONE.** End-to-end "bit-identical" claim removed; replaced with
   measured tolerance (1.4e-4 synthesis order) plus a native-peer architecture paragraph.
   CUDA suite count corrected 146/147 → 159 (measured: 158 passed + 1 failed = 159 checks).
   `docs/ACADEMIC_NUKE_ACOUSTICS.md` and `docs/G6ME_SCAN_FMM_BEM_NTT.md` carry the same
   production-path wording (65,610 voxels × 128 steps, GPU diagonal asserted, e2e at 1e-3).
3. **P1 `--head-mesh` silent personalization drop — DONE.** `crates/g6me-core/src/plus.rs:892`
   now retries `FmmBemSolver::new_with_cuda(&mesh, false)` on a GPU construction error and only
   drops BEM after a CPU failure, each with a visible `warning:` line and
   `record_cpu_fallback("bem-head-mesh", …)`. The earlier E0277 is gone: `Error` implements
   `Display` via `thiserror`, `.to_string()` compiles; `cargo check -p g6me-core --features cuda`
   exits 0 with 0 diagnostics.
4. **P1 verifier production-path proof — DONE (cuda group 18 → 20 checks, all passing).**
   - FMM: `FmmBemSolver::new_with_cuda(&mesh, true)` + `solid_angle_diag_on_gpu` asserted.
   - Raytrace histogram: direct `raytrace_hoa_histogram_on_cuda` entry (no fallback wrapper),
     `direct_kernel_path: true`.
   - FDTD: 65,610-voxel (54×45×27) × 128-step production grid, max diff < 1e-13.
   - End-to-end `g6me --mode plus --room control-room --cuda` vs CPU at 1e-3, asserting every
     stage reported GPU and zero fallbacks (`cuda_cli_end_to_end()`).
5. **P2 diagnostics — DONE.** New `crates/g6me-core/src/cuda_report.rs`; every offload point
   records `record_gpu` / `record_cpu_fallback` with reason; `print_summary(cli.cuda)` prints one
   `[cuda] stage <name>: GPU (n call(s); detail)` or `CPU fallback (reason)` line per stage once
   per render; Ozaki detail shows the runtime-resolved `actual_prec`; cached runtime-init failure
   is recorded once (`record_init_failure`) and printed once.
6. **P3 PCG rename — DONE.** `RayPcg64` → `RayPcg32` (`pcg32_next` / `pcg32_uniform`) in
   `room.rs`, `raytrace_hoa.cu`, `g6me-cuda/src/lib.rs`, `g6me-gui/src/main.rs`; XSH-RR comment
   corrected; the separate true `Pcg64` (u128 LCG, XSL-RR 128/64) doc corrected.
7. **No six-kernel contract regression — VERIFIED.** 20/20 cuda checks include 0-FMA PTX/SASS
   audits, `--fmad=true` sabotage control, per-arch Ozaki opcodes, bit-exact Goldilocks NTT,
   exact `u128` CRT Ozaki, Stockham FFT, BEM matvec, FDTD.

## Files changed (mine)

- New: `crates/g6me-core/src/cuda_report.rs`; `docs/handoffs/cuda-parity-audit.md` (this file).
- `crates/g6me-core/src/lib.rs` (`pub mod cuda_report;`)
- `crates/g6me-core/src/room.rs` (FDTD/raytrace dispatch + recording, `raytrace_hoa_histogram_on_cuda`, PCG32 rename)
- `crates/g6me-core/src/bem.rs` (`solid_angle_diag_on_gpu`, GPU/CPU diag recording, matvec recording)
- `crates/g6me-core/src/blackwell.rs` (Ozaki recording + actual precision, init-failure recording, new precision API)
- `crates/g6me-core/src/plus.rs` (NTT/FFT recording, head-mesh BEM fallback)
- `crates/g6me-cli/src/main.rs` (reference-mode idle warning, end-of-run stage report, 2 tests)
- `crates/g6me-verify/src/checks.rs` (production-path checks + e2e CLI check + PCG32 rename)
- `crates/g6me-cuda/kernels/raytrace_hoa.cu`, `crates/g6me-cuda/src/lib.rs`, `crates/g6me-gui/src/main.rs` (PCG32 rename)
- `README.md`, `docs/ACADEMIC_NUKE_ACOUSTICS.md`, `docs/G6ME_SCAN_FMM_BEM_NTT.md` (claims)

Shared files also carry concurrent agents' edits (do not revert): `cochlea.rs`,
`scattering.rs` (acoustics), `externalization.rs` (new, acoustics), `scan.rs`/`zbc.rs`
(scan), `checks.rs`/`g6me-verify/*` (claims-audit), `.github/workflows/ci.yml`, `lean/*`.

## Latest commands and results

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy --workspace --all-targets --features cuda --target-dir target-main -- -D warnings` | exit 0, 0 diagnostics (`/tmp/g6me_clippy4.log`) |
| `cargo clippy -p g6me-verify --all-targets --features cuda -- -D warnings` | exit 0, 0 diagnostics |
| `cargo check -p g6me-core --features cuda` | exit 0, 0 diagnostics |
| `cargo build --release --workspace --features cuda` | Finished in 1m 56s, 0 errors |
| `target-main/release/g6me-verify --group cuda` | **20 passed, 0 failed, 0 skipped** (`/tmp/g6me_verify_final.log`) |
| full `target-main/release/g6me-verify` (sibling present) | 158 passed, 1 failed, 0 skipped — only `cochlea / Baumgartner-Majdak Bayesian externalization` (`/tmp/g6me_verify_full.log:173`) |
| `cargo test -p g6me-core --features cuda --lib` | 305 passed, 3 failed, 2 ignored in 278.84 s (3 failures are acoustics-owned, below) |
| `cargo test -p g6me-cli -p g6me-verify -p g6me-cuda -p g6me-gui --features cuda` | CLI 27 passed, CUDA 4 passed, GUI 43 passed; g6me-verify bin 14 passed + `suite::verification_suite_passes` failed on the same cochlea check |
| `cargo test -p g6me-cli --features cuda cuda_with_` | 2 passed, 25 filtered |

CLI smoke (release binary, `target-main/release/g6me`):

- Reference mode, `--cuda`:
  `warning: --cuda requested but this mode has no GPU work — the reference renderer has no GPU stages; pass --mode plus (with --room control-room/theater or --head-mesh for room/BEM synthesis) to offload G6+ work. GPU stages run: 0.`
  and `[cuda] GPU stages run: 0 (this configuration has no GPU work)`; exit 0.
- `--cuda --mode plus --room control-room --room-quality preview --format wav --depth f32`
  on a 5.1 input: exit 0, `[cuda] stage fdtd: GPU (2 call(s); 52x43x26 voxels, 900 steps)`,
  `[cuda] stage fft: GPU (138 call(s))`,
  `[cuda] stage raytrace-hoa: GPU (11 call(s); 2048 rays x 48 bounces)`,
  `[cuda] GPU stages: 3; CPU-fallback stages: 0`.

## Precision API for scan-audit (implemented, messaged)

`crates/g6me-core/src/blackwell.rs`:

```rust
pub fn ozaki2_crt_gemm_with_cuda_precision(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    backend: TensorBackend,
    use_cuda: bool,
    precision: CudaTensorPrecision,
) -> (Vec<Vec<f64>>, Option<CudaTensorPrecision>)
```

`.1 = Some(actual)` when the device ran the multiply (runtime-resolved format, never merely the
requested one); `.1 = None` when the CPU reference path ran. `ozaki2_crt_gemm_with_cuda` remains
and delegates. Signature sent to `scan-audit` (message `025d75c3`); `scan.rs` was not edited.

## Open items

1. **Acoustics rewrite red tests (owned by acoustics-audit; do not chase).**
   - `cochlea::tests::baumgartner_majdak_externalization_ranks_room_above_dry_in_head` —
     "dry mono impulse should score low externalization (<0.20), got 0.487" (`cochlea.rs:526`)
   - `plus::tests::academic_nuke_hankel_room_slepian_externalization_integration` —
     "ControlRoom externalization (0.195) must exceed Pure (0.488)" (`plus.rs:3802`)
   - `scattering::tests::near_field_series_matches_high_precision_reference` —
     "near-field series error at 1000 Hz: 2.1675195318830834e-9" (`scattering.rs:489`)
   - verifier `cochlea / Baumgartner-Majdak Bayesian externalization` —
     "expected Studio (0.178) & ControlRoom (0.482) > Pure (0.180)"
2. **Native-peer violation (accepted by team-lead; NOT fixed — stop order).**
   `raytrace_hoa_histogram` (`crates/g6me-core/src/room.rs:1320`) is the production CPU fallback
   and calls `trace_single_ray` (`room.rs:1164`; parallel via `par_map_indexed` at 1337, serial
   at 1524). `raytrace_hoa.cu:10` states the CPU tracer "mirrors every floating-point operation,
   PCG32 draw and branch below line for line", so the bit-identical histogram gate passes by
   construction, not independent agreement. Proposed follow-up stream owning `room.rs` after
   acoustics: write a native CPU tracer (its own sampling/reduction order), relax the histogram
   gate to a measured tolerance, update README exception wording.
3. **Full workspace test tally pending.** `cargo test --features cuda` was last run before
   acoustics' latest edits: 305/3/2 on core. Re-run after acoustics declares done; if any of the
   three persists, escalate with the assertion text.
4. **Stale README counts to reconcile after the rewrites** (not CUDA parity): the
   workspace-test paragraph still says 375/385 tests, core 290/300, CLI 25; measured now are
   core 310 total (incl. 2 ignored), CLI 27, GUI 43, CUDA 4, verifier bin 15. The non-CUDA
   verifier counts ("129 checks", "reports 128") were not re-measured. Note the debug
   `suite::verification_suite_passes` reported 151 checks (sibling absent, 4 skips) while the
   release verifier with sibling reports 159 — reconcile before quoting either.
5. **Architecture refinement (bit-exact by specification, cite papers).** README was accepted
   with measured tolerances + the raytrace exception. Remaining CUDA-scope work is covered by
   item 2 and any future parity gate that must be justified against the source paper
   (DOI/arXiv/standard + equation), not blogs.

## First action for the successor

Wait for `acoustics-audit` to declare its branch finished, then run
`cargo test --features cuda --target-dir target-main` across the workspace and
`target-main/release/g6me-verify`; record the tally. If a failure other than the four
acoustics items appears, treat it as real and escalate to team-lead with the assertion text.
Then re-run `cargo fmt --all -- --check` and the workspace clippy gate and reconcile item 4.
Do not revert concurrent agents' edits; do not commit or push without instruction.

Tree state at handoff: 27 modified/untracked entries, working tree compiles
(`cargo check --workspace --all-targets --features cuda` and release build both clean);
only the acoustics-owned tests are red.
