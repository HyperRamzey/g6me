# Scan audit handoff — stopped by operator order

State: **work committed, verification incomplete and clippy red**. This handoff is written
after an explicit stop order; no further edits were made.

## Worktree and branch

- Worktree: `/mnt/aosp/Projects/g6me-scan-fix`
- Branch: `scan-audit-fixes` (created at base commit `e94322a`)
- Base: `e94322a feat(cuda): bit-exact production offload with exact Ozaki CRT and audit hardening`
- Commits:
  - `d2a099f` wip(cuda): local placeholder for actual-precision Ozaki-II GEMM API
  - `78c7821` feat(scan): DNG 1.7.1 conformance, report truth, honest claim names, OIS wiring
- No push, no commit to shared main.

`git diff --stat e94322a..HEAD` (before the handoff commit):

```
 README.md                         |    9 +-
 crates/g6me-cli/src/scan_main.rs  |   74 +-
 crates/g6me-core/src/blackwell.rs |   23 +-
 crates/g6me-core/src/burst3d.rs   |   44 +
 crates/g6me-core/src/dng.rs       | 1674 +++++++++++++++++++++++++++++++++----
 crates/g6me-core/src/scan.rs      |  303 ++++---
 docs/G6ME_SCAN_FMM_BEM_NTT.md     |    6 +-
 7 files changed, 1839 insertions(+), 294 deletions(-)
```

## Mandate item status

| # | Item | Status |
|---|------|--------|
| 1a | DNG 1.7.1 ActiveArea-relative CFA/BlackLevel phase | done, test + sabotage |
| 1b | Opcode composition in file order, List1/2/3 distinct | done, test + sabotage |
| 1c | GainMap area/plane/pitch/origin | done, test + sabotage |
| 1d | SampleFormat 339 float16/24/32 as float | done, test + sabotage |
| 1e | BlackLevelRepeatDim > 2x2 and DeltaH/V | done, test + sabotage |
| 1f | JXL edge-tile dimensions | done, test + sabotage |
| 2 | `schur_backend` carries actual GPU precision; `iterations` = accepted steps; false OpcodeList3-prior claim removed | done (CPU path exercised; GPU path compile-only here) |
| 3 | AprilTag3/ChArUco renames; 2DGS/PGSR renames | done everywhere editable; **bem.rs:509 and checks.rs still carry the old wording** (both were off-limits) |
| 4a | `interpolate_ois_at_scanline` dead code | wired into the CLI burst path (per-frame mid-row OIS shift) and dead `apply_ois_midframe` wrapper deleted |
| 4b | Dense micro-depth 262,144-pixel gate | tiled processing **not implemented**; CLI now prints the skip reason and the limit when it skips |
| 5 | Remesh cap parameter + target-miss report | **not done** — stop order arrived first; bem.rs untouched entirely |
| 6 | Constraints | bem.rs untouched, checks.rs untouched, cochlea/verifier/CUDA kernels untouched, no dependency changes, shared tree untouched |

### Precision-API dependency (cuda-parity-audit)

`cuda-parity-audit` owns `blackwell.rs` in the shared tree and has published the signature:

```rust
pub fn ozaki2_crt_gemm_with_cuda_precision(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    backend: TensorBackend,
    use_cuda: bool,
    precision: CudaTensorPrecision,
) -> (Vec<Vec<f64>>, Option<CudaTensorPrecision>)
```

`78c7821` consumes exactly that API in `scan.rs` (single call site in
`bundle_adjust_ozaki2_fp4_schur_with_cuda`). `d2a099f` is a local compile placeholder with the
same signature and must **not** be integrated when their version is present.

## Verification actually run (and what was not)

Environment: Rust 1.100.0 (from the clippy URL), unoptimized test profile, no CUDA feature.

- `cargo fmt --all` — succeeded; it rewrote the files before commit.
- `cargo clippy -p g6me-core -p g6me-cli --all-targets -- -D warnings` — **FAILED**, 4 errors,
  all in `dng.rs` (1 lib doc lint, 3 test lints):
  - `dng.rs:2147` doc list item without indentation (lib target; blocks the lib build under
    `-D warnings`)
  - `dng.rs:3999` manual implementation of `.is_multiple_of()`
  - `dng.rs:4427` and `dng.rs:4462` `needless_range_loop` (new test code)
  - Compilation aborts on the lib error, so the CLI target was not fully linted.
- `cargo test -p g6me-core --lib dng::` — 12 passed, 0 failed (run before the final
  honesty-report text edit; the edit changed only a string literal).
- `cargo test -p g6me-core --lib scan::` — 6 passed, 0 failed (run before the AprilTag/2DGS
  renames, the OIS wrapper deletion, and the `SchurBackend` display change; not re-run).
- `cargo test -p g6me-core --lib glomap_so3` — passed with 7 of 14 LM steps accepted.
- Sabotage evidence for the accepted-step counting: reverting `iterations: accepted_steps` to
  `iters` made the test fail with `got 14`.
- Each of the six DNG fixes was sabotage-verified in the earlier session (test fails without
  the fix, passes with it); the exact failure text was not preserved in this handoff.
- **Never executed:** the new `burst3d::tests::ois_interpolation_tracks_the_rolling_shutter_trajectory`,
  the full `cargo test -p g6me-core -p g6me-cli`, and any `g6me-scan` smoke run on a synthetic
  gate-passing burst. Do not read this handoff as a green build.

## Files deliberately untouched

- `crates/g6me-core/src/bem.rs` (including the `2DGS/PGSR` comment at line 509)
- `crates/g6me-verify/src/checks.rs` (including its `schur_backend == TensorBackend::Fp4E2m1Base13`
  comparison, which still compiles against the new `SchurBackend` via
  `impl PartialEq<TensorBackend>`, and its 2DGS/PGSR report strings)
- `crates/g6me-core/src/cochlea.rs`, `crates/g6me-cuda/**`, all other verifier sources
- `/mnt/aosp/Projects/g6me` (shared tree)

## Integration steps

From the integration branch (on top of the other fixers' commits, at or after `e94322a`):

```sh
git cherry-pick 78c7821
# If cuda-parity-audit's ozaki2_crt_gemm_with_cuda_precision is already present:
#   do NOT cherry-pick d2a099f (placeholder, duplicate definition).
# If it is absent:
git cherry-pick d2a099f
```

Expected conflicts: `dng.rs` if claims-audit also edited it; `scan.rs`/`scan_main.rs` if another
fixer touched the scan pipeline. The `README.md` and `docs/G6ME_SCAN_FMM_BEM_NTT.md` hunks are
text-only and conflict-prone in the same regions other claim fixes touch.

Before integrating, finish the red items:

1. Fix the four `dng.rs` clippy lints (doc indentation at 2147, `is_multiple_of` at 3999,
   `needless_range_loop` at 4427 and 4462).
2. Run `cargo test -p g6me-core -p g6me-cli` and the synthetic-burst `g6me-scan` smoke.
3. Remesh cap work (mandate item 5) is unimplemented; `isotropic_acoustic_remesh` still has its
   fixed 16,384-face cap and no target-miss report.

A pre-rename backup of the fixed `dng.rs` also exists at
`/home/sanjxz/.cache/g6me-audit/dng_fixed.rs` (outside the worktree, not needed for integration).
