# Session state — g6me fix wave + CUDA/room research

Snapshot: 2026-11-28 ~07:05 local. Branch `main`, HEAD `e94322a`.
Written because the operator is restarting Claude Code (possibly at a higher agent limit).

## Working tree (uncommitted by design — do not commit without explicit instruction)

Modified: `.github/workflows/ci.yml`, `README.md`, `crates/g6me-cli/src/main.rs`,
`g6me-core/src/{bem,blackwell,cochlea,lib,plus,room,scattering}.rs`,
`g6me-cuda/kernels/raytrace_hoa.cu`, `g6me-cuda/src/lib.rs`, `g6me-gui/src/main.rs`,
`g6me-verify/{Cargo.toml,src/checks.rs,src/main.rs}`,
`docs/ACADEMIC_NUKE_ACOUSTICS.md`, `docs/G6ME_SCAN_FMM_BEM_NTT.md`, `lean/README.md`.

Untracked: `crates/g6me-core/src/cuda_report.rs`, `crates/g6me-core/src/externalization.rs`,
`crates/g6me-verify/src/bin/`, `docs/CUDA_PERF_PLAN.md`, `docs/ROOM_PRESETS_RESEARCH.md`,
`lean/lean-toolchain`, `docs/handoffs/`.

## Active workstreams and last known state

| Stream | State | Handoff |
|---|---|---|
| acoustics-audit (ZBC/BEZ, AMT externalization, snowman, Duda, paraunitary naming) | Long pole. Mid-rewrite of `cochlea.rs`, `scattering.rs`, new `externalization.rs`. | `docs/handoffs/acoustics-audit.md` |
| cuda-parity-audit (inert `--cuda`, README, head-mesh fallback, verifier blind spots, precision API) | Seven items implemented; CUDA verifier group 20/20; release build clean. | `docs/handoffs/cuda-parity-audit.md` |
| claims-audit (asm floors, `--require-cuda`, Kulisch, Lean CI, docs honesty) | Verifier/Lean/docs green except acoustics-owned failures. | `docs/handoffs/claims-audit.md` |
| scan-audit (DNG 1.7.1, AprilTag/ChArUco, 2DGS/PGSR, OIS, remesh) | Isolated worktree `/mnt/aosp/Projects/g6me-scan-fix`, branch off `e94322a`. | `docs/handoffs/scan-audit.md` |

## Open blockers (exact)

1. `crates/g6me-core/src/externalization.rs` loops at `:498` and `:537` are `for i in 0..n`;
   workspace clippy `--features cuda -- -D warnings` exits 101 on both. Convert to
   `.iter().enumerate()`, keeping the `j >= 0 && (j as usize) < n` guard and summation order.
   The two `#[allow(clippy::needless_range_loop)]` attributes were removed; do not reintroduce.
2. `cargo test -p g6me-core --lib scattering`:
   `near_field_series_matches_high_precision_reference` fails at 1000 Hz with
   `2.1675195318830834e-9` > `1e-9` (adaptive-cutoff change in `rigid_sphere_green`).
3. Verifier Baumgartner gate: `expected Studio (0.178) & ControlRoom (0.482) > Pure (0.180)`;
   Studio is 0.178, below Pure. Make the AMT implementation match the paper or prove the
   expectation wrong; do not rewrite the gate or the doc claim.
4. `docs/ACADEMIC_NUKE_ACOUSTICS.md:88` still claims the strict beat until (3) is resolved.
5. README "129 checks" count and the CUDA check-count paragraph are stale; update after the tree
   is green. README "bit-identical end to end" must be replaced with the measured tolerance.
6. **Native-peer violation (new, not yet fixed):** the production CPU raytrace fallback is a
   line-for-line mirror of the CUDA kernel. `raytrace_hoa_histogram` (`room.rs:1320`) calls
   `trace_single_ray` (`room.rs:1164`, `:1337`), which `raytrace_hoa.cu:10` itself describes as
   mirroring every floating-point operation, PCG32 draw and branch. The bit-identical histogram
   gate therefore passes by construction, not by independent agreement. Required: a native CPU
   tracer with its own sampling/reduction order, and the histogram gate downgraded to a measured
   tolerance. `room.rs` is acoustics-owned; assign to a dedicated follow-up agent after the
   acoustics patch lands. README already flags this as pending, and the end-to-end
   "bit-identical" claim is gone (replaced by measured tolerances: FDTD 1e-13, BEM matvec 1e-13,
   FMM-BEM 1e-10, Stockham 1e-11, end-to-end 1e-3; exact only for Goldilocks NTT and Ozaki-II
   CRT).

## Established evidence (from agent reports; re-run to confirm after restart)

- `cargo fmt --all --check` exit 0.
- `cargo clippy --locked -p g6me-verify --all-targets --features cuda -- -D warnings` exit 101,
  only the two externalization loops.
- `cargo test -p g6me-verify`: 15 passed (116.5 s), before the latest acoustics edits.
- Full verifier: 140 passed, 1 failed (Baumgartner gate above).
- `g6me-verify --group asm --require-all`: 18 passed, 0 failed, 0 skipped; both FMA sabotage
  controls fired (Rust `fma_probe` 1 FMA, C `contract_fast` 64 FMA).
- `g6me-verify --group blackwell`: 7 passed; Kulisch randomized 64 cases, 0 mismatches, naive
  f64 wrong on 63/64.
- `lean/check_axioms.sh`: PASS, 266 theorems, axioms `['Quot.sound','propext']`.
- CUDA verifier group: 20 passed / 0 failed (cuda-parity-audit, before final rebuild).

## Architecture rules in force (operator, this session)

- CPU and GPU are **native peers** implementing the same feature on different hardware; neither
  emulates the other's internals. Production CPU fallbacks are real CPU algorithms; bit-exact
  mirrors are test-only oracles.
- **Bit-exactness is still a first-class goal**, achieved by specification: same algorithm,
  same operation order where the paper fixes it, strict IEEE ops (zero-FMA, no fast-math), same
  precision and summation order. Integer-exact stages must be bit-identical; float stages are
  driven to the smallest reproducible difference and gated at that bound.
- Every algorithm/spec claim must be checked against the **actual paper or standard**, fetched
  and cited by DOI/arXiv/standard number plus equation/section. Blogs and dataset cards are
  discovery aids only.
- Paper-literal naming: a paper's name may only appear on an implementation of that paper.
  Intentional approximations are renamed, not misattributed.
- No commit or push without explicit operator instruction.

## New research artifacts

- `docs/CUDA_PERF_PLAN.md` — B1–B7 baseline, fused NTT/UPOLA, graph capture, GPU GMRES,
  AN5D FDTD, fused cp.async Ozaki, cutile backend, Blackwell extras, cuBLAS emulated-FP64 probe,
  plus the "Native peers, not emulation" rule.
- `docs/ROOM_PRESETS_RESEARCH.md` — measured dataset matrix (OpenAIR 31.25 Hz–16 kHz, dEchorate,
  MeshRIR, AcousticRooms, 3D-MARCo, …), shoebox preset library (Musikverein/Boston/Concertgebouw,
  control room, theater), 10-band plan, FDTD/geometric split, calibration and ISO 3382 gates.

## Next-session dispatch plan (if the limit is raised to 8)

1. Finish the four current fixers; do not start new work in their files.
2. Speed §1 — fused single-launch NTT + VRAM-resident UPOLA (`g6me-cuda/src/lib.rs`,
   `plus.rs` UPOLA only; acoustics keeps cochlea/scattering).
3. Room presets B0/B1 — 10-band materials + per-band ray/ISM (`room.rs`, CUDA kernel); requires
   `room.rs` handed over from acoustics.
4. Validation/data — mirror the OpenAIR subset, ISO 3382 metric harness, dEchorate/MeshRIR
   loaders; new files only.
5. Native CPU raytrace — replace the production `trace_single_ray` mirror with an independent
   CPU tracer and convert the histogram gate to a measured tolerance (`room.rs`); owner must
   take room.rs after acoustics.
6. Read-only reviewer — audit diffs and parity claims; no edits.

Do not run two agents in the same file. Use worktrees for scan/speed. Keep at most 6 active
workers out of 8; reserve the rest for review.

## Resume checklist

1. Read `docs/handoffs/*.md` and this file.
2. `git status --short` and `git diff --stat` — confirm the expected files only.
3. `cargo check -p g6me-core --features cuda` before any new edit.
4. Resolve the three blockers above in order (loops → scattering → Baumgartner), then re-run
   fmt/clippy/tests/asm/blackwell/Lean and the full verifier.
