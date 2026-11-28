# claims-audit handoff

Snapshot: 2026-11-28, branch `main`, HEAD `e94322a`, uncommitted working tree
(concurrent acoustics/cuda fixers active). No commit or push was made.

## Verifier / Lean / docs items completed

1. `asm_guard` hardened (`crates/g6me-verify/src/checks.rs`):
   - required-module allowlist with instruction floors: stn 1500, enhance 1200,
     unmask 1000, upmix 250, plus 7000, limiter 800, room 5000, scattering 500,
     ntt 80, dsp 400; zero matching blocks is a Fail, below floor is a Fail.
   - sibling `g6me` CLI is required under `--require-all` (Skip otherwise).
   - Rust-side FMA sabotage control: new `crates/g6me-verify/src/bin/fma_probe.rs`
     (`#[target_feature(enable = "fma")]` + `mul_add`), scanned by `asm_guard`;
     absent probe = Fail under `--require-all`.
   - parser unit tests `module_patterns_are_length_prefixed_paths` and
     `module_parser_counts_blocks_instructions_and_fma`; sabotage check confirmed
     the parser test discriminates (`instrs += 2` produced (1,6,1) vs expected
     (1,3,1), then reverted).
2. `--require-cuda` added (`crates/g6me-verify/src/main.rs`); `--require-all`
   implies CUDA strictness only when built with `--features cuda`. Help text and
   module docs updated. `crates/g6me-verify/Cargo.toml` gained
   `default-run = "g6me-verify"` for the second binary.
3. Kulisch strengthened: randomized catastrophic-cancellation case (64 cases,
   +2^53 / unit terms / -2^53 at 2^-30, exact i128 reference) that also fails if
   the corpus does not discriminate. Tautological `reference_nsim_self` removed
   from `audit_cochlea` (no occurrences remain under `crates/`). `contraction()`
   now carries an explicit note that it tests the compiler/build flags, not DSP.
4. Lean CI: `lean/lean-toolchain` pinned to `leanprover/lean4:v4.27.0`; new `lean`
   job in `.github/workflows/ci.yml` installs elan and runs `./check_axioms.sh`;
   `lean/README.md` updated to match. YAML parses (jobs: linux, windows, linux-v3,
   lean, windows-gnu).
5. Docs honesty in `docs/ACADEMIC_NUKE_ACOUSTICS.md`:
   - `:17` Duda-Martens: removed "double-precision accuracy"; documents old
     `ka+32` cap error 1.4e-3 (1 kHz) / 6.8e-4 (4 kHz) at r=1.25a and the measured
     remaining 2.2e-9 at 1 kHz vs the 60-digit mpmath reference.
   - `:59`, `:71` FDN: renamed Orthogonal-Scattering FDN; states the matrix is
     paraunitary but the Schlecht-Habets paraunitary absorption-filter design is
     not implemented (one-pole HF + broadband gain).
   - `:87` 28-CF -> 24-CF (matches `cochlea.rs:31` and all
     `compute_neurogram(..., 24)` call sites).
   - `:98` "full IEEE-754 f64" -> "Emulates f64-precision ... not bit-for-bit".
   - `:108` Matsuoka: arXiv:2606.23698 (abstract fetched) is a Bailey six-step
     3-D transform; the crate implements the 4-step N1 x N2 factorization, now
     stated explicitly.
   - Ray orders (4-48 Preview / 4-72 Production), MagLS (1.2-2.2 kHz), Preview
     (500 Hz/0.115 m; 275 Hz/0.209 m) and Production (670 Hz/0.0858 m;
     437 Hz/0.132 m) were already correct and were re-verified against
     `room.rs:1714-1716`, `hoa.rs:275-276`, `plus.rs:1064-1065` and
     `room.rs` dims/`for_room_budget`.
6. `README.md:261` room-quality row now states per-room values (preview 500 Hz
   control-room / 275 Hz theater + 2,048 rays; production 670 Hz control-room /
   437 Hz theater + 50k rays), matching the code.

## Files changed by this stream

- `crates/g6me-verify/src/checks.rs`
- `crates/g6me-verify/src/main.rs`
- `crates/g6me-verify/Cargo.toml`
- `crates/g6me-verify/src/bin/fma_probe.rs` (new)
- `lean/lean-toolchain` (new)
- `.github/workflows/ci.yml`
- `lean/README.md`
- `crates/g6me-core/src/scattering.rs` (two excessive-precision test literals
  truncated; python3 confirms bit-identical f64: `True True`)
- `crates/g6me-core/src/externalization.rs` (net zero: two
  `#[allow(clippy::needless_range_loop)]` attributes were added, then removed on
  instruction; no allow remains. File is acoustics-owned; do not edit it.)
- `docs/ACADEMIC_NUKE_ACOUSTICS.md` (`:17`, `:59`, `:71`, `:87`, `:98`, `:108`)
- `README.md` (`:261`)
- `docs/handoffs/claims-audit.md` (this file)

## Evidence (last results; note staleness)

| command | result | exit |
|---|---|---|
| `cargo fmt --all --check` | clean | 0 |
| scoped `cargo clippy --locked -p g6me-verify --all-targets --features cuda -- -D warnings` | only `externalization.rs:498` (env_r) and `:537` (x_r); verifier crate clean | 101 |
| `cargo test --locked -p g6me-verify` | 15 passed, 0 failed (116.5 s), before latest acoustics edits | 0 |
| `cargo test --locked -p g6me-core --lib scattering` | 5 passed, 1 failed: `near_field_series_matches_high_precision_reference` 2.1675195318830834e-9 > 1e-9 | 101 |
| `cargo build --locked --release -p g6me-verify -p g6me-cli` | finished | 0 |
| `target/release/g6me-verify --group asm --require-all` | 18 passed, 0 failed, 0 skipped; module rows stn 17/6938, enhance 12/6472, unmask 11/6937, upmix 12/4663, plus 38/43349, limiter 12/4514, room 57/26054, scattering 10/2530, ntt 6/1508, dsp 1/938, all 0 FMA; verify 366/171331 and CLI 1011/258185 0 FMA; fma_probe caught 1 FMA; contract_fast 64 FMA | 0 |
| `target/release/g6me-verify --group blackwell` | 7 passed; randomized 64 cases, 0 Kulisch mismatches, naive f64 wrong 63/64 | 0 |
| `target/release/g6me-verify` (all groups) | 140 passed, 1 failed: cochlea Baumgartner "Studio (0.178) & ControlRoom (0.482) > Pure (0.180)" | 1 |
| `target/release/g6me-verify --group cuda --require-cuda` (default build) | 1 failed as designed: "built without --features cuda" | 1 |
| `cd lean && ./check_axioms.sh` | PURITY GATE: PASS, 266 theorems, axioms ['Quot.sound','propext']; log `target/lean-gate.log` | 0 |

RETRACTED: an earlier full-workspace clippy exit 0 was obtained while the two
`#[allow(clippy::needless_range_loop)]` attributes were present. It is NOT valid
green evidence and must not be reported as such.

The release binaries in `target/release` predate the latest acoustics edits;
rebuild before re-running asm/blackwell.

## Held gate and blocker

Workspace gate is held:
`cargo clippy --locked --workspace --all-targets --features cuda -- -D warnings`
still exits 101 on `crates/g6me-core/src/externalization.rs:498` and `:537`.
acoustics-audit owns the `.iter().enumerate()` conversion (keep the
`j >= 0 && (j as usize) < n` guard and summation order; do not reintroduce
allows). Also open: the `scattering` test above and the cochlea Baumgartner gate.
The lead is routing both. Do not re-run/claim the workspace gate until acoustics
confirms the file is quiet and the Baumgartner gate passes.

## Remaining README items

- `README.md:32` and `:45` still say "129 checks" and read as an end-to-end
  bit-identical claim. Scope them to the `bitexact` downmixer comparison when the
  suite count is measured, e.g. "the f64 downmixer is bit-identical to binaries
  compiled from the vendored C reference; `g6me-verify`'s `bitexact` group ... the
  full suite is N checks, all passing." (This edit was ordered before the stop but
  is NOT applied in the tree; the stop message assumed it was done.)
- Hold the "129 checks" -> current-count update until acoustics lands. Current
  measured total is 141 rows (140 pass + 1 fail; 141 passed expected when green).
  Occurrences to update: `:32`, `:45`, `:197`, `:637`, `:766`, `:767`.
- The no-sibling sentence at `:644-645` ("reports 128") is stale: with the new
  asm rows a bare `cargo run` should report 129 passed / 3 skipped (CLI, module
  coverage, fma_probe) — measure before writing.
- `:646` CUDA count paragraph ("159 checks") belongs to cuda-parity-audit; check
  it against the final measured cuda-feature total rather than duplicating it.
- The CUDA end-to-end wording is already tolerance-level and owned by
  cuda-parity-audit; do not rewrite it.

## Docs lines corrected vs still acoustics-owned

Corrected by this stream: ACADEMIC `:17`, `:59`, `:71`, `:87` (24-CF), `:98`,
`:108`; README `:261`.

Still acoustics-owned, do not edit until their fix lands:
- `docs/ACADEMIC_NUKE_ACOUSTICS.md:88` strict-beat Baumgartner claim; keep it
  until the verifier gate passes.
- Code comments/help text with the stale generic "production 670 Hz / dx <= 8.5 cm"
  claim: `room.rs:77-80`, `room.rs:107`, `g6me-cli/src/main.rs:177`,
  `g6me-gui/src/main.rs:1572-1573`, `plus.rs:1061`.
- `scattering.rs:9` and `:115` "< 1e-15" convergence claims (measured remaining
  error is 2.2e-9 at 1 kHz for r=1.25a).
- The 24-channel / 28-CF code-side naming is otherwise consistent; the doc is now
  24-CF.

## Exact first action for successor

1. Wait for team-lead/acoustics confirmation that `externalization.rs` is
   converted and the Baumgartner gate passes.
2. Run, capturing exit codes:
   `cargo fmt --all --check`;
   `cargo clippy --locked --workspace --all-targets --features cuda -- -D warnings`;
   `cargo test --locked -p g6me-verify`;
   `cargo build --locked --release -p g6me-verify -p g6me-cli`;
   `target/release/g6me-verify --group asm --require-all`;
   `target/release/g6me-verify --group blackwell`;
   `target/release/g6me-verify` (record the final count);
   `(cd lean && ./check_axioms.sh)`.
3. Update the README count/scoping items above from the measured values, then send
   the final evidence table. Do not touch `externalization.rs`; do not commit or
   push.

Tree state: uncommitted `main` with concurrent acoustics/cuda edits; workspace
clippy held on two externalization loops; Lean PASS; docs-honesty edits applied;
README count pending.
