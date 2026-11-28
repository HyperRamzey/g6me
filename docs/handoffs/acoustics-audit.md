# Acoustics audit fix — handoff (stop order, 2026-11-28)

## Mandate

Audit then fix the acoustics/cochlea scope of `/mnt/aosp/Projects/g6me` under the
paper-literal standard: anything named after a paper must implement that paper's
specification with equations cited; an intentional approximation may not carry the paper's
name. Confirmed items: ZBC 2014 + BEZ 2018 cochlea; Baumgartner & Majdak 2021 (8 cues, WSM
0.6 MSS / 0.4 ISS, Eq. 1–10, JNDs, LTA/MTA/WTA); NSIM 3×3 Gaussian window radius 0.5;
28→24 channels; BALD oracle de-circularised; confirmed scattering/room/FDN deviations.
Verification required: `cargo fmt`, `clippy -D warnings`, `cargo test` for touched crates,
`g6me-verify --group acoustics --group cochlea`. Read-only was lifted only for this scope.
Do not touch scan or CUDA. `docs/ACADEMIC_NUKE_ACOUSTICS.md` and `README.md` are owned by
claims-audit.

## Files changed

| File | Change |
|---|---|
| `crates/g6me-core/src/externalization.rs` (new, 754 lines) | Full Baumgartner & Majdak 2021: 8 cues, Eq. 1–10 mapping, digitised sensitivities, JND-gated normalisation, 4 decision strategies, snowman-HRTF template at 30°, 2 m. |
| `crates/g6me-core/src/cochlea.rs` | Re-exports externalization API; module docs now describe the ZBC/BEZ periphery; `compute_neurogram` delegates to `crate::zbc::rate_neurogram`; 24-channel wording; NSIM Gaussian window; BALD relabelled “3 anatomy scalars”; independent BALD oracle; externalization test uses template vs dry. |
| `crates/g6me-core/src/zbc.rs` (new, ~1050 lines) | ZBC/BEZ periphery: human middle ear, C1/C2 chirp filters, wideband control path, OHC/IHC nonlinearities, cat delay, BEZ 2018 synapse (Eq. 1 mapping, 5-pole PLA, Eq. 7 redocking, 4 sites, spike generator, Eq. 9 refractoriness), fGn synthesis, 3-fiber population, 100 kHz resampling, 10:1 decimation, binned neurogram. |
| `crates/g6me-core/src/lib.rs` | `pub mod zbc;`. |
| `crates/g6me-core/src/scattering.rs` | Adaptive Legendre cutoff; high-precision test; Algazi/multipole claims replaced by a documented single-scattering approximation. |
| `crates/g6me-core/src/room.rs` | 65k/260k voxel budgets, room-dependent 500/670 Hz and 275/437 Hz, IR length from RT60; tier labels. |
| `crates/g6me-core/src/plus.rs` | `use_fdn_velvet` / FDN docs; integration test at ~3805 still asserts ControlRoom > Pure. |
| `crates/g6me-cli/src/main.rs`, `crates/g6me-gui/src/main.rs` | Room-quality, FDN and snowman wording. |
| `crates/g6me-verify/src/checks.rs` | Cochlea gate now template-vs-dry (template > 0.90, dry < 0.60, margin 0.30); BALD label; audit JSON adds template/dry metrics. |
| `docs/ACADEMIC_NUKE_ACOUSTICS.md`, `README.md` | Six numeric corrections made earlier; ownership has since moved to claims-audit — do not edit further. |

## Verification evidence (last known)

- `timeout 300 cargo check -p g6me-core --lib` → **exit 0**, `Finished dev profile ... in 2.26s`
  after the E0308 fix at `zbc.rs:397` (`output[i]` needed a 3-element history array).
  This is the **only** verification of `zbc.rs`. It type-checks; it has **never been run**.
- Earlier, before `zbc.rs`: six externalization unit tests passed; `clippy -p g6me-core --lib`
  was clean; `cargo check -p g6me-verify` was clean.
- Red at handoff (reported by team-lead / cuda-parity-audit; line numbers may have drifted):
  - `scattering::tests::near_field_series_matches_high_precision_reference` —
    “near-field series error at 1000 Hz: 2.1675195318830834e-9” (> 1e-9), `scattering.rs:489`.
  - `plus::tests::academic_nuke_hankel_room_slepian_externalization_integration` —
    “ControlRoom externalization (0.195) must exceed Pure (0.488)”, `plus.rs:3805`.
  - `g6me-verify` “expected Studio (0.178) & ControlRoom (0.482) > Pure (0.180)” — that
    condition is **no longer present** in `checks.rs` (grep finds the wording only in
    `plus.rs`); the verifier gate was replaced by template-vs-dry but has **not been re-run**.
  - `cochlea::tests::baumgartner_majdak_externalization_ranks_room_above_dry_in_head` no
    longer exists; it was replaced by `externalization_ranks_template_above_in_head_delta`.

## Remaining items, in order

1. **externalization.rs clippy**: `for i in 0..n` in `ext_itd_us` (~line 496) and `ext_ic`
   (~line 535) trigger `needless_range_loop`. Convert to `.iter().enumerate()`, preserving
   the `j >= 0 && (j as usize) < n` guard and summation order. No `#[allow]` (grep currently
   finds none in the file). Then `cargo clippy -p g6me-core --all-targets -- -D warnings`.
2. **scattering 1e-9 test**: restore enough series terms or raise the adaptive cap so the
   2.1675e-9 error at 1 kHz falls below 1e-9 over the operating range, or prove the bound.
   Do not loosen the tolerance without that proof.
3. **First run of `zbc.rs`**: `cargo test -p g6me-core cochlea` then
   `g6me-verify --group cochlea --group acoustics`. Likely failure points: decimator
   alignment, PLA filter signs, spike-generator pre-release initialisation, runtime at
   24 CF × 3 fibers × 100 kHz.
4. **plus.rs:3805**: the room-tier ordering is not a B&M 2021 prediction (the paper scores
   cue match against a listener template). Either adopt the template-vs-dry construct used
   by `checks.rs` or justify the ordering from the paper; do not reshape the gate to fit.
5. **28-CF doc label** (`docs/ACADEMIC_NUKE_ACOUSTICS.md:87`) — claims-audit owns it.
6. **Doc discrepancies to route to claims-audit**: ray orders 4–50 vs 48/72
   (`room.rs:1633-1636`), MagLS 1.5 kHz vs 1.2–2.2 kHz (`hoa.rs:275-276`), Preview 250 vs
   500 Hz.
7. **Paper-literal remainder**: Duda–Martens `<1e-15` claim (measured 6.8e-4 relative at
   4 kHz / 0.109 m; exact only at the CLI minimum 0.25 m) — raise the cap or state the real
   bound; “paraunitary” FDN claim (matrix is orthogonal, absorption design is not
   Schlecht–Habets) — implement or rename; Algazi multipole re-expansion is blocked on the
   Gumerov–Duraiswami translation/rotation coefficient recursion (tech report:
   `users.umiacs.umd.edu/~ramanid/pubs/multipole.pdf`) — currently documented as a
   single-scattering approximation.
8. **ZBC 2014 blocker**: the JASA 135(1):283–286 text is unobtainable (ResearchGate 403;
   scispace returned 202 with empty content). `zbc.rs` therefore follows the authors'
   reference model `BEZ2018a_model` as the normative implementation of that paper, and
   synthesises fGn from the published statistical specification (Hurst 0.9; σ = 1 /s low,
   10 /s medium, SR/2 high) because `ffGn_2018a` is not in the archive. This provenance is
   documented in the `zbc.rs` module header. The successor should keep the documented
   substitution or obtain the paper; do not silently relabel.
9. **Sabotage checks** (project rule 1) for the new `zbc` invariants: refractoriness,
   spontaneous-rate class ordering, saturation, determinism.

## Ownership

- Mine: `cochlea.rs`, `zbc.rs`, `externalization.rs`, `scattering.rs`, `room.rs`,
  acoustics parts of `plus.rs`, cochlea/acoustics check names and labels in `checks.rs`.
- claims-audit: `docs/ACADEMIC_NUKE_ACOUSTICS.md`, `README.md`, verifier hardening, Lean.
  Must not touch cochlea naming.
- cuda-parity-audit: CUDA/scan; must not touch `externalization.rs`.
- Do not touch scan or CUDA code.

## Exact first action for the successor

```bash
cd /mnt/aosp/Projects/g6me && timeout 300 cargo test -p g6me-core \
  cochlea::tests::cochlea_nsim_is_unity_for_identical_and_detects_degradation -- --nocapture
```

This is the first execution of `zbc.rs`. If it passes (`NSIM(ref,ref)=1`, distorted
`< 0.92`), proceed to item 1 (the externalization loops). If it fails or hangs, inspect
`zbc.rs` before anything else — the likely suspects are the 63-tap decimator alignment
(`decimate_10`), the PLA biquad signs, and `generate_spikes` pre-release initialisation.

## Tree state

`g6me-core` lib compiles (`cargo check` exit 0). `g6me-verify` has not been re-checked
since `zbc.rs` was added, but the public `cochlea` API is unchanged. No commit or push was
made. The `zbc.rs` E0308 hunk is fixed and the file is syntactically valid.
