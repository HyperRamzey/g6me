# Patent-Clean Mastering & Spatial Enhancement Suite for `g6me`

This document describes the patent-clean replacement suite implemented inside `g6me` (`g6me-core`, `g6me-cli`, and `g6me-gui`) for a 5-plugin Nuendo mastering & spatialization chain, verified under the repository's deterministic IEEE-754 `f64`, zero-external-crate, zero-FMA contract and `/AGENTS.md` engineering discipline.

---

## 1. Patent Landscape & Clean-Room Design Boundaries

| Commercial Reference | Public Patent Status | Clean-Room Boundary Enforced in `g6me` |
| --- | --- | --- |
| **Waves RBass / MaxxBass** | Original MaxxBass patent (`US5930373A`, Shashoua et al.) is **expired**; newer stereo/adaptive speaker-distortion patents (`US9712916B2`, `CN110832881B`) remain active. | Uses independent per-channel Fierro–Välimäki Fuzzy STN decomposition + Mu–Gan–Tan phase-vocoder harmonic synthesis + Moliner–Rämö–Välimäki existing-harmonic awareness + Oo–Gan–Hawksford `FEXP1` transient NLD. Does not implement Waves' stereo cross-coupled virtual bass or speaker-excursion distortion models. |
| **Soundtheory Gullfoss Master** | Actively patented (`WO2019119028A1`, `AU2018387454B2`, priority Dec 2017) on proprietary filterbank/spectral-division dynamics. | Implements the open ISO-standardized human auditory loudness model (**Moore, Glasberg, Varathanathan & Schlittenlacher 2016**, **ISO 532-2:2017 / ISO 532-3:2022**) with 75-band `roex` ERB excitation, binaural inhibition, and zero-padded causal minimum-phase cepstral FIR realization (`crate::fft::minimum_phase_spectrum` + `half_kaiser_tail_apply`). |
| **NUGEN Halo Upmix** | Proprietary neural dialogue isolation and licensed ISOSTEM heritage (alongside Dolby/DTS/Fraunhofer/Zynaptiq upmix patent thickets). | Implements published pure-math Primary–Ambient Extraction (**He, Tan & Gan IEEE/ACM TASLP 2014/2015** PCA + ASE, **Song et al. IEEE TCE Jan 2025** APES ambient phase independence, and **Kraft & Zölzer DAFx-15/16** mutually-prime Schroeder all-pass decorrelators). |
| **Zynaptiq INTENSITY 2** | Actively patented (`US11079418`, `EP3271736B1`, `WO2016146265A1`, `JP6655269B2`) on spatio-temporal extended frequency transforms. | Uses the shared **Fierro & Välimäki (JAES 2023 / arXiv 2022)** dual-resolution Enhanced Fuzzy STN front-end (`crate::stn`) with bounded upward soft-knee perceptual contrast applied strictly to `Sines + Transients` while leaving the `Noise` residual untouched. |
| **FabFilter Pro-L 2 & NUGEN ISL2** | ISL2 is standard ITU-R BS.1770 true-peak brickwall limiting; Pro-L 2 has no major public software patents (proprietary closed-source tuning). | Retains `g6me`'s bit-exact default `Transparent` limiter and adds opt-in `Punchy` mode combining **Parker, Zavalishin & Le Bivic (DAFx-2016)** Residual ADAA2 `C^2` soft-knee clipping, **Signalsmith Audio (Luff 2022)** 2-stage cascaded boxcar `C^1` lookahead smoothing, and **Giannoulis, Massberg & Reiss (JAES 2012)** log-domain smooth branching release. |

All new stages are **strictly opt-in (`enable: false` by default)**, and `LimiterStyle::Transparent` remains the default so existing stock renders stay 100% bit-exact.

---

## 2. Pipeline Architecture & Shared-STN Synergy

```text
Input PCM (2.0 or 5.1/7.1)
  │
  ├─► [Optional] 2.0 → 7.1 PAE Upmixer (`upmix.rs`, stereo inputs only)
  │       He PCA+ASE + APES quadrature phase + Kraft–Zölzer 4-ch all-pass + LR4 LFE
  │
  ├─► G6 Reference / G6+ Binaural Downmix (`dsp.rs` / `plus.rs`) ──► Stereo f64
  │
  ├─► [Optional] Shared-STN Enhancer (`stn.rs` + `enhance.rs`)
  │       Single dual-resolution Fuzzy STN pass per channel → (Sines, Transients, Noise)
  │         • Virtual Bass: N=8192*s phase-vocoder harmonic synth on Sines + FEXP1 NLD on Transients
  │         • Micro-Detail: soft-knee upward contrast + 2.5 kHz presence shelf on Sines + Transients
  │         • Noise residual passes untouched → bit-exact null (< 1e-12) at zero drive
  │
  ├─► [Optional] Moore 2016 / ISO 532-2 Perceptual Unmasking EQ (`unmask.rs`)
  │       75-band ERB roex + binaural inhibition → minimum-phase cepstral FIR (zero-padded OLA)
  │
  └─► DC Block (opt) → Gain Trim → Brickwall True-Peak Limiter (`limiter.rs`: Transparent | Punchy + ADAA2)
```

---

## 3. DSP Implementations & Exact Mathematical Specifications

### 3.1 Shared Dual-Resolution Enhanced Fuzzy STN (`crates/g6me-core/src/stn.rs`)
Based on **Driedger, Müller & Disch (ISMIR 2014)** and **Fierro & Välimäki (JAES 2023 / arXiv 2022)**:

- **Two cascaded 75%-overlap `sqrt(0.5 * periodic_hann)` COLA STFT stages** (`sr_scale = next_pow2(ceil(fs / 48000))`):
  - **Stage 1 (Tonal)**: `N_1 = 4096 * sr_scale`, `H_1 = N_1 / 4`, horizontal median `~200 ms`, vertical median `~500 Hz`. Extracts `x_s` and exact sample-aligned residual `x_res = x_delayed - x_s`.
  - **Stage 2 (Transient)**: `N_2 = 512 * sr_scale`, `H_2 = N_2 / 4`, horizontal median `~25 ms`, vertical median `~1000 Hz`. Extracts `x_t` from `x_res` and defines `x_n = x_res_delayed - x_t`.
- **Squared-sine fuzzy membership** (`beta_L = 0.7`, `beta_U = 0.8`, enforcing `beta_L >= 0.5`):
  - `S_s = sin^2((pi/2) * clamp((r_s - beta_L) / (beta_U - beta_L), 0, 1))`, `S_t = sin^2(...)`, `S_n = 1 - S_s - S_t >= 0`.
  - Guarantees `S_s + S_t + S_n == 1.0` at every bin and exact time-domain null `|x_s + x_t + x_n - x_aligned| < 1e-12` across arbitrary chunk boundaries.
- **Exact causal latency**: Each stage has `total_latency = N + h_half * H - 1` (accounting for both `N - H + h_half * H` STFT/median delay and `H - 1` hop buffering).

### 3.2 Zero-IMD Psychoacoustic Virtual Bass + STN Micro-Detail (`crates/g6me-core/src/enhance.rs`)
- **Tonal Virtual Bass** (Moliner, Rämö & Välimäki DAFx-2020 + Mu, Gan & Tan ICASSP 2012/2013):
  - Operates strictly on `x_s` at `N = 8192 * sr_scale` (`5.86 Hz` bin spacing at 48 kHz — resolving bass semitones down to 30 Hz).
  - Unwraps instantaneous frequency `omega_0(k)` via hop-to-hop phase differences, projects harmonics `h in {2, 3, 4}` with normalized timbre-matched weights `w_h = exp(-harmonic_decay * (h - 2))`.
  - **Existing-harmonic awareness**: when harmonic bin `kh` is already a local spectral peak above `-22 dB` relative to target, boosts its existing main lobe in-phase; otherwise synthesizes the missing harmonic directly at exact fractional frequency `omega_h = h * omega_0` with continuous hop-to-hop phase propagation into `w[i]^2` COLA OLA.
  - Measured on a two-tone bass chord (`55.0 Hz A1 + 73.42 Hz D2`, Oo–Gan–Hawksford JAES 2011): achieves `~-7.4 dB` harmonic generation with `~-69 dB` IMD-to-harmonics (**> 70 dB cleaner** than naive `FEXP1` waveshaping).
- **Transient Virtual Bass**: Sub-bass LR4 low-pass (`cutoff_hz`) → Oo–Gan–Hawksford `FEXP1` asymmetric NLD (`0.75 * odd + 0.25 * HWR`) → LR4 band-pass (`cutoff_hz .. 4 * cutoff_hz`), applied strictly to `x_t` so kick drums gain punch without polluting bass sustain or noise floor.
- **STN Micro-Detail & Presence Enhancer**: Dual-envelope (`15/120 ms` tonal, `2/45 ms` transient) soft-knee upward contrast multiplier (`upward_detail_gain`) gated below `-60 dBFS` and smoothly tapering to unity at `0 dBFS`, shaped by a 1st-order `2.5 kHz` presence shelf (`presence_tilt_db`) on the lifted delta only. `x_n` is never boosted.

### 3.3 Auditory-Model Perceptual Unmasking Dynamic EQ (`crates/g6me-core/src/unmask.rs`)
Based on **Moore, Glasberg, Varathanathan & Schlittenlacher (Trends in Hearing 2016; JASA 2018; ISO 532-2:2017 / ISO 532-3:2022)**:

- **Outer + middle ear transfer** interpolated in log-frequency from ISO 532-2 Tables A.1 & A.2.
- **75-band ERB-rate cochlear filterbank** (`1.75 ..= 38.75 Cam`, `0.5 Cam` spacing) using level-dependent asymmetric `roex(p_l, p_u)` skirts (`p_l = p_51 - 0.35 * (p_51 / p_51_1k) * (L_in - 51)`).
- **Specific loudness & Moore 2016 binaural inhibition**:
  - `C = 0.0631`, `sigma = 1.8 Cam` Gaussian ERB smoothing, `p = 1.5978`:
    `INHIB_ipsi = 2 / (1 + sech(N'_contra_sm / N'_ipsi_sm)^1.5978)`, `N'_ipsi_inh = N'_ipsi / INHIB_ipsi`.
  - Yields exact `1.500x` diotic-to-monaural summation and `1.008 sone` on a 1 kHz 40 dB SPL diotic reference tone.
- **Dual temporal integration** (short-term `22 ms / 50 ms`, long-term `99 ms / 2000 ms`) compared against an equal-information pink-noise ERB reference (tilted by `brighten` `±3 dB`), with audibility gating (`E_self / (E_self + 25 * E_THRQ)`), spread-masking excess recovery, and loudness-neutral zero-mean clamping to `±max_gain_db`.
- **Causal minimum-phase FIR realization**: Zero-padded linear-convolution OLA (`W = 1024 * s`, `H = 256 * s`, `N_fft = 2048 * s`) using analysis-only `0.25 * (1 - cos(2*pi*i/W))` (`COLA error = 2.22e-16`), `crate::fft::minimum_phase_spectrum`, and `half_kaiser_tail_apply` (`<= W` causal taps). Latency = `W - H = 768 * s` (`16 ms` at 48 kHz).

### 3.4 Phase-Coherent 2.0 → 7.1 PAE Upmixer (`crates/g6me-core/src/upmix.rs`)
Based on **He, Tan & Gan (IEEE/ACM TASLP 2014/2015)**, **Song et al. (IEEE TCE Jan 2025)**, and **Kraft & Zölzer (DAFx-15/16)**:

- **2×2 Hermitian covariance eigendecomposition** (`150 ms` exponential smoothing + 5-bin triangular spectral smoothing) with Wishart-debiased squared ASE primary fraction `p_frac = ((lam1 - lam2) / lam1)^2`.
- **APES ambient phase independence**: rotates the correlated residual along `v_1` into opposite quadrature (`+j` on L, `-j` on R) and renormalizes ambient power to exact `E_in - E_p` (strict Parseval energy conservation).
- **Equal-power front triad (`FL / FC / FR`)**: azimuth centerness `2 * a * b * max(cos(phi), 0)` locks mono vocals to `FC` with `< -35 dB` rear leakage while routing diffuse uncorrelated stereo (`> 60%`) to surrounds.
- **Kraft–Zölzer 4-channel Schroeder all-pass decorrelators** (`SL / SR / BL / BR`): 4 mutually-prime delay cascades (`[29,67,131,227]`, `[37,79,149,251]`, `[43,89,167,277]`, `[53,103,191,307]` scaled by `fs/48000`), exact unit magnitude `|H(e^{jω})| = 1.0 ± 4.44e-16`, pairwise mono-fed `|ICC| < 0.06`, plus optional causal LR4 LFE crossover.
- Output matches `MASK_7_1` native channel order `[FL, FR, FC, LFE, BL, BR, SL, SR]`, feeding directly into `G6` / `G6+`.

### 3.5 "Pro-L 2 Punchy" Limiter Upgrade (`crates/g6me-core/src/limiter.rs`)
- **Residual ADAA2 `C^2` Soft-Knee Clipper** (Parker, Zavalishin & Le Bivic DAFx-2016):
  - Soft-knee polynomial/rational saturator `f(x)` that is **identically linear (`f(x) = x`) for `|x| <= t0`** (`0.000 dB` passband droop, `< 1e-12` error!) and `C^2`-smooth above `t0` approaching asymptotic ceiling `c_lim + 1.0 dB`.
  - Computes `y[n] = x[n - 1] + adaa2(r(x))` where `r(x) = f(x) - x` is the residual nonlinearity (`r(x) == 0` in the linear passband), avoiding catastrophic cancellation on small signals while suppressing high-frequency aliasing by `> 15 dB` over naive clipping (adds exactly 1 sample of latency).
- **Signalsmith 2-Stage Cascaded Boxcar Lookahead Smoother** (Geraint Luff 2022):
  - Two moving-average stages of lengths `L1 = ceil((ramp + 1)/2)` and `L2 = (ramp + 1) - L1 + 1` (`L1 + L2 - 1 == ramp + 1`) convolving the sliding lookahead minimum `min_{m in [j..=j+ramp]} sched[m]` (including pre-roll feeds from `j_signed = -ramp`) to produce a `C^1` parabolic S-curve with zero true-peak overshoot on sample 0 through end-of-file.
- **Giannoulis, Massberg & Reiss (JAES 2012) Log-Domain Smooth Branching**: Slow-stage attack/release smoothing runs in the log-gain domain in `Punchy` mode for constant dB/s musical release.

---

## 4. Verification & `/AGENTS.md` Sabotage Proofs

All invariants are verified by deterministic unit tests, `g6me-verify --require-all` (0 failures, 0 skips, 0 FMA instructions), and `/AGENTS.md` Sabotage Rule checks proving each test discriminates against regression:

| Module / Invariant | Verification Test | Sabotage Proof (Feature Disabled / Broken → Test Fails) |
| --- | --- | --- |
| **STN Perfect Reconstruction** | `stn_perfect_reconstruction_null_across_arbitrary_chunks` (`< 1e-12`) | Breaking `total_latency` alignment (`- (hop - 1)`) blows null error from `< 1e-13` to `1.815` (FAIL). |
| **Zero-IMD Virtual Bass** | `virtual_bass_beats_naive_nld_on_two_tone_imd_by_over_20_db` | Replacing phase-vocoder synthesis with naive `FEXP1` NLD degrades IMD-to-harmonics from `<-40 dB` to `+5.4 dB` (FAIL). |
| **STN Micro-Detail Noise Immunity** | `micro_detail_lifts_quiet_tonal_content_without_boosting_noise_floor` | Applying lift to `noise` stream instead of `sines + transients` boosts white noise floor by `> +3 dB` (FAIL). |
| **Moore 2016 Binaural Inhibition** | `moore_2016_sone_calibration_and_binaural_inhibition_exact` | Removing `INHIB_ipsi` division turns diotic/monaural summation from `1.500x` to `2.000x` (FAIL). |
| **PAE Center Lock & Decorrelation** | `center_mono_vocal_locks_to_fc_with_negligible_rear_leakage`, `kraft_zolzer_allpass_decorrelators_are_unit_magnitude_and_uncorrelated` | Setting identical delays across surround decorrelators drives pairwise `|ICC|` from `< 0.06` to `1.000` (FAIL). |
| **ADAA2 Aliasing Suppression & Hard Ceiling** | `adaa2_clipper_zero_passband_error_and_aliasing_suppression`, `punchy_limiter_holds_hard_true_peak_ceiling_and_shaves_transients` | Bypassing ADAA2 to memoryless `f(x)` drops aliasing suppression to `0.0 dB` (FAIL); gating `Punchy` smoother on `if live` breaks sample-0 ceiling by `+3.86 dB` (FAIL). |

---

## 5. CLI Usage Examples

```bash
# Full patent-clean mastering + spatial chain on a stereo source into G6+ Studio binaural:
g6me input_stereo.flac \
  --mode plus --plus-profile studio \
  --upmix --upmix-center 0.85 --upmix-surround 0.90 --upmix-rear 0.75 \
  --vbass --vbass-cutoff 100 --vbass-amount 0.45 --vbass-punch 0.60 \
  --detail --detail-intensity 0.35 --detail-tilt 2.0 \
  --unmask --unmask-recover 0.35 --unmask-tame 0.35 --unmask-max-db 4.5 \
  --limit-style punchy --limit-clip-db 2.0 --ceiling -1.0

# Verify bit-exact default path (all new stages off, Transparent limiter):
g6me input_7_1.flac --dry-run
```
