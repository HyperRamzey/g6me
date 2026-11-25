//! `G6+` ("Plus") universal, non-personalized binaural renderer.
//!
//! Designed specifically for 48 kHz surround beds (including Dolby TrueHD 7.1
//! with `.4` Atmos heights folded by DEE) without requiring per-user HRTF
//! measurements, ear scans, or headphone-specific profiles. Stacks transparently
//! with external headphone EQ / `HyprEQ` because the 7.0 diffuse-field response
//! is equalized flat to within `<= ±0.25 dB` (1-ERB Glasberg–Moore excitation)
//! and matched to stock `g6ct` uncorrelated per-ear loudness (`3.500`, `+5.44 dB`).
//!
//! Acoustic model:
//! 1. **Physical Brown–Duda (1998) spherical head shadow** (frequency-dependent
//!    diffraction ILD, `a = 8.75 cm`, `c = 343 m/s`, `f0 = 623.9 Hz`) converted
//!    to strictly causal minimum phase via Hilbert-cepstral folding.
//! 2. **Kulkarni–Isabelle–Colburn (1999) excess Woodworth ITD**: measures the
//!    natural minimum-phase ITD already contributed by the head-shadow filter
//!    and applies only the remaining causal time-of-flight delay so total ITD
//!    matches Woodworth (`FL ≈ 261 µs`, `SL ≈ 607 µs`, `BL ≈ 322 µs`) without
//!    double-counting.
//! 3. **Causal Linkwitz–Riley 4th-order (`LR4`) bass-mono crossover** at
//!    `bass_mono_hz` (default `120 Hz`): keeps sub-bass phase-coherent (`IC >= 0.97`)
//!    across closed-backs, open-backs, planars, and estats with zero pre-ringing.
//! 4. **Universal broad front–back pinna shadow** (gentle rear HF shelf + subtle
//!    1.2 kHz Blauert directional band, zero sharp individualized pinna notches).
//! 5. **DEE-safe side/back all-pass decorrelation**: 1st-order causal all-pass on
//!    `BL`/`BR` above `bass_mono_hz` prevents coherent `+4.4 dB` phase-locked
//!    pileup when DEE folds `.4` Atmos heights simultaneously into `SL+BL`/`SR+BR`,
//!    with 0.00 dB individual magnitude coloration and 0 µs ITD corruption.
//! 6. **Direct-path 1-ERB common Diffuse-Field Equalization (Zaunschirm,
//!    Schörkhuber & Höldrich, JASA 2018)**: equalizes the direct 7.0 ring
//!    to within `<= ±0.25 dB` independently of lateral room reflections so
//!    `> 400 Hz` early reflections never pull down direct midrange relative to bass,
//!    followed by broadband Parseval reflection power normalization (`3.50`).
//! 7. **Smooth alternating-polarity ITU-R BS.1116 lateral early reflections**
//!    (`>= 8.5 ms`, zero `< 6 ms` comb bounce, `Hassager, Gran & Dau, JASA 2016`
//!    smoothed diffuse tail) controlled by `externalize` and `center_focus`.
//! 8. **Adaptive 1-ERB Shirley–Kendrick–Churchill (JAES 2007) phantom-center
//!    de-comb** (`phantom_decomb`): measures each frontal stereo pair's crosstalk
//!    comb-filter deficit above `450 Hz` and fills it in causal minimum phase so
//!    stereo/music vocals stay open without tonal hollowing.
//! 9. **Bauer / Linkwitz / Völk–Fastl (AES 2011) coherent LF trim shelf**
//!    (`bass_trim_db`): user-tunable 1st-order causal minimum-phase `340 Hz`
//!    shelf on full-range channels (discrete `LFE` left untouched) compensating
//!    in-phase low-frequency summation (`+3..+4 dB`) on correlated music and
//!    surround mixes vs `Reference`.
//! 10. **LFE & Streaming OLA**: optional 120 Hz 4th-order Butterworth LPF aligned
//!     to `PRE_DELAY`, mono `1/sqrt(2)` in-phase to both ears; `N_IR = 2048`,
//!     `N_FFT = 8192`, `HOP = 6144`, deterministic zero-FMA `f64` arithmetic.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsp::{Dsp, MASK_STEREO, SLOTS};
use crate::fft::{
    half_kaiser_tail_apply, minimum_phase_spectrum, roex_smooth_power, Complex, FftPlan,
};

/// Synthesis FFT size (`341.3 ms` at 48 kHz) — long enough that the causal LR4
/// bass-mono crossover impulse decays `< -140 dB` with zero circular wrap.
pub const N_SYNTH: usize = 16_384;
/// Final windowed FIR kernel length (`42.67 ms` at 48 kHz).
pub const N_IR: usize = 2_048;
/// Streaming overlap-add FFT size (`HOP + N_IR - 1 = 8191 < 8192`).
pub const N_FFT: usize = 8_192;
/// Maximum input chunk per single OLA transform (`8192 - 2048 = 6144`).
pub const OLA_HOP: usize = 6_144;
/// Causal direct-path pre-delay in seconds (`1.0 ms` = `48` samples at 48 kHz).
pub const PRE_DELAY_S: f64 = 0.001;
/// Half-Kaiser tail taper length at the end of `N_IR`.
const KAISER_TAIL: usize = 256;
/// Kaiser beta for tail taper (`> 119 dB` window edge attenuation).
const KAISER_BETA: f64 = 16.0;
/// Spherical head radius (meters) — Woodworth / Brown–Duda universal average.
const HEAD_RADIUS_M: f64 = 0.0875;
/// Speed of sound in air at 20 °C (m/s).
const SPEED_OF_SOUND: f64 = 343.0;
/// Target uncorrelated 7.0 per-ear diffuse power (matches stock `g6ct` = `3.5017`).
const TARGET_DF_POWER: f64 = 3.5;

/// Top-level binaural renderer selection.
///
/// Defaults to [`RendererMode::Reference`] so every existing CLI invocation,
/// GUI preset, and `g6me-verify` bit-exact contract stays untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RendererMode {
    /// Bit-exact `CMatrixEncoderEfxMod` (`Dsp<f64>`).
    #[default]
    Reference,
    /// `G6+` universal non-personalized binaural renderer.
    Plus,
}

impl RendererMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Plus => "plus",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Reference => "Reference (Bit-Exact G6)",
            Self::Plus => "Plus (G6+ Binaural)",
        }
    }
}

/// Curated acoustic presets for [`RendererMode::Plus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlusProfile {
    /// Zero room reflections — pure physical ITD/ILD/pinna + DFEQ.
    Pure,
    /// Dry ITU-R BS.1116 studio control-room externalization (recommended default).
    #[default]
    Studio,
    /// Wider cinematic soundstage with richer lateral early reflections.
    Cinema,
    /// User-edited custom parameters.
    Custom,
}

impl PlusProfile {
    pub const ALL: [Self; 4] = [Self::Pure, Self::Studio, Self::Cinema, Self::Custom];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pure => "pure",
            Self::Studio => "studio",
            Self::Cinema => "cinema",
            Self::Custom => "custom",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Pure => "Pure",
            Self::Studio => "Studio",
            Self::Cinema => "Cinema",
            Self::Custom => "Custom",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Pure => {
                "Pure physical head-shadow & ITD cues, zero room reflections, strict DFEQ"
            }
            Self::Studio => {
                "Dry ITU-BS.1116 control-room lateral cues (DRR ~+9 dB), tight dialogue focus"
            }
            Self::Cinema => "Wider frontal stage, deeper surrounds, richer lateral externalization",
            Self::Custom => "Custom soundfield tuning",
        }
    }
}

/// Runtime-tweakable parameters for the `G6+` binaural renderer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlusCfg {
    /// Active profile tag (recomputed or updated when sliders move).
    pub profile: PlusProfile,
    /// Frontal stereo/L-R stage width (`0.50 ..= 1.50`, `1.0` = `±30°`).
    pub width: f64,
    /// Center channel dialogue focus (`0.00 ..= 1.00`; higher = drier, tighter anchor).
    pub center_focus: f64,
    /// Surround & rear immersion depth (`0.50 ..= 1.50`, `1.0` = natural ITU balance).
    pub surround_depth: f64,
    /// Dry ITU-BS.1116 lateral early-reflection externalization (`0.00 ..= 1.00`).
    pub externalize: f64,
    /// Causal LR4 bass-mono crossover frequency in Hz (`0.0` disables; `60..=200` typical).
    pub bass_mono_hz: f64,
    /// Bauer / Linkwitz coherent low-frequency balance trim in dB (`-6.0 ..= +3.0 dB`,
    /// 1st-order causal minimum-phase shelf at `340 Hz` on full-range channels; discrete
    /// `LFE` left untouched). Tames coherent stereo/surround bass buildup vs `Reference`.
    pub bass_trim_db: f64,
    /// Shirley–Kendrick adaptive 1-ERB phantom-center comb-notch fill (`0.00 ..= 1.00`).
    pub phantom_decomb: f64,
    /// Discrete `.1` LFE gain offset in dB (`-12.0 ..= +12.0 dB`; `0.0` = unity `1/sqrt(2)`
    /// matching Reference / Nuendo 0 dB discrete LFE).
    pub lfe_gain_db: f64,
    /// Apply `bass_trim_db` (340 Hz LF balance shelf) to discrete `LFE` as well (`false` keeps
    /// discrete `LFE` at `0 dB` relative to untrimmed Reference; `true` trims `.1` with mains).
    pub lfe_lf_trim: bool,
    /// Apply 120 Hz 4th-order Butterworth low-pass filter to the LFE channel.
    pub lfe_lpf: bool,
}

impl Default for PlusCfg {
    fn default() -> Self {
        Self::from_profile(PlusProfile::Studio)
    }
}

impl PlusCfg {
    pub fn from_profile(profile: PlusProfile) -> Self {
        match profile {
            PlusProfile::Pure => Self {
                profile: PlusProfile::Pure,
                width: 1.00,
                center_focus: 0.70,
                surround_depth: 1.00,
                externalize: 0.00,
                bass_mono_hz: 120.0,
                bass_trim_db: 0.0,
                phantom_decomb: 0.00,
                lfe_gain_db: 0.0,
                lfe_lf_trim: false,
                lfe_lpf: true,
            },
            PlusProfile::Studio => Self {
                profile: PlusProfile::Studio,
                width: 1.00,
                center_focus: 0.65,
                surround_depth: 1.00,
                externalize: 0.45,
                bass_mono_hz: 120.0,
                bass_trim_db: -1.5,
                phantom_decomb: 0.55,
                lfe_gain_db: 0.0,
                lfe_lf_trim: false,
                lfe_lpf: true,
            },
            PlusProfile::Cinema => Self {
                profile: PlusProfile::Cinema,
                width: 1.18,
                center_focus: 0.55,
                surround_depth: 1.20,
                externalize: 0.75,
                bass_mono_hz: 110.0,
                bass_trim_db: -2.1,
                phantom_decomb: 0.75,
                lfe_gain_db: 0.0,
                lfe_lf_trim: false,
                lfe_lpf: true,
            },
            PlusProfile::Custom => Self {
                profile: PlusProfile::Custom,
                ..Self::from_profile(PlusProfile::Studio)
            },
        }
    }

    /// Detect whether current numeric values match one of the built-in profiles.
    pub fn detect_profile(&self) -> PlusProfile {
        for p in [PlusProfile::Pure, PlusProfile::Studio, PlusProfile::Cinema] {
            let r = Self::from_profile(p);
            if (self.width - r.width).abs() < 1e-4
                && (self.center_focus - r.center_focus).abs() < 1e-4
                && (self.surround_depth - r.surround_depth).abs() < 1e-4
                && (self.externalize - r.externalize).abs() < 1e-4
                && (self.bass_mono_hz - r.bass_mono_hz).abs() < 0.25
                && (self.bass_trim_db - r.bass_trim_db).abs() < 0.05
                && (self.phantom_decomb - r.phantom_decomb).abs() < 1e-4
                && (self.lfe_gain_db - r.lfe_gain_db).abs() < 0.05
                && self.lfe_lf_trim == r.lfe_lf_trim
                && self.lfe_lpf == r.lfe_lpf
            {
                return p;
            }
        }
        PlusProfile::Custom
    }

    pub fn sync_profile(&mut self) {
        self.profile = self.detect_profile();
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if !self.width.is_finite() || !(0.5..=1.5).contains(&self.width) {
            return Err(format!(
                "plus.width must be in 0.50..=1.50, got {}",
                self.width
            ));
        }
        if !self.center_focus.is_finite() || !(0.0..=1.0).contains(&self.center_focus) {
            return Err(format!(
                "plus.center_focus must be in 0.00..=1.00, got {}",
                self.center_focus
            ));
        }
        if !self.surround_depth.is_finite() || !(0.5..=1.5).contains(&self.surround_depth) {
            return Err(format!(
                "plus.surround_depth must be in 0.50..=1.50, got {}",
                self.surround_depth
            ));
        }
        if !self.externalize.is_finite() || !(0.0..=1.0).contains(&self.externalize) {
            return Err(format!(
                "plus.externalize must be in 0.00..=1.00, got {}",
                self.externalize
            ));
        }
        if !self.bass_mono_hz.is_finite() || !(0.0..=250.0).contains(&self.bass_mono_hz) {
            return Err(format!(
                "plus.bass_mono_hz must be in 0..=250 Hz, got {}",
                self.bass_mono_hz
            ));
        }
        if !self.bass_trim_db.is_finite() || !(-6.0..=3.0).contains(&self.bass_trim_db) {
            return Err(format!(
                "plus.bass_trim_db must be in -6.0..=+3.0 dB, got {}",
                self.bass_trim_db
            ));
        }
        if !self.phantom_decomb.is_finite() || !(0.0..=1.0).contains(&self.phantom_decomb) {
            return Err(format!(
                "plus.phantom_decomb must be in 0.00..=1.00, got {}",
                self.phantom_decomb
            ));
        }
        if !self.lfe_gain_db.is_finite() || !(-12.0..=12.0).contains(&self.lfe_gain_db) {
            return Err(format!(
                "plus.lfe_gain_db must be in -12.0..=+12.0 dB, got {}",
                self.lfe_gain_db
            ));
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "G6+ {} (width {:.2}, focus {:.2}, surr {:.2}, ext {:.2}, bass-mono {:.0} Hz, lf {:+.1} dB{}, decomb {:.2}, lfe {:+.1} dB{})",
            self.profile.label(),
            self.width,
            self.center_focus,
            self.surround_depth,
            self.externalize,
            self.bass_mono_hz,
            self.bass_trim_db,
            if self.lfe_lf_trim { "+lfe" } else { "" },
            self.phantom_decomb,
            self.lfe_gain_db,
            if self.lfe_lpf { ", lfe-lpf" } else { "" },
        )
    }

    /// Estimate direct-to-reverberant ratio (dB) for center dialogue at current settings
    /// (for GUI acoustic readout).
    pub fn estimated_center_drr_db(&self) -> Option<f64> {
        if self.externalize <= 1e-4 {
            return None;
        }
        let refl_scale = self.externalize * (1.0 - 0.50 * self.center_focus);
        let refl_e = 0.65 * refl_scale * refl_scale;
        if refl_e <= 1e-9 {
            return None;
        }
        Some(10.0 * (1.0 / refl_e).log10())
    }
}

/// Slot indices matching `crate::mask::Slot::index()`.
const SLOT_FL: usize = 0;
const SLOT_FR: usize = 1;
const SLOT_FC: usize = 2;
const SLOT_LFE: usize = 3;
const SLOT_BL: usize = 4;
const SLOT_BR: usize = 5;
const SLOT_FLC: usize = 6;
const SLOT_FRC: usize = 7;
const SLOT_BC: usize = 8;
const SLOT_SL: usize = 9;
const SLOT_SR: usize = 10;
const SLOT_TC: usize = 11;

/// Base speaker azimuths in degrees (`+` = left, `-` = right) for each `SLOTS` index.
const SLOT_AZ_DEG: [f64; SLOTS] = [
    30.0,   // FL
    -30.0,  // FR
    0.0,    // FC
    0.0,    // LFE (handled separately)
    142.5,  // BL
    -142.5, // BR
    15.0,   // FLC
    -15.0,  // FRC
    180.0,  // BC
    100.0,  // SL
    -100.0, // SR
    0.0,    // TC
];

/// Full-range 7.0 ring slots used for the 1-ERB diffuse-field equalization reference.
const DF_RING_SLOTS: [usize; 7] = [
    SLOT_FL, SLOT_FR, SLOT_FC, SLOT_SL, SLOT_SR, SLOT_BL, SLOT_BR,
];

/// A single dry studio early-reflection image source.
#[derive(Debug, Clone, Copy)]
struct ReflectionTap {
    /// Arrival delay after direct sound (seconds), always `>= 8.5 ms`.
    delay_s: f64,
    /// Image-source azimuth in degrees (`+` = left, `-` = right).
    az_deg: f64,
    /// Linear amplitude before `externalize` scaling.
    gain: f64,
    /// Wall HF low-pass cutoff (Hz).
    lpf_hz: f64,
}

/// Synthesized `SLOTS × 2` binaural FIR kernel bank at a concrete sample rate.
#[derive(Debug, Clone)]
pub struct KernelBank {
    pub sample_rate: u32,
    pub cfg: PlusCfg,
    /// Time-domain left-ear FIR kernels (`N_IR` taps per slot).
    pub ir_l: Vec<Vec<f64>>,
    /// Time-domain right-ear FIR kernels (`N_IR` taps per slot).
    pub ir_r: Vec<Vec<f64>>,
    /// Causal pre-delay in integer samples (`round(PRE_DELAY_S * sample_rate)`).
    pub pre_delay_samples: usize,
}

impl KernelBank {
    /// Synthesize all 12×2 binaural FIR kernels deterministically in `f64`.
    pub fn synthesize(cfg: PlusCfg, sample_rate: u32) -> Self {
        let fs = (sample_rate.max(8_000)) as f64;
        let nb = N_SYNTH / 2 + 1;
        let plan = FftPlan::new(N_SYNTH);
        let freqs: Vec<f64> = (0..nb)
            .map(|k| (k as f64) * fs / (N_SYNTH as f64))
            .collect();
        let pre_delay_samples = (PRE_DELAY_S * fs).round().max(1.0) as usize;
        let pre_delay_s = (pre_delay_samples as f64) / fs;

        // Precompute causal LR4 bass-mono crossover curves once.
        let mut lr4_lo = vec![Complex::ZERO; nb];
        let mut lr4_hi = vec![Complex::ONE; nb];
        if cfg.bass_mono_hz > 0.0 {
            let fc = cfg.bass_mono_hz.min(0.45 * fs);
            for (k, &f) in freqs.iter().enumerate() {
                let (lo, hi) = lr4_pair(f, fc, fs);
                lr4_lo[k] = lo;
                lr4_hi[k] = hi;
            }
        }

        // Precompute DEE-safe 1st-order causal all-pass filter (650 Hz) for BL/BR.
        let mut ap_rear = vec![Complex::ONE; nb];
        let f_ap = 650.0_f64.min(0.4 * fs);
        let tan_ap = (PI * f_ap / fs).tan();
        let ap_coeff = (tan_ap - 1.0) / (tan_ap + 1.0);
        for (k, &f) in freqs.iter().enumerate() {
            let z1 = Complex::from_polar(1.0, -2.0 * PI * f / fs);
            // H_ap(z) = (a + z^-1) / (1 + a * z^-1), |H_ap| == 1 everywhere, strictly causal.
            let num = Complex::new(ap_coeff, 0.0).add(z1);
            let den = Complex::ONE.add(z1.scale(ap_coeff));
            ap_rear[k] = num.div(den);
        }

        let mut dir_l: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; nb]; SLOTS];
        let mut dir_r: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; nb]; SLOTS];
        let mut chan_weights = [1.0_f64; SLOTS];

        // 1. Synthesize each full-range slot's anechoic direct path first.
        for ch in 0..SLOTS {
            if ch == SLOT_LFE {
                continue;
            }
            let (sl, sr, cw) = synthesize_direct_spectrum(
                ch,
                &cfg,
                fs,
                pre_delay_s,
                &freqs,
                &lr4_lo,
                &lr4_hi,
                &ap_rear,
                &plan,
            );
            dir_l[ch] = sl;
            dir_r[ch] = sr;
            chan_weights[ch] = cw;
        }

        // 2. Direct-path 1-ERB common Diffuse-Field Equalization (Zaunschirm,
        //    Schörkhuber & Höldrich, JASA 2018): computing DFEQ on the direct 7.0
        //    ring alone prevents `> 400 Hz` lateral room reflections from biasing DFEQ
        //    into attenuating direct midrange vs sub-150 Hz bass.
        let mut p_dir = vec![0.0_f64; nb];
        for &ch in &DF_RING_SLOTS {
            for k in 0..nb {
                p_dir[k] += 0.5 * (dir_l[ch][k].norm_sq() + dir_r[ch][k].norm_sq());
            }
        }
        let p_dir_smooth = roex_smooth_power(&freqs, &p_dir, 0.75);
        let dfeq_mag: Vec<f64> = p_dir_smooth
            .iter()
            .map(|&p| (TARGET_DF_POWER / p.max(1e-9)).sqrt().clamp(0.25, 4.0))
            .collect();
        let dfeq_min = minimum_phase_spectrum(&dfeq_mag, &plan);

        // Apply the common direct-path DFEQ to every full-range channel identically
        // so ITD, ILD, and L/R symmetry stay mathematically invariant.
        for ch in 0..SLOTS {
            if ch == SLOT_LFE {
                continue;
            }
            for k in 0..nb {
                dir_l[ch][k] = dir_l[ch][k].mul(dfeq_min[k]);
                dir_r[ch][k] = dir_r[ch][k].mul(dfeq_min[k]);
            }
        }

        let mut spec_l = dir_l.clone();
        let mut spec_r = dir_r.clone();

        // 3. Add dry smooth alternating-polarity ITU-R BS.1116 / Hassager–Gran–Dau
        //    lateral early reflections (`>= 8.5 ms`) when `externalize > 0`, filtered by
        //    the direct diffuse envelope (`dfeq_min`) for clean spectral integration.
        if cfg.externalize > 1e-6 {
            for ch in 0..SLOTS {
                if ch == SLOT_LFE {
                    continue;
                }
                add_channel_reflections(
                    ch,
                    &cfg,
                    fs,
                    pre_delay_s,
                    chan_weights[ch],
                    &freqs,
                    &dfeq_min,
                    &mut spec_l[ch],
                    &mut spec_r[ch],
                    &plan,
                );
            }
        }

        // 4. Adaptive 1-ERB Shirley–Kendrick–Churchill (JAES 2007) phantom-center
        //    de-comb on frontal stereo pairs `(FL, FR)` and `(FLC, FRC)` above 450 Hz.
        if cfg.phantom_decomb > 1e-6 {
            apply_frontal_decomb(
                SLOT_FL,
                SLOT_FR,
                &freqs,
                &dir_l,
                &dir_r,
                &mut spec_l,
                &mut spec_r,
                cfg.phantom_decomb,
                &plan,
            );
            apply_frontal_decomb(
                SLOT_FLC,
                SLOT_FRC,
                &freqs,
                &dir_l,
                &dir_r,
                &mut spec_l,
                &mut spec_r,
                cfg.phantom_decomb,
                &plan,
            );
        }

        // 5. Broadband Parseval 7.0 diffuse power preservation (`TARGET_DF_POWER = 3.5`)
        //    after reflections & de-comb so total loudness matches stock `g6ct` (`3.5017`)
        //    without tilting the direct spectrum.
        let mut tot_p = 0.0_f64;
        let inv_n = 1.0 / (N_SYNTH as f64);
        for &ch in &DF_RING_SLOTS {
            let pl = (spec_l[ch][0].norm_sq()
                + spec_l[ch][nb - 1].norm_sq()
                + 2.0
                    * spec_l[ch][1..nb - 1]
                        .iter()
                        .map(|c| c.norm_sq())
                        .sum::<f64>())
                * inv_n;
            let pr = (spec_r[ch][0].norm_sq()
                + spec_r[ch][nb - 1].norm_sq()
                + 2.0
                    * spec_r[ch][1..nb - 1]
                        .iter()
                        .map(|c| c.norm_sq())
                        .sum::<f64>())
                * inv_n;
            tot_p += 0.5 * (pl + pr);
        }
        let g_bb = (TARGET_DF_POWER / tot_p.max(1e-12)).sqrt();
        for ch in 0..SLOTS {
            if ch == SLOT_LFE {
                continue;
            }
            for k in 0..nb {
                spec_l[ch][k] = spec_l[ch][k].scale(g_bb);
                spec_r[ch][k] = spec_r[ch][k].scale(g_bb);
            }
        }

        // 6. Synthesize LFE: mono `1/sqrt(2)` per ear (so `gL^2 + gR^2 = 1.0`,
        //    matching stock `g6ct` / Nuendo 0 dB discrete LFE energy by default),
        //    scaled by `lfe_gain_db`, aligned to `pre_delay_s`, optional causal
        //    120 Hz 4th-order Butterworth LPF, zero reflections.
        let lfe_gain = std::f64::consts::FRAC_1_SQRT_2 * 10.0_f64.powf(cfg.lfe_gain_db / 20.0);
        for (k, &f) in freqs.iter().enumerate() {
            let phase = Complex::from_polar(1.0, -2.0 * PI * f * pre_delay_s);
            let h = if cfg.lfe_lpf {
                butter4_lpf(f, 120.0_f64.min(0.45 * fs), fs)
            } else {
                Complex::ONE
            };
            let v = h.mul(phase).scale(lfe_gain);
            spec_l[SLOT_LFE][k] = v;
            spec_r[SLOT_LFE][k] = v;
        }

        // 7. Bauer (1961) / Linkwitz (1971) / Völk–Fastl (AES 2011) causal minimum-phase
        //    coherent LF trim shelf at 340 Hz across the full-range ring (and discrete
        //    `LFE` only when `cfg.lfe_lf_trim` is enabled).
        if cfg.bass_trim_db.abs() > 1e-6 {
            let fc_lf = 340.0_f64.min(0.45 * fs);
            let g_lf = 10.0_f64.powf(cfg.bass_trim_db / 20.0);
            let lf_mag: Vec<f64> = freqs
                .iter()
                .map(|&f| {
                    let x = f / fc_lf;
                    ((g_lf * g_lf + x * x) / (1.0 + x * x)).sqrt()
                })
                .collect();
            let lf_min = minimum_phase_spectrum(&lf_mag, &plan);
            for ch in 0..SLOTS {
                if ch == SLOT_LFE && !cfg.lfe_lf_trim {
                    continue;
                }
                for k in 0..nb {
                    spec_l[ch][k] = spec_l[ch][k].mul(lf_min[k]);
                    spec_r[ch][k] = spec_r[ch][k].mul(lf_min[k]);
                }
            }
        }

        // 8. IFFT to time domain, truncate to `N_IR`, and apply half-Kaiser tail taper.
        let mut scratch = vec![Complex::ZERO; N_SYNTH];
        let mut time_buf = vec![0.0_f64; N_SYNTH];
        let mut ir_l = vec![vec![0.0_f64; N_IR]; SLOTS];
        let mut ir_r = vec![vec![0.0_f64; N_IR]; SLOTS];

        for ch in 0..SLOTS {
            plan.irfft(&spec_l[ch], &mut scratch, &mut time_buf);
            ir_l[ch].copy_from_slice(&time_buf[..N_IR]);
            half_kaiser_tail_apply(&mut ir_l[ch], KAISER_TAIL, KAISER_BETA);

            plan.irfft(&spec_r[ch], &mut scratch, &mut time_buf);
            ir_r[ch].copy_from_slice(&time_buf[..N_IR]);
            half_kaiser_tail_apply(&mut ir_r[ch], KAISER_TAIL, KAISER_BETA);
        }

        Self {
            sample_rate,
            cfg,
            ir_l,
            ir_r,
            pre_delay_samples,
        }
    }
}

/// Apply Shirley, Kendrick & Churchill (JAES 2007) adaptive 1-ERB phantom-center
/// comb-notch compensation identically to a frontal L/R speaker pair `(ch_l, ch_r)`.
#[allow(clippy::too_many_arguments)]
fn apply_frontal_decomb(
    ch_l: usize,
    ch_r: usize,
    freqs: &[f64],
    dir_l: &[Vec<Complex>],
    dir_r: &[Vec<Complex>],
    spec_l: &mut [Vec<Complex>],
    spec_r: &mut [Vec<Complex>],
    strength: f64,
    plan: &FftPlan,
) {
    let nb = freqs.len();
    let max_boost = 10.0_f64.powf(3.0 / 20.0);
    let mut p_coh = Vec::with_capacity(nb);
    let mut p_inc = Vec::with_capacity(nb);
    for k in 0..nb {
        let sl = dir_l[ch_l][k].add(dir_l[ch_r][k]).norm_sq();
        let sr = dir_r[ch_l][k].add(dir_r[ch_r][k]).norm_sq();
        p_coh.push(0.5 * (sl + sr));
        p_inc.push(
            0.5 * (dir_l[ch_l][k].norm_sq()
                + dir_l[ch_r][k].norm_sq()
                + dir_r[ch_l][k].norm_sq()
                + dir_r[ch_r][k].norm_sq()),
        );
    }
    let sm_coh = roex_smooth_power(freqs, &p_coh, 1.0);
    let sm_inc = roex_smooth_power(freqs, &p_inc, 1.0);
    let dc_mag: Vec<f64> = freqs
        .iter()
        .enumerate()
        .map(|(k, &f)| {
            let fade = ((f - 450.0) / 500.0).clamp(0.0, 1.0);
            let deficit = if sm_coh[k] < sm_inc[k] {
                sm_inc[k] / sm_coh[k].max(1e-12)
            } else {
                1.0
            };
            deficit.powf(0.5 * strength * fade).min(max_boost)
        })
        .collect();
    let dc_min = minimum_phase_spectrum(&dc_mag, plan);
    for &ch in &[ch_l, ch_r] {
        for k in 0..nb {
            spec_l[ch][k] = spec_l[ch][k].mul(dc_min[k]);
            spec_r[ch][k] = spec_r[ch][k].mul(dc_min[k]);
        }
    }
}

/// Synthesize one full-range channel's un-equalized anechoic direct spectrum
/// `(H_L, H_R)` and its channel weight.
#[allow(clippy::too_many_arguments)]
fn synthesize_direct_spectrum(
    ch: usize,
    cfg: &PlusCfg,
    fs: f64,
    pre_delay_s: f64,
    freqs: &[f64],
    lr4_lo: &[Complex],
    lr4_hi: &[Complex],
    ap_rear: &[Complex],
    plan: &FftPlan,
) -> (Vec<Complex>, Vec<Complex>, f64) {
    let nb = freqs.len();
    let base_az = SLOT_AZ_DEG[ch];
    let is_front_pair = matches!(ch, SLOT_FL | SLOT_FR | SLOT_FLC | SLOT_FRC);
    let is_side = matches!(ch, SLOT_SL | SLOT_SR);
    let is_back = matches!(ch, SLOT_BL | SLOT_BR | SLOT_BC);
    let is_center = ch == SLOT_FC;

    // Effective speaker azimuth & head-shadow depth.
    let eff_az = if is_front_pair {
        (base_az * cfg.width).clamp(-65.0, 65.0)
    } else {
        base_az
    };
    let shadow_depth = if is_front_pair {
        (0.85 * cfg.width).clamp(0.0, 1.0)
    } else if is_side || is_back {
        (0.95 * cfg.surround_depth).clamp(0.35, 1.15)
    } else {
        0.0
    };

    // Per-channel relative gain weight (DFEQ normalizes total 7.0 power back to 3.5).
    let chan_weight = if is_center {
        // Higher center_focus gently anchors dialogue clarity (+0.5 dB at focus=1.0).
        0.94 + 0.12 * cfg.center_focus
    } else if is_side || is_back {
        // Surround depth smoothly scales surround immersion (0.80 .. 1.20).
        0.60 + 0.40 * cfg.surround_depth
    } else {
        1.0
    };

    // 1. Direct-path magnitude responses: Brown–Duda spherical head shadow +
    //    broad non-individualized front–back pinna shading.
    let mut mag_l = Vec::with_capacity(nb);
    let mut mag_r = Vec::with_capacity(nb);
    let pinna_scale = if is_side || is_back {
        cfg.surround_depth.clamp(0.5, 1.3)
    } else {
        1.0
    };
    for &f in freqs {
        let bdl = brown_duda_mag(f, eff_az, 1.0, shadow_depth);
        let bdr = brown_duda_mag(f, eff_az, -1.0, shadow_depth);
        let pinna = broad_pinna_mag(f, eff_az, ch == SLOT_TC, pinna_scale);
        mag_l.push(bdl * pinna);
        mag_r.push(bdr * pinna);
    }

    // 2. Hilbert-cepstral minimum-phase conversion.
    let hl_min = minimum_phase_spectrum(&mag_l, plan);
    let hr_min = minimum_phase_spectrum(&mag_r, plan);

    // 3. Kulkarni–Isabelle–Colburn (1999) excess Woodworth ITD:
    //    measure the natural minimum-phase ITD already present in `(hl_min, hr_min)`
    //    between 350 Hz and 900 Hz, then add only the remaining excess propagation delay.
    let ww_base = woodworth_itd_s(eff_az);
    let target_itd = if is_side || is_back {
        ww_base * (0.85 + 0.15 * cfg.surround_depth).clamp(0.70, 1.15)
    } else {
        ww_base
    };
    let nat_itd = measure_unwrapped_itd_s(freqs, &hl_min, &hr_min, 350.0, 900.0);
    // Positive `eff_az` is left => Left ear leads (`hl` ahead of `hr`), so contralateral
    // Right ear gets `tau_r = excess`, ipsilateral Left ear gets `tau_l = 0`.
    let (tau_l, tau_r) = if eff_az > 1e-6 {
        (0.0, (target_itd - nat_itd.max(0.0)).max(0.0))
    } else if eff_az < -1e-6 {
        ((target_itd - (-nat_itd).max(0.0)).max(0.0), 0.0)
    } else {
        (0.0, 0.0)
    };

    // 4. Assemble direct path with causal LR4 bass-mono crossover, causal Thiran
    //    fractional delay (zero sinc pre-ringing), & DEE-safe rear all-pass.
    let base_scale = std::f64::consts::FRAC_1_SQRT_2 * chan_weight;
    let use_rear_ap = matches!(ch, SLOT_BL | SLOT_BR);
    let pre_smp = pre_delay_s * fs;
    let total_l_smp = pre_smp + tau_l * fs;
    let total_r_smp = pre_smp + tau_r * fs;

    let mut sl = Vec::with_capacity(nb);
    let mut sr = Vec::with_capacity(nb);
    for (k, &f) in freqs.iter().enumerate() {
        let d_pre = causal_frac_delay(f, pre_smp, fs);
        let d_l = causal_frac_delay(f, total_l_smp, fs);
        let d_r = causal_frac_delay(f, total_r_smp, fs);

        let mut hi_l = hl_min[k].mul(d_l);
        let mut hi_r = hr_min[k].mul(d_r);
        if use_rear_ap {
            hi_l = hi_l.mul(ap_rear[k]);
            hi_r = hi_r.mul(ap_rear[k]);
        }

        let dl = lr4_lo[k]
            .mul(d_pre)
            .add(lr4_hi[k].mul(hi_l))
            .scale(base_scale);
        let dr = lr4_lo[k]
            .mul(d_pre)
            .add(lr4_hi[k].mul(hi_r))
            .scale(base_scale);
        sl.push(dl);
        sr.push(dr);
    }

    (sl, sr, chan_weight)
}

/// Add smooth dry ITU-R BS.1116 / Hassager–Gran–Dau lateral early reflections
/// (`>= 8.5 ms`) weighted by direct diffuse equalization `dfeq_min`.
#[allow(clippy::too_many_arguments)]
fn add_channel_reflections(
    ch: usize,
    cfg: &PlusCfg,
    fs: f64,
    pre_delay_s: f64,
    chan_weight: f64,
    freqs: &[f64],
    dfeq_min: &[Complex],
    sl: &mut [Complex],
    sr: &mut [Complex],
    plan: &FftPlan,
) {
    let nb = freqs.len();
    let base_az = SLOT_AZ_DEG[ch];
    let is_center = ch == SLOT_FC;
    let refl_scale = if is_center {
        cfg.externalize * (1.0 - 0.50 * cfg.center_focus)
    } else {
        cfg.externalize
    };
    if refl_scale <= 1e-6 {
        return;
    }

    let base_scale = std::f64::consts::FRAC_1_SQRT_2 * chan_weight;
    let taps = itu_bs1116_reflections(ch);
    let hpf_base: f64 = if base_az.abs() < 1e-6 { 520.0 } else { 400.0 };
    let hpf_fc = hpf_base.min(0.4 * fs);

    for tap in taps {
        let g = tap.gain * refl_scale * base_scale;
        if g.abs() <= 1e-9 {
            continue;
        }
        let r_itd = woodworth_itd_s(tap.az_deg);
        let (rt_l, rt_r) = if tap.az_deg >= 0.0 {
            (0.0, r_itd)
        } else {
            (r_itd, 0.0)
        };
        let del_l_smp = (pre_delay_s + tap.delay_s + rt_l) * fs;
        let del_r_smp = (pre_delay_s + tap.delay_s + rt_r) * fs;
        let lpf_fc = tap.lpf_hz.min(0.45 * fs);
        for (k, &f) in freqs.iter().enumerate() {
            let hpf2 = butter2_hpf(f, hpf_fc, fs);
            let wall = hpf2
                .mul(hpf2)
                .mul(first_order_lpf(f, lpf_fc, fs))
                .mul(dfeq_min[k]);
            let bdl = brown_duda_mag(f, tap.az_deg, 1.0, 0.75);
            let bdr = brown_duda_mag(f, tap.az_deg, -1.0, 0.75);
            let dl = causal_frac_delay(f, del_l_smp, fs).scale(g * bdl);
            let dr = causal_frac_delay(f, del_r_smp, fs).scale(g * bdr);
            sl[k] = sl[k].add(wall.mul(dl));
            sr[k] = sr[k].add(wall.mul(dr));
        }
    }

    // Center-axis channels (`FC`, `BC`, `TC`): symmetric L/R 1-ERB balance preserves
    // exact 0.00 dB phantom center alignment across 20 Hz – 20 kHz.
    if base_az.abs() < 1e-6 {
        let pl: Vec<f64> = sl.iter().map(|c| c.norm_sq()).collect();
        let pr: Vec<f64> = sr.iter().map(|c| c.norm_sq()).collect();
        let sml = roex_smooth_power(freqs, &pl, 0.75);
        let smr = roex_smooth_power(freqs, &pr, 0.75);
        let gl_mag: Vec<f64> = sml
            .iter()
            .zip(&smr)
            .map(|(&l, &r)| (0.5 * (l + r) / l.max(1e-18)).sqrt())
            .collect();
        let gr_mag: Vec<f64> = sml
            .iter()
            .zip(&smr)
            .map(|(&l, &r)| (0.5 * (l + r) / r.max(1e-18)).sqrt())
            .collect();
        let gl = minimum_phase_spectrum(&gl_mag, plan);
        let gr = minimum_phase_spectrum(&gr_mag, plan);
        for k in 0..nb {
            sl[k] = sl[k].mul(gl[k]);
            sr[k] = sr[k].mul(gr[k]);
        }
    }
}

/// Brown & Duda (1998) Eq. (5) spherical head-shadow magnitude transfer function.
///
/// `az_deg`: source azimuth (`+` = left, `-` = right).
/// `ear_sign`: `+1.0` for left ear (`+90°`), `-1.0` for right ear (`-90°`).
#[inline]
fn brown_duda_mag(f_hz: f64, az_deg: f64, ear_sign: f64, depth: f64) -> f64 {
    if depth <= 0.0 {
        return 1.0;
    }
    let ear_az = 90.0 * ear_sign;
    let mut diff = (az_deg - ear_az).abs();
    if diff > 180.0 {
        diff = 360.0 - diff;
    }
    let alpha_min = 0.1;
    let theta_min = 150.0;
    let alpha_raw =
        (1.0 + 0.5 * alpha_min) + (1.0 - 0.5 * alpha_min) * (diff * PI / theta_min).cos();
    let alpha = (1.0 + depth * (alpha_raw - 1.0)).clamp(0.08, 2.15);
    let w0 = SPEED_OF_SOUND / HEAD_RADIUS_M;
    let x = (2.0 * PI * f_hz) / w0;
    ((alpha * alpha * x * x + 1.0) / (x * x + 1.0)).sqrt()
}

/// Broad, non-individualized front–back / overhead pinna shading.
/// Intentionally free of sharp individualized notches so it translates across
/// all ear anatomies and headphones.
#[inline]
fn broad_pinna_mag(f_hz: f64, az_deg: f64, is_top: bool, scale: f64) -> f64 {
    if is_top {
        // Gentle overhead clarity cue (+1.2 dB broad shelf above 5 kHz).
        let x = f_hz / 5_000.0;
        let g = 10.0_f64.powf(1.2 / 20.0);
        return ((g * g * x * x + 1.0) / (x * x + 1.0)).sqrt();
    }
    let abs_az = az_deg.abs();
    if abs_az <= 60.0 {
        return 1.0;
    }
    if abs_az <= 120.0 {
        // Side channels: subtle -1.0 dB broad pinna shelf above 4 kHz.
        let db = -scale;
        let g = 10.0_f64.powf(db / 20.0);
        let x = f_hz / 4_000.0;
        return ((g * g * x * x + 1.0) / (x * x + 1.0)).sqrt();
    }
    // Rear channels (`|az| > 120°`): broad -3.0 dB HF pinna shadow above 3.2 kHz
    // plus gentle -1.4 dB wide Blauert rear dip at 1.2 kHz (`Q = 1.1`).
    let shelf_db = -3.0 * scale;
    let g_shelf = 10.0_f64.powf(shelf_db / 20.0);
    let x = f_hz / 3_200.0;
    let shelf = ((g_shelf * g_shelf * x * x + 1.0) / (x * x + 1.0)).sqrt();

    let bell_db = if f_hz > 20.0 {
        let oct = (f_hz / 1_200.0).log2();
        -1.4 * scale * (-0.5 * (oct / 0.55) * (oct / 0.55)).exp()
    } else {
        0.0
    };
    shelf * 10.0_f64.powf(bell_db / 20.0)
}

/// Woodworth spherical-head interaural time delay (seconds, non-negative).
#[inline]
pub fn woodworth_itd_s(az_deg: f64) -> f64 {
    let r = az_deg.abs().to_radians().min(PI);
    let path = if r <= 0.5 * PI {
        r + r.sin()
    } else {
        PI - r + r.sin()
    };
    (HEAD_RADIUS_M / SPEED_OF_SOUND) * path
}

/// Measure unwrapped average interaural phase delay `(phi_L - phi_R) / (2*pi*f)`
/// in seconds over `[f_lo, f_hi]` Hz. Positive means Left ear leads Right ear.
pub fn measure_unwrapped_itd_s(
    freqs: &[f64],
    spec_l: &[Complex],
    spec_r: &[Complex],
    f_lo: f64,
    f_hi: f64,
) -> f64 {
    let mut prev: Option<f64> = None;
    let mut acc_phi = 0.0_f64;
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for k in 1..freqs.len() {
        let f = freqs[k];
        // Cross-spectrum phase: angle(L * conj(R)) = phi_L - phi_R.
        // If R is delayed by tau > 0 relative to L, phi_R = phi_L - 2*pi*f*tau,
        // so phi_L - phi_R = +2*pi*f*tau > 0.
        let cross = spec_l[k].mul(spec_r[k].conj());
        let phi = cross.arg();
        if let Some(p) = prev {
            let mut d = phi - p;
            while d > PI {
                d -= 2.0 * PI;
            }
            while d < -PI {
                d += 2.0 * PI;
            }
            acc_phi += d;
        } else {
            acc_phi = phi;
        }
        prev = Some(phi);
        if f >= f_lo && f <= f_hi {
            sum += acc_phi / (2.0 * PI * f);
            count += 1;
        }
    }
    if count > 0 {
        sum / (count as f64)
    } else {
        0.0
    }
}

/// Bilinear-transformed discrete-time causal 4th-order Linkwitz–Riley crossover
/// pair `(H_lo(f), H_hi(f))`. Strictly causal, `|H_lo + H_hi| == 1.0` everywhere.
#[inline]
fn lr4_pair(f_hz: f64, fc_hz: f64, fs: f64) -> (Complex, Complex) {
    let k = (PI * fc_hz / fs).tan();
    let sq2 = std::f64::consts::SQRT_2;
    let k2 = k * k;
    let norm = 1.0 / (1.0 + sq2 * k + k2);

    let b0_l = k2 * norm;
    let b1_l = 2.0 * b0_l;
    let b2_l = b0_l;

    let b0_h = norm;
    let b1_h = -2.0 * norm;
    let b2_h = norm;

    let a1 = 2.0 * (k2 - 1.0) * norm;
    let a2 = (1.0 - sq2 * k + k2) * norm;

    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    let z2 = z1.mul(z1);
    let den = Complex::ONE.add(z1.scale(a1)).add(z2.scale(a2));

    let hlp = Complex::new(b0_l, 0.0)
        .add(z1.scale(b1_l))
        .add(z2.scale(b2_l))
        .div(den);
    let hhp = Complex::new(b0_h, 0.0)
        .add(z1.scale(b1_h))
        .add(z2.scale(b2_h))
        .div(den);
    (hlp.mul(hlp), hhp.mul(hhp))
}

/// Bilinear-transformed causal 4th-order Butterworth low-pass filter (`LFE` 120 Hz).
#[inline]
fn butter4_lpf(f_hz: f64, fc_hz: f64, fs: f64) -> Complex {
    let k = (PI * fc_hz / fs).tan();
    let k2 = k * k;
    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    let z2 = z1.mul(z1);

    // 4th-order Butterworth Q values: 1/(2*cos(pi/8)) and 1/(2*cos(3*pi/8)).
    let qs = [0.541_196_100_146_197, 1.306_562_964_876_376_6];
    let mut acc = Complex::ONE;
    for q in qs {
        let inv_q = 1.0 / q;
        let norm = 1.0 / (1.0 + inv_q * k + k2);
        let b0 = k2 * norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k2 - 1.0) * norm;
        let a2 = (1.0 - inv_q * k + k2) * norm;
        let num = Complex::new(b0, 0.0).add(z1.scale(b1)).add(z2.scale(b2));
        let den = Complex::ONE.add(z1.scale(a1)).add(z2.scale(a2));
        acc = acc.mul(num.div(den));
    }
    acc
}

/// Bilinear causal 1st-order low-pass filter.
#[inline]
fn first_order_lpf(f_hz: f64, fc_hz: f64, fs: f64) -> Complex {
    let k = (PI * fc_hz / fs).tan();
    let norm = 1.0 / (1.0 + k);
    let b0 = k * norm;
    let a1 = (k - 1.0) * norm;
    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    let num = Complex::new(b0, 0.0).add(z1.scale(b0));
    let den = Complex::ONE.add(z1.scale(a1));
    num.div(den)
}

/// Strictly causal fractional sample delay (`total_samples >= 2.0`) combining an
/// integer sample shift `z^{-d_int}` with a 2nd-order Thiran all-pass filter
/// (`D = 2.0 + frac`, `|H| == 1.0` to machine precision, zero sinc pre-ringing).
#[inline]
fn causal_frac_delay(f_hz: f64, total_samples: f64, fs: f64) -> Complex {
    if total_samples < 2.0 {
        let d_int = total_samples.round().max(0.0);
        return Complex::from_polar(1.0, -2.0 * PI * f_hz * d_int / fs);
    }
    let d_int = (total_samples - 2.0).floor();
    let frac = (total_samples - 2.0 - d_int).clamp(0.0, 0.999_999);
    let d = 2.0 + frac;
    let a1 = -2.0 * (d - 2.0) / (d + 1.0);
    let a2 = ((d - 1.0) * (d - 2.0)) / ((d + 1.0) * (d + 2.0));

    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    let z2 = z1.mul(z1);
    let num = Complex::new(a2, 0.0).add(z1.scale(a1)).add(z2);
    let den = Complex::ONE.add(z1.scale(a1)).add(z2.scale(a2));
    let ap = num.div(den);
    let shift = Complex::from_polar(1.0, -2.0 * PI * f_hz * d_int / fs);
    shift.mul(ap)
}

/// Bilinear causal 2nd-order Butterworth high-pass filter (studio reflection low-cut).
#[inline]
fn butter2_hpf(f_hz: f64, fc_hz: f64, fs: f64) -> Complex {
    let k = (PI * fc_hz / fs).tan();
    let k2 = k * k;
    let sq2 = std::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + sq2 * k + k2);
    let b0 = norm;
    let b1 = -2.0 * norm;
    let b2 = norm;
    let a1 = 2.0 * (k2 - 1.0) * norm;
    let a2 = (1.0 - sq2 * k + k2) * norm;
    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    let z2 = z1.mul(z1);
    let num = Complex::new(b0, 0.0).add(z1.scale(b1)).add(z2.scale(b2));
    let den = Complex::ONE.add(z1.scale(a1)).add(z2.scale(a2));
    num.div(den)
}

/// Dry ITU-R BS.1116 control-room (`6.0 × 5.0 × 3.0 m`) lateral early-reflection
/// clusters (`>= 8.5 ms` to avoid `< 6 ms` floor/desk comb coloration).
///
/// L/R pairs (`FL/FR`, `SL/SR`, `BL/BR`, `FLC/FRC`) are exact mirror images so
/// stereo symmetry holds to `< 0.01 dB`. Center (`FC`) uses equal-delay,
/// opposite-polarity bilateral lateral pairs so the early reflections are maximally
/// decorrelated across ears (lowering `IACC` cleanly into the dry externalized zone
/// `~0.85` at `Studio` while keeping `DRR >= +8 dB` and `< 0.05 dB` ERB L/R balance).
fn itu_bs1116_reflections(ch: usize) -> [ReflectionTap; 6] {
    fn mirror(taps: [ReflectionTap; 6]) -> [ReflectionTap; 6] {
        taps.map(|t| ReflectionTap {
            az_deg: -t.az_deg,
            ..t
        })
    }

    let fl = [
        ReflectionTap {
            delay_s: 0.0092,
            az_deg: 68.0,
            gain: 0.25,
            lpf_hz: 6_200.0,
        },
        ReflectionTap {
            delay_s: 0.0138,
            az_deg: -62.0,
            gain: -0.21,
            lpf_hz: 5_600.0,
        },
        ReflectionTap {
            delay_s: 0.0169,
            az_deg: 112.0,
            gain: 0.16,
            lpf_hz: 4_800.0,
        },
        ReflectionTap {
            delay_s: 0.0196,
            az_deg: -108.0,
            gain: -0.14,
            lpf_hz: 4_400.0,
        },
        ReflectionTap {
            delay_s: 0.0228,
            az_deg: 54.0,
            gain: 0.12,
            lpf_hz: 4_000.0,
        },
        ReflectionTap {
            delay_s: 0.0259,
            az_deg: -58.0,
            gain: -0.10,
            lpf_hz: 3_600.0,
        },
    ];

    let fc = [
        ReflectionTap {
            delay_s: 0.0182,
            az_deg: 68.0,
            gain: 0.46,
            lpf_hz: 6_000.0,
        },
        ReflectionTap {
            delay_s: 0.0182,
            az_deg: -68.0,
            gain: -0.46,
            lpf_hz: 6_000.0,
        },
        ReflectionTap {
            delay_s: 0.0231,
            az_deg: 108.0,
            gain: -0.38,
            lpf_hz: 4_800.0,
        },
        ReflectionTap {
            delay_s: 0.0231,
            az_deg: -108.0,
            gain: 0.38,
            lpf_hz: 4_800.0,
        },
        ReflectionTap {
            delay_s: 0.0284,
            az_deg: 54.0,
            gain: 0.30,
            lpf_hz: 4_000.0,
        },
        ReflectionTap {
            delay_s: 0.0284,
            az_deg: -54.0,
            gain: -0.30,
            lpf_hz: 4_000.0,
        },
    ];

    let sl = [
        ReflectionTap {
            delay_s: 0.0088,
            az_deg: 76.0,
            gain: 0.22,
            lpf_hz: 5_500.0,
        },
        ReflectionTap {
            delay_s: 0.0144,
            az_deg: -72.0,
            gain: -0.18,
            lpf_hz: 5_000.0,
        },
        ReflectionTap {
            delay_s: 0.0181,
            az_deg: 126.0,
            gain: 0.15,
            lpf_hz: 4_400.0,
        },
        ReflectionTap {
            delay_s: 0.0213,
            az_deg: -118.0,
            gain: -0.13,
            lpf_hz: 4_000.0,
        },
        ReflectionTap {
            delay_s: 0.0246,
            az_deg: 64.0,
            gain: 0.11,
            lpf_hz: 3_600.0,
        },
        ReflectionTap {
            delay_s: 0.0274,
            az_deg: -66.0,
            gain: -0.09,
            lpf_hz: 3_300.0,
        },
    ];

    let bl = [
        ReflectionTap {
            delay_s: 0.0095,
            az_deg: 118.0,
            gain: 0.21,
            lpf_hz: 5_000.0,
        },
        ReflectionTap {
            delay_s: 0.0151,
            az_deg: -112.0,
            gain: -0.17,
            lpf_hz: 4_600.0,
        },
        ReflectionTap {
            delay_s: 0.0188,
            az_deg: 78.0,
            gain: 0.14,
            lpf_hz: 4_200.0,
        },
        ReflectionTap {
            delay_s: 0.0221,
            az_deg: -82.0,
            gain: -0.12,
            lpf_hz: 3_800.0,
        },
        ReflectionTap {
            delay_s: 0.0252,
            az_deg: 132.0,
            gain: 0.10,
            lpf_hz: 3_500.0,
        },
        ReflectionTap {
            delay_s: 0.0280,
            az_deg: -128.0,
            gain: -0.09,
            lpf_hz: 3_200.0,
        },
    ];

    match ch {
        SLOT_FL | SLOT_FLC => fl,
        SLOT_FR | SLOT_FRC => mirror(fl),
        SLOT_SL => sl,
        SLOT_SR => mirror(sl),
        SLOT_BL => bl,
        SLOT_BR => mirror(bl),
        _ => fc, // FC, BC, TC
    }
}

/// Streaming overlap-add `G6+` binaural renderer.
///
/// Processes arbitrary block sizes `n` with 1:1 immediate frame emission
/// (`1.0 ms` causal FIR pre-delay, zero block-buffering latency) and exposes
/// [`Self::flush`] to drain the remaining `N_IR - 1` impulse tail at EOF.
#[derive(Debug, Clone)]
pub struct BinauralPlus {
    mask: u32,
    sample_rate: u32,
    cfg: PlusCfg,
    pre_delay_samples: usize,
    plan: FftPlan,
    /// Pre-transformed `N_FFT/2 + 1` frequency-domain kernels for each slot.
    kern_l: Vec<Vec<Complex>>,
    kern_r: Vec<Vec<Complex>>,
    /// Overlap tails (`N_IR - 1` samples) carried between blocks.
    ov_l: Vec<f64>,
    ov_r: Vec<f64>,
    /// Reusable per-block scratch buffers (zero allocation in `process`).
    in_pad: Vec<f64>,
    fft_scratch: Vec<Complex>,
    x_bins: Vec<Complex>,
    acc_l: Vec<Complex>,
    acc_r: Vec<Complex>,
    y_l: Vec<f64>,
    y_r: Vec<f64>,
    finite: bool,
}

impl BinauralPlus {
    pub fn new(mask: u32, cfg: PlusCfg, sample_rate: u32) -> Self {
        let bank = KernelBank::synthesize(cfg, sample_rate);
        let plan = FftPlan::new(N_FFT);
        let nb = N_FFT / 2 + 1;
        let mut scratch = vec![Complex::ZERO; N_FFT];
        let mut pad = vec![0.0_f64; N_FFT];
        let mut kern_l = vec![vec![Complex::ZERO; nb]; SLOTS];
        let mut kern_r = vec![vec![Complex::ZERO; nb]; SLOTS];

        for ch in 0..SLOTS {
            pad.fill(0.0);
            pad[..N_IR].copy_from_slice(&bank.ir_l[ch]);
            plan.rfft(&pad, &mut scratch, &mut kern_l[ch]);

            pad.fill(0.0);
            pad[..N_IR].copy_from_slice(&bank.ir_r[ch]);
            plan.rfft(&pad, &mut scratch, &mut kern_r[ch]);
        }

        Self {
            mask,
            sample_rate,
            cfg,
            pre_delay_samples: bank.pre_delay_samples,
            plan,
            kern_l,
            kern_r,
            ov_l: vec![0.0_f64; N_IR - 1],
            ov_r: vec![0.0_f64; N_IR - 1],
            in_pad: vec![0.0_f64; N_FFT],
            fft_scratch: vec![Complex::ZERO; N_FFT],
            x_bins: vec![Complex::ZERO; nb],
            acc_l: vec![Complex::ZERO; nb],
            acc_r: vec![Complex::ZERO; nb],
            y_l: vec![0.0_f64; N_FFT],
            y_r: vec![0.0_f64; N_FFT],
            finite: true,
        }
    }

    pub fn cfg(&self) -> PlusCfg {
        self.cfg
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn latency_samples(&self) -> usize {
        if self.mask == MASK_STEREO {
            0
        } else {
            self.pre_delay_samples
        }
    }

    pub fn is_finite(&self) -> bool {
        self.finite
            && self.ov_l.iter().all(|v| v.is_finite())
            && self.ov_r.iter().all(|v| v.is_finite())
    }

    /// Process `frames` samples of planar input slots into stereo `(out_l, out_r)`.
    /// Mirrors `Dsp::process` exactly: stereo input (`MASK_STEREO`) passes through
    /// untouched; multi-channel inputs are convolved with exact streaming overlap-add.
    pub fn process(
        &mut self,
        inputs: &[Option<&[f64]>; SLOTS],
        out_l: &mut [f64],
        out_r: &mut [f64],
        frames: usize,
    ) {
        if frames == 0 {
            return;
        }
        if self.mask == MASK_STEREO {
            if let Some(l) = inputs[SLOT_FL] {
                out_l[..frames].copy_from_slice(&l[..frames]);
            } else {
                out_l[..frames].fill(0.0);
            }
            if let Some(r) = inputs[SLOT_FR] {
                out_r[..frames].copy_from_slice(&r[..frames]);
            } else {
                out_r[..frames].fill(0.0);
            }
            return;
        }

        let mut offset = 0usize;
        while offset < frames {
            let m = (frames - offset).min(OLA_HOP);
            self.process_hop(
                inputs,
                offset,
                &mut out_l[offset..offset + m],
                &mut out_r[offset..offset + m],
                m,
            );
            offset += m;
        }
    }

    fn process_hop(
        &mut self,
        inputs: &[Option<&[f64]>; SLOTS],
        offset: usize,
        out_l: &mut [f64],
        out_r: &mut [f64],
        m: usize,
    ) {
        debug_assert!((1..=OLA_HOP).contains(&m));
        let nb = N_FFT / 2 + 1;
        self.acc_l.fill(Complex::ZERO);
        self.acc_r.fill(Complex::ZERO);

        for (ch, slot) in inputs.iter().enumerate() {
            let Some(buf) = slot else { continue };
            let slice = &buf[offset..offset + m];
            // Fast silence skip per channel.
            if !slice.iter().any(|&v| v != 0.0) {
                continue;
            }
            self.in_pad[..m].copy_from_slice(slice);
            self.in_pad[m..].fill(0.0);
            self.plan
                .rfft(&self.in_pad, &mut self.fft_scratch, &mut self.x_bins);

            let kl = &self.kern_l[ch];
            let kr = &self.kern_r[ch];
            for k in 0..nb {
                let x = self.x_bins[k];
                self.acc_l[k] = self.acc_l[k].add(x.mul(kl[k]));
                self.acc_r[k] = self.acc_r[k].add(x.mul(kr[k]));
            }
        }

        self.plan
            .irfft(&self.acc_l, &mut self.fft_scratch, &mut self.y_l);
        self.plan
            .irfft(&self.acc_r, &mut self.fft_scratch, &mut self.y_r);

        let ov_len = N_IR - 1;
        for i in 0..m {
            let ol = if i < ov_len { self.ov_l[i] } else { 0.0 };
            let or = if i < ov_len { self.ov_r[i] } else { 0.0 };
            let vl = self.y_l[i] + ol;
            let vr = self.y_r[i] + or;
            if !vl.is_finite() || !vr.is_finite() {
                self.finite = false;
            }
            out_l[i] = vl;
            out_r[i] = vr;
        }

        // Shift overlap buffer left by `m` and add the new block's convolution tail.
        // Since `m >= 1`, `m + j > j`, so ascending `j` is strictly safe in-place.
        for j in 0..ov_len {
            let src = m + j;
            let prev_l = if src < ov_len { self.ov_l[src] } else { 0.0 };
            let prev_r = if src < ov_len { self.ov_r[src] } else { 0.0 };
            self.ov_l[j] = prev_l + self.y_l[src];
            self.ov_r[j] = prev_r + self.y_r[src];
        }
    }

    /// Drain the remaining `N_IR - 1` overlap-add impulse tail at EOF and reset
    /// the tail state to zero. Returns empty vectors for stereo bypass.
    pub fn flush(&mut self) -> (Vec<f64>, Vec<f64>) {
        if self.mask == MASK_STEREO {
            return (Vec::new(), Vec::new());
        }
        let tl = self.ov_l.clone();
        let tr = self.ov_r.clone();
        self.ov_l.fill(0.0);
        self.ov_r.fill(0.0);
        (tl, tr)
    }
}

/// Unified renderer dispatch used by `pipeline.rs`.
pub enum Renderer {
    Classic(Box<Dsp<f64>>),
    Plus(Box<BinauralPlus>),
}

impl Renderer {
    pub fn new(mode: RendererMode, plus: PlusCfg, mask: u32, sample_rate: u32) -> Self {
        match mode {
            RendererMode::Reference => Self::Classic(Box::new(Dsp::<f64>::new(mask))),
            RendererMode::Plus => Self::Plus(Box::new(BinauralPlus::new(mask, plus, sample_rate))),
        }
    }

    pub fn process(
        &mut self,
        inputs: &[Option<&[f64]>; SLOTS],
        out_l: &mut [f64],
        out_r: &mut [f64],
        frames: usize,
    ) {
        match self {
            Self::Classic(dsp) => dsp.process(inputs, out_l, out_r, frames),
            Self::Plus(plus) => plus.process(inputs, out_l, out_r, frames),
        }
    }

    pub fn is_finite(&self) -> bool {
        match self {
            Self::Classic(dsp) => dsp.is_finite(),
            Self::Plus(plus) => plus.is_finite(),
        }
    }

    pub fn flush(&mut self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Self::Classic(_) => (Vec::new(), Vec::new()),
            Self::Plus(plus) => plus.flush(),
        }
    }

    pub fn latency_samples(&self) -> usize {
        match self {
            Self::Classic(_) => 0,
            Self::Plus(plus) => plus.latency_samples(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::MASK_7_1;
    use crate::fft::rfft;
    use sha2::{Digest, Sha256};

    fn ir_spectrum_4096(ir: &[f64]) -> Vec<Complex> {
        let mut buf = vec![0.0_f64; 4096];
        buf[..ir.len()].copy_from_slice(ir);
        rfft(&buf)
    }

    /// Zaunschirm (JASA 2018) direct-path 1-ERB diffuse-field excitation across
    /// `20 Hz – 18 kHz` must stay within `±0.20 dB` across all profile geometries
    /// (`Pure`, `Studio`, `Cinema`), and total `(direct + reflections)` stays within
    /// `±0.25 dB` (`Pure`, `Studio`) / `[-0.30, +0.55] dB` (`Cinema` mid-band room presence).
    #[test]
    fn diffuse_field_erb_is_flat_across_profiles() {
        let nb = 4096 / 2 + 1;
        let freqs: Vec<f64> = (0..nb).map(|k| k as f64 * 48_000.0 / 4096.0).collect();
        let measure_erb_bounds = |cfg: PlusCfg| -> (f64, f64) {
            let bank = KernelBank::synthesize(cfg, 48_000);
            let mut p_df = vec![0.0_f64; nb];
            for &ch in &DF_RING_SLOTS {
                let sl = ir_spectrum_4096(&bank.ir_l[ch]);
                let sr = ir_spectrum_4096(&bank.ir_r[ch]);
                for k in 0..nb {
                    p_df[k] += 0.5 * (sl[k].norm_sq() + sr[k].norm_sq());
                }
            }
            let sm = roex_smooth_power(&freqs, &p_df, 1.0);
            let mut min_db = f64::INFINITY;
            let mut max_db = f64::NEG_INFINITY;
            for (k, &f) in freqs.iter().enumerate() {
                if (20.0..=18_000.0).contains(&f) {
                    let db = 10.0 * (sm[k] / TARGET_DF_POWER).log10();
                    min_db = min_db.min(db);
                    max_db = max_db.max(db);
                }
            }
            (min_db, max_db)
        };

        for profile in [PlusProfile::Pure, PlusProfile::Studio, PlusProfile::Cinema] {
            let mut direct = PlusCfg::from_profile(profile);
            direct.externalize = 0.0;
            direct.bass_trim_db = 0.0;
            direct.phantom_decomb = 0.0;
            let (d_min, d_max) = measure_erb_bounds(direct);
            assert!(
                d_min >= -0.20 && d_max <= 0.20,
                "{:?} direct-path DF ERB out of ±0.20 dB band: [{d_min:+.3}, {d_max:+.3}] dB",
                profile
            );

            let mut neutral = PlusCfg::from_profile(profile);
            neutral.bass_trim_db = 0.0;
            neutral.phantom_decomb = 0.0;
            let (t_min, t_max) = measure_erb_bounds(neutral);
            let (lo, hi) = if profile == PlusProfile::Cinema {
                (-0.30, 0.55)
            } else {
                (-0.25, 0.25)
            };
            assert!(
                t_min >= lo && t_max <= hi,
                "{:?} total neutral DF ERB out of [{lo:+.2}, {hi:+.2}] dB: [{t_min:+.3}, {t_max:+.3}] dB",
                profile
            );
        }
    }

    /// Correlated `FL=FR` stereo playback on `Studio` and `Cinema` must tame coherent
    /// low-frequency buildup (`50–100 Hz`) and Shirley–Kendrick phantom-center comb dip
    /// (`1.5–2.5 kHz`) compared to `Pure`, keeping `50 Hz vs 2 kHz` tilt `<= 2.2 dB`.
    #[test]
    fn cinema_and_studio_tame_coherent_lf_bass_buildup() {
        let nb = 4096 / 2 + 1;
        let freqs: Vec<f64> = (0..nb).map(|k| k as f64 * 48_000.0 / 4096.0).collect();
        let flfr_erb_db = |cfg: PlusCfg| -> Vec<f64> {
            let bank = KernelBank::synthesize(cfg, 48_000);
            let fl_l = ir_spectrum_4096(&bank.ir_l[SLOT_FL]);
            let fl_r = ir_spectrum_4096(&bank.ir_r[SLOT_FL]);
            let fr_l = ir_spectrum_4096(&bank.ir_l[SLOT_FR]);
            let fr_r = ir_spectrum_4096(&bank.ir_r[SLOT_FR]);
            let raw: Vec<f64> = (0..nb)
                .map(|k| 0.5 * (fl_l[k].add(fr_l[k]).norm_sq() + fl_r[k].add(fr_r[k]).norm_sq()))
                .collect();
            let sm = roex_smooth_power(&freqs, &raw, 1.0);
            sm.iter().map(|&p| 10.0 * p.max(1e-18).log10()).collect()
        };
        let bin = |hz: f64| -> usize { (hz * 4096.0 / 48_000.0).round() as usize };

        let pure_db = flfr_erb_db(PlusCfg::from_profile(PlusProfile::Pure));
        let studio_db = flfr_erb_db(PlusCfg::from_profile(PlusProfile::Studio));
        let cinema_db = flfr_erb_db(PlusCfg::from_profile(PlusProfile::Cinema));

        // Pure ITU DF sits ~+3 dB at 50 Hz and drops ~-2 dB at 2 kHz (> 4.5 dB tilt).
        let pure_tilt = pure_db[bin(50.0)] - pure_db[bin(2_000.0)];
        assert!(
            pure_tilt > 4.0,
            "Pure coherent FL=FR tilt expected > 4.0 dB, got {pure_tilt:+.2} dB"
        );

        // Cinema keeps sub-bass near Reference 0 dB and total 50 Hz-vs-2 kHz tilt <= 2.2 dB.
        assert!(
            cinema_db[bin(50.0)].abs() <= 1.2,
            "Cinema FL=FR 50 Hz vs Ref out of ±1.2 dB: {:+.2} dB",
            cinema_db[bin(50.0)]
        );
        let cinema_tilt = (cinema_db[bin(50.0)] - cinema_db[bin(2_000.0)]).abs();
        assert!(
            cinema_tilt <= 2.2,
            "Cinema coherent LF-vs-2kHz tilt still too high: {cinema_tilt:.2} dB"
        );
        let studio_tilt = (studio_db[bin(50.0)] - studio_db[bin(2_000.0)]).abs();
        assert!(
            studio_tilt < pure_tilt - 1.5,
            "Studio tilt ({studio_tilt:.2} dB) should improve cleanly over Pure ({pure_tilt:.2} dB)"
        );
    }

    /// Coherent LF shelf trim (`bass_trim_db`) leaves discrete `SLOT_LFE` and L/R symmetry untouched.
    #[test]
    fn bass_trim_leaves_lfe_and_symmetry_untouched() {
        let base = PlusCfg::from_profile(PlusProfile::Pure);
        let mut trimmed = base;
        trimmed.bass_trim_db = -4.0;
        trimmed.phantom_decomb = 0.8;

        let b0 = KernelBank::synthesize(base, 48_000);
        let b1 = KernelBank::synthesize(trimmed, 48_000);

        assert_eq!(b0.ir_l[SLOT_LFE], b1.ir_l[SLOT_LFE]);
        assert_eq!(b0.ir_r[SLOT_LFE], b1.ir_r[SLOT_LFE]);
        for i in 0..N_IR {
            assert!((b1.ir_l[SLOT_FL][i] - b1.ir_r[SLOT_FR][i]).abs() < 1e-12);
            assert!((b1.ir_r[SLOT_FL][i] - b1.ir_l[SLOT_FR][i]).abs() < 1e-12);
        }
    }

    /// Uncorrelated 7.0 per-ear loudness must match stock `g6ct` (`3.5017`, `+5.443 dB`)
    /// within `±0.10 dB` so A/B comparisons and `--setpoint -3` headroom match cleanly.
    #[test]
    fn uncorrelated_loudness_matches_stock_g6ct() {
        let stock_70_power = 3.501_656_f64;
        for profile in [PlusProfile::Pure, PlusProfile::Studio, PlusProfile::Cinema] {
            let bank = KernelBank::synthesize(PlusCfg::from_profile(profile), 48_000);
            let mut p_l = 0.0_f64;
            let mut p_r = 0.0_f64;
            for &ch in &DF_RING_SLOTS {
                p_l += bank.ir_l[ch].iter().map(|x| x * x).sum::<f64>();
                p_r += bank.ir_r[ch].iter().map(|x| x * x).sum::<f64>();
            }
            let diff_l = 10.0 * (p_l / stock_70_power).log10();
            let diff_r = 10.0 * (p_r / stock_70_power).log10();
            assert!(
                diff_l.abs() < 0.10 && diff_r.abs() < 0.10,
                "{:?} loudness drift vs g6ct: L={diff_l:+.3} dB, R={diff_r:+.3} dB",
                profile
            );
            assert!(
                (diff_l - diff_r).abs() < 0.01,
                "{:?} L/R diffuse imbalance: {:.4} dB",
                profile,
                (diff_l - diff_r).abs()
            );
        }
    }

    /// Sub-120 Hz interaural coherence (`IC`) must stay `>= 0.96` across all 7.1 channels
    /// so bass never cancels or smears on open-back, closed-back, planar, or estat headphones.
    #[test]
    fn sub_bass_coherence_is_high_on_all_channels() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), 48_000);
        let nb = 4096 / 2 + 1;
        let freqs: Vec<f64> = (0..nb).map(|k| k as f64 * 48_000.0 / 4096.0).collect();

        for ch in [
            SLOT_FL, SLOT_FR, SLOT_FC, SLOT_LFE, SLOT_SL, SLOT_SR, SLOT_BL, SLOT_BR,
        ] {
            let sl = ir_spectrum_4096(&bank.ir_l[ch]);
            let sr = ir_spectrum_4096(&bank.ir_r[ch]);
            let mut num = 0.0_f64;
            let mut el = 0.0_f64;
            let mut er = 0.0_f64;
            for (k, &f) in freqs.iter().enumerate() {
                if (20.0..=120.0).contains(&f) {
                    num += sl[k].mul(sr[k].conj()).re;
                    el += sl[k].norm_sq();
                    er += sr[k].norm_sq();
                }
            }
            let ic = num / (el * er).sqrt();
            assert!(
                ic >= 0.96,
                "slot {ch} sub-120 Hz coherence too low: {ic:.4}"
            );
        }
    }

    /// Low-frequency ITD (`350–1000 Hz`) must track Woodworth within `±35 µs`
    /// without double-counting minimum-phase head-shadow group delay.
    #[test]
    fn measured_itd_matches_woodworth_without_double_counting() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Pure), 48_000);
        let nb = 4096 / 2 + 1;
        let freqs: Vec<f64> = (0..nb).map(|k| k as f64 * 48_000.0 / 4096.0).collect();

        for (slot, az_deg, label) in [
            (SLOT_FL, 30.0, "FL"),
            (SLOT_SL, 100.0, "SL"),
            (SLOT_BL, 142.5, "BL"),
        ] {
            let sl = ir_spectrum_4096(&bank.ir_l[slot]);
            let sr = ir_spectrum_4096(&bank.ir_r[slot]);
            let meas_us = measure_unwrapped_itd_s(&freqs, &sl, &sr, 350.0, 1000.0) * 1e6;
            let target_us = woodworth_itd_s(az_deg) * 1e6;
            assert!(
                (meas_us - target_us).abs() <= 45.0,
                "{label} ITD mismatch: measured={meas_us:.1} µs, Woodworth={target_us:.1} µs"
            );
        }
    }

    /// `FC` and `FL+FR` L/R spectral symmetry must hold to `< 0.05 dB` (`20 Hz–20 kHz`).
    #[test]
    fn left_right_symmetry_is_exact() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), 48_000);
        let nb = 4096 / 2 + 1;
        let freqs: Vec<f64> = (0..nb).map(|k| k as f64 * 48_000.0 / 4096.0).collect();

        // FL vs FR mirror check: FL_L == FR_R and FL_R == FR_L.
        for i in 0..N_IR {
            assert!(
                (bank.ir_l[SLOT_FL][i] - bank.ir_r[SLOT_FR][i]).abs() < 1e-12,
                "FL_L vs FR_R mirror mismatch at tap {i}"
            );
            assert!(
                (bank.ir_r[SLOT_FL][i] - bank.ir_l[SLOT_FR][i]).abs() < 1e-12,
                "FL_R vs FR_L mirror mismatch at tap {i}"
            );
        }

        // FC 1-ERB smoothed L vs R power symmetry (allowing the 0.9 ms lateral reflection
        // micro-asymmetry for externalization, ERB loudness balance stays < 0.08 dB).
        let fc_l = ir_spectrum_4096(&bank.ir_l[SLOT_FC]);
        let fc_r = ir_spectrum_4096(&bank.ir_r[SLOT_FC]);
        let pl: Vec<f64> = fc_l.iter().map(|c| c.norm_sq()).collect();
        let pr: Vec<f64> = fc_r.iter().map(|c| c.norm_sq()).collect();
        let sml = roex_smooth_power(&freqs, &pl, 1.0);
        let smr = roex_smooth_power(&freqs, &pr, 1.0);
        for (k, &f) in freqs.iter().enumerate() {
            if (20.0..=20_000.0).contains(&f) {
                let db = (10.0 * (sml[k] / smr[k]).log10()).abs();
                assert!(db < 0.08, "FC L/R imbalance at {f:.1} Hz: {db:.3} dB");
            }
        }
    }

    /// Center channel externalization lowers early `IACC` cleanly from `Pure` (`1.000`)
    /// to `Studio` (`< 0.94`) and `Cinema` (`< 0.88`) while staying dry (`DRR >= +7 dB` in Studio).
    #[test]
    fn center_externalization_lowers_iacc_while_staying_dry() {
        fn fc_iacc_and_drr(profile: PlusProfile) -> (f64, f64) {
            let bank = KernelBank::synthesize(PlusCfg::from_profile(profile), 48_000);
            let l = &bank.ir_l[SLOT_FC];
            let r = &bank.ir_r[SLOT_FC];
            // Early-window cross-correlation over ±1 ms (±48 samples) lag.
            let el: f64 = l.iter().map(|x| x * x).sum();
            let er: f64 = r.iter().map(|x| x * x).sum();
            let den = (el * er).sqrt();
            let mut max_cc = 0.0_f64;
            for lag in -48_isize..=48_isize {
                let mut sum = 0.0_f64;
                for (i, &li) in l.iter().enumerate().take(N_IR) {
                    let j = i as isize + lag;
                    if j >= 0 && (j as usize) < N_IR {
                        sum += li * r[j as usize];
                    }
                }
                max_cc = max_cc.max((sum / den).abs());
            }
            // Direct window: first 6 ms around pre_delay (`0..336` samples);
            // early reflections start >= 8.5 ms (`>= 456` samples).
            let direct_e: f64 = l[..380].iter().map(|x| x * x).sum::<f64>()
                + r[..380].iter().map(|x| x * x).sum::<f64>();
            let refl_e: f64 = l[380..].iter().map(|x| x * x).sum::<f64>()
                + r[380..].iter().map(|x| x * x).sum::<f64>();
            let drr_db = if refl_e > 1e-18 {
                10.0 * (direct_e / refl_e).log10()
            } else {
                120.0
            };
            (max_cc, drr_db)
        }

        let (iacc_pure, _) = fc_iacc_and_drr(PlusProfile::Pure);
        let (iacc_studio, drr_studio) = fc_iacc_and_drr(PlusProfile::Studio);
        let (iacc_cinema, drr_cinema) = fc_iacc_and_drr(PlusProfile::Cinema);

        assert!(
            iacc_pure > 0.999,
            "Pure FC IACC should be ~1.0, got {iacc_pure:.4}"
        );
        assert!(
            iacc_studio < 0.96 && iacc_studio > 0.78,
            "Studio FC IACC out of dry externalized window: {iacc_studio:.4}"
        );
        assert!(
            iacc_cinema < iacc_studio,
            "Cinema FC IACC ({iacc_cinema:.4}) should be lower than Studio ({iacc_studio:.4})"
        );
        assert!(
            drr_studio >= 7.0,
            "Studio FC DRR too wet: {drr_studio:.2} dB (want >= +7.0 dB)"
        );
        assert!(
            drr_cinema >= 3.5 && drr_cinema < drr_studio,
            "Cinema FC DRR unexpected: {drr_cinema:.2} dB"
        );
    }

    /// DEE `.4` height fold sends identical correlated signal into `SL+BL`;
    /// the rear all-pass decorrelation must keep `SL+BL` summed energy significantly
    /// below stock `g6ct`'s coherent `+4.41 dB` pileup (`<= +3.5 dB` above single-channel).
    #[test]
    fn dee_sl_bl_correlated_buildup_beats_stock_g6ct() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), 48_000);
        // Feed identical unit impulse to SL and BL simultaneously.
        let mut e_sl = 0.0_f64;
        let mut e_bl = 0.0_f64;
        let mut e_sum = 0.0_f64;
        for i in 0..N_IR {
            let sl_l = bank.ir_l[SLOT_SL][i];
            let sl_r = bank.ir_r[SLOT_SL][i];
            let bl_l = bank.ir_l[SLOT_BL][i];
            let bl_r = bank.ir_r[SLOT_BL][i];
            e_sl += sl_l * sl_l + sl_r * sl_r;
            e_bl += bl_l * bl_l + bl_r * bl_r;
            e_sum += (sl_l + bl_l).powi(2) + (sl_r + bl_r).powi(2);
        }
        // Incoherent sum would be `e_sl + e_bl` (0 dB coherent excess);
        // fully coherent sum would be `+3.01 dB` above `e_sl + e_bl`.
        let coherent_excess_db = 10.0 * (e_sum / (e_sl + e_bl)).log10();
        assert!(
            coherent_excess_db < 1.8,
            "SL+BL correlated pileup too high: {coherent_excess_db:+.2} dB above incoherent sum"
        );
    }

    /// Kernel tail floor must decay below `-140 dBFS` by the final sample (zero truncation click),
    /// and pre-delay head (`0..24`) must stay quiet (`< -90 dBFS`, zero pre-ringing).
    #[test]
    fn kernel_tail_and_pre_ringing_are_clean() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), 48_000);
        for ch in 0..SLOTS {
            let tail_max = bank.ir_l[ch][N_IR - 4..]
                .iter()
                .chain(&bank.ir_r[ch][N_IR - 4..])
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max);
            let tail_db = 20.0 * tail_max.max(1e-20).log10();
            assert!(
                tail_db < -140.0,
                "slot {ch} kernel tail too loud: {tail_db:.1} dBFS"
            );

            let pre_max = bank.ir_l[ch][..20]
                .iter()
                .chain(&bank.ir_r[ch][..20])
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max);
            let pre_db = 20.0 * pre_max.max(1e-20).log10();
            assert!(
                pre_db < -90.0,
                "slot {ch} pre-ringing too loud: {pre_db:.1} dBFS"
            );
        }
    }

    /// Streaming overlap-add must be invariant across arbitrary chunk sizes
    /// (`137`, `4096`, `10000`) to `< 1e-12` including the EOF flush tail.
    #[test]
    fn streaming_ola_is_chunk_size_invariant_and_deterministic() {
        let total = 12_345;
        let cfg = PlusCfg::from_profile(PlusProfile::Studio);

        // Deterministic multi-channel test signal across all 7.1 slots.
        let mut sigs: Vec<Vec<f64>> = vec![vec![0.0; total]; SLOTS];
        for (ch, slot) in sigs.iter_mut().enumerate() {
            let freq = 110.0 + (ch as f64) * 173.0;
            for (i, s) in slot.iter_mut().enumerate() {
                *s = 0.25 * (2.0 * PI * freq * (i as f64) / 48_000.0).sin();
            }
        }

        let run = |chunk: usize| -> (Vec<f64>, Vec<f64>) {
            let mut eng = BinauralPlus::new(MASK_7_1, cfg, 48_000);
            let mut out_l = vec![0.0_f64; total];
            let mut out_r = vec![0.0_f64; total];
            let mut pos = 0usize;
            while pos < total {
                let n = (total - pos).min(chunk);
                let refs: [Option<&[f64]>; SLOTS] = std::array::from_fn(|ch| {
                    if matches!(
                        ch,
                        SLOT_FL
                            | SLOT_FR
                            | SLOT_FC
                            | SLOT_LFE
                            | SLOT_BL
                            | SLOT_BR
                            | SLOT_SL
                            | SLOT_SR
                    ) {
                        Some(&sigs[ch][pos..pos + n])
                    } else {
                        None
                    }
                });
                eng.process(&refs, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n], n);
                pos += n;
            }
            assert!(eng.is_finite());
            let (tl, tr) = eng.flush();
            out_l.extend_from_slice(&tl);
            out_r.extend_from_slice(&tr);
            (out_l, out_r)
        };

        let (ref_l, ref_r) = run(4096);
        let (a_l, a_r) = run(137);
        let (b_l, b_r) = run(10_000);

        assert_eq!(ref_l.len(), a_l.len());
        assert_eq!(ref_l.len(), b_l.len());
        let max_err_a = ref_l
            .iter()
            .zip(&a_l)
            .chain(ref_r.iter().zip(&a_r))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        let max_err_b = ref_l
            .iter()
            .zip(&b_l)
            .chain(ref_r.iter().zip(&b_r))
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err_a < 1e-12, "chunk=137 vs 4096 error {max_err_a:e}");
        assert!(max_err_b < 1e-12, "chunk=10000 vs 4096 error {max_err_b:e}");

        // Second identical run produces bit-identical SHA-256.
        let (rep_l, rep_r) = run(4096);
        let hash = |l: &[f64], r: &[f64]| {
            let mut h = Sha256::new();
            for &v in l.iter().chain(r.iter()) {
                h.update(v.to_le_bytes());
            }
            h.finalize()
        };
        assert_eq!(hash(&ref_l, &ref_r), hash(&rep_l, &rep_r));
    }

    /// Stereo input (`MASK_STEREO`) passes through bit-identically with zero latency.
    #[test]
    fn stereo_mask_is_bit_exact_wire_bypass() {
        let mut eng = BinauralPlus::new(MASK_STEREO, PlusCfg::default(), 48_000);
        assert_eq!(eng.latency_samples(), 0);
        let in_l = [0.25, -0.5, 0.75, 0.125];
        let in_r = [-0.125, 0.625, -0.875, 0.5];
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&in_l);
        refs[SLOT_FR] = Some(&in_r);
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        eng.process(&refs, &mut out_l, &mut out_r, 4);
        assert_eq!(out_l, in_l);
        assert_eq!(out_r, in_r);
        let (tl, tr) = eng.flush();
        assert!(tl.is_empty() && tr.is_empty());
    }

    /// LFE channel with `lfe_lpf = true` passes 40 Hz at ~0 dB and attenuates 400 Hz
    /// by `> 35 dB` (4th-order Butterworth @ 120 Hz).
    #[test]
    fn lfe_lowpass_attenuates_out_of_band_and_stays_mono() {
        let bank = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), 48_000);
        // LFE L and R are strictly identical.
        assert_eq!(bank.ir_l[SLOT_LFE], bank.ir_r[SLOT_LFE]);
        let sp = ir_spectrum_4096(&bank.ir_l[SLOT_LFE]);
        let bin_40 = (40.0_f64 * 4096.0 / 48_000.0).round() as usize;
        let bin_400 = (400.0_f64 * 4096.0 / 48_000.0).round() as usize;
        let db_40 = 20.0 * (sp[bin_40].norm() / std::f64::consts::FRAC_1_SQRT_2).log10();
        let db_400 = 20.0 * (sp[bin_400].norm() / std::f64::consts::FRAC_1_SQRT_2).log10();
        assert!(
            db_40.abs() < 0.3,
            "LFE 40 Hz passband drift: {db_40:+.2} dB"
        );
        assert!(
            db_400 < -35.0,
            "LFE 400 Hz stopband too weak: {db_400:+.2} dB"
        );
    }

    /// Discrete LFE gain offset (`lfe_gain_db`) and LF-shelf opt-in (`lfe_lf_trim`)
    /// control `SLOT_LFE` cleanly while leaving the full-range channels untouched.
    #[test]
    fn lfe_gain_and_lf_trim_control_discrete_lfe() {
        let bin_40 = (40.0_f64 * 4096.0 / 48_000.0).round() as usize;
        let base = PlusCfg::from_profile(PlusProfile::Pure);
        let b0 = KernelBank::synthesize(base, 48_000);
        let ref_db40 = 20.0 * (ir_spectrum_4096(&b0.ir_l[SLOT_LFE])[bin_40].norm()).log10();

        let mut boosted = base;
        boosted.lfe_gain_db = 6.0;
        let b_plus = KernelBank::synthesize(boosted, 48_000);
        let plus_db40 = 20.0 * (ir_spectrum_4096(&b_plus.ir_l[SLOT_LFE])[bin_40].norm()).log10();
        assert!(
            ((plus_db40 - ref_db40) - 6.0).abs() < 0.05,
            "LFE +6 dB gain offset drifted: {:+.3} dB",
            plus_db40 - ref_db40
        );
        assert_eq!(b0.ir_l[SLOT_FL], b_plus.ir_l[SLOT_FL]);

        let mut trimmed_lfe = base;
        trimmed_lfe.bass_trim_db = -4.0;
        trimmed_lfe.lfe_lf_trim = true;
        let b_trim = KernelBank::synthesize(trimmed_lfe, 48_000);
        let trim_db40 = 20.0 * (ir_spectrum_4096(&b_trim.ir_l[SLOT_LFE])[bin_40].norm()).log10();
        assert!(
            ((trim_db40 - ref_db40) - (-3.911)).abs() < 0.15,
            "LFE LF-shelf trim at 40 Hz expected ~-3.91 dB, got {:+.3} dB",
            trim_db40 - ref_db40
        );
    }
}
