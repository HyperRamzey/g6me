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
//!    diffraction ILD, `a = 8.75 cm`, `c = 343 m/s`, `f0 = c/(pi*a) = 1247.8 Hz`)
//!    converted to strictly causal minimum phase via Hilbert-cepstral folding.
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

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use sha2::Digest;

use crate::atmosphere::AtmosphereCfg;
use crate::bem::{watertight_winding_repair, FmmBemSolver, TriangleMesh};
use crate::blackwell::CudaTensorPrecision;
use crate::dsp::{Dsp, MASK_STEREO, SLOTS};
use crate::fft::{
    half_kaiser_tail_apply, half_slepian_tail_apply, minimum_phase_spectrum, roex_smooth_power,
    Complex, FftPlan,
};
use crate::hoa::{kermit_canfield_warped_allpass, MagLsDecoder};
use crate::ntt::{
    field_to_i128, i64_to_field, mod_add, mod_mul, NttPlan, GOLDILOCKS_P, NTT_KERNEL_SCALE,
    NTT_PCM_SCALE,
};
use crate::room::{
    synthesize_hybrid_room_reflections_with_cuda, FdtdSolver3d, RoomGeometry, RoomPreset,
    RoomQuality,
};
use crate::scattering::{
    ear_canal_compensation_mag, snowman_hrtf, sph_to_cart, SnowmanGeom, DEFAULT_EAR_CANAL_MM,
    DEFAULT_HEAD_RADIUS_M,
};

/// Synthesis FFT size (`341.3 ms` at 48 kHz) — long enough that the causal LR4
/// bass-mono crossover impulse decays `< -140 dB` with zero circular wrap.
pub const N_SYNTH: usize = 16_384;
/// Final windowed anechoic FIR kernel length (`42.67 ms` at 48 kHz).
pub const N_IR: usize = 2_048;
/// Final windowed ControlRoom 3D Hybrid Room BRIR length (`341.33 ms` at 48 kHz = 8 UPOLA partitions).
pub const N_IR_ROOM: usize = 16_384;
/// Final windowed TheaterStage / long-RT60 3D Hybrid Room BRIR length (`682.67 ms` at 48 kHz = 16 UPOLA partitions).
pub const N_IR_THEATER: usize = 32_768;
/// Uniform Partitioned Overlap-Add (UPOLA) partition length at 48 kHz (`42.67 ms`).
pub const UPOLA_PART: usize = 2_048;
/// Streaming overlap-add FFT size for anechoic kernels (`HOP + N_IR - 1 = 8191 < 8192`).
pub const N_FFT: usize = 8_192;
/// Maximum input chunk per single OLA transform (`8192 - 2048 = 6144`).
pub const OLA_HOP: usize = 6_144;
/// Causal direct-path pre-delay in seconds (`1.0 ms` = `48` samples at 48 kHz).
pub const PRE_DELAY_S: f64 = 0.001;
/// Half-Kaiser tail taper length at the end of `N_IR` (48 kHz base).
const KAISER_TAIL: usize = 256;
/// Kaiser beta for tail taper (`> 119 dB` window edge attenuation).
const KAISER_BETA: f64 = 16.0;

/// Power-of-two time-support scale factor relative to 48 kHz (`1` for `<= 48 kHz`,
/// `2` for `88.2/96 kHz`, `4` for `176.4/192 kHz`, `8` for `352.8/384 kHz`).
/// Keeps `N_IR` at `~42.7 ms` (preserving all `>= 8.5..30 ms` early reflections and
/// LR4/LFE tails) and `N_SYNTH` at `~341 ms` across every supported sample rate.
#[inline]
fn sr_scale(sample_rate: u32) -> usize {
    let mult = (sample_rate.max(8_000) as f64 / 48_000.0).ceil() as usize;
    mult.max(1).next_power_of_two().min(8)
}
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

/// Acoustic diffraction physics solver for [`RendererMode::Plus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhysicsEngine {
    /// Fast analytic Brown–Duda (1998) spherical head shadow (default for instant synthesis).
    #[default]
    BrownDuda,
    /// Exact 3D Helmholtz rigid-sphere (`h_m^(2)(ka)`) + coupled Snowman head/torso solver
    /// (Duda & Martens 1998; Algazi et al. 2002) + Order-7 HOA MagLS + Hold SH tapering.
    HankelSnowmanMagLs,
}

impl PhysicsEngine {
    pub const ALL: [Self; 2] = [Self::BrownDuda, Self::HankelSnowmanMagLs];

    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "brown-duda" | "fast" | "analytic" => Ok(Self::BrownDuda),
            "hankel-magls" | "hankel" | "snowman" | "exact" | "magls" => {
                Ok(Self::HankelSnowmanMagLs)
            }
            other => Err(format!(
                "unknown physics engine `{other}` (expected: brown-duda, hankel-magls)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::BrownDuda => "brown-duda",
            Self::HankelSnowmanMagLs => "hankel-magls",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BrownDuda => "Brown-Duda (Fast)",
            Self::HankelSnowmanMagLs => "3D Hankel + Snowman + Order-7 MagLS",
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
    /// matching Reference 0 dB discrete LFE).
    pub lfe_gain_db: f64,
    /// Apply `bass_trim_db` (340 Hz LF balance shelf) to discrete `LFE` as well (`false` keeps
    /// discrete `LFE` at `0 dB` relative to untrimmed Reference; `true` trims `.1` with mains).
    pub lfe_lf_trim: bool,
    /// Apply 120 Hz 4th-order Butterworth low-pass filter to the LFE channel.
    pub lfe_lpf: bool,
    /// Acoustic wave-scattering engine (`BrownDuda` fast vs `HankelSnowmanMagLs` exact 3D).
    pub physics_engine: PhysicsEngine,
    /// Virtual speaker distance in meters (`0.25 ..= 10.0`, default `2.0` m; near-field parallax).
    pub distance_m: f64,
    /// Listener spherical head radius in meters (`0.065 ..= 0.115`, default `0.0875` m).
    pub head_radius_m: f64,
    /// Listener head-to-torso spacing in meters (`0.20 ..= 0.38`, default `0.26` m).
    pub torso_height_m: f64,
    /// Personalization high-frequency pinna shelf tilt in dB (`-4.0 ..= +4.0`, default `0.0`).
    pub pinna_shelf_db: f64,
    /// Individualized ear-canal length in mm for Webster 1D horn impedance (`25.0` = neutral,
    /// `15.0 ..= 38.0` active, `0.0` disables).
    pub ear_canal_mm: f64,
    /// Atmospheric thermodynamic & molecular `O₂`/`N₂` air state (Cramer 1993 / ISO 9613-1).
    pub atmosphere: AtmosphereCfg,
    /// Hybrid 3D Room Acoustics engine preset (`Anechoic`, `ControlRoom`, `TheaterStage`).
    pub room_preset: RoomPreset,
    /// Physical simulation fidelity for the 3D Hybrid Room solver (`Preview` vs `Production`).
    pub room_quality: RoomQuality,
    /// Replace Half-Kaiser tail taper with exact Slepian DPSS eigenvector window (`Slepian 1961`).
    pub use_slepian_tail: bool,
    /// Engage orthogonal-scattering FDN (lossless Schlecht-Habets prototype) + Velvet Noise
    /// late externalization.
    pub use_fdn_velvet: bool,
    /// Engage Kermit-Canfield & Abel (DAFx 2016) Bark-warped all-pass rear decorrelation.
    pub use_warped_decorr: bool,
    /// Execute binaural FIR Overlap-Add convolution strictly inside the 64-bit Goldilocks
    /// prime field (`Z / pZ`, `p = 2^64 - 2^32 + 1`) via Number-Theoretic Transforms (`--exact-ntt`,
    /// 0 ULPs of intermediate floating-point rounding error!).
    pub exact_ntt: bool,
    /// Optional NVIDIA CUDA GPU offload (`--cuda`, requires `--features cuda` & CUDA driver;
    /// falls back cleanly to deterministic CPU `f64` when no GPU is present).
    pub use_cuda: bool,
    /// Tensor Core precision for Ozaki-II CRT FP64 GEMM when CUDA GPU offload is active
    /// (`Auto` selects single-limb INT8 on `sm_61..sm_120a`; explicit NVFP4 is available only
    /// on `sm_120a`, and unsupported manual formats downgrade per `resolve_for_arch`).
    pub cuda_tensor_precision: CudaTensorPrecision,
}

impl Default for PlusCfg {
    fn default() -> Self {
        Self::from_profile(PlusProfile::Studio)
    }
}

impl PlusCfg {
    pub fn from_profile(profile: PlusProfile) -> Self {
        let base_adv = (
            PhysicsEngine::BrownDuda,
            2.0_f64,
            DEFAULT_HEAD_RADIUS_M,
            0.26_f64,
            0.0_f64,
            DEFAULT_EAR_CANAL_MM,
            AtmosphereCfg::default(),
            RoomPreset::Anechoic,
            false,
            false,
            false,
        );
        let (
            physics_engine,
            distance_m,
            head_radius_m,
            torso_height_m,
            pinna_shelf_db,
            ear_canal_mm,
            atmosphere,
            room_preset,
            use_slepian_tail,
            use_fdn_velvet,
            use_warped_decorr,
        ) = base_adv;
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
                physics_engine,
                distance_m,
                head_radius_m,
                torso_height_m,
                pinna_shelf_db,
                ear_canal_mm,
                atmosphere,
                room_preset,
                room_quality: RoomQuality::Preview,
                use_slepian_tail,
                use_fdn_velvet,
                use_warped_decorr,
                exact_ntt: false,
                use_cuda: false,
                cuda_tensor_precision: CudaTensorPrecision::Auto,
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
                physics_engine,
                distance_m,
                head_radius_m,
                torso_height_m,
                pinna_shelf_db,
                ear_canal_mm,
                atmosphere,
                room_preset,
                room_quality: RoomQuality::Preview,
                use_slepian_tail,
                use_fdn_velvet,
                use_warped_decorr,
                exact_ntt: false,
                use_cuda: false,
                cuda_tensor_precision: CudaTensorPrecision::Auto,
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
                physics_engine,
                distance_m,
                head_radius_m,
                torso_height_m,
                pinna_shelf_db,
                ear_canal_mm,
                atmosphere,
                room_preset,
                room_quality: RoomQuality::Preview,
                use_slepian_tail,
                use_fdn_velvet,
                use_warped_decorr,
                exact_ntt: false,
                use_cuda: false,
                cuda_tensor_precision: CudaTensorPrecision::Auto,
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
                && self.physics_engine == r.physics_engine
                && (self.distance_m - r.distance_m).abs() < 1e-3
                && (self.head_radius_m - r.head_radius_m).abs() < 1e-4
                && (self.torso_height_m - r.torso_height_m).abs() < 1e-3
                && (self.pinna_shelf_db - r.pinna_shelf_db).abs() < 1e-3
                && (self.ear_canal_mm - r.ear_canal_mm).abs() < 1e-2
                && self.room_preset == r.room_preset
                && self.room_quality == r.room_quality
                && self.use_slepian_tail == r.use_slepian_tail
                && self.use_fdn_velvet == r.use_fdn_velvet
                && self.use_warped_decorr == r.use_warped_decorr
                && self.exact_ntt == r.exact_ntt
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
        if !self.distance_m.is_finite() || !(0.25..=10.0).contains(&self.distance_m) {
            return Err(format!(
                "plus.distance_m must be in 0.25..=10.0 m, got {}",
                self.distance_m
            ));
        }
        if !self.head_radius_m.is_finite() || !(0.065..=0.115).contains(&self.head_radius_m) {
            return Err(format!(
                "plus.head_radius_m must be in 0.065..=0.115 m, got {}",
                self.head_radius_m
            ));
        }
        if !self.torso_height_m.is_finite() || !(0.20..=0.38).contains(&self.torso_height_m) {
            return Err(format!(
                "plus.torso_height_m must be in 0.20..=0.38 m, got {}",
                self.torso_height_m
            ));
        }
        if !self.pinna_shelf_db.is_finite() || !(-4.0..=4.0).contains(&self.pinna_shelf_db) {
            return Err(format!(
                "plus.pinna_shelf_db must be in -4.0..=+4.0 dB, got {}",
                self.pinna_shelf_db
            ));
        }
        if !self.ear_canal_mm.is_finite()
            || (self.ear_canal_mm != 0.0 && !(15.0..=38.0).contains(&self.ear_canal_mm))
        {
            return Err(format!(
                "plus.ear_canal_mm must be 0 or in 15.0..=38.0 mm, got {}",
                self.ear_canal_mm
            ));
        }
        self.atmosphere.validate()?;
        Ok(())
    }

    pub fn summary(&self) -> String {
        let mut s = format!(
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
        );
        if self.physics_engine == PhysicsEngine::HankelSnowmanMagLs
            || self.room_preset != RoomPreset::Anechoic
            || self.use_slepian_tail
            || self.use_fdn_velvet
            || self.exact_ntt
            || self.use_cuda
            || (self.distance_m - 2.0).abs() > 1e-3
        {
            let cuda_tag = if self.use_cuda {
                format!(", cuda:{}", self.cuda_tensor_precision.as_str())
            } else {
                String::new()
            };
            s.push_str(&format!(
                " [{}, {:.2}m, room:{}{}{}{}{}]",
                self.physics_engine.as_str(),
                self.distance_m,
                self.room_preset.as_str(),
                if self.use_slepian_tail {
                    ", slepian"
                } else {
                    ""
                },
                if self.use_fdn_velvet {
                    ", fdn+velvet"
                } else {
                    ""
                },
                if self.exact_ntt { ", exact-ntt" } else { "" },
                cuda_tag,
            ));
        }
        s
    }

    /// Estimate direct-to-reverberant ratio (dB) for center dialogue at current settings
    /// (for GUI acoustic readout).
    pub fn estimated_center_drr_db(&self) -> Option<f64> {
        if self.externalize <= 1e-4 && self.room_preset == RoomPreset::Anechoic {
            return None;
        }
        let ext = if self.room_preset != RoomPreset::Anechoic {
            self.externalize.max(0.45)
        } else {
            self.externalize
        };
        let refl_scale = ext * (1.0 - 0.50 * self.center_focus);
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

static KERNEL_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Arc<KernelBank>>>> = OnceLock::new();

fn kernel_cache_key(
    cfg: &PlusCfg,
    sample_rate: u32,
    custom_room: Option<&RoomGeometry>,
    custom_head_mesh: Option<&TriangleMesh>,
) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(sample_rate.to_le_bytes());
    if let Ok(cfg_json) = serde_json::to_vec(cfg) {
        h.update((cfg_json.len() as u64).to_le_bytes());
        h.update(&cfg_json);
    }
    if let Some(room) = custom_room {
        h.update([1u8]);
        if let Ok(rj) = serde_json::to_vec(room) {
            h.update((rj.len() as u64).to_le_bytes());
            h.update(&rj);
        }
    } else {
        h.update([0u8]);
    }
    if let Some(mesh) = custom_head_mesh {
        h.update([1u8]);
        h.update((mesh.vertices.len() as u64).to_le_bytes());
        for v in &mesh.vertices {
            for &c in v {
                h.update(c.to_bits().to_le_bytes());
            }
        }
        h.update((mesh.faces.len() as u64).to_le_bytes());
        for f in &mesh.faces {
            for &idx in f {
                h.update((idx as u64).to_le_bytes());
            }
        }
    } else {
        h.update([0u8]);
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

impl KernelBank {
    /// Return a cached `Arc<KernelBank>` keyed by the exact SHA-256 of
    /// `(PlusCfg, sample_rate, custom_room, custom_head_mesh)` so batch album renders
    /// compute FDTD/Ray-Tracing/FMM-BEM only once per unique acoustic configuration.
    pub fn synthesize_cached(
        cfg: PlusCfg,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
        custom_head_mesh: Option<&TriangleMesh>,
    ) -> Arc<Self> {
        let key = kernel_cache_key(&cfg, sample_rate, custom_room, custom_head_mesh);
        let map_mutex = KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Ok(guard) = map_mutex.lock() {
            if let Some(existing) = guard.get(&key) {
                return Arc::clone(existing);
            }
        }
        let bank = Arc::new(Self::synthesize_with_assets(
            cfg,
            sample_rate,
            custom_room,
            custom_head_mesh,
        ));
        if let Ok(mut guard) = map_mutex.lock() {
            // Cap cache to 16 entries to bound memory across interactive GUI sweeps.
            if guard.len() >= 16 {
                guard.clear();
            }
            guard.insert(key, Arc::clone(&bank));
        }
        bank
    }

    /// Synthesize all 12×2 binaural FIR kernels deterministically in `f64`.
    pub fn synthesize(cfg: PlusCfg, sample_rate: u32) -> Self {
        Self::synthesize_with_room(cfg, sample_rate, None)
    }

    /// Synthesize all 12×2 binaural FIR kernels with an optional custom 3D room JSON geometry.
    pub fn synthesize_with_room(
        cfg: PlusCfg,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
    ) -> Self {
        Self::synthesize_with_assets(cfg, sample_rate, custom_room, None)
    }

    /// Synthesize all 12×2 binaural FIR kernels with optional custom 3D room JSON geometry
    /// and/or a user-scanned 3D head/pinna `.obj` mesh (`--head-mesh`) solved via Gumerov
    /// FMM-BEM Helmholtz acoustic reciprocity!
    pub fn synthesize_with_assets(
        cfg: PlusCfg,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
        custom_head_mesh: Option<&TriangleMesh>,
    ) -> Self {
        let fs = (sample_rate.max(8_000)) as f64;
        let scale = sr_scale(sample_rate);
        let active_room_preview: Option<RoomGeometry> = if let Some(r) = custom_room {
            Some(r.clone())
        } else if cfg.room_preset != RoomPreset::Anechoic {
            Some(RoomGeometry::from_preset(cfg.room_preset))
        } else {
            None
        };
        let has_room = active_room_preview.is_some();
        let is_long_room = active_room_preview
            .as_ref()
            .map(|r| cfg.room_preset == RoomPreset::TheaterStage || r.target_rt60_s > 0.35)
            .unwrap_or(false);
        let n_ir = if is_long_room {
            N_IR_THEATER * scale
        } else if has_room {
            N_IR_ROOM * scale
        } else {
            N_IR * scale
        };
        let n_synth = if has_room {
            (n_ir * 2).max(N_SYNTH * scale)
        } else {
            N_SYNTH * scale
        };
        let kaiser_tail = if has_room {
            (KAISER_TAIL * 4) * scale
        } else {
            KAISER_TAIL * scale
        };
        let nb = n_synth / 2 + 1;
        let plan = FftPlan::new(n_synth);
        let freqs: Vec<f64> = (0..nb)
            .map(|k| (k as f64) * fs / (n_synth as f64))
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

        // Precompute rear all-pass decorrelator: either Kermit-Canfield & Abel (DAFx 2016)
        // Bark-warped 4-stage cascade or the classic 1st-order 650 Hz all-pass.
        let (ap_rear_l, ap_rear_r) =
            if cfg.use_warped_decorr || cfg.physics_engine == PhysicsEngine::HankelSnowmanMagLs {
                (
                    kermit_canfield_warped_allpass(&freqs, fs, 1.0),
                    kermit_canfield_warped_allpass(&freqs, fs, -1.0),
                )
            } else {
                let mut ap = vec![Complex::ONE; nb];
                let f_ap = 650.0_f64.min(0.4 * fs);
                let tan_ap = (PI * f_ap / fs).tan();
                let ap_coeff = (tan_ap - 1.0) / (tan_ap + 1.0);
                for (k, &f) in freqs.iter().enumerate() {
                    let z1 = Complex::from_polar(1.0, -2.0 * PI * f / fs);
                    let num = Complex::new(ap_coeff, 0.0).add(z1);
                    let den = Complex::ONE.add(z1.scale(ap_coeff));
                    ap[k] = num.div(den);
                }
                (ap.clone(), ap)
            };

        let active_room = active_room_preview;

        // Build Order-7 HOA MagLS decoder once when exact wave-scattering or hybrid room is active.
        let snowman_geom = SnowmanGeom::from_head_and_torso(cfg.head_radius_m, cfg.torso_height_m);
        let c_air = cfg.atmosphere.speed_of_sound_m_s();
        let magls: Option<MagLsDecoder> =
            if cfg.physics_engine == PhysicsEngine::HankelSnowmanMagLs || active_room.is_some() {
                Some(MagLsDecoder::build(
                    &freqs,
                    snowman_geom,
                    cfg.distance_m,
                    c_air,
                    true,
                    0.35,
                ))
            } else {
                None
            };

        let mut dir_l: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; nb]; SLOTS];
        let mut dir_r: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; nb]; SLOTS];
        let mut chan_weights = [1.0_f64; SLOTS];

        // If a real 3D head/pinna mesh (`--head-mesh`) was supplied, build the FMM-BEM Helmholtz
        // solver (auto-repairing open holes with Generalized Winding Numbers) and solve exact
        // personalized binaural spectra for all 11 full-range virtual speakers via reciprocity!
        let bem_direct: Option<Vec<(Vec<f64>, Vec<f64>)>> = custom_head_mesh.and_then(|raw_mesh| {
            let mesh = if raw_mesh.stats(c_air).is_watertight {
                raw_mesh.clone()
            } else {
                watertight_winding_repair(raw_mesh, 16)
            };
            let solver = match FmmBemSolver::new_with_cuda(&mesh, cfg.use_cuda) {
                Ok(solver) => solver,
                Err(gpu_err) if cfg.use_cuda => match FmmBemSolver::new_with_cuda(&mesh, false) {
                    Ok(solver) => {
                        eprintln!(
                            "warning: head-mesh BEM GPU construction failed ({gpu_err}); \
                             using the CPU BEM solver instead"
                        );
                        crate::cuda_report::record_cpu_fallback(
                            "bem-head-mesh",
                            gpu_err.to_string(),
                        );
                        solver
                    }
                    Err(cpu_err) => {
                        eprintln!(
                            "warning: head-mesh personalization disabled: BEM mesh rejected \
                             ({cpu_err})"
                        );
                        crate::cuda_report::record_cpu_fallback(
                            "bem-head-mesh",
                            cpu_err.to_string(),
                        );
                        return None;
                    }
                },
                Err(cpu_err) => {
                    eprintln!(
                        "warning: head-mesh personalization disabled: BEM mesh rejected \
                         ({cpu_err})"
                    );
                    crate::cuda_report::record_cpu_fallback("bem-head-mesh", cpu_err.to_string());
                    return None;
                }
            };
            let mut srcs = Vec::with_capacity(SLOTS);
            for (ch, &base_az) in SLOT_AZ_DEG.iter().enumerate() {
                let eff_az = if matches!(ch, SLOT_FL | SLOT_FR | SLOT_FLC | SLOT_FRC) {
                    (base_az * cfg.width).clamp(-55.0, 55.0)
                } else {
                    base_az
                };
                let el = if ch == SLOT_TC { 60.0 } else { 0.0 };
                srcs.push(sph_to_cart(eff_az, el, cfg.distance_m));
            }
            Some(solver.synthesize_binaural_irs_with_cuda(
                &srcs,
                sample_rate,
                n_ir,
                &cfg.atmosphere,
                cfg.use_cuda,
            ))
        });

        // 1. Synthesize each full-range slot's anechoic direct path first.
        for ch in 0..SLOTS {
            if ch == SLOT_LFE {
                continue;
            }
            if let Some(ref bem_irs) = bem_direct {
                let (ref bil, ref bir) = bem_irs[ch];
                let mut pad_l = vec![0.0_f64; n_synth];
                let mut pad_r = vec![0.0_f64; n_synth];
                let copy_len = bil.len().min(n_synth);
                pad_l[..copy_len].copy_from_slice(&bil[..copy_len]);
                pad_r[..copy_len].copy_from_slice(&bir[..copy_len]);
                let mut sc = vec![Complex::ZERO; n_synth];
                let mut sl = vec![Complex::ZERO; nb];
                let mut sr = vec![Complex::ZERO; nb];
                plan.rfft(&pad_l, &mut sc, &mut sl);
                plan.rfft(&pad_r, &mut sc, &mut sr);
                let cw = if ch == SLOT_FC {
                    0.94 + 0.12 * cfg.center_focus
                } else if matches!(ch, SLOT_SL | SLOT_SR | SLOT_BL | SLOT_BR) {
                    0.60 + 0.40 * cfg.surround_depth
                } else {
                    1.0
                };
                let g = std::f64::consts::FRAC_1_SQRT_2 * cw;
                for k in 0..nb {
                    sl[k] = sl[k].scale(g);
                    sr[k] = sr[k].scale(g);
                }
                dir_l[ch] = sl;
                dir_r[ch] = sr;
                chan_weights[ch] = cw;
                continue;
            }
            let ap_rear = if SLOT_AZ_DEG[ch] >= 0.0 {
                &ap_rear_l
            } else {
                &ap_rear_r
            };
            let (sl, sr, cw) = synthesize_direct_spectrum(
                ch,
                &cfg,
                fs,
                pre_delay_s,
                &freqs,
                &lr4_lo,
                &lr4_hi,
                ap_rear,
                snowman_geom,
                c_air,
                &plan,
            );
            dir_l[ch] = sl;
            dir_r[ch] = sr;
            chan_weights[ch] = cw;
        }

        // 2. Direct-path 1-ERB common Diffuse-Field Equalization (Zaunschirm,
        //    Schörkhuber & Höldrich, JASA 2018).
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

        // 3. Room / externalization reflections:
        //    - If a 3D Hybrid Room (`ControlRoom`, `TheaterStage`, or `--room-json`) is active:
        //      run 2 reciprocal 3D FDTD PDE simulations (Left ear & Right ear -> all 12 speakers!)
        //      + 3D ISM + BTM desk edge diffraction + PCG32 Monte Carlo Ray Tracing -> Order-7 MagLS
        //      (+ Paraunitary Scattering FDN & Velvet Noise).
        //    - Otherwise, when `externalize > 0`, add dry ITU-R BS.1116 lateral early reflections.
        if let (Some(room_geom), Some(decoder)) = (&active_room, &magls) {
            // Compute virtual speaker Cartesian coordinates inside the room.
            let mut spk_coords = [[0.0_f64; 3]; SLOTS];
            for (ch, coord) in spk_coords.iter_mut().enumerate() {
                let base_az = SLOT_AZ_DEG[ch];
                let eff_az = if matches!(ch, SLOT_FL | SLOT_FR | SLOT_FLC | SLOT_FRC) {
                    (base_az * cfg.width).clamp(-65.0, 65.0)
                } else {
                    base_az
                };
                let el = if ch == SLOT_TC { 60.0 } else { 0.0 };
                let rel = sph_to_cart(eff_az, el, cfg.distance_m);
                *coord = [
                    (room_geom.listener_m[0] + rel[0]).clamp(0.25, room_geom.dims_m[0] - 0.25),
                    (room_geom.listener_m[1] + rel[1]).clamp(0.25, room_geom.dims_m[1] - 0.25),
                    (room_geom.listener_m[2] + rel[2]).clamp(0.25, room_geom.dims_m[2] - 0.25),
                ];
            }

            // Acoustic Reciprocity shortcut: only 2 3D FDTD runs (L ear + R ear) record all 12 speakers!
            // Preview: 500 Hz @ 6 PPW (65k voxel budget, 0.165 s).
            // Production: 670 Hz @ 6 PPW (`dx <= 0.085 m`, 260k voxel budget, 0.35 s decay).
            let (fdtd_target_hz, fdtd_voxel_cap, fdtd_dur_s, fdtd_max_steps) =
                match cfg.room_quality {
                    RoomQuality::Preview => (500.0, 65_000usize, 0.165_f64, 1_100usize),
                    RoomQuality::Production => (670.0, 260_000usize, 0.35_f64, 3_200usize),
                };
            let fdtd =
                FdtdSolver3d::for_room_budget(room_geom, c_air, fdtd_target_hz, fdtd_voxel_cap);
            let ear_l_v = fdtd.coord_to_voxel([
                room_geom.listener_m[0],
                room_geom.listener_m[1] + cfg.head_radius_m,
                room_geom.listener_m[2],
            ]);
            let ear_r_v = fdtd.coord_to_voxel([
                room_geom.listener_m[0],
                room_geom.listener_m[1] - cfg.head_radius_m,
                room_geom.listener_m[2],
            ]);
            let spk_voxels: Vec<(usize, usize, usize)> =
                spk_coords.iter().map(|&p| fdtd.coord_to_voxel(p)).collect();
            let fdtd_steps = ((fdtd_dur_s / fdtd.dt_s).ceil() as usize).clamp(120, fdtd_max_steps);
            let fdtd_l_all =
                fdtd.simulate_with_cuda(ear_l_v, &spk_voxels, fdtd_steps, cfg.use_cuda);
            let fdtd_r_all =
                fdtd.simulate_with_cuda(ear_r_v, &spk_voxels, fdtd_steps, cfg.use_cuda);
            let tail_duration_s = ((n_ir as f64) / fs - pre_delay_s - 0.015).max(0.144);

            let ext_mix = cfg.externalize.max(0.55);
            for ch in 0..SLOTS {
                if ch == SLOT_LFE {
                    continue;
                }
                let refl_scale = if ch == SLOT_FC {
                    ext_mix * (1.0 - 0.45 * cfg.center_focus)
                } else {
                    ext_mix
                };
                let (rl, rr) = synthesize_hybrid_room_reflections_with_cuda(
                    room_geom,
                    &cfg.atmosphere,
                    decoder,
                    &freqs,
                    fs,
                    pre_delay_s,
                    spk_coords[ch],
                    Some(&fdtd_l_all[ch]),
                    Some(&fdtd_r_all[ch]),
                    fdtd.dt_s,
                    cfg.use_fdn_velvet,
                    cfg.room_quality,
                    tail_duration_s,
                    0x6736_4D45_0000_0001u64.wrapping_add(ch as u64 * 0x9E37_79B9),
                    &plan,
                    cfg.use_cuda,
                );
                let g = std::f64::consts::FRAC_1_SQRT_2 * chan_weights[ch] * refl_scale * 0.45;
                for k in 0..nb {
                    spec_l[ch][k] = spec_l[ch][k].add(rl[k].mul(dfeq_min[k]).scale(g));
                    spec_r[ch][k] = spec_r[ch][k].add(rr[k].mul(dfeq_min[k]).scale(g));
                }
            }
        } else if cfg.externalize > 1e-6 {
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
            // Optional Paraunitary Scattering FDN + Velvet Noise late shimmer in dry Studio/Cinema.
            if cfg.use_fdn_velvet {
                let tail_len = ((0.06 * fs).round() as usize).min(n_synth / 2);
                let mut fdn = crate::room::ScatteringFdn::new(sample_rate, 0.22, 0.12);
                let (fl, fr) = fdn.synthesize_stereo_tail(tail_len);
                let vl = crate::room::generate_velvet_noise(tail_len, sample_rate, 2_200.0, 0xF00D);
                let vr = crate::room::generate_velvet_noise(tail_len, sample_rate, 2_200.0, 0xCAFE);
                let mut tl = vec![0.0_f64; n_synth];
                let mut tr = vec![0.0_f64; n_synth];
                let start = ((pre_delay_s + 0.014) * fs).round() as usize;
                for i in 0..tail_len {
                    if start + i >= n_synth {
                        break;
                    }
                    let env = (-6.907_755 * (i as f64) / (0.20 * fs)).exp();
                    tl[start + i] = 0.7 * fl[i] + 0.3 * vl[i] * env * 0.035;
                    tr[start + i] = 0.7 * fr[i] + 0.3 * vr[i] * env * 0.035;
                }
                let mut scr = vec![Complex::ZERO; n_synth];
                let mut fdn_l = vec![Complex::ZERO; nb];
                let mut fdn_r = vec![Complex::ZERO; nb];
                plan.rfft(&tl, &mut scr, &mut fdn_l);
                plan.rfft(&tr, &mut scr, &mut fdn_r);
                for ch in 0..SLOTS {
                    if ch == SLOT_LFE {
                        continue;
                    }
                    let g = 0.10
                        * cfg.externalize
                        * chan_weights[ch]
                        * if ch == SLOT_FC {
                            1.0 - 0.5 * cfg.center_focus
                        } else {
                            1.0
                        };
                    for (k, &f) in freqs.iter().enumerate() {
                        let x_hp = f / 450.0;
                        let hp = (x_hp * x_hp) / (1.0 + x_hp * x_hp);
                        spec_l[ch][k] = spec_l[ch][k].add(fdn_l[k].mul(dfeq_min[k]).scale(g * hp));
                        spec_r[ch][k] = spec_r[ch][k].add(fdn_r[k].mul(dfeq_min[k]).scale(g * hp));
                    }
                }
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

        // 5. Audible-band Parseval 7.0 diffuse power preservation (`TARGET_DF_POWER = 3.5`).
        let f_ref = 24_000.0_f64.min(0.5 * fs);
        let k_end = ((f_ref * (n_synth as f64) / fs).round() as usize).clamp(1, nb - 1);
        let inv_band = 1.0 / ((2 * k_end) as f64);
        let mut tot_p = 0.0_f64;
        for &ch in &DF_RING_SLOTS {
            let pl = (spec_l[ch][0].norm_sq()
                + spec_l[ch][k_end].norm_sq()
                + 2.0
                    * spec_l[ch][1..k_end]
                        .iter()
                        .map(|c| c.norm_sq())
                        .sum::<f64>())
                * inv_band;
            let pr = (spec_r[ch][0].norm_sq()
                + spec_r[ch][k_end].norm_sq()
                + 2.0
                    * spec_r[ch][1..k_end]
                        .iter()
                        .map(|c| c.norm_sq())
                        .sum::<f64>())
                * inv_band;
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

        // 6. Synthesize LFE: mono `1/sqrt(2)` per ear, optional 120 Hz Butterworth LPF.
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

        // 7. Bauer / Linkwitz / Völk–Fastl (AES 2011) causal minimum-phase coherent LF trim shelf.
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

        // 7b. Optional 1D Webster Horn Equation Ear-Canal Impedance personalization (`ear_canal_mm != 25.0`).
        if cfg.ear_canal_mm > 0.0 && (cfg.ear_canal_mm - DEFAULT_EAR_CANAL_MM).abs() > 1e-3 {
            let canal_mag: Vec<f64> = freqs
                .iter()
                .map(|&f| ear_canal_compensation_mag(f, cfg.ear_canal_mm, c_air))
                .collect();
            let canal_min = minimum_phase_spectrum(&canal_mag, &plan);
            for ch in 0..SLOTS {
                if ch == SLOT_LFE {
                    continue;
                }
                for k in 0..nb {
                    spec_l[ch][k] = spec_l[ch][k].mul(canal_min[k]);
                    spec_r[ch][k] = spec_r[ch][k].mul(canal_min[k]);
                }
            }
        }

        // 8. IFFT to time domain, truncate to `n_ir`, and apply Half-Slepian DPSS or Half-Kaiser tail taper.
        let mut scratch = vec![Complex::ZERO; n_synth];
        let mut time_buf = vec![0.0_f64; n_synth];
        let mut ir_l = vec![vec![0.0_f64; n_ir]; SLOTS];
        let mut ir_r = vec![vec![0.0_f64; n_ir]; SLOTS];

        for ch in 0..SLOTS {
            plan.irfft(&spec_l[ch], &mut scratch, &mut time_buf);
            ir_l[ch].copy_from_slice(&time_buf[..n_ir]);
            if cfg.use_slepian_tail {
                half_slepian_tail_apply(&mut ir_l[ch], kaiser_tail, 5.0);
            } else {
                half_kaiser_tail_apply(&mut ir_l[ch], kaiser_tail, KAISER_BETA);
            }

            plan.irfft(&spec_r[ch], &mut scratch, &mut time_buf);
            ir_r[ch].copy_from_slice(&time_buf[..n_ir]);
            if cfg.use_slepian_tail {
                half_slepian_tail_apply(&mut ir_r[ch], kaiser_tail, 5.0);
            } else {
                half_kaiser_tail_apply(&mut ir_r[ch], kaiser_tail, KAISER_BETA);
            }
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
    snowman_geom: SnowmanGeom,
    c_air: f64,
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

    // 1. Direct-path magnitude responses: either exact 3D Spherical-Hankel + Coupled Snowman
    //    head/torso scattering + ISO 9613-1 O2/N2 air absorption, or classic Brown–Duda.
    let mut mag_l = Vec::with_capacity(nb);
    let mut mag_r = Vec::with_capacity(nb);
    let pinna_scale = if is_side || is_back {
        cfg.surround_depth.clamp(0.5, 1.3)
    } else {
        1.0
    };
    let el_deg = if ch == SLOT_TC { 60.0 } else { 0.0 };
    let user_pinna_g = 10.0_f64.powf(cfg.pinna_shelf_db / 20.0);

    if cfg.physics_engine == PhysicsEngine::HankelSnowmanMagLs {
        // Evaluate exact 3D Helmholtz rigid-sphere + Snowman torso on a dense ~180-anchor
        // frequency grid and log/linearly interpolate magnitudes across FFT bins for speed.
        let mut anchors: Vec<usize> = Vec::with_capacity(200);
        anchors.push(0);
        let mut cur = 0usize;
        let df = if nb > 1 { freqs[1] - freqs[0] } else { 10.0 };
        while cur + 1 < nb {
            let f = freqs[cur];
            let step_hz = if f < 1_000.0 {
                25.0
            } else if f < 4_000.0 {
                60.0
            } else {
                0.03 * f
            };
            let step = ((step_hz / df.max(1e-6)).round() as usize).max(1);
            cur = (cur + step).min(nb - 1);
            anchors.push(cur);
        }
        let mut ml_a = Vec::with_capacity(anchors.len());
        let mut mr_a = Vec::with_capacity(anchors.len());
        for &bin in &anchors {
            let f = freqs[bin];
            let hl = snowman_hrtf(
                f,
                snowman_geom,
                eff_az,
                el_deg,
                cfg.distance_m,
                1.0,
                c_air,
                true,
            )
            .norm();
            let hr = snowman_hrtf(
                f,
                snowman_geom,
                eff_az,
                el_deg,
                cfg.distance_m,
                -1.0,
                c_air,
                true,
            )
            .norm();
            let air = cfg.atmosphere.air_gain_linear(f, cfg.distance_m);
            let pinna = broad_pinna_mag(f, eff_az, ch == SLOT_TC, pinna_scale);
            let x_up = f / 4_000.0;
            let user_shelf = if cfg.pinna_shelf_db.abs() > 1e-6 {
                ((user_pinna_g * user_pinna_g * x_up * x_up + 1.0) / (x_up * x_up + 1.0)).sqrt()
            } else {
                1.0
            };
            ml_a.push((hl * air * pinna * user_shelf).max(1e-6));
            mr_a.push((hr * air * pinna * user_shelf).max(1e-6));
        }
        mag_l.resize(nb, 1.0);
        mag_r.resize(nb, 1.0);
        for seg in 0..(anchors.len() - 1) {
            let b0 = anchors[seg];
            let b1 = anchors[seg + 1];
            mag_l[b0] = ml_a[seg];
            mag_r[b0] = mr_a[seg];
            let span = b1 - b0;
            if span > 1 {
                let inv = 1.0 / (span as f64);
                for off in 1..span {
                    let t = (off as f64) * inv;
                    mag_l[b0 + off] = ml_a[seg] + t * (ml_a[seg + 1] - ml_a[seg]);
                    mag_r[b0 + off] = mr_a[seg] + t * (mr_a[seg + 1] - mr_a[seg]);
                }
            }
        }
        mag_l[nb - 1] = *ml_a.last().unwrap_or(&1.0);
        mag_r[nb - 1] = *mr_a.last().unwrap_or(&1.0);
    } else {
        for &f in freqs {
            let bdl = brown_duda_mag(f, eff_az, 1.0, shadow_depth);
            let bdr = brown_duda_mag(f, eff_az, -1.0, shadow_depth);
            let pinna = broad_pinna_mag(f, eff_az, ch == SLOT_TC, pinna_scale);
            let user_shelf = if cfg.pinna_shelf_db.abs() > 1e-6 {
                let x_up = f / 4_000.0;
                ((user_pinna_g * user_pinna_g * x_up * x_up + 1.0) / (x_up * x_up + 1.0)).sqrt()
            } else {
                1.0
            };
            mag_l.push(bdl * pinna * user_shelf);
            mag_r.push(bdr * pinna * user_shelf);
        }
    }

    // 2. Hilbert-cepstral minimum-phase conversion.
    let hl_min = minimum_phase_spectrum(&mag_l, plan);
    let hr_min = minimum_phase_spectrum(&mag_r, plan);

    // 3. Kulkarni–Isabelle–Colburn (1999) excess Woodworth ITD using exact speed of sound & head radius
    //    when HankelSnowmanMagLs is active (or stock constants in fast BrownDuda mode).
    let ww_base = if cfg.physics_engine == PhysicsEngine::HankelSnowmanMagLs {
        let r = eff_az.abs().to_radians().min(PI);
        let path = if r <= 0.5 * PI {
            r + r.sin()
        } else {
            PI - r + r.sin()
        };
        (cfg.head_radius_m / c_air) * path
    } else {
        woodworth_itd_s(eff_az)
    };
    let target_itd = if is_side || is_back {
        ww_base * (0.85 + 0.15 * cfg.surround_depth).clamp(0.70, 1.15)
    } else {
        ww_base
    };
    let nat_itd = measure_unwrapped_itd_s(freqs, &hl_min, &hr_min, 350.0, 900.0);
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

    // Precompute the unit-circle z^-1 phasors and the shared 4th-order reflection
    // high-pass + direct-path DFEQ envelope once per channel rather than per tap/bin.
    let z1s: Vec<Complex> = freqs
        .iter()
        .map(|&f| Complex::from_polar(1.0, -2.0 * PI * f / fs))
        .collect();
    let kh = (PI * hpf_fc / fs).tan();
    let kh2 = kh * kh;
    let sq2 = std::f64::consts::SQRT_2;
    let h_norm = 1.0 / (1.0 + sq2 * kh + kh2);
    let hb0 = h_norm;
    let hb1 = -2.0 * h_norm;
    let hb2 = h_norm;
    let ha1 = 2.0 * (kh2 - 1.0) * h_norm;
    let ha2 = (1.0 - sq2 * kh + kh2) * h_norm;
    let hpf4_dfeq: Vec<Complex> = (0..nb)
        .map(|k| {
            let z1 = z1s[k];
            let z2 = z1.mul(z1);
            let num = Complex::new(hb0, 0.0).add(z1.scale(hb1)).add(z2.scale(hb2));
            let den = Complex::ONE.add(z1.scale(ha1)).add(z2.scale(ha2));
            let hpf2 = num.div(den);
            hpf2.mul(hpf2).mul(dfeq_min[k])
        })
        .collect();

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
        let kl = (PI * lpf_fc / fs).tan();
        let l_norm = 1.0 / (1.0 + kl);
        let lb0 = kl * l_norm;
        let la1 = (kl - 1.0) * l_norm;
        for (k, &f) in freqs.iter().enumerate() {
            let z1 = z1s[k];
            let lpf = Complex::new(lb0, 0.0)
                .add(z1.scale(lb0))
                .div(Complex::ONE.add(z1.scale(la1)));
            let wall = hpf4_dfeq[k].mul(lpf);
            let bdl = brown_duda_mag(f, tap.az_deg, 1.0, 0.75);
            let bdr = brown_duda_mag(f, tap.az_deg, -1.0, 0.75);
            let dl = causal_frac_delay_z(f, del_l_smp, fs, z1).scale(g * bdl);
            let dr = causal_frac_delay_z(f, del_r_smp, fs, z1).scale(g * bdr);
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
    let w0 = 2.0 * SPEED_OF_SOUND / HEAD_RADIUS_M;
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

/// Strictly causal fractional sample delay (`total_samples >= 2.0`) combining an
/// integer sample shift `z^{-d_int}` with a 2nd-order Thiran all-pass filter
/// (`D = 2.0 + frac`, `|H| == 1.0` to machine precision, zero sinc pre-ringing).
#[inline]
fn causal_frac_delay(f_hz: f64, total_samples: f64, fs: f64) -> Complex {
    let z1 = Complex::from_polar(1.0, -2.0 * PI * f_hz / fs);
    causal_frac_delay_z(f_hz, total_samples, fs, z1)
}

#[inline]
fn causal_frac_delay_z(f_hz: f64, total_samples: f64, fs: f64, z1: Complex) -> Complex {
    if total_samples < 2.0 {
        let d_int = total_samples.round().max(0.0);
        return Complex::from_polar(1.0, -2.0 * PI * f_hz * d_int / fs);
    }
    let d_int = (total_samples - 2.0).floor();
    let frac = (total_samples - 2.0 - d_int).clamp(0.0, 0.999_999);
    let d = 2.0 + frac;
    let a1 = -2.0 * (d - 2.0) / (d + 1.0);
    let a2 = ((d - 1.0) * (d - 2.0)) / ((d + 1.0) * (d + 2.0));

    let z2 = z1.mul(z1);
    let num = Complex::new(a2, 0.0).add(z1.scale(a1)).add(z2);
    let den = Complex::ONE.add(z1.scale(a1)).add(z2.scale(a2));
    let ap = num.div(den);
    let shift = Complex::from_polar(1.0, -2.0 * PI * f_hz * d_int / fs);
    shift.mul(ap)
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
/// Uses single-partition OLA (`N_IR = 2048`, `N_FFT = 8192`) for anechoic presets and
/// **Zero-Latency Frequency-Domain Delay-Line Uniform Partitioned Overlap-Add (FDL-UPOLA)**
/// (`B = 2048`, `N_FFT = 4096`, `P = 8..16` partitions covering `341..683 ms` at 48 kHz)
/// for 3D Hybrid Room BRIRs — supporting BOTH FP64 Stockham FFT and 64-bit Goldilocks
/// `--exact-ntt` with strict 1:1 immediate frame emission and arbitrary chunk-size invariance.
/// Optional device-resident exact-NTT UPOLA state (`docs/CUDA_PERF_PLAN.md` section 1).
///
/// The delay line belongs to one stream, so `Clone` deliberately starts the copy without
/// device state: the stateless per-call integer NTT path is bit-identical and does not
/// silently share one FDL between two independently advancing streams.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct NttCudaHandle(Option<std::sync::Arc<std::sync::Mutex<g6me_cuda::CudaUpolaState>>>);

#[cfg(feature = "cuda")]
impl Clone for NttCudaHandle {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for NttCudaHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(_) => f.write_str("NttCudaHandle(device-resident)"),
            None => f.write_str("NttCudaHandle(none)"),
        }
    }
}

// Test-only device-stage failure seam for the device-resident UPOLA path. The alignment
// path must leave every host streaming field untouched when a device call fails, so a test
// arms a stage here, calls the hop, and compares the host state to a snapshot.
//
// Stage 1 = pre-write tail accumulate, stage 2 = final inverse/D2H. Thread-local so parallel
// tests cannot interfere; the no-op build is used in every non-test compilation and inlines
// away.
#[cfg(all(test, feature = "cuda"))]
thread_local! {
    static UPOLA_FAIL_STAGE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static UPOLA_INVERSE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline(always)]
fn upola_device_failpoint(stage: u8) -> Result<(), String> {
    #[cfg(all(test, feature = "cuda"))]
    {
        let hit = UPOLA_FAIL_STAGE.with(|f| {
            if f.get() == stage {
                f.set(0);
                true
            } else {
                false
            }
        });
        if hit {
            return Err(format!("injected device UPOLA failure at stage {stage}"));
        }
    }
    let _ = stage;
    Ok(())
}

#[inline(always)]
fn upola_note_inverse() {
    #[cfg(all(test, feature = "cuda"))]
    UPOLA_INVERSE_CALLS.with(|c| c.set(c.get() + 1));
}

#[derive(Debug, Clone)]
pub struct BinauralPlus {
    mask: u32,
    sample_rate: u32,
    cfg: PlusCfg,
    pre_delay_samples: usize,
    n_ir: usize,
    /// Number of uniform partitions (`1` for anechoic single-block OLA, `> 1` for FDL-UPOLA).
    n_parts: usize,
    /// Partition block size `B` (for UPOLA) or max OLA hop (for single-partition).
    ola_hop: usize,
    plan: FftPlan,
    /// Pre-transformed frequency-domain kernels `[part][slot][bin]` (`n_parts` partitions).
    kern_l: Vec<Vec<Vec<Complex>>>,
    kern_r: Vec<Vec<Vec<Complex>>>,
    /// Overlap tails (`n_ir - 1` for single-part, `2 * B - 1` for FDL-UPOLA).
    ov_l: Vec<f64>,
    ov_r: Vec<f64>,
    /// UPOLA Frequency-Domain Delay Line `[fdl_slot][ch][bin]` storing last `n_parts - 1`
    /// completed `B`-sample input spectra (FFT path).
    fdl_fft: Vec<Vec<Vec<Complex>>>,
    /// UPOLA per-channel current-block accumulation buffer `[ch][0..B]` for sub-block invariance.
    cur_in: Vec<Vec<f64>>,
    cur_fill: usize,
    fdl_head: usize,
    /// Reusable per-block scratch buffers (zero allocation in `process`).
    in_pad: Vec<f64>,
    fft_scratch: Vec<Complex>,
    x_bins: Vec<Complex>,
    acc_l: Vec<Complex>,
    acc_r: Vec<Complex>,
    y_l: Vec<f64>,
    y_r: Vec<f64>,
    /// Optional 64-bit Goldilocks prime-field NTT OLA / FDL-UPOLA engine (`--exact-ntt`).
    ntt_plan: Option<NttPlan>,
    ntt_kern_l: Vec<Vec<Vec<u64>>>,
    ntt_kern_r: Vec<Vec<Vec<u64>>>,
    ntt_fdl: Vec<Vec<Vec<u64>>>,
    ntt_ov_l: Vec<i128>,
    ntt_ov_r: Vec<i128>,
    ntt_x: Vec<u64>,
    ntt_acc_l: Vec<u64>,
    ntt_acc_r: Vec<u64>,
    ntt_kernel_scale: f64,
    /// VRAM-resident FDL-UPOLA state (`None` on CPU-only hosts or when the device path was
    /// declined); the per-call integer NTT path below stays the bit-identical fallback.
    #[cfg(feature = "cuda")]
    ntt_cuda: NttCudaHandle,
    finite: bool,
}

impl BinauralPlus {
    pub fn new(mask: u32, cfg: PlusCfg, sample_rate: u32) -> Self {
        Self::new_with_assets(mask, cfg, sample_rate, None, None)
    }

    pub fn new_with_room(
        mask: u32,
        cfg: PlusCfg,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
    ) -> Self {
        Self::new_with_assets(mask, cfg, sample_rate, custom_room, None)
    }

    pub fn new_with_assets(
        mask: u32,
        cfg: PlusCfg,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
        custom_head_mesh: Option<&TriangleMesh>,
    ) -> Self {
        let bank = KernelBank::synthesize_cached(cfg, sample_rate, custom_room, custom_head_mesh);
        let scale = sr_scale(sample_rate);
        let n_ir = bank.ir_l[0].len();
        let use_upola = n_ir > N_IR * scale;
        let (n_parts, ola_hop, n_fft, ov_len) = if use_upola {
            let b = UPOLA_PART * scale;
            let p = n_ir.div_ceil(b);
            (p, b, 2 * b, 2 * b - 1)
        } else {
            let nf = N_FFT * scale;
            (1usize, nf - n_ir, nf, n_ir - 1)
        };
        let plan = FftPlan::new(n_fft);
        let nb = n_fft / 2 + 1;
        let mut scratch = vec![Complex::ZERO; n_fft];
        let mut pad = vec![0.0_f64; n_fft];
        let mut kern_l = vec![vec![vec![Complex::ZERO; nb]; SLOTS]; n_parts];
        let mut kern_r = vec![vec![vec![Complex::ZERO; nb]; SLOTS]; n_parts];

        for p in 0..n_parts {
            let start = p * ola_hop;
            let len = if use_upola {
                ola_hop.min(n_ir.saturating_sub(start))
            } else {
                n_ir
            };
            for ch in 0..SLOTS {
                pad.fill(0.0);
                if len > 0 {
                    pad[..len].copy_from_slice(&bank.ir_l[ch][start..start + len]);
                }
                plan.rfft(&pad, &mut scratch, &mut kern_l[p][ch]);

                pad.fill(0.0);
                if len > 0 {
                    pad[..len].copy_from_slice(&bank.ir_r[ch][start..start + len]);
                }
                plan.rfft(&pad, &mut scratch, &mut kern_r[p][ch]);
            }
        }

        // Choose exact power-of-two NTT kernel quantization scale with guaranteed >= 4x headroom
        // below `GOLDILOCKS_P / 2` across all active channels and full-scale 24-bit PCM.
        let mut ntt_kernel_scale = NTT_KERNEL_SCALE;
        if cfg.exact_ntt {
            let total_l1: f64 = (0..SLOTS)
                .map(|ch| {
                    let l1_l: f64 = bank.ir_l[ch].iter().map(|v| v.abs()).sum();
                    let l1_r: f64 = bank.ir_r[ch].iter().map(|v| v.abs()).sum();
                    l1_l.max(l1_r)
                })
                .sum();
            let max_safe = 0.25 * (GOLDILOCKS_P as f64);
            while ntt_kernel_scale > 65_536.0
                && total_l1 * NTT_PCM_SCALE * ntt_kernel_scale > max_safe
            {
                ntt_kernel_scale *= 0.5;
            }
        }

        let fdl_slots = n_parts.saturating_sub(1);
        let fdl_fft = if !cfg.exact_ntt && fdl_slots > 0 {
            vec![vec![vec![Complex::ZERO; nb]; SLOTS]; fdl_slots]
        } else {
            Vec::new()
        };
        let cur_in = if n_parts > 1 {
            vec![vec![0.0_f64; ola_hop]; SLOTS]
        } else {
            Vec::new()
        };

        let (
            ntt_plan,
            ntt_kern_l,
            ntt_kern_r,
            ntt_fdl,
            ntt_ov_l,
            ntt_ov_r,
            ntt_x,
            ntt_acc_l,
            ntt_acc_r,
        ) = if cfg.exact_ntt {
            let np = NttPlan::new(n_fft);
            let mut nkl = vec![vec![vec![0u64; n_fft]; SLOTS]; n_parts];
            let mut nkr = vec![vec![vec![0u64; n_fft]; SLOTS]; n_parts];
            for p in 0..n_parts {
                let start = p * ola_hop;
                let len = if use_upola {
                    ola_hop.min(n_ir.saturating_sub(start))
                } else {
                    n_ir
                };
                for ch in 0..SLOTS {
                    for i in 0..len {
                        let ql = (bank.ir_l[ch][start + i] * ntt_kernel_scale).round() as i64;
                        let qr = (bank.ir_r[ch][start + i] * ntt_kernel_scale).round() as i64;
                        nkl[p][ch][i] = i64_to_field(ql);
                        nkr[p][ch][i] = i64_to_field(qr);
                    }
                    np.forward(&mut nkl[p][ch]);
                    np.forward(&mut nkr[p][ch]);
                }
            }
            let nfdl = if fdl_slots > 0 {
                vec![vec![vec![0u64; n_fft]; SLOTS]; fdl_slots]
            } else {
                Vec::new()
            };
            (
                Some(np),
                nkl,
                nkr,
                nfdl,
                vec![0i128; ov_len],
                vec![0i128; ov_len],
                vec![0u64; n_fft],
                vec![0u64; n_fft],
                vec![0u64; n_fft],
            )
        } else {
            (
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };

        // Device-resident FDL-UPOLA state (docs/CUDA_PERF_PLAN.md section 1): the NTT-domain
        // kernels, twiddle stages, delay line and both ear accumulators are uploaded once per
        // preset, so a streaming block no longer ships every 8192-word buffer down and back.
        #[cfg(feature = "cuda")]
        let ntt_cuda = {
            let mut handle = NttCudaHandle::default();
            if cfg.use_cuda {
                if let Some(np) = ntt_plan.as_ref() {
                    if let Some(rt) = crate::blackwell::shared_cuda_runtime() {
                        let mut flat_l = Vec::with_capacity(ntt_kern_l.len() * SLOTS * n_fft);
                        let mut flat_r = Vec::with_capacity(flat_l.capacity());
                        for parts in &ntt_kern_l {
                            for ch in parts {
                                flat_l.extend_from_slice(ch);
                            }
                        }
                        for parts in &ntt_kern_r {
                            for ch in parts {
                                flat_r.extend_from_slice(ch);
                            }
                        }
                        match rt.create_upola_state(
                            n_fft,
                            SLOTS,
                            n_parts,
                            np.inv_n(),
                            &flat_l,
                            &flat_r,
                            np.twiddles_fwd(),
                            np.twiddles_inv(),
                        ) {
                            Ok(state) => {
                                handle = NttCudaHandle(Some(std::sync::Arc::new(
                                    std::sync::Mutex::new(state),
                                )));
                            }
                            Err(e) => crate::cuda_report::record_cpu_fallback(
                                "ntt",
                                format!("device-resident UPOLA unavailable: {e}"),
                            ),
                        }
                    }
                }
            }
            handle
        };

        Self {
            mask,
            sample_rate,
            cfg,
            pre_delay_samples: bank.pre_delay_samples,
            n_ir,
            n_parts,
            ola_hop,
            plan,
            kern_l,
            kern_r,
            ov_l: vec![0.0_f64; ov_len],
            ov_r: vec![0.0_f64; ov_len],
            fdl_fft,
            cur_in,
            cur_fill: 0,
            fdl_head: 0,
            in_pad: vec![0.0_f64; n_fft],
            fft_scratch: vec![Complex::ZERO; n_fft],
            x_bins: vec![Complex::ZERO; nb],
            acc_l: vec![Complex::ZERO; nb],
            acc_r: vec![Complex::ZERO; nb],
            y_l: vec![0.0_f64; n_fft],
            y_r: vec![0.0_f64; n_fft],
            ntt_plan,
            ntt_kern_l,
            ntt_kern_r,
            ntt_fdl,
            ntt_ov_l,
            ntt_ov_r,
            ntt_x,
            ntt_acc_l,
            ntt_acc_r,
            ntt_kernel_scale,
            #[cfg(feature = "cuda")]
            ntt_cuda,
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
    /// untouched; multi-channel inputs are convolved with exact streaming overlap-add
    /// (single-partition OLA for anechoic, Zero-Latency FDL-UPOLA for long room BRIRs).
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
                if l[..frames].iter().any(|v| !v.is_finite()) {
                    self.finite = false;
                }
                out_l[..frames].copy_from_slice(&l[..frames]);
            } else {
                out_l[..frames].fill(0.0);
            }
            if let Some(r) = inputs[SLOT_FR] {
                if r[..frames].iter().any(|v| !v.is_finite()) {
                    self.finite = false;
                }
                out_r[..frames].copy_from_slice(&r[..frames]);
            } else {
                out_r[..frames].fill(0.0);
            }
            return;
        }

        let mut offset = 0usize;
        while offset < frames {
            let max_step = if self.n_parts > 1 {
                self.ola_hop - self.cur_fill
            } else {
                self.ola_hop
            };
            let m = (frames - offset).min(max_step);
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

    #[inline]
    fn ntt_forward_buf(use_cuda: bool, np: &NttPlan, buf: &mut [u64]) {
        #[cfg(feature = "cuda")]
        if use_cuda {
            match crate::blackwell::shared_cuda_runtime() {
                Some(rt) => match rt.run_goldilocks_ntt_transform(buf, np.twiddles_fwd(), None) {
                    Ok(out) => {
                        buf.copy_from_slice(&out);
                        crate::cuda_report::record_gpu("ntt", "");
                        return;
                    }
                    Err(e) => crate::cuda_report::record_cpu_fallback("ntt", e),
                },
                None => crate::cuda_report::record_cpu_fallback("ntt", "CUDA runtime unavailable"),
            }
        }
        let _ = use_cuda;
        np.forward(buf);
    }

    #[inline]
    fn ntt_inverse_buf(use_cuda: bool, np: &NttPlan, buf: &mut [u64]) {
        #[cfg(feature = "cuda")]
        if use_cuda {
            match crate::blackwell::shared_cuda_runtime() {
                Some(rt) => {
                    match rt.run_goldilocks_ntt_transform(buf, np.twiddles_inv(), Some(np.inv_n()))
                    {
                        Ok(out) => {
                            buf.copy_from_slice(&out);
                            crate::cuda_report::record_gpu("ntt", "");
                            return;
                        }
                        Err(e) => crate::cuda_report::record_cpu_fallback("ntt", e),
                    }
                }
                None => crate::cuda_report::record_cpu_fallback("ntt", "CUDA runtime unavailable"),
            }
        }
        let _ = use_cuda;
        np.inverse(buf);
    }

    #[inline]
    fn rfft_buf(
        use_cuda: bool,
        plan: &FftPlan,
        input: &[f64],
        scratch: &mut [Complex],
        out_bins: &mut [Complex],
    ) {
        #[cfg(feature = "cuda")]
        if use_cuda {
            match crate::blackwell::shared_cuda_runtime() {
                Some(rt) => {
                    // Real-input FFT: zero imaginary lane is produced on-device (no host zero
                    // upload) and the twiddles are device-resident across blocks.
                    match rt.run_stockham_fft_f64_real_input(input, false) {
                        Ok((re, im)) => {
                            for (k, slot) in out_bins.iter_mut().enumerate() {
                                *slot = Complex::new(re[k], im[k]);
                            }
                            crate::cuda_report::record_gpu("fft", "");
                            return;
                        }
                        Err(e) => crate::cuda_report::record_cpu_fallback("fft", e),
                    }
                }
                None => crate::cuda_report::record_cpu_fallback("fft", "CUDA runtime unavailable"),
            }
        }
        let _ = use_cuda;
        plan.rfft(input, scratch, out_bins);
    }

    #[inline]
    fn irfft_buf(
        use_cuda: bool,
        plan: &FftPlan,
        bins: &[Complex],
        scratch: &mut [Complex],
        out_real: &mut [f64],
    ) {
        #[cfg(feature = "cuda")]
        if use_cuda {
            match crate::blackwell::shared_cuda_runtime() {
                Some(rt) => {
                    let n = plan.len();
                    let nb = bins.len();
                    let mut re = vec![0.0_f64; n];
                    let mut im = vec![0.0_f64; n];
                    for k in 0..nb {
                        re[k] = bins[k].re;
                        im[k] = bins[k].im;
                    }
                    for k in 1..(n / 2) {
                        re[n - k] = bins[k].re;
                        im[n - k] = -bins[k].im;
                    }
                    match rt.run_stockham_fft_f64_real_output(&re, &im, true) {
                        Ok(out_re) => {
                            out_real.copy_from_slice(&out_re);
                            crate::cuda_report::record_gpu("fft", "");
                            return;
                        }
                        Err(e) => crate::cuda_report::record_cpu_fallback("fft", e),
                    }
                }
                None => crate::cuda_report::record_cpu_fallback("fft", "CUDA runtime unavailable"),
            }
        }
        let _ = use_cuda;
        plan.irfft(bins, scratch, out_real);
    }

    /// Push the completed `B`-sample `cur_in` block into the UPOLA Frequency-Domain Delay Line.
    ///
    /// The exact-NTT engine only stores the block's spectrum: its `p >= 1` tail is formed on
    /// demand at the next partial block's first hop, or folded into the next aligned block's
    /// single-inverse sum. The float engine retains the tail accumulate into `ov_l`/`ov_r`.
    fn push_completed_upola_block(&mut self) {
        let fdl_len = self.n_parts - 1;
        let head = self.fdl_head;
        let b = self.ola_hop;
        let ov_len = 2 * b - 1;
        let use_cuda = self.cfg.use_cuda;

        if let Some(ref np) = self.ntt_plan {
            // Write the completed block into the delay line. No inverse happens here: a
            // following partial block inverse-transforms the deferred tail at its first
            // hop, while a following aligned block folds it into its single U inverse.
            for ch in 0..SLOTS {
                if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                    continue;
                }
                self.ntt_x.fill(0);
                let src = &self.cur_in[ch];
                if src.iter().any(|&v| v != 0.0) {
                    for (d, &s) in self.ntt_x.iter_mut().take(b).zip(src.iter()) {
                        let q = (s * NTT_PCM_SCALE).round().clamp(-8_388_608.0, 8_388_607.0) as i64;
                        *d = i64_to_field(q);
                    }
                    Self::ntt_forward_buf(use_cuda, np, &mut self.ntt_x);
                }
                self.ntt_fdl[head][ch].copy_from_slice(&self.ntt_x);
            }
        } else {
            let nb = self.plan.len() / 2 + 1;
            for ch in 0..SLOTS {
                if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                    continue;
                }
                let src = &self.cur_in[ch];
                if src.iter().any(|&v| v != 0.0) {
                    self.in_pad[..b].copy_from_slice(src);
                    self.in_pad[b..].fill(0.0);
                    Self::rfft_buf(
                        use_cuda,
                        &self.plan,
                        &self.in_pad,
                        &mut self.fft_scratch,
                        &mut self.fdl_fft[head][ch],
                    );
                } else {
                    self.fdl_fft[head][ch].fill(Complex::ZERO);
                }
            }

            // Accumulate all tail partitions p = 1..n_parts with ONE inverse FFT per ear!
            self.acc_l.fill(Complex::ZERO);
            self.acc_r.fill(Complex::ZERO);
            for p in 1..self.n_parts {
                let slot_idx = (head + fdl_len - (p - 1)) % fdl_len;
                for ch in 0..SLOTS {
                    if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                        continue;
                    }
                    let xf = &self.fdl_fft[slot_idx][ch];
                    let kl = &self.kern_l[p][ch];
                    let kr = &self.kern_r[p][ch];
                    for k in 0..nb {
                        let xv = xf[k];
                        self.acc_l[k] = self.acc_l[k].add(xv.mul(kl[k]));
                        self.acc_r[k] = self.acc_r[k].add(xv.mul(kr[k]));
                    }
                }
            }
            Self::irfft_buf(
                use_cuda,
                &self.plan,
                &self.acc_l,
                &mut self.fft_scratch,
                &mut self.y_l,
            );
            Self::irfft_buf(
                use_cuda,
                &self.plan,
                &self.acc_r,
                &mut self.fft_scratch,
                &mut self.y_r,
            );
            for i in 0..ov_len {
                self.ov_l[i] += self.y_l[i];
                self.ov_r[i] += self.y_r[i];
            }
        }

        self.fdl_head = (head + 1) % fdl_len;
        self.cur_fill = 0;
    }

    /// Effective channel mask for the NTT accumulation loops: `SLOT_FL`/`SLOT_FR` are always
    /// included regardless of the speaker mask, exactly like the CPU loops' condition.
    #[cfg(feature = "cuda")]
    fn ntt_active_mask(&self) -> u32 {
        self.mask | (1u32 << SLOT_FL) | (1u32 << SLOT_FR)
    }

    /// Quantize one `f64` hop into the exact 24-bit PCM field elements used by the NTT
    /// (identical expression to the stateless path), zero-padding to `ntt_x.len()`.
    #[cfg(feature = "cuda")]
    fn ntt_quantize_hop(&mut self, slice: &[f64]) {
        self.ntt_x.fill(0);
        for (dst, &sample) in self.ntt_x.iter_mut().zip(slice.iter()) {
            let q = (sample * NTT_PCM_SCALE)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i64;
            *dst = i64_to_field(q);
        }
    }

    /// Device-resident NTT hop (`docs/CUDA_PERF_PLAN.md` section 1).
    ///
    /// Mirrors the integer branch of [`Self::process_hop`] operation-for-operation:
    /// forward transforms per active channel, exact pointwise accumulation, one batched
    /// inverse per ear, then the same overlap-add bookkeeping. All device work for the hop
    /// completes before any host streaming state changes, so an `Err` return leaves the
    /// caller free to fall back for this call.
    ///
    /// This method is only reachable when `self.ntt_plan.is_some()`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn process_hop_ntt_cuda(
        &mut self,
        state: &std::sync::Arc<std::sync::Mutex<g6me_cuda::CudaUpolaState>>,
        inputs: &[Option<&[f64]>; SLOTS],
        offset: usize,
        out_l: &mut [f64],
        out_r: &mut [f64],
        m: usize,
    ) -> Result<(), String> {
        let n_fft = self
            .ntt_plan
            .as_ref()
            .map(|p| p.len())
            .ok_or_else(|| "device UPOLA hop without an NTT plan".to_string())?;
        let use_upola = self.n_parts > 1;
        let full_aligned_upola = use_upola && self.cur_fill == 0 && m == self.ola_hop;
        let mask = self.ntt_active_mask();
        let head = self.fdl_head;
        let mut st = state
            .lock()
            .map_err(|_| "ntt cuda state poisoned".to_string())?;
        let inv_scale = 1.0 / (NTT_PCM_SCALE * self.ntt_kernel_scale);
        let ov_len = self.ntt_ov_l.len();

        if full_aligned_upola {
            // ---- completed whole hop: one inverse for the whole output block -----------
            // Standard single-inverse UPOLA: form U_j = sum_{p=0}^{P-1} x_{j-p} h_p in the
            // NTT domain, inverse once, emits out = U_j + (previous U tail), and keeps the
            // new overlap tail U_j[B..]. Partitions p >= 1 read the delay line *before*
            // this block is written, so the row that block j evicts is still available;
            // partition 0 is folded in from the freshly written row. The old split
            // (partition-0 inverse for this block + tail inverse for the next block)
            // launched two inverses and two synchronous D2H copies per aligned hop.
            upola_device_failpoint(1)?;
            st.accumulate_fdl_prev(head, mask)?;
            st.zero_fdl_row(head)?;
            for (ch, slot) in inputs.iter().enumerate() {
                if (mask >> ch) & 1 == 0 {
                    continue;
                }
                let Some(slice) = slot.map(|buf| &buf[offset..offset + m]) else {
                    continue;
                };
                if !slice.iter().any(|&v| v != 0.0) {
                    continue;
                }
                self.ntt_quantize_hop(slice);
                st.upload_fdl_row(head, ch, &self.ntt_x)?;
            }
            st.forward_fdl_row(head, mask)?;
            st.accumulate_fdl_add_p0(head, mask)?;
            upola_note_inverse();
            upola_device_failpoint(2)?;
            let u = st.inverse_acc_to_host()?;
            let (u_l, u_r) = u.split_at(n_fft);

            // ---- all device work for the hop is done: commit host state ----------------
            for i in 0..m {
                let ol = if i < ov_len { self.ntt_ov_l[i] } else { 0 };
                let or = if i < ov_len { self.ntt_ov_r[i] } else { 0 };
                let vl = ((field_to_i128(u_l[i]) + ol) as f64) * inv_scale;
                let vr = ((field_to_i128(u_r[i]) + or) as f64) * inv_scale;
                if !vl.is_finite() || !vr.is_finite() {
                    self.finite = false;
                }
                out_l[i] = vl;
                out_r[i] = vr;
            }
            let y_len = m + self.ola_hop - 1;
            for j in 0..ov_len {
                let src = m + j;
                self.ntt_ov_l[j] = if src < y_len {
                    field_to_i128(u_l[src])
                } else {
                    0
                };
                self.ntt_ov_r[j] = if src < y_len {
                    field_to_i128(u_r[src])
                } else {
                    0
                };
            }
            self.fdl_head = (head + 1) % (self.n_parts - 1);
            crate::cuda_report::record_gpu("ntt", "device-resident UPOLA");
            return Ok(());
        }

        // ---- partial hop, or the single-partition anechoic engine -------------------
        // A partial block emits its samples before the block completes, so the p >= 1
        // contribution of the *current* block, T_{j-1} = sum_{p=1}^{P-1} x_{j-p} h_p,
        // is inverse-transformed once at the first hop and added to the overlap buffer.
        // The aligned path folds the same sum into its single U inverse. At completion
        // the overlap buffer has advanced to U_j[B..] by construction (the partition-0
        // slice tails shift it there), so no second inverse is needed at the push.
        let block_start_partial = use_upola && self.cur_fill == 0 && m < self.ola_hop;
        let t_vec = if block_start_partial {
            upola_device_failpoint(1)?;
            st.accumulate_fdl_prev(head, mask)?;
            upola_note_inverse();
            upola_device_failpoint(2)?;
            Some(st.inverse_acc_to_host()?)
        } else {
            None
        };

        let mut contribution_mask = 0u32;
        for (ch, slot) in inputs.iter().enumerate() {
            if (mask >> ch) & 1 == 0 {
                continue;
            }
            let slice_opt = slot.map(|buf| &buf[offset..offset + m]);
            if use_upola {
                // Idempotent on retry: the same samples land at the same `cur_fill`.
                if let Some(sl) = slice_opt {
                    self.cur_in[ch][self.cur_fill..self.cur_fill + m].copy_from_slice(sl);
                } else {
                    self.cur_in[ch][self.cur_fill..self.cur_fill + m].fill(0.0);
                }
            }
            let Some(slice) = slice_opt else { continue };
            if !slice.iter().any(|&v| v != 0.0) {
                continue;
            }
            self.ntt_quantize_hop(slice);
            st.upload_scratch_row(ch, &self.ntt_x)?;
            contribution_mask |= 1u32 << ch;
        }
        let acc = if contribution_mask != 0 {
            st.forward_scratch(contribution_mask)?;
            st.mac_scratch_partition0(contribution_mask)?;
            upola_note_inverse();
            upola_device_failpoint(2)?;
            st.inverse_acc_to_host()?
        } else {
            // Zero input transforms and accumulates to the exact zero vector.
            vec![0u64; 2 * n_fft]
        };
        let (acc_l, acc_r) = acc.split_at(n_fft);

        // Device write of the just-completed block into the delay line (only when this hop
        // filled it). No inverse here: a following partial block inverse-transforms the
        // deferred tail at its first hop, and a following aligned block folds it into its
        // single U inverse.
        let push = use_upola && self.cur_fill + m == self.ola_hop;
        if push {
            self.write_completed_upola_block_cuda(&mut st, head)?;
        }

        // ---- commit host state (all fallible device work is complete) --------------
        if let Some(t) = t_vec.as_ref() {
            let (t_l, t_r) = t.split_at(n_fft);
            for i in 0..ov_len {
                self.ntt_ov_l[i] += field_to_i128(t_l[i]);
                self.ntt_ov_r[i] += field_to_i128(t_r[i]);
            }
        }
        for i in 0..m {
            let ol = if i < ov_len { self.ntt_ov_l[i] } else { 0 };
            let or = if i < ov_len { self.ntt_ov_r[i] } else { 0 };
            let vl = ((field_to_i128(acc_l[i]) + ol) as f64) * inv_scale;
            let vr = ((field_to_i128(acc_r[i]) + or) as f64) * inv_scale;
            if !vl.is_finite() || !vr.is_finite() {
                self.finite = false;
            }
            out_l[i] = vl;
            out_r[i] = vr;
        }
        let y_len = if use_upola {
            m + self.ola_hop - 1
        } else {
            n_fft
        };
        for j in 0..ov_len {
            let src = m + j;
            let prev_l = if src < ov_len { self.ntt_ov_l[src] } else { 0 };
            let prev_r = if src < ov_len { self.ntt_ov_r[src] } else { 0 };
            let add_l = if src < y_len {
                field_to_i128(acc_l[src])
            } else {
                0
            };
            let add_r = if src < y_len {
                field_to_i128(acc_r[src])
            } else {
                0
            };
            self.ntt_ov_l[j] = prev_l + add_l;
            self.ntt_ov_r[j] = prev_r + add_r;
        }
        if push {
            self.fdl_head = (head + 1) % (self.n_parts - 1);
            self.cur_fill = 0;
        } else if use_upola {
            self.cur_fill += m;
        }
        crate::cuda_report::record_gpu("ntt", "device-resident UPOLA");
        Ok(())
    }

    /// Device-side body of [`Self::push_completed_upola_block`]: transform the completed
    /// `ola_hop`-sample `cur_in` block into delay-line row `head`. The block's p >= 1
    /// contributions are not inverse-transformed here; a following partial block forms the
    /// deferred tail at its first hop, while a following aligned block folds it into its
    /// single U inverse.
    #[cfg(feature = "cuda")]
    fn write_completed_upola_block_cuda(
        &mut self,
        st: &mut g6me_cuda::CudaUpolaState,
        head: usize,
    ) -> Result<(), String> {
        let mask = self.ntt_active_mask();
        let b = self.ola_hop;
        st.zero_fdl_row(head)?;
        for ch in 0..SLOTS {
            if (mask >> ch) & 1 == 0 {
                continue;
            }
            let src = &self.cur_in[ch];
            if !src.iter().any(|&v| v != 0.0) {
                continue;
            }
            self.ntt_x.fill(0);
            for (d, &s) in self.ntt_x.iter_mut().take(b).zip(src.iter()) {
                let q = (s * NTT_PCM_SCALE).round().clamp(-8_388_608.0, 8_388_607.0) as i64;
                *d = i64_to_field(q);
            }
            st.upload_fdl_row(head, ch, &self.ntt_x)?;
        }
        st.forward_fdl_row(head, mask)
    }

    fn process_hop(
        &mut self,
        inputs: &[Option<&[f64]>; SLOTS],
        offset: usize,
        out_l: &mut [f64],
        out_r: &mut [f64],
        m: usize,
    ) {
        debug_assert!((1..=self.ola_hop).contains(&m));
        let use_upola = self.n_parts > 1;
        let full_aligned_upola = use_upola && self.cur_fill == 0 && m == self.ola_hop;
        let use_cuda = self.cfg.use_cuda;

        // Device-resident FDL-UPOLA path. It mirrors the integer NTT branch below exactly and
        // returns without touching host streaming state on error, so a failed device hop can
        // fall back to the stateless path for this call.
        #[cfg(feature = "cuda")]
        if use_cuda && self.ntt_plan.is_some() {
            if let Some(state) = self.ntt_cuda.0.clone() {
                match self.process_hop_ntt_cuda(&state, inputs, offset, out_l, out_r, m) {
                    Ok(()) => return,
                    Err(e) => {
                        crate::cuda_report::record_cpu_fallback("ntt", e);
                        self.ntt_cuda = NttCudaHandle::default();
                        // The host delay line was not maintained while the device owned it;
                        // zero the decay tails so the fallback cannot replay stale history.
                        self.ntt_ov_l.fill(0);
                        self.ntt_ov_r.fill(0);
                    }
                }
            }
        }

        if let Some(ref np) = self.ntt_plan {
            let n_fft = np.len();
            let head = self.fdl_head;
            let fdl_len = self.n_parts.saturating_sub(1);

            // A partial block emits its samples before the block completes, so its p >= 1
            // contribution T_{j-1} = sum_{p=1}^{P-1} x_{j-p} h_p is inverse-transformed
            // once at the first hop and added to the overlap buffer. The aligned path folds
            // the same sum into one U inverse below.
            if use_upola && !full_aligned_upola && self.cur_fill == 0 {
                self.ntt_acc_l.fill(0);
                self.ntt_acc_r.fill(0);
                for p in 1..self.n_parts {
                    let slot_idx = (head + fdl_len - p) % fdl_len;
                    for ch in 0..SLOTS {
                        if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                            continue;
                        }
                        let xf = &self.ntt_fdl[slot_idx][ch];
                        if !xf.iter().any(|&v| v != 0) {
                            continue;
                        }
                        let kl = &self.ntt_kern_l[p][ch];
                        let kr = &self.ntt_kern_r[p][ch];
                        for k in 0..n_fft {
                            let xv = xf[k];
                            self.ntt_acc_l[k] = mod_add(self.ntt_acc_l[k], mod_mul(xv, kl[k]));
                            self.ntt_acc_r[k] = mod_add(self.ntt_acc_r[k], mod_mul(xv, kr[k]));
                        }
                    }
                }
                Self::ntt_inverse_buf(use_cuda, np, &mut self.ntt_acc_l);
                Self::ntt_inverse_buf(use_cuda, np, &mut self.ntt_acc_r);
                let ov_len_t = self.ntt_ov_l.len();
                for i in 0..ov_len_t {
                    self.ntt_ov_l[i] += field_to_i128(self.ntt_acc_l[i]);
                    self.ntt_ov_r[i] += field_to_i128(self.ntt_acc_r[i]);
                }
            }

            self.ntt_acc_l.fill(0);
            self.ntt_acc_r.fill(0);
            if full_aligned_upola {
                // Standard single-inverse UPOLA: U_j = sum_{p=0}^{P-1} x_{j-p} h_p in the
                // NTT domain, one inverse, one overlap update. Partitions p >= 1 read the
                // delay line before block j overwrites the oldest row; partition 0 comes
                // from the freshly written row.
                for p in 1..self.n_parts {
                    let slot_idx = (head + fdl_len - p) % fdl_len;
                    for ch in 0..SLOTS {
                        if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                            continue;
                        }
                        let xf = &self.ntt_fdl[slot_idx][ch];
                        if !xf.iter().any(|&v| v != 0) {
                            continue;
                        }
                        let kl = &self.ntt_kern_l[p][ch];
                        let kr = &self.ntt_kern_r[p][ch];
                        for k in 0..n_fft {
                            let xv = xf[k];
                            self.ntt_acc_l[k] = mod_add(self.ntt_acc_l[k], mod_mul(xv, kl[k]));
                            self.ntt_acc_r[k] = mod_add(self.ntt_acc_r[k], mod_mul(xv, kr[k]));
                        }
                    }
                }
                for (ch, slot) in inputs.iter().enumerate() {
                    if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                        continue;
                    }
                    self.ntt_x.fill(0);
                    if let Some(slice) = slot.map(|buf| &buf[offset..offset + m]) {
                        if slice.iter().any(|&v| v != 0.0) {
                            for (dst, &sample) in self.ntt_x.iter_mut().take(m).zip(slice) {
                                let q = (sample * NTT_PCM_SCALE)
                                    .round()
                                    .clamp(-8_388_608.0, 8_388_607.0) as i64;
                                *dst = i64_to_field(q);
                            }
                            Self::ntt_forward_buf(use_cuda, np, &mut self.ntt_x);
                        }
                    }
                    let kl = &self.ntt_kern_l[0][ch];
                    let kr = &self.ntt_kern_r[0][ch];
                    for k in 0..n_fft {
                        let xv = self.ntt_x[k];
                        self.ntt_acc_l[k] = mod_add(self.ntt_acc_l[k], mod_mul(xv, kl[k]));
                        self.ntt_acc_r[k] = mod_add(self.ntt_acc_r[k], mod_mul(xv, kr[k]));
                    }
                    self.ntt_fdl[head][ch].copy_from_slice(&self.ntt_x);
                }
            } else {
                for (ch, slot) in inputs.iter().enumerate() {
                    if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                        continue;
                    }
                    let slice_opt = slot.map(|buf| &buf[offset..offset + m]);
                    if use_upola {
                        if let Some(sl) = slice_opt {
                            self.cur_in[ch][self.cur_fill..self.cur_fill + m].copy_from_slice(sl);
                        } else {
                            self.cur_in[ch][self.cur_fill..self.cur_fill + m].fill(0.0);
                        }
                    }
                    let Some(slice) = slice_opt else { continue };
                    if !slice.iter().any(|&v| v != 0.0) {
                        continue;
                    }
                    self.ntt_x.fill(0);
                    for (dst, &sample) in self.ntt_x.iter_mut().take(m).zip(slice) {
                        let q = (sample * NTT_PCM_SCALE)
                            .round()
                            .clamp(-8_388_608.0, 8_388_607.0) as i64;
                        *dst = i64_to_field(q);
                    }
                    Self::ntt_forward_buf(use_cuda, np, &mut self.ntt_x);
                    let kl = &self.ntt_kern_l[0][ch];
                    let kr = &self.ntt_kern_r[0][ch];
                    for k in 0..n_fft {
                        let xv = self.ntt_x[k];
                        self.ntt_acc_l[k] = mod_add(self.ntt_acc_l[k], mod_mul(xv, kl[k]));
                        self.ntt_acc_r[k] = mod_add(self.ntt_acc_r[k], mod_mul(xv, kr[k]));
                    }
                }
            }
            Self::ntt_inverse_buf(use_cuda, np, &mut self.ntt_acc_l);
            Self::ntt_inverse_buf(use_cuda, np, &mut self.ntt_acc_r);

            let inv_scale = 1.0 / (NTT_PCM_SCALE * self.ntt_kernel_scale);
            let ov_len = self.ntt_ov_l.len();
            for i in 0..m {
                let ol = if i < ov_len { self.ntt_ov_l[i] } else { 0 };
                let or = if i < ov_len { self.ntt_ov_r[i] } else { 0 };
                let vl = ((field_to_i128(self.ntt_acc_l[i]) + ol) as f64) * inv_scale;
                let vr = ((field_to_i128(self.ntt_acc_r[i]) + or) as f64) * inv_scale;
                if !vl.is_finite() || !vr.is_finite() {
                    self.finite = false;
                }
                out_l[i] = vl;
                out_r[i] = vr;
            }
            let y_len = if use_upola {
                m + self.ola_hop - 1
            } else {
                n_fft
            };
            for j in 0..ov_len {
                let src = m + j;
                let prev_l = if src < ov_len { self.ntt_ov_l[src] } else { 0 };
                let prev_r = if src < ov_len { self.ntt_ov_r[src] } else { 0 };
                let add_l = if src < y_len {
                    field_to_i128(self.ntt_acc_l[src])
                } else {
                    0
                };
                let add_r = if src < y_len {
                    field_to_i128(self.ntt_acc_r[src])
                } else {
                    0
                };
                self.ntt_ov_l[j] = prev_l + add_l;
                self.ntt_ov_r[j] = prev_r + add_r;
            }

            if full_aligned_upola {
                // The single inverse produced U_j; its second half is the complete overlap
                // tail for the next block (no separate tail inverse / add).
                for j in 0..ov_len {
                    let src = m + j;
                    self.ntt_ov_l[j] = if src < y_len {
                        field_to_i128(self.ntt_acc_l[src])
                    } else {
                        0
                    };
                    self.ntt_ov_r[j] = if src < y_len {
                        field_to_i128(self.ntt_acc_r[src])
                    } else {
                        0
                    };
                }
                self.fdl_head = (head + 1) % fdl_len;
            } else if use_upola {
                self.cur_fill += m;
                if self.cur_fill == self.ola_hop {
                    self.push_completed_upola_block();
                }
            }
            return;
        }

        let nb = self.plan.len() / 2 + 1;
        let head = self.fdl_head;
        self.acc_l.fill(Complex::ZERO);
        self.acc_r.fill(Complex::ZERO);

        for (ch, slot) in inputs.iter().enumerate() {
            if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                continue;
            }
            let slice_opt = slot.map(|buf| &buf[offset..offset + m]);
            if use_upola && !full_aligned_upola {
                if let Some(sl) = slice_opt {
                    self.cur_in[ch][self.cur_fill..self.cur_fill + m].copy_from_slice(sl);
                } else {
                    self.cur_in[ch][self.cur_fill..self.cur_fill + m].fill(0.0);
                }
            }
            let Some(slice) = slice_opt else {
                if full_aligned_upola {
                    self.fdl_fft[head][ch].fill(Complex::ZERO);
                }
                continue;
            };
            // Fast silence skip per channel.
            if !slice.iter().any(|&v| v != 0.0) {
                if full_aligned_upola {
                    self.fdl_fft[head][ch].fill(Complex::ZERO);
                }
                continue;
            }
            self.in_pad[..m].copy_from_slice(slice);
            self.in_pad[m..].fill(0.0);
            Self::rfft_buf(
                use_cuda,
                &self.plan,
                &self.in_pad,
                &mut self.fft_scratch,
                &mut self.x_bins,
            );
            if full_aligned_upola {
                self.fdl_fft[head][ch].copy_from_slice(&self.x_bins);
            }

            let kl = &self.kern_l[0][ch];
            let kr = &self.kern_r[0][ch];
            for k in 0..nb {
                let x = self.x_bins[k];
                self.acc_l[k] = self.acc_l[k].add(x.mul(kl[k]));
                self.acc_r[k] = self.acc_r[k].add(x.mul(kr[k]));
            }
        }

        Self::irfft_buf(
            use_cuda,
            &self.plan,
            &self.acc_l,
            &mut self.fft_scratch,
            &mut self.y_l,
        );
        Self::irfft_buf(
            use_cuda,
            &self.plan,
            &self.acc_r,
            &mut self.fft_scratch,
            &mut self.y_r,
        );

        let ov_len = self.ov_l.len();
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

        // Shift overlap buffer left by `m` and add partition 0's convolution tail.
        let y_len = if use_upola {
            m + self.ola_hop - 1
        } else {
            self.y_l.len()
        };
        for j in 0..ov_len {
            let src = m + j;
            let prev_l = if src < ov_len { self.ov_l[src] } else { 0.0 };
            let prev_r = if src < ov_len { self.ov_r[src] } else { 0.0 };
            let add_l = if src < y_len { self.y_l[src] } else { 0.0 };
            let add_r = if src < y_len { self.y_r[src] } else { 0.0 };
            self.ov_l[j] = prev_l + add_l;
            self.ov_r[j] = prev_r + add_r;
        }

        if full_aligned_upola {
            // Fast path: fdl_fft[head] is already transformed — sum p = 1..n_parts in 1 IFFT/ear!
            let fdl_len = self.n_parts - 1;
            self.acc_l.fill(Complex::ZERO);
            self.acc_r.fill(Complex::ZERO);
            for p in 1..self.n_parts {
                let slot_idx = (head + fdl_len - (p - 1)) % fdl_len;
                for ch in 0..SLOTS {
                    if ch != SLOT_FL && ch != SLOT_FR && (self.mask & (1u32 << ch)) == 0 {
                        continue;
                    }
                    let xf = &self.fdl_fft[slot_idx][ch];
                    let kl = &self.kern_l[p][ch];
                    let kr = &self.kern_r[p][ch];
                    for k in 0..nb {
                        let xv = xf[k];
                        self.acc_l[k] = self.acc_l[k].add(xv.mul(kl[k]));
                        self.acc_r[k] = self.acc_r[k].add(xv.mul(kr[k]));
                    }
                }
            }
            Self::irfft_buf(
                use_cuda,
                &self.plan,
                &self.acc_l,
                &mut self.fft_scratch,
                &mut self.y_l,
            );
            Self::irfft_buf(
                use_cuda,
                &self.plan,
                &self.acc_r,
                &mut self.fft_scratch,
                &mut self.y_r,
            );
            for i in 0..ov_len {
                self.ov_l[i] += self.y_l[i];
                self.ov_r[i] += self.y_r[i];
            }
            self.fdl_head = (head + 1) % fdl_len;
        } else if use_upola {
            self.cur_fill += m;
            if self.cur_fill == self.ola_hop {
                self.push_completed_upola_block();
            }
        }
    }

    /// Drain the remaining `n_ir - 1` overlap-add impulse tail at EOF and reset
    /// the tail state to zero. Returns empty vectors for stereo bypass.
    pub fn flush(&mut self) -> (Vec<f64>, Vec<f64>) {
        if self.mask == MASK_STEREO {
            return (Vec::new(), Vec::new());
        }
        if self.n_parts > 1 {
            let tail_len = self.n_ir - 1;
            let mut tl = vec![0.0_f64; tail_len];
            let mut tr = vec![0.0_f64; tail_len];
            let zeros: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            self.process(&zeros, &mut tl, &mut tr, tail_len);
            // Reset all FDL and overlap state cleanly.
            self.ov_l.fill(0.0);
            self.ov_r.fill(0.0);
            self.ntt_ov_l.fill(0);
            self.ntt_ov_r.fill(0);
            for s in &mut self.fdl_fft {
                for c in s {
                    c.fill(Complex::ZERO);
                }
            }
            for s in &mut self.ntt_fdl {
                for c in s {
                    c.fill(0);
                }
            }
            #[cfg(feature = "cuda")]
            if let Some(state) = self.ntt_cuda.0.as_ref() {
                if let Ok(mut st) = state.lock() {
                    let _ = st.reset();
                }
            }
            for c in &mut self.cur_in {
                c.fill(0.0);
            }
            self.cur_fill = 0;
            self.fdl_head = 0;
            return (tl, tr);
        }
        if self.ntt_plan.is_some() {
            let inv_scale = 1.0 / (NTT_PCM_SCALE * self.ntt_kernel_scale);
            let tl: Vec<f64> = self
                .ntt_ov_l
                .iter()
                .map(|&v| (v as f64) * inv_scale)
                .collect();
            let tr: Vec<f64> = self
                .ntt_ov_r
                .iter()
                .map(|&v| (v as f64) * inv_scale)
                .collect();
            self.ntt_ov_l.fill(0);
            self.ntt_ov_r.fill(0);
            return (tl, tr);
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
        Self::new_with_assets(mode, plus, mask, sample_rate, None, None)
    }

    pub fn new_with_room(
        mode: RendererMode,
        plus: PlusCfg,
        mask: u32,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
    ) -> Self {
        Self::new_with_assets(mode, plus, mask, sample_rate, custom_room, None)
    }

    pub fn new_with_assets(
        mode: RendererMode,
        plus: PlusCfg,
        mask: u32,
        sample_rate: u32,
        custom_room: Option<&RoomGeometry>,
        custom_head_mesh: Option<&TriangleMesh>,
    ) -> Self {
        match mode {
            RendererMode::Reference => Self::Classic(Box::new(Dsp::<f64>::new(mask))),
            RendererMode::Plus => Self::Plus(Box::new(BinauralPlus::new_with_assets(
                mask,
                plus,
                sample_rate,
                custom_room,
                custom_head_mesh,
            ))),
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

    /// High-sample-rate (`96 kHz`, `192 kHz`) kernels scale their FIR support to
    /// preserve all early reflections (`<= 30 ms`) and match `48 kHz` audible-band
    /// diffuse-field loudness within `±0.10 dB`.
    #[test]
    fn high_sample_rates_preserve_reflections_and_audible_loudness() {
        let cfg = PlusCfg::from_profile(PlusProfile::Studio);
        let b48 = KernelBank::synthesize(cfg, 48_000);
        let b96 = KernelBank::synthesize(cfg, 96_000);
        let b192 = KernelBank::synthesize(cfg, 192_000);

        assert_eq!(b48.ir_l[SLOT_FC].len(), N_IR);
        assert_eq!(b96.ir_l[SLOT_FC].len(), N_IR * 2);
        assert_eq!(b192.ir_l[SLOT_FC].len(), N_IR * 4);

        // FC's 28.4 ms lateral reflection (at ~29.4 ms with 1 ms pre-delay) must survive
        // intact at 96 kHz (sample ~2822) and 192 kHz (sample ~5645) with clean tail decay.
        let late96 = b96.ir_l[SLOT_FC][2700..3000]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            late96 > 1e-3,
            "96 kHz 28.4 ms FC reflection truncated or missing: peak {late96:e}"
        );
        let tail96_db = 20.0
            * b96.ir_l[SLOT_FC][b96.ir_l[SLOT_FC].len() - 4..]
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1e-20)
                .log10();
        assert!(
            tail96_db < -140.0,
            "96 kHz tail too loud: {tail96_db:.1} dB"
        );

        // Compare audible-band (20 Hz..18 kHz) 7.0 diffuse power between 48 kHz and 96/192 kHz.
        let audible_df_power = |bank: &KernelBank, n_fft: usize| -> f64 {
            let fs = bank.sample_rate as f64;
            let k_max = ((18_000.0 * n_fft as f64 / fs).round() as usize).min(n_fft / 2);
            let k_min = (20.0 * n_fft as f64 / fs).round() as usize;
            let mut sum = 0.0_f64;
            for &ch in &DF_RING_SLOTS {
                let mut pl = vec![0.0; n_fft];
                let mut pr = vec![0.0; n_fft];
                pl[..bank.ir_l[ch].len()].copy_from_slice(&bank.ir_l[ch]);
                pr[..bank.ir_r[ch].len()].copy_from_slice(&bank.ir_r[ch]);
                let sl = rfft(&pl);
                let sr = rfft(&pr);
                for k in k_min..=k_max {
                    sum += 0.5 * (sl[k].norm_sq() + sr[k].norm_sq());
                }
            }
            sum / ((k_max - k_min + 1) as f64)
        };
        let p48 = audible_df_power(&b48, 4096);
        let p96 = audible_df_power(&b96, 8192);
        let p192 = audible_df_power(&b192, 16384);
        let diff96 = 10.0 * (p96 / p48).log10();
        let diff192 = 10.0 * (p192 / p48).log10();
        assert!(
            diff96.abs() < 0.10,
            "96 kHz audible DF power drifted vs 48 kHz: {diff96:+.3} dB"
        );
        assert!(
            diff192.abs() < 0.10,
            "192 kHz audible DF power drifted vs 48 kHz: {diff192:+.3} dB"
        );
    }

    /// Brown & Duda (1998) Eq. (5) uses `w0 = 2c/a` (`f0 = 1247.8 Hz` for `a = 8.75 cm`),
    /// so at `f = f0` (`x = 1`) the head-shadow response equals `sqrt((alpha^2 + 1) / 2)`.
    #[test]
    fn brown_duda_matches_1998_eq5_cutoff_frequency() {
        let f0 = SPEED_OF_SOUND / (PI * HEAD_RADIUS_M);
        // Directly facing left ear (`az = +90°`, `diff = 0`): `alpha = 2.0`,
        // so at `f = f0` (`x = 1`), `|H| = sqrt((4 + 1) / 2) = sqrt(2.5)`.
        let ipsi = brown_duda_mag(f0, 90.0, 1.0, 1.0);
        let want_ipsi = 2.5_f64.sqrt();
        assert!(
            (ipsi - want_ipsi).abs() < 1e-12,
            "ipsilateral |H(f0)|={ipsi:.6}, expected sqrt(2.5)={want_ipsi:.6}"
        );
        // Contralateral shadow (`az = +90°`, right ear `-90°`, `diff = 180°`):
        // `alpha = 1.05 + 0.95 * cos(180/150 * pi) = 1.05 + 0.95 * cos(216°)`.
        let alpha_contra = 1.05 + 0.95 * (180.0 * PI / 150.0).cos();
        let contra = brown_duda_mag(f0, 90.0, -1.0, 1.0);
        let want_contra = ((alpha_contra * alpha_contra + 1.0) * 0.5).sqrt();
        assert!(
            (contra - want_contra).abs() < 1e-12,
            "contralateral |H(f0)|={contra:.6}, expected {want_contra:.6}"
        );
    }

    /// Stereo bypass must flag non-finite input samples, and multichannel mode must
    /// ignore input slots whose bits are cleared in `mask`.
    #[test]
    fn stereo_bypass_detects_nan_and_multichannel_respects_mask() {
        let mut st = BinauralPlus::new(MASK_STEREO, PlusCfg::default(), 48_000);
        let bad_l = [0.0, f64::NAN, 0.0, 0.0];
        let good_r = [0.0; 4];
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&bad_l);
        refs[SLOT_FR] = Some(&good_r);
        let mut out_l = [0.0; 4];
        let mut out_r = [0.0; 4];
        st.process(&refs, &mut out_l, &mut out_r, 4);
        assert!(
            !st.is_finite(),
            "stereo bypass failed to flag non-finite input"
        );

        // Mask has FL|FR|FC only; a loud impulse placed on BL must be ignored.
        let mask_3_0 = crate::dsp::CH_FL | crate::dsp::CH_FR | crate::dsp::CH_FC;
        let mut eng = BinauralPlus::new(mask_3_0, PlusCfg::from_profile(PlusProfile::Pure), 48_000);
        let mut bl = vec![0.0; 256];
        bl[0] = 1.0;
        let mut refs2: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs2[SLOT_BL] = Some(&bl);
        let mut ol = vec![0.0; 256];
        let mut or = vec![0.0; 256];
        eng.process(&refs2, &mut ol, &mut or, 256);
        assert!(
            ol.iter().chain(&or).all(|&v| v == 0.0),
            "unmasked BL slot leaked into output"
        );
    }

    /// End-to-end integration of the full acoustic-physics stack:
    /// Spherical-Hankel + Snowman torso + Order-7 MagLS + Hybrid 3D ControlRoom
    /// (2-run reciprocal 3D FDTD + ISM + BTM + PCG32 Ray Tracing) + Slepian DPSS tail
    /// + Paraunitary Scattering FDN/Velvet + Webster 1D Ear-Canal impedance.
    #[test]
    fn academic_nuke_hankel_room_slepian_externalization_integration() {
        let pure_cfg = PlusCfg::from_profile(PlusProfile::Pure);
        let mut nuke_cfg = PlusCfg::from_profile(PlusProfile::Studio);
        nuke_cfg.physics_engine = PhysicsEngine::HankelSnowmanMagLs;
        nuke_cfg.room_preset = RoomPreset::ControlRoom;
        nuke_cfg.distance_m = 2.0;
        nuke_cfg.use_slepian_tail = true;
        nuke_cfg.use_fdn_velvet = true;
        nuke_cfg.use_warped_decorr = true;
        nuke_cfg.ear_canal_mm = 27.0;
        nuke_cfg.atmosphere = crate::atmosphere::AtmosphereCfg {
            temp_c: 21.5,
            rh_pct: 45.0,
            co2_ppm: 420.0,
            pressure_kpa: 101.325,
        };
        nuke_cfg.validate().expect("nuke_cfg must validate");

        let bank_pure = KernelBank::synthesize(pure_cfg, 48_000);
        let bank_nuke = KernelBank::synthesize(nuke_cfg, 48_000);

        for ch in 0..SLOTS {
            assert!(bank_nuke.ir_l[ch].iter().all(|v| v.is_finite()));
            assert!(bank_nuke.ir_r[ch].iter().all(|v| v.is_finite()));
        }

        // Baumgartner–Majdak Bayesian Auditory Externalization must score the
        // 3D Hybrid ControlRoom higher than anechoic Pure on frontal speaker FL.
        let ext_pure = crate::cochlea::baumgartner_majdak_externalization(
            &bank_pure.ir_l[SLOT_FL],
            &bank_pure.ir_r[SLOT_FL],
            48_000,
        );
        let ext_nuke = crate::cochlea::baumgartner_majdak_externalization(
            &bank_nuke.ir_l[SLOT_FL],
            &bank_nuke.ir_r[SLOT_FL],
            48_000,
        );
        assert!(
            ext_nuke.score > ext_pure.score,
            "ControlRoom externalization ({:.3}) must exceed Pure ({:.3})",
            ext_nuke.score,
            ext_pure.score
        );
    }

    /// End-to-end integration of `--head-mesh` (3D FMM-BEM direct synthesis) +
    /// `--exact-ntt` (0-ULP Goldilocks prime-field NTT overlap-add streaming).
    #[test]
    fn head_mesh_fmm_bem_and_exact_ntt_end_to_end() {
        let mesh = crate::bem::TriangleMesh::procedural_head_with_pinnae(0.0875, 2);
        let mut cfg_fft = PlusCfg::from_profile(PlusProfile::Pure);
        cfg_fft.exact_ntt = false;
        let mut cfg_ntt = cfg_fft;
        cfg_ntt.exact_ntt = true;

        let mut eng_fft =
            BinauralPlus::new_with_assets(crate::dsp::MASK_5_1, cfg_fft, 48_000, None, Some(&mesh));
        let mut eng_ntt =
            BinauralPlus::new_with_assets(crate::dsp::MASK_5_1, cfg_ntt, 48_000, None, Some(&mesh));

        let n = 1024usize;
        let fl: Vec<f64> = (0..n)
            .map(|i| 0.3 * (2.0 * PI * 1000.0 * i as f64 / 48_000.0).sin())
            .collect();
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&fl);

        let mut l_fft = vec![0.0_f64; n];
        let mut r_fft = vec![0.0_f64; n];
        let mut l_ntt = vec![0.0_f64; n];
        let mut r_ntt = vec![0.0_f64; n];
        eng_fft.process(&refs, &mut l_fft, &mut r_fft, n);
        eng_ntt.process(&refs, &mut l_ntt, &mut r_ntt, n);
        let (tl_fft, tr_fft) = eng_fft.flush();
        let (tl_ntt, tr_ntt) = eng_ntt.flush();
        l_fft.extend_from_slice(&tl_fft);
        r_fft.extend_from_slice(&tr_fft);
        l_ntt.extend_from_slice(&tl_ntt);
        r_ntt.extend_from_slice(&tr_ntt);

        assert!(eng_fft.is_finite() && eng_ntt.is_finite());
        assert_eq!(l_fft.len(), l_ntt.len());
        let max_diff = l_fft
            .iter()
            .zip(&l_ntt)
            .chain(r_fft.iter().zip(&r_ntt))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 2e-6,
            "Goldilocks --exact-ntt diverged from f64 FFT on FMM-BEM head-mesh IR: max_diff={max_diff:e}"
        );
        // Ipsilateral left ear must be louder than contralateral right ear for FL (+30 deg)
        let rms_l = (l_ntt.iter().map(|v| v * v).sum::<f64>() / l_ntt.len() as f64).sqrt();
        let rms_r = (r_ntt.iter().map(|v| v * v).sum::<f64>() / r_ntt.len() as f64).sqrt();
        assert!(
            rms_l > rms_r * 1.1,
            "FMM-BEM head-mesh FL source should yield ipsilateral > contralateral energy (L={rms_l:.4}, R={rms_r:.4})"
        );
    }

    #[test]
    fn control_room_long_brir_and_exact_ntt_match_fft_without_wrap() {
        let mut cfg = PlusCfg::from_profile(PlusProfile::Studio);
        cfg.room_preset = RoomPreset::ControlRoom;
        cfg.physics_engine = PhysicsEngine::HankelSnowmanMagLs;

        let bank1 = KernelBank::synthesize_cached(cfg, 48_000, None, None);
        let bank2 = KernelBank::synthesize_cached(cfg, 48_000, None, None);
        assert!(
            Arc::ptr_eq(&bank1, &bank2),
            "SHA-256 BRIR cache must return identical Arc on repeat synthesis"
        );
        assert_eq!(
            bank1.ir_l[SLOT_FL].len(),
            N_IR_ROOM,
            "3D Hybrid ControlRoom must synthesize full 16384-tap (341.3 ms) UPOLA BRIR"
        );

        let mut cfg_ntt = cfg;
        cfg_ntt.exact_ntt = true;
        let mut eng_fft = BinauralPlus::new(crate::dsp::MASK_5_1, cfg, 48_000);
        let mut eng_fft_chunked = BinauralPlus::new(crate::dsp::MASK_5_1, cfg, 48_000);
        let mut eng_ntt = BinauralPlus::new(crate::dsp::MASK_5_1, cfg_ntt, 48_000);

        let n = 5000usize;
        let fl: Vec<f64> = (0..n)
            .map(|i| 0.45 * (2.0 * PI * 320.0 * i as f64 / 48_000.0).sin())
            .collect();
        let fr: Vec<f64> = (0..n)
            .map(|i| 0.35 * (2.0 * PI * 640.0 * i as f64 / 48_000.0).cos())
            .collect();
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&fl);
        refs[SLOT_FR] = Some(&fr);

        let mut l_fft = vec![0.0_f64; n];
        let mut r_fft = vec![0.0_f64; n];
        let mut l_ntt = vec![0.0_f64; n];
        let mut r_ntt = vec![0.0_f64; n];
        eng_fft.process(&refs, &mut l_fft, &mut r_fft, n);
        eng_ntt.process(&refs, &mut l_ntt, &mut r_ntt, n);
        let (tl_fft, tr_fft) = eng_fft.flush();
        let (tl_ntt, tr_ntt) = eng_ntt.flush();
        l_fft.extend_from_slice(&tl_fft);
        r_fft.extend_from_slice(&tr_fft);
        l_ntt.extend_from_slice(&tl_ntt);
        r_ntt.extend_from_slice(&tr_ntt);

        // Also feed odd 317-sample chunks to verify UPOLA arbitrary chunk-size invariance!
        let mut l_chk = vec![0.0_f64; n];
        let mut r_chk = vec![0.0_f64; n];
        let mut pos = 0usize;
        while pos < n {
            let step = (n - pos).min(317);
            let mut sub: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            sub[SLOT_FL] = Some(&fl[pos..pos + step]);
            sub[SLOT_FR] = Some(&fr[pos..pos + step]);
            eng_fft_chunked.process(
                &sub,
                &mut l_chk[pos..pos + step],
                &mut r_chk[pos..pos + step],
                step,
            );
            pos += step;
        }
        let (tl_chk, tr_chk) = eng_fft_chunked.flush();
        l_chk.extend_from_slice(&tl_chk);
        r_chk.extend_from_slice(&tr_chk);

        let chunk_diff = l_fft
            .iter()
            .zip(&l_chk)
            .chain(r_fft.iter().zip(&r_chk))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            chunk_diff < 1e-11,
            "FDL-UPOLA chunk-size invariance violated: max_diff={chunk_diff:e}"
        );

        let max_diff = l_fft
            .iter()
            .zip(&l_ntt)
            .chain(r_fft.iter().zip(&r_ntt))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 3e-6,
            "ControlRoom 16384-tap UPOLA --exact-ntt diverged from f64 FFT: max_diff={max_diff:e}"
        );
    }

    // ------------------------------------------------------------------
    // Live CUDA end-to-end CPU/GPU parity for the BinauralPlus UPOLA
    // convolution: FP64 Stockham FFT path and exact Goldilocks NTT path.
    // ------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_test_ready(test: &str) -> bool {
        if crate::blackwell::shared_cuda_runtime().is_some() {
            true
        } else {
            eprintln!("skipping {test}: no live CUDA GPU/runtime (CPU-only host)");
            false
        }
    }

    #[cfg(feature = "cuda")]
    fn parity_signal(n: usize, sample_rate: u32, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                0.37 * (2.0 * PI * (213.0 * t) + phase).sin()
                    + 0.19 * (2.0 * PI * 997.0 * t + 0.25).cos()
                    + 0.07 * (2.0 * PI * 4123.0 * t).sin()
            })
            .collect()
    }

    /// Build a renderer for the parity tests. `synthesize_cuda` selects whether the
    /// FIR kernel bank is synthesized with GPU FDTD/FMM-BEM offload; `process_cuda`
    /// selects the convolution-stage offload under test. For the NTT bit-exact test the
    /// bank is always synthesized on the CPU so that both engines quantize identical
    /// FIR taps (`use_cuda` also changes FDTD synthesis numerics by ~1e-17).
    #[cfg(feature = "cuda")]
    fn parity_render(
        mut cfg: PlusCfg,
        sample_rate: u32,
        synthesize_cuda: bool,
        process_cuda: bool,
        chunk: usize,
        fl: &[f64],
        fr: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        cfg.use_cuda = synthesize_cuda;
        let mut eng = BinauralPlus::new(crate::dsp::MASK_5_1, cfg, sample_rate);
        // Test-only knob: isolate the convolution-stage offload from the FIR-bank
        // synthesis path so the rendered kernels are bit-identical across engines.
        eng.cfg.use_cuda = process_cuda;

        let n = fl.len();
        let mut out_l = vec![0.0_f64; n];
        let mut out_r = vec![0.0_f64; n];
        let mut pos = 0usize;
        while pos < n {
            let step = (n - pos).min(chunk);
            let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            refs[SLOT_FL] = Some(&fl[pos..pos + step]);
            refs[SLOT_FR] = Some(&fr[pos..pos + step]);
            eng.process(
                &refs,
                &mut out_l[pos..pos + step],
                &mut out_r[pos..pos + step],
                step,
            );
            pos += step;
        }
        let (tl, tr) = eng.flush();
        out_l.extend_from_slice(&tl);
        out_r.extend_from_slice(&tr);
        assert!(eng.is_finite(), "renderer reported a non-finite state");
        (out_l, out_r)
    }

    #[cfg(feature = "cuda")]
    fn max_abs_pair_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "renderer output lengths must match");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Full BinauralPlus pipeline (kernel synthesis + single-partition OLA and FDL-UPOLA)
    /// rendered on CPU `f64` vs live CUDA Stockham FFT: max |d| <= 1e-11, no NaN/Inf.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_plus_stockham_pipeline_matches_cpu_across_rates_and_chunks() {
        if !cuda_test_ready("cuda_plus_stockham_pipeline_matches_cpu_across_rates_and_chunks") {
            return;
        }
        let anechoic = PlusCfg::from_profile(PlusProfile::Pure);
        let mut control_room = PlusCfg::from_profile(PlusProfile::Studio);
        control_room.room_preset = RoomPreset::ControlRoom;
        control_room.physics_engine = PhysicsEngine::HankelSnowmanMagLs;

        // (cfg, sample_rate, chunk, frames, synthesize_with_cuda).
        //
        // The anechoic cases construct the GPU engine with `cfg.use_cuda = true` exactly as a
        // user would; Anechoic/Pure has no FDTD or ray-traced room, so the FIR bank is
        // bit-identical and the only CPU/GPU difference is the convolution under test.
        // Room presets also offload FDTD/ray tracing during FIR synthesis, which changes the
        // synthesized BRIR (different summation order, measured up to 1.4e-4 in the rendered
        // output) — a property of the synthesis path, not of the UPOLA convolution. Those
        // cases therefore hold synthesis CPU-identical and toggle only the convolution stage.
        let cases = [
            (anechoic, 44_100u32, 1024usize, 2048usize, true),
            (anechoic, 48_000, 317, 2048, true),
            (anechoic, 48_000, 1, 384, true),
            (anechoic, 96_000, 2048, 4096, true),
            (control_room, 48_000, 317, 2048, false),
            (control_room, 48_000, 2048, 4096, false),
        ];

        for (cfg, rate, chunk, frames, synth_cuda) in cases {
            let fl = parity_signal(frames, rate, 0.0);
            let fr = parity_signal(frames, rate, 1.3);
            let (cl, cr) = parity_render(cfg, rate, false, false, chunk, &fl, &fr);
            let (gl, gr) = parity_render(cfg, rate, synth_cuda, true, chunk, &fl, &fr);
            let max_diff = max_abs_pair_diff(&cl, &gl).max(max_abs_pair_diff(&cr, &gr));
            let peak = cl.iter().chain(&cr).fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                cl.iter().chain(&cr).all(|v| v.is_finite())
                    && gl.iter().chain(&gr).all(|v| v.is_finite()),
                "rate={rate} chunk={chunk}: non-finite renderer output"
            );
            assert!(peak > 1e-6, "rate={rate} chunk={chunk}: signal is silent");
            assert!(
                max_diff <= 1e-11,
                "rate={rate} chunk={chunk} synth_cuda={synth_cuda}: CPU/GPU Stockham FFT \
                 pipeline diverged: max|d|={max_diff:e} (limit 1e-11)"
            );
            eprintln!(
                "cuda stockham parity: rate={rate} chunk={chunk} frames={frames} \
                 synth_cuda={synth_cuda} max|d|={max_diff:.3e} peak={peak:.3e}"
            );
        }
    }

    /// Full BinauralPlus pipeline through `--exact-ntt`: live CUDA Goldilocks NTT must be
    /// bit-identical (`to_bits`) to the CPU prime-field transform for every sample.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_plus_goldilocks_ntt_pipeline_is_bit_identical_to_cpu() {
        if !cuda_test_ready("cuda_plus_goldilocks_ntt_pipeline_is_bit_identical_to_cpu") {
            return;
        }
        let anechoic = PlusCfg::from_profile(PlusProfile::Pure);
        let mut control_room = PlusCfg::from_profile(PlusProfile::Studio);
        control_room.room_preset = RoomPreset::ControlRoom;
        control_room.physics_engine = PhysicsEngine::HankelSnowmanMagLs;

        // (cfg, sample_rate, chunk, frames)
        let cases = [
            (anechoic, 48_000u32, 1usize, 384usize),
            (anechoic, 48_000, 317, 1536),
            (anechoic, 96_000, 2048, 4096),
            (control_room, 48_000, 317, 2048),
            (control_room, 48_000, 2048, 4096),
        ];

        for (mut cfg, rate, chunk, frames) in cases {
            cfg.exact_ntt = true;
            let fl = parity_signal(frames, rate, 0.0);
            let fr = parity_signal(frames, rate, 1.3);
            let (cl, cr) = parity_render(cfg, rate, false, false, chunk, &fl, &fr);
            let (gl, gr) = parity_render(cfg, rate, false, true, chunk, &fl, &fr);
            assert_eq!(cl.len(), gl.len());
            assert_eq!(cr.len(), gr.len());
            let mut mismatches = 0usize;
            let mut max_diff = 0.0_f64;
            for (a, b) in cl.iter().zip(&gl).chain(cr.iter().zip(&gr)) {
                if a.to_bits() != b.to_bits() {
                    mismatches += 1;
                    max_diff = max_diff.max((a - b).abs());
                }
            }
            assert!(
                gl.iter().chain(&gr).all(|v| v.is_finite()),
                "rate={rate} chunk={chunk}: non-finite Goldilocks NTT output"
            );
            assert_eq!(
                mismatches, 0,
                "rate={rate} chunk={chunk}: GPU Goldilocks NTT must be bit-identical to CPU, \
                 {mismatches} sample(s) differ, max|d|={max_diff:e}"
            );
            eprintln!(
                "cuda goldilocks NTT parity: rate={rate} chunk={chunk} frames={frames} \
                 samples={} all bit-identical",
                cl.len() + cr.len()
            );
        }
    }

    /// Build a ControlRoom `--exact-ntt --cuda` engine whose device-resident UPOLA state
    /// exists, or `None` on a CPU-only host.
    #[cfg(feature = "cuda")]
    fn control_room_exact_ntt_cuda_engine() -> Option<BinauralPlus> {
        let mut cfg = PlusCfg::from_profile(PlusProfile::Studio);
        cfg.room_preset = RoomPreset::ControlRoom;
        cfg.physics_engine = PhysicsEngine::HankelSnowmanMagLs;
        cfg.exact_ntt = true;
        cfg.use_cuda = true;
        let eng = BinauralPlus::new(crate::dsp::MASK_5_1, cfg, 48_000);
        if eng.ntt_cuda.0.is_none() {
            eprintln!("skipping: device-resident UPOLA state unavailable");
            return None;
        }
        assert!(eng.n_parts > 1, "ControlRoom must use multi-partition UPOLA");
        Some(eng)
    }

    /// Aligned device hops must use exactly one inverse NTT launch + one synchronous D2H
    /// copy per hop (single-inverse UPOLA), not the old partition-0 + tail pair.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_upola_aligned_hop_uses_one_inverse_and_d2h() {
        let Some(mut eng) = control_room_exact_ntt_cuda_engine() else {
            return;
        };
        if !cuda_test_ready("cuda_upola_aligned_hop_uses_one_inverse_and_d2h") {
            return;
        }
        let state = eng.ntt_cuda.0.clone().expect("device UPOLA state");
        let m = eng.ola_hop;
        let fl = parity_signal(m, 48_000, 0.0);
        let fr = parity_signal(m, 48_000, 1.1);
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&fl);
        refs[SLOT_FR] = Some(&fr);
        let mut out_l = vec![0.0_f64; m];
        let mut out_r = vec![0.0_f64; m];
        UPOLA_INVERSE_CALLS.with(|c| c.set(0));
        let hops = 4usize;
        for hop in 0..hops {
            eng.process_hop_ntt_cuda(&state, &refs, 0, &mut out_l, &mut out_r, m)
                .unwrap_or_else(|e| panic!("aligned device UPOLA hop {hop}: {e}"));
        }
        let calls = UPOLA_INVERSE_CALLS.with(std::cell::Cell::get);
        assert_eq!(
            calls, hops,
            "aligned hops must issue one inverse NTT + one D2H each, got {calls} for {hops} hops"
        );
    }

    /// A device failure on the aligned hop must leave every host streaming field untouched:
    /// out buffers, overlap tails, delay-line head, block fill and the finite flag.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_upola_device_failure_leaves_host_state_untouched() {
        let Some(mut eng) = control_room_exact_ntt_cuda_engine() else {
            return;
        };
        if !cuda_test_ready("cuda_upola_device_failure_leaves_host_state_untouched") {
            return;
        }
        let state = eng.ntt_cuda.0.clone().expect("device UPOLA state");
        let m = eng.ola_hop;
        let fl = parity_signal(m, 48_000, 2.0);
        let fr = parity_signal(m, 48_000, 0.7);
        let mut refs: [Option<&[f64]>; SLOTS] = [None; SLOTS];
        refs[SLOT_FL] = Some(&fl);
        refs[SLOT_FR] = Some(&fr);
        let mut out_l = vec![0.0_f64; m];
        let mut out_r = vec![0.0_f64; m];
        // One successful hop so the overlap tail and delay line are non-trivial.
        eng.process_hop_ntt_cuda(&state, &refs, 0, &mut out_l, &mut out_r, m)
            .expect("warm-up device UPOLA hop");
        let ov_l = eng.ntt_ov_l.clone();
        let ov_r = eng.ntt_ov_r.clone();
        let head = eng.fdl_head;
        let fill = eng.cur_fill;
        let finite = eng.finite;

        for stage in [1u8, 2] {
            let sentinel_l = vec![123.25_f64; m];
            let sentinel_r = vec![-77.5_f64; m];
            out_l.copy_from_slice(&sentinel_l);
            out_r.copy_from_slice(&sentinel_r);
            UPOLA_FAIL_STAGE.with(|f| f.set(stage));
            let err = eng
                .process_hop_ntt_cuda(&state, &refs, 0, &mut out_l, &mut out_r, m)
                .expect_err("injected device failure must propagate");
            assert!(
                err.contains("injected"),
                "stage {stage}: unexpected error text: {err}"
            );
            assert_eq!(
                out_l, sentinel_l,
                "stage {stage}: out_l mutated before the device call failed"
            );
            assert_eq!(
                out_r, sentinel_r,
                "stage {stage}: out_r mutated before the device call failed"
            );
            assert_eq!(
                eng.ntt_ov_l, ov_l,
                "stage {stage}: left overlap tail mutated on device failure"
            );
            assert_eq!(
                eng.ntt_ov_r, ov_r,
                "stage {stage}: right overlap tail mutated on device failure"
            );
            assert_eq!(eng.fdl_head, head, "stage {stage}: fdl_head mutated");
            assert_eq!(eng.cur_fill, fill, "stage {stage}: cur_fill mutated");
            assert_eq!(eng.finite, finite, "stage {stage}: finite flag mutated");
        }
    }
}
