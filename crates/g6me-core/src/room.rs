//! Hybrid 3D Room Acoustics Engine:
//! - **Layer 1 (> 500 Hz)**: Allen & Berkley (JASA 1979) / Savioja & Svensson (JASA 2015)
//!   3D Image-Source Method (orders 1–3) with octave-band wall absorption & `AtmosphereCfg`
//!   molecular `O₂`/`N₂` air absorption, plus deterministic native Monte Carlo Acoustic
//!   Ray Tracing (orders 4–50, Lambertian / ISO-17497 scattering -> Order-7 HOA -> MagLS).
//! - **Layer 2 (Analytic Edge Diffraction)**: Svensson, Fred & Vanderkooy (JASA 1999)
//!   Biot–Tolstoy–Medwin (BTM) secondary-source edge diffraction + Kirsch & Ewert
//!   (IEEE/ACM TASLP 2023) universal diffraction filter for mixing-desk / baffle edges.
//! - **Layer 3 (< 500 Hz)**: Hamilton & Bilbao (IEEE/ACM TASLP 2017) 3D Wave-Equation
//!   FDTD PDE solver using **Acoustic Reciprocity** (only 2 runs: L ear & R ear, sampling
//!   all speaker positions simultaneously) with porous bass-trap boundary impedances.
//! - **Late Externalization**: Schlecht & Habets (IEEE/ACM TASLP 2017 & IEEE TSP 2020)
//!   Paraunitary Scattering Feedback Delay Network (exact unit-circle lossless eigenvalue
//!   bounds + `T_60(f)` absorption) + Välimäki, Lehtonen & Takanen (IEEE TASLP 2013)
//!   maximally sparse ternary `{-1, 0, +1}` Velvet Noise late diffusion.

use crate::atmosphere::AtmosphereCfg;
use crate::fft::{Complex, FftPlan};
use crate::hoa::{eval_sh_order7, MagLsDecoder, HOA_CH};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Virtual listening room acoustic preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoomPreset {
    /// Free-field anechoic (no hybrid room reflections; uses direct head/torso diffraction).
    #[default]
    Anechoic,
    /// ITU-R BS.1116 reference mastering control room (`6.0 × 5.0 × 3.0 m`, `RT60 ≈ 0.25 s`,
    /// porous bass traps, rear QRD diffuser, virtual mixing console edge).
    ControlRoom,
    /// Large theatrical reference mixing stage (`12.0 × 9.5 × 5.2 m`, `RT60 ≈ 0.42 s`).
    TheaterStage,
}

impl RoomPreset {
    pub const ALL: [RoomPreset; 3] = [
        RoomPreset::Anechoic,
        RoomPreset::ControlRoom,
        RoomPreset::TheaterStage,
    ];

    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "anechoic" | "dry" | "off" | "none" => Ok(Self::Anechoic),
            "control-room" | "control" | "studio" | "bs1116" => Ok(Self::ControlRoom),
            "theater" | "theatre" | "theater-stage" | "cinema" | "stage" => Ok(Self::TheaterStage),
            other => Err(format!(
                "unknown room preset `{other}` (expected: anechoic, control-room, theater)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anechoic => "anechoic",
            Self::ControlRoom => "control-room",
            Self::TheaterStage => "theater",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Anechoic => "Anechoic",
            Self::ControlRoom => "Control Room (BS.1116)",
            Self::TheaterStage => "Theater Stage",
        }
    }
}

/// Tier-A shoebox preset library (`docs/ROOM_PRESETS_RESEARCH.md` §4).
///
/// The three legacy [`RoomPreset`] rooms are included so one enum addresses the
/// whole library; the five additional rooms are the measured/shoebox halls and
/// the standard design rooms. Every preset fits the engine's six-plane shoebox
/// geometry and the `1.5..=60 m` dimension validator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShoeboxPreset {
    /// ITU-R BS.1116 mastering control room (legacy [`RoomPreset::ControlRoom`]).
    ControlRoom,
    /// Theatrical reference mixing stage (legacy [`RoomPreset::TheaterStage`]).
    TheaterStage,
    /// Small studio live room, `8.0 × 6.0 × 4.0 m`, design RT `0.45 s`.
    StudioLiveRoom,
    /// Lecture theatre, `15.0 × 10.0 × 6.0 m`, design RT `0.8 s`.
    LectureTheatre,
    /// Cinema dubbing stage, `20.0 × 12.0 × 8.0 m`, design RT `0.5 s`.
    Cinema,
    /// Vienna Musikverein Grosser Saal, shoebox `48.8 × 19.1 × 17.75 m`.
    Musikverein,
    /// Boston Symphony Hall, shoebox `38.1 × 22.9 × 18.6 m`.
    BostonSymphonyHall,
    /// Amsterdam Concertgebouw Grote Zaal, shoebox `44 × 28 × 17 m`.
    Concertgebouw,
}

impl ShoeboxPreset {
    pub const ALL: [ShoeboxPreset; 8] = [
        Self::ControlRoom,
        Self::TheaterStage,
        Self::StudioLiveRoom,
        Self::LectureTheatre,
        Self::Cinema,
        Self::Musikverein,
        Self::BostonSymphonyHall,
        Self::Concertgebouw,
    ];

    pub const HALLS: [ShoeboxPreset; 3] = [
        Self::Musikverein,
        Self::BostonSymphonyHall,
        Self::Concertgebouw,
    ];

    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "control-room" | "control" | "bs1116" => Ok(Self::ControlRoom),
            "theater" | "theatre" | "theater-stage" | "stage" => Ok(Self::TheaterStage),
            "studio" | "studio-live-room" | "live-room" => Ok(Self::StudioLiveRoom),
            "lecture" | "lecture-theatre" | "lecture-theater" => Ok(Self::LectureTheatre),
            "cinema" | "cinema-dubbing-stage" | "dubbing-stage" => Ok(Self::Cinema),
            "musikverein" | "vienna" | "grosser-saal" => Ok(Self::Musikverein),
            "boston" | "symphony-hall" | "boston-symphony-hall" => Ok(Self::BostonSymphonyHall),
            "concertgebouw" | "amsterdam" | "grote-zaal" => Ok(Self::Concertgebouw),
            other => Err(format!(
                "unknown shoebox preset `{other}` (expected: control-room, theater, studio, \
                 lecture, cinema, musikverein, boston, concertgebouw)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ControlRoom => "control-room",
            Self::TheaterStage => "theater",
            Self::StudioLiveRoom => "studio",
            Self::LectureTheatre => "lecture",
            Self::Cinema => "cinema",
            Self::Musikverein => "musikverein",
            Self::BostonSymphonyHall => "boston",
            Self::Concertgebouw => "concertgebouw",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ControlRoom => "Control Room (BS.1116)",
            Self::TheaterStage => "Theater Stage",
            Self::StudioLiveRoom => "Studio Live Room",
            Self::LectureTheatre => "Lecture Theatre",
            Self::Cinema => "Cinema Dubbing Stage",
            Self::Musikverein => "Vienna Musikverein, Grosser Saal",
            Self::BostonSymphonyHall => "Boston Symphony Hall",
            Self::Concertgebouw => "Amsterdam Concertgebouw, Grote Zaal",
        }
    }

    /// The legacy three-way preset, when one exists.
    pub fn legacy_room_preset(self) -> Option<RoomPreset> {
        match self {
            Self::ControlRoom => Some(RoomPreset::ControlRoom),
            Self::TheaterStage => Some(RoomPreset::TheaterStage),
            _ => None,
        }
    }

    /// Bridge from the legacy UI/CLI enum.
    pub fn from_room_preset(preset: RoomPreset) -> Option<Self> {
        match preset {
            RoomPreset::Anechoic => None,
            RoomPreset::ControlRoom => Some(Self::ControlRoom),
            RoomPreset::TheaterStage => Some(Self::TheaterStage),
        }
    }
}

/// Provenance of a preset's per-band `RT60` target curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Rt60Model {
    /// No per-band curve; only a mid-frequency target (or standard allowance)
    /// is asserted. Used by the legacy control-room and theater-stage presets,
    /// whose bass is owned by the FDTD boundary model.
    #[default]
    MidBandOnly,
    /// Per-band statistical (Eyring) design curve computed from the preset's
    /// material table; the mid-band value is calibrated to the measured or
    /// standard target. Bands below `geometric_bands_from_hz` are `None`
    /// because the FDTD wave solver owns them (research doc §2 method table).
    EyringDesign,
}

/// Documented acoustic targets and preset metadata (research doc §4/§6).
///
/// Values in `rt60_bands_s`, `mid_rt60_s`, `mid_rt60_alt_s`, `mid_edt_s`,
/// `strength_g_db`, `clarity_c80_db`, `bass_lf`, `one_minus_iacc` and
/// `itdg_ms` are transcribed from `docs/ROOM_PRESETS_RESEARCH.md` (Beranek
/// 2004 via akuTEK Table 2) where the doc gives them; the doc is the source of
/// truth and the preset tests compare against those exact numbers.
///
/// `rt60_bands_s[i] == None` means the band is not a geometric target: the
/// FDTD wave model owns it below `geometric_bands_from_hz`, and a per-band
/// statistical value would be an unmeasured extrapolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoomAcousticTargets {
    /// Band centres for every per-band array in this struct.
    pub bands_hz: [f64; ROOM_BAND_COUNT],
    /// Per-band geometric RT60 design target (s); `None` below the FDTD handoff.
    pub rt60_bands_s: [Option<f64>; ROOM_BAND_COUNT],
    /// Per-band measured EDT (s), where a source gives it. The research doc
    /// tabulates only the mid-frequency EDT for the three halls.
    pub edt_bands_s: [Option<f64>; ROOM_BAND_COUNT],
    /// Occupied mid-frequency reverberation time `T` (s), the measured target.
    pub mid_rt60_s: f64,
    /// Documented design range `[lo, hi]` (s) when the source gives a range
    /// rather than a measured single value (studio/lecture/cinema).
    pub mid_rt60_range_s: Option<[f64; 2]>,
    /// Alternate documented mid-band RT (e.g. Concertgebouw `2.2 s` occupied
    /// according to the hall, vs Beranek's `2.0 s`).
    pub mid_rt60_alt_s: Option<f64>,
    /// Mid-frequency early decay time `EDT` (s).
    pub mid_edt_s: Option<f64>,
    /// Strength `G` (dB).
    pub strength_g_db: Option<f64>,
    /// Clarity `C80` (dB) at mid frequencies.
    pub clarity_c80_db: Option<f64>,
    /// Beranek bass ratio / low-frequency factor `LF`.
    pub bass_lf: Option<f64>,
    /// Binaural inter-aural cross-correlation complement `1 - IACC`.
    pub one_minus_iacc: Option<f64>,
    /// Initial time-delay gap (ms).
    pub itdg_ms: Option<f64>,
    /// Acoustic volume (m³) quoted by the source. For the halls this differs
    /// from `L × W × H`; it is metadata, not the geometric volume.
    pub acoustic_volume_m3: Option<f64>,
    /// Lowest octave band whose target is the geometric (ray/ISM) model. The
    /// FDTD wave solver owns the bands below it.
    pub geometric_bands_from_hz: f64,
    /// Explanation of what the per-band curve is (and is not).
    pub model: Rt60Model,
    /// Reference source position `(x, y, z)` for the ISO 3382 geometric checks
    /// (ITDG). Chosen to reproduce the documented ITDG in the shoebox, not a
    /// surveyed stage coordinate.
    pub reference_source_m: Option<[f64; 3]>,
    /// Eyring mid-band fit residual `rt60(1 kHz) - mid_rt60_s` (s), when the
    /// material table was fitted to the mid target (research doc §5 B4).
    pub fit_residual_s: Option<f64>,
    /// Data provenance string.
    pub source: String,
}

impl Default for RoomAcousticTargets {
    fn default() -> Self {
        Self {
            bands_hz: ROOM_BANDS_HZ,
            rt60_bands_s: [None; ROOM_BAND_COUNT],
            edt_bands_s: [None; ROOM_BAND_COUNT],
            mid_rt60_s: 0.25,
            mid_rt60_range_s: None,
            mid_rt60_alt_s: None,
            mid_edt_s: None,
            strength_g_db: None,
            clarity_c80_db: None,
            bass_lf: None,
            one_minus_iacc: None,
            itdg_ms: None,
            acoustic_volume_m3: None,
            geometric_bands_from_hz: 500.0,
            model: Rt60Model::MidBandOnly,
            reference_source_m: None,
            fit_residual_s: None,
            source: String::new(),
        }
    }
}

/// Physical simulation fidelity for the 3D Hybrid Room solver (FDTD + Ray Tracing + UPOLA).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoomQuality {
    /// Fast interactive/CI fidelity: 65k-voxel FDTD (500 Hz at 6 PPW in the 6x5x3 m
    /// control room, 275 Hz in the 12x9.5x5.2 m theater) + 2,048 parallel native-tracer rays.
    #[default]
    Preview,
    /// Full mastering production fidelity: 260k-voxel FDTD (670 Hz at 6 PPW in the
    /// control room, 437 Hz in the theater) + 50,000 parallel deterministic native
    /// Monte Carlo rays across all CPU cores. The IR length is set by the room preset's
    /// RT60, not by the quality tier.
    Production,
}

impl RoomQuality {
    pub const ALL: [RoomQuality; 2] = [RoomQuality::Preview, RoomQuality::Production];

    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "preview" | "fast" | "draft" => Ok(Self::Preview),
            "production" | "prod" | "full" | "high" | "reference" => Ok(Self::Production),
            other => Err(format!(
                "unknown room quality `{other}` (expected: preview, production)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preview => "preview",
            Self::Production => "production",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Preview => "Preview (Fast, 2k rays / 65k-voxel FDTD)",
            Self::Production => "Production (50k rays / 260k-voxel FDTD)",
        }
    }
}

/// Deterministic PCG64 (128-bit LCG, XSL-RR 128/64 output) PRNG for bit-exact
/// reproducible Velvet Noise sequences and for the native CPU stochastic ray tracer.
/// (The test-only `RayPcg32` oracle separately mirrors the PCG stream in
/// `raytrace_hoa.cu`; it is not used by the production CPU path.)
#[derive(Debug, Clone, Copy)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

impl Pcg64 {
    const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: ((stream as u128) << 1) | 1,
        };
        let _ = rng.next_u64();
        rng.state = rng.state.wrapping_add(seed as u128);
        let _ = rng.next_u64();
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let xorshifted = ((old >> 64) ^ old) as u64;
        let rot = (old >> 122) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const INV_2_53: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * INV_2_53
    }
}

/// Finite diffracting wedge / mixing-desk edge for BTM analytic edge diffraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffractionEdge {
    /// Start endpoint `(x, y, z)` in room coordinates (m).
    pub p0: [f64; 3],
    /// End endpoint `(x, y, z)` in room coordinates (m).
    pub p1: [f64; 3],
    /// Exterior wedge angle `theta_w` in radians (e.g. `1.5 * PI` for a 90° desk edge).
    pub wedge_angle_rad: f64,
    /// Optional unit tangent direction lying along wedge face 0 (perpendicular to edge `p1 - p0`).
    /// When omitted, defaults to the horizontal normal perpendicular to the edge (e.g. `+x` tabletop).
    #[serde(default)]
    pub face0_dir: Option<[f64; 3]>,
}

/// Number of octave bands in the ten-band room model (`31.5 Hz` .. `16 kHz`).
pub const ROOM_BAND_COUNT: usize = 10;
/// ISO 266 preferred octave-band centres used by the ten-band room model: the
/// OpenAIR reverberation-time band set (`31.25 Hz` is represented by the ISO
/// octave centre `31.5 Hz`).
pub const ROOM_BANDS_HZ: [f64; ROOM_BAND_COUNT] = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
];
/// Index of the `1 kHz` octave band. The pre-ten-band single-band room paths
/// (`absorption[3]`) evaluate exactly this band.
pub const ROOM_BAND_MID: usize = 5;

/// Log-frequency (octave) interpolation weight of `band` at `f_hz`.
///
/// This is the piecewise-linear partition basis on the ten-band
/// `log2(f)` grid (the same grid [`band_value_at`] interpolates on): inside the
/// interval between two adjacent band centres, the two bracketing weights sum
/// to exactly `1.0`, and the two edge bands hold full weight outside the covered
/// range. Normalising each interval by its actual width matters because the ISO
/// preferred centres are nominal octaves: `31.5 → 63` is exactly one octave but
/// `63 → 125` is `0.9887` octaves, so unnormalised triangles would sum to
/// `1.007` near 63 Hz and boost that region of the late-tail spectrum.
pub fn octave_band_weight(f_hz: f64, band: usize) -> f64 {
    if band >= ROOM_BAND_COUNT || !f_hz.is_finite() || f_hz <= 0.0 {
        return 0.0;
    }
    if f_hz <= ROOM_BANDS_HZ[0] {
        return if band == 0 { 1.0 } else { 0.0 };
    }
    if f_hz >= ROOM_BANDS_HZ[ROOM_BAND_COUNT - 1] {
        return if band == ROOM_BAND_COUNT - 1 {
            1.0
        } else {
            0.0
        };
    }
    let mut idx = 0usize;
    while idx + 1 < ROOM_BAND_COUNT && f_hz > ROOM_BANDS_HZ[idx + 1] {
        idx += 1;
    }
    let t =
        (f_hz / ROOM_BANDS_HZ[idx]).log2() / (ROOM_BANDS_HZ[idx + 1] / ROOM_BANDS_HZ[idx]).log2();
    if band == idx {
        1.0 - t
    } else if band == idx + 1 {
        t
    } else {
        0.0
    }
}

/// Monotone log-frequency interpolation of a ten-band table.
///
/// Uses Fritsch–Carlson PCHIP in `log10(f)` (see [`crate::pchip`]) rather than
/// piecewise-linear interpolation on the raw `log2(f)` axis. The nominal ISO
/// centres are only *approximately* octave-spaced: `63 → 125 Hz` spans
/// `0.9887` octaves, so an `log2(f)`-un-normalised linear segment approaches
/// `y(63) + 0.9887·(y(125) − y(63))` from the left and restarts at `y(125)` on
/// the right — a step of ~1.1 % of the 63→125 rise at 125 Hz. PCHIP is C¹ and,
/// being shape-preserving, keeps absorption inside `[0.01, 0.99]` and
/// scattering inside `[0, 1]` without overshoot.
#[inline]
fn band_value_at(values: &[f64; ROOM_BAND_COUNT], f_hz: f64) -> f64 {
    crate::pchip::pchip_eval_fixed(&ROOM_BANDS_HZ, values, f_hz)
}

/// Expand a legacy six-band `[125, 250, 500, 1k, 2k, 4k]` table to ten bands by
/// log-frequency interpolation, holding the 125 Hz value at 31.5/63 Hz and the
/// 4 kHz value at 8/16 kHz. This is the backward-compatibility mapping for
/// `--room-json` files written before the ten-band model.
pub fn bands_six_to_ten(six: &[f64; 6]) -> [f64; ROOM_BAND_COUNT] {
    const SIX_BANDS: [f64; 6] = [125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0];
    std::array::from_fn(|i| {
        let f = ROOM_BANDS_HZ[i];
        if f <= SIX_BANDS[0] {
            six[0]
        } else if f >= SIX_BANDS[5] {
            six[5]
        } else {
            let mut idx = 0usize;
            while idx + 1 < 6 && f > SIX_BANDS[idx + 1] {
                idx += 1;
            }
            let t = (f / SIX_BANDS[idx]).log2();
            six[idx] + t * (six[idx + 1] - six[idx])
        }
    })
}

/// Ten-band octave wall acoustic material (`31.5 Hz` .. `16 kHz`).
///
/// `absorption` is the energy absorption coefficient per octave band and
/// `scattering` the ISO-17497 diffuse-scattering coefficient per band. Both are
/// interpolated in `log2(f)` when a continuously frequency-dependent value is
/// required (image-source taps, per-bin synthesis weights).
///
/// Deserialization accepts the pre-B0 six-band `absorption` array and scalar
/// `scattering` so existing room JSON keeps loading unchanged.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct WallMaterial {
    /// Energy absorption coefficients `alpha in [0.01, 0.99]` at
    /// `[31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k] Hz`.
    pub absorption: [f64; ROOM_BAND_COUNT],
    /// ISO-17497 diffuse scattering coefficient `s in [0.0, 1.0]` per band.
    pub scattering: [f64; ROOM_BAND_COUNT],
}

/// Untagged band-table decoder: ten bands (current) or six bands (legacy).
#[derive(Deserialize)]
#[serde(untagged)]
enum BandArrayDe {
    Ten([f64; ROOM_BAND_COUNT]),
    Six([f64; 6]),
}

impl BandArrayDe {
    fn into_ten(self) -> [f64; ROOM_BAND_COUNT] {
        match self {
            Self::Ten(v) => v,
            Self::Six(v) => bands_six_to_ten(&v),
        }
    }
}

/// Untagged scattering decoder: ten bands, six bands, or a legacy scalar.
#[derive(Deserialize)]
#[serde(untagged)]
enum BandSpectrumDe {
    Ten([f64; ROOM_BAND_COUNT]),
    Six([f64; 6]),
    One(f64),
}

impl BandSpectrumDe {
    fn into_ten(self) -> [f64; ROOM_BAND_COUNT] {
        match self {
            Self::Ten(v) => v,
            Self::Six(v) => bands_six_to_ten(&v),
            Self::One(v) => [v; ROOM_BAND_COUNT],
        }
    }
}

fn de_absorption<'de, D>(de: D) -> std::result::Result<[f64; ROOM_BAND_COUNT], D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    Ok(BandArrayDe::deserialize(de)?.into_ten())
}

fn de_scattering<'de, D>(de: D) -> std::result::Result<[f64; ROOM_BAND_COUNT], D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    Ok(BandSpectrumDe::deserialize(de)?.into_ten())
}

#[derive(Deserialize)]
struct WallMaterialDe {
    #[serde(deserialize_with = "de_absorption")]
    absorption: [f64; ROOM_BAND_COUNT],
    #[serde(default, deserialize_with = "de_scattering")]
    scattering: [f64; ROOM_BAND_COUNT],
}

impl<'de> Deserialize<'de> for WallMaterial {
    fn deserialize<D>(de: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = WallMaterialDe::deserialize(de)?;
        Ok(Self {
            absorption: raw.absorption,
            scattering: raw.scattering,
        })
    }
}

impl WallMaterial {
    /// Uniform material across all ten bands (used by the anechoic preset).
    pub const fn uniform(absorption: f64, scattering: f64) -> Self {
        Self {
            absorption: [absorption; ROOM_BAND_COUNT],
            scattering: [scattering; ROOM_BAND_COUNT],
        }
    }

    /// Clamped band absorption `alpha_b in [0.01, 0.99]`.
    #[inline]
    pub fn alpha(&self, band: usize) -> f64 {
        self.absorption[band.min(ROOM_BAND_COUNT - 1)].clamp(0.01, 0.99)
    }

    /// Band energy reflection coefficient `1 - alpha_b`.
    #[inline]
    pub fn refl_energy(&self, band: usize) -> f64 {
        1.0 - self.alpha(band)
    }

    /// Clamped band ISO-17497 scattering coefficient.
    #[inline]
    pub fn scatter(&self, band: usize) -> f64 {
        self.scattering[band.min(ROOM_BAND_COUNT - 1)].clamp(0.0, 1.0)
    }

    /// Amplitude reflection coefficient `sqrt(1 - alpha(f))` log-interpolated
    /// between the ten octave-band absorption values.
    pub fn refl_amp(&self, f_hz: f64) -> f64 {
        (1.0 - band_value_at(&self.absorption, f_hz).clamp(0.01, 0.99)).sqrt()
    }

    /// Diffuse-scattering coefficient interpolated to `f_hz`.
    pub fn scatter_at(&self, f_hz: f64) -> f64 {
        band_value_at(&self.scattering, f_hz).clamp(0.0, 1.0)
    }
}

/// Full 3D room geometry & acoustic material definition (loadable via `--room-json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomGeometry {
    pub name: String,
    /// Room shoebox dimensions `[Lx, Ly, Lz]` in meters (`x` = front-back, `y` = left-right, `z` = height).
    pub dims_m: [f64; 3],
    /// Listener head center `[x, y, z]` in room coordinates (m).
    pub listener_m: [f64; 3],
    /// Wall materials in order: `[front(+x), back(-x), left(+y), right(-y), floor(-z), ceiling(+z)]`.
    pub walls: [WallMaterial; 6],
    /// Porous bass-trap normalized boundary admittance `beta = (1-R)/(1+R)` for `< 500 Hz` FDTD.
    pub bass_trap_admittance: f64,
    /// Target reverberation time `RT60` (seconds) at mid frequencies.
    pub target_rt60_s: f64,
    /// Documented per-band / ISO 3382 targets and provenance (research doc §4/§6).
    #[serde(default)]
    pub targets: Option<RoomAcousticTargets>,
    /// Optional finite diffracting edges (mixing console front lip, monitor baffle).
    #[serde(default)]
    pub edges: Vec<DiffractionEdge>,
}

// ---------------------------------------------------------------------------
// Tier-A shoebox material library (design values; research doc §4).
//
// These are *design* octave-band tables, not measured material samples. The
// per-band RT target of every calibrated preset is the Eyring value these
// tables produce, with the mid-band value gated against the measured/standard
// target in the tests. The eight tables cover the surface classes used by the
// Tier-A rooms: plaster, wood panelling, coffered ceiling, occupied seats,
// porous panel, QRD diffuser, wood floor, ceiling cloud, carpet and cinema
// fabric.
// ---------------------------------------------------------------------------

/// Rigid plaster / masonry wall with mild low-frequency panel absorption.
fn mat_plaster() -> WallMaterial {
    WallMaterial {
        absorption: [0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.06, 0.05, 0.05, 0.05],
        scattering: [0.05, 0.05, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08],
    }
}

/// Wood-panelled hall wall.
fn mat_wood_panel() -> WallMaterial {
    WallMaterial {
        absorption: [0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.14],
        scattering: [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.25, 0.24],
    }
}

/// Coffered wood ceiling.
fn mat_coffered_ceiling() -> WallMaterial {
    WallMaterial {
        absorption: [0.15, 0.14, 0.12, 0.10, 0.10, 0.12, 0.14, 0.16, 0.18, 0.18],
        scattering: [0.15, 0.18, 0.22, 0.28, 0.32, 0.35, 0.38, 0.40, 0.40, 0.38],
    }
}

/// Fully occupied upholstered seating (audience), per m² of floor.
fn mat_occupied_seats() -> WallMaterial {
    WallMaterial {
        absorption: [0.65, 0.68, 0.72, 0.80, 0.85, 0.88, 0.90, 0.90, 0.88, 0.84],
        scattering: [0.10, 0.12, 0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.32, 0.30],
    }
}

/// Porous broadband studio absorber (front/side walls).
fn mat_acoustic_panel() -> WallMaterial {
    WallMaterial {
        absorption: [0.15, 0.25, 0.40, 0.45, 0.45, 0.45, 0.44, 0.42, 0.38, 0.34],
        scattering: [0.15, 0.20, 0.28, 0.35, 0.40, 0.42, 0.42, 0.40, 0.38, 0.36],
    }
}

/// QRD skyline diffuser (back wall): high ISO-17497 scattering at mid/high bands.
fn mat_qrd_diffuser() -> WallMaterial {
    WallMaterial {
        absorption: [0.05, 0.06, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.14],
        scattering: [0.20, 0.30, 0.55, 0.75, 0.85, 0.88, 0.90, 0.90, 0.88, 0.84],
    }
}

/// Wood floor (empty).
fn mat_wood_floor() -> WallMaterial {
    WallMaterial {
        absorption: [0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.10],
        scattering: [0.05, 0.06, 0.08, 0.12, 0.15, 0.18, 0.20, 0.20, 0.20, 0.18],
    }
}

/// Suspended ceiling cloud.
fn mat_ceiling_cloud() -> WallMaterial {
    WallMaterial {
        absorption: [0.12, 0.22, 0.36, 0.40, 0.42, 0.42, 0.40, 0.38, 0.35, 0.32],
        scattering: [0.10, 0.15, 0.25, 0.32, 0.38, 0.40, 0.40, 0.38, 0.36, 0.34],
    }
}

/// Heavy carpet.
fn mat_carpet() -> WallMaterial {
    WallMaterial {
        absorption: [0.05, 0.08, 0.14, 0.20, 0.26, 0.28, 0.30, 0.30, 0.28, 0.25],
        scattering: [0.05, 0.06, 0.08, 0.12, 0.18, 0.22, 0.25, 0.25, 0.24, 0.22],
    }
}

/// Fabric-covered cinema wall / ceiling treatment. The 1 kHz value is fitted to
/// the Tier-A cinema design target (`0.50 s` in `20 × 12 × 8 m`) together with
/// the carpeted floor; the surrounding bands keep a smooth fabric curve.
fn mat_cinema_fabric() -> WallMaterial {
    WallMaterial {
        absorption: [0.12, 0.18, 0.30, 0.42, 0.48, 0.515, 0.50, 0.46, 0.42, 0.38],
        scattering: [0.10, 0.14, 0.20, 0.28, 0.32, 0.35, 0.35, 0.34, 0.32, 0.30],
    }
}

impl RoomGeometry {
    pub fn from_preset(preset: RoomPreset) -> Self {
        match preset {
            RoomPreset::Anechoic => Self {
                name: "Anechoic Chamber".to_string(),
                dims_m: [6.0, 5.0, 3.0],
                listener_m: [3.0, 2.5, 1.3],
                walls: [WallMaterial::uniform(0.99, 0.0); 6],
                bass_trap_admittance: 0.95,
                target_rt60_s: 0.02,
                targets: None,
                edges: Vec::new(),
            },
            RoomPreset::ControlRoom => {
                // ITU-R BS.1116-3 mastering control room: porous broadband absorbers on front/sides,
                // QRD skyline diffuser on rear wall, damped floor/ceiling, mixing console edge.
                // The 1 kHz column (index 5) and the mid-band scattering values are unchanged
                // from the pre-ten-band preset, so existing room gates keep their calibrated
                // material point.
                let front = WallMaterial {
                    absorption: [0.30, 0.36, 0.42, 0.55, 0.68, 0.74, 0.78, 0.82, 0.84, 0.86],
                    scattering: [0.18, 0.20, 0.22, 0.24, 0.25, 0.25, 0.26, 0.28, 0.28, 0.27],
                };
                let back_qrd = WallMaterial {
                    absorption: [0.28, 0.31, 0.35, 0.38, 0.42, 0.45, 0.50, 0.56, 0.58, 0.60],
                    scattering: [0.20, 0.30, 0.50, 0.70, 0.78, 0.78, 0.80, 0.82, 0.80, 0.76],
                };
                let side = WallMaterial {
                    absorption: [0.26, 0.32, 0.38, 0.48, 0.58, 0.65, 0.70, 0.75, 0.77, 0.79],
                    scattering: [0.22, 0.26, 0.30, 0.33, 0.35, 0.35, 0.36, 0.38, 0.38, 0.37],
                };
                let floor = WallMaterial {
                    absorption: [0.08, 0.10, 0.12, 0.16, 0.24, 0.34, 0.44, 0.52, 0.54, 0.56],
                    scattering: [0.08, 0.10, 0.12, 0.14, 0.15, 0.15, 0.16, 0.18, 0.18, 0.17],
                };
                let ceiling = WallMaterial {
                    absorption: [0.28, 0.34, 0.40, 0.55, 0.70, 0.76, 0.80, 0.84, 0.86, 0.88],
                    scattering: [0.20, 0.24, 0.27, 0.29, 0.30, 0.30, 0.31, 0.33, 0.33, 0.32],
                };
                Self {
                    name: "ITU-R BS.1116 Control Room".to_string(),
                    dims_m: [6.0, 5.0, 3.0],
                    listener_m: [2.6, 2.5, 1.25],
                    walls: [front, back_qrd, side, side, floor, ceiling],
                    bass_trap_admittance: 0.16,
                    target_rt60_s: 0.25,
                    targets: Some(standard_mid_band_targets(
                        0.25,
                        90.0,
                        Some([0.20, 0.40]),
                        "ITU-R BS.1116-3 / EBU Tech 3276: 0.2–0.4 s, \
                         T = 0.25·(V/100)^(1/3) ≈ 0.241 s averaged 200 Hz–4 kHz, \
                         +0.3 s allowed at 63 Hz",
                    )),
                    edges: vec![DiffractionEdge {
                        // Virtual mixing desk front edge 0.75 m in front of listener, 0.85 m high.
                        p0: [3.35, 1.65, 0.85],
                        p1: [3.35, 3.35, 0.85],
                        wedge_angle_rad: 1.5 * PI,
                        face0_dir: Some([1.0, 0.0, 0.0]),
                    }],
                }
            }
            RoomPreset::TheaterStage => {
                // 1 kHz column (index 5) and mid-band scattering unchanged from the
                // pre-ten-band preset.
                let wall = WallMaterial {
                    absorption: [0.16, 0.20, 0.24, 0.32, 0.40, 0.46, 0.52, 0.60, 0.62, 0.64],
                    scattering: [0.25, 0.32, 0.38, 0.42, 0.45, 0.45, 0.47, 0.50, 0.50, 0.48],
                };
                let back = WallMaterial {
                    absorption: [0.20, 0.25, 0.30, 0.38, 0.46, 0.52, 0.58, 0.65, 0.67, 0.69],
                    scattering: [0.35, 0.45, 0.55, 0.62, 0.65, 0.65, 0.67, 0.70, 0.70, 0.68],
                };
                let floor = WallMaterial {
                    absorption: [0.10, 0.14, 0.18, 0.28, 0.42, 0.55, 0.65, 0.72, 0.74, 0.76],
                    scattering: [0.30, 0.38, 0.44, 0.48, 0.50, 0.50, 0.52, 0.55, 0.55, 0.52],
                };
                Self {
                    name: "Reference Theater Mixing Stage".to_string(),
                    dims_m: [12.0, 9.5, 5.2],
                    listener_m: [5.2, 4.75, 1.45],
                    walls: [wall, back, wall, wall, floor, wall],
                    bass_trap_admittance: 0.12,
                    target_rt60_s: 0.42,
                    targets: Some(standard_mid_band_targets(
                        0.42,
                        592.8,
                        None,
                        "Reference theatrical mixing stage design target (research doc §4); \
                         no public measured IR set, marked design target",
                    )),
                    edges: vec![DiffractionEdge {
                        p0: [6.2, 3.4, 0.90],
                        p1: [6.2, 6.1, 0.90],
                        wedge_angle_rad: 1.5 * PI,
                        face0_dir: Some([1.0, 0.0, 0.0]),
                    }],
                }
            }
        }
    }

    /// Build one of the eight Tier-A shoebox presets.
    pub fn from_shoebox_preset(preset: ShoeboxPreset) -> Self {
        match preset {
            ShoeboxPreset::ControlRoom => Self::from_preset(RoomPreset::ControlRoom),
            ShoeboxPreset::TheaterStage => Self::from_preset(RoomPreset::TheaterStage),
            ShoeboxPreset::StudioLiveRoom => {
                let geom = Self {
                    name: "Studio Live Room".to_string(),
                    dims_m: [8.0, 6.0, 4.0],
                    listener_m: [5.2, 3.0, 1.7],
                    // Front wall is wood-panelled (less absorbent than the side
                    // panels) so the Eyring 1 kHz fit lands on the 0.45 s design
                    // target instead of 14% below it.
                    walls: [
                        mat_wood_panel(),
                        mat_qrd_diffuser(),
                        mat_acoustic_panel(),
                        mat_acoustic_panel(),
                        mat_wood_floor(),
                        mat_ceiling_cloud(),
                    ],
                    bass_trap_admittance: 0.20,
                    target_rt60_s: 0.45,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_design_targets(
                    0.45,
                    [0.35, 0.55],
                    192.0,
                    1_000.0,
                    "Tier-A small studio live room; RT design range 0.35–0.55 s calibrated to \
                     BUT/ACE small rooms (research doc §4); material table is a design fit, not \
                     measured",
                )
            }
            ShoeboxPreset::LectureTheatre => {
                let geom = Self {
                    name: "Lecture Theatre".to_string(),
                    dims_m: [15.0, 10.0, 6.0],
                    listener_m: [10.5, 5.0, 1.2],
                    // Wood-panelled ceiling: with the carpeted floor and side
                    // panels this lands on the 0.8 s speech design target.
                    walls: [
                        mat_plaster(),
                        mat_qrd_diffuser(),
                        mat_acoustic_panel(),
                        mat_acoustic_panel(),
                        mat_carpet(),
                        mat_wood_panel(),
                    ],
                    bass_trap_admittance: 0.12,
                    target_rt60_s: 0.80,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_design_targets(
                    0.80,
                    [0.70, 0.90],
                    900.0,
                    500.0,
                    "Tier-A lecture theatre, speech clarity; RT design range 0.70–0.90 s, \
                     ACE/BUT cross-check (research doc §4); material table is a design fit",
                )
            }
            ShoeboxPreset::Cinema => {
                let geom = Self {
                    name: "Cinema Dubbing Stage".to_string(),
                    dims_m: [20.0, 12.0, 8.0],
                    listener_m: [14.0, 6.0, 1.2],
                    walls: [
                        mat_cinema_fabric(),
                        mat_cinema_fabric(),
                        mat_cinema_fabric(),
                        mat_cinema_fabric(),
                        mat_carpet(),
                        mat_cinema_fabric(),
                    ],
                    bass_trap_admittance: 0.18,
                    target_rt60_s: 0.50,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_design_targets(
                    0.50,
                    [0.40, 0.60],
                    1_920.0,
                    500.0,
                    "Tier-A cinema dubbing stage; RT design range 0.40–0.60 s (research doc §4); \
                     material table is a design fit, not measured",
                )
            }
            ShoeboxPreset::Musikverein => {
                let geom = Self {
                    name: "Vienna Musikverein, Grosser Saal".to_string(),
                    dims_m: [48.8, 19.1, 17.75],
                    listener_m: [46.5, 9.55, 2.0],
                    walls: [
                        mat_plaster(),
                        mat_plaster(),
                        mat_plaster(),
                        mat_plaster(),
                        mat_occupied_seats(),
                        mat_coffered_ceiling(),
                    ],
                    bass_trap_admittance: 0.05,
                    target_rt60_s: 2.0,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_hall_targets(HallTargetInput {
                    mid_rt60_s: 2.0,
                    mid_rt60_alt_s: None,
                    mid_edt_s: 2.0,
                    strength_g_db: 4.0,
                    clarity_c80_db: -1.0,
                    bass_lf: 0.17,
                    one_minus_iacc: Some(0.64),
                    itdg_ms: 12.0,
                    acoustic_volume_m3: 15_000.0,
                    reference_source_m: [4.0, 9.55, 2.0],
                    source: "Beranek 2004 via akuTEK Table 2 (T/EDT/G/C80/LF/1−IACC/ITDG); \
                                 geometry Wikipedia (Musikverein); 15 000 m³ is the quoted \
                                 acoustic volume, not L·W·H = 16 544 m³",
                })
            }
            ShoeboxPreset::BostonSymphonyHall => {
                let geom = Self {
                    name: "Boston Symphony Hall".to_string(),
                    dims_m: [38.1, 22.9, 18.6],
                    listener_m: [35.5, 11.45, 2.0],
                    walls: [
                        mat_wood_panel(),
                        mat_wood_panel(),
                        mat_wood_panel(),
                        mat_wood_panel(),
                        mat_occupied_seats(),
                        mat_coffered_ceiling(),
                    ],
                    bass_trap_admittance: 0.05,
                    target_rt60_s: 1.9,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_hall_targets(HallTargetInput {
                    mid_rt60_s: 1.9,
                    mid_rt60_alt_s: None,
                    mid_edt_s: 1.9,
                    strength_g_db: 2.0,
                    clarity_c80_db: 0.0,
                    bass_lf: 0.22,
                    one_minus_iacc: Some(0.65),
                    itdg_ms: 15.0,
                    acoustic_volume_m3: 18_750.0,
                    reference_source_m: [5.0, 11.45, 2.0],
                    source: "Beranek 2004 via akuTEK Table 2 (T/EDT/G/C80/LF/1−IACC/ITDG); \
                                 geometry Wikipedia (Symphony Hall, Boston); 18 750 m³ is the \
                                 quoted acoustic volume, not L·W·H = 16 228 m³",
                })
            }
            ShoeboxPreset::Concertgebouw => {
                let geom = Self {
                    name: "Amsterdam Concertgebouw, Grote Zaal".to_string(),
                    dims_m: [44.0, 28.0, 17.0],
                    listener_m: [40.4, 14.0, 2.0],
                    walls: [
                        mat_plaster(),
                        mat_plaster(),
                        mat_plaster(),
                        mat_plaster(),
                        mat_occupied_seats(),
                        mat_coffered_ceiling(),
                    ],
                    bass_trap_admittance: 0.05,
                    target_rt60_s: 2.0,
                    targets: None,
                    edges: Vec::new(),
                };
                geom.with_hall_targets(HallTargetInput {
                    mid_rt60_s: 2.0,
                    mid_rt60_alt_s: Some(2.2),
                    mid_edt_s: 1.9,
                    strength_g_db: 3.0,
                    clarity_c80_db: -1.0,
                    bass_lf: 0.18,
                    one_minus_iacc: None,
                    itdg_ms: 21.0,
                    acoustic_volume_m3: 18_780.0,
                    reference_source_m: [5.0, 14.0, 2.0],
                    source: "Beranek 2004 via akuTEK Table 2 (T/EDT/G/C80/LF/ITDG); \
                                 Concertgebouw geometry and RT 2.0 s occupied (Beranek) / \
                                 2.2 s occupied, 2.8 s empty (hall), Wikipedia; 18 780 m³ is \
                                 the quoted acoustic volume, not L·W·H = 20 944 m³",
                })
            }
        }
    }

    /// Quoted acoustic volume when the preset carries one, else `L·W·H`.
    pub fn acoustic_volume_m3(&self) -> f64 {
        self.targets
            .as_ref()
            .and_then(|t| t.acoustic_volume_m3)
            .unwrap_or(self.dims_m[0] * self.dims_m[1] * self.dims_m[2])
    }

    /// Total interior surface area of the shoebox (m²).
    pub fn surface_area_m2(&self) -> f64 {
        let [l, w, h] = self.dims_m;
        2.0 * (l * w + l * h + w * h)
    }

    /// Per-wall areas in `RoomGeometry` wall order (m²).
    fn wall_areas_m2(&self) -> [f64; 6] {
        let [l, w, h] = self.dims_m;
        [w * h, w * h, l * h, l * h, l * w, l * w]
    }

    /// Eyring (Norris–Eyring) reverberation time of one octave band (s):
    ///
    /// `T = 0.161 V / ( -S ln(1 - alpha_bar) + 4 m V )`
    ///
    /// with `alpha_bar` the six-wall area-weighted absorption at `ROOM_BANDS_HZ[band]`
    /// and `m` the ISO 9613-1 energy attenuation coefficient in `1/m`
    /// (`m = -2 ln(|H_air(1 m)|)` from [`AtmosphereCfg::air_gain_linear`]).
    pub fn eyring_rt60_band(&self, atm: &AtmosphereCfg, band: usize) -> f64 {
        let b = band.min(ROOM_BAND_COUNT - 1);
        let f = ROOM_BANDS_HZ[b];
        let areas = self.wall_areas_m2();
        let s: f64 = areas.iter().sum();
        let absorbed: f64 = areas
            .iter()
            .enumerate()
            .map(|(i, a)| a * self.walls[i].alpha(b))
            .sum();
        let alpha_bar = (absorbed / s).clamp(1e-4, 0.999);
        let v = self.dims_m[0] * self.dims_m[1] * self.dims_m[2];
        let m = -2.0 * atm.air_gain_linear(f, 1.0).ln();
        let denom = -s * (1.0 - alpha_bar).ln() + 4.0 * m * v;
        0.161 * v / denom.max(1e-12)
    }

    /// Eyring reverberation time for all ten octave bands (s).
    pub fn eyring_rt60_bands(&self, atm: &AtmosphereCfg) -> [f64; ROOM_BAND_COUNT] {
        std::array::from_fn(|b| self.eyring_rt60_band(atm, b))
    }

    /// Attach a calibrated Eyring per-band design target to a hall preset whose
    /// mid-band T is a measured value from the research doc.
    fn with_hall_targets(mut self, input: HallTargetInput<'_>) -> Self {
        let atm = AtmosphereCfg::default();
        let eyring = self.eyring_rt60_bands(&atm);
        // The FDTD wave solver owns everything below 250 Hz in the hall-scale
        // 260k-voxel grid (achieved 6-PPW cutoff 132–144 Hz): the geometric
        // targets start at the next octave, 250 Hz.
        let geometric_bands_from_hz = 250.0;
        let bands = std::array::from_fn(|i| {
            (ROOM_BANDS_HZ[i] >= geometric_bands_from_hz).then_some(eyring[i])
        });
        let fit_residual_s = bands[ROOM_BAND_MID].map(|t| t - input.mid_rt60_s);
        self.targets = Some(RoomAcousticTargets {
            bands_hz: ROOM_BANDS_HZ,
            rt60_bands_s: bands,
            edt_bands_s: [None; ROOM_BAND_COUNT],
            mid_rt60_s: input.mid_rt60_s,
            mid_rt60_alt_s: input.mid_rt60_alt_s,
            mid_edt_s: Some(input.mid_edt_s),
            strength_g_db: Some(input.strength_g_db),
            clarity_c80_db: Some(input.clarity_c80_db),
            bass_lf: Some(input.bass_lf),
            one_minus_iacc: input.one_minus_iacc,
            itdg_ms: Some(input.itdg_ms),
            mid_rt60_range_s: None,
            acoustic_volume_m3: Some(input.acoustic_volume_m3),
            geometric_bands_from_hz,
            model: Rt60Model::EyringDesign,
            reference_source_m: Some(input.reference_source_m),
            fit_residual_s,
            source: input.source.to_string(),
        });
        self
    }

    /// Attach an Eyring per-band design target to a standard/design room that
    /// has no measured per-band source.
    fn with_design_targets(
        mut self,
        mid_rt60_s: f64,
        range: [f64; 2],
        volume_m3: f64,
        geometric_bands_from_hz: f64,
        source: &str,
    ) -> Self {
        let atm = AtmosphereCfg::default();
        let eyring = self.eyring_rt60_bands(&atm);
        let bands = std::array::from_fn(|i| {
            (ROOM_BANDS_HZ[i] >= geometric_bands_from_hz).then_some(eyring[i])
        });
        let fit_residual_s = bands[ROOM_BAND_MID].map(|t| t - mid_rt60_s);
        self.targets = Some(RoomAcousticTargets {
            bands_hz: ROOM_BANDS_HZ,
            rt60_bands_s: bands,
            edt_bands_s: [None; ROOM_BAND_COUNT],
            mid_rt60_s,
            mid_rt60_alt_s: None,
            mid_edt_s: None,
            strength_g_db: None,
            clarity_c80_db: None,
            bass_lf: None,
            one_minus_iacc: None,
            itdg_ms: None,
            mid_rt60_range_s: Some(range),
            acoustic_volume_m3: Some(volume_m3),
            geometric_bands_from_hz,
            model: Rt60Model::EyringDesign,
            reference_source_m: None,
            fit_residual_s,
            source: source.to_string(),
        });
        self
    }

    pub fn from_json_file(path: &std::path::Path) -> std::result::Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read room json `{}`: {e}", path.display()))?;
        let geom: Self = serde_json::from_str(&text)
            .map_err(|e| format!("invalid room json `{}`: {e}", path.display()))?;
        geom.validate()?;
        Ok(geom)
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        for (i, &d) in self.dims_m.iter().enumerate() {
            if !d.is_finite() || !(1.5..=60.0).contains(&d) {
                return Err(format!("room.dims_m[{i}] must be in 1.5..=60.0 m, got {d}"));
            }
        }
        for i in 0..3 {
            let p = self.listener_m[i];
            if !p.is_finite() || p <= 0.2 || p >= self.dims_m[i] - 0.2 {
                return Err(format!(
                    "room.listener_m[{i}]={p} must lie strictly inside room dims {:?}",
                    self.dims_m
                ));
            }
        }
        if !self.target_rt60_s.is_finite() || !(0.01..=4.0).contains(&self.target_rt60_s) {
            return Err(format!(
                "room.target_rt60_s must be in 0.01..=4.0 s, got {}",
                self.target_rt60_s
            ));
        }
        for (wi, wall) in self.walls.iter().enumerate() {
            for (bi, &alpha) in wall.absorption.iter().enumerate() {
                if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                    return Err(format!(
                        "room.walls[{wi}].absorption[{bi}]={alpha} must be in 0.0..=1.0"
                    ));
                }
            }
            for (bi, &s) in wall.scattering.iter().enumerate() {
                if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                    return Err(format!(
                        "room.walls[{wi}].scattering[{bi}]={s} must be in 0.0..=1.0"
                    ));
                }
            }
        }
        if !self.bass_trap_admittance.is_finite()
            || !(0.0..=1.0).contains(&self.bass_trap_admittance)
        {
            return Err(format!(
                "room.bass_trap_admittance must be in 0.0..=1.0, got {}",
                self.bass_trap_admittance
            ));
        }
        if let Some(t) = &self.targets {
            if t.bands_hz != ROOM_BANDS_HZ {
                return Err(format!(
                    "room.targets.bands_hz must be the ten ISO octaves {:?}, got {:?}",
                    ROOM_BANDS_HZ, t.bands_hz
                ));
            }
            if !t.mid_rt60_s.is_finite() || !(0.01..=6.0).contains(&t.mid_rt60_s) {
                return Err(format!(
                    "room.targets.mid_rt60_s must be in 0.01..=6.0 s, got {}",
                    t.mid_rt60_s
                ));
            }
            if let Some([lo, hi]) = t.mid_rt60_range_s {
                if !lo.is_finite() || !hi.is_finite() || lo <= 0.0 || hi < lo {
                    return Err(format!(
                        "room.targets.mid_rt60_range_s must be [lo > 0, hi >= lo], got {lo}..{hi}"
                    ));
                }
            }
            for (bi, value) in t.rt60_bands_s.iter().enumerate() {
                if let Some(v) = value {
                    if !v.is_finite() || !(0.005..=60.0).contains(v) {
                        return Err(format!(
                            "room.targets.rt60_bands_s[{bi}]={v} must be in 0.005..=60.0 s"
                        ));
                    }
                }
            }
            for (bi, value) in t.edt_bands_s.iter().enumerate() {
                if let Some(v) = value {
                    if !v.is_finite() || !(0.005..=60.0).contains(v) {
                        return Err(format!(
                            "room.targets.edt_bands_s[{bi}]={v} must be in 0.005..=60.0 s"
                        ));
                    }
                }
            }
            if !t.geometric_bands_from_hz.is_finite()
                || !(20.0..=20_000.0).contains(&t.geometric_bands_from_hz)
            {
                return Err(format!(
                    "room.targets.geometric_bands_from_hz must be in 20..=20 000 Hz, got {}",
                    t.geometric_bands_from_hz
                ));
            }
            if let Some(p) = t.reference_source_m {
                for i in 0..3 {
                    if !p[i].is_finite() || p[i] < 0.0 || p[i] > self.dims_m[i] {
                        return Err(format!(
                            "room.targets.reference_source_m[{i}]={} must be inside dims {:?}",
                            p[i], self.dims_m
                        ));
                    }
                }
            }
            if let Some(v) = t.acoustic_volume_m3 {
                if !v.is_finite() || v <= 0.0 {
                    return Err(format!(
                        "room.targets.acoustic_volume_m3={v} must be finite and positive"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Inputs to [`RoomGeometry::with_hall_targets`] (keeps the preset table readable).
struct HallTargetInput<'a> {
    mid_rt60_s: f64,
    mid_rt60_alt_s: Option<f64>,
    mid_edt_s: f64,
    strength_g_db: f64,
    clarity_c80_db: f64,
    bass_lf: f64,
    one_minus_iacc: Option<f64>,
    itdg_ms: f64,
    acoustic_volume_m3: f64,
    reference_source_m: [f64; 3],
    source: &'a str,
}

/// Mid-band-only target used by the legacy control-room and theater presets.
///
/// `range_s` is the acceptance window when a standard specifies one (ITU-R
/// BS.1116 gives 0.2–0.4 s for the control room); design-target presets whose
/// research entry is a single value carry `None`.
fn standard_mid_band_targets(
    mid_rt60_s: f64,
    volume_m3: f64,
    range_s: Option<[f64; 2]>,
    source: &str,
) -> RoomAcousticTargets {
    RoomAcousticTargets {
        bands_hz: ROOM_BANDS_HZ,
        rt60_bands_s: [None; ROOM_BAND_COUNT],
        edt_bands_s: [None; ROOM_BAND_COUNT],
        mid_rt60_s,
        mid_rt60_alt_s: None,
        mid_edt_s: None,
        strength_g_db: None,
        clarity_c80_db: None,
        bass_lf: None,
        one_minus_iacc: None,
        itdg_ms: None,
        mid_rt60_range_s: range_s,
        acoustic_volume_m3: Some(volume_m3),
        geometric_bands_from_hz: 1_000.0,
        model: Rt60Model::MidBandOnly,
        reference_source_m: None,
        fit_residual_s: None,
        source: source.to_string(),
    }
}

// ============================================================================
// 1. Paraunitary Scattering FDN (Schlecht & Habets TASLP 2017 / TSP 2020)
//    + Velvet Noise Late Diffusion (Välimäki, Lehtonen & Takanen TASLP 2013)
// ============================================================================

/// 8-channel Paraunitary Scattering Feedback Delay Network (`N = 8`).
///
/// Couples 8 mutually-prime delay lines through an exact orthogonal Hadamard–Householder
/// paraunitary feedback matrix `U` (`U^T U == I` to `< 1e-14`, guaranteeing exact
/// unit-circle lossless eigenvalues when absorption is disabled) plus per-line 1-pole
/// `T_60(f)` absorption filters (`g_dc`, `g_nyq`).
#[derive(Debug, Clone)]
pub struct ScatteringFdn {
    delays: [usize; 8],
    lines: [Vec<f64>; 8],
    pos: [usize; 8],
    /// Per-line broadband decay gain (`1.0` for exact lossless mode).
    g_line: [f64; 8],
    /// Per-line 1-pole HF damping filter coefficient `p in [0, 0.85]`.
    damp_p: [f64; 8],
    damp_z: [f64; 8],
    /// Cascaded 2-stage paraunitary Householder/Hadamard scattering matrix (`8×8`, exact orthogonal).
    pub matrix: [[f64; 8]; 8],
}

impl ScatteringFdn {
    /// Construct an exact orthogonal `8×8` paraunitary feedback matrix by composing
    /// normalized Walsh–Hadamard `H_8 / sqrt(8)` with a Householder reflection `I - 2 v v^T`.
    pub fn paraunitary_matrix() -> [[f64; 8]; 8] {
        // Normalized 8x8 Walsh-Hadamard matrix.
        let s = 1.0 / (8.0_f64).sqrt();
        let mut h = [[0.0_f64; 8]; 8];
        for (i, row) in h.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let bits = (i & j).count_ones();
                *cell = if bits % 2 == 0 { s } else { -s };
            }
        }
        // Unit Householder reflection vector v (asymmetric to break trivial symmetry).
        let v_raw = [0.42, -0.31, 0.53, 0.19, -0.44, 0.27, -0.35, 0.16];
        let v_norm = v_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v: [f64; 8] = v_raw.map(|x| x / v_norm);
        let mut r = [[0.0_f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                let delta = if i == j { 1.0 } else { 0.0 };
                r[i][j] = delta - 2.0 * v[i] * v[j];
            }
        }
        // Product U = R * H (composition of two orthogonal matrices is strictly orthogonal!).
        let mut u = [[0.0_f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                let mut acc = 0.0_f64;
                for k in 0..8 {
                    acc += r[i][k] * h[k][j];
                }
                u[i][j] = acc;
            }
        }
        u
    }

    /// Create a Scattering FDN at `sample_rate` with `rt60_dc_s` and `rt60_nyq_s`.
    /// Passing `rt60_dc_s == f64::INFINITY && rt60_nyq_s == f64::INFINITY` creates the
    /// exact **lossless** prototype (`sum(out^2) == sum(in^2)` to `< 1e-13`).
    pub fn new(sample_rate: u32, rt60_dc_s: f64, rt60_nyq_s: f64) -> Self {
        let fs = (sample_rate.max(8_000)) as f64;
        // Mutually coprime delay lengths scaled from 48 kHz (~9 ms .. 29 ms).
        let base_primes: [usize; 8] = [443, 521, 613, 719, 829, 953, 1_091, 1_249];
        let sr_ratio = fs / 48_000.0;
        let mut delays = [0usize; 8];
        for i in 0..8 {
            delays[i] = ((base_primes[i] as f64) * sr_ratio).round().max(16.0) as usize;
        }
        let lines = std::array::from_fn(|i| vec![0.0_f64; delays[i]]);

        let lossless = !rt60_dc_s.is_finite() && !rt60_nyq_s.is_finite();
        let mut g_line = [1.0_f64; 8];
        let mut damp_p = [0.0_f64; 8];
        if !lossless {
            let t_dc = rt60_dc_s.clamp(0.05, 5.0);
            let t_hi = rt60_nyq_s.clamp(0.03, t_dc);
            for i in 0..8 {
                let tau_s = (delays[i] as f64) / fs;
                // -60 dB in T60 -> gain = 10^(-3 * tau / T60)
                let g_dc = 10.0_f64.powf(-3.0 * tau_s / t_dc);
                let g_hi = 10.0_f64.powf(-3.0 * tau_s / t_hi);
                g_line[i] = g_dc;
                // 1-pole filter H(z) = (1-p)/(1 - p z^-1): |H(pi)| = (1-p)/(1+p) = g_hi/g_dc
                let r = (g_hi / g_dc.max(1e-12)).clamp(0.05, 1.0);
                damp_p[i] = ((1.0 - r) / (1.0 + r)).clamp(0.0, 0.85);
            }
        }

        Self {
            delays,
            lines,
            pos: [0; 8],
            g_line,
            damp_p,
            damp_z: [0.0; 8],
            matrix: Self::paraunitary_matrix(),
        }
    }

    /// Process one 8-channel vector step through the FDN, returning the current delay-line outputs.
    pub fn tick(&mut self, input: [f64; 8]) -> [f64; 8] {
        let mut y = [0.0_f64; 8];
        for (i, line) in self.lines.iter().enumerate() {
            let raw = line[self.pos[i]];
            // Frequency-dependent T60 absorption filter.
            let p = self.damp_p[i];
            let filt = if p > 0.0 {
                let out = (1.0 - p) * raw + p * self.damp_z[i];
                self.damp_z[i] = out;
                out
            } else {
                raw
            };
            y[i] = filt * self.g_line[i];
        }

        // Orthogonal paraunitary scattering mix U * y + input.
        for (i, &inp) in input.iter().enumerate() {
            let mut fb = inp;
            for (j, &yj) in y.iter().enumerate() {
                fb += self.matrix[i][j] * yj;
            }
            self.lines[i][self.pos[i]] = fb;
            self.pos[i] += 1;
            if self.pos[i] >= self.delays[i] {
                self.pos[i] = 0;
            }
        }
        y
    }

    /// Synthesize a stereo decorrelated late-tail impulse response `(ir_l, ir_r)` of length `len`.
    pub fn synthesize_stereo_tail(&mut self, len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut ir_l = vec![0.0_f64; len];
        let mut ir_r = vec![0.0_f64; len];
        let inj = [
            0.353_553_39,
            -0.353_553_39,
            0.353_553_39,
            -0.353_553_39,
            0.353_553_39,
            -0.353_553_39,
            0.353_553_39,
            -0.353_553_39,
        ];
        // Orthogonal L/R readout taps for minimal broadband IACC.
        let out_l = [0.5, 0.0, -0.5, 0.0, 0.5, 0.0, -0.5, 0.0];
        let out_r = [0.0, 0.5, 0.0, -0.5, 0.0, 0.5, 0.0, -0.5];
        for n in 0..len {
            let inp = if n == 0 { inj } else { [0.0; 8] };
            let y = self.tick(inp);
            let mut sl = 0.0_f64;
            let mut sr = 0.0_f64;
            for i in 0..8 {
                sl += out_l[i] * y[i];
                sr += out_r[i] * y[i];
            }
            ir_l[n] = sl;
            ir_r[n] = sr;
        }
        (ir_l, ir_r)
    }
}

/// Generate a maximally sparse ternary `{-1, 0, +1}` Velvet Noise sequence
/// (Välimäki, Lehtonen & Takanen, IEEE TASLP 2013) with exact grid spacing `td` samples
/// (`density_hz` impulses per second, typically `2000 Hz`) and deterministic `Pcg64` seed.
pub fn generate_velvet_noise(len: usize, sample_rate: u32, density_hz: f64, seed: u64) -> Vec<f64> {
    let mut out = vec![0.0_f64; len];
    if len == 0 {
        return out;
    }
    let fs = (sample_rate.max(8_000)) as f64;
    let td = (fs / density_hz.clamp(200.0, 8_000.0)).max(2.0);
    let mut rng = Pcg64::new(seed, 0x5645_4C56_4554_4E31);
    let mut m = 0usize;
    loop {
        let base = (m as f64) * td;
        let idx = (base + rng.next_f64() * (td - 1.0)).floor() as usize;
        if idx >= len {
            break;
        }
        let sign = if (rng.next_u64() & 1) == 0 { 1.0 } else { -1.0 };
        out[idx] = sign;
        m += 1;
    }
    out
}

// ============================================================================
// 2. Layer 2: Analytic Biot–Tolstoy–Medwin (BTM) Edge Diffraction
//    (Svensson, Fred & Vanderkooy, JASA 1999) + Kirsch & Ewert (TASLP 2023)
// ============================================================================

/// Evaluate the Svensson–Fred–Vanderkooy (JASA 1999) BTM secondary-source line-integral
/// impulse response for a finite diffracting wedge `edge` from `src` to `rec`, accumulated
/// into `ir` at `sample_rate` and speed of sound `c_m_s`.
///
/// Integrates the exact BTM directivity factor `beta(theta_s, theta_r, nu)` over `N_SEG`
/// edge segments (`nu = PI / wedge_angle_rad`, with `nu = 2/3` for a 90° mixing desk).
#[allow(clippy::too_many_arguments)]
pub fn add_btm_edge_diffraction(
    edge: &DiffractionEdge,
    src: [f64; 3],
    rec: [f64; 3],
    c_m_s: f64,
    sample_rate: u32,
    t_direct_s: f64,
    pre_delay_s: f64,
    ir: &mut [f64],
) {
    let fs = sample_rate as f64;
    let ex = edge.p1[0] - edge.p0[0];
    let ey = edge.p1[1] - edge.p0[1];
    let ez = edge.p1[2] - edge.p0[2];
    let edge_len = (ex * ex + ey * ey + ez * ez).sqrt();
    if edge_len < 1e-6 || ir.is_empty() {
        return;
    }
    let e_hat = [ex / edge_len, ey / edge_len, ez / edge_len];
    let theta_w = edge.wedge_angle_rad.clamp(1.05 * PI, 1.99 * PI);
    let nu = PI / theta_w;

    // Construct an orthonormal cylindrical frame `(u_hat, v_hat, e_hat)` around the wedge edge
    // where `u_hat` points along wedge face 0 (`theta = 0`) and `v_hat = normalize(u_hat x e_hat)`
    // sweeps through the exterior air region `theta in [0, theta_w]`.
    let cross3 = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let norm3 = |a: [f64; 3]| -> [f64; 3] {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let u_raw = if let Some(f0) = edge.face0_dir {
        let proj = f0[0] * e_hat[0] + f0[1] * e_hat[1] + f0[2] * e_hat[2];
        [
            f0[0] - proj * e_hat[0],
            f0[1] - proj * e_hat[1],
            f0[2] - proj * e_hat[2],
        ]
    } else if e_hat[2].abs() < 0.9 {
        cross3(e_hat, [0.0, 0.0, 1.0])
    } else {
        cross3(e_hat, [1.0, 0.0, 0.0])
    };
    let u_hat = norm3(u_raw);
    let v_hat = norm3(cross3(u_hat, e_hat));

    const N_SEG: usize = 48;
    let dz = edge_len / (N_SEG as f64);

    // Distance from direct path for relative level normalization.
    let d_dir = ((src[0] - rec[0]).powi(2) + (src[1] - rec[1]).powi(2) + (src[2] - rec[2]).powi(2))
        .sqrt()
        .max(0.05);

    for seg in 0..N_SEG {
        let u = (seg as f64 + 0.5) * dz;
        let ep = [
            edge.p0[0] + e_hat[0] * u,
            edge.p0[1] + e_hat[1] * u,
            edge.p0[2] + e_hat[2] * u,
        ];
        let vs = [src[0] - ep[0], src[1] - ep[1], src[2] - ep[2]];
        let vr = [rec[0] - ep[0], rec[1] - ep[1], rec[2] - ep[2]];
        let m_dist = (vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2])
            .sqrt()
            .max(1e-4);
        let l_dist = (vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2])
            .sqrt()
            .max(1e-4);

        // Total propagation time relative to direct wave.
        let t_path = (m_dist + l_dist) / c_m_s;
        let t_rel = pre_delay_s + (t_path - t_direct_s);
        if t_rel < 0.0 {
            continue;
        }
        let smp_f = t_rel * fs;
        let i0 = smp_f.floor() as usize;
        if i0 + 1 >= ir.len() {
            continue;
        }
        let frac = smp_f - (i0 as f64);

        // Cylindrical coordinates `(zs, rs, theta_s)` and `(zr, rr, theta_r)` around the wedge edge.
        let zs = vs[0] * e_hat[0] + vs[1] * e_hat[1] + vs[2] * e_hat[2];
        let zr = vr[0] * e_hat[0] + vr[1] * e_hat[1] + vr[2] * e_hat[2];
        let rs_v = [
            vs[0] - zs * e_hat[0],
            vs[1] - zs * e_hat[1],
            vs[2] - zs * e_hat[2],
        ];
        let rr_v = [
            vr[0] - zr * e_hat[0],
            vr[1] - zr * e_hat[1],
            vr[2] - zr * e_hat[2],
        ];
        let rs = (rs_v[0] * rs_v[0] + rs_v[1] * rs_v[1] + rs_v[2] * rs_v[2])
            .sqrt()
            .max(1e-5);
        let rr = (rr_v[0] * rr_v[0] + rr_v[1] * rr_v[1] + rr_v[2] * rr_v[2])
            .sqrt()
            .max(1e-5);

        let su = rs_v[0] * u_hat[0] + rs_v[1] * u_hat[1] + rs_v[2] * u_hat[2];
        let sv = rs_v[0] * v_hat[0] + rs_v[1] * v_hat[1] + rs_v[2] * v_hat[2];
        let ru = rr_v[0] * u_hat[0] + rr_v[1] * u_hat[1] + rr_v[2] * u_hat[2];
        let rv = rr_v[0] * v_hat[0] + rr_v[1] * v_hat[1] + rr_v[2] * v_hat[2];
        let theta_s = sv.atan2(su).rem_euclid(2.0 * PI).clamp(0.0, theta_w);
        let theta_r = rv.atan2(ru).rem_euclid(2.0 * PI).clamp(0.0, theta_w);

        // Svensson, Fred & Vanderkooy (JASA 1999) Eq. (6)-(8):
        // cosh(eta) = (m*l + zs*zr) / (rs*rr), and the 4 symmetric wedge face angles
        // `pi +- theta_s +- theta_r` (preserving exact Helmholtz reciprocity!).
        let cosh_eta = ((m_dist * l_dist + zs * zr) / (rs * rr)).max(1.000_001);
        let eta = cosh_eta.acosh();
        let cosh_nu_eta = (nu * eta).cosh();

        let angles = [
            PI + theta_s + theta_r,
            PI + theta_s - theta_r,
            PI - theta_s + theta_r,
            PI - theta_s - theta_r,
        ];
        let mut beta = 0.0_f64;
        for &phi in &angles {
            let den = (cosh_nu_eta - (nu * phi).cos()).max(0.08);
            beta += (nu * phi).sin() / den;
        }

        // BTM secondary-source amplitude normalized to free-field direct pressure (1 / (4*pi*d_dir)).
        let amp = (-nu / (4.0 * PI)) * beta * (dz * d_dir / (m_dist * l_dist));
        // Kirsch & Ewert (TASLP 2023) physical damping limit prevents unphysical pole spikes.
        let amp_clamped = amp.clamp(-0.08, 0.08);
        ir[i0] += amp_clamped * (1.0 - frac);
        ir[i0 + 1] += amp_clamped * frac;
    }
}

// ============================================================================
// 3. Layer 3 (< 500 Hz): 3D Wave-Equation FDTD PDE Solver
//    (Hamilton & Bilbao, IEEE/ACM TASLP 2017) with Acoustic Reciprocity
// ============================================================================

/// Configuration & state for the 3D Standard Rectilinear (SRL) Wave-Equation FDTD Solver.
#[derive(Debug, Clone)]
pub struct FdtdSolver3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dt_s: f64,
    pub lambda: f64,
    pub beta_wall: f64,
}

impl FdtdSolver3d {
    /// Construct a Courant-stable (`lambda = c * dt / dx = 0.55 < 1/sqrt(3)`) 3D FDTD grid
    /// covering `geom.dims_m` with spatial step `dx_m` (`0.114 m` resolves up to `504 Hz` at
    /// `6` points-per-wavelength for ITU-R BS.1116 control rooms).
    pub fn new(geom: &RoomGeometry, c_m_s: f64, dx_m: f64) -> Self {
        let dx = dx_m.clamp(0.07, 0.40);
        let nx = ((geom.dims_m[0] / dx).round() as usize).clamp(8, 128);
        let ny = ((geom.dims_m[1] / dx).round() as usize).clamp(8, 128);
        let nz = ((geom.dims_m[2] / dx).round() as usize).clamp(6, 96);
        let lambda = 0.55_f64; // Strictly below 3D SRL stability limit 1/sqrt(3) ≈ 0.57735
        let dt_s = lambda * dx / c_m_s;
        Self {
            nx,
            ny,
            nz,
            dx_m: dx,
            dt_s,
            lambda,
            beta_wall: geom.bass_trap_admittance.clamp(0.01, 0.98),
        }
    }

    /// Construct a 3D FDTD grid targeting `target_hz` at `6` points-per-wavelength while
    /// keeping the total voxel count within `max_voxels` for large theatrical halls.
    ///
    /// - For `ControlRoom` (`6.0 × 5.0 × 3.0 m = 90 m³`, `max_voxels = 65_000`), selects
    ///   `dx = 0.114 m` (`53 × 44 × 26 = 60,632` voxels, `f_6ppw = 504 Hz`!).
    /// - For `ControlRoom` Production (`target_hz = 670 Hz`, `max_voxels = 260_000`), selects
    ///   `dx = 0.085 m` (`71 × 59 × 35 = 146,615` voxels, `f_6ppw = 676 Hz`!).
    /// - For `TheaterStage` (`12.0 × 9.5 × 5.2 m = 593 m³`, Schroeder freq `53 Hz`), selects
    ///   `dx = 0.209 m` (`57 × 45 × 25 = 64,125` voxels, `f_6ppw = 275 Hz`, `> 5×` Schroeder!).
    pub fn for_room_budget(
        geom: &RoomGeometry,
        c_m_s: f64,
        target_hz: f64,
        max_voxels: usize,
    ) -> Self {
        let dx_target = c_m_s / (6.0 * target_hz.clamp(100.0, 720.0));
        let vol = geom.dims_m[0] * geom.dims_m[1] * geom.dims_m[2];
        let dx_budget = (vol / (max_voxels.max(4_096) as f64)).cbrt();
        Self::new(geom, c_m_s, dx_target.max(dx_budget))
    }

    /// Exact 6 points-per-wavelength valid bandwidth of this 3D FDTD grid in Hz.
    #[inline]
    pub fn max_freq_6ppw(&self, c_m_s: f64) -> f64 {
        c_m_s / (6.0 * self.dx_m)
    }

    #[inline]
    pub fn coord_to_voxel(&self, p: [f64; 3]) -> (usize, usize, usize) {
        let ix = ((p[0] / self.dx_m).round() as isize).clamp(1, (self.nx - 2) as isize) as usize;
        let iy = ((p[1] / self.dx_m).round() as isize).clamp(1, (self.ny - 2) as isize) as usize;
        let iz = ((p[2] / self.dx_m).round() as isize).clamp(1, (self.nz - 2) as isize) as usize;
        (ix, iy, iz)
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    /// Run ONE 3D FDTD wave simulation with an impulse injected at `src_voxel`, recording
    /// the pressure time-series simultaneously at ALL `receivers` voxels!
    ///
    /// By **Acoustic Reciprocity**, placing the source at the Left ear (run 1) and Right ear
    /// (run 2) records the low-frequency modal room response for ALL 12 virtual speakers
    /// in just 2 FDTD runs!
    /// Run the 3D FDTD wave simulation, optionally offloading the zero-FMA 7-point SRL
    /// stencil to a live NVIDIA GPU via `g6me-cuda` when `use_cuda` is true and `--features cuda`
    /// is compiled in. Falls back deterministically to the CPU solver when no GPU is
    /// attached.
    pub fn simulate_with_cuda(
        &self,
        src_voxel: (usize, usize, usize),
        receivers: &[(usize, usize, usize)],
        steps: usize,
        use_cuda: bool,
    ) -> Vec<Vec<f64>> {
        #[cfg(feature = "cuda")]
        if use_cuda {
            match crate::blackwell::shared_cuda_runtime() {
                Some(rt) => match self.simulate_on_cuda(&rt, src_voxel, receivers, steps) {
                    Ok(out) => {
                        crate::cuda_report::record_gpu(
                            "fdtd",
                            format!("{}x{}x{} voxels, {steps} steps", self.nx, self.ny, self.nz),
                        );
                        return out;
                    }
                    Err(e) => crate::cuda_report::record_cpu_fallback("fdtd", e),
                },
                None => crate::cuda_report::record_cpu_fallback("fdtd", "CUDA runtime unavailable"),
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = use_cuda;
        self.simulate(src_voxel, receivers, steps)
    }

    /// Execute the entire 3D SRL FDTD wave simulation device-resident on a live CUDA runtime.
    #[cfg(feature = "cuda")]
    pub fn simulate_on_cuda(
        &self,
        rt: &g6me_cuda::G6meCudaRuntime,
        src_voxel: (usize, usize, usize),
        receivers: &[(usize, usize, usize)],
        steps: usize,
    ) -> std::result::Result<Vec<Vec<f64>>, String> {
        let total = self.nx * self.ny * self.nz;
        let p_prev = vec![0.0_f64; total];
        let mut p_curr = vec![0.0_f64; total];
        let lam2 = self.lambda * self.lambda;
        let half_lam_beta = 0.5 * self.lambda * self.beta_wall;
        let mut inv_den = [0.0_f64; 7];
        for (kb, slot) in inv_den.iter_mut().enumerate() {
            *slot = 1.0 / (1.0 + half_lam_beta * (kb as f64));
        }
        let src_id = self.idx(src_voxel.0, src_voxel.1, src_voxel.2);
        if steps > 0 {
            p_curr[src_id] = 1.0;
        }
        let rec_ids: Vec<usize> = receivers
            .iter()
            .map(|&(i, j, k)| self.idx(i, j, k))
            .collect();
        rt.run_fdtd_3d_simulation(
            &p_prev,
            &p_curr,
            &inv_den,
            self.nx,
            self.ny,
            self.nz,
            lam2,
            half_lam_beta,
            &rec_ids,
            steps,
        )
    }

    pub fn simulate(
        &self,
        src_voxel: (usize, usize, usize),
        receivers: &[(usize, usize, usize)],
        steps: usize,
    ) -> Vec<Vec<f64>> {
        let total = self.nx * self.ny * self.nz;
        let mut p_prev = vec![0.0_f64; total];
        let mut p_curr = vec![0.0_f64; total];
        let mut p_next = vec![0.0_f64; total];
        let lam2 = self.lambda * self.lambda;
        let center_c = 2.0 - 6.0 * lam2;
        let half_lam_beta = 0.5 * self.lambda * self.beta_wall;

        let src_id = self.idx(src_voxel.0, src_voxel.1, src_voxel.2);
        let rec_ids: Vec<usize> = receivers
            .iter()
            .map(|&(i, j, k)| self.idx(i, j, k))
            .collect();
        let mut out: Vec<Vec<f64>> = vec![vec![0.0_f64; steps]; receivers.len()];

        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let stride_x = ny * nz;

        #[allow(clippy::needless_range_loop)]
        for step in 0..steps {
            if step == 0 {
                p_curr[src_id] += 1.0;
            }
            for (r, &rid) in rec_ids.iter().enumerate() {
                out[r][step] = p_curr[rid];
            }

            // Fast branchless interior 7-point 3D SRL stencil (`1..nx-1, 1..ny-1, 1..nz-1`).
            for i in 1..(nx - 1) {
                let i_off = i * stride_x;
                for j in 1..(ny - 1) {
                    let ij_off = i_off + j * nz;
                    for k in 1..(nz - 1) {
                        let id = ij_off + k;
                        let sum_nb = p_curr[id - stride_x]
                            + p_curr[id + stride_x]
                            + p_curr[id - nz]
                            + p_curr[id + nz]
                            + p_curr[id - 1]
                            + p_curr[id + 1];
                        p_next[id] = center_c * p_curr[id] - p_prev[id] + lam2 * sum_nb;
                    }
                }
            }

            // Locally-reacting porous bass-trap boundary planes (Hamilton & Bilbao 2017).
            let update_bnd = |i: usize, j: usize, k: usize, p_next: &mut [f64]| {
                let id = (i * ny + j) * nz + k;
                let pc = p_curr[id];
                let mut lap = 0.0_f64;
                let mut k_bnd = 0u32;
                if i > 0 {
                    lap += p_curr[id - stride_x] - pc;
                } else {
                    k_bnd += 1;
                }
                if i + 1 < nx {
                    lap += p_curr[id + stride_x] - pc;
                } else {
                    k_bnd += 1;
                }
                if j > 0 {
                    lap += p_curr[id - nz] - pc;
                } else {
                    k_bnd += 1;
                }
                if j + 1 < ny {
                    lap += p_curr[id + nz] - pc;
                } else {
                    k_bnd += 1;
                }
                if k > 0 {
                    lap += p_curr[id - 1] - pc;
                } else {
                    k_bnd += 1;
                }
                if k + 1 < nz {
                    lap += p_curr[id + 1] - pc;
                } else {
                    k_bnd += 1;
                }
                let g = half_lam_beta * (k_bnd as f64);
                p_next[id] = (2.0 * pc + lam2 * lap - (1.0 - g) * p_prev[id]) / (1.0 + g);
            };

            // x = 0 and x = nx - 1 faces
            for j in 0..ny {
                for k in 0..nz {
                    update_bnd(0, j, k, &mut p_next);
                    update_bnd(nx - 1, j, k, &mut p_next);
                }
            }
            // y = 0 and y = ny - 1 faces (excluding already-updated x faces)
            for i in 1..(nx - 1) {
                for k in 0..nz {
                    update_bnd(i, 0, k, &mut p_next);
                    update_bnd(i, ny - 1, k, &mut p_next);
                }
            }
            // z = 0 and z = nz - 1 faces (excluding already-updated x and y faces)
            for i in 1..(nx - 1) {
                for j in 1..(ny - 1) {
                    update_bnd(i, j, 0, &mut p_next);
                    update_bnd(i, j, nz - 1, &mut p_next);
                }
            }

            std::mem::swap(&mut p_prev, &mut p_curr);
            std::mem::swap(&mut p_curr, &mut p_next);
        }
        out
    }
}

// ============================================================================
// 4. Layer 1 (> 500 Hz): 3D Image-Source Method (Orders 1–3) +
//    Deterministic native Monte Carlo Ray Tracing (Orders 4–50 -> Order-7 HOA)
// ============================================================================

/// A single 3D geometric image-source reflection tap.
#[derive(Debug, Clone, Copy)]
pub struct ImageSourceTap {
    pub delay_s: f64,
    pub az_deg: f64,
    pub el_deg: f64,
    pub dist_m: f64,
    /// Bounce count on each of the 6 walls `[+x, -x, +y, -y, -z, +z]`.
    pub wall_hits: [u8; 6],
    /// Per-band energy reflection product `∏_w (1 - alpha_{w,b})^{hits_w}` over
    /// all wall bounces of this tap, using the ten-band material table. This is
    /// the per-band image-source transport weight; the caller adds molecular
    /// air absorption and geometric spreading per frequency.
    pub band_reflection_energy: [f64; ROOM_BAND_COUNT],
}

impl ImageSourceTap {
    /// Per-band amplitude reflection product `sqrt(energy_b)`.
    #[inline]
    pub fn band_reflection_amp(&self, band: usize) -> f64 {
        self.band_reflection_energy[band.min(ROOM_BAND_COUNT - 1)].sqrt()
    }
}

/// Allen & Berkley (JASA 1979) 3D shoebox Image-Source Method up to `max_order` (`<= 3`).
///
/// Computes exact reflection coordinates, arrival delays relative to direct path,
/// incidence angles `(az_deg, el_deg)` in the listener's head frame (`+x` front, `+y` left,
/// `+z` up), propagation distance, and per-wall bounce counts for frequency-dependent
/// material + molecular air attenuation.
pub fn image_sources_3d(
    geom: &RoomGeometry,
    src_m: [f64; 3],
    c_m_s: f64,
    max_order: i32,
) -> Vec<ImageSourceTap> {
    let l = geom.dims_m;
    let rec = geom.listener_m;
    let d_dir =
        ((src_m[0] - rec[0]).powi(2) + (src_m[1] - rec[1]).powi(2) + (src_m[2] - rec[2]).powi(2))
            .sqrt()
            .max(0.05);
    let t_dir = d_dir / c_m_s;

    let mut taps = Vec::new();
    for nx in -max_order..=max_order {
        for ny in -max_order..=max_order {
            for nz in -max_order..=max_order {
                let order = nx.abs() + ny.abs() + nz.abs();
                if order == 0 || order > max_order {
                    continue;
                }
                let mirror_coord = |s: f64, dim: f64, n: i32| -> f64 {
                    if n % 2 == 0 {
                        (n as f64) * dim + s
                    } else {
                        ((n + 1) as f64) * dim - s
                    }
                };
                let ix = mirror_coord(src_m[0], l[0], nx);
                let iy = mirror_coord(src_m[1], l[1], ny);
                let iz = mirror_coord(src_m[2], l[2], nz);

                let dx = ix - rec[0];
                let dy = iy - rec[1];
                let dz = iz - rec[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.05);
                let delay_s = dist / c_m_s - t_dir;
                if delay_s < 0.002 {
                    continue;
                }
                let horiz = (dx * dx + dy * dy).sqrt().max(1e-9);
                let az_deg = dy.atan2(dx).to_degrees();
                let el_deg = dz.atan2(horiz).to_degrees();

                // Wall hit counts on [+x, -x, +y, -y, -z, +z].
                let split_hits = |n: i32| -> (u8, u8) {
                    if n >= 0 {
                        (((n + 1) / 2) as u8, (n / 2) as u8)
                    } else {
                        (((-n) / 2) as u8, ((-n + 1) / 2) as u8)
                    }
                };
                let (hx_pos, hx_neg) = split_hits(nx);
                let (hy_pos, hy_neg) = split_hits(ny);
                let (hz_pos, hz_neg) = split_hits(nz);

                let wall_hits = [hx_pos, hx_neg, hy_pos, hy_neg, hz_neg, hz_pos];
                let band_reflection_energy = std::array::from_fn(|band| {
                    wall_hits
                        .iter()
                        .enumerate()
                        .fold(1.0_f64, |acc, (w, &hits)| {
                            if hits == 0 {
                                acc
                            } else {
                                acc * geom.walls[w].refl_energy(band).powi(hits as i32)
                            }
                        })
                });
                taps.push(ImageSourceTap {
                    delay_s,
                    az_deg,
                    el_deg,
                    dist_m: dist,
                    wall_hits,
                    band_reflection_energy,
                });
            }
        }
    }
    taps
}

/// **Test-only** bit-exact Rust mirror of the PCG stream in
/// `crates/g6me-cuda/kernels/raytrace_hoa.cu`.
///
/// The kernel advances a 64-bit LCG state and emits an XSH-RR-permuted 32-bit word,
/// consuming two 64-bit words per `f64` uniform (`hi` supplies 21 bits, `lo` 32). The
/// seeding, draw order and arithmetic below are identical to `pcg32_next` /
/// `pcg32_uniform`; this oracle exists solely to prove that the CUDA kernel still
/// produces the records it is specified to produce. It is **not** on any production
/// path: [`raytrace_hoa_histogram`] uses the independent native CPU tracer below.
#[cfg(all(test, feature = "cuda"))]
#[derive(Debug, Clone, Copy)]
struct RayPcg32 {
    state: u64,
    inc: u64,
}

#[cfg(all(test, feature = "cuda"))]
impl RayPcg32 {
    /// Kernel seeding: `inc = 2*ray_id + 1`, `state = seed ^ ((ray_id + 1) * golden)`,
    /// then one discarded output.
    fn new(seed: u64, ray_id: usize) -> Self {
        let rid = ray_id as u64;
        let mut rng = Self {
            state: seed ^ rid.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            inc: (rid << 1) | 1,
        };
        let _ = rng.next_u64();
        rng
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        u64::from(xorshifted.rotate_right(rot))
    }

    /// Uniform `f64` in `[0.0, 1.0)`, matching `pcg32_uniform` exactly.
    #[inline]
    fn uniform(&mut self) -> f64 {
        let hi = self.next_u64();
        let lo = self.next_u64();
        let bits = ((hi & 0x1F_FFFF) << 32) | lo;
        (bits as f64) * 1.110_223_024_625_156_5e-16 // 2^-53
    }
}

/// One per-bounce trace record produced by the **test-only** kernel mirror oracle,
/// ABI-compatible with the `(ray_delays, ray_energies, ray_dirs)` arrays returned by
/// `G6meCudaRuntime::run_raytrace_hoa_pass_with_scattering`.
///
/// `delay_s == -1.0` / `energy == 0.0` marks a bounce with no receiver-sphere hit.
#[cfg(all(test, feature = "cuda"))]
#[derive(Debug, Clone, Copy)]
struct RayHitRecord {
    delay_s: f64,
    energy: f64,
    /// Direction from the listener back toward the incoming ray (`-dir`).
    dir: [f64; 3],
}

/// Accumulate one per-bounce **GPU** ray record into the Order-7 HOA histogram.
///
/// This is the GPU host reduction used by [`raytrace_hoa_histogram_on_runtime`] and
/// [`raytrace_hoa_histogram_with_cuda`]: records arrive already in the kernel's
/// `(ray ascending, bounce ascending)` layout and are summed in that order. The
/// production CPU path ([`raytrace_hoa_histogram`]) does **not** call it — it runs the
/// independent native tracer with its own per-chunk reduction. The only CPU caller is
/// the `#[cfg(all(test, feature = "cuda"))]` bit-exact mirror oracle, which validates
/// the kernel itself. Bounces `1..=3` are skipped: orders 1–3 come from the exact
/// image-source method.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn accumulate_hoa_hit(
    hist: &mut [[f64; HOA_CH]],
    bounce: usize,
    delay_s: f64,
    energy: f64,
    inc: [f64; 3],
    atm: &AtmosphereCfg,
    c_m_s: f64,
    n_time_bins: usize,
    bin_dt_s: f64,
) {
    accumulate_hoa_hit_at(
        hist,
        bounce,
        delay_s,
        energy,
        inc,
        atm,
        c_m_s,
        RAY_AIR_REFERENCE_HZ,
        n_time_bins,
        bin_dt_s,
    );
}

/// [`accumulate_hoa_hit`] at an explicit reference frequency. The per-band
/// reduction passes the band centre (`ROOM_BANDS_HZ[band]`), so each band's
/// molecular air absorption is evaluated at its own frequency instead of the
/// legacy broadband 2 kHz reference.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn accumulate_hoa_hit_at(
    hist: &mut [[f64; HOA_CH]],
    bounce: usize,
    delay_s: f64,
    energy: f64,
    inc: [f64; 3],
    atm: &AtmosphereCfg,
    c_m_s: f64,
    f_air_hz: f64,
    n_time_bins: usize,
    bin_dt_s: f64,
) {
    // Orders 4..=max_bounces (0-indexed `bounce >= 3`) only.
    if bounce < 3 || delay_s <= 0.0 || energy <= 0.0 {
        return;
    }
    let bin = (delay_s / bin_dt_s) as usize;
    if bin >= n_time_bins {
        return;
    }
    let az = inc[1].atan2(inc[0]);
    let el = inc[2].clamp(-1.0, 1.0).asin();
    let sh = eval_sh_order7(az, el);
    // `raytrace_hoa.cu` reports arrival time; recover the travelled distance for
    // molecular air absorption exactly as the GPU host reduction does.
    let d_hit = delay_s * c_m_s;
    let air = atm.air_gain_linear(f_air_hz, d_hit);
    let w = energy * air * air;
    for c_idx in 0..HOA_CH {
        hist[bin][c_idx] += w * sh[c_idx];
    }
}

/// **Test-only oracle:** trace one kernel-PCG32 ray (`ray_id`) through the shoebox
/// room, mirroring `raytrace_hoa.cu` line for line.
///
/// Every floating-point operation, PCG draw and branch mirrors the CUDA kernel
/// (Marsaglia unit-sphere launch, same wall order, same receiver-sphere test, same wall
/// absorption, same energy floor, same Lambertian scatter), so the returned records are
/// bit-identical to the GPU's per-bounce arrays. This is used only by the CUDA oracle
/// test that keeps the kernel honest; it is not a CPU production implementation.
#[cfg(all(test, feature = "cuda"))]
fn trace_single_ray(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    ray_id: usize,
    ray_init_energy: f64,
    max_bounces: usize,
    seed: u64,
) -> Vec<RayHitRecord> {
    let c = atm.speed_of_sound_m_s();
    let l = geom.dims_m;
    let [rx, ry, rz] = geom.listener_m;
    let rec_rad = 0.45_f64; // Acoustic receiver sphere radius (m)
    let rec_rad2 = rec_rad * rec_rad;
    let mut rng = RayPcg32::new(seed, ray_id);
    let mut out = Vec::new();

    // Uniform random direction on the unit sphere via Marsaglia (1972) rejection,
    // identical draws to the kernel launch sampler.
    let (mut dx, mut dy, mut dz) = loop {
        let ux = 2.0 * rng.uniform() - 1.0;
        let uy = 2.0 * rng.uniform() - 1.0;
        let s = ux * ux + uy * uy;
        if s >= 1.0 || s == 0.0 {
            continue;
        }
        let f = 2.0 * (1.0 - s).sqrt();
        break (ux * f, uy * f, 1.0 - 2.0 * s);
    };

    let [mut px, mut py, mut pz] = src_m;
    let [lx, ly, lz] = l;
    let mut total_dist = 0.0_f64;
    let mut energy = ray_init_energy;
    let energy_floor = 1e-7 * ray_init_energy;

    for _ in 0..max_bounces {
        let mut hit = RayHitRecord {
            delay_s: -1.0,
            energy: 0.0,
            dir: [0.0; 3],
        };

        // Earliest intersection with the six shoebox planes (kernel branch order).
        let tx = if dx > 1e-12 {
            (lx - px) / dx
        } else if dx < -1e-12 {
            -px / dx
        } else {
            1e30
        };
        let ty = if dy > 1e-12 {
            (ly - py) / dy
        } else if dy < -1e-12 {
            -py / dy
        } else {
            1e30
        };
        let tz = if dz > 1e-12 {
            (lz - pz) / dz
        } else if dz < -1e-12 {
            -pz / dz
        } else {
            1e30
        };

        let mut t_hit = tx;
        let mut wall = if dx > 0.0 { 0usize } else { 1 };
        if ty < t_hit {
            t_hit = ty;
            wall = if dy > 0.0 { 2 } else { 3 };
        }
        if tz < t_hit {
            t_hit = tz;
            wall = if dz > 0.0 { 5 } else { 4 };
        }
        if t_hit <= 1e-9 || t_hit > 1e20 {
            out.push(hit);
            break;
        }

        // Receiver-sphere overlap on the segment `[0, t_hit]` (kernel arithmetic).
        let ocx = rx - px;
        let ocy = ry - py;
        let ocz = rz - pz;
        let proj = ocx * dx + ocy * dy + ocz * dz;
        if proj > 0.0 && proj < t_hit {
            let cx = ocx - proj * dx;
            let cy = ocy - proj * dy;
            let cz = ocz - proj * dz;
            let d2 = cx * cx + cy * cy + cz * cz;
            if d2 <= rec_rad2 {
                hit.delay_s = (total_dist + proj) / c;
                hit.energy = energy;
                hit.dir = [-dx, -dy, -dz];
            }
        }
        out.push(hit);

        // Advance to the wall (same clamp as the kernel) and apply mid-band absorption.
        px = (px + t_hit * dx).clamp(1e-6, lx - 1e-6);
        py = (py + t_hit * dy).clamp(1e-6, ly - 1e-6);
        pz = (pz + t_hit * dz).clamp(1e-6, lz - 1e-6);
        total_dist += t_hit;
        energy *= geom.walls[wall].refl_energy(ROOM_BAND_MID);

        // Specular reflection (sign flip), then the shared energy-floor break.
        if wall < 2 {
            dx = -dx;
        } else if wall < 4 {
            dy = -dy;
        } else {
            dz = -dz;
        }
        if energy < energy_floor {
            break;
        }

        // ISO-17497 Lambertian diffuse scattering blend. Cosine-weighted hemisphere via
        // Marsaglia unit-disk rejection plus the Lambert equal-area lift `z = sqrt(1 - s)`;
        // no libm trig, so this branch is bit-reproducible on both CPU and GPU.
        if rng.uniform() < geom.walls[wall].scatter(ROOM_BAND_MID) {
            let (sx, sy, sz) = loop {
                let ux = 2.0 * rng.uniform() - 1.0;
                let uy = 2.0 * rng.uniform() - 1.0;
                let s = ux * ux + uy * uy;
                if s >= 1.0 || s == 0.0 {
                    continue;
                }
                break (ux, uy, (1.0 - s).sqrt());
            };
            (dx, dy, dz) = match wall {
                0 => (-sz, sx, sy),
                1 => (sz, sx, sy),
                2 => (sx, -sz, sy),
                3 => (sx, sz, sy),
                4 => (sx, sy, sz),
                _ => (sx, sy, -sz),
            };
        }
    }
    out
}

/// Relative energy cutoff for a scattering ray, as a fraction of its launch energy.
const RAY_ENERGY_FLOOR_REL: f64 = 1e-7;
/// Acoustic receiver sphere radius (m). Shared physical specification binding both peers.
const RAY_RECEIVER_RADIUS_M: f64 = 0.45;
/// Reference frequency for molecular air absorption in the ray histogram (Hz).
const RAY_AIR_REFERENCE_HZ: f64 = 2_000.0;
/// Rays per partial-sum chunk in the native CPU reduction. Fixed, so the histogram does
/// not depend on core count or scheduling.
const CPU_REDUCE_CHUNK: usize = 64;

/// One late-arrival record produced by the native CPU tracer.
///
/// Only arrivals at bounce order >= 4 are emitted; orders 1–3 are supplied by the exact
/// image-source method. `dir` points from the listener back toward the incoming ray
/// (`-d`), the directional convention of [`eval_sh_order7`].
struct NativeArrival {
    delay_s: f64,
    energy: f64,
    dir: [f64; 3],
}

/// Trace one ray with the **native CPU** stochastic tracer for one octave band.
///
/// This is an independent CPU algorithm, not a reproduction of the CUDA kernel's
/// instruction stream:
///
/// * RNG: deterministic PCG64 ([`Pcg64`]), streamed by `ray_id`.
/// * Launch: stratified equal-solid-angle sampling — one jittered latitude band per ray
///   (`Δz = 2/n_rays`, so every ray owns an equal-area band) plus a jittered
///   golden-ratio azimuth, instead of per-ray rejection sampling.
/// * Geometry: slab ray/AABB exit test (`1/d` multiply), instead of the kernel's
///   per-plane division branches.
/// * Receiver: analytic ray/sphere quadratic discriminant, instead of the kernel's
///   projection-plus-perpendicular-distance test; both record the closest-approach
///   time as the shared delay specification.
/// * Scatter: Malley's inversion for a cosine-weighted hemisphere (`r = √u₁`,
///   `φ = 2πu₂`), instead of Marsaglia rejection plus the equal-area lift.
///
/// The physical specification is shared with the GPU peer: same shoebox, source,
/// listener, band-specific wall energy reflection `1 − clamp(α_b)`, band-specific
/// ISO-17497 scattering probability, energy floor, receiver radius, and
/// `(delay, energy, direction)` observable. Bands share the same per-ray random
/// stream (common random numbers): where their scattering thresholds agree the
/// paths coincide and only the energy differs, which is the ten-band transport
/// specification. The CPU and GPU Monte Carlo streams differ, so the two
/// implementations agree statistically, not bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn trace_native_ray_band(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    ray_id: usize,
    n_rays: usize,
    max_bounces: usize,
    seed: u64,
    band: usize,
) -> Vec<NativeArrival> {
    let band = band.min(ROOM_BAND_COUNT - 1);
    let c = atm.speed_of_sound_m_s();
    let [lx, ly, lz] = geom.dims_m;
    let [rx, ry, rz] = geom.listener_m;
    let rec_r2 = RAY_RECEIVER_RADIUS_M * RAY_RECEIVER_RADIUS_M;
    let wall_refl: [f64; 6] = std::array::from_fn(|w| geom.walls[w].refl_energy(band));
    let mut rng = Pcg64::new(seed, ray_id as u64);

    // Stratified equal-solid-angle launch: ray `ray_id` owns the latitude band
    // z ∈ [1 - 2(k+1)/n, 1 - 2k/n] and draws a uniform point inside it.
    let z = 1.0 - 2.0 * ((ray_id as f64 + rng.next_f64()) / (n_rays as f64));
    let phi = 2.0 * PI * (ray_id as f64 * 0.618_033_988_749_894_9 + rng.next_f64()).fract();
    let r_xy = (1.0 - z * z).max(0.0).sqrt();
    let mut d = [r_xy * phi.cos(), r_xy * phi.sin(), z];

    let [mut px, mut py, mut pz] = src_m;
    let mut energy = 1.0 / (n_rays as f64);
    let energy_floor = RAY_ENERGY_FLOOR_REL * energy;
    let mut total_dist = 0.0_f64;
    let mut out = Vec::new();

    for bounce in 0..max_bounces {
        // Slab method: on each axis the exit is the larger of the two plane
        // intersections; the smallest such exit is the first wall hit.
        let mut t_hit = f64::INFINITY;
        let mut wall = 0usize;
        if d[0] != 0.0 {
            let inv = 1.0 / d[0];
            t_hit = ((lx - px) * inv).max(-px * inv);
            wall = if d[0] > 0.0 { 0 } else { 1 };
        }
        if d[1] != 0.0 {
            let inv = 1.0 / d[1];
            let far = ((ly - py) * inv).max(-py * inv);
            if far < t_hit {
                t_hit = far;
                wall = if d[1] > 0.0 { 2 } else { 3 };
            }
        }
        if d[2] != 0.0 {
            let inv = 1.0 / d[2];
            let far = ((lz - pz) * inv).max(-pz * inv);
            if far < t_hit {
                t_hit = far;
                wall = if d[2] > 0.0 { 5 } else { 4 };
            }
        }
        if !t_hit.is_finite() || t_hit <= 1e-9 || t_hit > 1e20 {
            break;
        }

        // Analytic receiver-sphere test via the ray/sphere quadratic. The shared
        // physical observable is the time of **closest approach** to the listener
        // center (`b`), recorded when the ray passes within the receiver radius and the
        // closest point lies on the current segment. `disc >= 0` is exactly the
        // within-radius condition; the quadratic is the CPU's own algebraic route to it.
        let ox = rx - px;
        let oy = ry - py;
        let oz = rz - pz;
        let b = ox * d[0] + oy * d[1] + oz * d[2];
        let disc = b * b - (ox * ox + oy * oy + oz * oz - rec_r2);
        if bounce >= 3 && disc >= 0.0 && b > 0.0 && b < t_hit {
            out.push(NativeArrival {
                delay_s: (total_dist + b) / c,
                energy,
                dir: [-d[0], -d[1], -d[2]],
            });
        }

        // Advance onto the wall, keeping the point strictly inside the box.
        px = (px + t_hit * d[0]).clamp(1e-6, lx - 1e-6);
        py = (py + t_hit * d[1]).clamp(1e-6, ly - 1e-6);
        pz = (pz + t_hit * d[2]).clamp(1e-6, lz - 1e-6);
        total_dist += t_hit;
        energy *= wall_refl[wall];

        match wall {
            0 | 1 => d[0] = -d[0],
            2 | 3 => d[1] = -d[1],
            _ => d[2] = -d[2],
        }
        if energy < energy_floor {
            break;
        }

        // ISO-17497 diffuse scattering: cosine-weighted hemisphere by inversion
        // around the wall normal, with this band's scattering probability.
        if rng.next_f64() < geom.walls[wall].scatter(band) {
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let r = u1.max(0.0).sqrt();
            let ang = 2.0 * PI * u2;
            let (sx, sy) = (r * ang.cos(), r * ang.sin());
            let sz = (1.0 - u1).max(0.0).sqrt();
            // Inward wall normal: wall 0 (+x) scatters toward -x, wall 1 (-x) toward
            // +x, wall 2 (+y) toward -y, wall 3 (-y) toward +y, wall 4 (floor) toward
            // +z, wall 5 (ceiling) toward -z.
            d = match wall {
                0 => [-sz, sx, sy],
                1 => [sz, sx, sy],
                2 => [sx, -sz, sy],
                3 => [sx, sz, sy],
                4 => [sx, sy, sz],
                _ => [sx, sy, -sz],
            };
        }
    }
    out
}

/// Reduce per-ray native arrival lists to an Order-7 HOA histogram with the CPU's own
/// association order: each fixed [`CPU_REDUCE_CHUNK`]-ray block is summed into a fresh
/// partial histogram, then partials are folded in ascending chunk order. This differs
/// from the GPU's single ray-major running accumulator and is independent of thread
/// count and scheduling.
fn reduce_native_arrivals(
    per_ray: &[Vec<NativeArrival>],
    atm: &AtmosphereCfg,
    n_time_bins: usize,
    bin_dt_s: f64,
) -> Vec<[f64; HOA_CH]> {
    reduce_native_arrivals_at(per_ray, atm, n_time_bins, bin_dt_s, RAY_AIR_REFERENCE_HZ)
}

/// [`reduce_native_arrivals`] with an explicit molecular-air reference frequency.
fn reduce_native_arrivals_at(
    per_ray: &[Vec<NativeArrival>],
    atm: &AtmosphereCfg,
    n_time_bins: usize,
    bin_dt_s: f64,
    f_air_hz: f64,
) -> Vec<[f64; HOA_CH]> {
    let c = atm.speed_of_sound_m_s();
    let mut hist = vec![[0.0_f64; HOA_CH]; n_time_bins];
    if n_time_bins == 0 {
        return hist;
    }
    for chunk in per_ray.chunks(CPU_REDUCE_CHUNK) {
        let mut partial = vec![[0.0_f64; HOA_CH]; n_time_bins];
        for arrivals in chunk {
            for a in arrivals {
                if a.delay_s <= 0.0 || a.energy <= 0.0 {
                    continue;
                }
                let bin = (a.delay_s / bin_dt_s) as usize;
                if bin >= n_time_bins {
                    continue;
                }
                let az = a.dir[1].atan2(a.dir[0]);
                let el = a.dir[2].clamp(-1.0, 1.0).asin();
                let sh = eval_sh_order7(az, el);
                let air = atm.air_gain_linear(f_air_hz, a.delay_s * c);
                let w = a.energy * air * air;
                for (cell, s) in partial[bin].iter_mut().zip(sh.iter()) {
                    *cell += w * *s;
                }
            }
        }
        for (dst, src) in hist.iter_mut().zip(partial.iter()) {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    }
    hist
}

/// Ten-band HOA histogram row: one Order-7 HOA vector per octave band.
pub type RoomBandHistogram = [[f64; HOA_CH]; ROOM_BAND_COUNT];

#[allow(clippy::too_many_arguments)]
fn raytrace_hoa_histogram_band(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
    band: usize,
) -> Vec<[f64; HOA_CH]> {
    if n_rays == 0 || n_time_bins == 0 {
        return vec![[0.0_f64; HOA_CH]; n_time_bins];
    }
    let per_ray = crate::util::par_map_indexed(n_rays, |ray_id| {
        trace_native_ray_band(geom, atm, src_m, ray_id, n_rays, max_bounces, seed, band)
    });
    reduce_native_arrivals_at(
        &per_ray,
        atm,
        n_time_bins,
        bin_dt_s,
        ROOM_BANDS_HZ[band.min(ROOM_BAND_COUNT - 1)],
    )
}

#[allow(clippy::too_many_arguments)]
fn raytrace_hoa_histogram_band_serial(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
    band: usize,
) -> Vec<[f64; HOA_CH]> {
    if n_rays == 0 || n_time_bins == 0 {
        return vec![[0.0_f64; HOA_CH]; n_time_bins];
    }
    let per_ray: Vec<Vec<NativeArrival>> = (0..n_rays)
        .map(|ray_id| {
            trace_native_ray_band(geom, atm, src_m, ray_id, n_rays, max_bounces, seed, band)
        })
        .collect();
    reduce_native_arrivals_at(
        &per_ray,
        atm,
        n_time_bins,
        bin_dt_s,
        ROOM_BANDS_HZ[band.min(ROOM_BAND_COUNT - 1)],
    )
}

/// Native CPU Monte Carlo Acoustic Ray Tracer (orders 4..=max_bounces) with
/// Lambertian / ISO-17497 diffuse wall scattering, binning late directional energy
/// into an Order-7 HOA (`64`-channel) impulse histogram ready for MagLS decoding.
///
/// This legacy single-band entry point evaluates the `1 kHz` octave
/// ([`ROOM_BAND_MID`]) with the pre-ten-band broadband air-absorption reference
/// ([`RAY_AIR_REFERENCE_HZ`]). It remains the compatibility slice used by the
/// verifier and by the single-band CPU/GPU gates; the production hybrid tail
/// uses [`raytrace_hoa_histogram_per_band`].
///
/// The CPU peer is an independent implementation (`trace_native_ray_band`):
/// stratified equal-solid-angle launch, PCG64 streams, slab ray/box intersection,
/// analytic ray/sphere receiver, Malley cosine-weighted scatter, and a chunked
/// partial-sum reduction (`reduce_native_arrivals`). It shares only the physical
/// specification with `raytrace_hoa.cu`, so the two paths agree within a measured
/// Monte Carlo tolerance rather than bit-for-bit.
///
/// Parallelized across logical CPU cores via [`crate::util::par_map_indexed`]; the
/// reduction association is fixed, so the output is bit-identical regardless of CPU
/// core count or scheduling.
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> Vec<[f64; HOA_CH]> {
    if n_rays == 0 || n_time_bins == 0 {
        return vec![[0.0_f64; HOA_CH]; n_time_bins];
    }
    let per_ray = crate::util::par_map_indexed(n_rays, |ray_id| {
        trace_native_ray_band(
            geom,
            atm,
            src_m,
            ray_id,
            n_rays,
            max_bounces,
            seed,
            ROOM_BAND_MID,
        )
    });
    // Legacy broadband air reference (2 kHz), preserving the pre-ten-band gate.
    reduce_native_arrivals(&per_ray, atm, n_time_bins, bin_dt_s)
}

/// Ten-band native CPU Monte Carlo ray tracing (research doc §5 B1).
///
/// Every octave band owns an independent trace with its own wall absorption and
/// ISO-17497 scattering probability. Bands share the per-ray random stream
/// (common random numbers): where their scattering thresholds agree, paths
/// coincide and only the band energy differs. Molecular air absorption is
/// applied at the band centre, not at a single broadband reference. Returns one
/// [`RoomBandHistogram`] per time bin (`hist[bin][band][channel]`).
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_per_band(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> Vec<RoomBandHistogram> {
    let band_hists: Vec<Vec<[f64; HOA_CH]>> = (0..ROOM_BAND_COUNT)
        .map(|band| {
            raytrace_hoa_histogram_band(
                geom,
                atm,
                src_m,
                n_rays,
                max_bounces,
                n_time_bins,
                bin_dt_s,
                seed,
                band,
            )
        })
        .collect();
    let mut rows = vec![[[0.0_f64; HOA_CH]; ROOM_BAND_COUNT]; n_time_bins];
    for (bin, row) in rows.iter_mut().enumerate() {
        for (dst, hist) in row.iter_mut().zip(&band_hists) {
            *dst = hist[bin];
        }
    }
    rows
}

/// Strictly single-threaded reference evaluation of
/// [`raytrace_hoa_histogram_per_band`], used to verify thread-count invariance.
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_per_band_serial(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> Vec<RoomBandHistogram> {
    let band_hists: Vec<Vec<[f64; HOA_CH]>> = (0..ROOM_BAND_COUNT)
        .map(|band| {
            raytrace_hoa_histogram_band_serial(
                geom,
                atm,
                src_m,
                n_rays,
                max_bounces,
                n_time_bins,
                bin_dt_s,
                seed,
                band,
            )
        })
        .collect();
    let mut rows = vec![[[0.0_f64; HOA_CH]; ROOM_BAND_COUNT]; n_time_bins];
    for (bin, row) in rows.iter_mut().enumerate() {
        for (dst, hist) in row.iter_mut().zip(&band_hists) {
            *dst = hist[bin];
        }
    }
    rows
}

/// Run the GPU raytrace pass and rebuild the HOA histogram on the host. `Err` means
/// the GPU pass itself failed, which lets a caller that must prove the kernel ran
/// distinguish this from the deterministic CPU result.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn raytrace_hoa_histogram_on_runtime(
    rt: &g6me_cuda::G6meCudaRuntime,
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> std::result::Result<Vec<[f64; HOA_CH]>, String> {
    let c = atm.speed_of_sound_m_s();
    // raytrace_hoa.cu uses the same RoomGeometry wall ordering
    // (+x=0, -x=1, +y=2, -y=3, -z=4, +z=5), so the material tables pass through.
    let wall_refl = std::array::from_fn(|idx| geom.walls[idx].refl_energy(ROOM_BAND_MID));
    let wall_scatter = std::array::from_fn(|idx| geom.walls[idx].scatter(ROOM_BAND_MID));
    let (delays, energies, dirs) = rt.run_raytrace_hoa_pass_with_scattering(
        geom.dims_m,
        src_m,
        geom.listener_m,
        wall_refl,
        wall_scatter,
        n_rays,
        max_bounces,
        c,
        RAY_RECEIVER_RADIUS_M,
        seed,
    )?;
    let mut hist = vec![[0.0_f64; HOA_CH]; n_time_bins];
    // GPU-side reduction: the kernel's ray-major / bounce-ascending record layout is
    // folded with one running accumulator. The CPU peer deliberately uses a different
    // association (chunked partial sums), so parity is bounded by the measured
    // Monte Carlo tolerance rather than by bit identity.
    for ray_id in 0..n_rays {
        let base = ray_id * max_bounces;
        for bounce in 0..max_bounces {
            let idx = base + bounce;
            let inc = [dirs[idx * 3], dirs[idx * 3 + 1], dirs[idx * 3 + 2]];
            accumulate_hoa_hit(
                &mut hist,
                bounce,
                delays[idx],
                energies[idx],
                inc,
                atm,
                c,
                n_time_bins,
                bin_dt_s,
            );
        }
    }
    Ok(hist)
}

/// GPU-only raytrace histogram: fails instead of falling back to the CPU. This is the
/// entry point for parity checks that must prove the device kernel produced the
/// result.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_on_cuda(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> std::result::Result<Vec<[f64; HOA_CH]>, String> {
    let rt = crate::blackwell::shared_cuda_runtime()
        .ok_or_else(|| "CUDA runtime unavailable".to_string())?;
    raytrace_hoa_histogram_on_runtime(
        &rt,
        geom,
        atm,
        src_m,
        n_rays,
        max_bounces,
        n_time_bins,
        bin_dt_s,
        seed,
    )
}

/// Optional GPU-offloaded deterministic Monte Carlo Acoustic Ray Tracer
/// (`raytrace_hoa.cu`). Uses `g6me-cuda` when `use_cuda` is true and a live NVIDIA GPU
/// is detected, otherwise falls back deterministically to the independent native CPU
/// tracer [`raytrace_hoa_histogram`]. Every dispatch is recorded in
/// [`crate::cuda_report`].
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_with_cuda(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
    use_cuda: bool,
) -> Vec<[f64; HOA_CH]> {
    #[cfg(feature = "cuda")]
    if use_cuda && n_rays > 0 && n_time_bins > 0 && max_bounces > 0 {
        match crate::blackwell::shared_cuda_runtime() {
            Some(rt) => match raytrace_hoa_histogram_on_runtime(
                &rt,
                geom,
                atm,
                src_m,
                n_rays,
                max_bounces,
                n_time_bins,
                bin_dt_s,
                seed,
            ) {
                Ok(hist) => {
                    crate::cuda_report::record_gpu(
                        "raytrace-hoa",
                        format!("{n_rays} rays x {max_bounces} bounces"),
                    );
                    return hist;
                }
                Err(e) => crate::cuda_report::record_cpu_fallback("raytrace-hoa", e),
            },
            None => {
                crate::cuda_report::record_cpu_fallback("raytrace-hoa", "CUDA runtime unavailable")
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = use_cuda;
    raytrace_hoa_histogram(
        geom,
        atm,
        src_m,
        n_rays,
        max_bounces,
        n_time_bins,
        bin_dt_s,
        seed,
    )
}

/// Run the GPU raytrace pass once per octave band and rebuild the ten-band HOA
/// histogram on the host (research doc §5 B1).
///
/// `raytrace_hoa.cu` is the band-native peer: each launch receives that band's
/// six wall reflection coefficients and six ISO-17497 scattering probabilities
/// and applies them at every bounce. Bands share `seed` (common random numbers),
/// so where their scattering thresholds agree the trajectories coincide and
/// only the energy differs. The host reduction evaluates ISO 9613-1 molecular
/// absorption at the band centre. `Err` means at least one GPU launch failed;
/// no band is silently produced on the CPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn raytrace_hoa_histogram_per_band_on_runtime(
    rt: &g6me_cuda::G6meCudaRuntime,
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> std::result::Result<Vec<RoomBandHistogram>, String> {
    let c = atm.speed_of_sound_m_s();
    let mut rows = vec![[[0.0_f64; HOA_CH]; ROOM_BAND_COUNT]; n_time_bins];
    for band in 0..ROOM_BAND_COUNT {
        let wall_refl = std::array::from_fn(|w| geom.walls[w].refl_energy(band));
        let wall_scatter = std::array::from_fn(|w| geom.walls[w].scatter(band));
        let (delays, energies, dirs) = rt.run_raytrace_hoa_pass_with_scattering(
            geom.dims_m,
            src_m,
            geom.listener_m,
            wall_refl,
            wall_scatter,
            n_rays,
            max_bounces,
            c,
            RAY_RECEIVER_RADIUS_M,
            seed,
        )?;
        let mut hist = vec![[0.0_f64; HOA_CH]; n_time_bins];
        for ray_id in 0..n_rays {
            let base = ray_id * max_bounces;
            for bounce in 0..max_bounces {
                let idx = base + bounce;
                let inc = [dirs[idx * 3], dirs[idx * 3 + 1], dirs[idx * 3 + 2]];
                accumulate_hoa_hit_at(
                    &mut hist,
                    bounce,
                    delays[idx],
                    energies[idx],
                    inc,
                    atm,
                    c,
                    ROOM_BANDS_HZ[band],
                    n_time_bins,
                    bin_dt_s,
                );
            }
        }
        for (bin, row) in hist.iter().enumerate() {
            rows[bin][band] = *row;
        }
    }
    Ok(rows)
}

/// GPU-only ten-band raytrace histogram: fails instead of falling back to the CPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_per_band_on_cuda(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> std::result::Result<Vec<RoomBandHistogram>, String> {
    let rt = crate::blackwell::shared_cuda_runtime()
        .ok_or_else(|| "CUDA runtime unavailable".to_string())?;
    raytrace_hoa_histogram_per_band_on_runtime(
        &rt,
        geom,
        atm,
        src_m,
        n_rays,
        max_bounces,
        n_time_bins,
        bin_dt_s,
        seed,
    )
}

/// Optional GPU-offloaded ten-band Monte Carlo ray tracing. Uses the per-band
/// `raytrace_hoa.cu` launches when `use_cuda` is true and a live NVIDIA GPU is
/// detected, otherwise falls back deterministically to
/// [`raytrace_hoa_histogram_per_band`]. Every dispatch is recorded in
/// [`crate::cuda_report`].
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_per_band_with_cuda(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
    use_cuda: bool,
) -> Vec<RoomBandHistogram> {
    #[cfg(feature = "cuda")]
    if use_cuda && n_rays > 0 && n_time_bins > 0 && max_bounces > 0 {
        match crate::blackwell::shared_cuda_runtime() {
            Some(rt) => match raytrace_hoa_histogram_per_band_on_runtime(
                &rt,
                geom,
                atm,
                src_m,
                n_rays,
                max_bounces,
                n_time_bins,
                bin_dt_s,
                seed,
            ) {
                Ok(hist) => {
                    crate::cuda_report::record_gpu(
                        "raytrace-hoa-bands",
                        format!("{ROOM_BAND_COUNT} bands x {n_rays} rays x {max_bounces} bounces"),
                    );
                    return hist;
                }
                Err(e) => crate::cuda_report::record_cpu_fallback("raytrace-hoa-bands", e),
            },
            None => crate::cuda_report::record_cpu_fallback(
                "raytrace-hoa-bands",
                "CUDA runtime unavailable",
            ),
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = use_cuda;
    raytrace_hoa_histogram_per_band(
        geom,
        atm,
        src_m,
        n_rays,
        max_bounces,
        n_time_bins,
        bin_dt_s,
        seed,
    )
}

/// Strictly single-threaded reference evaluation of [`raytrace_hoa_histogram`] (used to
/// verify bit-for-bit thread-count invariance of the native CPU tracer).
#[allow(clippy::too_many_arguments)]
pub fn raytrace_hoa_histogram_serial(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    src_m: [f64; 3],
    n_rays: usize,
    max_bounces: usize,
    n_time_bins: usize,
    bin_dt_s: f64,
    seed: u64,
) -> Vec<[f64; HOA_CH]> {
    if n_rays == 0 || n_time_bins == 0 {
        return vec![[0.0_f64; HOA_CH]; n_time_bins];
    }
    let per_ray: Vec<Vec<NativeArrival>> = (0..n_rays)
        .map(|ray_id| {
            trace_native_ray_band(
                geom,
                atm,
                src_m,
                ray_id,
                n_rays,
                max_bounces,
                seed,
                ROOM_BAND_MID,
            )
        })
        .collect();
    reduce_native_arrivals(&per_ray, atm, n_time_bins, bin_dt_s)
}

/// Per-band W-channel energy envelope `sqrt(sum_i h_i²)` for every 4 ms
/// histogram bin, plus each band's own maximum.
///
/// The envelope modulates the late tail, so normalising each band by its own
/// maximum preserves that band's decay rate. The pre-fix code divided every band
/// by one global maximum and floored the result at `0.15` (−16.5 dB), at which
/// point a strongly absorbing band stopped decaying and its tail inherited
/// `base_l`/`base_r`'s broadband rate instead of its own RT60.
fn band_energy_envelopes(
    hist_bands: &[RoomBandHistogram],
    n_bins: usize,
) -> (Vec<Vec<f64>>, [f64; ROOM_BAND_COUNT]) {
    let mut band_bin_energies = vec![vec![0.0_f64; n_bins]; ROOM_BAND_COUNT];
    let mut band_max_e = [0.0_f64; ROOM_BAND_COUNT];
    for (t, row) in hist_bands.iter().take(n_bins).enumerate() {
        for band in 0..ROOM_BAND_COUNT {
            let e = row[band].iter().map(|x| x * x).sum::<f64>().sqrt();
            band_bin_energies[band][t] = e;
            band_max_e[band] = band_max_e[band].max(e);
        }
    }
    (band_bin_energies, band_max_e)
}

/// Per-band relative amplitude envelopes in `[0, 1]`, each band divided by its
/// own maximum, alongside the raw maxima for the "any late energy" gate.
///
/// This is the production form of the envelope modulation: a band whose maximum
/// is at or below the numerical floor (`1e-12`) normalises to all zeros instead
/// of dividing 0/0, and every other band keeps its own decay slope.
fn band_relative_envelopes(
    hist_bands: &[RoomBandHistogram],
    n_bins: usize,
) -> (Vec<Vec<f64>>, [f64; ROOM_BAND_COUNT]) {
    let (energies, maxima) = band_energy_envelopes(hist_bands, n_bins);
    let relative = energies
        .iter()
        .enumerate()
        .map(|(band, energies)| {
            let scale = if maxima[band] > 1e-12 {
                1.0 / maxima[band]
            } else {
                0.0
            };
            energies.iter().map(|e| (e * scale).sqrt()).collect()
        })
        .collect();
    (relative, maxima)
}

/// High-pass shading applied to the decoded late-tail spectrum above
/// `f_xover` (below it the FDTD modal layer owns the response).
///
/// Deliberately frequency-only: molecular air absorption is **not** applied
/// here. The ten-band ray reduction already weights every arrival by
/// `air_gain_linear(f_band, path_length)²` (ISO 9613-1 at the band centre), so
/// the extra `air_gain_linear(f, zone_dist_m.min(60))` factor this function
/// replaced attenuated the late tail twice.
#[inline]
fn tail_highpass_shading(f_hz: f64, f_xover: f64) -> f64 {
    let x_hp = f_hz / f_xover.max(200.0);
    (x_hp * x_hp) / (1.0 + x_hp * x_hp)
}

/// Synthesize a complete **Hybrid 3D Room BRIR** (`spec_l_add`, `spec_r_add`) for a virtual
/// speaker at `(az_deg, el_deg, distance_m)` by fusing:
/// 1. 3D ISM early reflections (orders 1–2) with wall absorption + molecular `O₂`/`N₂` air,
/// 2. Analytic BTM edge diffraction around room/desk edges,
/// 3. Reciprocal 3D FDTD modal room response below `500 Hz` (passed in from the 2-run
///    acoustic-reciprocity FDTD cache),
/// 4. Deterministic native Monte Carlo ray-traced Order-7 HOA late reflections decoded via MagLS,
///    optionally enriched with Paraunitary Scattering FDN + Velvet Noise diffusion!
#[allow(clippy::too_many_arguments)]
pub fn synthesize_hybrid_room_reflections(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    magls: &MagLsDecoder,
    freqs: &[f64],
    fs: f64,
    pre_delay_s: f64,
    src_m: [f64; 3],
    fdtd_l_low: Option<&[f64]>,
    fdtd_r_low: Option<&[f64]>,
    fdtd_dt_s: f64,
    use_fdn_velvet: bool,
    quality: RoomQuality,
    tail_duration_s: f64,
    seed: u64,
    plan: &FftPlan,
) -> (Vec<Complex>, Vec<Complex>) {
    synthesize_hybrid_room_reflections_with_cuda(
        geom,
        atm,
        magls,
        freqs,
        fs,
        pre_delay_s,
        src_m,
        fdtd_l_low,
        fdtd_r_low,
        fdtd_dt_s,
        use_fdn_velvet,
        quality,
        tail_duration_s,
        seed,
        plan,
        false,
    )
}

/// Same as [`synthesize_hybrid_room_reflections`] with opt-in zero-FMA CUDA GPU offload for
/// stochastic ray tracing (`raytrace_hoa.cu`; GPU path is the CUDA peer of the native CPU
/// tracer, parity bounded by a measured Monte Carlo tolerance).
#[allow(clippy::too_many_arguments)]
pub fn synthesize_hybrid_room_reflections_with_cuda(
    geom: &RoomGeometry,
    atm: &AtmosphereCfg,
    magls: &MagLsDecoder,
    freqs: &[f64],
    fs: f64,
    pre_delay_s: f64,
    src_m: [f64; 3],
    fdtd_l_low: Option<&[f64]>,
    fdtd_r_low: Option<&[f64]>,
    fdtd_dt_s: f64,
    use_fdn_velvet: bool,
    quality: RoomQuality,
    tail_duration_s: f64,
    seed: u64,
    plan: &FftPlan,
    use_cuda: bool,
) -> (Vec<Complex>, Vec<Complex>) {
    let nb = freqs.len();
    let n_fft = (nb - 1) * 2;
    let c = atm.speed_of_sound_m_s();
    let mut out_l = vec![Complex::ZERO; nb];
    let mut out_r = vec![Complex::ZERO; nb];

    let d_dir = ((src_m[0] - geom.listener_m[0]).powi(2)
        + (src_m[1] - geom.listener_m[1]).powi(2)
        + (src_m[2] - geom.listener_m[2]).powi(2))
    .sqrt()
    .max(0.1);
    let t_dir = d_dir / c;

    // Compute physical FDTD 6-PPW crossover frequency directly from `fdtd_dt_s` (`lambda = 0.55`).
    let f_xover = if fdtd_l_low.is_some() && fdtd_dt_s > 1e-6 {
        (0.55 / (6.0 * fdtd_dt_s)).clamp(150.0, 680.0)
    } else {
        500.0
    };
    let tail_dur = tail_duration_s.clamp(0.144, ((n_fft as f64) / fs) - pre_delay_s - 0.02);

    // 1. Layer 1a: 3D Image-Source Method (orders 1–3, closing the 3rd-order gap before ray tracing).
    let ism_taps = image_sources_3d(geom, src_m, c, 3);
    for tap in &ism_taps {
        if tap.delay_s > tail_dur {
            continue;
        }
        // Geometric 1/r attenuation relative to direct sound.
        let geom_atten = (d_dir / tap.dist_m.max(0.2)).min(0.85);
        // Decode tap incidence direction through Order-7 MagLS decoder.
        let (hl_dir, hr_dir) = magls.decode_direction(tap.az_deg, tap.el_deg);
        let delay_total = pre_delay_s + tap.delay_s;
        for (k, &f) in freqs.iter().enumerate() {
            // High-pass ISM above `f_xover` where FDTD owns the modal bass when present.
            let x_hp = f / f_xover;
            let hp = if fdtd_l_low.is_some() {
                (x_hp * x_hp) / (1.0 + x_hp * x_hp)
            } else {
                1.0
            };
            let mut wall_g = 1.0_f64;
            for (w_idx, &hits) in tap.wall_hits.iter().enumerate() {
                if hits > 0 {
                    wall_g *= geom.walls[w_idx].refl_amp(f).powi(hits as i32);
                }
            }
            let air_g = atm.air_gain_linear(f, tap.dist_m);
            let gain = geom_atten * wall_g * air_g * hp;
            let ph = Complex::from_polar(gain, -2.0 * PI * f * delay_total);
            out_l[k] = out_l[k].add(hl_dir[k].mul(ph));
            out_r[k] = out_r[k].add(hr_dir[k].mul(ph));
        }
    }

    // 2. Layer 2: Analytic BTM Edge Diffraction around desk/baffle edges (time domain -> FFT).
    if !geom.edges.is_empty() {
        let mut btm_l = vec![0.0_f64; n_fft];
        let mut btm_r = vec![0.0_f64; n_fft];
        let ear_l = [
            geom.listener_m[0],
            geom.listener_m[1] + 0.0875,
            geom.listener_m[2],
        ];
        let ear_r = [
            geom.listener_m[0],
            geom.listener_m[1] - 0.0875,
            geom.listener_m[2],
        ];
        for edge in &geom.edges {
            add_btm_edge_diffraction(
                edge,
                src_m,
                ear_l,
                c,
                fs as u32,
                t_dir,
                pre_delay_s,
                &mut btm_l,
            );
            add_btm_edge_diffraction(
                edge,
                src_m,
                ear_r,
                c,
                fs as u32,
                t_dir,
                pre_delay_s,
                &mut btm_r,
            );
        }
        let mut scratch = vec![Complex::ZERO; n_fft];
        let mut bl_spec = vec![Complex::ZERO; nb];
        let mut br_spec = vec![Complex::ZERO; nb];
        plan.rfft(&btm_l, &mut scratch, &mut bl_spec);
        plan.rfft(&btm_r, &mut scratch, &mut br_spec);
        for k in 0..nb {
            // Gentle desk-diffraction blend (preserves flat DFEQ while adding real physical edge cue).
            out_l[k] = out_l[k].add(bl_spec[k].scale(0.35));
            out_r[k] = out_r[k].add(br_spec[k].scale(0.35));
        }
    }

    // 3. Layer 1b: Parallel deterministic native Monte Carlo Ray Tracing ->
    //    time-varying Order-7 HOA spatial-temporal histogram (`4 ms` bins across `tail_dur`)
    //    decoded through MagLS in 4 directional time zones modulated by the measured ray-traced
    //    energy envelope + Paraunitary Scattering FDN / ternary Velvet Noise diffusion.
    let bin_dt_s = 0.004; // 4 ms HOA energy histogram bins
    let n_bins = ((tail_dur / bin_dt_s).round() as usize).clamp(36, 200);
    let (n_rays, max_bounces) = match quality {
        RoomQuality::Preview => (2_048usize, 48usize),
        RoomQuality::Production => (50_000usize, 72usize),
    };
    let hist_bands = raytrace_hoa_histogram_per_band_with_cuda(
        geom,
        atm,
        src_m,
        n_rays,
        max_bounces,
        n_bins,
        bin_dt_s,
        seed,
        use_cuda,
    );
    // Per-band W-channel relative amplitude envelopes, each band normalised by
    // its own maximum. A numerical floor only (~1e-12): a band with no late
    // energy contributes nothing rather than dividing 0/0.
    let (band_rel_env, band_max_e) = band_relative_envelopes(&hist_bands, n_bins);
    let max_bin_e = band_max_e.iter().copied().fold(0.0_f64, f64::max);

    if max_bin_e > 1e-12 {
        let tail_len = ((tail_dur * fs).round() as usize).min(n_fft.saturating_sub(512));
        let (base_l, base_r) = if use_fdn_velvet {
            let mut fdn =
                ScatteringFdn::new(fs as u32, geom.target_rt60_s, geom.target_rt60_s * 0.55);
            let (fl, fr) = fdn.synthesize_stereo_tail(tail_len);
            let vl = generate_velvet_noise(tail_len, fs as u32, 2_200.0, seed ^ 0xA1);
            let vr = generate_velvet_noise(tail_len, fs as u32, 2_200.0, seed ^ 0xB2);
            let mut bl = vec![0.0_f64; tail_len];
            let mut br = vec![0.0_f64; tail_len];
            for i in 0..tail_len {
                let env = (-6.907_755 * (i as f64) / (geom.target_rt60_s * fs)).exp();
                bl[i] = 0.65 * fl[i] + 0.35 * vl[i] * env * 0.04;
                br[i] = 0.65 * fr[i] + 0.35 * vr[i] * env * 0.04;
            }
            (bl, br)
        } else {
            let vl = generate_velvet_noise(tail_len, fs as u32, 2_200.0, seed ^ 0xC3);
            let vr = generate_velvet_noise(tail_len, fs as u32, 2_200.0, seed ^ 0xD4);
            let mut bl = vec![0.0_f64; tail_len];
            let mut br = vec![0.0_f64; tail_len];
            for i in 0..tail_len {
                let env = (-6.907_755 * (i as f64) / (geom.target_rt60_s * fs)).exp();
                bl[i] = vl[i] * env * 0.03;
                br[i] = vr[i] * env * 0.03;
            }
            (bl, br)
        };

        // Octave-band spectral partition: adjacent triangular weights sum to
        // one at every frequency, so the per-band tails combine into the same
        // overall tail level the single-band renderer used.
        let band_weights: Vec<[f64; ROOM_BAND_COUNT]> = freqs
            .iter()
            .map(|&f| std::array::from_fn(|band| octave_band_weight(f, band)))
            .collect();

        const N_ZONES: usize = 4;
        let bins_per_zone = n_bins / N_ZONES;
        let pre_smp = ((pre_delay_s + 0.012) * fs).round() as usize;
        let mut scratch = vec![Complex::ZERO; n_fft];
        let mut sl_zone = vec![Complex::ZERO; nb];
        let mut sr_zone = vec![Complex::ZERO; nb];
        let mut zone_l = vec![0.0_f64; n_fft];
        let mut zone_r = vec![0.0_f64; n_fft];

        for z in 0..N_ZONES {
            let b0 = z * bins_per_zone;
            let b1 = if z + 1 == N_ZONES {
                n_bins
            } else {
                (z + 1) * bins_per_zone
            };
            // High-pass only (FDTD owns the modal band). Molecular air
            // absorption is owned by the ray reduction, which weights every
            // arrival by `air_gain_linear(f_band, path_length)²`; a second
            // zone-mean air gain here attenuated the late tail twice.
            let hp_gain: Vec<f64> = freqs
                .iter()
                .map(|&f| tail_highpass_shading(f, f_xover))
                .collect();

            for band in 0..ROOM_BAND_COUNT {
                // A band with no recorded late energy contributes nothing; skip
                // its FFT rather than render a silent zone.
                if band_max_e[band] <= 1e-12 {
                    continue;
                }
                let mut sh_zone = [0.0_f64; HOA_CH];
                for row in hist_bands.iter().take(b1).skip(b0) {
                    for (acc, &val) in sh_zone.iter_mut().zip(row[band].iter()) {
                        *acc += val;
                    }
                }
                let z_norm = sh_zone.iter().map(|x| x * x).sum::<f64>().sqrt();
                if z_norm <= 1e-12 {
                    continue;
                }
                for x in &mut sh_zone {
                    *x /= z_norm;
                }
                let (hl_hoa, hr_hoa) = magls.decode_sh_spectrum(&sh_zone);

                zone_l.fill(0.0);
                zone_r.fill(0.0);
                let s0 = ((b0 as f64) * bin_dt_s * fs).round() as usize;
                let s1 = (((b1 as f64) * bin_dt_s * fs).round() as usize).min(tail_len);
                let xfade = ((0.004 * fs).round() as usize).max(16);
                for i in s0..s1 {
                    let dst = pre_smp + i;
                    if dst >= n_fft {
                        break;
                    }
                    let bin_idx = (((i as f64) / fs) / bin_dt_s).floor() as usize;
                    // Each band relative to its own peak: the decay rate (and so
                    // the RT60) of this band's envelope survives into the tail.
                    let e_rel = band_rel_env[band][bin_idx.min(n_bins - 1)];
                    // Raised-cosine zone edge fade prevents clicks between HOA spatial zones.
                    let mut win = 1.0_f64;
                    if z > 0 && i - s0 < xfade {
                        win *= 0.5 * (1.0 - (PI * ((i - s0) as f64) / (xfade as f64)).cos());
                    }
                    if z + 1 < N_ZONES && s1 - i <= xfade {
                        win *= 0.5 * (1.0 - (PI * ((s1 - i) as f64) / (xfade as f64)).cos());
                    }
                    zone_l[dst] = base_l[i] * e_rel * win;
                    zone_r[dst] = base_r[i] * e_rel * win;
                }

                plan.rfft(&zone_l, &mut scratch, &mut sl_zone);
                plan.rfft(&zone_r, &mut scratch, &mut sr_zone);
                for (k, _f) in freqs.iter().enumerate() {
                    let w = band_weights[k][band];
                    if w <= 0.0 {
                        continue;
                    }
                    let scale = 0.18 * hp_gain[k] * w;
                    out_l[k] = out_l[k].add(sl_zone[k].mul(hl_hoa[k]).scale(scale));
                    out_r[k] = out_r[k].add(sr_zone[k].mul(hr_hoa[k]).scale(scale));
                }
            }
        }
    }

    // 4. Layer 3 (< f_xover): Stitch reciprocal 3D FDTD modal response via 4th-order lowpass.
    if let (Some(fl_raw), Some(fr_raw)) = (fdtd_l_low, fdtd_r_low) {
        let mut td_l = vec![0.0_f64; n_fft];
        let mut td_r = vec![0.0_f64; n_fft];
        let fdtd_dur = ((fl_raw.len() as f64) * fdtd_dt_s).min(tail_dur);
        let max_len = (fdtd_dur * fs).round() as usize;
        let pre_smp = (pre_delay_s * fs).round() as usize;
        // Resample FDTD time series from dt_s to 1/fs via linear interpolation & strip direct arrival.
        let peak_l = fl_raw
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let peak_r = fr_raw
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        let peak = peak_l.max(peak_r);
        let direct_skip_s = t_dir + 0.008;
        for i in 0..max_len.min(n_fft.saturating_sub(pre_smp)) {
            let t = direct_skip_s + (i as f64) / fs;
            let idx_f = t / fdtd_dt_s;
            let i0 = idx_f.floor() as usize;
            if i0 + 1 >= fl_raw.len() || i0 + 1 >= fr_raw.len() {
                break;
            }
            let frac = idx_f - (i0 as f64);
            let vl = fl_raw[i0] * (1.0 - frac) + fl_raw[i0 + 1] * frac;
            let vr = fr_raw[i0] * (1.0 - frac) + fr_raw[i0 + 1] * frac;
            td_l[pre_smp + i] = (vl / peak) * 0.12;
            td_r[pre_smp + i] = (vr / peak) * 0.12;
        }
        let mut scratch = vec![Complex::ZERO; n_fft];
        let mut fl_spec = vec![Complex::ZERO; nb];
        let mut fr_spec = vec![Complex::ZERO; nb];
        plan.rfft(&td_l, &mut scratch, &mut fl_spec);
        plan.rfft(&td_r, &mut scratch, &mut fr_spec);
        for (k, &f) in freqs.iter().enumerate() {
            // 4th-order low-pass below `f_xover` (modal regime) and DC high-pass above 25 Hz.
            let x_lp = f / f_xover;
            let lp = 1.0 / (1.0 + x_lp * x_lp);
            let x_dc = f / 25.0;
            let dc = (x_dc * x_dc) / (1.0 + x_dc * x_dc);
            let w = lp * lp * dc;
            out_l[k] = out_l[k].add(fl_spec[k].scale(w));
            out_r[k] = out_r[k].add(fr_spec[k].scale(w));
        }
    }

    (out_l, out_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paraunitary_fdn_matrix_is_exact_orthogonal_and_lossless_conserves_energy() {
        let u = ScatteringFdn::paraunitary_matrix();
        let mut max_err = 0.0_f64;
        for i in 0..8 {
            for j in 0..8 {
                let mut dot = 0.0_f64;
                for row in &u {
                    dot += row[i] * row[j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((dot - expected).abs());
            }
        }
        assert!(
            max_err < 1e-13,
            "FDN scattering matrix U^T U != I: max_err={max_err:e}"
        );

        // Lossless FDN (RT60 = inf) must conserve exact total delay-line energy!
        let mut fdn = ScatteringFdn::new(48_000, f64::INFINITY, f64::INFINITY);
        let inj = [1.0, -0.5, 0.25, 0.75, -0.4, 0.3, -0.8, 0.2];
        let e_in: f64 = inj.iter().map(|x| x * x).sum();
        let _ = fdn.tick(inj);
        for _ in 0..4_000 {
            let _ = fdn.tick([0.0; 8]);
        }
        let e_stored: f64 = fdn
            .lines
            .iter()
            .flat_map(|line| line.iter())
            .map(|x| x * x)
            .sum();
        assert!(
            (e_stored - e_in).abs() < 1e-11,
            "lossless paraunitary FDN energy drifted: in={e_in}, stored={e_stored}"
        );
    }

    #[test]
    fn velvet_noise_is_strictly_ternary_sparse_and_deterministic() {
        let v1 = generate_velvet_noise(4800, 48_000, 2_000.0, 0x1234_5678);
        let v2 = generate_velvet_noise(4800, 48_000, 2_000.0, 0x1234_5678);
        assert_eq!(v1, v2, "Velvet Noise must be bit-exact deterministic");
        let mut nz = 0usize;
        for &s in &v1 {
            assert!(
                s == 0.0 || s == 1.0 || s == -1.0,
                "non-ternary sample in Velvet Noise: {s}"
            );
            if s != 0.0 {
                nz += 1;
            }
        }
        // 0.1 s at 2000 Hz density => ~200 non-zero pulses out of 4800 samples (~4.2% sparsity).
        assert!((180..=220).contains(&nz), "unexpected pulse count: {nz}");
    }

    #[test]
    fn fdtd_3d_wave_pde_satisfies_exact_acoustic_reciprocity() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let solver = FdtdSolver3d::new(&geom, 344.854, 0.35);
        let p_ear = solver.coord_to_voxel([2.6, 2.58, 1.25]);
        let p_spk = solver.coord_to_voxel([4.3, 3.5, 1.3]);
        let h_12 = &solver.simulate(p_ear, &[p_spk], 90)[0];
        let h_21 = &solver.simulate(p_spk, &[p_ear], 90)[0];
        let peak = h_12.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            peak > 1e-4,
            "3D FDTD wave did not reach receiver: peak={peak:e}"
        );
        let max_diff = h_12
            .iter()
            .zip(h_21.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-13,
            "3D FDTD acoustic reciprocity violated: max_diff={max_diff:e}"
        );
    }

    #[test]
    fn btm_edge_diffraction_is_causal_and_nonzero_around_mixing_desk() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let src: [f64; 3] = [4.3, 3.5, 1.25];
        let rec: [f64; 3] = [2.6, 2.58, 1.25];
        let d_dir =
            ((src[0] - rec[0]).powi(2) + (src[1] - rec[1]).powi(2) + (src[2] - rec[2]).powi(2))
                .sqrt();
        let t_dir = d_dir / 344.854;
        let pre_s: f64 = 0.005;
        let pre_smp = (pre_s * 48_000.0).round() as usize;
        let mut ir = vec![0.0_f64; 1024];
        add_btm_edge_diffraction(
            &geom.edges[0],
            src,
            rec,
            344.854,
            48_000,
            t_dir,
            pre_s,
            &mut ir,
        );
        // Strictly causal: zero energy before direct arrival pre_smp!
        let pre_e: f64 = ir[..pre_smp].iter().map(|x| x * x).sum();
        let post_e: f64 = ir[pre_smp..].iter().map(|x| x * x).sum();
        assert_eq!(pre_e, 0.0, "BTM edge diffraction had non-causal energy");
        assert!(
            post_e > 1e-8,
            "BTM desk edge produced zero diffraction energy"
        );
    }

    #[test]
    fn native_raytracer_is_thread_count_invariant_and_populates_order7_hoa_histogram() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let h_par = raytrace_hoa_histogram(&geom, &atm, src, 512, 24, 24, 0.004, 0xCAFE_BABE);
        let h_ser =
            raytrace_hoa_histogram_serial(&geom, &atm, src, 512, 24, 24, 0.004, 0xCAFE_BABE);
        assert_eq!(h_par.len(), h_ser.len());
        for (b, (rp, rs)) in h_par.iter().zip(&h_ser).enumerate() {
            for (c, (&x, &y)) in rp.iter().zip(rs.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "native raytracer parallel vs serial bit mismatch at bin={b} ch={c}"
                );
            }
        }
        let tot_w: f64 = h_par.iter().map(|b| b[0]).sum();
        assert!(
            tot_w > 1e-6,
            "ray tracer recorded zero HOA W-channel energy"
        );
        // The seed must actually drive the PCG64 streams: a seed-plumbing regression
        // would otherwise leave this deterministic tracer silently seed-independent.
        let h_other = raytrace_hoa_histogram(&geom, &atm, src, 512, 24, 24, 0.004, 0xCAFE_BABF);
        assert!(
            h_par.iter().zip(&h_other).any(|(a, b)| a
                .iter()
                .zip(b.iter())
                .any(|(&x, &y)| x.to_bits() != y.to_bits())),
            "native raytracer ignored the seed"
        );
    }

    /// Physical agreement metrics between two HOA histograms. Cell-by-cell L1 is not
    /// usable: the observable has `n_bins × 64` mostly-sparse cells, far more than the
    /// number of late receiver hits, so two independent Monte Carlo streams disagree on
    /// most fine cells even with identical physics. These aggregates are the physical
    /// quantities the two peers must reproduce:
    ///
    /// * `max_abs_cell` — largest absolute per-(bin, channel) difference (reported only).
    /// * `rel_w_total` — relative difference of total omnidirectional W energy
    ///   (`W = channel 0 = 1` for N3D Order-0 SH, so `Σ_t h[t][0]` is captured energy).
    /// * `band_l1` — L1 distance between the normalized W-energy temporal envelopes in
    ///   `n_bands` equal-width time bands (`Σ_b |p_cpu[b] - p_gpu[b]|` with each
    ///   envelope normalized to unit sum).
    /// * `chan_l1` — L1 distance between the normalized directional distributions
    ///   `E_c / E_0` over all 64 HOA channels, with `E_c = Σ_t h[t][c]`. This is
    ///   reported but not gated: the higher-order moments of a diffuse late field are
    ///   zero-mean, so two independent streams differ there by pure √N noise.
    /// * `order_l1` — L1 distance between the normalized per-SH-order powers
    ///   `P_n = Σ_{c∈order n} E_c²`. Aggregating `2n+1` channels per order makes this
    ///   the directional-HOA statistic that carries a reproducible signal.
    /// * `mean_t_diff_s` — absolute difference of the energy-weighted mean arrival time
    ///   `Σ_b t_b W_b / Σ_b W_b` (bin centers). The 8-band envelope is nearly flat across
    ///   the mixing time, so it is insensitive to a global timing offset; this metric
    ///   pins the time axis on which both peers must agree.
    #[cfg(feature = "cuda")]
    #[derive(Debug, Clone, Copy, Default)]
    struct HistAgreement {
        max_abs_cell: f64,
        rel_w_total: f64,
        band_l1: f64,
        chan_l1: f64,
        order_l1: f64,
        mean_t_diff_s: f64,
    }

    #[cfg(feature = "cuda")]
    fn histogram_agreement(
        cpu: &[[f64; HOA_CH]],
        gpu: &[[f64; HOA_CH]],
        n_bands: usize,
        bin_dt_s: f64,
    ) -> HistAgreement {
        let n_bins = cpu.len();
        let mut max_abs_cell = 0.0_f64;
        let mut w_cpu = 0.0_f64;
        let mut w_gpu = 0.0_f64;
        let mut tw_cpu = 0.0_f64;
        let mut tw_gpu = 0.0_f64;
        let mut e_cpu = [0.0_f64; HOA_CH];
        let mut e_gpu = [0.0_f64; HOA_CH];
        for (bin, (a, b)) in cpu.iter().zip(gpu).enumerate() {
            for (c, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                max_abs_cell = max_abs_cell.max((x - y).abs());
                e_cpu[c] += x;
                e_gpu[c] += y;
            }
            let t = (bin as f64 + 0.5) * bin_dt_s;
            w_cpu += a[0];
            w_gpu += b[0];
            tw_cpu += t * a[0];
            tw_gpu += t * b[0];
        }
        let mean_t_diff_s =
            (tw_cpu / w_cpu.max(f64::MIN_POSITIVE) - tw_gpu / w_gpu.max(f64::MIN_POSITIVE)).abs();
        let band_of = |bin: usize| (bin * n_bands / n_bins.max(1)).min(n_bands - 1);
        let mut band_cpu = vec![0.0_f64; n_bands];
        let mut band_gpu = vec![0.0_f64; n_bands];
        for (bin, (a, b)) in cpu.iter().zip(gpu).enumerate() {
            band_cpu[band_of(bin)] += a[0];
            band_gpu[band_of(bin)] += b[0];
        }
        let sum_c: f64 = band_cpu.iter().sum();
        let sum_g: f64 = band_gpu.iter().sum();
        let band_l1 = band_cpu
            .iter()
            .zip(&band_gpu)
            .map(|(a, b)| {
                (a / sum_c.max(f64::MIN_POSITIVE) - b / sum_g.max(f64::MIN_POSITIVE)).abs()
            })
            .sum();
        let chan_l1 = e_cpu
            .iter()
            .zip(&e_gpu)
            .map(|(a, b)| {
                (a / w_cpu.max(f64::MIN_POSITIVE) - b / w_gpu.max(f64::MIN_POSITIVE)).abs()
            })
            .sum();
        let mut order_cpu = [0.0_f64; 8];
        let mut order_gpu = [0.0_f64; 8];
        for n in 0..8usize {
            for c in n * n..(n + 1) * (n + 1) {
                order_cpu[n] += e_cpu[c] * e_cpu[c];
                order_gpu[n] += e_gpu[c] * e_gpu[c];
            }
        }
        let p_cpu: f64 = order_cpu.iter().sum();
        let p_gpu: f64 = order_gpu.iter().sum();
        let order_l1 = order_cpu
            .iter()
            .zip(&order_gpu)
            .map(|(a, b)| {
                (a / p_cpu.max(f64::MIN_POSITIVE) - b / p_gpu.max(f64::MIN_POSITIVE)).abs()
            })
            .sum();
        HistAgreement {
            max_abs_cell,
            rel_w_total: (w_cpu - w_gpu).abs()
                / w_cpu.abs().max(w_gpu.abs()).max(f64::MIN_POSITIVE),
            band_l1,
            chan_l1,
            order_l1,
            mean_t_diff_s,
        }
    }

    /// Bit-exact oracle histogram from the test-only kernel mirror.
    #[cfg(all(test, feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn oracle_raytrace_hoa_histogram(
        geom: &RoomGeometry,
        atm: &AtmosphereCfg,
        src_m: [f64; 3],
        n_rays: usize,
        max_bounces: usize,
        n_time_bins: usize,
        bin_dt_s: f64,
        seed: u64,
    ) -> Vec<[f64; HOA_CH]> {
        let mut hist = vec![[0.0_f64; HOA_CH]; n_time_bins];
        let c = atm.speed_of_sound_m_s();
        let ray_init_energy = 1.0 / (n_rays as f64);
        for ray_id in 0..n_rays {
            let hits =
                trace_single_ray(geom, atm, src_m, ray_id, ray_init_energy, max_bounces, seed);
            for (bounce, hit) in hits.iter().enumerate() {
                accumulate_hoa_hit(
                    &mut hist,
                    bounce,
                    hit.delay_s,
                    hit.energy,
                    hit.dir,
                    atm,
                    c,
                    n_time_bins,
                    bin_dt_s,
                );
            }
        }
        hist
    }

    /// Live CPU-vs-GPU agreement of the stochastic raytrace -> Order-7 HOA histogram,
    /// with the production Control Room ISO-17497 scattering coefficients (0.15–0.78)
    /// active on every wall.
    ///
    /// The CPU path is the independent native tracer ([`trace_native_ray`]: stratified
    /// equal-solid-angle launch, PCG64, slab intersection, ray/sphere receiver, chunked
    /// reduction); the GPU path is the zero-FMA `raytrace_hoa.cu` kernel. The two Monte
    /// Carlo streams are independent, so bit identity is neither expected nor claimed.
    /// The gate bounds the sampling envelope measured across the seeds below, plus a
    /// stated margin. Skips cleanly when no NVIDIA GPU is present.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn cuda_raytrace_hoa_histogram_agrees_with_native_cpu_within_measured_tolerance() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP cuda_raytrace_hoa_histogram_agrees_with_native_cpu_within_measured_tolerance: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        const SEEDS: [u64; 8] = [
            0xCAFE_BABE,
            0x5EED_1234,
            0x0D15_EA5E,
            0x6736_4D45_5241_5954,
            0x0123_4567_89AB_CDEF,
            0xA5A5_5A5A_5A5A_A5A5,
            0x0BAD_C0DE_F00D_D00D,
            0x1357_9BDF_2468_ACE0,
        ];
        const N_BANDS: usize = 8;
        // Measured worst over the 8 seeds at this exact configuration (65 536 rays x
        // 24 bounces): rel W total = 0.0526, band L1 = 0.0688, order L1 = 0.0995
        // (RTX 5070, 2026-11-28). Bounds = 1.5 x measured max + 0.02 absolute margin
        // for seed-to-seed Monte Carlo scatter.
        const MAX_REL_W_TOTAL_TOL: f64 = 0.13;
        const MAX_BAND_L1_TOL: f64 = 0.13;
        const MAX_ORDER_L1_TOL: f64 = 0.18;
        // Energy-weighted mean arrival time must agree far more tightly than the
        // envelope: the worst of the 8 seeds is 0.32 ms, so the bound is
        // 1.5 x 0.32 ms + 1.0 ms margin, rounded up to 1.5 ms. This pins the time axis
        // the coarse band envelope leaves loose.
        const MAX_MEAN_T_DIFF_TOL: f64 = 0.0015;
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s) = (65_536usize, 24usize, 128usize, 0.002_f64);

        let mut worst = HistAgreement::default();
        for &seed in &SEEDS {
            let h_cpu = raytrace_hoa_histogram(
                &geom,
                &atm,
                src,
                n_rays,
                max_bounces,
                n_bins,
                bin_dt_s,
                seed,
            );
            let h_gpu = raytrace_hoa_histogram_on_cuda(
                &geom,
                &atm,
                src,
                n_rays,
                max_bounces,
                n_bins,
                bin_dt_s,
                seed,
            )
            .unwrap_or_else(|e| panic!("direct GPU kernel path failed for seed {seed:#x}: {e}"));
            let m = histogram_agreement(&h_cpu, &h_gpu, N_BANDS, bin_dt_s);
            eprintln!(
                "ControlRoom seed={seed:#x}: max cell |d| = {:.3e}, rel W total = {:.4}, \
                 band L1 = {:.4}, chan L1 = {:.4}, order L1 = {:.4}, mean t |d| = {:.2} ms \
                 ({n_rays} rays x {max_bounces} bounces x {n_bins} bins)",
                m.max_abs_cell,
                m.rel_w_total,
                m.band_l1,
                m.chan_l1,
                m.order_l1,
                m.mean_t_diff_s * 1e3
            );
            worst.max_abs_cell = worst.max_abs_cell.max(m.max_abs_cell);
            worst.rel_w_total = worst.rel_w_total.max(m.rel_w_total);
            worst.band_l1 = worst.band_l1.max(m.band_l1);
            worst.chan_l1 = worst.chan_l1.max(m.chan_l1);
            worst.order_l1 = worst.order_l1.max(m.order_l1);
            worst.mean_t_diff_s = worst.mean_t_diff_s.max(m.mean_t_diff_s);
        }
        eprintln!(
            "ControlRoom worst over {} seeds: max cell |d| = {:.3e}, rel W total = {:.4} \
             (tol {MAX_REL_W_TOTAL_TOL}), band L1 = {:.4} (tol {MAX_BAND_L1_TOL}), chan L1 = \
             {:.4}, order L1 = {:.4} (tol {MAX_ORDER_L1_TOL}), mean t |d| = {:.2} ms (tol {:.1} \
             ms)",
            SEEDS.len(),
            worst.max_abs_cell,
            worst.rel_w_total,
            worst.band_l1,
            worst.chan_l1,
            worst.order_l1,
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
        assert!(
            worst.rel_w_total < MAX_REL_W_TOTAL_TOL,
            "native CPU vs GPU raytrace W-energy rel diff = {:.4} >= {MAX_REL_W_TOTAL_TOL}",
            worst.rel_w_total
        );
        assert!(
            worst.band_l1 < MAX_BAND_L1_TOL,
            "native CPU vs GPU raytrace temporal-envelope L1 = {:.4} >= {MAX_BAND_L1_TOL}",
            worst.band_l1
        );
        assert!(
            worst.order_l1 < MAX_ORDER_L1_TOL,
            "native CPU vs GPU raytrace per-order power L1 = {:.4} >= {MAX_ORDER_L1_TOL}",
            worst.order_l1
        );
        assert!(
            worst.mean_t_diff_s < MAX_MEAN_T_DIFF_TOL,
            "native CPU vs GPU raytrace mean arrival-time diff = {:.3} ms >= {:.1} ms",
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
        assert!(
            worst.max_abs_cell.is_finite(),
            "CPU/GPU histogram comparison produced a non-finite cell difference"
        );
    }

    /// Second live agreement scenario: large theater stage (different geometry, wall
    /// reflectances and scattering probabilities).
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn cuda_raytrace_hoa_histogram_agrees_with_native_cpu_on_theater_stage() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP cuda_raytrace_hoa_histogram_agrees_with_native_cpu_on_theater_stage: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        const SEEDS: [u64; 8] = [
            0xCAFE_BABE,
            0x5EED_1234,
            0x0D15_EA5E,
            0x6736_4D45_5241_5954,
            0x0123_4567_89AB_CDEF,
            0xA5A5_5A5A_5A5A_A5A5,
            0x0BAD_C0DE_F00D_D00D,
            0x1357_9BDF_2468_ACE0,
        ];
        const N_BANDS: usize = 8;
        // Measured worst over the 8 seeds at this exact configuration (65 536 rays x
        // 16 bounces): rel W total = 0.0673, band L1 = 0.0631, order L1 = 0.0571
        // (RTX 5070, 2026-11-28). Bounds = 1.5 x measured max + 0.02 absolute margin
        // for seed-to-seed Monte Carlo scatter.
        const MAX_REL_W_TOTAL_TOL: f64 = 0.13;
        const MAX_BAND_L1_TOL: f64 = 0.13;
        const MAX_ORDER_L1_TOL: f64 = 0.18;
        // Energy-weighted mean arrival time: the worst of the 8 seeds is 1.06 ms, so the
        // bound is 1.5 x 1.06 ms + 1.0 ms margin, rounded up to 2.6 ms.
        const MAX_MEAN_T_DIFF_TOL: f64 = 0.0026;
        let geom = RoomGeometry::from_preset(RoomPreset::TheaterStage);
        let atm = AtmosphereCfg::default();
        let src = [6.2, 4.0, 1.35];
        let (n_rays, max_bounces, n_bins, bin_dt_s) = (65_536usize, 16usize, 128usize, 0.002_f64);

        let mut worst = HistAgreement::default();
        for &seed in &SEEDS {
            let h_cpu = raytrace_hoa_histogram(
                &geom,
                &atm,
                src,
                n_rays,
                max_bounces,
                n_bins,
                bin_dt_s,
                seed,
            );
            let h_gpu = raytrace_hoa_histogram_on_cuda(
                &geom,
                &atm,
                src,
                n_rays,
                max_bounces,
                n_bins,
                bin_dt_s,
                seed,
            )
            .unwrap_or_else(|e| panic!("direct GPU kernel path failed for seed {seed:#x}: {e}"));
            let m = histogram_agreement(&h_cpu, &h_gpu, N_BANDS, bin_dt_s);
            eprintln!(
                "TheaterStage seed={seed:#x}: max cell |d| = {:.3e}, rel W total = {:.4}, \
                 band L1 = {:.4}, chan L1 = {:.4}, order L1 = {:.4}, mean t |d| = {:.2} ms \
                 ({n_rays} rays x {max_bounces} bounces x {n_bins} bins)",
                m.max_abs_cell,
                m.rel_w_total,
                m.band_l1,
                m.chan_l1,
                m.order_l1,
                m.mean_t_diff_s * 1e3
            );
            worst.max_abs_cell = worst.max_abs_cell.max(m.max_abs_cell);
            worst.rel_w_total = worst.rel_w_total.max(m.rel_w_total);
            worst.band_l1 = worst.band_l1.max(m.band_l1);
            worst.chan_l1 = worst.chan_l1.max(m.chan_l1);
            worst.order_l1 = worst.order_l1.max(m.order_l1);
            worst.mean_t_diff_s = worst.mean_t_diff_s.max(m.mean_t_diff_s);
        }
        eprintln!(
            "TheaterStage worst over {} seeds: max cell |d| = {:.3e}, rel W total = {:.4} \
             (tol {MAX_REL_W_TOTAL_TOL}), band L1 = {:.4} (tol {MAX_BAND_L1_TOL}), chan L1 = \
             {:.4}, order L1 = {:.4} (tol {MAX_ORDER_L1_TOL}), mean t |d| = {:.2} ms (tol {:.1} \
             ms)",
            SEEDS.len(),
            worst.max_abs_cell,
            worst.rel_w_total,
            worst.band_l1,
            worst.chan_l1,
            worst.order_l1,
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
        assert!(
            worst.rel_w_total < MAX_REL_W_TOTAL_TOL,
            "TheaterStage native CPU vs GPU raytrace W-energy rel diff = {:.4} >= \
             {MAX_REL_W_TOTAL_TOL}",
            worst.rel_w_total
        );
        assert!(
            worst.band_l1 < MAX_BAND_L1_TOL,
            "TheaterStage native CPU vs GPU raytrace temporal-envelope L1 = {:.4} >= \
             {MAX_BAND_L1_TOL}",
            worst.band_l1
        );
        assert!(
            worst.order_l1 < MAX_ORDER_L1_TOL,
            "TheaterStage native CPU vs GPU raytrace per-order power L1 = {:.4} >= \
             {MAX_ORDER_L1_TOL}",
            worst.order_l1
        );
        assert!(
            worst.mean_t_diff_s < MAX_MEAN_T_DIFF_TOL,
            "TheaterStage native CPU vs GPU raytrace mean arrival-time diff = {:.3} ms >= \
             {:.1} ms",
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
        assert!(
            worst.max_abs_cell.is_finite(),
            "TheaterStage CPU/GPU histogram comparison produced a non-finite cell difference"
        );
    }

    /// The test-only bit-exact mirror (`RayPcg32` + `trace_single_ray` +
    /// [`accumulate_hoa_hit`]) still reproduces the CUDA kernel record for record. This
    /// keeps `raytrace_hoa.cu` itself validated even though the production CPU path is
    /// now an independent native tracer.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn cuda_raytrace_kernel_matches_the_test_only_bit_exact_oracle() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP cuda_raytrace_kernel_matches_the_test_only_bit_exact_oracle: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [1.5, 2.0, 1.2];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) = (
            512usize,
            12usize,
            128usize,
            0.002_f64,
            0x6736_4D45_5241_5954_u64,
        );
        let oracle = oracle_raytrace_hoa_histogram(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let gpu = raytrace_hoa_histogram_on_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .expect("direct GPU kernel path failed");
        for (bin, (a, b)) in oracle.iter().zip(&gpu).enumerate() {
            for (c, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "kernel vs bit-exact oracle mismatch at bin={bin} ch={c}: \
                     oracle={x:e} gpu={y:e}"
                );
            }
        }
        let energy: f64 = oracle.iter().flat_map(|b| b.iter()).map(|x| x.abs()).sum();
        assert!(
            energy > 1e-6,
            "kernel/oracle parity run recorded no HOA energy (sum |h| = {energy:e})"
        );
    }

    #[test]
    fn btm_svensson_wedge_angles_obey_helmholtz_reciprocity_and_fdtd_resolves_500hz() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let src: [f64; 3] = [4.35, 3.25, 1.30];
        let rec: [f64; 3] = [2.55, 2.15, 1.15];
        let d_dir =
            ((src[0] - rec[0]).powi(2) + (src[1] - rec[1]).powi(2) + (src[2] - rec[2]).powi(2))
                .sqrt();
        let t_dir = d_dir / 344.854;
        let mut ir_sr = vec![0.0_f64; 1024];
        let mut ir_rs = vec![0.0_f64; 1024];
        add_btm_edge_diffraction(
            &geom.edges[0],
            src,
            rec,
            344.854,
            48_000,
            t_dir,
            0.005,
            &mut ir_sr,
        );
        add_btm_edge_diffraction(
            &geom.edges[0],
            rec,
            src,
            344.854,
            48_000,
            t_dir,
            0.005,
            &mut ir_rs,
        );
        let max_diff = ir_sr
            .iter()
            .zip(&ir_rs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-13,
            "Svensson BTM diffraction violated Helmholtz reciprocity: max_diff={max_diff:e}"
        );

        let fdtd = FdtdSolver3d::for_room_budget(&geom, 344.854, 500.0, 65_000);
        assert!(
            fdtd.max_freq_6ppw(344.854) >= 499.0,
            "ControlRoom FDTD 6-PPW bandwidth must reach 500 Hz, got {:.1} Hz (dx={:.3} m)",
            fdtd.max_freq_6ppw(344.854),
            fdtd.dx_m
        );
    }

    // ------------------------------------------------------------------
    // Ten-band room model, Tier-A preset library and per-band transport.
    // ------------------------------------------------------------------

    #[test]
    fn octave_band_weights_partition_unity_and_hold_at_edges() {
        assert_eq!(ROOM_BAND_COUNT, 10);
        assert_eq!(
            ROOM_BANDS_HZ,
            [31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0]
        );
        assert_eq!(ROOM_BAND_MID, 5);
        assert_eq!(ROOM_BANDS_HZ[ROOM_BAND_MID], 1_000.0);

        // Partition of unity over a dense log sweep: every frequency is covered
        // by exactly one full weight, split between the two adjacent bands.
        let mut f = 20.0_f64;
        while f <= 20_000.0 {
            let sum: f64 = (0..ROOM_BAND_COUNT).map(|b| octave_band_weight(f, b)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "octave weights at {f} Hz sum to {sum}, not 1"
            );
            f *= 1.01;
        }
        // Geometric midpoint between 1 kHz and 2 kHz is an exact 50/50 split.
        let f_mid = (1_000.0_f64 * 2_000.0).sqrt();
        assert!((octave_band_weight(f_mid, 5) - 0.5).abs() < 1e-12);
        assert!((octave_band_weight(f_mid, 6) - 0.5).abs() < 1e-12);
        // Edge bands hold full weight outside the covered range.
        assert_eq!(octave_band_weight(10.0, 0), 1.0);
        assert_eq!(octave_band_weight(30_000.0, 9), 1.0);
        assert_eq!(octave_band_weight(31.5, 0), 1.0);
        assert_eq!(octave_band_weight(16_000.0, 9), 1.0);
        assert_eq!(octave_band_weight(f64::NAN, 3), 0.0);
        assert_eq!(octave_band_weight(1_000.0, ROOM_BAND_COUNT), 0.0);
    }

    /// The material tables interpolated by `band_value_at` used to step at
    /// 125 Hz: the un-normalised `log2(f)` segment from 63 Hz approaches
    /// `y(63) + 0.9887·(y(125) − y(63))`, while the interval above starts at
    /// `y(125)`. With `dy = 0.45` that is a 5.2e-3 step; PCHIP must hold the
    /// left and right limits to `y(125)` within 1e-9.
    #[test]
    fn band_value_interpolation_is_continuous_at_nominal_band_edges() {
        let values = [0.10, 0.15, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60];
        for &edge in &ROOM_BANDS_HZ[1..ROOM_BAND_COUNT - 1] {
            let left = band_value_at(&values, edge * (1.0 - 1e-12));
            let right = band_value_at(&values, edge * (1.0 + 1e-12));
            let at = band_value_at(&values, edge);
            assert!(
                (left - at).abs() < 1e-9,
                "left limit at {edge} Hz is {left}, knot is {at}"
            );
            assert!(
                (right - at).abs() < 1e-9,
                "right limit at {edge} Hz is {right}, knot is {at}"
            );
        }
        // The old piecewise-linear form differed from the knot by 5.19e-3 here.
        let at_125 = band_value_at(&values, 125.0);
        assert!((at_125 - values[2]).abs() < 1e-12);
    }

    /// PCHIP is shape-preserving: a monotone table stays monotone, and a table
    /// inside the physical ranges cannot overshoot them.
    #[test]
    fn band_value_interpolation_is_monotone_without_overshoot() {
        let absorption = [0.05, 0.07, 0.10, 0.13, 0.20, 0.30, 0.42, 0.55, 0.66, 0.75];
        // Non-monotone table with a sharp 125 Hz peak: no ringing past 0.90.
        let scattering = [0.10, 0.20, 0.90, 0.50, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05];
        let mut f = ROOM_BANDS_HZ[0];
        let mut prev = band_value_at(&absorption, f);
        while f <= ROOM_BANDS_HZ[ROOM_BAND_COUNT - 1] {
            let a = band_value_at(&absorption, f);
            assert!(
                a >= prev - 1e-12,
                "absorption interpolation dipped at {f} Hz: {prev} -> {a}"
            );
            assert!(
                (0.01..=0.99).contains(&a),
                "absorption left [0.01, 0.99] at {f} Hz: {a}"
            );
            let s = band_value_at(&scattering, f);
            assert!(
                (0.0..=1.0).contains(&s),
                "scattering left [0, 1] at {f} Hz: {s}"
            );
            prev = a;
            f *= 1.003;
        }
    }

    #[test]
    fn legacy_six_band_room_json_expands_to_ten_bands() {
        // Pre-ten-band `--room-json` schema: six-band absorption and a scalar
        // scattering coefficient. Must keep loading with the documented
        // log-frequency expansion and the 1 kHz column unchanged.
        let wall = r#"{"absorption": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60], "scattering": 0.25}"#;
        let json = format!(
            r#"{{
                "name": "Legacy Six Band Room",
                "dims_m": [6.0, 5.0, 3.0],
                "listener_m": [2.6, 2.5, 1.25],
                "walls": [{wall}, {wall}, {wall}, {wall}, {wall}, {wall}],
                "bass_trap_admittance": 0.16,
                "target_rt60_s": 0.25
            }}"#
        );
        let geom: RoomGeometry = serde_json::from_str(&json).expect("legacy room JSON must load");
        geom.validate().expect("expanded legacy room must validate");
        assert!(geom.targets.is_none());
        let m = &geom.walls[0];
        // 125, 250, 500, 1k, 2k, 4k map onto bands 2..=7 exactly; 31.5/63 hold
        // the 125 Hz value and 8k/16k hold the 4 kHz value.
        assert_eq!(
            m.absorption,
            [0.10, 0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.60, 0.60]
        );
        for (band, &s) in m.scattering.iter().enumerate() {
            assert_eq!(s, 0.25, "legacy scalar scattering lost at band {band}");
        }

        // Current ten-band schema round-trips. serde_json's default (non
        // `float_roundtrip`) float parser is specified within 1 ULP, so the
        // check is 2 ULP absolute; every coefficient must survive.
        let ten = WallMaterial {
            absorption: std::array::from_fn(|b| 0.05 + 0.05 * b as f64),
            scattering: std::array::from_fn(|b| 0.10 + 0.01 * b as f64),
        };
        let text = serde_json::to_string(&ten).expect("serialize ten-band material");
        let back: WallMaterial = serde_json::from_str(&text).expect("reload ten-band material");
        for (band, (&x, &y)) in back
            .absorption
            .iter()
            .zip(ten.absorption.iter())
            .enumerate()
        {
            let tol = 2.0 * f64::EPSILON * y.abs().max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "ten-band absorption band {band} round-trip {x} != {y}"
            );
        }
        for (band, (&x, &y)) in back
            .scattering
            .iter()
            .zip(ten.scattering.iter())
            .enumerate()
        {
            let tol = 2.0 * f64::EPSILON * y.abs().max(1.0);
            assert!(
                (x - y).abs() <= tol,
                "ten-band scattering band {band} round-trip {x} != {y}"
            );
        }
    }

    /// One row of the research doc §4 Tier-A table: preset, dimensions, quoted
    /// acoustic volume, mid T, then EDT/G/C80/LF/1−IACC/ITDG.
    #[allow(clippy::type_complexity)]
    type PresetDocRow = (
        ShoeboxPreset,
        [f64; 3],
        f64,
        f64,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
        Option<f64>,
    );

    #[test]
    fn shoebox_preset_metadata_matches_research_doc_tables() {
        for &preset in &ShoeboxPreset::ALL {
            let geom = RoomGeometry::from_shoebox_preset(preset);
            geom.validate()
                .unwrap_or_else(|e| panic!("{} preset invalid: {e}", preset.label()));
            assert!(
                geom.targets.is_some(),
                "{} preset carries no acoustic targets",
                preset.label()
            );
        }

        // Tier-A table (`docs/ROOM_PRESETS_RESEARCH.md` §4), transcribed exactly.
        let cases: [PresetDocRow; 6] = [
            (
                ShoeboxPreset::Musikverein,
                [48.8, 19.1, 17.75],
                15_000.0,
                2.0,
                Some(2.0),
                Some(4.0),
                Some(-1.0),
                Some(0.17),
                Some(0.64),
                Some(12.0),
            ),
            (
                ShoeboxPreset::BostonSymphonyHall,
                [38.1, 22.9, 18.6],
                18_750.0,
                1.9,
                Some(1.9),
                Some(2.0),
                Some(0.0),
                Some(0.22),
                Some(0.65),
                Some(15.0),
            ),
            (
                ShoeboxPreset::Concertgebouw,
                [44.0, 28.0, 17.0],
                18_780.0,
                2.0,
                Some(1.9),
                Some(3.0),
                Some(-1.0),
                Some(0.18),
                None,
                Some(21.0),
            ),
            (
                ShoeboxPreset::StudioLiveRoom,
                [8.0, 6.0, 4.0],
                192.0,
                0.45,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            (
                ShoeboxPreset::LectureTheatre,
                [15.0, 10.0, 6.0],
                900.0,
                0.80,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            (
                ShoeboxPreset::Cinema,
                [20.0, 12.0, 8.0],
                1_920.0,
                0.50,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ];
        for &(preset, dims, volume, t_mid, edt, g, c80, lf, iacc, itdg) in &cases {
            let geom = RoomGeometry::from_shoebox_preset(preset);
            let t = geom.targets.as_ref().unwrap();
            assert_eq!(geom.dims_m, dims, "{} dimensions", preset.label());
            assert_eq!(
                geom.acoustic_volume_m3(),
                volume,
                "{} quoted acoustic volume",
                preset.label()
            );
            assert_eq!(t.mid_rt60_s, t_mid, "{} T", preset.label());
            assert_eq!(t.mid_edt_s, edt, "{} EDT", preset.label());
            assert_eq!(t.strength_g_db, g, "{} G", preset.label());
            assert_eq!(t.clarity_c80_db, c80, "{} C80", preset.label());
            assert_eq!(t.bass_lf, lf, "{} LF", preset.label());
            assert_eq!(t.one_minus_iacc, iacc, "{} 1−IACC", preset.label());
            assert_eq!(t.itdg_ms, itdg, "{} ITDG", preset.label());
        }

        // Curve provenance and FDTD handoff per preset class.
        let model = |p| RoomGeometry::from_shoebox_preset(p).targets.unwrap().model;
        let geo_from = |p| {
            RoomGeometry::from_shoebox_preset(p)
                .targets
                .unwrap()
                .geometric_bands_from_hz
        };
        assert_eq!(model(ShoeboxPreset::ControlRoom), Rt60Model::MidBandOnly);
        assert_eq!(model(ShoeboxPreset::TheaterStage), Rt60Model::MidBandOnly);
        for p in [
            ShoeboxPreset::StudioLiveRoom,
            ShoeboxPreset::LectureTheatre,
            ShoeboxPreset::Cinema,
            ShoeboxPreset::Musikverein,
            ShoeboxPreset::BostonSymphonyHall,
            ShoeboxPreset::Concertgebouw,
        ] {
            assert_eq!(model(p), Rt60Model::EyringDesign, "{} model", p.label());
        }
        assert_eq!(geo_from(ShoeboxPreset::ControlRoom), 1_000.0);
        assert_eq!(geo_from(ShoeboxPreset::TheaterStage), 1_000.0);
        assert_eq!(geo_from(ShoeboxPreset::StudioLiveRoom), 1_000.0);
        assert_eq!(geo_from(ShoeboxPreset::LectureTheatre), 500.0);
        assert_eq!(geo_from(ShoeboxPreset::Cinema), 500.0);
        for p in ShoeboxPreset::HALLS {
            assert_eq!(geo_from(p), 250.0, "{} hall FDTD handoff", p.label());
        }

        // Documented ranges, and no range where the source gives one number.
        let range = |p| {
            RoomGeometry::from_shoebox_preset(p)
                .targets
                .unwrap()
                .mid_rt60_range_s
        };
        assert_eq!(range(ShoeboxPreset::ControlRoom), Some([0.20, 0.40]));
        assert_eq!(range(ShoeboxPreset::TheaterStage), None);
        assert_eq!(range(ShoeboxPreset::StudioLiveRoom), Some([0.35, 0.55]));
        assert_eq!(range(ShoeboxPreset::LectureTheatre), Some([0.70, 0.90]));
        assert_eq!(range(ShoeboxPreset::Cinema), Some([0.40, 0.60]));
        for p in ShoeboxPreset::HALLS {
            assert_eq!(range(p), None, "{}", p.label());
        }
        assert_eq!(
            RoomGeometry::from_shoebox_preset(ShoeboxPreset::Concertgebouw)
                .targets
                .unwrap()
                .mid_rt60_alt_s,
            Some(2.2)
        );
    }

    #[test]
    fn eyring_design_targets_are_reproducible_and_respond_to_material_changes() {
        let atm = AtmosphereCfg::default();
        let mut design_presets = 0usize;
        for &preset in &ShoeboxPreset::ALL {
            let geom = RoomGeometry::from_shoebox_preset(preset);
            let t = geom.targets.as_ref().unwrap();
            let recomputed = geom.eyring_rt60_bands(&atm);
            for (band, stored) in t.rt60_bands_s.iter().enumerate() {
                let expected = (t.model == Rt60Model::EyringDesign
                    && ROOM_BANDS_HZ[band] >= t.geometric_bands_from_hz)
                    .then_some(recomputed[band]);
                match (stored, expected) {
                    (None, None) => {}
                    (&Some(s), Some(e)) => assert!(
                        (s - e).abs() < 1e-12,
                        "{} band {band}: stored target {s} != Eyring material value {e}",
                        preset.label()
                    ),
                    (s, e) => panic!(
                        "{} band {band}: stored target {s:?} but geometric model expects {e:?}",
                        preset.label()
                    ),
                }
            }
            if t.model == Rt60Model::EyringDesign {
                design_presets += 1;
                let mid = t.rt60_bands_s[ROOM_BAND_MID]
                    .expect("Eyring design preset must target the 1 kHz band");
                let residual = mid - t.mid_rt60_s;
                assert!(
                    (residual - t.fit_residual_s.unwrap()).abs() < 1e-12,
                    "{} fit_residual_s does not match T(1 kHz) - T_mid",
                    preset.label()
                );
                assert!(
                    residual.abs() / t.mid_rt60_s <= 0.05,
                    "{} Eyring 1 kHz fit residual {residual:.3} s ({:.2}%) exceeds 5%",
                    preset.label(),
                    100.0 * residual.abs() / t.mid_rt60_s
                );
                let first_geo = t
                    .rt60_bands_s
                    .iter()
                    .flatten()
                    .next()
                    .copied()
                    .expect("Eyring design preset must have a geometric band");
                eprintln!(
                    "{}: T_mid={:.2} s, Eyring T(1k)={:.3} s, residual={:+.2}%, \
                     T(lowest geometric)={:.3}, T(8k)={:.3} s",
                    preset.label(),
                    t.mid_rt60_s,
                    mid,
                    100.0 * residual / t.mid_rt60_s,
                    first_geo,
                    t.rt60_bands_s[8].expect("8 kHz is always a geometric band")
                );
            }

            // Sabotage (research doc §5 B5): a 10% change to one band's floor
            // absorption must move the design target this gate reads, so a
            // hard-coded RT curve cannot pass. The 1 kHz floor is the largest
            // surface, making this the most conservative perturbation.
            if t.model == Rt60Model::EyringDesign {
                let mut sabotaged = geom.clone();
                let alpha = sabotaged.walls[4].absorption[ROOM_BAND_MID];
                sabotaged.walls[4].absorption[ROOM_BAND_MID] = (alpha * 1.10).min(0.99);
                let t_orig = geom.eyring_rt60_band(&atm, ROOM_BAND_MID);
                let t_sab = sabotaged.eyring_rt60_band(&atm, ROOM_BAND_MID);
                let rel = (t_sab - t_orig).abs() / t_orig;
                assert!(
                    rel > 1e-3,
                    "{}: 10% floor-alpha perturbation moved Eyring 1 kHz RT by only {:.4}%",
                    preset.label(),
                    100.0 * rel
                );
            }
        }
        assert_eq!(design_presets, 6, "expected six EyringDesign presets");
    }

    #[test]
    fn hall_itdg_matches_documented_first_reflection_gap() {
        let atm = AtmosphereCfg::default();
        let c = atm.speed_of_sound_m_s();
        let first_gap_ms = |geom: &RoomGeometry, src: [f64; 3]| -> f64 {
            image_sources_3d(geom, src, c, 1)
                .iter()
                .filter(|tap| tap.wall_hits[4] == 0) // floor excluded from Beranek's ITDG
                .map(|tap| tap.delay_s * 1_000.0)
                .fold(f64::INFINITY, f64::min)
        };
        for (preset, doc_ms) in [
            (ShoeboxPreset::Musikverein, 12.0),
            (ShoeboxPreset::BostonSymphonyHall, 15.0),
            (ShoeboxPreset::Concertgebouw, 21.0),
        ] {
            let geom = RoomGeometry::from_shoebox_preset(preset);
            let t = geom.targets.as_ref().unwrap();
            assert_eq!(t.itdg_ms, Some(doc_ms), "{} ITDG metadata", preset.label());
            let src = t
                .reference_source_m
                .expect("hall preset must carry a reference source");
            let geometric_ms = first_gap_ms(&geom, src);
            eprintln!(
                "{}: documented ITDG {doc_ms:.1} ms, shoebox first non-floor reflection \
                 {geometric_ms:.3} ms",
                preset.label()
            );
            assert!(
                geometric_ms.is_finite() && (geometric_ms - doc_ms).abs() <= 0.25,
                "{}: shoebox ITDG {geometric_ms:.3} ms deviates from documented {doc_ms} ms \
                 by more than 0.25 ms",
                preset.label()
            );
        }

        // Sabotage: a 1 m source move must move the geometric gap, so this gate
        // is measuring geometry and not echoing a stored constant.
        let geom = RoomGeometry::from_shoebox_preset(ShoeboxPreset::Musikverein);
        let src = geom.targets.as_ref().unwrap().reference_source_m.unwrap();
        let moved = [src[0] + 1.0, src[1], src[2]];
        let delta = (first_gap_ms(&geom, moved) - first_gap_ms(&geom, src)).abs();
        assert!(
            delta > 0.1,
            "moving the reference source by 1 m changed the geometric ITDG by only {delta:.3} ms"
        );
    }

    #[test]
    fn image_source_taps_carry_per_band_energy_and_isolate_material_band_changes() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let src = [4.3, 3.5, 1.3];
        let c = AtmosphereCfg::default().speed_of_sound_m_s();
        let taps = image_sources_3d(&geom, src, c, 2);
        assert!(!taps.is_empty(), "control-room ISM produced no taps");
        for tap in &taps {
            for band in 0..ROOM_BAND_COUNT {
                let expected = tap
                    .wall_hits
                    .iter()
                    .enumerate()
                    .fold(1.0_f64, |acc, (w, &h)| {
                        if h == 0 {
                            acc
                        } else {
                            acc * geom.walls[w].refl_energy(band).powi(h as i32)
                        }
                    });
                assert!(
                    (tap.band_reflection_energy[band] - expected).abs() < 1e-12,
                    "ISM tap band {band} energy {} != wall product {expected}",
                    tap.band_reflection_energy[band]
                );
                assert!((0.0..=1.0).contains(&tap.band_reflection_energy[band]));
            }
        }

        // Band isolation: perturbing the ceiling's 8 kHz alpha (ROOM_BANDS_HZ[8])
        // must change only band 8, and only on taps that actually hit the ceiling.
        let mut sabotaged = geom.clone();
        let alpha = sabotaged.walls[5].absorption[8];
        sabotaged.walls[5].absorption[8] = (alpha * 1.10).min(0.99);
        let sab_taps = image_sources_3d(&sabotaged, src, c, 2);
        assert_eq!(taps.len(), sab_taps.len());
        let mut ceiling_taps = 0usize;
        for (a, b) in taps.iter().zip(&sab_taps) {
            assert_eq!(a.wall_hits, b.wall_hits);
            assert_eq!(a.delay_s.to_bits(), b.delay_s.to_bits());
            if a.wall_hits[5] == 0 {
                for band in 0..ROOM_BAND_COUNT {
                    assert_eq!(
                        a.band_reflection_energy[band].to_bits(),
                        b.band_reflection_energy[band].to_bits(),
                        "non-ceiling tap changed in band {band}"
                    );
                }
            } else {
                ceiling_taps += 1;
                assert!(
                    (a.band_reflection_energy[8] - b.band_reflection_energy[8]).abs() > 1e-6,
                    "ceiling tap's 8 kHz energy did not respond to a ceiling alpha change"
                );
                for band in 0..ROOM_BAND_COUNT {
                    if band != 8 {
                        assert_eq!(
                            a.band_reflection_energy[band].to_bits(),
                            b.band_reflection_energy[band].to_bits(),
                            "ceiling tap leaked an 8 kHz material change into band {band}"
                        );
                    }
                }
            }
        }
        assert!(
            ceiling_taps > 0,
            "control-room ISM produced no ceiling reflection taps"
        );
    }

    #[test]
    fn per_band_histogram_isolates_band_material_changes_and_is_thread_invariant() {
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (256usize, 12usize, 16usize, 0.004_f64, 0x05EE_D0B1_u64);

        let base = raytrace_hoa_histogram_per_band(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let serial = raytrace_hoa_histogram_per_band_serial(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        assert_eq!(base.len(), n_bins);
        for (bin, (a, b)) in base.iter().zip(&serial).enumerate() {
            for band in 0..ROOM_BAND_COUNT {
                for (ch, (&x, &y)) in a[band].iter().zip(b[band].iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "per-band parallel/serial bit mismatch at bin={bin} band={band} ch={ch}"
                    );
                }
            }
        }

        // Band isolation: only band 8 (8 kHz, ROOM_BANDS_HZ[8]) may see the
        // ceiling change.
        let mut sabotaged = geom.clone();
        let alpha = sabotaged.walls[5].absorption[8];
        sabotaged.walls[5].absorption[8] = (alpha * 1.10).min(0.99);
        let pert = raytrace_hoa_histogram_per_band(
            &sabotaged,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let mut band8_changed = false;
        for (bin, (a, b)) in base.iter().zip(&pert).enumerate() {
            for band in 0..ROOM_BAND_COUNT {
                if band == 8 {
                    if a[band]
                        .iter()
                        .zip(b[band].iter())
                        .any(|(&x, &y)| x.to_bits() != y.to_bits())
                    {
                        band8_changed = true;
                    }
                } else {
                    for (ch, (&x, &y)) in a[band].iter().zip(b[band].iter()).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "ceiling 8 kHz change leaked into bin={bin} band={band} ch={ch}"
                        );
                    }
                }
            }
        }
        assert!(
            band8_changed,
            "ceiling 8 kHz alpha change did not alter the 8 kHz per-band histogram"
        );
    }

    #[test]
    fn per_band_histogram_applies_band_centre_air_absorption() {
        // Uniform material isolates the molecular-air term: the ten bands see
        // identical reflection/scattering tables and identical ray paths, so the
        // only band-to-band difference in the recorded energy is ISO 9613-1 air
        // absorption evaluated at each band centre. A single broadband air
        // reference (the pre-ten-band behaviour) would leave W(16 kHz) ≈ W(32 Hz).
        let geom = RoomGeometry {
            name: "Uniform Air Test Room".to_string(),
            dims_m: [20.0, 15.0, 10.0],
            listener_m: [10.0, 7.5, 2.0],
            walls: [WallMaterial::uniform(0.05, 0.10); 6],
            bass_trap_admittance: 0.05,
            target_rt60_s: 1.0,
            targets: None,
            edges: Vec::new(),
        };
        geom.validate()
            .expect("uniform air-test room must validate");
        let atm = AtmosphereCfg::default();
        let src = [2.0, 7.5, 2.0];
        let rows =
            raytrace_hoa_histogram_per_band(&geom, &atm, src, 1_024, 32, 128, 0.0025, 0xA17_0B1u64);
        let w = |band: usize| rows.iter().map(|row| row[band][0]).sum::<f64>();
        let (w_low, w_high) = (w(0), w(ROOM_BAND_COUNT - 1));
        assert!(
            w_low > 1e-9 && w_high > 1e-9,
            "air-absorption test room recorded no energy: W(32 Hz)={w_low:e}, W(16 kHz)={w_high:e}"
        );
        let ratio = w_high / w_low;
        eprintln!(
            "uniform room W(16 kHz)/W(31.5 Hz) = {ratio:.4} (ISO 9613-1 at the band centres)"
        );
        assert!(
            ratio < 0.75,
            "16 kHz late energy is not attenuated relative to 31.5 Hz (ratio {ratio:.4}); \
             the per-band air reference is not wired"
        );
    }

    /// T30 from a per-band relative amplitude envelope: least-squares slope of
    /// `20·log10(e_rel)` over the −5 … −35 dB portion of the decay after the
    /// envelope's peak, scaled to a 60 dB decay time. `None` when the envelope
    /// does not cover 30 dB (which is exactly what the old `0.15` floor caused).
    fn t30_from_amplitude_envelope(e_rel: &[f64], bin_dt_s: f64) -> Option<f64> {
        let peak = e_rel.iter().copied().fold(0.0_f64, f64::max);
        if peak <= 0.0 {
            return None;
        }
        let peak_idx = e_rel.iter().position(|&e| e == peak)?;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (i, &e) in e_rel.iter().enumerate().skip(peak_idx) {
            if e <= 0.0 {
                continue;
            }
            let level = 20.0 * (e / peak).log10();
            if (-35.0..=-5.0).contains(&level) {
                xs.push((i - peak_idx) as f64 * bin_dt_s);
                ys.push(level);
            }
        }
        if xs.len() < 8 {
            return None;
        }
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
        let slope = sxy / sxx;
        (slope < 0.0).then(|| -60.0 / slope)
    }

    /// The envelope of a strongly absorbing band must keep its own decay: the
    /// pre-fix code divided every band by one global maximum and clamped the
    /// relative amplitude at `0.15` (−16.48 dB), at which point that band
    /// flat-lined and inherited `base_l`/`base_r`'s broadband rate.
    #[test]
    fn per_band_envelope_keeps_absorbing_band_decay_below_minus_16_5_db() {
        const BAND: usize = 5; // 1 kHz: wall absorption dominates, air is negligible.
        let mut walls = [WallMaterial::uniform(0.05, 0.50); 6];
        for w in &mut walls {
            w.absorption[BAND] = 0.30;
        }
        let geom = RoomGeometry {
            name: "One Absorbing Band".to_string(),
            dims_m: [6.0, 5.0, 3.0],
            listener_m: [2.6, 2.5, 1.25],
            walls,
            bass_trap_admittance: 0.16,
            target_rt60_s: 0.25,
            targets: None,
            edges: Vec::new(),
        };
        geom.validate()
            .expect("one-absorbing-band room must validate");
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.3];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (8_192usize, 128usize, 200usize, 0.004_f64, 0x0BA5_E5EDu64);
        let hist = raytrace_hoa_histogram_per_band(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let (envelopes, maxima) = band_relative_envelopes(&hist, n_bins);
        assert!(maxima[BAND] > 0.0, "absorbing band recorded no late energy");
        let e_rel = &envelopes[BAND];
        // Last non-zero sample, so a tail that simply ran out of rays cannot
        // masquerade as a deep decay.
        let min_positive = e_rel
            .iter()
            .copied()
            .filter(|&e| e > 0.0)
            .fold(f64::INFINITY, f64::min);
        let min_db = 20.0 * min_positive.log10();
        assert!(
            min_positive < 0.10,
            "band {BAND} envelope bottomed at {min_positive:.4} ({min_db:.1} dB); \
             the old global clamp pinned it at 0.15 (−16.48 dB)"
        );
        let target = geom.eyring_rt60_band(&atm, BAND);
        let t30 = t30_from_amplitude_envelope(e_rel, bin_dt_s).unwrap_or_else(|| {
            panic!(
                "band {BAND} envelope does not span the −5…−35 dB T30 window \
                 (min {min_positive:.4}); the decay floor is back"
            )
        });
        eprintln!(
            "absorbing band {} Hz: T30 = {t30:.4} s, Eyring target = {target:.4} s, \
             envelope min = {min_positive:.3e} ({min_db:.1} dB)",
            ROOM_BANDS_HZ[BAND]
        );
        assert!(
            (t30 / target - 1.0).abs() < 0.20,
            "absorbing band T30 = {t30:.4} s vs Eyring target {target:.4} s \
             (ratio {:.3})",
            t30 / target
        );
    }

    /// A band with no recorded energy must normalise to all zeros (0/0 handled
    /// by the `1e-12` numerical floor), and a single non-zero bin must
    /// normalise to exactly 1.0 with no NaN anywhere.
    #[test]
    fn band_envelope_normalises_empty_bands_to_zero_without_nan() {
        let empty = vec![[[0.0_f64; HOA_CH]; ROOM_BAND_COUNT]; 4];
        let (env, maxima) = band_relative_envelopes(&empty, 4);
        assert_eq!(maxima, [0.0_f64; ROOM_BAND_COUNT]);
        for band in &env {
            assert!(band.iter().all(|&e| e == 0.0), "empty band is not zero");
        }

        let mut one = vec![[[0.0_f64; HOA_CH]; ROOM_BAND_COUNT]; 4];
        one[2][3][0] = 5.0;
        let (env, maxima) = band_relative_envelopes(&one, 4);
        assert_eq!(maxima[3], 5.0);
        assert_eq!(env[3][2], 1.0);
        assert_eq!(env[3][0], 0.0);
        for band in &env {
            assert!(band.iter().all(|e| e.is_finite()), "NaN in envelope");
        }
    }

    /// Per-path air attenuation is owned by the ray reduction: one synthetic
    /// arrival at each of two distances must carry exactly the ISO 9613-1
    /// energy attenuation `10^(−α(f)·d/10)` at the band centre.
    #[test]
    fn ray_reduction_applies_iso9613_air_once_at_two_distances() {
        let atm = AtmosphereCfg::default();
        let c = atm.speed_of_sound_m_s();
        let band = 8; // 8 kHz: air absorption is large enough to measure.
        let f = ROOM_BANDS_HZ[band];
        let alpha = atm.alpha_db_per_m(f);
        let bin_dt_s = 0.01;
        let n_bins = 64usize;
        let mut w = Vec::new();
        for &d in &[5.0_f64, 40.0] {
            let per_ray = vec![vec![NativeArrival {
                delay_s: d / c,
                energy: 1.0,
                dir: [0.0, 0.0, 1.0],
            }]];
            let hist = reduce_native_arrivals_at(&per_ray, &atm, n_bins, bin_dt_s, f);
            let bin = (d / c / bin_dt_s) as usize;
            w.push(hist[bin][0]);
            let sh = eval_sh_order7(0.0, std::f64::consts::FRAC_PI_2);
            let want = sh[0] * 10.0_f64.powf(-alpha * d / 10.0);
            assert!(
                (hist[bin][0] / want - 1.0).abs() < 1e-12,
                "ISO 9613-1 at {d} m, band {band}: W = {:e}, analytic = {want:e} \
                 (α = {alpha:.6e} dB/m)",
                hist[bin][0]
            );
        }
        let want_ratio = 10.0_f64.powf(alpha * 35.0 / 10.0);
        let got_ratio = w[0] / w[1];
        eprintln!(
            "8 kHz ISO 9613-1 α = {alpha:.5e} dB/m: W(5 m)/W(40 m) = {got_ratio:.4} \
             (analytic {want_ratio:.4})"
        );
        assert!((got_ratio / want_ratio - 1.0).abs() < 1e-12);
    }

    /// The late-tail spectral shading must be high-pass only. The old
    /// `hp_air` vector multiplied it by a second ISO 9613-1 air gain over the
    /// zone mean distance (capped at 60 m), so every late arrival was attenuated
    /// twice. At 16 kHz / 60 m that omitted factor is large.
    #[test]
    fn late_tail_shading_has_no_second_air_factor() {
        let atm = AtmosphereCfg::default();
        let f_xover = 500.0;
        for f in [1_000.0_f64, 8_000.0, 16_000.0] {
            let x = f / f_xover;
            let want = x * x / (1.0 + x * x);
            let got = tail_highpass_shading(f, f_xover);
            assert!(
                (got - want).abs() < 1e-15,
                "shading at {f} Hz = {got}, high-pass = {want}"
            );
        }
        let air_60m = atm.air_gain_linear(16_000.0, 60.0);
        eprintln!(
            "16 kHz ISO 9613-1 air gain over 60 m = {air_60m:.4} (double-counted before the fix)"
        );
        assert!(
            air_60m < 0.75,
            "test premise: 16 kHz / 60 m air gain {air_60m:.4} must be a measurable factor"
        );
        let hp = tail_highpass_shading(16_000.0, f_xover);
        assert!(
            (hp - hp * air_60m).abs() > 1e-3,
            "re-adding the zone air factor must be detectable at 16 kHz"
        );
    }

    /// CPU ↔ CUDA ten-band envelope parity. The two Monte Carlo streams differ,
    /// so the envelopes agree statistically; the integer-indexed bin/band
    /// normalisation itself is exact and deterministic for identical input
    /// (checked first). The float tolerance is the smallest bound that covers
    /// the measured difference on this host (0.0287).
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn per_band_cuda_envelope_parity_matches_native_at_measured_tolerance() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP per_band_cuda_envelope_parity_matches_native_at_measured_tolerance: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        // Measured worst over the ten bands on this host: 0.0287 (band 7).
        const MAX_NORMALIZED_ENVELOPE_L1: f64 = 0.05;
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (16_384usize, 24usize, 128usize, 0.002_f64, 0xE17E_10BEu64);

        let cpu = raytrace_hoa_histogram_per_band(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let gpu = raytrace_hoa_histogram_per_band_on_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .unwrap_or_else(|e| panic!("per-band CUDA raytrace failed: {e}"));

        let (cpu_env, cpu_max) = band_relative_envelopes(&cpu, n_bins);
        let (gpu_env, gpu_max) = band_relative_envelopes(&gpu, n_bins);
        // Deterministic integer-indexed stages: identical input must reproduce
        // the envelopes and maxima bit-for-bit.
        let (again_env, again_max) = band_relative_envelopes(&cpu, n_bins);
        assert_eq!(cpu_max, again_max, "band maxima are not deterministic");
        assert_eq!(cpu_env, again_env, "envelopes are not deterministic");

        let mut worst_l1 = 0.0_f64;
        for band in 0..ROOM_BAND_COUNT {
            assert!(
                cpu_max[band] > 0.0 && gpu_max[band] > 0.0,
                "band {band} ({:.0} Hz) has no late W energy: cpu={:e}, gpu={:e}",
                ROOM_BANDS_HZ[band],
                cpu_max[band],
                gpu_max[band]
            );
            let l1: f64 = cpu_env[band]
                .iter()
                .zip(gpu_env[band].iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>()
                / n_bins as f64;
            eprintln!(
                "band {band:2} ({:5.0} Hz): normalized-envelope L1 = {l1:.4}",
                ROOM_BANDS_HZ[band]
            );
            worst_l1 = worst_l1.max(l1);
        }
        eprintln!(
            "worst normalized-envelope L1 = {worst_l1:.4} (tol {MAX_NORMALIZED_ENVELOPE_L1})"
        );
        assert!(
            worst_l1 < MAX_NORMALIZED_ENVELOPE_L1,
            "native CPU vs GPU normalized band-envelope L1 = {worst_l1:.4} >= \
             {MAX_NORMALIZED_ENVELOPE_L1}"
        );
    }

    /// The per-band GPU stage must be genuinely GPU-resident: the
    /// `_with_cuda` wrapper dispatches the GPU-only kernel path, records
    /// `stage raytrace-hoa-bands: GPU` with no `CPU fallback`, and returns
    /// bit-identically what the direct GPU entry point returns. A host fallback
    /// would both record a fallback line and produce a statistically different
    /// native histogram, so neither can pass.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn per_band_gpu_stage_is_gpu_resident_and_records_no_fallback() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP per_band_gpu_stage_is_gpu_resident_and_records_no_fallback: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (4_096usize, 16usize, 32usize, 0.004_f64, 0x6D0B_11E5u64);

        let via_wrapper = raytrace_hoa_histogram_per_band_with_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
            true,
        );

        let report = crate::cuda_report::summary();
        let line = report
            .lines
            .iter()
            .find(|l| l.contains("stage raytrace-hoa-bands:"))
            .unwrap_or_else(|| {
                panic!(
                    "per-band GPU stage was not recorded at all: {:?}",
                    report.lines
                )
            });
        assert!(
            line.contains(": GPU ("),
            "per-band raytrace did not run on the GPU: {line}"
        );
        assert!(
            !line.contains("CPU fallback"),
            "per-band raytrace fell back to the CPU: {line}"
        );

        let direct = raytrace_hoa_histogram_per_band_on_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .unwrap_or_else(|e| panic!("direct per-band CUDA raytrace failed: {e}"));
        assert_eq!(
            via_wrapper, direct,
            "wrapper output is not bit-identical to the GPU-only entry point"
        );
        assert!(
            via_wrapper
                .iter()
                .any(|row| row.iter().any(|band| band.iter().any(|&x| x != 0.0))),
            "GPU per-band trace recorded no energy"
        );
    }

    /// Live GPU peer of the ten-band transport: one kernel launch per octave
    /// band, each with that band's wall tables and the same seed. The gate uses
    /// the same measured Monte Carlo bounds as the single-band CUDA gate.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn per_band_cuda_histogram_agrees_with_native_cpu_within_measured_tolerance() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP per_band_cuda_histogram_agrees_with_native_cpu_within_measured_tolerance: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        const MAX_REL_W_TOTAL_TOL: f64 = 0.13;
        const MAX_BAND_L1_TOL: f64 = 0.13;
        const MAX_ORDER_L1_TOL: f64 = 0.18;
        const MAX_MEAN_T_DIFF_TOL: f64 = 0.0015;
        const N_BANDS: usize = 8;
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (65_536usize, 24usize, 128usize, 0.002_f64, 0xCAFE_BABEu64);

        let cpu = raytrace_hoa_histogram_per_band(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        );
        let gpu = raytrace_hoa_histogram_per_band_on_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .unwrap_or_else(|e| panic!("per-band CUDA raytrace failed: {e}"));

        let mut worst = HistAgreement::default();
        for band in 0..ROOM_BAND_COUNT {
            let cpu_band: Vec<[f64; HOA_CH]> = cpu.iter().map(|row| row[band]).collect();
            let gpu_band: Vec<[f64; HOA_CH]> = gpu.iter().map(|row| row[band]).collect();
            let w_cpu: f64 = cpu_band.iter().map(|row| row[0]).sum();
            let w_gpu: f64 = gpu_band.iter().map(|row| row[0]).sum();
            assert!(
                w_cpu > 1e-9 && w_gpu > 1e-9,
                "band {band} ({:.0} Hz) has no late W energy: cpu={w_cpu:e}, gpu={w_gpu:e}",
                ROOM_BANDS_HZ[band]
            );
            let m = histogram_agreement(&cpu_band, &gpu_band, N_BANDS, bin_dt_s);
            eprintln!(
                "band {band:2} ({:5.0} Hz): rel W = {:.4}, band L1 = {:.4}, order L1 = {:.4}, \
                 mean t |d| = {:.3} ms",
                ROOM_BANDS_HZ[band],
                m.rel_w_total,
                m.band_l1,
                m.order_l1,
                m.mean_t_diff_s * 1e3
            );
            worst.rel_w_total = worst.rel_w_total.max(m.rel_w_total);
            worst.band_l1 = worst.band_l1.max(m.band_l1);
            worst.order_l1 = worst.order_l1.max(m.order_l1);
            worst.mean_t_diff_s = worst.mean_t_diff_s.max(m.mean_t_diff_s);
            worst.max_abs_cell = worst.max_abs_cell.max(m.max_abs_cell.abs());
        }
        eprintln!(
            "per-band worst over {} bands: rel W = {:.4} (tol {MAX_REL_W_TOTAL_TOL}), band L1 = \
             {:.4} (tol {MAX_BAND_L1_TOL}), order L1 = {:.4} (tol {MAX_ORDER_L1_TOL}), mean t |d| \
             = {:.3} ms (tol {:.1} ms)",
            ROOM_BAND_COUNT,
            worst.rel_w_total,
            worst.band_l1,
            worst.order_l1,
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
        assert!(
            worst.rel_w_total < MAX_REL_W_TOTAL_TOL,
            "native CPU vs GPU per-band W-energy rel diff = {:.4} >= {MAX_REL_W_TOTAL_TOL}",
            worst.rel_w_total
        );
        assert!(
            worst.band_l1 < MAX_BAND_L1_TOL,
            "native CPU vs GPU per-band temporal-envelope L1 = {:.4} >= {MAX_BAND_L1_TOL}",
            worst.band_l1
        );
        assert!(
            worst.order_l1 < MAX_ORDER_L1_TOL,
            "native CPU vs GPU per-band per-order power L1 = {:.4} >= {MAX_ORDER_L1_TOL}",
            worst.order_l1
        );
        assert!(
            worst.mean_t_diff_s < MAX_MEAN_T_DIFF_TOL,
            "native CPU vs GPU per-band mean arrival-time diff = {:.3} ms >= {:.1} ms",
            worst.mean_t_diff_s * 1e3,
            MAX_MEAN_T_DIFF_TOL * 1e3
        );
    }

    /// Every GPU band launch must be driven by its own wall tables: a
    /// single-band material change may alter only that band's histogram, and
    /// the other nine launches (same seed, same tables) must stay bit-identical.
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    fn per_band_cuda_perturbation_isolates_the_changed_band() {
        if crate::blackwell::shared_cuda_runtime().is_none() {
            eprintln!(
                "SKIP per_band_cuda_perturbation_isolates_the_changed_band: \
                 no live NVIDIA GPU detected"
            );
            return;
        }
        let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
        let atm = AtmosphereCfg::default();
        let src = [4.3, 3.5, 1.25];
        let (n_rays, max_bounces, n_bins, bin_dt_s, seed) =
            (4_096usize, 16usize, 32usize, 0.004_f64, 0xBA9D_5EEDu64);
        let base = raytrace_hoa_histogram_per_band_on_cuda(
            &geom,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .unwrap_or_else(|e| panic!("base per-band CUDA raytrace failed: {e}"));
        let mut sabotaged = geom.clone();
        let alpha = sabotaged.walls[5].absorption[8];
        sabotaged.walls[5].absorption[8] = (alpha * 1.10).min(0.99);
        let pert = raytrace_hoa_histogram_per_band_on_cuda(
            &sabotaged,
            &atm,
            src,
            n_rays,
            max_bounces,
            n_bins,
            bin_dt_s,
            seed,
        )
        .unwrap_or_else(|e| panic!("perturbed per-band CUDA raytrace failed: {e}"));

        let mut band8_changed = false;
        for (bin, (a, b)) in base.iter().zip(&pert).enumerate() {
            for band in 0..ROOM_BAND_COUNT {
                if band == 8 {
                    if a[band]
                        .iter()
                        .zip(b[band].iter())
                        .any(|(&x, &y)| x.to_bits() != y.to_bits())
                    {
                        band8_changed = true;
                    }
                } else {
                    for (ch, (&x, &y)) in a[band].iter().zip(b[band].iter()).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "GPU ceiling 8 kHz change leaked into bin={bin} band={band} ch={ch}"
                        );
                    }
                }
            }
        }
        assert!(
            band8_changed,
            "GPU ceiling 8 kHz alpha change did not alter the 8 kHz per-band histogram"
        );
    }
}
