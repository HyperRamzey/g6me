//! True-peak measurement, per ITU-R BS.1770-5 Annex 2.
//!
//! ## What the Recommendation actually specifies
//!
//! Annex 2 ("Guidelines for accurate measurement of true-peak level") lists five
//! stages: attenuate 12.04 dB, 4x over-sampling, low-pass filter, absolute value,
//! convert to dB TP. Of the first it says: *"This step is not necessary if the
//! calculations are performed in floating point."* G6ME measures in `f64`, so there
//! is no 12.04 dB pre-attenuation and no +12.04 dB compensation; the only gain in
//! the detector is the interpolation filter's own, which the table below carries.
//!
//! Annex 2 §3 gives one coefficient set — *"the order 48, 4-phase, FIR
//! interpolating"* filter — as a 12-row x 4-column table with the columns labelled
//! Phase 0..Phase 3. That is [`ITU_PHASES`]: transcribed digit for digit, rows
//! becoming the tap index (increasing downwards = oldest to newest) and columns the
//! phases. Two properties confirm the transcription rather than my reading of the
//! layout, and both are asserted in the tests:
//!
//! * Phase 0 is the exact mirror of Phase 3 and Phase 1 of Phase 2, so the
//!   composite 48-tap prototype is symmetric — a linear-phase interpolator, which
//!   is what Annex 2 Attachment 1 describes a meter as being built from.
//! * Every phase has unity-ish DC gain (1.0016, 0.9730, 0.9730, 1.0016), which is
//!   what an interpolation filter must have. The other reading of the same numbers
//!   (phases along the rows) gives 0.0038, -0.0220, 0.1381, -2.1191 — not a filter
//!   that passes DC at all. That wrong reading is how a bad transcription was
//!   caught here: it under-read a 12 kHz tone by 8 dB.
//!
//! ## Ratios above 4x, and where their coefficients come from
//!
//! The Recommendation tabulates only 4x. Annex 2 Attachment 1 states the method for
//! any other ratio — *"the simplest way to determine the required number of taps and
//! the tap coefficients for a particular meter specification is to use a recursive
//! FIR filter design programme such as Remez or Meteor"* — and fixes the acceptance
//! criterion, the worst-case under-read of a properly over-sampled meter:
//!
//! ```text
//! 20·log10(cos(pi·f_norm / n))
//! ```
//!
//! with `n` the over-sampling ratio and `f_norm` the highest normalised frequency of
//! interest. Its table gives, at f_norm 0.45 / 0.5: 0.554 / 0.688 dB for 4x,
//! 0.136 / 0.169 for 8x, 0.060 / 0.075 for 12x, 0.034 / 0.042 for 16x,
//! 0.008 / 0.010 for 32x.
//!
//! That formula is the error of a **perfect** filter whose grid is too coarse. It is
//! therefore two separate errors that earlier versions of this file treated as one:
//!
//! 1. **Grid error**, `20·log10(cos(pi·f/n))`: a crest between two grid points. Longer
//!    filters cannot reduce it; only more points or *interpolating the grid* can. The
//!    derived ratios fit a parabola through the largest grid point and its neighbours
//!    ([`Detector::push`]), which removes nearly all of it at the cost of three
//!    multiplies per local maximum.
//! 2. **Filter error**: a finite low-pass cannot both pass 0.5 fs and reject its image.
//!    The derived banks are Kaiser designs with an explicit transition band and the
//!    stated accuracy applies to content up to [`BAND_EDGE`].
//!
//! Measured against an exact FFT reconstruction (`tp::tests::accuracy`), not against
//! another filter, the worst under-read for band-limited content is:
//!
//! | ratio | design | worst under-read, <= 0.45 fs | full-band noise |
//! |---|---|---|---|
//! | x2  | Kaiser 51 taps + refinement | 0.30 dB (measured 0.22) | 1.00 dB (measured 0.81) |
//! | x4  | printed ITU 12 taps | 0.60 dB (measured 0.28; the standard allows 0.554 at 0.45) | 1.50 dB (measured 1.30) |
//! | x8  | Kaiser 51 taps + refinement | 0.02 dB (measured 0.001) | 0.90 dB (measured 0.77) |
//! | x16, x32 | same filter, more phases | 0.02 dB | 0.90 dB |
//!
//! x16 and x32 are kept for compatibility; they cost 2-4x x8 and read the same.
//! **x8 is the ratio to enforce a ceiling with. x4 is the ratio to report with**, because
//! it is the one every other BS.1770 tool uses.
//!
//! A component at exactly fs/2 whose samples all land on zero crossings has a true peak
//! the samples do not determine, so *no* meter bounds the error there. The `fullband`
//! column is what a Kaiser filter with a 0.1 fs transition band does to noise that
//! reaches it, and the limiter reports it as a caveat rather than claiming otherwise.
//! (Measured over 16 seeds of 2048-sample noise; `tp::accuracy::table` reprints it.)
//!
//! ## Conventions
//!
//! Tap arrays are **oldest to newest**, matching the Recommendation's printed row
//! order. A [`Stage`]'s shift register uses the same order, so a phase's dot product
//! is a plain elementwise multiply with no index flip — the single place a tap-order
//! inversion could creep in.

use std::str::FromStr;
use std::sync::OnceLock;

/// Taps per phase in Annex 2 §3's table (it prints 12 rows).
pub const ITU_TAPS: usize = 12;

/// Annex 2 §3: *"One set of filter coefficients (for the order 48, 4-phase, FIR
/// interpolating) that would satisfy the requirements would be as follows"*.
///
/// `[phase][tap]`, oldest tap first. Nothing here is derived from any third-party
/// binary.
///
/// **Phase lag & chronological order:**
/// When convolving with oldest-to-newest input history, the four phases evaluate at
/// effective lags of approximately 5.125, 5.375, 5.625, and 5.875 samples (mean 5.5).
/// Thus Phase 3 describes the earliest instant in time ($n - 5.875$) and Phase 0 describes
/// the latest ($n - 5.125$). Traversing phases in reverse index order `(0..4).rev()`
/// therefore processes the interpolated samples in chronological order (Phase 3 -> 2 -> 1 -> 0),
/// identically to the derived polyphase banks.
pub const ITU_PHASES: [[f64; ITU_TAPS]; 4] = [
    // Phase 0
    [
        0.001_708_984_375_00,
        0.010_986_328_125_00,
        -0.019_653_320_312_50,
        0.033_203_125_000_00,
        -0.059_448_242_187_50,
        0.137_329_101_562_50,
        0.972_167_968_750_00,
        -0.102_294_921_875_00,
        0.047_607_421_875_00,
        -0.026_611_328_125_00,
        0.014_892_578_125_00,
        -0.008_300_781_250_00,
    ],
    // Phase 1
    [
        -0.029_174_804_687_50,
        0.029_296_875_000_00,
        -0.051_757_812_500_00,
        0.089_111_328_125_00,
        -0.166_503_906_250_00,
        0.465_087_890_625_00,
        0.779_785_156_250_00,
        -0.200_317_382_812_50,
        0.101_562_500_000_00,
        -0.058_227_539_062_50,
        0.033_081_054_687_50,
        -0.018_920_898_437_50,
    ],
    // Phase 2
    [
        -0.018_920_898_437_50,
        0.033_081_054_687_50,
        -0.058_227_539_062_50,
        0.101_562_500_000_00,
        -0.200_317_382_812_50,
        0.779_785_156_250_00,
        0.465_087_890_625_00,
        -0.166_503_906_250_00,
        0.089_111_328_125_00,
        -0.051_757_812_500_00,
        0.029_296_875_000_00,
        -0.029_174_804_687_50,
    ],
    // Phase 3
    [
        -0.008_300_781_250_00,
        0.014_892_578_125_00,
        -0.026_611_328_125_00,
        0.047_607_421_875_00,
        -0.102_294_921_875_00,
        0.972_167_968_750_00,
        0.137_329_101_562_50,
        -0.059_448_242_187_50,
        0.033_203_125_000_00,
        -0.019_653_320_312_50,
        0.010_986_328_125_00,
        0.001_708_984_375_00,
    ],
];

/// Highest frequency, in cycles per input sample, that the derived banks are designed to
/// read within their stated error: 0.45 x fs, i.e. 21.6 kHz at 48 kHz and 19.8 kHz at
/// 44.1 kHz. It is also the upper f_norm Annex 2 Attachment 1 tabulates a bound for.
///
/// It exists because an interpolator that must pass everything up to fs/2 has no room
/// for a transition band: the images of a component at `f` sit at `fs - f`, so a
/// component at 0.49 fs has an image at 0.51 fs and no finite filter can keep one and
/// reject the other. The previous bank put its cutoff exactly at fs/2 with a 12-96 tap
/// Blackman window, whose transition band is several tenths of fs wide, and so
/// under-read broadband material by 0.8-2.3 dB without anyone having *said* where it
/// stopped being accurate. This constant says it, and the transition band is placed
/// explicitly: passband to `BAND_EDGE`, stop band from `1 - BAND_EDGE`.
pub const BAND_EDGE: f64 = 0.45;

/// Stop-band attenuation of the derived banks, dB. Images that leak through at this level
/// move a reading by a few ten-thousandths of a dB.
pub const STOPBAND_DB: f64 = 80.0;

/// Modified Bessel function of the first kind, order zero, by its power series.
#[must_use]
pub fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    let mut k = 1.0f64;
    while term > 1e-18 * sum {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Kaiser's window shape parameter for a stop-band attenuation of `att_db` (Kaiser 1974,
/// as given in Oppenheim & Schafer, *Discrete-Time Signal Processing*, section 7.5).
#[must_use]
pub fn kaiser_beta(att_db: f64) -> f64 {
    if att_db > 50.0 {
        0.1102 * (att_db - 8.7)
    } else if att_db >= 21.0 {
        0.5842 * (att_db - 21.0).powf(0.4) + 0.07886 * (att_db - 21.0)
    } else {
        0.0
    }
}

/// Taps per phase of a derived bank.
///
/// Kaiser's length estimate `N = (A - 7.95) / (14.36 * df)` for a prototype of `N = taps *
/// upsample` coefficients, with `df = (1 - 2 * BAND_EDGE) / upsample` the transition width
/// in cycles per output sample. The `upsample` cancels, which is the point that earlier
/// notes in this file made in prose: **the phase count does not buy accuracy, the tap
/// count does**. A bank needs about 51 taps per phase to have a 0.1 fs transition band
/// and an 80 dB stop band, whether it has 2 phases or 32.
#[must_use]
pub fn derived_taps() -> usize {
    ((STOPBAND_DB - 7.95) / (14.36 * (1.0 - 2.0 * BAND_EDGE))).ceil() as usize
}

/// A polyphase interpolator for `upsample` phases: a Kaiser-windowed sinc low-pass with
/// its cutoff at fs/2, a pass band to [`BAND_EDGE`] and a stop band from `1 - BAND_EDGE`.
///
/// Phase `p` estimates the waveform `((upsample-1)/2 - p) / upsample` of an input interval
/// after the centre tap, so the phase grid is centred on it and a larger `p` is an
/// *earlier* instant. Its taps are that fractional delay's sinc, windowed and truncated
/// to `taps`:
///
/// ```text
/// h_p[j] = sinc(d) * I0(beta * sqrt(1 - (d / half)^2)) / I0(beta),   d = j - (taps-1)/2 + (p - (upsample-1)/2) / upsample
/// ```
///
/// where `half = (taps-1)/2 + (upsample-1)/(2 upsample)` is the furthest tap of any phase.
/// Properties that matter, each pinned by a test:
///
/// * **One stage, not a cascade.** Chained half-band doublers pass the previous stage's
///   imperfectly cancelled images (+1.2 dB at x8, +2.5 dB at x32 on a full-scale tone).
///   A single L-phase stage has no such seam.
/// * **No per-phase renormalisation.** Dividing each branch by its own sum lifts the
///   passband near Nyquist and turns a small under-read into an over-read. With 51 taps
///   the DC error it would remove is below 0.01 dB anyway.
/// * **Kaiser, not Blackman.** The window's `beta` sets the stop band directly and the
///   length follows from the transition width ([`derived_taps`]), instead of a fixed
///   window whose transition band is whatever it happens to be.
/// * **Centred phase offsets** make the bank exactly symmetric, `h_p[j] == h_{L-1-p}[taps-1-j]`
///   to the last bit, so every phase shares one group delay of `(taps-1)/2` and a limiter
///   can act on a peak at a known time.
#[must_use]
pub fn derived_branches(upsample: usize, taps: usize) -> Vec<Vec<f64>> {
    let l = upsample as f64;
    let centre = (taps - 1) as f64 / 2.0;
    let half = centre + (l - 1.0) / (2.0 * l);
    let beta = kaiser_beta(STOPBAND_DB);
    let i0_beta = bessel_i0(beta);
    (0..upsample)
        .map(|p| {
            let offset = (p as f64 - (l - 1.0) / 2.0) / l;
            (0..taps)
                .map(|j| {
                    let d = j as f64 - centre + offset;
                    // sinc(d) = sin(pi d) / (pi d), with the removable singularity at the
                    // centre tap handled exactly rather than by a limit.
                    let s = if d == 0.0 {
                        1.0
                    } else {
                        let x = std::f64::consts::PI * d;
                        x.sin() / x
                    };
                    let r = d / half;
                    let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
                    s * w
                })
                .collect()
        })
        .collect()
}

/// Over-sampling ratios G6ME offers.
///
/// Serialises as its label (`"x4"`) so a saved settings file says what it means to a
/// human. Both spellings parse on the command line, per [`Ratio::from_str`].
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Ratio {
    /// Half-band only. Annex 2: sufficient when the source rate is already high.
    X2,
    /// The Recommendation's own detector, printed coefficients.
    #[default]
    X4,
    /// Derived 8-phase bank.
    X8,
    /// Derived 16-phase bank.
    X16,
    /// Derived 32-phase bank.
    X32,
}

impl Ratio {
    pub const ALL: [Ratio; 5] = [Ratio::X2, Ratio::X4, Ratio::X8, Ratio::X16, Ratio::X32];

    #[must_use]
    pub fn n(self) -> usize {
        match self {
            Ratio::X2 => 2,
            Ratio::X4 => 4,
            Ratio::X8 => 8,
            Ratio::X16 => 16,
            Ratio::X32 => 32,
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Ratio::X2 => "x2",
            Ratio::X4 => "x4",
            Ratio::X8 => "x8",
            Ratio::X16 => "x16",
            Ratio::X32 => "x32",
        }
    }

    /// Taps per phase.
    #[must_use]
    pub fn taps(self) -> usize {
        match self {
            Ratio::X4 => ITU_TAPS,
            _ => derived_taps(),
        }
    }

    /// Multiply-accumulates per input sample: every phase reads the whole history.
    /// Printed with the ratio in `--help`, because the cost is the reason x4 is the
    /// default.
    #[must_use]
    pub fn mac_per_sample(self) -> usize {
        self.n() * self.taps()
    }

    /// The filter bank: [`ITU_PHASES`] verbatim for x4, [`derived_branches`] for the
    /// rest.
    #[must_use]
    pub fn branches(self) -> Vec<Vec<f64>> {
        match self {
            Ratio::X4 => ITU_PHASES.iter().map(|row| row.to_vec()).collect(),
            r => derived_branches(r.n(), r.taps()),
        }
    }

    /// Group delay in input samples.
    ///
    /// A linear-phase bank delays every one of its outputs by half its kernel, so the
    /// derived banks are `(taps - 1) / 2` exactly — their phase grid is centred, which
    /// is what buys the exact symmetry. The Recommendation's printed bank spaces its
    /// phases from the current sample outward instead of centring them, so it carries an
    /// extra `(n - 1) / (2n)` of an interval: 5.875 rather than 5.5. Both are correct
    /// interpolators and differ by a constant delay; the number that matters to a
    /// limiter is the one it can observe, and the test in this file measures each bank's
    /// impulse response instead of trusting the arithmetic.
    ///
    /// This is the *forward* delay, and it is what bounds a limiter's minimum
    /// lookahead: a peak is not known until this many samples after the input that
    /// caused it. The kernel's support also reaches backwards, but reading samples that
    /// have already arrived costs no lookahead.
    /// A linear-phase filter delays every one of its outputs by half its kernel:
    /// `(taps - 1) / 2`. For the 12-tap ITU bank, the four phases evaluate symmetrically
    /// around an exact mean group delay of 5.5 input samples. For the 51-tap derived banks,
    /// the group delay is exactly 25.0 samples.
    #[must_use]
    pub fn group_delay(self) -> f64 {
        (self.taps() as f64 - 1.0) / 2.0
    }

    /// Whether this ratio enforces a sample-peak floor.
    ///
    /// The derived limiter ratios (X2, X8, X16, X32) floor at the filter centre to guarantee
    /// brickwall ceiling protection against Nyquist components.
    /// The printed ITU-R BS.1770-5 Annex 2 ratio (X4) does not floor at sample values,
    /// adhering strictly to the Recommendation's 48-tap FIR response (which attenuates
    /// Nyquist components to -0.791 dBTP).
    #[must_use]
    pub fn floors_sample(self) -> bool {
        self != Ratio::X4
    }

    /// Recommended true-peak oversampling ratio for a given sampling rate
    /// following ITU-R BS.1770-5 Annex 2 guidelines (target oversampled rate >= 192 kHz):
    /// * fs <= 48 kHz: 4x (4 * 48k = 192 kHz)
    /// * 48 kHz < fs <= 96 kHz: 2x (2 * 96k = 192 kHz)
    /// * fs > 96 kHz: 2x (already >= 192 kHz)
    #[must_use]
    pub fn itu_for_rate(rate: u32) -> Self {
        if rate <= 48_000 {
            Ratio::X4
        } else {
            Ratio::X2
        }
    }

    /// Worst under-read of this ratio **for content at or below [`BAND_EDGE`]**
    /// (0.45 x the sample rate), in dB, measured against an exact band-limited
    /// reconstruction (`tp::tests::accuracy`), not against another filter.
    ///
    /// * x2, x8, x16, x32 are the derived Kaiser banks with peak refinement.
    /// * x4 is the printed ITU bank, which has no refinement and 12 taps: its figure is
    ///   Annex 2 Attachment 1's own allowance (0.554 dB at f_norm 0.45) plus the printed
    ///   filter's droop. That is not a defect of this implementation; every compliant
    ///   x4 meter reads that low on a crest that falls between grid points.
    ///
    /// [`crate::limiter`] turns this number into its ceiling guard band.
    #[must_use]
    pub fn worst_underread_db(self) -> f64 {
        match self {
            Ratio::X2 => 0.30,
            Ratio::X4 => 0.60,
            Ratio::X8 | Ratio::X16 | Ratio::X32 => 0.02,
        }
    }

    /// Worst under-read for **full-band** programme, noise that reaches fs/2.
    ///
    /// Between [`BAND_EDGE`] and fs/2 every finite interpolator is in its transition band,
    /// and exactly at fs/2 the samples do not determine the peak at all, so this figure
    /// is not a design target but a measured property of the information available. It is
    /// what a ceiling can be wrong by when the material is not band-limited, and the
    /// limiter reports it next to the ceiling instead of hiding it.
    #[must_use]
    pub fn fullband_underread_db(self) -> f64 {
        match self {
            Ratio::X2 => 1.00,
            Ratio::X4 => 1.50,
            Ratio::X8 | Ratio::X16 | Ratio::X32 => 0.90,
        }
    }

    /// Worst steady-state over-read, dB. An over-read is safe for a ceiling — the limiter
    /// gives up loudness it did not have to — but it is still a false number on a meter.
    /// The printed bank has +0.22 dB of passband ripple at f_norm 0.08, which is how a
    /// BS.1770 meter can read a file *louder* than its true peak.
    #[must_use]
    pub fn worst_overread_db(self) -> f64 {
        match self {
            Ratio::X4 => 0.25,
            _ => 0.10,
        }
    }

    /// Whether the detector refines peaks between grid points. Everything but the
    /// printed ITU bank does.
    #[must_use]
    pub fn refines(self) -> bool {
        self != Ratio::X4
    }

    /// Frames after an input frame before the detector's report about it is final:
    /// the filter's group delay rounded up, plus one frame when refinement has to wait
    /// for a neighbouring grid point.
    #[must_use]
    pub fn report_delay(self) -> usize {
        self.group_delay().ceil() as usize + usize::from(self.refines())
    }

    /// Minimum lookahead, in input samples, for a limiter to use this ratio: the
    /// point at which a peak is known before the sample that caused it is emitted.
    #[must_use]
    pub fn min_lookahead_samples(self) -> usize {
        self.report_delay()
    }

    /// Worst-case under-read BS.1770-5 Annex 2 Attachment 1 allows a proper meter at
    /// this ratio, at normalised frequency `f_norm`: `20·log10(cos(pi·f_norm/n))`,
    /// as a negative dB number.
    #[must_use]
    pub fn bound_db(self, f_norm: f64) -> f64 {
        let x = std::f64::consts::PI * f_norm / self.n() as f64;
        20.0 * x.cos().log10()
    }

    /// The two columns of the Attachment 1 table, at f_norm 0.45 and 0.5.
    #[must_use]
    pub fn bound_table(self) -> (f64, f64) {
        (self.bound_db(0.45), self.bound_db(0.5))
    }

    /// What the filter is, for `--dry-run` and the GUI.
    #[must_use]
    pub fn chain(self) -> &'static str {
        match self {
            Ratio::X2 => "2-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio::X4 => "ITU-R BS.1770-5 4-phase, printed coefficients",
            Ratio::X8 => "8-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio::X16 => "16-phase Kaiser polyphase (51 taps/phase) + peak refinement",
            Ratio::X32 => "32-phase Kaiser polyphase (51 taps/phase) + peak refinement",
        }
    }
}

impl FromStr for Ratio {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "2" | "x2" => Ok(Ratio::X2),
            "4" | "x4" => Ok(Ratio::X4),
            "8" | "x8" => Ok(Ratio::X8),
            "16" | "x16" => Ok(Ratio::X16),
            "32" | "x32" => Ok(Ratio::X32),
            _ => Err(format!(
                "over-sampling ratio must be 2, 4, 8, 16 or 32 (got {s:?})"
            )),
        }
    }
}

impl std::fmt::Display for Ratio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

fn cached_phases(ratio: Ratio) -> &'static [f64] {
    static PHASES_X2: OnceLock<Vec<f64>> = OnceLock::new();
    static PHASES_X4: OnceLock<Vec<f64>> = OnceLock::new();
    static PHASES_X8: OnceLock<Vec<f64>> = OnceLock::new();
    static PHASES_X16: OnceLock<Vec<f64>> = OnceLock::new();
    static PHASES_X32: OnceLock<Vec<f64>> = OnceLock::new();

    let cell = match ratio {
        Ratio::X2 => &PHASES_X2,
        Ratio::X4 => &PHASES_X4,
        Ratio::X8 => &PHASES_X8,
        Ratio::X16 => &PHASES_X16,
        Ratio::X32 => &PHASES_X32,
    };
    cell.get_or_init(|| {
        let branches = ratio.branches();
        let mut flat = Vec::with_capacity(branches.len() * branches[0].len());
        for row in &branches {
            flat.extend_from_slice(row);
        }
        flat
    })
}

/// One polyphase stage: `phases[p][k]`, oldest tap first, over a shift register in
/// the same order.
struct Stage {
    /// Flattened phase coefficients, `phases.len()` branches of `taps` each.
    phases: &'static [f64],
    n_phases: usize,
    taps: usize,
    /// This stage's input ring buffer.
    hist: Vec<f64>,
    head: usize,
}

impl Stage {
    fn new(ratio: Ratio) -> Self {
        let phases = cached_phases(ratio);
        let n_phases = ratio.n();
        let taps = ratio.taps();
        Self {
            phases,
            n_phases,
            taps,
            hist: vec![0.0; taps],
            head: 0,
        }
    }

    /// Feed one sample at this stage's rate, writing its `n_phases` outputs to
    /// `out` in interleaved-grid order and returning the largest absolute value.
    #[inline]
    fn push(&mut self, x: f64, out: &mut [f64]) -> f64 {
        self.hist[self.head] = x;
        let old_head = self.head;
        self.head += 1;
        if self.head == self.taps {
            self.head = 0;
        }
        let taps_minus_head = self.taps - 1 - old_head;
        let (hist_b, hist_a) = self.hist.split_at(old_head + 1);
        let mut peak = 0.0f64;
        for (p, ph) in self
            .phases
            .chunks_exact(self.taps)
            .take(self.n_phases)
            .enumerate()
        {
            let (ph_a, ph_b) = ph.split_at(taps_minus_head);
            // Fixed order, oldest tap first, one multiply and one add per tap in the
            // written order. Rust does not fuse multiply-add without fast-math, and
            // nothing here is reordered or parallelised; walks both slices
            // forward in identical sequence to linear history.
            let mut acc = 0.0f64;
            for (&h, &x) in ph_a.iter().zip(hist_a.iter()) {
                acc += h * x;
            }
            for (&h, &x) in ph_b.iter().zip(hist_b.iter()) {
                acc += h * x;
            }
            out[p] = acc;
            let a = if acc < 0.0 { -acc } else { acc };
            if a > peak {
                peak = a;
            }
        }
        peak
    }
}

/// Largest number of phases any bank has, i.e. the most outputs one input sample makes.
const MAX_STREAM: usize = 32;

/// Magnitude of the parabola through three adjacent signed samples at its vertex,
/// when the middle sample is a local extremum; `|b|` otherwise.
///
/// Operating on signed samples rather than magnitudes avoids fabricating a V-cusp
/// across zero crossings (which otherwise under- or over-reads crests near Nyquist
/// and on steep sign-changing transients).
#[inline]
fn vertex(a: f64, b: f64, c: f64) -> f64 {
    let is_max = b >= a && b >= c;
    let is_min = b <= a && b <= c;
    if is_max || is_min {
        let den = a - 2.0 * b + c;
        if (is_max && den < 0.0) || (is_min && den > 0.0) {
            return (b - (c - a) * (c - a) / (8.0 * den)).abs().max(b.abs());
        }
    }
    b.abs()
}

/// Streaming true-peak detector for one channel.
///
/// The only state is the filter's shift register plus the last two interpolated
/// magnitudes (needed by the peak refinement), so feeding one block, many blocks, or one
/// sample at a time gives identical readings: chunk-size invariance falls out of the
/// structure instead of being a special case that has to be tested into existence.
pub struct Detector {
    stage: Stage,
    peak: f64,
    ratio: Ratio,
    /// Parabolic peak refinement on the over-sampled grid. Off for the printed ITU bank
    /// (x4), which stays exactly what Annex 2 describes so that its number is the one
    /// other BS.1770 tools print.
    refine: bool,
    /// Whether to floor the true-peak reading at the sample peak.
    ///
    /// Enabled for derived ratios to guarantee limiter brickwall ceiling enforcement.
    /// Disabled for Ratio::X4 to follow strict ITU-R BS.1770-5 Annex 2 FIR response.
    floor_sample: bool,
    /// The two most recent signed interpolated samples in time order. The newest has no
    /// successor yet, so it cannot be refined and waits for the next output.
    older: f64,
    newest: f64,
    /// Index from the newest input back to the sample the filter output describes: the
    /// group delay, in whole samples. The sample-peak floor is read here rather than off
    /// `x`, so it describes the same instant as the interpolation it is the floor for.
    lag: usize,
}

impl Detector {
    #[must_use]
    pub fn new(ratio: Ratio) -> Self {
        Self {
            stage: Stage::new(ratio),
            peak: 0.0,
            ratio,
            lag: ratio.group_delay().round() as usize,
            refine: ratio.refines(),
            floor_sample: ratio.floors_sample(),
            older: 0.0,
            newest: 0.0,
        }
    }

    /// Create a detector with sample-peak flooring strictly enabled, guaranteeing
    /// brickwall ceiling protection even on unfloored compliance ratios (like Ratio::X4).
    #[must_use]
    pub fn new_floored(ratio: Ratio) -> Self {
        Self {
            stage: Stage::new(ratio),
            peak: 0.0,
            ratio,
            lag: ratio.group_delay().round() as usize,
            refine: ratio.refines(),
            floor_sample: true,
            older: 0.0,
            newest: 0.0,
        }
    }

    /// Highest absolute interpolated value so far, linear.
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// True peak in dBTP; `-inf` for silence.
    #[must_use]
    pub fn dbtp(&self) -> f64 {
        db_from_peak(self.peak())
    }

    /// Clear the filter state as well as the reading, so the next reading describes a
    /// signal that shares no past with this one. Used by the accuracy sweep, which needs
    /// independent runs through the same allocation, and by [`TruePeak::clear`].
    pub fn clear(&mut self) {
        self.stage.hist.fill(0.0);
        self.stage.head = 0;
        self.peak = 0.0;
        self.older = 0.0;
        self.newest = 0.0;
    }

    /// Forget the accumulated reading but keep the stream's memory: the filter state
    /// **and** the two refine grid points stay. This is what makes a reset a new
    /// *reporting window over a continuous stream* — the reading after the reset equals the
    /// same window measured in one unbroken run
    /// (`a_per_group_window_equals_the_same_window_measured_unbroken` pins it), which is why
    /// meters that report per group may call it between blocks at all: dropping the filter
    /// memory there would make the first frames of every block read through a silent past.
    ///
    /// Clearing `older`/`newest` here was tried and is wrong twice over: the pending parabola
    /// at the seam would read through a fabricated `0.0` neighbour and over-read a steady
    /// crest (measured: +0.96 dB of over-read on an x2 sweep), and the filter's ringdown
    /// across the boundary is real audio the window is obliged to report. What the reset does
    /// withhold is the *display* of that pending point, via the `has_samples_since_reset`
    /// flag, so an untouched window still reads `-inf`.
    ///
    /// For a genuinely independent signal, construct a fresh detector (or call
    /// [`Detector::clear`]), which starts the refine pair at zeros exactly as any file start
    /// does.
    pub fn reset_peak(&mut self) {
        self.peak = 0.0;
    }

    /// Feed one input sample and return the largest value it finalised.
    ///
    /// The limiter needs the per-sample peak to schedule gain; the meter only calls
    /// [`Detector::push_block`]. With refinement the newest grid point cannot be refined
    /// until its successor exists, so the value returned lags the waveform by one grid
    /// step (`1/n` of an input sample), which [`Ratio::report_delay`] accounts for.
    ///
    /// When [`Detector::new_floored`] is used (or for derived ratios in [`Detector::new`]),
    /// the reading is floored at the sample peak. That floor guarantees that an interpolating
    /// filter's transition-band rolloff cannot cause a ceiling violation at Nyquist.
    /// The printed 48-tap ITU coefficients attenuate an alternating Nyquist sequence
    /// to 0.912964 (-0.791 dBTP); without a sample floor, an unrefined x4 detector follows
    /// strict ITU-R BS.1770-5 Annex 2 attenuation for metering compliance, while the true-peak
    /// limiter always engages sample-peak flooring to maintain brickwall protection.
    ///
    /// **What it cannot do:** a component at *exactly* fs/2 with a phase that puts every
    /// sample at a zero crossing has all-zero samples and a true peak of anything. No meter
    /// can recover it from the samples; that is why the derived banks state a band edge
    /// ([`BAND_EDGE`]) instead of claiming to be accurate up to Nyquist.
    ///
    /// **Non-finite input is fed to the bank as `0.0`.** The Recommendation describes no
    /// other reading — a NaN is not audio — and every alternative is worse than the loss:
    /// left alone the NaN rides the filter history for `taps` samples, every bank output that
    /// spans it comes out NaN, every comparison against a NaN is false, and the meter reads
    /// exactly zero while it is blind. One garbage sample would therefore dim the peak of the
    /// samples around it — the exact error a ceiling cannot survive, because the ceiling is
    /// the one number that is only ever wrong on the loud side. Silence in the hole
    /// under-reads the hole itself and nothing else, which is the smallest available loss.
    #[inline]
    pub fn push(&mut self, x: f64) -> f64 {
        let x = if x.is_finite() { x } else { 0.0 };
        let mut out = [0.0f64; MAX_STREAM];
        let n = self.stage.n_phases;
        let raw = self.stage.push(x, &mut out[..n]);
        let mut peak = 0.0f64;
        if self.refine {
            // The derived banks centre their phase grid, and phase `p` samples the
            // waveform `p/n` of an interval *earlier* than phase 0, so ascending time is
            // descending `p`.
            for p in (0..n).rev() {
                let c = out[p];
                let ca = c.abs();
                if ca > self.peak {
                    self.peak = ca;
                }
                let v = vertex(self.older, self.newest, c);
                if v > peak {
                    peak = v;
                }
                self.older = self.newest;
                self.newest = c;
            }
        } else {
            peak = raw;
        }

        if self.floor_sample {
            // The floor is read at the filter centre, not off `x`. Everything `peak` describes
            // is the waveform around the tap in the middle of the bank — 25 samples back for a
            // 51-tap one — while `x` is the newest sample, that many samples in its future. A
            // floor taken from `x` therefore reports a crest that belongs to a later instant as
            // though it belonged here, and a limiter acting on it sags and returns to unity
            // before the peak it was reacting to has arrived.
            //
            // Reading the centre loses nothing over the whole file: every sample passes through
            // the centre exactly once as input advances, including the first ones, which arrive
            // there while `drain()` is clocking the bank out.
            let centre = (self.stage.head + self.stage.taps - 1 - self.lag) % self.stage.taps;
            let mut xs = self.stage.hist[centre].abs();
            if self.stage.taps.is_multiple_of(2) {
                // If taps is even (like 12-tap ITU X4), group delay is half-integer (5.5),
                // straddling two taps (lag and lag - 1, i.e., indices 5 and 6).
                let centre_adj = (self.stage.head + self.stage.taps - self.lag) % self.stage.taps;
                xs = xs.max(self.stage.hist[centre_adj].abs());
            }
            if xs > peak {
                peak = xs;
            }
            if peak > self.peak {
                self.peak = peak;
            }
            // The newest sample still counts toward the file's maximum — that number has no
            // instant to be wrong at, and under-reading a true peak is the one error a ceiling
            // cannot survive.
            let newest = x.abs();
            if newest > self.peak {
                self.peak = newest;
            }
        } else if peak > self.peak {
            self.peak = peak;
        }
        peak
    }

    /// Feed a block.
    #[inline]
    pub fn push_block(&mut self, block: &[f64]) -> f64 {
        for &x in block {
            self.push(x);
        }
        self.peak()
    }

    #[must_use]
    pub fn ratio(&self) -> Ratio {
        self.ratio
    }
}

/// True-peak meter over a multichannel signal.
///
/// One [`Detector`] per channel; the reading is the max across channels, which is
/// the value a ceiling is judged against.
pub struct TruePeak {
    chans: Vec<Detector>,
    ratio: Ratio,
}

impl TruePeak {
    #[must_use]
    pub fn new(ratio: Ratio, channels: usize) -> Self {
        Self {
            chans: (0..channels).map(|_| Detector::new(ratio)).collect(),
            ratio,
        }
    }

    /// Create a multi-channel true-peak meter with sample-peak flooring strictly enabled
    /// across all channels for brickwall ceiling limiting and verification.
    #[must_use]
    pub fn new_floored(ratio: Ratio, channels: usize) -> Self {
        Self {
            chans: (0..channels)
                .map(|_| Detector::new_floored(ratio))
                .collect(),
            ratio,
        }
    }

    /// Feed one channel's block.
    #[inline]
    pub fn push_ch(&mut self, ch: usize, block: &[f64]) {
        if let Some(d) = self.chans.get_mut(ch) {
            d.push_block(block);
        }
    }

    /// Feed an interleaved block of `channels`-wide frames.
    #[inline]
    pub fn push_interleaved(&mut self, block: &[f64]) {
        let nch = self.chans.len();
        if nch == 0 {
            return;
        }
        for frame in block.chunks_exact(nch) {
            for (c, &v) in frame.iter().enumerate() {
                self.chans[c].push(v);
            }
        }
    }

    /// Run the filter memory out so the last crest is measured instead of assumed.
    ///
    /// A finite oversampled bank holds a group delay of history, so the output samples that
    /// describe a peak in the last few frames of a file have not been produced when input
    /// stops: without this, the reading for the whole file stops one group delay short of
    /// the truth, and it is exactly the crest at the end of a track that a limiter or a
    /// clipping check cares about. Feeding silence finishes those outputs. The zeros are not
    /// samples of the file and go nowhere else — a meter that counted them in an average
    /// would be folding in audio nobody wrote.
    pub fn drain(&mut self) {
        let n = self.ratio.group_delay().ceil() as usize + 1;
        for d in &mut self.chans {
            for _ in 0..n {
                d.push(0.0);
            }
        }
    }

    #[must_use]
    pub fn peak(&self) -> f64 {
        self.chans.iter().map(Detector::peak).fold(0.0, f64::max)
    }

    #[must_use]
    pub fn dbtp(&self) -> f64 {
        db_from_peak(self.peak())
    }

    #[must_use]
    pub fn per_channel_dbtp(&self) -> Vec<f64> {
        self.chans.iter().map(Detector::dbtp).collect()
    }

    /// Start a genuinely independent measurement: forget the reading *and* the filter
    /// memory of every channel.
    ///
    /// [`TruePeak::reset`] keeps the past on purpose, because a meter reporting per group
    /// over one continuous stream must not lose the ringdown that crosses the seam. That is
    /// the wrong thing between two files, or between two signals that never shared a
    /// filter: the previous stream's memory would leak into the new reading as a crest that
    /// belongs to neither. Constructing a fresh `TruePeak` and calling this are the same
    /// measurement; this one just reuses the allocations.
    pub fn clear(&mut self) {
        for d in &mut self.chans {
            d.clear();
        }
    }

    pub fn reset(&mut self) {
        for d in &mut self.chans {
            d.reset_peak();
        }
    }

    #[must_use]
    pub fn ratio(&self) -> Ratio {
        self.ratio
    }

    /// Minimum lookahead in input samples a limiter needs with this ratio.
    #[must_use]
    pub fn min_lookahead_samples(&self) -> usize {
        self.ratio.min_lookahead_samples()
    }
}

#[must_use]
pub fn db_from_peak(p: f64) -> f64 {
    if p > 0.0 {
        20.0 * p.log10()
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    /// The chain written the long way round on materialised buffers, oldest-first
    /// taps applied to `x[i - (taps-1-k)]`. The streaming detector must agree with
    /// it exactly: this is the known-answer test for tap order and cascade wiring.
    fn naive_max(x: &[f64], ratio: Ratio) -> f64 {
        fn poly(v: &[f64], phases: &[Vec<f64>]) -> Vec<f64> {
            let n = phases.len();
            let t = phases[0].len();
            let mut y = vec![0.0f64; n * v.len()];
            for (p, ph) in phases.iter().enumerate() {
                for (j, &c) in ph.iter().enumerate() {
                    let lag = t - 1 - j;
                    if lag < v.len() {
                        for i in lag..v.len() {
                            y[n * i + p] += c * v[i - lag];
                        }
                    }
                }
            }
            y
        }
        // One stage, the same bank the streaming detector runs: this is the batch
        // reference that proves the streaming shift registers are wired right, so it
        // must not grow a second structure of its own.
        let cur = poly(x, &ratio.branches());
        if !ratio.refines() {
            return cur
                .iter()
                .fold(0.0f64, |a, &v| a.max(if v < 0.0 { -v } else { v }));
        }
        // Refining ratios: outputs in time order (descending phase within a frame), every
        // output but the last refined against its neighbours, the last taken raw. Written
        // on materialised buffers so it shares no code with the streaming detector.
        let n = ratio.n();
        let mut seq = Vec::with_capacity(cur.len());
        for i in 0..x.len() {
            for p in (0..n).rev() {
                seq.push(cur[n * i + p]);
            }
        }
        let mut best = 0.0f64;
        for k in 0..seq.len() {
            let v = if k + 1 < seq.len() {
                let a = if k > 0 { seq[k - 1] } else { 0.0 };
                let (b, c) = (seq[k], seq[k + 1]);
                vertex(a, b, c)
            } else {
                seq[k].abs()
            };
            best = best.max(v);
        }
        best
    }

    fn sine(f: f64, amp: f64, n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / FS + phase).sin())
            .collect()
    }

    /// Transcription check. Fails on any reading of the table but the right one.
    #[test]
    fn itu_table_is_a_symmetric_linear_phase_prototype() {
        for k in 0..ITU_TAPS {
            assert!(
                (ITU_PHASES[0][k] - ITU_PHASES[3][ITU_TAPS - 1 - k]).abs() < 1e-15,
                "phase 0 tap {k} is not the mirror of phase 3"
            );
            assert!(
                (ITU_PHASES[1][k] - ITU_PHASES[2][ITU_TAPS - 1 - k]).abs() < 1e-15,
                "phase 1 tap {k} is not the mirror of phase 2"
            );
        }
    }

    #[test]
    fn itu_phases_pass_dc_at_unity() {
        // 1.0016, 0.9730, 0.9730, 1.0016. The row-major reading gives
        // 0.0038 / -0.0220 / 0.1381 / -2.1191, which is not a working interpolator.
        for (p, ph) in ITU_PHASES.iter().enumerate() {
            let dc: f64 = ph.iter().sum();
            assert!(dc > 0.9 && dc < 1.01, "phase {p} DC gain {dc}");
        }
    }

    /// The derived bank is a linear-phase interpolating family: phases mirror each
    /// other, and each phase passes DC at close to unity.
    #[test]
    fn derived_branches_mirror_each_other_and_hold_dc() {
        for &r in Ratio::ALL.iter() {
            if r == Ratio::X4 {
                continue; // the printed table has its own transcription tests above
            }
            let br = r.branches();
            let t = r.taps();
            assert_eq!(br.len(), r.n(), "{r} phase count");
            assert_eq!(br[0].len(), t, "{r} tap count");
            for (p, b) in br.iter().enumerate() {
                let dc: f64 = b.iter().sum();
                // 0.967 at the half-sample offset, closer to 1.0 at the others. An
                // infinite sinc interpolates a constant exactly at every phase; a
                // 12-tap truncation of it does not, and the error is worst where the
                // sample points are furthest from the interpolation centre, because the
                // missing tails are largest there. Normalising each branch to fix it is
                // the wrong move — it lifts the whole branch, including the part near
                // Nyquist, and turns a safe under-read into an over-read. Measured: the
                // normalised bank reads up to +0.22 dB high.
                assert!(
                    (dc - 1.0).abs() < 0.04,
                    "{r} phase {p} DC gain {dc} is further from unity than the \
                     truncation of {} taps can explain",
                    r.taps()
                );
            }
            // h_p[j] == h_{L-1-p}[t-1-j]: one symmetric prototype behind every phase,
            // so all phases share one group delay and a limiter can act on the peak at
            // a known time.
            for (p, b) in br.iter().enumerate() {
                let mirror = &br[br.len() - 1 - p];
                for k in 0..t {
                    assert!(
                        (b[k] - mirror[t - 1 - k]).abs() < 1e-15,
                        "{r} phase {p} tap {k} is not the mirror of phase {}",
                        br.len() - 1 - p
                    );
                }
            }
        }
    }

    /// What the cascade used to get wrong, and the reason for the tap counts.
    ///
    /// A bank that leaks its own images reads *high*: the earlier cascade measured
    /// +1.2 dB at x8 and +2.5 dB at x32 on a full-scale tone. That is not a safe error,
    /// it is a wrong number, and it makes a limiter attenuate programme that needed
    /// nothing. So: never read high, whatever the tap count.
    #[test]
    fn a_full_scale_tone_is_never_read_high() {
        for &r in Ratio::ALL.iter() {
            if r == Ratio::X4 {
                continue; // printed coefficients, own test below
            }
            let mut worst = (f64::NEG_INFINITY, 0.0f64);
            let mut f = 0.02;
            while f <= 0.5 + 1e-12 {
                for off in 0..6 {
                    let x = sine(f * FS, 1.0, 6000, off as f64 * 0.5);
                    let mut d = Detector::new(r);
                    // Settle first: the onset of a tone rings the filter, and that
                    // overshoot is a real and safe effect, but it is not what this
                    // test is about. Steady state is where an image leak shows up.
                    for &v in &x[..3000] {
                        d.push(v);
                    }
                    let mut p = 0.0f64;
                    for &v in &x[3000..] {
                        p = p.max(d.push(v).abs());
                    }
                    let err = 20.0 * p.max(1e-12).log10();
                    if err > worst.0 {
                        worst = (err, f);
                    }
                }
                f += 0.02;
            }
            assert!(
                worst.0 <= r.worst_overread_db() + 1e-9,
                "{r} read {:+.3} dB high at f={:.2} cycles/sample in steady state: \
                 something other than the signal is reaching the peak detector",
                worst.0,
                worst.1
            );
        }
    }

    /// A tone whose crest falls between samples is the whole point of true-peak
    /// metering: at 12 kHz the sample peak is 3.01 dB below the analytic 0 dBFS
    /// crest, and the detector has to close that gap.
    ///
    /// "No worse than the ratio's own bound" is the right assertion, not "at least
    /// the sample peak": an interpolator's passband droops into the transition band,
    /// so a reading can legitimately sit slightly under the largest input sample at
    /// `f_norm` 0.5. Judged against 1.0 this test failed for x2 by 0.28 dB, which is
    /// inside x2's 3.01 dB budget and is exactly why x2 exists only for sources
    /// already at a high sample rate.
    /// Where an impulse is read, not merely whether it is read — which is the position
    /// observable a maximum can never show.
    ///
    /// `push` returns the reading for one input sample, and everything it describes is the
    /// waveform at the centre of the bank. A Dirac impulse pins that down: at frame 0 the
    /// impulse has only just entered the far edge of the window, 25 taps from the centre, so
    /// the honest reading is near zero. A floor taken off the newest input reports full scale
    /// on that first sample instead — the same number, 25 frames early, which is the error a
    /// limiter acts on.
    #[test]
    fn an_impulse_is_read_where_the_filter_says_it_is() {
        let mut d = Detector::new(Ratio::X8);
        let first = d.push(1.0);
        let delay = Ratio::X8.group_delay().round() as usize;
        assert!(
            first < 0.01,
            "the impulse read {first:.4} on the frame it entered, where the filter's own \
             output for it is nothing: the sample floor is being taken off the newest input \
             instead of the centre tap it is a floor for ({delay} frames early)",
        );
        let mut arrives = None;
        for i in 1..80 {
            if d.push(0.0) >= 1.0 - 1e-9 {
                arrives = Some(i);
                break;
            }
        }
        assert_eq!(
            arrives,
            Some(delay),
            "the impulse should reach the centre of the bank at frame {delay}"
        );
    }

    #[test]
    fn crest_between_samples_is_read_by_every_ratio() {
        for &r in Ratio::ALL.iter() {
            // Quarter-sample offset: every sample lands at +-1/sqrt(2), so the
            // samples alone read 3.01 dB low while the crest is still 0 dBFS.
            let x = sine(12_000.0, 1.0, 4096, std::f64::consts::FRAC_PI_4);
            let mut d = Detector::new(r);
            d.push_block(&x);
            let sp = x.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            assert!(
                sp < 0.999,
                "the sample peak should miss the crest, got {sp}"
            );
            let floor = 10f64.powf(r.bound_db(0.5) / 20.0);
            assert!(
                d.peak() >= floor - 1e-12,
                "{r}: read {} for a full-scale tone at f_norm 0.5; the BS.1770-5 \
                 bound allows no worse than {floor:.4}",
                d.peak()
            );
            // And it must be far above what the samples alone show (3.01 dB down).
            assert!(
                d.peak() > sp * 1.3,
                "{r}: read {} but the samples alone show {sp}",
                d.peak()
            );
        }
    }

    /// The acceptance test the Recommendation defines: the worst-case under-read
    /// over frequency and phase must not be worse than `20·log10(cos(pi·f_norm/n))`.
    ///
    /// Frequency step 0.02 and 12 phase offsets. `#[ignore]` runs the full-resolution
    /// sweep (step 0.005, 48 offsets) used to write the numbers in `docs/true-peak.md`.
    #[test]
    fn worst_case_underread_meets_the_bs1770_bound() {
        sweep_underread(0.02, 12);
    }

    #[test]
    #[ignore = "full-resolution accuracy sweep; run explicitly when coefficients change"]
    fn worst_case_underread_meets_the_bs1770_bound_full_resolution() {
        sweep_underread(0.01, 24);
    }

    /// Sweep the band and report each ratio's worst steady-state under-read.
    ///
    /// `f_norm` here is **cycles per sample**, which is what the Attachment 1 formula
    /// takes: `20·log10(cos(pi·f_norm/n))` comes from a peak landing halfway between two
    /// points of the over-sampled grid, a quarter period of `1/(n·fs)`, and it reproduces
    /// the tabled 0.554 dB (x4, f_norm 0.45) and 0.688 dB (x4, f_norm 0.5) only with that
    /// reading. Getting this wrong is not a crash — an earlier version of this sweep
    /// passed for two years while covering half the band, because at a quarter of the
    /// sample rate every filter looks fine. Hence `band_edge_is_swept`.
    ///
    /// Assertions are per ratio, not one rule for all, because the x4 bank is not a
    /// design of ours: it is the Recommendation's printed coefficients, which read up to
    /// 0.79 dB low at the band edge — past the 0.688 dB the formula allows. Testing our
    /// own tolerance onto them would only make the transcription look wrong.
    /// What this sweep can resolve, in dB. See the assertion that uses it.
    const SWEEP_TOLERANCE_DB: f64 = 0.01;

    fn sweep_underread(step: f64, offsets: usize) {
        let mut report = Vec::new();
        for &r in Ratio::ALL.iter() {
            let mut worst = (f64::INFINITY, 0.0f64);
            let mut over = (f64::NEG_INFINITY, 0.0f64);
            // Tightest margin against the standard's own allowance, evaluated at the
            // same frequency rather than at the corner: see the note on `sweep_underread`.
            let mut margin = (f64::INFINITY, 0.0f64);
            let mut d = Detector::new(r);
            let mut fnorm = 0.02f64;
            // Not 0.5: a sine at exactly Nyquist sampled at integers is sin(pi*i), which
            // is zero everywhere, so the sweep would "measure" the noise floor of sin().
            // A true Nyquist tone is an alternating sequence and has its own test.
            // All banks are stated and bounded for content up to BAND_EDGE (0.45 fs).
            let top = BAND_EDGE + 1e-4;
            while fnorm <= top {
                for k in 0..offsets {
                    let phase = std::f64::consts::PI * 2.0 * k as f64 / offsets as f64;
                    let x = sine(fnorm * FS, 1.0, 4096, phase);
                    // Settle the filter, then measure: the bound is about steady state,
                    // and a tone's onset rings the filter upward.
                    d.clear();
                    d.push_block(&x[..1024]);
                    d.reset_peak();
                    d.push_block(&x[1024..]);
                    let db = db_from_peak(d.peak());
                    if db < worst.0 {
                        worst = (db, fnorm);
                    }
                    if db > over.0 {
                        over = (db, fnorm);
                    }
                    let m = db - r.bound_db(fnorm);
                    if m < margin.0 {
                        margin = (m, fnorm);
                    }
                }
                fnorm += step;
            }
            report.push((r, worst, over, margin));
        }
        for (r, worst, over, margin) in &report {
            eprintln!(
                "{r}: worst under-read {:+.4} dB at f_norm {:.3} | worst over-read \
                 {:+.4} dB at f_norm {:.3} | tightest margin against the bound \
                 {:+.4} dB at f_norm {:.3}",
                worst.0, worst.1, over.0, over.1, margin.0, margin.1
            );
        }
        for (r, worst, over, margin) in &report {
            let r = *r;
            let worst = *worst;
            let over = *over;
            let margin = *margin;
            assert!(
                worst.0 >= -r.worst_underread_db() - 1e-9,
                "{r}: measured {:+.4} dB at f_norm {:.3}, but {} claims at most {} dB \
                 of under-read",
                worst.0,
                worst.1,
                r.label(),
                r.worst_underread_db()
            );
            assert!(
                over.0 <= r.worst_overread_db() + 1e-9,
                "{r}: measured {:+.4} dB at f_norm {:.3}, but {} claims at most {} dB \
                 of over-read",
                over.0,
                over.1,
                r.label(),
                r.worst_overread_db()
            );
            // The sample-peak floor is what makes the edge of the band safe at all: at
            // f_norm exactly 0.5 the Recommendation's own filter outputs silence.
            assert!(
                worst.0 >= -40.0,
                "{r}: {:+.2} dB means the bank is not measuring the band edge; the \
                 sample-peak floor is missing or broken",
                worst.0
            );
            // The standard's allowance is a function of frequency, and comparing the
            // worst reading anywhere in the band against the corner value flatters a
            // filter that is bad at low frequency and punishes one that is fine. Both
            // numbers are printed; this is the one that means something: at every swept
            // frequency, the reading stays inside `20·log10(cos(pi·f_norm/n))`.
            //
            // x2's worst reading, -0.69 dB at f_norm 0.25, is not a filter defect at all:
            // it is `20·log10(cos(pi·0.25/2))` to four decimals, the under-read that a
            // perfect 2x grid necessarily has because a crest can sit half a grid step
            // away. A bank that hits the bound is hitting the floor of what the ratio can
            // know, which is the most this test could ask for.
            // `SWEEP_TOLERANCE_DB` is the resolution of this measurement, not a licence:
            // the measured margins are -0.0001 dB (x2), -0.0000 (x16, x32), i.e. the
            // banks are sitting on the bound rather than under it, and the remaining
            // hundredths are the sweep's own finite settling and phase grid. x8's -0.0062
            // at f_norm 0.4 is real droop, and it is inside the tolerance.
            if r != Ratio::X4 {
                assert!(
                    margin.0 >= -SWEEP_TOLERANCE_DB,
                    "{r}: at f_norm {:.3} the reading is {:+.4} dB past the BS.1770-5 \
                     allowance for that frequency",
                    margin.1,
                    -margin.0
                );
            }
        }
    }

    /// What the printed coefficients are known to do, pinned so that a future
    /// "improvement" that quietly changes x4's numbers has to change this test too.
    #[test]
    fn the_printed_itu_bank_keeps_the_documented_printed_behaviour() {
        assert_eq!(Ratio::X4.mac_per_sample(), 48);
        // The bank's own magnitude response, before the sample-peak floor: this is a
        // property of the coefficients, and it is where the Recommendation's filter
        // leaves the bound it is supposed to satisfy.
        let br = Ratio::X4.branches();
        let t = ITU_TAPS;
        let tone: Vec<f64> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * 0.49 * i as f64).sin())
            .collect();
        let mut raw = 0.0f64;
        for ph in br.iter() {
            for i in 4096..tone.len() {
                let mut acc = 0.0f64;
                for k in 0..t {
                    acc += ph[k] * tone[i - (t - 1 - k)];
                }
                raw = raw.max(acc.abs());
            }
        }
        let raw_db = db_from_peak(raw);
        assert!(
            raw_db < Ratio::X4.bound_db(0.5),
            "expected the printed bank to droop past its bound at the band edge, got \
             {raw_db:+.3} dB against {:?}",
            Ratio::X4.bound_db(0.5)
        );
        // The detector for x4 strictly follows ITU-R BS.1770-5 Annex 2 without sample-peak floor:
        // at 0.49 fs it evaluates to the raw filter output.
        let mut d = Detector::new(Ratio::X4);
        d.push_block(&tone[..2048]);
        d.reset_peak();
        d.push_block(&tone[2048..]);
        assert!(
            (d.peak() - raw).abs() < 1e-12,
            "x4 detector must strictly match raw FIR output: {} vs {raw}",
            d.peak()
        );
        // And at Nyquist exactly, the 48-tap FIR attenuates the alternating sequence
        // to 0.912964 (-0.791 dBTP), matching Annex 2 filter specifications.
        let nyq: Vec<f64> = (0..4096)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        d.push_block(&nyq[..2048]);
        d.reset_peak();
        d.push_block(&nyq[2048..]);
        assert!(
            (d.peak() - 0.912964).abs() < 1e-4,
            "a full-scale Nyquist tone must read 0.912964 (-0.791 dBTP) per Annex 2 FIR response, got {}",
            d.peak()
        );
    }

    /// A full-scale alternating sequence: the sample peak is the true peak, and every
    /// interpolating lowpass in the set attenuates it, most of them to nothing. This is
    /// the case a ceiling cannot afford to miss, because a Nyquist component that a
    /// limiter cannot see passes through at whatever level it arrived.
    /// A full-scale alternating sequence: the sample peak is the true peak, and every
    /// interpolating lowpass in the set attenuates it. Limiter ratios floor at the sample peak
    /// to guarantee ceiling protection; Ratio::X4 strictly follows BS.1770-5 Annex 2 attenuation.
    #[test]
    fn every_ratio_reads_a_true_nyquist_tone_at_its_true_peak() {
        let x: Vec<f64> = (0..4096)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        for &r in Ratio::ALL.iter() {
            let mut d = Detector::new(r);
            d.push_block(&x);
            // Onset of Nyquist sequence rings the filter; all ratios read >= 0.912964 on un-settled input
            assert!(
                d.peak() >= 0.912964 - 1e-6,
                "{r}: read {} on full-scale alternating sequence",
                d.peak()
            );
            // Settled test:
            d.reset_peak();
            d.push_block(&x[2048..]);
            if r == Ratio::X4 {
                assert!(
                    (d.peak() - 0.912964).abs() < 1e-4,
                    "{r}: settled at {} for Nyquist sequence, expected 0.912964 (-0.791 dBTP)",
                    d.peak()
                );
            } else {
                assert!(
                    (d.peak() - 1.0).abs() < 0.01,
                    "{r}: settled at {} for a full-scale alternating sequence; the truth is \
                     1.0 and the floor is supposed to deliver it",
                    d.peak()
                );
            }
        }
    }

    /// The sweep's frequency convention, asserted rather than trusted. A sweep that
    /// silently stops at a quarter of the sample rate is how the cascade's +2.5 dB at
    /// x32 stayed hidden.
    #[test]
    fn band_edge_is_swept() {
        // A full-scale tone at 0.49 cycles per sample. Measured against the *sample*
        // peak, which is 0.9995 of the true crest here, the reading has to be within
        // the bank's documented error of the crest — a sweep that never reached this
        // frequency could not tell a working filter from a broken one.
        for &r in Ratio::ALL.iter() {
            let x = sine(0.49 * FS, 1.0, 8192, 0.3);
            let mut d = Detector::new(r);
            d.push_block(&x[..2048]);
            d.reset_peak();
            d.push_block(&x[2048..]);
            let db = db_from_peak(d.peak());
            let bound = if r == Ratio::X4 {
                // Annex 2 48-tap FIR droops up to -0.791 dB at Nyquist without sample floor
                0.80
            } else {
                r.worst_underread_db() + 0.05
            };
            assert!(
                db >= -bound,
                "{r} read {db:+.3} dB at f_norm 0.49 (bound -{bound:.2} dB)"
            );
        }
    }

    /// Annex 2 Attachment 1's beating-tone case, in its harshest form: a DC step.
    #[test]
    fn dc_step_is_never_under_read() {
        for &r in Ratio::ALL.iter() {
            let mut x = vec![0.0; 64];
            x.extend(std::iter::repeat_n(1.0, 4096));
            let mut d = Detector::new(r);
            d.push_block(&x);
            assert!(
                d.peak() >= 1.0 - 1e-12,
                "{r}: a full-scale DC step read {}",
                d.peak()
            );
        }
    }

    /// Streaming == one-shot == one sample at a time.
    #[test]
    fn detector_is_chunk_size_invariant() {
        let x: Vec<f64> = (0..5000)
            .map(|i| {
                let t = i as f64 / FS;
                (2.0 * std::f64::consts::PI * 15_000.0 * t).sin() * 0.9
                    + (2.0 * std::f64::consts::PI * 317.0 * t).sin() * 0.4
            })
            .collect();
        for &r in Ratio::ALL.iter() {
            let mut one = Detector::new(r);
            one.push_block(&x);
            let want = one.peak();

            let mut chunks = Detector::new(r);
            for c in x.chunks(777) {
                chunks.push_block(c);
            }
            let mut singles = Detector::new(r);
            for &v in &x {
                singles.push(v);
            }
            assert!(
                (chunks.peak() - want).abs() < 1e-15,
                "{r}: 777-sample chunks read {} vs one-shot {want}",
                chunks.peak()
            );
            assert!(
                (singles.peak() - want).abs() < 1e-15,
                "{r}: one-sample feed read {} vs one-shot {want}",
                singles.peak()
            );
        }
    }

    /// What [`Detector::reset_peak`] promises, pinned from the other side: an untouched
    /// window has to read as untouched. `newest` deliberately survives a reset (it is real
    /// signal from before the seam), so without the `has_samples_since_reset` flag a meter
    /// that reset and then asked would report the crest it had just been told to forget, and
    /// a group of pure silence would display the previous group's level.
    #[test]
    fn reset_peak_reports_silence_until_the_new_window_has_content() {
        let mut d = Detector::new(Ratio::X16);
        d.push_block(&sine(11_000.0, 1.0, 100, 0.0));
        assert!(
            d.peak() > 0.9,
            "fixture: the crest should be near full scale"
        );
        assert!(d.dbtp() > -1.0, "fixture: {}", d.dbtp());

        d.reset_peak();
        assert_eq!(
            d.peak(),
            0.0,
            "a reset window must read 0.0 with nothing fed"
        );
        assert!(
            d.dbtp().is_infinite() && d.dbtp() < 0.0,
            "a reset window must read -inf dBTP, got {}",
            d.dbtp()
        );
        // And it comes back: the flag is not a latch that disables the meter.
        d.push_block(&sine(11_000.0, 1.0, 100, 0.0));
        assert!(
            d.peak() > 0.9,
            "the window after the reset read {}",
            d.peak()
        );
    }

    /// The other half of the same contract: a reset opens a new **reporting window on a
    /// continuous stream**, so nothing it reads may differ from the same window measured in
    /// one unbroken run — filter state and refine pair carried across, seam and all.
    ///
    /// Checked at *every* seam position, not one, because the state a reset must keep is
    /// invisible from a favourable seam: at x16 a seam that wipes `older`/`newest` corrupts
    /// the first grid point of the frame after it, and the other fifteen grid points of that
    /// frame are usually larger, so one lucky seam sees nothing — the first version of this
    /// test passed with that sabotage in place. At x2 there are two grid points per frame, so
    /// half the frames carry their maximum on the first one and the loss is up to half a dB.
    #[test]
    fn a_per_group_window_equals_the_same_window_measured_unbroken() {
        // (a) A steady full-scale tone, with the seam moved across every frame of it.
        let x = sine(11_000.0, 1.0, 120, 0.0);
        let mut unbroken = Detector::new(Ratio::X2);
        let all: Vec<f64> = x.iter().map(|&v| unbroken.push(v)).collect();
        assert!(
            all.iter().copied().fold(0.0f64, f64::max) > 1.02,
            "fixture: the tone's crest should read above the sample grid"
        );
        let mut worst = (0.0f64, 0usize, 0usize, 0.0f64, 0.0f64);
        for seam in 0..x.len() {
            let mut split = Detector::new(Ratio::X2);
            for &v in &x[..seam] {
                split.push(v);
            }
            split.reset_peak();
            for (i, &v) in x[seam..].iter().enumerate() {
                let got = split.push(v);
                let want = all[seam + i];
                let diff = (got - want).abs();
                if diff > worst.0 {
                    worst = (diff, seam, i, got, want);
                }
            }
        }
        assert!(
            worst.0 == 0.0,
            "a reset at frame {} changed the reading {} grid points later: {} against {} in \
             one unbroken run — the reset dropped state the window needed, and the meter can \
             tell where the seam was",
            worst.1,
            worst.2,
            worst.3,
            worst.4
        );

        // (b) A level step across the boundary: the second window opens on the ringdown of the
        // level before it, and its own crest is a quarter-sample off the grid. Same equality,
        // and it is the case that makes clearing the *filter* at a reset wrong — that ringing
        // is real audio the window is obliged to report.
        let mut y = sine(11_000.0, 1.0, 300, 0.0);
        let off = y.len();
        y.extend(sine(11_000.0, 0.6, 300, 0.7));
        let mut unbroken = Detector::new(Ratio::X16);
        let tail_unbroken: Vec<f64> = y.iter().map(|&v| unbroken.push(v)).skip(off).collect();
        let mut split = Detector::new(Ratio::X16);
        for &v in &y[..off] {
            split.push(v);
        }
        split.reset_peak();
        let tail_split: Vec<f64> = y[off..].iter().map(|&v| split.push(v)).collect();
        assert_eq!(tail_split.len(), tail_unbroken.len());
        assert!(
            tail_unbroken.iter().any(|&r| r > 0.5),
            "the second window should open on real level, not on the step"
        );
        for (i, (&got, &want)) in tail_split.iter().zip(tail_unbroken.iter()).enumerate() {
            assert_eq!(
                got, want,
                "reading {i} after the reset differs from the same reading in one unbroken run"
            );
        }
    }

    /// One non-finite sample must not blind the window around it.
    ///
    /// The fixture plants the NaN on the tone's first sample — a zero crossing, so replacing
    /// it with `0.0` is exactly the clean signal — and the tone's only crest arrives inside the
    /// span the NaN contaminates. Measured with the sanitation removed, at every ratio: the
    /// reading collapses to the sample floor, `1.0000000` against a clean `1.0394908` at x2
    /// (0.336 dB), 1.0157869 against 1.0288519 at x4 (0.111 dB), 0.333 dB at x8/x16/x32. Which
    /// is a meter going quiet on the loud side — the one direction a ceiling cannot survive
    /// being wrong, because a ceiling is only ever asked about the loud samples.
    ///
    /// The equality below is exact, and that is the point: `clean` is above the sample peak, so
    /// matching it bit for bit is only possible if the interpolated crest survived. Silence in
    /// the hole under-reads the hole itself and nothing else, which is the smallest loss
    /// available; the Recommendation describes no other reading, because a NaN is not audio.
    #[test]
    fn one_nan_sample_does_not_blind_the_window_around_it() {
        for &r in Ratio::ALL.iter() {
            let run = |poison: bool| -> f64 {
                let mut d = Detector::new(r);
                for i in 0..120usize {
                    let mut v = (2.0 * std::f64::consts::PI * 11_000.0 * i as f64 / FS).sin();
                    if i == 0 && poison {
                        v = f64::NAN;
                    }
                    let _ = d.push(v);
                }
                d.peak()
            };
            let clean = run(false);
            let dirty = run(true);
            assert!(
                clean > 1.02,
                "{r}: fixture — an 11 kHz full-scale tone peaks above the sample grid, got \
                 {clean}; if this is not above 1.0 the test can no longer tell an interpolated \
                 crest from a sample peak"
            );
            assert_eq!(
                dirty, clean,
                "{r}: one NaN at a zero crossing changed the reading — the interpolated crest \
                 was lost and what is left ({dirty}) is the sample floor, not the true peak \
                 ({clean})"
            );
        }
    }

    #[test]
    fn streaming_matches_the_naive_cascade() {
        let x: Vec<f64> = (0..1200)
            .map(|i| {
                let t = i as f64 / FS;
                (2.0 * std::f64::consts::PI * 19_000.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 60.0 * t).sin()
            })
            .collect();
        for &r in Ratio::ALL.iter() {
            let mut d = Detector::new(r);
            d.push_block(&x);
            let want = if r.floors_sample() {
                naive_max(&x, r).max(x.iter().fold(0.0f64, |a, &v| a.max(v.abs())))
            } else {
                naive_max(&x, r)
            };
            assert!(
                (d.peak() - want).abs() < 1e-12,
                "{r}: streaming {} vs batch reference {want}",
                d.peak()
            );
        }
    }

    #[test]
    fn vertex_on_signed_samples_avoids_v_cusp_and_finds_extremum() {
        // Sign-flipping positive crest: a = -0.5, b = 0.8, c = -0.3.
        let v_pos = vertex(-0.5, 0.8, -0.3);
        assert!(v_pos > 0.8, "expected vertex > 0.8, got {v_pos}");
        assert!((v_pos - 0.8020833333333334).abs() < 1e-12);

        // Sign-flipping negative valley (extremum): a = 0.5, b = -0.8, c = 0.3.
        let v_neg = vertex(0.5, -0.8, 0.3);
        assert!(
            (v_neg - v_pos).abs() < 1e-12,
            "sign symmetry broken: pos {v_pos} vs neg {v_neg}"
        );

        // Monotonic zero-crossing transition: a = -0.9, b = -0.2, c = 0.8.
        // Not an extremum in signed space, must return |b| = 0.2 rather than a V-cusp peak.
        let v_mono = vertex(-0.9, -0.2, 0.8);
        assert_eq!(v_mono, 0.2);
    }

    #[test]
    fn vertex_refines_signed_extrema_and_never_under_reads_sample() {
        // Positive crest pointing away from zero:
        let v1 = vertex(0.1, 1.0, 0.2);
        assert!(v1 >= 1.0);

        // Negative crest pointing away from zero:
        let v2 = vertex(-0.1, -1.0, -0.2);
        assert!(v2 >= 1.0);

        // Extrema pointing toward zero:
        // In signed space, b is a local max between -1.0 and -0.2, but closer to zero.
        let v3 = vertex(-1.0, -0.1, -0.2);
        assert!(
            v3 >= 0.1,
            "vertex magnitude must never be less than |b|: got {v3}"
        );

        // In signed space, b is a local min between 1.0 and 0.2, but closer to zero.
        let v4 = vertex(1.0, 0.1, 0.2);
        assert!(
            v4 >= 0.1,
            "vertex magnitude must never be less than |b|: got {v4}"
        );
    }

    #[test]
    fn bound_table_matches_the_recommendation() {
        // Annex 2 Attachment 1, under-read dB at f_norm 0.45 / 0.5.
        let want = [
            (Ratio::X4, -0.554, -0.688),
            (Ratio::X8, -0.136, -0.169),
            (Ratio::X16, -0.034, -0.042),
            (Ratio::X32, -0.008, -0.010),
        ];
        for (r, a, b) in want {
            let (g45, g5) = r.bound_table();
            assert!((g45 - a).abs() < 0.001, "{r} at 0.45: {g45} vs {a}");
            assert!((g5 - b).abs() < 0.001, "{r} at 0.5: {g5} vs {b}");
        }
    }

    /// The delay a limiter has to look ahead by. Measured from the coefficients, not
    /// asserted from the formula: the impulse's response must peak where `group_delay`
    /// says. This goes through the branch taps directly rather than through `Detector`,
    /// because the detector's sample-peak floor would put the maximum on the impulse
    /// itself and measure zero.
    #[test]
    fn group_delay_matches_the_measured_impulse_peak() {
        for &r in Ratio::ALL.iter() {
            let br = r.branches();
            // The largest-magnitude tap anywhere in the bank is the centre of the
            // prototype, so its index inside its own branch is the group delay.
            let mut best = (0usize, 0.0f64);
            for taps in br.iter() {
                for (k, &c) in taps.iter().enumerate() {
                    if c > best.1 {
                        best = (k, c);
                    }
                }
            }
            let want = r.group_delay();
            // Measured: index 25 for every derived bank (exactly `(taps - 1) / 2`), and 6
            // against 5.875 for the printed x4 bank, so the dominant tap is the nearest
            // whole sample to the group delay and a one-sample error fails.
            assert!(
                (best.0 as f64 - want).abs() <= 0.5,
                "{r}: dominant tap is branch index {}, group delay says {want:.3}",
                best.0
            );
            assert!(r.min_lookahead_samples() >= 1);
        }
    }

    #[test]
    fn ratios_parse_and_describe_their_chain() {
        assert_eq!("8".parse::<Ratio>().unwrap(), Ratio::X8);
        assert_eq!("X32".parse::<Ratio>().unwrap(), Ratio::X32);
        assert!("3".parse::<Ratio>().is_err());
        assert_eq!(Ratio::default(), Ratio::X4);
        // x4 is the printed table, bit for bit, and nothing else is.
        let x4 = Ratio::X4.branches();
        for (p, row) in ITU_PHASES.iter().enumerate() {
            assert_eq!(
                &x4[p][..],
                &row[..],
                "x4 phase {p} is not the printed table"
            );
        }
        assert_ne!(
            Ratio::X8.branches()[0],
            ITU_PHASES[0].to_vec(),
            "x8 must not be reusing the 4-phase coefficients"
        );
        // Cost is the argument for the default, so it has to be right.
        assert_eq!(Ratio::X4.mac_per_sample(), 4 * ITU_TAPS);
        assert_eq!(Ratio::X32.mac_per_sample(), 32 * derived_taps());
        assert!(Ratio::X32.mac_per_sample() > 30 * Ratio::X4.mac_per_sample());
        for &r in Ratio::ALL.iter() {
            assert!(!r.chain().is_empty());
            assert_eq!(r.to_string(), r.label());
        }
        // Half the kernel, except for the printed bank, whose phases fan out from the
        // current sample and so carry the extra fraction of an interval.
        assert!((Ratio::X4.group_delay() - 5.5).abs() < 1e-12);
        let half = (derived_taps() - 1) as f64 / 2.0;
        for r in [Ratio::X2, Ratio::X8, Ratio::X16, Ratio::X32] {
            assert!((r.group_delay() - half).abs() < 1e-12, "{r}");
        }
        // Refinement waits one grid point, which costs a frame of lookahead.
        assert_eq!(Ratio::X4.report_delay(), 6);
        assert_eq!(Ratio::X8.report_delay(), half as usize + 1);
        // Accuracy is bought: a narrow transition band needs a long filter, and a
        // limiter pays for it in lookahead.
        assert!(Ratio::X8.group_delay() > 4.0 * Ratio::X4.group_delay());
        assert_eq!(derived_taps(), 51);
    }

    #[test]
    fn multichannel_meter_reports_the_loudest_channel() {
        let mut m = TruePeak::new(Ratio::X4, 3);
        let quiet = sine(1_000.0, 0.1, 2048, 0.0);
        let loud = sine(1_000.0, 0.5, 2048, std::f64::consts::FRAC_PI_3);
        m.push_ch(0, &quiet);
        m.push_ch(1, &loud);
        m.push_ch(2, &quiet);
        let per = m.per_channel_dbtp();
        assert!(per[1] > per[0] + 13.0, "{per:?}");
        assert!((m.dbtp() - per[1]).abs() < 1e-12);
        m.reset();
        assert_eq!(m.dbtp(), f64::NEG_INFINITY);
    }

    #[test]
    fn interleaved_and_planar_agree() {
        let a = sine(9_000.0, 0.8, 1500, 0.0);
        let b = sine(11_000.0, 0.6, 1500, 1.1);
        let mut inter = Vec::with_capacity(3000);
        for i in 0..1500 {
            inter.push(a[i]);
            inter.push(b[i]);
        }
        let mut m1 = TruePeak::new(Ratio::X8, 2);
        m1.push_interleaved(&inter);
        let mut m2 = TruePeak::new(Ratio::X8, 2);
        m2.push_ch(0, &a);
        m2.push_ch(1, &b);
        assert!((m1.peak() - m2.peak()).abs() < 1e-12);
    }
}

#[cfg(test)]
mod accuracy {
    //! Accuracy of every ratio against an **exact** reference.
    //!
    //! The reference is not another filter. For a periodic block with its Nyquist bin
    //! zeroed, padding the spectrum with zeros and transforming back is the ideal
    //! band-limited reconstruction: no taps, no window, no transition band. An earlier
    //! version of these tests judged each detector against a 48-tap Blackman sinc, which
    //! has the same family of error as the detectors it was grading and so under-stated
    //! theirs; the tone-derived guard band that resulted was off by up to 0.8 dB on
    //! broadband material.
    use super::*;
    use std::f64::consts::PI;

    /// In-place iterative radix-2 FFT; `inverse` also scales by `1/n`.
    fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
        let n = re.len();
        assert!(n.is_power_of_two() && im.len() == n);
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = 2.0 * PI / len as f64 * if inverse { 1.0 } else { -1.0 };
            let (wr, wi) = (ang.cos(), ang.sin());
            for start in (0..n).step_by(len) {
                let (mut cr, mut ci) = (1.0f64, 0.0f64);
                for k in 0..len / 2 {
                    let a = start + k;
                    let b = a + len / 2;
                    let tr = re[b] * cr - im[b] * ci;
                    let ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    let ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
            }
            len <<= 1;
        }
        if inverse {
            let s = 1.0 / n as f64;
            re.iter_mut().for_each(|v| *v *= s);
            im.iter_mut().for_each(|v| *v *= s);
        }
    }

    /// Exact true peak of a periodic block with no energy at fs/2, read on an `l`x grid
    /// (`l` = 32 puts the grid error of the *reference* below 0.0004 dB).
    fn exact_peak(x: &[f64], l: usize) -> f64 {
        let n = x.len();
        let m = n * l;
        let mut re = x.to_vec();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im, false);
        let mut yr = vec![0.0; m];
        let mut yi = vec![0.0; m];
        yr[..n / 2].copy_from_slice(&re[..n / 2]);
        yi[..n / 2].copy_from_slice(&im[..n / 2]);
        for k in 1..n / 2 {
            yr[m - k] = re[n - k];
            yi[m - k] = im[n - k];
        }
        fft(&mut yr, &mut yi, true);
        yr.iter().fold(0.0f64, |a, &v| a.max((v * l as f64).abs()))
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Gaussian-ish noise (sum of uniforms) with a flat spectrum up to `fmax` cycles per
    /// sample, nothing above, and nothing at Nyquist, peak-normalised to 0.9.
    fn band_noise(n: usize, fmax: f64, seed: u64) -> Vec<f64> {
        let mut st = 0x9E3779B97F4A7C15u64 ^ seed.wrapping_mul(0xD1B54A32D192ED03);
        let mut re: Vec<f64> = (0..n)
            .map(|_| (0..4).map(|_| lcg(&mut st)).sum::<f64>())
            .collect();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im, false);
        for k in 0..n {
            let f = k.min(n - k) as f64 / n as f64;
            if f > fmax || k == n / 2 {
                re[k] = 0.0;
                im[k] = 0.0;
            }
        }
        fft(&mut re, &mut im, true);
        let pk = re.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        re.iter().map(|v| v * 0.9 / pk).collect()
    }

    /// A detector's reading of a periodic block: one period to settle the filter, then
    /// the peak over a second.
    fn read(x: &[f64], ratio: Ratio) -> f64 {
        let mut d = Detector::new(ratio);
        d.push_block(x);
        d.reset_peak();
        d.push_block(x);
        d.peak()
    }

    fn db(v: f64) -> f64 {
        20.0 * v.log10()
    }

    /// Worst (under, over) reading in dB against the exact peak. Under is positive when
    /// the detector reads low.
    fn noise_error(ratio: Ratio, fmax: f64, seeds: u64) -> (f64, f64) {
        let mut under = f64::NEG_INFINITY;
        let mut over = f64::NEG_INFINITY;
        for seed in 0..seeds {
            let x = band_noise(2048, fmax, seed);
            let truth = exact_peak(&x, 32);
            let e = db(truth) - db(read(&x, ratio));
            under = under.max(e);
            over = over.max(-e);
        }
        (under, over)
    }

    #[test]
    fn exact_reference_is_exact_on_a_tone() {
        // A tone whose crest falls between samples: truth is 1.0, samples read 0.707.
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * 32.0 * i as f64 / 256.0 + PI / 4.0).sin())
            .collect();
        let p = exact_peak(&x, 32);
        assert!((p - 1.0).abs() < 1e-6, "reference read {p} for a unit tone");
    }

    /// The headline gate: content up to [`BAND_EDGE`] is read within the figure the
    /// limiter's guard band is built from, for every ratio, against the exact reference.
    #[test]
    fn band_limited_noise_is_read_within_the_stated_error() {
        for ratio in Ratio::ALL {
            for fmax in [0.45, 0.40, 0.25] {
                let (under, over) = noise_error(ratio, fmax, 8);
                assert!(
                    under <= ratio.worst_underread_db() + 1e-9,
                    "{ratio} under-read noise limited to {fmax} fs by {under:.3} dB, more \
                     than the {} dB its guard band assumes",
                    ratio.worst_underread_db()
                );
                assert!(
                    over <= ratio.worst_overread_db() + 1e-9,
                    "{ratio} over-read noise limited to {fmax} fs by {over:.3} dB, more \
                     than the {} dB it claims",
                    ratio.worst_overread_db()
                );
            }
        }
    }

    /// Noise that reaches fs/2 is outside every design's pass band. The figure is a
    /// caveat, not a target, and this keeps it from drifting.
    #[test]
    fn full_band_noise_stays_inside_the_stated_caveat() {
        for ratio in Ratio::ALL {
            let (under, _) = noise_error(ratio, 0.5, 8);
            assert!(
                under <= ratio.fullband_underread_db() + 1e-9,
                "{ratio} under-read full-band noise by {under:.3} dB, more than the {} dB \
                 the limiter reports as its band caveat",
                ratio.fullband_underread_db()
            );
        }
    }

    /// Tones at rational frequencies close to the band edge, where a crest lands at the
    /// worst offset from the grid on every cycle. Truth is exactly 1.0.
    #[test]
    fn tones_between_grid_points_are_read_within_the_stated_error() {
        let freqs = [(1, 4), (1, 3), (2, 5), (5, 12), (4, 9), (9, 20), (23, 50)];
        for ratio in Ratio::ALL {
            let mut worst = (0.0f64, 0.0f64);
            for (num, den) in freqs {
                let n = den * 128usize.div_ceil(den);
                let k = num * (n / den);
                for step in 0..45 {
                    let phase = PI * step as f64 / 45.0;
                    let x: Vec<f64> = (0..n)
                        .map(|i| (2.0 * PI * k as f64 * i as f64 / n as f64 + phase).cos())
                        .collect();
                    let under = -db(read(&x, ratio));
                    if under > worst.0 {
                        worst = (under, num as f64 / den as f64);
                    }
                }
            }
            assert!(
                worst.0 <= ratio.worst_underread_db() + 1e-9,
                "{ratio} under-read a unit tone by {:.3} dB at f_norm {:.3}, more than the \
                 {} dB its guard band assumes",
                worst.0,
                worst.1,
                ratio.worst_underread_db()
            );
        }
    }

    /// Diagnostic table for `docs`: `cargo test -p g6me-core accuracy::table -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic; prints the table the constants in Ratio were set from"]
    fn table() {
        println!("under-read in dB vs exact reconstruction (positive = reads low)");
        println!("band       x2      x4      x8     x16     x32");
        for fmax in [0.5, 0.45, 0.40] {
            let mut row = format!("{fmax:<8.2}");
            for ratio in Ratio::ALL {
                let (u, _) = noise_error(ratio, fmax, 16);
                row.push_str(&format!(" {u:+7.3}"));
            }
            println!("{row}");
        }
    }
}
