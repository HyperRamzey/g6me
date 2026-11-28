//! ISO 3382-1 room-acoustic metrics for measured or rendered impulse responses.
//!
//! Definitions implemented (ISO 3382-1:2009, "Acoustics — Measurement of room
//! acoustic parameters — Part 1: Performance spaces"):
//!
//! * **EDT** — early decay time, from the decay curve between 0 dB and −10 dB,
//!   `EDT = 6 · (t(-10) − t(0))` (implemented as a least-squares slope, ×60).
//! * **T20** — decay time from −5 dB to −25 dB, slope ×60.
//! * **T30** — decay time from −5 dB to −35 dB, slope ×60.
//! * **C80** — clarity, `10·log10(E(0..80 ms) / E(80 ms..end))`.
//! * **D50** — definition, `E(0..50 ms) / E(0..end)`.
//! * **LF** — early lateral energy fraction, ISO 3382-1:2009 Eq. A.14:
//!   `LF = ∫_5ms^80ms p_L² dt / ∫_0^80ms p² dt`, both integrals over the first
//!   80 ms, with the figure-of-eight pressure `p_L` and the omnidirectional
//!   pressure `p`. For native B-format (W, X, Y, Z) `p_L` is Y and `p` is W,
//!   which is the definition. A stereo pair carries no figure-of-eight channel,
//!   so the harness keeps that calculation only under the explicit name
//!   `lateral_fraction_stereo_approx`, where `p_L = (L − R)/√2` and
//!   `p = (L + R)/√2`; that value is an approximation with no defined relation
//!   to the ISO quantity and is never reported as LF.
//! * **IACC** — interaural cross-correlation coefficient, the maximum absolute
//!   normalized cross-correlation of L and R over ±1 ms, evaluated over the
//!   first 80 ms after the direct sound.
//!
//! The decay curve is the Schroeder backward integral of the squared impulse
//! response, accumulated with Neumaier compensated summation (a local port of
//! HyprEQ's `accum::RunningSum`) so a loud direct sound followed by a long
//! quiet tail does not absorb the tail. A linear least-squares fit of the decay
//! in dB is used over each stated range, because the standard specifies a
//! straight-line fit rather than a two-point difference; on an exactly
//! exponential decay the two agree.
//!
//! Mono/stereo handling: mono responses produce C80/D50/decay values and no LF
//! or IACC (a single pressure channel carries no interaural or lateral
//! information). Stereo responses produce both spatial values, with LF under
//! the explicitly approximate stereo definition above. Four-channel B-format
//! uses W as the omni signal, gives the ISO LF from W and Y, and reports no
//! IACC (there are no ear signals to correlate).
//!
//! Limitations stated rather than hidden: no noise-floor subtraction (Lundeby),
//! and the octave bands use the verification-grade power-complementary bank in
//! `filterbank`; it follows the IEC 61260-1 class-1 magnitude mask at the
//! tabulated breakpoints but is not a certified IEC 61260 instrument.

use crate::filterbank::{OctaveBank, OCTAVE_CENTERS_HZ};

/// An impulse response with de-interleaved channels.
#[derive(Clone, Debug)]
pub struct Rir {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f64>>,
}

impl Rir {
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn layout(&self) -> ChannelLayout {
        match self.channels.len() {
            0 => ChannelLayout::Empty,
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            4 => ChannelLayout::BFormat,
            n => ChannelLayout::Multi(n as u16),
        }
    }
}

/// How many channels an impulse response has, and therefore which metrics are
/// physically available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelLayout {
    Empty,
    Mono,
    Stereo,
    BFormat,
    Multi(u16),
}

impl ChannelLayout {
    pub fn label(self) -> String {
        match self {
            ChannelLayout::Empty => "empty".into(),
            ChannelLayout::Mono => "mono".into(),
            ChannelLayout::Stereo => "stereo".into(),
            ChannelLayout::BFormat => "b-format".into(),
            ChannelLayout::Multi(n) => format!("{n} ch"),
        }
    }

    /// LF and IACC need a left/right pair. B-format and arbitrary multichannel
    /// layouts do not define one.
    pub fn has_stereo_pair(self) -> bool {
        matches!(self, ChannelLayout::Stereo)
    }

    /// Native B-format (W, X, Y, Z) carries the omnidirectional pressure in W
    /// and the left-right figure-of-eight pressure in Y, which is what ISO
    /// 3382-1 Eq. A.14 needs.
    pub fn has_bformat(self) -> bool {
        matches!(self, ChannelLayout::BFormat)
    }
}

/// One ISO 3382-1 metric set. `None` means the metric is undefined for this
/// signal (too short, silent, mono input for a spatial metric, …), never zero
/// as a stand-in for "unknown".
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metrics {
    pub t20_s: Option<f64>,
    pub t30_s: Option<f64>,
    pub edt_s: Option<f64>,
    pub c80_db: Option<f64>,
    pub d50: Option<f64>,
    pub lf: Option<f64>,
    pub iacc: Option<Iacc>,
}

/// IACC together with the lag at which the maximum occurs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iacc {
    pub value: f64,
    pub lag_samples: isize,
    pub lag_ms: f64,
}

/// Metrics for one octave band.
#[derive(Clone, Debug)]
pub struct BandRow {
    /// Nominal ISO 266 octave centre frequency.
    pub band_hz: f64,
    /// False when the band's upper edge lies at or above Nyquist (or the band is
    /// too narrow to contain an FFT bin at this response length).
    pub supported: bool,
    pub metrics: Metrics,
}

/// The full analysis of one impulse response.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub sample_rate: u32,
    pub layout: ChannelLayout,
    pub duration_s: f64,
    /// Sample index that was treated as the direct-sound arrival, before any
    /// alignment shift. Reported so a shifted analysis is reproducible.
    pub onset_sample: usize,
    pub broadband: Metrics,
    pub bands: Vec<BandRow>,
}

/// Neumaier compensated running sum, a local port of HyprEQ's
/// `accum::RunningSum`: each term is added once to the running total and once
/// to a correction that captures the rounding of that addition, so small terms
/// are not absorbed by a large partial sum. Non-finite input propagates.
#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    sum: f64,
    comp: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        if !value.is_finite() || !self.sum.is_finite() {
            self.sum += value;
            self.comp = 0.0;
            return;
        }
        let next = self.sum + value;
        if !next.is_finite() {
            self.sum = next;
            self.comp = 0.0;
            return;
        }
        self.comp += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.comp
    }
}

/// `Σ x[i]²` with compensated summation.
fn sum_squares(x: &[f64]) -> f64 {
    let mut acc = CompensatedSum::default();
    for v in x {
        acc.add(v * v);
    }
    acc.total()
}

/// Schroeder backward integral of `x²`: `edc[i] = Σ_{k=i}^{N-1} x[k]²`,
/// accumulated with compensated summation.
pub fn schroeder_edc(x: &[f64]) -> Vec<f64> {
    let mut edc = vec![0.0; x.len()];
    let mut acc = CompensatedSum::default();
    for (out, sample) in edc.iter_mut().rev().zip(x.iter().rev()) {
        acc.add(sample * sample);
        *out = acc.total();
    }
    edc
}

/// Decay time from the least-squares slope of `10·log10(edc/edc[0])` between
/// `hi_db` and `lo_db` (both negative or zero), scaled to 60 dB.
///
/// Returns `None` when the decay curve does not reach `lo_db`, when fewer than
/// four samples lie in the range, or when the fitted slope is not negative.
pub fn decay_time_db(edc: &[f64], sample_rate: f64, hi_db: f64, lo_db: f64) -> Option<f64> {
    if edc.len() < 8 || !sample_rate.is_finite() || sample_rate <= 0.0 || hi_db <= lo_db {
        return None;
    }
    let e0 = edc[0];
    if !e0.is_finite() || e0 <= 0.0 {
        return None;
    }
    let db_at = |i: usize| 10.0 * (edc[i] / e0).log10();

    let i_hi = (0..edc.len()).find(|&i| {
        let db = db_at(i);
        db.is_finite() && db <= hi_db
    })?;
    let i_lo = (i_hi..edc.len()).find(|&i| {
        let db = db_at(i);
        db.is_finite() && db <= lo_db
    })?;
    // Two points always fit a line; require a real interval so a degenerate
    // curve cannot masquerade as a measured decay.
    if i_lo < i_hi + 4 {
        return None;
    }

    let n = (i_lo - i_hi + 1) as f64;
    let mean_x = (i_hi + i_lo) as f64 / 2.0;
    let mut mean_y = 0.0;
    for i in i_hi..=i_lo {
        mean_y += db_at(i);
    }
    mean_y /= n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in i_hi..=i_lo {
        let dx = i as f64 - mean_x;
        num += dx * (db_at(i) - mean_y);
        den += dx * dx;
    }
    if !den.is_finite() || den <= 0.0 {
        return None;
    }
    let slope = num / den; // dB per sample
    if !slope.is_finite() || slope >= 0.0 {
        return None;
    }
    Some(-60.0 / (slope * sample_rate))
}

/// Energy in `[0, split)` and `[split, end)` of a squared signal, with
/// compensated summation in both ranges.
fn split_energy(x: &[f64], split: usize) -> (f64, f64) {
    let split = split.min(x.len());
    (sum_squares(&x[..split]), sum_squares(&x[split..]))
}

fn split_index(sample_rate: f64, t_s: f64) -> usize {
    if !sample_rate.is_finite() || sample_rate <= 0.0 || !t_s.is_finite() || t_s <= 0.0 {
        return 0;
    }
    (t_s * sample_rate).round() as usize
}

/// Clarity in dB for an integration boundary of `t_s` (C80 for 80 ms, C50 for
/// 50 ms): `10·log10(E_early / E_late)`.
///
/// `origin_samples` is the sample treated as t = 0 — 0 for the broadband
/// response, the band filter's energy-centroid delay for an octave band. The
/// split sits at `origin + t_s`; everything before it, including the leading
/// edge of the direct sound, is early. For a pure delay `D`, `x` delayed by
/// `D` with `origin_samples = D` gives exactly the undelayed result.
pub fn clarity_db(x: &[f64], sample_rate: f64, t_s: f64, origin_samples: usize) -> Option<f64> {
    let (early, late) = split_energy(x, origin_samples + split_index(sample_rate, t_s));
    if !early.is_finite() || early <= 0.0 || !late.is_finite() || late <= 0.0 {
        return None;
    }
    let c = 10.0 * (early / late).log10();
    c.is_finite().then_some(c)
}

/// Definition for an integration boundary of `t_s` (D50 for 50 ms):
/// `E_early / (E_early + E_late)`. `origin_samples` has the same meaning as in
/// [`clarity_db`].
pub fn definition(x: &[f64], sample_rate: f64, t_s: f64, origin_samples: usize) -> Option<f64> {
    let (early, late) = split_energy(x, origin_samples + split_index(sample_rate, t_s));
    let total = early + late;
    if !total.is_finite() || total <= 0.0 || !early.is_finite() || early < 0.0 {
        return None;
    }
    Some((early / total).clamp(0.0, 1.0))
}

/// **Approximate** lateral energy fraction from a stereo pair:
/// `Σ_{5ms..80ms} ((L−R)/√2)² / Σ_{0..80ms} ((L+R)/√2)²`.
///
/// This is not ISO 3382-1 Eq. A.14. A stereo pair carries no figure-of-eight
/// pressure channel, and `(L−R)/√2` has no defined relation to one; the window
/// is the ISO one, but the spatial interpretation is not. It is kept for
/// datasets that ship L/R only, under a name that says so. The constant 1/2
/// cancels in the ratio and is omitted from the arithmetic.
/// `origin_samples` is the sample treated as t = 0, as in [`clarity_db`].
pub fn lateral_fraction_stereo_approx(
    left: &[f64],
    right: &[f64],
    sample_rate: f64,
    origin_samples: usize,
) -> Option<f64> {
    if left.len() != right.len()
        || left.len() < 16
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
    {
        return None;
    }
    let start = (origin_samples + split_index(sample_rate, 0.005)).min(left.len());
    let end = (origin_samples + split_index(sample_rate, 0.080)).min(left.len());
    if end <= start {
        return None;
    }
    let mut lateral = CompensatedSum::default();
    for i in start..end {
        let d = left[i] - right[i];
        lateral.add(d * d);
    }
    let mut omni = CompensatedSum::default();
    for i in 0..end {
        let s = left[i] + right[i];
        omni.add(s * s);
    }
    let (lateral, omni) = (lateral.total(), omni.total());
    if !omni.is_finite() || omni <= 0.0 || !lateral.is_finite() {
        return None;
    }
    Some((lateral / omni).clamp(0.0, 1.0))
}

/// ISO 3382-1:2009 Eq. A.14 lateral energy fraction from native B-format:
/// `LF = Σ_{5ms..80ms} Y² / Σ_{0ms..80ms} W²`.
///
/// W is the omnidirectional pressure and Y the left-right figure-of-eight
/// pressure (FuMa channel order W, X, Y, Z). Any common scale factor on W and Y
/// cancels in the ratio. `origin_samples` is the sample treated as t = 0, as in
/// [`clarity_db`].
pub fn lateral_fraction_bformat(
    w: &[f64],
    y: &[f64],
    sample_rate: f64,
    origin_samples: usize,
) -> Option<f64> {
    if w.len() != y.len() || w.len() < 16 || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return None;
    }
    let start = (origin_samples + split_index(sample_rate, 0.005)).min(w.len());
    let end = (origin_samples + split_index(sample_rate, 0.080)).min(w.len());
    if end <= start {
        return None;
    }
    let lateral = sum_squares(&y[start..end]);
    let omni = sum_squares(&w[..end]);
    if !omni.is_finite() || omni <= 0.0 || !lateral.is_finite() {
        return None;
    }
    Some((lateral / omni).clamp(0.0, 1.0))
}

/// IACC: maximum absolute normalized cross-correlation of L and R over ±1 ms,
/// evaluated on the first 80 ms. Zero-padded correlation (one channel is
/// outside its window treated as zero) with both channels' window energies as
/// the normalizer. `origin_samples` is the sample treated as t = 0, as in
/// [`clarity_db`].
pub fn iacc(left: &[f64], right: &[f64], sample_rate: f64, origin_samples: usize) -> Option<Iacc> {
    if left.len() != right.len()
        || left.len() < 16
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
    {
        return None;
    }
    let window = (origin_samples + split_index(sample_rate, 0.080)).min(left.len());
    if window < 16 {
        return None;
    }
    let max_lag = split_index(sample_rate, 0.001).min(window - 1) as isize;
    if max_lag < 1 {
        return None;
    }
    let el = sum_squares(&left[..window]);
    let er = sum_squares(&right[..window]);
    if !el.is_finite() || el <= 0.0 || !er.is_finite() || er <= 0.0 {
        return None;
    }
    let denom = (el * er).sqrt();
    let mut best = 0.0f64;
    let mut best_lag = 0isize;
    for lag in -max_lag..=max_lag {
        let t0 = if lag < 0 { (-lag) as usize } else { 0 };
        let t1 = if lag > 0 {
            window - lag as usize
        } else {
            window
        };
        if t0 >= t1 {
            continue;
        }
        let mut acc = CompensatedSum::default();
        for t in t0..t1 {
            acc.add(left[t] * right[(t as isize + lag) as usize]);
        }
        let c = (acc.total() / denom).abs();
        if c > best {
            best = c;
            best_lag = lag;
        }
    }
    Some(Iacc {
        value: best.clamp(0.0, 1.0),
        lag_samples: best_lag,
        lag_ms: best_lag as f64 / sample_rate * 1000.0,
    })
}

/// The sample at which the direct sound arrives: the first sample reaching 10 %
/// of the peak absolute amplitude. Returns 0 for silence.
pub fn onset_sample(x: &[f64]) -> usize {
    let peak = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if !peak.is_finite() || peak <= 0.0 {
        return 0;
    }
    let threshold = peak * 0.1;
    x.iter().position(|v| v.abs() >= threshold).unwrap_or(0)
}

/// Omnidirectional pressure for a channel layout: the channel itself for mono,
/// `(L+R)/√2` for stereo, the W channel for B-format (FuMa order W, X, Y, Z),
/// and the energetic RMS sum for arbitrary multichannel inputs (documented, not
/// an ISO measurement).
fn omni_signal(channels: &[Vec<f64>]) -> Vec<f64> {
    match channels {
        [] => Vec::new(),
        [only] => only.clone(),
        [l, r] => l
            .iter()
            .zip(r.iter())
            .map(|(a, b)| (a + b) * std::f64::consts::FRAC_1_SQRT_2)
            .collect(),
        [w, _, _, _] => w.clone(),
        _ => {
            let n = channels[0].len();
            let scale = 1.0 / (channels.len() as f64).sqrt();
            let mut out = vec![0.0; n];
            for ch in channels {
                for (o, v) in out.iter_mut().zip(ch.iter()) {
                    *o += v * scale;
                }
            }
            out
        }
    }
}

/// Which physical signals back the spatial metrics.
#[derive(Clone, Copy)]
enum Spatial<'a> {
    /// Stereo pair; LF is the explicitly approximate difference-channel value.
    Stereo { left: &'a [f64], right: &'a [f64] },
    /// Native B-format: W is omni, Y is the figure-of-eight pressure.
    BFormat { w: &'a [f64], y: &'a [f64] },
}

fn metrics_of(
    omni: &[f64],
    spatial: Option<Spatial<'_>>,
    sample_rate: f64,
    origin_samples: usize,
) -> Metrics {
    let edc = schroeder_edc(omni);
    let (lf, iacc) = match spatial {
        Some(Spatial::Stereo { left, right }) => (
            lateral_fraction_stereo_approx(left, right, sample_rate, origin_samples),
            iacc(left, right, sample_rate, origin_samples),
        ),
        // No ear signals exist in B-format; IACC stays undefined.
        Some(Spatial::BFormat { w, y }) => (
            lateral_fraction_bformat(w, y, sample_rate, origin_samples),
            None,
        ),
        None => (None, None),
    };
    Metrics {
        // The decay fits are self-referenced to edc[0], so the filter delay
        // does not move them.
        t20_s: decay_time_db(&edc, sample_rate, -5.0, -25.0),
        t30_s: decay_time_db(&edc, sample_rate, -5.0, -35.0),
        edt_s: decay_time_db(&edc, sample_rate, 0.0, -10.0),
        c80_db: clarity_db(omni, sample_rate, 0.080, origin_samples),
        d50: definition(omni, sample_rate, 0.050, origin_samples),
        lf,
        iacc,
    }
}

/// Analyse every octave band from 31.5 Hz to 16 kHz that the sample rate can
/// represent, plus the broadband response.
///
/// `align` shifts the response so the detected direct-sound onset becomes
/// sample 0 (the ISO time origin). Without it, index 0 is used as t = 0, which
/// is only correct for pre-aligned responses.
///
/// Spatial channels follow the layout: stereo filters both channels for the
/// approximate LF and IACC; B-format filters W and Y and reports the ISO
/// Eq. A.14 LF.
pub fn analyze(rir: &Rir, align: bool) -> Result<Analysis, String> {
    let frames = rir.frames();
    if rir.channels.is_empty() || rir.sample_rate == 0 {
        return Err("impulse response has no channels or a zero sample rate".into());
    }
    if frames < 64 {
        return Err(format!(
            "impulse response has {frames} frames; at least 64 are needed for a decay fit"
        ));
    }
    if rir.channels.iter().any(|c| c.len() != frames) {
        return Err("channels have differing lengths".into());
    }

    let layout = rir.layout();
    let mut channels = rir.channels.clone();
    let onset = onset_sample(&omni_signal(&channels));
    if align && onset > 0 {
        for ch in &mut channels {
            ch.drain(..onset);
        }
    }
    let frames = channels[0].len();
    let fs = rir.sample_rate as f64;
    let omni = omni_signal(&channels);
    let spatial = if layout.has_bformat() {
        Some(Spatial::BFormat {
            w: channels[0].as_slice(),
            y: channels[2].as_slice(),
        })
    } else if layout.has_stereo_pair() {
        Some(Spatial::Stereo {
            left: channels[0].as_slice(),
            right: channels[1].as_slice(),
        })
    } else {
        None
    };

    let broadband = metrics_of(&omni, spatial, fs, 0);

    let bank = OctaveBank::new(frames, fs);
    let spec_omni = bank.forward(&omni);
    let spec_lr = match spatial {
        Some(Spatial::Stereo { left, right }) => Some((bank.forward(left), bank.forward(right))),
        _ => None,
    };
    let spec_y = match spatial {
        Some(Spatial::BFormat { y, .. }) => Some(bank.forward(y)),
        _ => None,
    };

    let mut bands = Vec::with_capacity(OCTAVE_CENTERS_HZ.len());
    for (band, center) in OCTAVE_CENTERS_HZ.iter().enumerate() {
        if !bank.supported(band) {
            bands.push(BandRow {
                band_hz: *center,
                supported: false,
                metrics: Metrics::default(),
            });
            continue;
        }
        let kernel = bank
            .min_phase_spectrum(band)
            .ok_or_else(|| format!("band {center} Hz: filter design failed"))?;
        // The causal filter delays the band's direct sound; anchor this band's
        // ISO windows to the kernel's energy centroid.
        let origin = bank.kernel_delay_samples(&kernel);
        let mut band_omni = bank.apply(&spec_omni, &kernel);
        band_omni.truncate(frames);
        let band_metrics = if let Some((sl, sr)) = &spec_lr {
            let mut bl = bank.apply(sl, &kernel);
            bl.truncate(frames);
            let mut br = bank.apply(sr, &kernel);
            br.truncate(frames);
            metrics_of(
                &band_omni,
                Some(Spatial::Stereo {
                    left: &bl,
                    right: &br,
                }),
                fs,
                origin,
            )
        } else if let Some(sy) = &spec_y {
            let mut by = bank.apply(sy, &kernel);
            by.truncate(frames);
            metrics_of(
                &band_omni,
                Some(Spatial::BFormat {
                    w: &band_omni,
                    y: &by,
                }),
                fs,
                origin,
            )
        } else {
            metrics_of(&band_omni, None, fs, origin)
        };
        bands.push(BandRow {
            band_hz: *center,
            supported: true,
            metrics: band_metrics,
        });
    }

    Ok(Analysis {
        sample_rate: rir.sample_rate,
        layout,
        duration_s: frames as f64 / fs,
        onset_sample: onset,
        broadband,
        bands,
    })
}

/// Human label for a nominal octave centre.
pub fn band_label(hz: f64) -> String {
    if hz >= 1000.0 {
        let k = hz / 1000.0;
        if (k - k.round()).abs() < 1e-9 {
            format!("{} kHz", k.round() as i64)
        } else {
            format!("{k} kHz")
        }
    } else if (hz - hz.round()).abs() < 1e-9 {
        format!("{} Hz", hz.round() as i64)
    } else {
        format!("{hz} Hz")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A response whose energy decays geometrically: `x[n] = a^n`.
    ///
    /// The Schroeder curve is then exactly linear in dB to within the truncated
    /// geometric tail, so T20/T30/EDT have the closed-form answer
    /// `T = −3 / (log10(a) · fs)`, and every energy split is a geometric sum.
    fn geometric_rir(t60_s: f64, sample_rate: f64, seconds: f64) -> (Vec<f64>, f64) {
        let n = (seconds * sample_rate).round() as usize;
        let a = 10f64.powf(-3.0 / (t60_s * sample_rate));
        let mut v = Vec::with_capacity(n);
        let mut cur = 1.0;
        for _ in 0..n {
            v.push(cur);
            cur *= a;
        }
        (v, a)
    }

    fn geometric_sum(q: f64, m: usize) -> f64 {
        // Σ_{k=0}^{m-1} q^k
        if (1.0 - q).abs() < f64::EPSILON {
            m as f64
        } else {
            (1.0 - q.powi(m as i32)) / (1.0 - q)
        }
    }

    #[test]
    fn geometric_decay_recovers_known_t20_t30_edt() {
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.5, fs, 1.8);
        let edc = schroeder_edc(&x);
        for (name, hi, lo) in [
            ("EDT", 0.0, -10.0),
            ("T20", -5.0, -25.0),
            ("T30", -5.0, -35.0),
        ] {
            let t = decay_time_db(&edc, fs, hi, lo)
                .unwrap_or_else(|| panic!("{name} did not resolve on an exact exponential"));
            let rel = (t - 0.5).abs() / 0.5;
            assert!(
                rel < 1e-6,
                "{name} = {t:.9} s, expected 0.5 s (relative error {rel:.3e})"
            );
        }
    }

    #[test]
    fn decay_metrics_are_none_on_degenerate_responses() {
        assert!(decay_time_db(&[], 48_000.0, -5.0, -25.0).is_none());
        let silence = vec![0.0; 4096];
        assert!(decay_time_db(&schroeder_edc(&silence), 48_000.0, -5.0, -25.0).is_none());
        // A single impulse drops to −∞ immediately: no interval to fit.
        let mut impulse = vec![0.0; 4096];
        impulse[0] = 1.0;
        assert!(decay_time_db(&schroeder_edc(&impulse), 48_000.0, -5.0, -25.0).is_none());
        // A response whose total decay stays above −25 dB cannot produce a T20.
        let short: Vec<f64> = (0..10).map(|i| 1.0 - 0.05 * i as f64).collect();
        let short_db = {
            let edc = schroeder_edc(&short);
            10.0 * (edc[edc.len() - 1] / edc[0]).log10()
        };
        assert!(short_db > -25.0, "fixture only decays to {short_db:.3} dB");
        assert!(decay_time_db(&schroeder_edc(&short), 48_000.0, -5.0, -25.0).is_none());
    }

    #[test]
    fn clarity_and_definition_match_the_closed_form() {
        let fs = 48_000.0;
        let (x, a) = geometric_rir(0.4, fs, 1.2);
        let q = a * a;
        let n = x.len();
        let etot = geometric_sum(q, n);
        let e80 = geometric_sum(q, (0.080 * fs).round() as usize);
        let e50 = geometric_sum(q, (0.050 * fs).round() as usize);
        let want_c80 = 10.0 * (e80 / (etot - e80)).log10();
        let want_d50 = e50 / etot;

        let got_c80 = clarity_db(&x, fs, 0.080, 0).expect("C80");
        let got_d50 = definition(&x, fs, 0.050, 0).expect("D50");
        assert!(
            (got_c80 - want_c80).abs() < 1e-9,
            "C80 {got_c80:.9} dB vs closed form {want_c80:.9} dB"
        );
        assert!(
            (got_d50 - want_d50).abs() < 1e-12,
            "D50 {got_d50:.12} vs closed form {want_d50:.12}"
        );
        assert!((0.0..=1.0).contains(&got_d50));
    }

    #[test]
    fn stereo_lateral_fraction_matches_the_geometric_closed_form() {
        let fs = 48_000.0;
        let (x, a) = geometric_rir(0.5, fs, 1.8);
        let q = a * a;
        let n = x.len();
        let m5 = (0.005 * fs).round() as usize;
        let m80 = (0.080 * fs).round() as usize;
        // ISO 3382-1 Eq. A.14 puts both integrals in the first 80 ms.
        let windowed = geometric_sum(q, m80);
        let want = (windowed - geometric_sum(q, m5)) / windowed;

        // Fully lateral difference channel: R = 0.
        let zeros = vec![0.0; n];
        let got = lateral_fraction_stereo_approx(&x, &zeros, fs, 0).expect("LF with R=0");
        assert!(
            (got - want).abs() < 1e-12,
            "LF {got:.12} vs closed form {want:.12}"
        );
        // The pre-fix whole-record denominator reports a smaller number; this
        // guards against silently reverting to it.
        let whole_record = (windowed - geometric_sum(q, m5)) / geometric_sum(q, n);
        assert!(
            got > whole_record + 1e-3,
            "LF {got:.6} is indistinguishable from the whole-record denominator {whole_record:.6}"
        );

        // Centred image: L == R means no lateral energy at all.
        let centred = lateral_fraction_stereo_approx(&x, &x, fs, 0).expect("LF with L=R");
        assert_eq!(centred, 0.0, "L == R must give exactly zero lateral energy");

        let silent = lateral_fraction_stereo_approx(&zeros, &zeros, fs, 0);
        assert!(silent.is_none(), "silence has no defined LF");
    }

    #[test]
    fn bformat_lateral_fraction_is_iso_eq_a14() {
        let fs = 48_000.0;
        let (w, a) = geometric_rir(0.5, fs, 1.8);
        let q = a * a;
        let n = w.len();
        let m5 = (0.005 * fs).round() as usize;
        let m80 = (0.080 * fs).round() as usize;
        // Y = 0.5·W: a purely lateral field at half the omni amplitude, so the
        // Eq. A.14 value is 0.25 × the W-only window ratio.
        let windowed = geometric_sum(q, m80);
        let want = 0.25 * (windowed - geometric_sum(q, m5)) / windowed;
        let zeros = vec![0.0; n];
        let y: Vec<f64> = w.iter().map(|v| 0.5 * v).collect();
        let rir = Rir {
            sample_rate: 48_000,
            channels: vec![w.clone(), zeros.clone(), y, zeros.clone()],
        };
        let an = analyze(&rir, true).expect("B-format analysis");
        assert_eq!(an.layout, ChannelLayout::BFormat);
        let lf = an.broadband.lf.expect("B-format must report the ISO LF");
        assert!((lf - want).abs() < 1e-12, "LF {lf:.12} vs {want:.12}");
        assert!(
            an.broadband.iacc.is_none(),
            "B-format has no ear signals, so IACC must stay undefined"
        );

        // W is the omni signal: the non-spatial metrics must equal a mono
        // analysis of W exactly.
        let mono = analyze(
            &Rir {
                sample_rate: 48_000,
                channels: vec![w],
            },
            true,
        )
        .expect("mono analysis");
        assert_eq!(an.broadband.edt_s, mono.broadband.edt_s);
        assert_eq!(an.broadband.t30_s, mono.broadband.t30_s);
        assert_eq!(an.broadband.c80_db, mono.broadband.c80_db);
        assert_eq!(an.broadband.d50, mono.broadband.d50);

        // Without a Y signal there is no lateral energy at all.
        let lateral_zero = Rir {
            sample_rate: 48_000,
            channels: vec![vec![1.0; n], zeros.clone(), zeros.clone(), zeros.clone()],
        };
        let flat = analyze(&lateral_zero, false).expect("B-format with silent Y");
        assert_eq!(flat.broadband.lf, Some(0.0));
    }

    #[test]
    fn compensated_sums_keep_bits_a_naive_walk_drops() {
        // The Schroeder integral walks backward, so the absorbing large term
        // is last in the record: after 1e16 is in the accumulator each 1.0 is
        // exactly half an ulp and a naive sum loses all of them to
        // ties-to-even. The exact total, 1e16 + 2000, is representable.
        let mut x = vec![1.0; 2001];
        x[2000] = 1e8;
        let edc = schroeder_edc(&x);
        assert_eq!(edc[0], 1e16 + 2000.0, "compensated EDC must keep the tail");
        let naive: f64 = x.iter().rev().map(|v| v * v).sum();
        assert_eq!(naive, 1e16, "fixture must actually lose the tail naively");

        // split_energy walks forward, so the same fixture puts the large term
        // first. Bound: the compensated result is exact (equality); the naive
        // result is off by the full 1000.
        let mut y = vec![1.0; 2001];
        y[0] = 1e8;
        let (early, late) = split_energy(&y, 1001);
        assert_eq!(early, 1e16 + 1000.0);
        assert_eq!(late, 1000.0);
        let naive_early: f64 = y[..1001].iter().map(|v| v * v).sum();
        assert_eq!(
            naive_early, 1e16,
            "fixture must actually lose the tail naively"
        );
    }

    #[test]
    fn origin_shift_compensates_a_pure_delay_exactly() {
        // Prepending D zeros and passing origin = D must reproduce the
        // undelayed metrics: the ISO windows are anchored to the direct sound,
        // not to the array start. This is the mechanism `analyze` uses for the
        // causal band filters, whose kernels carry the direct sound's energy
        // centroid a few milliseconds late.
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.4, fs, 0.6);
        let d = 1000; // 20.8 ms at 48 kHz
        let mut delayed = vec![0.0; d];
        delayed.extend_from_slice(&x);

        let c0 = clarity_db(&x, fs, 0.080, 0).expect("C80");
        let c1 = clarity_db(&delayed, fs, 0.080, d).expect("C80 delayed");
        assert!((c0 - c1).abs() < 1e-12, "C80 {c0:.12} vs {c1:.12}");
        let d0 = definition(&x, fs, 0.050, 0).expect("D50");
        let d1 = definition(&delayed, fs, 0.050, d).expect("D50 delayed");
        assert!((d0 - d1).abs() < 1e-12, "D50 {d0:.12} vs {d1:.12}");

        let zeros = vec![0.0; x.len()];
        let mut delayed_zeros = vec![0.0; d];
        delayed_zeros.extend_from_slice(&zeros);
        let lf0 = lateral_fraction_stereo_approx(&x, &zeros, fs, 0).expect("LF");
        let lf1 =
            lateral_fraction_stereo_approx(&delayed, &delayed_zeros, fs, d).expect("LF delayed");
        assert!((lf0 - lf1).abs() < 1e-12, "LF {lf0:.12} vs {lf1:.12}");

        let i0 = iacc(&x, &x, fs, 0).expect("IACC");
        let i1 = iacc(&delayed, &delayed, fs, d).expect("IACC delayed");
        assert!((i0.value - i1.value).abs() < 1e-12);
        assert_eq!(i0.lag_samples, i1.lag_samples);
    }

    #[test]
    fn iacc_is_one_for_identical_and_delayed_impulses() {
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.5, fs, 0.2);
        let same = iacc(&x, &x, fs, 0).expect("IACC identical");
        assert!((same.value - 1.0).abs() < 1e-12, "IACC {same:?}");
        assert_eq!(same.lag_samples, 0);

        let negated: Vec<f64> = x.iter().map(|v| -v).collect();
        let inverted = iacc(&x, &negated, fs, 0).expect("IACC inverted");
        assert!(
            (inverted.value - 1.0).abs() < 1e-12,
            "|ρ| is used: {inverted:?}"
        );
        assert_eq!(inverted.lag_samples, 0);

        // A 1 ms interaural delay is the maximum of the correlation.
        let mut left = vec![0.0; 256];
        let mut right = vec![0.0; 256];
        left[0] = 1.0;
        right[48] = 1.0; // 1 ms at 48 kHz
        let delayed = iacc(&left, &right, fs, 0).expect("IACC delayed");
        assert_eq!(delayed.lag_samples, 48);
        assert!((delayed.value - 1.0).abs() < 1e-12, "{delayed:?}");
        assert!((delayed.lag_ms - 1.0).abs() < 1e-12);

        let swapped = iacc(&right, &left, fs, 0).expect("IACC swapped");
        assert_eq!(swapped.lag_samples, -48);
    }

    #[test]
    fn iacc_of_independent_noise_is_below_one() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let left: Vec<f64> = (0..4800).map(|_| next()).collect();
        let right: Vec<f64> = (0..4800).map(|_| next()).collect();
        let c = iacc(&left, &right, 48_000.0, 0).expect("IACC noise");
        assert!(
            c.value < 0.95,
            "independent noise produced IACC {}",
            c.value
        );
        assert!((0.0..=1.0).contains(&c.value));
    }

    #[test]
    fn onset_detection_finds_the_direct_sound() {
        let mut x = vec![0.0; 4096];
        x[100] = 0.5;
        x[101] = -1.0;
        assert_eq!(onset_sample(&x), 100);
        assert_eq!(onset_sample(&[0.0; 8]), 0);
    }

    #[test]
    fn mono_has_no_spatial_metrics_and_stereo_does() {
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.4, fs, 1.0);
        let mono = Rir {
            sample_rate: 48_000,
            channels: vec![x.clone()],
        };
        let a = analyze(&mono, true).expect("mono analysis");
        assert_eq!(a.layout, ChannelLayout::Mono);
        assert!(a.broadband.lf.is_none());
        assert!(a.broadband.iacc.is_none());
        assert!(a.broadband.t30_s.is_some());
        assert!(a.broadband.c80_db.is_some());

        let stereo = Rir {
            sample_rate: 48_000,
            channels: vec![x.clone(), x],
        };
        let a = analyze(&stereo, true).expect("stereo analysis");
        assert_eq!(a.layout, ChannelLayout::Stereo);
        assert_eq!(a.broadband.lf, Some(0.0));
        assert!(a.broadband.iacc.is_some());
    }

    #[test]
    fn octave_bands_are_reported_from_31_5_hz_to_16_khz() {
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.4, fs, 0.5);
        let an = analyze(
            &Rir {
                sample_rate: 48_000,
                channels: vec![x],
            },
            true,
        )
        .unwrap();
        assert_eq!(an.bands.len(), 10);
        assert_eq!(an.bands[0].band_hz, 31.5);
        assert_eq!(an.bands[9].band_hz, 16_000.0);
        // 16 kHz is representable at 48 kHz (upper edge 22.6 kHz < 24 kHz).
        assert!(an.bands[9].supported);
    }

    /// Re-measurement fixture for the filter-bank change: prints the broadband
    /// and selected octave-band values so the before/after deltas are
    /// reproducible with `cargo test -- --nocapture`.
    #[test]
    fn synthetic_case_metrics_snapshot() {
        let fs = 48_000.0;
        let (x, _) = geometric_rir(0.5, fs, 1.8);
        let rir = Rir {
            sample_rate: 48_000,
            channels: vec![x.clone(), vec![0.0; x.len()]],
        };
        let a = analyze(&rir, true).expect("analysis");
        let bb = &a.broadband;
        println!(
            "snapshot broadband: edt={:?} t30={:?} c80={:?} d50={:?} lf={:?}",
            bb.edt_s, bb.t30_s, bb.c80_db, bb.d50, bb.lf
        );
        for row in &a.bands {
            if !matches!(row.band_hz, 125.0 | 500.0 | 1000.0 | 2000.0) {
                continue;
            }
            println!(
                "snapshot band {:>8}: edt={:?} t30={:?} c80={:?} d50={:?} lf={:?}",
                band_label(row.band_hz),
                row.metrics.edt_s,
                row.metrics.t30_s,
                row.metrics.c80_db,
                row.metrics.d50,
                row.metrics.lf
            );
        }
        assert!(bb.c80_db.is_some() && bb.d50.is_some() && bb.edt_s.is_some());
    }

    #[test]
    fn full_pipeline_recovers_the_envelope_decay_in_bands() {
        // Deterministic noise under a known exponential envelope. Unlike the
        // geometric DC decay above this exercises the octave filter bank end to
        // end: each band's T30 must still track the envelope's T60.
        let fs = 48_000.0;
        let t60 = 1.0;
        let n = (2.0 * fs) as usize;
        let a = 10f64.powf(-3.0 / (t60 * fs));
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut env = 1.0f64;
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            x.push(noise * env);
            env *= a;
        }
        let an = analyze(
            &Rir {
                sample_rate: 48_000,
                channels: vec![x],
            },
            true,
        )
        .expect("analysis");
        let bb = an.broadband.t30_s.expect("broadband T30");
        assert!(
            (bb - t60).abs() / t60 < 0.05,
            "broadband T30 {bb:.4} s should track the 1 s envelope"
        );
        for row in &an.bands {
            if !row.supported || !(125.0..=8000.0).contains(&row.band_hz) {
                continue;
            }
            let t = row.metrics.t30_s.unwrap_or_else(|| {
                panic!(
                    "band {} has no T30 after filtering",
                    band_label(row.band_hz)
                )
            });
            assert!(
                (t - t60).abs() / t60 < 0.10,
                "band {} T30 {t:.4} s should track 1 s",
                band_label(row.band_hz)
            );
        }
    }
}
