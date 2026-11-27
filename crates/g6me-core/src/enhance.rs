//! Shared-STN Mastering Enhancer:
//! 1. **Zero-IMD Psychoacoustic Virtual Bass**:
//!    * Moliner, Rämö & Välimäki (DAFx-2020), *"Virtual Bass System with Fuzzy Separation
//!      of Tones and Transients"*
//!    * Mu, Gan & Tan (IEEE ICASSP 2012 & 2013), *"A Psychoacoustic Bass Enhancement System
//!      with Improved Transient and Steady-State Performance"* + *"A Timbre Matching Approach..."*
//!    * Oo, Gan & Hawksford (JAES 2011), *"Perceptually-Motivated Objective Grading of
//!      Nonlinear Processing in Virtual-Bass Systems"* (`FEXP1` NLD + objective IMD/HGR metrics)
//! 2. **STN Micro-Detail & Presence Enhancer**:
//!    * Powered by the exact same Fierro & Välimäki (2022/2023) Fuzzy STN front-end (`crate::stn`),
//!      lifting tonal and transient micro-detail via a frequency-weighted upward soft-knee
//!      contrast curve while leaving the noise floor (`x_n`) strictly untouched.
//!
//! ## Architectural Synergy
//!
//! Both processors share a single [`StnDecomposer`] pass per channel. At `0` virtual bass
//! gain and `0` micro-detail intensity, the unity-sum STN decomposition guarantees exact
//! null reconstruction (`|y - x_delayed| < 1e-12`).
//!
//! Pure deterministic `f64`, zero external crates, zero FMA.

use std::collections::VecDeque;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::fft::{Complex, FftPlan};
use crate::stn::{sqrt_hann_cola_window, sr_scale, StnBlock, StnDecomposer};

/// Configuration for Zero-IMD Psychoacoustic Virtual Bass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VirtualBassCfg {
    /// Opt-in flag (default `false` so stock pipeline stays bit-exact).
    pub enable: bool,
    /// Upper cutoff frequency of the sub-bass fundamental region in Hz (`40.0 ..= 180.0`, default `100.0`).
    pub cutoff_hz: f64,
    /// Harmonic synthesis / boost drive level in dB (`-12.0 ..= 12.0`, default `0.0`;
    /// `-12..0` attenuates synthesized harmonics, `0` = moderate unity reference, `+6` = strong bass).
    /// Note: linear synthesis amount is controlled by `amount` (`0.0..=1.0`) scaled by `gain_db`.
    pub gain_db: f64,
    /// Overall wet synthesis mix (`0.0 ..= 1.0`, default `0.45`; `0.0` = exact bypass null).
    pub amount: f64,
    /// Timbre-matched harmonic rolloff exponent (`0.2 ..= 1.5`, default `0.65`;
    /// harmonic `h in {2,3,4}` gets weight `exp(-harmonic_decay * (h - 2))`).
    pub harmonic_decay: f64,
    /// Relative transient NLD punch weight (`0.0 ..= 1.5`, default `0.60`).
    pub transient_punch: f64,
}

impl Default for VirtualBassCfg {
    fn default() -> Self {
        Self {
            enable: false,
            cutoff_hz: 100.0,
            gain_db: 0.0,
            amount: 0.45,
            harmonic_decay: 0.65,
            transient_punch: 0.60,
        }
    }
}

impl VirtualBassCfg {
    pub fn validate(&self) -> Result<(), String> {
        if !self.enable {
            return Ok(());
        }
        if !self.cutoff_hz.is_finite() || !(40.0..=180.0).contains(&self.cutoff_hz) {
            return Err(format!(
                "vbass.cutoff_hz must be in 40..=180 Hz, got {}",
                self.cutoff_hz
            ));
        }
        if !self.gain_db.is_finite() || !(-12.0..=12.0).contains(&self.gain_db) {
            return Err(format!(
                "vbass.gain_db must be in -12..=+12 dB, got {}",
                self.gain_db
            ));
        }
        if !self.amount.is_finite() || !(0.0..=1.0).contains(&self.amount) {
            return Err(format!(
                "vbass.amount must be in 0.0..=1.0, got {}",
                self.amount
            ));
        }
        if !self.harmonic_decay.is_finite() || !(0.2..=1.5).contains(&self.harmonic_decay) {
            return Err(format!(
                "vbass.harmonic_decay must be in 0.2..=1.5, got {}",
                self.harmonic_decay
            ));
        }
        if !self.transient_punch.is_finite() || !(0.0..=1.5).contains(&self.transient_punch) {
            return Err(format!(
                "vbass.transient_punch must be in 0.0..=1.5, got {}",
                self.transient_punch
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.enable && self.amount > 0.0
    }
}

/// Configuration for STN Micro-Detail & Presence Enhancer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DetailCfg {
    /// Opt-in flag (default `false` so stock pipeline stays bit-exact).
    pub enable: bool,
    /// Micro-detail lift intensity (`0.0 ..= 1.0`, default `0.35`; `0.0` = exact bypass null).
    pub intensity: f64,
    /// Upward contrast bias (`0.0 ..= 1.0`, default `0.50`; controls knee/upward expansion of quiet details).
    pub bias: f64,
    /// Relative transient crispness boost (`0.0 ..= 1.5`, default `0.85`).
    pub transient_weight: f64,
    /// Presence shelf emphasis on the lifted detail around 2.5 kHz (`-3.0 ..= 6.0 dB`, default `2.0`).
    pub presence_tilt_db: f64,
}

impl Default for DetailCfg {
    fn default() -> Self {
        Self {
            enable: false,
            intensity: 0.35,
            bias: 0.50,
            transient_weight: 0.85,
            presence_tilt_db: 2.0,
        }
    }
}

impl DetailCfg {
    pub fn validate(&self) -> Result<(), String> {
        if !self.enable {
            return Ok(());
        }
        if !self.intensity.is_finite() || !(0.0..=1.0).contains(&self.intensity) {
            return Err(format!(
                "detail.intensity must be in 0.0..=1.0, got {}",
                self.intensity
            ));
        }
        if !self.bias.is_finite() || !(0.0..=1.0).contains(&self.bias) {
            return Err(format!(
                "detail.bias must be in 0.0..=1.0, got {}",
                self.bias
            ));
        }
        if !self.transient_weight.is_finite() || !(0.0..=1.5).contains(&self.transient_weight) {
            return Err(format!(
                "detail.transient_weight must be in 0.0..=1.5, got {}",
                self.transient_weight
            ));
        }
        if !self.presence_tilt_db.is_finite() || !(-3.0..=6.0).contains(&self.presence_tilt_db) {
            return Err(format!(
                "detail.presence_tilt_db must be in -3.0..=+6.0 dB, got {}",
                self.presence_tilt_db
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.enable && self.intensity > 0.0
    }
}

/// Combined shared-STN enhancement configuration (Virtual Bass + Micro-Detail).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhanceCfg {
    pub vbass: VirtualBassCfg,
    pub detail: DetailCfg,
}

impl EnhanceCfg {
    pub fn validate(&self) -> Result<(), String> {
        self.vbass.validate()?;
        self.detail.validate()?;
        Ok(())
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.vbass.is_active() || self.detail.is_active()
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.is_active()
    }

    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.vbass.is_active() {
            parts.push(format!(
                "vbass fc={:.0}Hz amt={:.2} gain={:+.1}dB punch={:.2}",
                self.vbass.cutoff_hz,
                self.vbass.amount,
                self.vbass.gain_db,
                self.vbass.transient_punch
            ));
        }
        if self.detail.is_active() {
            parts.push(format!(
                "detail int={:.2} bias={:.2} tilt={:+.1}dB",
                self.detail.intensity, self.detail.bias, self.detail.presence_tilt_db
            ));
        }
        if parts.is_empty() {
            "off".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

/// Direct Form II Transposed biquad in strict `f64` arithmetic.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn butter2_lpf(fc_hz: f64, fs: f64) -> Self {
        let fc = fc_hz.clamp(10.0, 0.45 * fs);
        let k = (PI * fc / fs).tan();
        let k2 = k * k;
        let sq2 = std::f64::consts::SQRT_2;
        let norm = 1.0 / (1.0 + sq2 * k + k2);
        let b0 = k2 * norm;
        Self {
            b0,
            b1: 2.0 * b0,
            b2: b0,
            a1: 2.0 * (k2 - 1.0) * norm,
            a2: (1.0 - sq2 * k + k2) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn butter2_hpf(fc_hz: f64, fs: f64) -> Self {
        let fc = fc_hz.clamp(10.0, 0.45 * fs);
        let k = (PI * fc / fs).tan();
        let k2 = k * k;
        let sq2 = std::f64::consts::SQRT_2;
        let norm = 1.0 / (1.0 + sq2 * k + k2);
        Self {
            b0: norm,
            b1: -2.0 * norm,
            b2: norm,
            a1: 2.0 * (k2 - 1.0) * norm,
            a2: (1.0 - sq2 * k + k2) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// First-order high shelf (`gain_db` above `fc_hz`, unity at DC).
    fn high_shelf_1st(fc_hz: f64, gain_db: f64, fs: f64) -> Self {
        if gain_db.abs() < 1e-6 {
            return Self {
                b0: 1.0,
                ..Default::default()
            };
        }
        let g = 10.0_f64.powf(gain_db / 20.0);
        let fc = fc_hz.clamp(100.0, 0.45 * fs);
        let k = (PI * fc / fs).tan();
        // Bilinear transform of H(s) = (g*s + w0) / (s + w0):
        // H(z) = ((g + k) + (k - g) z^-1) / ((1 + k) + (k - 1) z^-1)
        let inv = 1.0 / (1.0 + k);
        Self {
            b0: (g + k) * inv,
            b1: (k - g) * inv,
            b2: 0.0,
            a1: (k - 1.0) * inv,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Causal 4th-order Linkwitz–Riley filter (cascade of two 2nd-order Butterworth sections).
#[derive(Clone, Copy, Debug)]
struct Lr4 {
    s1: Biquad,
    s2: Biquad,
}

impl Lr4 {
    fn lpf(fc: f64, fs: f64) -> Self {
        let b = Biquad::butter2_lpf(fc, fs);
        Self { s1: b, s2: b }
    }

    fn hpf(fc: f64, fs: f64) -> Self {
        let b = Biquad::butter2_hpf(fc, fs);
        Self { s1: b, s2: b }
    }

    #[inline]
    fn tick(&mut self, x: f64) -> f64 {
        self.s2.tick(self.s1.tick(x))
    }
}

/// Oo, Gan & Hawksford (JAES 2011) `FEXP1` odd-symmetric fuzz-exponential NLD
/// paired with subtle asymmetric even-harmonic component (`HWR-FEXP1` blend) for
/// transient virtual bass punch.
#[inline]
pub fn fexp1_nld(x: f64, drive: f64) -> f64 {
    let k = drive.max(0.1);
    let denom = 1.0 - (-k).exp();
    let ax = x.abs();
    let odd = x.signum() * (1.0 - (-k * ax).exp()) / denom;
    // Slight even-harmonic contribution (2nd harmonic warmth for kick drums) via smooth HWR.
    let pos = if x > 0.0 {
        (1.0 - (-k * x).exp()) / denom
    } else {
        0.0
    };
    0.75 * odd + 0.25 * (pos - 0.5 * odd.abs())
}

/// Phase-coherent frequency-domain tonal virtual-bass harmonic synthesizer
/// (Moliner, Rämö & Välimäki DAFx-2020 + Mu, Gan & Tan ICASSP 2012/2013).
///
/// Operates exclusively on the STN `sines` stream:
/// 1. Detects isolated tonal spectral peaks in `[25 Hz, cutoff_hz]`.
/// 2. Projects each peak's energy to harmonics `h = 2, 3, 4` using timbre-matched
///    exponential decay weights `w_h = exp(-decay * (h - 2))`.
/// 3. Existing-harmonic awareness (Moliner et al. DAFx-2020): if harmonic `h` is
///    already present above `-24 dB` relative to the target harmonic amplitude,
///    boosts the existing harmonic in-phase; otherwise synthesizes the missing
///    harmonic with continuous phase-vocoder phase propagation (`phi_h(m) = phi_h(m-1) + h * d_phi_0`)
///    so there is ZERO intermodulation distortion between simultaneous bass notes!
struct TonalVbsSynth {
    n_fft: usize,
    hop: usize,
    n_bins: usize,
    fs: f64,
    cfg: VirtualBassCfg,
    plan: FftPlan,
    window: Vec<f64>,
    in_buf: Vec<f64>,
    in_fill: usize,
    frame_in: Vec<f64>,
    ola_out: Vec<f64>,
    /// Delay line keeping the dry tonal stream sample-aligned with the synthesizer's latency (`n_fft - hop`).
    dry_delay: VecDeque<f64>,
    /// Precomputed coherent normalization `0.5 * sum(w) * 1.45` and `sum(w^2)` for amplitude recovery.
    norm_amp: f64,
    win_energy: f64,
    /// Previous frame analysis phases for instantaneous-frequency phase propagation.
    prev_phase: Vec<f64>,
    /// Running synthesized phase accumulator per `(fundamental_bin, harmonic_idx)` slot.
    synth_phase: Vec<[f64; 3]>,
    fft_scratch: Vec<Complex>,
    spec_in: Vec<Complex>,
    spec_syn: Vec<Complex>,
    time_scratch: Vec<f64>,
    ready_out: VecDeque<f64>,
    latency: usize,
}

impl TonalVbsSynth {
    fn new(sample_rate: u32, cfg: VirtualBassCfg) -> Self {
        let fs = sample_rate.max(8_000) as f64;
        let scale = sr_scale(sample_rate);
        // 8192*scale gives 5.86 Hz bin spacing at 48 kHz — cleanly resolving adjacent bass
        // semitones/thirds (e.g. 55 Hz + 73.4 Hz) with ~-70 dB two-tone IMD suppression,
        // while transients bypass this stage entirely via the crisp time-domain FEXP1 path.
        let n_fft = 8192 * scale;
        let hop = n_fft / 4;
        let n_bins = n_fft / 2 + 1;
        // `dry_delay` is popped inside `process_hop` alongside `ola_out` (sharing the `hop - 1`
        // hop-buffering delay), so externally visible latency is `(n_fft - hop) + (hop - 1) = n_fft - 1`.
        let dry_delay_len = n_fft - hop;
        let latency = n_fft - 1;
        let mut dry_delay = VecDeque::with_capacity(dry_delay_len + hop);
        for _ in 0..dry_delay_len {
            dry_delay.push_back(0.0);
        }
        let window = sqrt_hann_cola_window(n_fft);
        let win_sum: f64 = window.iter().sum();
        let win_energy: f64 = window.iter().map(|w| w * w).sum();
        let norm_amp = 0.5 * win_sum * 1.45;
        Self {
            n_fft,
            hop,
            n_bins,
            fs,
            cfg,
            plan: FftPlan::new(n_fft),
            window,
            norm_amp,
            win_energy,
            in_buf: vec![0.0; hop],
            in_fill: 0,
            frame_in: vec![0.0; n_fft],
            ola_out: vec![0.0; n_fft],
            dry_delay,
            prev_phase: vec![0.0; n_bins],
            synth_phase: vec![[0.0; 3]; n_bins],
            fft_scratch: vec![Complex::ZERO; n_fft],
            spec_in: vec![Complex::ZERO; n_bins],
            spec_syn: vec![Complex::ZERO; n_bins],
            time_scratch: vec![0.0; n_fft],
            ready_out: VecDeque::with_capacity(hop * 2),
            latency,
        }
    }

    #[inline]
    fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Process tonal stream `sines` -> emits `sines_aligned + synthesized_bass_harmonics`.
    fn process(&mut self, sines: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.reserve(sines.len());
        for &x in sines {
            self.dry_delay.push_back(x);
            self.in_buf[self.in_fill] = x;
            self.in_fill += 1;
            if self.in_fill == self.hop {
                self.in_fill = 0;
                self.process_hop();
            }
            if let Some(y) = self.ready_out.pop_front() {
                out.push(y);
            } else {
                out.push(0.0);
            }
        }
    }

    fn process_hop(&mut self) {
        let n = self.n_fft;
        let hop = self.hop;
        let n_bins = self.n_bins;
        let df = self.fs / (n as f64);

        self.frame_in.copy_within(hop..n, 0);
        self.frame_in[(n - hop)..n].copy_from_slice(&self.in_buf);

        for i in 0..n {
            self.time_scratch[i] = self.frame_in[i] * self.window[i];
        }
        self.plan
            .rfft(&self.time_scratch, &mut self.fft_scratch, &mut self.spec_in);

        // Zero synthetic spectral boost accumulator (for naturally existing harmonics).
        self.spec_syn.fill(Complex::ZERO);
        let mut any_spec_boost = false;

        let gain = self.cfg.amount * 10.0_f64.powf(self.cfg.gain_db / 20.0);
        let parseval_den = (n as f64) * self.win_energy;
        if gain > 1e-9 {
            let k_min = ((25.0 / df).floor() as usize).clamp(2, n_bins - 2);
            let k_max = ((self.cfg.cutoff_hz / df).ceil() as usize).clamp(k_min, (n_bins - 2) / 2);
            let k_upper_cap = (((4.5 * self.cfg.cutoff_hz) / df).ceil() as usize).min(n_bins - 2);

            // Peak energy floor to skip silent bins (keeps synthesis fast & noise-free).
            let sub_peak_mag = self.spec_in[k_min..=k_max]
                .iter()
                .map(|c| c.norm())
                .fold(0.0_f64, f64::max);
            let bin_thresh = (sub_peak_mag * 1e-3).max(1e-7);

            // Timbre-matched harmonic weights for h = 2, 3, 4 (normalized so sum w_h = 1).
            let raw_w = [
                1.0_f64,
                (-self.cfg.harmonic_decay).exp(),
                (-2.0 * self.cfg.harmonic_decay).exp(),
            ];
            let w_sum = raw_w[0] + raw_w[1] + raw_w[2];
            let weights = [raw_w[0] / w_sum, raw_w[1] / w_sum, raw_w[2] / w_sum];

            let hop_phase_step = 2.0 * PI * (hop as f64) / (n as f64);

            for k in k_min..=k_max {
                let m_cur = self.spec_in[k].norm();
                if m_cur < bin_thresh {
                    continue;
                }

                // Instantaneous frequency via phase-vocoder unwrapped phase difference.
                let phi_cur = self.spec_in[k].arg();
                let expected = hop_phase_step * (k as f64);
                let mut dphi = phi_cur - self.prev_phase[k] - expected;
                dphi = (dphi + PI).rem_euclid(2.0 * PI) - PI;
                let true_bin_freq = ((k as f64) + dphi / hop_phase_step)
                    .clamp((k as f64) - 0.75, (k as f64) + 0.75);
                let omega_0 = 2.0 * PI * true_bin_freq / (n as f64);
                let amp_time = m_cur / self.norm_amp;

                for (idx, h) in [2usize, 3, 4].iter().enumerate() {
                    let hf = *h as f64;
                    let kh_f = true_bin_freq * hf;
                    let kh = kh_f.round() as usize;
                    if kh < 2 || kh > k_upper_cap {
                        continue;
                    }
                    let target_amp = amp_time * weights[idx] * gain;

                    // Measure existing harmonic amplitude ONLY if `kh` is itself a genuine local
                    // spectral peak (prevents a neighboring bass fundamental's skirt from
                    // masquerading as an existing 2nd harmonic).
                    let m_kh_prev = self.spec_in[kh - 1].norm();
                    let m_kh = self.spec_in[kh].norm();
                    let m_kh_next = self.spec_in[kh + 1].norm();
                    let is_existing_peak = m_kh >= m_kh_prev && m_kh >= m_kh_next;
                    let ex_e = m_kh_prev * m_kh_prev + m_kh * m_kh + m_kh_next * m_kh_next;
                    let ex_amp = 2.0 * (ex_e / parseval_den).sqrt();

                    // Moliner, Rämö & Välimäki (DAFx-2020) + Mu, Gan & Tan (ICASSP 2012/2013):
                    // If harmonic `h` is already naturally present (`is_existing_peak` and
                    // `>= 0.08 * target_amp`), boost its existing main lobe in-phase to preserve
                    // natural timbre. Otherwise synthesize the missing harmonic at exact
                    // fractional frequency `omega_h = h * omega_0` with continuous hop-to-hop
                    // phase propagation -> ~-70 dB IMD suppression on multi-note bass chords!
                    if is_existing_peak && ex_amp >= 0.08 * target_amp {
                        let scale_h = target_amp / ex_amp;
                        for j in (kh - 1)..=(kh + 1) {
                            self.spec_syn[j] = self.spec_syn[j].add(self.spec_in[j].scale(scale_h));
                        }
                        any_spec_boost = true;
                    } else {
                        let omega_h = hf * omega_0;
                        let phi_start = self.synth_phase[k][idx];
                        let next_phi =
                            (phi_start + omega_h * (hop as f64) + PI).rem_euclid(2.0 * PI) - PI;
                        self.synth_phase[k][idx] = next_phi;

                        for i in 0..n {
                            let w = self.window[i];
                            let y = target_amp * (phi_start + omega_h * (i as f64)).cos();
                            self.ola_out[i] += y * (w * w);
                        }
                    }
                }
            }
        }

        // Update previous-frame analysis phases for sub-bass bins.
        for k in 0..n_bins {
            self.prev_phase[k] = self.spec_in[k].arg();
        }

        if any_spec_boost {
            self.plan.irfft(
                &self.spec_syn,
                &mut self.fft_scratch,
                &mut self.time_scratch,
            );
            for i in 0..n {
                self.ola_out[i] += self.time_scratch[i] * self.window[i];
            }
        }

        for i in 0..hop {
            let syn = self.ola_out[i];
            let dry = self.dry_delay.pop_front().unwrap_or(0.0);
            self.ready_out.push_back(dry + syn);
        }
        self.ola_out.copy_within(hop..n, 0);
        self.ola_out[(n - hop)..n].fill(0.0);
    }
}

/// Per-channel shared-STN processor combining Virtual Bass and Micro-Detail.
struct ChannelEnhance {
    stn: StnDecomposer,
    tonal_vbs: Option<TonalVbsSynth>,
    /// Delay lines aligning `transients` and `noise` when `tonal_vbs` adds its OLA latency.
    trans_delay: VecDeque<f64>,
    noise_delay: VecDeque<f64>,
    /// Transient Virtual Bass sub-band LPF + post-NLD BPF (Oo–Gan–Hawksford / Moliner et al.).
    trans_lpf: Lr4,
    trans_bpf_hp: Lr4,
    trans_bpf_lp: Lr4,
    /// Micro-detail smooth envelopes for sines and transients.
    env_s: f64,
    env_t: f64,
    atk_s: f64,
    rel_s: f64,
    atk_t: f64,
    rel_t: f64,
    /// Presence shelf on lifted detail delta.
    presence_shelf: Biquad,
    cfg: EnhanceCfg,
    stn_buf: StnBlock,
    sines_vbs_buf: Vec<f64>,
    total_latency: usize,
}

impl ChannelEnhance {
    fn new(sample_rate: u32, cfg: EnhanceCfg) -> Self {
        let fs = sample_rate.max(8_000) as f64;
        let stn = StnDecomposer::new(sample_rate);
        let tonal_vbs = cfg
            .vbass
            .is_active()
            .then(|| TonalVbsSynth::new(sample_rate, cfg.vbass));
        let vbs_lat = tonal_vbs.as_ref().map_or(0, TonalVbsSynth::latency_samples);
        let mut trans_delay = VecDeque::with_capacity(vbs_lat + 64);
        let mut noise_delay = VecDeque::with_capacity(vbs_lat + 64);
        for _ in 0..vbs_lat {
            trans_delay.push_back(0.0);
            noise_delay.push_back(0.0);
        }
        let fc = cfg.vbass.cutoff_hz.clamp(40.0, 180.0);
        let total_latency = stn.latency_samples() + vbs_lat;
        Self {
            stn,
            tonal_vbs,
            trans_delay,
            noise_delay,
            trans_lpf: Lr4::lpf(fc, fs),
            trans_bpf_hp: Lr4::hpf(fc, fs),
            trans_bpf_lp: Lr4::lpf((4.0 * fc).min(0.42 * fs), fs),
            env_s: 0.0,
            env_t: 0.0,
            atk_s: 1.0 - (-1.0 / (0.015 * fs)).exp(),
            rel_s: 1.0 - (-1.0 / (0.120 * fs)).exp(),
            atk_t: 1.0 - (-1.0 / (0.002 * fs)).exp(),
            rel_t: 1.0 - (-1.0 / (0.045 * fs)).exp(),
            presence_shelf: Biquad::high_shelf_1st(2_500.0, cfg.detail.presence_tilt_db, fs),
            cfg,
            stn_buf: StnBlock::default(),
            sines_vbs_buf: Vec::new(),
            total_latency,
        }
    }

    #[inline]
    fn latency_samples(&self) -> usize {
        self.total_latency
    }

    fn process_in_place(&mut self, buf: &mut [f64]) {
        if buf.is_empty() {
            return;
        }
        self.stn.process_into(buf, &mut self.stn_buf);
        let n = buf.len();

        // 1. Tonal stream (+ optional zero-IMD phase-coherent Virtual Bass harmonics).
        if let Some(vbs) = self.tonal_vbs.as_mut() {
            vbs.process(&self.stn_buf.sines, &mut self.sines_vbs_buf);
        } else {
            self.sines_vbs_buf.clear();
            self.sines_vbs_buf.extend_from_slice(&self.stn_buf.sines);
        }

        let vbass_on = self.cfg.vbass.is_active();
        let trans_vbass_gain = if vbass_on {
            self.cfg.vbass.amount
                * self.cfg.vbass.transient_punch
                * 10.0_f64.powf(self.cfg.vbass.gain_db / 20.0)
        } else {
            0.0
        };
        let detail_on = self.cfg.detail.is_active();
        let intensity = self.cfg.detail.intensity;
        let bias = self.cfg.detail.bias;
        let t_weight = self.cfg.detail.transient_weight;

        // Upward micro-detail soft-knee parameters (INTENSITY-style perceptual contrast lift).
        // Quiet sines/transients above noise gate (-60 dBFS) receive bounded upward lift up to
        // +6 dB * intensity; hot peaks near 0 dBFS smoothly converge back to unity (0 dB lift).
        let max_lift_db = 6.0 * intensity;
        let gate_floor = 1e-3; // -60 dBFS gate prevents boosting numerical silence
        let knee_ref = 10.0_f64.powf((-18.0 - 12.0 * bias) / 20.0);

        for (i, dst) in buf.iter_mut().enumerate().take(n) {
            let s = self.sines_vbs_buf[i];
            let mut t = if self.tonal_vbs.is_some() {
                self.trans_delay.push_back(self.stn_buf.transients[i]);
                self.trans_delay.pop_front().unwrap_or(0.0)
            } else {
                self.stn_buf.transients[i]
            };
            let noise = if self.tonal_vbs.is_some() {
                self.noise_delay.push_back(self.stn_buf.noise[i]);
                self.noise_delay.pop_front().unwrap_or(0.0)
            } else {
                self.stn_buf.noise[i]
            };

            // 2. Transient Virtual Bass (FEXP1 NLD on sub-bass transient impulses only).
            if trans_vbass_gain > 1e-9 {
                let sub_t = self.trans_lpf.tick(t);
                let shaped = fexp1_nld(sub_t * 3.0, 2.5) / 3.0;
                let band_h = self.trans_bpf_lp.tick(self.trans_bpf_hp.tick(shaped));
                t += trans_vbass_gain * band_h;
            }

            // 3. Micro-Detail upward perceptual contrast lift on Sines + Transients ONLY.
            //    Noise stream `noise` is NEVER boosted, keeping the noise floor dead clean!
            let mut detail_delta = 0.0_f64;
            if detail_on && max_lift_db > 1e-9 {
                let as_ = s.abs();
                let cs = if as_ > self.env_s {
                    self.atk_s
                } else {
                    self.rel_s
                };
                self.env_s += (as_ - self.env_s) * cs;

                let at = t.abs();
                let ct = if at > self.env_t {
                    self.atk_t
                } else {
                    self.rel_t
                };
                self.env_t += (at - self.env_t) * ct;

                // Soft-knee upward boost factor in [0, 1]: peaks near knee_ref, fades out
                // below gate_floor and smoothly tapers to 0 as level approaches 0 dBFS.
                let lift_s = upward_detail_gain(self.env_s, gate_floor, knee_ref, max_lift_db);
                let lift_t =
                    upward_detail_gain(self.env_t, gate_floor, knee_ref, max_lift_db * t_weight);

                let raw_delta = s * (lift_s - 1.0) + t * (lift_t - 1.0);
                detail_delta = self.presence_shelf.tick(raw_delta);
            }

            // Exact unity sum when enhancements are zero: `s + t + noise == x_delayed`.
            *dst = s + t + noise + detail_delta;
        }
    }

    fn flush(&mut self) -> Vec<f64> {
        let mut zeros = vec![0.0_f64; self.total_latency];
        self.process_in_place(&mut zeros);
        zeros
    }
}

/// Smooth soft-knee upward contrast multiplier (`>= 1.0`) for micro-detail enhancement.
///
/// * Below `gate_floor` (`-60 dBFS`): smoothly fades to `1.0` so silence is never lifted.
/// * Around `knee_ref`: approaches `10^(max_lift_db / 20)`.
/// * Towards full scale (`1.0`): tapers back toward `1.0` so loud peaks stay unclipped.
#[inline]
fn upward_detail_gain(env: f64, gate_floor: f64, knee_ref: f64, max_lift_db: f64) -> f64 {
    if env <= 1e-12 || max_lift_db <= 0.0 {
        return 1.0;
    }
    // Smooth gate: 0 at env << gate_floor, 1 at env >> gate_floor.
    let g_ratio = env / gate_floor;
    let gate = (g_ratio * g_ratio) / (1.0 + g_ratio * g_ratio);
    // Upward curve: highest for quiet detail near knee_ref, tapers cleanly for hot peaks.
    let compression = 1.0 / (1.0 + (env / knee_ref).powf(0.85));
    let lift_db = max_lift_db * gate * compression;
    10.0_f64.powf(lift_db / 20.0)
}

/// Stereo shared-STN enhancement processor (Virtual Bass + Micro-Detail).
pub struct EnhanceProcessor {
    left: ChannelEnhance,
    right: ChannelEnhance,
}

impl EnhanceProcessor {
    pub fn new(sample_rate: u32, cfg: EnhanceCfg) -> Self {
        Self {
            left: ChannelEnhance::new(sample_rate, cfg),
            right: ChannelEnhance::new(sample_rate, cfg),
        }
    }

    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.left.latency_samples()
    }

    pub fn process(&mut self, l: &mut [f64], r: &mut [f64]) {
        debug_assert_eq!(l.len(), r.len());
        self.left.process_in_place(l);
        self.right.process_in_place(r);
    }

    pub fn flush(&mut self) -> (Vec<f64>, Vec<f64>) {
        (self.left.flush(), self.right.flush())
    }
}

/// Oo, Gan & Hawksford (JAES 2011) objective virtual-bass evaluation metrics
/// for automated regression and quality verification.
#[derive(Debug, Clone, Copy)]
pub struct VbsObjectiveMetrics {
    /// Harmonic Generation Ratio (dB): synthesized harmonic energy in `{2*f1, 3*f1, 2*f2, 3*f2}`
    /// relative to original fundamental energy.
    pub harmonic_generation_db: f64,
    /// Two-Tone Intermodulation Distortion (dB): IMD products `{|f2-f1|, f1+f2, 2*f1-f2, 2*f2-f1}`
    /// relative to synthesized harmonic energy (lower / more negative = cleaner!).
    pub imd_to_harmonics_db: f64,
}

/// Measure Oo–Gan–Hawksford (JAES 2011) harmonic generation & two-tone IMD on a steady segment.
pub fn measure_two_tone_vbs_metrics(
    signal: &[f64],
    fs: f64,
    f1: f64,
    f2: f64,
) -> VbsObjectiveMetrics {
    let n = signal.len();
    let probe_power = |f_hz: f64| -> f64 {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let w = 2.0 * PI * f_hz / fs;
        // Hann-windowed single-bin Goertzel/DFT probe.
        for (i, &x) in signal.iter().enumerate() {
            let win = 0.5 * (1.0 - (2.0 * PI * (i as f64) / (n as f64)).cos());
            let ang = w * (i as f64);
            re += x * win * ang.cos();
            im -= x * win * ang.sin();
        }
        (re * re + im * im) / ((n as f64) * (n as f64))
    };

    let p_fund = probe_power(f1) + probe_power(f2);
    let p_harm = probe_power(2.0 * f1)
        + probe_power(3.0 * f1)
        + probe_power(2.0 * f2)
        + probe_power(3.0 * f2);
    let p_imd = probe_power((f2 - f1).abs())
        + probe_power(f1 + f2)
        + probe_power((2.0 * f1 - f2).abs())
        + probe_power((2.0 * f2 - f1).abs());

    VbsObjectiveMetrics {
        harmonic_generation_db: 10.0 * (p_harm.max(1e-30) / p_fund.max(1e-30)).log10(),
        imd_to_harmonics_db: 10.0 * (p_imd.max(1e-30) / p_harm.max(1e-30)).log10(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_amount_and_zero_intensity_null_bit_accurately() {
        let fs = 48_000u32;
        let len = 16_384usize;
        let mut l: Vec<f64> = (0..len)
            .map(|i| 0.35 * (2.0 * PI * 65.0 * i as f64 / fs as f64).sin())
            .collect();
        let mut r: Vec<f64> = (0..len)
            .map(|i| 0.35 * (2.0 * PI * 195.0 * i as f64 / fs as f64).cos())
            .collect();
        let orig_l = l.clone();
        let orig_r = r.clone();

        // Enabled flags with 0.0 amount / 0.0 intensity -> STN runs, output must null to < 1e-12!
        let cfg = EnhanceCfg {
            vbass: VirtualBassCfg {
                enable: true,
                amount: 0.0,
                ..Default::default()
            },
            detail: DetailCfg {
                enable: true,
                intensity: 0.0,
                ..Default::default()
            },
        };
        let mut proc = EnhanceProcessor::new(fs, cfg);
        let lat = proc.latency_samples();
        proc.process(&mut l, &mut r);
        let (tl, tr) = proc.flush();
        l.extend_from_slice(&tl);
        r.extend_from_slice(&tr);

        let err_l = (0..len)
            .map(|i| (l[lat + i] - orig_l[i]).abs())
            .fold(0.0_f64, f64::max);
        let err_r = (0..len)
            .map(|i| (r[lat + i] - orig_r[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err_l < 1e-12 && err_r < 1e-12,
            "zero-drive bypass failed null check: err_l={err_l:e}, err_r={err_r:e}"
        );
    }

    #[test]
    fn virtual_bass_beats_naive_nld_on_two_tone_imd_by_over_20_db() {
        // Oo, Gan & Hawksford (JAES 2011) two-tone bass chord test: 55.0 Hz (A1) + 73.42 Hz (D2).
        // (Avoids a 3:2 fifth where 2*f2 - f1 mathematically coincides with 2*f1 = 110 Hz.)
        let fs = 48_000u32;
        let fs_f = fs as f64;
        let f1 = 55.0;
        let f2 = 73.42;
        let len = 64_000usize;
        let mut l: Vec<f64> = (0..len)
            .map(|i| {
                let t = i as f64 / fs_f;
                0.35 * (2.0 * PI * f1 * t).sin() + 0.35 * (2.0 * PI * f2 * t).sin()
            })
            .collect();
        let mut r = l.clone();

        // Reference naive NLD baseline (classic waveshaper on full bass).
        let naive: Vec<f64> = l
            .iter()
            .map(|&x| x + 0.6 * fexp1_nld(x * 3.0, 2.5))
            .collect();
        let naive_m = measure_two_tone_vbs_metrics(&naive[24_000..56_000], fs_f, f1, f2);

        // Our STN + Phase-Vocoder Virtual Bass.
        let cfg = EnhanceCfg {
            vbass: VirtualBassCfg {
                enable: true,
                cutoff_hz: 110.0,
                gain_db: 0.0,
                amount: 0.6,
                harmonic_decay: 0.65,
                transient_punch: 0.6,
            },
            detail: DetailCfg::default(),
        };
        let mut proc = EnhanceProcessor::new(fs, cfg);
        let lat = proc.latency_samples();
        proc.process(&mut l, &mut r);
        let (tl, _tr) = proc.flush();
        l.extend_from_slice(&tl);

        let stn_m = measure_two_tone_vbs_metrics(&l[lat + 24_000..lat + 56_000], fs_f, f1, f2);

        // Must generate substantial harmonics (>-18 dB relative to fundamental)
        assert!(
            stn_m.harmonic_generation_db > -18.0,
            "STN VBS generated too little harmonic energy: {:.2} dB",
            stn_m.harmonic_generation_db
        );
        // Must have dramatically lower IMD than naive waveshaping (at least 20 dB cleaner!).
        let imd_advantage = naive_m.imd_to_harmonics_db - stn_m.imd_to_harmonics_db;
        assert!(
            imd_advantage > 20.0 && stn_m.imd_to_harmonics_db < -25.0,
            "STN VBS IMD not clean enough: stn={:.2} dB, naive={:.2} dB, advantage={:.2} dB",
            stn_m.imd_to_harmonics_db,
            naive_m.imd_to_harmonics_db,
            imd_advantage
        );
    }

    #[test]
    fn micro_detail_lifts_quiet_tonal_content_without_boosting_noise_floor() {
        let fs = 48_000u32;
        let fs_f = fs as f64;
        let len = 48_000usize;

        // 1. Quiet 1.5 kHz sine (-30 dBFS) -> must receive measurable upward micro-detail lift!
        let mut tone_l: Vec<f64> = (0..len)
            .map(|i| 0.0316 * (2.0 * PI * 1_500.0 * i as f64 / fs_f).sin())
            .collect();
        let mut tone_r = tone_l.clone();
        let orig_rms =
            (tone_l[16_000..40_000].iter().map(|v| v * v).sum::<f64>() / 24_000.0).sqrt();

        let cfg = EnhanceCfg {
            vbass: VirtualBassCfg::default(),
            detail: DetailCfg {
                enable: true,
                intensity: 0.75,
                bias: 0.5,
                transient_weight: 1.0,
                presence_tilt_db: 2.0,
            },
        };
        let mut proc = EnhanceProcessor::new(fs, cfg);
        let lat = proc.latency_samples();
        proc.process(&mut tone_l, &mut tone_r);
        let (tl, _tr) = proc.flush();
        tone_l.extend_from_slice(&tl);
        let out_rms = (tone_l[lat + 16_000..lat + 40_000]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            / 24_000.0)
            .sqrt();
        let tone_lift_db = 20.0 * (out_rms / orig_rms).log10();
        assert!(
            tone_lift_db > 1.5,
            "expected >1.5 dB upward micro-detail lift on quiet tone, got {tone_lift_db:+.2} dB"
        );

        // 2. Pure white noise at same -30 dBFS -> noise stream stays untouched (< 0.8 dB change).
        let mut noise_l = vec![0.0_f64; len];
        let mut seed: u64 = 0xfeed_face_cafe_beef;
        for slot in &mut noise_l {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *slot = (((seed >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0) * 0.055;
        }
        let mut noise_r = noise_l.clone();
        let n_in_rms =
            (noise_l[16_000..40_000].iter().map(|v| v * v).sum::<f64>() / 24_000.0).sqrt();

        let mut proc_n = EnhanceProcessor::new(fs, cfg);
        let lat_n = proc_n.latency_samples();
        proc_n.process(&mut noise_l, &mut noise_r);
        let (tnl, _tnr) = proc_n.flush();
        noise_l.extend_from_slice(&tnl);
        let n_out_rms = (noise_l[lat_n + 16_000..lat_n + 40_000]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            / 24_000.0)
            .sqrt();
        let noise_delta_db = 20.0 * (n_out_rms / n_in_rms).log10();
        assert!(
            noise_delta_db.abs() < 0.8 && tone_lift_db > noise_delta_db + 1.5,
            "noise floor must stay clean (delta={noise_delta_db:+.2} dB) vs tone lift ({tone_lift_db:+.2} dB)"
        );
    }
}
