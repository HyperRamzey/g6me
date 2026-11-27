//! Fierro & Välimäki (JAES 2023 / arXiv 2022) Enhanced Fuzzy Sines–Transients–Noise (STN)
//! decomposition built on Driedger, Müller & Disch (ISMIR 2014) dual-resolution HPSS.
//!
//! ## Architectural role
//!
//! This shared front-end decomposes any mono `f64` stream into three strictly
//! unity-summing streams `(sines, transients, noise)`:
//!
//! * **Stage 1 (coarse tonal resolution, `N_1 = 4096 * sr_scale`, `H_1 = N_1 / 4`)**:
//!   extracts narrowband sinusoidal components `x_s` via horizontal (`~200 ms`) and
//!   vertical (`~500 Hz`) spectrogram median filtering + squared-sine fuzzy membership.
//!   Exact time-aligned residual `x_res = x_delayed - x_s`.
//! * **Stage 2 (fine transient resolution, `N_2 = 512 * sr_scale`, `H_2 = N_2 / 4`)**:
//!   extracts impulsive percussive transients `x_t` from `x_res`, and defines the
//!   noise/texture stream as the exact residual `x_n = x_res_delayed - x_t`.
//!
//! ## Exact Perfect Reconstruction Guarantee
//!
//! 1. Within each STFT stage, the squared-sine fuzzy membership transition
//!    `f(r; beta_l, beta_u) = sin^2((pi/2) * (r - beta_l) / (beta_u - beta_l))`
//!    with `beta_l >= 0.5` enforces `S_s + S_t <= 1` and `S_n = 1 - S_s - S_t >= 0`,
//!    so `S_s + S_t + S_n == 1.0` at every time-frequency bin.
//! 2. Across the 2-stage cascade, `x_s_aligned + x_t + x_n == x_aligned` to within
//!    floating-point rounding (`< 1e-12`, `< -240 dB`), independent of chunk boundaries.
//!
//! Pure deterministic `f64`, zero external crates, zero FMA contraction.

use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::fft::{Complex, FftPlan};

/// Default lower fuzzy transition threshold `beta_L` (Fierro & Välimäki 2022/2023).
pub const DEFAULT_BETA_L: f64 = 0.7;
/// Default upper fuzzy transition threshold `beta_U` (Fierro & Välimäki 2022/2023).
pub const DEFAULT_BETA_U: f64 = 0.8;

/// Power-of-two sample-rate scale relative to 48 kHz (`1` at <=48 kHz, `2` at 96 kHz, `4` at 192 kHz).
#[inline]
pub fn sr_scale(sample_rate: u32) -> usize {
    let mult = (sample_rate.max(8_000) as f64 / 48_000.0).ceil() as usize;
    mult.max(1).next_power_of_two().min(8)
}

/// Fierro & Välimäki (JAES 2023) squared-sine fuzzy membership transition function.
///
/// Maps anisotropy indicator `r in [0, 1]` smoothly across `[beta_l, beta_u]`:
/// * `0.0` for `r <= beta_l`
/// * `sin^2((pi / 2) * (r - beta_l) / (beta_u - beta_l))` for `beta_l < r < beta_u`
/// * `1.0` for `r >= beta_u`
#[inline]
pub fn fuzzy_sin2(r: f64, beta_l: f64, beta_u: f64) -> f64 {
    if r <= beta_l {
        0.0
    } else if r >= beta_u {
        1.0
    } else {
        let span = (beta_u - beta_l).max(1e-12);
        let t = (r - beta_l) / span;
        let s = (0.5 * PI * t).sin();
        s * s
    }
}

/// Compute unity-summing `(S_s, S_t, S_n)` fuzzy masks from horizontal median `m_h`
/// and vertical median `m_v` magnitudes.
///
/// Requires `beta_l >= 0.5` and `beta_u > beta_l` so `S_s` and `S_t` cannot both be
/// positive simultaneously, guaranteeing `S_n = 1 - S_s - S_t >= 0` and
/// `S_s + S_t + S_n == 1.0` identically.
#[inline]
pub fn fuzzy_masks(m_h: f64, m_v: f64, beta_l: f64, beta_u: f64) -> (f64, f64, f64) {
    let denom = m_h + m_v;
    if denom <= 1e-30 {
        return (0.0, 0.0, 1.0);
    }
    let r_s = m_h / denom;
    let r_t = 1.0 - r_s;
    let bl = beta_l.clamp(0.5, 0.99);
    let bu = beta_u.clamp(bl + 1e-6, 1.0);
    let s_s = fuzzy_sin2(r_s, bl, bu);
    let s_t = fuzzy_sin2(r_t, bl, bu);
    let s_n = (1.0 - s_s - s_t).max(0.0);
    (s_s, s_t, s_n)
}

/// Exact median of a non-empty slice (using a caller-owned scratch buffer to avoid allocation).
#[inline]
fn median_into(vals: &[f64], scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend_from_slice(vals);
    let n = scratch.len();
    debug_assert!(n > 0);
    let mid = n / 2;
    scratch.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n % 2 == 1 {
        scratch[mid]
    } else {
        let hi = scratch[mid];
        let lo = scratch[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// Build an exact 75%-overlap COLA `sqrt(0.5 * periodic_hann)` window of length `n`.
///
/// Satisfies `sum_{m=0..3} w[k + m * (n/4)]^2 == 1.0` to `< 1e-15` for all `k in 0..n/4`.
pub fn sqrt_hann_cola_window(n: usize) -> Vec<f64> {
    let inv_n = 1.0 / (n as f64);
    (0..n)
        .map(|i| {
            let hann = 0.5 * (1.0 - (2.0 * PI * (i as f64) * inv_n).cos());
            (0.5 * hann).sqrt()
        })
        .collect()
}

/// Single-resolution streaming STFT median-filter extractor.
///
/// Extracts either the sinusoidal component (`extract_sines == true`) or the transient
/// component (`extract_sines == false`), and simultaneously yields the exact
/// time-aligned input sample (`delayed_input`) so downstream residual subtraction
/// `delayed_input - extracted` is sample-accurate to `< 1e-13`.
struct StftMedianStage {
    n_fft: usize,
    hop: usize,
    n_bins: usize,
    h_half: usize,
    v_half: usize,
    beta_l: f64,
    beta_u: f64,
    extract_sines: bool,
    plan: FftPlan,
    window: Vec<f64>,
    /// Incoming sample accumulator (`hop` samples trigger one STFT frame).
    in_buf: Vec<f64>,
    in_fill: usize,
    /// Sliding ring of `n_fft` latest input samples (pre-padded with `n_fft - hop` zeros).
    frame_in: Vec<f64>,
    /// FIFO of complex STFT frames + magnitudes for centered horizontal median (`2*h_half + 1`).
    spec_ring: VecDeque<Vec<Complex>>,
    mag_ring: VecDeque<Vec<f64>>,
    /// Exact input delay line aligned to total stage latency `(n_fft - hop) + h_half * hop`.
    input_delay: VecDeque<f64>,
    /// Overlap-add output buffer of length `n_fft`.
    ola_out: Vec<f64>,
    /// Reusable FFT / median scratch buffers.
    fft_scratch: Vec<Complex>,
    spec_scratch: Vec<Complex>,
    time_scratch: Vec<f64>,
    m_h_buf: Vec<f64>,
    m_v_buf: Vec<f64>,
    med_vals: Vec<f64>,
    med_scratch: Vec<f64>,
    /// Ready aligned output pairs `(extracted, delayed_input)`.
    ready_ext: VecDeque<f64>,
    ready_in: VecDeque<f64>,
    /// Number of initial pre-latency zero samples to discard before aligning output.
    /// Wait: we DO NOT discard them internally; we report `latency_samples()` and emit
    /// strictly causal 1-to-1 samples (latency zeros followed by aligned signal), and
    /// `flush()` drains the final `latency_samples()` samples!
    total_latency: usize,
    /// Number of input frames seen so far (for centering the horizontal median queue).
    frames_pushed: usize,
}

impl StftMedianStage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_fft: usize,
        h_half: usize,
        v_half: usize,
        beta_l: f64,
        beta_u: f64,
        extract_sines: bool,
    ) -> Self {
        assert!(n_fft.is_power_of_two() && n_fft >= 16);
        let hop = n_fft / 4;
        let n_bins = n_fft / 2 + 1;
        // `input_delay` is popped inside `process_hop` into `ready_in`, so it shares the
        // `hop - 1` hop-buffering delay with `ready_ext`. Pre-filling `(n_fft - hop) + h_half * hop`
        // aligns `out_in` to `out_ext` to < 1e-15; the externally visible causal latency of both
        // streams is `(n_fft - hop) + h_half * hop + (hop - 1) == n_fft + h_half * hop - 1`.
        let in_delay_len = (n_fft - hop) + h_half * hop;
        let total_latency = in_delay_len + (hop - 1);
        let mut input_delay = VecDeque::with_capacity(in_delay_len + hop);
        for _ in 0..in_delay_len {
            input_delay.push_back(0.0);
        }
        // Pre-seed the left half of the horizontal median window (`h_half` zero frames)
        // so that frame 0 sits at index `h_half` once `h_half` future frames arrive.
        let win_len = 2 * h_half + 1;
        let mut spec_ring = VecDeque::with_capacity(win_len + 1);
        let mut mag_ring = VecDeque::with_capacity(win_len + 1);
        for _ in 0..h_half {
            spec_ring.push_back(vec![Complex::ZERO; n_bins]);
            mag_ring.push_back(vec![0.0; n_bins]);
        }
        Self {
            n_fft,
            hop,
            n_bins,
            h_half,
            v_half,
            beta_l,
            beta_u,
            extract_sines,
            plan: FftPlan::new(n_fft),
            window: sqrt_hann_cola_window(n_fft),
            in_buf: vec![0.0; hop],
            in_fill: 0,
            frame_in: vec![0.0; n_fft],
            spec_ring,
            mag_ring,
            input_delay,
            ola_out: vec![0.0; n_fft],
            fft_scratch: vec![Complex::ZERO; n_fft],
            spec_scratch: vec![Complex::ZERO; n_bins],
            time_scratch: vec![0.0; n_fft],
            m_h_buf: vec![0.0; n_bins],
            m_v_buf: vec![0.0; n_bins],
            med_vals: Vec::with_capacity((2 * h_half + 1).max(2 * v_half + 1)),
            med_scratch: Vec::with_capacity((2 * h_half + 1).max(2 * v_half + 1)),
            ready_ext: VecDeque::with_capacity(hop * 2),
            ready_in: VecDeque::with_capacity(hop * 2),
            total_latency,
            frames_pushed: 0,
        }
    }

    #[inline]
    fn latency_samples(&self) -> usize {
        self.total_latency
    }

    /// Push `input.len()` samples and produce `input.len()` aligned `(extracted, delayed_input)` pairs.
    fn process(&mut self, input: &[f64], out_ext: &mut Vec<f64>, out_in: &mut Vec<f64>) {
        out_ext.clear();
        out_in.clear();
        out_ext.reserve(input.len());
        out_in.reserve(input.len());

        for &x in input {
            self.input_delay.push_back(x);
            self.in_buf[self.in_fill] = x;
            self.in_fill += 1;
            if self.in_fill == self.hop {
                self.in_fill = 0;
                self.process_hop();
            }
            if let (Some(e), Some(d)) = (self.ready_ext.pop_front(), self.ready_in.pop_front()) {
                out_ext.push(e);
                out_in.push(d);
            } else {
                // Before the first full hop completes, emit causal pre-roll zeros
                // while keeping `input_delay` strictly synchronized.
                out_ext.push(0.0);
                out_in.push(0.0);
            }
        }
    }

    /// Process one completed `hop` of input samples into the STFT/median/OLA pipeline.
    fn process_hop(&mut self) {
        let n = self.n_fft;
        let hop = self.hop;
        let n_bins = self.n_bins;

        // Shift `frame_in` left by `hop` and append the new `hop` samples.
        self.frame_in.copy_within(hop..n, 0);
        self.frame_in[(n - hop)..n].copy_from_slice(&self.in_buf);

        // Apply analysis window and compute forward RFFT.
        for i in 0..n {
            self.time_scratch[i] = self.frame_in[i] * self.window[i];
        }
        let mut spec = vec![Complex::ZERO; n_bins];
        self.plan
            .rfft(&self.time_scratch, &mut self.fft_scratch, &mut spec);
        let mag: Vec<f64> = spec.iter().map(|c| c.norm()).collect();

        self.spec_ring.push_back(spec);
        self.mag_ring.push_back(mag);
        self.frames_pushed += 1;

        let win_len = 2 * self.h_half + 1;
        if self.spec_ring.len() < win_len {
            // Still filling lookahead frames for the horizontal median: emit `hop` zeros.
            for _ in 0..hop {
                let d = self.input_delay.pop_front().unwrap_or(0.0);
                self.ready_ext.push_back(0.0);
                self.ready_in.push_back(d);
            }
            return;
        }

        // Center frame is at index `self.h_half` in `spec_ring` / `mag_ring`.
        let center = self.h_half;

        // Horizontal median per bin across `win_len` time frames.
        for k in 0..n_bins {
            self.med_vals.clear();
            for m in 0..win_len {
                self.med_vals.push(self.mag_ring[m][k]);
            }
            self.m_h_buf[k] = median_into(&self.med_vals, &mut self.med_scratch);
        }

        // Vertical median per bin across `[k - v_half ..= k + v_half]` of center frame.
        let center_mag = &self.mag_ring[center];
        for k in 0..n_bins {
            let lo = k.saturating_sub(self.v_half);
            let hi = (k + self.v_half).min(n_bins - 1);
            self.m_v_buf[k] = median_into(&center_mag[lo..=hi], &mut self.med_scratch);
        }

        // Apply squared-sine fuzzy mask to center complex spectrum.
        let center_spec = &self.spec_ring[center];
        for (k, dst) in self.spec_scratch.iter_mut().enumerate().take(n_bins) {
            let (s_s, s_t, _s_n) =
                fuzzy_masks(self.m_h_buf[k], self.m_v_buf[k], self.beta_l, self.beta_u);
            let mask = if self.extract_sines { s_s } else { s_t };
            *dst = center_spec[k].scale(mask);
        }

        // Pop oldest frame to maintain ring size `win_len - 1` before next push.
        self.spec_ring.pop_front();
        self.mag_ring.pop_front();

        // Inverse RFFT + synthesis window + overlap-add.
        self.plan.irfft(
            &self.spec_scratch,
            &mut self.fft_scratch,
            &mut self.time_scratch,
        );
        for i in 0..n {
            self.ola_out[i] += self.time_scratch[i] * self.window[i];
        }

        // Emit the oldest `hop` completed OLA samples paired with aligned `input_delay`.
        for i in 0..hop {
            let ext_s = self.ola_out[i];
            let in_s = self.input_delay.pop_front().unwrap_or(0.0);
            self.ready_ext.push_back(ext_s);
            self.ready_in.push_back(in_s);
        }
        self.ola_out.copy_within(hop..n, 0);
        self.ola_out[(n - hop)..n].fill(0.0);
    }
}

/// Output of one streaming STN decomposition block.
///
/// Always satisfies `sines[i] + transients[i] + noise[i] == delayed_input[i]`
/// to `< 1e-12` (`< -240 dB`).
#[derive(Debug, Clone, Default)]
pub struct StnBlock {
    pub sines: Vec<f64>,
    pub transients: Vec<f64>,
    pub noise: Vec<f64>,
}

impl StnBlock {
    pub fn len(&self) -> usize {
        self.sines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sines.is_empty()
    }
}

/// Two-stage cascading Enhanced Fuzzy STN decomposer (Fierro & Välimäki 2022/2023).
///
/// * Stage 1 (`N_1 = 4096 * sr_scale`, `~85.3 ms` at 48 kHz) extracts tonal `sines`
///   with horizontal median `~200 ms` and vertical median `~500 Hz`.
/// * Stage 2 (`N_2 = 512 * sr_scale`, `~10.7 ms` at 48 kHz) decomposes the exact
///   Stage 1 residual `x - x_s` into sharp `transients` and residual `noise`.
pub struct StnDecomposer {
    stage1: StftMedianStage,
    stage2: StftMedianStage,
    /// Delay line keeping Stage 1 `sines` sample-aligned with Stage 2's latency.
    sines_delay: VecDeque<f64>,
    /// Reusable stage-1 buffers.
    s1_sines: Vec<f64>,
    s1_in: Vec<f64>,
    s1_res: Vec<f64>,
    s2_trans: Vec<f64>,
    s2_res_in: Vec<f64>,
    /// Total causal latency in samples (`stage1.latency + stage2.latency`).
    total_latency: usize,
}

impl StnDecomposer {
    /// Create a new streaming STN decomposer for `sample_rate` Hz using Fierro & Välimäki
    /// defaults (`beta_L = 0.7`, `beta_U = 0.8`).
    pub fn new(sample_rate: u32) -> Self {
        Self::with_thresholds(sample_rate, DEFAULT_BETA_L, DEFAULT_BETA_U)
    }

    /// Create a new streaming STN decomposer with custom fuzzy transition thresholds.
    pub fn with_thresholds(sample_rate: u32, beta_l: f64, beta_u: f64) -> Self {
        let fs = sample_rate.max(8_000) as f64;
        let scale = sr_scale(sample_rate);
        let n1 = 4096 * scale;
        let h1 = n1 / 4;
        // Stage 1 median lengths: ~200 ms horizontal, ~500 Hz vertical (Fierro & Välimäki).
        let h1_half = ((0.200 * fs / (h1 as f64) - 1.0) * 0.5)
            .round()
            .clamp(2.0, 8.0) as usize;
        let df1 = fs / (n1 as f64);
        let v1_half = ((500.0 / df1 - 1.0) * 0.5).round().clamp(4.0, 32.0) as usize;

        let n2 = 512 * scale;
        let h2 = n2 / 4;
        // Stage 2 median lengths: ~25 ms horizontal, ~1000 Hz vertical for crisp transient isolation.
        let h2_half = ((0.025 * fs / (h2 as f64) - 1.0) * 0.5)
            .round()
            .clamp(2.0, 8.0) as usize;
        let df2 = fs / (n2 as f64);
        let v2_half = ((1000.0 / df2 - 1.0) * 0.5).round().clamp(3.0, 16.0) as usize;

        let stage1 = StftMedianStage::new(n1, h1_half, v1_half, beta_l, beta_u, true);
        let stage2 = StftMedianStage::new(n2, h2_half, v2_half, beta_l, beta_u, false);
        let lat2 = stage2.latency_samples();
        let mut sines_delay = VecDeque::with_capacity(lat2 + h1);
        for _ in 0..lat2 {
            sines_delay.push_back(0.0);
        }
        let total_latency = stage1.latency_samples() + lat2;
        Self {
            stage1,
            stage2,
            sines_delay,
            s1_sines: Vec::new(),
            s1_in: Vec::new(),
            s1_res: Vec::new(),
            s2_trans: Vec::new(),
            s2_res_in: Vec::new(),
            total_latency,
        }
    }

    /// Exact total causal latency in samples.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.total_latency
    }

    /// Decompose one input block into aligned `(sines, transients, noise)` streams.
    pub fn process(&mut self, input: &[f64]) -> StnBlock {
        let mut out = StnBlock::default();
        self.process_into(input, &mut out);
        out
    }

    /// Decompose one input block into a caller-provided [`StnBlock`] (reusing allocations).
    pub fn process_into(&mut self, input: &[f64], out: &mut StnBlock) {
        let n = input.len();
        self.stage1
            .process(input, &mut self.s1_sines, &mut self.s1_in);

        // Residual entering Stage 2: `x_res = x_aligned - x_s`.
        self.s1_res.clear();
        self.s1_res.reserve(n);
        for i in 0..n {
            self.s1_res.push(self.s1_in[i] - self.s1_sines[i]);
        }

        // Stage 2 extracts sharp transients `x_t` from `x_res` and returns aligned `x_res`.
        self.stage2
            .process(&self.s1_res, &mut self.s2_trans, &mut self.s2_res_in);

        // Align Stage 1 `sines` to Stage 2's latency; define `noise = x_res_aligned - x_t`.
        out.sines.clear();
        out.transients.clear();
        out.noise.clear();
        out.sines.reserve(n);
        out.transients.reserve(n);
        out.noise.reserve(n);

        for i in 0..n {
            self.sines_delay.push_back(self.s1_sines[i]);
            let s = self.sines_delay.pop_front().unwrap_or(0.0);
            let t = self.s2_trans[i];
            let n_val = self.s2_res_in[i] - t;
            out.sines.push(s);
            out.transients.push(t);
            out.noise.push(n_val);
        }
    }

    /// Drain the final `latency_samples()` frames of programme held in the analysis buffers.
    pub fn flush(&mut self) -> StnBlock {
        let zeros = vec![0.0_f64; self.total_latency];
        self.process(&zeros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_masks_sum_to_unity_and_stay_non_negative() {
        for i in 0..=1000 {
            let m_h = (i as f64) * 0.001;
            let m_v = 1.0 - m_h;
            let (s_s, s_t, s_n) = fuzzy_masks(m_h, m_v, DEFAULT_BETA_L, DEFAULT_BETA_U);
            assert!(
                s_s >= 0.0 && s_t >= 0.0 && s_n >= 0.0,
                "negative mask at i={i}: ({s_s}, {s_t}, {s_n})"
            );
            assert!(
                (s_s + s_t + s_n - 1.0).abs() < 1e-15,
                "unity-sum violated at i={i}: sum={}",
                s_s + s_t + s_n
            );
        }
        // Extreme tonal -> pure sine; extreme vertical -> pure transient; equal -> pure noise.
        assert_eq!(fuzzy_masks(10.0, 0.1, 0.7, 0.8), (1.0, 0.0, 0.0));
        assert_eq!(fuzzy_masks(0.1, 10.0, 0.7, 0.8), (0.0, 1.0, 0.0));
        assert_eq!(fuzzy_masks(1.0, 1.0, 0.7, 0.8), (0.0, 0.0, 1.0));
    }

    #[test]
    fn stn_perfect_reconstruction_null_across_arbitrary_chunks() {
        let fs = 48_000u32;
        let len = 24_000usize;
        // Complex test signal: low sine + mid sine + periodic impulses + deterministic pseudo-noise.
        let mut x = vec![0.0_f64; len];
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        for (i, slot) in x.iter_mut().enumerate() {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = ((seed >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0;
            let t = i as f64 / fs as f64;
            let tone = 0.4 * (2.0 * PI * 80.0 * t).sin() + 0.25 * (2.0 * PI * 1_100.0 * t).sin();
            let imp = if i % 4800 == 1200 { 0.8 } else { 0.0 };
            *slot = tone + imp + 0.08 * u;
        }

        let mut dec = StnDecomposer::new(fs);
        let lat = dec.latency_samples();
        let mut sum_out = Vec::with_capacity(len + lat);

        // Feed in irregular prime chunk sizes to verify streaming invariance.
        let chunks = [313usize, 1009, 4096, 777, 2048];
        let mut pos = 0usize;
        let mut ci = 0usize;
        while pos < len {
            let take = chunks[ci % chunks.len()].min(len - pos);
            let blk = dec.process(&x[pos..pos + take]);
            for i in 0..blk.len() {
                sum_out.push(blk.sines[i] + blk.transients[i] + blk.noise[i]);
            }
            pos += take;
            ci += 1;
        }
        let tail = dec.flush();
        for i in 0..tail.len() {
            sum_out.push(tail.sines[i] + tail.transients[i] + tail.noise[i]);
        }

        assert_eq!(sum_out.len(), len + lat);
        let max_err = (0..len)
            .map(|i| (sum_out[lat + i] - x[i]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-12,
            "STN perfect reconstruction null failed: max_err={max_err:e}"
        );
    }

    #[test]
    fn stn_separates_pure_sine_impulse_and_noise() {
        let fs = 48_000u32;
        let len = 48_000usize; // 1.0 second

        // 1. Pure 440 Hz steady sine -> must land predominantly in `sines`.
        let sine: Vec<f64> = (0..len)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f64 / fs as f64).sin())
            .collect();
        let mut dec = StnDecomposer::new(fs);
        let lat = dec.latency_samples();
        let b1 = dec.process(&sine);
        let t1 = dec.flush();
        let e_s: f64 = b1
            .sines
            .iter()
            .chain(&t1.sines)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let e_t: f64 = b1
            .transients
            .iter()
            .chain(&t1.transients)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let e_n: f64 = b1
            .noise
            .iter()
            .chain(&t1.noise)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let frac_s = e_s / (e_s + e_t + e_n);
        assert!(
            frac_s > 0.95,
            "pure sine should route >95% energy to sines, got {:.2}% (S={e_s:.3}, T={e_t:.3}, N={e_n:.3})",
            frac_s * 100.0
        );

        // 2. Isolated Kronecker impulse -> must land predominantly in `transients`.
        let mut imp = vec![0.0_f64; len];
        imp[24_000] = 1.0;
        let mut dec = StnDecomposer::new(fs);
        let b2 = dec.process(&imp);
        let t2 = dec.flush();
        let e_s: f64 = b2.sines.iter().chain(&t2.sines).map(|v| v * v).sum();
        let e_t: f64 = b2
            .transients
            .iter()
            .chain(&t2.transients)
            .map(|v| v * v)
            .sum();
        let e_n: f64 = b2.noise.iter().chain(&t2.noise).map(|v| v * v).sum();
        let frac_t = e_t / (e_s + e_t + e_n);
        assert!(
            frac_t > 0.85,
            "isolated impulse should route >85% energy to transients, got {:.2}% (S={e_s:.4}, T={e_t:.4}, N={e_n:.4})",
            frac_t * 100.0
        );

        // 3. White noise -> must land predominantly in `noise`.
        let mut noise = vec![0.0_f64; len];
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        for slot in &mut noise {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *slot = (((seed >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0) * 0.25;
        }
        let mut dec = StnDecomposer::new(fs);
        let b3 = dec.process(&noise);
        let t3 = dec.flush();
        let e_s: f64 = b3
            .sines
            .iter()
            .chain(&t3.sines)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let e_t: f64 = b3
            .transients
            .iter()
            .chain(&t3.transients)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let e_n: f64 = b3
            .noise
            .iter()
            .chain(&t3.noise)
            .skip(lat + 8000)
            .take(32000)
            .map(|v| v * v)
            .sum();
        let frac_n = e_n / (e_s + e_t + e_n);
        assert!(
            frac_n > 0.75,
            "white noise should route >75% energy to noise, got {:.2}% (S={e_s:.4}, T={e_t:.4}, N={e_n:.4})",
            frac_n * 100.0
        );
    }
}
