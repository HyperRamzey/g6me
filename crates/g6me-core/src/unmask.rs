//! Auditory-model perceptual unmasking dynamic EQ based on Moore, Glasberg,
//! Varathanathan & Schlittenlacher (Trends in Hearing 2016; JASA 2018;
//! standardized in ISO 532-2:2017 / ISO 532-3:2022).
//!
//! ## 100% Patent-Clean Perceptual Architecture
//!
//! Instead of proprietary filterbanks or naive multiband pumping, this module
//! implements the open ISO-standardized human auditory loudness model:
//!
//! 1. **Outer + middle ear combined transfer** (ISO 532-2 Tables A.1 & A.2).
//! 2. **75-band ERB-rate cochlear excitation pattern** (`1.75..=38.75 Cam`, `0.5 Cam`
//!    spacing) with level-dependent asymmetric `roex(p_l, p_u)` auditory filters.
//! 3. **ISO 532-2 specific loudness `N'(ERB)`** with low-level cochlear gain `G(f)`,
//!    threshold excitation `E_THRQ(f)`, and **Moore 2016 binaural inhibition**:
//!    `INHIB_ipsi = 2 / (1 + sech(N'_contra_sm / N'_ipsi_sm)^1.5978)` using
//!    `sigma = 1.8 Cam` Gaussian ERB smoothing (`N'_ipsi_inh = N'_ipsi / INHIB_ipsi`,
//!    yielding the exact `1.500x` diotic-to-monaural binaural summation ratio).
//! 4. **Dual temporal integration** (short-term `22 ms / 50 ms` vs. long-term
//!    `99 ms / 2000 ms`).
//! 5. **Perceptual unmasking target & minimum-phase FIR realization**: compares
//!    short-term specific loudness and spread-masking excess against an
//!    equal-information (pink-noise) ERB loudness reference, clamps bounded
//!    loudness-neutral gains to `±max_gain_db`, and synthesizes a strictly causal
//!    minimum-phase FIR via [`crate::fft::minimum_phase_spectrum`] with
//!    [`crate::fft::half_kaiser_tail_apply`] in zero-padded linear-convolution OLA.
//!
//! Pure deterministic `f64`, zero external crates, zero FMA contraction.

use std::f64::consts::PI;

use crate::fft::{half_kaiser_tail_apply, minimum_phase_spectrum, Complex, FftPlan};
use crate::stn::sr_scale;

/// Number of ERB bands (`1.75` to `38.75 Cam` in `0.5 Cam` steps, matching ISO 532-2).
pub const ERB_BANDS: usize = 75;
/// Starting ERB-rate number in Cam.
pub const ERB_START_CAM: f64 = 1.75;
/// ERB-rate step in Cam.
pub const ERB_STEP_CAM: f64 = 0.5;
/// Moore et al. (2016) binaural inhibition calibration constant `C` so a 1 kHz 40 dB SPL
/// diotic tone integrates to `~1.0 sone` after binaural inhibition (`1.500x` summation).
pub const MOORE_2016_C: f64 = 0.0631;
/// Moore et al. (2016) binaural inhibition exponent `p`.
pub const MOORE_2016_INHIB_P: f64 = 1.5978;
/// Moore et al. (2016) Gaussian ERB smoothing standard deviation (Cam).
pub const MOORE_2016_INHIB_SIGMA_CAM: f64 = 1.8;
/// Nominal mastering SPL calibration corresponding to `0 dBFS` sinusoidal peak.
pub const NOMINAL_FULL_SCALE_SPL_DB: f64 = 100.0;

/// Configuration for the auditory-model perceptual unmasking dynamic EQ.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnmaskCfg {
    /// Opt-in switch (`false` by default to preserve bit-exact default downmix).
    pub enable: bool,
    /// Recovery strength for masked auditory bands (`0.0..=1.0`, default `0.35`).
    pub recover: f64,
    /// Taming strength for overly dominant masking bands (`0.0..=1.0`, default `0.35`).
    pub tame: f64,
    /// Maximum per-band cut or boost magnitude in dB (`0.5..=12.0`, default `4.5`).
    pub max_gain_db: f64,
    /// Balance bias shifting target toward recovery (`+1.0`) or taming (`-1.0`).
    pub bias: f64,
    /// Spectral tilt of the equal-information ERB reference (`-1.0..=1.0`, `±3 dB`).
    pub brighten: f64,
}

impl Default for UnmaskCfg {
    fn default() -> Self {
        Self {
            enable: false,
            recover: 0.35,
            tame: 0.35,
            max_gain_db: 4.5,
            bias: 0.0,
            brighten: 0.0,
        }
    }
}

impl UnmaskCfg {
    pub fn validate(&self) -> Result<(), String> {
        if !self.enable {
            return Ok(());
        }
        if !(self.recover.is_finite() && (0.0..=1.0).contains(&self.recover)) {
            return Err(format!(
                "unmask recover must be in 0.0..=1.0 (got {})",
                self.recover
            ));
        }
        if !(self.tame.is_finite() && (0.0..=1.0).contains(&self.tame)) {
            return Err(format!(
                "unmask tame must be in 0.0..=1.0 (got {})",
                self.tame
            ));
        }
        if !(self.max_gain_db.is_finite() && (0.5..=12.0).contains(&self.max_gain_db)) {
            return Err(format!(
                "unmask max_gain_db must be in 0.5..=12.0 dB (got {})",
                self.max_gain_db
            ));
        }
        if !(self.bias.is_finite() && (-1.0..=1.0).contains(&self.bias)) {
            return Err(format!(
                "unmask bias must be in -1.0..=1.0 (got {})",
                self.bias
            ));
        }
        if !(self.brighten.is_finite() && (-1.0..=1.0).contains(&self.brighten)) {
            return Err(format!(
                "unmask brighten must be in -1.0..=1.0 (got {})",
                self.brighten
            ));
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        if !self.enable {
            return "off".to_string();
        }
        format!(
            "Moore-2016 ERB unmasking (recover {:.0}%, tame {:.0}%, max ±{:.1} dB, bias {:+.2}, brighten {:+.2})",
            self.recover * 100.0,
            self.tame * 100.0,
            self.max_gain_db,
            self.bias,
            self.brighten
        )
    }
}

/// Convert frequency in Hz to ERB-rate number (Cam) per Glasberg & Moore (1990) / ISO 532-2.
#[inline]
pub fn hz_to_cam(f_hz: f64) -> f64 {
    21.366 * (0.004368 * f_hz.max(0.0) + 1.0).log10()
}

/// Convert ERB-rate number (Cam) to frequency in Hz per ISO 532-2.
#[inline]
pub fn cam_to_hz(cam: f64) -> f64 {
    (10.0_f64.powf(cam / 21.366) - 1.0) / 0.004368
}

/// ISO 532-2 / Moore (2016) auditory filter sharpness `p_51(f)` at 51 dB SPL/ERB.
#[inline]
pub fn p51(f_hz: f64) -> f64 {
    let f = f_hz.max(1.0);
    4.0 * f / (24.673 * (0.004368 * f + 1.0))
}

fn interp_log_table(f_hz: f64, xs: &[f64], ys: &[f64]) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if f_hz <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if f_hz >= xs[last] {
        return ys[last];
    }
    let lf = f_hz.ln();
    for i in 0..last {
        if f_hz >= xs[i] && f_hz <= xs[i + 1] {
            let t = (lf - xs[i].ln()) / (xs[i + 1].ln() - xs[i].ln());
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    ys[last]
}

/// ISO 532-2 combined outer-ear (free-field frontal, Table A.1) minus middle-ear
/// attenuation (Table A.2) transfer function in dB.
pub fn outer_middle_ear_db(f_hz: f64) -> f64 {
    const FREQS: [f64; 31] = [
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
        500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
        6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    ];
    // Free-field outer ear gain (dB)
    const FREE_FIELD: [f64; 31] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.9, 1.4, 1.6, 1.7, 2.5, 2.7, 2.6,
        2.6, 3.2, 5.2, 6.6, 12.0, 16.8, 10.5, 4.7, 1.8, -1.6, -2.5, -5.0, -8.0,
    ];
    // Middle-ear attenuation (dB)
    const MIDDLE_EAR: [f64; 31] = [
        39.6, 32.0, 25.85, 21.4, 18.5, 15.9, 14.1, 12.4, 11.0, 9.6, 8.3, 7.4, 6.2, 4.8, 3.8, 3.3,
        2.9, 2.6, 2.6, 4.5, 5.4, 6.1, 8.5, 10.4, 7.3, 7.0, 6.6, 7.0, 9.2, 10.2, 12.2,
    ];
    let ff = interp_log_table(f_hz, &FREQS, &FREE_FIELD);
    let me = interp_log_table(f_hz, &FREQS, &MIDDLE_EAR);
    ff - me
}

/// Low-level cochlear amplifier gain `G(f)` in dB relative to 500 Hz (ISO 532-2).
fn g_low_db(f_hz: f64) -> f64 {
    if f_hz >= 500.0 {
        return 0.0;
    }
    const FREQS: [f64; 11] = [
        50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    ];
    const G_DB: [f64; 11] = [
        -22.5, -18.8, -15.3, -12.0, -9.2, -6.5, -4.4, -2.8, -1.5, -0.6, 0.0,
    ];
    interp_log_table(f_hz, &FREQS, &G_DB)
}

/// Precomputed ISO 532-2 / Moore et al. (2016) auditory model state.
#[derive(Debug, Clone)]
pub struct AuditoryModel {
    pub cams: [f64; ERB_BANDS],
    pub fc_hz: [f64; ERB_BANDS],
    pub p51_band: [f64; ERB_BANDS],
    pub g_lin: [f64; ERB_BANDS],
    pub alpha: [f64; ERB_BANDS],
    pub a_const: [f64; ERB_BANDS],
    pub a_pow_alpha: [f64; ERB_BANDS],
    pub e_thrq: [f64; ERB_BANDS],
    pub n_at_thrq: [f64; ERB_BANDS],
    /// Gaussian kernel for binaural inhibition (`sigma = 1.8 Cam`, normalized).
    gauss_kernel: Vec<f64>,
    gauss_half: usize,
    /// Equal-information (pink-noise) normalized reference specific loudness profile.
    pub pink_ref_norm: [f64; ERB_BANDS],
}

impl Default for AuditoryModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditoryModel {
    pub fn new() -> Self {
        let mut cams = [0.0; ERB_BANDS];
        let mut fc_hz = [0.0; ERB_BANDS];
        let mut p51_band = [0.0; ERB_BANDS];
        let mut g_lin = [0.0; ERB_BANDS];
        let mut alpha = [0.0; ERB_BANDS];
        let mut a_const = [0.0; ERB_BANDS];
        let mut a_pow_alpha = [0.0; ERB_BANDS];
        let mut e_thrq = [0.0; ERB_BANDS];
        let mut n_at_thrq = [0.0; ERB_BANDS];

        for i in 0..ERB_BANDS {
            let c = ERB_START_CAM + ERB_STEP_CAM * (i as f64);
            let f = cam_to_hz(c);
            cams[i] = c;
            fc_hz[i] = f;
            p51_band[i] = p51(f);
            let gdb = g_low_db(f);
            let g = 10.0_f64.powf(gdb / 10.0);
            let al = 0.2 + 0.0033 * (-gdb);
            let ac = 4.7568 + 0.21 * (-gdb);
            let eth = 2.31 * 10.0_f64.powf((3.5 - 0.1 * gdb) / 10.0) / 2.2387;
            let ac_pow = ac.powf(al);
            let n_th = MOORE_2016_C * ((g * eth + ac).powf(al) - ac_pow);

            g_lin[i] = g;
            alpha[i] = al;
            a_const[i] = ac;
            a_pow_alpha[i] = ac_pow;
            e_thrq[i] = eth;
            n_at_thrq[i] = n_th;
        }

        // Gaussian smoothing kernel across ERB bands for Moore 2016 binaural inhibition.
        let sigma_bins = MOORE_2016_INHIB_SIGMA_CAM / ERB_STEP_CAM;
        let gauss_half = (3.5 * sigma_bins).ceil() as usize;
        let mut gauss_kernel = Vec::with_capacity(2 * gauss_half + 1);
        let mut ksum = 0.0;
        for d in -(gauss_half as isize)..=(gauss_half as isize) {
            let x = (d as f64) / sigma_bins;
            let w = (-0.5 * x * x).exp();
            gauss_kernel.push(w);
            ksum += w;
        }
        for w in &mut gauss_kernel {
            *w /= ksum;
        }

        let mut model = Self {
            cams,
            fc_hz,
            p51_band,
            g_lin,
            alpha,
            a_const,
            a_pow_alpha,
            e_thrq,
            n_at_thrq,
            gauss_kernel,
            gauss_half,
            pink_ref_norm: [1.0 / (ERB_BANDS as f64); ERB_BANDS],
        };

        // Precompute equal-information (pink-noise at 65 dB SPL overall) reference specific loudness.
        let mut pink_e = [0.0_f64; ERB_BANDS];
        for (i, &f) in model.fc_hz.iter().enumerate() {
            let erb_w = 24.673 * (0.004368 * f + 1.0);
            // Pink noise spectral density ~ 1/f; power in 1 ERB ~ erb_w / f.
            let om_pow = 10.0_f64.powf(outer_middle_ear_db(f) / 10.0);
            pink_e[i] = 1e6 * (erb_w / f) * om_pow;
        }
        let mut n_pink = [0.0_f64; ERB_BANDS];
        for i in 0..ERB_BANDS {
            n_pink[i] = model.specific_loudness_monaural(i, pink_e[i]);
        }
        let sum_pink: f64 = n_pink.iter().sum::<f64>().max(1e-18);
        for (dst, &n) in model.pink_ref_norm.iter_mut().zip(n_pink.iter()) {
            *dst = n / sum_pink;
        }
        model
    }

    /// Monaural ISO 532-2 specific loudness `N'(i)` in sones/Cam from linear cochlear excitation `e`.
    #[inline]
    pub fn specific_loudness_monaural(&self, band: usize, e: f64) -> f64 {
        if e <= 1e-12 || !e.is_finite() {
            return 0.0;
        }
        let eth = self.e_thrq[band];
        if e < eth {
            let r = 2.0 * e / (e + eth);
            self.n_at_thrq[band] * r * r.sqrt()
        } else {
            let val = (self.g_lin[band] * e + self.a_const[band]).powf(self.alpha[band])
                - self.a_pow_alpha[band];
            (MOORE_2016_C * val).max(0.0)
        }
    }

    /// Apply Moore, Glasberg, Varathanathan & Schlittenlacher (2016) binaural inhibition
    /// to monaural specific loudness vectors `(n_l, n_r)`.
    ///
    /// Returns `(n_l_inhibited, n_r_inhibited, total_binaural_sones)`.
    ///
    /// Uses `INHIB_ipsi = 2 / (1 + sech(N'_contra_sm / N'_ipsi_sm)^1.5978)` and
    /// divides `N'_ipsi / INHIB_ipsi`, matching the exact `1.500x` diotic summation
    /// and `1.000x` monaural asymptote.
    pub fn apply_binaural_inhibition(
        &self,
        n_l: &[f64; ERB_BANDS],
        n_r: &[f64; ERB_BANDS],
        out_l: &mut [f64; ERB_BANDS],
        out_r: &mut [f64; ERB_BANDS],
    ) -> f64 {
        let mut sm_l = [0.0_f64; ERB_BANDS];
        let mut sm_r = [0.0_f64; ERB_BANDS];
        let hk = self.gauss_half as isize;
        for i in 0..ERB_BANDS {
            let mut acc_l = 0.0;
            let mut acc_r = 0.0;
            let mut wsum = 0.0;
            for (k_idx, &w) in self.gauss_kernel.iter().enumerate() {
                let j = (i as isize) + (k_idx as isize) - hk;
                if j >= 0 && (j as usize) < ERB_BANDS {
                    let ju = j as usize;
                    acc_l += w * n_l[ju];
                    acc_r += w * n_r[ju];
                    wsum += w;
                }
            }
            let inv = if wsum > 0.0 { 1.0 / wsum } else { 1.0 };
            sm_l[i] = acc_l * inv;
            sm_r[i] = acc_r * inv;
        }

        let mut total = 0.0;
        for i in 0..ERB_BANDS {
            let inhib_l = binaural_inhib_factor(sm_l[i], sm_r[i]);
            let inhib_r = binaural_inhib_factor(sm_r[i], sm_l[i]);
            out_l[i] = n_l[i] / inhib_l;
            out_r[i] = n_r[i] / inhib_r;
            total += (out_l[i] + out_r[i]) * ERB_STEP_CAM;
        }
        total
    }

    /// Compute cochlear excitation pattern (`E_total` with asymmetric level-dependent
    /// `roex(p_l, p_u)` plus narrow `E_self` local band energy) from one-sided FFT
    /// power bins (calibrated to linear SPL power units `10^(L_dB/10)`).
    pub fn compute_excitation(
        &self,
        bin_freqs: &[f64],
        bin_om_power: &[f64],
        e_total: &mut [f64; ERB_BANDS],
        e_self: &mut [f64; ERB_BANDS],
    ) {
        e_total.fill(0.0);
        e_self.fill(0.0);
        let p51_1k = p51(1000.0);

        // First aggregate FFT bins into the 75 ERB-band input levels (fast, O(nb)) so
        // the asymmetric level-dependent roex spread runs in O(75^2) = 5625 ops/hop.
        let mut band_in = [0.0_f64; ERB_BANDS];
        for (k, &p_om) in bin_om_power.iter().enumerate() {
            if p_om <= 0.0 {
                continue;
            }
            let f = bin_freqs[k];
            if f < 20.0 || f > 18_000.0 {
                continue;
            }
            let c = hz_to_cam(f);
            let idx_f = (c - ERB_START_CAM) / ERB_STEP_CAM;
            let idx = idx_f.round().clamp(0.0, (ERB_BANDS - 1) as f64) as usize;
            band_in[idx] += p_om;
        }

        // Level-dependent asymmetric roex(p_l, p_u) cochlear excitation spread.
        for src in 0..ERB_BANDS {
            let p_src = band_in[src];
            if p_src <= 1e-18 {
                continue;
            }
            e_self[src] += p_src;
            let f_src = self.fc_hz[src];
            let l_x = 10.0 * p_src.max(1e-18).log10();
            let p_ref = self.p51_band[src];
            let p_u = p_ref;
            let p_l = (p_ref - 0.35 * (p_ref / p51_1k) * (l_x - 51.0)).max(0.1 * p_ref);

            for (dst, e_dst) in e_total.iter_mut().enumerate() {
                if dst == src {
                    *e_dst += p_src;
                    continue;
                }
                let f_dst = self.fc_hz[dst];
                // ISO 532-2: filter centered at f_dst evaluated at component f_src:
                // upper skirt of filter affects frequencies below f_dst (upward spread of masking
                // comes from the shallow lower skirt p_l when f_src < f_dst!).
                let g_rel = (f_src - f_dst).abs() / f_dst;
                let p = if f_src < f_dst { p_l } else { p_u };
                let pg = p * g_rel;
                if pg < 20.0 {
                    let w = (1.0 + pg) * (-pg).exp();
                    *e_dst += p_src * w;
                }
            }
        }
    }
}

/// Moore et al. (2016) binaural inhibition denominator `INHIB_ipsi in [1.0, 2.0]`.
#[inline]
pub fn binaural_inhib_factor(n_ipsi_sm: f64, n_contra_sm: f64) -> f64 {
    if n_contra_sm <= 1e-18 {
        return 1.0;
    }
    if n_ipsi_sm <= 1e-18 {
        return 2.0;
    }
    let ratio = (n_contra_sm / n_ipsi_sm).min(40.0);
    let sech = 1.0 / ratio.cosh();
    let denom = 1.0 + sech.powf(MOORE_2016_INHIB_P);
    2.0 / denom
}

/// Streaming stereo Moore-2016 perceptual unmasking dynamic EQ using zero-padded
/// linear-convolution overlap-add with strictly causal minimum-phase FIR updates.
pub struct UnmaskEq {
    cfg: UnmaskCfg,
    sample_rate: u32,
    win_len: usize,
    hop: usize,
    n_fft: usize,
    nb: usize,
    plan: FftPlan,
    /// Analysis-only COLA window (`0.5 * periodic_hann` at 75% overlap, sum == 1.0).
    win_a: Vec<f64>,
    /// Precomputed bin frequencies (Hz) and outer+middle ear linear power weights.
    bin_freqs: Vec<f64>,
    bin_om_pow: Vec<f64>,
    /// Power calibration scale converting windowed FFT `|X|^2` to linear SPL power (`10^(L/10)`).
    spl_scale: f64,
    /// Auditory model (75 ERB bands + binaural inhibition).
    model: AuditoryModel,
    /// Per-band short-term (`22 ms / 50 ms`) and long-term (`99 ms / 2000 ms`) specific loudness.
    st_loudness: [f64; ERB_BANDS],
    lt_loudness: [f64; ERB_BANDS],
    /// Smoothed EQ gain in dB per ERB band.
    smooth_gain_db: [f64; ERB_BANDS],
    /// Temporal smoothing coefficients per hop.
    st_atk: f64,
    st_rel: f64,
    lt_atk: f64,
    lt_rel: f64,
    gain_smooth: f64,
    /// Input sliding frame buffers (length `win_len`).
    in_l: Vec<f64>,
    in_r: Vec<f64>,
    in_fill: usize,
    /// Output overlap-add accumulators (length `n_fft`).
    ola_l: Vec<f64>,
    ola_r: Vec<f64>,
    /// Reusable FFT & spectral scratch buffers.
    frame_buf: Vec<f64>,
    fft_scratch: Vec<Complex>,
    spec_l: Vec<Complex>,
    spec_r: Vec<Complex>,
    bin_pow_l: Vec<f64>,
    bin_pow_r: Vec<f64>,
    mag_target: Vec<f64>,
    fir_ir: Vec<f64>,
    h_min: Vec<Complex>,
    out_frame: Vec<f64>,
    /// Cached peak gain excursion for diagnostics/tests.
    max_abs_gain_db: f64,
    /// Frames consumed (for exact flush accounting).
    fed: usize,
    emitted: usize,
}

impl UnmaskEq {
    pub fn new(cfg: UnmaskCfg, sample_rate: u32) -> Self {
        let scale = sr_scale(sample_rate);
        let win_len = 1024 * scale;
        let hop = win_len / 4;
        let n_fft = 2 * win_len;
        let nb = n_fft / 2 + 1;
        let plan = FftPlan::new(n_fft);

        // Analysis-only 75%-overlap COLA window: `0.25 * (1 - cos(2*pi*i/W))` sums to 1.000000.
        let inv_w = 1.0 / (win_len as f64);
        let win_a: Vec<f64> = (0..win_len)
            .map(|i| 0.25 * (1.0 - (2.0 * PI * (i as f64) * inv_w).cos()))
            .collect();
        // Coherent sum of analysis window: `sum(win_a) = W / 4`.
        // For a unit-amplitude sine `sin(2*pi*f*t)`, single-sided FFT peak has `|X[k]| = 0.5 * sum(win_a)`.
        // So `|X[k]|^2 / (0.25 * (sum(win_a))^2) == 1.0` for 0 dBFS peak sine!
        let w_sum: f64 = win_a.iter().sum();
        let spl_ref_pow = 10.0_f64.powf(NOMINAL_FULL_SCALE_SPL_DB / 10.0);
        let spl_scale = spl_ref_pow / (0.25 * w_sum * w_sum);

        let df = (sample_rate as f64) / (n_fft as f64);
        let bin_freqs: Vec<f64> = (0..nb).map(|k| (k as f64) * df).collect();
        let bin_om_pow: Vec<f64> = bin_freqs
            .iter()
            .map(|&f| 10.0_f64.powf(outer_middle_ear_db(f.max(20.0)) / 10.0))
            .collect();

        let dt = (hop as f64) / (sample_rate.max(8_000) as f64);
        let coef = |tau_ms: f64| 1.0 - (-dt / (tau_ms * 1e-3)).exp();

        Self {
            cfg,
            sample_rate,
            win_len,
            hop,
            n_fft,
            nb,
            plan,
            win_a,
            bin_freqs,
            bin_om_pow,
            spl_scale,
            model: AuditoryModel::new(),
            st_loudness: [0.0; ERB_BANDS],
            lt_loudness: [0.0; ERB_BANDS],
            smooth_gain_db: [0.0; ERB_BANDS],
            st_atk: coef(22.0),
            st_rel: coef(50.0),
            lt_atk: coef(99.0),
            lt_rel: coef(2000.0),
            gain_smooth: coef(45.0),
            in_l: vec![0.0; win_len],
            in_r: vec![0.0; win_len],
            in_fill: 0,
            ola_l: vec![0.0; n_fft],
            ola_r: vec![0.0; n_fft],
            frame_buf: vec![0.0; n_fft],
            fft_scratch: vec![Complex::ZERO; n_fft],
            spec_l: vec![Complex::ZERO; nb],
            spec_r: vec![Complex::ZERO; nb],
            bin_pow_l: vec![0.0; nb],
            bin_pow_r: vec![0.0; nb],
            mag_target: vec![1.0; nb],
            fir_ir: vec![0.0; n_fft],
            h_min: vec![Complex::ONE; nb],
            out_frame: vec![0.0; n_fft],
            max_abs_gain_db: 0.0,
            fed: 0,
            emitted: 0,
        }
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Algorithmic streaming latency in samples (`win_len - hop`).
    #[inline]
    pub fn latency_samples(&self) -> usize {
        self.win_len - self.hop
    }

    /// Peak absolute per-band EQ gain applied so far (in dB).
    #[inline]
    pub fn max_abs_gain_db(&self) -> f64 {
        self.max_abs_gain_db
    }

    /// Process planar stereo slices `left` and `right`, appending output samples to `out_l`/`out_r`.
    pub fn process(
        &mut self,
        left: &[f64],
        right: &[f64],
        out_l: &mut Vec<f64>,
        out_r: &mut Vec<f64>,
    ) {
        debug_assert_eq!(left.len(), right.len());
        let len = left.len();
        if len == 0 {
            return;
        }
        self.fed += len;
        let mut idx = 0;
        let write_start = self.win_len - self.hop;
        while idx < len {
            let take = (self.hop - self.in_fill).min(len - idx);
            let dst = write_start + self.in_fill;
            self.in_l[dst..dst + take].copy_from_slice(&left[idx..idx + take]);
            self.in_r[dst..dst + take].copy_from_slice(&right[idx..idx + take]);
            self.in_fill += take;
            idx += take;

            if self.in_fill == self.hop {
                self.process_hop(out_l, out_r);
                self.in_l.copy_within(self.hop..self.win_len, 0);
                self.in_r.copy_within(self.hop..self.win_len, 0);
                self.in_fill = 0;
            }
        }
    }

    /// Drain remaining algorithmic latency samples so total emitted samples equal `fed + latency`.
    /// Note: emits exactly `fed - emitted` remaining samples aligned after skipping initial
    /// pre-roll, OR callers that keep latency simply drain `latency_samples()` zeros.
    /// Matches `StnDecomposer` / `EnhanceProcessor` convention: emits `fed + latency - emitted`
    /// total samples (i.e. feeds `latency` zeros and flushes partial hop).
    pub fn flush(&mut self, out_l: &mut Vec<f64>, out_r: &mut Vec<f64>) {
        let target_total = self.fed + self.latency_samples();
        let emitted_before = self.emitted;
        if emitted_before >= target_total {
            return;
        }
        let need_zeros = self.latency_samples() + self.hop;
        let zeros = vec![0.0_f64; need_zeros];
        let mut tmp_l = Vec::new();
        let mut tmp_r = Vec::new();
        let saved_fed = self.fed;
        self.process(&zeros, &zeros, &mut tmp_l, &mut tmp_r);
        self.fed = saved_fed;
        let take = target_total.saturating_sub(emitted_before).min(tmp_l.len());
        out_l.extend_from_slice(&tmp_l[..take]);
        out_r.extend_from_slice(&tmp_r[..take]);
        self.emitted = emitted_before + take;
    }

    /// Compute the bounded perceptual unmasking minimum-phase transfer function `H_min`
    /// for the current frame.
    fn update_min_phase_filter(&mut self) {
        if self.cfg.recover == 0.0 && self.cfg.tame == 0.0 && self.cfg.brighten == 0.0 {
            self.h_min.fill(Complex::ONE);
            return;
        }

        // Convert complex FFT bins into outer+middle-ear-weighted linear SPL power.
        for k in 0..self.nb {
            let s = self.spl_scale * self.bin_om_pow[k];
            self.bin_pow_l[k] = self.spec_l[k].norm_sq() * s;
            self.bin_pow_r[k] = self.spec_r[k].norm_sq() * s;
        }

        let mut e_tot_l = [0.0_f64; ERB_BANDS];
        let mut e_self_l = [0.0_f64; ERB_BANDS];
        let mut e_tot_r = [0.0_f64; ERB_BANDS];
        let mut e_self_r = [0.0_f64; ERB_BANDS];

        self.model.compute_excitation(
            &self.bin_freqs,
            &self.bin_pow_l,
            &mut e_tot_l,
            &mut e_self_l,
        );
        self.model.compute_excitation(
            &self.bin_freqs,
            &self.bin_pow_r,
            &mut e_tot_r,
            &mut e_self_r,
        );

        let mut n_mono_l = [0.0_f64; ERB_BANDS];
        let mut n_mono_r = [0.0_f64; ERB_BANDS];
        for i in 0..ERB_BANDS {
            n_mono_l[i] = self.model.specific_loudness_monaural(i, e_tot_l[i]);
            n_mono_r[i] = self.model.specific_loudness_monaural(i, e_tot_r[i]);
        }

        let mut n_inh_l = [0.0_f64; ERB_BANDS];
        let mut n_inh_r = [0.0_f64; ERB_BANDS];
        let total_sones =
            self.model
                .apply_binaural_inhibition(&n_mono_l, &n_mono_r, &mut n_inh_l, &mut n_inh_r);

        // Update short-term (22/50 ms) and long-term (99/2000 ms) binaural specific loudness.
        let mut st_sum = 0.0;
        let mut lt_sum = 0.0;
        for i in 0..ERB_BANDS {
            let n_inst = n_inh_l[i] + n_inh_r[i];
            let a_st = if n_inst > self.st_loudness[i] {
                self.st_atk
            } else {
                self.st_rel
            };
            self.st_loudness[i] += (n_inst - self.st_loudness[i]) * a_st;

            let a_lt = if n_inst > self.lt_loudness[i] {
                self.lt_atk
            } else {
                self.lt_rel
            };
            self.lt_loudness[i] += (n_inst - self.lt_loudness[i]) * a_lt;

            st_sum += self.st_loudness[i];
            lt_sum += self.lt_loudness[i];
        }

        // Gate silently when total loudness is near silence (< 0.02 sone) so noise floor isn't boosted.
        let mut target_db = [0.0_f64; ERB_BANDS];
        if total_sones > 0.02 && st_sum > 1e-6 {
            let ref_scale = 0.65 * st_sum + 0.35 * lt_sum.max(st_sum);
            let max_g = self.cfg.max_gain_db;
            let bias_db = self.cfg.bias * 1.5;

            let mut weighted_gain_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..ERB_BANDS {
                // Tilt equal-information pink reference by `brighten` (±3 dB across ERB span).
                let norm_pos = (i as f64) / ((ERB_BANDS - 1) as f64) - 0.5;
                let tilt_lin = 10.0_f64.powf((self.cfg.brighten * 3.0 * norm_pos) / 20.0);
                let n_target = (ref_scale * self.model.pink_ref_norm[i] * tilt_lin).max(1e-6);
                let n_cur = self.st_loudness[i];

                // Audibility gate: only boost bands that actually contain signal energy above threshold.
                let e_local = 0.5 * (e_self_l[i] + e_self_r[i]);
                let e_tot = 0.5 * (e_tot_l[i] + e_tot_r[i]);
                let audibility =
                    (e_local / (e_local + 25.0 * self.model.e_thrq[i])).clamp(0.0, 1.0);

                // Perceptual loudness deviation in dB (sone^2 ~ intensity^0.4 -> ~20 dB/decade).
                let ratio = (n_cur + 1e-6) / n_target;
                let dev_db = 20.0 * ratio.log10();

                // Spread-masking excess: when neighboring bands inject excitation `e_tot > e_local`
                // onto an audible band `i`, recovery lifts band `i` out of the masker's skirt.
                let spread_mask_db = if e_local > 1e-12 && e_tot > e_local {
                    (10.0 * (e_tot / e_local).log10()).min(18.0) * audibility
                } else {
                    0.0
                };

                let raw_db = if dev_db > 0.0 {
                    // Dominant masking band -> tame.
                    -self.cfg.tame * (0.45 * dev_db) + bias_db * 0.25
                } else {
                    // Masked / recessed band -> recover (gated by audibility).
                    self.cfg.recover * (0.40 * (-dev_db) * audibility + 0.35 * spread_mask_db)
                        + bias_db * 0.25
                };

                // Add pure brighten shelf component smoothly when requested.
                let b_db = self.cfg.brighten * 3.0 * norm_pos;
                let g = (raw_db + b_db).clamp(-max_g, max_g);
                target_db[i] = g;

                let w = n_cur + 1e-4;
                weighted_gain_sum += w * g;
                weight_sum += w;
            }

            // Loudness-neutral mean-zero removal (weighted by current specific loudness) unless
            // bias deliberately tilts net balance.
            let mean_db = if weight_sum > 0.0 {
                (weighted_gain_sum / weight_sum) * (1.0 - 0.35 * self.cfg.bias.abs())
            } else {
                0.0
            };
            for g in &mut target_db {
                *g = (*g - mean_db).clamp(-max_g, max_g);
            }
        }

        // Smooth ERB gains across adjacent bands (3-tap `[0.25, 0.5, 0.25]`) and in time.
        let mut spat_db = [0.0_f64; ERB_BANDS];
        for i in 0..ERB_BANDS {
            let prev = target_db[i.saturating_sub(1)];
            let next = target_db[(i + 1).min(ERB_BANDS - 1)];
            spat_db[i] = 0.25 * prev + 0.5 * target_db[i] + 0.25 * next;
        }
        for (smooth, &spat) in self.smooth_gain_db.iter_mut().zip(spat_db.iter()) {
            *smooth += (spat - *smooth) * self.gain_smooth;
            let a = smooth.abs();
            if a > self.max_abs_gain_db {
                self.max_abs_gain_db = a;
            }
        }

        // Interpolate smooth ERB dB curve onto linear FFT magnitude grid (DC and Nyquist tapered).
        self.mag_target[0] = 1.0;
        for k in 1..self.nb {
            let f = self.bin_freqs[k];
            let g_db = if f <= self.model.fc_hz[0] {
                let t = (f / self.model.fc_hz[0]).clamp(0.0, 1.0);
                t * self.smooth_gain_db[0]
            } else if f >= self.model.fc_hz[ERB_BANDS - 1] {
                let span =
                    (0.5 * (self.sample_rate as f64) - self.model.fc_hz[ERB_BANDS - 1]).max(100.0);
                let t = (1.0 - (f - self.model.fc_hz[ERB_BANDS - 1]) / span).clamp(0.0, 1.0);
                t * self.smooth_gain_db[ERB_BANDS - 1]
            } else {
                let c = hz_to_cam(f);
                let pos = ((c - ERB_START_CAM) / ERB_STEP_CAM).clamp(0.0, (ERB_BANDS - 1) as f64);
                let i0 = (pos.floor() as usize).min(ERB_BANDS - 2);
                let frac = pos - (i0 as f64);
                self.smooth_gain_db[i0]
                    + frac * (self.smooth_gain_db[i0 + 1] - self.smooth_gain_db[i0])
            };
            self.mag_target[k] = 10.0_f64.powf(g_db / 20.0);
        }

        // Synthesize strictly causal minimum-phase FIR of length <= win_len via Hilbert cepstrum,
        // apply half-Kaiser tail taper, and zero-pad back to n_fft so OLA is exact linear convolution.
        let h_raw = minimum_phase_spectrum(&self.mag_target, &self.plan);
        self.plan
            .irfft(&h_raw, &mut self.fft_scratch, &mut self.fir_ir);
        half_kaiser_tail_apply(&mut self.fir_ir[..self.win_len], self.win_len / 4, 8.0);
        for v in &mut self.fir_ir[self.win_len..] {
            *v = 0.0;
        }
        self.plan
            .rfft(&self.fir_ir, &mut self.fft_scratch, &mut self.h_min);
    }

    fn process_hop(&mut self, out_l: &mut Vec<f64>, out_r: &mut Vec<f64>) {
        // Window + zero-pad left channel into n_fft.
        for i in 0..self.win_len {
            self.frame_buf[i] = self.in_l[i] * self.win_a[i];
        }
        for i in self.win_len..self.n_fft {
            self.frame_buf[i] = 0.0;
        }
        self.plan
            .rfft(&self.frame_buf, &mut self.fft_scratch, &mut self.spec_l);

        // Window + zero-pad right channel into n_fft.
        for i in 0..self.win_len {
            self.frame_buf[i] = self.in_r[i] * self.win_a[i];
        }
        for i in self.win_len..self.n_fft {
            self.frame_buf[i] = 0.0;
        }
        self.plan
            .rfft(&self.frame_buf, &mut self.fft_scratch, &mut self.spec_r);

        // Compute Moore-2016 perceptual unmasking minimum-phase filter H_min.
        self.update_min_phase_filter();

        // Linear convolution in frequency domain + overlap-add for Left.
        for k in 0..self.nb {
            self.spec_l[k] = self.spec_l[k].mul(self.h_min[k]);
        }
        self.plan
            .irfft(&self.spec_l, &mut self.fft_scratch, &mut self.out_frame);
        for i in 0..self.n_fft {
            self.ola_l[i] += self.out_frame[i];
        }

        // Linear convolution in frequency domain + overlap-add for Right.
        for k in 0..self.nb {
            self.spec_r[k] = self.spec_r[k].mul(self.h_min[k]);
        }
        self.plan
            .irfft(&self.spec_r, &mut self.fft_scratch, &mut self.out_frame);
        for i in 0..self.n_fft {
            self.ola_r[i] += self.out_frame[i];
        }

        out_l.extend_from_slice(&self.ola_l[..self.hop]);
        out_r.extend_from_slice(&self.ola_r[..self.hop]);
        self.emitted += self.hop;

        self.ola_l.copy_within(self.hop..self.n_fft, 0);
        self.ola_r.copy_within(self.hop..self.n_fft, 0);
        for i in (self.n_fft - self.hop)..self.n_fft {
            self.ola_l[i] = 0.0;
            self.ola_r[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moore_2016_sone_calibration_and_binaural_inhibition_exact() {
        let model = AuditoryModel::new();
        // Pure 1 kHz tone at 40 dB SPL (free-field -> outer+middle ear = 0.0 dB at 1 kHz).
        let mut e_40 = [0.0_f64; ERB_BANDS];
        let p_1k = p51(1000.0);
        for (i, &f) in model.fc_hz.iter().enumerate() {
            let g_rel = (f - 1000.0).abs() / 1000.0;
            let p = if f < 1000.0 {
                (p_1k - 0.35 * (40.0 - 51.0)).max(0.1 * p_1k)
            } else {
                p_1k
            };
            let w = (1.0 + p * g_rel) * (-p * g_rel).exp();
            e_40[i] = 1e4 * w;
        }
        let mut n_mono = [0.0_f64; ERB_BANDS];
        for i in 0..ERB_BANDS {
            n_mono[i] = model.specific_loudness_monaural(i, e_40[i]);
        }
        let zeros = [0.0_f64; ERB_BANDS];
        let mut out_l = [0.0_f64; ERB_BANDS];
        let mut out_r = [0.0_f64; ERB_BANDS];

        let sones_monaural =
            model.apply_binaural_inhibition(&n_mono, &zeros, &mut out_l, &mut out_r);
        let sones_diotic =
            model.apply_binaural_inhibition(&n_mono, &n_mono, &mut out_l, &mut out_r);

        // ISO 532-2 / Moore 2016: 1 kHz 40 dB SPL diotic == ~1.00 sone.
        assert!(
            (sones_diotic - 1.0).abs() < 0.05,
            "1 kHz 40 dB SPL diotic expected ~1.0 sone, got {sones_diotic:.4}"
        );
        // Binaural inhibition ratio (diotic / monaural) == 1.500x (NOT 2.0x!).
        let bin_ratio = sones_diotic / sones_monaural;
        assert!(
            (bin_ratio - 1.5).abs() < 1e-3,
            "Moore 2016 binaural summation ratio expected 1.500, got {bin_ratio:.6}"
        );

        // +10 dB (50 dB SPL) must approximately double perceived loudness (~2.0x).
        let mut n_50 = [0.0_f64; ERB_BANDS];
        for i in 0..ERB_BANDS {
            n_50[i] = model.specific_loudness_monaural(i, e_40[i] * 10.0);
        }
        let sones_50 = model.apply_binaural_inhibition(&n_50, &n_50, &mut out_l, &mut out_r);
        let dbl = sones_50 / sones_diotic;
        assert!(
            (1.85..=2.15).contains(&dbl),
            "+10 dB loudness doubling ratio expected ~2.0, got {dbl:.3}"
        );
    }

    #[test]
    fn zero_recover_and_tame_is_exact_bypass() {
        let sr = 48_000;
        let cfg = UnmaskCfg {
            enable: true,
            recover: 0.0,
            tame: 0.0,
            max_gain_db: 4.5,
            bias: 0.0,
            brighten: 0.0,
        };
        let mut eq = UnmaskEq::new(cfg, sr);
        let n = 12_000;
        let left: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.5 * (2.0 * PI * 317.0 * t).sin() + 0.3 * (2.0 * PI * 2_411.0 * t).cos()
            })
            .collect();
        let right: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.4 * (2.0 * PI * 521.0 * t).sin() - 0.25 * (2.0 * PI * 4_129.0 * t).sin()
            })
            .collect();

        let mut out_l = Vec::new();
        let mut out_r = Vec::new();
        eq.process(&left, &right, &mut out_l, &mut out_r);
        eq.flush(&mut out_l, &mut out_r);

        let lat = eq.latency_samples();
        let mut max_err = 0.0_f64;
        for i in 0..n {
            max_err = max_err
                .max((out_l[i + lat] - left[i]).abs())
                .max((out_r[i + lat] - right[i]).abs());
        }
        assert!(
            max_err < 1e-12,
            "zero-gain unmask bypass error too large: {max_err:e}"
        );
    }

    #[test]
    fn unmasking_tames_dominant_masker_and_recovers_masked_probe() {
        let sr = 48_000;
        let cfg = UnmaskCfg {
            enable: true,
            recover: 0.85,
            tame: 0.85,
            max_gain_db: 6.0,
            bias: 0.0,
            brighten: 0.0,
        };
        let mut eq = UnmaskEq::new(cfg, sr);

        // 1.5 s signal: loud 1 kHz masker (-3 dBFS) + quiet 2.5 kHz masked probe (-27 dBFS).
        let n = 72_000;
        let f_mask = 1_000.0;
        let f_probe = 2_500.0;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.70 * (2.0 * PI * f_mask * t).sin() + 0.045 * (2.0 * PI * f_probe * t).sin()
            })
            .collect();

        let mut out_l = Vec::new();
        let mut out_r = Vec::new();
        eq.process(&sig, &sig, &mut out_l, &mut out_r);
        eq.flush(&mut out_l, &mut out_r);

        // Measure steady-state tone amplitudes via DFT projection on the final 0.5 s.
        let lat = eq.latency_samples();
        let start = n - 24_000;
        let tone_amp = |buf: &[f64], f: f64| -> f64 {
            let mut re = 0.0;
            let mut im = 0.0;
            let m = 24_000;
            for (k, &s) in buf[start..start + m].iter().enumerate() {
                let ang = 2.0 * PI * f * ((start + k) as f64) / (sr as f64);
                re += s * ang.cos();
                im += s * ang.sin();
            }
            2.0 * re.hypot(im) / (m as f64)
        };

        let in_mask = tone_amp(&sig, f_mask);
        let in_probe = tone_amp(&sig, f_probe);
        let out_mask = tone_amp(&out_l[lat..], f_mask);
        let out_probe = tone_amp(&out_l[lat..], f_probe);

        let ratio_in_db = 20.0 * (in_probe / in_mask).log10();
        let ratio_out_db = 20.0 * (out_probe / out_mask).log10();
        let unmask_gain_db = ratio_out_db - ratio_in_db;

        assert!(
            unmask_gain_db > 1.5,
            "expected probe-to-masker clarity improvement > 1.5 dB, got {unmask_gain_db:.2} dB \
             (in={ratio_in_db:.2}, out={ratio_out_db:.2}, mask_delta={:.2} dB, probe_delta={:.2} dB)",
            20.0 * (out_mask / in_mask).log10(),
            20.0 * (out_probe / in_probe).log10(),
        );
        assert!(
            eq.max_abs_gain_db() <= 6.0001,
            "unmask exceeded max_gain_db bound: {:.3} dB",
            eq.max_abs_gain_db()
        );
    }

    #[test]
    fn minimum_phase_fir_is_strictly_causal_and_chunk_invariant() {
        let sr = 48_000;
        let cfg = UnmaskCfg {
            enable: true,
            recover: 0.6,
            tame: 0.6,
            max_gain_db: 4.5,
            bias: 0.1,
            brighten: 0.2,
        };
        let n = 8_192;
        let left: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.5 * (2.0 * PI * 440.0 * t).sin() + 0.15 * (2.0 * PI * 3_100.0 * t).sin()
            })
            .collect();
        let right: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.45 * (2.0 * PI * 440.0 * t + 0.3).sin() + 0.12 * (2.0 * PI * 5_200.0 * t).cos()
            })
            .collect();

        // Single block
        let mut eq1 = UnmaskEq::new(cfg.clone(), sr);
        let mut l1 = Vec::new();
        let mut r1 = Vec::new();
        eq1.process(&left, &right, &mut l1, &mut r1);
        eq1.flush(&mut l1, &mut r1);

        // Verify FIR causality on eq1's final minimum-phase filter
        let front_e: f64 = eq1.fir_ir[..eq1.win_len].iter().map(|x| x * x).sum();
        let back_e: f64 = eq1.fir_ir[eq1.win_len..].iter().map(|x| x * x).sum();
        assert_eq!(back_e, 0.0, "zero-padded FIR tail must be strictly zero");
        assert!(front_e > 0.2, "FIR energy unexpectedly tiny: {front_e}");
        assert!(
            eq1.fir_ir[0].abs() > 0.3,
            "minimum-phase FIR should concentrate peak at tap 0, got {}",
            eq1.fir_ir[0]
        );

        // Odd chunk sizes
        let mut eq2 = UnmaskEq::new(cfg, sr);
        let mut l2 = Vec::new();
        let mut r2 = Vec::new();
        let mut pos = 0;
        let chunks = [1, 13, 97, 311, 512, 17, 1009];
        let mut ci = 0;
        while pos < n {
            let step = chunks[ci % chunks.len()].min(n - pos);
            eq2.process(
                &left[pos..pos + step],
                &right[pos..pos + step],
                &mut l2,
                &mut r2,
            );
            pos += step;
            ci += 1;
        }
        eq2.flush(&mut l2, &mut r2);

        assert_eq!(l1.len(), l2.len());
        let diff = l1
            .iter()
            .zip(l2.iter())
            .chain(r1.iter().zip(r2.iter()))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-13, "chunk-size invariance drift: {diff:e}");
    }
}
