//! Zilany–Bruce–Carney (JASA 2014) auditory periphery with the Bruce–Erfani–Zilany
//! (Hearing Research 2018) synapse, spike generator and neurogram.
//!
//! Signal path per characteristic frequency (CF):
//! 1. Human middle ear: three biquad sections (Pascal et al. 1998 fit, as used by the
//!    authors' reference model).
//! 2. Dual-resonance nonlinear basilar membrane: a third-order wideband gammatone control
//!    path drives the OHC Boltzmann nonlinearity (asymmetry 7), a 600 Hz low-pass and the
//!    `NLafterohc` time-constant law; the signal path is a tenth-order chirp filter (C1)
//!    whose poles are steered by that control path, in parallel with the C2 chirp filter,
//!    whose damping is fixed at the OHC-impaired value `1/bmTaumin` (the reference calls
//!    `C2ChirpFilt` with `1/ratiobm`).
//! 3. IHC: `NLogarithm` transduction (slope 0.1, asymmetry 3) for C1, square-law C2
//!    transduction, a 3 kHz seventh-order membrane low-pass and the cat cochlear delay.
//! 4. BEZ 2018 synapse: Eq. 1 IHC→PLA mapping with the Eq. 2–6 CF/SR factors and the
//!    `3·SR` constant, fractional Gaussian noise, the two power-law adaptation paths
//!    (5-pole approximation of α = 1.5e-6 s, β = 5e-4 s and α = 1e-2 s, β = 1e-1 s),
//!    adaptive mean redocking Eq. 7, four release sites with the Eq. 8 release integral,
//!    and spike generation with absolute plus release-rate-dependent relative
//!    refractoriness (Eq. 9).
//!
//! Provenance: the ZBC 2014 JASA paper text could not be retrieved; the middle-ear, BM
//! and IHC equations and parameters follow the authors' reference implementation
//! (`BEZ2018a_model/model_IHC_BEZ2018a.c`), which is the normative implementation of that
//! paper. The synapse follows the published BEZ 2018 equations and Table 1. The reference
//! distribution's `ffGn_2018a` generator shapes circulant-embedded fGn and resamples it
//! from a 0.1 s grid, whereas this port synthesizes full-band fGn by FFT at the synapse
//! rate with the same Hurst exponent 0.9 and the same per-class standard deviations
//! (1 /s low, 10 /s medium, SR/2 high); the two are statistically equivalent in mean and
//! variance but not in bandwidth. Stimuli are band-limited resampled to the model's
//! 100 kHz internal rate, as `generate_neurogram_BEZ2018a.m` requires.
//!
//! Input calibration: the reference model takes a pressure waveform in pascals. This port
//! keeps that convention explicitly through [`PASCAL_PER_UNIT`]: one input unit is one
//! pascal, so a full-scale sinusoid (peak amplitude 1.0) is [`FULL_SCALE_PEAK_SPL_DB`]
//! (93.98 dB SPL peak, 90.97 dB SPL RMS). Attenuate the signal (or scale the units) to
//! drive the model at lower levels; the IHC asymmetry uses the same 20 µPa reference.

use crate::fft::{i0, Complex, FftPlan};
use std::f64::consts::{PI, TAU};

/// Internal sampling rate of the reference model (Hz).
pub const ZBC_INTERNAL_FS: f64 = 100_000.0;
/// Reference pressure of the dB SPL scale (Pa), as used by the IHC asymmetry.
const REFERENCE_PRESSURE_PA: f64 = 20e-6;
/// Sound-pressure calibration of the model input: one input unit is this many
/// pascals. The reference model is driven by a pressure waveform in Pa, so the
/// natural convention is one unit per pascal.
pub const PASCAL_PER_UNIT: f64 = 1.0;
/// SPL of a full-scale input sample (peak amplitude [`PASCAL_PER_UNIT`]).
///
/// `20*log10(1 Pa / 20 uPa) = 93.9794 dB SPL` peak, i.e. `90.9691 dB SPL` RMS
/// for a sinusoid. The reference's `NLogarithm` uses the same 20 uPa reference,
/// so no further scaling is applied inside the model.
pub const FULL_SCALE_PEAK_SPL_DB: f64 = 93.979_400_086_720_37;
/// Number of synaptic release sites per fiber.
const N_SITES: usize = 4;
/// Short-term adaptation time constant of the mean redocking time (s).
const T_RD_TAU: f64 = 60.0e-3;
/// Resting mean redocking time (s).
const T_RD_REST: f64 = 14.0e-3;
/// Redocking-time jump per redocking event (s).
const T_RD_JUMP: f64 = 0.4e-3;

/// One auditory-nerve fiber of the model population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiberSpec {
    pub cf_hz: f64,
    /// Spontaneous discharge rate drawn from the BEZ 2018 Table 1 class distribution.
    pub spont_rate: f64,
    /// Absolute refractory period (s), uniform on [0.2085, 0.6915] ms.
    pub tabs_s: f64,
    /// Baseline mean relative refractory period (s), uniform on [0.131, 0.894] ms.
    pub trel_s: f64,
    pub seed: u64,
}

/// Deterministic xorshift64* generator (the model is stochastic but reproducible).
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on the open interval (0, 1).
    #[inline]
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Band-limited resampling with a Kaiser-windowed sinc kernel (32-tap support).
fn resample_to(x: &[f64], fs_in: f64, fs_out: f64) -> Vec<f64> {
    if (fs_in - fs_out).abs() < 1e-9 || x.is_empty() {
        return x.to_vec();
    }
    let n_out = ((x.len() as f64) * fs_out / fs_in).round() as usize;
    // Normalized cutoff (cycles per input sample), kept below the lower Nyquist.
    let fc = 0.95 * 0.5 * fs_out.min(fs_in) / fs_in;
    const HALF: isize = 16;
    let beta = 8.0;
    let i0b = i0(beta);
    let mut y = vec![0.0_f64; n_out];
    for (j, slot) in y.iter_mut().enumerate() {
        let p = j as f64 * fs_in / fs_out;
        let lo = (p - HALF as f64).ceil() as isize;
        let hi = (p + HALF as f64).floor() as isize;
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for i in lo..=hi {
            if i < 0 || i as usize >= x.len() {
                continue;
            }
            let d = p - i as f64;
            let sinc = if d.abs() < 1e-12 {
                1.0
            } else {
                let a = TAU * fc * d;
                a.sin() / a
            };
            let r = d / HALF as f64;
            let w = if r.abs() <= 1.0 {
                i0(beta * (1.0 - r * r).sqrt()) / i0b
            } else {
                0.0
            };
            let h = 2.0 * fc * sinc * w;
            acc += x[i as usize] * h;
            wsum += h;
        }
        *slot = if wsum.abs() > 1e-12 { acc / wsum } else { acc };
    }
    y
}

/// Fractional Gaussian noise with the published Hurst exponent, normalized to `sigma`.
///
/// FFT spectral synthesis: white Gaussian noise is shaped by `|f|^{(1-2H)/2}` on the
/// positive-frequency grid (DC removed), inverse transformed, and rescaled to exactly the
/// requested sample standard deviation. This reproduces the long-range correlation and
/// variance of the published fGn specification; it is not sample-identical to the
/// reference Paxson generator, which is not distributed with the model archive.
fn fractional_gaussian_noise(n: usize, hurst: f64, sigma: f64, rng: &mut Rng) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let nfft = n.next_power_of_two().max(16);
    let plan = FftPlan::new(nfft);
    let mut buf = vec![Complex::ZERO; nfft];
    for v in buf.iter_mut() {
        *v = Complex::new(rng.gaussian(), 0.0);
    }
    plan.forward(&mut buf);
    // Amplitude shaping: sqrt of the fGn power spectral density |f|^(1-2H).
    for (k, v) in buf.iter_mut().enumerate().skip(1) {
        let f = (k as f64 / nfft as f64).min(1.0 - k as f64 / nfft as f64);
        *v = v.scale(f.powf(0.5 * (1.0 - 2.0 * hurst)));
    }
    buf[0] = Complex::ZERO;
    plan.inverse(&mut buf);
    let mut out: Vec<f64> = buf.iter().take(n).map(|c| c.re).collect();
    let mean = out.iter().sum::<f64>() / n as f64;
    for v in out.iter_mut() {
        *v -= mean;
    }
    let var = out.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let g = if var > 0.0 { sigma / var.sqrt() } else { 0.0 };
    for v in out.iter_mut() {
        *v *= g;
    }
    out
}

/// Healthy auditory-nerve population used by `generate_neurogram_BEZ2018a.m`: one low-,
/// one medium- and one high-spontaneous-rate fiber per CF. `SR` is drawn from the
/// truncated Gaussian distributions of BEZ 2018 Table 1; `tabs` and `trel` share one
/// uniform draw, as specified in the paper.
pub fn population_for(cfs: &[f64]) -> Vec<[FiberSpec; 3]> {
    let mut rng = Rng::new(0x2018_5EED_0000_0001);
    let mut out = Vec::with_capacity(cfs.len());
    for (ch, &cf) in cfs.iter().enumerate() {
        let mut fibers = [FiberSpec {
            cf_hz: cf,
            spont_rate: 4.0,
            tabs_s: 0.4e-3,
            trel_s: 0.4e-3,
            seed: 1,
        }; 3];
        for (class, slot) in fibers.iter_mut().enumerate() {
            let (mean, std, lo, hi) = match class {
                0 => (0.1, 0.1, 1.0e-3, 0.2),
                1 => (4.0, 4.0, 0.2, 18.0),
                _ => (70.0, 30.0, 18.0, 180.0),
            };
            let mut spont = mean;
            for _ in 0..64 {
                let v = mean + std * rng.gaussian();
                if v >= lo && v <= hi {
                    spont = v;
                    break;
                }
            }
            let u = rng.uniform();
            *slot = FiberSpec {
                cf_hz: cf,
                spont_rate: spont,
                tabs_s: 0.2085e-3 + u * (0.6915e-3 - 0.2085e-3),
                trel_s: 0.1310e-3 + u * (0.8940e-3 - 0.1310e-3),
                seed: 0x9E37_79B9_7F4A_7C15 ^ ((ch as u64) << 32) ^ ((class as u64) << 8) ^ 0x5A17,
            };
        }
        out.push(fibers);
    }
    out
}

// ---------------------------------------------------------------------------
// Middle ear (human, Pascal et al. 1998 fit as used by the reference model)
// ---------------------------------------------------------------------------

struct MiddleEar {
    m11: f64,
    m12: f64,
    m13: f64,
    m14: f64,
    m15: f64,
    m16: f64,
    m21: f64,
    m22: f64,
    m23: f64,
    m24: f64,
    m25: f64,
    m26: f64,
    m31: f64,
    m32: f64,
    m33: f64,
    m34: f64,
    m35: f64,
    m36: f64,
    megainmax: f64,
    y1: [f64; 2],
    y2: [f64; 2],
    y3: [f64; 2],
    x1: f64,
    x2: f64,
}

impl MiddleEar {
    fn new(tdres: f64) -> Self {
        // Prewarping at 1 kHz, exactly as the reference implementation.
        let c = TAU * 1_000.0 / (PI * 1_000.0 * tdres).tan();
        let c2 = c * c;
        Self {
            m11: 1.0 / (c2 + 5.9761e3 * c + 2.5255e7),
            m12: -2.0 * c2 + 2.0 * 2.5255e7,
            m13: c2 - 5.9761e3 * c + 2.5255e7,
            m14: c2 + 5.6665e3 * c,
            m15: -2.0 * c2,
            m16: c2 - 5.6665e3 * c,
            m21: 1.0 / (c2 + 6.4255e3 * c + 1.3975e8),
            m22: -2.0 * c2 + 2.0 * 1.3975e8,
            m23: c2 - 6.4255e3 * c + 1.3975e8,
            m24: c2 + 5.8934e3 * c + 1.7926e8,
            m25: -2.0 * c2 + 2.0 * 1.7926e8,
            m26: c2 - 5.8934e3 * c + 1.7926e8,
            m31: 1.0 / (c2 + 2.4891e4 * c + 1.2700e9),
            m32: -2.0 * c2 + 2.0 * 1.2700e9,
            m33: c2 - 2.4891e4 * c + 1.2700e9,
            m34: 3.1137e3 * c + 6.9768e8,
            m35: 2.0 * 6.9768e8,
            m36: -3.1137e3 * c + 6.9768e8,
            megainmax: 2.0,
            y1: [0.0; 2],
            y2: [0.0; 2],
            y3: [0.0; 2],
            x1: 0.0,
            x2: 0.0,
        }
    }

    fn step(&mut self, x: f64) -> f64 {
        let y1 = self.m11
            * (self.m14 * x + self.m15 * self.x1 + self.m16 * self.x2
                - self.m12 * self.y1[0]
                - self.m13 * self.y1[1]);
        let y2 = self.m21
            * (self.m24 * y1 + self.m25 * self.y1[0] + self.m26 * self.y1[1]
                - self.m22 * self.y2[0]
                - self.m23 * self.y2[1]);
        let y3 = self.m31
            * (self.m34 * y2 + self.m35 * self.y2[0] + self.m36 * self.y2[1]
                - self.m32 * self.y3[0]
                - self.m33 * self.y3[1]);
        self.x2 = self.x1;
        self.x1 = x;
        self.y1 = [y1, self.y1[0]];
        self.y2 = [y2, self.y2[0]];
        self.y3 = [y3, self.y3[0]];
        y3 / self.megainmax
    }
}

// ---------------------------------------------------------------------------
// Tenth-order nonlinear chirp-gammatone filters C1 / C2
// ---------------------------------------------------------------------------

/// Returns the three independent pole pairs `(real, imag)` (p1, p3, p5).
fn chirp_poles(sigma0: f64, ipw: f64, ipb: f64, rpa: f64) -> ((f64, f64), (f64, f64), (f64, f64)) {
    let p1 = (-sigma0, ipw);
    let p5 = (p1.0 - rpa, p1.1 - ipb);
    let p3 = (0.5 * (p1.0 + p5.0), 0.5 * (p1.1 + p5.1));
    (p1, p3, p5)
}

struct ChirpFilter {
    cf_omega: f64,
    fs_bilinear: f64,
    sigma0: f64,
    ipw: f64,
    ipb: f64,
    rpa: f64,
    init_phase: f64,
    norm_gain: f64,
    input: [[f64; 3]; 6],
    output: [[f64; 3]; 6],
}

impl ChirpFilter {
    fn new(cf: f64, tdres: f64, taumax: f64) -> Self {
        let sigma0 = 1.0 / taumax;
        let ipw = 1.01 * cf * TAU - 50.0;
        let ipb = 0.2343 * TAU * cf - 1104.0;
        let rpa = 10f64.powf(0.9 * cf.log10() + 0.55) + 2000.0;
        let pzero = 10f64.powf(0.7 * cf.log10() + 1.6) + 500.0;
        let cf_omega = TAU * cf;
        let fs_bilinear = TAU * cf / (TAU * cf * tdres / 2.0).tan();
        let (p1, p3, p5) = chirp_poles(sigma0, ipw, ipb, rpa);
        let rzero = -pzero;
        let mut init_phase = 0.0;
        for &(pre, pim) in &[p1, p3, p5, p1, p5] {
            init_phase += (cf_omega / -rzero).atan()
                - ((cf_omega - pim) / -pre).atan()
                - ((cf_omega + pim) / -pre).atan();
        }
        let mut gain = 1.0;
        for &(pre, pim) in &[
            p1,
            (p1.0, -p1.1),
            p3,
            (p3.0, -p3.1),
            p5,
            (p5.0, -p5.1),
            p1,
            (p1.0, -p1.1),
            p5,
            (p5.0, -p5.1),
        ] {
            gain *= (cf_omega - pim).powi(2) + pre * pre;
        }
        let norm_gain = gain.sqrt() / ((cf_omega * cf_omega + rzero * rzero).sqrt()).powi(5);
        Self {
            cf_omega,
            fs_bilinear,
            sigma0,
            ipw,
            ipb,
            rpa,
            init_phase,
            norm_gain,
            input: [[0.0; 3]; 6],
            output: [[0.0; 3]; 6],
        }
    }

    fn step(&mut self, x: f64, sigma_shift: f64) -> f64 {
        let (p1, p3, p5) = chirp_poles(self.sigma0 + sigma_shift, self.ipw, self.ipb, self.rpa);
        let cfw = self.cf_omega;
        let mut phase = 0.0;
        for &(pre, pim) in &[p1, p3, p5, p1, p5] {
            phase -= ((cfw - pim) / -pre).atan() + ((cfw + pim) / -pre).atan();
        }
        let rzero = -cfw / ((self.init_phase - phase) / 5.0).tan();
        self.input[0] = [x, self.input[0][0], self.input[0][1]];
        for (i, &(pre, pim)) in [p1, p3, p5, p1, p5].iter().enumerate() {
            let temp = (self.fs_bilinear - pre).powi(2) + pim * pim;
            let dy = self.input[i][0] * (self.fs_bilinear - rzero)
                - 2.0 * rzero * self.input[i][1]
                - (self.fs_bilinear + rzero) * self.input[i][2]
                + 2.0
                    * self.output[i][0]
                    * (self.fs_bilinear * self.fs_bilinear - pre * pre - pim * pim)
                - self.output[i][1] * ((self.fs_bilinear + pre).powi(2) + pim * pim);
            let dy = dy / temp;
            self.input[i + 1] = [dy, self.output[i][0], self.output[i][1]];
            self.output[i] = [dy, self.output[i][0], self.output[i][1]];
        }
        self.output[4][0] * self.norm_gain / 4.0
    }
}

// ---------------------------------------------------------------------------
// Control-path third-order nonlinear gammatone, low-pass ladders, nonlinearities
// ---------------------------------------------------------------------------

struct WbGammaTone {
    phase: f64,
    f: [Complex; 4],
    fl: [Complex; 4],
}

impl WbGammaTone {
    fn new() -> Self {
        Self {
            phase: 0.0,
            f: [Complex::ZERO; 4],
            fl: [Complex::ZERO; 4],
        }
    }

    fn step(
        &mut self,
        x: f64,
        tdres: f64,
        centerfreq: f64,
        tau: f64,
        gain: f64,
        order: usize,
    ) -> f64 {
        self.phase -= TAU * centerfreq * tdres;
        let dtmp = tau * 2.0 / tdres;
        let c1 = (dtmp - 1.0) / (dtmp + 1.0);
        let c2 = 1.0 / (dtmp + 1.0);
        let rot = Complex::from_polar(1.0, self.phase);
        let unrot = Complex::from_polar(1.0, -self.phase);
        self.f[0] = rot.scale(x);
        for j in 1..=order {
            self.f[j] = self.f[j - 1]
                .add(self.fl[j - 1])
                .scale(c2 * gain)
                .add(self.fl[j].scale(c1));
        }
        let out = unrot.mul(self.f[order]).re;
        self.fl = self.f;
        out
    }
}

/// Cascaded one-pole all-pole low-pass of order `n` (bilinear, matched to the reference).
struct LadderLowpass {
    order: usize,
    y: Vec<f64>,
    yl: Vec<f64>,
}

impl LadderLowpass {
    fn new(order: usize) -> Self {
        Self {
            order,
            y: vec![0.0; order + 1],
            yl: vec![0.0; order + 1],
        }
    }

    fn step(&mut self, x: f64, tdres: f64, fc: f64, gain: f64) -> f64 {
        let c = 2.0 / tdres;
        let c1 = (c - TAU * fc) / (c + TAU * fc);
        let c2 = TAU * fc / (TAU * fc + c);
        self.y[0] = x * gain;
        for i in 0..self.order {
            self.y[i + 1] = c1 * self.yl[i + 1] + c2 * (self.y[i] + self.yl[i]);
        }
        self.yl.copy_from_slice(&self.y);
        self.y[self.order]
    }
}

/// OHC Boltzmann nonlinearity, normalized to a maximum of one.
fn boltzman(x: f64, asym: f64, s0: f64, s1: f64, x1: f64) -> f64 {
    let shift = 1.0 / (1.0 + asym);
    let x0 = s0 * (((1.0 / shift - 1.0) / (1.0 + (x1 / s1).exp())).ln());
    let out1 = 1.0 / (1.0 + (-(x - x0) / s0).exp() * (1.0 + (-(x - x1) / s1).exp())) - shift;
    out1 / (1.0 - shift)
}

/// Time-constant law after the OHC low-pass (the `NLafterohc` function).
fn nl_after_ohc(x: f64, taumin: f64, taumax: f64, asym: f64) -> f64 {
    let r = taumin / taumax;
    let min_r = if r < 0.05 { 0.5 * r } else { 0.05 };
    let dc = (asym - 1.0) / (asym + 1.0) / 2.0 - min_r;
    let r1 = r - min_r;
    let s0 = -dc / (r1 / (1.0 - min_r)).ln();
    let out = taumax * (min_r + (1.0 - min_r) * (-x.abs() / s0).exp());
    out.clamp(taumin, taumax)
}

/// IHC logarithmic transduction function.
fn n_logarithm(x: f64, slope: f64, asym: f64) -> f64 {
    const CORNER: f64 = 80.0;
    let strength = 20.0e6 / 10f64.powf(CORNER / 20.0);
    let mut xx = (1.0 + strength * x.abs()).ln() * slope;
    if x < 0.0 {
        let splx = 20.0 * (-x / REFERENCE_PRESSURE_PA).log10();
        let asym_t = asym - (asym - 1.0) / (1.0 + (splx / 5.0).exp());
        xx = -xx / asym_t;
    }
    xx
}

/// Gain and integer group delay of the control-path low-pass at the shifted CF.
fn gain_groupdelay(tdres: f64, centerfreq: f64, cf: f64, tau: f64) -> (f64, i32) {
    let tmpcos = (TAU * (centerfreq - cf) * tdres).cos();
    let dtmp2 = tau * 2.0 / tdres;
    let c1 = (dtmp2 - 1.0) / (dtmp2 + 1.0);
    let c2 = 1.0 / (dtmp2 + 1.0);
    let tmp1 = 1.0 + c1 * c1 - 2.0 * c1 * tmpcos;
    let tmp2 = 2.0 * c2 * c2 * (1.0 + tmpcos);
    let wb_gain = (tmp1 / tmp2).sqrt();
    let grd = (0.5 - (c1 * c1 - c1 * tmpcos) / (1.0 + c1 * c1 - 2.0 * c1 * tmpcos)).floor() as i32;
    (wb_gain, grd)
}

// ---------------------------------------------------------------------------
// Complete IHC signal path for one CF
// ---------------------------------------------------------------------------

struct CochlearChannel {
    cf: f64,
    tdres: f64,
    me: MiddleEar,
    wb: WbGammaTone,
    ohc_lp: LadderLowpass,
    ihc_lp: LadderLowpass,
    c1: ChirpFilter,
    c2: ChirpFilter,
    centerfreq: f64,
    bm_taumax: f64,
    bm_taumin: f64,
    /// Pole shift of the C2 filter: the reference passes `1/ratiobm` as `fcohc`,
    /// so C2 runs with the OHC completely impaired (`p1.x = -1/bmTaumin`).
    c2_rsigma: f64,
    tau_wb_max: f64,
    tau_wb_min: f64,
    wbgain_init: f64,
    tauwb: f64,
    wbgain: f64,
    last_tmpgain: f64,
    tmpgain: Vec<f64>,
    delay_samples: usize,
}

impl CochlearChannel {
    fn new(cf: f64, tdres: f64) -> Self {
        let bmplace = (35.0 / 2.1) * (1.0 + cf / 165.4).log10();
        let centerfreq = 165.4 * (10f64.powf((bmplace + 1.2) / (35.0 / 2.1)) - 1.0);
        let gain = (26.0 * ((2.2 * (cf / 0.6e3).log10() + 0.15).tanh() + 1.0)).clamp(15.0, 60.0);
        // Human Q10 from Shera et al. (PNAS 2002), as used for species == 2.
        let q10 = (cf / 1000.0).powf(0.3) * 12.7 * 0.505 + 0.2085;
        let bw = cf / q10;
        let taumax = 2.0 / (TAU * bw);
        let ratio = 10f64.powf(-gain / (20.0 * 3.0));
        let taumin = taumax * ratio;
        let bm_ratio = 10f64.powf(-gain / (20.0 * 2.5));
        let bm_taumax = taumax / 0.7;
        let bm_taumin = bm_taumax * bm_ratio;
        let tau_wb_max = taumin + 0.2 * (taumax - taumin);
        let tau_wb_min = tau_wb_max / taumax * taumin;
        let (wbgain_init, _) = gain_groupdelay(tdres, centerfreq, cf, tau_wb_max);
        let delay = 3.0 * (-(11.9 * (0.80 + cf / 456.0).log10()) / 12.5).exp() * 1e-3;
        let delay_samples = (delay / tdres).ceil().max(0.0) as usize;
        Self {
            cf,
            tdres,
            me: MiddleEar::new(tdres),
            wb: WbGammaTone::new(),
            ohc_lp: LadderLowpass::new(2),
            ihc_lp: LadderLowpass::new(7),
            c1: ChirpFilter::new(cf, tdres, bm_taumax),
            c2: ChirpFilter::new(cf, tdres, bm_taumax),
            centerfreq,
            bm_taumax,
            bm_taumin,
            c2_rsigma: 1.0 / bm_taumin - 1.0 / bm_taumax,
            tau_wb_max,
            tau_wb_min,
            wbgain_init,
            tauwb: tau_wb_max,
            wbgain: wbgain_init,
            last_tmpgain: wbgain_init,
            tmpgain: Vec::new(),
            delay_samples,
        }
    }

    /// Returns the delayed IHC receptor potential at the internal 100 kHz rate.
    fn process(&mut self, x: &[f64]) -> Vec<f64> {
        let n = x.len();
        self.tmpgain = vec![0.0; n];
        self.tauwb = self.tau_wb_max;
        self.wbgain = self.wbgain_init;
        self.last_tmpgain = self.wbgain_init;
        let mut out = vec![0.0_f64; n];
        for (k, &px) in x.iter().enumerate() {
            let meout = self.me.step(px);
            let wbout1 = self.wb.step(
                meout,
                self.tdres,
                self.centerfreq,
                self.tauwb,
                self.wbgain,
                3,
            );
            let wbout = (self.tauwb / self.tau_wb_max).powi(3) * wbout1 * 10e3 * (cf_max(self.cf));
            let ohc_nonlin = boltzman(wbout, 7.0, 12.0, 5.0, 5.0);
            let ohcout = self.ohc_lp.step(ohc_nonlin, self.tdres, 600.0, 1.0);
            let tauc1 = nl_after_ohc(ohcout, self.bm_taumin, self.bm_taumax, 7.0);
            let rsigma = 1.0 / tauc1 - 1.0 / self.bm_taumax;
            let tauwb_new = self.tau_wb_max
                + (tauc1 - self.bm_taumax) * (self.tau_wb_max - self.tau_wb_min)
                    / (self.bm_taumax - self.bm_taumin);
            let (wb_gain, grd) = gain_groupdelay(self.tdres, self.centerfreq, self.cf, tauwb_new);
            let idx = k as i64 + grd as i64;
            if idx >= 0 && (idx as usize) < n {
                self.tmpgain[idx as usize] = wb_gain;
            }
            if self.tmpgain[k] == 0.0 {
                self.tmpgain[k] = self.last_tmpgain;
            }
            self.wbgain = self.tmpgain[k];
            self.last_tmpgain = self.wbgain;
            self.tauwb = tauwb_new;

            let c1 = self.c1.step(meout, rsigma);
            // C2 is the parallel path with the OHC completely impaired (reference
            // calls C2ChirpFilt with `1/ratiobm`), so its damping is 1/bmTaumin.
            let c2 = self.c2.step(meout, self.c2_rsigma);
            let c1_v = n_logarithm(c1, 0.1, 3.0);
            let c2_v = -n_logarithm(c2 * c2.abs() * self.cf / 10.0 * self.cf / 2e3, 0.2, 1.0);
            out[k] = self.ihc_lp.step(c1_v + c2_v, self.tdres, 3000.0, 1.0);
        }
        let mut delayed = vec![0.0_f64; n];
        for i in self.delay_samples..n {
            delayed[i] = out[i - self.delay_samples];
        }
        delayed
    }
}

#[inline]
fn cf_max(cf: f64) -> f64 {
    (cf / 5e3).max(1.0)
}

// ---------------------------------------------------------------------------
// BEZ 2018 synapse: mapping, power-law adaptation, decimation
// ---------------------------------------------------------------------------

/// Second-order IIR section in the direct form used by the reference PLA filters:
/// `y[k] = b0 x[k] + b1 x[k-1] + b2 x[k-2] - a1 y[k-1] - a2 y[k-2]`.
#[derive(Clone, Copy)]
struct Iir2 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Iir2 {
    fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Five-pole approximation of the two power-law adaptation paths (BEZ 2018, Sec. 2).
struct PowerLawAdaptation {
    n1: Iir2,
    n2: Iir2,
    n3: Iir2,
    m1: Iir2,
    m2: Iir2,
    m3: Iir2,
    m4: Iir2,
    m5: Iir2,
    i1: f64,
    i2: f64,
}

impl PowerLawAdaptation {
    fn new() -> Self {
        Self {
            n1: Iir2::new(
                1.0e-3,
                -0.994466986569624e-3,
                2.347e-12,
                -1.992127932802320,
                0.992140616993846,
            ),
            n2: Iir2::new(
                1.0,
                -1.997855276593802,
                0.997855827934345,
                -1.999195329360981,
                0.999195402928777,
            ),
            n3: Iir2::new(
                1.0,
                0.798261718184977,
                0.199131619874064,
                0.798261718183851,
                0.199131619873480,
            ),
            m1: Iir2::new(
                0.2,
                -0.0346984006638638,
                3.45967592e-8,
                -0.491115852967412,
                0.055050209956838,
            ),
            m2: Iir2::new(
                1.0,
                -0.803462163297112,
                0.154962026341513,
                -1.084520302502860,
                0.288760329320566,
            ),
            m3: Iir2::new(
                1.0,
                -1.416084732997016,
                0.496615555008723,
                -1.588427084535629,
                0.628138993662508,
            ),
            m4: Iir2::new(
                1.0,
                -1.830362725074550,
                0.836399964176882,
                -1.886287488516458,
                0.888972875389923,
            ),
            m5: Iir2::new(
                1.0,
                -1.983165053215032,
                0.983193027347456,
                -1.989549282714008,
                0.989558985673023,
            ),
            i1: 0.0,
            i2: 0.0,
        }
    }

    /// One 10 kHz sample of the two-path power-law adaptation, including the fGn drive.
    #[inline]
    fn step(&mut self, x: f64, noise: f64) -> f64 {
        let sout1 = (x + noise - 0.15 * self.i1).max(0.0);
        let sout2 = (x - 1000.0 * self.i2).max(0.0);
        self.i2 = self.n3.step(self.n2.step(self.n1.step(sout2)));
        self.i1 = self.m5.step(
            self.m4
                .step(self.m3.step(self.m2.step(self.m1.step(sout1)))),
        );
        sout1 + sout2
    }
}

/// Eq. 2–6 CF and SR factors of the IHC→PLA mapping.
fn synapse_cf_factor(cf: f64, spont: f64) -> (f64, f64) {
    let cfslope = spont.powf(0.19) * 10f64.powf(-0.87);
    let cfconst = 0.1 * spont.log10().powi(2) + 0.56 * spont.log10() - 0.84;
    let cfsat = 10f64.powf(cfslope * 8965.5 / 1e3 + cfconst);
    let cf_factor = cfsat.min(10f64.powf(cfslope * cf / 1e3 + cfconst)) * 2.0;
    let mult = (4.3 - cf / 5000.0).max(2.95 * (1.5 - spont / 100.0).max(1.0));
    (cf_factor, mult)
}

/// Eq. 1 mapping with the preferred shifted-softplus exponential-like function.
fn map_ihc(v: f64, cf_factor: f64, mult: f64) -> f64 {
    const P1SP: f64 = 0.00172;
    const P2SP: f64 = 1165.0;
    let mut ex = P1SP * ((1.0 + (P2SP * v).exp()).ln() - 2f64.ln());
    if !ex.is_finite() {
        ex = P1SP * (P2SP * v - 2f64.ln());
    }
    let mag = 10f64.powf(0.9 * (ex.abs() * cf_factor).log10() + mult);
    if v < 0.0 {
        -mag
    } else {
        mag
    }
}

/// Taps of the synapse decimation filter (Kaiser-windowed sinc, cutoff 4.5 kHz).
const DECIMATE_10_TAPS: usize = 63;
/// Group delay of the symmetric decimation filter, in 100 kHz samples.
const DECIMATE_10_HALF: usize = DECIMATE_10_TAPS / 2;

/// Anti-aliased 10:1 decimation (Kaiser-windowed sinc, 63 taps, cutoff 4.5 kHz).
///
/// Output `j` is the filter centred on input sample `10*j + DECIMATE_10_HALF`,
/// i.e. it carries a group delay of [`DECIMATE_10_HALF`] input samples that the
/// caller must compensate ([`upsample_10`] does so).
fn decimate_10(x: &[f64]) -> Vec<f64> {
    const N: usize = DECIMATE_10_TAPS;
    const HALF: usize = DECIMATE_10_HALF;
    let fc = 4500.0 / ZBC_INTERNAL_FS;
    let mut h = vec![0.0_f64; N];
    let mut sum = 0.0;
    for (i, v) in h.iter_mut().enumerate() {
        let k = i as f64 - HALF as f64;
        let sinc = if k.abs() < 1e-12 {
            1.0
        } else {
            (TAU * fc * k).sin() / (PI * k)
        };
        let r = k / HALF as f64;
        let w = if r.abs() <= 1.0 {
            i0(5.0 * (1.0 - r * r).sqrt()) / i0(5.0)
        } else {
            0.0
        };
        *v = 2.0 * fc * sinc * w;
        sum += *v;
    }
    for v in h.iter_mut() {
        *v /= sum;
    }
    let n_out = if x.len() > N {
        (x.len() - N) / 10 + 1
    } else {
        0
    };
    (0..n_out)
        .map(|j| {
            let c = j * 10 + HALF;
            (0..N).map(|i| h[i] * x[c + i - HALF]).sum()
        })
        .collect()
}

/// Linear 10 → 100 kHz interpolation of the decimated synapse drive.
///
/// `syn10[j]` is centred on input sample `10*j + DECIMATE_10_HALF` (see
/// [`decimate_10`]), so the read index subtracts that group delay: `out[i]`
/// interpolates the drive at `pad` sample `i + delaypoint`, matching the
/// reference's delay-compensated `resample()`.
fn upsample_10(syn10: &[f64], n: usize, delaypoint: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    let last = syn10.last().copied().unwrap_or(0.0);
    for (i, slot) in out.iter_mut().enumerate() {
        // delaypoint >= 375 for the supported CF range (<= 20 kHz), so the
        // saturating subtraction only matters for unsupported CFs.
        let idx = ((i + delaypoint) as isize - DECIMATE_10_HALF as isize).max(0) as usize;
        let z = idx / 10;
        let b = idx % 10;
        *slot = if z + 1 < syn10.len() {
            syn10[z] + (syn10[z + 1] - syn10[z]) * (b as f64 / 10.0)
        } else {
            last
        };
    }
    out
}

/// Synapse release rate (`Sout`, sp/s) at the internal 100 kHz rate for one fiber.
fn synapse_output(vihc: &[f64], spec: &FiberSpec, tdres: f64) -> Vec<f64> {
    let n = vihc.len();
    if n == 0 {
        return Vec::new();
    }
    let (cf_factor, mult) = synapse_cf_factor(spec.cf_hz, spec.spont_rate);
    let delaypoint = (7500.0 / (spec.cf_hz / 1e3)).floor() as usize;
    let mut mapped = Vec::with_capacity(n);
    for &v in vihc {
        mapped.push(map_ihc(v, cf_factor, mult) + 3.0 * spec.spont_rate);
    }
    // Pre-history and tail padding as in the reference Synapse(): the first
    // `delaypoint` samples repeat the mapped drive at the start of the signal.
    //
    // The reference's tail loop (`model_Synapse_BEZ2018a.c`, third `powerLawIn`
    // loop) adds `3*spont` *per sample*, which is a rate ramp, not a resting
    // value: at CF = 1 kHz with SR = 70 it reaches 3.15e6 sp/s, and the
    // decimation filter's 31-sample tail sees the first samples of that ramp in
    // the last output samples. The tail is padding, not physiology, so it holds
    // the final mapped drive (edge replication) instead; in silence that is the
    // resting `3*spont` offset.
    let total = n + 3 * delaypoint;
    let mut pad = vec![mapped[0]; total];
    pad[delaypoint..delaypoint + n].copy_from_slice(&mapped);
    let tail = pad[delaypoint + n - 1];
    for slot in &mut pad[delaypoint + n..] {
        *slot = tail;
    }
    let samp = decimate_10(&pad);
    if samp.is_empty() {
        return vec![0.0; n];
    }
    let m = ((n + 2 * delaypoint) as f64 * tdres * 10_000.0).floor() as usize;
    let m = m.min(samp.len());
    // Published fGn standard deviation per spontaneous-rate class.
    let sigma = if spec.spont_rate < 0.2 {
        1.0
    } else if spec.spont_rate < 18.0 {
        10.0
    } else {
        spec.spont_rate / 2.0
    };
    let mut noise_rng = Rng::new(spec.seed ^ 0xD1B5_4A32_D192_ED03);
    let noise = fractional_gaussian_noise(m, 0.9, sigma, &mut noise_rng);
    let mut pla = PowerLawAdaptation::new();
    let mut syn10 = Vec::with_capacity(m);
    for k in 0..m {
        syn10.push(pla.step(samp[k], noise[k]));
    }
    // Upsample 10 -> 100 kHz by linear interpolation, then advance past the
    // pre-history. `upsample_10` compensates the decimator's 31-sample group delay.
    upsample_10(&syn10, n, delaypoint)
}

/// Sample-index clock for one release site's redocking interval.
///
/// The reference compares `floor(one_site_redock/tdres)` with
/// `floor(elapsed/tdres)`. Accumulating `elapsed` in f64 and re-dividing drifts
/// across integer boundaries (413 additions of 1e-5 s floor to 412, so the
/// comparison also holds on the next sample), which delays the `t_rd_jump` and
/// can fire it twice on consecutive samples. Counting whole samples reproduces
/// the intended comparison exactly while leaving the RNG draw order untouched.
#[derive(Clone, Copy, Default)]
struct RedockClock {
    /// Redocking interval in whole internal samples.
    redock_step: i64,
    /// Whole samples elapsed since the last release.
    step: i64,
}

impl RedockClock {
    /// Arm the clock for a freshly drawn redocking time (s).
    #[inline]
    fn arm(&mut self, redock_s: f64, tdres: f64) {
        self.redock_step = (redock_s / tdres).floor() as i64;
        self.step = 0;
    }

    /// Advance one internal sample. Returns true on the single sample at which
    /// the floored redocking interval is reached.
    #[inline]
    fn tick(&mut self) -> bool {
        let fired = self.step == self.redock_step;
        self.step += 1;
        fired
    }
}

/// BEZ 2018 spike generator: four release sites, adaptive redocking Eq. 7, absolute and
/// release-rate-dependent relative refractoriness Eq. 9.
fn generate_spikes(synout: &[f64], spec: &FiberSpec, tdres: f64, rng: &mut Rng) -> Vec<f64> {
    let n = synout.len();
    if n == 0 {
        return Vec::new();
    }
    let t_rd_init = T_RD_REST + 0.02e-3 * spec.spont_rate - T_RD_JUMP;

    let mut one_site_redock = [0.0_f64; N_SITES];
    let mut redock = [RedockClock::default(); N_SITES];
    let mut pre_release = [0.0_f64; N_SITES];
    for i in 0..N_SITES {
        one_site_redock[i] = -t_rd_init * rng.uniform().ln();
        redock[i].arm(one_site_redock[i], tdres);
        let guess = (N_SITES as f64 / synout[0].max(0.1) + t_rd_init) * rng.uniform().ln() / tdres;
        pre_release[i] = (-(n as f64)).max(guess.ceil());
    }
    pre_release.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k_init = pre_release[0] as i64;

    let mut previous_release = [0.0_f64; N_SITES];
    for i in 0..N_SITES {
        previous_release[i] = pre_release[i] * tdres;
    }
    let mut elapsed = [0.0_f64; N_SITES];
    let mut xsum = [0.0_f64; N_SITES];
    let mut unit_interval = [0i64; N_SITES];
    let mut current_refractory = k_init as f64 * tdres;
    let mut previous_redocking = t_rd_init;
    let mut current_redocking = t_rd_init;
    let mut t_rd_decay = true;
    let mut rd_first = false;
    let last = n - 1;

    let mut spikes = Vec::new();
    let mut k = k_init;
    while k < n as i64 {
        let ki = (k.max(0) as usize).min(last);
        for site in 0..N_SITES {
            if (k as f64) > pre_release[site] {
                if redock[site].tick() {
                    current_redocking = previous_redocking + T_RD_JUMP;
                    previous_redocking = current_redocking;
                    t_rd_decay = false;
                    rd_first = true;
                }
                elapsed[site] += tdres;
            }
            if elapsed[site] >= one_site_redock[site] {
                xsum[site] += synout[ki] / N_SITES as f64;
            }
            if xsum[site] >= unit_interval[site] as f64 && (k as f64) >= pre_release[site] {
                one_site_redock[site] = -current_redocking * rng.uniform().ln();
                redock[site].arm(one_site_redock[site], tdres);
                let release = previous_release[site] + elapsed[site];
                elapsed[site] = 0.0;
                if release >= current_refractory {
                    if release >= 0.0 {
                        spikes.push(release);
                    }
                    let trel_k = (spec.trel_s * 100.0 / synout[ki]).min(spec.trel_s);
                    let tref = spec.tabs_s - trel_k * rng.uniform().ln();
                    current_refractory = release + tref;
                }
                previous_release[site] = release;
                xsum[site] = 0.0;
                unit_interval[site] = (-rng.uniform().ln() / tdres) as i64;
            }
        }
        if t_rd_decay && rd_first {
            current_redocking =
                previous_redocking - (tdres / T_RD_TAU) * (previous_redocking - T_RD_REST);
            previous_redocking = current_redocking;
        } else {
            t_rd_decay = true;
        }
        k += 1;
    }
    spikes
}

fn fiber_spike_times_100k(x100: &[f64], spec: FiberSpec) -> Vec<f64> {
    let tdres = 1.0 / ZBC_INTERNAL_FS;
    let mut channel = CochlearChannel::new(spec.cf_hz, tdres);
    let vihc = channel.process(x100);
    let synout = synapse_output(&vihc, &spec, tdres);
    let mut rng = Rng::new(spec.seed ^ 0xF6E1_0000_0000_0000);
    generate_spikes(&synout, &spec, tdres, &mut rng)
}

/// Convert model input samples to pascals ([`PASCAL_PER_UNIT`]).
#[inline]
fn to_pascals(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| v * PASCAL_PER_UNIT).collect()
}

/// Spike times (s) of one model fiber in response to `x`.
///
/// `x` is the pressure waveform in units of [`PASCAL_PER_UNIT`] (one unit = 1 Pa,
/// so full scale is 93.98 dB SPL peak).
pub fn fiber_spike_times(x: &[f64], sample_rate: u32, spec: FiberSpec) -> Vec<f64> {
    let x_pa = to_pascals(x);
    let x100 = resample_to(&x_pa, sample_rate.max(8_000) as f64, ZBC_INTERNAL_FS);
    fiber_spike_times_100k(&x100, spec)
}

/// Binned auditory-nerve neurogram (`sp/s`) from the three-fiber population per CF.
///
/// Each CF contributes one low-, one medium- and one high-spontaneous-rate fiber; the
/// binned spike counts of the three fibers are averaged, matching the healthy population
/// used by `generate_neurogram_BEZ2018a.m`. `x` is the pressure waveform in units of
/// [`PASCAL_PER_UNIT`] (one unit = 1 Pa).
pub fn rate_neurogram(x: &[f64], sample_rate: u32, cfs: &[f64], frame_ms: f64) -> Vec<Vec<f64>> {
    let fs = sample_rate.max(8_000) as f64;
    let frame_len = ((frame_ms * 1e-3 * fs).round() as usize).max(16);
    let n_frames = x.len() / frame_len;
    let mut rates = vec![vec![0.0_f64; n_frames]; cfs.len()];
    if n_frames == 0 || x.is_empty() {
        return rates;
    }
    let frame_s = frame_len as f64 / fs;
    let x_pa = to_pascals(x);
    let x100 = resample_to(&x_pa, fs, ZBC_INTERNAL_FS);
    let population = population_for(cfs);
    let tdres = 1.0 / ZBC_INTERNAL_FS;
    for (ch, fibers) in population.iter().enumerate() {
        let mut channel = CochlearChannel::new(cfs[ch], tdres);
        let vihc = channel.process(&x100);
        for spec in fibers {
            let synout = synapse_output(&vihc, spec, tdres);
            let mut rng = Rng::new(spec.seed ^ 0xF6E1_0000_0000_0000);
            for t in generate_spikes(&synout, spec, tdres, &mut rng) {
                let f = (t / frame_s).floor();
                if f >= 0.0 && (f as usize) < n_frames {
                    rates[ch][f as usize] += 1.0 / (3.0 * frame_s);
                }
            }
        }
    }
    rates
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnostic: dump the Rust cochlear output for a 1 kHz, 1 Pa tone so it can
    /// be compared against the compiled `model_IHC_BEZ2018a.c` reference.
    #[test]
    fn dump_ihc_reference_comparison() {
        let n = 5000;
        let tdres = 1.0 / ZBC_INTERNAL_FS;
        let x: Vec<f64> = (0..n)
            .map(|k| (6.28318530717959 * 1000.0 * k as f64 * tdres).sin())
            .collect();
        let mut channel = CochlearChannel::new(1000.0, tdres);
        let vihc = channel.process(&x);
        let mut s = String::new();
        for v in vihc {
            s.push_str(&format!("{v:.17e}\n"));
        }
        std::fs::write("/home/sanjxz/.cache/g6me-papers/zbc/rust_vihc_1k.txt", s)
            .expect("write diagnostic dump");
    }

    /// Diagnostic: dump the Rust C1 chirp filter for a unit sine at CF, rsigma = 0.
    #[test]
    fn dump_chirp_reference_comparison() {
        let n = 5000;
        let tdres: f64 = 1.0 / ZBC_INTERNAL_FS;
        let cf: f64 = 1000.0;
        let gain = (26.0 * ((2.2 * (cf / 0.6e3).log10() + 0.15).tanh() + 1.0)).clamp(15.0, 60.0);
        let q10 = (cf / 1000.0).powf(0.3) * 12.7 * 0.505 + 0.2085;
        let taumax = 2.0 / (TAU * (cf / q10));
        let bm_taumax = taumax / 0.7;
        let mut filter = ChirpFilter::new(cf, tdres, bm_taumax);
        let mut s = String::new();
        for k in 0..n {
            let x = (6.28318530717959 * cf * k as f64 * tdres).sin();
            let y = filter.step(x, 0.0);
            s.push_str(&format!("{y:.17e}\n"));
        }
        std::fs::write("/home/sanjxz/.cache/g6me-papers/zbc/rust_chirp_1k.txt", s)
            .expect("write chirp dump");
        let _ = gain;
    }

    /// Diagnostic: trace the Rust cochlear loop against the instrumented C.
    #[test]
    fn dump_ihc_trace() {
        let n = 5000;
        let tdres = 1.0 / ZBC_INTERNAL_FS;
        let mut ch = CochlearChannel::new(1000.0, tdres);
        ch.tmpgain = vec![0.0; n];
        ch.tauwb = ch.tau_wb_max;
        ch.wbgain = ch.wbgain_init;
        ch.last_tmpgain = ch.wbgain_init;
        let mut out = vec![0.0_f64; n];
        for k in 0..n {
            let px = (6.28318530717959 * 1000.0 * k as f64 * tdres).sin();
            let meout = ch.me.step(px);
            let wbout1 = ch.wb.step(meout, tdres, ch.centerfreq, ch.tauwb, ch.wbgain, 3);
            let wbout = (ch.tauwb / ch.tau_wb_max).powi(3) * wbout1 * 10e3 * cf_max(ch.cf);
            let ohc_nonlin = boltzman(wbout, 7.0, 12.0, 5.0, 5.0);
            let ohcout = ch.ohc_lp.step(ohc_nonlin, tdres, 600.0, 1.0);
            let tauc1 = nl_after_ohc(ohcout, ch.bm_taumin, ch.bm_taumax, 7.0);
            let rsigma = 1.0 / tauc1 - 1.0 / ch.bm_taumax;
            let tauwb_new = ch.tau_wb_max
                + (tauc1 - ch.bm_taumax) * (ch.tau_wb_max - ch.tau_wb_min)
                    / (ch.bm_taumax - ch.bm_taumin);
            let (wb_gain, grd) = gain_groupdelay(tdres, ch.centerfreq, ch.cf, tauwb_new);
            let idx = k as i64 + grd as i64;
            if idx >= 0 && (idx as usize) < n {
                ch.tmpgain[idx as usize] = wb_gain;
            }
            if ch.tmpgain[k] == 0.0 {
                ch.tmpgain[k] = ch.last_tmpgain;
            }
            ch.wbgain = ch.tmpgain[k];
            ch.last_tmpgain = ch.wbgain;
            ch.tauwb = tauwb_new;
            let c1 = ch.c1.step(meout, rsigma);
            let c2 = ch.c2.step(meout, ch.c2_rsigma);
            let c1_v = n_logarithm(c1, 0.1, 3.0);
            let c2_v = -n_logarithm(c2 * c2.abs() * ch.cf / 10.0 * ch.cf / 2e3, 0.2, 1.0);
            out[k] = ch.ihc_lp.step(c1_v + c2_v, tdres, 3000.0, 1.0);
            if matches!(k, 200 | 201 | 202 | 300 | 301 | 302 | 1000 | 3000 | 4999) {
                eprintln!(
                    "RDBG {k} meout={meout:.17e} wbout1={wbout1:.17e} wbout={wbout:.17e} ohcnl={ohc_nonlin:.17e} ohcout={ohcout:.17e} tmptauc1={tauc1:.17e} rsigma={rsigma:.17e} wbgain={wb_gain:.17e} grd={grd} tauwb={tauwb_new:.17e} c1={c1:.17e} c2={c2:.17e} c1v={c1_v:.17e} c2v={c2_v:.17e} vihc={:.17e}",
                    out[k]
                );
            }
        }
    }

    /// Diagnostic: measure resting rates, tone-driven rates, latency and chirp gain.
    #[test]
    fn diagnostics_measure() {
        let tdres = 1.0 / ZBC_INTERNAL_FS;
        let spec = FiberSpec {
            cf_hz: 1000.0,
            spont_rate: 70.0,
            tabs_s: 0.45e-3,
            trel_s: 0.5e-3,
            seed: 0x5EED_0001,
        };
        // A. Resting synapse drive from silence.
        let n = 100_000;
        let vihc = vec![0.0_f64; n];
        let syn = synapse_output(&vihc, &spec, tdres);
        let mean_all = syn.iter().sum::<f64>() / n as f64;
        let tail = &syn[n - 5_000..];
        let tm = tail.iter().sum::<f64>() / 5_000.0;
        let ts = (tail.iter().map(|v| (v - tm) * (v - tm)).sum::<f64>() / 5_000.0).sqrt();
        let max = syn.iter().cloned().fold(f64::MIN, f64::max);
        eprintln!(
            "REST mean_all={mean_all:.6} tail_mean={tm:.6} tail_std={ts:.6} max={max:.6}"
        );
        // B. Constant-drive discharge rate.
        for drive in [105.348_363_458_386_9, 210.0, 315.0] {
            let synout = vec![drive; 200_000];
            let mut rng = Rng::new(0xABCD_EF01);
            let spikes = generate_spikes(&synout, &spec, tdres, &mut rng);
            eprintln!(
                "DC drive={drive:.3} rate={:.3}",
                spikes.len() as f64 / 2.0
            );
        }
        // C. Full-scale 1 kHz tone (94 dB SPL peak) rates per SR class.
        let fs = 100_000.0;
        let m = 100_000;
        let tone: Vec<f64> = (0..m)
            .map(|i| (TAU * 1000.0 * i as f64 / fs).sin())
            .collect();
        for sr in [0.1, 4.0, 70.0] {
            let mut s = spec;
            s.spont_rate = sr;
            let spikes = fiber_spike_times(&tone, fs as u32, s);
            let count = spikes.iter().filter(|&&t| (0.1..0.9).contains(&t)).count();
            eprintln!("TONE sr={sr} rate={:.2}", count as f64 / 0.8);
        }
        // D. Decimator + interpolation latency on a 1 kHz sine.
        let delaypoint = 7_500;
        let np = 4_000;
        let total = np + 3 * delaypoint;
        let pad: Vec<f64> = (0..total)
            .map(|k| (TAU * 100.0 * (k as f64 - delaypoint as f64) * tdres).sin())
            .collect();
        let syn10 = decimate_10(&pad);
        let out = upsample_10(&syn10, np, delaypoint);
        let mut best = (0i32, f64::MIN);
        let mut corrs = Vec::new();
        for lag in -50i32..=50 {
            let mut acc = 0.0;
            for i in 200..np - 200 {
                let j = i as i32 - lag;
                let r = (TAU * 100.0 * j as f64 * tdres).sin();
                acc += out[i] * r;
            }
            corrs.push((lag, acc));
            if acc > best.1 {
                best = (lag, acc);
            }
        }
        eprintln!("LAG best={} corr={:.6}", best.0, best.1);
        for lag in [0, 1, 30, 31, 32] {
            let c = corrs.iter().find(|(l, _)| *l == lag).unwrap().1;
            eprintln!("LAGC lag={lag} corr={c:.6}");
        }
        let w = TAU * 100.0 * tdres;
        let (mut sa, mut sb) = (0.0_f64, 0.0_f64);
        for i in 400..np - 400 {
            sa += out[i] * (w * i as f64).sin();
            sb += out[i] * (w * i as f64).cos();
        }
        let phase = sb.atan2(sa);
        eprintln!("PHASE lag_samples={:.6}", -phase / w);
        // E. Chirp gain at CF.
        let cf: f64 = 1000.0;
        let gain = (26.0 * ((2.2 * (cf / 0.6e3).log10() + 0.15).tanh() + 1.0)).clamp(15.0, 60.0);
        let q10 = (cf / 1000.0).powf(0.3) * 12.7 * 0.505 + 0.2085;
        let taumax = 2.0 / (TAU * (cf / q10));
        let bm_taumax = taumax / 0.7;
        let mut chirp = ChirpFilter::new(cf, tdres, bm_taumax);
        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        for k in 0..5_000 {
            let y = chirp.step((TAU * cf * k as f64 * tdres).sin(), 0.0);
            if k >= 3_000 {
                mn = mn.min(y);
                mx = mx.max(y);
            }
        }
        eprintln!("CHIRP gain={:.9} gain_const={gain:.6}", (mx - mn) / 2.0);
    }
}
