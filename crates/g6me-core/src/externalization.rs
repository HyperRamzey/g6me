//! Auditory externalization model of Baumgartner & Majdak (2021), "Decision making in
//! auditory externalization perception: model predictions for static conditions",
//! bioRxiv 2020.04.30.068817 (CC-BY-NC-ND), Eqs. 1–10.
//!
//! Eight cues are evaluated against an internal template:
//! `MSS`, `ISS`, `MSSD`, `ISSD`, `ITIT`, `IC`, `MI`, and `ITSD` (the interaural
//! time-spectral deviation, defined by analogy with `ISSD` on the frequency-dependent
//! ITD profile). The default decision stage is the weighted-sum model (WSM) with the
//! published weights 0.6 `MSS` / 0.4 `ISS` and 0 for the remaining cues; the dynamic
//! LTA / MTA / WTA selection strategies are also available.
//!
//! The score is the paper's template-similarity rating (Eqs. 4, 7 and 10), not a
//! room-benefit metric: the model has no reverberation term, and the paper restricts
//! its predictions to static conditions, leaving extension to naturally reverberant
//! stimuli open. A room tier is therefore not predicted to outscore an anechoic tier
//! merely by adding reflections.
//!
//! **Template substitution.** The paper compares against listener-specific HRTF/BRIR
//! templates that are not available here. This implementation synthesizes the template
//! from our own physical head/torso model (`scattering::snowman_hrtf`) at the
//! caller-supplied lateral angle (default 30°, the front-left production slot),
//! elevation 0, range 1 m. The target's broadband ITD enters only the ITIT cue; it does
//! not steer the template. This is the documented substitution for the paper's
//! listener-specific DTF template.

use crate::fft::{erb_hz, Complex, FftPlan};
use crate::scattering::{snowman_hrtf, SnowmanGeom};
use std::f64::consts::PI;

/// Rating scale range `e_range` of Eq. 10 (paper's normalized 1–5 scale).
pub const EXT_E_RANGE: f64 = 3.78;
/// Rating scale offset `e_offset` of Eq. 10 (rating of an in-head image).
pub const EXT_E_OFFSET: f64 = 1.0;
/// Interaural level-difference just-noticeable difference (Eqs. 1, 4, 5).
pub const EXT_ILD_JND_DB: f64 = 1.0;
/// Interaural time-difference just-noticeable difference (Eq. 5), microseconds.
pub const EXT_ITD_JND_US: f64 = 20.0;
/// Binaural-weighting steepness `\Phi` of Eq. 9, degrees.
pub const EXT_PHI_DEG: f64 = 13.0;
/// Assumed presentation level of the stationarity approximation, dB SPL.
pub const EXT_SPL_DB: f64 = 70.0;
/// Within-band audibility floor, dB SPL.
pub const EXT_AUDIBLE_FLOOR_DB: f64 = 20.0;
/// Speed of sound used for the template HRTF, m/s.
pub const EXT_SPEED_OF_SOUND: f64 = 343.0;
/// Template source distance, m (the production renderer's default 2 m).
pub const EXT_TEMPLATE_DISTANCE_M: f64 = 2.0;

/// WSM cue weights from Baumgartner & Majdak (2021) Fig. 4b, ordered
/// `[MSS, ISS, MSSD, ISSD, ITIT, IC, MI, ITSD]`. The paper reports 0.6 / 0.4 for the
/// two spectral-shape cues and < 0.005 (set to zero) for the rest.
pub const EXT_WEIGHTS: [f64; 8] = [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Cue sensitivities `S_cue` of Eq. 10, digitized from the paper's Fig. 3a (the values
/// are only published as a logarithmic colour map; these are the geometric means across
/// the five modelled experiments). ITSD is absent from Fig. 3a and uses the ISSD value.
pub const EXT_SENSITIVITY: [f64; 8] = [
    0.1195, 0.0867, 0.0628, 0.1406, 0.0550, 0.0186, 0.003_72, 0.1406,
];

/// Decision stage of Fig. 1d/e.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecisionStrategy {
    /// Static weighted sum (paper's best strategy).
    #[default]
    WeightedSum,
    /// Winner-takes-all: largest cue rating (minimum expectation error).
    WinnerTakesAll,
    /// Median-takes-all: median cue rating.
    MedianTakesAll,
    /// Loser-takes-all: smallest cue rating (maximum expectation error).
    LoserTakesAll,
}

/// Single-cue expectation errors `d_cue` (dimensionless, Eqs. 1–9).
#[derive(Debug, Clone, Copy, Default)]
pub struct CueErrors {
    pub mss: f64,
    pub iss: f64,
    pub mssd: f64,
    pub issd: f64,
    pub itit: f64,
    pub ic: f64,
    pub mi: f64,
    pub itsd: f64,
}

impl CueErrors {
    fn as_array(self) -> [f64; 8] {
        [
            self.mss, self.iss, self.mssd, self.issd, self.itit, self.ic, self.mi, self.itsd,
        ]
    }
}

/// Output of the Baumgartner & Majdak (2021) externalization model.
#[derive(Debug, Clone, Copy)]
pub struct ExternalizationReport {
    /// Normalized externalization score in `[0, 1]` (0 = inside the head, 1 = fully
    /// externalized at the template's reference rating).
    pub score: f64,
    /// Normalized monaural spectral-shape rating (`MSS`).
    pub monaural_cue: f64,
    /// Normalized interaural spectral-shape rating (`ISS`).
    pub binaural_cue: f64,
    /// Broadband interaural coherence of the target (`IC`).
    pub iacc: f64,
    /// Early-to-direct reflection energy ratio (dB), diagnostic only.
    pub early_refl_db: f64,
    /// Raw expectation errors of all eight cues.
    pub cue_errors: CueErrors,
    /// Raw Eq. 10 ratings `e_cue`.
    pub cue_ratings: [f64; 8],
    /// Lateral angle of the template, degrees (left positive).
    pub azimuth_deg: f64,
}

/// Default template azimuth: the front-left speaker of the 12-speaker ring (30° left),
/// matching the 30–50° source azimuths of the paper's experiments.
pub const EXT_DEFAULT_AZIMUTH_DEG: f64 = 30.0;

/// Evaluate the externalization model with the published weighted-sum decision stage
/// and the default 30° template direction.
pub fn baumgartner_majdak_externalization(
    ir_l: &[f64],
    ir_r: &[f64],
    sample_rate: u32,
) -> ExternalizationReport {
    externalization_with_strategy(ir_l, ir_r, sample_rate, DecisionStrategy::WeightedSum)
}

/// Evaluate the externalization model with an explicit decision strategy.
pub fn externalization_with_strategy(
    ir_l: &[f64],
    ir_r: &[f64],
    sample_rate: u32,
    strategy: DecisionStrategy,
) -> ExternalizationReport {
    externalization_with_azimuth(ir_l, ir_r, sample_rate, EXT_DEFAULT_AZIMUTH_DEG, strategy)
}

/// Evaluate the externalization model against a template at an explicit lateral angle.
pub fn externalization_with_azimuth(
    ir_l: &[f64],
    ir_r: &[f64],
    sample_rate: u32,
    azimuth_deg: f64,
    strategy: DecisionStrategy,
) -> ExternalizationReport {
    let n = ir_l.len().min(ir_r.len());
    if n < 32 {
        return ExternalizationReport {
            score: 0.0,
            monaural_cue: 0.0,
            binaural_cue: 0.0,
            iacc: 1.0,
            early_refl_db: -120.0,
            cue_errors: CueErrors::default(),
            cue_ratings: [EXT_E_OFFSET; 8],
            azimuth_deg: 0.0,
        };
    }
    let fs = (sample_rate.max(8_000)) as f64;
    let cfs = ext_band_cfs(fs);

    // 1. Target analysis of the direct-path portion (first 3.8 ms after the direct sound,
    //    the direct-path definition of the paper's Exp. III), so that late reverberation
    //    does not masquerade as a spectral-shape deviation from the internal DTF.
    let direct_l = ext_direct_path(&ir_l[..n], fs);
    let direct_r = ext_direct_path(&ir_r[..n], fs);
    let (pow_tl, band_tl) = ext_analyze(&direct_l, fs, &cfs);
    let (pow_tr, band_tr) = ext_analyze(&direct_r, fs, &cfs);

    // 2. Internal template: our own HRTF model for the intended source direction
    //    (documented substitution for the listener-specific DTF).
    let env_tl = ext_envelope(&ext_lowpass_3k(&ir_l[..n], fs), fs);
    let env_tr = ext_envelope(&ext_lowpass_3k(&ir_r[..n], fs), fs);
    let itd_target = ext_itd_us(&env_tl, &env_tr, fs);
    let azimuth_deg = azimuth_deg.clamp(-90.0, 90.0);
    let (tpl_l, tpl_r) = ext_template_ir(n, fs, azimuth_deg);
    let (pow_ml, band_ml) = ext_analyze(&ext_direct_path(&tpl_l, fs), fs, &cfs);
    let (pow_mr, band_mr) = ext_analyze(&ext_direct_path(&tpl_r, fs), fs, &cfs);

    // 3. Common gain: the template's loudest band is presented at 70 dB SPL and the
    //    target keeps its level relative to the template so that MI stays meaningful.
    let peak = pow_ml
        .iter()
        .chain(pow_mr.iter())
        .cloned()
        .fold(1e-30_f64, f64::max);
    let gain_db = EXT_SPL_DB - 10.0 * peak.log10();
    let to_db = |p: &[f64]| -> Vec<f64> {
        p.iter()
            .map(|&v| (10.0 * v.max(1e-30).log10() + gain_db).max(EXT_AUDIBLE_FLOOR_DB))
            .collect()
    };
    let p_tl = to_db(&pow_tl);
    let p_tr = to_db(&pow_tr);
    let p_ml = to_db(&pow_ml);
    let p_mr = to_db(&pow_mr);

    // 4. Cue errors.
    let iid_t: Vec<f64> = p_tl.iter().zip(&p_tr).map(|(l, r)| l - r).collect();
    let iid_m: Vec<f64> = p_ml.iter().zip(&p_mr).map(|(l, r)| l - r).collect();

    let msg_tl = ext_msg(&p_tl);
    let msg_tr = ext_msg(&p_tr);
    let msg_ml = ext_msg(&p_ml);
    let msg_mr = ext_msg(&p_mr);

    let mss_l = ext_abs_norm(&msg_tl, &msg_ml, EXT_ILD_JND_DB, 0.5);
    let mss_r = ext_abs_norm(&msg_tr, &msg_mr, EXT_ILD_JND_DB, 0.5);
    let mss = ext_binaural_weight(mss_l, mss_r, azimuth_deg);

    let mssd_l = ext_sd_ratio(&msg_tl, &msg_ml);
    let mssd_r = ext_sd_ratio(&msg_tr, &msg_mr);
    let mssd = ext_binaural_weight(mssd_l, mssd_r, azimuth_deg);

    let iss = ext_abs_norm(&iid_t, &iid_m, EXT_ILD_JND_DB, EXT_ILD_JND_DB);
    let issd = ext_sd_ratio(&iid_t, &iid_m);

    let mi_tl = ext_rms_db(&ir_l[..n]) + gain_db;
    let mi_tr = ext_rms_db(&ir_r[..n]) + gain_db;
    let mi_ml = ext_rms_db(&tpl_l) + gain_db;
    let mi_mr = ext_rms_db(&tpl_r) + gain_db;
    let mi = 0.5
        * (ext_jnd_ratio(mi_tl, mi_ml, EXT_ILD_JND_DB, EXT_SPL_DB)
            + ext_jnd_ratio(mi_tr, mi_mr, EXT_ILD_JND_DB, EXT_SPL_DB));

    let ic_target = ext_ic(&ir_l[..n], &ir_r[..n], fs);
    let ic_template = ext_ic(&tpl_l, &tpl_r, fs);
    let ic = ext_jnd_ratio(ic_target, ic_template, 0.0, ic_template.max(1e-3));

    let itd_template = ext_itd_us(
        &ext_envelope(&ext_lowpass_3k(&tpl_l, fs), fs),
        &ext_envelope(&ext_lowpass_3k(&tpl_r, fs), fs),
        fs,
    );
    let iid_t_bb = ext_mean(&iid_t);
    let iid_m_bb = ext_mean(&iid_m);
    let itit = ext_itit(itd_target, itd_template, iid_t_bb, iid_m_bb);

    let itsd = {
        let bt = ext_band_itds(&band_tl, &band_tr, &cfs, fs);
        let bm = ext_band_itds(&band_ml, &band_mr, &cfs, fs);
        ext_sd_ratio(&bt, &bm)
    };

    let cue_errors = CueErrors {
        mss,
        iss,
        mssd,
        issd,
        itit,
        ic,
        mi,
        itsd,
    };

    // 5. Eq. 10 mapping and Fig. 1 decision stage.
    let cue_ratings = ext_map_all(cue_errors.as_array());
    let raw = match strategy {
        DecisionStrategy::WeightedSum => cue_ratings
            .iter()
            .zip(EXT_WEIGHTS.iter())
            .map(|(e, w)| e * w)
            .sum::<f64>(),
        DecisionStrategy::WinnerTakesAll => cue_ratings
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
        DecisionStrategy::LoserTakesAll => {
            cue_ratings.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        DecisionStrategy::MedianTakesAll => {
            let mut r = cue_ratings;
            r.sort_by(f64::total_cmp);
            0.5 * (r[3] + r[4])
        }
    };
    let norm = |e: f64| ((e - EXT_E_OFFSET) / EXT_E_RANGE).clamp(0.0, 1.0);

    // Early-reflection diagnostic (unchanged, not one of the paper's cues).
    let early_refl_db = ext_early_reflection_db(&ir_l[..n], &ir_r[..n], fs);

    ExternalizationReport {
        score: norm(raw),
        monaural_cue: norm(cue_ratings[0]),
        binaural_cue: norm(cue_ratings[1]),
        iacc: ic_target,
        early_refl_db,
        cue_errors,
        cue_ratings,
        azimuth_deg,
    }
}

/// Eq. 10: `e_cue = 2 e_range / (1 + exp(d_cue / S_cue)) + e_offset`.
fn ext_map_with_sensitivity(d: f64, s_cue: f64) -> f64 {
    2.0 * EXT_E_RANGE / (1.0 + (d / s_cue).exp()) + EXT_E_OFFSET
}

// The array mapping needs the per-index sensitivities; do it explicitly so the ordering
// of the cue vector is visible at the mapping site.
fn ext_map_all(errs: [f64; 8]) -> [f64; 8] {
    [
        ext_map_with_sensitivity(errs[0], EXT_SENSITIVITY[0]),
        ext_map_with_sensitivity(errs[1], EXT_SENSITIVITY[1]),
        ext_map_with_sensitivity(errs[2], EXT_SENSITIVITY[2]),
        ext_map_with_sensitivity(errs[3], EXT_SENSITIVITY[3]),
        ext_map_with_sensitivity(errs[4], EXT_SENSITIVITY[4]),
        ext_map_with_sensitivity(errs[5], EXT_SENSITIVITY[5]),
        ext_map_with_sensitivity(errs[6], EXT_SENSITIVITY[6]),
        ext_map_with_sensitivity(errs[7], EXT_SENSITIVITY[7]),
    ]
}

fn ext_band_cfs(fs: f64) -> Vec<f64> {
    let f_min = 100.0_f64;
    let f_max = 16_000.0_f64.min(0.45 * fs);
    let e_min = 21.4 * (0.004_37 * f_min + 1.0).log10();
    let e_max = 21.4 * (0.004_37 * f_max + 1.0).log10();
    let n = ((e_max - e_min).floor() as usize).max(1) + 1;
    (0..n)
        .map(|i| {
            let e = e_min + i as f64;
            (10.0_f64.powf(e / 21.4) - 1.0) / 0.004_37
        })
        .take_while(|f| *f <= f_max * 1.000_1)
        .collect()
}

/// Direct-path portion: from 0.5 ms before the peak to 3.8 ms after it, zero elsewhere.
fn ext_direct_path(x: &[f64], fs: f64) -> Vec<f64> {
    let n = x.len();
    let peak = (0..n)
        .max_by(|&a, &b| (x[a] * x[a]).total_cmp(&(x[b] * x[b])))
        .unwrap_or(0);
    let start = peak.saturating_sub((0.000_5 * fs) as usize);
    let end = (peak + (0.003_8 * fs) as usize).min(n);
    let mut out = vec![0.0_f64; n];
    out[start..end].copy_from_slice(&x[start..end]);
    out
}

/// 4th-order gammatone band signal (same cascade used by the cochlea model).
fn ext_gammatone_band(x: &[f64], fs: f64, cf: f64) -> Vec<f64> {
    let bw = 1.019 * erb_hz(cf);
    let r = (-2.0 * PI * bw / fs).exp();
    let pole = Complex::from_polar(r, 2.0 * PI * cf / fs);
    let gain = (1.0 - r).powi(4);
    let mut state = [Complex::ZERO; 4];
    let mut out = Vec::with_capacity(x.len());
    for &sample in x {
        let mut cur = Complex::new(sample * gain, 0.0);
        for st in &mut state {
            cur = cur.add(st.mul(pole));
            *st = cur;
        }
        out.push(cur.re);
    }
    out
}

fn ext_analyze(x: &[f64], fs: f64, cfs: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut powers = Vec::with_capacity(cfs.len());
    let mut bands = Vec::with_capacity(cfs.len());
    for &cf in cfs {
        let b = ext_gammatone_band(x, fs, cf);
        let ms = b.iter().map(|v| v * v).sum::<f64>() / (b.len().max(1) as f64);
        powers.push(ms);
        bands.push(b);
    }
    (powers, bands)
}

/// Spectral gradient restricted by an elevated arctangent, Eq. 6.
fn ext_msg(p: &[f64]) -> Vec<f64> {
    p.windows(2)
        .map(|w| ((w[1] - w[0]) - PI / 2.0).atan() + PI / 2.0)
        .collect()
}

/// Eq. 4/7: threshold `JND`, then mean of `|a-b| / max(|b|, floor)`.
fn ext_abs_norm(a: &[f64], b: &[f64], jnd: f64, floor: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (&ai, &bi) in a.iter().zip(b).take(n) {
        let d = (ai - bi).abs();
        if d > jnd {
            sum += d / bi.abs().max(floor);
        }
    }
    sum / n as f64
}

/// Eq. 3/8: relative difference of standard deviations across frequency.
fn ext_sd_ratio(a: &[f64], b: &[f64]) -> f64 {
    let sd_a = ext_sd(a);
    let sd_b = ext_sd(b);
    (sd_a - sd_b).abs() / sd_b.max(1e-6)
}

fn ext_sd(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    let mu = ext_mean(x);
    (x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / x.len() as f64).sqrt()
}

fn ext_mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

/// Eq. 1: JND-zeroed level difference normalized by the template level.
fn ext_jnd_ratio(target: f64, template: f64, jnd: f64, floor: f64) -> f64 {
    let delta = (target - template).abs();
    let gated = if delta > jnd { delta } else { 0.0 };
    gated / template.abs().max(floor)
}

/// Eq. 5: ITIT expectation error with the 20 µs / 1 dB JNDs.
fn ext_itit(itd_t: f64, itd_m: f64, iid_t: f64, iid_m: f64) -> f64 {
    let d_itd = itd_t - itd_m;
    let d_iid = iid_t - iid_m;
    let g_itd = if d_itd.abs() > EXT_ITD_JND_US {
        d_itd
    } else {
        0.0
    };
    let g_iid = if d_iid.abs() > EXT_ILD_JND_DB {
        d_iid
    } else {
        0.0
    };
    (g_itd / itd_m.abs().max(EXT_ITD_JND_US) - g_iid / iid_m.abs().max(EXT_ILD_JND_DB)).abs()
}

/// Eq. 9 binaural weighting: ipsilateral ear dominates with lateral eccentricity.
fn ext_binaural_weight(d_l: f64, d_r: f64, phi_deg: f64) -> f64 {
    d_r + (d_l - d_r) / (1.0 + (-phi_deg / EXT_PHI_DEG).exp())
}

fn ext_rms_db(x: &[f64]) -> f64 {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64;
    10.0 * ms.max(1e-30).log10()
}

/// 4th-order Butterworth low-pass at 3 kHz (two cascaded biquads), Eq. 5 ITD input.
fn ext_lowpass_3k(x: &[f64], fs: f64) -> Vec<f64> {
    let fc = 3_000.0_f64.min(0.45 * fs);
    let k = (PI * fc / fs).tan();
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let norm = 1.0 / (1.0 + k / q + k * k);
    let b0 = k * k * norm;
    let b1 = 2.0 * b0;
    let b2 = b0;
    let a1 = 2.0 * (k * k - 1.0) * norm;
    let a2 = (1.0 - k / q + k * k) * norm;
    let mut y = x.to_vec();
    for _ in 0..2 {
        let mut z1 = 0.0;
        let mut z2 = 0.0;
        for v in &mut y {
            let input = *v;
            let out = b0 * input + z1;
            z1 = b1 * input - a1 * out + z2;
            z2 = b2 * input - a2 * out;
            *v = out;
        }
    }
    y
}

/// Temporal energy envelope: rectification + 1 ms one-pole smoothing.
fn ext_envelope(x: &[f64], fs: f64) -> Vec<f64> {
    let a = (-1.0 / (0.001 * fs)).exp();
    let mut state = 0.0;
    x.iter()
        .map(|&v| {
            state = (1.0 - a) * v.abs() + a * state;
            state
        })
        .collect()
}

/// Envelope ITD in microseconds: lag maximizing the normalized envelope correlation.
/// Positive values mean the left ear leads (source on the left).
fn ext_itd_us(env_l: &[f64], env_r: &[f64], fs: f64) -> f64 {
    let n = env_l.len().min(env_r.len());
    if n < 8 {
        return 0.0;
    }
    let max_lag = ((1e-3 * fs) as usize).clamp(1, n / 2);
    let e_l: f64 = env_l[..n].iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let e_r: f64 = env_r[..n].iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let denom = (e_l * e_r).sqrt();
    let mut corr = vec![0.0_f64; 2 * max_lag + 1];
    let mut best = (max_lag as isize, f64::NEG_INFINITY);
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut acc = 0.0;
        for (i, &ri) in env_r[..n].iter().enumerate() {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < n {
                acc += env_l[j as usize] * ri;
            }
        }
        let c = acc / denom;
        corr[(lag + max_lag as isize) as usize] = c;
        if c > best.1 {
            best = (lag, c);
        }
    }
    let idx = (best.0 + max_lag as isize) as usize;
    let mut frac = 0.0;
    if idx > 0 && idx + 1 < corr.len() {
        let (a, b, c) = (corr[idx - 1], corr[idx], corr[idx + 1]);
        let den = a - 2.0 * b + c;
        if den.abs() > 1e-12 {
            frac = 0.5 * (a - c) / den;
        }
    }
    // `lag` aligns the left envelope onto a later right envelope, i.e. it is negative
    // when the left ear leads; negate so that positive ITD means a left-side source.
    -(best.0 as f64 + frac) / fs * 1e6
}

/// Broadband interaural coherence: maximum normalized correlation over ±1 ms.
fn ext_ic(x_l: &[f64], x_r: &[f64], fs: f64) -> f64 {
    let n = x_l.len().min(x_r.len());
    if n < 8 {
        return 1.0;
    }
    let max_lag = ((1e-3 * fs) as usize).clamp(1, n / 2);
    let e_l: f64 = x_l[..n].iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let e_r: f64 = x_r[..n].iter().map(|v| v * v).sum::<f64>().max(1e-30);
    let denom = (e_l * e_r).sqrt();
    let mut best = f64::NEG_INFINITY;
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let mut acc = 0.0;
        for (i, &ri) in x_r[..n].iter().enumerate() {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < n {
                acc += x_l[j as usize] * ri;
            }
        }
        best = best.max(acc.abs() / denom);
    }
    best.clamp(0.0, 1.0)
}

fn ext_band_itds(band_l: &[Vec<f64>], band_r: &[Vec<f64>], cfs: &[f64], fs: f64) -> Vec<f64> {
    band_l
        .iter()
        .zip(band_r)
        .zip(cfs)
        .filter(|(_, cf)| **cf <= 3_000.0)
        .map(|((l, r), _)| {
            let el = ext_envelope(l, fs);
            let er = ext_envelope(r, fs);
            ext_itd_us(&el, &er, fs)
        })
        .collect()
}

/// Synthesize the internal DTF template pair from our own snowman HRTF model.
///
/// This is the documented substitution for the paper's listener-specific HRTF/BRIR
/// template: the anechoic externalized response of the production renderer's physical
/// head/torso model at `az_deg`, elevation 0, and the default 2 m source distance.
pub fn externalization_template_ir(n: usize, fs: f64, az_deg: f64) -> (Vec<f64>, Vec<f64>) {
    ext_template_ir(n, fs, az_deg)
}

fn ext_template_ir(n: usize, fs: f64, az_deg: f64) -> (Vec<f64>, Vec<f64>) {
    let n_fft = n.next_power_of_two().clamp(512, 8_192);
    let plan = FftPlan::new(n_fft);
    let geom = SnowmanGeom::default();
    let mut bins_l = vec![Complex::ZERO; n_fft / 2 + 1];
    let mut bins_r = vec![Complex::ZERO; n_fft / 2 + 1];
    for k in 0..=n_fft / 2 {
        let f = k as f64 * fs / n_fft as f64;
        let (hl, hr) = if k == 0 {
            (Complex::ONE, Complex::ONE)
        } else {
            (
                snowman_hrtf(
                    f,
                    geom,
                    az_deg,
                    0.0,
                    EXT_TEMPLATE_DISTANCE_M,
                    1.0,
                    EXT_SPEED_OF_SOUND,
                    true,
                ),
                snowman_hrtf(
                    f,
                    geom,
                    az_deg,
                    0.0,
                    EXT_TEMPLATE_DISTANCE_M,
                    -1.0,
                    EXT_SPEED_OF_SOUND,
                    true,
                ),
            )
        };
        bins_l[k] = hl;
        bins_r[k] = hr;
    }
    let mut scratch = vec![Complex::ZERO; n_fft];
    let mut out_l = vec![0.0_f64; n_fft];
    let mut out_r = vec![0.0_f64; n_fft];
    plan.irfft(&bins_l, &mut scratch, &mut out_l);
    plan.irfft(&bins_r, &mut scratch, &mut out_r);
    // The normalized HRTF of the near ear is advanced in time (its path is shorter), so a
    // circular inverse FFT would place it at the end of the buffer. Apply a common 2 ms
    // delay to both ears; the interaural difference is preserved and both impulses become
    // causal inside the analysis window.
    let shift = ((0.002 * fs).round() as usize).min(n_fft - 1);
    let delayed = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0_f64; n_fft];
        for (i, &v) in x.iter().enumerate() {
            y[(i + shift) % n_fft] = v;
        }
        y
    };
    (delayed(&out_l), delayed(&out_r))
}

fn ext_early_reflection_db(ir_l: &[f64], ir_r: &[f64], fs: f64) -> f64 {
    let n = ir_l.len().min(ir_r.len());
    let peak_idx = (0..n)
        .max_by(|&a, &b| {
            let ea = ir_l[a] * ir_l[a] + ir_r[a] * ir_r[a];
            let eb = ir_l[b] * ir_l[b] + ir_r[b] * ir_r[b];
            ea.total_cmp(&eb)
        })
        .unwrap_or(0);
    let dir_lo = peak_idx.saturating_sub((0.001 * fs) as usize);
    let dir_hi = (peak_idx + (0.004 * fs) as usize).min(n);
    let refl_lo = (peak_idx + (0.005 * fs) as usize).min(n);
    let refl_hi = (peak_idx + (0.080 * fs) as usize).min(n);
    let e_dir: f64 = (dir_lo..dir_hi)
        .map(|i| ir_l[i] * ir_l[i] + ir_r[i] * ir_r[i])
        .sum::<f64>()
        .max(1e-30);
    let e_refl: f64 = (refl_lo..refl_hi)
        .map(|i| ir_l[i] * ir_l[i] + ir_r[i] * ir_r[i])
        .sum();
    10.0 * (e_refl.max(1e-18) / e_dir).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template_pair(az_deg: f64) -> (Vec<f64>, Vec<f64>) {
        ext_template_ir(8_192, 48_000.0, az_deg)
    }

    #[test]
    fn eq10_mapping_has_published_endpoints() {
        // d = 0 -> 2*range/(1+1) + offset = range + offset -> normalized 1.0.
        assert!((ext_map_with_sensitivity(0.0, 0.1) - (EXT_E_RANGE + EXT_E_OFFSET)).abs() < 1e-12);
        // d -> inf -> offset -> normalized 0.0.
        assert!((ext_map_with_sensitivity(1e6, 0.1) - EXT_E_OFFSET).abs() < 1e-9);
        // Monotone decreasing in d.
        let mut prev = f64::INFINITY;
        for d in [0.0, 0.1, 0.5, 1.0, 4.0] {
            let e = ext_map_with_sensitivity(d, 0.1);
            assert!(e < prev);
            prev = e;
        }
    }

    #[test]
    fn eq9_weighting_favours_ipsilateral_ear() {
        // phi = +90 deg (left) -> weight entirely on the left error.
        let w = ext_binaural_weight(0.8, 0.2, 90.0);
        assert!((w - 0.8).abs() < 1e-3, "left-dominant weight {w}");
        let w0 = ext_binaural_weight(0.8, 0.2, 0.0);
        assert!((w0 - 0.5).abs() < 1e-12, "frontal weight {w0}");
    }

    #[test]
    fn jnd_zeros_sub_threshold_differences() {
        let a = [10.0, 12.0, 14.0, 16.0];
        let within = [10.5, 12.4, 14.0, 15.6]; // < 1 dB everywhere
        assert_eq!(ext_abs_norm(&a, &within, 1.0, 1.0), 0.0);
        let beyond = [12.0, 12.0, 14.0, 16.0]; // 2 dB on one band
        assert!(ext_abs_norm(&a, &beyond, 1.0, 1.0) > 0.0);
    }

    #[test]
    fn template_itd_sign_and_magnitude_follow_azimuth() {
        let (l, r) = template_pair(40.0);
        let itd = ext_itd_us(
            &ext_envelope(&ext_lowpass_3k(&l, 48_000.0), 48_000.0),
            &ext_envelope(&ext_lowpass_3k(&r, 48_000.0), 48_000.0),
            48_000.0,
        );
        assert!(
            itd > 100.0,
            "left source must yield positive ITD, got {itd}"
        );
        let (l2, r2) = template_pair(-40.0);
        let itd2 = ext_itd_us(
            &ext_envelope(&ext_lowpass_3k(&l2, 48_000.0), 48_000.0),
            &ext_envelope(&ext_lowpass_3k(&r2, 48_000.0), 48_000.0),
            48_000.0,
        );
        assert!(
            itd2 < -100.0,
            "right source must yield negative ITD, got {itd2}"
        );
    }

    #[test]
    fn externalized_template_scores_above_in_head_delta() {
        let fs = 48_000u32;
        let (l, r) = template_pair(35.0);
        let e_ext = externalization_with_azimuth(&l, &r, fs, 35.0, DecisionStrategy::WeightedSum);
        assert!(
            e_ext.score > 0.75,
            "template vs itself should externalize, got {:.3}",
            e_ext.score
        );
        let mut dry_l = vec![0.0; 4096];
        let mut dry_r = vec![0.0; 4096];
        dry_l[200] = 1.0;
        dry_r[200] = 1.0;
        let e_dry = baumgartner_majdak_externalization(&dry_l, &dry_r, fs);
        assert!(
            e_dry.score < e_ext.score - 0.25,
            "in-head delta ({:.3}) must score well below externalized template ({:.3})",
            e_dry.score,
            e_ext.score
        );
    }

    #[test]
    fn decision_strategies_obey_ordering() {
        let fs = 48_000u32;
        let (mut l, mut r) = template_pair(30.0);
        l[100] += 0.3;
        r[3000] -= 0.2;
        let wsm = externalization_with_strategy(&l, &r, fs, DecisionStrategy::WeightedSum);
        let wta = externalization_with_strategy(&l, &r, fs, DecisionStrategy::WinnerTakesAll);
        let mta = externalization_with_strategy(&l, &r, fs, DecisionStrategy::MedianTakesAll);
        let lta = externalization_with_strategy(&l, &r, fs, DecisionStrategy::LoserTakesAll);
        assert!(wta.score >= mta.score - 1e-12);
        assert!(mta.score >= lta.score - 1e-12);
        assert!(wsm.score >= 0.0 && wsm.score <= 1.0);
        assert_eq!(wta.cue_ratings.len(), 8);
    }
}
