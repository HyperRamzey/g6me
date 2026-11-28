//! Auditory externalization model of Baumgartner & Majdak (2021), "Decision making in
//! auditory externalization perception: model predictions for static conditions",
//! Acta Acustica 5, 59, DOI 10.1051/aacus/2021053 (CC-BY), Eqs. 1–10.
//!
//! Seven cues are evaluated against an internal template — exactly the cue set of the
//! paper: `MSS`, `ISS`, `MSSD`, `ISSD`, `ITIT`, `IC`, and `MI`. The AMT's `ITSD` (the
//! interaural temporal standard deviation of ILDs across 20 ms frames, defined by
//! Li, Baumgartner & Peissig (2020), Acta Acustica 4, 21, Eq. 5) is not one of the
//! seven cues evaluated by the 2021 model and is not implemented here.
//!
//! The default decision stage is the weighted-sum model (WSM) with the published
//! weights 0.6 `MSS` / 0.4 `ISS` and 0 for the remaining cues (Fig. 4b); the dynamic
//! LTA / MTA / WTA selection strategies are also available.
//!
//! The score is the paper's template-similarity rating (Eqs. 4, 7 and 10), not a
//! room-benefit metric: the model has no reverberation term, and the paper restricts
//! its predictions to static conditions, leaving extension to naturally reverberant
//! stimuli open. A room tier is therefore not predicted to outscore an anechoic tier
//! merely by adding reflections.
//!
//! **Template substitution (TEMPLATE_LIMITED).** The paper compares against
//! listener-specific HRTF/BRIR templates that are not bundled here. The built-in
//! template is synthesized from our own physical head/torso model
//! (`scattering::snowman_hrtf`), which has **no pinna**, at the caller-supplied lateral
//! angle (default 30°, the front-left production slot), elevation 0, range 2 m. Because
//! Eq. 6/7 MSS relies on pinna-notch spectral contrasts, this pinna-free template
//! cannot reproduce the paper's hypothetical diotic benchmark of ~0.2 (Sec. 3.4): an
//! in-head delta yields `d_MSS ≈ 0.04` and a score near 0.5 instead. The verifier's
//! in-head gate for this default is therefore
//! [`EXT_TEMPLATE_LIMITED_DRY_GATE`] (0.60), which is explicitly non-paper-literal.
//! Callers that have a pinna-bearing HRTF pair — e.g. the FMM-BEM HRTF solved on a
//! `g6me-scan` head mesh — should pass it to
//! [`externalization_with_template_ir`]; that path removes the template limitation.
//! The target's broadband ITD enters only the ITIT cue; it does not steer the template.

use crate::fft::{erb_hz, Complex, FftPlan};
use crate::scattering::{snowman_hrtf, SnowmanGeom};
use std::f64::consts::PI;

/// Rating scale range `e_range` of Eq. 10 for the paper's normalized 0–1 rating scale:
/// at `d = 0` the mapping is `e_range + e_offset = 1`, and it tends to `e_offset = 0`
/// as `d → ∞`. The absolute scale cancels in the normalized score.
pub const EXT_E_RANGE: f64 = 1.0;
/// Rating scale offset `e_offset` of Eq. 10 (rating of an in-head image), 0 on the
/// paper's normalized scale.
pub const EXT_E_OFFSET: f64 = 0.0;
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

/// Numerical guard for the template denominators of Eqs. 1–5 and 7–8. The paper
/// specifies no floor; this value only prevents division by exactly zero.
const EXT_DENOM_FLOOR: f64 = 1e-12;

/// Non-paper-literal in-head gate for the pinna-free default template (0.60).
///
/// The paper's hypothetical diotic benchmark is an externalization score of ~0.2 with a
/// listener-specific (pinna-bearing) template (Sec. 3.4). The built-in snowman template
/// has no pinna, so its `MSS` error for an in-head delta is only ~0.04 and the score is
/// ~0.49; the verifier therefore gates the default path at 0.60. This constant names
/// that deviation instead of presenting 0.60 as the paper's threshold. Supply a pinna
/// HRTF through [`externalization_with_template_ir`] to gate at the paper's value.
pub const EXT_TEMPLATE_LIMITED_DRY_GATE: f64 = 0.60;

/// WSM cue weights from Baumgartner & Majdak (2021) Fig. 4b and Sec. 3.3, ordered
/// `[MSS, ISS, MSSD, ISSD, ITIT, IC, MI]`. The paper reports about 0.6 for the monaural
/// spectral-shape cue and 0.4 for the interaural spectral-shape cue, with all other
/// weights < 0.005 (set to zero).
pub const EXT_WEIGHTS: [f64; 7] = [0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Cue sensitivities `S_cue` of Eq. 10, ordered as in [`EXT_WEIGHTS`], digitized from
/// the paper's Fig. 3a (the values are published only as a logarithmic colour map;
/// these are the geometric means across the five modelled experiments).
pub const EXT_SENSITIVITY: [f64; 7] = [0.1195, 0.0867, 0.0628, 0.1406, 0.0550, 0.0186, 0.003_72];

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

/// Single-cue expectation errors `d_cue` (dimensionless, Eqs. 1–9), ordered as in
/// [`EXT_WEIGHTS`].
#[derive(Debug, Clone, Copy, Default)]
pub struct CueErrors {
    pub mss: f64,
    pub iss: f64,
    pub mssd: f64,
    pub issd: f64,
    pub itit: f64,
    pub ic: f64,
    pub mi: f64,
}

impl CueErrors {
    fn as_array(self) -> [f64; 7] {
        [
            self.mss, self.iss, self.mssd, self.issd, self.itit, self.ic, self.mi,
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
    /// Raw expectation errors of all seven cues.
    pub cue_errors: CueErrors,
    /// Raw Eq. 10 ratings `e_cue`, ordered as in [`EXT_WEIGHTS`].
    pub cue_ratings: [f64; 7],
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
///
/// The template is the built-in pinna-free snowman HRTF. See the module docs
/// (TEMPLATE_LIMITED) before comparing absolute scores with the paper's.
pub fn externalization_with_azimuth(
    ir_l: &[f64],
    ir_r: &[f64],
    sample_rate: u32,
    azimuth_deg: f64,
    strategy: DecisionStrategy,
) -> ExternalizationReport {
    let n = ir_l.len().min(ir_r.len());
    let azimuth_deg = azimuth_deg.clamp(-90.0, 90.0);
    let fs = (sample_rate.max(8_000)) as f64;
    let (tpl_l, tpl_r) = if n < 32 {
        (Vec::new(), Vec::new())
    } else {
        ext_template_ir(n, fs, azimuth_deg)
    };
    externalization_core(
        ir_l,
        ir_r,
        &tpl_l,
        &tpl_r,
        sample_rate,
        azimuth_deg,
        strategy,
    )
}

/// Evaluate the model against a caller-supplied template HRIR/BRIR pair.
///
/// `template_l` / `template_r` must be the externalized (anechoic) impulse responses
/// for the same source direction as the target. Pass a listener-specific, pinna-bearing
/// HRTF pair — for example the FMM-BEM HRTF solved on a `g6me-scan` head mesh — to get a
/// paper-faithful prediction; the built-in snowman template has no pinna and is
/// TEMPLATE_LIMITED. The template arrays are analyzed as supplied and may have a
/// different length from the target.
pub fn externalization_with_template_ir(
    ir_l: &[f64],
    ir_r: &[f64],
    template_l: &[f64],
    template_r: &[f64],
    sample_rate: u32,
    azimuth_deg: f64,
    strategy: DecisionStrategy,
) -> ExternalizationReport {
    externalization_core(
        ir_l,
        ir_r,
        template_l,
        template_r,
        sample_rate,
        azimuth_deg.clamp(-90.0, 90.0),
        strategy,
    )
}

fn externalization_core(
    ir_l: &[f64],
    ir_r: &[f64],
    tpl_l: &[f64],
    tpl_r: &[f64],
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
            cue_ratings: [EXT_E_OFFSET; 7],
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
    let pow_tl = ext_analyze(&direct_l, fs, &cfs);
    let pow_tr = ext_analyze(&direct_r, fs, &cfs);

    // 2. Internal template: caller-supplied HRTF/BRIR pair. The default entry point
    //    synthesizes it from the pinna-free snowman model (TEMPLATE_LIMITED).
    let env_tl = ext_envelope(&ext_lowpass_3k(&ir_l[..n], fs), fs);
    let env_tr = ext_envelope(&ext_lowpass_3k(&ir_r[..n], fs), fs);
    let itd_target = ext_itd_us(&env_tl, &env_tr, fs);
    let pow_ml = ext_analyze(&ext_direct_path(tpl_l, fs), fs, &cfs);
    let pow_mr = ext_analyze(&ext_direct_path(tpl_r, fs), fs, &cfs);

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

    let mss_l = ext_abs_norm(&msg_tl, &msg_ml, EXT_ILD_JND_DB);
    let mss_r = ext_abs_norm(&msg_tr, &msg_mr, EXT_ILD_JND_DB);
    let mss = ext_binaural_weight(mss_l, mss_r, azimuth_deg);

    let mssd_l = ext_sd_ratio(&msg_tl, &msg_ml);
    let mssd_r = ext_sd_ratio(&msg_tr, &msg_mr);
    let mssd = ext_binaural_weight(mssd_l, mssd_r, azimuth_deg);

    let iss = ext_abs_norm(&iid_t, &iid_m, EXT_ILD_JND_DB);
    let issd = ext_sd_ratio(&iid_t, &iid_m);

    let mi_tl = ext_rms_db(&ir_l[..n]) + gain_db;
    let mi_tr = ext_rms_db(&ir_r[..n]) + gain_db;
    let mi_ml = ext_rms_db(tpl_l) + gain_db;
    let mi_mr = ext_rms_db(tpl_r) + gain_db;
    let mi = 0.5
        * (ext_jnd_ratio(mi_tl, mi_ml, EXT_ILD_JND_DB)
            + ext_jnd_ratio(mi_tr, mi_mr, EXT_ILD_JND_DB));

    let ic_target = ext_ic(&ir_l[..n], &ir_r[..n], fs);
    let ic_template = ext_ic(tpl_l, tpl_r, fs);
    let ic = ext_jnd_ratio(ic_target, ic_template, 0.0);

    let itd_template = ext_itd_us(
        &ext_envelope(&ext_lowpass_3k(tpl_l, fs), fs),
        &ext_envelope(&ext_lowpass_3k(tpl_r, fs), fs),
        fs,
    );
    let iid_t_bb = ext_mean(&iid_t);
    let iid_m_bb = ext_mean(&iid_m);
    let itit = ext_itit(itd_target, itd_template, iid_t_bb, iid_m_bb);

    let cue_errors = CueErrors {
        mss,
        iss,
        mssd,
        issd,
        itit,
        ic,
        mi,
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
            r[3]
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
fn ext_map_all(errs: [f64; 7]) -> [f64; 7] {
    [
        ext_map_with_sensitivity(errs[0], EXT_SENSITIVITY[0]),
        ext_map_with_sensitivity(errs[1], EXT_SENSITIVITY[1]),
        ext_map_with_sensitivity(errs[2], EXT_SENSITIVITY[2]),
        ext_map_with_sensitivity(errs[3], EXT_SENSITIVITY[3]),
        ext_map_with_sensitivity(errs[4], EXT_SENSITIVITY[4]),
        ext_map_with_sensitivity(errs[5], EXT_SENSITIVITY[5]),
        ext_map_with_sensitivity(errs[6], EXT_SENSITIVITY[6]),
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

fn ext_analyze(x: &[f64], fs: f64, cfs: &[f64]) -> Vec<f64> {
    cfs.iter()
        .map(|&cf| {
            let b = ext_gammatone_band(x, fs, cf);
            b.iter().map(|v| v * v).sum::<f64>() / (b.len().max(1) as f64)
        })
        .collect()
}

/// Spectral gradient restricted by an elevated arctangent, Eq. 6.
fn ext_msg(p: &[f64]) -> Vec<f64> {
    p.windows(2)
        .map(|w| ((w[1] - w[0]) - PI / 2.0).atan() + PI / 2.0)
        .collect()
}

/// Eq. 4/7: zero differences at or below `jnd`, then mean of `|a-b| / |b|`.
/// The paper specifies no denominator floor; [`EXT_DENOM_FLOOR`] is numerical only.
fn ext_abs_norm(a: &[f64], b: &[f64], jnd: f64) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for (&ai, &bi) in a.iter().zip(b).take(n) {
        let d = (ai - bi).abs();
        if d > jnd {
            sum += d / bi.abs().max(EXT_DENOM_FLOOR);
        }
    }
    sum / n as f64
}

/// Eq. 3/8: relative difference of standard deviations across frequency.
fn ext_sd_ratio(a: &[f64], b: &[f64]) -> f64 {
    let sd_a = ext_sd(a);
    let sd_b = ext_sd(b);
    (sd_a - sd_b).abs() / sd_b.max(EXT_DENOM_FLOOR)
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

/// Eq. 1/2: JND-gated absolute difference normalized by the template magnitude.
fn ext_jnd_ratio(target: f64, template: f64, jnd: f64) -> f64 {
    let delta = (target - template).abs();
    let gated = if delta > jnd { delta } else { 0.0 };
    gated / template.abs().max(EXT_DENOM_FLOOR)
}

/// Signed denominator guard that preserves the template sign (Eq. 5).
fn ext_signed_denom(x: f64) -> f64 {
    if x.abs() >= EXT_DENOM_FLOOR {
        x
    } else if x < 0.0 {
        -EXT_DENOM_FLOOR
    } else {
        EXT_DENOM_FLOOR
    }
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
    (g_itd / ext_signed_denom(itd_m) - g_iid / ext_signed_denom(iid_m)).abs()
}

/// Eq. 9 binaural weighting: ipsilateral ear dominates with lateral eccentricity.
fn ext_binaural_weight(d_l: f64, d_r: f64, phi_deg: f64) -> f64 {
    d_r + (d_l - d_r) / (1.0 + (-phi_deg / EXT_PHI_DEG).exp())
}

fn ext_rms_db(x: &[f64]) -> f64 {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64;
    10.0 * ms.max(1e-30).log10()
}

/// Section quality factors of a 4th-order Butterworth response:
/// `Q_k = 1 / (2 cos((2k+1)π/8))`, k = 0, 1.
const EXT_BUTTERWORTH_Q: [f64; 2] = [0.541_196_100_146_196_9, 1.306_562_964_876_376_4];

/// 4th-order Butterworth low-pass at 3 kHz (two cascaded biquads with
/// `Q = 0.5412` and `1.3066`; -3.01 dB at 3 kHz, 24 dB/oct), Eq. 5 ITD input.
fn ext_lowpass_3k(x: &[f64], fs: f64) -> Vec<f64> {
    let fc = 3_000.0_f64.min(0.45 * fs);
    let k = (PI * fc / fs).tan();
    let mut y = x.to_vec();
    for &q in &EXT_BUTTERWORTH_Q {
        let norm = 1.0 / (1.0 + k / q + k * k);
        let b0 = k * k * norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k * k - 1.0) * norm;
        let a2 = (1.0 - k / q + k * k) * norm;
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

/// Synthesize the default internal-DTF template pair from our own snowman HRTF model.
///
/// This is the pinna-free substitution described in the module docs
/// (TEMPLATE_LIMITED): the anechoic externalized response of the production renderer's
/// physical head/torso model at `az_deg`, elevation 0, and the default 2 m source
/// distance. It is not a listener-specific HRTF and cannot reproduce the paper's
/// pinna-driven MSS contrasts; use [`externalization_with_template_ir`] with a
/// pinna-bearing pair when a paper-faithful prediction is required.
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
        assert_eq!(ext_abs_norm(&a, &within, 1.0), 0.0);
        let beyond = [12.0, 12.0, 14.0, 16.0]; // 2 dB on one band
        assert!(ext_abs_norm(&a, &beyond, 1.0) > 0.0);
    }

    /// The 2021 paper evaluates exactly seven cues, and the WSM uses only MSS/ISS.
    #[test]
    fn seven_cue_model_matches_paper() {
        assert_eq!(EXT_WEIGHTS.len(), 7);
        assert_eq!(EXT_SENSITIVITY.len(), 7);
        assert!((EXT_WEIGHTS.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(EXT_WEIGHTS[0], 0.6);
        assert_eq!(EXT_WEIGHTS[1], 0.4);
        assert!(EXT_WEIGHTS[2..].iter().all(|&w| w == 0.0));
        let errs = CueErrors {
            mss: 1.0,
            iss: 2.0,
            mssd: 3.0,
            issd: 4.0,
            itit: 5.0,
            ic: 6.0,
            mi: 7.0,
        };
        assert_eq!(errs.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    /// Eq. 7 normalizes by `|MSG_template|` with no 0.5 floor. A 0.2-rad template
    /// gradient with a 1.2-rad excess gives exactly 6.0; the removed floor gave 2.4.
    #[test]
    fn eq7_mss_uses_the_template_gradient_without_a_floor() {
        let template = [0.2, 0.2];
        let target = [1.4, 1.4];
        let d = ext_abs_norm(&target, &template, 1.0);
        assert!((d - 6.0).abs() < 1e-12, "Eq. 7 denominator: got {d}");
        let within = [0.8, 0.2];
        assert_eq!(ext_abs_norm(&within, &template, 1.0), 0.0);
    }

    /// Eqs. 1/2 divide by the template magnitude, not by the JND.
    #[test]
    fn eq1_eq2_normalize_by_the_template() {
        assert!((ext_jnd_ratio(70.0, 60.0, 1.0) - 10.0 / 60.0).abs() < 1e-12);
        assert!((ext_jnd_ratio(0.5, 1.0, 0.0) - 0.5).abs() < 1e-12);
    }

    /// Eq. 5 divides by the signed template ITD/IID. With `ITD_template = -200 µs` and
    /// `IID_template = +5 dB` the exact error is 1.5; absolute denominators give 4.5.
    #[test]
    fn eq5_itit_preserves_signed_template_denominators() {
        let d = ext_itit(100.0, -200.0, -10.0, 5.0);
        assert!(
            (d - 1.5).abs() < 1e-12,
            "Eq. 5 signed denominators: got {d}"
        );
    }

    /// The Eq. 5 ITD input is a 4th-order Butterworth low-pass: -3.01 dB at 3 kHz and a
    /// 24 dB/oct asymptotic slope. A 4th-order Linkwitz-Riley cascade (`Q = 1/sqrt(2)`
    /// twice) would be -6.02 dB at 3 kHz.
    #[test]
    fn butterworth_lowpass_is_minus_3db_at_3k_with_24db_per_octave() {
        let fs = 192_000.0;
        let n = 65_536;
        let mut impulse = vec![0.0_f64; n];
        impulse[0] = 1.0;
        let y = ext_lowpass_3k(&impulse, fs);
        let bins = crate::fft::rfft(&y);
        let bin_hz = fs / n as f64;
        let mag_db = |f: f64| {
            let k = (f / bin_hz).round() as usize;
            20.0 * bins[k].norm().max(1e-300).log10()
        };
        assert!(
            (mag_db(3_000.0) + 3.010_3).abs() < 0.05,
            "Butterworth must be -3.01 dB at fc, got {:.3} dB",
            mag_db(3_000.0)
        );
        let slope = mag_db(12_000.0) - mag_db(6_000.0);
        assert!(
            (slope + 24.0).abs() < 1.0,
            "4th-order slope must be ~24 dB/oct, got {slope:.2}"
        );
    }

    /// Test-only RBJ notch biquad (direct form II transposed), used to synthesize an
    /// illustrative pinna-like template. This is NOT a measured HRTF and is not used to
    /// claim a paper value; it demonstrates only that a template with pinna-like
    /// spectral contrasts moves MSS in the documented direction.
    fn notch_biquad(x: &[f64], fs: f64, f0: f64, q: f64) -> Vec<f64> {
        let w0 = 2.0 * PI * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b0 = 1.0 / a0;
        let b1 = -2.0 * w0.cos() / a0;
        let b2 = b0;
        let a1 = -2.0 * w0.cos() / a0;
        let a2 = (1.0 - alpha) / a0;
        let (mut z1, mut z2) = (0.0, 0.0);
        x.iter()
            .map(|&v| {
                let out = b0 * v + z1;
                z1 = b1 * v - a1 * out + z2;
                z2 = b2 * v - a2 * out;
                out
            })
            .collect()
    }

    /// The pluggable-template path is what removes the TEMPLATE_LIMITED limitation.
    /// With pinna-like notches in the template, the flat in-head delta develops a real
    /// Eq. 7 MSS error and the score drops below the documented gate.
    #[test]
    fn pinna_like_template_lowers_the_in_head_score() {
        let fs = 48_000u32;
        let az = 30.0;
        let (tpl_l, tpl_r) = ext_template_ir(4096, fs as f64, az);
        let notched_l = notch_biquad(
            &notch_biquad(&tpl_l, fs as f64, 5_000.0, 8.0),
            fs as f64,
            9_000.0,
            8.0,
        );
        let notched_r = notch_biquad(
            &notch_biquad(&tpl_r, fs as f64, 5_000.0, 8.0),
            fs as f64,
            9_000.0,
            8.0,
        );
        let mut dry_l = vec![0.0; 4096];
        dry_l[240] = 1.0;
        let dry_r = dry_l.clone();
        let smooth = externalization_with_template_ir(
            &dry_l,
            &dry_r,
            &tpl_l,
            &tpl_r,
            fs,
            az,
            DecisionStrategy::WeightedSum,
        );
        let pinna = externalization_with_template_ir(
            &dry_l,
            &dry_r,
            &notched_l,
            &notched_r,
            fs,
            az,
            DecisionStrategy::WeightedSum,
        );
        assert!(
            pinna.cue_errors.mss > smooth.cue_errors.mss * 1.5,
            "pinna-like notches must raise the MSS error ({:.4} vs {:.4})",
            pinna.cue_errors.mss,
            smooth.cue_errors.mss
        );
        assert!(
            pinna.score < smooth.score,
            "notched template ({:.3}) must lower the in-head score vs smooth ({:.3})",
            pinna.score,
            smooth.score
        );
        assert!(
            pinna.score < EXT_TEMPLATE_LIMITED_DRY_GATE,
            "pinna-like template must pass the non-paper-literal gate, got {:.3}",
            pinna.score
        );
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
            e_ext.score > 0.99,
            "template vs itself must be the maximum rating, got {:.3}",
            e_ext.score
        );
        let mut dry_l = vec![0.0; 4096];
        let mut dry_r = vec![0.0; 4096];
        dry_l[200] = 1.0;
        dry_r[200] = 1.0;
        let e_dry = baumgartner_majdak_externalization(&dry_l, &dry_r, fs);
        // The built-in template is pinna-free (TEMPLATE_LIMITED): the paper's ~0.2
        // benchmark cannot be reached, so the documented 0.60 gate applies. A
        // pinna-bearing template through externalization_with_template_ir is required
        // for the paper's value.
        assert!(
            e_dry.score < EXT_TEMPLATE_LIMITED_DRY_GATE,
            "in-head delta ({:.3}) must stay below the TEMPLATE_LIMITED gate ({:.2})",
            e_dry.score,
            EXT_TEMPLATE_LIMITED_DRY_GATE
        );
        assert!(
            e_ext.score > e_dry.score + 0.25,
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
        assert_eq!(wta.cue_ratings.len(), 7);
    }

    /// The WSM score is exactly the documented 0.6 MSS / 0.4 ISS weighted sum, so
    /// zeroing the dominant MSS weight changes the score by exactly its weighted
    /// contribution (the documented direction: removing a positive cue rating lowers
    /// the score).
    #[test]
    fn zeroing_a_cue_weight_moves_the_score_as_documented() {
        let fs = 48_000u32;
        let mut dry_l = vec![0.0; 4096];
        dry_l[200] = 1.0;
        let dry_r = dry_l.clone();
        let e = baumgartner_majdak_externalization(&dry_l, &dry_r, fs);
        let wsm: f64 = e
            .cue_ratings
            .iter()
            .zip(EXT_WEIGHTS.iter())
            .map(|(r, w)| r * w)
            .sum();
        assert!(
            (e.score - wsm).abs() < 1e-12,
            "score ({:.6}) must equal the weighted sum ({:.6})",
            e.score,
            wsm
        );
        // The in-head delta's MSS rating is strictly inside the scale, so the cue
        // contributes a positive amount that zeroing its weight removes.
        assert!(
            e.cue_ratings[0] > 0.0 && e.cue_ratings[0] < 1.0,
            "MSS rating must be strictly inside the scale, got {:.4}",
            e.cue_ratings[0]
        );
        let without_mss = wsm - EXT_WEIGHTS[0] * e.cue_ratings[0];
        assert!(
            without_mss < wsm,
            "zeroing the MSS weight ({:.6}) must lower the score below {:.6}",
            without_mss,
            wsm
        );
    }
}
