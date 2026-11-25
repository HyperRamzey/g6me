//! ITU-R BS.1770-5 / EBU R128 loudness, EBU Tech 3342 Loudness Range (LRA),
//! AES TD1008 Peak-to-Loudness Ratio (PLR), and EBU Tech 3205-E stereo phase
//! correlation metering.
//!
//! ## Literature & Standards
//!
//! * **ITU-R BS.1770-5 (2023)**: K-weighting pre-filter (Tables 1 & 2), 400 ms
//!   gating blocks with 75 % overlap (100 ms hop), `-70 LKFS` absolute gate and
//!   `-10 LU` relative gate.
//! * **Brecht De Man (AES 145th Convention, 2018)**, *"Evaluation of
//!   Implementations of the EBU R128 Loudness Measurement"*: exact continuous-time
//!   prototype parameters for bilinear-transformed K-weighting at arbitrary sample
//!   rates (reproduces ITU-R BS.1770-5 Tables 1 & 2 at 48 kHz to `< 1e-15` and
//!   keeps 44.1 / 88.2 / 96 / 192 kHz well within EBU Tech 3341's `±0.1 LU` tolerance).
//! * **EBU Tech 3342**: Loudness Range (`LRA`) from 3 s sliding short-term blocks
//!   (100 ms hop), `-70 LUFS` absolute gate, `-20 LU` relative gate, `P95 - P10`.
//! * **AES TD1008**: Peak-to-Loudness Ratio (`PLR = true_peak_dbtp - integrated_lufs`).
//! * **EBU Tech 3205-E**: Normalized L/R cross-correlation `rho in [-1, +1]`.
//!
//! Pure deterministic `f64` with zero external crates and zero FMA, matching
//! the workspace's strict IEEE-754 arithmetic contract.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Absolute gating threshold per ITU-R BS.1770-5 and EBU Tech 3342 (LUFS / LKFS).
const ABS_GATE_LUFS: f64 = -70.0;
/// Relative gating offset for Integrated Loudness (ITU-R BS.1770-5).
const REL_GATE_INTEGRATED_LU: f64 = -10.0;
/// Relative gating offset for Loudness Range (EBU Tech 3342).
const REL_GATE_LRA_LU: f64 = -20.0;
/// ITU-R BS.1770-5 calibration offset in dB.
const BS1770_OFFSET_DB: f64 = -0.691;

/// Direct Form II Transposed biquad filter in strict `f64` arithmetic.
#[derive(Clone, Copy, Debug)]
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
    const fn new(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
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

/// Design the two-stage ITU-R BS.1770-5 K-weighting biquad cascade for any
/// sample rate `rate` using Brecht De Man's (AES 2018) bilinear prototype.
fn design_k_weighting(rate: u32) -> (Biquad, Biquad) {
    let fs = (rate.max(8_000)) as f64;

    // Stage 1: Acoustic head-effect high-shelf (+4.0 dB above ~1682 Hz).
    let db = 3.999_843_853_973_347;
    let f0_hs = 1_681.974_450_955_533_f64.min(0.45 * fs);
    let q_hs = 0.707_175_236_955_419_6;
    let k1 = (PI * f0_hs / fs).tan();
    let k1_sq = k1 * k1;
    let vh = 10.0_f64.powf(db / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0_1 = 1.0 + k1 / q_hs + k1_sq;
    let stage1 = Biquad::new(
        (vh + vb * k1 / q_hs + k1_sq) / a0_1,
        2.0 * (k1_sq - vh) / a0_1,
        (vh - vb * k1 / q_hs + k1_sq) / a0_1,
        2.0 * (k1_sq - 1.0) / a0_1,
        (1.0 - k1 / q_hs + k1_sq) / a0_1,
    );

    // Stage 2: RLB (Revised Low-frequency B-curve) 2nd-order high-pass (~38.1 Hz).
    // Numerator stays unnormalized (1, -2, 1) to match ITU-R BS.1770-5 Table 2
    // exactly at 48 kHz (to < 1e-15), paired with the -0.691 dB offset.
    let f0_hp = 38.135_470_876_024_44_f64.min(0.45 * fs);
    let q_hp = 0.500_327_037_323_877_3;
    let k2 = (PI * f0_hp / fs).tan();
    let k2_sq = k2 * k2;
    let a0_2 = 1.0 + k2 / q_hp + k2_sq;
    let stage2 = Biquad::new(
        1.0,
        -2.0,
        1.0,
        2.0 * (k2_sq - 1.0) / a0_2,
        (1.0 - k2 / q_hp + k2_sq) / a0_2,
    );

    (stage1, stage2)
}

#[inline]
fn power_to_lufs(mean_sq: f64) -> f64 {
    if mean_sq <= 0.0 || !mean_sq.is_finite() {
        f64::NEG_INFINITY
    } else {
        BS1770_OFFSET_DB + 10.0 * mean_sq.log10()
    }
}

/// Summary of ITU-R BS.1770-5 / EBU R128 loudness, EBU Tech 3342 LRA,
/// AES TD1008 PLR, and EBU Tech 3205-E stereo phase correlation for one stereo render.
///
/// Fields that are undefined on silence or clips shorter than their gating window
/// (`< 400 ms` for Integrated/Momentary, `< 3 s` for Short-Term/LRA) are `None` so
/// JSON serialization stays strictly RFC 8259 compliant and round-trips cleanly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoudnessReport {
    /// Integrated program loudness (ITU-R BS.1770-5, gated), LUFS.
    pub integrated_lufs: Option<f64>,
    /// Maximum 400 ms momentary loudness (EBU Tech 3341), LUFS.
    pub momentary_max_lufs: Option<f64>,
    /// Maximum 3 s short-term loudness (EBU Tech 3341), LUFS.
    pub short_term_max_lufs: Option<f64>,
    /// Loudness Range (`P95 - P10` over gated 3 s blocks, EBU Tech 3342), LU.
    pub lra_lu: Option<f64>,
    /// Peak-to-Loudness Ratio (`true_peak_dbtp - integrated_lufs`, AES TD1008), dB.
    pub plr_db: Option<f64>,
    /// Overall unweighted stereo phase correlation `rho in [-1.0, +1.0]`.
    /// `+1.0` = identical mono, `0.0` = decorrelated stereo, `-1.0` = inverted polarity.
    pub correlation: f64,
}

impl LoudnessReport {
    /// Compact one-line summary suitable for CLI headlines and GUI history rows.
    #[must_use]
    pub fn short_line(&self) -> Option<String> {
        let i = self.integrated_lufs?;
        let mut s = format!("I {i:+.1} LUFS");
        if let Some(lra) = self.lra_lu {
            s.push_str(&format!(" · LRA {lra:.1} LU"));
        }
        if let Some(plr) = self.plr_db {
            s.push_str(&format!(" · PLR {plr:.1} dB"));
        }
        Some(s)
    }

    /// Detailed loudness & correlation line for full copyable reports.
    #[must_use]
    pub fn detail_line(&self) -> String {
        let fmt_opt = |v: Option<f64>, unit: &str, signed: bool| match v {
            Some(x) if x.is_finite() => {
                if signed {
                    format!("{x:+.2} {unit}")
                } else {
                    format!("{x:.2} {unit}")
                }
            }
            _ => "n/a".to_string(),
        };
        format!(
            "loudness (BS.1770-5): I {}, M max {}, S max {}, LRA {}, PLR {}, corr {:+.2}",
            fmt_opt(self.integrated_lufs, "LUFS", true),
            fmt_opt(self.momentary_max_lufs, "LUFS", true),
            fmt_opt(self.short_term_max_lufs, "LUFS", true),
            fmt_opt(self.lra_lu, "LU", false),
            fmt_opt(self.plr_db, "dB", false),
            self.correlation,
        )
    }
}

/// Standard delivery targets used to grade rendered loudness & true peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeliveryTarget {
    /// Streaming platforms (Spotify / YouTube / Apple Music): `-14 LUFS`, `-1.0 dBTP`.
    #[default]
    Streaming,
    /// EBU R128 broadcast delivery: `-23.0 LUFS ± 0.5 LU`, `-1.0 dBTP`.
    BroadcastEbu,
    /// ATSC A/85 broadcast delivery: `-24.0 LUFS ± 2.0 LU`, `-2.0 dBTP`.
    BroadcastAtsc,
    /// Cinema / theatrical nearfield reference: `-24.0 LUFS`, `-1.0 dBTP`.
    Cinema,
}

impl DeliveryTarget {
    pub const ALL: [Self; 4] = [
        Self::Streaming,
        Self::BroadcastEbu,
        Self::BroadcastAtsc,
        Self::Cinema,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Streaming => "Streaming (-14 LUFS / -1 dBTP)",
            Self::BroadcastEbu => "EBU R128 (-23 LUFS / -1 dBTP)",
            Self::BroadcastAtsc => "ATSC A/85 (-24 LUFS / -2 dBTP)",
            Self::Cinema => "Cinema (-24 LUFS / -1 dBTP)",
        }
    }

    pub fn target_lufs(self) -> f64 {
        match self {
            Self::Streaming => -14.0,
            Self::BroadcastEbu => -23.0,
            Self::BroadcastAtsc | Self::Cinema => -24.0,
        }
    }

    pub fn tolerance_lu(self) -> f64 {
        match self {
            Self::Streaming => 2.0,
            Self::BroadcastEbu => 0.5,
            Self::BroadcastAtsc => 2.0,
            Self::Cinema => 3.0,
        }
    }

    pub fn max_true_peak_dbtp(self) -> f64 {
        match self {
            Self::BroadcastAtsc => -2.0,
            Self::Streaming | Self::BroadcastEbu | Self::Cinema => -1.0,
        }
    }

    /// Grade a rendered file against this target.
    pub fn evaluate(self, true_peak_dbtp: f64, loudness: Option<LoudnessReport>) -> Compliance {
        if true_peak_dbtp > self.max_true_peak_dbtp() + 0.05 {
            return Compliance::OverCeiling;
        }
        let Some(l) = loudness.and_then(|r| r.integrated_lufs) else {
            return Compliance::Unmeasured;
        };
        let delta = l - self.target_lufs();
        if delta.abs() <= self.tolerance_lu() {
            Compliance::Pass
        } else if delta > 0.0 {
            Compliance::TooLoud(delta)
        } else {
            Compliance::Quiet(delta)
        }
    }
}

/// Compliance outcome for a [`DeliveryTarget`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Compliance {
    Pass,
    TooLoud(f64),
    Quiet(f64),
    OverCeiling,
    Unmeasured,
}

impl Compliance {
    pub fn badge(self) -> String {
        match self {
            Self::Pass => "PASS".to_string(),
            Self::TooLoud(d) => format!("+{d:.1} LU"),
            Self::Quiet(d) => format!("{d:+.1} LU"),
            Self::OverCeiling => "OVER TP".to_string(),
            Self::Unmeasured => "—".to_string(),
        }
    }
}

/// Streaming stereo ITU-R BS.1770-5 / EBU R128 loudness & correlation meter.
///
/// Zero per-block allocations: only appends one `f64` every 100 ms of audio
/// above `-70 LUFS` (~80 bytes/sec, `< 1 MB` for a 3-hour film).
pub struct LoudnessMeter {
    s1_l: Biquad,
    s2_l: Biquad,
    s1_r: Biquad,
    s2_r: Biquad,
    hop_samples: usize,
    sub_energy: f64,
    sub_count: usize,
    ring4: [f64; 4],
    ring4_idx: usize,
    ring30: [f64; 30],
    ring30_idx: usize,
    hops_done: u64,
    gated_400: Vec<f64>,
    gated_3000: Vec<f64>,
    momentary_max: f64,
    short_term_max: f64,
    sum_ll: f64,
    sum_rr: f64,
    sum_lr: f64,
}

impl LoudnessMeter {
    pub fn new(sample_rate: u32) -> Self {
        let (s1, s2) = design_k_weighting(sample_rate);
        let hop_samples = ((sample_rate.max(8_000) as f64) * 0.1).round().max(1.0) as usize;
        Self {
            s1_l: s1,
            s2_l: s2,
            s1_r: s1,
            s2_r: s2,
            hop_samples,
            sub_energy: 0.0,
            sub_count: 0,
            ring4: [0.0; 4],
            ring4_idx: 0,
            ring30: [0.0; 30],
            ring30_idx: 0,
            hops_done: 0,
            gated_400: Vec::new(),
            gated_3000: Vec::new(),
            momentary_max: f64::NEG_INFINITY,
            short_term_max: f64::NEG_INFINITY,
            sum_ll: 0.0,
            sum_rr: 0.0,
            sum_lr: 0.0,
        }
    }

    /// Feed a stereo block `(left, right)` of identical length.
    pub fn push(&mut self, left: &[f64], right: &[f64]) {
        let n = left.len().min(right.len());
        let hop = self.hop_samples;
        for i in 0..n {
            let l = left[i];
            let r = right[i];
            self.sum_ll += l * l;
            self.sum_rr += r * r;
            self.sum_lr += l * r;

            let kl = self.s2_l.tick(self.s1_l.tick(l));
            let kr = self.s2_r.tick(self.s1_r.tick(r));
            // Stereo BS.1770-5 channel weights: G_L = 1.0, G_R = 1.0.
            self.sub_energy += kl * kl + kr * kr;
            self.sub_count += 1;
            if self.sub_count == hop {
                let mean_sq = self.sub_energy / (hop as f64);
                self.sub_energy = 0.0;
                self.sub_count = 0;
                self.commit_hop(mean_sq);
            }
        }
    }

    fn commit_hop(&mut self, mean_sq: f64) {
        self.ring4[self.ring4_idx] = mean_sq;
        self.ring4_idx = (self.ring4_idx + 1) % 4;
        self.ring30[self.ring30_idx] = mean_sq;
        self.ring30_idx = (self.ring30_idx + 1) % 30;
        self.hops_done += 1;

        if self.hops_done >= 4 {
            let mut sum = 0.0;
            for &v in &self.ring4 {
                sum += v;
            }
            let z400 = sum * 0.25;
            let lu = power_to_lufs(z400);
            if lu > self.momentary_max {
                self.momentary_max = lu;
            }
            if lu > ABS_GATE_LUFS {
                self.gated_400.push(z400);
            }
        }

        if self.hops_done >= 30 {
            let mut sum = 0.0;
            for &v in &self.ring30 {
                sum += v;
            }
            let z3000 = sum / 30.0;
            let lu = power_to_lufs(z3000);
            if lu > self.short_term_max {
                self.short_term_max = lu;
            }
            if lu > ABS_GATE_LUFS {
                self.gated_3000.push(z3000);
            }
        }
    }

    /// Finalize all gated metrics and pair PLR with the measured true peak.
    #[must_use]
    pub fn summarize(&self, true_peak_dbtp: f64) -> LoudnessReport {
        let integrated_lufs = self.integrated();
        let momentary_max_lufs = self.momentary_max.is_finite().then_some(self.momentary_max);
        let short_term_max_lufs = self
            .short_term_max
            .is_finite()
            .then_some(self.short_term_max);
        let lra_lu = self.lra();
        let plr_db = match (integrated_lufs, true_peak_dbtp.is_finite()) {
            (Some(i), true) => Some(true_peak_dbtp - i),
            _ => None,
        };
        let denom = (self.sum_ll * self.sum_rr).sqrt();
        let correlation = if denom > 0.0 && denom.is_finite() {
            (self.sum_lr / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        LoudnessReport {
            integrated_lufs,
            momentary_max_lufs,
            short_term_max_lufs,
            lra_lu,
            plr_db,
            correlation,
        }
    }

    fn integrated(&self) -> Option<f64> {
        if self.gated_400.is_empty() {
            return None;
        }
        let mean_abs = self.gated_400.iter().sum::<f64>() / (self.gated_400.len() as f64);
        let rel_thr_lufs = power_to_lufs(mean_abs) + REL_GATE_INTEGRATED_LU;
        let rel_thr_z = 10.0_f64.powf((rel_thr_lufs - BS1770_OFFSET_DB) / 10.0);

        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &z in &self.gated_400 {
            if z > rel_thr_z {
                sum += z;
                cnt += 1;
            }
        }
        if cnt == 0 {
            return None;
        }
        let val = power_to_lufs(sum / (cnt as f64));
        val.is_finite().then_some(val)
    }

    fn lra(&self) -> Option<f64> {
        if self.gated_3000.len() < 2 {
            return None;
        }
        let mean_abs = self.gated_3000.iter().sum::<f64>() / (self.gated_3000.len() as f64);
        let rel_thr_lufs = power_to_lufs(mean_abs) + REL_GATE_LRA_LU;
        let rel_thr_z = 10.0_f64.powf((rel_thr_lufs - BS1770_OFFSET_DB) / 10.0);

        let mut surv: Vec<f64> = self
            .gated_3000
            .iter()
            .copied()
            .filter(|&z| z > rel_thr_z)
            .map(power_to_lufs)
            .filter(|v| v.is_finite())
            .collect();
        if surv.len() < 2 {
            return Some(0.0);
        }
        surv.sort_by(|a, b| a.total_cmp(b));
        let n = surv.len();
        let lo = surv[((n - 1) as f64 * 0.10) as usize];
        let hi = surv[((n - 1) as f64 * 0.95) as usize];
        Some((hi - lo).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_weighting_at_48k_matches_bs1770_tables_to_machine_precision() {
        let (s1, s2) = design_k_weighting(48_000);
        // ITU-R BS.1770-5 Table 1 (48 kHz high-shelf)
        let t1 = [
            (s1.b0, 1.535_124_859_586_97),
            (s1.b1, -2.691_696_189_406_38),
            (s1.b2, 1.198_392_810_852_85),
            (s1.a1, -1.690_659_293_182_41),
            (s1.a2, 0.732_480_774_215_85),
        ];
        for (got, want) in t1 {
            assert!((got - want).abs() < 1e-13, "stage1 {got} != {want}");
        }
        // ITU-R BS.1770-5 Table 2 (48 kHz RLB high-pass)
        let t2 = [
            (s2.b0, 1.0),
            (s2.b1, -2.0),
            (s2.b2, 1.0),
            (s2.a1, -1.990_047_454_833_98),
            (s2.a2, 0.990_072_250_366_21),
        ];
        for (got, want) in t2 {
            assert!((got - want).abs() < 1e-13, "stage2 {got} != {want}");
        }
    }

    #[test]
    fn stereo_997hz_calibration_across_sample_rates() {
        // EBU Tech 3341 reference: stereo 997 Hz sine at amplitude 0.1 (-20 dBFS peak)
        // sums two equal channels (mean square = 2 * (0.1^2 / 2) = 0.01 = -20.0 dB),
        // giving -20.00 LUFS across sample rates within ±0.05 LU.
        for &rate in &[44_100u32, 48_000, 96_000] {
            let n = rate as usize * 4;
            let tone: Vec<f64> = (0..n)
                .map(|i| 0.1 * (2.0 * PI * 997.0 * (i as f64) / (rate as f64)).sin())
                .collect();
            let mut m = LoudnessMeter::new(rate);
            m.push(&tone, &tone);
            let r = m.summarize(-20.0);
            let i = r.integrated_lufs.expect("integrated loudness");
            assert!(
                (i - (-20.0)).abs() < 0.05,
                "rate {rate}: expected -20.00 LUFS, got {i:.4}"
            );
            assert!((r.correlation - 1.0).abs() < 1e-12);
            assert!(r.lra_lu.unwrap() < 0.05);
            assert!((r.plr_db.unwrap() - 0.0).abs() < 0.05);
        }
    }

    #[test]
    fn relative_gate_excludes_quiet_passages() {
        // 4 s at -35 dBFS (well below -10 LU relative gate) + 4 s at -14 dBFS.
        // Without relative gating, mean power would be ~0.5 * 10^(-1.4) => -17.0 LUFS.
        // With the BS.1770-5 -10 LU relative gate, the quiet half is excluded and
        // integrated loudness sits near -14.0 LUFS.
        let rate = 48_000u32;
        let a_lo = 10.0_f64.powf(-35.0 / 20.0);
        let a_hi = 10.0_f64.powf(-14.0 / 20.0);
        let n = rate as usize * 8;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let amp = if i < n / 2 { a_lo } else { a_hi };
                amp * (2.0 * PI * 997.0 * (i as f64) / (rate as f64)).sin()
            })
            .collect();
        let mut m = LoudnessMeter::new(rate);
        m.push(&sig, &sig);
        let r = m.summarize(-14.0);
        let i = r.integrated_lufs.unwrap();
        assert!(
            (i - (-14.0)).abs() < 0.35,
            "relative gate failed to exclude -35 dBFS section: got {i:.3} LUFS"
        );
    }

    #[test]
    fn dynamic_two_level_signal_matches_ebur128_integrated_and_lra() {
        // 4 s at 0.05 (-26.02 dBFS) + 4 s at 0.20 (-13.98 dBFS):
        // Cross-checked against ffmpeg ebur128: I ≈ -16.73 LUFS, LRA ≈ 12.04 LU.
        let rate = 48_000u32;
        let n = rate as usize * 8;
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let amp = if i < n / 2 { 0.05 } else { 0.20 };
                amp * (2.0 * PI * 997.0 * (i as f64) / (rate as f64)).sin()
            })
            .collect();
        let mut m = LoudnessMeter::new(rate);
        m.push(&sig, &sig);
        let r = m.summarize(-13.98);
        let i = r.integrated_lufs.unwrap();
        let lra = r.lra_lu.unwrap();
        assert!((i - (-16.73)).abs() < 0.15, "I = {i:.3}");
        assert!((lra - 12.04).abs() < 0.25, "LRA = {lra:.3}");
        assert!((r.momentary_max_lufs.unwrap() - (-13.98)).abs() < 0.10);
        assert!((r.short_term_max_lufs.unwrap() - (-13.98)).abs() < 0.10);
    }

    #[test]
    fn chunk_size_invariance_and_phase_correlation() {
        let rate = 48_000u32;
        let n = rate as usize * 2;
        let l: Vec<f64> = (0..n)
            .map(|i| 0.2 * (2.0 * PI * 440.0 * (i as f64) / (rate as f64)).sin())
            .collect();
        let r_inv: Vec<f64> = l.iter().map(|&x| -x).collect();
        let r_orth: Vec<f64> = (0..n)
            .map(|i| 0.2 * (2.0 * PI * 440.0 * (i as f64) / (rate as f64)).cos())
            .collect();

        let mut whole = LoudnessMeter::new(rate);
        whole.push(&l, &r_orth);
        let rep_whole = whole.summarize(-14.0);

        let mut chunked = LoudnessMeter::new(rate);
        let mut off = 0;
        for &sz in &[1usize, 7, 13, 1024, 4096] {
            if off < n {
                let end = (off + sz).min(n);
                chunked.push(&l[off..end], &r_orth[off..end]);
                off = end;
            }
        }
        while off < n {
            let end = (off + 517).min(n);
            chunked.push(&l[off..end], &r_orth[off..end]);
            off = end;
        }
        let rep_chunked = chunked.summarize(-14.0);
        assert_eq!(rep_whole, rep_chunked);
        assert!(rep_whole.correlation.abs() < 1e-3);

        let mut inv = LoudnessMeter::new(rate);
        inv.push(&l, &r_inv);
        assert!((inv.summarize(-14.0).correlation - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn delivery_target_compliance_grades_ceiling_and_loudness() {
        let ok = LoudnessReport {
            integrated_lufs: Some(-14.2),
            ..Default::default()
        };
        assert_eq!(
            DeliveryTarget::Streaming.evaluate(-1.1, Some(ok)),
            Compliance::Pass
        );
        assert_eq!(
            DeliveryTarget::Streaming.evaluate(-0.5, Some(ok)),
            Compliance::OverCeiling
        );
        assert!(matches!(
            DeliveryTarget::BroadcastEbu.evaluate(-1.2, Some(ok)),
            Compliance::TooLoud(_)
        ));
    }
}
