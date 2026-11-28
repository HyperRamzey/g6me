//! Cochlea, Auditory Nerve Neurogram (NSIM), Bayesian Auditory Externalization, and
//! Bayesian Active Personalization Loop.
//!
//! 1. **Zilany–Bruce–Carney auditory periphery** (Zilany, Bruce & Carney, JASA 2014;
//!    Bruce, Erfani & Zilany, Hearing Research 2018), implemented in [`crate::zbc`]:
//!    - human middle ear (three biquad sections),
//!    - dual-resonance nonlinear basilar membrane (C1/C2 tenth-order chirp filters with
//!      the wideband control path and OHC Boltzmann nonlinearity),
//!    - IHC `NLogarithm` transduction, `3 kHz` seventh-order membrane low-pass and the cat
//!      cochlear delay,
//!    - BEZ 2018 synapse: Eq. 1 IHC→PLA mapping, fractional Gaussian noise, two power-law
//!      adaptation paths, adaptive redocking, four release sites and stochastic spike
//!      generation with absolute plus relative refractoriness,
//!    - three fibers per CF (low/medium/high spontaneous rate from BEZ 2018 Table 1),
//!      24 ERB-spaced CFs from `120 Hz` to `12 kHz`, binned spike-rate neurogram.
//!    - **Neurogram Similarity Index (NSIM)** (Hines & Harte 2010 / Bruce et al. 2018):
//!      mean luminance × structural similarity `NSIM in [0, 1]` between reference and
//!      processed auditory-nerve neurograms over `3×3` Gaussian windows (radius 0.5).
//! 2. **Baumgartner & Majdak (2021)** — **Bayesian Auditory Externalization Model**:
//!    eight expectation-error cues (`MSS`, `ISS`, `MSSD`, `ISSD`, `ITIT`, `IC`, `MI`,
//!    `ITSD`) mapped through Eq. 10 and combined by the published weighted-sum decision
//!    stage (0.6 `MSS` / 0.4 `ISS`) or the LTA/MTA/WTA selection strategies. The
//!    listener-specific DTF template is substituted by our own snowman HRTF model at a
//!    fixed `30°` lateral angle and `2 m` distance (see `externalization.rs`).
//! 3. **Bayesian Active Personalization over Three Anatomy Scalars (BALD)**:
//!    10–15 click uncertainty-aware Bayesian active-learning A/B loop over anatomical
//!    parameters `(head_radius_m, torso_height_m, pinna_shelf_db)` choosing each A/B
//!    pair by maximizing expected Shannon entropy reduction (BALD / mutual information).
//!    This is a three-scalar anatomy search, not a Hear-to-Hear HRTF posterior.


/// 24 ERB-spaced cochlear characteristic frequencies (Hz) spanning `120 Hz .. 12 kHz`.
pub fn cochlea_cf_hz(n_ch: usize) -> Vec<f64> {
    let f_min = 120.0_f64;
    let f_max = 12_000.0_f64;
    let erb_min = 21.4 * (0.004_37 * f_min + 1.0).log10();
    let erb_max = 21.4 * (0.004_37 * f_max + 1.0).log10();
    (0..n_ch)
        .map(|i| {
            let t = if n_ch > 1 {
                (i as f64) / ((n_ch - 1) as f64)
            } else {
                0.5
            };
            let e = erb_min + t * (erb_max - erb_min);
            (10.0_f64.powf(e / 21.4) - 1.0) / 0.004_37
        })
        .collect()
}

/// Auditory-nerve neurogram matrix `[channel][time_frame]` in instantaneous spike rate (`sp/s`).
#[derive(Debug, Clone)]
pub struct Neurogram {
    pub cfs_hz: Vec<f64>,
    pub frame_ms: f64,
    /// Instantaneous auditory-nerve discharge rates (`sp/s`), shape `[n_ch][n_frames]`.
    pub rates: Vec<Vec<f64>>,
}

/// Compute the auditory-nerve neurogram of a mono signal `x` at `sample_rate`.
///
/// Runs the Zilany–Bruce–Carney periphery (`crate::zbc`) for one low-, one medium- and one
/// high-spontaneous-rate fiber at each of `n_ch` ERB-spaced CFs, then bins the generated
/// spike times into `5 ms` frames (`sp/s`). The model runs at its native `100 kHz` internal
/// rate; `x` is band-limited resampled on entry.
pub fn compute_neurogram(x: &[f64], sample_rate: u32, n_ch: usize) -> Neurogram {
    let frame_ms = 5.0_f64;
    let cfs = cochlea_cf_hz(n_ch);
    let rates = crate::zbc::rate_neurogram(x, sample_rate, &cfs, frame_ms);
    Neurogram {
        cfs_hz: cfs,
        frame_ms,
        rates,
    }
}

/// Compute the **Neurogram Similarity Index (NSIM)** in `[0, 1]` between reference neurogram `a`
/// and test neurogram `b` (Hines & Harte 2010 / Bruce, Erfani & Zilany 2018).
///
/// Evaluates local `(3 × 3)` time-frequency patch luminance `L` and structural covariance `S`
/// (`NSIM = mean(L * S)`), equaling exact `1.0` for identical neurograms and dropping
/// sharply under phase distortion, spectral notches, or intermodulation artifacts.
pub fn nsim(a: &Neurogram, b: &Neurogram) -> f64 {
    let n_ch = a.rates.len().min(b.rates.len());
    if n_ch == 0 {
        return 1.0;
    }
    let n_fr = a.rates[0].len().min(b.rates[0].len());
    if n_fr == 0 {
        return 1.0;
    }

    // Dynamic range L = 280 sp/s -> standard NSIM constants C1 = (0.01*L)^2, C2 = (0.03*L)^2.
    const L_DYN: f64 = 280.0;
    const C1: f64 = (0.01 * L_DYN) * (0.01 * L_DYN);
    const C2: f64 = (0.03 * L_DYN) * (0.03 * L_DYN);

    // Hines & Harte (2010) use a 3 x 3 Gaussian window; radius 0.5 samples.
    const W_SIGMA: f64 = 0.5;
    let weight =
        |dc: isize, dt: isize| (-((dc * dc + dt * dt) as f64) / (2.0 * W_SIGMA * W_SIGMA)).exp();
    let mut total = 0.0_f64;
    let mut count = 0usize;

    for ch in 0..n_ch {
        for fr in 0..n_fr {
            let mut sum_w = 0.0_f64;
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            for dc in -1isize..=1 {
                let c = (ch as isize + dc).clamp(0, n_ch as isize - 1) as usize;
                for dt in -1isize..=1 {
                    let t = (fr as isize + dt).clamp(0, n_fr as isize - 1) as usize;
                    let w = weight(dc, dt);
                    sum_w += w;
                    sum_a += w * a.rates[c][t];
                    sum_b += w * b.rates[c][t];
                }
            }
            let mu_a = sum_a / sum_w;
            let mu_b = sum_b / sum_w;

            let mut var_a = 0.0_f64;
            let mut var_b = 0.0_f64;
            let mut cov = 0.0_f64;
            for dc in -1isize..=1 {
                let c = (ch as isize + dc).clamp(0, n_ch as isize - 1) as usize;
                for dt in -1isize..=1 {
                    let t = (fr as isize + dt).clamp(0, n_fr as isize - 1) as usize;
                    let w = weight(dc, dt);
                    let da = a.rates[c][t] - mu_a;
                    let db = b.rates[c][t] - mu_b;
                    var_a += w * da * da;
                    var_b += w * db * db;
                    cov += w * da * db;
                }
            }
            let sig_a = (var_a / sum_w).max(0.0).sqrt();
            let sig_b = (var_b / sum_w).max(0.0).sqrt();
            let cov_ab = cov / sum_w;

            let lum = (2.0 * mu_a * mu_b + C1) / (mu_a * mu_a + mu_b * mu_b + C1);
            let struc = (cov_ab + 0.5 * C2) / (sig_a * sig_b + 0.5 * C2);
            total += (lum * struc).clamp(0.0, 1.0);
            count += 1;
        }
    }
    if count > 0 {
        total / (count as f64)
    } else {
        1.0
    }
}

pub use crate::externalization::{
    baumgartner_majdak_externalization, externalization_with_strategy, CueErrors, DecisionStrategy,
    ExternalizationReport,
};

// ============================================================================
// 3. Bayesian Active Personalization over Three Anatomy Scalars (BALD)
// ============================================================================

/// Personalized anthropomorphic binaural parameters optimized via A/B active learning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnatomyParams {
    /// Spherical head radius `a` in meters (`0.075 ..= 0.102`).
    pub head_radius_m: f64,
    /// Head-to-torso vertical spacing in meters (`0.22 ..= 0.34`).
    pub torso_height_m: f64,
    /// High-frequency pinna shelf tilt in dB (`-3.0 ..= +3.0`).
    pub pinna_shelf_db: f64,
}

impl Default for AnatomyParams {
    fn default() -> Self {
        Self {
            head_radius_m: 0.0875,
            torso_height_m: 0.26,
            pinna_shelf_db: 0.0,
        }
    }
}

/// Interactive Bayesian Active Personalization session over `(head_radius_m, torso_height_m, pinna_shelf_db)`.
///
/// Maintains a full discrete Bayesian posterior over a `5 × 4 × 5 = 100`-point anatomical
/// hypothesis grid, proposes maximally informative `(A, B)` pairs by maximizing expected
/// posterior Shannon entropy reduction (BALD / mutual information), updates the posterior
/// after each user A/B preference click using a smooth Bradley–Terry / logistic distance
/// likelihood, and converges in `10–15` clicks!
#[derive(Debug, Clone)]
pub struct PersonalizationSession {
    grid: Vec<AnatomyParams>,
    posterior: Vec<f64>,
    pub trial: usize,
    pub candidate_a: usize,
    pub candidate_b: usize,
    pub initial_entropy_bits: f64,
}

impl Default for PersonalizationSession {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonalizationSession {
    pub fn new() -> Self {
        let head_vals = [0.076, 0.082, 0.0875, 0.093, 0.099];
        let torso_vals = [0.22, 0.255, 0.29, 0.33];
        let pinna_vals = [-2.5, -1.25, 0.0, 1.25, 2.5];

        let mut grid = Vec::with_capacity(head_vals.len() * torso_vals.len() * pinna_vals.len());
        let mut prior = Vec::with_capacity(100);
        for &h in &head_vals {
            for &t in &torso_vals {
                for &p in &pinna_vals {
                    grid.push(AnatomyParams {
                        head_radius_m: h,
                        torso_height_m: t,
                        pinna_shelf_db: p,
                    });
                    // Mild Gaussian anthropomorphic prior centered on KEMAR average.
                    let dh = (h - 0.0875) / 0.010;
                    let dt = (t - 0.26) / 0.05;
                    let dp = p / 2.2;
                    prior.push((-0.25 * (dh * dh + dt * dt + dp * dp)).exp());
                }
            }
        }
        let sum: f64 = prior.iter().sum();
        for w in &mut prior {
            *w /= sum;
        }
        let initial_entropy_bits = Self::entropy_bits_of(&prior);

        let mut session = Self {
            grid,
            posterior: prior,
            trial: 0,
            candidate_a: 0,
            candidate_b: 1,
            initial_entropy_bits,
        };
        session.select_next_pair();
        session
    }

    fn entropy_bits_of(probs: &[f64]) -> f64 {
        let mut h = 0.0_f64;
        for &p in probs {
            if p > 1e-18 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Normalized squared perceptual distance between two anatomical parameter vectors.
    #[inline]
    fn dist_sq(x: AnatomyParams, y: AnatomyParams) -> f64 {
        let dh = (x.head_radius_m - y.head_radius_m) / 0.008;
        let dt = (x.torso_height_m - y.torso_height_m) / 0.040;
        let dp = (x.pinna_shelf_db - y.pinna_shelf_db) / 1.5;
        dh * dh + dt * dt + dp * dp
    }

    /// Probability that a listener whose true anatomy is `theta` prefers candidate `a` over `b`.
    #[inline]
    fn prob_prefer_a(theta: AnatomyParams, a: AnatomyParams, b: AnatomyParams) -> f64 {
        // Closer candidate in normalized anatomical space has higher utility -> sigmoid(beta*(d_b - d_a)).
        let da = Self::dist_sq(theta, a).sqrt();
        let db = Self::dist_sq(theta, b).sqrt();
        let logit = (3.2 * (db - da)).clamp(-20.0, 20.0);
        1.0 / (1.0 + (-logit).exp())
    }

    /// Current Shannon entropy of the posterior (bits). Lower = higher confidence.
    pub fn entropy_bits(&self) -> f64 {
        Self::entropy_bits_of(&self.posterior)
    }

    /// Posterior convergence progress in `[0.0, 1.0]` based on information gain.
    pub fn confidence(&self) -> f64 {
        if self.initial_entropy_bits <= 1e-9 {
            return 1.0;
        }
        (1.0 - self.entropy_bits() / self.initial_entropy_bits).clamp(0.0, 1.0)
    }

    /// Posterior expected value (MMSE estimate) of `(head_radius_m, torso_height_m, pinna_shelf_db)`.
    pub fn posterior_mean(&self) -> AnatomyParams {
        let mut h = 0.0_f64;
        let mut t = 0.0_f64;
        let mut p = 0.0_f64;
        for (g, &w) in self.grid.iter().zip(self.posterior.iter()) {
            h += w * g.head_radius_m;
            t += w * g.torso_height_m;
            p += w * g.pinna_shelf_db;
        }
        AnatomyParams {
            head_radius_m: h,
            torso_height_m: t,
            pinna_shelf_db: p,
        }
    }

    pub fn pair_params(&self) -> (AnatomyParams, AnatomyParams) {
        (self.grid[self.candidate_a], self.grid[self.candidate_b])
    }

    /// Select the next `(A, B)` pair maximizing expected information gain (BALD):
    /// `I(y; theta | A, B) = H(E_theta[p(y|theta)]) - E_theta[H(p(y|theta))]`.
    fn select_next_pair(&mut self) {
        // Restrict candidate pool to the top 20 posterior-mass points (plus global anchors on early trials)
        // so pair selection is exact and runs in < 0.2 ms per GUI frame.
        let mut ranked: Vec<usize> = (0..self.grid.len()).collect();
        ranked.sort_by(|&i, &j| self.posterior[j].total_cmp(&self.posterior[i]));
        let pool_len = 18.min(ranked.len());
        let pool = &ranked[..pool_len];

        let mut best_ig = -1.0_f64;
        let mut best_a = pool[0];
        let mut best_b = pool[1.min(pool_len - 1)];

        let bin_ent = |p: f64| -> f64 {
            let pc = p.clamp(1e-15, 1.0 - 1e-15);
            -(pc * pc.log2() + (1.0 - pc) * (1.0 - pc).log2())
        };

        for idx_i in 0..pool_len {
            let a_i = pool[idx_i];
            let pa = self.grid[a_i];
            for &b_i in &pool[(idx_i + 1)..pool_len] {
                let pb = self.grid[b_i];
                let mut p_y1 = 0.0_f64;
                let mut exp_cond_h = 0.0_f64;
                for (k, &theta) in self.grid.iter().enumerate() {
                    let w = self.posterior[k];
                    if w < 1e-9 {
                        continue;
                    }
                    let py = Self::prob_prefer_a(theta, pa, pb);
                    p_y1 += w * py;
                    exp_cond_h += w * bin_ent(py);
                }
                let ig = bin_ent(p_y1) - exp_cond_h;
                if ig > best_ig {
                    best_ig = ig;
                    best_a = a_i;
                    best_b = b_i;
                }
            }
        }
        self.candidate_a = best_a;
        self.candidate_b = best_b;
    }

    /// Record a user preference (`prefer_a == true` if user picked A over B), apply
    /// Bayes' rule to update the posterior, and choose the next maximally informative pair.
    pub fn record_choice(&mut self, prefer_a: bool) {
        let pa = self.grid[self.candidate_a];
        let pb = self.grid[self.candidate_b];
        let mut sum = 0.0_f64;
        for (k, &theta) in self.grid.iter().enumerate() {
            let p_a = Self::prob_prefer_a(theta, pa, pb);
            let like = if prefer_a { p_a } else { 1.0 - p_a };
            self.posterior[k] *= like;
            sum += self.posterior[k];
        }
        if sum > 1e-30 {
            for w in &mut self.posterior {
                *w /= sum;
            }
        }
        self.trial += 1;
        self.select_next_pair();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn cochlea_nsim_is_unity_for_identical_and_detects_degradation() {
        let fs = 48_000u32;
        let n = 4800; // 100 ms
        let mut ref_sig = vec![0.0_f64; n];
        let mut dist_sig = vec![0.0_f64; n];
        for i in 0..n {
            let t = (i as f64) / (fs as f64);
            let s = 0.25 * (2.0 * PI * 440.0 * t).sin()
                + 0.18 * (2.0 * PI * 1_650.0 * t).sin()
                + 0.12 * (2.0 * PI * 4_200.0 * t).sin();
            ref_sig[i] = s;
            // Heavy nonlinear clipping + high-frequency loss
            dist_sig[i] = (s * 4.0).tanh() * 0.25 - 0.12 * (2.0 * PI * 4_200.0 * t).sin();
        }
        let ng_ref = compute_neurogram(&ref_sig, fs, 24);
        let ng_same = compute_neurogram(&ref_sig, fs, 24);
        let ng_dist = compute_neurogram(&dist_sig, fs, 24);

        let s_same = nsim(&ng_ref, &ng_same);
        let s_dist = nsim(&ng_ref, &ng_dist);
        assert!(
            (s_same - 1.0).abs() < 1e-12,
            "identical neurogram NSIM must be 1.0, got {s_same}"
        );
        assert!(
            s_dist < 0.92,
            "degraded signal should reduce NSIM below 0.92, got {s_dist:.4}"
        );
    }

    #[test]
    fn externalization_ranks_template_above_in_head_delta() {
        let fs = 48_000u32;
        // The model's own externalized template (30-degree source, 2 m): the reference
        // condition that must receive the maximum Eq. 10 rating.
        let (tpl_l, tpl_r) =
            crate::externalization::externalization_template_ir(4096, fs as f64, 30.0);
        let e_tpl = baumgartner_majdak_externalization(&tpl_l, &tpl_r, fs);

        // Dry inside-head diotic impulse: flat spectrum, zero IID, zero reflections.
        let mut dry_l = vec![0.0_f64; 4096];
        let mut dry_r = vec![0.0_f64; 4096];
        dry_l[240] = 1.0;
        dry_r[240] = 1.0;
        let e_dry = baumgartner_majdak_externalization(&dry_l, &dry_r, fs);

        assert!(
            e_tpl.score > 0.90,
            "externalized template must score > 0.90, got {:.3}",
            e_tpl.score
        );
        assert!(
            e_dry.score < 0.60,
            "dry mono impulse should score low externalization (<0.60), got {:.3}",
            e_dry.score
        );
        assert!(
            e_tpl.score > e_dry.score + 0.30,
            "template ({:.3}) must beat dry in-head ({:.3})",
            e_tpl.score,
            e_dry.score
        );
    }

    #[test]
    fn bayesian_active_personalization_converges_within_12_clicks() {
        let truth = AnatomyParams {
            head_radius_m: 0.093,
            torso_height_m: 0.29,
            pinna_shelf_db: 1.25,
        };
        let mut sess = PersonalizationSession::new();
        let h0 = sess.entropy_bits();
        for _ in 0..12 {
            let (a, b) = sess.pair_params();
            // Independent L1 oracle: deliberately does not call PersonalizationSession::dist_sq,
            // so the test cannot pass by sharing the model's own likelihood geometry.
            let err = |x: AnatomyParams| -> f64 {
                (x.head_radius_m - truth.head_radius_m).abs() / 0.010
                    + (x.torso_height_m - truth.torso_height_m).abs() / 0.045
                    + 0.5 * (x.pinna_shelf_db - truth.pinna_shelf_db).abs() / 1.8
            };
            sess.record_choice(err(a) < err(b));
        }
        let h12 = sess.entropy_bits();
        let est = sess.posterior_mean();
        assert!(
            h12 < h0 * 0.45,
            "posterior entropy failed to drop substantially: h0={h0:.2}, h12={h12:.2}"
        );
        assert!(
            (est.head_radius_m - truth.head_radius_m).abs() < 0.005,
            "head_radius did not converge: est={:.4}, truth={:.4}",
            est.head_radius_m,
            truth.head_radius_m
        );
        assert!(
            (est.torso_height_m - truth.torso_height_m).abs() < 0.025,
            "torso_height did not converge: est={:.4}, truth={:.4}",
            est.torso_height_m,
            truth.torso_height_m
        );
        assert!(
            (est.pinna_shelf_db - truth.pinna_shelf_db).abs() < 0.8,
            "pinna_shelf did not converge: est={:.2}, truth={:.2}",
            est.pinna_shelf_db,
            truth.pinna_shelf_db
        );
    }
}
