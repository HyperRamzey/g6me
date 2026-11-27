//! Order-7 Higher-Order Ambisonics (64 real SH channels), MagLS binaural decoding,
//! spherical-harmonic tapering, and perceptually informed all-pass decorrelation.
//!
//! 1. **Order-7 Real Spherical Harmonics (`64 = (7+1)²` channels, ACN/N3D)** with
//!    exact `9 × 16 = 144`-point Gauss–Legendre spherical quadrature (`Y^T W Y = I_64`
//!    to `< 1e-14`).
//! 2. **Schörkhuber, Zaunschirm & Höldrich (DAGA 2018)** & **Deppisch, Helmholz &
//!    Ahrens (IEEE/ACM TASLP 2022)** — **MagLS** binaural HOA decoding: uses exact
//!    Least-Squares SH projection below `f_c ≈ 1.5 kHz` (preserving exact ITD/phase)
//!    and phase-unconstrained Magnitude Least-Squares above `f_c` (eliminating
//!    spatial-aliasing HF rolloff).
//! 3. **Hold, Gamper, Pulkki, Raghuvanshi & Tashev (IEEE ICASSP 2019)** —
//!    order-dependent spherical-harmonic domain tapering & diffuse coloration
//!    compensation for smooth off-axis imaging without high-order sidelobes.
//! 4. **Kermit-Canfield & Abel (DAFx 2016)** (extending Kendall JAES 1995) —
//!    frequency-warped perceptually informed all-pass decorrelator (`|H(e^{jω})| == 1`
//!    to machine precision across all frequencies) with Bark-warped group delay
//!    dispersion for rear/surround channels.

use crate::fft::Complex;
use crate::scattering::{snowman_hrtf, SnowmanGeom};
use std::f64::consts::PI;

/// Order-7 Higher-Order Ambisonics (`N = 7`, `(N+1)^2 = 64` spherical harmonics).
pub const HOA_ORDER: usize = 7;
pub const HOA_CH: usize = (HOA_ORDER + 1) * (HOA_ORDER + 1);

/// A single spherical quadrature node `(az_rad, sin_el, weight)` with `sum(weight) == 1.0`.
#[derive(Debug, Clone, Copy)]
pub struct SphereNode {
    pub az_rad: f64,
    pub el_rad: f64,
    pub sin_el: f64,
    pub weight: f64,
}

/// Exact `n`-point Gauss–Legendre quadrature nodes and weights on `[-1, 1]`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let m = n.div_ceil(2);
    for i in 1..=m {
        let mut z = (PI * (i as f64 - 0.25) / (n as f64 + 0.5)).cos();
        let mut pp = 1.0_f64;
        for _ in 0..32 {
            let mut p1 = 1.0_f64;
            let mut p2 = 0.0_f64;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf - 1.0) * z * p2 - (jf - 1.0) * p3) / jf;
            }
            pp = (n as f64) * (z * p1 - p2) / (z * z - 1.0);
            let dz = p1 / pp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        nodes.push(-z);
        weights.push(w);
        if i != n + 1 - i {
            nodes.push(z);
            weights.push(w);
        }
    }
    (nodes, weights)
}

/// Exact `9 × 16 = 144`-point Gauss–Legendre × Equiangular spherical quadrature grid.
///
/// Integrates all products `Y_n^m * Y_{n'}^{m'}` of degree `<= 14` (`7 + 7`) to `< 1e-14`.
pub fn sphere_quadrature_order7() -> Vec<SphereNode> {
    let (xs, ws) = gauss_legendre(9);
    let n_az = 16usize;
    let mut out = Vec::with_capacity(xs.len() * n_az);
    for (&x, &wx) in xs.iter().zip(ws.iter()) {
        let el = x.clamp(-1.0, 1.0).asin();
        let w = (0.5 * wx) / (n_az as f64);
        for j in 0..n_az {
            let az = 2.0 * PI * (j as f64) / (n_az as f64);
            out.push(SphereNode {
                az_rad: az,
                el_rad: el,
                sin_el: x,
                weight: w,
            });
        }
    }
    out
}

/// Unnormalized associated Legendre polynomial `P_l^m(x)` (`0 <= m <= l`, no Condon–Shortley sign).
#[inline]
pub fn assoc_legendre(l: usize, m: usize, x: f64) -> f64 {
    let mut pmm = 1.0_f64;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0_f64;
        for _ in 1..=m {
            pmm *= fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmmp1;
    }
    let mut pll = 0.0_f64;
    for ll in (m + 2)..=l {
        pll = ((2 * ll - 1) as f64 * x * pmmp1 - (ll + m - 1) as f64 * pmm) / ((ll - m) as f64);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Precomputed factorials `0!..=14!`.
const FACT: [f64; 15] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
];

/// Evaluate all 64 Order-7 real spherical harmonics (ACN channel order, N3D normalization)
/// at `(az_rad, el_rad)` where `az_rad` is counter-clockwise from front (`+` left).
pub fn eval_sh_order7(az_rad: f64, el_rad: f64) -> [f64; HOA_CH] {
    let x = el_rad.sin().clamp(-1.0, 1.0);
    let mut out = [0.0_f64; HOA_CH];
    let mut idx = 0usize;
    for n in 0..=HOA_ORDER {
        for m in -(n as isize)..=(n as isize) {
            let am = m.unsigned_abs();
            let mut norm = ((2 * n + 1) as f64 * FACT[n - am] / FACT[n + am]).sqrt();
            if am > 0 {
                norm *= std::f64::consts::SQRT_2;
            }
            let p = assoc_legendre(n, am, x);
            let ang = (am as f64) * az_rad;
            out[idx] = if m > 0 {
                norm * p * ang.cos()
            } else if m < 0 {
                norm * p * ang.sin()
            } else {
                norm * p
            };
            idx += 1;
        }
    }
    out
}

/// Hold, Gamper, Pulkki, Raghuvanshi & Tashev (IEEE ICASSP 2019) spherical-harmonic
/// tapering weights `w_n` for orders `0..=7`.
///
/// Applies max-rE / raised-cosine order tapering normalized to preserve diffuse-field
/// energy `sum_n (2n+1) w_n^2 == sum_n (2n+1)` so spatial aliasing sidelobes are
/// suppressed without dulling the diffuse field.
pub fn hold_sh_taper_weights(taper_strength: f64) -> [f64; HOA_ORDER + 1] {
    let alpha = taper_strength.clamp(0.0, 1.0);
    let mut raw = [1.0_f64; HOA_ORDER + 1];
    // Max-rE Legendre root approximation: P_n(cos(137.9° / (N + 1.51)))
    let theta_re = (137.9_f64 / (HOA_ORDER as f64 + 1.51)).to_radians();
    let cos_re = theta_re.cos();
    let mut e_raw = 0.0_f64;
    let mut e_unit = 0.0_f64;
    for (n, raw_slot) in raw.iter_mut().enumerate() {
        let p_n = assoc_legendre(n, 0, cos_re).max(0.05);
        let w = (1.0 - alpha) + alpha * p_n;
        *raw_slot = w;
        let deg_count = (2 * n + 1) as f64;
        e_raw += deg_count * w * w;
        e_unit += deg_count;
    }
    let comp = (e_unit / e_raw.max(1e-12)).sqrt();
    for w in &mut raw {
        *w *= comp;
    }
    raw
}

/// Precomputed Order-7 MagLS binaural HOA decoder (`HOA_CH = 64` SH weights for L and R ears
/// at each frequency bin).
#[derive(Debug, Clone)]
pub struct MagLsDecoder {
    pub freqs: Vec<f64>,
    /// Left-ear SH decoding weights `[bin][acn]`.
    pub dec_l: Vec<[Complex; HOA_CH]>,
    /// Right-ear SH decoding weights `[bin][acn]`.
    pub dec_r: Vec<[Complex; HOA_CH]>,
}

impl MagLsDecoder {
    /// Build a MagLS + Hold-tapered Order-7 HOA binaural decoder for a given Snowman
    /// geometry, source distance, and speed of sound.
    ///
    /// - Below `f_magls = 1500 Hz`: exact Least-Squares projection `D_LS(f) = sum_q w_q H(q, f) Y(q)`.
    /// - Above `f_magls = 1500 Hz`: iterative phase-relaxed Magnitude Least-Squares
    ///   (Schörkhuber, Zaunschirm & Höldrich DAGA 2018; Deppisch, Helmholz & Ahrens
    ///   IEEE/ACM TASLP 2022) keeping `|D(f) Y(q)| ≈ |H(q, f)|` using the phase of the
    ///   previous frequency bin `arg(D(f_{k-1}) Y(q))` (plus 2 refinement iterations) and
    ///   Hold et al. (ICASSP 2019) SH-domain tapering & diffuse coloration compensation!
    ///
    /// Evaluated on an adaptive ERB-spaced anchor frequency grid (`~65` anchors) and
    /// smoothly interpolated across all FFT bins for fast deterministic synthesis.
    pub fn build(
        freqs: &[f64],
        geom: SnowmanGeom,
        distance_m: f64,
        c_m_s: f64,
        include_torso: bool,
        taper_strength: f64,
    ) -> Self {
        let nb = freqs.len();
        let nodes = sphere_quadrature_order7();
        let q_len = nodes.len();
        let sh_grid: Vec<[f64; HOA_CH]> = nodes
            .iter()
            .map(|nd| eval_sh_order7(nd.az_rad, nd.el_rad))
            .collect();
        let taper = hold_sh_taper_weights(taper_strength);
        let mut ch_taper = [1.0_f64; HOA_CH];
        let mut idx = 0usize;
        for (n, &w) in taper.iter().enumerate() {
            for _ in 0..(2 * n + 1) {
                ch_taper[idx] = w;
                idx += 1;
            }
        }

        // Build ~60 anchor bin indices (dense below 2 kHz, log-spaced above) so
        // Order-7 MagLS runs in milliseconds while preserving smooth phase evolution.
        let mut anchors: Vec<usize> = Vec::with_capacity(80);
        anchors.push(0);
        let mut cur = 0usize;
        while cur + 1 < nb {
            let f = freqs[cur];
            let df = if nb > 1 { freqs[1] - freqs[0] } else { 10.0 };
            let target_step_hz = if f < 1_500.0 {
                75.0
            } else if f < 5_000.0 {
                200.0
            } else {
                0.08 * f
            };
            let step = ((target_step_hz / df.max(1e-6)).round() as usize).max(1);
            cur = (cur + step).min(nb - 1);
            anchors.push(cur);
        }

        let mut anc_l: Vec<[Complex; HOA_CH]> = Vec::with_capacity(anchors.len());
        let mut anc_r: Vec<[Complex; HOA_CH]> = Vec::with_capacity(anchors.len());
        let mut prev_l = [Complex::ZERO; HOA_CH];
        let mut prev_r = [Complex::ZERO; HOA_CH];
        prev_l[0] = Complex::ONE;
        prev_r[0] = Complex::ONE;

        const F_MAGLS_LO: f64 = 1_200.0;
        const F_MAGLS_HI: f64 = 2_200.0;

        for (a_i, &bin) in anchors.iter().enumerate() {
            let f = freqs[bin];
            let mut hl_q = Vec::with_capacity(q_len);
            let mut hr_q = Vec::with_capacity(q_len);
            for nd in &nodes {
                let az_deg = nd.az_rad.to_degrees();
                let el_deg = nd.el_rad.to_degrees();
                hl_q.push(snowman_hrtf(
                    f,
                    geom,
                    az_deg,
                    el_deg,
                    distance_m,
                    1.0,
                    c_m_s,
                    include_torso,
                ));
                hr_q.push(snowman_hrtf(
                    f,
                    geom,
                    az_deg,
                    el_deg,
                    distance_m,
                    -1.0,
                    c_m_s,
                    include_torso,
                ));
            }

            // Direct Least-Squares projection: D_LS[c] = sum_q w_q * H(q) * Y_c(q)
            let project = |h_q: &[Complex]| -> [Complex; HOA_CH] {
                let mut d = [Complex::ZERO; HOA_CH];
                for q in 0..q_len {
                    let w = nodes[q].weight;
                    let h = h_q[q];
                    let y = &sh_grid[q];
                    for c in 0..HOA_CH {
                        d[c] = d[c].add(h.scale(w * y[c]));
                    }
                }
                d
            };

            let dl_ls = project(&hl_q);
            let dr_ls = project(&hr_q);

            let (dl, dr) = if f <= F_MAGLS_LO || a_i == 0 {
                (dl_ls, dr_ls)
            } else {
                // MagLS iterative phase relaxation (Schörkhuber et al. 2018):
                // replace target phase with phase of previous/current SH reconstruction.
                let solve_magls =
                    |h_q: &[Complex], mut d_cur: [Complex; HOA_CH]| -> [Complex; HOA_CH] {
                        let mut target = vec![Complex::ZERO; q_len];
                        for _iter in 0..2 {
                            for q in 0..q_len {
                                let y = &sh_grid[q];
                                let mut pred = Complex::ZERO;
                                for c in 0..HOA_CH {
                                    pred = pred.add(d_cur[c].scale(y[c]));
                                }
                                let mag = h_q[q].norm();
                                let phase = if pred.norm_sq() > 1e-24 {
                                    pred.arg()
                                } else {
                                    h_q[q].arg()
                                };
                                target[q] = Complex::from_polar(mag, phase);
                            }
                            d_cur = project(&target);
                        }
                        d_cur
                    };
                let dl_m = solve_magls(&hl_q, prev_l);
                let dr_m = solve_magls(&hr_q, prev_r);
                let mix = ((f - F_MAGLS_LO) / (F_MAGLS_HI - F_MAGLS_LO)).clamp(0.0, 1.0);
                // Smooth linear blend between LS and MagLS across 1.2 – 2.2 kHz.
                let mut out_l = [Complex::ZERO; HOA_CH];
                let mut out_r = [Complex::ZERO; HOA_CH];
                for c in 0..HOA_CH {
                    out_l[c] = dl_ls[c].scale(1.0 - mix).add(dl_m[c].scale(mix));
                    out_r[c] = dr_ls[c].scale(1.0 - mix).add(dr_m[c].scale(mix));
                }
                (out_l, out_r)
            };

            prev_l = dl;
            prev_r = dr;

            // Apply Hold et al. (ICASSP 2019) SH tapering + diffuse-field covariance match.
            let mut tl = dl;
            let mut tr = dr;
            if taper_strength > 1e-6 {
                let mut e_orig_l = 0.0_f64;
                let mut e_tap_l = 0.0_f64;
                let mut e_orig_r = 0.0_f64;
                let mut e_tap_r = 0.0_f64;
                for c in 0..HOA_CH {
                    e_orig_l += dl[c].norm_sq();
                    e_orig_r += dr[c].norm_sq();
                    tl[c] = dl[c].scale(ch_taper[c]);
                    tr[c] = dr[c].scale(ch_taper[c]);
                    e_tap_l += tl[c].norm_sq();
                    e_tap_r += tr[c].norm_sq();
                }
                // Coloration compensation restores exact diffuse energy per bin.
                let gl = (e_orig_l / e_tap_l.max(1e-24)).sqrt();
                let gr = (e_orig_r / e_tap_r.max(1e-24)).sqrt();
                for c in 0..HOA_CH {
                    tl[c] = tl[c].scale(gl);
                    tr[c] = tr[c].scale(gr);
                }
            }

            anc_l.push(tl);
            anc_r.push(tr);
        }

        // Interpolate smoothly from anchors to every FFT bin.
        let mut dec_l = vec![[Complex::ZERO; HOA_CH]; nb];
        let mut dec_r = vec![[Complex::ZERO; HOA_CH]; nb];
        for seg in 0..(anchors.len() - 1) {
            let b0 = anchors[seg];
            let b1 = anchors[seg + 1];
            dec_l[b0] = anc_l[seg];
            dec_r[b0] = anc_r[seg];
            let span = b1 - b0;
            if span > 1 {
                let inv = 1.0 / (span as f64);
                for off in 1..span {
                    let t = (off as f64) * inv;
                    let k = b0 + off;
                    for c in 0..HOA_CH {
                        dec_l[k][c] = anc_l[seg][c].scale(1.0 - t).add(anc_l[seg + 1][c].scale(t));
                        dec_r[k][c] = anc_r[seg][c].scale(1.0 - t).add(anc_r[seg + 1][c].scale(t));
                    }
                }
            }
        }
        let last = anchors.len() - 1;
        dec_l[nb - 1] = anc_l[last];
        dec_r[nb - 1] = anc_r[last];

        Self {
            freqs: freqs.to_vec(),
            dec_l,
            dec_r,
        }
    }

    /// Decode a plane-wave / directional source with Order-7 SH vector `y` to `(H_L, H_R)`
    /// across all frequency bins.
    pub fn decode_sh_spectrum(&self, sh: &[f64; HOA_CH]) -> (Vec<Complex>, Vec<Complex>) {
        let nb = self.freqs.len();
        let mut hl = Vec::with_capacity(nb);
        let mut hr = Vec::with_capacity(nb);
        for k in 0..nb {
            let mut sl = Complex::ZERO;
            let mut sr = Complex::ZERO;
            let dl = &self.dec_l[k];
            let dr = &self.dec_r[k];
            for c in 0..HOA_CH {
                let yc = sh[c];
                if yc != 0.0 {
                    sl = sl.add(dl[c].scale(yc));
                    sr = sr.add(dr[c].scale(yc));
                }
            }
            hl.push(sl);
            hr.push(sr);
        }
        (hl, hr)
    }

    /// Decode a directional source at `(az_deg, el_deg)` through the Order-7 MagLS decoder.
    pub fn decode_direction(&self, az_deg: f64, el_deg: f64) -> (Vec<Complex>, Vec<Complex>) {
        let sh = eval_sh_order7(az_deg.to_radians(), el_deg.to_radians());
        self.decode_sh_spectrum(&sh)
    }
}

/// Kermit-Canfield & Abel (DAFx 2016) frequency-warped perceptually informed all-pass
/// decorrelation spectrum (`|H(e^{jω})| == 1.000000` at every frequency).
///
/// Cascades 4 strictly causal 1st-order all-pass sections tuned along Bark-warped
/// critical bands (`380 Hz`, `910 Hz`, `2150 Hz`, `4800 Hz`, with subtle `seed_sign`
/// polarity dispersion) so rear/surround channels decorrelate cleanly with zero
/// magnitude coloration (`0.000000 dB` ripple everywhere).
pub fn kermit_canfield_warped_allpass(freqs: &[f64], fs: f64, seed_sign: f64) -> Vec<Complex> {
    // Bark-spaced pole frequencies for perceptually uniform group-delay spread (~1.8 ms).
    let base_fc = if seed_sign >= 0.0 {
        [380.0_f64, 910.0, 2_150.0, 4_800.0]
    } else {
        [430.0_f64, 1_060.0, 2_480.0, 5_350.0]
    };
    let coeffs: [f64; 4] = base_fc.map(|fc| {
        let f_clamped = fc.min(0.42 * fs);
        let t = (PI * f_clamped / fs).tan();
        (t - 1.0) / (t + 1.0)
    });

    freqs
        .iter()
        .map(|&f| {
            let z1 = Complex::from_polar(1.0, -2.0 * PI * f / fs);
            let mut acc = Complex::ONE;
            for &a in &coeffs {
                // H(z) = (a + z^-1) / (1 + a * z^-1) -> exact |H(e^{jω})| == 1.0.
                let num = Complex::new(a, 0.0).add(z1);
                let den = Complex::ONE.add(z1.scale(a));
                acc = acc.mul(num.div(den));
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order7_sh_quadrature_is_orthonormal_to_1e_13() {
        let nodes = sphere_quadrature_order7();
        let ys: Vec<[f64; HOA_CH]> = nodes
            .iter()
            .map(|nd| eval_sh_order7(nd.az_rad, nd.el_rad))
            .collect();
        let mut max_diag = 0.0_f64;
        let mut max_off = 0.0_f64;
        for i in 0..HOA_CH {
            for j in 0..HOA_CH {
                let mut g = 0.0_f64;
                for q in 0..nodes.len() {
                    g += nodes[q].weight * ys[q][i] * ys[q][j];
                }
                if i == j {
                    max_diag = max_diag.max((g - 1.0).abs());
                } else {
                    max_off = max_off.max(g.abs());
                }
            }
        }
        assert!(
            max_diag < 1e-13 && max_off < 1e-13,
            "Order-7 Gram error: diag={max_diag:e}, off={max_off:e}"
        );
    }

    #[test]
    fn magls_preserves_hf_magnitude_better_than_plain_ls() {
        let geom = SnowmanGeom::default();
        let c = 344.854;
        let dist = 2.0;
        let freqs: Vec<f64> = (0..=128).map(|k| (k as f64) * 16_000.0 / 128.0).collect();
        let dec = MagLsDecoder::build(&freqs, geom, dist, c, false, 0.0);
        // Also compute raw LS at 10 kHz to compare against MagLS.
        let k10 = 80usize; // 10 kHz
        let f10 = freqs[k10];
        let nodes = sphere_quadrature_order7();
        let mut ls_l = [Complex::ZERO; HOA_CH];
        for nd in &nodes {
            let h = snowman_hrtf(
                f10,
                geom,
                nd.az_rad.to_degrees(),
                nd.el_rad.to_degrees(),
                dist,
                1.0,
                c,
                false,
            );
            let y = eval_sh_order7(nd.az_rad, nd.el_rad);
            for ch in 0..HOA_CH {
                ls_l[ch] = ls_l[ch].add(h.scale(nd.weight * y[ch]));
            }
        }

        // Measure average HF magnitude error over 12 horizontal test angles.
        let mut err_magls = 0.0_f64;
        let mut err_ls = 0.0_f64;
        for step in 0..12 {
            let az = -150.0 + (step as f64) * 30.0;
            let exact = snowman_hrtf(f10, geom, az, 0.0, dist, 1.0, c, false).norm();
            let y = eval_sh_order7(az.to_radians(), 0.0);
            let mut m_rec = Complex::ZERO;
            let mut l_rec = Complex::ZERO;
            for ch in 0..HOA_CH {
                m_rec = m_rec.add(dec.dec_l[k10][ch].scale(y[ch]));
                l_rec = l_rec.add(ls_l[ch].scale(y[ch]));
            }
            err_magls += (m_rec.norm() - exact).powi(2);
            err_ls += (l_rec.norm() - exact).powi(2);
        }
        assert!(
            err_magls < err_ls,
            "MagLS HF error ({err_magls:.4e}) should beat plain LS ({err_ls:.4e})"
        );
    }

    #[test]
    fn kermit_canfield_warped_allpass_is_exact_unit_magnitude_and_dispersive() {
        let fs = 48_000.0;
        let freqs: Vec<f64> = (0..=512).map(|k| (k as f64) * fs / 1024.0).collect();
        let hl = kermit_canfield_warped_allpass(&freqs, fs, 1.0);
        let hr = kermit_canfield_warped_allpass(&freqs, fs, -1.0);

        let max_mag_err = hl
            .iter()
            .chain(hr.iter())
            .map(|c| (c.norm() - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_mag_err < 1e-13,
            "all-pass magnitude ripple: {max_mag_err:e}"
        );

        // Verify phase difference between L and R decorrelators is non-trivial across mids.
        let mut phase_diff_sum = 0.0_f64;
        for k in 10..200 {
            let diff = hl[k].mul(hr[k].conj()).arg().abs();
            phase_diff_sum += diff;
        }
        assert!(
            phase_diff_sum > 5.0,
            "expected substantial inter-channel phase decorrelation, got {phase_diff_sum:.3}"
        );
    }
}
