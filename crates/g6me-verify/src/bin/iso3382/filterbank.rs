//! Octave-band analysis bank: an exact power partition realized as causal
//! minimum-phase filters.
//!
//! **Magnitude design.** Bands are placed at the exact base-2 ISO 266 octave
//! centres (`31.25 Hz … 16 kHz`, labelled with the nominal `31.5 Hz … 16 kHz`
//! values) and are separated by a power-complementary crossfade in log2(f):
//! adjacent band powers `P` satisfy `P(t) + P(1 - t) = 1`, so
//! `Σ_bands W_b(f)² = 1` at every FFT bin from DC to Nyquist. The crossfade
//! shape is `P(t) = (1-t)³ / (t³ + (1-t)³)` for `t = |log2(f/f_c)| < 1`; it is
//! C² at the band joins, is exactly −3.01 dB at the nominal band edges
//! `f_c/√2` and `f_c·√2` of every interior edge, and stays inside the
//! IEC 61260-1 class-1 octave-band relative-attenuation limits at every
//! tabulated breakpoint (≤ 0.85 dB at 3/8 octave where class 1 allows 1.3 dB,
//! and 14.5 dB at 3/4 octave where class 1 requires 9.75 dB); its
//! filter-integrated response is +0.02 dB against the class-1 ±0.3 dB limit.
//! The outermost bands are extended flat to DC and to Nyquist so the bank
//! tiles the whole spectrum (without that fill the sub-15.6 Hz and
//! supra-16 kHz remainders would make the band energies sum to less than the
//! signal energy); this boundary treatment is why the 31.5 Hz band responds
//! to sub-audio content and the 16 kHz band to content above 22.6 kHz.
//!
//! **Causal realization.** Each band magnitude is factored by the real-cepstrum
//! minimum-phase method (Oppenheim & Schafer, *Discrete-Time Signal
//! Processing*, §12.5): the folded cepstrum of `ln|H|` gives a causal,
//! front-loaded impulse response whose DFT magnitude equals the designed mask.
//! A small amplitude floor `ε = 1e-6` (−120 dB) keeps the logarithm finite and
//! costs `≈ 1e-11` of extra band energy. The previous implementation used a
//! symmetric zero-phase mask and cropped the inverse transform to
//! `[0, len)`, which discarded the negative-time half of the direct-sound
//! response: 49.0–49.9 % of the band energy of a direct impulse between 63 Hz
//! and 1 kHz, biasing C80, D50 and EDT. A causal filter keeps all of that
//! energy at t ≥ 0 and needs no time-origin shift.
//!
//! The bank is a verification-grade reference, not a certified IEC 61260-1
//! instrument: it matches the class-1 magnitude mask at the tabulated
//! breakpoints by design and is an exact energy partition, but no measurement
//! uncertainty analysis or conformance test per IEC 61260-2/-3 is performed.

use g6me_core::fft::{Complex, FftPlan};
use std::f64::consts::PI;

/// ISO 266 *nominal* octave-band centre frequencies, 31.5 Hz to 16 kHz.
/// These are labels; the filters are placed at the exact base-2 centres.
pub const OCTAVE_CENTERS_HZ: [f64; 10] = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

/// Exact base-2 octave centres used by the filter design (IEC 61260-1 Annex E
/// base-two exact mid-band frequencies, rounded to the nominal labels above).
const DESIGN_CENTERS_HZ: [f64; 10] = [
    31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Amplitude floor for the cepstral logarithm (−120 dB). The realized band
/// power is `W² + ε²` instead of `W²`, adding ≈1e-11 to the total band energy.
const MASK_FLOOR: f64 = 1e-6;

/// Octave-band edges `(f_centre/√2, f_centre·√2)` for a nominal centre.
pub fn octave_band_edges(centre_hz: f64) -> (f64, f64) {
    (centre_hz / SQRT_2, centre_hz * SQRT_2)
}

/// Crossfade power for a band at `t = |log2(f/f_c)|` octaves from its centre.
///
/// `power(0) = 1`, `power(1/2) = 1/2` (−3.01 dB), `power(1) = 0`, and the
/// complementary band at the same frequency carries `1 - power(t)`; the C²
/// shape keeps the class-1 passband tolerance at 1/4 and 3/8 octave.
fn crossfade_power(t: f64) -> f64 {
    if !t.is_finite() || t >= 1.0 {
        return 0.0;
    }
    let a = t.abs();
    let q = 1.0 - a;
    let p = q * q * q;
    p / (a * a * a + p)
}

/// Smallest power of two `>= n`, with a floor of 2 (the FFT plan's minimum).
pub fn next_pow2(n: usize) -> usize {
    let mut p = 2usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// FFT-domain octave filter bank for one signal length and sample rate.
pub struct OctaveBank {
    /// Zero-padded transform length (>= 2·len, so masking is linear convolution).
    n: usize,
    len: usize,
    fs: f64,
    plan: FftPlan,
    /// Inclusive one-sided bin range of non-zero weight for each supported band.
    ranges: [Option<(usize, usize)>; 10],
}

impl OctaveBank {
    pub fn new(len: usize, fs: f64) -> Self {
        let n = next_pow2(len.saturating_mul(2));
        let plan = FftPlan::new(n);
        let mut ranges = [None; 10];
        let nyquist_bin = n / 2;
        if fs > 0.0 {
            for (band, (slot, centre)) in ranges
                .iter_mut()
                .zip(DESIGN_CENTERS_HZ.iter())
                .enumerate()
            {
                let (lo, hi) = octave_band_edges(*centre);
                // The upper nominal edge must stay strictly below Nyquist,
                // otherwise the octave is not representable at this rate.
                if hi >= fs / 2.0 || lo <= 0.0 {
                    continue;
                }
                let k_hi = if band == 9 {
                    nyquist_bin
                } else {
                    ((hi * n as f64 / fs).floor() as usize).min(nyquist_bin)
                };
                let k_lo = if band == 0 {
                    0
                } else {
                    (lo * n as f64 / fs).ceil() as usize
                };
                if k_lo <= k_hi && k_lo <= nyquist_bin {
                    *slot = Some((k_lo, k_hi));
                }
            }
        }
        Self {
            n,
            len,
            fs,
            plan,
            ranges,
        }
    }

    /// Transform length used by this bank.
    pub fn fft_len(&self) -> usize {
        self.n
    }

    /// Inclusive FFT-bin range for a supported band, `None` if unsupported.
    pub fn band_bins(&self, band: usize) -> Option<(usize, usize)> {
        self.ranges.get(band).copied().flatten()
    }

    pub fn supported(&self, band: usize) -> bool {
        self.band_bins(band).is_some()
    }

    /// Design amplitude weight for bin `k` in `band`, on the unit-power
    /// partition: `Σ_bands band_weight(band, k)² = 1` for every `k`.
    pub fn band_weight(&self, band: usize, k: usize) -> f64 {
        if k > self.n / 2 || !self.fs.is_finite() || self.fs <= 0.0 {
            return 0.0;
        }
        let Some(&centre) = DESIGN_CENTERS_HZ.get(band) else {
            return 0.0;
        };
        let f = k as f64 * self.fs / self.n as f64;
        // Outside the outermost centres the bank is filled flat by the nearest
        // edge band, so the partition covers DC .. Nyquist exactly.
        if f < DESIGN_CENTERS_HZ[0] {
            return f64::from(u8::from(band == 0));
        }
        if f > DESIGN_CENTERS_HZ[9] {
            return f64::from(u8::from(band == 9));
        }
        let t = (f / centre).log2().abs();
        crossfade_power(t).sqrt()
    }

    /// Forward transform of a zero-padded copy of `x`.
    pub fn forward(&self, x: &[f64]) -> Vec<Complex> {
        let mut input = vec![0.0; self.n];
        let take = x.len().min(self.len);
        input[..take].copy_from_slice(&x[..take]);
        let mut scratch = vec![Complex::ZERO; self.n];
        let mut out = vec![Complex::ZERO; self.n / 2 + 1];
        self.plan.rfft(&input, &mut scratch, &mut out);
        out
    }

    /// Minimum-phase frequency response of `band`, from the real-cepstrum
    /// factor of the designed magnitude `sqrt(W² + ε²)`.
    ///
    /// The returned one-sided spectrum is the DFT of a causal real impulse
    /// response: multiplying a signal spectrum by it filters the signal with
    /// the designed magnitude and no acausal energy.
    pub fn min_phase_spectrum(&self, band: usize) -> Option<Vec<Complex>> {
        self.band_bins(band)?;
        let nb = self.n / 2 + 1;
        let mut logspec = vec![Complex::ZERO; nb];
        for (k, slot) in logspec.iter_mut().enumerate() {
            let w = self.band_weight(band, k);
            *slot = Complex::new(0.5 * (w * w + MASK_FLOOR * MASK_FLOOR).ln(), 0.0);
        }
        let mut scratch = vec![Complex::ZERO; self.n];
        // Real cepstrum of the log magnitude: even in quefrency.
        let mut cepstrum = vec![0.0; self.n];
        self.plan.irfft(&logspec, &mut scratch, &mut cepstrum);
        // Fold it onto the causal half: keep c[0] and c[N/2], double the rest.
        let mut folded = vec![0.0; self.n];
        folded[0] = cepstrum[0];
        for j in 1..self.n / 2 {
            folded[j] = 2.0 * cepstrum[j];
        }
        folded[self.n / 2] = cepstrum[self.n / 2];
        let mut logh = vec![Complex::ZERO; nb];
        self.plan.rfft(&folded, &mut scratch, &mut logh);
        let mut out = vec![Complex::ZERO; nb];
        for (dst, src) in out.iter_mut().zip(logh.iter()) {
            *dst = src.exp();
        }
        Some(out)
    }

    /// Apply a band kernel to a signal spectrum; the result is the full
    /// zero-padded linear-convolution output of length `n`.
    pub fn apply(&self, spec: &[Complex], kernel: &[Complex]) -> Vec<f64> {
        debug_assert_eq!(spec.len(), self.n / 2 + 1);
        debug_assert_eq!(kernel.len(), self.n / 2 + 1);
        let mut masked = vec![Complex::ZERO; self.n / 2 + 1];
        for ((slot, s), h) in masked.iter_mut().zip(spec.iter()).zip(kernel.iter()) {
            *slot = s.mul(*h);
        }
        let mut scratch = vec![Complex::ZERO; self.n];
        let mut out = vec![0.0; self.n];
        self.plan.irfft(&masked, &mut scratch, &mut out);
        out
    }

    /// Full-length band output for `spec` (convenience wrapper that builds the
    /// kernel on demand).
    pub fn inverse_band_full(&self, spec: &[Complex], band: usize) -> Option<Vec<f64>> {
        let kernel = self.min_phase_spectrum(band)?;
        Some(self.apply(spec, &kernel))
    }

    /// Band output truncated to the original signal length.
    pub fn inverse_band(&self, spec: &[Complex], band: usize) -> Option<Vec<f64>> {
        let mut out = self.inverse_band_full(spec, band)?;
        out.truncate(self.len);
        Some(out)
    }
}

#[cfg(test)]
fn filter_all(bank: &OctaveBank, x: &[f64]) -> Vec<Option<Vec<f64>>> {
    let spec = bank.forward(x);
    (0..OCTAVE_CENTERS_HZ.len())
        .map(|b| bank.inverse_band(&spec, b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn every_band_is_octave_wide() {
        for centre in DESIGN_CENTERS_HZ {
            let (lo, hi) = octave_band_edges(centre);
            assert!(
                (hi / lo - 2.0).abs() < 1e-12,
                "{} Hz band ratio is {}",
                centre,
                hi / lo
            );
            assert!(
                (lo * hi - centre * centre).abs() < 1e-9 * centre * centre,
                "{} Hz edges are not geometric about the centre",
                centre
            );
        }
        // The design centres are exactly one octave apart, which is what makes
        // the crossfade power-complementary without per-bin renormalization.
        for pair in DESIGN_CENTERS_HZ.windows(2) {
            assert_eq!(pair[1], 2.0 * pair[0]);
        }
    }

    #[test]
    fn crossfade_is_power_complementary_and_hits_the_nominal_edges() {
        for t in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875] {
            let p = crossfade_power(t);
            assert!(
                (p + crossfade_power(1.0 - t) - 1.0).abs() < 1e-15,
                "power at {t} and its complement do not sum to one"
            );
        }
        assert!((crossfade_power(0.0) - 1.0).abs() < 1e-15);
        assert!((crossfade_power(0.5) - 0.5).abs() < 1e-15);
        assert_eq!(crossfade_power(1.0), 0.0);
        // Class-1 margins, verified against ANSI S1.11-2004 Table 1 (the US
        // adoption of IEC 61260): at 1/4 octave class 1 allows 0.6 dB, at
        // 3/8 octave 1.3 dB, at 3/4 octave it requires at least 9.75 dB.
        let db = |t: f64| -10.0 * crossfade_power(t).log10();
        assert!(db(0.25) <= 0.6, "1/4-octave attenuation {} dB", db(0.25));
        assert!(db(0.375) <= 1.3, "3/8-octave attenuation {} dB", db(0.375));
        assert!(db(0.5) <= 5.0, "edge attenuation {} dB", db(0.5));
        assert!(db(0.75) >= 9.75, "3/4-octave attenuation {} dB", db(0.75));
        assert!(db(7.0 / 8.0) >= 13.6, "7/8-octave attenuation {} dB", db(7.0 / 8.0));
    }

    #[test]
    fn band_weights_partition_the_spectrum() {
        let fs = 48_000.0;
        let bank = OctaveBank::new(4096, fs);
        for k in 0..=bank.n / 2 {
            let sum: f64 = (0..10)
                .map(|b| {
                    let w = bank.band_weight(b, k);
                    w * w
                })
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "bin {k} ({} Hz): Σ W² = {sum:.15}",
                k as f64 * fs / bank.n as f64
            );
        }
    }

    #[test]
    fn impulse_band_energy_matches_parseval() {
        let fs = 48_000.0;
        let len = 4096;
        let bank = OctaveBank::new(len, fs);
        let mut impulse = vec![0.0; len];
        impulse[0] = 1.0;
        let spec = bank.forward(&impulse);
        let n = bank.n as f64;
        let mut checked = 0;
        for band in 0..10 {
            let Some(full) = bank.inverse_band_full(&spec, band) else {
                continue;
            };
            // Parseval for a real one-sided spectrum: interior bins count
            // twice (positive and negative frequency); DC and Nyquist once.
            // The realized magnitude is sqrt(W² + ε²), not W.
            let realized = |k: usize| {
                let w = bank.band_weight(band, k);
                w * w + MASK_FLOOR * MASK_FLOOR
            };
            let mut expected = realized(0) + realized(bank.n / 2);
            expected += 2.0 * (1..bank.n / 2).map(realized).sum::<f64>();
            let expected = expected / n;
            let got: f64 = full.iter().map(|v| v * v).sum();
            assert!(
                (got - expected).abs() < 1e-12 * expected,
                "band {band}: energy {got:.12} vs Parseval {expected:.12}"
            );
            checked += 1;
        }
        assert_eq!(checked, 10, "all ten bands are representable at 48 kHz");
    }

    #[test]
    fn minimum_phase_kernel_matches_the_designed_magnitude() {
        let fs = 48_000.0;
        let bank = OctaveBank::new(4096, fs);
        for band in 0..10 {
            let kernel = bank.min_phase_spectrum(band).expect("supported band");
            for (k, h) in kernel.iter().enumerate() {
                let w = bank.band_weight(band, k);
                let want = (w * w + MASK_FLOOR * MASK_FLOOR).sqrt();
                assert!(
                    (h.norm() - want).abs() < 1e-12,
                    "band {band} bin {k}: |H| {} vs design {want}",
                    h.norm()
                );
            }
        }
    }

    #[test]
    fn direct_impulse_energy_is_preserved_in_the_1_khz_band() {
        // The old zero-phase mask + [0, len) crop retained only ~51 % here.
        let fs = 48_000.0;
        let len = 4096;
        let bank = OctaveBank::new(len, fs);
        let mut impulse = vec![0.0; len];
        impulse[0] = 1.0;
        let spec = bank.forward(&impulse);
        let full = bank.inverse_band_full(&spec, 5).expect("1 kHz band");
        let cropped = bank.inverse_band(&spec, 5).expect("1 kHz band");
        let total: f64 = full.iter().map(|v| v * v).sum();
        let kept: f64 = cropped.iter().map(|v| v * v).sum();
        let retained = kept / total;
        assert!(
            retained >= 0.999,
            "1 kHz impulse retained {:.6} of its band energy (need >= 0.999)",
            retained
        );
    }

    #[test]
    fn band_energies_sum_to_the_total() {
        // Exact power partition: for a unit impulse the ten full-length band
        // outputs carry the impulse's energy. The epsilon floor costs 1e-11.
        let fs = 48_000.0;
        let len = 4096;
        let bank = OctaveBank::new(len, fs);
        let mut impulse = vec![0.0; len];
        impulse[0] = 1.0;
        let spec = bank.forward(&impulse);
        let total: f64 = impulse.iter().map(|v| v * v).sum();
        let mut sum = 0.0;
        for band in 0..10 {
            let full = bank.inverse_band_full(&spec, band).expect("band");
            sum += full.iter().map(|v| v * v).sum::<f64>();
        }
        assert!(
            (sum - total).abs() < 1e-9,
            "band energies {sum:.15} vs total {total:.15}"
        );
    }

    #[test]
    fn tone_at_band_centre_passes_and_neighbours_reject_it() {
        let fs = 48_000.0;
        // 0.2 s at 48 kHz. The 1 kHz tone is not an FFT bin of the
        // zero-padded transform, so a little of it leaks through the finite
        // record; the leak is not filter response and falls as 1/length
        // (8.5e-4 into the 500 Hz band here).
        let len = 9600;
        let bank = OctaveBank::new(len, fs);
        let tone: Vec<f64> = (0..len)
            .map(|i| (TAU * 1000.0 * i as f64 / fs).sin())
            .collect();
        let bands = filter_all(&bank, &tone);
        let energy = |b: usize| -> f64 {
            bands[b]
                .as_ref()
                .map(|v| v.iter().map(|s| s * s).sum())
                .unwrap_or(0.0)
        };
        let total: f64 = tone.iter().map(|v| v * v).sum();
        let pass = energy(5) / total; // 1 kHz band
        assert!(pass > 0.98, "1 kHz band passed only {pass:.12} of a 1 kHz tone");
        assert!(
            energy(4) / total < 1e-3,
            "500 Hz band leaked {} of the 1 kHz tone",
            energy(4) / total
        );
        assert!(
            energy(6) / total < 1e-3,
            "2 kHz band leaked {} of the 1 kHz tone",
            energy(6) / total
        );
        // The filter itself has a null at the neighbouring octave centre; the
        // residual above is finite-record leakage, not mask response.
        let k_tone = (1000.0 * bank.n as f64 / fs).round() as usize;
        assert_eq!(bank.band_weight(4, k_tone), 0.0);
        assert_eq!(bank.band_weight(6, k_tone), 0.0);
    }

    #[test]
    fn band_availability_follows_nyquist() {
        // 44.1 kHz cannot represent the 16 kHz octave (upper edge 22.6 kHz).
        let bank = OctaveBank::new(4096, 44_100.0);
        assert!(!bank.supported(9), "16 kHz band must be unavailable at 44.1 kHz");
        assert!(bank.supported(8), "8 kHz band is available at 44.1 kHz");
        // 32 kHz loses only the 16 kHz octave: the 8 kHz upper edge, 11.3 kHz,
        // is still below its 16 kHz Nyquist.
        let bank = OctaveBank::new(4096, 32_000.0);
        assert!(!bank.supported(9));
        assert!(bank.supported(8));
        assert!(bank.supported(7));
        // 16 kHz sample rate loses everything from 8 kHz up.
        let bank = OctaveBank::new(4096, 16_000.0);
        assert!(!bank.supported(8));
        assert!(bank.supported(7), "4 kHz band upper edge 5.66 kHz < 8 kHz");
    }

    #[test]
    fn filtering_keeps_signal_length() {
        let fs = 48_000.0;
        let len = 4096;
        let bank = OctaveBank::new(len, fs);
        let mut impulse = vec![0.0; len];
        impulse[0] = 1.0;
        let out = bank.inverse_band(&bank.forward(&impulse), 4).unwrap();
        assert_eq!(out.len(), len, "filtered output must keep the input length");
        let full = bank.inverse_band_full(&bank.forward(&impulse), 4).unwrap();
        assert_eq!(full.len(), bank.fft_len());
    }
}
