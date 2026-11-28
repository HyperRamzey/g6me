//! Ideal rectangular octave-band filtering in the FFT domain.
//!
//! ISO 3382-1 band metrics are normally measured through IEC 61260 class-1
//! octave filters. This harness uses zero-phase FFT masks with a raised-cosine
//! taper across each octave instead: no passband ripple, exactly known weights,
//! and an impulse response that decays fast enough not to leak the direct sound
//! into the late decay. A brick-wall (rectangular) mask is *not* used because
//! its 1/t impulse-response tail makes the loud onset dominate the Schroeder
//! integral at late times; on a synthetic 1 s decay that biases the 125 Hz T30
//! to 2.3 s. The tapered filters are a defensible reference for synthetic
//! verification, not a claim of IEC 61260 conformance; measured-dataset tables
//! produced with them are reference values, not certified ones.
//!
//! Each band is octave-wide: edges at the centre frequency divided and
//! multiplied by √2 (ISO 266 preferred octave centres 31.5 Hz … 16 kHz), with
//! weight 1 at the centre falling as a half cosine to the edges.
//! A band is unsupported when its upper edge is not below Nyquist or when it
//! contains no FFT bin at the zero-padded transform length.

use g6me_core::fft::{Complex, FftPlan};
use std::f64::consts::PI;

/// ISO 266 preferred octave-band centre frequencies, 31.5 Hz to 16 kHz.
pub const OCTAVE_CENTERS_HZ: [f64; 10] = [
    31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0,
];

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Octave-band edges `(f_centre/√2, f_centre·√2)`.
pub fn octave_band_edges(centre_hz: f64) -> (f64, f64) {
    (centre_hz / SQRT_2, centre_hz * SQRT_2)
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
    /// Inclusive one-sided bin range for each supported band.
    ranges: [Option<(usize, usize)>; 10],
}

impl OctaveBank {
    pub fn new(len: usize, fs: f64) -> Self {
        let n = next_pow2(len.saturating_mul(2));
        let plan = FftPlan::new(n);
        let mut ranges = [None; 10];
        if fs > 0.0 {
            for (slot, centre) in ranges.iter_mut().zip(OCTAVE_CENTERS_HZ.iter()) {
                let (lo, hi) = octave_band_edges(*centre);
                // The upper edge must stay strictly below Nyquist, otherwise
                // the band is not representable at this sample rate.
                if hi >= fs / 2.0 || lo <= 0.0 {
                    continue;
                }
                let k_lo = (lo * n as f64 / fs).ceil() as usize;
                let k_hi = (hi * n as f64 / fs).floor() as usize;
                if k_lo <= k_hi && k_hi < n / 2 {
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

    /// Inclusive FFT-bin range for a supported band, `None` if unsupported.
    pub fn band_bins(&self, band: usize) -> Option<(usize, usize)> {
        self.ranges.get(band).copied().flatten()
    }

    pub fn supported(&self, band: usize) -> bool {
        self.band_bins(band).is_some()
    }

    /// Raised-cosine weight for bin `k` in `band`: 0 outside the octave, 1 at
    /// the centre, half-cosine tapers to each edge. The smooth taper is what
    /// keeps the impulse response from ringing across the whole record.
    pub fn band_weight(&self, band: usize, k: usize) -> f64 {
        let Some((k_lo, k_hi)) = self.band_bins(band) else {
            return 0.0;
        };
        if k < k_lo || k > k_hi || !self.fs.is_finite() || self.fs <= 0.0 {
            return 0.0;
        }
        let Some(&centre) = OCTAVE_CENTERS_HZ.get(band) else {
            return 0.0;
        };
        let f = k as f64 * self.fs / self.n as f64;
        let lo = centre / SQRT_2;
        let hi = centre * SQRT_2;
        if f <= lo || f >= hi {
            return 0.0;
        }
        let w = if f <= centre {
            0.5 - 0.5 * (PI * (f - lo) / (centre - lo)).cos()
        } else {
            0.5 + 0.5 * (PI * (f - centre) / (hi - centre)).cos()
        };
        w.clamp(0.0, 1.0)
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

    /// Inverse transform of `spec` weighted into `band`, at the full
    /// zero-padded length. This is the complete linear-convolution result.
    pub fn inverse_band_full(&self, spec: &[Complex], band: usize) -> Option<Vec<f64>> {
        let (k_lo, k_hi) = self.band_bins(band)?;
        let mut masked = vec![Complex::ZERO; self.n / 2 + 1];
        for (k, slot) in masked.iter_mut().enumerate().take(k_hi + 1).skip(k_lo) {
            let w = self.band_weight(band, k);
            if w != 0.0 {
                *slot = spec[k].scale(w);
            }
        }
        let mut scratch = vec![Complex::ZERO; self.n];
        let mut out = vec![0.0; self.n];
        self.plan.irfft(&masked, &mut scratch, &mut out);
        Some(out)
    }

    /// Inverse transform of `spec` with only `band`'s bins retained. The result
    /// is truncated to the original signal length.
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
        for centre in OCTAVE_CENTERS_HZ {
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
        for (band, centre) in OCTAVE_CENTERS_HZ.iter().enumerate() {
            let Some(full) = bank.inverse_band_full(&spec, band) else {
                continue;
            };
            let (k_lo, k_hi) = bank.band_bins(band).unwrap();
            // Parseval for a real one-sided spectrum: each interior bin
            // contributes twice (positive and negative frequency). These bands
            // never touch DC or Nyquist, so full-length energy is 2·Σw²/n.
            let expected = 2.0
                * (k_lo..=k_hi)
                    .map(|k| bank.band_weight(band, k).powi(2))
                    .sum::<f64>()
                / n;
            let got: f64 = full.iter().map(|v| v * v).sum();
            assert!(
                (got - expected).abs() < 1e-12 * expected,
                "band {centre}: energy {got:.12} vs Parseval {expected:.12}"
            );
            checked += 1;
        }
        assert_eq!(checked, 10, "all ten bands are representable at 48 kHz");
    }

    #[test]
    fn tone_at_band_centre_passes_and_neighbours_reject_it() {
        let fs = 48_000.0;
        // 0.1 s at 48 kHz gives integer cycles for 1 kHz, so the rectangular
        // window leaks no energy outside the 1 kHz bins.
        let len = 4800;
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
        // The zero-padded 1 kHz tone spreads over a few bins, and the tapered
        // mask rolls off towards 707 Hz/1.414 kHz, so 98.5% is the expected pass.
        assert!(pass > 0.98, "1 kHz band passed only {:.12} of a 1 kHz tone", pass);
        assert!(energy(4) / total < 1e-3, "500 Hz band leaked the 1 kHz tone");
        assert!(
            energy(6) / total < 1e-3,
            "2 kHz band leaked the 1 kHz tone"
        );
    }

    #[test]
    fn band_availability_follows_nyquist() {
        // 44.1 kHz cannot represent the 16 kHz octave (upper edge 22.63 kHz).
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
    fn filtering_is_zero_phase_and_keeps_signal_length() {
        let fs = 48_000.0;
        let len = 4096;
        let mid = len / 2;
        let bank = OctaveBank::new(len, fs);
        // A centred impulse produces the filter's impulse response, which a
        // real (zero-phase) mask must place symmetrically about the impulse.
        let mut impulse = vec![0.0; len];
        impulse[mid] = 1.0;
        let out = bank.inverse_band(&bank.forward(&impulse), 4).unwrap();
        assert_eq!(out.len(), len, "filtered output must keep the input length");
        let peak = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, mid, "500 Hz impulse response peak is not zero-phase");
        for j in 1..mid {
            assert!(
                (out[mid - j] - out[mid + j]).abs() < 1e-12,
                "response is not symmetric at ±{j} samples"
            );
        }
    }
}
