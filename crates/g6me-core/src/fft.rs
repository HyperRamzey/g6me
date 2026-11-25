//! Pure-Rust deterministic `f64` FFT, Hilbert-cepstral minimum-phase synthesis,
//! Kaiser windowing, and Glasberg–Moore (1990) ERB/roex psychoacoustic smoothing.
//!
//! Zero external crates, zero runtime SIMD/FMA dispatch — compiles under the
//! workspace's strict IEEE-754 arithmetic contract so `g6me-verify` stays clean
//! across x86_64 and aarch64.
//!
//! Minimum-phase cepstral folding, Kaiser-Bessel (`i0`, `kaiser_beta`,
//! `half_kaiser_tail_apply`), and `roex(p)` ERB smoothing are adapted from
//! `HyprEQ` (`HyperRamzey/HyprEQ`).

use std::f64::consts::PI;

/// Simple `f64` complex number with explicit non-FMA arithmetic.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }

    #[inline]
    pub fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }

    #[inline]
    pub fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }

    #[inline]
    pub fn div(self, o: Self) -> Self {
        let d = o.norm_sq();
        Self {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }
}

/// Iterative in-place radix-2 Cooley-Tukey FFT (`inverse == false`) or unscaled
/// IFFT (`inverse == true`, divided by `n`). `buf.len()` must be a power of two.
pub fn fft_in_place(buf: &mut [Complex], inverse: bool) {
    let n = buf.len();
    debug_assert!(n.is_power_of_two(), "FFT size must be a power of two");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let ang_step = sign * 2.0 * PI / (len as f64);
        // Compute twiddles per k from sin/cos rather than repeated complex
        // multiplication to eliminate cumulative rounding drift on large N.
        let mut i = 0usize;
        while i < n {
            for k in 0..half {
                let ang = ang_step * (k as f64);
                let w = Complex::new(ang.cos(), ang.sin());
                let u = buf[i + k];
                let v = buf[i + k + half].mul(w);
                buf[i + k] = u.add(v);
                buf[i + k + half] = u.sub(v);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for c in buf.iter_mut() {
            *c = c.scale(inv_n);
        }
    }
}

/// Cached radix-2 FFT plan holding precomputed twiddle tables so streaming
/// overlap-add blocks execute with zero trigonometric calls in the hot loop.
#[derive(Debug, Clone)]
pub struct FftPlan {
    n: usize,
    bitrev: Vec<usize>,
    /// Twiddle factors for forward FFT flattened by stage: stage `s` (`half = 1 << s`)
    /// occupies `[half - 1 .. 2*half - 1]` (Wait: indexed by `offset + k` where
    /// `offset = half - 1`).
    twiddles_fwd: Vec<Complex>,
    twiddles_inv: Vec<Complex>,
}

impl FftPlan {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2, "FFT size must be power of 2 >= 2");
        let mut bitrev = vec![0usize; n];
        let mut j = 0usize;
        for slot in bitrev.iter_mut() {
            *slot = j;
            let mut bit = n >> 1;
            while (j & bit) != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
        }
        let mut twiddles_fwd = Vec::with_capacity(n);
        let mut twiddles_inv = Vec::with_capacity(n);
        let mut len = 2usize;
        while len <= n {
            let half = len >> 1;
            let step = 2.0 * PI / (len as f64);
            for k in 0..half {
                let ang = step * (k as f64);
                let c = ang.cos();
                let s = ang.sin();
                twiddles_fwd.push(Complex::new(c, -s));
                twiddles_inv.push(Complex::new(c, s));
            }
            len <<= 1;
        }
        Self {
            n,
            bitrev,
            twiddles_fwd,
            twiddles_inv,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn forward(&self, buf: &mut [Complex]) {
        self.execute(buf, false);
    }

    pub fn inverse(&self, buf: &mut [Complex]) {
        self.execute(buf, true);
    }

    fn execute(&self, buf: &mut [Complex], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(buf.len(), n);
        for i in 0..n {
            let j = self.bitrev[i];
            if i < j {
                buf.swap(i, j);
            }
        }
        let tw = if inverse {
            &self.twiddles_inv
        } else {
            &self.twiddles_fwd
        };
        let mut len = 2usize;
        let mut tw_off = 0usize;
        while len <= n {
            let half = len >> 1;
            let mut i = 0usize;
            while i < n {
                for k in 0..half {
                    let u = buf[i + k];
                    let v = buf[i + k + half].mul(tw[tw_off + k]);
                    buf[i + k] = u.add(v);
                    buf[i + k + half] = u.sub(v);
                }
                i += len;
            }
            tw_off += half;
            len <<= 1;
        }
        if inverse {
            let inv_n = 1.0 / (n as f64);
            for c in buf.iter_mut() {
                *c = c.scale(inv_n);
            }
        }
    }

    /// Real-to-half-complex FFT (`n` real samples -> `n/2 + 1` bins).
    pub fn rfft(&self, input: &[f64], scratch: &mut [Complex], out: &mut [Complex]) {
        let n = self.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(scratch.len(), n);
        debug_assert_eq!(out.len(), n / 2 + 1);
        for (dst, &src) in scratch.iter_mut().zip(input.iter()) {
            *dst = Complex::new(src, 0.0);
        }
        self.forward(scratch);
        out.copy_from_slice(&scratch[..n / 2 + 1]);
    }

    /// Half-complex-to-real IFFT (`n/2 + 1` bins -> `n` real samples).
    pub fn irfft(&self, bins: &[Complex], scratch: &mut [Complex], out: &mut [f64]) {
        let n = self.n;
        let nb = n / 2 + 1;
        debug_assert_eq!(bins.len(), nb);
        debug_assert_eq!(scratch.len(), n);
        debug_assert_eq!(out.len(), n);
        scratch[0] = Complex::new(bins[0].re, 0.0);
        for k in 1..(nb - 1) {
            scratch[k] = bins[k];
            scratch[n - k] = bins[k].conj();
        }
        scratch[nb - 1] = Complex::new(bins[nb - 1].re, 0.0);
        self.inverse(scratch);
        for (dst, src) in out.iter_mut().zip(scratch.iter()) {
            *dst = src.re;
        }
    }
}

/// Real-to-half-complex convenience helper (allocates a plan).
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    let plan = FftPlan::new(n);
    let mut scratch = vec![Complex::ZERO; n];
    let mut out = vec![Complex::ZERO; n / 2 + 1];
    plan.rfft(input, &mut scratch, &mut out);
    out
}

/// Half-complex-to-real convenience helper (allocates a plan).
pub fn irfft(bins: &[Complex]) -> Vec<f64> {
    let n = (bins.len() - 1) * 2;
    let plan = FftPlan::new(n);
    let mut scratch = vec![Complex::ZERO; n];
    let mut out = vec![0.0; n];
    plan.irfft(bins, &mut scratch, &mut out);
    out
}

/// Hilbert-cepstral minimum-phase complex spectrum from a one-sided linear
/// magnitude array `mag_bins` of length `N/2 + 1` (ported from `HyprEQ/src/ir.rs`).
///
/// Reconstructs `H_min(k) = |H(k)| * exp(j * phi_min(k))` whose time-domain
/// impulse response is causal and concentrates energy as early as possible.
pub fn minimum_phase_spectrum(mag_bins: &[f64], plan: &FftPlan) -> Vec<Complex> {
    let nb = mag_bins.len();
    let n = (nb - 1) * 2;
    assert_eq!(plan.len(), n);

    let mut cep = vec![Complex::ZERO; n];
    for (k, &m) in mag_bins.iter().enumerate() {
        let lv = m.max(1e-12).ln();
        cep[k] = Complex::new(lv, 0.0);
        if k > 0 && k < nb - 1 {
            cep[n - k] = Complex::new(lv, 0.0);
        }
    }

    plan.inverse(&mut cep);

    // Causal cepstral folding: keep DC & Nyquist, double positive quefrencies,
    // zero negative quefrencies.
    let half = n / 2;
    cep[0] = Complex::new(cep[0].re, 0.0);
    for c in &mut cep[1..half] {
        *c = Complex::new(2.0 * c.re, 0.0);
    }
    cep[half] = Complex::new(cep[half].re, 0.0);
    for c in &mut cep[(half + 1)..n] {
        *c = Complex::ZERO;
    }

    plan.forward(&mut cep);
    cep[..nb].iter().map(|c| c.exp()).collect()
}

/// Modified Bessel function of the first kind, order 0: $I_0(x)$
/// (ported from `HyprEQ/src/window.rs`).
pub fn i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = 0.5 * x;
    let half_x2 = half_x * half_x;
    for k in 1..64 {
        let kf = k as f64;
        term *= half_x2 / (kf * kf);
        sum += term;
        if term <= 1e-16 * sum {
            break;
        }
    }
    sum
}

/// Kaiser $\beta$ parameter for desired stopband attenuation in dB
/// (ported from `HyprEQ/src/window.rs`).
pub fn kaiser_beta(att_db: f64) -> f64 {
    if att_db > 50.0 {
        0.1102 * (att_db - 8.7)
    } else if att_db >= 21.0 {
        0.5842 * (att_db - 21.0).powf(0.4) + 0.07886 * (att_db - 21.0)
    } else {
        0.0
    }
}

/// Apply the falling half of a Kaiser window to the final `tail` samples of `ir`,
/// leaving `ir[..len - tail]` untouched (`HyprEQ/src/window.rs` pattern).
pub fn half_kaiser_tail_apply(ir: &mut [f64], tail: usize, beta: f64) {
    let len = ir.len();
    if tail == 0 || len == 0 {
        return;
    }
    let t = tail.min(len);
    let start = len - t;
    let denom = i0(beta);
    let m = (2 * t - 1) as f64;
    let half_m = 0.5 * m;
    for i in 0..t {
        // Sample the second half `[t .. 2*t - 1]` of a symmetric length `2*t`
        // Kaiser window so `i == 0` is ~1.0 and `i == t - 1` hits the edge floor.
        let idx = (t + i) as f64;
        let r = (idx - half_m) / half_m;
        let arg = (1.0 - r * r).max(0.0).sqrt();
        let w = i0(beta * arg) / denom;
        ir[start + i] *= w;
    }
}

/// Glasberg & Moore (1990) equivalent rectangular bandwidth in Hz
/// (ported from `HyprEQ/src/psycho.rs`).
#[inline]
pub fn erb_hz(f_hz: f64) -> f64 {
    24.7 * (4.37e-3 * f_hz.max(0.0) + 1.0)
}

/// Smooth a linear power spectrum `power` (defined at `freqs` Hz) through
/// Glasberg & Moore (1990) symmetric `roex(p)` auditory filters scaled by
/// `erb_mult` (e.g. `1.0` = 1-ERB psychoacoustic excitation).
pub fn roex_smooth_power(freqs: &[f64], power: &[f64], erb_mult: f64) -> Vec<f64> {
    let n = freqs.len();
    debug_assert_eq!(n, power.len());
    let mut out = vec![0.0_f64; n];
    if n == 0 {
        return out;
    }
    let scale = erb_mult.max(0.1);
    for i in 0..n {
        let fc = freqs[i];
        if fc <= 0.0 {
            out[i] = power[i];
            continue;
        }
        // roex(p) has ERB = 4 * fc / p  =>  p = 4 * fc / (scale * ERB(fc))
        let bw = scale * erb_hz(fc);
        let p = 4.0 * fc / bw;
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        // Cutoff at g_max where (1 + p*g)*exp(-p*g) < 1e-6 (p*g ~ 17).
        let df_max = (17.0 / p) * fc;
        let f_lo = (fc - df_max).max(0.0);
        let f_hi = fc + df_max;

        // Linear scan over the narrow support around `i` using bin spacing.
        let mut j = i;
        loop {
            let fj = freqs[j];
            if fj < f_lo {
                break;
            }
            let g = (fc - fj) / fc;
            let pg = p * g;
            let w = (1.0 + pg) * (-pg).exp();
            num += w * power[j];
            den += w;
            if j == 0 {
                break;
            }
            j -= 1;
        }
        for j in (i + 1)..n {
            let fj = freqs[j];
            if fj > f_hi {
                break;
            }
            let g = (fj - fc) / fc;
            let pg = p * g;
            let w = (1.0 + pg) * (-pg).exp();
            num += w * power[j];
            den += w;
        }
        out[i] = if den > 0.0 { num / den } else { power[i] };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_round_trip_and_parseval() {
        let n = 1024;
        let plan = FftPlan::new(n);
        let input: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                0.7 * (2.0 * PI * 7.0 * t).sin() + 0.3 * (2.0 * PI * 53.0 * t + 0.4).cos()
            })
            .collect();

        let mut scratch = vec![Complex::ZERO; n];
        let mut bins = vec![Complex::ZERO; n / 2 + 1];
        let mut recon = vec![0.0; n];
        plan.rfft(&input, &mut scratch, &mut bins);
        plan.irfft(&bins, &mut scratch, &mut recon);

        let max_err = input
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-13, "FFT round-trip error {max_err:e}");

        // Parseval: sum |x|^2 == (1/N) * (|X[0]|^2 + |X[N/2]|^2 + 2 * sum_{1..N/2-1} |X[k]|^2)
        let e_time: f64 = input.iter().map(|x| x * x).sum();
        let e_freq: f64 = (bins[0].norm_sq()
            + bins[n / 2].norm_sq()
            + 2.0 * bins[1..n / 2].iter().map(|c| c.norm_sq()).sum::<f64>())
            / (n as f64);
        assert!(
            (e_time - e_freq).abs() < 1e-11,
            "Parseval mismatch: time={e_time}, freq={e_freq}"
        );
    }

    #[test]
    fn plan_matches_direct_fft() {
        let n = 512;
        let plan = FftPlan::new(n);
        let orig: Vec<Complex> = (0..n)
            .map(|i| Complex::new(((i * 37) % 101) as f64 * 0.01 - 0.5, ((i * 19) % 89) as f64 * 0.01))
            .collect();
        let mut a = orig.clone();
        let mut b = orig;
        fft_in_place(&mut a, false);
        plan.forward(&mut b);
        let diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| x.sub(*y).norm())
            .fold(0.0_f64, f64::max);
        assert!(diff < 1e-12, "plan vs direct FFT diff {diff:e}");
    }

    #[test]
    fn minimum_phase_reconstructs_magnitude_and_is_causal() {
        let n = 4096;
        let nb = n / 2 + 1;
        let plan = FftPlan::new(n);
        // Smooth 1st-order shelf magnitude response.
        let mag: Vec<f64> = (0..nb)
            .map(|k| {
                let f = k as f64 * 48_000.0 / n as f64;
                let x = f / 1000.0;
                ((0.25 * x * x + 1.0) / (x * x + 1.0)).sqrt()
            })
            .collect();
        let h = minimum_phase_spectrum(&mag, &plan);

        let max_mag_err = h
            .iter()
            .zip(mag.iter())
            .map(|(c, &m)| (c.norm() - m).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_mag_err < 1e-11,
            "min-phase magnitude mismatch {max_mag_err:e}"
        );

        let mut scratch = vec![Complex::ZERO; n];
        let mut ir = vec![0.0; n];
        plan.irfft(&h, &mut scratch, &mut ir);
        // Causal energy: almost all energy in the first half, virtually zero in the back half.
        let front_e: f64 = ir[..n / 2].iter().map(|x| x * x).sum();
        let back_e: f64 = ir[n / 2..].iter().map(|x| x * x).sum();
        assert!(
            back_e / front_e < 1e-12,
            "non-causal energy ratio: {}",
            back_e / front_e
        );
        // Minimum-phase peak occurs right at sample 0.
        assert!(
            ir[0].abs() > 0.4,
            "expected sample-0 impulse peak, got {}",
            ir[0]
        );
    }

    #[test]
    fn half_kaiser_tail_preserves_head_and_attenuates_end() {
        let mut ir = vec![1.0; 2048];
        half_kaiser_tail_apply(&mut ir, 384, 16.0);
        // Head untouched.
        assert_eq!(ir[0], 1.0);
        assert_eq!(ir[1663], 1.0);
        // Tail monotonically decays to exact Kaiser edge floor `1 / I0(16) ≈ 1.12e-6` (-119 dB).
        assert!(ir[1664] > 0.999);
        assert!(ir[2047] < 2e-6, "end tap too high: {}", ir[2047]);
        for i in 1665..2048 {
            assert!(ir[i] <= ir[i - 1], "tail not monotonic at {i}");
        }
    }

    #[test]
    fn roex_preserves_flat_spectrum() {
        let freqs: Vec<f64> = (0..513).map(|k| k as f64 * 48_000.0 / 1024.0).collect();
        let flat = vec![3.5_f64; freqs.len()];
        let sm = roex_smooth_power(&freqs, &flat, 1.0);
        for (i, &v) in sm.iter().enumerate() {
            assert!(
                (v - 3.5).abs() < 1e-12,
                "flat spectrum distorted at bin {i}: {v}"
            );
        }
    }
}
