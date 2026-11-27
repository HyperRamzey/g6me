//! Sample-format vocabulary shared by the ffmpeg stages, plus render statistics.

use serde::{Deserialize, Serialize};

/// PCM format used on the wire between a stage and ffmpeg.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BitDepth {
    S16,
    S24,
    S32,
    F32,
    F64,
}

impl BitDepth {
    pub const ALL: [BitDepth; 5] = [
        BitDepth::S16,
        BitDepth::S24,
        BitDepth::S32,
        BitDepth::F32,
        BitDepth::F64,
    ];

    /// ffmpeg decoder/muxer name for raw PCM of this format.
    pub fn codec(self) -> &'static str {
        match self {
            BitDepth::S16 => "pcm_s16le",
            BitDepth::S24 => "pcm_s24le",
            BitDepth::S32 => "pcm_s32le",
            BitDepth::F32 => "pcm_f32le",
            BitDepth::F64 => "pcm_f64le",
        }
    }

    /// `aresample`/`aformat` sample-format token.
    pub fn swr_fmt(self) -> &'static str {
        match self {
            BitDepth::S16 => "s16",
            BitDepth::S24 => "s32", // swr has no s24; see `quantize_note`
            BitDepth::S32 => "s32",
            BitDepth::F32 => "flt",
            BitDepth::F64 => "dbl",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, BitDepth::F32 | BitDepth::F64)
    }

    pub fn label(self) -> &'static str {
        match self {
            BitDepth::S16 => "16-bit int",
            BitDepth::S24 => "24-bit int",
            BitDepth::S32 => "32-bit int",
            BitDepth::F32 => "32-bit float",
            BitDepth::F64 => "64-bit float",
        }
    }

    /// Why 24-bit is quantized here and not by a filter: libswresample has no `s24`
    /// sample format (`S24` maps to `s32` above), and ffmpeg's 24-bit encoders keep the
    /// top 24 bits of that `s32` -- a floor, half an LSB low on average, dither
    /// included. [`Grid24`] rounds onto the 24-bit grid first, so the encoder's
    /// truncation drops only zeros.
    pub fn quantize_note() -> &'static str {
        "24-bit is rounded (and dithered) by g6me before the encoder; ffmpeg's own s32->s24 \
         step truncates"
    }
}

/// 9-tap Wannamaker (JAES 1992) F-weighted psychoacoustic noise-shaping FIR
/// coefficients `h[1..=9]` for standard sample rates (`<= 50 kHz`).
///
/// The error-feedback transfer function is `H_e(z) = 1 - sum_{k=1..9} h[k] z^-k`,
/// which pushes quantization noise out of the 1–5 kHz peak ear-sensitivity band
/// toward Nyquist while staying minimum-phase and unconditionally stable.
pub const WANNAMAKER_9_TAPS: [f64; 9] = [
    2.412, -3.370, 3.937, -4.174, 3.353, -2.205, 1.281, -0.569, 0.0847,
];

/// Terhardt (1979) / ISO 226 perceptual sensitivity power weight `|W(f)|^2`
/// with smooth ultrasonic roll-off above 20 kHz, used to synthesize
/// Helmrich–Holters–Zölzer (2007) minimum-phase error-feedback filters via
/// Levinson–Durbin at high sample rates (`> 50 kHz`, e.g. 88.2–384 kHz).
fn perceptual_weight_sq(f_hz: f64) -> f64 {
    if f_hz <= 0.0 {
        return 1e-4;
    }
    let fk = f_hz / 1000.0;
    let d = fk - 3.3;
    let mut tq = 3.64 * fk.powf(-0.8) - 6.5 * (-0.6 * d * d).exp() + 1e-3 * fk * fk * fk * fk;
    if f_hz > 20_000.0 {
        tq += 40.0 * (f_hz / 20_000.0).log10();
    }
    let w_db = (-(tq + 6.0)).clamp(-90.0, 20.0);
    10.0_f64.powf(w_db / 10.0)
}

/// Design a 9-tap minimum-phase error-feedback FIR `h[1..=9]` for `sample_rate`.
///
/// - For `sample_rate <= 50_000`: exact 9-tap Wannamaker (1992) F-weighted filter.
/// - For `sample_rate > 50_000`: 9-tap Wiener-optimal minimum-phase predictor
///   solved via Levinson–Durbin recursion on the perceptual power spectrum
///   `|W(f)|^2 + 5e-4`, pushing quantization noise cleanly into the ultrasonic band.
pub fn design_wannamaker_taps(sample_rate: u32) -> [f64; 9] {
    if sample_rate <= 50_000 {
        return WANNAMAKER_9_TAPS;
    }
    const ORDER: usize = 9;
    const N: usize = 2048;
    const REG: f64 = 5e-4;
    let sr = f64::from(sample_rate);
    let mut spec = [0.0f64; N / 2 + 1];
    for (i, s) in spec.iter_mut().enumerate() {
        let f = (i as f64) * sr / (N as f64);
        *s = perceptual_weight_sq(f) + REG;
    }
    let mut r = [0.0f64; ORDER + 1];
    let inv_n = 2.0 / (N as f64);
    for (k, rk) in r.iter_mut().enumerate() {
        let nyq_sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let mut s = 0.5 * (spec[0] + spec[N / 2] * nyq_sign);
        let step = 2.0 * std::f64::consts::PI * (k as f64) / (N as f64);
        for (i, &sp) in spec.iter().enumerate().take(N / 2).skip(1) {
            s += sp * (step * (i as f64)).cos();
        }
        *rk = s * inv_n;
    }
    let mut a = [0.0f64; ORDER + 1];
    let mut e = r[0].max(1e-18);
    for m in 1..=ORDER {
        let mut acc = r[m];
        for j in 1..m {
            acc += a[j] * r[m - j];
        }
        let k = (-acc / e).clamp(-0.9999, 0.9999);
        let mut a_new = a;
        a_new[m] = k;
        for j in 1..m {
            a_new[j] = a[j] + k * a[m - j];
        }
        a = a_new;
        e *= 1.0 - k * k;
    }
    let mut h = [0.0f64; ORDER];
    for j in 0..ORDER {
        h[j] = -a[j + 1];
    }
    h
}

/// In-app stereo integer quantizer with optional TPDF dither or Wannamaker (1992) /
/// Helmrich–Holters–Zölzer (2007) 9-tap psychoacoustic error-feedback noise shaping.
///
/// Supports both 16-bit (`BitDepth::S16`) and 24-bit (`BitDepth::S24`) targets so
/// psychoacoustic noise shaping works cleanly at 16-bit and 24-bit across 44.1–384 kHz
/// without relying on libswresample's 24-bit-unstable noise shapers.
#[derive(Clone, Debug)]
pub struct PcmQuantizer {
    dither: Dither,
    scale: f64,
    max_val: f64,
    rng: u64,
    /// Previous uniform draw per channel, for high-passed triangular dither.
    prev: [f64; 2],
    /// 9-tap error-feedback coefficients (active when `dither == Dither::Wannamaker`).
    taps: [f64; 9],
    /// Quantization error history per channel (`hist[ch][0]` is `e[n-1]`).
    hist: [[f64; 9]; 2],
}

impl PcmQuantizer {
    /// Dithers this in-app quantizer implements directly.
    pub fn supports(d: Dither) -> bool {
        matches!(
            d,
            Dither::None
                | Dither::Rectangular
                | Dither::Triangular
                | Dither::TriangularHp
                | Dither::Wannamaker
        )
    }

    /// Construct a quantizer for `depth` (`S16` or `S24`), `dither`, and `sample_rate`.
    /// Returns `None` if `depth` is not `S16`/`S24` or `dither` is not supported in-app.
    pub fn new(depth: BitDepth, dither: Dither, sample_rate: u32) -> Option<Self> {
        if !Self::supports(dither) {
            return None;
        }
        let (scale, max_val) = match depth {
            BitDepth::S16 => (32_768.0, 32_767.0),
            BitDepth::S24 => (8_388_608.0, 8_388_607.0),
            _ => return None,
        };
        let taps = if dither == Dither::Wannamaker {
            design_wannamaker_taps(sample_rate)
        } else {
            [0.0; 9]
        };
        Some(Self {
            dither,
            scale,
            max_val,
            // Fixed seed: two renders of one file stay byte-identical.
            rng: 0x9E37_79B9_7F4A_7C15,
            prev: [0.0; 2],
            taps,
            hist: [[0.0; 9]; 2],
        })
    }

    /// Uniform in [-0.5, 0.5), xorshift64*.
    #[inline]
    fn uniform(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    #[inline]
    fn quantize_sample(&mut self, ch: usize, x: f64) -> f64 {
        let xs = x * self.scale;
        match self.dither {
            Dither::None => xs.round_ties_even().clamp(-self.scale, self.max_val) / self.scale,
            Dither::Rectangular => {
                let d = self.uniform();
                (xs + d).round_ties_even().clamp(-self.scale, self.max_val) / self.scale
            }
            Dither::Triangular => {
                let d = self.uniform() + self.uniform();
                (xs + d).round_ties_even().clamp(-self.scale, self.max_val) / self.scale
            }
            Dither::TriangularHp => {
                let u = self.uniform();
                let d = u - self.prev[ch];
                self.prev[ch] = u;
                (xs + d).round_ties_even().clamp(-self.scale, self.max_val) / self.scale
            }
            Dither::Wannamaker => {
                // Explicit loop without FMA: fb = sum_{k=0..9} h[k] * e[n-1-k]
                let mut fb = 0.0;
                for k in 0..9 {
                    let prod = self.taps[k] * self.hist[ch][k];
                    fb += prod;
                }
                let v = xs - fb;
                // Nonsubtractive 2-LSB peak-to-peak TPDF dither inside the feedback loop.
                let d = self.uniform() + self.uniform();
                let q = (v + d).round_ties_even().clamp(-self.scale, self.max_val);
                // Overload-safe error clamp prevents feedback windup at ±FS clip while
                // remaining completely transparent under normal operation (|e| <= 1.5 LSB).
                let e = (q - v).clamp(-2.0, 2.0);
                let h = &mut self.hist[ch];
                for k in (1..9).rev() {
                    h[k] = h[k - 1];
                }
                h[0] = e;
                q / self.scale
            }
            _ => xs.round_ties_even().clamp(-self.scale, self.max_val) / self.scale,
        }
    }

    /// Quantize both stereo channels in place, frame by frame.
    pub fn process(&mut self, l: &mut [f64], r: &mut [f64]) {
        for (a, b) in l.iter_mut().zip(r.iter_mut()) {
            *a = self.quantize_sample(0, *a);
            *b = self.quantize_sample(1, *b);
        }
    }
}

/// Rounds a stereo stream onto the 24-bit grid, optionally with dither or
/// Wannamaker psychoacoustic noise shaping, before it reaches an encoder that
/// would otherwise truncate it (see [`BitDepth::quantize_note`]).
#[derive(Clone, Debug)]
pub struct Grid24(PcmQuantizer);

impl Grid24 {
    /// Dithers this quantizer can apply at 24 bits.
    pub fn supports(d: Dither) -> bool {
        PcmQuantizer::supports(d)
    }

    /// Construct at default 48 kHz; `None` for a dither [`Grid24::supports`] refuses.
    pub fn new(dither: Dither) -> Option<Self> {
        Self::with_sample_rate(dither, 48_000)
    }

    /// Construct with explicit `sample_rate` (selects standard Wannamaker vs
    /// high-rate Levinson–Durbin perceptual shaping for `Dither::Wannamaker`).
    pub fn with_sample_rate(dither: Dither, sample_rate: u32) -> Option<Self> {
        PcmQuantizer::new(BitDepth::S24, dither, sample_rate).map(Self)
    }

    /// Quantize both channels in place, frame by frame.
    pub fn process(&mut self, l: &mut [f64], r: &mut [f64]) {
        self.0.process(l, r);
    }
}

/// Noise-shaping / dither method.
///
/// `None` is the default and keeps renders reproducible. `Wannamaker` runs
/// g6me's in-app 9-tap Wannamaker (1992) / Helmrich–Holters–Zölzer (2007)
/// psychoacoustic error-feedback noise shaper at 16-bit and 24-bit across
/// 44.1–384 kHz; the remaining methods map directly to ffmpeg's `aresample`
/// `dither_method` (or in-app TPDF/RPDF for 24-bit).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Dither {
    #[default]
    None,
    Rectangular,
    Triangular,
    TriangularHp,
    /// In-app 9-tap Wannamaker (JAES 1992) F-weighted (`<= 50 kHz`) /
    /// Levinson–Durbin ISO-226 (`> 50 kHz`) psychoacoustic noise shaping.
    Wannamaker,
    Lipshitz,
    Shibata,
    LowShibata,
    HighShibata,
    FWeighted,
    ModifiedEWeighted,
    ImprovedEWeighted,
}

impl Dither {
    pub const ALL: [Dither; 12] = [
        Dither::None,
        Dither::Rectangular,
        Dither::Triangular,
        Dither::TriangularHp,
        Dither::Wannamaker,
        Dither::Lipshitz,
        Dither::Shibata,
        Dither::LowShibata,
        Dither::HighShibata,
        Dither::FWeighted,
        Dither::ModifiedEWeighted,
        Dither::ImprovedEWeighted,
    ];

    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Dither::None => "none",
            Dither::Rectangular => "rectangular",
            Dither::Triangular => "triangular",
            Dither::TriangularHp => "triangular_hp",
            Dither::Wannamaker => "wannamaker",
            Dither::Lipshitz => "lipshitz",
            Dither::Shibata => "shibata",
            Dither::LowShibata => "low_shibata",
            Dither::HighShibata => "high_shibata",
            Dither::FWeighted => "f_weighted",
            Dither::ModifiedEWeighted => "modified_e_weighted",
            Dither::ImprovedEWeighted => "improved_e_weighted",
        }
    }

    /// True for dithers handled exclusively by g6me's in-app [`PcmQuantizer`]
    /// rather than ffmpeg's `aresample`.
    pub fn is_in_app_only(self) -> bool {
        matches!(self, Dither::Wannamaker)
    }

    pub fn label(self) -> &'static str {
        match self {
            Dither::None => "none (reproducible)",
            Dither::Wannamaker => "wannamaker (psychoacoustic)",
            other => other.ffmpeg_name(),
        }
    }
}

/// Full-scale reference for PCM: `1.0` maps to 0 dBFS.
pub fn dbfs(x: f64) -> f64 {
    if x > 0.0 && x.is_finite() {
        20.0 * x.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Incremental SHA-256 over rendered output, so two runs can be compared byte
/// for byte without keeping both files.
///
/// A thin wrapper so callers cannot accidentally construct one that reports a
/// digest of nothing.
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::new())
    }

    pub fn update(&mut self, bytes: &[u8]) {
        use sha2::Digest;
        self.0.update(bytes);
    }

    pub fn finish_hex(self) -> String {
        use sha2::Digest;
        let out = self.0.finalize();
        let mut s = String::with_capacity(out.len() * 2);
        for b in out {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_the_known_test_vector() {
        // The empty-input and "abc" digests are the standard FIPS 180-4 vectors.
        assert_eq!(
            Sha256::new().finish_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut h = Sha256::new();
        h.update(b"a");
        h.update(b"bc");
        assert_eq!(
            h.finish_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn codec_names_are_populated() {
        for d in BitDepth::ALL {
            assert!(!d.codec().is_empty());
        }
    }

    #[test]
    fn dither_names_are_what_ffmpeg_parses() {
        // ffmpeg rejects unknown constants loudly, so a typo here would fail
        // every render; these are the exact spellings from its option list.
        assert_eq!(Dither::LowShibata.ffmpeg_name(), "low_shibata");
        assert_eq!(Dither::FWeighted.ffmpeg_name(), "f_weighted");
        assert_eq!(
            Dither::ModifiedEWeighted.ffmpeg_name(),
            "modified_e_weighted"
        );
        assert_eq!(Dither::default(), Dither::None);
        assert_eq!(Dither::None.ffmpeg_name(), "none");
    }

    #[test]
    fn dbfs_conversions() {
        assert_eq!(dbfs(1.0), 0.0);
        assert_eq!(dbfs(0.0), f64::NEG_INFINITY);
        assert!((dbfs(0.5) + 6.0206).abs() < 0.01);
    }

    /// What ffmpeg's 24-bit encoders do with the `s32` they are handed.
    fn encoder_s24(x: f64) -> i64 {
        let s32 = (x * 2147483648.0)
            .round_ties_even()
            .clamp(-2147483648.0, 2147483647.0) as i64;
        s32 >> 8
    }

    fn error_after_encoder(d: Dither) -> (f64, f64) {
        let mut g = Grid24::new(d).unwrap();
        let mut seed = 1u64;
        let src: Vec<f64> = (0..20_000)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e-3
            })
            .collect();
        let (mut l, mut r) = (src.clone(), src.clone());
        g.process(&mut l, &mut r);
        let e: Vec<f64> = l
            .iter()
            .zip(&src)
            .map(|(&q, &x)| encoder_s24(q) as f64 - x * 8_388_608.0)
            .collect();
        let mean = e.iter().sum::<f64>() / e.len() as f64;
        let max = e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        (mean, max)
    }

    #[test]
    fn grid24_output_survives_the_encoders_truncation_unbiased() {
        // Without the grid, the encoder alone sits half an LSB low.
        let raw: f64 = (0..1000)
            .map(|i| encoder_s24(i as f64 * 1.37e-7) as f64 - i as f64 * 1.37e-7 * 8_388_608.0)
            .sum::<f64>()
            / 1000.0;
        assert!(raw < -0.4, "the bias this exists for: {raw}");

        let (mean, max) = error_after_encoder(Dither::None);
        assert!(mean.abs() < 0.02 && max <= 0.5 + 1e-9, "none: {mean} {max}");
        for d in [
            Dither::Rectangular,
            Dither::Triangular,
            Dither::TriangularHp,
        ] {
            let (mean, max) = error_after_encoder(d);
            assert!(mean.abs() < 0.03, "{d:?} biased: {mean}");
            assert!(
                max > 0.5 && max <= 1.5 + 1e-9,
                "{d:?} noise out of range: {max}"
            );
        }
    }

    #[test]
    fn grid24_clamps_full_scale_and_refuses_noise_shaping() {
        let mut g = Grid24::new(Dither::None).unwrap();
        let (mut l, mut r) = (vec![1.0, -1.0, 2.0], vec![0.0; 3]);
        g.process(&mut l, &mut r);
        assert_eq!(
            l,
            vec![8_388_607.0 / 8_388_608.0, -1.0, 8_388_607.0 / 8_388_608.0]
        );
        assert!(Grid24::new(Dither::Shibata).is_none());
        assert!(Grid24::new(Dither::LowShibata).is_none());
        assert!(Grid24::new(Dither::Wannamaker).is_some());
    }

    #[test]
    fn wannamaker_noise_shaping_drops_1_to_5khz_floor_and_preserves_minus90_dbfs_sine() {
        fn band_power_db(err: &[f64], sr: u32) -> f64 {
            let start = 4096usize;
            let len = 4096usize;
            let mut sum_p = 0.0;
            let mut cnt = 0usize;
            for f in (1100..5000).step_by(100) {
                let w = 2.0 * std::f64::consts::PI * (f as f64) / (sr as f64);
                let mut re = 0.0;
                let mut im = 0.0;
                for (k, &v) in err[start..start + len].iter().enumerate() {
                    let ang = w * (k as f64);
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
                sum_p += (re * re + im * im) / ((len as f64) * (len as f64));
                cnt += 1;
            }
            10.0 * (sum_p / (cnt as f64) + 1e-30).log10()
        }

        for &(sr, min_drop_db) in &[(48_000u32, 15.0f64), (96_000, 18.0), (192_000, 20.0)] {
            let n = 12_288usize;
            let amp = 10.0_f64.powf(-90.0 / 20.0);
            let src: Vec<f64> = (0..n)
                .map(|i| {
                    amp * (2.0 * std::f64::consts::PI * 1000.0 * (i as f64) / (sr as f64)).sin()
                })
                .collect();

            let mut l_w = src.clone();
            let mut r_w = src.clone();
            let mut qw = PcmQuantizer::new(BitDepth::S16, Dither::Wannamaker, sr).unwrap();
            qw.process(&mut l_w, &mut r_w);

            let mut l_t = src.clone();
            let mut r_t = src.clone();
            let mut qt = PcmQuantizer::new(BitDepth::S16, Dither::Triangular, sr).unwrap();
            qt.process(&mut l_t, &mut r_t);

            // Verify every output sample lies exactly on the 16-bit grid.
            for &v in &l_w {
                let q = v * 32_768.0;
                assert!((q - q.round()).abs() < 1e-9, "off S16 grid: {v}");
            }

            let err_w: Vec<f64> = l_w
                .iter()
                .zip(&src)
                .map(|(&q, &x)| (q - x) * 32_768.0)
                .collect();
            let err_t: Vec<f64> = l_t
                .iter()
                .zip(&src)
                .map(|(&q, &x)| (q - x) * 32_768.0)
                .collect();
            let drop_db = band_power_db(&err_t, sr) - band_power_db(&err_w, sr);
            assert!(
                drop_db >= min_drop_db,
                "sr={sr}: expected >= {min_drop_db} dB noise-floor drop in 1-5 kHz, got {drop_db:.2} dB"
            );

            // Verify -90 dBFS 1 kHz fundamental is recovered cleanly without harmonic distortion
            // (2nd/3rd harmonics stay far below the fundamental).
            let start = 4096usize;
            let len = 8192usize;
            let tone_mag = |sig: &[f64], freq_hz: f64| -> f64 {
                let w = 2.0 * std::f64::consts::PI * freq_hz / (sr as f64);
                let mut re = 0.0;
                let mut im = 0.0;
                for (k, &v) in sig[start..start + len].iter().enumerate() {
                    let ang = w * (k as f64);
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
                2.0 * (re * re + im * im).sqrt() / (len as f64)
            };
            let fund = tone_mag(&l_w, 1000.0);
            let h2 = tone_mag(&l_w, 2000.0);
            let h3 = tone_mag(&l_w, 3000.0);
            assert!(
                (fund / amp - 1.0).abs() < 0.12,
                "sr={sr}: -90 dBFS 1 kHz fundamental drifted: got {fund:.3e} vs {amp:.3e}"
            );
            assert!(
                h2 < fund * 0.2 && h3 < fund * 0.2,
                "sr={sr}: unexpected harmonic distortion at -90 dBFS: h2={h2:.3e} h3={h3:.3e} fund={fund:.3e}"
            );
        }

        // Overload recovery & 24-bit grid exactness check.
        let mut g24 = Grid24::with_sample_rate(Dither::Wannamaker, 48_000).unwrap();
        let mut l = vec![1.0; 256];
        l.extend(vec![0.0; 2048]);
        let mut r = l.clone();
        g24.process(&mut l, &mut r);
        assert!(l.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0));
        let tail_max = l[512..].iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        assert!(
            tail_max < 1e-5,
            "overload recovery tail too large at 24-bit: {tail_max}"
        );
    }
}
