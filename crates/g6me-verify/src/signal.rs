//! Deterministic test signals, shared by every check.
//!
//! Determinism is not a nicety here: a verification report that cannot be
//! re-generated sample-for-sample proves nothing about a failure someone else
//! saw. Every generator is a pure function of its parameters, seeded from a
//! counter rather than a clock, and uses no floating-point transcendental where a
//! cheap exact recurrence will do — `sin` is fine to *call* (both sides call the
//! same one) but the signal itself must not depend on libm's rounding choices in
//! a way that changes between the two binaries being compared, so amplitudes are
//! quantised to f32 immediately after generation.

/// What kind of signal to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// One impulse per channel, at a distinct frame each: measures delay,
    /// polarity and channel routing in one pass.
    Impulse,
    /// Per-channel linear chirp: excites every filter over the whole band.
    Sweep,
    /// Deterministic pseudo-noise: the closest thing to real material.
    Noise,
    /// Inharmonic partial sum — long, loud, and the one that accumulates filter
    /// state the way a recording does.
    Partials,
    /// Near-full-scale alternating signal: headroom and clipping behaviour.
    Hot,
    /// Constant +1/-1 per channel: DC through the recursive sections.
    Dc,
    /// Exact zeros.
    Silence,
    /// Values around the f64 denormal boundary, where a build that turns on
    /// flush-to-zero would visibly diverge.
    Tiny,
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Impulse => "impulse",
            Kind::Sweep => "chirp",
            Kind::Noise => "noise",
            Kind::Partials => "partials",
            Kind::Hot => "hot",
            Kind::Dc => "dc",
            Kind::Silence => "silence",
            Kind::Tiny => "tiny",
        }
    }

    /// The set used for the bit-exactness sweep.
    pub fn all() -> &'static [Kind] {
        &[
            Kind::Impulse,
            Kind::Sweep,
            Kind::Noise,
            Kind::Partials,
            Kind::Hot,
            Kind::Dc,
            Kind::Silence,
            Kind::Tiny,
        ]
    }
}

/// Small fast deterministic PRNG (xorshift64*), so a signal is reproducible
/// across processes and platforms.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        u * 2.0 - 1.0
    }
}

/// Generate `frames * channels` interleaved f32 samples.
pub fn generate(kind: Kind, frames: usize, channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    let mut v = vec![0.0f32; frames * ch];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (frames as u64) << 8 ^ channels as u64);
    match kind {
        Kind::Impulse => {
            // Two impulses per channel, at distinct frames, with a mix of signs and
            // magnitudes: routing, polarity, gain and filter state between transients
            // are all visible in one pass. A single impulse would let a DSP that
            // forgot its state between bursts slip through.
            for c in 0..ch {
                let f1 = (c * 137 + 29).min(frames.saturating_sub(1));
                let f2 = (f1 / 2 + 7).min(frames.saturating_sub(1));
                v[f1 * ch + c] = if c % 2 == 0 { 0.75 } else { -0.5 };
                v[f2 * ch + c] = if c % 3 == 0 { -0.25 } else { 0.125 };
            }
            // Plus one full-scale impulse in channel 0 for an absolute reference.
            if frames > 7 {
                v[7 * ch] = 1.0;
            }
        }
        Kind::Sweep => {
            for c in 0..ch {
                let f0 = 20.0f64 * (1.0 + c as f64);
                let f1 = 4000.0f64 * (1.0 + c as f64 * 0.25);
                let mut phase = 0.0f64;
                for i in 0..frames {
                    let t = i as f64 / frames as f64;
                    let freq = f0 * (f1 / f0).powf(t);
                    phase += std::f64::consts::TAU * freq / 48000.0;
                    if phase > std::f64::consts::TAU {
                        phase -= std::f64::consts::TAU;
                    }
                    v[i * ch + c] = (phase.sin() * 0.4) as f32;
                }
            }
        }
        Kind::Noise => {
            for s in v.iter_mut() {
                *s = rng.unit() * 0.3;
            }
        }
        Kind::Partials => {
            // Five inharmonic partials per channel with slow amplitude drift.
            for c in 0..ch {
                let base = 110.0f64 * (1.0 + 0.37 * c as f64);
                let mut acc = vec![0.0f32; frames];
                for k in 1..=5u32 {
                    let mut ph = 0.0f64;
                    let step = std::f64::consts::TAU * base * (k as f64 * 1.31) / 48000.0;
                    for (i, a) in acc.iter_mut().enumerate() {
                        ph += step;
                        if ph > std::f64::consts::TAU {
                            ph -= std::f64::consts::TAU;
                        }
                        let env = 0.5 + 0.5 * (i as f64 * 1e-5).sin();
                        *a += (ph.sin() * 0.18 * env / k as f64) as f32;
                    }
                }
                // A little noise on top, so nothing is ever exactly periodic.
                for i in 0..frames {
                    v[i * ch + c] = acc[i] + rng.unit() * 0.01;
                }
            }
        }
        Kind::Hot => {
            for i in 0..frames {
                for c in 0..ch {
                    let s = if (i + c) % 2 == 0 {
                        0.999969
                    } else {
                        -0.999939
                    };
                    v[i * ch + c] = s;
                }
            }
        }
        Kind::Dc => {
            for i in 0..frames {
                for c in 0..ch {
                    v[i * ch + c] = if c % 3 == 0 { -0.25 } else { 0.5 };
                }
            }
        }
        Kind::Silence => {}
        Kind::Tiny => {
            // Straddle the double denormal threshold (2.2e-308) and the f32 one.
            for i in 0..frames {
                for c in 0..ch {
                    let e = 1.0e-38f32 * (1.0 + ((i % 7) as f32) * 0.1);
                    v[i * ch + c] = if (i + c) % 2 == 0 { e } else { -e * 0.5 };
                }
            }
        }
    }
    v
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for s in v {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

pub fn f32_from_bytes(b: &[u8]) -> Vec<f32> {
    // A trailing partial sample is dropped rather than guessed at: a truncated raw
    // stream is a bug upstream, and inventing a sample would hide it.
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// How two renderings differ, at the bit level.
#[derive(Clone, Copy, Debug, Default)]
pub struct Diff {
    pub samples: usize,
    pub differing: usize,
    /// Largest |a-b| over finite pairs.
    pub max_abs: f64,
    /// Largest distance in representable f32 steps (ULPs).
    pub max_ulp: i64,
    /// Largest |a-b| / max(|a|,|b|) over non-zero finite pairs. Raw ULP distance is
    /// meaningless near zero (an ulp there is ~1e-45), so relative error is what
    /// says whether a difference actually matters.
    pub max_rel: f64,
    /// Samples that differ only in sign of zero, which compare equal numerically
    /// but are a real bit difference worth reporting.
    pub signed_zero: usize,
    /// First differing frame index, for a reproducer.
    pub first_frame: Option<usize>,
}

impl Diff {
    pub fn exact(&self) -> bool {
        self.differing == 0
    }

    /// Signal-to-difference ratio in dB, `None` when the difference is zero
    /// (which is not "infinite quality", it is "identical").
    pub fn snr_db(&self) -> Option<f64> {
        if self.differing == 0 {
            return None;
        }
        Some(-20.0 * self.max_abs.log10())
    }
}

/// Compare two stereo f32 streams bit-for-bit.
pub fn compare(a: &[f32], b: &[f32]) -> Diff {
    let mut d = Diff::default();
    let n = a.len().min(b.len());
    d.samples = n / 2;
    if a.len() != b.len() {
        d.differing = (a.len().max(b.len()) - n) / 2;
        d.first_frame = Some(d.samples);
        d.max_abs = f64::INFINITY;
        return d;
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let frame = i / 2;
        if x.to_bits() == y.to_bits() {
            continue;
        }
        if x == y {
            // +0.0 vs -0.0
            d.signed_zero += 1;
            d.differing += 1;
            d.first_frame.get_or_insert(frame);
            continue;
        }
        d.differing += 1;
        d.first_frame.get_or_insert(frame);
        if x.is_finite() && y.is_finite() {
            let diff = (x - y).abs() as f64;
            d.max_abs = d.max_abs.max(diff);
            let scale = x.abs().max(y.abs()) as f64;
            if scale > 0.0 {
                d.max_rel = d.max_rel.max(diff / scale);
            }
            let ux = x.to_bits() as i32;
            let uy = y.to_bits() as i32;
            // Ordered f32 space: negative floats count backwards.
            let ox = if ux < 0 {
                0x8000_0000i64 - ux as i64
            } else {
                ux as i64
            };
            let oy = if uy < 0 {
                0x8000_0000i64 - uy as i64
            } else {
                uy as i64
            };
            d.max_ulp = d.max_ulp.max((ox - oy).abs());
        } else {
            d.max_abs = f64::INFINITY;
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_are_reproducible() {
        for k in Kind::all() {
            let a = generate(*k, 512, 8);
            let b = generate(*k, 512, 8);
            assert_eq!(a, b, "{} must be deterministic", k.label());
            assert_eq!(a.len(), 512 * 8, "{} length", k.label());
        }
    }

    #[test]
    fn signals_are_not_all_zero_or_nan() {
        for k in Kind::all() {
            let v = generate(*k, 512, 8);
            assert!(
                v.iter().all(|s| s.is_finite()),
                "{} produced a non-finite sample",
                k.label()
            );
            if *k != Kind::Silence {
                assert!(
                    v.iter().any(|s| *s != 0.0),
                    "{} is silent, so it would verify nothing",
                    k.label()
                );
            }
        }
    }

    #[test]
    fn impulse_lands_in_its_own_channel_and_frame() {
        let v = generate(Kind::Impulse, 4096, 8);
        let mut seen = [0usize; 8];
        for f in 0..4096 {
            for c in 0..8 {
                if v[f * 8 + c] != 0.0 {
                    seen[c] += 1;
                }
            }
        }
        for (c, n) in seen.iter().enumerate() {
            assert!(*n >= 2, "channel {c} got only {n} non-zero samples");
        }
    }

    #[test]
    fn round_trips_through_bytes() {
        let v = generate(Kind::Noise, 300, 2);
        assert_eq!(f32_from_bytes(&f32_bytes(&v)), v);
    }

    #[test]
    fn compare_counts_bits_not_just_values() {
        let a = vec![1.0f32, -0.0, 1.0, 1.0];
        let b = vec![1.0f32, 0.0, 1.0000001, 1.0];
        let d = compare(&a, &b);
        assert_eq!(d.samples, 2);
        assert_eq!(d.differing, 2, "signed zero and a real change");
        assert_eq!(d.signed_zero, 1);
        assert_eq!(
            d.first_frame,
            Some(0),
            "first_frame points at the first byte difference, and a signed zero is one"
        );
        assert_eq!(d.max_ulp, 1, "one f32 step apart");
        assert!(!d.exact());
        assert!(d.snr_db().is_some());

        let same = compare(&a, &a);
        assert!(same.exact() && same.snr_db().is_none());
        // A length mismatch is a hard difference, not a silent truncation.
        let short = compare(&a, &a[..2]);
        assert!(!short.exact());
    }

    #[test]
    fn ulp_distance_handles_negatives() {
        // -1.0 and the next representable value towards zero are one step apart,
        // even though their bit patterns are far apart as unsigned integers.
        let one = (-1.0f32).next_up();
        let d = compare(&[-1.0f32, 0.0], &[one, 0.0]);
        assert_eq!(d.max_ulp, 1);
    }
}
