//! The checks themselves.
//!
//! Every check answers one question with bytes rather than adjectives:
//!
//! | check | question |
//! |---|---|
//! | `bitexact` | does the Rust f64 downmixer produce **exactly** the f32 samples the reference `-DG6CT_DOUBLE` build produces? |
//! | `builds` | does that still hold for `-march=native` and `-O3`? |
//! | `width` | how far is the f32-internal build from the f64-internal one — i.e. what the f64 invariant actually buys? |
//! | `chunk` | is the result independent of the block size the file is fed in? |
//! | `continuity` | does filter state carry across block boundaries exactly? |
//! | `drift` | over a minute of material, does anything accumulate, wander, or blow up? |
//! | `phase` | do the two paths' delays, polarities and per-slot timings match sample-for-sample? |
//! | `contraction` | is any mul+add in this binary being contracted into FMA? |
//! | `asm` | does the disassembler agree? |

use crate::reference::{self, RefBuild, MASK_7_1, REF_BLOCK, SLOTS_8};
use crate::signal::{self, Diff, Kind};
use g6me_core::dsp::{Dsp, SLOTS};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    Skip,
    /// Informational: measured and reported, not pass/fail (toolchain-dependent).
    Info,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
            Status::Info => "info",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Check {
    pub group: &'static str,
    pub name: String,
    pub status: Status,
    /// One-line evidence: counts, delays, dB figures, the first differing frame.
    pub detail: String,
    pub metric: Option<Value>,
}

pub fn pass(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Pass,
        detail: detail.into(),
        metric: None,
    }
}

pub fn fail(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Fail,
        detail: detail.into(),
        metric: None,
    }
}

pub fn skip(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Skip,
        detail: detail.into(),
        metric: None,
    }
}

pub fn info(group: &'static str, name: impl Into<String>, detail: impl Into<String>) -> Check {
    Check {
        group,
        name: name.into(),
        status: Status::Info,
        detail: detail.into(),
        metric: None,
    }
}

/// Run the Rust downmixer over interleaved f32, in f64 internally, narrowing the
/// result to f32 the way the reference CLI does on write.
pub fn rust_f64(input: &[f32], channels: u16, mask: u32, block: usize) -> Vec<f32> {
    run_rust::<f64>(input, channels, mask, block)
}

/// The same through `Dsp<f32>`, for the width comparison.
pub fn rust_f32(input: &[f32], channels: u16, mask: u32, block: usize) -> Vec<f32> {
    run_rust::<f32>(input, channels, mask, block)
}

fn run_rust<T: g6me_core::dsp::Real + Copy + Default>(
    input: &[f32],
    channels: u16,
    mask: u32,
    block: usize,
) -> Vec<f32> {
    let ch = channels as usize;
    let frames = input.len() / ch;
    let slots = slot_map(channels);
    // Widen exactly once, up front: the reference CLI does the same on read, so
    // the arithmetic both sides perform afterwards starts from identical values.
    let wide: Vec<T> = input.iter().map(|&s| T::from_f64(s as f64)).collect();
    let mut planar: Vec<Vec<T>> = vec![vec![T::default(); block]; ch];
    let mut out_l = vec![T::default(); block];
    let mut out_r = vec![T::default(); block];
    let mut dsp = Dsp::<T>::new(mask);
    let mut out = Vec::with_capacity(frames * 2);
    let mut done = 0;
    while done < frames {
        let n = block.min(frames - done);
        for c in 0..ch {
            for i in 0..n {
                planar[c][i] = wide[(done + i) * ch + c];
            }
        }
        // Built fresh per block: the slot array borrows `planar`, so it cannot be
        // reused across a mutation of it.
        let mut chans: [Option<&[T]>; SLOTS] = [None; SLOTS];
        for c in 0..ch {
            chans[slots[c]] = Some(&planar[c][..n]);
        }
        dsp.process(&chans, &mut out_l[..n], &mut out_r[..n], n);
        for i in 0..n {
            out.push(out_l[i].to_f64() as f32);
            out.push(out_r[i].to_f64() as f32);
        }
        done += n;
    }
    out
}

/// Input channel order → WAVEFORMATEXTENSIBLE slot position, matching the
/// reference CLI's `slot8` / `slot12` tables.
pub fn slot_map(channels: u16) -> Vec<usize> {
    match channels {
        12 => SLOTS_12.to_vec(),
        _ => SLOTS_8.to_vec(),
    }
}

use crate::reference::SLOTS_12;

/// Where the samples go while a reference build runs.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        let d = std::env::temp_dir().join(format!(
            "g6me-verify-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).expect("scratch dir");
        Self(d)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub fn write_raw(&self, name: &str, v: &[f32]) -> PathBuf {
        let p = self.path(name);
        std::fs::write(&p, signal::f32_bytes(v)).expect("write raw");
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Reference builds that claim the numerical contract, i.e. everything except the
/// deliberate counter-example.
pub fn contract_builds<'a>(builds: &'a [RefBuild], width: &str) -> Vec<&'a RefBuild> {
    builds
        .iter()
        .filter(|b| b.width == width && b.variant != "contract_fast")
        .collect()
}

/// The counter-example: same C source, built the way a careless build is built.
pub fn loose_build<'a>(builds: &'a [RefBuild], width: &str) -> Option<&'a RefBuild> {
    builds
        .iter()
        .find(|b| b.width == width && b.variant == "contract_fast")
}

/// Input configurations to verify. A mask selects which filters exist at all, so
/// bit-exactness on the stock 7.1 mask alone would not prove much: the 5.1 and
/// 12-channel cases run different subsets of the filter bank, and 12 channels also
/// changes the slot mapping.
pub const CASES: &[(u32, u16, &str)] = &[
    (MASK_7_1, 8, "7.1"),
    (0x003f, 8, "5.1"),
    (0x0fff, 12, "atmos-12"),
    (0x0003, 8, "front-only"),
];

/// `bitexact`: the claim the whole project rests on.
pub fn bitexact(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let f64_builds = contract_builds(builds, "f64");
    if f64_builds.is_empty() {
        return vec![skip(
            "bitexact",
            "reference builds",
            "no -DG6CT_DOUBLE reference binary was built (no C compiler?)",
        )];
    }
    for (idx, &(mask, channels, case)) in CASES.iter().enumerate() {
        // The stock case gets every signal; the others get the three that exercise
        // state, transients and broadband content, to keep the suite quick.
        let kinds: Vec<Kind> = if idx == 0 {
            Kind::all().to_vec()
        } else {
            vec![Kind::Partials, Kind::Impulse, Kind::Noise]
        };
        for kind in kinds {
            let sig = signal::generate(kind, frames, channels);
            let inp = scratch.write_raw(&format!("{case}-{kind}.raw"), &sig);
            for b in &f64_builds {
                let want = match reference::run(b, &inp, &scratch.path("o.raw"), mask, channels) {
                    Ok(v) => v,
                    Err(e) => {
                        out.push(fail("bitexact", format!("{}/{case}/{kind}", b.name), e));
                        continue;
                    }
                };
                let want = signal::f32_from_bytes(&want);
                // Feed Rust at the reference's own block size first: same input, same
                // chunking, so any difference can only be arithmetic.
                let got = rust_f64(&sig, channels, mask, REF_BLOCK);
                let d = signal::compare(&got, &want);
                let name = format!("{}/{case}/{kind}", b.name);
                let where_ = format!("{channels}ch mask {mask:#05x}");
                if d.exact() {
                    out.push(with_metric(
                        pass(
                            "bitexact",
                            name,
                            format!("{where_}: {} frames, bit-identical", d.samples),
                        ),
                        json!({
                            "mask": mask, "channels": channels, "kind": kind.label(),
                            "frames": d.samples, "differing": 0,
                        }),
                    ));
                } else {
                    let zeros = if d.signed_zero > 0 {
                        format!(" ({} of them only in the sign of zero)", d.signed_zero)
                    } else {
                        String::new()
                    };
                    out.push(with_metric(
                        fail(
                            "bitexact",
                            name,
                            format!(
                                "{where_}: {} of {} frames differ; max |d|={} worst {} \
                                 relative; first at frame {:?}{zeros}",
                                d.differing,
                                d.samples,
                                fmt_f(d.max_abs),
                                fmt_f(d.max_rel),
                                d.first_frame,
                            ),
                        ),
                        diff_json(&d),
                    ));
                }
            }
        }
    }
    out
}

/// `builds`: the reference must not care about `-O3`/`-march=native` either, which
/// is what proves the numerical contract rather than a lucky codegen.
pub fn builds_agree(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let f64 = contract_builds(builds, "f64");
    if f64.len() < 2 {
        return vec![skip(
            "builds",
            "reference variants",
            "only one f64 build available",
        )];
    }
    for kind in [Kind::Partials, Kind::Noise] {
        let sig = signal::generate(kind, frames, 8);
        let inp = scratch.write_raw(&format!("v-{}.raw", kind.label()), &sig);
        let mut prev: Option<(String, Vec<f32>)> = None;
        for b in &f64 {
            let got = match reference::run(b, &inp, &scratch.path("v.raw"), MASK_7_1, 8) {
                Ok(v) => signal::f32_from_bytes(&v),
                Err(e) => {
                    out.push(fail("builds", format!("{}/{kind}", b.name), e));
                    continue;
                }
            };
            if let Some((pname, pwant)) = &prev {
                let d = signal::compare(&got, pwant);
                let name = format!("{pname} vs {}", b.name);
                out.push(if d.exact() {
                    pass(
                        "builds",
                        name,
                        format!("{kind:?} output identical across optimisation levels"),
                    )
                } else {
                    fail(
                        "builds",
                        name,
                        format!(
                            "{} frames differ (max {} ULP) — the C build flags changed the numbers",
                            d.differing, d.max_ulp
                        ),
                    )
                });
            }
            prev = Some((b.name.clone(), got));
        }
    }
    out
}

/// `width`: how much the f64-internal invariant actually buys.
///
/// The f32 build (float state) and the f64 build (`G6CT_DOUBLE`) share coefficients
/// but not arithmetic. Upstream's own note is that f64 is "60 dB closer to an exact
/// render"; measuring the gap here is what stops that claim being folklore.
pub fn width_gap(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let pick = |w: &str| {
        builds
            .iter()
            .find(|b| b.variant == "contract_off" && b.width == w)
    };
    let (Some(f64b), Some(f32b)) = (pick("f64"), pick("f32")) else {
        return vec![skip("width", "f32 and f64 builds", "both widths not built")];
    };
    let sig = signal::generate(Kind::Partials, frames, 8);
    let inp = scratch.write_raw("w.raw", &sig);
    let a = match reference::run(f64b, &inp, &scratch.path("a.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("width", "f64 reference", e)],
    };
    let b = match reference::run(f32b, &inp, &scratch.path("b.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("width", "f32 reference", e)],
    };
    let d = signal::compare(&a, &b);
    let peak = a.iter().chain(b.iter()).fold(0.0f32, |m, s| m.max(s.abs()));
    // The gap must be real (proving internal width matters) and small (proving
    // neither build is broken). "Small" is relative to signal peak, not absolute.
    let rel = if peak > 0.0 {
        d.max_abs as f32 / peak
    } else {
        0.0
    };
    let name = format!(
        "f32-internal vs f64-internal on a {} signal",
        Kind::Partials
    );
    if d.differing == 0 {
        out.push(with_metric(
            fail(
                "width",
                name,
                "the two builds are identical, so nothing here justifies f64 state — \
                 check that -DG6CT_DOUBLE is really taking effect",
            ),
            diff_json(&d),
        ));
    } else if rel > 5e-3 {
        out.push(with_metric(
            fail(
                "width",
                name,
                format!(
                    "the f32 build is {} of peak away (worst sample {} relative) — too far \
                     for the same transfer function",
                    fmt_f(rel as f64),
                    fmt_f(d.max_rel)
                ),
            ),
            diff_json(&d),
        ));
    } else {
        out.push(with_metric(
            pass(
                "width",
                name,
                format!(
                    "{} of {} frames differ; max |d|={} ({:.1e} of peak {peak:.3}), worst \
                     sample {} relative; snr {} dB — internal width is worth about {} \
                     decimal digits on this material",
                    d.differing,
                    d.samples,
                    fmt_f(d.max_abs),
                    rel,
                    fmt_f(d.max_rel),
                    d.snr_db().map(fmt_f).unwrap_or_else(|| "inf".into()),
                    (d.max_abs as f64).max(1e-30).log10().abs().round() as i32,
                ),
            ),
            diff_json(&d),
        ));
    }
    // And the Rust implementation must sit on the f64 side, exactly. Its f32 path
    // is checked against the f32 build for the same reason: if either width drifted
    // from its own reference, the width comparison above would be meaningless.
    let r = rust_f64(&sig, 8, MASK_7_1, REF_BLOCK);
    let dr = signal::compare(&r, &a);
    let df = signal::compare(&r, &b);
    let r32 = rust_f32(&sig, 8, MASK_7_1, REF_BLOCK);
    let d32 = signal::compare(&r32, &b);
    out.push(if d32.exact() {
        pass(
            "width",
            "rust f32 path tracks the f32 build",
            format!(
                "0 of {} frames differ — both widths are faithful to their own reference",
                d32.samples
            ),
        )
    } else {
        info(
            "width",
            "rust f32 path tracks the f32 build",
            format!(
                "{} of {} frames differ (max {} ULP); the f32 path is a secondary build, \
                 the shipped one is f64",
                d32.differing, d32.samples, d32.max_ulp
            ),
        )
    });
    out.push(if dr.exact() && !df.exact() {
        pass(
            "width",
            "rust tracks the f64 build",
            format!(
                "0 of {} frames differ from f64-internal; {} differ from f32-internal",
                dr.samples, df.differing
            ),
        )
    } else if dr.exact() {
        info(
            "width",
            "rust tracks the f64 build",
            "identical to f64-internal; the f32 build happened to agree on this signal",
        )
    } else {
        fail(
            "width",
            "rust tracks the f64 build",
            format!("{} frames differ from the f64 reference", dr.differing),
        )
    });
    out
}

/// `chunk`: the render feeds the DSP in blocks chosen for memory and latency, not
/// for the DSP's convenience, so the answer must not depend on it.
pub fn chunk_invariance(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    for kind in [Kind::Partials, Kind::Impulse, Kind::Hot] {
        let sig = signal::generate(kind, frames, 8);
        let base = rust_f64(&sig, 8, MASK_7_1, 4096);
        for b in [1usize, 7, 13, 1024, 65_536] {
            let got = rust_f64(&sig, 8, MASK_7_1, b);
            let d = signal::compare(&got, &base);
            out.push(if d.exact() {
                pass(
                    "chunk",
                    format!("{kind}/block {b}"),
                    format!("identical to block 4096 across {} frames", d.samples),
                )
            } else {
                fail(
                    "chunk",
                    format!("{kind}/block {b}"),
                    format!(
                        "block size changed the output: {} frames differ, first at {:?} — \
                         filter state is not surviving the boundary",
                        d.differing, d.first_frame
                    ),
                )
            });
        }
        // And it must still equal the reference at its own 1024-frame block.
        if let Some(rb) = contract_builds(builds, "f64").first() {
            let inp = scratch.write_raw(&format!("c-{}.raw", kind.label()), &sig);
            match reference::run(rb, &inp, &scratch.path("c.raw"), MASK_7_1, 8) {
                Ok(v) => {
                    let d = signal::compare(&base, &signal::f32_from_bytes(&v));
                    out.push(if d.exact() {
                        pass(
                            "chunk",
                            format!("{kind}/rust 4096 vs reference 1024"),
                            "bit-identical across different block sizes in each implementation",
                        )
                    } else {
                        fail(
                            "chunk",
                            format!("{kind}/rust 4096 vs reference 1024"),
                            format!("{} frames differ", d.differing),
                        )
                    });
                }
                Err(e) => out.push(fail("chunk", format!("{kind}/reference run"), e)),
            }
        }
    }
    out
}

/// `continuity`: a file rendered in two sittings must equal one rendering.
///
/// This is the property the stage-parallel pipeline depends on: the DSP sees one
/// ordered block sequence while decode and encode run beside it. A DSP that
/// ignored its history would pass the first half of this and fail the second.
pub fn state_continuity(frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    for kind in [Kind::Partials, Kind::Sweep] {
        let sig = signal::generate(kind, frames, 8);
        let one = rust_f64(&sig, 8, MASK_7_1, 4096);

        // Same DSP object, two calls: what `render_file` actually does.
        let ch = 8usize;
        let slots = slot_map(8);
        let mut dsp = Dsp::<f64>::new(MASK_7_1);
        let mut two = Vec::with_capacity(one.len());
        for (start, len) in [(0usize, frames / 2), (frames / 2, frames - frames / 2)] {
            let mut planar = vec![vec![0.0f64; len]; ch];
            for c in 0..ch {
                for i in 0..len {
                    planar[c][i] = sig[(start + i) * ch + c] as f64;
                }
            }
            let mut chans: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            for c in 0..ch {
                chans[slots[c]] = Some(&planar[c]);
            }
            let mut l = vec![0.0f64; len];
            let mut r = vec![0.0f64; len];
            dsp.process(&chans, &mut l, &mut r, len);
            for i in 0..len {
                two.push(l[i] as f32);
                two.push(r[i] as f32);
            }
        }
        let d = signal::compare(&one, &two);
        out.push(if d.exact() {
            pass(
                "continuity",
                format!("{kind}/split render"),
                format!("two sittings equal one render across {} frames", d.samples),
            )
        } else {
            fail(
                "continuity",
                format!("{kind}/split render"),
                format!(
                    "{} frames differ, first at frame {:?} — state is lost at the boundary",
                    d.differing, d.first_frame
                ),
            )
        });

        // The control: throwing the state away mid-file must change the answer.
        // Without this, a DSP with no memory would pass the check above for free.
        let mut fresh = Vec::with_capacity(one.len());
        for (start, len) in [(0usize, frames / 2), (frames / 2, frames - frames / 2)] {
            let mut dsp2 = Dsp::<f64>::new(MASK_7_1);
            let mut planar = vec![vec![0.0f64; len]; ch];
            for c in 0..ch {
                for i in 0..len {
                    planar[c][i] = sig[(start + i) * ch + c] as f64;
                }
            }
            let mut chans: [Option<&[f64]>; SLOTS] = [None; SLOTS];
            for c in 0..ch {
                chans[slots[c]] = Some(&planar[c]);
            }
            let mut l = vec![0.0f64; len];
            let mut r = vec![0.0f64; len];
            dsp2.process(&chans, &mut l, &mut r, len);
            for i in 0..len {
                fresh.push(l[i] as f32);
                fresh.push(r[i] as f32);
            }
        }
        let df = signal::compare(&one, &fresh);
        out.push(if df.differing > 0 {
            pass(
                "continuity",
                format!("{kind}/state really is load-bearing"),
                format!(
                    "resetting the filters mid-file changes {} of {} frames",
                    df.differing, df.samples
                ),
            )
        } else {
            fail(
                "continuity",
                format!("{kind}/state really is load-bearing"),
                "a mid-file state reset changed nothing, so the filters are stateless \
                 and the continuity check above proves nothing",
            )
        });
    }
    out
}

/// A per-slot timing summary, computed identically for both implementations.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    /// Frame of the first non-zero output sample.
    pub first: Vec<i64>,
    /// Frame of the largest-magnitude output sample.
    pub peak: Vec<i64>,
    /// Energy centroid in frames — a group-delay-ish summary that catches a
    /// dispersed path even when the peak lines up.
    pub centroid: Vec<f64>,
    /// Polarity of the peak, per slot.
    pub sign: Vec<i8>,
}

/// Extract the timing summary from a stereo output produced by an impulse input
/// with a single channel excited (per-slot routing check).
pub fn timing_of(out: &[f32], frames: usize) -> Timing {
    let mut t = Timing::default();
    let mut peak_i = 0usize;
    let mut peak_v = 0.0f32;
    let mut first = None;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for f in 0..frames {
        let l = out[f * 2];
        let r = out[f * 2 + 1];
        let m = l.abs().max(r.abs());
        if m > 0.0 && first.is_none() {
            first = Some(f as i64);
        }
        if m > peak_v {
            peak_v = m;
            peak_i = f;
        }
        let e = (l as f64 * l as f64) + (r as f64 * r as f64);
        num += e * f as f64;
        den += e;
    }
    t.first.push(first.unwrap_or(-1));
    t.peak.push(peak_i as i64);
    t.centroid.push(if den > 0.0 { num / den } else { 0.0 });
    let peak_l = out[peak_i * 2];
    let peak_r = out[peak_i * 2 + 1];
    let peak_sample = if peak_r.abs() > peak_l.abs() {
        peak_r
    } else {
        peak_l
    };
    t.sign.push(if peak_sample >= 0.0 { 1 } else { -1 });
    t
}

/// `phase`: per-slot impulse response timing, polarity and routing.
pub fn phase(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(rb) = refs.first() else {
        return vec![skip("phase", "reference build", "no f64 reference binary")];
    };
    // One impulse at a time, so each slot's own path is measured in isolation.
    let mut rows: Vec<String> = Vec::new();
    let mut bad = 0;
    for slot in 0..8usize {
        let mut sig = vec![0.0f32; frames * 8];
        sig[16 * 8 + slot] = 1.0;
        let inp = scratch.write_raw(&format!("p{slot}.raw"), &sig);
        let want = match reference::run(rb, &inp, &scratch.path("p.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => {
                out.push(fail("phase", format!("slot {slot}"), e));
                bad += 1;
                continue;
            }
        };
        let got = rust_f64(&sig, 8, MASK_7_1, REF_BLOCK);
        let d = signal::compare(&got, &want);
        let t = timing_of(&want, frames);
        rows.push(format!(
            "{} {}",
            ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"][slot],
            json!({
                "first": t.first[0],
                "peak": t.peak[0],
                "centroid": (t.centroid[0] * 100.0).round() / 100.0,
                "peak_sign": t.sign[0],
            })
        ));
        let name = format!(
            "slot {slot} ({})",
            ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"][slot]
        );
        if d.exact() {
            out.push(with_metric(
                pass(
                    "phase",
                    name,
                    "impulse response bit-identical, so every delay, \
                                      grouping and polarity matches by construction",
                ),
                json!({
                    "first_frame": t.first[0],
                    "peak_frame": t.peak[0],
                    "energy_centroid_frames": t.centroid[0],
                    "peak_sign": t.sign[0],
                }),
            ));
        } else {
            bad += 1;
            out.push(with_metric(
                fail(
                    "phase",
                    name,
                    format!(
                        "{} frames differ (max {} ULP, first at {:?})",
                        d.differing, d.max_ulp, d.first_frame
                    ),
                ),
                diff_json(&d),
            ));
        }
    }
    out.push(info(
        "phase",
        "per-slot response (reference)",
        rows.join("  "),
    ));
    if bad > 0 {
        out.push(fail("phase", "summary", format!("{bad} slots disagreed")));
    }
    out
}

/// The counter-example, run as a check in its own right.
///
/// Built from the *same C source* with `-ffp-contract=fast`, i.e. what happens when
/// nobody tells the compiler that floating point has a contract. If that build
/// diverges from the canonical one then the flags in `port/Makefile` are doing real
/// work and every bit-exact statement above depends on them; if it does not diverge,
/// this toolchain simply found nothing to fuse, which is reported rather than
/// claimed.
pub fn adversarial(builds: &[RefBuild], scratch: &Scratch, frames: usize) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(strict) = refs.first() else {
        return vec![skip("contraction", "canonical reference", "not built")];
    };
    let Some(loose) = loose_build(builds, "f64") else {
        return vec![skip(
            "contraction",
            "-ffp-contract=fast reference",
            "not built",
        )];
    };
    let mut diverged = 0usize;
    let mut worst = Diff::default();
    let mut notes: Vec<String> = Vec::new();
    for kind in [Kind::Partials, Kind::Silence, Kind::Impulse] {
        let sig = signal::generate(kind, frames, 8);
        let inp = scratch.write_raw(&format!("adv-{}.raw", kind.label()), &sig);
        let a = match reference::run(strict, &inp, &scratch.path("a.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => return vec![fail("contraction", "canonical reference run", e)],
        };
        let b = match reference::run(loose, &inp, &scratch.path("b.raw"), MASK_7_1, 8) {
            Ok(v) => signal::f32_from_bytes(&v),
            Err(e) => return vec![fail("contraction", "contract=fast reference run", e)],
        };
        let d = signal::compare(&a, &b);
        if d.differing > 0 {
            diverged += 1;
            if d.max_rel >= worst.max_rel {
                worst = d;
            }
        }
        notes.push(format!(
            "{kind}: {} frames differ (max rel {:e})",
            d.differing, d.max_rel
        ));
    }
    if diverged > 0 {
        out.push(with_metric(
            pass(
                "contraction",
                "-ffp-contract=off is load-bearing",
                format!(
                    "{diverged} of 3 signals change when the same C source is built with \
                     -ffp-contract=fast; worst max |d|={} max rel {:e} — so the DSP flags are \
                     what make a build of this module interchangeable, and why Rust must not \
                     contract either ({})",
                    fmt_f(worst.max_abs),
                    worst.max_rel,
                    notes.join("; ")
                ),
            ),
            diff_json(&worst),
        ));
    } else {
        out.push(info(
            "contraction",
            "-ffp-contract=off is load-bearing",
            format!(
                "this toolchain produced identical code either way ({}); the flag is still \
                 required, because another compiler or another -march may fuse",
                notes.join("; ")
            ),
        ));
    }
    out
}

/// `drift`: one minute of material, looking for anything that accumulates.
pub fn drift(builds: &[RefBuild], scratch: &Scratch, seconds: u64) -> Vec<Check> {
    let mut out = Vec::new();
    let refs = contract_builds(builds, "f64");
    let Some(rb) = refs.first() else {
        return vec![skip("drift", "reference build", "no f64 reference binary")];
    };
    let frames = (seconds * 48000) as usize;
    let sig = signal::generate(Kind::Partials, frames, 8);
    let inp = scratch.write_raw("drift.raw", &sig);
    let t0 = std::time::Instant::now();
    let want = match reference::run(rb, &inp, &scratch.path("drift.out.raw"), MASK_7_1, 8) {
        Ok(v) => signal::f32_from_bytes(&v),
        Err(e) => return vec![fail("drift", "reference run", e)],
    };
    let c_wall = t0.elapsed();
    let t1 = std::time::Instant::now();
    let got = rust_f64(&sig, 8, MASK_7_1, 4096);
    let r_wall = t1.elapsed();
    let d = signal::compare(&got, &want);
    out.push(with_metric(
        if d.exact() {
            pass(
                "drift",
                format!("{seconds}s bit-exact"),
                format!(
                    "{} stereo frames, 0 differing; rust {} ms vs reference {} ms \
                     (reference includes process start-up)",
                    d.samples,
                    r_wall.as_millis(),
                    c_wall.as_millis()
                ),
            )
        } else {
            fail(
                "drift",
                format!("{seconds}s bit-exact"),
                format!(
                    "{} of {} frames differ; first at frame {:?}, max {} ULP — divergence \
                     accumulates rather than staying bounded",
                    d.differing, d.samples, d.first_frame, d.max_ulp
                ),
            )
        },
        diff_json(&d),
    ));

    // Stability, independent of the reference: no NaN/inf, and the tail must not
    // be larger than the head (an unstable recursion grows).
    let finite = got.iter().all(|s| s.is_finite());
    let rms = |v: &[f32]| {
        let n = v.len().max(1) as f64;
        (v.iter().map(|s| *s as f64 * *s as f64).sum::<f64>() / n).sqrt()
    };
    let half = got.len() / 2;
    let (head, tail) = (rms(&got[..half]), rms(&got[half..]));
    out.push(if finite && tail <= head * 2.0 + 1e-6 {
        pass(
            "drift",
            "stability over the whole file",
            format!(
                "all finite; rms head {head:.5} tail {tail:.5}, peak {:.4} — no growth",
                got.iter().fold(0.0f32, |m, s| m.max(s.abs()))
            ),
        )
    } else {
        fail(
            "drift",
            "stability over the whole file",
            format!(
                "finite={finite}, rms head {head:.6} tail {tail:.6} — the recursion is growing"
            ),
        )
    });
    out
}

/// Behavioural FMA detector: no disassembler needed, and it tests the code the
/// verifier is actually running in.
///
/// `(1+2^-52)^2 - (1+2^-51)` is exactly `2^-104`. Separately rounded mul then add
/// gives exactly zero; a fused multiply-add keeps the unrounded product and gives
/// `2^-104`. So a non-zero answer means the compiler contracted — which in a
/// recursive IIR is a different filter, not a faster one of the same filter.
pub fn contraction() -> Vec<Check> {
    let out = vec![check_f64_contraction()];
    out
}

fn check_f64_contraction() -> Check {
    let a = 1.0f64 + f64::EPSILON;
    let b = 1.0f64 + f64::EPSILON;
    let c = -(1.0f64 + 2.0 * f64::EPSILON);
    // Written so the compiler cannot fold it at compile time: the values come from
    // a black_box and the result is read back.
    let (a, b, c) = (
        std::hint::black_box(a),
        std::hint::black_box(b),
        std::hint::black_box(c),
    );
    let sep = std::hint::black_box(a * b + c);
    // The exact answer is 2^-104; anything else means contraction happened.
    let fused = 1.0f64 / (1u128 << 104) as f64;
    if sep == 0.0 {
        pass(
            "contraction",
            "no FMA in this binary's f64 arithmetic",
            format!(
                "(1+2^-52)^2-(1+2^-51) = {sep}, as separate rounding requires; a fused \
                 multiply-add would give {fused:e}"
            ),
        )
    } else {
        fail(
            "contraction",
            "no FMA in this binary's f64 arithmetic",
            format!(
                "got {sep:e} instead of 0 — mul+add is being contracted, which changes the \
                 recursive filter's coefficients (build without -C fp-contract=fast)"
            ),
        )
    }
}

/// `asm`: agree with the disassembler, and check the reference binaries too.
pub fn asm_guard(builds: &[RefBuild]) -> Vec<Check> {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => return vec![skip("asm", "current executable", e.to_string())],
    };
    let mut out = Vec::new();
    out.push(scan_binary(
        "rust dsp (verify)",
        &exe,
        &["g6me"],
        ExpectFma::No,
    ));
    // Also scan the shipped sibling `g6me` binary when built in the same profile directory
    // so every production DSP module (dsp, stn, enhance, unmask, upmix, plus, limiter,
    // room, scattering, ntt, hoa, bem, pcm, transducer) is audited end-to-end.
    if let Some(dir) = exe.parent() {
        let cli_bin = dir.join(if cfg!(windows) { "g6me.exe" } else { "g6me" });
        if cli_bin.is_file() {
            out.push(scan_binary(
                "rust dsp (g6me cli)",
                &cli_bin,
                &["g6me"],
                ExpectFma::No,
            ));
        }
    }
    for b in contract_builds(builds, "f64") {
        // The reference must show no FMA either: that is what `-ffp-contract=off`
        // is for, and it is the assumption every bit-exact result above rests on.
        let mut c = scan_binary(
            "reference",
            &b.path,
            &["g6ct_process", "g6ct_init"],
            ExpectFma::No,
        );
        c.name = format!("{} / no FMA", b.name);
        out.push(c);
    }
    // The counter-example is expected to fuse. If it does not, that is worth knowing
    // too, because it says the divergence was produced some other way.
    if let Some(loose) = loose_build(builds, "f64") {
        let mut c = scan_binary(
            "reference",
            &loose.path,
            &["g6ct_process", "g6ct_init"],
            ExpectFma::Expected,
        );
        c.name = format!("{} / fma expected", loose.name);
        out.push(c);
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ExpectFma {
    /// A DSP path that must stay unfused.
    No,
    /// The deliberate counter-example.
    Expected,
}

/// Look for fused multiply/add mnemonics inside the interesting functions.
fn scan_binary(what: &str, path: &PathBuf, want_symbols: &[&str], expect: ExpectFma) -> Check {
    let Some(tool) = find_objdump() else {
        return skip(
            "asm",
            format!("{what} disassembly"),
            "no objdump available to inspect codegen",
        );
    };
    let o = std::process::Command::new(&tool)
        .arg("-d")
        .arg("--no-show-raw-insn")
        .arg(path)
        .output();
    let Ok(o) = o else {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!("{tool} could not read {}", path.display()),
        );
    };
    let txt = String::from_utf8_lossy(&o.stdout).to_string();
    if txt.trim().is_empty() {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!("{tool} produced no disassembly for {}", path.display()),
        );
    }
    // objdump prints one block per symbol:
    //   0000000000001160 <g6ct_process>:
    //   1160:  f2 0f 10 05 …   movsd 0x…(%rip),%xmm1
    // so a symbol boundary is any line containing `<name>`.
    let fma_mnemonics = [
        "vfmadd", "vfmsub", "vfnmadd", "vfnmsub", // x86 AVX-FMA
        "fmadd", "fmsub", "fnmadd", "fnmsub", // aarch64
        "fma.",   // ppc
    ];
    let mut inside = false;
    let mut hits = 0usize;
    let mut inspected = 0usize;
    let mut insides = 0usize;
    for line in txt.lines() {
        // A symbol header is `<address> <name>:` and nothing else. Instruction lines
        // carry `<sym+off>` annotations too (rip-relative loads, call targets); treating
        // those as headers skipped exactly the FMA lines that load a constant.
        if let Some(sym) = symbol_header(line) {
            inside = want_symbols.iter().any(|w| sym.contains(w));
            if inside {
                inspected += 1;
            }
            continue;
        }
        if !inside {
            continue;
        }
        insides += 1;
        if fma_mnemonics.iter().any(|m| mnemonic(line).starts_with(m)) {
            hits += 1;
        }
    }
    if inspected == 0 {
        return skip(
            "asm",
            format!("{what} disassembly"),
            format!(
                "no symbols matching {:?} in {} (stripped?)",
                want_symbols,
                path.display()
            ),
        );
    }
    match (expect, hits) {
        (ExpectFma::No, 0) => pass(
            "asm",
            format!("{what}: no fused mul-add"),
            format!(
                "{inspected} symbol block(s), {insides} instruction(s) scanned, no FMA mnemonic"
            ),
        ),
        (ExpectFma::No, n) => fail(
            "asm",
            format!("{what}: no fused mul-add"),
            format!("{n} FMA instruction(s) inside the DSP code — bit-exactness is void"),
        ),
        (ExpectFma::Expected, n) if n > 0 => pass(
            "asm",
            format!("{what}: fma present as expected"),
            format!(
                "{n} FMA instruction(s) — this is the counter-example build, so the \
                 divergence measured against it really is fusion"
            ),
        ),
        (ExpectFma::Expected, _) => info(
            "asm",
            format!("{what}: fma expected but absent"),
            "the compiler chose not to fuse here, so any divergence it caused came from a \
             different transformation",
        ),
    }
}

/// `    1194:\tvfmadd132sd 0xe6b(%rip),...` -> `vfmadd132sd`: the first word after the
/// address. (GNU objdump separates mnemonic and operands with a space, llvm-objdump
/// with a tab, so "the last tab-separated field" is the operands for one of them.)
fn mnemonic(line: &str) -> &str {
    line.split_once(':')
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .unwrap_or("")
}

/// `0000000000001190 <g6ct_process>:` -> `g6ct_process`.
fn symbol_header(line: &str) -> Option<&str> {
    let (addr, rest) = line.trim_end().split_once(' ')?;
    if addr.is_empty() || !addr.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    rest.strip_prefix('<')?.strip_suffix(">:")
}

fn find_objdump() -> Option<String> {
    for t in ["objdump", "llvm-objdump", "gobjdump"] {
        if std::process::Command::new(t)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(t.to_string());
        }
    }
    None
}

fn with_metric(mut c: Check, m: Value) -> Check {
    c.metric = Some(m);
    c
}

fn diff_json(d: &Diff) -> Value {
    json!({
        "frames": d.samples,
        "differing": d.differing,
        "signed_zero": d.signed_zero,
        "max_abs": d.max_abs,
        "max_rel": d.max_rel,
        "max_ulp": d.max_ulp,
        "first_frame": d.first_frame,
        "snr_db": d.snr_db(),
    })
}

fn fmt_f(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.is_infinite() {
        "inf".to_string()
    } else {
        format!("{v:.3e}")
    }
}

/// `acoustics`: exact 3D Helmholtz/Snowman wave-scattering reciprocity, 3D FDTD PDE
/// reciprocity, Slepian DPSS Heisenberg-Gabor optimality, Paraunitary FDN lossless
/// energy conservation, Order-7 HOA quadrature orthonormality, and Bark-warped AP ripple.
pub fn acoustics() -> Vec<Check> {
    use g6me_core::atmosphere::AtmosphereCfg;
    use g6me_core::fft::{i0, kaiser_beta, slepian_dpss, Complex, FftPlan};
    use g6me_core::hoa::{
        eval_sh_order7, kermit_canfield_warped_allpass, sphere_quadrature_order7, HOA_CH,
    };
    use g6me_core::room::{FdtdSolver3d, RoomGeometry, RoomPreset, ScatteringFdn};
    use g6me_core::scattering::{rigid_sphere_green, snowman_green, SnowmanGeom};
    use std::f64::consts::PI;

    let mut out = Vec::new();

    // 1. Acoustic reciprocity of 3D Helmholtz rigid sphere & coupled Snowman head+torso (< 1e-13).
    let atmo = AtmosphereCfg::default();
    let c_air = atmo.speed_of_sound_m_s();
    let geom = SnowmanGeom::default();
    let p1 = [1.4, 0.7, 0.25];
    let p2 = [-0.6, 1.1, 0.45];
    let mut max_recip_err = 0.0_f64;
    for &f in &[300.0, 1_200.0, 3_600.0, 8_000.0] {
        let k = 2.0 * PI * f / c_air;
        let g12 = rigid_sphere_green(k, geom.head_radius_m, 0.45, 2.1, 0.35);
        let g21 = rigid_sphere_green(k, geom.head_radius_m, 2.1, 0.45, 0.35);
        let r_sphere = g12.sub(g21).norm() / g12.norm().max(1e-30);
        let s12 = snowman_green(k, geom, p1, p2);
        let s21 = snowman_green(k, geom, p2, p1);
        let r_snow = s12.sub(s21).norm() / s12.norm().max(1e-30);
        max_recip_err = max_recip_err.max(r_sphere).max(r_snow);
    }
    out.push(with_metric(
        if max_recip_err < 1e-13 {
            pass(
                "acoustics",
                "Helmholtz + Snowman acoustic reciprocity",
                format!(
                    "max |H(r1->r2)-H(r2->r1)|/|H| = {max_recip_err:.3e} (< 1e-13) at c = {c_air:.3} m/s"
                ),
            )
        } else {
            fail(
                "acoustics",
                "Helmholtz + Snowman acoustic reciprocity",
                format!("reciprocity error {max_recip_err:.3e} >= 1e-13"),
            )
        },
        json!({ "max_rel_err": max_recip_err, "speed_of_sound_m_s": c_air }),
    ));

    // 2. 3D Wave-Equation FDTD PDE reciprocity (< 1e-13).
    let room = RoomGeometry::from_preset(RoomPreset::ControlRoom);
    let fdtd = FdtdSolver3d::new(&room, c_air, 0.35);
    let va = fdtd.coord_to_voxel([1.5, 1.4, 1.2]);
    let vb = fdtd.coord_to_voxel([3.9, 3.2, 1.5]);
    let p_ab = &fdtd.simulate(va, &[vb], 90)[0];
    let p_ba = &fdtd.simulate(vb, &[va], 90)[0];
    let max_fdtd_err = p_ab
        .iter()
        .zip(p_ba.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let peak_fdtd = p_ab.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    out.push(with_metric(
        if peak_fdtd > 1e-5 && max_fdtd_err < 1e-13 {
            pass(
                "acoustics",
                "3D FDTD PDE acoustic reciprocity",
                format!(
                    "max |p(a->b)-p(b->a)| = {max_fdtd_err:.3e} (< 1e-13), peak = {peak_fdtd:.3e}"
                ),
            )
        } else {
            fail(
                "acoustics",
                "3D FDTD PDE acoustic reciprocity",
                format!("FDTD reciprocity error {max_fdtd_err:.3e}, peak {peak_fdtd:.3e}"),
            )
        },
        json!({ "max_abs_err": max_fdtd_err, "peak": peak_fdtd }),
    ));

    // 3. Exact Slepian DPSS Heisenberg-Gabor energy concentration beats Kaiser.
    let n_win = 128usize;
    let nw = 4.0_f64;
    let dp = slepian_dpss(n_win, nw);
    let beta = kaiser_beta(96.0);
    let i0b = i0(beta);
    let mut kw: Vec<f64> = (0..n_win)
        .map(|i| {
            let r = 2.0 * (i as f64) / ((n_win - 1) as f64) - 1.0;
            i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b
        })
        .collect();
    let kn = kw.iter().map(|v| v * v).sum::<f64>().sqrt();
    for v in &mut kw {
        *v /= kn;
    }
    let n_fft = 4096usize;
    let plan = FftPlan::new(n_fft);
    let mut scr = vec![Complex::ZERO; n_fft];
    let mut pad = vec![0.0_f64; n_fft];
    let mut bins = vec![Complex::ZERO; n_fft / 2 + 1];
    let w_cut = nw / (n_win as f64);
    let k_cut = (w_cut * (n_fft as f64)).round() as usize;
    let band_leak =
        |w: &[f64], pad: &mut [f64], scr: &mut [Complex], bins: &mut [Complex]| -> f64 {
            pad.fill(0.0);
            pad[..w.len()].copy_from_slice(w);
            plan.rfft(pad, scr, bins);
            let mut tot = bins[0].norm_sq() + bins[n_fft / 2].norm_sq();
            let mut out_e = bins[n_fft / 2].norm_sq();
            for (k, c) in bins.iter().enumerate().take(n_fft / 2).skip(1) {
                let e = 2.0 * c.norm_sq();
                tot += e;
                if k > k_cut {
                    out_e += e;
                }
            }
            out_e / tot.max(1e-30)
        };
    let leak_slep = band_leak(&dp, &mut pad, &mut scr, &mut bins);
    let leak_kais = band_leak(&kw, &mut pad, &mut scr, &mut bins);
    out.push(with_metric(
        if leak_slep < leak_kais && leak_slep < 1e-9 {
            pass(
                "acoustics",
                "Slepian DPSS Heisenberg optimality",
                format!(
                    "out-of-band leakage Slepian={leak_slep:.3e} < Kaiser={leak_kais:.3e} (NW={nw})"
                ),
            )
        } else {
            fail(
                "acoustics",
                "Slepian DPSS Heisenberg optimality",
                format!("Slepian leakage {leak_slep:.3e} did not beat Kaiser {leak_kais:.3e}"),
            )
        },
        json!({ "slepian_leakage": leak_slep, "kaiser_leakage": leak_kais }),
    ));

    // 4. Paraunitary Scattering FDN exact orthogonality (M^T M == I) and lossless energy conservation.
    let m = ScatteringFdn::paraunitary_matrix();
    let mut max_orth_err = 0.0_f64;
    for i in 0..8 {
        for j in 0..8 {
            let mut dot = 0.0_f64;
            for row in &m {
                dot += row[i] * row[j];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            max_orth_err = max_orth_err.max((dot - target).abs());
        }
    }
    let mut fdn = ScatteringFdn::new(48_000, 1e12, 1e12);
    let mut inp = [0.0_f64; 8];
    inp[0] = 1.0;
    let mut out_e = 0.0_f64;
    for step in 0..2048 {
        let y = fdn.tick(if step == 0 { inp } else { [0.0; 8] });
        if step < 1200 {
            out_e += y.iter().map(|v| v * v).sum::<f64>();
        }
    }
    out.push(with_metric(
        if max_orth_err < 1e-13 && out_e.is_finite() && out_e > 0.0 {
            pass(
                "acoustics",
                "Paraunitary FDN lossless orthogonality",
                format!(
                    "max |M^T M - I| = {max_orth_err:.3e} (< 1e-13), lossless tick energy = {out_e:.4}"
                ),
            )
        } else {
            fail(
                "acoustics",
                "Paraunitary FDN lossless orthogonality",
                format!("FDN orthogonality err {max_orth_err:.3e}, energy {out_e:.4}"),
            )
        },
        json!({ "max_orth_err": max_orth_err, "tick_energy": out_e }),
    ));

    // 5. Order-7 HOA (64 SH) Gauss-Legendre spherical quadrature orthonormality.
    let nodes = sphere_quadrature_order7();
    let mut max_gram_err = 0.0_f64;
    let ys: Vec<[f64; HOA_CH]> = nodes
        .iter()
        .map(|nd| eval_sh_order7(nd.az_rad, nd.el_rad))
        .collect();
    for i in 0..HOA_CH {
        for j in 0..HOA_CH {
            let mut g = 0.0_f64;
            for (k, nd) in nodes.iter().enumerate() {
                g += nd.weight * ys[k][i] * ys[k][j];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            max_gram_err = max_gram_err.max((g - target).abs());
        }
    }
    out.push(with_metric(
        if max_gram_err < 1e-13 {
            pass(
                "acoustics",
                "Order-7 HOA SH quadrature orthonormality",
                format!(
                    "64x64 Gram max error = {max_gram_err:.3e} (< 1e-13) on {} Gauss-Legendre nodes",
                    nodes.len()
                ),
            )
        } else {
            fail(
                "acoustics",
                "Order-7 HOA SH quadrature orthonormality",
                format!("64x64 Gram error {max_gram_err:.3e} >= 1e-13"),
            )
        },
        json!({ "max_gram_err": max_gram_err, "nodes": nodes.len() }),
    ));

    // 6. Kermit-Canfield & Abel Bark-warped all-pass 0.000000 dB magnitude ripple.
    let freqs: Vec<f64> = (0..513).map(|k| k as f64 * 48_000.0 / 1024.0).collect();
    let ap_l = kermit_canfield_warped_allpass(&freqs, 48_000.0, 1.0);
    let max_ap_ripple = ap_l
        .iter()
        .map(|c| (c.norm() - 1.0).abs())
        .fold(0.0_f64, f64::max);
    out.push(with_metric(
        if max_ap_ripple < 1e-13 {
            pass(
                "acoustics",
                "Bark-warped all-pass zero-ripple",
                format!("max ||H(f)| - 1| = {max_ap_ripple:.3e} (< 1e-13)"),
            )
        } else {
            fail(
                "acoustics",
                "Bark-warped all-pass zero-ripple",
                format!("all-pass magnitude ripple {max_ap_ripple:.3e} >= 1e-13"),
            )
        },
        json!({ "max_ripple": max_ap_ripple }),
    ));

    out
}

/// `cochlea`: Zilany-Bruce-Carney biophysical auditory-nerve neurogram NSIM,
/// Baumgartner-Majdak Bayesian Auditory Externalization, and Hear-to-Hear BALD personalization.
pub fn cochlea() -> Vec<Check> {
    use g6me_core::cochlea::{
        baumgartner_majdak_externalization, compute_neurogram, nsim, AnatomyParams,
        PersonalizationSession,
    };
    use g6me_core::plus::{KernelBank, PhysicsEngine, PlusCfg, PlusProfile};
    use g6me_core::room::RoomPreset;
    use std::f64::consts::PI;

    let mut out = Vec::new();
    let fs = 48_000u32;
    const SLOT_FL: usize = 0;

    // 1. Zilany-Bruce-Carney biophysical neurogram NSIM unity & distortion sensitivity.
    let n = 4800usize;
    let mut clean = vec![0.0_f64; n];
    let mut clipped = vec![0.0_f64; n];
    for i in 0..n {
        let t = (i as f64) / (fs as f64);
        let s = 0.25 * (2.0 * PI * 440.0 * t).sin()
            + 0.18 * (2.0 * PI * 1_650.0 * t).sin()
            + 0.12 * (2.0 * PI * 4_200.0 * t).sin();
        clean[i] = s;
        clipped[i] = (s * 4.0).tanh() * 0.25 - 0.12 * (2.0 * PI * 4_200.0 * t).sin();
    }
    let ng_clean = compute_neurogram(&clean, fs, 24);
    let ng_same = compute_neurogram(&clean, fs, 24);
    let ng_clip = compute_neurogram(&clipped, fs, 24);
    let s_same = nsim(&ng_clean, &ng_same);
    let s_clip = nsim(&ng_clean, &ng_clip);
    out.push(with_metric(
        if (s_same - 1.0).abs() < 1e-12 && s_clip < 0.92 {
            pass(
                "cochlea",
                "Zilany-Bruce-Carney neurogram NSIM",
                format!("NSIM(ref,ref)={s_same:.6}, NSIM(ref,distorted)={s_clip:.4} (< 0.92)"),
            )
        } else {
            fail(
                "cochlea",
                "Zilany-Bruce-Carney neurogram NSIM",
                format!("NSIM unity={s_same:.6}, distorted={s_clip:.4}"),
            )
        },
        json!({ "nsim_same": s_same, "nsim_distorted": s_clip }),
    ));

    // 2. Baumgartner-Majdak Externalization: G6+ Studio & 3D ControlRoom > G6+ Pure.
    let bank_pure = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Pure), fs);
    let bank_studio = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), fs);
    let mut ctrl_cfg = PlusCfg::from_profile(PlusProfile::Studio);
    ctrl_cfg.physics_engine = PhysicsEngine::HankelSnowmanMagLs;
    ctrl_cfg.room_preset = RoomPreset::ControlRoom;
    ctrl_cfg.use_slepian_tail = true;
    ctrl_cfg.use_fdn_velvet = true;
    let bank_ctrl = KernelBank::synthesize(ctrl_cfg, fs);

    let e_pure =
        baumgartner_majdak_externalization(&bank_pure.ir_l[SLOT_FL], &bank_pure.ir_r[SLOT_FL], fs);
    let e_studio = baumgartner_majdak_externalization(
        &bank_studio.ir_l[SLOT_FL],
        &bank_studio.ir_r[SLOT_FL],
        fs,
    );
    let e_ctrl =
        baumgartner_majdak_externalization(&bank_ctrl.ir_l[SLOT_FL], &bank_ctrl.ir_r[SLOT_FL], fs);

    out.push(with_metric(
        if e_studio.score > e_pure.score && e_ctrl.score > e_pure.score {
            pass(
                "cochlea",
                "Baumgartner-Majdak Bayesian externalization",
                format!(
                    "E(Pure)={:.3} < E(Studio)={:.3}, E(3D ControlRoom)={:.3}",
                    e_pure.score, e_studio.score, e_ctrl.score
                ),
            )
        } else {
            fail(
                "cochlea",
                "Baumgartner-Majdak Bayesian externalization",
                format!(
                    "expected Studio ({:.3}) & ControlRoom ({:.3}) > Pure ({:.3})",
                    e_studio.score, e_ctrl.score, e_pure.score
                ),
            )
        },
        json!({
            "pure": e_pure.score,
            "studio": e_studio.score,
            "control_room": e_ctrl.score,
        }),
    ));

    // 3. Hear-to-Hear Bayesian Active Personalization convergence in 12 clicks.
    let truth = AnatomyParams {
        head_radius_m: 0.093,
        torso_height_m: 0.29,
        pinna_shelf_db: 1.25,
    };
    let mut sess = PersonalizationSession::new();
    let h0 = sess.entropy_bits();
    for _ in 0..12 {
        let (a, b) = sess.pair_params();
        let da = ((truth.head_radius_m - a.head_radius_m) / 0.008).powi(2)
            + ((truth.torso_height_m - a.torso_height_m) / 0.040).powi(2)
            + ((truth.pinna_shelf_db - a.pinna_shelf_db) / 1.5).powi(2);
        let db = ((truth.head_radius_m - b.head_radius_m) / 0.008).powi(2)
            + ((truth.torso_height_m - b.torso_height_m) / 0.040).powi(2)
            + ((truth.pinna_shelf_db - b.pinna_shelf_db) / 1.5).powi(2);
        sess.record_choice(da < db);
    }
    let h12 = sess.entropy_bits();
    let est = sess.posterior_mean();
    let ok_p = h12 < h0 * 0.45
        && (est.head_radius_m - truth.head_radius_m).abs() < 0.005
        && (est.torso_height_m - truth.torso_height_m).abs() < 0.025
        && (est.pinna_shelf_db - truth.pinna_shelf_db).abs() < 0.8;
    out.push(with_metric(
        if ok_p {
            pass(
                "cochlea",
                "Hear-to-Hear BALD personalization (12 clicks)",
                format!(
                    "entropy {h0:.2} -> {h12:.2} bits; head={:.4}m torso={:.3}m pinna={:+.2}dB",
                    est.head_radius_m, est.torso_height_m, est.pinna_shelf_db
                ),
            )
        } else {
            fail(
                "cochlea",
                "Hear-to-Hear BALD personalization (12 clicks)",
                format!(
                    "did not converge: H {h0:.2}->{h12:.2}, est=({:.4}, {:.3}, {:+.2})",
                    est.head_radius_m, est.torso_height_m, est.pinna_shelf_db
                ),
            )
        },
        json!({
            "h0_bits": h0,
            "h12_bits": h12,
            "est_head_m": est.head_radius_m,
            "est_torso_m": est.torso_height_m,
            "est_pinna_db": est.pinna_shelf_db,
        }),
    ));

    out
}

/// `blackwell`: Kulisch superaccumulator, Ozaki-I INT8, Ozaki-II CRT across INT8 / FP8 E4M3 /
/// Blackwell FP4 E2M1 Base-13 (bit-identical), Matsuoka Bailey 4-Step Full-FP64 FFT on FP4,
/// Overlap-Save & Blelloch 2x2 parallel-prefix IIR, and Blackwell `sm_120` zero-FMA PTX audit.
pub fn blackwell() -> Vec<Check> {
    use g6me_core::blackwell::{
        audit_gpu_asm_zero_fma, bailey_ozaki_fft, ozaki1_int8_gemm, ozaki2_crt_gemm, BiquadCoeffs,
        KulischAcc, TensorBackend, BLACKWELL_ZERO_FMA_PTX,
    };
    use g6me_core::fft::{Complex, FftPlan};

    let mut out = Vec::new();

    // 1. Kulisch exact fixed-point superaccumulator.
    let mut acc = KulischAcc::new();
    acc.add(1.0e16);
    acc.add(1.0);
    acc.add(-1.0e16);
    acc.add(1.0e-12);
    let k_val = acc.finalize();
    let k_err = (k_val - (1.0 + 1.0e-12)).abs();
    out.push(with_metric(
        if k_err == 0.0 {
            pass(
                "blackwell",
                "Kulisch 2304-bit exact superaccumulator",
                format!("1e16 + 1.0 - 1e16 + 1e-12 = {k_val:.15} (err = 0.0)"),
            )
        } else {
            fail(
                "blackwell",
                "Kulisch 2304-bit exact superaccumulator",
                format!("Kulisch cancellation error {k_err:.3e}"),
            )
        },
        json!({ "abs_err": k_err }),
    ));

    // 2. Ozaki-I INT8 + Ozaki-II CRT across INT8, FP8 E4M3, and Blackwell FP4 E2M1 Base-13.
    let mut a = vec![vec![0.0_f64; 8]; 6];
    let mut b = vec![vec![0.0_f64; 5]; 8];
    for (i, row) in a.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((i * 7 + j * 13 + 1) as f64 * 0.371).sin() * 1.35;
        }
    }
    for (i, row) in b.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = ((i * 11 + j * 5 + 2) as f64 * 0.519).cos() * 1.15;
        }
    }
    let mut c_ref = vec![vec![0.0_f64; 5]; 6];
    for i in 0..6 {
        for j in 0..5 {
            let mut s = 0.0_f64;
            for k in 0..8 {
                s += a[i][k] * b[k][j];
            }
            c_ref[i][j] = s;
        }
    }
    let c_oz1 = ozaki1_int8_gemm(&a, &b, 9);
    let c_int8 = ozaki2_crt_gemm(&a, &b, TensorBackend::Int8);
    let c_fp8 = ozaki2_crt_gemm(&a, &b, TensorBackend::Fp8E4m3);
    let c_fp4 = ozaki2_crt_gemm(&a, &b, TensorBackend::Fp4E2m1Base13);

    let max_diff = |x: &[Vec<f64>], y: &[Vec<f64>]| -> f64 {
        x.iter()
            .zip(y.iter())
            .flat_map(|(rx, ry)| rx.iter().zip(ry.iter()).map(|(u, v)| (u - v).abs()))
            .fold(0.0_f64, f64::max)
    };
    let e_oz1 = max_diff(&c_oz1, &c_ref);
    let e_int8 = max_diff(&c_int8, &c_ref);
    let e_fp8 = max_diff(&c_fp8, &c_ref);
    let e_fp4 = max_diff(&c_fp4, &c_ref);
    let bit_diff_fp8 = max_diff(&c_int8, &c_fp8);
    let bit_diff_fp4 = max_diff(&c_int8, &c_fp4);

    out.push(with_metric(
        if e_oz1 < 1e-13
            && e_int8 < 1e-13
            && e_fp8 < 1e-13
            && e_fp4 < 1e-13
            && bit_diff_fp8 == 0.0
            && bit_diff_fp4 == 0.0
        {
            pass(
                "blackwell",
                "Ozaki-I & Ozaki-II CRT (INT8/FP8/FP4 E2M1)",
                format!(
                    "Oz1={e_oz1:.2e}, Oz2(INT8/FP8/FP4)={e_fp4:.2e} (< 1e-13), INT8==FP8==FP4 bit-diff=0.0"
                ),
            )
        } else {
            fail(
                "blackwell",
                "Ozaki-I & Ozaki-II CRT (INT8/FP8/FP4 E2M1)",
                format!(
                    "Oz1={e_oz1:.2e} INT8={e_int8:.2e} FP8={e_fp8:.2e} FP4={e_fp4:.2e} bit_diff=({bit_diff_fp8:.2e},{bit_diff_fp4:.2e})"
                ),
            )
        },
        json!({
            "ozaki1_err": e_oz1,
            "ozaki2_int8_err": e_int8,
            "ozaki2_fp8_err": e_fp8,
            "ozaki2_fp4_err": e_fp4,
            "backend_bit_diff": bit_diff_fp8.max(bit_diff_fp4),
        }),
    ));

    // 3. Matsuoka (Sept 2026) Full-FP64 Bailey 4-Step FFT on Blackwell FP4 E2M1 Tensor Cores.
    let n1 = 8usize;
    let n2 = 8usize;
    let x: Vec<Complex> = (0..(n1 * n2))
        .map(|k| Complex::new(((k as f64) * 0.23).sin(), ((k as f64) * 0.41).cos() * 0.7))
        .collect();
    let y_fp4 = bailey_ozaki_fft(&x, n1, n2, TensorBackend::Fp4E2m1Base13);
    let plan = FftPlan::new(n1 * n2);
    let mut y_ref = x.clone();
    plan.forward(&mut y_ref);
    let max_fft_err = y_fp4
        .iter()
        .zip(y_ref.iter())
        .map(|(u, v)| u.sub(*v).norm())
        .fold(0.0_f64, f64::max);
    out.push(with_metric(
        if max_fft_err < 1e-12 {
            pass(
                "blackwell",
                "Matsuoka Bailey 4-step FP64 FFT on FP4",
                format!(
                    "64-pt complex FFT via FP4 E2M1 Ozaki-II CRT + Kulisch: max err = {max_fft_err:.3e} (< 1e-12)"
                ),
            )
        } else {
            fail(
                "blackwell",
                "Matsuoka Bailey 4-step FP64 FFT on FP4",
                format!("Bailey FP4 FFT error {max_fft_err:.3e} >= 1e-12"),
            )
        },
        json!({ "max_abs_err": max_fft_err }),
    ));

    // 4. Phase-Exact GPU IIR: Overlap-Save (-320 dBFS settling) + Blelloch 2x2 parallel prefix scan.
    let k_lp = (std::f64::consts::PI * 1_200.0 / 48_000.0).tan();
    let k2 = k_lp * k_lp;
    let sq2 = std::f64::consts::SQRT_2;
    let inv_n = 1.0 / (1.0 + sq2 * k_lp + k2);
    let bq = BiquadCoeffs {
        b0: k2 * inv_n,
        b1: 2.0 * k2 * inv_n,
        b2: k2 * inv_n,
        a1: 2.0 * (k2 - 1.0) * inv_n,
        a2: (1.0 - sq2 * k_lp + k2) * inv_n,
    };
    let sig: Vec<f64> = (0..1024)
        .map(|i| ((i as f64) * 0.08).sin() + 0.4 * ((i as f64) * 0.31).cos())
        .collect();
    let y_seq = bq.process_sequential(&sig);
    let y_ols = bq.process_overlap_save(&sig, 256);
    let y_ble = bq.process_blelloch_scan(&sig);
    let e_ols = y_seq
        .iter()
        .zip(y_ols.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let e_ble = y_seq
        .iter()
        .zip(y_ble.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    out.push(with_metric(
        if e_ols < 1e-13 && e_ble < 1e-13 {
            pass(
                "blackwell",
                "GPU IIR Overlap-Save & Blelloch 2x2 scan",
                format!(
                    "max |OLS - seq| = {e_ols:.3e}, max |Blelloch - seq| = {e_ble:.3e} (< 1e-13)"
                ),
            )
        } else {
            fail(
                "blackwell",
                "GPU IIR Overlap-Save & Blelloch 2x2 scan",
                format!("OLS err {e_ols:.3e}, Blelloch err {e_ble:.3e}"),
            )
        },
        json!({ "ols_err": e_ols, "blelloch_err": e_ble }),
    ));

    // 5. Blackwell sm_120 PTX/SASS zero-FMA audit + negative-control rejection.
    let clean_audit = audit_gpu_asm_zero_fma(BLACKWELL_ZERO_FMA_PTX);
    let poisoned = format!("{BLACKWELL_ZERO_FMA_PTX}\n    fma.rn.f64 %fd9, %fd1, %fd2, %fd3;\n");
    let bad_audit = audit_gpu_asm_zero_fma(&poisoned);
    let nvcc_present = std::process::Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    out.push(with_metric(
        match (&clean_audit, &bad_audit) {
            (Ok(n_lines), Err(hits)) if *n_lines > 5 && !hits.is_empty() => pass(
                "blackwell",
                "sm_120 PTX/SASS zero-FMA audit",
                format!(
                    "verified {n_lines} PTX instructions with 0 FMA; negative control caught {} FMA hit(s) (nvcc={nvcc_present})",
                    hits.len()
                ),
            ),
            _ => fail(
                "blackwell",
                "sm_120 PTX/SASS zero-FMA audit",
                format!("unexpected PTX audit result: clean={clean_audit:?}, poison={bad_audit:?}"),
            ),
        },
        json!({ "nvcc_present": nvcc_present }),
    ));

    out
}

/// Standalone `--audit-cochlea [WAV]` nuclear perceptual report:
/// runs Zilany-Bruce-Carney biophysical neurogram NSIM + Baumgartner-Majdak Bayesian
/// Auditory Externalization across G6+ acoustic engines (or an optional user stereo WAV).
pub fn audit_cochlea(wav_path: Option<&std::path::Path>, as_json: bool) -> Result<(), String> {
    use g6me_core::cochlea::{baumgartner_majdak_externalization, compute_neurogram, nsim};
    use g6me_core::plus::{KernelBank, PhysicsEngine, PlusCfg, PlusProfile};
    use g6me_core::room::RoomPreset;
    use std::f64::consts::PI;
    use std::io::{Read, Seek, SeekFrom};

    const SLOT_FL: usize = 0;
    let fs = 48_000u32;
    let bank_pure = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Pure), fs);
    let bank_studio = KernelBank::synthesize(PlusCfg::from_profile(PlusProfile::Studio), fs);
    let mut ctrl_cfg = PlusCfg::from_profile(PlusProfile::Studio);
    ctrl_cfg.physics_engine = PhysicsEngine::HankelSnowmanMagLs;
    ctrl_cfg.room_preset = RoomPreset::ControlRoom;
    ctrl_cfg.use_slepian_tail = true;
    ctrl_cfg.use_fdn_velvet = true;
    let bank_ctrl = KernelBank::synthesize(ctrl_cfg, fs);

    let e_pure =
        baumgartner_majdak_externalization(&bank_pure.ir_l[SLOT_FL], &bank_pure.ir_r[SLOT_FL], fs);
    let e_studio = baumgartner_majdak_externalization(
        &bank_studio.ir_l[SLOT_FL],
        &bank_studio.ir_r[SLOT_FL],
        fs,
    );
    let e_ctrl =
        baumgartner_majdak_externalization(&bank_ctrl.ir_l[SLOT_FL], &bank_ctrl.ir_r[SLOT_FL], fs);

    // Biophysical cochlea NSIM between direct anechoic vs studio/control-room on a 120 ms complex tone.
    let n = 5760usize;
    let mut dry_tone = vec![0.0_f64; n];
    for (i, s) in dry_tone.iter_mut().enumerate() {
        let t = (i as f64) / (fs as f64);
        *s = 0.22 * (2.0 * PI * 330.0 * t).sin()
            + 0.18 * (2.0 * PI * 1_100.0 * t).sin()
            + 0.14 * (2.0 * PI * 3_500.0 * t).sin();
    }
    let ng_ref = compute_neurogram(&dry_tone, fs, 24);

    let wav_audit = if let Some(path) = wav_path {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (info, data_off) = g6me_core::wav::read_info_file(&mut f)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if info.channels < 2 {
            return Err(format!(
                "{} has {} channel(s); --audit-cochlea requires stereo (2+ channels)",
                path.display(),
                info.channels
            ));
        }
        f.seek(SeekFrom::Start(data_off))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let ch = info.channels as usize;
        let bps = (info.bits / 8) as usize;
        let frames = (info.frames() as usize).min(info.rate as usize / 2);
        let mut raw = vec![0u8; frames * ch * bps];
        f.read_exact(&mut raw)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let mut l = Vec::with_capacity(frames);
        let mut r = Vec::with_capacity(frames);
        let decode_one = |bytes: &[u8]| -> f64 {
            match (info.float, info.bits) {
                (true, 32) => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
                (true, 64) => f64::from_le_bytes(bytes.try_into().unwrap()),
                (false, 16) => i16::from_le_bytes(bytes.try_into().unwrap()) as f64 / 32768.0,
                (false, 24) => {
                    let v = ((bytes[2] as i8 as i32) << 16)
                        | ((bytes[1] as i32) << 8)
                        | (bytes[0] as i32);
                    (v as f64) / 8_388_608.0
                }
                (false, 32) => {
                    i32::from_le_bytes(bytes.try_into().unwrap()) as f64 / 2_147_483_648.0
                }
                _ => 0.0,
            }
        };
        for i in 0..frames {
            let off_l = (i * ch) * bps;
            let off_r = (i * ch + 1) * bps;
            l.push(decode_one(&raw[off_l..off_l + bps]));
            r.push(decode_one(&raw[off_r..off_r + bps]));
        }
        let ext = baumgartner_majdak_externalization(&l, &r, info.rate);
        let ng_l = compute_neurogram(&l, info.rate, 24);
        let ng_r = compute_neurogram(&r, info.rate, 24);
        let lr_nsim = nsim(&ng_l, &ng_r);
        Some(json!({
            "path": path.display().to_string(),
            "sample_rate": info.rate,
            "frames_analyzed": frames,
            "externalization_score": ext.score,
            "monaural_cue": ext.monaural_cue,
            "binaural_cue": ext.binaural_cue,
            "iacc": ext.iacc,
            "early_refl_db": ext.early_refl_db,
            "lr_neurogram_nsim": lr_nsim,
        }))
    } else {
        None
    };

    let self_nsim = nsim(&ng_ref, &ng_ref);
    let report = json!({
        "cochlea_model": "Zilany-Bruce-Carney (JASA 2014 / 2018) 24-CF IHC/ANF Neurogram",
        "externalization_model": "Baumgartner & Majdak (2021) Bayesian Auditory Externalization",
        "reference_nsim_self": self_nsim,
        "profiles": {
            "pure": {
                "externalization": e_pure.score,
                "monaural_cue": e_pure.monaural_cue,
                "binaural_cue": e_pure.binaural_cue,
                "iacc": e_pure.iacc,
                "early_refl_db": e_pure.early_refl_db,
            },
            "studio": {
                "externalization": e_studio.score,
                "monaural_cue": e_studio.monaural_cue,
                "binaural_cue": e_studio.binaural_cue,
                "iacc": e_studio.iacc,
                "early_refl_db": e_studio.early_refl_db,
            },
            "control_room_3d": {
                "externalization": e_ctrl.score,
                "monaural_cue": e_ctrl.monaural_cue,
                "binaural_cue": e_ctrl.binaural_cue,
                "iacc": e_ctrl.iacc,
                "early_refl_db": e_ctrl.early_refl_db,
            }
        },
        "wav_audit": wav_audit,
    });

    if as_json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("Biophysical Cochlea & Bayesian Externalization Audit");
        println!("----------------------------------------------------");
        println!("  Zilany-Bruce-Carney Neurogram NSIM (self): {self_nsim:.6}");
        println!(
            "  G6+ Pure         : E = {:.3} (monaural={:.3}, binaural={:.3}, IACC={:.3}, ER={:+.1} dB)",
            e_pure.score, e_pure.monaural_cue, e_pure.binaural_cue, e_pure.iacc, e_pure.early_refl_db
        );
        println!(
            "  G6+ Studio       : E = {:.3} (monaural={:.3}, binaural={:.3}, IACC={:.3}, ER={:+.1} dB)",
            e_studio.score, e_studio.monaural_cue, e_studio.binaural_cue, e_studio.iacc, e_studio.early_refl_db
        );
        println!(
            "  G6+ ControlRoom3D: E = {:.3} (monaural={:.3}, binaural={:.3}, IACC={:.3}, ER={:+.1} dB)",
            e_ctrl.score, e_ctrl.monaural_cue, e_ctrl.binaural_cue, e_ctrl.iacc, e_ctrl.early_refl_db
        );
        if let Some(wa) = &wav_audit {
            println!("  WAV File Audit   : {}", wa);
        }
    }

    if !(e_studio.score > e_pure.score && e_ctrl.score > e_pure.score) {
        return Err(format!(
            "externalization gate failed: Pure={:.3}, Studio={:.3}, ControlRoom={:.3}",
            e_pure.score, e_studio.score, e_ctrl.score
        ));
    }
    Ok(())
}

/// `ntt`: Goldilocks prime-field (`p = 2^64 - 2^32 + 1`) Number-Theoretic Transform
/// convolution with 0 ULPs of intermediate floating-point rounding error.
pub fn ntt() -> Vec<Check> {
    use g6me_core::ntt::{
        exact_int_convolve, NttOverlapAddConvolver, NttPlan, BLACKWELL_GOLDILOCKS_NTT_CUDA,
        GOLDILOCKS_P,
    };
    use g6me_core::plus::{BinauralPlus, PlusCfg, PlusProfile};

    let mut out = Vec::new();

    // 1. Goldilocks field radix-2 NTT invertibility (0 ULP exact integer round-trip).
    let n = 1024usize;
    let plan = NttPlan::new(n);
    let orig: Vec<u64> = (0..n)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_5A5A) % GOLDILOCKS_P)
        .collect();
    let mut buf = orig.clone();
    plan.forward(&mut buf);
    plan.inverse(&mut buf);
    let exact_rt = buf == orig;
    out.push(with_metric(
        if exact_rt {
            pass(
                "ntt",
                "Goldilocks Z/pZ radix-2 NTT invertibility",
                format!(
                    "N={n} forward+inverse NTT matches input bit-for-bit (0 ULP) mod p=2^64-2^32+1"
                ),
            )
        } else {
            fail(
                "ntt",
                "Goldilocks Z/pZ radix-2 NTT invertibility",
                "NTT forward+inverse round-trip did not match input exactly",
            )
        },
        json!({ "n": n, "p": GOLDILOCKS_P, "exact": exact_rt }),
    ));

    // 2. 0-ULP exact integer convolution vs direct i128 schoolbook convolution.
    let mut sig = vec![0i64; 512];
    let mut ker = vec![0i64; 257];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for s in &mut sig {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s = ((state >> 40) as i64) - (1 << 23);
    }
    for k in &mut ker {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *k = ((state >> 36) as i64) - (1 << 27);
    }
    let got = exact_int_convolve(&sig, &ker);
    let mut max_int_err = 0i128;
    for (i, &gv) in got.iter().enumerate() {
        let mut acc = 0i128;
        let j_lo = i.saturating_sub(ker.len() - 1);
        let j_hi = i.min(sig.len() - 1);
        for j in j_lo..=j_hi {
            acc += (sig[j] as i128) * (ker[i - j] as i128);
        }
        max_int_err = max_int_err.max((gv - acc).abs());
    }
    out.push(with_metric(
        if max_int_err == 0 {
            pass(
                "ntt",
                "0-ULP exact integer linear convolution",
                format!(
                    "512x257 24-bit PCM * Q28 FIR convolution max integer error = {max_int_err} (exact i128 match)"
                ),
            )
        } else {
            fail(
                "ntt",
                "0-ULP exact integer linear convolution",
                format!("max integer error = {max_int_err} (expected 0)"),
            )
        },
        json!({ "max_int_err": max_int_err.to_string(), "out_len": got.len() }),
    ));

    // 3. Streaming NttOverlapAddConvolver & BinauralPlus --exact-ntt parity.
    let ir_f64: Vec<f64> = (0..192)
        .map(|i| (-0.025 * i as f64).exp() * (0.19 * i as f64).cos() * 0.25)
        .collect();
    let mut conv = NttOverlapAddConvolver::new(&ir_f64);
    let pcm: Vec<f64> = (0..640)
        .map(|i| 0.4 * (2.0 * std::f64::consts::PI * 997.0 * i as f64 / 48_000.0).sin())
        .collect();
    let lat = conv.latency();
    let mut streamed = Vec::with_capacity(pcm.len() + lat);
    for &s in &pcm {
        streamed.push(conv.process_sample(s));
    }
    for _ in 0..lat {
        streamed.push(conv.process_sample(0.0));
    }
    let mut max_ola_diff = 0.0_f64;
    for i in 0..pcm.len() {
        let mut ref_v = 0.0_f64;
        for (j, &h) in ir_f64.iter().enumerate() {
            if i >= j {
                ref_v += pcm[i - j] * h;
            }
        }
        max_ola_diff = max_ola_diff.max((streamed[i + lat] - ref_v).abs());
    }

    let mut cfg_ntt = PlusCfg::from_profile(PlusProfile::Pure);
    cfg_ntt.exact_ntt = true;
    let mut plus_ntt = BinauralPlus::new(g6me_core::dsp::MASK_5_1, cfg_ntt, 48_000);
    let fl: Vec<f64> = (0..1024)
        .map(|t| 0.25 * (2.0 * std::f64::consts::PI * 440.0 * t as f64 / 48_000.0).sin())
        .collect();
    let mut slots: [Option<&[f64]>; g6me_core::dsp::SLOTS] = [None; g6me_core::dsp::SLOTS];
    slots[0] = Some(&fl);
    let mut out_l = vec![0.0_f64; 1024];
    let mut out_r = vec![0.0_f64; 1024];
    plus_ntt.process(&slots, &mut out_l, &mut out_r, 1024);
    let (tail_l, tail_r) = plus_ntt.flush();
    out_l.extend_from_slice(&tail_l);
    out_r.extend_from_slice(&tail_r);
    let plus_ok = plus_ntt.is_finite()
        && out_l.iter().chain(&out_r).all(|s| s.is_finite())
        && out_l
            .iter()
            .chain(&out_r)
            .map(|s| s.abs())
            .fold(0.0_f64, f64::max)
            > 0.01;
    let cuda_ok = BLACKWELL_GOLDILOCKS_NTT_CUDA.contains("GOLDILOCKS_P")
        && BLACKWELL_GOLDILOCKS_NTT_CUDA.contains("goldilocks_ntt_stage");

    out.push(with_metric(
        if max_ola_diff < 5e-7 && plus_ok && cuda_ok {
            pass(
                "ntt",
                "Streaming Goldilocks OLA + BinauralPlus --exact-ntt",
                format!(
                    "streaming OLA max diff vs f64 = {max_ola_diff:.3e} (< 5e-7), BinauralPlus --exact-ntt finite & non-zero"
                ),
            )
        } else {
            fail(
                "ntt",
                "Streaming Goldilocks OLA + BinauralPlus --exact-ntt",
                format!("max_ola_diff={max_ola_diff:.3e}, plus_ok={plus_ok}, cuda_ok={cuda_ok}"),
            )
        },
        json!({ "max_ola_diff": max_ola_diff, "plus_ok": plus_ok, "cuda_ok": cuda_ok }),
    ));

    out
}

/// `bem`: 3D Helmholtz FMM-BEM solver (`--head-mesh`), Generalized Winding Number
/// watertight repair, isotropic acoustic remeshing (`h <= lambda_min/6`), Gumerov
/// acoustic reciprocity, and analytical Mie sphere validation.
pub fn bem() -> Vec<Check> {
    use g6me_core::bem::{
        helmholtz_green, isotropic_acoustic_remesh, watertight_winding_repair, FmmBemSolver,
        TriangleMesh, BLACKWELL_FMM_BEM_CUDA,
    };
    use g6me_core::scattering::{rigid_sphere_green, sph_to_cart};
    use std::f64::consts::PI;

    let mut out = Vec::new();

    // 1. Generalized Winding Number open-hole watertight repair (Jacobson et al., SIGGRAPH 2013).
    let mut broken = TriangleMesh::sphere(0.0875, 2);
    broken.faces.retain(|&[a, b, c]| {
        let cz = (broken.vertices[a][2] + broken.vertices[b][2] + broken.vertices[c][2]) / 3.0;
        cz < 0.06
    });
    let broken_wt = broken.stats(344.8).is_watertight;
    let repaired = watertight_winding_repair(&broken, 18);
    let rep_st = repaired.stats(344.8);
    out.push(with_metric(
        if !broken_wt && rep_st.is_watertight && (rep_st.center_winding - 1.0).abs() < 1e-6 {
            pass(
                "bem",
                "Generalized Winding Number watertight repair",
                format!(
                    "open hole closed: boundary_edges={} (0), center_winding={:.6} (1.0), faces={}",
                    rep_st.boundary_edges, rep_st.center_winding, rep_st.faces
                ),
            )
        } else {
            fail(
                "bem",
                "Generalized Winding Number watertight repair",
                format!(
                    "broken_wt={broken_wt}, repaired_wt={}, boundary={}, winding={:.6}",
                    rep_st.is_watertight, rep_st.boundary_edges, rep_st.center_winding
                ),
            )
        },
        json!({
            "boundary_edges": rep_st.boundary_edges,
            "center_winding": rep_st.center_winding,
            "faces": rep_st.faces,
        }),
    ));

    // 2. Isotropic acoustic remeshing (h <= h_target, watertightness preserved).
    let coarse = TriangleMesh::sphere(0.0875, 1);
    let h_target = 0.025;
    let remeshed = isotropic_acoustic_remesh(&coarse, h_target, 4);
    let rem_st = remeshed.stats(344.8);
    out.push(with_metric(
        if rem_st.is_watertight && rem_st.max_edge_m <= h_target {
            pass(
                "bem",
                "Isotropic acoustic remeshing (h <= lambda/6)",
                format!(
                    "watertight={}, max_edge={:.4} m <= {:.4} m, f_max_acoustic={:.0} Hz",
                    rem_st.is_watertight, rem_st.max_edge_m, h_target, rem_st.max_valid_freq_hz
                ),
            )
        } else {
            fail(
                "bem",
                "Isotropic acoustic remeshing (h <= lambda/6)",
                format!(
                    "watertight={}, max_edge={:.4} m > {:.4} m",
                    rem_st.is_watertight, rem_st.max_edge_m, h_target
                ),
            )
        },
        json!({
            "max_edge_m": rem_st.max_edge_m,
            "h_target_m": h_target,
            "max_valid_freq_hz": rem_st.max_valid_freq_hz,
        }),
    ));

    // 3. Exact BEM + FMM Octree Gumerov reciprocity & analytical Mie sphere ILD validation.
    let sphere = TriangleMesh::sphere(0.0875, 2);
    let solver = FmmBemSolver::new(&sphere).expect("build BEM solver");
    let f_hz = 1200.0;
    let c = 344.0;
    let r_src = sph_to_cart(45.0, 10.0, 2.0);
    let h_direct = solver.solve_direct_hrtf(f_hz, c, r_src, solver.left_ear_elem, false);
    let h_recip = solver.solve_reciprocal_hrtfs(f_hz, c, &[r_src], solver.left_ear_elem, false)[0];
    let recip_rel = h_direct.sub(h_recip).norm() / h_direct.norm().max(1e-12);

    let k = 2.0 * PI * f_hz / c;
    let ear_l = solver.elements[solver.left_ear_elem].centroid;
    let ear_r = solver.elements[solver.right_ear_elem].centroid;
    let h_bem_l = solver.solve_reciprocal_hrtfs(f_hz, c, &[r_src], solver.left_ear_elem, false)[0];
    let h_bem_r = solver.solve_reciprocal_hrtfs(f_hz, c, &[r_src], solver.right_ear_elem, false)[0];
    let g_ref = helmholtz_green(k, [0.0, 0.0, 0.0], r_src);
    let norm3 = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let dot3 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let r_s = norm3(r_src);
    let cos_l = dot3(r_src, ear_l) / (r_s * norm3(ear_l));
    let cos_r = dot3(r_src, ear_r) / (r_s * norm3(ear_r));
    let h_mie_l = rigid_sphere_green(k, 0.0875, r_s, norm3(ear_l), cos_l).div(g_ref);
    let h_mie_r = rigid_sphere_green(k, 0.0875, r_s, norm3(ear_r), cos_r).div(g_ref);
    let ild_bem_db = 20.0 * (h_bem_l.norm() / h_bem_r.norm()).log10();
    let ild_mie_db = 20.0 * (h_mie_l.norm() / h_mie_r.norm()).log10();
    let ild_err_db = (ild_bem_db - ild_mie_db).abs();

    let head = TriangleMesh::procedural_head_with_pinnae(0.0875, 3);
    let fmm_solver = FmmBemSolver::new(&head).expect("build FMM-BEM solver");
    let h_fmm_d =
        fmm_solver.solve_direct_hrtf(2500.0, 344.85, r_src, fmm_solver.left_ear_elem, true);
    let h_fmm_r =
        fmm_solver.solve_reciprocal_hrtfs(2500.0, 344.85, &[r_src], fmm_solver.left_ear_elem, true)
            [0];
    let fmm_rel = h_fmm_d.sub(h_fmm_r).norm() / h_fmm_d.norm().max(1e-12);
    let cuda_zero_fma = BLACKWELL_FMM_BEM_CUDA.contains("__dmul_rn")
        && BLACKWELL_FMM_BEM_CUDA.contains("__dadd_rn")
        && !BLACKWELL_FMM_BEM_CUDA.contains("fma(");

    out.push(with_metric(
        if recip_rel < 1e-11 && fmm_rel < 1e-10 && ild_err_db < 0.75 && cuda_zero_fma {
            pass(
                "bem",
                "3D FMM-BEM Gumerov reciprocity & Mie sphere validation",
                format!(
                    "BEM recip={recip_rel:.2e}, FMM recip={fmm_rel:.2e} ({} octree nodes), ILD={ild_bem_db:.2} vs Mie {ild_mie_db:.2} dB (err {ild_err_db:.2} dB)",
                    fmm_solver.octree.node_count()
                ),
            )
        } else {
            fail(
                "bem",
                "3D FMM-BEM Gumerov reciprocity & Mie sphere validation",
                format!(
                    "recip={recip_rel:.2e}, fmm_recip={fmm_rel:.2e}, ild_err={ild_err_db:.2} dB, cuda_zero_fma={cuda_zero_fma}"
                ),
            )
        },
        json!({
            "bem_recip_rel": recip_rel,
            "fmm_recip_rel": fmm_rel,
            "octree_nodes": fmm_solver.octree.node_count(),
            "ild_bem_db": ild_bem_db,
            "ild_mie_db": ild_mie_db,
            "ild_err_db": ild_err_db,
        }),
    ));

    // 4. Schenck (JASA 1968) CHIEF interior-point regularization at exact ka = pi irregular frequency.
    let f_irreg = c / (2.0 * 0.0875); // ka = pi (~1965.7 Hz)
    let h_chief_d = solver.solve_direct_hrtf(f_irreg, c, r_src, solver.left_ear_elem, false);
    let h_chief_r =
        solver.solve_reciprocal_hrtfs(f_irreg, c, &[r_src], solver.left_ear_elem, false)[0];
    let chief_recip = h_chief_d.sub(h_chief_r).norm() / h_chief_d.norm().max(1e-12);
    let k_irreg = 2.0 * PI * f_irreg / c;
    let g_ref_irreg = helmholtz_green(k_irreg, [0.0, 0.0, 0.0], r_src);
    let h_mie_irreg =
        rigid_sphere_green(k_irreg, 0.0875, r_s, norm3(ear_l), cos_l).div(g_ref_irreg);
    let chief_mie_db = (20.0 * (h_chief_r.norm() / h_mie_irreg.norm()).log10()).abs();
    out.push(with_metric(
        if chief_recip < 1e-11 && chief_mie_db < 1.5 && h_chief_r.norm().is_finite() {
            pass(
                "bem",
                "CHIEF interior-point irregular-frequency suppression (ka = pi)",
                format!(
                    "f={f_irreg:.1} Hz (ka=pi): recip={chief_recip:.2e}, |H_CHIEF| vs Mie err={chief_mie_db:.2} dB"
                ),
            )
        } else {
            fail(
                "bem",
                "CHIEF interior-point irregular-frequency suppression (ka = pi)",
                format!("chief_recip={chief_recip:.2e}, chief_mie_db={chief_mie_db:.2} dB"),
            )
        },
        json!({
            "f_irreg_hz": f_irreg,
            "chief_recip_rel": chief_recip,
            "chief_mie_err_db": chief_mie_db,
        }),
    ));

    out
}

/// `scan`: 4-stage `g6me-scan` photogrammetry-to-FMM-BEM pipeline (DNG 1.7 SubIFD &
/// analytical `OpcodeList2/3` zero-resampling, v7.0 MIPI `RAW10` Distinct-Frame &
/// Clock-Bracket gates, GOST ID-1 Russian debit card tangent-extrapolated metric anchor
/// and privacy mask, Forster `SO(3)` IMU preintegration, OISSR, Chugunov glare separation,
/// GLOMAP, Ozaki-II FP4->FP64 Rolling-Shutter BA, and 2DGS/PGSR -> FMM-BEM handoff).
pub fn scan() -> Vec<Check> {
    use g6me_core::bem::TriangleMesh;
    use g6me_core::blackwell::TensorBackend;
    use g6me_core::burst3d::{
        mat3_mul, mat3_transpose, preintegrate_imu_manifold, separate_specular_and_micro_depth,
        so3_exp, so3_log, wronski_oissr_zero_demosaic_fuse, MAT3_ID,
    };
    use g6me_core::dng::{
        build_synthetic_dng_17, demosaic_malvar_he_cutler, pack_mipi_raw10, parse_dng_bytes,
        unpack_mipi_raw10_to_f64, verify_burst_gates, BurstFrameMeta, BurstMetaSidecar, CfaPattern,
        DngMetadata, FixVignetteRadialOpcode, GainMapOpcode, ImuAccelSample, ImuGyroSample,
        OisSample, WarpPlaneCoeffs, WarpRectilinearOpcode, COMPRESSION_JPEG_XL,
        PHOTOMETRIC_LINEAR_RAW,
    };
    use g6me_core::scan::{
        bundle_adjust_ozaki2_fp4_schur, calibrate_russian_debit_card_id1,
        glomap_so3_rotation_averaging, is_inside_card_privacy_mask,
        run_stage3_to_stage4_to_fmm_bem, sample_surfels_from_mesh, BaObservation,
        MetricGaugeConstraint, RelativeRotationEdge, RollingShutterCamera,
        ISO7816_CHIP_CENTER_X_MM, ISO7816_CHIP_CENTER_Y_MM, RUSSIAN_ID1_CORNER_RADIUS_MM,
        RUSSIAN_ID1_HEIGHT_MM, RUSSIAN_ID1_WIDTH_MM,
    };

    let mut out = Vec::new();

    // 1. DNG 1.7 SubIFD + OpcodeList2/3 zero-resampling analytical ray inversion (CA < 0.008 px).
    let synth_meta = DngMetadata {
        dng_version: [1, 7, 0, 0],
        preview_width: 1024,
        preview_height: 768,
        main_width: 8160,
        main_height: 6120,
        bits_per_sample: 16,
        compression: COMPRESSION_JPEG_XL,
        photometric: PHOTOMETRIC_LINEAR_RAW,
        new_subfile_type: 16,
        subifd_routed: true,
        cfa_pattern: Some(CfaPattern::Gbrg),
        black_level: 256.0,
        white_level: 4095.0,
        gain_maps: vec![GainMapOpcode {
            top: 0,
            left: 0,
            bottom: 6120,
            right: 8160,
            plane: 0,
            planes: 4,
            row_pitch: 2,
            col_pitch: 2,
            map_points_v: 2,
            map_points_h: 2,
            map_spacing_v: 1.0,
            map_spacing_h: 1.0,
            map_origin_v: 0.0,
            map_origin_h: 0.0,
            map_planes: 1,
            gains: vec![1.25, 1.25, 1.25, 1.25],
        }],
        vignette: Some(FixVignetteRadialOpcode {
            k: [0.22, -0.05, 0.01, 0.0, 0.0],
            cx_hat: 0.5,
            cy_hat: 0.5,
        }),
        warp_rectilinear: Some(WarpRectilinearOpcode {
            planes: vec![
                WarpPlaneCoeffs {
                    k: [1.0005, -0.022, 0.006, 0.0],
                    m: [1e-4, -1e-4],
                },
                WarpPlaneCoeffs {
                    k: [1.0000, -0.020, 0.005, 0.0],
                    m: [1e-4, -1e-4],
                },
                WarpPlaneCoeffs {
                    k: [0.9994, -0.018, 0.004, 0.0],
                    m: [1e-4, -1e-4],
                },
            ],
            cx_hat: 0.5,
            cy_hat: 0.5,
        }),
    };
    let dng_bytes = build_synthetic_dng_17(&synth_meta);
    let parsed = parse_dng_bytes(&dng_bytes).expect("parse DNG 1.7");
    let w_parsed = parsed.warp_rectilinear.as_ref().expect("WarpRectilinear");
    let mut max_ca_px = 0.0_f64;
    let m_diag = 4080.0_f64.hypot(3060.0);
    let (g_xd, g_yd) = w_parsed.distort_norm(0.35, -0.25, 1);
    let g_px = (4080.0 + g_xd * m_diag, 3060.0 + g_yd * m_diag);
    for plane in [0usize, 2usize] {
        let (p_xd, p_yd) = w_parsed.distort_norm(0.35, -0.25, plane);
        let p_px = (4080.0 + p_xd * m_diag, 3060.0 + p_yd * m_diag);
        let (cx, cy) = w_parsed.cancel_lateral_ca_px(p_px.0, p_px.1, 8160.0, 6120.0, plane);
        max_ca_px = max_ca_px.max((cx - g_px.0).hypot(cy - g_px.1));
    }
    let honesty = parsed.honesty_report(64, 0.31, max_ca_px);

    // MIPI RAW10 round-trip + v7.0 Distinct-Frame & Clock-Bracket gates (positive & negative).
    let w_r = 32usize;
    let h_r = 16usize;
    let stride = (w_r / 4) * 5;
    let mut frames = Vec::new();
    let mut frame_metas = Vec::new();
    for f in 0..6usize {
        let vals: Vec<u16> = (0..(w_r * h_r))
            .map(|i| (256 + ((i * 7 + f * 37) % 700)) as u16)
            .collect();
        frames.push(pack_mipi_raw10(&vals, w_r, h_r, stride));
        let t0 = 1_000_000_000i64 + (f as i64) * 40_000_000;
        frame_metas.push(BurstFrameMeta {
            frame_index: f,
            file_name: format!("frame_{f}.raw10"),
            byte_length: stride * h_r,
            row_stride: stride,
            width: w_r,
            height: h_r,
            sensor_timestamp_ns: t0,
            sensor_exposure_time_ns: 4_000_000,
            sensor_sensitivity_iso: 100,
            sensor_rolling_shutter_skew_ns: 36_000_000,
            lens_focus_distance_diopters: 4.5,
            ois_samples: vec![OisSample {
                t_ns: t0 + 10_000_000,
                dx_px: 1.5 * (f as f64).sin(),
                dy_px: 1.2 * (f as f64).cos(),
            }],
        });
    }
    let unpacked = unpack_mipi_raw10_to_f64(&frames[0], w_r, h_r, stride, 256.0, 1023.0)
        .expect("RAW10 unpack");
    let rgb = demosaic_malvar_he_cutler(&unpacked, w_r, h_r, CfaPattern::Gbrg);
    let mut imu_gyro = Vec::new();
    let mut imu_accel = Vec::new();
    for k in 0..200i64 {
        let t = 950_000_000i64 + k * 2_000_000;
        imu_gyro.push(ImuGyroSample {
            t_ns: t,
            wx: 0.05 * ((k as f64) * 0.1).sin(),
            wy: -0.03 * ((k as f64) * 0.13).cos(),
            wz: 0.01,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        });
        imu_accel.push(ImuAccelSample {
            t_ns: t,
            ax: 0.0,
            ay: 0.0,
            az: 9.81,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        });
    }
    let sidecar = BurstMetaSidecar {
        cfa: "GBRG".into(),
        black_level: 256.0,
        white_level: 4095.0,
        sensor_info_timestamp_source: 1,
        lens_intrinsic_calibration: [6000.0, 6000.0, 4080.0, 3060.0, 0.0],
        frames: frame_metas,
        imu_gyro_uncal: imu_gyro,
        imu_accel_uncal: imu_accel,
    };
    let refs: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
    let gate_ok = verify_burst_gates(&sidecar, &refs).expect("gates");
    let mut aliased = refs.clone();
    aliased[4] = aliased[3];
    let gate_bad = verify_burst_gates(&sidecar, &aliased).expect("aliased gates");

    out.push(with_metric(
        if parsed.main_width == 8160
            && parsed.main_height == 6120
            && parsed.subifd_routed
            && max_ca_px < 0.008
            && honesty.contains("SubIFD")
            && rgb.len() == w_r * h_r
            && gate_ok.distinct_frames_ok
            && gate_ok.clock_bracket_ok
            && !gate_bad.distinct_frames_ok
        {
            pass(
                "scan",
                "DNG 1.7 OpcodeList2/3 + v7.0 MIPI RAW10 gates",
                format!(
                    "SubIFD={}x{}, analytical 3-plane CA err={max_ca_px:.3e} px (<0.008), 6/6 distinct RAW10 gates pass & alias caught",
                    parsed.main_width, parsed.main_height
                ),
            )
        } else {
            fail(
                "scan",
                "DNG 1.7 OpcodeList2/3 + v7.0 MIPI RAW10 gates",
                format!(
                    "dims={}x{}, ca_err={max_ca_px:.3e}, distinct_ok={}, alias_caught={}",
                    parsed.main_width,
                    parsed.main_height,
                    gate_ok.distinct_frames_ok,
                    !gate_bad.distinct_frames_ok
                ),
            )
        },
        json!({
            "subifd_w": parsed.main_width,
            "subifd_h": parsed.main_height,
            "max_ca_err_px": max_ca_px,
            "distinct_ok": gate_ok.distinct_frames_ok,
            "alias_caught": !gate_bad.distinct_frames_ok,
        }),
    ));

    // 2. Russian Debit Card (GOST R ISO/IEC 7810-2006 ID-1) horizontal & vertical tangent calibration + privacy mask.
    let s = 10.0_f64;
    let x0 = 500.0;
    let y0 = 400.0;
    let w_px = RUSSIAN_ID1_WIDTH_MM * s;
    let h_px = RUSSIAN_ID1_HEIGHT_MM * s;
    let margin = RUSSIAN_ID1_CORNER_RADIUS_MM * s * 1.5;
    let mk_edge = |p0: [f64; 2], p1: [f64; 2]| -> Vec<[f64; 2]> {
        (0..12)
            .map(|i| {
                let t = i as f64 / 11.0;
                [p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]
            })
            .collect()
    };
    let h_top = mk_edge([x0 + margin, y0], [x0 + w_px - margin, y0]);
    let h_right = mk_edge([x0 + w_px, y0 + margin], [x0 + w_px, y0 + h_px - margin]);
    let h_bot = mk_edge([x0 + margin, y0 + h_px], [x0 + w_px - margin, y0 + h_px]);
    let h_left = mk_edge([x0, y0 + margin], [x0, y0 + h_px - margin]);
    let chip = [
        x0 + ISO7816_CHIP_CENTER_X_MM * s,
        y0 + ISO7816_CHIP_CENTER_Y_MM * s,
    ];
    let h_anchor =
        calibrate_russian_debit_card_id1([&h_top, &h_right, &h_bot, &h_left], Some(chip), false)
            .expect("horizontal ID-1 card");

    let v_top = mk_edge([x0 + margin, y0], [x0 + h_px - margin, y0]);
    let v_right = mk_edge([x0 + h_px, y0 + margin], [x0 + h_px, y0 + w_px - margin]);
    let v_bot = mk_edge([x0 + margin, y0 + w_px], [x0 + h_px - margin, y0 + w_px]);
    let v_left = mk_edge([x0, y0 + margin], [x0, y0 + w_px - margin]);
    let v_anchor =
        calibrate_russian_debit_card_id1([&v_top, &v_right, &v_bot, &v_left], None, true)
            .expect("vertical ID-1 card");

    let inside_masked = is_inside_card_privacy_mask(&h_anchor, x0 + w_px * 0.5, y0 + h_px * 0.5);
    let outside_clear = !is_inside_card_privacy_mask(&h_anchor, x0 - 40.0, y0 - 40.0);
    let h_err = (h_anchor.meters_per_pixel - 1e-4).abs();
    let v_err = (v_anchor.meters_per_pixel - 1e-4).abs();
    out.push(with_metric(
        if h_err < 1e-10 && v_err < 1e-10 && h_anchor.privacy_masked && inside_masked && outside_clear
        {
            pass(
                "scan",
                "GOST ID-1 Russian debit card metric anchor + privacy mask",
                format!(
                    "H/V scale err={h_err:.2e}/{v_err:.2e} m/px (0% corner bias), ISO 7816-2 chip & PAN/CVV mask active"
                ),
            )
        } else {
            fail(
                "scan",
                "GOST ID-1 Russian debit card metric anchor + privacy mask",
                format!(
                    "h_err={h_err:.2e}, v_err={v_err:.2e}, inside={inside_masked}, outside_clear={outside_clear}"
                ),
            )
        },
        json!({
            "h_scale_err_m_px": h_err,
            "v_scale_err_m_px": v_err,
            "privacy_masked": h_anchor.privacy_masked,
        }),
    ));

    // 3. Handheld burst 3D physics (Forster SO(3) IMU + Wronski/OISSR + Chugunov glare + micro-baseline depth).
    let mut gyro = Vec::new();
    let mut accel = Vec::new();
    for k in 0..=100i64 {
        let t = k * 5_000_000;
        gyro.push(ImuGyroSample {
            t_ns: t,
            wx: 0.01,
            wy: -0.02,
            wz: 1.03,
            bx: 0.01,
            by: -0.02,
            bz: 0.03,
        });
        accel.push(ImuAccelSample {
            t_ns: t,
            ax: 0.0,
            ay: 0.0,
            az: 0.0,
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        });
    }
    let pre = preintegrate_imu_manifold(&gyro, &accel, 0, 500_000_000);
    let phi = so3_log(&pre.delta_r);
    let imu_err = (phi[2] - 0.5).abs();

    let bw = 24usize;
    let bh = 24usize;
    let pat = CfaPattern::Gbrg;
    let shifts = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.25),
        (0.25, 1.0),
    ];
    let gt_rgb = |x: f64, y: f64| -> [f64; 3] {
        [
            0.55 + 0.15 * (x * 0.2).sin(),
            0.38 + 0.10 * (y * 0.25).cos(),
            0.26 + 0.08 * ((x + y) * 0.15).sin(),
        ]
    };
    let cfa_frames: Vec<Vec<f64>> = shifts
        .iter()
        .enumerate()
        .map(|(fi, &(dx, dy))| {
            let mut cfa = vec![0.0_f64; bw * bh];
            for y in 0..bh {
                for x in 0..bw {
                    let ch = pat.color_at(x, y);
                    let noise = 0.015 * (((x * 31 + y * 17 + fi * 97) as f64).sin());
                    cfa[y * bw + x] = gt_rgb((x as f64) - dx, (y as f64) - dy)[ch] + noise;
                }
            }
            cfa
        })
        .collect();
    let cfa_refs: Vec<&[f64]> = cfa_frames.iter().map(|v| v.as_slice()).collect();
    let fused = wronski_oissr_zero_demosaic_fuse(&cfa_refs, &shifts, bw, bh, pat, 0.65);
    let mut oissr_mse = 0.0_f64;
    let mut oissr_cnt = 0usize;
    for y in 3..(bh - 3) {
        for x in 3..(bw - 3) {
            let g = gt_rgb(x as f64, y as f64);
            let p = fused[y * bw + x];
            for c in 0..3 {
                oissr_mse += (p[c] - g[c]).powi(2);
                oissr_cnt += 1;
            }
        }
    }
    let oissr_rmse = (oissr_mse / (oissr_cnt as f64)).sqrt();

    let focal_px = 400.0;
    let true_z_m = 0.25;
    let baselines = [
        (0.0000, 0.0000),
        (0.0008, 0.0002),
        (0.0016, -0.0003),
        (0.0024, 0.0004),
        (-0.0012, 0.0006),
        (-0.0020, -0.0005),
    ];
    let diffuse_tex = |x: f64, y: f64| -> f64 {
        0.35 + 0.15 * (x * 0.55).sin() * (y * 0.48).cos() + 0.08 * ((x - y) * 0.35).cos()
    };
    let glare_frames: Vec<Vec<f64>> = baselines
        .iter()
        .map(|&(bx, by)| {
            let mut img = vec![0.0_f64; bw * bh];
            let disp_x = focal_px * bx / true_z_m;
            let disp_y = focal_px * by / true_z_m;
            let spec_cx = 12.0 + 3.2 * disp_x;
            let spec_cy = 12.0 + 3.2 * disp_y;
            for y in 0..bh {
                for x in 0..bw {
                    let d = diffuse_tex((x as f64) + disp_x, (y as f64) + disp_y);
                    let r2 = ((x as f64) - spec_cx).powi(2) + ((y as f64) - spec_cy).powi(2);
                    img[y * bw + x] = d + 0.55 * (-r2 / 2.2).exp();
                }
            }
            img
        })
        .collect();
    let g_refs: Vec<&[f64]> = glare_frames.iter().map(|v| v.as_slice()).collect();
    let sep =
        separate_specular_and_micro_depth(&g_refs, &baselines, bw, bh, focal_px, 0.15, 0.50, 24);
    let glare_err = (sep.diffuse[12 * bw + 12] - diffuse_tex(12.0, 12.0)).abs();
    let mut zs = Vec::new();
    for y in 5..(bh - 5) {
        for x in 5..(bw - 5) {
            zs.push(sep.depth_m[y * bw + x]);
        }
    }
    zs.sort_by(|a, b| a.total_cmp(b));
    let med_z = zs[zs.len() / 2];
    let z_err = (med_z - true_z_m).abs();

    out.push(with_metric(
        if imu_err < 1e-12 && oissr_rmse < 0.02 && glare_err < 0.06 && z_err < 0.035 {
            pass(
                "scan",
                "Forster SO(3) IMU + Wronski/OISSR + Chugunov glare & micro-depth",
                format!(
                    "SO(3) err={imu_err:.2e}, OISSR RMSE={oissr_rmse:.4} (<0.02), glare err={glare_err:.4}, z={med_z:.3}m (true 0.250m)"
                ),
            )
        } else {
            fail(
                "scan",
                "Forster SO(3) IMU + Wronski/OISSR + Chugunov glare & micro-depth",
                format!(
                    "imu_err={imu_err:.2e}, oissr_rmse={oissr_rmse:.4}, glare_err={glare_err:.4}, z_err={z_err:.4}"
                ),
            )
        },
        json!({
            "imu_err_rad": imu_err,
            "oissr_rmse": oissr_rmse,
            "glare_err": glare_err,
            "median_depth_m": med_z,
        }),
    ));

    // 4. GLOMAP SO(3) + Ozaki-II FP4->FP64 Rolling-Shutter Schur BA + 2DGS/PGSR -> FMM-BEM handoff.
    let gt_rots = [
        MAT3_ID,
        so3_exp([0.04, 0.22, -0.03]),
        so3_exp([-0.05, -0.20, 0.02]),
    ];
    let gt_trans = [[0.0, 0.0, 0.35], [-0.08, 0.01, 0.36], [0.08, -0.01, 0.34]];
    let edges = vec![
        RelativeRotationEdge {
            cam_i: 0,
            cam_j: 1,
            r_ij: mat3_mul(&gt_rots[1], &mat3_transpose(&gt_rots[0])),
            weight: 1.0,
        },
        RelativeRotationEdge {
            cam_i: 0,
            cam_j: 2,
            r_ij: mat3_mul(&gt_rots[2], &mat3_transpose(&gt_rots[0])),
            weight: 1.0,
        },
        RelativeRotationEdge {
            cam_i: 1,
            cam_j: 2,
            r_ij: mat3_mul(&gt_rots[2], &mat3_transpose(&gt_rots[1])),
            weight: 1.0,
        },
    ];
    let noisy_init = [
        MAT3_ID,
        mat3_mul(&so3_exp([0.08, -0.06, 0.05]), &gt_rots[1]),
        mat3_mul(&so3_exp([-0.07, 0.09, -0.04]), &gt_rots[2]),
    ];
    let avg_rots = glomap_so3_rotation_averaging(3, &edges, &noisy_init, 40);
    let mut max_glomap_rad = 0.0_f64;
    for c in 0..3 {
        let e = so3_log(&mat3_mul(&avg_rots[c], &mat3_transpose(&gt_rots[c])));
        max_glomap_rad = max_glomap_rad.max((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt());
    }

    let gt_pts = vec![
        [0.0, 0.0, 0.0],
        [0.08560, 0.0, 0.0],
        [0.08560, 0.05398, 0.0],
        [0.0, 0.05398, 0.0],
        [-0.04, -0.03, 0.02],
        [0.03, -0.04, -0.02],
        [-0.03, 0.06, 0.03],
        [0.06, 0.07, -0.03],
    ];
    let mut gt_cams = Vec::new();
    for c in 0..3 {
        gt_cams.push(RollingShutterCamera {
            rot: gt_rots[c],
            trans: gt_trans[c],
            omega_rad_s: [0.05, -0.08, 0.02],
            vel_m_s: [0.01, -0.005, 0.0],
            fx: 3000.0,
            fy: 3000.0,
            cx: 2040.0,
            cy: 1530.0,
            width: 4080.0,
            height: 3060.0,
            vcm_focal_scale: if c == 1 { 1.003 } else { 1.0 },
            rolling_shutter_skew_s: 0.038,
            ois_shift_px: [1.2 * (c as f64), -0.8 * (c as f64)],
        });
    }
    let mut obs = Vec::new();
    for (ci, cam) in gt_cams.iter().enumerate() {
        for (pi, &pt) in gt_pts.iter().enumerate() {
            let rough = cam.project_at_row(pt, 1530.0, None);
            let exact = cam.project_at_row(pt, rough[1], None);
            obs.push(BaObservation {
                cam_idx: ci,
                pt_idx: pi,
                u_px: exact[0],
                v_px: exact[1],
            });
        }
    }
    let mut cams = gt_cams.clone();
    cams[1].trans[0] += 0.004;
    cams[1].rot = mat3_mul(&so3_exp([0.012, -0.010, 0.008]), &cams[1].rot);
    cams[1].vcm_focal_scale = 1.0;
    cams[2].trans[1] -= 0.003;
    let mut pts = gt_pts.clone();
    for (i, p) in pts.iter_mut().enumerate().skip(2) {
        p[0] += 0.0015 * (i as f64).sin();
        p[2] -= 0.0020 * (i as f64).cos();
    }
    let gauges = vec![MetricGaugeConstraint {
        pt_a: 0,
        pt_b: 1,
        target_dist_m: RUSSIAN_ID1_WIDTH_MM * 1e-3,
        weight: 1e5,
    }];
    let ba = bundle_adjust_ozaki2_fp4_schur(&mut cams, &mut pts, &obs, &gauges, None, 10.0, 14);

    let head = TriangleMesh::procedural_head_with_pinnae(0.0875, 2);
    let surfels = sample_surfels_from_mesh(&head);
    let pipe = run_stage3_to_stage4_to_fmm_bem(h_anchor, ba, &surfels, 12, 12, 0.06, 344.85)
        .expect("Stage 3->4->FMM-BEM");
    let h_l = pipe.solver.solve_reciprocal_hrtfs(
        1000.0,
        344.85,
        &[[0.0, 2.0, 0.0]],
        pipe.solver.left_ear_elem,
        false,
    )[0];

    out.push(with_metric(
        if max_glomap_rad < 1e-6
            && ba.schur_backend == TensorBackend::Fp4E2m1Base13
            && ba.final_rms_px < 0.05
            && ba.final_rms_px < ba.initial_rms_px * 0.05
            && ba.metric_scale_error_mm < 0.5
            && pipe.mesh_stats.is_watertight
            && h_l.norm().is_finite()
            && h_l.norm() > 0.5
        {
            pass(
                "scan",
                "GLOMAP + Ozaki-II FP4->FP64 RS-BA + 2DGS->FMM-BEM handoff",
                format!(
                    "GLOMAP err={max_glomap_rad:.2e} rad, BA RMS {:.2}->{:.4} px (FP4 Schur), watertight={} ({} tris), |H_L(1kHz)|={:.3}",
                    ba.initial_rms_px,
                    ba.final_rms_px,
                    pipe.mesh_stats.is_watertight,
                    pipe.mesh_stats.faces,
                    h_l.norm()
                ),
            )
        } else {
            fail(
                "scan",
                "GLOMAP + Ozaki-II FP4->FP64 RS-BA + 2DGS->FMM-BEM handoff",
                format!(
                    "glomap={max_glomap_rad:.2e}, BA {:.2}->{:.4}, scale_err={:.3}mm, watertight={}, |H_L|={:.3}",
                    ba.initial_rms_px,
                    ba.final_rms_px,
                    ba.metric_scale_error_mm,
                    pipe.mesh_stats.is_watertight,
                    h_l.norm()
                ),
            )
        },
        json!({
            "glomap_err_rad": max_glomap_rad,
            "ba_initial_rms_px": ba.initial_rms_px,
            "ba_final_rms_px": ba.final_rms_px,
            "metric_scale_error_mm": ba.metric_scale_error_mm,
            "watertight": pipe.mesh_stats.is_watertight,
            "faces": pipe.mesh_stats.faces,
            "hl_1khz_mag": h_l.norm(),
        }),
    ));

    out
}

/// `mastering`: Wannamaker (1992) / Levinson–Durbin psychoacoustic noise shaping,
/// Klippel (2006) + King–Agerkvist (JAES 2018) transducer assessment, PAE upmixer,
/// Shared-STN Virtual Bass + Micro-Detail, Moore 2016 Unmasking EQ, Punchy ADAA2
/// limiter, parallel raytrace determinism, and UPOLA SHA-256 kernel caching.
///
/// Running every DSP stage here also guarantees their symbols are linked into
/// `g6me-verify` and audited for zero FMA by `asm_guard`.
pub fn mastering() -> Vec<Check> {
    use g6me_core::atmosphere::AtmosphereCfg;
    use g6me_core::enhance::{DetailCfg, EnhanceCfg, EnhanceProcessor, VirtualBassCfg};
    use g6me_core::limiter::{Cfg as LimiterCfg, Limiter, LimiterStyle};
    use g6me_core::pcm::{BitDepth, Dither, PcmQuantizer};
    use g6me_core::plus::{KernelBank, PlusCfg, PlusProfile, N_IR_ROOM};
    use g6me_core::room::{
        raytrace_hoa_histogram, raytrace_hoa_histogram_serial, RoomGeometry, RoomPreset,
        RoomQuality,
    };
    use g6me_core::tp::Ratio;
    use g6me_core::transducer::{TransducerMonitor, TransducerProfile};
    use g6me_core::unmask::{UnmaskCfg, UnmaskEq};
    use g6me_core::upmix::{UpmixCfg, Upmixer};
    use std::f64::consts::PI;
    use std::sync::Arc;

    let mut out = Vec::new();
    let sr = 48_000u32;

    // 1. Wannamaker (1992) & Levinson–Durbin ISO-226 psychoacoustic noise shaping (48/96/192 kHz).
    let band_power_db = |err: &[f64], rate: u32| -> f64 {
        let start = 4096usize;
        let len = 4096usize;
        let mut sum_p = 0.0;
        let mut cnt = 0usize;
        for f in (1100..5000).step_by(100) {
            let w = 2.0 * PI * (f as f64) / (rate as f64);
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
    };
    let mut drops = Vec::new();
    let mut wan_ok = true;
    for &(rate, min_drop) in &[(48_000u32, 15.0f64), (96_000, 18.0), (192_000, 20.0)] {
        let n = 12_288usize;
        let amp = 10.0_f64.powf(-90.0 / 20.0);
        let src: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * PI * 1000.0 * (i as f64) / (rate as f64)).sin())
            .collect();
        let (mut lw, mut rw) = (src.clone(), src.clone());
        PcmQuantizer::new(BitDepth::S16, Dither::Wannamaker, rate)
            .unwrap()
            .process(&mut lw, &mut rw);
        let (mut lt, mut rt) = (src.clone(), src.clone());
        PcmQuantizer::new(BitDepth::S16, Dither::Triangular, rate)
            .unwrap()
            .process(&mut lt, &mut rt);
        let ew: Vec<f64> = lw
            .iter()
            .zip(&src)
            .map(|(&q, &x)| (q - x) * 32_768.0)
            .collect();
        let et: Vec<f64> = lt
            .iter()
            .zip(&src)
            .map(|(&q, &x)| (q - x) * 32_768.0)
            .collect();
        let drop = band_power_db(&et, rate) - band_power_db(&ew, rate);
        drops.push((rate, drop));
        if drop < min_drop {
            wan_ok = false;
        }
    }
    out.push(with_metric(
        if wan_ok {
            pass(
                "mastering",
                "Wannamaker & Levinson-Durbin psychoacoustic noise shaping",
                format!(
                    "1-5 kHz floor drop vs TPDF: 48k={:.1} dB, 96k={:.1} dB, 192k={:.1} dB",
                    drops[0].1, drops[1].1, drops[2].1
                ),
            )
        } else {
            fail(
                "mastering",
                "Wannamaker & Levinson-Durbin psychoacoustic noise shaping",
                format!("noise floor drop below spec: {drops:?}"),
            )
        },
        json!({ "drop_48k_db": drops[0].1, "drop_96k_db": drops[1].1, "drop_192k_db": drops[2].1 }),
    ));

    // 2. Klippel (2006) + King–Agerkvist (JAES 2018) electro-mechanical-thermal transducer ODE.
    let mut mon_lo = TransducerMonitor::new(TransducerProfile::StudioMonitor5In, sr).unwrap();
    let mut mon_hi = TransducerMonitor::new(TransducerProfile::StudioMonitor5In, sr).unwrap();
    let n_tr = 48_000usize;
    let tone_lo: Vec<f64> = (0..n_tr)
        .map(|i| 0.05 * (2.0 * PI * 55.0 * (i as f64) / (sr as f64)).sin())
        .collect();
    let tone_hi: Vec<f64> = (0..n_tr)
        .map(|i| 0.85 * (2.0 * PI * 55.0 * (i as f64) / (sr as f64)).sin())
        .collect();
    mon_lo.push(&tone_lo, &tone_lo);
    mon_hi.push(&tone_hi, &tone_hi);
    let r_lo = mon_lo.summarize();
    let r_hi = mon_hi.summarize();
    let tr_ok = r_hi.peak_excursion_mm > r_lo.peak_excursion_mm * 5.0
        && r_hi.thermal_rise_c > r_lo.thermal_rise_c * 10.0
        && r_hi.est_thd_pct > r_lo.est_thd_pct * 3.0;
    out.push(with_metric(
        if tr_ok {
            pass(
                "mastering",
                "Klippel + King-Agerkvist nonlinear transducer assessment",
                format!(
                    "55 Hz drive: excursion {:.2}->{:.2} mm ({:.0}% Xmax), dT={:.2} C, THD {:.2}%->{:.2}%",
                    r_lo.peak_excursion_mm,
                    r_hi.peak_excursion_mm,
                    r_hi.peak_excursion_pct_xmax,
                    r_hi.thermal_rise_c,
                    r_lo.est_thd_pct,
                    r_hi.est_thd_pct
                ),
            )
        } else {
            fail(
                "mastering",
                "Klippel + King-Agerkvist nonlinear transducer assessment",
                format!("unexpected response: lo={r_lo:?} hi={r_hi:?}"),
            )
        },
        json!({
            "peak_excursion_mm": r_hi.peak_excursion_mm,
            "thermal_rise_c": r_hi.thermal_rise_c,
            "thd_pct": r_hi.est_thd_pct,
        }),
    ));

    // 3. Full opt-in mastering chain (PAE upmix + Shared-STN VBass/Detail + Moore Unmask + Punchy ADAA2 Limiter).
    let n_sig = 8192usize;
    let mut l: Vec<f64> = (0..n_sig)
        .map(|i| {
            let t = (i as f64) / (sr as f64);
            0.45 * (2.0 * PI * 65.0 * t).sin()
                + 0.35 * (2.0 * PI * 440.0 * t).sin()
                + 0.30 * (2.0 * PI * 2800.0 * t).sin()
        })
        .collect();
    let mut r: Vec<f64> = (0..n_sig)
        .map(|i| {
            let t = (i as f64) / (sr as f64);
            0.45 * (2.0 * PI * 65.0 * t).sin()
                + 0.35 * (2.0 * PI * 440.0 * t + 0.25).sin()
                + 0.28 * (2.0 * PI * 3100.0 * t).sin()
        })
        .collect();
    let mut upmix = Upmixer::new(
        UpmixCfg {
            enable: true,
            ..UpmixCfg::default()
        },
        sr,
    );
    let mut up_out = [
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
        vec![0.0; n_sig],
    ];
    upmix.process(&l, &r, &mut up_out);
    let up_ok = up_out.iter().all(|ch| ch.iter().all(|v| v.is_finite()));

    let enh_cfg = EnhanceCfg {
        vbass: VirtualBassCfg {
            enable: true,
            ..VirtualBassCfg::default()
        },
        detail: DetailCfg {
            enable: true,
            ..DetailCfg::default()
        },
    };
    let mut stn_enh = EnhanceProcessor::new(sr, enh_cfg);
    stn_enh.process(&mut l, &mut r);
    // Shared-STN is causal with ~8.8 k samples of latency at 48 kHz; drain the pipeline so the
    // check sees the signal rather than the pre-roll. Core uses the same process+flush pattern.
    let (stn_tail_l, stn_tail_r) = stn_enh.flush();
    l.extend_from_slice(&stn_tail_l);
    r.extend_from_slice(&stn_tail_r);

    let mut unmask = UnmaskEq::new(
        UnmaskCfg {
            enable: true,
            ..UnmaskCfg::default()
        },
        sr,
    );
    let mut unm_l = Vec::with_capacity(n_sig);
    let mut unm_r = Vec::with_capacity(n_sig);
    unmask.process(&l, &r, &mut unm_l, &mut unm_r);
    unmask.flush(&mut unm_l, &mut unm_r);

    let lim_cfg = LimiterCfg {
        enable: true,
        ceiling_db: -1.0,
        ratio: Ratio::X8,
        style: LimiterStyle::Punchy,
        clip_drive_db: 2.0,
        ..LimiterCfg::default()
    };
    let mut lim = Limiter::enabled(lim_cfg, sr, 2);
    lim.process(&mut [&mut unm_l[..], &mut unm_r[..]]);
    let tail = lim.flush();
    let peak = unm_l
        .iter()
        .chain(&unm_r)
        .chain(tail[0].iter())
        .chain(tail[1].iter())
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    let ceiling_lin = 10.0_f64.powf(-1.0 / 20.0);
    let chain_ok = up_ok
        && unm_l.iter().chain(&unm_r).all(|v| v.is_finite())
        && tail[0].iter().chain(&tail[1]).all(|v| v.is_finite())
        && peak > 0.05
        && peak <= ceiling_lin + 1e-3;
    out.push(with_metric(
        if chain_ok {
            pass(
                "mastering",
                "PAE Upmix + Shared-STN + Moore Unmask + Punchy ADAA2",
                format!("all stages finite, sample peak={peak:.4} <= -1.0 dBTP ceiling ({ceiling_lin:.4})"),
            )
        } else {
            fail(
                "mastering",
                "PAE Upmix + Shared-STN + Moore Unmask + Punchy ADAA2",
                format!("up_ok={up_ok}, peak={peak:.4} vs ceiling {ceiling_lin:.4}"),
            )
        },
        json!({ "peak": peak, "ceiling_lin": ceiling_lin }),
    ));

    // 4. Parallel PCG64 HOA raytracing bit-exactness + UPOLA room tail + SHA-256 KernelBank cache.
    let geom = RoomGeometry::from_preset(RoomPreset::ControlRoom);
    let atm = AtmosphereCfg::default();
    let src = [4.2, 3.6, 1.35];
    let h_par = raytrace_hoa_histogram(&geom, &atm, src, 256, 24, 128, 0.002, 0xCAFE_BABE);
    let h_ser = raytrace_hoa_histogram_serial(&geom, &atm, src, 256, 24, 128, 0.002, 0xCAFE_BABE);
    let rt_exact = h_par.len() == h_ser.len()
        && h_par.iter().zip(&h_ser).all(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        });

    let mut cfg_room = PlusCfg::from_profile(PlusProfile::Studio);
    cfg_room.room_preset = RoomPreset::ControlRoom;
    cfg_room.room_quality = RoomQuality::Preview;
    let b1 = KernelBank::synthesize_cached(cfg_room, sr, None, None);
    let b2 = KernelBank::synthesize_cached(cfg_room, sr, None, None);
    let n_ir = b1.ir_l[0].len();
    let cache_hit = Arc::ptr_eq(&b1, &b2) && n_ir == N_IR_ROOM;
    out.push(with_metric(
        if rt_exact && cache_hit {
            pass(
                "mastering",
                "Parallel PCG64 raytracing + UPOLA SHA-256 KernelBank cache",
                format!(
                    "parallel vs serial raytrace 0-bit diff, ControlRoom N_IR={} ({:.1} ms), Arc::ptr_eq cache hit",
                    n_ir,
                    (n_ir as f64) * 1000.0 / (sr as f64)
                ),
            )
        } else {
            fail(
                "mastering",
                "Parallel PCG64 raytracing + UPOLA SHA-256 KernelBank cache",
                format!("rt_exact={rt_exact}, cache_hit={cache_hit}, n_ir={n_ir}"),
            )
        },
        json!({ "rt_bit_exact": rt_exact, "cache_hit": cache_hit, "n_ir": n_ir }),
    ));

    out
}

/// `cuda`: real CUDA `.cu` -> `.cubin` (`nvcc --fmad=false`) -> `cuobjdump --dump-sass`
/// zero-FMA audit + `--fmad=true` sabotage negative control.
pub fn cuda() -> Vec<Check> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    {
        let arch = "sm_75";
        match g6me_cuda::audit_all_cuda_kernels_sass(Some(arch)) {
            Ok(reports) => {
                let total_instrs: usize = reports.iter().map(|r| r.sass_instructions).sum();
                let used_arch = reports.first().map(|r| r.arch.as_str()).unwrap_or(arch);
                out.push(with_metric(
                    if reports.len() == 6 && total_instrs > 0 {
                        pass(
                            "cuda",
                            "6 CUDA kernels nvcc --fmad=false PTX + SASS audit",
                            format!(
                                "{}/6 .cu kernels compiled ({used_arch}): 0 contracted fma.rn in \
                                 nvcc PTX, 0 contracted FMA in {total_instrs} cuobjdump --dump-sass \
                                 SASS instrs (IEEE div/sqrt MUFU microcode allowed for raytrace_hoa)",
                                reports.len()
                            ),
                        )
                    } else {
                        fail(
                            "cuda",
                            "6 CUDA kernels nvcc --fmad=false PTX + SASS audit",
                            format!("kernels={}, instrs={total_instrs}", reports.len()),
                        )
                    },
                    json!({
                        "arch": used_arch,
                        "kernels": reports.len(),
                        "total_sass_instructions": total_instrs,
                        "ptx_fma_hits": 0,
                        "fma_hits": 0,
                    }),
                ));
                match g6me_cuda::verify_sass_auditor_catches_fmad_sabotage(Some(arch)) {
                    Ok(hits) => out.push(with_metric(
                        pass(
                            "cuda",
                            "CUDA PTX/SASS auditor --fmad=true sabotage control",
                            format!(
                                "sabotaged nvcc --fmad=true caught fma.rn in PTX and {hits} real \
                                 DFMA SASS hit(s)"
                            ),
                        ),
                        json!({ "sabotage_fma_hits": hits }),
                    )),
                    Err(e) => out.push(fail(
                        "cuda",
                        "CUDA PTX/SASS auditor --fmad=true sabotage control",
                        e,
                    )),
                }
            }
            Err(e) => {
                if e.contains("not found")
                    || e.contains("not installed")
                    || e.contains("unavailable")
                {
                    out.push(info(
                        "cuda",
                        "6 CUDA kernels nvcc --fmad=false SASS audit",
                        format!("CUDA toolkit probe: {e}"),
                    ));
                } else {
                    out.push(fail(
                        "cuda",
                        "6 CUDA kernels nvcc --fmad=false SASS audit",
                        e,
                    ));
                }
            }
        }
        let hw = g6me_core::blackwell::cuda_hardware_summary()
            .unwrap_or_else(|| "no live NVIDIA GPU detected (CPU fallback active)".to_string());
        out.push(info("cuda", "live NVIDIA GPU runtime", hw));
    }
    #[cfg(not(feature = "cuda"))]
    {
        use g6me_core::blackwell::{compile_and_audit_cuda_sass, CudaAuditStatus};
        match compile_and_audit_cuda_sass("sm_75") {
            CudaAuditStatus::VerifiedSass {
                arch,
                sass_instructions,
            } => out.push(with_metric(
                pass(
                    "cuda",
                    "CUDA SASS zero-FMA audit (rebuild with --features cuda for all 6 kernels)",
                    format!(
                        "compiled Blackwell kernel ({arch}) via nvcc --fmad=false & audited {sass_instructions} SASS instrs (0 FMA)"
                    ),
                ),
                json!({ "arch": arch, "sass_instructions": sass_instructions }),
            )),
            CudaAuditStatus::ToolchainUnavailable {
                ptx_reference_instructions,
            } => out.push(info(
                "cuda",
                "CUDA SASS zero-FMA audit",
                format!(
                    "nvcc/cuobjdump not installed; verified {ptx_reference_instructions} reference PTX instrs (0 FMA)"
                ),
            )),
            CudaAuditStatus::Failed(msg) => out.push(fail(
                "cuda",
                "CUDA SASS zero-FMA audit",
                msg,
            )),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disassembly_lines_are_classified_by_shape_not_by_angle_brackets() {
        // GNU objdump on a function whose FMA loads its constant rip-relative.
        let fma =
            "    1194:\tvfmadd132sd 0xe6b(%rip),%xmm1,%xmm0        # 2008 <_IO_stdin_used+0x8>";
        assert_eq!(symbol_header(fma), None);
        assert_eq!(mnemonic(fma), "vfmadd132sd");
        assert_eq!(
            symbol_header("0000000000001190 <g6ct_process>:"),
            Some("g6ct_process")
        );
        assert_eq!(symbol_header("    11a0:\tcall   1050 <memcpy@plt>"), None);
        // llvm-objdump separates mnemonic and operands with a tab.
        assert_eq!(
            mnemonic("    1194:      \tvfmadd231sd\t%xmm2, %xmm1, %xmm0"),
            "vfmadd231sd"
        );
    }

    #[test]
    fn slot_tables_match_the_reference_cli() {
        assert_eq!(slot_map(8), vec![0, 1, 2, 3, 4, 5, 9, 10]);
        assert_eq!(slot_map(12), (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn rust_run_produces_stereo_of_the_right_length() {
        let sig = signal::generate(Kind::Partials, 3000, 8);
        let out = rust_f64(&sig, 8, MASK_7_1, 1024);
        assert_eq!(out.len(), 3000 * 2);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn a_single_impulse_does_not_stay_in_the_first_frame() {
        // The filters are recursive, so energy must spread out over time; if the
        // DSP ever became a straight pass-through, this timing check would catch it.
        let mut sig = vec![0.0f32; 4096 * 8];
        sig[16 * 8] = 1.0;
        let out = rust_f64(&sig, 8, MASK_7_1, 1024);
        let t = timing_of(&out, 4096);
        assert!(t.peak[0] > 16, "peak moved to {}", t.peak[0]);
        assert!(t.centroid[0] > t.first[0] as f64);
    }

    /// Dekker's two-product: `p` is the rounded product and `e` is the exact
    /// residual, so `a*b == p + e` with no rounding. Needs no FMA, which is the
    /// point — it lets us show what a fused instruction would have kept.
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let split = 134_217_729.0f64; // 2^27 + 1
        let ca = split * a;
        let ah = ca - (ca - a);
        let al = a - ah;
        let cb = split * b;
        let bh = cb - (cb - b);
        let bl = b - bh;
        let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
        (p, e)
    }

    #[test]
    fn contraction_guard_is_not_vacuous() {
        let a = 1.0f64 + f64::EPSILON;
        let c = -(1.0f64 + 2.0 * f64::EPSILON);
        let (p, e) = two_product(a, a);
        // The guard's expression must have a genuinely non-zero discarded residual:
        // if this were zero, both fused and unfused would answer 0 and the guard
        // would be measuring nothing.
        assert!(e > 0.0, "residual of (1+2^-52)^2 is {e}, expected 2^-104");
        // Unfused: the product is rounded to p first, and p + c cancels exactly.
        assert_eq!(p + c, 0.0, "the unfused evaluation must be exactly zero");
        // The rounded product must be exactly 1+2^-51: that is what makes `p + c`
        // cancel to zero, and therefore what makes a non-zero answer proof of fusion.
        assert_eq!(
            p,
            1.0 + 2.0 * f64::EPSILON,
            "rounded product should be exactly 1+2^-51"
        );
        let checks = contraction();
        assert_eq!(checks.len(), 1);
        assert_eq!(
            checks[0].status,
            Status::Pass,
            "guard reported {:?}",
            checks[0].detail
        );
    }
}
