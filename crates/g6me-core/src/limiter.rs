//! True-peak ceiling limiter, and the metering of its own effect.
//!
//! ## Where it sits
//!
//! Immediately after the downmixer and before ffmpeg receives a byte, so it is
//! applied to the same `f64` samples the matrix produced and *before* any
//! quantisation to s16/s24/f32. A ceiling applied after quantisation would be
//! applied to a signal that has already been clipped by the format.
//!
//! ## Shape of the gain computer
//!
//! The side chain is the [`crate::tp`] detector: a zero-phase-equivalent linear-phase
//! FIR oversampled peak estimate. There is deliberately **no IIR in the side chain** —
//! no RMS detector, no feedback, no one-pole in the peak path — because an IIR has a
//! group delay of its own that the limiter cannot see ahead of, and a limiter whose
//! idea of "now" is smeared cannot honour a hard ceiling.
//!
//! Gain is channel-linked: one gain for every channel of the group. Per-channel gains
//! would change the level relationships the matrix exists to create, which makes the
//! processor change the thing the product is.
//!
//! ```text
//!  input[i] ──► per-channel TP detector ──► p[i] ──► window minimum over the
//!            (group max, linked)                      lookahead ──► target gain
//!        │                                                          │
//!        └──► delay line, L = lookahead + group delay ──► × gain ◄──┘
//! ```
//!
//! The audio is delayed by `L` so the gain applied to a sample was computed from a
//! peak that had already been measured. That is the whole content of "lookahead":
//! without the delay the gain can only react to a peak after the sample that caused
//! it has been emitted.
//!
//! ## Why the lookahead has a floor
//!
//! Three things have to fit inside the lookahead, or the ceiling is a suggestion:
//!
//! 1. the interpolator's group delay ([`Ratio::group_delay`]) — a peak is not
//!    measured until its energy has emerged from the FIR;
//! 2. the attack ramp ([`ATTACK_MS`]) — a gain step is a discontinuity in the output
//!    and is ramped linearly to avoid clicking, and a ramp has to be finished before
//!    the peak it is protecting arrives;
//! 3. one sample, so "arriving before" means strictly before.
//!
//! [`Cfg::lookahead_samples`] applies that floor, and [`Cfg::effective_lookahead_ms`]
//! reports what was actually used: a request below the floor is raised, never
//! silently honoured short. At 48 kHz and x4 the floor is 0.24 ms, so the whole
//! 0.2–2.0 ms range is available on the common settings; at 8 kHz with x32 it rises
//! to 1.05 ms and the report says so.
//!
//! ## When it is off
//!
//! [`Limiter::disabled`] returns a limiter that never attenuates and adds no delay, and
//! the pipeline skips the stage entirely, so the byte stream is the one G6ME produced
//! before this stage existed. Nothing in the disabled path touches a sample.

use crate::tp::{Detector, Ratio, TruePeak};

/// Linear attack time. Short enough to catch a single-cycle transient, long enough
/// that the gain change is not a step in the output.
pub const ATTACK_MS: f64 = 0.5;

/// Time for the gain to return to unity after a peak, as a one-pole time constant.
/// Too short and the limiter pumps on every drum; too long and it rides below unity
/// through material that needed no limiting at all.
pub const RELEASE_MS: f64 = 120.0;

/// The limiter's own share of the ceiling guard band, in dB.
///
/// Not a fudge factor, and not the whole guard. A limiter multiplies a *neighbourhood* of
/// samples by a gain that is itself moving, so the interpolated peak of the output is not
/// simply the input peak times the gain: a peak sitting on a rising gain ends up a few
/// ten-thousandths to a few hundred-thousandths of a dB above the target, and how much
/// depends on the slope the lookahead allows. Aiming slightly low buys that difference
/// back with room to spare and costs a level change no listener has ever measured.
///
/// The rest of the guard belongs to the meter: [`Ratio::worst_underread_db`] is how much
/// a peak the side chain cannot see can be worth, and no amount of lookahead recovers it.
/// [`Cfg::guard_db`] adds the two, so a declared ceiling means the same thing at every
/// ratio instead of silently meaning "the ceiling as x2 sees it".
pub const GAIN_SLEW_GUARD_DB: f64 = 0.02;

/// Lowest ceiling offered. Below this the processor is a attenuator, not a limiter,
/// and `--gain` already exists for that.
pub const MIN_CEILING_DB: f64 = -24.0;

/// Highest ceiling: digital full scale, i.e. "make sure nothing exceeds 0 dBTP".
pub const MAX_CEILING_DB: f64 = 0.0;

/// Shortest gain ramp that is a ramp rather than a step, in frames.
pub const MIN_RAMP_FRAMES: usize = 2;

/// Smallest lookahead the UI and CLI offer, in ms.
pub const MIN_LOOKAHEAD_MS: f64 = 0.2;

/// Largest lookahead offered. Past this the processor is audible as pre-ringing of
/// the gain envelope rather than as protection.
pub const MAX_LOOKAHEAD_MS: f64 = 2.0;

/// Settings for the true-peak limiter.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cfg {
    /// Off by default: the whole point of G6ME is that nothing happens unless asked.
    pub enable: bool,
    /// Output ceiling in dBTP.
    pub ceiling_db: f64,
    /// Over-sampling ratio of the side chain and of the post-limiter measurement.
    pub ratio: Ratio,
    /// Requested lookahead in ms, clamped to [`MIN_LOOKAHEAD_MS`]..=[`MAX_LOOKAHEAD_MS`].
    pub lookahead_ms: f64,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            enable: false,
            // -1 dBTP: the value a delivery spec asks for when it asks for anything,
            // and far enough below 0 that a consumer's own 4x resampler cannot clip.
            ceiling_db: -1.0,
            ratio: Ratio::X4,
            lookahead_ms: 1.0,
        }
    }
}

impl Cfg {
    /// Validate, with an error that says what to type instead.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enable {
            return Ok(());
        }
        if !(self.ceiling_db.is_finite()
            && self.ceiling_db >= MIN_CEILING_DB
            && self.ceiling_db <= MAX_CEILING_DB)
        {
            return Err(format!(
                "true-peak ceiling must be between {MIN_CEILING_DB} and \
                 {MAX_CEILING_DB} dBTP (got {})",
                self.ceiling_db
            ));
        }
        if !(self.lookahead_ms.is_finite()
            && self.lookahead_ms >= MIN_LOOKAHEAD_MS
            && self.lookahead_ms <= MAX_LOOKAHEAD_MS)
        {
            return Err(format!(
                "lookahead must be between {MIN_LOOKAHEAD_MS} and {MAX_LOOKAHEAD_MS} ms \
                 (got {self})",
                self = self.lookahead_ms
            ));
        }
        Ok(())
    }

    /// Lookahead in samples before the floor is applied.
    #[must_use]
    pub fn requested_lookahead_samples(&self, rate: u32) -> usize {
        (self.lookahead_ms * 1e-3 * rate as f64).round() as usize
    }

    /// Floor: the interpolator's memory plus the shortest ramp that is not a step.
    ///
    /// Two things have to fit in the lookahead before the gain is even allowed to try
    /// for the ceiling:
    ///
    /// * the interpolation filter's own memory (`delta`) — a peak's interpolated value
    ///   is a weighted sum over about that many input samples either side, and none of
    ///   it is measurable until the filter has produced it;
    /// * a ramp of at least two frames, because a one-frame gain change is a step and
    ///   a step in the gain is a click in the output.
    ///
    /// [`Cfg::ramp_frames`] then takes whatever is left, so the schedule can always
    /// reach the gain it needs in time — that is the arithmetic the ceiling guarantee
    /// rests on, not a hope.
    #[must_use]
    pub fn min_lookahead_samples(&self) -> usize {
        self.ratio.group_delay().ceil() as usize + MIN_RAMP_FRAMES + 1
    }

    /// Lookahead actually used, after the floor.
    #[must_use]
    pub fn lookahead_samples(&self, rate: u32) -> usize {
        self.requested_lookahead_samples(rate)
            .max(self.min_lookahead_samples())
            .max(1)
    }

    /// How far below the requested ceiling the gain aims: what the side chain's filter
    /// can fail to see, plus what a moving gain adds on top. See
    /// [`GAIN_SLEW_GUARD_DB`] for both halves and the table of what each ratio costs.
    #[must_use]
    pub fn guard_db(&self) -> f64 {
        self.ratio.worst_underread_db() + GAIN_SLEW_GUARD_DB
    }

    /// Frames over which the gain is allowed to fall, and the reason lookahead is
    /// audible: the ramp is whatever the lookahead can spare after the interpolator's
    /// memory, capped at [`ATTACK_MS`].
    ///
    /// `lookahead - delta` is exactly the room the schedule has to bring the gain down
    /// before a peak's protection window opens, so using all of it is both the
    /// smoothest choice and the only one that is always fast enough.
    #[must_use]
    pub fn ramp_frames(&self, rate: u32) -> usize {
        let la = self.lookahead_samples(rate);
        // One frame less than the room, and the reason is a proof rather than a
        // preference: the schedule for an emitted frame is exact only if every
        // neighbourhood value that can still deepen it is already known, and that
        // leaves `lookahead - delta - 1` frames of ramp.
        let room = la
            .saturating_sub(self.ratio.group_delay().ceil() as usize)
            .saturating_sub(1);
        let preferred = (ATTACK_MS * 1e-3 * rate as f64).round() as usize;
        room.max(MIN_RAMP_FRAMES)
            .min(preferred.max(MIN_RAMP_FRAMES))
    }

    /// The lookahead the processor really has, in ms. Differs from the request when
    /// the interpolator or the attack ramp needed more room, and `--dry-run` prints
    /// both so the difference is never hidden.
    #[must_use]
    pub fn effective_lookahead_ms(&self, rate: u32) -> f64 {
        self.lookahead_samples(rate) as f64 / rate as f64 * 1e3
    }

    /// True when the requested lookahead had to be raised.
    #[must_use]
    pub fn lookahead_was_raised(&self, rate: u32) -> bool {
        self.lookahead_samples(rate) > self.requested_lookahead_samples(rate)
    }

    /// Total added latency in samples: lookahead plus the interpolator's delay,
    /// because the gain for a sample comes from a peak the detector emits late.
    #[must_use]
    pub fn latency_samples(&self, rate: u32) -> usize {
        self.lookahead_samples(rate) + self.ratio.group_delay().ceil() as usize
    }
}

/// What one file's limiter pass measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Report {
    /// True peak at the limiter input, dBTP.
    pub pre_dbtp: f64,
    /// True peak at the limiter output, dBTP — measured, not computed.
    pub post_dbtp: f64,
    /// Largest gain reduction applied, in dB (positive number, 0 when untouched).
    pub max_gain_reduction_db: f64,
    /// Ceiling in force, dBTP.
    pub ceiling_db: f64,
    /// How far below that ceiling the gain aims, and why: meter error plus gain-slew
    /// error, per [`GAIN_SLEW_GUARD_DB`]. A file limited at -1 dBTP with `guard_db` of
    /// 0.72 is expected to measure around -1.72, and that is the point, not a bug.
    pub guard_db: f64,
    /// Lookahead in force, ms.
    pub effective_lookahead_ms: f64,
    /// Lookahead the user asked for, ms.
    pub requested_lookahead_ms: f64,
    /// Added latency, ms.
    pub latency_ms: f64,
    /// Added latency, samples.
    pub latency_samples: usize,
    /// Frames the gain ramp takes in force.
    pub ramp_frames: usize,
    /// Side-chain ratio.
    pub ratio: Ratio,
    /// Frames of gain history observed (0 when the limiter never engaged).
    pub engaged_frames: u64,
}

impl Report {
    /// Did the output respect the ceiling? Tolerance covers the meter's own
    /// quantisation, not a real overshoot.
    #[must_use]
    pub fn within_ceiling(&self) -> bool {
        self.post_dbtp <= self.ceiling_db
    }
}

/// Streaming true-peak limiter over planar blocks of one group.
///
/// The gain schedule is computed in three passes over a bounded ring, so memory does
/// not grow with file length and the result does not depend on how the input is cut
/// into blocks:
///
/// 1. **required gain** per input frame, from the side-chain peak the detector reports
///    for that frame;
/// 2. **neighbourhood minimum** over `±delta` frames, because a peak's interpolated
///    value is built from that many samples either side and every one of them is
///    scaled by the gain of its own frame;
/// 3. **backward ramp pass** `g[j] = min(g_a[j], g[j + 1] + slope)`, which is what
///    makes the gain *arrive* at the required depth instead of stepping there, and is
///    the reason no overshoot is possible rather than merely unlikely.
///
/// A one-pole release follower runs forward at emission, and only ever pulls the gain
/// *down* toward the schedule, so it cannot break the ceiling.
pub struct Limiter {
    cfg: Cfg,
    rate: u32,
    /// Side-chain detector per channel.
    side: Vec<Detector>,
    /// Post-limiter measurement, same ratio as the side chain.
    out_meter: TruePeak,
    /// Audio delay lines, one per channel, length `latency`.
    lines: Vec<Vec<f64>>,
    /// Write/read cursor in the delay lines.
    pos: usize,
    /// Total latency: lookahead + interpolator memory.
    latency: usize,
    /// Interpolator memory in frames: how far either side of a peak has to be covered.
    delta: usize,
    /// Frames per slot ring.
    ring_len: usize,
    /// Required gain per input frame, and the scheduled gain, on the same ring.
    req_ring: Vec<f64>,
    sched_ring: Vec<f64>,
    /// Frames fed to the side chain, and the first input frame not yet scheduled.
    fed: u64,
    scheduled: u64,
    /// Gain in force after the release follower.
    gain: f64,
    /// One-pole release coefficient.
    release_coef: f64,
    /// Fastest gain change allowed per frame, in linear terms.
    slope: f64,
    /// Amplitude the gain aims at — the ceiling less [`Cfg::guard_db`].
    ceiling_lin: f64,
    report: Report,
    engaged: bool,
}

impl Limiter {
    /// A limiter that does nothing and adds no delay. Cheaper than configuring one
    /// off, and it is what the pipeline uses when `enable` is false.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            cfg: Cfg::default(),
            rate: 48_000,
            side: Vec::new(),
            out_meter: TruePeak::new(Ratio::X4, 0),
            lines: Vec::new(),
            pos: 0,
            latency: 0,
            delta: 0,
            ring_len: 0,
            req_ring: Vec::new(),
            sched_ring: Vec::new(),
            fed: 0,
            scheduled: 0,
            gain: 1.0,
            release_coef: 1.0,
            slope: 1.0,
            ceiling_lin: 1.0,
            report: Report::default(),
            engaged: false,
        }
    }

    #[must_use]
    pub fn enabled(cfg: Cfg, rate: u32, channels: usize) -> Self {
        let delta = cfg.ratio.group_delay().ceil() as usize;
        let _lookahead = cfg.lookahead_samples(rate);
        let latency = cfg.latency_samples(rate);
        let ramp = cfg.ramp_frames(rate);
        // Every live index needs a slot: the emission cursor, the scheduling cursor
        // `delta + ramp` behind it, and the neighbourhood read `delta` ahead of the
        // scheduling cursor. Rounded up with room to spare.
        let ring_len = (latency + 2 * delta + ramp + 8).max(16);
        Self {
            cfg,
            rate,
            side: (0..channels).map(|_| Detector::new(cfg.ratio)).collect(),
            out_meter: TruePeak::new(cfg.ratio, channels),
            lines: vec![vec![0.0; latency.max(1)]; channels.max(1)],
            pos: 0,
            latency,
            delta,
            ring_len,
            req_ring: vec![1.0; ring_len],
            sched_ring: vec![1.0; ring_len],
            fed: 0,
            scheduled: 0,
            gain: 1.0,
            // One-pole release: g += (1 - g) * (1 - coef), so the gain returns to
            // unity without a step.
            release_coef: (-1.0 / (RELEASE_MS * 1e-3 * rate as f64)).exp(),
            slope: 1.0 / ramp as f64,
            ceiling_lin: 10f64.powf((cfg.ceiling_db - cfg.guard_db()) / 20.0),
            report: Report {
                ceiling_db: cfg.ceiling_db,
                guard_db: cfg.guard_db(),
                effective_lookahead_ms: cfg.effective_lookahead_ms(rate),
                requested_lookahead_ms: cfg.lookahead_ms,
                latency_ms: latency as f64 / rate as f64 * 1e3,
                latency_samples: latency,
                ramp_frames: ramp,
                ratio: cfg.ratio,
                pre_dbtp: f64::NEG_INFINITY,
                post_dbtp: f64::NEG_INFINITY,
                max_gain_reduction_db: 0.0,
                engaged_frames: 0,
            },
            engaged: false,
        }
    }

    /// Sample rate the limiter was built for.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.rate
    }

    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.cfg.enable
    }

    /// Added latency in samples, for the report and the GUI.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.latency
    }

    /// Post-limiter true peak per channel, dBTP.
    ///
    /// [`Report::post_dbtp`] is the group maximum, because that is what the ceiling is
    /// judged against — one gain for the whole group means one number that has to hold.
    /// A meter panel wants the breakdown, and the difference is not cosmetic: on 5.1
    /// material the group maximum and the loudest single channel can disagree by several
    /// dB once the LFE is in the group.
    #[must_use]
    pub fn per_channel_post_dbtp(&self) -> Vec<f64> {
        self.out_meter.per_channel_dbtp()
    }

    #[must_use]
    pub fn report(&self) -> Report {
        let mut r = self.report;
        r.pre_dbtp =
            crate::tp::db_from_peak(self.side.iter().map(Detector::peak).fold(0.0, f64::max));
        r.post_dbtp = self.out_meter.dbtp();
        r
    }

    /// Slot for an input-frame index on the ring.
    fn slot(&self, j: u64) -> usize {
        (j % self.ring_len as u64) as usize
    }

    /// Passes 2 and 3 for every input frame up to (but excluding) `limit`.
    ///
    /// Pass 3 is a *backwards* recurrence — `g[j] = min(g_a[j], g[j + 1] + slope)` —
    /// so the segment is walked right to left and seeded with the value already
    /// scheduled at `limit`, which was finalised on an earlier call. Scheduling a
    /// segment in increasing order would read a slot that has not been written yet,
    /// which is exactly the kind of error a ring buffer makes easy and a test for
    /// chunk-size invariance makes impossible to keep.
    fn schedule_to(&mut self, limit: u64) {
        if limit <= self.scheduled {
            return;
        }
        let delta = self.delta as u64;
        let slope = self.slope;
        // Seed with "the future demands nothing", which is exact for every frame this
        // routine is allowed to finalise: a demand further away than the ramp can
        // travel is worth at least 1.0 by the time it arrives, and 1.0 is no limiting
        // at all. Reading the slot at `limit` instead would read an index that was
        // never written, which is the ring-buffer mistake to make here.
        let mut acc = 1.0f64;
        for m in (self.scheduled..limit).rev() {
            // Pass 2: the gain has to sit at the required depth across the whole
            // neighbourhood the peak is built from, not just at its own frame.
            let lo = m.saturating_sub(delta);
            let mut ga = 1.0f64;
            for k in lo..=(m + delta) {
                let v = self.req_ring[self.slot(k)];
                if v < ga {
                    ga = v;
                }
            }
            // Pass 3: arrive at that depth no faster than `slope` per frame.
            acc = ga.min(acc + slope).clamp(0.0, 1.0);
            let slot = self.slot(m);
            self.sched_ring[slot] = acc;
            if acc < 1.0 {
                if !self.engaged {
                    self.engaged = true;
                }
                let gr = -20.0 * acc.max(f64::MIN_POSITIVE).log10();
                if gr > self.report.max_gain_reduction_db {
                    self.report.max_gain_reduction_db = gr;
                }
            }
        }
        self.scheduled = limit;
    }

    /// The release follower: fall instantly onto the schedule (which is itself
    /// slope-limited, so this is not a step), rise one-pole toward unity.
    fn advance_gain(&mut self, target: f64) {
        let g = self.gain;
        self.gain = if target < g {
            target
        } else {
            (g + (1.0 - g) * (1.0 - self.release_coef)).min(target)
        };
        if self.gain < 1.0 {
            self.report.engaged_frames += 1;
        }
    }

    /// Process one block of planar samples in place. Every channel in `planar` must be
    /// the same length.
    /// # Input
    ///
    /// One mutable slice per channel, all the same length, processed in place. Slices
    /// rather than `Vec`s because the render loop hands over a window of buffers it
    /// already owns; making it copy a block to satisfy the signature would be pure cost
    /// on the one path that has to stay ahead of a live device.
    pub fn process(&mut self, planar: &mut [&mut [f64]]) {
        if !self.cfg.enable || planar.is_empty() {
            return;
        }
        let n = planar[0].len();
        debug_assert!(
            planar.iter().all(|p| p.len() == n),
            "ragged block into the limiter"
        );
        let nch = self.side.len().min(planar.len());
        let latency = self.latency;
        for i in 0..n {
            // --- pass 1: linked required gain for the frame the side chain reports --
            let mut peak = 0.0f64;
            for (det, buf) in self.side[..nch].iter_mut().zip(planar.iter()) {
                let p = det.push(buf[i]);
                if p > peak {
                    peak = p;
                }
            }
            self.fed += 1;
            // The detector reports a peak `delta` frames after the input frame it
            // describes; the first `delta` feeds describe nothing yet.
            if self.fed > self.delta as u64 {
                let m = self.fed - 1 - self.delta as u64;
                let req = if peak > self.ceiling_lin {
                    self.ceiling_lin / peak
                } else {
                    1.0
                };
                let slot = self.slot(m);
                self.req_ring[slot] = req;
            }

            // A frame's schedule can still be deepened by input arriving up to
            // `2 * delta` frames later (its own neighbourhood reaches `delta` ahead,
            // and the peak it protects is reported `delta` late), so it is final only
            // once that input has been fed.
            let limit = self.fed.saturating_sub(self.delta as u64 * 2);
            self.schedule_to(limit);

            // --- emit one frame: the delay line, scaled by the follower's gain --
            // The frame coming out is input frame `fed - 1 - latency`, and its
            // schedule is already final because the scheduling cursor trails the feed
            // by `2 * delta` while the emission cursor trails it by `latency`, which
            // is larger by construction (Cfg::min_lookahead_samples).
            let j = self.fed.saturating_sub(latency as u64 + 1);
            let target = if self.fed > latency as u64 {
                self.sched_ring[self.slot(j)].clamp(0.0, 1.0)
            } else {
                1.0
            };
            self.advance_gain(target);
            let g = self.gain;
            let slot = self.pos;
            for (line, buf) in self.lines[..nch].iter_mut().zip(planar.iter_mut()) {
                let old = line[slot];
                line[slot] = buf[i];
                buf[i] = old * g;
            }
            self.pos = if slot + 1 >= latency.max(1) {
                0
            } else {
                slot + 1
            };
        }
        // Measure what came out, with the detector the ceiling was judged by.
        for (c, buf) in planar[..nch].iter().enumerate() {
            self.out_meter.push_ch(c, &buf[..n]);
        }
    }

    /// Drain the delay line at end of file, returning the trailing frames. They are
    /// real output — the last `latency` samples of a limited file live here — and
    /// dropping them would shorten every limited render.
    pub fn flush(&mut self) -> Vec<Vec<f64>> {
        if !self.cfg.enable {
            return Vec::new();
        }
        let nch = self.lines.len().min(self.side.len());
        let latency = self.latency;
        // After the last sample the signal is silence: say so on the ring, or the
        // slots the tail reads are whatever was there `ring_len` frames ago.
        for m in self.fed..(self.fed + self.delta as u64 * 2 + latency as u64 + 2) {
            let slot = self.slot(m);
            self.req_ring[slot] = 1.0;
        }
        self.schedule_to(self.fed + self.delta as u64 * 2 + latency as u64 + 1);
        let mut out = vec![Vec::new(); nch];
        for _ in 0..latency {
            let j = self.fed.saturating_sub(latency as u64 + 1);
            let target = if self.fed > latency as u64 {
                self.sched_ring[self.slot(j)].clamp(0.0, 1.0)
            } else {
                1.0
            };
            self.advance_gain(target);
            self.fed += 1;
            let slot = self.pos;
            for (line, o) in self.lines[..nch].iter_mut().zip(out.iter_mut()) {
                let old = line[slot];
                line[slot] = 0.0;
                o.push(old * self.gain);
            }
            self.pos = if slot + 1 >= latency.max(1) {
                0
            } else {
                slot + 1
            };
        }
        // The tail is programme, not debris: it is the last `latency` frames of the
        // file, delayed, and a peak can absolutely live in it. Measuring only what
        // `process` emitted would report the post-limiter true peak of everything
        // except the end of the file, which is exactly where a limiter's own report has
        // to be right — so the tail goes through the same detector the ceiling was
        // judged by before the report is allowed to be read.
        for (c, buf) in out.iter().enumerate() {
            self.out_meter.push_ch(c, buf);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn cfg(ceiling: f64, ratio: Ratio, lookahead: f64) -> Cfg {
        Cfg {
            enable: true,
            ceiling_db: ceiling,
            ratio,
            lookahead_ms: lookahead,
        }
    }

    fn burst(f: f64, amp: f64, n: usize, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * std::f64::consts::PI * f * i as f64 / FS + phase).sin())
            .collect()
    }

    /// Material designed to break a limiter: full-scale bursts at frequencies whose
    /// crests fall between samples, plus single-sample impulses whose interpolated
    /// neighbours are the actual peak.
    fn worst_case(n: usize) -> Vec<f64> {
        let mut v = vec![0.0f64; n];
        let tones = [1_000.0f64, 6_000.0, 11_000.0, 12_000.0, 15_500.0, 23_000.0];
        for (k, &f) in tones.iter().enumerate() {
            let seg = n / tones.len();
            let start = k * seg;
            let end = ((k + 1) * seg).min(n);
            let x = burst(f, 1.0, end - start, k as f64 * 0.37);
            v[start..end].copy_from_slice(&x);
        }
        // Impulses: a single sample at full scale has interpolated neighbours that
        // a sample-peak-only limiter never sees.
        for i in (0..n).step_by(997) {
            v[i] = 1.0;
        }
        v
    }

    fn run(cfg: Cfg, planar: &[Vec<f64>], chunk: usize) -> (Vec<Vec<f64>>, Report) {
        let mut lim = Limiter::enabled(cfg, FS as u32, planar.len());
        let mut out = vec![Vec::new(); planar.len()];
        // The limiter works in place, so the fixture keeps its own copy to feed it.
        let mut data: Vec<Vec<f64>> = planar.to_vec();
        let mut i = 0usize;
        while i < data[0].len() {
            let end = (i + chunk).min(data[0].len());
            let mut block: Vec<&mut [f64]> = data.iter_mut().map(|d| &mut d[i..end]).collect();
            lim.process(&mut block);
            for (c, b) in block.iter().enumerate() {
                out[c].extend_from_slice(b);
            }
            i = end;
        }
        let tail = lim.flush();
        for (c, t) in tail.iter().enumerate() {
            out[c].extend_from_slice(t);
        }
        let r = lim.report();
        (out, r)
    }

    /// The headline claim: measured output true peak never exceeds the ceiling, for
    /// every ratio, ceiling and lookahead on deliberately hostile material.
    ///
    /// Every case runs even when one fails, and the message names the worst offender,
    /// because "the first case that failed" is a much less useful number than "the
    /// largest overshoot anywhere in the matrix".
    #[test]
    fn output_true_peak_never_exceeds_the_ceiling() {
        let x = worst_case(4096);
        let mut worst = (f64::NEG_INFINITY, String::new());
        // (how far below the ceiling the best case landed, that case's guard, its name)
        let mut landing = (f64::INFINITY, 0.0f64, String::new());
        let mut cases = 0usize;
        for &ratio in Ratio::ALL.iter() {
            for &ceiling in &[-1.0f64, -3.0, -5.0, -12.0] {
                for &la in &[MIN_LOOKAHEAD_MS, 1.0, MAX_LOOKAHEAD_MS] {
                    let (out, r) = run(cfg(ceiling, ratio, la), std::slice::from_ref(&x), 333);
                    cases += 1;
                    let excess = r.post_dbtp - ceiling;
                    if excess < landing.0 {
                        landing = (excess, r.guard_db, format!("{ratio} at {ceiling} dBTP"));
                    }
                    if excess > worst.0 {
                        worst = (
                            excess,
                            format!(
                                "{ratio} at {ceiling} dBTP, {la} ms requested / {:.3} ms \
                                 effective, gain reduction {:.3} dB",
                                r.effective_lookahead_ms, r.max_gain_reduction_db
                            ),
                        );
                    }
                    assert_eq!(out.len(), 1, "channel count changed: {r:?}");
                    assert_eq!(
                        out[0].len(),
                        x.len() + r.latency_samples,
                        "output length must be input plus latency, {ratio} {la} ms"
                    );
                }
            }
        }
        assert!(
            worst.0 <= 0.0,
            "{cases} cases; worst overshoot {:+.6} dB over the ceiling, at {}",
            worst.0,
            worst.1
        );
        // The ceiling is a ceiling, but it is not supposed to be a vague one: a limiter
        // that lands 6 dB below the target is a volume knob with extra steps. Each case
        // has to come within 1 dB above its own guard band, which is what the ratio's
        // optics cost, and the worst case across the matrix is printed so the number is
        // visible rather than assumed.
        assert!(
            landing.0 > -(landing.1 + 1.0),
            "worst landing {:+.3} dB below the ceiling, at {}, where the guard band is \
             {:.3} dB: that is squashing, not limiting",
            landing.0,
            landing.1,
            landing.2
        );
    }

    /// A limiter that never engages must leave the signal alone apart from the delay.
    #[test]
    fn quiet_input_passes_through_at_unity_gain() {
        let x = burst(440.0, 0.01, 4096, 0.0);
        let (_, r) = run(cfg(-1.0, Ratio::X4, 1.0), std::slice::from_ref(&x), 512);
        assert_eq!(r.max_gain_reduction_db, 0.0, "{r:?}");
        assert_eq!(r.engaged_frames, 0, "{r:?}");
        assert_eq!(r.post_dbtp, r.pre_dbtp, "gain moved the signal: {r:?}");
    }

    /// Chunk-size invariance: the same bytes out for any block sequence, including
    /// one sample at a time.
    #[test]
    fn limiter_is_chunk_size_invariant() {
        let x = worst_case(2048);
        let (a, ra) = run(cfg(-3.0, Ratio::X8, 0.7), std::slice::from_ref(&x), 4096);
        for chunk in [1usize, 7, 512, 1021] {
            let (b, rb) = run(cfg(-3.0, Ratio::X8, 0.7), std::slice::from_ref(&x), chunk);
            assert_eq!(a[0].len(), b[0].len(), "chunk {chunk}");
            let diff = a[0]
                .iter()
                .zip(b[0].iter())
                .fold(0.0f64, |m, (&u, &v)| m.max((u - v).abs()));
            assert!(diff == 0.0, "chunk {chunk} differs by {diff}");
            assert_eq!(ra.max_gain_reduction_db, rb.max_gain_reduction_db);
        }
    }

    /// A gain step is a discontinuity in the output, so the attack is a ramp. Fed a
    /// flat-top block of 1.0 samples, the output *is* the gain, so the descent can be
    /// measured directly off the samples.
    #[test]
    fn gain_moves_on_a_ramp_not_a_step() {
        let c = cfg(-6.0, Ratio::X4, 1.0);
        let ramp = c.ramp_frames(48_000) as f64;
        let mut x = vec![0.0f64; 2048];
        for v in x.iter_mut().skip(1024) {
            *v = 1.0;
        }
        let (out, r) = run(c, &[x], 256);
        let lin = 10f64.powf(-6.0 / 20.0);
        assert!(r.max_gain_reduction_db > 5.0, "{r:?}");
        assert!(r.post_dbtp <= -6.0, "{r:?}");
        let o = &out[0];
        let first = o.iter().position(|&v| v.abs() > 1e-9).expect("no output");
        // Nothing above the ceiling, anywhere.
        assert!(
            o[first..].iter().all(|&v| v.abs() <= lin + 1e-12),
            "output exceeds the ceiling"
        );
        // The gain arrives at its destination by ramping: the largest single
        // frame-to-frame change is bounded by the slope, and the descent takes more
        // than one frame.
        let mut biggest = 0.0f64;
        let mut steps = 0usize;
        for w in o[first..first + 400].windows(2) {
            let d = (w[1] - w[0]).abs();
            if d > 1e-12 {
                steps += 1;
            }
            biggest = biggest.max(d);
        }
        assert!(
            biggest <= 1.0 / ramp + 1e-9,
            "gain moved {biggest} in one frame; slope allows {}",
            1.0 / ramp
        );
        assert!(steps >= 2, "gain moved in {steps} step, not a ramp");
    }

    /// Below the floor the request is raised, reported as raised, and never silently
    /// shortened.
    #[test]
    fn lookahead_floor_is_enforced_and_reported() {
        let c = cfg(-1.0, Ratio::X32, MIN_LOOKAHEAD_MS);
        let rate = 8_000u32;
        assert!(c.lookahead_was_raised(rate), "{c:?} at {rate}");
        assert!(
            c.effective_lookahead_ms(rate) > MIN_LOOKAHEAD_MS,
            "effective {} ms should exceed the request",
            c.effective_lookahead_ms(rate)
        );
        assert_eq!(
            c.lookahead_samples(rate),
            c.min_lookahead_samples(),
            "the floor should be what is used"
        );
        // Accuracy costs delay, and the floor is where that shows up: a 96-tap-per-phase
        // 32x bank is 47.5 samples deep, so at 48 kHz it cannot honour a 0.2 ms request
        // and says so. Every ratio must still get at least its floor, and the effective
        // value is what the UI prints — a slider that silently did something other than
        // what it shows is worse than one with a narrower range.
        for &r in Ratio::ALL.iter() {
            for &rt in &[8_000u32, 48_000, 96_000, 192_000] {
                for &la in &[MIN_LOOKAHEAD_MS, 1.0, MAX_LOOKAHEAD_MS] {
                    let k = cfg(-1.0, r, la);
                    assert!(
                        k.lookahead_samples(rt) >= k.min_lookahead_samples(),
                        "{r} at {rt}: {la} ms gives {} frames, the filter needs {}",
                        k.lookahead_samples(rt),
                        k.min_lookahead_samples()
                    );
                    assert!(
                        k.latency_samples(rt) >= k.min_lookahead_samples(),
                        "{r} at {rt}: latency must at least cover the filter's memory"
                    );
                    // The claim `lookahead_was_raised` makes, checked against the frames.
                    assert_eq!(
                        k.lookahead_was_raised(rt),
                        k.requested_lookahead_samples(rt) < k.min_lookahead_samples(),
                        "{r} at {rt} with {la} ms"
                    );
                }
            }
        }
        // Which ratios can honour the bottom of the range at a normal rate, and which
        // cannot. This is the honest shape of the control: x2 and x4 are shallow enough
        // that 0.2 ms is real, x8 and up are not, and the UI has to say so rather than
        // pretend the slider reaches 0.2 ms for every setting.
        let min_at_48k = |r: Ratio| cfg(-1.0, r, MIN_LOOKAHEAD_MS).lookahead_was_raised(48_000);
        assert!(
            !min_at_48k(Ratio::X2),
            "x2 should fit in {} ms",
            MIN_LOOKAHEAD_MS
        );
        assert!(
            !min_at_48k(Ratio::X4),
            "x4 should fit in {} ms",
            MIN_LOOKAHEAD_MS
        );
        assert!(min_at_48k(Ratio::X8), "x8 is deeper than its memory");
        assert!(min_at_48k(Ratio::X16), "x16 is deeper than its memory");
        assert!(min_at_48k(Ratio::X32), "x32 is deeper than its memory");
        // The top of the range stays usable where the sample rate is normal.
        assert!(!cfg(-1.0, Ratio::X4, MAX_LOOKAHEAD_MS).lookahead_was_raised(192_000));
        assert!(!cfg(-1.0, Ratio::X32, MAX_LOOKAHEAD_MS).lookahead_was_raised(192_000));
    }

    /// Two runs must produce identical bytes.
    #[test]
    fn limiter_is_deterministic() {
        let x = worst_case(1500);
        let (a, _) = run(cfg(-2.0, Ratio::X16, 0.4), std::slice::from_ref(&x), 222);
        let (b, _) = run(cfg(-2.0, Ratio::X16, 0.4), &[x], 222);
        assert_eq!(a[0], b[0]);
    }

    /// Linked gain: the channel balance is preserved, because a limiter that
    /// attenuated channels independently would rewrite the matrix's output.
    #[test]
    fn gain_is_linked_across_channels() {
        let loud = burst(12_000.0, 1.0, 2048, std::f64::consts::FRAC_PI_4);
        let quiet = burst(12_000.0, 0.001, 2048, std::f64::consts::FRAC_PI_4);
        let (out, _) = run(
            cfg(-6.0, Ratio::X4, 1.0),
            &[loud.clone(), quiet.clone()],
            512,
        );
        let lat = cfg(-6.0, Ratio::X4, 1.0).latency_samples(48_000);
        // Compare the tail where both channels are limited: the ratio between them
        // must be the ratio between their inputs (0.001/1.0), to within the release
        // envelope's own movement.
        let n = out[0].len();
        let mut worst = 0.0f64;
        for i in (lat + 600..n).step_by(7) {
            if out[0][i].abs() > 1e-6 && out[1][i].abs() > 1e-9 {
                let want = quiet[i.saturating_sub(lat)] / loud[i.saturating_sub(lat)];
                let got = out[1][i] / out[0][i];
                if want.abs() > 1e-9 {
                    worst = worst.max((got / want - 1.0).abs());
                }
            }
        }
        assert!(worst < 0.25, "channel balance moved by {worst}");
    }

    /// With the limiter disabled nothing is delayed and nothing is changed.
    #[test]
    fn disabled_limiter_is_an_identity() {
        let mut x = worst_case(1000);
        let before = x.clone();
        let mut lim = Limiter::disabled();
        let mut block: [&mut [f64]; 1] = [x.as_mut_slice()];
        lim.process(&mut block);
        assert_eq!(x, before, "disabled limiter changed samples");
        assert_eq!(lim.flush().len(), 0, "disabled limiter added frames");
        assert_eq!(lim.latency_samples(), 0);
        assert!(!lim.is_enabled());
    }

    #[test]
    fn config_rejects_silly_values_with_an_actionable_message() {
        assert!(cfg(-1.0, Ratio::X4, 1.0).validate().is_ok());
        assert!(cfg(1.0, Ratio::X4, 1.0).validate().is_err());
        assert!(cfg(-25.0, Ratio::X4, 1.0).validate().is_err());
        assert!(cfg(-1.0, Ratio::X4, 0.05).validate().is_err());
        assert!(cfg(-1.0, Ratio::X4, 3.0).validate().is_err());
        assert!(cfg(f64::NAN, Ratio::X4, 1.0).validate().is_err());
        // Disabled, nothing is checked: an unset ceiling must not block a render.
        assert!(Cfg::default().validate().is_ok());
    }

    /// The end of the file is inside the ceiling claim, not outside it.
    #[test]
    fn a_peak_in_the_flushed_tail_is_measured() {
        let lat = cfg(-1.0, Ratio::X4, 1.0).latency_samples(48_000);
        let mut lim = Limiter::enabled(cfg(-1.0, Ratio::X4, 1.0), 48_000, 1);
        // Silence, then a full-scale crest in the last few samples: those frames leave
        // through flush(), so a report that metered only process() would call this file
        // silent.
        let mut buf = vec![0.0f64; lat * 2];
        let last = buf.len() - 1;
        buf[last] = 0.9;
        let mut block: [&mut [f64]; 1] = [buf.as_mut_slice()];
        lim.process(&mut block);
        let tail = lim.flush();
        let r = lim.report();
        assert!(!tail[0].is_empty(), "flush returned nothing");
        let tail_peak = tail[0].iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(tail_peak > 0.5, "the crest did not reach the tail");
        assert!(
            r.post_dbtp > -2.0,
            "post-limiter true peak {:+.2} dBTP does not know about the tail crest \
             at {:+.2} dBFS",
            r.post_dbtp,
            crate::tp::db_from_peak(tail_peak)
        );
    }

    /// Trailing frames are output, not debris.
    #[test]
    fn flush_returns_the_delayed_tail() {
        let lat = cfg(-1.0, Ratio::X4, 1.0).latency_samples(48_000);
        let mut lim = Limiter::enabled(cfg(-1.0, Ratio::X4, 1.0), 48_000, 1);
        let mut buf = vec![0.5f64; 100];
        let mut block: [&mut [f64]; 1] = [buf.as_mut_slice()];
        lim.process(&mut block);
        let tail = lim.flush();
        assert_eq!(tail[0].len(), lat);
        assert_eq!(block[0].len(), 100, "process must not change block length");
        // The whole render is input plus latency: nothing lost, nothing invented.
        let all: usize = block[0].iter().chain(tail[0].iter()).count();
        assert_eq!(all, 100 + lat);
    }

    /// Multichannel: every channel is measured, and the report's post reading is the
    /// group maximum.
    #[test]
    fn multichannel_output_is_measured_as_a_group() {
        let a = burst(12_000.0, 1.0, 2048, 0.4);
        let b = burst(12_000.0, 0.9, 2048, 1.2);
        let c = burst(12_000.0, 0.5, 2048, 2.0);
        let (_, r) = run(cfg(-3.0, Ratio::X8, 1.0), &[a, b, c], 512);
        assert!(r.post_dbtp <= -3.0 + 1e-6, "{r:?}");
        assert!(r.pre_dbtp > 0.0, "input was over the ceiling: {r:?}");
        assert!(r.max_gain_reduction_db > 2.0, "{r:?}");
        assert!(r.engaged_frames > 0);
    }
}

#[cfg(test)]
mod diag {
    use super::*;
    use std::f64::consts::PI;

    /// Flatness of each ratio's magnitude response, plus its response to a single
    /// impulse. A peak-hold meter is only honest if the passband is flat: a cascade
    /// that reads +2 dB on an impulse will read high on click-heavy material too.
    #[test]
    fn response_by_class() {
        for &r in Ratio::ALL.iter() {
            // impulse
            let mut d = Detector::new(r);
            let mut p_imp = 0.0f64;
            for i in 0..400 {
                p_imp = p_imp.max(d.push(if i == 200 { 1.0 } else { 0.0 }));
            }
            // tones: settle, then measure the steady-state peak
            let mut worst_low = 0.0f64;
            let mut worst_hi = 0.0f64;
            let mut f = 0.01f64;
            while f <= 0.5001 {
                let mut lo = (0.0f64, 0.0f64);
                for off in 0..16 {
                    let mut d = Detector::new(r);
                    d.clear();
                    for i in 0..400 {
                        let _ = d.push((2.0 * PI * f * i as f64 + off as f64).sin());
                    }
                    let mut p = 0.0f64;
                    for i in 0..2000 {
                        let v = (2.0 * PI * f * (i + 400) as f64 / 1.0 + off as f64).sin();
                        p = p.max(d.push(v).abs());
                    }
                    lo = (lo.0.max(p), lo.1);
                }
                if f < 0.2 {
                    worst_low = worst_low.max(lo.0);
                } else {
                    worst_hi = worst_hi.max(lo.0);
                }
                f += 0.01;
            }
            println!(
                "{r:?}  impulse {:+.3} dB   tone passband lo {:+.3} dB  hi {:+.3} dB",
                20.0 * p_imp.log10(),
                20.0 * worst_low.log10(),
                20.0 * worst_hi.log10()
            );
        }
    }
}
