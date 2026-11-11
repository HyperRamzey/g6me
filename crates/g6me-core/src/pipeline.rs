//! One file, end to end: probe → decode child → f64 → downmix → encode child.
//!
//! The two ffmpeg children are independent processes and we are the pipe between
//! them, which is what makes the whole thing stream: peak memory is one block, so
//! a 3-hour Atmos concert costs the same as a 3-second clip. The downmixer is
//! driven on one thread per file, so its recursive filter state is a straight-line
//! sequence — no chunk stitching, no crossfade, nothing that could drift phase
//! (see [`crate::dsp`]).
//!
//! Optional stages (resample, IR convolution, dither, bit-depth) are ffmpeg
//! filters rather than Rust code, so each one is visible in the emitted command
//! line and independently checkable with `--dry-run`.

use crate::child::Proc;
use crate::dsp::{Dsp, SLOTS};
use crate::error::{Error, Result};
use crate::limiter::{Limiter, Report as LimiterReport};
use crate::mask::{format_order, is_downmixable, mask_of, order_from_layout, Slot};
use crate::pcm::Sha256;
use crate::probe::{probe_audio, AudioInfo, Caps, Tools};
use crate::stages::{
    effective_input_rate, input_args, input_stage, loss_chain, output_args, InputStage, RenderCfg,
};
use crate::tp::{db_from_peak, Ratio, TruePeak};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Progress callback, called from the worker thread: must be cheap and must not
/// block (the GUI bumps a counter).
pub trait Sink: Send + Sync {
    fn update(&self, frames: u64, bytes_out: u64);
}

/// A tap on the rendered blocks, used by live monitoring.
///
/// It sees exactly what the encoder would: the same filters, the same downmixer
/// state, the same block sequence. A tap that blocks slows the pump down, which
/// is how playing a file to a device stays in step with the device instead of
/// running ahead into an ever-growing buffer.
pub trait SampleTap: Send + Sync {
    /// One processed block. `planar` is indexed by [`crate::dsp::SLOTS`] and
    /// carries the *input* side, `l`/`r` the stereo output — together the ten
    /// levels the VST insert reported.
    fn tap(&self, planar: &[Option<&[f64]>; SLOTS], l: &[f64], r: &[f64], n: usize);

    /// Called once per file before its first block, with the rate the render will
    /// actually run at. A meter needs it because its release is
    /// `exp(-n / (0.30 * sample_rate))`: quote the wrong rate and a 44.1 kHz file
    /// falls 8 % slower than the plugin it is supposed to resemble.
    fn will_render(&self, _sample_rate: u32) {}
}

pub struct RenderOpts<'a> {
    pub tools: &'a Tools,
    pub caps: &'a Caps,
    /// Cooperative cancellation, shared with every stage of the render.
    ///
    /// Owned (`Arc`, not `&`) because the decode and encode stages run on their
    /// own threads and each has to poll it independently of the others.
    pub cancel: Option<Arc<AtomicBool>>,
    pub progress: Option<Arc<dyn Sink>>,
    pub tap: Option<Arc<dyn SampleTap>>,
}

impl<'a> RenderOpts<'a> {
    pub fn new(tools: &'a Tools, caps: &'a Caps) -> Self {
        Self {
            tools,
            caps,
            cancel: None,
            progress: None,
            tap: None,
        }
    }
}

/// What happened to one file, for the report, the log and the GUI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub source: PathBuf,
    pub output: PathBuf,
    pub source_info: AudioInfo,
    pub order: Vec<Slot>,
    pub mask: u32,
    pub input_stage: String,
    pub ir: Option<String>,
    pub working_rate: u32,
    pub output_rate: u32,
    pub frames: u64,
    pub seconds: f64,
    pub wall_ms: u128,
    pub realtime_factor: f64,
    pub pipe_bytes_in: u64,
    pub output_bytes: u64,
    pub encoder: String,
    pub sha256: Option<String>,
    /// SHA-256 of the finished file, which is the number `sha256sum` prints.
    ///
    /// Both hashes are reported because they answer different questions, and confusing
    /// them has a cost: `sha256` covers the samples handed to the encoder, so two runs
    /// compare equal even when the container differs (a WAV and a FLAC of one render
    /// share it), while `file_sha256` covers the bytes on disk and is what an evidence
    /// log or a dedupe pass needs. Only the second is reproducible with an external tool.
    pub file_sha256: Option<String>,
    pub warnings: Vec<String>,
    pub input_cmd: Vec<String>,
    pub encode_cmd: Vec<String>,
    pub chain: Vec<String>,
    /// True peak of the samples handed to the encoder, dBTP. This is the file's true
    /// peak, not an estimate taken before the last gain stage.
    pub true_peak_dbtp: f64,
    /// The same measurement per channel, in output channel order.
    pub per_channel_true_peak_dbtp: Vec<f64>,
    /// Which detector produced the two numbers above, because a dBTP without its ratio
    /// is not a comparable number.
    pub true_peak_ratio: Ratio,
    /// What the limiter did, when it was asked to do anything.
    pub limiter: Option<LimiterReport>,
    /// Sample peak per channel of the delivered output, dBFS. Distinguished from the
    /// true peak on purpose: a file can sit at -1 dBFS in samples and +0.8 dBTP between
    /// them, and only one of those two numbers is what a sample-rate converter clips on.
    pub sample_peak_dbfs: Vec<f64>,
    /// RMS per channel of the delivered output, dBFS: `sqrt(mean(x^2))` over the whole
    /// file, which is the mean-square figure the source meter showed next to its peak.
    /// Silence reports `f64::NEG_INFINITY`, not a lie.
    pub rms_dbfs: Vec<f64>,
}

impl Report {
    pub fn one_line(&self) -> String {
        let mut s = format!(
            "{} -> {}  [{}] {} frames  {:.1}s  {:.1}x realtime  {}",
            self.source
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            self.output
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            format_order(&self.order),
            self.frames,
            self.seconds,
            self.realtime_factor,
            crate::util::human_bytes(self.output_bytes)
        );
        // The true peak belongs on the headline line: it is the only number there that
        // describes the audio rather than the job, and it is quoted with its ratio
        // because a x2 reading and a x32 reading of one file differ by a third of a dB —
        // more than most delivery specs are graded on.
        s.push_str(&format!(
            "  tp {:+.2} dBTP @{}",
            self.true_peak_dbtp,
            self.true_peak_ratio.label()
        ));
        if let Some(l) = &self.limiter {
            s.push_str(&format!(
                "  limited to {:+.2} dBTP (peak reduction {:.2} dB)",
                l.ceiling_db, l.max_gain_reduction_db
            ));
        }
        s
    }

    /// The whole true-peak story for one file, as text, for the log and for the GUI's
    /// meter history.
    ///
    /// This exists because the numbers are worthless after a render unless they stay
    /// next to the file they describe: what the ceiling was asked to be, what the
    /// detector could actually see, what the limiter really did, and what came out.
    /// Every figure here is a measurement of the delivered stream.
    #[must_use]
    pub fn peak_line(&self) -> String {
        let chans = self
            .per_channel_true_peak_dbtp
            .iter()
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut s = format!(
            "true peak {:+.2} dBTP @{}  per channel: {}",
            self.true_peak_dbtp,
            self.true_peak_ratio.label(),
            chans
        );
        if let Some(l) = &self.limiter {
            s.push_str(&format!(
                "\nlimiter: ceiling {:+.2} dBTP @{}; aims {:.2} dB low ({:.2} dB meter + {:.2} dB gain slew)",
                l.ceiling_db,
                l.ratio.label(),
                l.guard_db,
                l.ratio.worst_underread_db(),
                l.guard_db - l.ratio.worst_underread_db()
            ));
            s.push_str(&format!(
                "; lookahead {:.2} ms asked, {:.2} ms used; latency {} samples ({:.3} ms), gain ramp {} frames",
                l.requested_lookahead_ms,
                l.effective_lookahead_ms,
                l.latency_samples,
                l.latency_ms,
                l.ramp_frames
            ));
            s.push_str(&format!(
                "; pre {:+.2} dBTP -> post {:+.2} dBTP ({} the ceiling); peak reduction {:.2} dB over {} engaged frames",
                l.pre_dbtp,
                l.post_dbtp,
                if l.within_ceiling() {
                    "measured within"
                } else {
                    "OVER"
                },
                l.max_gain_reduction_db,
                l.engaged_frames
            ));
        }
        s
    }
}

/// Decide the interleaved channel order for a stream.
///
/// Priority: an explicit user order (the only way to handle a container that lies
/// about its layout), then the layout name ffprobe reported, then the conventions
/// containers actually follow. If none of those work we refuse instead of guessing
/// — guessing is how a surround channel ends up in the front slot and the rear
/// information in the wrong ear.
pub fn resolve_order(cfg: &RenderCfg, info: &AudioInfo) -> Result<Vec<Slot>> {
    let order = if let Some(o) = &cfg.order {
        if o.len() != info.channels {
            return Err(Error::OrderMismatch {
                channels: info.channels,
                given: o.len(),
            });
        }
        if mask_of(o).count_ones() as usize != o.len() {
            return Err(Error::msg("channel order contains a duplicate channel"));
        }
        o.clone()
    } else if let Some(o) = order_from_layout(info.layout.as_deref(), info.channels) {
        o
    } else {
        return Err(Error::MaskUnknown {
            layout: info.layout.clone(),
            channels: info.channels,
        });
    };
    let mask = mask_of(&order);
    if !is_downmixable(&order) {
        return Err(Error::NoFrontPair(mask));
    }
    Ok(order)
}

/// Render `src` to `dst`.
/// What [`PeakStage`] measured about the delivered stream.
struct PeakSummary {
    true_peak_dbtp: f64,
    per_channel_true_peak_dbtp: Vec<f64>,
    sample_peak_dbfs: Vec<f64>,
    rms_dbfs: Vec<f64>,
    limiter: Option<LimiterReport>,
}

/// The last thing to touch a sample, and the pair of numbers that describe it.
///
/// Its own type because the two jobs belong together and neither reads well inline: the
/// limiter is the reason a ceiling claim is worth making, the meter is the reason it can
/// be checked. Keeping them in one place also makes the interesting part explicit rather
/// than implicit — when both want the same ratio, one measurement serves both.
struct PeakStage {
    tp: TruePeak,
    limiter: Option<Limiter>,
    /// The limiter's own post-limiter measurement is the file's measurement.
    shared: bool,
    /// Per channel: held sample peak, running sum of squares, frames seen.
    accum: [Accum; 2],
}

/// Peak and mean-square, held per channel for the whole file.
///
/// The live meter bars decay to rest within a third of a second, which is right for a
/// bar and useless as a record: once a file is finished the numbers that describe it have
/// to exist somewhere other than the screen.
#[derive(Clone, Copy, Default)]
struct Accum {
    peak: f64,
    sum_sq: f64,
    frames: u64,
}

impl Accum {
    fn push(&mut self, buf: &[f64]) {
        let mut p = self.peak;
        let mut ss = self.sum_sq;
        // Written as an explicit loop, in this order, because the sum is a reduction:
        // it must not be reassociated or contracted into an FMA any more than the
        // filters are. It is a meter, but a meter that moves between runs is a meter
        // nobody trusts.
        for &x in buf {
            let a = x.abs();
            if a > p {
                p = a;
            }
            ss += x * x;
        }
        self.peak = p;
        self.sum_sq = ss;
        self.frames += buf.len() as u64;
    }

    fn peak_db(&self) -> f64 {
        db_from_peak(self.peak)
    }

    fn rms_db(&self) -> f64 {
        if self.frames == 0 {
            return f64::NEG_INFINITY;
        }
        db_from_peak((self.sum_sq / self.frames as f64).sqrt())
    }
}

impl PeakStage {
    fn new(cfg: &RenderCfg, rate: u32) -> Self {
        let limiter = cfg
            .limiter
            .enable
            .then(|| Limiter::enabled(cfg.limiter, rate, 2));
        Self {
            tp: TruePeak::new(cfg.true_peak, 2),
            shared: cfg.limiter.enable && cfg.limiter.ratio == cfg.true_peak,
            limiter,
            accum: std::array::from_fn(|_| Accum::default()),
        }
    }

    /// Limit, then measure, in that order. Stereo, in place.
    fn process(&mut self, left: &mut [f64], right: &mut [f64]) {
        if let Some(l) = self.limiter.as_mut() {
            let mut chans: [&mut [f64]; 2] = [left, right];
            l.process(&mut chans);
        }
        if !self.shared {
            self.tp.push_ch(0, left);
            self.tp.push_ch(1, right);
        }
        // Sample peak and mean-square are always accumulated, shared meter or not: two
        // passes over a block that is already in cache, and they are the numbers the
        // history panel is built from.
        for (acc, buf) in self.accum.iter_mut().zip([left, right]) {
            acc.push(buf);
        }
    }

    /// Frames the limiter was still holding when input ended. Programme, not debris.
    fn finish_input(&mut self) -> Vec<Vec<f64>> {
        self.limiter
            .as_mut()
            .map(Limiter::flush)
            .unwrap_or_default()
    }

    /// The report's true-peak numbers, and what the limiter did if it ran.
    fn summarize(&self) -> PeakSummary {
        let report = self.limiter.as_ref().map(Limiter::report);
        let (true_peak_dbtp, per_channel_true_peak_dbtp) = match (&report, self.shared) {
            (Some(r), true) => (
                r.post_dbtp,
                self.limiter
                    .as_ref()
                    .map(Limiter::per_channel_post_dbtp)
                    .unwrap_or_default(),
            ),
            _ => (self.tp.dbtp(), self.tp.per_channel_dbtp()),
        };
        PeakSummary {
            true_peak_dbtp,
            per_channel_true_peak_dbtp,
            sample_peak_dbfs: self.accum.iter().map(Accum::peak_db).collect(),
            rms_dbfs: self.accum.iter().map(Accum::rms_db).collect(),
            limiter: report,
        }
    }
}

pub fn render_file(src: &Path, dst: &Path, cfg: &RenderCfg, opts: &RenderOpts) -> Result<Report> {
    let started = Instant::now();
    let mut warnings = Vec::new();

    let to_disk = !cfg.out.format.is_null();
    if to_disk {
        // Checked first and unconditionally: `--force` must never be able to
        // truncate the input while its own output is being written.
        if crate::util::same_file(src, dst) {
            return Err(Error::SameFile(src.to_path_buf()));
        }
        if dst.exists() && !cfg.out.overwrite {
            return Err(Error::Exists(dst.to_path_buf()));
        }
        if let Some(parent) = dst.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
    }

    let plan = plan_file(src, dst, cfg, opts.tools, opts.caps)?;
    cfg.out.validate(opts.caps)?;
    if let Some(t) = &opts.tap {
        t.will_render(plan.working_rate);
    }
    let Plan {
        info,
        order,
        mask,
        channels,
        working_rate: work_rate,
        stage,
        chain,
        input_args: in_args,
        encode_cmd: enc_args,
        ir_info,
        warnings: plan_warnings,
        ..
    } = plan;
    warnings.extend(plan_warnings);

    cfg.limiter
        .validate()
        .map_err(|e| Error::msg(format!("true-peak limiter: {e}")))?;

    // ---- true peak, and the limiter that owns it -----------------------------------
    //
    // This sits after every stage that can move a sample and before the encoder, so the
    // reported dBTP is a property of the file on disk. Metering the input, or the
    // downmixer's output before the limiter touched it, would report a number the
    // delivered file does not have.
    let mut peak = PeakStage::new(cfg, work_rate);

    let mut producer = Proc::spawn_reader("decode", &opts.tools.ffmpeg, &in_args)?;
    let mut consumer = EncoderSink::new(cfg, opts, &enc_args)?;

    // ---- three concurrent stages, one file -------------------------------
    //
    // Decode, DSP and encode run at the same time; the file costs the *slowest*
    // stage rather than the sum of all three. The DSP is still exactly one thread
    // walking one contiguous block sequence, so the recursive filter state is
    // unchanged and the output is what a single-threaded render would produce.
    let block = cfg.block.clamp(1, 1 << 20);
    let qblocks = cfg.queue_blocks.clamp(1, 4096);
    let (dec_tx, dec_rx) = crate::stage::stage::<Decoded>(qblocks);
    let (enc_tx, enc_rx) = crate::stage::stage::<Encoded>(qblocks);
    let bytes_out = Arc::new(AtomicU64::new(0));
    let aborted = Arc::new(AtomicBool::new(false));

    let reader = std::thread::Builder::new()
        .name("g6me-decode".into())
        .spawn({
            let cancel = opts.cancel.clone();
            move || decode_stage(&mut producer, &dec_tx, channels, block, cancel)
        })
        .map_err(Error::Io)?;
    let writer = std::thread::Builder::new()
        .name("g6me-encode".into())
        .spawn({
            let cancel = opts.cancel.clone();
            let abort = aborted.clone();
            let bytes = bytes_out.clone();
            let hasher = cfg.sha256.then(Sha256::new);
            move || encode_stage(&mut consumer, &enc_rx, hasher, &bytes, abort, cancel)
        })
        .map_err(Error::Io)?;

    // Stage 2 — the downmixer, on the calling thread. Nothing else touches `dsp`.
    let mut dsp = Dsp::<f64>::new(mask);
    // Reused across the whole file: one allocation, no per-block allocator churn
    // on a path that is supposed to stay ahead of a live device.
    let mut inter = vec![0f64; block * channels];
    let mut planar: [Vec<f64>; SLOTS] = std::array::from_fn(|_| Vec::with_capacity(block));
    let mut out_l = vec![0f64; block];
    let mut out_r = vec![0f64; block];
    let mut frames: u64 = 0;
    let mut pipe_bytes_in: u64 = 0;
    // Which slot each interleaved column belongs to.
    let slots: Vec<usize> = order.iter().map(|s| s.index()).collect();

    // Stop everything, join the peers, then report. Every exit from this loop
    // goes through here, so no path can leave a child process or a thread behind.
    macro_rules! bail {
        ($err:expr) => {{
            aborted.store(true, Ordering::Relaxed);
            // Unblocks the decoder (its next send fails, so it kills ffmpeg) and
            // the encoder (which aborts instead of finalising a partial file).
            dec_rx.close();
            enc_tx.close();
            let (rd, wr) = join_stages(reader, writer);
            let _ = rd.result;
            let _ = wr.result;
            if to_disk {
                // Only now, with both children confirmed gone: deleting the file
                // while an encoder still holds it open just makes it recreate it.
                let _ = std::fs::remove_file(dst);
            }
            return Err($err);
        }};
    }

    loop {
        if let Some(c) = &opts.cancel {
            if c.load(Ordering::Relaxed) {
                bail!(Error::Cancelled);
            }
        }
        let Some(dec) = dec_rx.recv() else { break };
        let n = dec.frames;
        bytes_to_f64_into(&dec.bytes, &mut inter[..n * channels]);

        // A short tail happens only at EOF, and is fine: the state simply runs
        // out mid-block, because the block sequence is contiguous.
        deinterleave(&inter[..n * channels], n, channels, &slots, &mut planar);
        let refs: [Option<&[f64]>; SLOTS] = std::array::from_fn(|i| {
            if planar[i].is_empty() {
                None
            } else {
                Some(&planar[i][..n])
            }
        });

        // Always hand the block to the downmixer, including the two-channel case:
        // `Dsp` implements the stereo bypass itself, and a second copy of that
        // logic here would be one more place for a channel to go wrong.
        dsp.process(&refs, &mut out_l[..n], &mut out_r[..n], n);
        if !dsp.is_finite() {
            bail!(Error::NonFinite { frames });
        }

        // Last stage to touch a sample, and deliberately before the tap: a monitor
        // should hear what the file will contain, ceiling and all, and the latency it
        // hears is real — which is why the report states it.
        peak.process(&mut out_l[..n], &mut out_r[..n]);

        // The tap runs before the encoder so a monitor that paces itself on a
        // device FIFO sets the tempo of the whole chain.
        if let Some(t) = &opts.tap {
            t.tap(&refs, &out_l[..n], &out_r[..n], n);
        }

        let mut out_buf = Vec::with_capacity(n * 16);
        interleave_f64(&out_l[..n], &out_r[..n], &mut out_buf);
        if !enc_tx.send(Encoded { bytes: out_buf }) {
            // The encoder died. Its own error is the informative one.
            let wr = {
                aborted.store(true, Ordering::Relaxed);
                dec_rx.close();
                join_stages(reader, writer).1
            };
            if to_disk {
                let _ = std::fs::remove_file(dst);
            }
            return Err(wr
                .result
                .err()
                .unwrap_or_else(|| Error::msg("encode stage stopped unexpectedly")));
        }
        frames += n as u64;
        pipe_bytes_in += dec.bytes.len() as u64;
        if let Some(p) = &opts.progress {
            p.update(frames, bytes_out.load(Ordering::Relaxed));
        }
    }

    // The limiter holds the last `latency` frames of the file back. They are programme,
    // not debris, and they have to reach the encoder before the stream closes: a file
    // that lost its tail would be audibly truncated at exactly the point where the
    // limiter added delay. If the encoder has already died the send fails quietly and the
    // join below reports its error, which is the informative one.
    let tail = peak.finish_input();
    if let (Some(tl), Some(tr)) = (tail.first(), tail.get(1)) {
        if !tl.is_empty() {
            frames += tl.len() as u64;
            let mut tb = Vec::with_capacity(tl.len() * 4);
            interleave_f64(tl, tr, &mut tb);
            let _ = enc_tx.send(Encoded { bytes: tb });
        }
    }

    // Normal end of the decode stream: close the DSP's ends and let both peers
    // finish their children properly (encode finalises the file).
    drop(dec_rx);
    enc_tx.close();
    let (rd, wr) = join_stages(reader, writer);

    if frames == 0 {
        // An empty result nearly always means the decode child died and printed
        // why; surface its message instead of shipping a silent file.
        let why = match (rd.result.err(), wr.result.err()) {
            (Some(e), _) => e.to_string(),
            (None, Some(e)) => e.to_string(),
            (None, None) => "no audio samples were produced".to_string(),
        };
        if to_disk {
            let _ = std::fs::remove_file(dst);
        }
        return Err(Error::ChildFailed {
            stage: "decode",
            code: -1,
            msg: if why.trim().is_empty() {
                "no audio samples were produced".to_string()
            } else {
                why
            },
        });
    }
    // The DSP stage has no error of its own left to report; the children do.
    if let Some(e) = wr.result.err() {
        if to_disk {
            let _ = std::fs::remove_file(dst);
        }
        return Err(e);
    }
    if let Some(e) = rd.result.err() {
        if to_disk {
            let _ = std::fs::remove_file(dst);
        }
        return Err(e);
    }

    let output_bytes = if to_disk {
        std::fs::metadata(dst).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };
    let encoder_name = wr.encoder;
    let sha256 = wr.sha256;
    // Hashed here, after the encoder child has been waited on and its file closed,
    // because a hash of a half-written file is worse than no hash: it looks like evidence.
    let file_sha256 = if cfg.sha256 && to_disk {
        hash_file(dst).ok()
    } else {
        None
    };

    let peak_summary = peak.summarize();

    let wall = started.elapsed();
    let seconds = frames as f64 / work_rate as f64;
    Ok(Report {
        source: src.to_path_buf(),
        output: if to_disk {
            dst.to_path_buf()
        } else {
            PathBuf::new()
        },
        source_info: info,
        order,
        mask,
        input_stage: stage.label().to_string(),
        ir: cfg.ir.enable.then(|| {
            format!(
                "{} ({} Hz, {} ch)",
                cfg.ir
                    .file
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
                ir_info.as_ref().map(|i| i.sample_rate).unwrap_or(0),
                ir_info.as_ref().map(|i| i.channels).unwrap_or(0)
            )
        }),
        working_rate: work_rate,
        output_rate: cfg.out.resample.unwrap_or(work_rate),
        frames,
        seconds,
        wall_ms: wall.as_millis(),
        realtime_factor: if wall.as_secs_f64() > 0.0 {
            seconds / wall.as_secs_f64()
        } else {
            0.0
        },
        pipe_bytes_in,
        output_bytes,
        encoder: encoder_name,
        sha256,
        file_sha256,
        warnings,
        input_cmd: in_args,
        encode_cmd: enc_args,
        chain,
        true_peak_dbtp: peak_summary.true_peak_dbtp,
        per_channel_true_peak_dbtp: peak_summary.per_channel_true_peak_dbtp,
        true_peak_ratio: cfg.true_peak,
        limiter: peak_summary.limiter,
        sample_peak_dbfs: peak_summary.sample_peak_dbfs,
        rms_dbfs: peak_summary.rms_dbfs,
    })
}

/// What a render *would* do, worked out before any sample moves.
///
/// This is the single place that turns a [`RenderCfg`] into ffmpeg command lines,
/// and [`render_file`] consumes it rather than recomputing anything — so `--dry-run`
/// shows exactly the command line that will run, not an approximation of it.
#[derive(Clone, Debug)]
pub struct Plan {
    pub info: AudioInfo,
    pub order: Vec<Slot>,
    pub mask: u32,
    pub channels: usize,
    pub working_rate: u32,
    pub stage: InputStage,
    /// Plain-language description of every operation on the samples, in order.
    pub chain: Vec<String>,
    pub input_args: Vec<String>,
    pub encode_cmd: Vec<String>,
    pub ir_info: Option<AudioInfo>,
    pub warnings: Vec<String>,
}

/// Work out the whole chain for `src` without rendering anything.
///
/// Every check that could make a render fail later (unknown layout, missing soxr,
/// IR file absent, 24-bit FLAC without the reference encoder) happens here, so a
/// dry run or a queue scan reports the real problem up front.
pub fn plan_file(
    src: &Path,
    dst: &Path,
    cfg: &RenderCfg,
    tools: &Tools,
    caps: &Caps,
) -> Result<Plan> {
    let mut warnings = Vec::new();
    let info = probe_audio(tools, src, cfg.stream)?;
    if info.channels == 0 {
        return Err(Error::MaskUnknown {
            layout: info.layout.clone(),
            channels: 0,
        });
    }
    let order = resolve_order(cfg, &info)?;
    let mask = mask_of(&order);
    let channels = order.len();
    let work_rate = effective_input_rate(cfg, &info);
    let (stage, _) = input_stage(cfg, &info, mask);
    let chain = loss_chain(cfg, &info, mask);

    if !caps.soxr && (stage == InputStage::Soxr || stage == InputStage::SoxrAndDepth) {
        return Err(Error::NoSoxr);
    }
    if cfg.ir.enable && !caps.afir {
        return Err(Error::NoAfir);
    }

    // IR metadata: needed to decide whether the IR needs a rate hop, and to say
    // out loud what a long linear-phase IR does to timing.
    let ir_info = if cfg.ir.enable {
        let p = cfg
            .ir
            .file
            .as_ref()
            .ok_or_else(|| Error::msg("IR calibration is enabled but no IR file is set"))?;
        if !p.is_file() {
            return Err(Error::IrMissing(p.to_path_buf()));
        }
        let i = probe_audio(tools, p, 0)?;
        if i.channels != 1 {
            warnings.push(format!(
                "IR is {} channels; the graph folds it to mono before convolution",
                i.channels
            ));
        }
        if i.sample_rate != work_rate {
            warnings.push(format!(
                "IR is {} Hz but the working rate is {} Hz, so the IR is resampled to match",
                i.sample_rate, work_rate
            ));
        }
        if i.duration > 0.05 {
            warnings.push(format!(
                "IR is {:.1} ms long: linear-phase convolution adds ~{:.0} ms of group delay \
                 and advances transients; use output latency (~half the IR length) if that matters",
                i.duration * 1000.0,
                i.duration * 500.0
            ));
        }
        Some(i)
    } else {
        None
    };

    let (input_args, _, _) = input_args(cfg, &info, mask, src, ir_info.as_ref())?;
    let encode_cmd = output_args(cfg, work_rate, dst, src, caps)?;
    Ok(Plan {
        info,
        order,
        mask,
        channels,
        working_rate: work_rate,
        stage,
        chain,
        input_args,
        encode_cmd,
        ir_info,
        warnings,
    })
}

/// One decoded block: interleaved `pcm_f64le`, an exact whole number of frames.
struct Decoded {
    bytes: Vec<u8>,
    frames: usize,
}

/// One rendered block: interleaved stereo `pcm_f64le`, what the encoder receives.
///
/// Just the bytes: the frame count is the DSP stage's business (it reports it),
/// and re-shipping it here would be a second number that can disagree with the first.
struct Encoded {
    bytes: Vec<u8>,
}

/// Result of the decode stage.
struct DecodeDone {
    result: Result<()>,
}

/// Result of the encode stage, including the things only it knows: which encoder
/// ran and the hash of the samples it was handed.
struct EncoderDone {
    result: Result<()>,
    encoder: String,
    sha256: Option<String>,
}

/// Stage 1: drain the decoder's stdout into blocks, as fast as it produces them.
///
/// Running this on its own thread is what keeps the ffmpeg child busy while the
/// DSP is working, instead of letting its stdout pipe fill and stall it.
fn decode_stage(
    producer: &mut Proc,
    tx: &crate::stage::Tx<Decoded>,
    channels: usize,
    block: usize,
    cancel: Option<Arc<AtomicBool>>,
) -> DecodeDone {
    let stride = channels * 8;
    let size = block * stride;
    let mut raw = vec![0u8; size];
    loop {
        if let Some(c) = &cancel {
            if c.load(Ordering::Relaxed) {
                producer.kill();
                return DecodeDone {
                    result: Err(Error::Cancelled),
                };
            }
        }
        raw.resize(size, 0);
        let raw_bytes = match read_fill(producer, &mut raw) {
            Ok(n) => n,
            Err(e) => {
                producer.kill();
                return DecodeDone { result: Err(e) };
            }
        };
        if raw_bytes == 0 {
            break;
        }
        if raw_bytes % stride != 0 {
            producer.kill();
            return DecodeDone {
                result: Err(Error::msg(format!(
                    "decode produced a partial frame ({raw_bytes} bytes, not a multiple of {stride})"
                ))),
            };
        }
        raw.truncate(raw_bytes);
        if !tx.send(Decoded {
            bytes: raw,
            frames: raw_bytes / stride,
        }) {
            // Downstream gave up; stop reading and take the child with us.
            producer.kill();
            return DecodeDone {
                result: Err(Error::Cancelled),
            };
        }
        // `send` moved the buffer; allocate the next one. The queue is bounded, so
        // at most `queue_blocks + 1` of these exist at any moment.
        raw = Vec::new();
    }
    DecodeDone {
        result: producer.finish(),
    }
}

/// Stage 3: feed the encoder, hash what it was given, and finish (or abort) it.
///
/// This thread owns the output file's lifecycle: on an abort it kills the child
/// and leaves the file un-finalised for the caller to delete, so a cancelled run
/// never leaves a playable-but-truncated file behind.
fn encode_stage(
    consumer: &mut EncoderSink,
    rx: &crate::stage::Rx<Encoded>,
    mut hasher: Option<Sha256>,
    bytes: &Arc<AtomicU64>,
    abort: Arc<AtomicBool>,
    cancel: Option<Arc<AtomicBool>>,
) -> EncoderDone {
    let mut fail: Option<Error> = None;
    loop {
        if let Some(c) = &cancel {
            if c.load(Ordering::Relaxed) {
                abort.store(true, Ordering::Relaxed);
                break;
            }
        }
        let Some(enc) = rx.recv() else { break };
        if let Err(e) = consumer.write(&enc.bytes, hasher.as_mut()) {
            fail = Some(e);
            abort.store(true, Ordering::Relaxed);
            break;
        }
        bytes.store(consumer.bytes_written(), Ordering::Relaxed);
    }
    if abort.load(Ordering::Relaxed) {
        rx.drain_discard();
        consumer.abort();
        return EncoderDone {
            result: Err(fail.unwrap_or(Error::Cancelled)),
            encoder: String::new(),
            sha256: None,
        };
    }
    let (encoder, res) = match consumer.finish() {
        Ok(n) => (n, Ok(())),
        Err(e) => (String::new(), Err(e)),
    };
    // A write that failed earlier is the more specific message than a child
    // exiting nonzero because its stdin was cut short.
    EncoderDone {
        result: if let Some(e) = fail { Err(e) } else { res },
        encoder,
        sha256: hasher.map(|h| h.finish_hex()),
    }
}

/// SHA-256 of a file, streamed, so a 3-hour render costs the same memory as a clip.
fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| Error::msg(format!("cannot open {} to hash it: {e}", path.display())))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| Error::msg(format!("cannot read {} to hash it: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finish_hex())
}

/// Wait for both peer stages, never leaking a thread even if one panicked.
fn join_stages(
    reader: std::thread::JoinHandle<DecodeDone>,
    writer: std::thread::JoinHandle<EncoderDone>,
) -> (DecodeDone, EncoderDone) {
    let rd = reader.join().unwrap_or_else(|_| DecodeDone {
        result: Err(Error::msg("decode stage panicked")),
    });
    let wr = writer.join().unwrap_or_else(|_| EncoderDone {
        result: Err(Error::msg("encode stage panicked")),
        encoder: String::new(),
        sha256: None,
    });
    (rd, wr)
}

/// Read exactly `buf.len()` bytes, or fewer only at EOF.
fn read_fill(p: &mut Proc, buf: &mut [u8]) -> Result<usize> {
    let mut off = 0;
    while off < buf.len() {
        match p.read(&mut buf[off..]) {
            Ok(0) => break,
            Ok(n) => off += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Io(e)),
        }
    }
    Ok(off)
}

/// Decode little-endian `f64` samples into `dst`.
///
/// The decoder was asked for `pcm_f64le`, so this is the whole conversion: no
/// scaling, no rounding, no quantization — every integer the decoder produced is
/// exactly representable in `f64`, which is why the f64 path costs nothing.
fn bytes_to_f64_into(src: &[u8], dst: &mut [f64]) {
    debug_assert_eq!(src.len() % 8, 0);
    for (o, c) in dst.iter_mut().zip(src.chunks_exact(8)) {
        let mut a = [0u8; 8];
        a.copy_from_slice(c);
        *o = f64::from_le_bytes(a);
    }
}

/// Spread interleaved frames into one buffer per [`crate::dsp::SLOTS`] slot.
///
/// The reference indexes its channel table by WAVEFORMATEXTENSIBLE bit position,
/// so this is where the stream's declared order becomes the module's slot order.
/// Slots the stream does not carry stay empty and are read as silence.
fn deinterleave(
    inter: &[f64],
    n: usize,
    channels: usize,
    slots: &[usize],
    planar: &mut [Vec<f64>],
) {
    for v in planar.iter_mut() {
        v.clear();
    }
    for &s in slots {
        planar[s].resize(n, 0.0);
    }
    for i in 0..n {
        let frame = &inter[i * channels..i * channels + channels];
        for (k, &s) in slots.iter().enumerate() {
            planar[s][i] = frame[k];
        }
    }
}

fn interleave_f64(l: &[f64], r: &[f64], out: &mut Vec<u8>) {
    out.reserve(l.len() * 16);
    for (a, b) in l.iter().zip(r) {
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
    }
}

/// The consuming end: one ffmpeg encode child per file.
///
/// There used to be a second shape here — ffmpeg writing raw 24-bit PCM into the
/// reference `flac` binary — because ffmpeg's FLAC encoder was believed to write
/// only 16 or 32 bits. It does not: fed 24-bit samples it emits a genuine
/// bps=24 stream, and decoding it gives back the same samples the reference
/// encoder's output gives (`scripts/check-flac-depth.sh` proves it). The extra
/// child and its pump thread existed only for a limitation that is not there, and
/// they were the source of the worst bug this project ever shipped, so the
/// encode side is one process again.
enum EncoderSink {
    /// Monitoring: samples are dropped after the tap, because a device is the
    /// only destination.
    Null {
        bytes: u64,
    },
    Plain {
        p: Proc,
        name: String,
        bytes: u64,
    },
}

impl EncoderSink {
    fn new(cfg: &RenderCfg, opts: &RenderOpts, enc_args: &[String]) -> Result<Self> {
        if cfg.out.format.is_null() {
            return Ok(Self::Null { bytes: 0 });
        }
        Ok(Self::Plain {
            p: Proc::spawn_writer("encode", &opts.tools.ffmpeg, enc_args)?,
            name: format!("ffmpeg {}", cfg.out.format.label()),
            bytes: 0,
        })
    }

    #[allow(unused_variables)]
    fn write(&mut self, buf: &[u8], hash: Option<&mut Sha256>) -> Result<()> {
        // The hash covers the samples handed to the encoder, which is what makes
        // two runs comparable regardless of how the container was written.
        if let Some(h) = hash {
            h.update(buf);
        }
        match self {
            Self::Null { bytes } => *bytes += buf.len() as u64,
            Self::Plain { p, name, bytes } => {
                if let Err(e) = p.write(buf) {
                    // The child's own stderr is the actual reason; a bare EPIPE
                    // tells the user nothing.
                    let tail = p.err_tail();
                    return Err(encoder_pipe_err(name, &tail, e));
                }
                *bytes += buf.len() as u64;
            }
        }
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        match self {
            Self::Null { bytes } => *bytes,
            Self::Plain { bytes, .. } => *bytes,
        }
    }

    fn finish(&mut self) -> Result<String> {
        match self {
            Self::Null { .. } => Ok("none (monitor only)".to_string()),
            Self::Plain { p, name, .. } => {
                p.finish()?;
                Ok(std::mem::take(name))
            }
        }
    }

    fn abort(&mut self) {
        match self {
            Self::Null { .. } => {}
            Self::Plain { p, .. } => p.kill(),
        }
    }
}

/// A closed pipe on the write side means the consumer died early; its stderr is
/// the real reason.
fn encoder_pipe_err(what: &str, err: &str, e: std::io::Error) -> Error {
    let said = err.trim();
    if said.is_empty() {
        Error::msg(format!(
            "the {what} encoder closed its input ({e}); it wrote no file"
        ))
    } else {
        Error::msg(format!("the {what} encoder died: {said}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcm::BitDepth;
    use crate::probe::probe_caps;
    use crate::stages::OutFormat;

    /// Locate ffmpeg and probe it once per test. Tests skip rather than fail on
    /// a machine with no ffmpeg, so the suite stays green on a bare CI runner.
    fn tools_ok() -> Option<(Tools, Caps)> {
        let t = Tools::discover();
        if !t.is_usable() {
            eprintln!("skipping: no ffmpeg on PATH");
            return None;
        }
        let caps = probe_caps(&t);
        Some((t, caps))
    }

    /// Run `f` on a helper thread and fail if it has not finished in `budget`.
    ///
    /// A stalled pipe is the failure mode this crate has actually hit, and a
    /// stalled test hangs CI instead of reporting anything. So the stall itself is
    /// what these tests assert on.
    fn with_watchdog<T, F>(budget: std::time::Duration, what: &str, f: F) -> T
    where
        T: std::fmt::Debug + Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        let label = what.to_string();
        std::thread::Builder::new()
            .name("g6me-watchdog-subject".into())
            .spawn(move || {
                let _ = tx.send(f());
            })
            .unwrap();
        match rx.recv_timeout(budget) {
            Ok(v) => v,
            Err(_) => panic!(
                "{label} did not finish in {}s — it is stalled",
                budget.as_secs()
            ),
        }
    }

    fn workdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("g6me-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// 8-channel WAV with an impulse at a distinct frame in each channel.
    fn impulse_wav(path: &Path, channels: u16, rate: u32) {
        let n = (rate / 4) as usize;
        let mut pcm = vec![0u8; n * channels as usize * 4];
        for c in 0..channels as usize {
            let frame = c * 97 + 13;
            let off = (frame * channels as usize + c) * 4;
            pcm[off..off + 4].copy_from_slice(&1_073_741_823i32.to_le_bytes());
        }
        crate::wav::write_pcm_s32(path, rate, channels, &pcm).unwrap();
    }

    /// The stage that could reorder channels is the *input* stage (decode plus
    /// whatever filters it runs), so that is where identity is asserted: an
    /// impulse in each of the 8 slots must arrive in the matching interleaved
    /// column and at the matching frame.
    #[test]
    fn input_stage_keeps_every_channel_in_its_own_slot() {
        let Some((tools, _caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("ident");
        let w = dir.join("imp8.wav");
        impulse_wav(&w, 8, 48000);

        let info = probe_audio(&tools, &w, 0).unwrap();
        let order = resolve_order(&RenderCfg::with_defaults(), &info).unwrap();
        assert_eq!(order.len(), 8);
        let mask = mask_of(&order);

        let (args, rate, ch) =
            input_args(&RenderCfg::with_defaults(), &info, mask, &w, None).unwrap();
        assert_eq!((rate, ch), (48000, 8));
        let samples = spawn_f64(&tools, &args);
        assert_eq!(samples.len(), 12000 * 8);

        let mut got = Vec::new();
        for c in 0..8usize {
            let mut best = (0usize, 0.0f64);
            for (frame, f) in samples.chunks_exact(8).enumerate() {
                if f[c].abs() > best.1 {
                    best = (frame, f[c].abs());
                }
            }
            assert!(best.1 > 0.49, "channel {c} lost its impulse");
            got.push(best.0);
        }
        let want: Vec<usize> = (0..8).map(|c| c * 97 + 13).collect();
        assert_eq!(got, want, "no channel may be reordered, folded or delayed");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End to end, front left must stay front left and must not swap with the
    /// right ear — the failure mode that matters for a binaural downmixer.
    #[test]
    fn stereo_left_and_right_never_swap() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("swap");
        // Distinct impulse frames per channel: 13 in FL, 110 in FR.
        let rate = 48000u32;
        let n = (rate / 4) as usize;
        let mut pcm = vec![0u8; n * 2 * 4];
        for (c, frame) in [(0usize, 13usize), (1, 110)] {
            let off = (frame * 2 + c) * 4;
            pcm[off..off + 4].copy_from_slice(&1_073_741_823i32.to_le_bytes());
        }
        let w = dir.join("swap.wav");
        crate::wav::write_pcm_s32(&w, rate, 2, &pcm).unwrap();

        let out = dir.join("swap.flac");
        let opts = RenderOpts::new(&tools, &caps);
        render_file(&w, &out, &RenderCfg::with_defaults(), &opts).unwrap();

        let back = ffmpeg_to_f64(&tools, &out);
        let peak_frame = |c: usize| {
            back.chunks_exact(2)
                .enumerate()
                .max_by(|(_, a), (_, b)| a[c].abs().total_cmp(&b[c].abs()))
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(peak_frame(0), 13, "FL stayed in the left output");
        assert_eq!(peak_frame(1), 110, "FR stayed in the right output");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Run a produced command line and collect its f64 stream.
    fn spawn_f64(tools: &Tools, args: &[String]) -> Vec<f64> {
        let mut p = Proc::spawn_reader("decode", &tools.ffmpeg, args).unwrap();
        let mut raw = Vec::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = p.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n]);
        }
        p.finish().unwrap();
        assert_eq!(raw.len() % 8, 0);
        raw.chunks_exact(8)
            .map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a)
            })
            .collect()
    }

    fn ffmpeg_to_f64(tools: &Tools, path: &Path) -> Vec<f64> {
        let mut args = vec![
            "-v".to_string(),
            "error".into(),
            "-nostdin".into(),
            "-i".into(),
        ];
        args.push(path.display().to_string());
        args.extend(
            ["-f", "f64le", "-c:a", "pcm_f64le", "-"]
                .iter()
                .map(|s| s.to_string()),
        );
        let mut p = Proc::spawn_reader("decode", &tools.ffmpeg, &args).unwrap();
        let mut raw = Vec::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = p.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&buf[..n]);
        }
        p.finish().unwrap();
        assert_eq!(raw.len() % 8, 0);
        raw.chunks_exact(8)
            .map(|c| {
                let mut a = [0u8; 8];
                a.copy_from_slice(c);
                f64::from_le_bytes(a)
            })
            .collect()
    }

    #[test]
    fn refuses_to_overwrite_the_input() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("same");
        let w = dir.join("x.wav");
        impulse_wav(&w, 2, 48000);
        let cfg = RenderCfg::with_defaults();
        let opts = RenderOpts::new(&tools, &caps);
        assert!(matches!(
            render_file(&w, &w, &cfg, &opts),
            Err(Error::SameFile(_))
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn existing_output_is_protected() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("exists");
        let w = dir.join("y.wav");
        impulse_wav(&w, 2, 48000);
        let out = dir.join("y.flac");
        std::fs::write(&out, b"keep me").unwrap();
        let opts = RenderOpts::new(&tools, &caps);
        assert!(matches!(
            render_file(&w, &out, &RenderCfg::with_defaults(), &opts),
            Err(Error::Exists(_))
        ));
        assert_eq!(std::fs::read(&out).unwrap(), b"keep me");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stereo_input_passes_through_bit_exact() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("bypass");
        // 2-channel input reaches the bypass branch (mask == FL|FR).
        let w = dir.join("st.wav");
        impulse_wav(&w, 2, 48000);
        let out = dir.join("st.flac");
        let mut cfg = RenderCfg::with_defaults();
        cfg.sha256 = true;
        let opts = RenderOpts::new(&tools, &caps);
        let r = render_file(&w, &out, &cfg, &opts).unwrap();
        assert_eq!(r.mask, crate::dsp::MASK_STEREO);
        let a = r.sha256.clone().unwrap();
        // The two hashes answer different questions, and a reader who assumes they are
        // the same number will chase a phantom: `sha256` covers the sample stream, so it
        // is deliberately *not* what `sha256sum` prints for the file.
        let on_disk = {
            use crate::pcm::Sha256;
            let mut h = Sha256::new();
            h.update(&std::fs::read(&out).unwrap());
            h.finish_hex()
        };
        assert_eq!(
            r.file_sha256.as_deref(),
            Some(on_disk.as_str()),
            "file_sha256 must be the hash of the bytes on disk"
        );
        assert_ne!(
            a, on_disk,
            "the stream hash and the file hash are different scopes; if this ever flips, \
             one of them stopped meaning what it says"
        );
        // Same input, same config, same hash: rendering is deterministic.
        let out2 = dir.join("st2.flac");
        let mut cfg2 = cfg.clone();
        cfg2.out.overwrite = true;
        let r2 = render_file(&w, &out2, &cfg2, &opts).unwrap();
        assert_eq!(
            a,
            r2.sha256.unwrap(),
            "FLAC is deterministic, so hashes must match"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn renders_the_real_atmos_sample_end_to_end() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let src = Path::new("/mnt/aosp/xlam/Atmos/AgesOfPain.mka");
        if !src.is_file() {
            eprintln!("skipping: no sample media");
            return;
        }
        let dir = workdir("e2e");
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 8 };
        cfg.out.depth = BitDepth::S24;
        cfg.sha256 = true;
        let out = dir.join("e2e.flac");
        let opts = RenderOpts::new(&tools, &caps);
        let r = render_file(src, &out, &cfg, &opts).unwrap();
        eprintln!("{}", r.one_line());
        eprintln!("chain:\n  {}", r.chain.join("\n  "));
        assert!(r.frames > 100_000);
        assert!(r.output_bytes > 1_000_000);
        assert!(
            r.realtime_factor > 0.5,
            "render should keep ahead of realtime, got {}",
            r.realtime_factor
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truehd_24bit_input_arrives_unchanged() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let src = Path::new("/mnt/aosp/xlam/Atmos/AgesOfPain.mka");
        if !src.is_file() {
            return;
        }
        // The default config must not resample: the report says "decode only".
        let dir = workdir("exact");
        let opts = RenderOpts::new(&tools, &caps);
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Wav;
        cfg.out.depth = BitDepth::S32;
        let out = dir.join("t.wav");
        let r = render_file(src, &out, &cfg, &opts).unwrap();
        assert_eq!(r.input_stage, InputStage::Passthrough.label());
        assert_eq!(r.working_rate, 48000);
        assert_eq!(r.output_rate, 48000, "no resample happened");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The default output: 24-bit FLAC written by ffmpeg. This route used to run
    /// as two children (`ffmpeg | flac`) and hung twice — once on `flac` options it
    /// rejected, once on a consumer that died without closing its channel. The
    /// second failure mode belongs to the pipe, not to the number of children, so
    /// the test stays: it asserts the file is really bps=24 *and* that the render
    /// finished inside a watchdog.
    #[test]
    fn flac24_encodes_a_long_file_without_stalling() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        if !caps.flac24 {
            eprintln!("skipping: this ffmpeg failed the 24-bit FLAC round-trip probe");
            return;
        }
        let dir = workdir("flac24");
        let w = dir.join("in.wav");
        // 30 s of 8-channel audio: long enough that flac would have written
        // megabytes of statistics had its stderr not been drained while it ran.
        let frames = 30 * 48000usize;
        let mut pcm = Vec::with_capacity(frames * 8 * 4);
        for i in 0..frames {
            let v: i32 = ((i as i32).wrapping_mul(7) & 0xffff) << 8;
            for c in 0..8i32 {
                pcm.extend_from_slice(&(v.wrapping_mul(c + 1)).to_le_bytes());
            }
        }
        crate::wav::write_pcm_s32(&w, 48000, 8, &pcm).unwrap();

        let out = dir.join("out.flac");
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 5 };
        cfg.out.depth = BitDepth::S24;
        // The watchdog thread cannot borrow, so owned copies go in and the
        // originals stay behind for the assertions.
        let (t2, c2, cfg2, w2, out2) = (
            tools.clone(),
            caps.clone(),
            cfg.clone(),
            w.clone(),
            out.clone(),
        );
        let r = with_watchdog(
            std::time::Duration::from_secs(120),
            "the 24-bit FLAC render",
            move || {
                let opts = RenderOpts::new(&t2, &c2);
                render_file(&w2, &out2, &cfg2, &opts)
            },
        )
        .unwrap();
        assert!(
            r.encoder.to_lowercase().contains("flac"),
            "encoder: {}",
            r.encoder
        );
        let (rate, ch, bps) = crate::probe::flac_streaminfo(&out).expect("FLAC stream info");
        assert_eq!((rate, ch, bps), (48000, 2, 24), "a true 24-bit FLAC file");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The live monitor on a real device clock, through the shared pipeline.
    ///
    /// Samples go to a sink that throws them away (`Monitor::open_quiet`), so this
    /// exercises cpal, the tap, the pacing and the meters **without making a sound**.
    /// On a machine with no discard device (CI) it skips; it never falls back to a
    /// device anybody could hear.
    #[test]
    fn monitor_plays_a_render_to_a_discard_device() {
        let Some((_, info)) = crate::monitor::quiet_device() else {
            eprintln!("skipping: no sample-swallowing output device here");
            return;
        };
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("monitor");
        let w = dir.join("in.wav");
        // 4 s of tone in 8 channels, loud enough for the meters to see.
        let frames = 4 * 48000usize;
        let mut pcm = Vec::with_capacity(frames * 8 * 4);
        for i in 0..frames {
            let v = (1_500_000_000i32).wrapping_mul(((i % 97) as i32 - 48) / 48);
            for _ in 0..8 {
                pcm.extend_from_slice(&v.to_le_bytes());
            }
        }
        crate::wav::write_pcm_s32(&w, 48000, 8, &pcm).unwrap();

        let meters = Arc::new(crate::meter::Meters::new());
        let mon = Arc::new(
            crate::monitor::Monitor::open_quiet(2048, meters.clone())
                .expect("a discard device must open"),
        );
        assert!(
            is_discard_sink(mon.device()),
            "not a sink: {:?}",
            mon.device().name
        );
        mon.start().expect("the clock must start");

        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Null;
        // Follow the sink's clock, exactly as the CLI does when they disagree.
        cfg.out.resample = (info.sample_rate != 48000).then_some(info.sample_rate);
        let (tools, caps, cfg, w) = (tools.clone(), caps.clone(), cfg.clone(), w);
        let out = dir.join("must-not-exist.flac");
        let mon2 = mon.clone();
        let out2 = out.clone();
        let r = with_watchdog(
            std::time::Duration::from_secs(90),
            "the monitored render",
            move || {
                let mut opts = RenderOpts::new(&tools, &caps);
                opts.tap = Some(mon2.tap());
                render_file(&w, &out2, &cfg, &opts)
            },
        )
        .expect("a monitored render must finish");
        mon.stop();

        assert_eq!(r.output_bytes, 0, "monitoring writes no file");
        assert!(!out.exists(), "OutFormat::Null must not create an output");
        let s = mon.stats();
        assert!(s.played_frames > 0, "the device clock ran: {s:?}");
        assert!(
            mon.meters().peaks().iter().any(|&p| p > 0.0),
            "the meters saw the signal"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn is_discard_sink(d: &crate::monitor::DeviceInfo) -> bool {
        crate::monitor::is_discard(d)
    }

    /// An encoder child that dies at once must **abort** the render, not strand the
    /// DSP on a stdin pipe nothing will ever read again.
    #[test]
    fn a_dead_encoder_aborts_instead_of_hanging() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };

        let dir = workdir("deadenc");
        let w = dir.join("in.wav");
        impulse_wav(&w, 8, 48000);
        // An output path that cannot be opened makes ffmpeg exit at init while the
        // decoder is still feeding us — the encoder-is-gone case, with a working
        // ffmpeg on the decode side so the decode cannot mask it.
        let out = dir.join("out.flac");
        std::fs::create_dir(&out).expect("output path made into a directory");
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 5 };
        cfg.out.depth = BitDepth::S24;
        let (t2, c2, cfg2, w2, out2) = (
            tools.clone(),
            caps.clone(),
            cfg.clone(),
            w.clone(),
            out.clone(),
        );
        let e = with_watchdog(
            std::time::Duration::from_secs(60),
            "the render against a dead encoder",
            move || {
                let opts = RenderOpts::new(&t2, &c2);
                render_file(&w2, &out2, &cfg2, &opts)
            },
        )
        .expect_err("an encoder that exits immediately cannot succeed");
        let msg = e.to_string().to_lowercase();
        assert!(
            msg.contains("flac"),
            "the error must name the encoder that died: {msg}"
        );
        assert!(
            !msg.trim().ends_with(':') && !msg.trim().ends_with("aborted"),
            "the error must carry a reason, not trail off: {msg}"
        );
        assert!(
            std::fs::read_dir(&out).map(|d| d.count()).unwrap_or(0) == 0,
            "a failed encode leaves no file behind: {msg}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A near-full-scale impulse is the cleanest true-peak fixture available: the
    /// inter-sample peak of an isolated impulse *is* its sample value, so a detector that
    /// reads materially above it is reporting its own ringing.
    fn impulse_dbfs() -> f64 {
        crate::tp::db_from_peak(1_073_741_823.0 / (1u32 << 31) as f64)
    }

    fn tone_wav(path: &Path, channels: u16, rate: u32, seconds: f64) {
        let n = (rate as f64 * seconds) as usize;
        let amp = 0.99f64;
        let mut pcm = vec![0u8; n * channels as usize * 4];
        for i in 0..n {
            // 440 Hz on one channel, 997 Hz on the other: two crest factors, and a beat
            // between them that puts inter-sample peaks well above either sample peak.
            for c in 0..channels as usize {
                let f = if c == 0 { 440.0 } else { 997.0 };
                let v = amp * (2.0 * std::f64::consts::PI * f * i as f64 / rate as f64).sin();
                let q = (v * (1i64 << 31) as f64).clamp(-2147483648.0, 2147483647.0) as i32;
                let off = (i * channels as usize + c) * 4;
                pcm[off..off + 4].copy_from_slice(&q.to_le_bytes());
            }
        }
        crate::wav::write_pcm_s32(path, rate, channels, &pcm).unwrap();
    }

    #[test]
    fn a_render_reports_the_true_peak_of_what_it_wrote() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("tp-meter");
        let w = dir.join("imp.wav");
        impulse_wav(&w, 2, 48000);
        let opts = RenderOpts::new(&tools, &caps);
        let r = render_file(
            &w,
            &dir.join("imp.flac"),
            &RenderCfg::with_defaults(),
            &opts,
        )
        .unwrap();
        // Nothing limited it, so the file's true peak is the impulse's own level.
        assert!(
            (r.true_peak_dbtp - impulse_dbfs()).abs() < 0.05,
            "reported {:+.2} dBTP, impulse sits at {:+.2} dBFS",
            r.true_peak_dbtp,
            impulse_dbfs()
        );
        assert_eq!(r.per_channel_true_peak_dbtp.len(), 2);
        assert_eq!(r.true_peak_ratio, Ratio::X4);
        assert!(
            r.limiter.is_none(),
            "a default render must not report a limiter"
        );
        assert!(r.one_line().contains("tp "), "{}", r.one_line());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Metering is a measurement. Choosing a different ratio may not move a byte, which
    /// is what lets the GUI offer x32 without invalidating a previously published hash.
    #[test]
    fn the_meter_ratio_changes_no_output_bytes() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("tp-ratio");
        let w = dir.join("tone.wav");
        tone_wav(&w, 2, 48000, 0.05);
        let opts = RenderOpts::new(&tools, &caps);
        let mut bytes: Vec<Vec<u8>> = Vec::new();
        let mut read: Vec<f64> = Vec::new();
        for ratio in [Ratio::X2, Ratio::X4, Ratio::X32] {
            let mut cfg = RenderCfg::with_defaults();
            cfg.true_peak = ratio;
            let out = dir.join(format!("x{}.flac", ratio.n()));
            let r = render_file(&w, &out, &cfg, &opts).unwrap();
            bytes.push(std::fs::read(&out).unwrap());
            read.push(r.true_peak_dbtp);
        }
        assert_eq!(bytes[0], bytes[1], "x2 and x4 wrote different files");
        assert_eq!(bytes[1], bytes[2], "x4 and x32 wrote different files");
        // The whole point of the ratio knob: x2 must under-read where x32 does not.
        assert!(
            read[0] <= read[2] + 1e-9,
            "x2 read {:+.2} above x32's {:+.2}",
            read[0],
            read[2]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The claim a user actually cares about, checked on the file rather than inside the
    /// limiter: decode what was written, measure it independently, stay under the ceiling.
    #[test]
    fn the_finished_file_respects_the_ceiling() {
        let Some((tools, caps)) = tools_ok() else {
            return;
        };
        let dir = workdir("tp-limit");
        let w = dir.join("tone.wav");
        tone_wav(&w, 2, 48000, 0.4);
        let opts = RenderOpts::new(&tools, &caps);

        let mut cfg = RenderCfg::with_defaults();
        cfg.limiter = crate::limiter::Cfg {
            enable: true,
            ceiling_db: -6.0,
            ratio: Ratio::X4,
            lookahead_ms: 1.0,
        };
        let out = dir.join("limited.wav");
        let r = render_file(&w, &out, &cfg, &opts).unwrap();

        let l = r.limiter.expect("a limited render must report the limiter");
        assert!(
            l.post_dbtp <= -6.0 + 0.01,
            "limiter reported post {:+.2} dBTP against a -6.00 dBTP ceiling",
            l.post_dbtp
        );
        assert!(l.pre_dbtp > -6.0, "the input was never over the ceiling");
        assert!(l.max_gain_reduction_db > 0.0, "the limiter never engaged");

        // Independent measurement of the bytes on disk, with a detector that shares no
        // state with the one that made the gain decisions.
        let pcm = ffmpeg_to_f64(&tools, &out);
        let mut tp = TruePeak::new(Ratio::X4, 2);
        tp.push_interleaved(&pcm);
        assert!(
            tp.dbtp() <= -6.0 + 0.05,
            "the file measures {:+.2} dBTP against a -6.00 dBTP ceiling",
            tp.dbtp()
        );

        // Delayed output is longer output, and the report has to say so.
        let mut plain = RenderCfg::with_defaults();
        plain.true_peak = Ratio::X4;
        let r0 = render_file(&w, &dir.join("plain.wav"), &plain, &opts).unwrap();
        assert_eq!(
            r.frames,
            r0.frames + l.latency_samples as u64,
            "the limiter's tail did not reach the file"
        );
        assert!(r.peak_line().contains("limiter: ceiling -6.00 dBTP"));
        assert!(
            r.peak_line().contains("measured within"),
            "{}",
            r.peak_line()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
