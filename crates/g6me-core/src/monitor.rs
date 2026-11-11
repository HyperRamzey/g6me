//! Live output monitoring — the VST insert's behaviour, outside a host.
//!
//! The rendered chain is written to a device instead of a file by plugging a
//! [`MonitorTap`] into [`crate::pipeline::RenderOpts::tap`] and choosing
//! [`crate::stages::OutFormat::Null`]. There is deliberately *no second DSP path
//! here*: monitoring runs the same pump, the same filters and the same
//! downmixer state as a render, so what you hear is what the file will contain.
//!
//! ## Pacing
//!
//! A render loop with no destination runs far faster than a sound card, so the
//! buffer would either grow without bound or need a drop policy that changes
//! what you hear. Instead [`Feed::push`] waits on a condvar until the device
//! callback has drained enough: the audio clock sets the tempo of the whole
//! pipeline, which is also what makes the meter timings meaningful. Nothing is
//! ever dropped while the queue is within capacity, and underruns are counted
//! rather than hidden.
//!
//! ## Sample rate
//!
//! The device dictates the rate. If a file's rate differs, enable
//! [`crate::stages::ResampleCfg`] with [`Monitor::sample_rate`] as the target —
//! the resample then appears in the loss chain and in the ffmpeg command line
//! exactly as it does for a render, instead of being a silent side-effect of
//! monitoring.

use crate::error::{Error, Result};
use crate::meter::Meters;
use crate::pipeline::SampleTap;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, FromSample, Host, OutputCallbackInfo, SampleFormat, SizedSample, Stream,
    StreamConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Default queue depth in frames (~85 ms at 48 kHz): long enough to survive a
/// compositor hiccup, short enough that the meter is not describing the past.
pub const DEFAULT_QUEUE_FRAMES: usize = 4096;

/// One selectable output device.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    pub sample_rate: u32,
    pub channels: u16,
    /// Short human name of the device's native sample format.
    pub format: String,
}

impl DeviceInfo {
    pub fn label(&self) -> String {
        format!(
            "{}{} [{} Hz, {} ch, {}]",
            self.name,
            if self.is_default { " (default)" } else { "" },
            self.sample_rate,
            self.channels,
            self.format
        )
    }
}

fn fmt_name(f: SampleFormat) -> String {
    match f {
        SampleFormat::F32 => "f32",
        SampleFormat::F64 => "f64",
        SampleFormat::I16 => "s16",
        SampleFormat::I32 => "s32",
        SampleFormat::U8 => "u8",
        _ => "other",
    }
    .to_string()
}

fn host() -> Result<Host> {
    Ok(cpal::default_host())
}

/// Name/id fragments that mean "this output goes nowhere".
///
/// ALSA's `null`, JACK/PipeWire's `auto_null`, ffmpeg-style `discard`, the ALSA
/// `dummy` driver, and macOS BlackHole all swallow samples. They are the only
/// outputs automated tests are allowed to open.
pub const DISCARD_HINTS: &[&str] = &[
    "discard",
    "null",
    "dummy",
    "blackhole",
    "auto_null",
    "sink-input",
];

/// Would sound played to this device be heard in the room?
///
/// Deliberately conservative: an unknown device counts as audible, so a device
/// that cannot be recognised as a sink is refused by the test path instead of
/// put on the user's speakers.
pub fn is_discard(info: &DeviceInfo) -> bool {
    let hay = format!("{} {}", info.name, info.id).to_lowercase();
    DISCARD_HINTS.iter().any(|h| hay.contains(h))
}

/// May this process make sound at all?
///
/// Off unless someone asked for it in their own shell. Nothing in `cargo test`
/// or CI sets it, so the suite cannot put audio on a machine's speakers; a
/// developer who wants the live check runs `G6ME_ALLOW_AUDIO_OUTPUT=1 cargo test`.
pub fn audio_output_allowed() -> bool {
    ["G6ME_ALLOW_AUDIO_OUTPUT", "G6ME_AUDIO_TEST"]
        .iter()
        .any(|k| {
            matches!(
                std::env::var(k).ok().as_deref().map(str::trim),
                Some("1") | Some("true") | Some("yes")
            )
        })
}

/// A device that can be used without anyone hearing anything: an output whose
/// name says it swallows samples.
///
/// Returns `None` when no such device exists, which is the normal state on a CI
/// runner — callers skip the live check rather than falling back to a real one.
pub fn quiet_device() -> Option<(cpal::Device, DeviceInfo)> {
    let h = host().ok()?;
    for d in h.output_devices().ok()? {
        let Ok(cfg) = d.default_output_config() else {
            continue;
        };
        let id = d.id().map(|i| i.to_string()).unwrap_or_default();
        let name = d
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        let info = DeviceInfo {
            is_default: h
                .default_output_device()
                .and_then(|x| x.id().ok())
                .map(|x| x.to_string())
                == Some(id.clone()),
            sample_rate: cfg.sample_rate(),
            channels: cfg.channels(),
            format: fmt_name(cfg.sample_format()),
            name,
            id,
        };
        if is_discard(&info) {
            return Some((d, info));
        }
    }
    None
}

/// Every output device, with the default one first.
pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let h = host()?;
    let def_id = h
        .default_output_device()
        .and_then(|d| d.id().ok())
        .map(|i| i.to_string());
    let mut out = Vec::new();
    for d in h
        .output_devices()
        .map_err(|e| Error::Device(e.to_string()))?
    {
        let id = d.id().map(|i| i.to_string()).unwrap_or_default();
        let name = d
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "(unnamed)".to_string());
        let cfg = d.default_output_config().ok();
        let info = DeviceInfo {
            is_default: def_id.as_deref() == Some(id.as_str()),
            name,
            id,
            sample_rate: cfg.map(|c| c.sample_rate()).unwrap_or(48000),
            channels: cfg.map(|c| c.channels()).unwrap_or(2),
            format: cfg.map(|c| fmt_name(c.sample_format())).unwrap_or_default(),
        };
        out.push(info);
    }
    out.sort_by_key(|d| (!d.is_default, d.name.clone()));
    Ok(out)
}

/// Find a device by id, or by a case-insensitive substring of its name, or the
/// default when `want` is empty.
pub fn find_device(want: &str) -> Result<(cpal::Device, DeviceInfo)> {
    let h = host()?;
    let mut all = Vec::new();
    for d in h
        .output_devices()
        .map_err(|e| Error::Device(e.to_string()))?
    {
        let id = d.id().map(|i| i.to_string()).unwrap_or_default();
        let name = d
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        all.push((id, name, d));
    }
    if all.is_empty() {
        return Err(Error::Device("no output devices were found".into()));
    }
    let want = want.trim();
    let pick = if want.is_empty() {
        h.default_output_device()
            .and_then(|d| {
                all.iter()
                    .find(|(_, _, kept)| kept == &d)
                    .map(|(_, _, d)| d.clone())
            })
            .or_else(|| all.first().map(|(_, _, d)| d.clone()))
            .ok_or_else(|| Error::Device("no default output device".into()))?
    } else {
        let lower = want.to_lowercase();
        all.iter()
            .find(|(id, name, _)| id == want || name.to_lowercase().contains(&lower))
            .map(|(_, _, d)| d.clone())
            .ok_or_else(|| {
                Error::Device(format!(
                    "no output device matching '{want}'. available: {}",
                    all.iter()
                        .map(|(_, n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?
    };
    let cfg = pick
        .default_output_config()
        .map_err(|e| Error::Device(format!("device config: {e}")))?;
    let info = DeviceInfo {
        id: pick.id().map(|i| i.to_string()).unwrap_or_default(),
        name: pick
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "(unnamed)".into()),
        is_default: false,
        sample_rate: cfg.sample_rate(),
        channels: cfg.channels(),
        format: fmt_name(cfg.sample_format()),
    };
    Ok((pick, info))
}

/// The hand-off between the render thread and the audio callback.
struct Feed {
    /// Interleaved stereo f32 waiting to be played.
    q: Mutex<VecDeque<f32>>,
    ready: Condvar,
    /// Queue high-water mark in frames; `push` waits here.
    capacity_frames: usize,
    closed: AtomicBool,
    played_frames: AtomicU64,
    underruns: AtomicU64,
    /// Blocks the callback had to fill with silence.
    silent_frames: AtomicU64,
}

impl Feed {
    fn new(capacity_frames: usize) -> Self {
        Self {
            q: Mutex::new(VecDeque::with_capacity(capacity_frames * 2)),
            ready: Condvar::new(),
            capacity_frames: capacity_frames.max(64),
            closed: AtomicBool::new(false),
            played_frames: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            silent_frames: AtomicU64::new(0),
        }
    }

    fn queued_frames(&self) -> usize {
        self.q.lock().map(|q| q.len() / 2).unwrap_or(0)
    }

    /// Append one block, waiting for room. Returns `false` once the monitor is
    /// closed, which is how a stopped monitor unwinds the render loop.
    fn push(&self, l: &[f32], r: &[f32]) -> bool {
        let need = l.len().max(r.len());
        let mut q = self.q.lock().unwrap_or_else(|e| e.into_inner());
        // Wait for room. The `!q.is_empty()` test is what keeps this from
        // deadlocking when a single block is larger than the whole queue: with
        // nothing queued, the block is accepted rather than refused forever.
        while !self.closed.load(Ordering::Acquire)
            && !q.is_empty()
            && q.len() / 2 + need > self.capacity_frames
        {
            let (next, _) = self
                .ready
                .wait_timeout(q, std::time::Duration::from_millis(250))
                .unwrap_or_else(|e| e.into_inner());
            q = next;
        }
        if self.closed.load(Ordering::Acquire) {
            return false;
        }
        for i in 0..need {
            q.push_back(l.get(i).copied().unwrap_or(0.0));
            q.push_back(r.get(i).copied().unwrap_or(0.0));
        }
        drop(q);
        self.ready.notify_all();
        true
    }

    /// Audio-callback side. Never blocks: if the lock is held or the queue is
    /// short, the rest of the block is silence and it is counted.
    fn drain(&self, out: &mut [f32]) {
        if out.is_empty() {
            return;
        }
        let mut written = 0usize;
        if let Ok(mut q) = self.q.try_lock() {
            for slot in out.iter_mut() {
                match q.pop_front() {
                    Some(v) => {
                        *slot = v;
                        written += 1;
                    }
                    None => break,
                }
            }
            drop(q);
            self.ready.notify_all();
        }
        let missing = out.len() - written;
        if missing > 0 {
            self.underruns.fetch_add(1, Ordering::Relaxed);
            self.silent_frames
                .fetch_add(missing as u64 / 2, Ordering::Relaxed);
            for slot in &mut out[written..] {
                *slot = 0.0;
            }
        }
        self.played_frames
            .fetch_add(written as u64 / 2, Ordering::Relaxed);
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.ready.notify_all();
    }
}

/// Live monitor attached to one output device.
pub struct Monitor {
    feed: Arc<Feed>,
    /// Behind a lock so a `Monitor` can be shared as an `Arc`: the tap handed to
    /// the render thread and the start/stop handle used by the UI are the same
    /// object, and `Arc<Monitor>` cannot hand out `&mut`.
    stream: Mutex<Option<Stream>>,
    info: DeviceInfo,
    meters: Arc<Meters>,
}

impl Monitor {
    /// Open a device and prepare (but do not start) the stream.
    ///
    /// `queue_frames` is the latency budget: samples may sit in the queue up to
    /// that many frames deep before the render thread waits.
    ///
    /// This puts sound out of a speaker. Callers that only want to exercise the
    /// path use [`Monitor::open_quiet`].
    pub fn open(want_device: &str, queue_frames: usize, meters: Arc<Meters>) -> Result<Self> {
        let (device, info) = find_device(want_device)?;
        Self::open_device(device, info, queue_frames, meters)
    }

    /// Open a **sink that swallows samples** — never the default device.
    ///
    /// This is what tests and headless self-checks use. It refuses to fall back:
    /// if there is no discard device, it errors, because the alternative is
    /// playing a test signal out of somebody's speakers or headphones.
    pub fn open_quiet(queue_frames: usize, meters: Arc<Meters>) -> Result<Self> {
        let (device, info) = quiet_device().ok_or_else(|| {
            Error::Device(
                "no sample-swallowing output device (ALSA null/discard/dummy) exists; \
                 refusing to open a audible device for a self-check"
                    .into(),
            )
        })?;
        Self::open_device(device, info, queue_frames, meters)
    }

    /// Open a device and prepare the stream, from a device the caller resolved.
    fn open_device(
        device: cpal::Device,
        info: DeviceInfo,
        queue_frames: usize,
        meters: Arc<Meters>,
    ) -> Result<Self> {
        let cfg = device
            .default_output_config()
            .map_err(|e| Error::Device(format!("default config: {e}")))?;
        let config = StreamConfig {
            channels: 2,
            sample_rate: cfg.sample_rate(),
            buffer_size: match cfg.buffer_size() {
                cpal::SupportedBufferSize::Range { min, max } => {
                    let want = (queue_frames as u32).clamp(*min, *max);
                    BufferSize::Fixed(want)
                }
                _ => BufferSize::Default,
            },
        };
        let feed = Arc::new(Feed::new(queue_frames));
        let stream = build_stream(&device, config, cfg.sample_format(), feed.clone())?;
        meters.set_format(cfg.sample_rate() as f32, queue_frames / 8);
        Ok(Self {
            feed,
            stream: Mutex::new(Some(stream)),
            info,
            meters,
        })
    }

    /// Start the device clock. From here the render thread is paced by it.
    pub fn start(&self) -> Result<()> {
        let mut guard = self.stream.lock().unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            Some(s) => s.play().map_err(|e| Error::Device(format!("play: {e}"))),
            None => Err(Error::Device("monitor was already stopped".into())),
        }
    }

    /// Stop and release the device. Unblocks a render thread waiting in `push`.
    pub fn stop(&self) {
        self.feed.close();
        *self.stream.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    pub fn is_running(&self) -> bool {
        self.stream.lock().map(|s| s.is_some()).unwrap_or(false)
            && !self.feed.closed.load(Ordering::Acquire)
    }

    /// The device's clock — the rate the rendered chain must be at.
    pub fn sample_rate(&self) -> u32 {
        self.info.sample_rate
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn meters(&self) -> &Arc<Meters> {
        &self.meters
    }

    /// Queue fill + error counters for the GUI.
    pub fn stats(&self) -> MonitorStats {
        MonitorStats {
            queued_frames: self.feed.queued_frames(),
            played_frames: self.feed.played_frames.load(Ordering::Relaxed),
            underruns: self.feed.underruns.load(Ordering::Relaxed),
            silent_frames: self.feed.silent_frames.load(Ordering::Relaxed),
            capacity_frames: self.feed.capacity_frames,
        }
    }

    /// A tap that feeds this monitor; hand it to
    /// [`crate::pipeline::RenderOpts::tap`] with `OutFormat::Null` to play a file.
    pub fn tap(self: &Arc<Self>) -> Arc<MonitorTap> {
        self.tap_for()
    }

    /// Push stereo f32 directly (for a source that is not a render).
    pub fn push(&self, l: &[f32], r: &[f32]) -> bool {
        self.feed.push(l, r)
    }

    /// A tap that also meters, sized for the device clock.
    fn tap_for(self: &Arc<Self>) -> Arc<MonitorTap> {
        Arc::new(MonitorTap {
            feed: self.feed.clone(),
            meters: self.meters.clone(),
            sample_rate: self.info.sample_rate as f32,
        })
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        // Release the device and never leave a render thread parked on a feed
        // nobody is draining any more.
        self.feed.close();
        *self.stream.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

/// Counters shown next to the meters.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct MonitorStats {
    /// Frames waiting to be played right now.
    pub queued_frames: usize,
    /// Frames the device has actually consumed.
    pub played_frames: u64,
    /// Callbacks that had to be padded with silence.
    pub underruns: u64,
    /// Total frames of that padding.
    pub silent_frames: u64,
    /// The latency budget the queue was opened with.
    pub capacity_frames: usize,
}

impl MonitorStats {
    /// Queue fill as a fraction of the budget; the GUI draws this as a bar and
    /// warns when it sits near full (the render is outrunning the device).
    pub fn fill(&self) -> f32 {
        if self.capacity_frames == 0 {
            0.0
        } else {
            (self.queued_frames as f64 / self.capacity_frames as f64) as f32
        }
    }
}

/// Sample tap that forwards rendered blocks to a [`Monitor`].
pub struct MonitorTap {
    feed: Arc<Feed>,
    meters: Arc<Meters>,
    sample_rate: f32,
}

impl SampleTap for MonitorTap {
    fn tap(&self, planar: &[Option<&[f64]>; crate::dsp::SLOTS], l: &[f64], r: &[f64], n: usize) {
        // The same ten meters the plugin exposed, fed from the same block the
        // encoder would have received. Block sizes vary here, so the release
        // constant is recomputed per block: exp(-n / (0.30 * sr)) stays exact.
        self.meters.set_format(self.sample_rate, n);
        self.meters.observe_block(planar, l, r, n);

        let mut fl = Vec::with_capacity(n);
        let mut fr = Vec::with_capacity(n);
        for i in 0..n {
            fl.push(clamp1(l[i] as f32));
            fr.push(clamp1(r[i] as f32));
        }
        self.feed.push(&fl, &fr);
    }
}

/// Keep the device out of wrap-around territory: integer backends convert
/// out-of-range floats by truncating, which turns a hot downmix into noise.
fn clamp1(v: f32) -> f32 {
    if v > 1.0 {
        1.0
    } else if v < -1.0 {
        -1.0
    } else if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Build the output stream for whatever sample type the device wants.
fn build_stream(
    device: &cpal::Device,
    config: StreamConfig,
    format: SampleFormat,
    feed: Arc<Feed>,
) -> Result<Stream> {
    let err = |ctx: &'static str| {
        move |e: cpal::Error| {
            // An xrun here is the same event the feed already counted; anything else
            // is worth a line in the log.
            eprintln!("monitor {ctx}: {e}");
        }
    };
    match format {
        SampleFormat::F32 => open::<f32>(device, config, feed, err("f32")),
        SampleFormat::F64 => open::<f64>(device, config, feed, err("f64")),
        SampleFormat::I16 => open::<i16>(device, config, feed, err("s16")),
        SampleFormat::I32 => open::<i32>(device, config, feed, err("s32")),
        SampleFormat::U8 => open::<u8>(device, config, feed, err("u8")),
        other => Err(Error::Device(format!(
            "unsupported device sample format {other:?}; this build handles f32, f64, s16, s32, u8"
        ))),
    }
}

fn open<T>(
    device: &cpal::Device,
    config: StreamConfig,
    feed: Arc<Feed>,
    err_fn: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let ch = config.channels.max(1) as usize;
    // Only the first two channels of the device are fed from the queue; the rest
    // are duplicates, so a 5.1 or 7.1 stream does not play silence on the rear.
    let pairs = ch.min(2);
    let mut scratch: Vec<f32> = Vec::new();
    let callback = move |data: &mut [T], _info: &OutputCallbackInfo| {
        let frames = data.len() / ch;
        scratch.clear();
        scratch.resize(frames * ch, 0.0);
        feed.drain(&mut scratch[..frames * pairs]);
        for (i, frame) in data.chunks_mut(ch).enumerate() {
            let l = scratch[i * 2];
            let r = scratch[i * 2 + 1];
            for (c, slot) in frame.iter_mut().enumerate() {
                let src = if pairs == 1 {
                    // Mono device: fold both ears rather than dropping one.
                    (l + r) * 0.5
                } else if c & 1 == 0 {
                    l
                } else {
                    r
                };
                *slot = T::from_sample(src);
            }
        }
    };
    device
        .build_output_stream(config, callback, err_fn, None)
        .map_err(|e| Error::Device(format!("open stream: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feed_paces_the_writer_and_never_drops_within_capacity() {
        let f = Feed::new(100);
        let l = vec![0.25f32; 40];
        let r = vec![-0.25f32; 40];
        assert!(f.push(&l, &r));
        assert_eq!(f.queued_frames(), 40);
        // Second block fits (80 <= 100); a third would have to wait, so drain.
        assert!(f.push(&l, &r));
        assert_eq!(f.queued_frames(), 80);
        // Ask for 100 frames when only 80 are queued: everything queued must be
        // played, in order, and only the tail may be silence.
        let mut out = vec![7.0f32; 200];
        f.drain(&mut out);
        assert_eq!(f.queued_frames(), 0, "nothing may be left behind");
        assert_eq!(f.played_frames.load(Ordering::Relaxed), 80);
        assert_eq!(&out[..4], &[0.25, -0.25, 0.25, -0.25][..], "stereo order");
        assert_eq!(&out[158..160], &[0.25, -0.25][..], "last queued frame");
        assert!(
            out[160..].iter().all(|&v| v == 0.0),
            "the underrun tail is silence"
        );
        assert_eq!(f.underruns.load(Ordering::Relaxed), 1);
        assert_eq!(f.silent_frames.load(Ordering::Relaxed), 20);
    }

    #[test]
    fn closing_wakes_a_waiting_writer() {
        let f = Arc::new(Feed::new(64));
        let big = vec![0.1f32; 32];
        assert!(f.push(&big, &big));
        assert!(f.push(&big, &big));
        let t = {
            let f = f.clone();
            std::thread::spawn(move || f.push(&big, &big))
        };
        std::thread::sleep(std::time::Duration::from_millis(60));
        f.close();
        assert!(
            !t.join().unwrap(),
            "a closed feed must refuse samples, not block forever"
        );
    }

    #[test]
    fn drain_without_lock_contention_produces_silence_and_counts() {
        let f = Feed::new(16);
        let mut out = vec![7.0f32; 8];
        f.drain(&mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(f.underruns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn clamp1_tames_hot_and_nonfinite_samples() {
        assert_eq!(clamp1(9.0), 1.0);
        assert_eq!(clamp1(-9.0), -1.0);
        assert_eq!(clamp1(f32::NAN), 0.0);
        assert_eq!(clamp1(f32::INFINITY), 1.0);
    }

    /// Real hardware only when present: proves enumeration works on this box.
    /// Enumeration opens nothing, so it makes no sound.
    #[test]
    fn lists_output_devices_when_audio_is_available() {
        match list_devices() {
            Ok(v) => {
                for d in &v {
                    eprintln!(
                        "{}{}",
                        d.label(),
                        if is_discard(d) {
                            " [swallows samples]"
                        } else {
                            ""
                        }
                    );
                }
            }
            Err(e) => eprintln!("no audio backend here: {e}"),
        }
    }

    #[test]
    fn discard_matching_is_conservative() {
        let named = |name: &str, id: &str| DeviceInfo {
            is_default: false,
            name: name.into(),
            id: id.into(),
            sample_rate: 48000,
            channels: 2,
            format: "f32".into(),
        };
        assert!(is_discard(&named(
            "Discard all samples (playback)",
            "alsa:null"
        )));
        assert!(
            is_discard(&named("auto_null", "pipewire:0")),
            "PipeWire sink"
        );
        assert!(is_discard(&named("BlackHole 2ch", "coreaudio:1")));
        // Anything that could be speakers must NOT match.
        assert!(!is_discard(&named(
            "HD-Audio Generic, ALCS1200A Analog",
            "alsa:sysdefault"
        )));
        assert!(!is_discard(&named("WH-1000XM5", "bluez:1")), "headphones");
        assert!(!is_discard(&named("", "")), "unknown is treated as audible");
    }

    /// The rule that keeps a test suite off somebody's headphones: audio output is
    /// off unless the operator asked for it, and CI never asks.
    #[test]
    fn audio_output_is_off_unless_explicitly_allowed() {
        for k in ["G6ME_ALLOW_AUDIO_OUTPUT", "G6ME_AUDIO_TEST"] {
            std::env::remove_var(k);
        }
        assert!(!audio_output_allowed());
        // A value that is not an affirmative must not enable it either.
        std::env::set_var("G6ME_ALLOW_AUDIO_OUTPUT", "0");
        assert!(!audio_output_allowed());
        std::env::set_var("G6ME_ALLOW_AUDIO_OUTPUT", "1");
        assert!(audio_output_allowed());
        std::env::remove_var("G6ME_ALLOW_AUDIO_OUTPUT");
    }

    /// `open_quiet` must only ever land on a sink that swallows samples — and if
    /// there is none, it must refuse instead of picking the default device.
    #[test]
    fn quiet_monitor_never_opens_an_audible_device() {
        let Some((_, info)) = quiet_device() else {
            eprintln!("skipping: no discard device on this machine");
            return;
        };
        assert!(is_discard(&info), "quiet_device picked {}", info.name);
        assert!(
            !info.is_default || is_discard(&info),
            "the default device is not a sink: {}",
            info.name
        );
        let m = Monitor::open_quiet(1024, Arc::new(Meters::new()))
            .expect("a discard device should open");
        assert!(is_discard(m.device()));
        m.stop();
    }

    // The end-to-end live render (real clock, real tap) lives in `pipeline`'s tests,
    // where the render fixtures are, as
    // `pipeline::monitor_plays_a_render_to_a_discard_device`. It is not duplicated
    // here, and nothing in this module opens a device except via `open_quiet`.
}
