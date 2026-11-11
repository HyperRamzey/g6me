//! Batch engine: many files and folders, one output per input.
//!
//! ## Threading model
//!
//! The downmixer is a cascade of recursive all-pass sections, so its state is a
//! sequence: split one file across threads and you have to reconstruct that
//! sequence across a boundary, which is exactly what changes phase behaviour. So
//! **DSP stays single-threaded per file** — one [`crate::dsp::Dsp`] owned by one
//! worker for the whole file — and the parallelism lives *between* files: N
//! workers, each rendering a different file, each with its own two ffmpeg
//! children.
//!
//! Workers pull from a shared cursor rather than a queue. There is nothing to
//! co-ordinate beyond "give me the next index", so a slow 3-hour Atmos file
//! cannot block the four short files behind it.
//!
//! ## Failures
//!
//! A file that cannot be decoded, or whose layout cannot be determined, is
//! recorded and skipped; the queue continues. Only cancellation stops the batch,
//! and it stops between blocks rather than mid-write, so a cancelled run leaves
//! no half-written output behind ([`crate::pipeline`] deletes it).

use crate::error::Error;
use crate::pcm::BitDepth;
use crate::pipeline::{render_file, RenderOpts, Report, SampleTap, Sink};
use crate::probe::{Caps, Tools};
use crate::stages::{OutFormat, RenderCfg};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Containers we will attempt, chosen by extension because deciding by probing
/// every file in a 40 000-file library would take longer than rendering it.
/// Anything ffmpeg can decode is accepted at render time regardless.
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mka", "mkv", "webm", "m4a", "m4b", "mp4", "mov", "m4v", "flac", "wav", "wave", "rf64", "aiff",
    "aif", "aifc", "afc", "w64", "caf", "wma", "ape", "mla", "tta", "wv", "tak", "oga", "ogg",
    "opus", "spx", "ac3", "eac3", "dts", "dtshd", "dff", "dsf", "aac", "mp3", "mp2", "m4r", "amr",
    "3gp", "3g2", "voc", "au", "snd", "raw",
];

/// Files that are definitely not audio, so a folder scan does not offer them.
pub const SKIP_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "nfo", "txt", "md", "cue", "log", "srt", "ass",
    "ssa", "sub", "pdf", "zip", "rar", "7z", "exe", "dll", "json", "xml", "yml", "yaml", "csv",
    "dat", "jpg", "cover", "jpeg",
];

/// What one file in the queue is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Queued,
    Running,
    Done,
    Failed,
    /// Left alone on purpose (output exists and `--force` was not given).
    Skipped,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Queued => "queued",
            Phase::Running => "running",
            Phase::Done => "done",
            Phase::Failed => "failed",
            Phase::Skipped => "skipped",
        }
    }
    pub fn is_terminal(self) -> bool {
        matches!(self, Phase::Done | Phase::Failed | Phase::Skipped)
    }
}

/// One queued file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub src: PathBuf,
    pub dst: PathBuf,
    pub phase: Phase,
    /// Frames rendered so far, for a progress bar that moves mid-file.
    pub frames: u64,
    pub error: Option<String>,
    pub report: Option<Report>,
}

impl Item {
    pub fn new(src: PathBuf, dst: PathBuf) -> Self {
        Self {
            src,
            dst,
            phase: Phase::Queued,
            frames: 0,
            error: None,
            report: None,
        }
    }

    pub fn seconds(&self) -> f64 {
        self.report.as_ref().map(|r| r.seconds).unwrap_or(0.0)
    }
}

/// Makes the meter tap for one file, given that file's index in the queue.
///
/// Per file on purpose. One meter shared by four workers would display a level
/// that belongs to none of them — the peak would be the loudest of four different
/// pieces of music, which is not a measurement of anything.
pub type TapFor = dyn Fn(usize) -> Option<Arc<dyn SampleTap>> + Send + Sync;

/// [`TapFor`] wrapped so [`BatchCfg`] can keep deriving `Debug`: a closure has
/// nothing honest to print, so it says whether one is there and stops.
#[derive(Clone, Default)]
pub struct MeterFactory(pub Option<std::sync::Arc<TapFor>>);

impl std::fmt::Debug for MeterFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match &self.0 {
            Some(_) => "MeterFactory(set)",
            None => "MeterFactory(none)",
        })
    }
}

/// Inputs to one batch run.
#[derive(Clone, Debug)]
pub struct BatchCfg {
    /// Folders and files, in the order given.
    pub inputs: Vec<PathBuf>,
    /// `None` writes each output next to its input.
    pub out_dir: Option<PathBuf>,
    /// Scan folders recursively (default, and what a music library expects).
    pub recursive: bool,
    /// How many files to render at once. DSP stays single-threaded per file.
    pub workers: usize,
    pub render: RenderCfg,
    /// Refuse to start if nothing was found (instead of a silent no-op).
    pub fail_on_empty: bool,
    /// Live levels for each file as it renders. `None` renders blind, which is
    /// what the CLI does.
    pub taps: MeterFactory,
}

impl Default for BatchCfg {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            out_dir: None,
            recursive: true,
            workers: default_workers(),
            render: RenderCfg::with_defaults(),
            fail_on_empty: false,
            taps: MeterFactory::default(),
        }
    }
}

/// Sensible default: enough workers to keep two ffmpeg children plus the DSP
/// busy per file, without pretending a hard drive scales linearly.
pub fn default_workers() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    // Each worker is a decoder + DSP + encoder; four of those saturate a laptop,
    // and a 64-core box gains nothing past that (the DSP is the bottleneck).
    cores.clamp(1, 4)
}

/// Extension of a path, lowercased, or "" for no extension.
fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default()
}

/// True when this looks like an audio file we should offer.
pub fn looks_like_audio(p: &Path) -> bool {
    let e = ext_of(p);
    if e.is_empty() {
        return false;
    }
    if SKIP_EXTENSIONS.contains(&e.as_str()) {
        return false;
    }
    AUDIO_EXTENSIONS.contains(&e.as_str())
}

/// Where a file's output goes.
///
/// With an output directory the input tree is mirrored under it, so a batch over
/// `~/Music` lands as `~/Music-flac/Artist/Album/track.flac` instead of 40 000
/// files in one heap.
pub fn output_for(src: &Path, base: &Path, out_dir: Option<&Path>, ext: &str) -> PathBuf {
    let stem = src
        .file_name()
        .map(|s| {
            let s = s.to_string_lossy().to_string();
            match s.rfind('.') {
                Some(i) if i > 0 => s[..i].to_string(),
                _ => s,
            }
        })
        .unwrap_or_else(|| "output".to_string());
    let root = out_dir.unwrap_or_else(|| src.parent().unwrap_or(Path::new(".")));
    let subdir = match out_dir {
        Some(_) => relative_within(src.parent().unwrap_or(Path::new(".")), base)
            .unwrap_or(Path::new(""))
            .to_path_buf(),
        None => PathBuf::new(),
    };
    let mut out = root.to_path_buf();
    out.push(subdir);
    out.push(format!("{stem}.{ext}"));
    out
}

/// `child` inside `root`, or None when it is not below it.
fn relative_within<'a>(child: &'a Path, root: &Path) -> Option<&'a Path> {
    let (c, r) = (child.as_os_str(), root.as_os_str());
    if c == r {
        return Some(Path::new(""));
    }
    let mut cpath = child;
    if cpath.starts_with(root) {
        cpath = cpath.strip_prefix(root).unwrap_or(Path::new(""));
        Some(cpath)
    } else {
        None
    }
}

/// Expand files and folders into a de-duplicated queue.
///
/// `base` is the folder whose structure is mirrored into `out_dir` — the first
/// folder given on the command line, which is what a user means by "this tree".
pub fn discover(inputs: &[PathBuf], out_dir: Option<&Path>, ext: &str) -> Vec<Item> {
    let mut seen: Vec<PathBuf> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    // The mirror root is the first directory argument.
    let base = inputs
        .iter()
        .find(|p| p.is_dir())
        .cloned()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let mut stack: Vec<PathBuf> = Vec::new();
    for input in inputs {
        if input.is_dir() {
            stack.push(input.clone());
        } else if input.is_file() {
            push_file(input, &base, out_dir, ext, &mut seen, &mut items);
        }
    }
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // unreadable directory: skip it, do not fail the batch
        };
        let mut names: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        // Deterministic order, so two runs of the same batch queue identically.
        names.sort();
        for p in names {
            if p.is_dir() {
                stack.push(p);
            } else if looks_like_audio(&p) {
                push_file(&p, &base, out_dir, ext, &mut seen, &mut items);
            }
        }
    }
    items
}

fn push_file(
    src: &Path,
    base: &Path,
    out_dir: Option<&Path>,
    ext: &str,
    seen: &mut Vec<PathBuf>,
    items: &mut Vec<Item>,
) {
    let canon = std::fs::canonicalize(src).unwrap_or_else(|_| src.to_path_buf());
    if seen.iter().any(|s| s == &canon) {
        return; // the same album reached through two arguments or a symlink
    }
    seen.push(canon);
    let dst = output_for(src, base, out_dir, ext);
    items.push(Item::new(src.to_path_buf(), dst));
}

/// Live counters, read by the UI thread while workers render.
#[derive(Default)]
pub struct Counters {
    pub total: usize,
    pub done: AtomicUsize,
    pub failed: AtomicUsize,
    pub skipped: AtomicUsize,
    pub running: AtomicUsize,
    pub frames: AtomicU64,
    pub bytes: AtomicU64,
    pub seconds: AtomicU64,
}

impl Counters {
    /// Seconds of audio completed, for an ETA.
    pub fn seconds_done(&self) -> f64 {
        f64::from_bits(self.seconds.load(Ordering::Relaxed))
    }

    pub fn set_seconds_done(&self, v: f64) {
        self.seconds.store(v.to_bits(), Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CountersView {
        CountersView {
            total: self.total,
            done: self.done.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            running: self.running.load(Ordering::Relaxed),
            frames: self.frames.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            seconds_done: self.seconds_done(),
        }
    }
}

/// Serializable view of [`Counters`].
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct CountersView {
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
    pub running: usize,
    pub frames: u64,
    pub bytes: u64,
    pub seconds_done: f64,
}

impl CountersView {
    /// Finished items over total, 0..=1.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            ((self.done + self.failed + self.skipped) as f64 / self.total as f64) as f32
        }
    }

    /// Files still to finish.
    pub fn remaining(&self) -> usize {
        self.total
            .saturating_sub(self.done + self.failed + self.skipped)
    }

    /// Wall-clock estimate from the average rendered-second-per-second-of-wall
    /// time observed so far. `None` until there is a completed file to learn from.
    pub fn eta(&self, elapsed: Duration, per_item_seconds: f64) -> Option<Duration> {
        let finished = self.done + self.failed;
        if finished == 0 || self.seconds_done <= 0.0 || elapsed.is_zero() {
            return None;
        }
        let rate = self.seconds_done / elapsed.as_secs_f64();
        if rate <= 0.0 {
            return None;
        }
        let left = self.remaining() as f64 * per_item_seconds.max(1.0);
        Some(Duration::from_secs_f64(
            (left / rate).clamp(0.0, 86_400.0 * 7.0),
        ))
    }
}

/// Bump the shared counters from inside a render.
struct Feed {
    frames: Arc<AtomicU64>,
    bytes: Arc<AtomicU64>,
}

impl Sink for Feed {
    fn update(&self, frames: u64, bytes_out: u64) {
        self.frames.store(frames, Ordering::Relaxed);
        self.bytes.store(bytes_out, Ordering::Relaxed);
    }
}

/// Outcome of a whole batch.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
    pub cancelled: bool,
    pub elapsed: Duration,
    pub items: Vec<Item>,
}

impl Outcome {
    pub fn one_line(&self) -> String {
        format!(
            "{} ok, {} failed, {} skipped in {:.1}s",
            self.done,
            self.failed,
            self.skipped,
            self.elapsed.as_secs_f64()
        )
    }

    /// Non-zero exit code for the CLI when something failed.
    pub fn exit_code(&self) -> i32 {
        if self.cancelled {
            130
        } else if self.failed > 0 {
            1
        } else {
            0
        }
    }
}

/// A running batch, shared with the caller so it can show progress and cancel.
pub struct Batch {
    pub items: Arc<Mutex<Vec<Item>>>,
    pub counters: Arc<Counters>,
    pub cancel: Arc<AtomicBool>,
    started: Instant,
    handles: Vec<std::thread::JoinHandle<()>>,
}

/// Called after each item reaches a terminal state. Must be cheap: it runs on a
/// worker thread between files.
pub type Progress = Arc<dyn Fn(usize, &Item) + Send + Sync>;

impl Drop for Batch {
    fn drop(&mut self) {
        // A batch dropped without `finish` must not keep rendering files nobody is
        // waiting for. The workers own their own clones, so signalling is all this
        // can safely do from a destructor.
        self.cancel();
    }
}

impl Batch {
    /// Queue and start rendering. Returns as soon as the workers are running.
    pub fn start(
        items: Vec<Item>,
        cfg: &BatchCfg,
        tools: Tools,
        caps: Caps,
        progress: Option<Progress>,
    ) -> Self {
        let total = items.len();
        let items = Arc::new(Mutex::new(items));
        let counters = Arc::new(Counters {
            total,
            ..Counters::default()
        });
        let cancel = Arc::new(AtomicBool::new(false));
        let cursor = Arc::new(AtomicUsize::new(0));
        let workers = cfg.workers.clamp(1, 64).min(total.max(1));

        // One ffmpeg child pair per worker, and each ffmpeg is told not to slice
        // its own filtergraph across cores: N workers x M filter threads is how a
        // batch ends up slower than a single thread.
        let mut cfg = cfg.clone();
        if cfg.render.out.filter_threads.is_none() && workers > 1 {
            cfg.render.out.filter_threads = Some(1);
        }

        let mut handles = Vec::with_capacity(workers);
        for id in 0..workers {
            let items = items.clone();
            let counters = counters.clone();
            let cancel_w = cancel.clone();
            let cursor = cursor.clone();
            let tools = tools.clone();
            let caps = caps.clone();
            let cfg = cfg.clone();
            let progress = progress.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("g6me-worker-{id}"))
                .spawn(move || {
                    loop {
                        if cancel_w.load(Ordering::Relaxed) {
                            break;
                        }
                        let idx = cursor.fetch_add(1, Ordering::Relaxed);
                        if idx >= total {
                            break;
                        }
                        let (src, dst) = {
                            let guard = items.lock().unwrap_or_else(|e| e.into_inner());
                            (guard[idx].src.clone(), guard[idx].dst.clone())
                        };
                        {
                            let mut guard = items.lock().unwrap_or_else(|e| e.into_inner());
                            guard[idx].phase = Phase::Running;
                        }
                        counters.running.fetch_add(1, Ordering::Relaxed);

                        let frames = Arc::new(AtomicU64::new(0));
                        let bytes = Arc::new(AtomicU64::new(0));
                        let feed = Arc::new(Feed {
                            frames: frames.clone(),
                            bytes: bytes.clone(),
                        });
                        // Fresh tap per file: its meter belongs to this file alone.
                        let tap = cfg.taps.0.as_ref().and_then(|make| make(idx));
                        let opts = RenderOpts {
                            tools: &tools,
                            caps: &caps,
                            cancel: Some(cancel_w.clone()),
                            progress: Some(feed),
                            tap,
                        };
                        let result = render_file(&src, &dst, &cfg.render, &opts);

                        counters.running.fetch_sub(1, Ordering::Relaxed);
                        let mut guard = items.lock().unwrap_or_else(|e| e.into_inner());
                        let item = &mut guard[idx];
                        item.frames = frames.load(Ordering::Relaxed);
                        match result {
                            Ok(report) => {
                                counters.done.fetch_add(1, Ordering::Relaxed);
                                counters.frames.fetch_add(report.frames, Ordering::Relaxed);
                                counters
                                    .bytes
                                    .fetch_add(report.output_bytes, Ordering::Relaxed);
                                counters.set_seconds_done(counters.seconds_done() + report.seconds);
                                item.phase = Phase::Done;
                                item.report = Some(report);
                                item.error = None;
                            }
                            Err(e) => {
                                // An existing output is a decision, not a fault.
                                if matches!(e, Error::Exists(_)) {
                                    counters.skipped.fetch_add(1, Ordering::Relaxed);
                                    item.phase = Phase::Skipped;
                                } else {
                                    counters.failed.fetch_add(1, Ordering::Relaxed);
                                    item.phase = Phase::Failed;
                                }
                                item.error = Some(e.to_string());
                                item.report = None;
                            }
                        }
                        let done = item.clone();
                        drop(guard);
                        if let Some(p) = &progress {
                            p(idx, &done);
                        }
                    }
                })
                .map(|h| handles.push(h));
            if let Err(e) = spawned {
                // Better to stop than to quietly render on fewer threads than the
                // user asked for.
                eprintln!("worker {id} could not start: {e}");
                cancel.store(true, Ordering::Relaxed);
                break;
            }
        }

        Self {
            items,
            counters,
            cancel,
            started: Instant::now(),
            handles,
        }
    }

    /// Ask every worker to stop at the next item boundary, and every in-flight
    /// render to stop at the next block.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Copy of the queue for display.
    pub fn snapshot(&self) -> Vec<Item> {
        self.items.lock().map(|i| i.clone()).unwrap_or_default()
    }

    /// Join the workers and produce the final result.
    ///
    /// Workers retire by themselves once the cursor passes the end of the queue,
    /// so this does *not* cancel: cancelling here would abandon whatever is still
    /// queued and report a clean run that rendered almost nothing.
    pub fn finish(mut self) -> Outcome {
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        let view = self.counters.snapshot();
        let done = view.done;
        Outcome {
            done,
            failed: view.failed,
            skipped: view.skipped,
            cancelled: self.is_cancelled(),
            elapsed: self.started.elapsed(),
            items: self.snapshot(),
        }
    }

    /// Run a batch to completion (the CLI path).
    pub fn run(
        items: Vec<Item>,
        cfg: &BatchCfg,
        tools: Tools,
        caps: Caps,
        progress: Option<Progress>,
    ) -> Outcome {
        if items.is_empty() {
            return Outcome::default();
        }
        Batch::start(items, cfg, tools, caps, progress).finish()
    }
}

/// Choose the encoder from the requested format and depth, so callers do not
/// have to know that 24-bit FLAC needs a different binary.
pub fn encoder_for(fmt: &OutFormat, depth: BitDepth) -> OutFormat {
    match fmt {
        OutFormat::Flac { level } if depth == BitDepth::S24 => OutFormat::Flac { level: *level },
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::probe_caps;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("g6me-batch-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"fake audio").unwrap();
    }

    /// A batch over `dir` that may overwrite and does not copy metadata (the
    /// metadata copy needs a real source file, which several of these tests fake).
    fn failing_cfg(dir: PathBuf, workers: usize) -> BatchCfg {
        BatchCfg {
            inputs: vec![dir],
            workers,
            render: RenderCfg {
                out: crate::stages::OutCfg {
                    overwrite: true,
                    copy_metadata: false,
                    ..Default::default()
                },
                ..RenderCfg::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn discovery_walks_mirrors_and_dedupes() {
        let root = tmp("walk");
        let music = root.join("Music");
        touch(&music.join("Artist/Album/01.mka"));
        touch(&music.join("Artist/Album/02.mka"));
        touch(&music.join("Artist/Album/cover.jpg"));
        touch(&music.join("Artist/Album/notes.txt"));
        touch(&music.join("Artist/Album/03.flac"));
        touch(&root.join("loose.m4a"));

        let out = root.join("out");
        let items = discover(std::slice::from_ref(&music), Some(&out), "flac");
        let names: Vec<String> = items
            .iter()
            .map(|i| {
                i.dst
                    .strip_prefix(&out)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(items.len(), 3, "cover/txt must not appear, got {names:?}");
        assert!(
            names.iter().any(|n| n.ends_with("Artist/Album/01.flac")),
            "tree must be mirrored, got {names:?}"
        );

        // Passing the same folder twice must not queue anything twice.
        let twice = discover(&[music.clone(), music], Some(&out), "flac");
        assert_eq!(twice.len(), 3);

        // Loose file argument: output goes next to it.
        let loose = discover(&[root.join("loose.m4a")], None, "flac");
        assert_eq!(loose.len(), 1);
        assert_eq!(
            loose[0].dst.file_name().unwrap().to_string_lossy(),
            "loose.flac"
        );
        assert_eq!(loose[0].dst.parent(), Some(root.as_path()));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn extension_filters_are_exclusive() {
        assert!(looks_like_audio(Path::new("a.mka")));
        assert!(looks_like_audio(Path::new("a.DSF")));
        assert!(!looks_like_audio(Path::new("cover.JPG")));
        assert!(!looks_like_audio(Path::new("notes.txt")));
        assert!(!looks_like_audio(Path::new("noextension")));
    }

    #[test]
    fn counters_view_math() {
        let c = Counters::default();
        let mut v = c.snapshot();
        assert_eq!(v.fraction(), 0.0);
        assert_eq!(v.remaining(), 0);
        v.total = 10;
        v.done = 4;
        v.failed = 1;
        assert!((v.fraction() - 0.5).abs() < 1e-6);
        assert_eq!(v.remaining(), 5);
        // An ETA needs observed throughput, not just a count of finished files.
        assert!(v.eta(Duration::from_secs(10), 60.0).is_none());
        v.seconds_done = 300.0;
        let eta = v
            .eta(Duration::from_secs(10), 60.0)
            .expect("throughput known");
        // 300 s of audio in 10 s of wall time = 30x; 5 files x 60 s = 300 s left.
        assert!((eta.as_secs_f64() - 10.0).abs() < 0.5, "got {eta:?}");
        assert!(CountersView::default()
            .eta(Duration::from_secs(10), 60.0)
            .is_none());
    }

    #[test]
    fn workers_stay_within_reason() {
        let w = default_workers();
        assert!((1..=4).contains(&w), "got {w}");
    }

    /// The real thing: three generated files, two workers, FLAC out.
    #[test]
    fn batch_renders_a_folder_and_reports_each_file() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("run");
        let src_dir = dir.join("in");
        std::fs::create_dir_all(&src_dir).unwrap();
        for i in 0..3 {
            let p = src_dir.join(format!("t{i}.wav"));
            crate::wav::write_pcm_s32(&p, 48000, 8, &vec![(i as u8) + 1; (48000 / 8) * 8 * 4])
                .unwrap();
        }
        let out = dir.join("out");
        let mut cfg = failing_cfg(src_dir, 2);
        cfg.out_dir = Some(out.clone());

        let items = discover(&cfg.inputs, cfg.out_dir.as_deref(), "flac");
        assert_eq!(items.len(), 3);
        let done = Arc::new(AtomicUsize::new(0));
        let sink = {
            let done = done.clone();
            Arc::new(move |_i: usize, it: &Item| {
                assert_eq!(
                    it.phase,
                    Phase::Done,
                    "{}: {:?}",
                    it.src.display(),
                    it.error
                );
                done.fetch_add(1, Ordering::Relaxed);
            })
        } as Progress;
        let o = Batch::run(items, &cfg, tools.clone(), caps.clone(), Some(sink));
        eprintln!("batch: {}", o.one_line());
        assert_eq!(
            o.done,
            3,
            "{:?}",
            o.items
                .iter()
                .map(|i| (&i.dst, &i.error))
                .collect::<Vec<_>>()
        );
        assert_eq!(o.failed, 0);
        assert_eq!(done.load(Ordering::Relaxed), 3);
        for i in 0..3 {
            assert!(out.join(format!("t{i}.flac")).is_file());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A corrupt file must fail alone; the queue keeps going.
    #[test]
    fn one_bad_file_does_not_stop_the_batch() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("bad");
        std::fs::write(
            dir.join("broken.mka"),
            b"this is not a matroska file at all",
        )
        .unwrap();
        crate::wav::write_pcm_s32(&dir.join("good.wav"), 48000, 2, &vec![0u8; 4800]).unwrap();
        let cfg = failing_cfg(dir.clone(), 2);
        let items = discover(&cfg.inputs, None, "flac");
        assert_eq!(items.len(), 2);
        let o = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(
            o.done,
            1,
            "the good file must still render: {:?}",
            o.items
                .iter()
                .map(|i| (&i.src, &i.phase, &i.error))
                .collect::<Vec<_>>()
        );
        assert_eq!(o.failed, 1);
        assert_eq!(o.exit_code(), 1);
        assert!(dir.join("good.flac").is_file());
        assert!(
            !dir.join("broken.flac").exists(),
            "no output for a failed file"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn existing_outputs_are_skipped_not_clobbered() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("exists");
        crate::wav::write_pcm_s32(&dir.join("a.wav"), 48000, 2, &vec![0u8; 4800]).unwrap();
        let out = dir.join("a.flac");
        std::fs::write(&out, b"precious").unwrap();
        let cfg = BatchCfg {
            inputs: vec![dir.clone()],
            workers: 1,
            ..Default::default()
        };
        let items = discover(&cfg.inputs, None, "flac");
        let o = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(o.skipped, 1, "{:?}", o.items[0].error);
        assert_eq!(o.done, 0);
        assert_eq!(std::fs::read(&out).unwrap(), b"precious");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cancel_stops_the_batch_and_leaves_no_partial_files() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("cancel");
        for i in 0..12 {
            crate::wav::write_pcm_s32(
                &dir.join(format!("c{i}.wav")),
                48000,
                8,
                &vec![3u8; 48000 * 8 * 4],
            )
            .unwrap();
        }
        let cfg = failing_cfg(dir.clone(), 2);
        let items = discover(&cfg.inputs, None, "flac");
        let b = Batch::start(items, &cfg, tools, caps, None);
        std::thread::sleep(Duration::from_millis(150));
        b.cancel();
        let o = b.finish();
        assert!(o.cancelled);
        assert!(
            o.done < 12,
            "cancelling after 150 ms should not have finished everything"
        );
        // Every file that exists must be a complete render, never a truncated one.
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().map(|e| e == "flac").unwrap_or(false) {
                let meta = std::fs::metadata(&p).unwrap();
                assert!(meta.len() > 100, "suspiciously small output {:?}", p);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_runs_of_the_same_batch_are_identical() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("determinism");
        crate::wav::write_pcm_s32(&dir.join("d.wav"), 44100, 6, &vec![9u8; 44100 * 6 * 4]).unwrap();
        let mut cfg = failing_cfg(dir.clone(), 1);
        cfg.render.sha256 = true;
        let items = discover(&cfg.inputs, None, "flac");
        let first = Batch::run(items.clone(), &cfg, tools.clone(), caps.clone(), None);
        let h1 = first.items[0]
            .report
            .as_ref()
            .and_then(|r| r.sha256.clone())
            .expect("hash requested");
        let second = Batch::run(items, &cfg, tools, caps, None);
        let h2 = second.items[0]
            .report
            .as_ref()
            .and_then(|r| r.sha256.clone())
            .expect("hash requested");
        assert_eq!(h1, h2, "same input and config must give the same bytes");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Meters are per file. A GUI spawns one level strip per rendering file, so the
    /// factory must be called exactly once per file, and each set must have seen
    /// that file's own samples.
    #[test]
    fn one_meter_tap_per_rendering_file() {
        let Some((tools, caps)) = tools_and_caps() else {
            return;
        };
        let dir = tmp("meters");
        for i in 0..3 {
            crate::wav::write_pcm_s32(
                &dir.join(format!("m{i}.wav")),
                48000,
                8,
                &vec![200u8; 48000 * 8 * 4 / 4],
            )
            .unwrap();
        }
        let mut cfg = failing_cfg(dir.clone(), 2);
        cfg.out_dir = Some(dir.join("out"));

        let seen = Arc::new(Mutex::new(Vec::<usize>::new()));
        let meters: Arc<Mutex<Vec<std::sync::Arc<crate::meter::Meters>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let (seen_make, meters_make) = (seen.clone(), meters.clone());
        cfg.taps = MeterFactory(Some(Arc::new(move |idx: usize| {
            seen_make.lock().unwrap().push(idx);
            let m = std::sync::Arc::new(crate::meter::Meters::new());
            meters_make.lock().unwrap().push(m.clone());
            Some(Arc::new(crate::meter::MeterTap::new(m)))
        })));

        let items = discover(&cfg.inputs, cfg.out_dir.as_deref(), "flac");
        assert_eq!(items.len(), 3);
        let outcome = Batch::run(items, &cfg, tools, caps, None);
        assert_eq!(
            outcome.done,
            3,
            "{:?}",
            outcome.items.iter().map(|i| &i.error).collect::<Vec<_>>()
        );

        let calls = seen.lock().unwrap().clone();
        assert_eq!(calls.len(), 3, "one tap per file, got {calls:?}");
        let mut calls = calls;
        calls.sort_unstable();
        assert_eq!(calls, vec![0, 1, 2], "each file got its own tap");
        let ms = meters.lock().unwrap();
        assert_eq!(ms.len(), 3);
        for m in ms.iter() {
            assert!(m.frames() > 0, "a meter that saw no samples is not a meter");
            assert!(
                m.peaks().iter().any(|p| *p > 0.0),
                "the file is a constant tone; something should be lit"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_factory_renders_blind() {
        // The CLI never asks for levels; the default factory must stay empty and
        // must not change what is written.
        let cfg = BatchCfg::default();
        assert!(cfg.taps.0.is_none());
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("MeterFactory(none)"), "{dbg}");
    }

    fn tools_and_caps() -> Option<(Tools, Caps)> {
        let t = Tools::discover();
        if !t.is_usable() {
            eprintln!("skipping: no ffmpeg on PATH");
            return None;
        }
        let caps = probe_caps(&t);
        Some((t, caps))
    }
}
