//! Background work: the batch render.
//!
//! It runs on its own thread and talks to the UI through a channel plus a
//! `request_repaint`. Nothing here touches egui state directly, because an
//! immediate-mode UI rebuilt every frame cannot be mutated from another thread —
//! and because a render that blocks the UI thread freezes the window, which on
//! Windows looks exactly like a crash.
//!
//! Levels reach the window the same way: `BatchCfg.taps` carries a factory that
//! gives each file its own meter, so the numbers on screen are the ones the
//! encoder saw. Nothing in this file opens an audio device.

use g6me_core::batch::{self, Batch, BatchCfg, Counters, Item, Outcome, Phase};
use g6me_core::pipeline::Report;
use g6me_core::probe::{Caps, Tools};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Severity, so the log can colour and count problems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}

/// What the UI is told happened.
#[derive(Clone, Debug)]
pub enum Event {
    /// A queued file reached a terminal state.
    ///
    /// `report` rides along with the success so the meter history can be written down
    /// while the numbers are fresh. The live bars decay to rest in a third of a second —
    /// correct for a bar, useless as a record — so without this the peak of a finished
    /// file exists nowhere after the run.
    ItemState {
        index: usize,
        phase: Phase,
        note: String,
        report: Option<Box<Report>>,
    },
    /// The batch finished; the UI stops showing a spinner.
    Finished(Summary),
    /// The startup capability probe finished. It runs on its own thread because it
    /// launches ffmpeg several times, and a window that appears a second later
    /// beats a window that appears frozen.
    Caps(Caps, String),
}

/// Final tally of one batch, in the shape the status bar wants.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub done: usize,
    pub failed: usize,
    pub skipped: usize,
    pub cancelled: bool,
    pub seconds: f64,
    pub bytes: u64,
    pub error: Option<String>,
}

impl Summary {
    pub fn from_outcome(o: &Outcome, wall: Duration) -> Self {
        Self {
            done: o.done,
            failed: o.failed,
            skipped: o.skipped,
            cancelled: o.cancelled,
            seconds: wall.as_secs_f64(),
            bytes: o
                .items
                .iter()
                .filter_map(|i| i.report.as_ref())
                .map(|r| r.output_bytes)
                .sum(),
            error: None,
        }
    }

    pub fn one_line(&self) -> String {
        if let Some(e) = &self.error {
            return format!("run refused: {e}");
        }
        format!(
            "{} ok, {} failed, {} skipped in {} → {} of output{}",
            self.done,
            self.failed,
            self.skipped,
            g6me_core::util::human_time(self.seconds),
            g6me_core::util::human_bytes(self.bytes),
            if self.cancelled { " (cancelled)" } else { "" }
        )
    }
}

/// Owns the render thread and the channel it reports on.
pub struct Runner {
    tx: mpsc::Sender<Event>,
    rx: Arc<Mutex<mpsc::Receiver<Event>>>,
    /// Shared with every stage of every render, so one press stops the whole run.
    cancel: Arc<AtomicBool>,
    counters: Arc<Mutex<Option<Arc<Counters>>>>,
    running: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    ctx: Arc<Mutex<Option<egui::Context>>>,
    started: Mutex<Option<Instant>>,
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}

impl Runner {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            cancel: Arc::new(AtomicBool::new(false)),
            counters: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            worker: Mutex::new(None),
            ctx: Arc::new(Mutex::new(None)),
            started: Mutex::new(None),
        }
    }

    /// The sending half of the event channel, for the few things that report from
    /// outside the batch workers: the startup probe and the monitor thread.
    pub fn sender(&self) -> mpsc::Sender<Event> {
        self.tx.clone()
    }

    /// Wall time since the current run started, for the ETA.
    pub fn elapsed(&self) -> Option<Duration> {
        self.started
            .lock()
            .ok()
            .and_then(|s| s.map(|t| t.elapsed()))
    }

    /// Remember the context so worker threads can ask for a repaint. Without this
    /// an egui window only updates when the mouse moves over it, and a progress bar
    /// that updates on mouse movement is not a progress bar.
    pub fn set_context(&self, ctx: egui::Context) {
        if let Ok(mut c) = self.ctx.lock() {
            *c = Some(ctx);
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Live counters, for the progress bar and the ETA.
    pub fn counters(&self) -> Option<g6me_core::batch::CountersView> {
        self.counters
            .lock()
            .ok()
            .and_then(|c| c.as_ref().map(|c| c.snapshot()))
    }

    /// Start a batch. Fails fast on an unusable queue rather than half-running it.
    ///
    /// Hands back the exact item list the workers were given, so the table the user
    /// sees and the queue being rendered cannot drift apart — progress events carry
    /// an index into this list.
    pub fn start(
        &self,
        cfg: &BatchCfg,
        tools: Tools,
        caps: Caps,
    ) -> std::result::Result<Vec<Item>, String> {
        if self.is_running() {
            return Err("a run is already in progress".into());
        }
        let ext = cfg.render.out.format.extension().to_string();
        let items = batch::discover(&cfg.inputs, cfg.out_dir.as_deref(), &ext, cfg.recursive);
        if items.is_empty() {
            return Err(
                "no audio files found — check the queue, the extension filter, and whether \
                 the folder is readable"
                    .into(),
            );
        }
        let list = items.clone();
        if let Ok(mut s) = self.started.lock() {
            *s = Some(Instant::now());
        }
        self.cancel.store(false, Ordering::Relaxed);
        self.running.store(true, Ordering::Relaxed);

        let tx = self.tx.clone();
        // An Arc rather than a closure, because both the progress hook and the worker
        // hold one. `set_context` ran at startup, so the slot is filled by now and the
        // closures outlive any single frame.
        let wake: Arc<dyn Fn() + Send + Sync> = {
            let slot = self.ctx.clone();
            Arc::new(move || {
                if let Ok(c) = slot.lock() {
                    if let Some(ctx) = c.as_ref() {
                        ctx.request_repaint();
                    }
                }
            })
        };
        // Cheap: runs on a worker thread between files, so only a channel send and
        // a repaint request.
        let wake_progress = wake.clone();
        let progress: batch::Progress = Arc::new(move |index: usize, item: &Item| {
            let note = match &item.error {
                Some(e) => e.clone(),
                None => item
                    .report
                    .as_ref()
                    .map(|r| {
                        format!(
                            "{} frames · {}",
                            r.frames,
                            g6me_core::util::human_bytes(r.output_bytes)
                        )
                    })
                    .unwrap_or_default(),
            };
            let _ = tx.send(Event::ItemState {
                index,
                phase: item.phase,
                note,
                // Boxed: a Report is a dozen PathBufs, Vecs and strings, and this event
                // is cloned through a channel for every file in a queue of thousands.
                report: item.report.clone().map(Box::new),
            });
            wake_progress();
        });

        let batch = Batch::start(items, cfg, tools, caps, Some(progress));
        if let Ok(mut c) = self.counters.lock() {
            *c = Some(batch.counters.clone());
        }
        let running = self.running.clone();
        let tx2 = self.tx.clone();
        let h = std::thread::spawn(move || {
            let started = Instant::now();
            let outcome = batch.finish();
            let mut s = Summary::from_outcome(&outcome, started.elapsed());
            if outcome.cancelled {
                s.error = None;
            }
            let _ = tx2.send(Event::Finished(s));
            running.store(false, Ordering::Relaxed);
            wake();
        });
        if let Ok(mut w) = self.worker.lock() {
            *w = Some(h);
        }
        Ok(list)
    }

    /// Ask for a stop. The current file finishes its current block, removes its
    /// partial output, and the queue is abandoned.
    pub fn cancel(&self) {
        if self.is_running() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Collect and clear pending events. Called once per frame.
    pub fn drain(&self) -> Vec<Event> {
        let mut out = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(e) = rx.try_recv() {
                out.push(e);
            }
        }
        out
    }
}
