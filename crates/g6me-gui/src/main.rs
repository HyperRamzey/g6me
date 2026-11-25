//! G6 Media Encoder — desktop front end.
//!
//! Everything here is presentation. The queue, the render, the filters, the
//! meters' release curve all live in `g6me-core`, which is the same library the CLI
//! drives, so a setting that behaves one way in the CLI behaves that way here. Two
//! things in particular are worth stating because they are the reasons this window
//! exists:
//!
//! * **the levels are the render's own.** Each file that starts rendering gets a
//!   meter, and the render feeds it the block it handed the encoder. There is no
//!   second "just measure it" path to drift out of agreement with the real one.
//! * **this window never makes a sound.** It writes files and it measures them. No
//!   audio device is opened anywhere in this crate, so nothing here can put a
//!   downmix into someone's headphones by accident, and the test suite is silent by
//!   construction rather than by convention.
//!
//! The layout is deliberately fixed — queue left, controls centre, meters and
//! progress along the bottom — because a mixer's muscle memory is the point.

// A released Windows GUI must not drag a console window along with it. Debug builds
// keep the console, because a panic that prints nowhere is a worse trade than one
// extra window while developing.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod runner;
mod settings;
mod theme;
mod widgets;

use crate::runner::{Event, Level, PickTarget, Runner, Summary};
use crate::settings::{describe, Format, MasteringPreset, Settings};
use crate::theme::Palette;
use crate::widgets::{Browser, Mode};
use eframe::CreationContext;
use g6me_core::batch::{self, Phase};
use g6me_core::loudness::{Compliance, DeliveryTarget, LoudnessReport};
use g6me_core::mask;
use g6me_core::meter::{MeterTap, Meters};
use g6me_core::pcm::{BitDepth, Dither};
use g6me_core::pipeline;
use g6me_core::pipeline::Report;
use g6me_core::plus::{woodworth_itd_s, PlusCfg, PlusProfile, RendererMode};
use g6me_core::probe::{probe_caps, Caps, Tools};
use g6me_core::stages::Gtype;
use g6me_core::util;
use std::path::PathBuf;
use std::sync::Mutex;

/// Filter for the queue list in the left sidebar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum QueueFilter {
    #[default]
    All,
    Queued,
    Done,
    Failed,
    Skipped,
}

impl QueueFilter {
    const ALL: [Self; 5] = [
        Self::All,
        Self::Queued,
        Self::Done,
        Self::Failed,
        Self::Skipped,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Queued => "queued",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }

    fn matches(self, phase: Phase) -> bool {
        match self {
            Self::All => true,
            Self::Queued => matches!(phase, Phase::Queued | Phase::Running),
            Self::Done => phase == Phase::Done,
            Self::Failed => phase == Phase::Failed,
            Self::Skipped => phase == Phase::Skipped,
        }
    }
}

fn row_matches_filter(row: &Row, filter: QueueFilter, query: &str) -> bool {
    if !filter.matches(row.phase) {
        return false;
    }
    let q = query.trim();
    if q.is_empty() {
        return true;
    }
    let q_low = q.to_lowercase();
    let name = row
        .src
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| row.src.display().to_string().to_lowercase());
    name.contains(&q_low) || row.shout().to_lowercase().contains(&q_low)
}

/// Which settings page is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Renderer,
    Output,
    TruePeak,
    Resample,
    Ir,
    Advanced,
}

impl Tab {
    const ALL: [Tab; 6] = [
        Tab::Renderer,
        Tab::Output,
        Tab::TruePeak,
        Tab::Resample,
        Tab::Ir,
        Tab::Advanced,
    ];

    fn label(self) -> &'static str {
        match self {
            Tab::Renderer => "Renderer",
            Tab::Output => "Output",
            Tab::TruePeak => "True peak",
            Tab::Resample => "Resample",
            Tab::Ir => "IR calibration",
            Tab::Advanced => "Advanced",
        }
    }
}

/// One meter per file that is rendering, in the order the workers picked them up.
///
/// Per file rather than one for the app: four files sharing a meter would show the
/// loudest of four different pieces of music, which is a level that belongs to none
/// of them.
#[derive(Clone, Default)]
struct Cards(std::sync::Arc<std::sync::Mutex<Vec<CardState>>>);

struct CardState {
    index: usize,
    meters: std::sync::Arc<Meters>,
    finished: bool,
}

impl Cards {
    /// The meter for file `index`, creating it the first time that file starts.
    fn open(&self, index: usize) -> std::sync::Arc<Meters> {
        let mut v = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = v.iter_mut().find(|c| c.index == index) {
            c.finished = false;
            return c.meters.clone();
        }
        let m = std::sync::Arc::new(Meters::new());
        v.push(CardState {
            index,
            meters: m.clone(),
            finished: false,
        });
        m
    }

    fn finish(&self, index: usize) {
        if let Some(c) = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter_mut()
            .find(|c| c.index == index)
        {
            c.finished = true;
        }
    }

    fn list(&self) -> Vec<(usize, std::sync::Arc<Meters>, bool)> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|c| (c.index, c.meters.clone(), c.finished))
            .collect()
    }

    fn clear(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }

    fn live(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .filter(|c| !c.finished)
            .count()
    }
}

/// One row of the queue table.
struct Row {
    src: PathBuf,
    dst: PathBuf,
    phase: Phase,
    note: String,
    /// Why this file's destination is not the obvious one, from discovery.
    ///
    /// Kept apart from `note` on purpose: `note` is overwritten by the runner the moment
    /// the file starts, and a file that had to be written somewhere other than where its
    /// name pointed is worth knowing before it runs, not just after.
    collision: String,
}

impl Row {
    /// The one line to show beside the file name: the result if the file has run, and
    /// otherwise the reason its output path moved.
    fn shout(&self) -> &str {
        if self.note.is_empty() {
            &self.collision
        } else {
            &self.note
        }
    }

    fn tooltip(&self) -> String {
        let mut s = format!("\u{2192} {}", self.dst.display());
        if !self.collision.is_empty() {
            s.push('\n');
            s.push_str(&self.collision);
        }
        if !self.note.is_empty() {
            s.push('\n');
            s.push_str(&self.note);
        }
        s
    }
}

pub struct App {
    settings: Settings,
    tools: Tools,
    /// Filled in by a background probe: `probe_caps` runs ffmpeg several times and
    /// a window that appears a second later is better than one that appears frozen.
    caps: Mutex<Option<Caps>>,
    runner: Runner,
    cards: Cards,
    palette: Palette,
    /// What is installed in the context right now, so the palette is rewritten on a
    /// change rather than sixty times a second.
    installed: Option<Palette>,
    tab: Tab,
    queue: Vec<Row>,
    queue_filter: QueueFilter,
    queue_search: String,
    pick: Option<PickTarget>,
    browser: Browser,
    /// Editable/pasteable path buffers kept synced with `settings.out_dir` / `settings.ir_file`.
    out_dir_text: String,
    ir_file_text: String,
    log: Vec<(Level, String)>,
    summary: Option<Summary>,
    /// Every file that has finished this session, with the levels it was measured at.
    ///
    /// This is the reason the report carries true peak, sample peak and RMS at all. The
    /// live bars fall to rest in a third of a second, which is what a bar should do and
    /// the opposite of what a record needs: after a render the user asks "did that one
    /// come in over the ceiling", and the answer has to still be on the screen. Kept
    /// across runs, cleared by hand, newest first.
    history: Vec<MeterRow>,
    /// Delivery loudness/true-peak target used to grade finished files in `history`.
    delivery_target: DeliveryTarget,
    status: String,
    show_caps: bool,
    dry_run: Mutex<Option<DryRun>>,
    log_scroll: bool,
    last_repaint: std::time::Instant,
}

/// A planned render, shown without writing anything.
#[derive(Clone)]
struct DryRun {
    src: PathBuf,
    lines: Vec<String>,
    error: Option<String>,
}

/// One finished file, as the meters remembered it.
///
/// The strings are built once, when the file finishes, rather than formatted every frame:
/// a queue of two thousand files redraws sixty times a second and should not be asked to
/// re-render two thousand float formats to do it.
struct MeterRow {
    /// Output filename (basename), for structured CSV/JSON export.
    filename: String,
    /// Compact levels for the list row.
    levels: String,
    /// The same measurement in full, for copying.
    text: String,
    /// Measured true peak, detector ratio, sample peak, RMS, and BS.1770-5 loudness.
    true_peak_dbtp: f64,
    true_peak_ratio: String,
    sample_peak_dbfs: Vec<f64>,
    rms_dbfs: Vec<f64>,
    loudness: Option<LoudnessReport>,
    /// Came in at or over 0 dBTP, or failed to render.
    hot: bool,
    ok: bool,
}

/// A dB figure the user can read, including the one value that has no logarithm.
fn db(v: f64) -> String {
    if v == f64::NEG_INFINITY {
        "−inf".to_string()
    } else if !v.is_finite() {
        "n/a".to_string()
    } else {
        format!("{v:+.2}")
    }
}

/// RFC 4180 CSV field escaping.
fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn fmt_csv_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x:.3}"),
        _ => String::new(),
    }
}

/// Export session history as RFC 4180 CSV, newest first, graded against `target`.
fn history_csv(rows: &[MeterRow], target: DeliveryTarget) -> String {
    let mut out = String::from(
        "file,true_peak_dbtp,tp_ratio,sample_peak_l_dbfs,sample_peak_r_dbfs,\
         rms_l_dbfs,rms_r_dbfs,integrated_lufs,momentary_max_lufs,\
         short_term_max_lufs,lra_lu,plr_db,correlation,compliance\n",
    );
    for row in rows.iter().rev() {
        let comp = target.evaluate(row.true_peak_dbtp, row.loudness).badge();
        let l = row.loudness;
        let fields = [
            csv_escape(&row.filename),
            fmt_csv_f64(row.true_peak_dbtp.is_finite().then_some(row.true_peak_dbtp)),
            csv_escape(&row.true_peak_ratio),
            fmt_csv_f64(row.sample_peak_dbfs.first().copied()),
            fmt_csv_f64(row.sample_peak_dbfs.get(1).copied()),
            fmt_csv_f64(row.rms_dbfs.first().copied()),
            fmt_csv_f64(row.rms_dbfs.get(1).copied()),
            fmt_csv_f64(l.and_then(|x| x.integrated_lufs)),
            fmt_csv_f64(l.and_then(|x| x.momentary_max_lufs)),
            fmt_csv_f64(l.and_then(|x| x.short_term_max_lufs)),
            fmt_csv_f64(l.and_then(|x| x.lra_lu)),
            fmt_csv_f64(l.and_then(|x| x.plr_db)),
            fmt_csv_f64(l.map(|x| x.correlation)),
            csv_escape(&comp),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

/// Export session history as pretty-printed JSON, newest first, graded against `target`.
fn history_json(rows: &[MeterRow], target: DeliveryTarget) -> String {
    let finite_or_null = |v: Option<f64>| match v {
        Some(x) if x.is_finite() => serde_json::json!(x),
        _ => serde_json::Value::Null,
    };
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .rev()
        .map(|row| {
            let comp = target.evaluate(row.true_peak_dbtp, row.loudness).badge();
            serde_json::json!({
                "file": row.filename,
                "true_peak_dbtp": finite_or_null(Some(row.true_peak_dbtp)),
                "true_peak_ratio": row.true_peak_ratio,
                "sample_peak_dbfs": row.sample_peak_dbfs.iter().map(|&x| finite_or_null(Some(x))).collect::<Vec<_>>(),
                "rms_dbfs": row.rms_dbfs.iter().map(|&x| finite_or_null(Some(x))).collect::<Vec<_>>(),
                "loudness": row.loudness,
                "delivery_target": target.label(),
                "compliance": comp,
            })
        })
        .collect();
    serde_json::to_string_pretty(&arr).unwrap_or_else(|_| "[]".to_string())
}

impl MeterRow {
    fn from_report(r: &Report) -> Self {
        let name = r
            .output
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let list = |v: &[f64]| {
            if v.is_empty() {
                "—".to_string()
            } else {
                v.iter().map(|x| db(*x)).collect::<Vec<_>>().join("/")
            }
        };
        let mut levels = format!(
            "{} — tp {} dBTP @{} · peak {} dBFS · rms {} dBFS",
            name,
            db(r.true_peak_dbtp),
            r.true_peak_ratio.label(),
            list(&r.sample_peak_dbfs),
            list(&r.rms_dbfs)
        );
        if let Some(l) = r.loudness {
            if let Some(short) = l.short_line() {
                levels.push_str(" · ");
                levels.push_str(&short);
                levels.push_str(&format!(" · ρ {:+.2}", l.correlation));
            }
        }
        // DC follows the rule the CLI headline uses: measured on every file, printed in the
        // full copy text on every file, and on the row a person scans only when some channel
        // is over the threshold where an offset is a finding instead of a floor.
        if let Some(n) = Report::dc_notice(&r.dc_offset) {
            levels.push_str(" · ");
            levels.push_str(&n);
        }
        let hot = r.true_peak_dbtp > -0.05
            || r.limiter.as_ref().is_some_and(|l| !l.within_ceiling())
            // A file whose own decoded measurement disagrees with the stream is as urgent
            // as one that overhangs the ceiling: the headline number is not the number on
            // disk, and the row that says "-3.61 dBTP" about a file that clamped to zero is
            // the row nobody should have to read closely to distrust.
            || r.delivered_note.is_some();
        Self {
            filename: name.clone(),
            levels,
            text: format!("{}\n{}", name, r.peak_line()),
            true_peak_dbtp: r.true_peak_dbtp,
            true_peak_ratio: r.true_peak_ratio.label().to_string(),
            sample_peak_dbfs: r.sample_peak_dbfs.clone(),
            rms_dbfs: r.rms_dbfs.clone(),
            loudness: r.loudness,
            hot,
            ok: true,
        }
    }
}

impl App {
    fn new(cc: &CreationContext<'_>) -> Self {
        let settings = Settings::default();
        let out_dir_text = settings.out_dir.display().to_string();
        let ir_file_text = settings.ir_file.display().to_string();
        let mut app = Self {
            settings,
            tools: Tools::discover(),
            caps: Mutex::new(None),
            runner: Runner::new(),
            cards: Cards::default(),
            palette: Palette::default(),
            installed: None,
            tab: Tab::Renderer,
            queue: Vec::new(),
            queue_filter: QueueFilter::All,
            queue_search: String::new(),
            pick: None,
            browser: Browser::default(),
            out_dir_text,
            ir_file_text,
            log: Vec::new(),
            summary: None,
            history: Vec::new(),
            delivery_target: DeliveryTarget::default(),
            status: "idle".to_string(),
            show_caps: false,
            dry_run: Mutex::new(None),
            log_scroll: true,
            last_repaint: std::time::Instant::now(),
        };
        app.runner.set_context(cc.egui_ctx.clone());
        // Palette before the first frame: the window should not flash dark while a
        // light-mode desktop asks for light.
        app.palette.apply(&cc.egui_ctx);
        app.installed = Some(app.palette);
        app.push_log(
            Level::Info,
            format!(
                "G6 Media Encoder — {} cores, up to {} files at once",
                util::cpu_count(),
                Settings::max_threads()
            ),
        );
        // Probe capabilities off the UI thread, reporting through the same channel the
        // batch uses. Until it lands, Start stays disabled: offering "24-bit FLAC" on a
        // machine with no reference encoder is exactly the mistake this project already
        // made once and had to unwind.
        let tools = app.tools.clone();
        let tx = app.runner.sender();
        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || {
            let caps = probe_caps(&tools);
            let note = caps.summary();
            let _ = tx.send(Event::Caps(caps, note));
            ctx.request_repaint();
        });
        app
    }

    fn caps(&self) -> Option<Caps> {
        self.caps.lock().ok().and_then(|c| c.clone())
    }

    fn push_log(&mut self, level: Level, msg: impl Into<String>) {
        let msg = msg.into();
        if msg.trim().is_empty() {
            return;
        }
        // A batch can emit hundreds of lines; keep the tail and say so.
        const KEEP: usize = 400;
        self.log.push((level, msg));
        if self.log.len() > KEEP {
            let dropped = self.log.len() - KEEP;
            self.log.drain(..dropped);
            self.log.insert(
                0,
                (Level::Info, format!("… {dropped} earlier line(s) dropped")),
            );
        }
    }

    // ---- panels ------------------------------------------------------------

    fn top_panel(&mut self, ctx: &egui::Context) {
        let caps = self.caps();
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("G6 Media Encoder");
                ui.separator();
                match &caps {
                    None => {
                        ui.weak("probing ffmpeg…");
                    }
                    Some(c) => {
                        let flag = |b: bool| if b { "yes" } else { "NO" };
                        ui.label(format!(
                            "ffmpeg {} · soxr {} · afir {} · flac {} · 24-bit {}",
                            c.ffmpeg_version,
                            flag(c.soxr),
                            flag(c.afir),
                            flag(c.flac),
                            flag(c.flac24),
                        ));
                        if ui.link("details").clicked() {
                            self.show_caps = !self.show_caps;
                        }
                    }
                }
            });
            ui.horizontal(|ui| {
                let running = self.runner.is_running();
                let ready = caps.is_some() && !self.queue.is_empty();
                let start = ui
                    .add_enabled(!running && ready, egui::Button::new("▶ Start"))
                    .on_hover_text("Start rendering the queued files (Ctrl+Enter).");
                if start.clicked() {
                    self.start_run();
                }
                if ui
                    .add_enabled(running, egui::Button::new("■ Cancel"))
                    .on_hover_text(
                        "Stops after the current block and removes partial outputs, so a \
                         cancelled run leaves nothing half-written (Esc).",
                    )
                    .clicked()
                {
                    self.runner.cancel();
                    self.status = "cancelling…".to_string();
                }
                ui.separator();
                let mut t = self.settings.threads as u32;
                // The ceiling is the shared one from `util`, so the slider cannot offer
                // a number the CLI would reject.
                let max = Settings::max_threads() as u32;
                ui.add_enabled(
                    !running,
                    egui::Slider::new(&mut t, 1..=max.max(1)).text("files at once"),
                );
                self.settings.threads = util::clamp_threads(t as usize);
                ui.separator();
                ui.monospace(describe(&self.settings));
                ui.separator();
                // Here rather than in a tab: it is a window property, like the title
                // bar, not a rendering setting, and hunting for it in a page is the
                // worst place to look once the screen is bright enough to hurt.
                // "system" is not a colour, so the button says what it currently means.
                let cur = match self.palette {
                    Palette::Auto => format!(
                        "system ({})",
                        match self.palette.resolved(ctx) {
                            egui::Theme::Dark => "dark",
                            egui::Theme::Light => "light",
                        }
                    ),
                    p => p.label().to_string(),
                };
                egui::ComboBox::from_id_salt("palette")
                    .selected_text(format!("theme: {cur}"))
                    .show_ui(ui, |ui| {
                        for p in Palette::ALL {
                            ui.selectable_value(&mut self.palette, p, p.label());
                        }
                    })
                    .response
                    .on_hover_text(
                        "system follows the desktop's light/dark setting. pure black sets every \
                         large surface to #000 for OLED panels; widgets keep a faint fill so \
                         they stay visible.",
                    );
            });
            ui.add_space(3.0);
        });
    }

    fn bottom_panel(&mut self, ctx: &egui::Context) {
        let counters = self.runner.counters();
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(c) = &counters {
                    let bar = egui::ProgressBar::new(c.fraction()).text(format!(
                        "{}/{} files · {} running",
                        c.done + c.failed + c.skipped,
                        c.total,
                        c.running
                    ));
                    ui.add(bar);
                    ui.separator();
                    ui.label(format!(
                        "{} · {} of audio rendered",
                        util::human_bytes(c.bytes),
                        util::human_time(c.seconds_done)
                    ));
                    // ETA from the observed throughput so far, using the average
                    // duration of a completed file for the ones still queued.
                    let per_item = if c.done > 0 {
                        c.seconds_done / c.done as f64
                    } else {
                        0.0
                    };
                    if let Some(eta) = self.runner.elapsed().and_then(|e| c.eta(e, per_item)) {
                        ui.weak(format!(
                            "about {} left",
                            util::human_time(eta.as_secs_f64())
                        ));
                    }
                } else {
                    ui.weak("nothing has run yet");
                }
                if let Some(s) = &self.summary {
                    let col = if s.failed > 0 {
                        egui::Color32::from_rgb(226, 140, 90)
                    } else {
                        egui::Color32::from_rgb(120, 190, 120)
                    };
                    ui.colored_label(col, s.one_line());
                }
            });
            ui.separator();
            self.meters_panel(ui, ctx);
        });
    }

    /// One bordered card per file that is rendering, in a scrollable list.
    ///
    /// The area takes a slice of the window height and the cards inside it share
    /// that slice: one file rendering gets a tall card, eight get eight shorter ones
    /// and the list starts scrolling. Nothing here is pinned at a size that only
    /// happens to suit the screen it was written on.
    fn meters_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let cards = self.cards.list();
        // Measured from the window, not the screen. `screen_rect` was the bug: on a
        // 4 K display it hands the meter strip 420 px whether the window is 1 400 px
        // tall or 500, and in a short window that is the entire UI.
        let win_h = ctx
            .input(|i| i.raw.viewport().inner_rect)
            .map(|r| r.height())
            .filter(|h| *h > 0.0)
            .unwrap_or(800.0);
        // A third of the window, but always leaving 300 px for the top bar, the queue
        // and the settings page, and never less than one readable card.
        let area = (win_h * 0.34)
            .clamp(96.0, 340.0)
            .min((win_h - 300.0).max(96.0));
        if cards.is_empty() {
            ui.vertical(|ui| {
                ui.set_height(area.min(48.0));
                ui.weak(
                    "no file is rendering — each file gets its own card with ten levels the \
                     moment it starts",
                );
            });
        } else {
            // Equal slices, with a floor: below about 84 px the bars stop being
            // readable and scrolling is better than squashing them.
            let card_h = (area / cards.len() as f32).clamp(84.0, 240.0);
            egui::ScrollArea::vertical()
                .max_height(area)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (idx, meters, finished) in cards {
                        let (title, status) = self.card_label(idx, finished);
                        let card = widgets::Card {
                            title: &title,
                            status: &status,
                            finished,
                        };
                        // A card is per file, so clicking it clears that file's holds
                        // and nobody else's.
                        if widgets::meter_card(ui, &meters, &card, card_h) {
                            meters.reset();
                        }
                        ui.add_space(5.0);
                    }
                });
        }
        self.history_panel(ui, ctx);
        ui.horizontal(|ui| {
            ui.weak(widgets::caption(self.cards.live()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(self.status.clone());
            });
        });
    }

    /// The retained numbers, one line per finished file.
    ///
    /// Text rather than bars, deliberately: a bar is a memory of a moment and cannot be
    /// read off a screen two seconds later, copied into a delivery note, or compared
    /// between two runs. Every figure is what was measured on the bytes that were
    /// written, so this is a record of the output, not of the rendering.
    fn history_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.history.is_empty() {
            return;
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.weak(format!("finished files — {}", self.history.len()));
            egui::ComboBox::from_id_salt("delivery-target")
                .selected_text(self.delivery_target.label())
                .show_ui(ui, |ui| {
                    for t in DeliveryTarget::ALL {
                        ui.selectable_value(&mut self.delivery_target, t, t.label());
                    }
                })
                .response
                .on_hover_text(
                    "Delivery loudness & true-peak standard used to grade finished files \
                     (ITU-R BS.1770-5 / EBU R128 / ATSC A/85).",
                );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("copy JSON")
                    .on_hover_text(
                        "Copy finished-file measurements and compliance as JSON, newest first.",
                    )
                    .clicked()
                {
                    ctx.copy_text(history_json(&self.history, self.delivery_target));
                    self.status = format!("copied {} row(s) as JSON", self.history.len());
                }
                if ui
                    .small_button("copy CSV")
                    .on_hover_text(
                        "Copy finished-file measurements and compliance as CSV, newest first.",
                    )
                    .clicked()
                {
                    ctx.copy_text(history_csv(&self.history, self.delivery_target));
                    self.status = format!("copied {} row(s) as CSV", self.history.len());
                }
                if ui
                    .small_button("copy all")
                    .on_hover_text("Copy every line as plain text, newest first.")
                    .clicked()
                {
                    let text = self.history_text();
                    ctx.copy_text(text);
                    self.status = format!("copied {} row(s) as text", self.history.len());
                }
                if ui
                    .small_button("clear")
                    .on_hover_text("Forget this session's history.")
                    .clicked()
                {
                    self.history.clear();
                }
            });
        });
        let rows = self.history.len().min(12) as f32;
        egui::ScrollArea::vertical()
            .max_height((rows * 17.0).clamp(34.0, 210.0))
            .auto_shrink([false, false])
            .stick_to_bottom(false)
            .show(ui, |ui| {
                // Drawn newest first: the file that just finished is the one being asked
                // about, and a list that grows downwards pushes it off screen.
                for row in self.history.iter().rev() {
                    let comp = self
                        .delivery_target
                        .evaluate(row.true_peak_dbtp, row.loudness);
                    let badge_col = match comp {
                        Compliance::Pass => widgets::tone(
                            ui,
                            egui::Color32::from_rgb(96, 176, 92),
                            egui::Color32::from_rgb(30, 122, 60),
                        ),
                        Compliance::OverCeiling => widgets::tone(
                            ui,
                            egui::Color32::from_rgb(240, 110, 96),
                            egui::Color32::from_rgb(176, 26, 16),
                        ),
                        Compliance::TooLoud(_) | Compliance::Quiet(_) => widgets::tone(
                            ui,
                            egui::Color32::from_rgb(216, 178, 58),
                            egui::Color32::from_rgb(168, 128, 14),
                        ),
                        Compliance::Unmeasured => ui.visuals().weak_text_color(),
                    };
                    ui.horizontal(|ui| {
                        ui.colored_label(widgets::status_dot(ui, row.hot, row.ok), "●");
                        ui.label(
                            egui::RichText::new(format!("[{}]", comp.badge()))
                                .monospace()
                                .size(10.0)
                                .color(badge_col),
                        )
                        .on_hover_text(format!("Graded against {}", self.delivery_target.label()));
                        // Selectable so a single line can be copied without the button.
                        ui.add(egui::Label::new(&row.levels).selectable(true));
                        if ui
                            .small_button("⧉")
                            .on_hover_text("Copy this file's full measurement as text.")
                            .clicked()
                        {
                            ctx.copy_text(row.text.clone());
                        }
                    });
                }
            });
    }

    /// The whole history as plain text, newest first, for a log or a delivery note.
    fn history_text(&self) -> String {
        let mut out = String::new();
        for row in self.history.iter().rev() {
            out.push_str(&row.text);
            out.push('\n');
        }
        out
    }

    /// Title and status for a card, read from the queue row: a card must never
    /// claim to be a different file than the table says it is.
    fn card_label(&self, index: usize, finished: bool) -> (String, String) {
        let Some(row) = self.queue.get(index) else {
            return (format!("file #{index}"), String::new());
        };
        let title = row
            .src
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| row.src.display().to_string());
        let status = if finished {
            truncate(row.shout(), 44)
        } else {
            "rendering".to_string()
        };
        (title, status)
    }

    fn queue_panel(&mut self, ctx: &egui::Context) {
        let running = self.runner.is_running();
        let failed_count = self
            .queue
            .iter()
            .filter(|r| r.phase == Phase::Failed)
            .count();
        egui::SidePanel::left("queue")
            .default_width(360.0)
            .min_width(260.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Queue ({})", self.queue.len()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add_enabled(!running, egui::Button::new("clear"))
                            .clicked()
                        {
                            self.queue.clear();
                            self.settings.inputs.clear();
                        }
                        if failed_count > 0
                            && ui
                                .add_enabled(
                                    !running,
                                    egui::Button::new(format!("retry failed ({failed_count})")),
                                )
                                .on_hover_text("Re-run only the files that failed in the last batch.")
                                .clicked()
                        {
                            self.retry_failed();
                        }
                    });
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!running, egui::Button::new("add audio files…"))
                        .on_hover_text("Open system file picker (or drag & drop files onto the window)")
                        .clicked()
                    {
                        self.open_picker(ctx, PickTarget::Queue, Mode::AudioFile);
                    }
                    if ui
                        .add_enabled(!running, egui::Button::new("add folder…"))
                        .on_hover_text("Open system folder picker (or drag & drop folders onto the window)")
                        .clicked()
                    {
                        self.open_picker(ctx, PickTarget::Queue, Mode::Dir);
                    }
                    #[cfg(feature = "native-dialogs")]
                    if ui
                        .add_enabled(!running, egui::Button::new("in-app…"))
                        .on_hover_text("Open built-in cross-platform browser (for minimal sessions without a portal)")
                        .clicked()
                    {
                        self.open_in_app_picker(PickTarget::Queue, Mode::AudioFile);
                    }
                });
                let mut rescan = false;
                if ui
                    .checkbox(&mut self.settings.recursive, "scan folders recursively")
                    .changed()
                {
                    rescan = true;
                }
                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut self.settings.use_out_dir, "write outputs to")
                        .changed()
                    {
                        rescan = true;
                    }
                    if ui
                        .add_enabled(!running, egui::Button::new("…"))
                        .on_hover_text("choose an output folder")
                        .clicked()
                    {
                        self.open_picker(ctx, PickTarget::OutDir, Mode::Dir);
                    }
                    if ui
                        .small_button("open folder")
                        .on_hover_text("Open the current output folder in the system file manager.")
                        .clicked()
                    {
                        self.open_output_folder();
                    }
                });
                if self.settings.use_out_dir {
                    ui.horizontal(|ui| {
                        let resp = ui.add_enabled(
                            !running,
                            egui::TextEdit::singleline(&mut self.out_dir_text)
                                .desired_width(ui.available_width())
                                .font(egui::FontId::monospace(11.0))
                                .hint_text("paste an output folder path…"),
                        );
                        if resp.changed() {
                            self.settings.out_dir = widgets::clean_pasted_path(&self.out_dir_text);
                            rescan = true;
                        }
                    });
                } else {
                    ui.weak("outputs go next to each input");
                }
                if rescan && !running {
                    self.rebuild_queue();
                }
                if !self.queue.is_empty() {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.queue_search)
                                .desired_width((ui.available_width() - 26.0).max(80.0))
                                .hint_text("filter queue by filename…"),
                        );
                        if !self.queue_search.is_empty() && ui.small_button("✕").clicked() {
                            self.queue_search.clear();
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        for f in QueueFilter::ALL {
                            let n = self
                                .queue
                                .iter()
                                .filter(|r| f.matches(r.phase))
                                .count();
                            let label = if f == QueueFilter::All || n > 0 {
                                format!("{} ({n})", f.label())
                            } else {
                                f.label().to_string()
                            };
                            ui.selectable_value(&mut self.queue_filter, f, label);
                        }
                    });
                }
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.queue.is_empty() {
                            ui.weak(
                                "Nothing queued.\n\nDrag and drop audio files or folders onto \
                                 this window, or add them above — anything ffmpeg can decode \
                                 works (.mka TrueHD/Atmos, .wav, .flac, .mkv, .m4a…).",
                            );
                            return;
                        }
                        let mut remove: Option<usize> = None;
                        let mut shown = 0usize;
                        for (i, r) in self.queue.iter().enumerate() {
                            if !row_matches_filter(r, self.queue_filter, &self.queue_search) {
                                continue;
                            }
                            shown += 1;
                            let col = match r.phase {
                                Phase::Queued => egui::Color32::GRAY,
                                Phase::Running => egui::Color32::from_rgb(120, 170, 220),
                                Phase::Done => egui::Color32::from_rgb(120, 190, 120),
                                Phase::Failed => egui::Color32::from_rgb(226, 110, 90),
                                Phase::Skipped => egui::Color32::from_rgb(200, 170, 90),
                            };
                            let head = format!(
                                "{}  {}",
                                r.phase.label(),
                                r.src
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| r.src.display().to_string())
                            );
                            ui.horizontal(|ui| {
                                ui.colored_label(col, head);
                                if !r.shout().is_empty() {
                                    ui.weak(truncate(r.shout(), 40));
                                }
                                if !running && ui.small_button("✕").clicked() {
                                    remove = Some(i);
                                }
                            })
                            .response
                            .on_hover_text(r.tooltip());
                        }
                        if shown == 0 {
                            ui.weak("No queued files match the current filter.");
                        }
                        // Removed after the loop: the table is borrowed while it is
                        // being drawn.
                        if let Some(i) = remove {
                            let row = self.queue.remove(i);
                            self.settings.inputs.retain(|p| {
                                // A folder entry stays: dropping one file out of a
                                // scanned tree is not the same as un-queuing it.
                                *p != row.src || p.is_dir()
                            });
                        }
                    });
            });
    }

    fn central_panel(&mut self, ctx: &egui::Context) {
        let caps = self.caps();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                for t in Tab::ALL {
                    ui.selectable_value(&mut self.tab, t, t.label());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("dry run")
                        .on_hover_text(
                            "Plan the first queued file: show the filter chain and the exact \
                             commands, write nothing (Ctrl+D).",
                        )
                        .clicked()
                    {
                        self.run_dry_run();
                    }
                });
            });
            ui.separator();
            // The log gets a fixed slice of the height and the settings scroll inside
            // what is left. Leaving this to `auto_shrink([false, false])` let the
            // settings area claim every pixel and pushed the log out of the window.
            const LOG_BUDGET: f32 = 168.0;
            let settings_h = (ui.available_height() - LOG_BUDGET).max(120.0);
            egui::ScrollArea::vertical()
                .max_height(settings_h)
                .auto_shrink([false, false])
                .show(ui, |ui| match self.tab {
                    Tab::Renderer => self.tab_renderer(ui),
                    Tab::Output => self.tab_output(ui, caps.as_ref()),
                    Tab::TruePeak => self.tab_true_peak(ui),
                    Tab::Resample => self.tab_resample(ui, caps.as_ref()),
                    Tab::Ir => self.tab_ir(ui, caps.as_ref()),
                    Tab::Advanced => self.tab_advanced(ui),
                });
            ui.separator();
            self.log_panel(ui);
        });
    }

    /// Nugen Halo Upmix–inspired binaural renderer & soundfield profile controls.
    fn tab_renderer(&mut self, ui: &mut egui::Ui) {
        let running = self.runner.is_running();
        ui.horizontal(|ui| {
            ui.label("engine");
            ui.add_enabled_ui(!running, |ui| {
                ui.selectable_value(
                    &mut self.settings.renderer,
                    RendererMode::Reference,
                    RendererMode::Reference.display_name(),
                )
                .on_hover_text(
                    "Legacy Sound BlasterX G6 all-pass downmix (bit-exact hardware clone): \
                     direct, punchy, with sharper surround phase contrast and no HRTF/DFEQ shaping.",
                );
                ui.selectable_value(
                    &mut self.settings.renderer,
                    RendererMode::Plus,
                    RendererMode::Plus.display_name(),
                )
                .on_hover_text(
                    "G6+ physical binaural renderer: speaker-like out-of-head width, natural \
                     head shadow/ITD, and neutral 1-ERB diffuse-field EQ that preserves HyprEQ tonality.",
                );
            });
        });
        ui.add_space(6.0);

        let live_in = self
            .cards
            .list()
            .into_iter()
            .find(|(_, _, finished)| !finished)
            .map(|(_, meters, _)| std::array::from_fn(|i| meters.param(i)));

        ui.horizontal_top(|ui| {
            widgets::soundfield_radar(
                ui,
                self.settings.renderer,
                &self.settings.plus,
                live_in,
                228.0,
            );
            ui.add_space(12.0);
            ui.vertical(|ui| {
                if self.settings.renderer == RendererMode::Reference {
                    ui.strong("Bit-Exact G6 Reference Engine");
                    ui.add_space(4.0);
                    ui.label(
                        "Runs the untouched CMatrixEncoderEfxMod 64-bit matrix with exact \
                         sample-for-sample parity against vendor/g6ct.",
                    );
                    ui.add_space(6.0);
                    ui.weak(
                        "Switch to Plus (G6+ Binaural) for the universal non-personalized \
                         renderer: physical Woodworth ITD, Brown–Duda spherical head shadow, \
                         causal LR4 bass-mono, DEE-safe rear decorrelation, dry ITU-BS.1116 \
                         lateral externalization, and strict 1-ERB diffuse-field equalization \
                         that stacks cleanly with HyprEQ.",
                    );
                    ui.add_space(8.0);
                    if ui
                        .add_enabled(
                            !running,
                            egui::Button::new("Switch to G6+ Plus (Studio) →"),
                        )
                        .clicked()
                    {
                        self.settings.renderer = RendererMode::Plus;
                        self.settings.plus = PlusCfg::from_profile(PlusProfile::Studio);
                    }
                    return;
                }

                // Plus mode: presets + real-time tweakable soundfield sliders.
                ui.horizontal(|ui| {
                    ui.label("profile");
                    ui.add_enabled_ui(!running, |ui| {
                        for p in [PlusProfile::Pure, PlusProfile::Studio, PlusProfile::Cinema] {
                            let active = self.settings.plus.profile == p;
                            let tip = match p {
                                PlusProfile::Pure => {
                                    "Pure anechoic speaker sphere: zero room reflections, strict \
                                     0 dB LF balance, and no de-comb. Sounds maximally direct/dry, \
                                     though close stereo mixes can feel closer to the head and warmer in bass."
                                }
                                PlusProfile::Studio => {
                                    "Balanced nearfield control room: subtle lateral space (+9 dB DRR), \
                                     -1.5 dB LF balance to clear stereo bass buildup, and 55% phantom \
                                     de-comb for natural, unboxed vocals."
                                }
                                PlusProfile::Cinema => {
                                    "Wide theatrical stage (±35°), richer surrounds (1.20x), deeper \
                                     room cues, -2.1 dB LF balance, and 75% de-comb. Big spatial scale \
                                     without boomy low-end."
                                }
                                PlusProfile::Custom => "",
                            };
                            if ui
                                .selectable_label(active, p.display_name())
                                .on_hover_text(tip)
                                .clicked()
                            {
                                self.settings.plus = PlusCfg::from_profile(p);
                            }
                        }
                    });
                    if self.settings.plus.profile == PlusProfile::Custom {
                        ui.colored_label(
                            egui::Color32::from_rgb(76, 194, 232),
                            "• Custom",
                        );
                    }
                });
                ui.weak(self.settings.plus.profile.description());
                ui.add_space(6.0);

                let mut changed = false;
                ui.add_enabled_ui(!running, |ui| {
                    let p = &mut self.settings.plus;
                    let fl_deg = (30.0 * p.width).clamp(15.0, 65.0);
                    ui.horizontal(|ui| {
                        ui.label("stage width");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.width, 0.50..=1.50)
                                    .step_by(0.01)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Spreads or narrows the front L/R speakers (±15° to ±45°; 1.00 = ±30°). \
                                 Higher makes guitars, pads, and panning feel wider around your head; \
                                 lower pulls lead focus and punch tighter inward.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        ui.weak(format!("±{fl_deg:.0}°"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("center focus");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.center_focus, 0.00..=1.00)
                                    .step_by(0.01)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Locks center dialogue crisp and forward by drying early reflections on FC. \
                                 Higher sounds closer, clearer, and better anchored; lower blends speech \
                                 more softly into the room.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("surround depth");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.surround_depth, 0.50..=1.50)
                                    .step_by(0.01)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Scales side and rear channel level (SL/SR/BL/BR) relative to the front. \
                                 Raise for deeper 360° movie envelopment and Atmos height immersion; \
                                 lower if ambient surrounds sit too loud.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("externalize");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.externalize, 0.00..=1.00)
                                    .step_by(0.01)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Blends smooth, comb-free lateral studio reflections (>= 8.8 ms). \
                                 Higher moves the soundstage out in front of your face like room speakers; \
                                 0.00 gives dry, close in-head monitors.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("bass mono");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.bass_mono_hz, 0.0..=250.0)
                                    .step_by(5.0)
                                    .fixed_decimals(0)
                                    .suffix(" Hz"),
                            )
                            .on_hover_text(
                                "Sums sub-bass below this cutoff cleanly to center mono (4th-order LR4; 0 = off). \
                                 Firms up kick impact and eliminates phase-hollow bass across headphones; \
                                 0 keeps raw stereo/surround bass phase.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("LF balance");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.bass_trim_db, -6.0..=3.0)
                                    .step_by(0.1)
                                    .fixed_decimals(1)
                                    .suffix(" dB"),
                            )
                            .on_hover_text(
                                "Bauer / Linkwitz 340 Hz shelf trimming coherent in-phase bass buildup on \
                                 full-range channels. Negative dB (-1.5 to -2.5) clears boxy warmth and \
                                 boominess vs Reference; positive dB thickens low-end weight.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("phantom de-comb");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.phantom_decomb, 0.00..=1.00)
                                    .step_by(0.01)
                                    .fixed_decimals(2),
                            )
                            .on_hover_text(
                                "Shirley–Kendrick adaptive 1-ERB fill for the 1.6–2.0 kHz comb notch when \
                                 centered L/R signals overlap. Higher restores open vocal presence and snare \
                                 snap without adding harsh cymbal treble.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("LFE level");
                        if ui
                            .add(
                                egui::Slider::new(&mut p.lfe_gain_db, -12.0..=12.0)
                                    .step_by(0.1)
                                    .fixed_decimals(1)
                                    .suffix(" dB"),
                            )
                            .on_hover_text(
                                "Discrete `.1` LFE gain offset (-12 to +12 dB). 0.0 dB preserves exact \
                                 Reference / Nuendo discrete LFE unity (when no bass is routed as objects \
                                 into LFE); boost (+6 to +10 dB) for theatrical `.1` rumble or cut if LFE overpowers.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(
                                &mut p.lfe_lpf,
                                "LFE 120 Hz LPF",
                            )
                            .on_hover_text(
                                "Rolls off discrete `.1` LFE above 120 Hz (phase-aligned 4th-order Butterworth). \
                                 Keeps LFE strictly sub-bass rumble and prevents upper-bass bleed from muddying vocals.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                        if ui
                            .checkbox(
                                &mut p.lfe_lf_trim,
                                "apply LF balance to LFE",
                            )
                            .on_hover_text(
                                "Off (default): discrete `.1` LFE ignores the LF balance shelf and stays at your \
                                 exact LFE level (0.0 dB for Nuendo discrete LFE). On: applies the same 340 Hz \
                                 LF balance shelf to `.1` so sub-bass tracks the main bed.",
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    });
                });
                if changed {
                    self.settings.plus.sync_profile();
                }

                ui.add_space(6.0);
                let fl_az = (30.0 * self.settings.plus.width).clamp(15.0, 65.0);
                let fl_itd_us = woodworth_itd_s(fl_az) * 1e6;
                let drr = match self.settings.plus.estimated_center_drr_db() {
                    Some(d) => format!("{d:+.1} dB"),
                    None => "∞ (dry)".to_string(),
                };
                ui.monospace(format!(
                    "FL ITD {fl_itd_us:.0} µs · FC DRR {drr} · LF {:+.1} dB{} · LFE {:+.1} dB · decomb {:.0}% · DFEQ ±0.20 dB",
                    self.settings.plus.bass_trim_db,
                    if self.settings.plus.lfe_lf_trim {
                        "+LFE"
                    } else {
                        ""
                    },
                    self.settings.plus.lfe_gain_db,
                    self.settings.plus.phantom_decomb * 100.0,
                ));
                ui.weak(
                    "Universal non-personalized cues (no ear scan or SOFA needed). Direct-path \
                     1-ERB DFEQ + Bauer LF balance preserve HyprEQ tonality across stereo & 7.1.",
                );
                if let Err(e) = self.settings.plus.validate() {
                    ui.colored_label(egui::Color32::from_rgb(226, 110, 90), e);
                }
            });
        });
    }

    fn tab_output(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        ui.horizontal(|ui| {
            ui.label("container");
            let prev_fmt = self.settings.format;
            for f in Format::ALL {
                let tip = match f {
                    Format::Flac => {
                        "Lossless compressed audio: bit-for-bit identical sound to WAV at smaller file size."
                    }
                    Format::Wav => {
                        "Uncompressed PCM/float audio: bit-for-bit identical sound at 16/24-bit, plus \
                         supports 32-bit float to preserve overs above 0 dBFS without clipping."
                    }
                };
                ui.selectable_value(&mut self.settings.format, f, f.label())
                    .on_hover_text(tip);
            }
            if self.settings.format != prev_fmt && !self.runner.is_running() {
                self.rebuild_queue();
            }
            if self.settings.format == Format::Flac {
                ui.separator();
                ui.label("level");
                let mut l = self.settings.flac_level;
                // ffmpeg's FLAC encoder counts 0..=12 and refuses 13; the reference
                // flac CLI's -0..-8 presets are a different scale and not in use.
                if ui
                    .add(egui::DragValue::new(&mut l).range(0..=12))
                    .on_hover_text(
                        "Lossless compression effort only: zero sonic difference across 0..12.",
                    )
                    .changed()
                {
                    self.settings.flac_level = l.min(12);
                }
                ui.weak("(0 fast … 12 smallest)");
            }
        });
        ui.horizontal(|ui| {
            ui.label("bit depth");
            egui::ComboBox::from_id_salt("depth")
                .selected_text(self.settings.depth.label())
                .show_ui(ui, |ui| {
                    for d in BitDepth::ALL {
                        let why = depth_note(d, self.settings.format, caps);
                        let label = match why {
                            Some(w) => format!("{} — {}", d.label(), w),
                            None => d.label().to_string(),
                        };
                        ui.selectable_value(&mut self.settings.depth, d, label);
                    }
                })
                .response
                .on_hover_text(
                    "Sets output quantization floor: 24-bit (-144 dB) is transparent; 16-bit raises \
                     background hiss slightly on quiet tails; 32-bit float keeps peaks above 0 dBFS intact.",
                );
            if self.settings.depth == BitDepth::S24 && self.settings.format == Format::Flac {
                // Not a claim about a version string: the probe encodes a known
                // 24-bit block with this ffmpeg and checks the bytes come back.
                let ok = caps.map(|c| c.flac24).unwrap_or(true);
                ui.colored_label(
                    if ok {
                        egui::Color32::from_rgb(120, 190, 120)
                    } else {
                        egui::Color32::from_rgb(226, 110, 90)
                    },
                    if ok {
                        "24-bit FLAC verified lossless on this ffmpeg"
                    } else {
                        "this ffmpeg failed the 24-bit FLAC check — see capabilities"
                    },
                );
            }
        });
        if self.settings.format == Format::Flac {
            ui.weak(
                "FLAC carries 16- or 24-bit integers only: ffmpeg writes a genuine bps=24 \
                   stream, but it cannot carry 32-bit PCM or float — WAV does both.",
            );
        }
        ui.horizontal(|ui| {
            ui.label("output dither");
            egui::ComboBox::from_id_salt("out-dither")
                .selected_text(self.settings.dither.label())
                .show_ui(ui, |ui| {
                    for d in Dither::ALL {
                        let invalid_24 = self.settings.depth == BitDepth::S24
                            && !g6me_core::pcm::Grid24::supports(d);
                        let label = if invalid_24 {
                            format!("{} — 16-bit only", d.label())
                        } else {
                            d.label().to_string()
                        };
                        ui.selectable_value(&mut self.settings.dither, d, label);
                    }
                })
                .response
                .on_hover_text(
                    "Applied when quantizing down to the output bit depth. Noise-shaping dithers \
                     (Lipshitz, Shibata, E/F-weighted) are restricted to 16-bit by ffmpeg's libswresample.",
                );
            if self.settings.dither != Dither::None {
                if self.settings.depth == BitDepth::S24
                    && !g6me_core::pcm::Grid24::supports(self.settings.dither)
                {
                    ui.colored_label(
                        egui::Color32::from_rgb(226, 110, 90),
                        "24-bit supports rectangular / triangular / triangular_hp only",
                    );
                } else {
                    ui.weak("deterministic: the same input gives the same output every run");
                }
            }
        });
        ui.checkbox(&mut self.settings.copy_metadata, "copy source metadata");
        ui.checkbox(
            &mut self.settings.overwrite,
            "overwrite existing outputs (otherwise they are skipped)",
        );
        ui.checkbox(
            &mut self.settings.sha256,
            "hash each output (the sample stream, and the file as sha256sum sees it)",
        );
        ui.separator();
        ui.weak(
            "The internal maths is 64-bit float in both the downmixer and the convolver, and \
             that is not a choice: it is what makes the output match the reference build for \
             build. Bit depth here only decides how the result is written.",
        );
    }

    /// True-peak metering and limiting.
    ///
    /// The consequences shown here — guard band, latency, the lookahead actually used —
    /// are computed by the core from the same functions the render uses, not restated,
    /// because a panel that recalculates them is a panel that will quietly disagree with
    /// the file it produced.
    fn tab_true_peak(&mut self, ui: &mut egui::Ui) {
        use g6me_core::limiter::Cfg as LCfg;
        use g6me_core::tp::Ratio;

        let running = self.runner.is_running();
        ui.horizontal_wrapped(|ui| {
            ui.label("presets");
            ui.add_enabled_ui(!running, |ui| {
                for preset in MasteringPreset::ALL {
                    if ui
                        .small_button(preset.label())
                        .on_hover_text(preset.description())
                        .clicked()
                    {
                        self.settings.apply_mastering_preset(preset);
                    }
                }
            });
        });
        ui.separator();

        // The rate the DSP runs at: the resampler's target when resampling is on, and
        // otherwise whatever the source is. Lookahead in ms and latency in samples trade
        // on that number, so it is stated rather than assumed.
        let rate = if self.settings.resample {
            self.settings.rate
        } else {
            48_000
        };
        ui.horizontal(|ui| {
            ui.label("measure at");
            egui::ComboBox::from_id_salt("tp-ratio")
                .selected_text(self.settings.tp_ratio.label())
                .show_ui(ui, |ui| {
                    for r in Ratio::ALL {
                        ui.selectable_value(
                            &mut self.settings.tp_ratio,
                            r,
                            format!(
                                "{} — {} MAC/sample, ≤{} dB under-read",
                                r.label(),
                                r.mac_per_sample(),
                                r.worst_underread_db()
                            ),
                        );
                    }
                })
                .response
                .on_hover_text(
                    "Reporting detector oversampling only: changes no sound or output bytes; higher \
                     ratios catch inter-sample peaks more accurately.",
                );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.weak(format!("at {}", self.settings.tp_ratio.chain()));
            });
        });
        ui.weak(format!(
            "Every render reports the true peak of what it wrote, in dBTP. The ratio is the \
             detector's over-sampling factor: a lower one cannot see a crest that falls \
             between its grid points, so it under-reads — x2 by up to {} dB. Metering changes \
             no output bytes.",
            Ratio::X2.worst_underread_db()
        ));
        ui.add_space(6.0);
        // Above the limiter, and above its early return, because DC does not care whether
        // limiting is on: an offset ships in the file either way. It earns its keep under a
        // limiter too, since a limiter reads an offset as level and will hold gain down for
        // a signal that never clips. Every render reports the file's own `dc_offset`, so the
        // box gets ticked from a measurement rather than from a hunch.
        ui.checkbox(&mut self.settings.dc_block, "block DC (5 Hz high-pass)")
            .on_hover_text(
                "Removes DC offset below 5 Hz without altering audible bass, mids, or treble. \
                 Prevents silent DC bias from prematurely triggering the limiter or clicking at edits.",
            );
        ui.add_space(6.0);
        ui.checkbox(&mut self.settings.limiter, "limit the true peak")
            .on_hover_text(
                "Transparent two-stage lookahead limiter catching inter-sample overs above your ceiling \
                 so DACs and AAC/Opus encoders don't clip or crackle on hot surround peaks.",
            );
        if !self.settings.limiter {
            ui.horizontal(|ui| {
                ui.label("gain trim");
                let mut g = self.settings.gain_db;
                if ui
                    .add(
                        egui::DragValue::new(&mut g)
                            .range(-24.0..=0.0)
                            .speed(0.05)
                            .suffix(" dB"),
                    )
                    .on_hover_text("Constant gain staging applied to every sample before encoding (0 dB = untouched).")
                    .changed()
                {
                    self.settings.gain_db = g.clamp(-24.0, 0.0);
                }
                if self.settings.gain_db == 0.0 && !self.settings.dc_block {
                    ui.weak("Off: files are measured, never changed.");
                } else {
                    ui.weak("Limiter off: only DC block / gain trim above apply.");
                }
            });
            return;
        }
        let st = &mut self.settings;
        // One field, two effects: the trim and the ceiling. It is the first control here
        // because it is the one that decides what the others are for — with a setpoint in
        // play the limiter is only ever catching peaks, since the level came down as a
        // constant gain on the way in.
        ui.horizontal(|ui| {
            ui.label("setpoint");
            let mut sp = st.setpoint_db.unwrap_or(st.ceiling_db);
            if ui
                .add(
                    egui::DragValue::new(&mut sp)
                        .range(-24.0..=0.0)
                        .speed(0.05)
                        .suffix(" dBTP"),
                )
                .on_hover_text(
                    "Lowers overall level cleanly by this dB and caps peaks there too. \
                     Keeps dynamic punch natural by padding headroom so the limiter only catches stray crests.",
                )
                .changed()
            {
                st.setpoint_db = Some(sp.clamp(-24.0, 0.0));
            }
            if let Some(sp) = st.setpoint_db {
                ui.label(format!(
                    "({sp:+.2} dB trim & {sp:+.2} dBTP ceiling; click separate to decouple)"
                ));
                if ui
                    .button("separate")
                    .on_hover_text("Decouples pre-gain trim from peak ceiling so you can preserve raw loudness.")
                    .clicked()
                {
                    st.setpoint_db = None;
                }
            } else {
                ui.label("gain trim");
                let mut g = st.gain_db;
                if ui
                    .add(
                        egui::DragValue::new(&mut g)
                            .range(-24.0..=0.0)
                            .speed(0.05)
                            .suffix(" dB"),
                    )
                    .on_hover_text("Constant gain applied before the limiter (0 dB = unity).")
                    .changed()
                {
                    st.gain_db = g.clamp(-24.0, 0.0);
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("ceiling");
            let mut c = st.setpoint_db.unwrap_or(st.ceiling_db);
            if ui
                .add_enabled(
                    st.setpoint_db.is_none(),
                    egui::DragValue::new(&mut c)
                        .range(-24.0..=0.0)
                        .speed(0.05)
                        .suffix(" dBTP"),
                )
                .on_hover_text(
                    "Maximum peak level allowed cleanly out of the limiter (-1.0 to -0.3 dBTP avoids DAC \
                     reconstruction clipping while keeping full transient snap).",
                )
                .changed()
            {
                st.ceiling_db = c.clamp(-24.0, 0.0);
            }
            ui.label("lookahead");
            let mut la = st.lookahead_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut la)
                        .range(
                            g6me_core::limiter::MIN_LOOKAHEAD_MS
                                ..=g6me_core::limiter::MAX_LOOKAHEAD_MS,
                        )
                        .speed(0.02)
                        .suffix(" ms"),
                )
                .on_hover_text(
                    "Pre-reads upcoming transients so gain ducks smoothly right before peaks arrive, \
                     avoiding harsh clicky clipping distortion.",
                )
                .changed()
            {
                st.lookahead_ms = la;
            }
            ui.label("limit at");
            // The label used to read "same as meter", which is how a fallback to the meter's x4
            // passed for a design decision for as long as it did. The two defaults are different
            // numbers on purpose, so the label says which one it is, derived from the constant.
            let default_label = format!("default ({})", g6me_core::limiter::DEFAULT_RATIO.label());
            egui::ComboBox::from_id_salt("limit-ratio")
                .selected_text(match st.limit_ratio {
                    None => default_label.clone(),
                    Some(r) => r.label().to_string(),
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut st.limit_ratio, None, default_label.as_str());
                    for r in Ratio::ALL {
                        ui.selectable_value(&mut st.limit_ratio, Some(r), r.label());
                    }
                })
                .response
                .on_hover_text(
                    "Oversampling precision inside the limiter detector: higher catches sharp inter-sample \
                     overs cleanly so crisp highs never saturate.",
                );
        });
        // The slow stage's shape and how much of it the channels share. Both linking
        // controls live here rather than behind a menu, because the damage they prevent
        // (a stereo image that pumps) is only noticed after a render, and the fix is two
        // numbers.
        ui.horizontal(|ui| {
            ui.label("attack");
            let mut a = st.attack_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut a)
                        .range(
                            g6me_core::limiter::MIN_ATTACK_MS..=g6me_core::limiter::MAX_ATTACK_MS,
                        )
                        .speed(0.5)
                        .suffix(" ms"),
                )
                .on_hover_text(
                    "How fast the slow leveling stage ducks sustained loud bursts: slower preserves \
                     drum punch; faster softens continuous loudness earlier.",
                )
                .changed()
            {
                st.attack_ms = a;
            }
            ui.label("release");
            let mut r = st.release_ms;
            if ui
                .add(
                    egui::DragValue::new(&mut r)
                        .range(
                            g6me_core::limiter::MIN_RELEASE_MS..=g6me_core::limiter::MAX_RELEASE_MS,
                        )
                        .speed(2.0)
                        .suffix(" ms"),
                )
                .on_hover_text(
                    "How fast level recovers after a loud peak: too short sounds pumped/gritty on bass; \
                     longer feels transparent and smooth.",
                )
                .changed()
            {
                st.release_ms = r;
            }
            ui.label("link");
            let mut lt = st.link_transient_pct;
            if ui
                .add(
                    egui::DragValue::new(&mut lt)
                        .range(0.0..=100.0)
                        .speed(1.0)
                        .suffix("% transient"),
                )
                .on_hover_text(
                    "How much L and R duck fast peaks together: high linking keeps centered kicks/snares \
                     dead-center without pulling sideways.",
                )
                .changed()
            {
                st.link_transient_pct = lt;
            }
            let mut lr = st.link_release_pct;
            if ui
                .add(
                    egui::DragValue::new(&mut lr)
                        .range(0.0..=100.0)
                        .speed(1.0)
                        .suffix("% release"),
                )
                .on_hover_text(
                    "How much L and R share slow recovery: keeping this moderate prevents loud hits on \
                     one side from dipping the quiet opposite ear.",
                )
                .changed()
            {
                st.link_release_pct = lr;
            }
        });
        let cfg = LCfg {
            enable: true,
            ceiling_db: st.setpoint_db.unwrap_or(st.ceiling_db),
            ratio: st.limiter_ratio(),
            lookahead_ms: st.lookahead_ms,
            attack_ms: st.attack_ms,
            release_ms: st.release_ms,
            link_transient: st.link_transient_pct / 100.0,
            link_release: st.link_release_pct / 100.0,
        };
        let lat = cfg.latency_samples(rate);
        let asked = cfg.lookahead_ms;
        let used = cfg.effective_lookahead_ms(rate);
        ui.weak(format!(
            "two stages: the lookahead one catches peaks, then a level one with a {:.0} ms \
             attack and {:.0} ms release rides under it. Peaks are ducked in both channels \
             {:.0}% together, level {:.0}%.",
            cfg.attack_ms, cfg.release_ms, st.link_transient_pct, st.link_release_pct
        ));
        // The ceiling named in this sentence has to be the one the box is set to. It used to
        // print a literal "-1.00 dBTP" beside a number calculated from the actual ceiling, so
        // at the default -3.00 it read "a -1.00 dBTP ceiling lands near -3.62 dBTP" — one
        // claim, two ceilings, and a shortfall that belonged to the meter's 4x rather than to
        // the 8x on offer. Everything here is now read off the config the render will use.
        ui.weak(format!(
            "aims {:.2} dB below the ceiling ({:.2} dB the meter cannot see + {:.2} dB of \
             gain slew), so the {:.2} dBTP ceiling set above lands near {:.2} dBTP.",
            cfg.guard_db(),
            cfg.ratio.worst_underread_db(),
            cfg.guard_db() - cfg.ratio.worst_underread_db(),
            cfg.ceiling_db,
            cfg.ceiling_db - cfg.guard_db()
        ));
        ui.weak(format!(
            "latency {} samples ({:.3} ms) at {} Hz, prepended at the start of the file. {}",
            lat,
            lat as f64 / rate as f64 * 1000.0,
            rate,
            if used > asked + 1e-6 {
                format!(
                    "{} ms lookahead asked for is shallower than the {} filter's own memory \
                     ({} samples), so {:.2} ms is used.",
                    asked,
                    cfg.ratio.label(),
                    cfg.ratio.group_delay().ceil() as usize,
                    used
                )
            } else {
                format!("{:.2} ms of lookahead is honoured as asked.", used)
            }
        ));
        if cfg.ratio != st.tp_ratio {
            ui.weak(format!(
                "Meter and limiter are running different ratios, so each measures once \
                 more than they would agree: the report's dBTP is the {} reading, the \
                 ceiling is held by the {} one.",
                st.tp_ratio.label(),
                cfg.ratio.label()
            ));
        }
        if let Err(e) = cfg.validate() {
            ui.colored_label(egui::Color32::from_rgb(226, 110, 90), e);
        }
    }

    fn tab_resample(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        let soxr = caps.map(|c| c.soxr).unwrap_or(true);
        if !soxr {
            ui.colored_label(
                egui::Color32::from_rgb(226, 140, 90),
                "This ffmpeg has no libsoxr, so resampling is unavailable. Everything else \
                 still works, and the default path is resampleless anyway.",
            );
            ui.add_enabled_ui(false, |ui| {
                self.resample_controls(ui);
            });
        } else {
            self.resample_controls(ui);
        }
    }

    fn resample_controls(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.settings.resample,
            "resample the input before the downmix (libsoxr)",
        )
        .on_hover_text(
            "Converts source sample rate before binaural filtering so crossfeed, phase, and \
             DFEQ timbre stay true on 44.1 / 96 / 192 kHz files.",
        );
        ui.weak(
            "Rate conversion happens before the downmix, never after: the filter coefficients \
             model the hardware at its own rate, so resampling the stereo result would be a \
             different curve.",
        );
        ui.add_enabled_ui(self.settings.resample, |ui| {
            ui.horizontal(|ui| {
                ui.label("target rate");
                let mut r = self.settings.rate;
                if ui
                    .add(
                        egui::DragValue::new(&mut r)
                            .speed(100)
                            .range(8_000..=384_000),
                    )
                    .on_hover_text(
                        "48000 Hz matches native Reference G6 / Dolby Atmos filter timing cleanly; \
                         higher rates extend ultrasonic bandwidth.",
                    )
                    .changed()
                {
                    self.settings.rate = r;
                }
                ui.label("Hz");
                for preset in [44_100u32, 48_000, 88_200, 96_000, 192_000] {
                    if ui
                        .selectable_label(self.settings.rate == preset, format!("{preset}"))
                        .clicked()
                    {
                        self.settings.rate = preset;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.settings.use_resample_depth, "quantize to")
                    .on_hover_text(
                        "Truncates resampled audio to 16 or 32 bits before downmixing. Keep off (32-bit \
                         float internal) for cleanest headroom and lowest noise.",
                    );
                egui::ComboBox::from_id_salt("rsdepth")
                    .selected_text(self.settings.resample_depth.label())
                    .show_ui(ui, |ui| {
                        for d in [BitDepth::S16, BitDepth::S32] {
                            ui.selectable_value(&mut self.settings.resample_depth, d, d.label());
                        }
                    });
                ui.weak(g6me_core::pcm::BitDepth::quantize_note());
            });
            ui.horizontal(|ui| {
                ui.label("soxr precision");
                let mut p = self.settings.precision;
                if ui
                    .add(
                        egui::Slider::new(&mut p, 15.0..=33.0)
                            .text("bits")
                            .step_by(1.0),
                    )
                    .on_hover_text(
                        "Controls anti-aliasing steepness and stopband depth: 33 bits virtually \
                         eliminates treble foldback/aliasing glare.",
                    )
                    .changed()
                {
                    self.settings.precision = p;
                }
                ui.checkbox(&mut self.settings.cheby, "Chebyshev passband")
                    .on_hover_text(
                        "Trades tiny passband ripple for a steeper top-octave cutoff. Leave off for \
                         flattest frequency and phase linearity.",
                    );
            });
            ui.horizontal(|ui| {
                ui.label("resample dither");
                egui::ComboBox::from_id_salt("resample-dither")
                    .selected_text(self.settings.resample_dither.label())
                    .show_ui(ui, |ui| {
                        for d in Dither::ALL {
                            ui.selectable_value(&mut self.settings.resample_dither, d, d.label());
                        }
                    })
                    .response
                    .on_hover_text(
                        "Dither applied inside the input resampler when quantizing before the downmix \
                         (separate from output dither on the Output tab).",
                    );
                if self.settings.resample_dither != Dither::None {
                    ui.weak("deterministic: the same input gives the same output every run");
                }
            });
        });
    }

    fn tab_ir(&mut self, ui: &mut egui::Ui, caps: Option<&Caps>) {
        let afir = caps.map(|c| c.afir).unwrap_or(true);
        if !afir {
            ui.colored_label(
                egui::Color32::from_rgb(226, 140, 90),
                "This ffmpeg has no afir filter, so IR calibration is unavailable.",
            );
        }
        ui.checkbox(
            &mut self.settings.ir,
            "convolve every input channel with a headphone correction IR",
        )
        .on_hover_text(
            "Applies your headphone/room correction EQ impulse response (e.g. AutoEQ / HyprEQ) to \
             every speaker before binaural downmixing.",
        );
        ui.weak(
            "Same shape mpv uses: mono IR, applied to all input channels before the downmix, \
             with no normalisation by default so a measured curve keeps its measured level.",
        );
        ui.add_enabled_ui(self.settings.ir, |ui| {
            ui.horizontal(|ui| {
                ui.label("IR file");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.ir_file_text)
                        .desired_width((ui.available_width() - 96.0).max(160.0))
                        .font(egui::FontId::monospace(11.0))
                        .hint_text("paste an IR .wav path or click …"),
                );
                if resp.changed() {
                    self.settings.ir_file = widgets::clean_pasted_path(&self.ir_file_text);
                }
                if ui
                    .button("…")
                    .on_hover_text("Choose impulse response file")
                    .clicked()
                {
                    let ctx = ui.ctx().clone();
                    self.open_picker(&ctx, PickTarget::Ir, Mode::AnyFile);
                }
                #[cfg(feature = "native-dialogs")]
                if ui
                    .button("in-app")
                    .on_hover_text("Open built-in file browser")
                    .clicked()
                {
                    self.open_in_app_picker(PickTarget::Ir, Mode::AnyFile);
                }
            });
            ui.horizontal(|ui| {
                ui.label("irnorm");
                let mut n = self.settings.ir_norm;
                if ui
                    .add(egui::DragValue::new(&mut n).speed(0.05).range(-1.0..=2.0))
                    .on_hover_text(
                        "Rescales the IR's loudness before convolving. Leave at -1 for measured EQ \
                         curves so target bass/treble gain stays calibrated.",
                    )
                    .changed()
                {
                    self.settings.ir_norm = n;
                }
                if ui
                    .selectable_label(self.settings.ir_norm == -1.0, "none (-1)")
                    .on_hover_text("Disables IR loudness rescaling so exact filter gain is preserved.")
                    .clicked()
                {
                    self.settings.ir_norm = -1.0;
                }
                ui.weak("(-1 = raw level, what a measured filter needs)");
            });
            ui.horizontal(|ui| {
                ui.label("wet");
                let mut w = self.settings.ir_wet;
                if ui
                    .add(egui::Slider::new(&mut w, 0.0..=2.0).step_by(0.05))
                    .on_hover_text(
                        "Controls how strongly the EQ/room IR shapes the sound (1.00 = full effect).",
                    )
                    .changed()
                {
                    self.settings.ir_wet = w;
                }
                ui.checkbox(&mut self.settings.use_ir_dry, "dry")
                    .on_hover_text(
                        "Blends unprocessed direct signal back in to tame extreme headphone EQ or reverb.",
                    );
                if self.settings.use_ir_dry {
                    let mut d = self.settings.ir_dry;
                    ui.add(egui::Slider::new(&mut d, 0.0..=2.0).step_by(0.05))
                        .on_hover_text("Dry blend level alongside the convolved wet signal.");
                    self.settings.ir_dry = d;
                }
            });
            ui.horizontal(|ui| {
                ui.label("graph type");
                let mut cur = self.settings.ir_gtype;
                egui::ComboBox::from_id_salt("gtype")
                    .selected_text(match cur {
                        None => "default".to_string(),
                        Some(g) => g.ffmpeg_name().to_string(),
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut cur, None, "default");
                        for g in [
                            Gtype::Peak,
                            Gtype::Dc,
                            Gtype::Gn,
                            Gtype::Ac,
                            Gtype::Rms,
                            Gtype::None,
                        ] {
                            ui.selectable_value(&mut cur, Some(g), g.ffmpeg_name());
                        }
                    })
                    .response
                    .on_hover_text(
                        "Selects how ffmpeg normalizes overall IR gain (dc/rms/peak/none) when irnorm >= 0.",
                    );
                self.settings.ir_gtype = cur;
                ui.label("partition");
                let mut mn = self.settings.ir_minp;
                let mut mx = self.settings.ir_maxp;
                if ui
                    .add(egui::DragValue::new(&mut mn).range(64..=65536))
                    .on_hover_text("FFT block partition size in ffmpeg afir (speed only; sonically identical).")
                    .changed()
                {
                    self.settings.ir_minp = mn;
                }
                ui.label("…");
                if ui
                    .add(egui::DragValue::new(&mut mx).range(64..=65536))
                    .on_hover_text("Maximum FFT partition size in ffmpeg afir (speed only; sonically identical).")
                    .changed()
                {
                    self.settings.ir_maxp = mx;
                }
            });
            ui.horizontal(|ui| {
                ui.label("convolution length");
                let mut l = self.settings.ir_length;
                // A fraction, because that is the unit FFmpeg's afir defines. The old
                // 0..8 "seconds" control happily produced values afir refuses outright
                // ("out of range [0 - 1]", exit 222), which surfaced as a failed render
                // with a filter-graph message that never mentioned this control.
                ui.add(
                    egui::DragValue::new(&mut l)
                        .speed(0.02)
                        .range(0.0..=1.0)
                        .fixed_decimals(2),
                )
                .on_hover_text(
                    "Fraction of the IR tail used (0 or 1 = whole IR). Shortening truncates low-bass \
                     resolution and room decay.",
                );
                self.settings.ir_length = l;
                ui.weak("(fraction of the IR; 0 or 1 = whole)");
            });
            ui.weak(
                "Convolution runs in double precision, fixed — the mpv equivalent of \
                 precision=double.",
            );
        });
    }

    fn tab_advanced(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("channel order");
            let before = self.settings.order_text.clone();
            ui.add_sized(
                [ui.available_width() - 90.0, 0.0],
                egui::TextEdit::singleline(&mut self.settings.order_text)
                    .font(egui::FontId::monospace(11.0))
                    .hint_text("leave empty to trust the decoder"),
            );
            if before != self.settings.order_text {
                if let Err(e) = self.settings.apply_order_text() {
                    self.status = e;
                }
            }
        });
        ui.weak(format!(
            "known names: {}",
            mask::format_order(&mask::Slot::ALL)
        ));
        if let Some(o) = &self.settings.order {
            ui.label(format!(
                "override in effect: {} → mask {}",
                mask::format_order(o),
                mask::format_mask_hex(mask::mask_of(o))
            ));
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("block");
            let mut b = self.settings.block;
            if ui
                .add(egui::DragValue::new(&mut b).range(64..=262_144))
                .changed()
            {
                self.settings.block = b;
            }
            ui.label("frames · queue");
            let mut q = self.settings.queue_blocks;
            if ui
                .add(egui::DragValue::new(&mut q).range(1..=256))
                .changed()
            {
                self.settings.queue_blocks = q;
            }
            ui.label("blocks");
            if ui
                .selectable_label(
                    self.settings.block == 4096 && self.settings.queue_blocks == 16,
                    "default",
                )
                .clicked()
            {
                self.settings.block = 4096;
                self.settings.queue_blocks = 16;
            }
        });
        ui.weak(format!(
            "peak memory ≈ {} per file (decode and encode overlap the downmixer; the \
             downmixer itself stays single-threaded)",
            util::human_bytes(self.settings.peak_bytes_per_file())
        ));
        ui.horizontal(|ui| {
            ui.label("audio stream");
            let mut s = self.settings.stream;
            ui.add(egui::DragValue::new(&mut s).range(0..=31));
            self.settings.stream = s;
            ui.weak("(0 = first audio stream)");
        });
        ui.separator();
        let running = self.runner.is_running();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!running, egui::Button::new("reset settings to defaults"))
                .on_hover_text(
                    "Restores Renderer, Output, True peak, Resample, IR, and Advanced controls to \
                     defaults while keeping your queued files and output folder.",
                )
                .clicked()
            {
                self.settings.reset_preserving_queue();
                self.out_dir_text = self.settings.out_dir.display().to_string();
                self.ir_file_text = self.settings.ir_file.display().to_string();
                self.rebuild_queue();
                self.push_log(Level::Info, "reset settings to defaults (queue preserved)");
            }
            ui.weak("shortcuts: Ctrl+Enter start · Esc cancel · Ctrl+D dry run");
        });
        ui.separator();
        ui.weak(
            "Decode, downmix and encode run as three concurrent stages per file. Filter \
             threads inside ffmpeg are pinned to one when more than one file runs at once, \
             because N files × M ffmpeg threads is how a batch ends up slower than one thread.",
        );
    }

    fn log_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("Log");
            if ui.small_button("clear").clicked() {
                self.log.clear();
            }
            ui.checkbox(&mut self.log_scroll, "follow");
        });
        egui::ScrollArea::vertical()
            .id_salt("log")
            .max_height(140.0)
            .stick_to_bottom(self.log_scroll)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (lvl, msg) in &self.log {
                    let col = match lvl {
                        Level::Info => egui::Color32::from_gray(180),
                        Level::Warn => egui::Color32::from_rgb(226, 180, 90),
                        Level::Error => egui::Color32::from_rgb(226, 120, 100),
                    };
                    ui.label(egui::RichText::new(msg).color(col).monospace().size(11.0));
                }
                if self.log.is_empty() {
                    ui.weak("(nothing logged yet)");
                }
            });
    }

    fn windows(&mut self, ctx: &egui::Context) {
        // Built-in browser window: always available as fallback (or via "in-app…") for
        // minimal Wayland/X11 sessions without an XDG desktop portal.
        let pick = self.pick.take();
        if let Some(p) = pick {
            let title = match p {
                PickTarget::Queue => "Add to queue",
                PickTarget::OutDir => "Choose output folder",
                PickTarget::Ir => "Choose impulse response",
            };
            let mut open = true;
            let mut done: Option<Vec<PathBuf>> = None;
            egui::Window::new(title)
                .open(&mut open)
                .default_pos([220.0, 160.0])
                .default_size([520.0, 420.0])
                .show(ctx, |ui| {
                    self.browser.ui(ui, |paths| done = Some(paths));
                });
            self.pick = if open { Some(p) } else { None };
            if let Some(paths) = done {
                self.apply_picked(p, paths, "added");
            }
        }

        if self.show_caps {
            let caps = self.caps();
            let mut open = self.show_caps;
            egui::Window::new("Capabilities")
                .open(&mut open)
                .show(ctx, |ui| match &caps {
                    None => {
                        ui.weak("still probing…");
                    }
                    Some(c) => {
                        ui.label(format!("ffmpeg {}", c.ffmpeg_version));
                        for (k, v) in [
                            ("libsoxr resampler", c.soxr),
                            ("afir convolution", c.afir),
                            ("FLAC encode", c.flac),
                            ("lavfi synthetic sources", c.lavfi),
                            ("24-bit FLAC round-trip", c.flac24),
                        ] {
                            let (mark, col) = if v {
                                ("available", egui::Color32::from_rgb(120, 190, 120))
                            } else {
                                ("missing", egui::Color32::from_rgb(226, 120, 100))
                            };
                            ui.horizontal(|ui| {
                                ui.label(k);
                                ui.colored_label(col, mark);
                            });
                        }

                        for n in &c.notes {
                            ui.weak(n.clone());
                        }
                    }
                });
            self.show_caps = open;
        }

        let dr = self.dry_run.lock().ok().and_then(|d| d.clone());
        if let Some(dr) = dr {
            let mut open = true;
            egui::Window::new(format!("Dry run — {}", widgets::short_path(&dr.src, 40)))
                .open(&mut open)
                .default_pos([260.0, 200.0])
                .default_size([680.0, 420.0])
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            if let Some(e) = &dr.error {
                                ui.colored_label(egui::Color32::from_rgb(226, 120, 100), e.clone());
                            }
                            for l in &dr.lines {
                                ui.label(egui::RichText::new(l).monospace().size(11.0));
                            }
                        });
                });
            if !open {
                if let Ok(mut d) = self.dry_run.lock() {
                    *d = None;
                }
            }
        }
    }

    // ---- actions -----------------------------------------------------------

    /// Open a native Windows/Linux/macOS OS file/folder dialog on a background
    /// thread so the UI never blocks; falls back to the in-app browser when built
    /// with `--no-default-features`.
    fn open_picker(&mut self, ctx: &egui::Context, target: PickTarget, mode: Mode) {
        #[cfg(feature = "native-dialogs")]
        {
            let tx = self.runner.sender();
            let ctx = ctx.clone();
            let start_dir: Option<PathBuf> = match target {
                PickTarget::OutDir if self.settings.out_dir.is_dir() => {
                    Some(self.settings.out_dir.clone())
                }
                PickTarget::Ir => self
                    .settings
                    .ir_file
                    .parent()
                    .filter(|p| p.is_dir())
                    .map(PathBuf::from),
                _ => None,
            };
            std::thread::spawn(move || {
                let mut dlg = rfd::FileDialog::new();
                if let Some(dir) = start_dir {
                    dlg = dlg.set_directory(dir);
                }
                let paths: Vec<PathBuf> = match (target, mode) {
                    (PickTarget::Queue, Mode::AudioFile) => dlg
                        .set_title("Add audio files to queue")
                        .add_filter(
                            "Audio / Media",
                            &[
                                "mka", "mkv", "wav", "flac", "mlp", "thd", "ac3", "eac3", "dts",
                                "m4a", "mp4", "ogg", "opus", "wv", "aiff", "aif", "caf",
                            ],
                        )
                        .add_filter("All files", &["*"])
                        .pick_files()
                        .unwrap_or_default(),
                    (_, Mode::Dir) => dlg
                        .set_title(if target == PickTarget::OutDir {
                            "Choose output folder"
                        } else {
                            "Add folder to queue"
                        })
                        .pick_folder()
                        .into_iter()
                        .collect(),
                    (PickTarget::Ir, _) => dlg
                        .set_title("Choose impulse response")
                        .add_filter("Impulse Response / Audio", &["wav", "flac", "aiff", "aif"])
                        .add_filter("All files", &["*"])
                        .pick_file()
                        .into_iter()
                        .collect(),
                    _ => dlg.pick_files().unwrap_or_default(),
                };
                if !paths.is_empty() {
                    let _ = tx.send(Event::Picked { target, paths });
                    ctx.request_repaint();
                }
            });
        }
        #[cfg(not(feature = "native-dialogs"))]
        {
            let _ = ctx;
            self.open_in_app_picker(target, mode);
        }
    }

    fn open_in_app_picker(&mut self, target: PickTarget, mode: Mode) {
        self.pick = Some(target);
        self.browser.mode = mode;
        match target {
            PickTarget::Queue => self.browser.set_single(None),
            PickTarget::Ir => self.browser.set_single(Some(self.settings.ir_file.clone())),
            PickTarget::OutDir => {}
        }
    }

    fn apply_picked(&mut self, target: PickTarget, paths: Vec<PathBuf>, verb: &str) {
        match target {
            PickTarget::Queue => {
                let mut added = 0usize;
                for path in paths {
                    if !self.settings.inputs.contains(&path) {
                        self.settings.inputs.push(path);
                        added += 1;
                    }
                }
                if added > 0 {
                    self.push_log(
                        Level::Info,
                        format!(
                            "{verb} {added} {} to the queue",
                            if added == 1 { "item" } else { "items" }
                        ),
                    );
                }
            }
            PickTarget::OutDir => {
                if let Some(path) = paths.into_iter().next() {
                    self.settings.use_out_dir = true;
                    self.out_dir_text = path.display().to_string();
                    self.settings.out_dir = path;
                }
            }
            PickTarget::Ir => {
                if let Some(path) = paths.into_iter().next() {
                    self.ir_file_text = path.display().to_string();
                    self.settings.ir_file = path;
                }
            }
        }
        if target != PickTarget::Ir {
            self.rebuild_queue();
        }
    }

    /// Handle OS drag-and-drop of audio files or folders onto the window.
    fn handle_drops(&mut self, ctx: &egui::Context) {
        let running = self.runner.is_running();
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if hovering {
            let rect = ctx.screen_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop-overlay"),
            ));
            painter.rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(12, 18, 28, 190),
            );
            let inner = rect.shrink(28.0);
            let accent = if running {
                egui::Color32::from_rgb(226, 140, 90)
            } else {
                egui::Color32::from_rgb(76, 194, 232)
            };
            painter.rect_stroke(
                inner,
                12.0,
                egui::Stroke::new(2.0_f32, accent),
                egui::StrokeKind::Inside,
            );
            let msg = if running {
                "Render in progress — wait or cancel before dropping files"
            } else {
                "Drop audio files or folders to add to queue"
            };
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                msg,
                egui::FontId::proportional(20.0),
                egui::Color32::WHITE,
            );
        }

        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            if running {
                self.push_log(
                    Level::Warn,
                    "ignored dropped files while a render is already running",
                );
            } else {
                self.apply_picked(PickTarget::Queue, dropped, "dropped");
            }
        }
    }

    fn rebuild_queue(&mut self) {
        let cfg = self.settings.batch_cfg();
        let ext = cfg.render.out.format.extension().to_string();
        let items = batch::discover(&cfg.inputs, cfg.out_dir.as_deref(), &ext, cfg.recursive);
        self.queue = items
            .iter()
            .map(|i| Row {
                src: i.src.clone(),
                dst: i.dst.clone(),
                phase: i.phase,
                note: String::new(),
                collision: i.note.clone().unwrap_or_default(),
            })
            .collect();
        let skipped = self
            .queue
            .iter()
            .filter(|r| r.phase == Phase::Skipped)
            .count();
        self.status = format!(
            "{} file(s) queued{}",
            self.queue.len(),
            if skipped > 0 {
                format!(", {skipped} already have output")
            } else {
                String::new()
            }
        );
    }

    fn start_run(&mut self) {
        let explicit_items: Vec<batch::Item> = self
            .queue
            .iter()
            .map(|r| {
                let mut it = batch::Item::new(r.src.clone(), r.dst.clone());
                if !r.collision.is_empty() {
                    it.note = Some(r.collision.clone());
                }
                it
            })
            .collect();
        self.start_with_items(explicit_items, "start");
    }

    fn retry_failed(&mut self) {
        let failed_items: Vec<batch::Item> = self
            .queue
            .iter()
            .filter(|r| r.phase == Phase::Failed)
            .map(|r| {
                let mut it = batch::Item::new(r.src.clone(), r.dst.clone());
                if !r.collision.is_empty() {
                    it.note = Some(r.collision.clone());
                }
                it
            })
            .collect();
        if failed_items.is_empty() {
            self.status = "no failed files to retry".to_string();
            return;
        }
        self.start_with_items(failed_items, "retry failed");
    }

    fn start_with_items(&mut self, explicit_items: Vec<batch::Item>, verb: &str) {
        let caps = match self.caps() {
            Some(c) => c,
            None => {
                self.status = "still probing ffmpeg — one moment".to_string();
                return;
            }
        };
        let problems = self.settings.problems(&caps);
        if !problems.is_empty() {
            for p in &problems {
                self.push_log(Level::Error, p.clone());
            }
            self.summary = Some(Summary {
                error: Some(problems.join("; ")),
                ..Default::default()
            });
            self.status = "not started — fix the problems above".to_string();
            return;
        }
        let mut cfg = self.settings.batch_cfg();
        // Meters for the render itself: a card per file, made as a worker picks the
        // file up, fed every block that goes to the encoder.
        cfg.taps = self.meter_factory();
        self.cards.clear();
        self.summary = None;
        match self
            .runner
            .start_items(explicit_items, &cfg, self.tools.clone(), caps)
        {
            Ok(items) => {
                self.queue = items
                    .iter()
                    .map(|i| Row {
                        src: i.src.clone(),
                        dst: i.dst.clone(),
                        phase: i.phase,
                        note: String::new(),
                        collision: i.note.clone().unwrap_or_default(),
                    })
                    .collect();
                self.status = format!("rendering {} file(s)", items.len());
                self.push_log(
                    Level::Info,
                    format!(
                        "{verb} · {} · {} file(s)",
                        describe(&self.settings),
                        items.len()
                    ),
                );
            }
            Err(e) => {
                self.push_log(Level::Error, e.clone());
                self.summary = Some(Summary {
                    error: Some(e),
                    ..Default::default()
                });
                self.status = "not started".to_string();
            }
        }
    }

    /// Open the active output folder (or the first queued destination's parent) in the OS
    /// file manager on a detached thread so the UI never blocks.
    fn open_output_folder(&mut self) {
        let dir = if self.settings.use_out_dir {
            Some(self.settings.out_dir.clone())
        } else {
            self.queue
                .first()
                .and_then(|r| r.dst.parent().map(PathBuf::from))
        };
        let Some(dir) = dir.filter(|d| d.is_dir()) else {
            self.push_log(
                Level::Warn,
                "no existing output folder to open — queue a file or pick an output folder first",
            );
            return;
        };
        std::thread::spawn(move || {
            #[cfg(target_os = "linux")]
            let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(&dir).spawn();
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("explorer").arg(&dir).spawn();
        });
    }

    fn run_dry_run(&mut self) {
        let Some(first) = self.queue.first() else {
            self.status = "queue something first".to_string();
            return;
        };
        let Some(caps) = self.caps() else {
            self.status = "still probing ffmpeg".to_string();
            return;
        };
        let cfg = self.settings.render_cfg();
        let src = first.src.clone();
        let dst = first.dst.clone();
        let tools = self.tools.clone();
        let out = match pipeline::plan_file(&src, &dst, &cfg, &tools, &caps) {
            Ok(p) => {
                let mut lines = vec![
                    format!(
                        "source   {}{}  [{} ch, {} Hz, {}, {}]",
                        p.info.codec_name,
                        p.info
                            .profile
                            .as_deref()
                            .map(|x| format!(" ({x})"))
                            .unwrap_or_default(),
                        p.info.channels,
                        p.info.sample_rate,
                        p.info.layout.clone().unwrap_or_else(|| "?".into()),
                        p.info
                            .bits
                            .map(|b| format!("{b}-bit"))
                            .unwrap_or_else(|| "? bit".into()),
                    ),
                    format!(
                        "order    {}  mask {}  stage {}",
                        mask::format_order(&p.order),
                        mask::format_mask_hex(p.mask),
                        p.stage.label()
                    ),
                    format!("rate     {} Hz working", p.working_rate),
                    String::new(),
                    "chain".to_string(),
                ];
                for step in &p.chain {
                    lines.push(format!("  {step}"));
                }
                lines.push(String::new());
                lines.push("decode".to_string());
                lines.push(format!("  {}", p.input_args.join(" ")));
                lines.push(String::new());
                lines.push("encode".to_string());
                lines.push(format!("  {}", p.encode_cmd.join(" ")));
                for w in &p.warnings {
                    lines.push(format!("warning: {w}"));
                }
                DryRun {
                    src,
                    lines,
                    error: None,
                }
            }
            Err(e) => DryRun {
                src,
                lines: vec![
                    "This file cannot be rendered with the current settings.".to_string(),
                    String::new(),
                    format!("  {e}"),
                ],
                error: Some(e.to_string()),
            },
        };
        if let Ok(mut d) = self.dry_run.lock() {
            *d = Some(out);
        }
        self.status = "dry run".to_string();
    }

    /// One meter per file, handed to the batch as it starts each one.
    ///
    /// This is the whole reason the app can show levels without making a sound: the
    /// render thread passes each block to the meter on its way to the encoder, so the
    /// picture is the same arithmetic that is being written to disk — not a parallel
    /// "just measure it" path that can drift away from it.
    fn meter_factory(&self) -> g6me_core::batch::MeterFactory {
        let cards = self.cards.clone();
        g6me_core::batch::MeterFactory(Some(std::sync::Arc::new(move |index: usize| {
            // No redraw hook here on purpose: the window already runs a 20 fps repaint
            // clock while a batch is live (see `update`), and asking for a repaint on
            // every block — a file goes through ~400 of them a second at speed — would
            // only spin the UI faster than the eye can read.
            let tap: std::sync::Arc<dyn g6me_core::pipeline::SampleTap> =
                std::sync::Arc::new(MeterTap::new(cards.open(index)));
            Some(tap)
        })))
    }

    fn drain_events(&mut self) {
        for e in self.runner.drain() {
            match e {
                Event::ItemState {
                    index,
                    phase,
                    note,
                    report,
                } => {
                    if let Some(r) = self.queue.get_mut(index) {
                        r.phase = phase;
                        r.note = note.clone();
                    }
                    if let Some(r) = report {
                        self.history.push(MeterRow::from_report(&r));
                        // A library is thousands of files; the list is a record of the
                        // session, not a database. Oldest rows go first.
                        const HISTORY_CAP: usize = 4000;
                        if self.history.len() > HISTORY_CAP {
                            self.history.remove(0);
                        }
                    }
                    if phase == Phase::Failed {
                        self.push_log(Level::Error, note);
                    }
                    // A finished file's card stays on screen, dimmed, with its last
                    // peaks on it: the clipping you missed while it ran is still worth
                    // seeing after it stopped. It goes when the next run starts.
                    if matches!(phase, Phase::Done | Phase::Failed | Phase::Skipped) {
                        self.cards.finish(index);
                    }
                }
                Event::Finished(s) => {
                    self.push_log(
                        if s.failed > 0 {
                            Level::Warn
                        } else {
                            Level::Info
                        },
                        s.one_line(),
                    );
                    self.status = "idle".to_string();
                    self.summary = Some(s);
                }
                Event::Caps(caps, note) => {
                    // Handling the probe here rather than in a callback keeps every UI
                    // state change on the UI thread.
                    self.status = note.clone();
                    if let Ok(mut c) = self.caps.lock() {
                        *c = Some(caps);
                    }
                    // Now that the caps are known, the queue can be scanned: which
                    // files already have an output depends on the chosen extension.
                    self.rebuild_queue();
                }
                Event::Picked { target, paths } => {
                    if !self.runner.is_running() {
                        self.apply_picked(target, paths, "added");
                    }
                }
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

/// Note shown next to a bit-depth choice, or `None` when there is nothing to warn
/// about.
fn depth_note(d: BitDepth, f: Format, caps: Option<&Caps>) -> Option<&'static str> {
    if f != Format::Flac {
        return None;
    }
    // FLAC is an integer container and ffmpeg's encoder stops at 24 bits, so these
    // combinations are refused by the validator rather than quietly downgraded.
    if d.is_float() {
        return Some("not available in FLAC — use WAV");
    }
    if d == BitDepth::S32 {
        return Some("not available in FLAC — use WAV");
    }
    if d == BitDepth::S24 && caps.map(|c| !c.flac24).unwrap_or(false) {
        return Some("unverified on this ffmpeg");
    }
    None
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();
        // Global keyboard shortcuts when no text input has focus.
        if !ctx.wants_keyboard_input() {
            let (start_key, dry_key, esc_key) = ctx.input(|i| {
                let cmd = i.modifiers.command || i.modifiers.ctrl;
                (
                    cmd && i.key_pressed(egui::Key::Enter),
                    cmd && i.key_pressed(egui::Key::D),
                    i.key_pressed(egui::Key::Escape),
                )
            });
            let running = self.runner.is_running();
            if esc_key && running {
                self.runner.cancel();
                self.status = "cancelling…".to_string();
            } else if start_key && !running && self.caps().is_some() && !self.queue.is_empty() {
                self.start_run();
            } else if dry_key && !running && !self.queue.is_empty() {
                self.run_dry_run();
            }
        }
        // The palette is installed on a change, not every frame: `apply` rewrites
        // egui's stored styles, and doing that sixty times a second would throw away
        // the caching it keeps them for.
        if self.installed != Some(self.palette) {
            self.palette.apply(ctx);
            self.installed = Some(self.palette);
        }
        self.top_panel(ctx);
        self.bottom_panel(ctx);
        self.queue_panel(ctx);
        self.central_panel(ctx);
        self.windows(ctx);
        self.handle_drops(ctx);
        // The meters decay on the render thread, so a live meter needs a redraw
        // clock. 20 fps is enough to read and cheap enough to leave running; when
        // nothing is active the window goes quiet.
        if self.runner.is_running() {
            self.last_repaint = std::time::Instant::now();
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if self.last_repaint.elapsed() < std::time::Duration::from_millis(400) {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 820.0])
            .with_min_inner_size([900.0, 620.0])
            .with_title("G6 Media Encoder"),
        // Glow, not Wgpu: this has to come up on an old Windows GPU driver and on a
        // Wayland compositor with no hardware acceleration, and it does so on GL.
        ..Default::default()
    };
    eframe::run_native(
        "G6 Media Encoder",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real report from a real render, decoded back.
    ///
    /// A fixture rather than a literal, for two reasons: `Report` has thirty fields and a
    /// hand-written one would be all noise, and this way a field that stops reaching the
    /// panel — a rename that drops DC, or the decoded-back pass going unreported — fails a
    /// test instead of quietly vanishing from the copy text.
    fn rendered() -> Report {
        let raw = include_str!("../testdata/clamped.json");
        match serde_json::from_str::<Report>(raw) {
            Ok(r) => r,
            Err(e) => panic!("the fixture no longer matches Report: {e}"),
        }
    }

    /// The meters decay to rest inside a third of a second. What a finished file is
    /// remembered by has to be the numbers, not the memory of a bar.
    #[test]
    fn a_finished_file_keeps_its_numbers_after_the_render() {
        let r = rendered();
        assert!(r.delivered.is_some(), "fixture must carry the decode-back");
        let row = MeterRow::from_report(&r);
        for want in ["tp ", "dBTP", "peak ", "rms "] {
            assert!(
                row.levels.contains(want),
                "the row lost {want}: {}",
                row.levels
            );
        }
        assert!(
            row.levels.contains(".wav"),
            "a row with no file: {}",
            row.levels
        );
        // The full text is what "copy all" hands to a log or a delivery note, so it carries
        // the parts that make a number checkable: the detector ratio, the DC, and the file's
        // own measurement next to the stream's.
        for want in ["@x4", "dc:", "file on disk", "WARNING"] {
            assert!(row.text.contains(want), "the copy text lost {want}");
        }
        // …and the compact row stays quiet about an offset this file does not have: 1.9e-4
        // is measured, reported in the text above, and not a finding.
        assert!(
            !row.levels.contains("dc "),
            "an immaterial offset took the compact row: {}",
            row.levels
        );
    }

    /// A row that reads calm about a file whose own decoded measurement disagrees with its
    /// report is a row that will be trusted anyway.
    #[test]
    fn a_file_disagreeing_with_its_own_report_shows_as_hot() {
        let r = rendered();
        assert!(r.delivered_note.is_some(), "fixture must disagree");
        assert!(
            MeterRow::from_report(&r).hot,
            "a clamped file is listed as calm"
        );

        let mut calm = r.clone();
        calm.delivered_note = None;
        calm.true_peak_dbtp = -20.0;
        calm.limiter = None;
        assert!(
            !MeterRow::from_report(&calm).hot,
            "a cold file is flagged hot, which trains the colour to be ignored"
        );
    }

    /// The dB figure with no logarithm still has to print as itself.
    #[test]
    fn db_prints_the_value_that_has_no_logarithm() {
        assert_eq!(db(f64::NEG_INFINITY), "\u{2212}inf");
        assert_eq!(db(f64::NAN), "n/a");
        assert_eq!(db(-3.0), "-3.00");
    }

    #[test]
    fn meter_row_includes_loudness_metrics_and_exports_csv_and_json() {
        let mut r1 = rendered();
        r1.output = PathBuf::from("/out/track, \"1\".flac");
        r1.true_peak_dbtp = -1.2;
        r1.delivered_note = None;
        r1.loudness = Some(LoudnessReport {
            integrated_lufs: Some(-14.3),
            momentary_max_lufs: Some(-10.1),
            short_term_max_lufs: Some(-12.0),
            lra_lu: Some(6.4),
            plr_db: Some(13.1),
            correlation: 0.87,
        });
        let row1 = MeterRow::from_report(&r1);
        for want in ["I -14.3 LUFS", "LRA 6.4 LU", "PLR 13.1 dB", "ρ +0.87"] {
            assert!(
                row1.levels.contains(want),
                "missing {want} in levels: {}",
                row1.levels
            );
        }
        assert!(row1.text.contains("loudness (BS.1770-5)"));

        let mut r2 = rendered();
        r2.output = PathBuf::from("/out/newer.flac");
        r2.true_peak_dbtp = -0.2;
        r2.loudness = Some(LoudnessReport {
            integrated_lufs: Some(-11.0),
            ..Default::default()
        });
        let row2 = MeterRow::from_report(&r2);
        let history = vec![row1, row2];

        let csv = history_csv(&history, DeliveryTarget::Streaming);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 rows: {csv}");
        assert!(lines[0].starts_with("file,true_peak_dbtp,"));
        // Newest first (row2 before row1), with RFC 4180 quoting on row1's filename.
        assert!(lines[1].starts_with("newer.flac,"));
        assert!(lines[1].ends_with(",OVER TP"));
        assert!(lines[2].starts_with("\"track, \"\"1\"\".flac\","));
        assert!(lines[2].ends_with(",PASS"));

        let json_str = history_json(&history, DeliveryTarget::Streaming);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
        let arr = parsed.as_array().expect("JSON array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["file"], "newer.flac");
        assert_eq!(arr[0]["compliance"], "OVER TP");
        assert_eq!(arr[1]["file"], "track, \"1\".flac");
        assert_eq!(arr[1]["compliance"], "PASS");
    }

    #[test]
    fn queue_filter_and_search_match_expected_rows() {
        let rows = [
            Row {
                src: PathBuf::from("/music/Atmos_Track_01.mka"),
                dst: PathBuf::from("/out/Atmos_Track_01.flac"),
                phase: Phase::Done,
                note: "100 frames".into(),
                collision: String::new(),
            },
            Row {
                src: PathBuf::from("/music/Concert_Live.wav"),
                dst: PathBuf::from("/out/Concert_Live.flac"),
                phase: Phase::Failed,
                note: "decode error".into(),
                collision: String::new(),
            },
            Row {
                src: PathBuf::from("/music/Pending_Mix.flac"),
                dst: PathBuf::from("/out/Pending_Mix.flac"),
                phase: Phase::Running,
                note: String::new(),
                collision: String::new(),
            },
        ];
        assert!(row_matches_filter(&rows[0], QueueFilter::Done, "atmos"));
        assert!(!row_matches_filter(&rows[0], QueueFilter::Failed, "atmos"));
        assert!(row_matches_filter(&rows[1], QueueFilter::Failed, "DECODE"));
        assert!(row_matches_filter(&rows[2], QueueFilter::Queued, ""));
        assert!(!row_matches_filter(&rows[2], QueueFilter::Skipped, ""));
    }
}
