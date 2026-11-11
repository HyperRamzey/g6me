//! Error type for the whole crate.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ffmpeg not found: checked the G6ME_FFMPEG environment variable and PATH")]
    FfmpegMissing,

    #[error("ffprobe not found (needed to read {0})")]
    FfprobeMissing(PathBuf),

    #[error("ffprobe failed for {path}: {msg}")]
    ProbeFailed { path: String, msg: String },

    #[error("no audio stream in {0}")]
    NoAudio(PathBuf),

    #[error(
        "unsupported channel layout '{layout}' with {channels} channels on {} — \
         pass an explicit channel order instead of guessing",
        .path.display()
    )]
    UnsupportedLayout {
        path: PathBuf,
        layout: String,
        channels: usize,
    },

    #[error("stream has no front left/right pair (mask {0:#06x}); nothing to downmix")]
    NoFrontPair(u32),

    #[error("this ffmpeg build has no libsoxr, so resampling is unavailable — passthrough only")]
    NoSoxr,

    #[error("this ffmpeg build has no 'afir' filter, so IR calibration is unavailable")]
    NoAfir,

    #[error("cannot work with a file and its own output being the same file: {}", .0.display())]
    SameFile(PathBuf),

    #[error("IR file not found: {}", .0.display())]
    IrMissing(PathBuf),

    #[error(
        "could not tell which channel is which: layout '{}' with {channels} channels — \
         set an explicit channel order",
        .layout.as_deref().unwrap_or("none reported")
    )]
    MaskUnknown {
        layout: Option<String>,
        channels: usize,
    },

    #[error("channel order lists {given} channels but the stream has {channels}")]
    OrderMismatch { channels: usize, given: usize },

    #[error("output already exists: {} (use --force)", .0.display())]
    Exists(PathBuf),

    #[error("cannot read input {}: {msg}", .path.display())]
    InputUnreadable { path: PathBuf, msg: String },

    #[error("{stage} failed (exit {code}):\n{msg}")]
    ChildFailed {
        stage: &'static str,
        code: i32,
        msg: String,
    },

    #[error("{stage} produced no output — input may be shorter than one block or corrupt")]
    NoData { stage: &'static str },

    #[error("downmixer state went non-finite after {frames} frames — source contains NaN/Inf")]
    NonFinite { frames: u64 },

    #[error("cancelled")]
    Cancelled,

    #[error("{0}")]
    Msg(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[cfg(feature = "monitor")]
    #[error("audio device: {0}")]
    Device(String),
}

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Msg(s.into())
    }

    /// True for the conditions that should not stop a batch: a bad or
    /// already-done file is reported and the queue moves on.
    pub fn is_per_file(&self) -> bool {
        matches!(
            self,
            Error::NoAudio(_)
                | Error::ProbeFailed { .. }
                | Error::UnsupportedLayout { .. }
                | Error::MaskUnknown { .. }
                | Error::OrderMismatch { .. }
                | Error::IrMissing(_)
                | Error::NoFrontPair(_)
                | Error::Exists(_)
                | Error::InputUnreadable { .. }
                | Error::ChildFailed { .. }
                | Error::NoData { .. }
                | Error::NonFinite { .. }
        )
    }
}

pub type Result<T> = std::result::Result<T, Error>;
