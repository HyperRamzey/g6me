//! # g6me-core
//!
//! The engine behind **G6 Media Encoder** (G6ME): a Rust port of the G6
//! 7.1/5.1 → binaural-stereo matrix downmixer, plus the file and live-monitoring
//! machinery around it.
//!
//! ## The numerical contract
//!
//! [`dsp`] is a bit-exact port of `vendor/g6ct/g6ct.c`, which is itself a
//! bit-exact C port of `CMatrixEncoderEfxMod` from the original USB audio
//! `KSUSBAPO64.dll` (provenance in `vendor/g6ct/UPSTREAM.md`). That exactness is
//! the product: the
//! whole reason this exists is to reproduce the hardware's phase behaviour
//! byte-for-byte. So the DSP path forbids, by construction:
//!
//! * fused multiply-add (changes rounding in the recursive all-pass state)
//! * fast-math / reassociation / contraction (reorders the sum)
//! * denormal flushing (silently rewrites tiny state values)
//! * any vectorization that is not provably bit-identical
//!
//! [`dsp::Real`] is the sealed sample type: `f32` reproduces the Windows DLL and
//! VST bit-for-bit, `f64` is the same transfer function with less floating-point
//! noise, and [`dsp::assert_coefficients`] pins every coefficient's exact bit
//! pattern so a transcription slip fails the build instead of drifting phase.
//!
//! ## I/O
//!
//! Real containers are handled by ffmpeg in a subprocess ([`probe`],
//! [`stages`], [`pipeline`]) — that keeps TrueHD/Atmos decoding possible without
//! linking a GPL libav into this binary, and makes every optional stage visible
//! as a command line you can check with `--dry-run`. [`wav`] covers the few
//! places where spawning a process for a header is silly.
//!
//! Nothing is applied that the user did not ask for: by default the chain is
//! `decode → f64 → downmix → encode`, with resampling, bit-depth reduction,
//! dither and IR convolution each behind their own opt-in control
//! ([`stages::RenderCfg`], [`stages::loss_chain`]).
//!
//! ## Batch and monitoring
//!
//! [`batch`] runs a queue of files and folders with DSP single-threaded per file
//! and parallelism *across* files, which is what the recursive state requires.
//! [`monitor`] (feature `monitor`) plays a live output device with the same
//! meters and meter semantics as the VST insert ([`meter`]).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod batch;
pub mod child;
pub mod dsp;
pub mod error;
pub mod limiter;
pub mod mask;
pub mod meter;
pub mod pcm;
pub mod pipeline;
pub mod probe;
pub mod stage;
pub mod stages;
pub mod tp;
pub mod util;
pub mod wav;

#[cfg(feature = "monitor")]
pub mod monitor;

pub use error::{Error, Result};

/// Version of the vendored C reference this crate is verified against.
pub const UPSTREAM_REF: &str = "g6-re/port g6ct.c (bit-exact port of KSUSBAPO64.dll)";

/// Crate version, for the GUI/CLI about box.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
