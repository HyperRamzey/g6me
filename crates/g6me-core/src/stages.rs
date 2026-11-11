//! Stage configuration and the exact ffmpeg command lines each stage emits.
//!
//! Everything here is a pure function of the config, which buys two things:
//! `--dry-run` can print precisely what will run, and the tests can assert the
//! filter strings (a typo in `dither_method` makes ffmpeg fail every render, so
//! it belongs in a test).
//!
//! ## Defaults are loss-neutral on purpose
//!
//! With the stock configuration the pipeline is
//! `decode -> f64 -> downmix -> encode`, with no resampler, no dither and no
//! convolution anywhere: each of those steps is opt-in, one control each. The
//! input side asks ffmpeg for `pcm_f64le`, which is an exact widening of the
//! decoder's `s32` output (all 2^31 integers are representable in f64, and the
//! scale factor is a power of two), so the downmixer sees the source values
//! unchanged.
//!
//! ## Channel identity
//!
//! Every input chain starts with `aformat=channel_layouts=<mask hex>`, which
//! *declares* the layout of the decoded stream instead of letting swresample
//! guess it. `crate::tests::channel_identity` proves with an impulse per channel
//! that this neither reorders nor folds any channel.

use crate::error::{Error, Result};
use crate::limiter::Cfg as LimiterCfg;
use crate::mask::format_mask_hex;
use crate::pcm::{BitDepth, Dither};
use crate::probe::{AudioInfo, Caps};
use crate::tp::Ratio;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Input-side normalization: rate conversion and (optionally) a bit-depth
/// reduction *before* the downmixer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResampleCfg {
    pub enable: bool,
    pub rate: u32,
    /// `None` keeps the source precision. Restricted to what libswresample can
    /// express as a filter format — see [`BitDepth::quantize_note`].
    pub depth: Option<BitDepth>,
    /// soxr precision in bits, 15..=33.
    pub precision: f64,
    pub dither: Dither,
    pub cheby: bool,
}

impl Default for ResampleCfg {
    fn default() -> Self {
        Self {
            enable: false,
            rate: 48000,
            depth: None,
            precision: 33.0,
            dither: Dither::None,
            cheby: false,
        }
    }
}

impl ResampleCfg {
    /// Formats a caller may ask for on the input side. 24-bit is deliberately
    /// absent: swresample has no `s24` sample format, so 24-bit only exists at
    /// the encoder.
    pub fn allowed_depths() -> [BitDepth; 4] {
        [BitDepth::S16, BitDepth::S32, BitDepth::F32, BitDepth::F64]
    }

    fn effective_depth(&self) -> Option<BitDepth> {
        self.depth.filter(|d| Self::allowed_depths().contains(d))
    }
}

/// afir's IR auto-gain types, by the exact names ffmpeg parses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Gtype {
    None,
    Peak,
    Dc,
    Gn,
    Ac,
    Rms,
}

impl Gtype {
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Gtype::None => "none",
            Gtype::Peak => "peak",
            Gtype::Dc => "dc",
            Gtype::Gn => "gn",
            Gtype::Ac => "ac",
            Gtype::Rms => "rms",
        }
    }
}

/// `afir`'s `precision`, fixed to double.
///
/// The convolver and the downmixer both run their state and accumulation in f64,
/// and that is an invariant rather than a setting: a 200 ms IR convolved in f32
/// loses the detail a measured headphone correction exists to apply, and the
/// downmixer's recursive sections are bit-exact against the reference C port only
/// in its `G6CT_DOUBLE` (f64) build. Bit depth at the *output* is a user choice;
/// arithmetic inside the app is not. There is therefore no `float` option here.
pub const AFIR_PRECISION: &str = "double";

/// Linear-phase headphone correction, applied per input channel before the
/// downmixer — the same thing `mpv`'s
/// `af=lavfi=[amovie=ir.wav[ir2];[in][ir2]afir=...]` does.
///
/// The defaults reproduce that graph option-for-option
/// (`irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double`); knobs
/// left `None` are *omitted* from the filter string so ffmpeg's own defaults
/// apply, which is what keeps parity with an mpv config.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IrCfg {
    pub enable: bool,
    pub file: Option<PathBuf>,
    /// `irnorm`, -1..=2. `-1` = no auto gain (raw IR), which is what a measured
    /// correction filter needs.
    pub irnorm: Option<f64>,
    pub gtype: Option<Gtype>,
    pub wet: Option<f64>,
    pub dry: Option<f64>,
    pub length: Option<f64>,
    pub link: Option<bool>,
    pub minp: u32,
    pub maxp: u32,
}

impl Default for IrCfg {
    fn default() -> Self {
        Self {
            enable: false,
            file: None,
            irnorm: Some(-1.0),
            gtype: None,
            wet: Some(1.0),
            dry: None,
            length: None,
            link: None,
            minp: 4096,
            maxp: 4096,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutFormat {
    /// FLAC, written by ffmpeg's own encoder.
    ///
    /// Depth is 16 or 24 bit integer. ffmpeg's encoder advertises `s16 s32` as
    /// its input sample formats and derives the stream's bits-per-sample from what
    /// it is fed: fed 24-bit samples it writes a genuine `bps=24` stream, and
    /// decoding that stream returns the input bytes exactly (`flac -t` says ok).
    /// It cannot carry 32-bit PCM — fed true 32-bit it still writes bps=24 and
    /// drops the low byte of every sample — so 32-bit output is WAV only, and
    /// asking for 32-bit FLAC is rejected rather than quietly downgraded.
    ///
    /// `level` is ffmpeg's FLAC compression level, 0..=12 (12 packs smallest);
    /// 13 is refused by the encoder at init, which `g6me --caps` demonstrates.
    Flac {
        level: u32,
    },
    Wav,
    /// Write no file at all. This is what live monitoring uses: the same pump,
    /// the same filters and the same downmixer, with the samples handed to a
    /// device instead of an encoder — so what you hear is what a render produces
    /// rather than a second implementation of the chain.
    Null,
}

impl OutFormat {
    pub fn label(&self) -> String {
        match self {
            OutFormat::Flac { level } => format!("FLAC (level {level})"),
            OutFormat::Wav => "WAV".to_string(),
            OutFormat::Null => "no file (monitor only)".to_string(),
        }
    }
    pub fn extension(&self) -> &'static str {
        match self {
            OutFormat::Flac { .. } => "flac",
            OutFormat::Wav => "wav",
            OutFormat::Null => "",
        }
    }

    /// True when nothing is written to disk.
    pub fn is_null(&self) -> bool {
        matches!(self, OutFormat::Null)
    }
}

/// Output-side encoding, including the optional second resample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutCfg {
    pub format: OutFormat,
    pub depth: BitDepth,
    /// `None` = passthrough: no resampling, output at the working rate.
    pub resample: Option<u32>,
    pub precision: f64,
    pub dither: Dither,
    pub copy_metadata: bool,
    pub overwrite: bool,
    /// `-filter_complex_threads`; 1 keeps a multi-file batch from oversubscribing
    /// the CPU with one slice-threaded ffmpeg per file.
    pub filter_threads: Option<u32>,
}

impl Default for OutCfg {
    fn default() -> Self {
        Self {
            format: OutFormat::Flac { level: 8 },
            depth: BitDepth::S24,
            resample: None,
            precision: 33.0,
            dither: Dither::None,
            copy_metadata: true,
            overwrite: false,
            filter_threads: None,
        }
    }
}

impl OutCfg {
    /// Validate a format/depth pair against what is actually available.
    pub fn validate(&self, caps: &Caps) -> Result<()> {
        if self.format.is_null() {
            return Ok(());
        }
        if let OutFormat::Flac { level } = &self.format {
            if self.depth.is_float() {
                return Err(Error::msg(
                    "FLAC is integer-only; choose WAV for float output",
                ));
            }
            if !matches!(level, 0..=12) {
                // Measured against ffmpeg n9.0.2: 0..=12 encode, 13 and up fail at
                // encoder init with "invalid compression level". The reference flac
                // CLI's -0..-8 presets are a different scale and no longer in play.
                return Err(Error::msg(
                    "FLAC compression level must be 0..=12 (ffmpeg's FLAC encoder; \
                     12 is the smallest file, 13 is refused)",
                ));
            }
            if self.depth == BitDepth::S24 && !caps.flac24 {
                return Err(Error::msg(
                    "this ffmpeg was not verified able to write a lossless 24-bit FLAC \
                     (run `g6me --caps` for the reason). Use '--depth 16', or \
                     '--format wav --depth 24' which never touches the FLAC encoder.",
                ));
            }
            if self.depth == BitDepth::S32 {
                return Err(Error::msg(
                    "32-bit FLAC is not available: ffmpeg's FLAC encoder writes at \
                     most 24-bit and drops the low byte of every 32-bit sample, so \
                     the file would be labelled 24-bit while looking 32-bit. Use \
                     '--format flac --depth 24' for the lossless int file, or \
                     '--format wav --depth 32' / '--depth f32' for the full width \
                     of the DSP output.",
                ));
            }
        }
        if self.resample == Some(0) {
            return Err(Error::msg("output sample rate must be > 0"));
        }
        Ok(())
    }
}

/// Everything one render needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RenderCfg {
    pub resample: ResampleCfg,
    pub ir: IrCfg,
    pub out: OutCfg,
    /// Override the decoder-reported channel order (interleaved order).
    pub order: Option<Vec<crate::mask::Slot>>,
    /// Frames per processing block.
    pub block: usize,
    /// Blocks allowed to be in flight between the decode, DSP and encode stages.
    ///
    /// This is the whole memory bound of a render: peak is roughly
    /// `queue_blocks * block * channels * 8` bytes per file, so a 3-hour Atmos
    /// concert streams at the same cost as a 3-second clip. Deep enough to absorb
    /// a scheduler hiccup, shallow enough that cancelling does not wait for a
    /// queue to drain.
    pub queue_blocks: usize,
    /// Which audio stream to take, 0-based.
    pub stream: usize,
    /// Over-sampling ratio the true peak is measured at, for every render.
    ///
    /// Measured always rather than behind a flag: it is the one number in a report that
    /// says something about the signal instead of about the pipeline, and the cost is
    /// [`Ratio::mac_per_sample`] multiply-adds per sample per channel — 48 at the
    /// default x4, which is a fraction of what the downmixer already does.
    ///
    /// The reading is of the signal the encoder receives, so it is the same number a
    /// BS.1770 consumer gets from the finished file, not a pre-limiter estimate.
    #[serde(default)]
    pub true_peak: Ratio,
    /// True-peak limiter, applied after the downmixer and before the encoder.
    ///
    /// Off by default, like every other stage here. When enabled it is the last thing
    /// that touches the samples, because a ceiling asserted before the last gain stage
    /// is not a ceiling.
    #[serde(default)]
    pub limiter: LimiterCfg,
    /// Hash every finished output, twice over: the samples handed to the encoder
    /// (container-independent, so a WAV and a FLAC of one render agree) and the file on
    /// disk (what `sha256sum` prints, so the number is checkable with a tool the user
    /// already has). See [`crate::pipeline::Report::file_sha256`].
    pub sha256: bool,
}

impl RenderCfg {
    pub fn with_defaults() -> Self {
        Self {
            block: 4096,
            ..Default::default()
        }
    }
}

/// What the input stage will actually do, after reconciling the config with the
/// source. `Noop` means "enabled, but there is nothing to do", which the report
/// says out loud instead of pretending a resampler ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputStage {
    Passthrough,
    Noop,
    Soxr,
    SoxrAndDepth,
    DepthOnly,
}

impl InputStage {
    pub fn label(self) -> &'static str {
        match self {
            InputStage::Passthrough => "decode only",
            InputStage::Noop => "enabled, no-op (same rate, no dither)",
            InputStage::Soxr => "soxr resample",
            InputStage::SoxrAndDepth => "soxr resample + depth",
            InputStage::DepthOnly => "depth only",
        }
    }
}

/// Decide + build the filter steps for the input chain. Steps are joined with
/// `,` for `-af`, or embedded in a `-filter_complex` when the IR stage is on.
pub fn input_stage(cfg: &RenderCfg, info: &AudioInfo, mask: u32) -> (InputStage, Vec<String>) {
    let want_rate = cfg.resample.enable && cfg.resample.rate != info.sample_rate;
    let want_dither = cfg.resample.enable && cfg.resample.dither != Dither::None;
    let depth = if cfg.resample.enable {
        cfg.resample.effective_depth()
    } else {
        None
    };
    // A depth request equal to what the decoder already produced is not a step.
    let depth =
        depth.filter(|d| !matches!(d, BitDepth::S32 | BitDepth::F64) || info.bits == Some(32));

    let stage = match (want_rate || want_dither, depth.is_some()) {
        (true, true) => InputStage::SoxrAndDepth,
        (true, false) => InputStage::Soxr,
        (false, true) => InputStage::DepthOnly,
        (false, false) if cfg.resample.enable => InputStage::Noop,
        (false, false) => InputStage::Passthrough,
    };

    // Declared identity of the decoded stream; first so every later filter sees a
    // known layout instead of inferring one.
    let mut steps = vec![format!("aformat=channel_layouts={}", format_mask_hex(mask))];

    match stage {
        InputStage::Passthrough | InputStage::Noop => {}
        InputStage::Soxr | InputStage::SoxrAndDepth => {
            let mut s = format!(
                "aresample={}:resampler=soxr:precision={}:dither_method={}",
                cfg.resample.rate,
                cfg.resample.precision as u32,
                cfg.resample.dither.ffmpeg_name()
            );
            if cfg.resample.cheby {
                s.push_str(":cheby=true");
            }
            steps.push(s);
        }
        InputStage::DepthOnly => {
            steps.push(format!(
                "aresample=dither_method={}",
                cfg.resample.dither.ffmpeg_name()
            ));
        }
    }

    if let Some(d) = depth {
        steps.push(format!("aformat=sample_fmts={}", d.swr_fmt()));
    }

    (stage, steps)
}

/// afir options, in the order mpv writes them, omitting what the user left alone.
pub fn afir_filter(cfg: &IrCfg) -> String {
    let mut f = String::from("afir=irfmt=mono");
    if let Some(v) = cfg.irnorm {
        f.push_str(&format!(":irnorm={v}"));
    }
    if let Some(g) = cfg.gtype {
        f.push_str(&format!(":gtype={}", g.ffmpeg_name()));
    }
    if let Some(v) = cfg.wet {
        f.push_str(&format!(":wet={v}"));
    }
    if let Some(v) = cfg.dry {
        f.push_str(&format!(":dry={v}"));
    }
    if let Some(v) = cfg.length {
        f.push_str(&format!(":length={v}"));
    }
    if let Some(l) = cfg.link {
        f.push_str(&format!(":irlink={}", if l { "true" } else { "false" }));
    }
    f.push_str(&format!(":minp={}:maxp={}", cfg.minp, cfg.maxp));
    f.push_str(&format!(":precision={AFIR_PRECISION}"));
    f
}

/// Command line for the producing (decode) child. Reads PCM to us on stdout as
/// interleaved `f64le`, `mask.count_ones()` channels, source rate (or the
/// resampled rate).
pub fn input_args(
    cfg: &RenderCfg,
    info: &AudioInfo,
    mask: u32,
    src: &Path,
    ir_info: Option<&AudioInfo>,
) -> Result<(Vec<String>, u32, usize)> {
    let (_, steps) = input_stage(cfg, info, mask);
    let work_rate = if cfg.resample.enable {
        cfg.resample.rate
    } else {
        info.sample_rate
    };
    let channels = mask.count_ones() as usize;

    let mut args: Vec<String> = vec!["-v".into(), "error".into(), "-nostdin".into()];
    if let Some(t) = cfg.out.filter_threads {
        args.push("-filter_complex_threads".into());
        args.push(t.to_string());
    }
    args.push("-i".into());
    args.push(src.display().to_string());

    if cfg.ir.enable {
        let ir_path = cfg
            .ir
            .file
            .as_ref()
            .ok_or_else(|| Error::msg("IR calibration is enabled but no IR file is set"))?
            .display()
            .to_string();
        args.push("-i".into());
        args.push(ir_path);

        // afir convolves in the main path's time base; an IR recorded at another
        // rate is brought to it, which is the only case where we resample an IR.
        let mut ir_steps = vec!["aformat=sample_fmts=dbl:channel_layouts=mono".to_string()];
        if let Some(i) = ir_info {
            if i.sample_rate != work_rate {
                ir_steps.insert(
                    0,
                    format!(
                        "aresample={}:resampler=soxr:precision=33:dither_method=none",
                        work_rate
                    ),
                );
            }
        }
        let mut main = steps.clone();
        main.push("aformat=sample_fmts=dbl".into());
        let graph = format!(
            "[0:a:{}]{}[am];[1:a:0]{}[ir];[am][ir]{}[o]",
            cfg.stream,
            main.join(","),
            ir_steps.join(","),
            afir_filter(&cfg.ir)
        );
        args.push("-filter_complex".into());
        args.push(graph);
        args.push("-map".into());
        args.push("[o]".into());
    } else {
        // Always pass -af, even when the only step is the layout declaration:
        // for a container that reports no layout it pins the channel count we
        // expect instead of letting the muxer guess.
        args.push("-af".into());
        args.push(steps.join(","));
    }

    args.push("-f".into());
    args.push("f64le".into());
    args.push("-c:a".into());
    args.push("pcm_f64le".into());
    args.push("-".into());
    Ok((args, work_rate, channels))
}

/// What the encoder will be fed: interleaved `f64le`, 2 channels, `work_rate`.
pub fn encoder_pump_args(work_rate: u32) -> Vec<String> {
    vec![
        "-f".into(),
        "f64le".into(),
        "-c:a".into(),
        "pcm_f64le".into(),
        "-ar".into(),
        work_rate.to_string(),
        "-ch_layout".into(),
        "stereo".into(),
        "-i".into(),
        "pipe:0".into(),
    ]
}

/// Command line for the consuming (encode) child — one ffmpeg process, whatever
/// the format.
pub fn output_args(
    cfg: &RenderCfg,
    work_rate: u32,
    out_path: &Path,
    src: &Path,
    caps: &Caps,
) -> Result<Vec<String>> {
    let o = &cfg.out;
    o.validate(caps)?;

    if o.format.is_null() {
        // No encoder child at all; the pump hands samples to the tap.
        return Ok(Vec::new());
    }
    let mut args: Vec<String> = vec!["-v".into(), "error".into(), "-nostdin".into()];
    if let Some(t) = o.filter_threads {
        args.push("-filter_complex_threads".into());
        args.push(t.to_string());
    }
    args.extend(encoder_pump_args(work_rate));
    if o.copy_metadata {
        // Second input, read for its header/tags only.
        args.push("-i".into());
        args.push(src.display().to_string());
    }

    let need_resample = o.resample.is_some_and(|r| r != work_rate);
    let need_dither = o.dither != Dither::None;
    if need_resample || need_dither {
        let rate = o.resample.unwrap_or(work_rate);
        let mut s = if need_resample {
            format!(
                "aresample={rate}:resampler=soxr:precision={}:dither_method={}",
                o.precision as u32,
                o.dither.ffmpeg_name()
            )
        } else {
            format!("aresample=dither_method={}", o.dither.ffmpeg_name())
        };
        if o.resample.is_some() && !need_resample && o.precision > 0.0 {
            s.push_str(&format!(":precision={}", o.precision as u32));
        }
        args.push("-af".into());
        args.push(s);
    }
    if o.copy_metadata {
        args.push("-map".into());
        args.push("0:a".into());
        args.push("-map_metadata".into());
        args.push("1".into());
        args.push("-map_metadata:s:0".into());
        args.push("1:s:0".into());
    }
    args.push("-y".into());

    match &o.format {
        OutFormat::Flac { level } => {
            args.push("-c:a".into());
            args.push("flac".into());
            args.push("-compression_level".into());
            args.push(level.to_string());
            // Ask for the depth explicitly instead of letting the encoder pick: a
            // 24-bit request lands on s32 (libswresample has no s24 format) and the
            // encoder then derives bps=24 from it, which is what makes the stream
            // genuinely 24-bit rather than a widened one.
            args.push("-sample_fmt".into());
            args.push(o.depth.swr_fmt().into());
            args.push(out_path.display().to_string());
        }
        OutFormat::Wav => {
            args.push("-c:a".into());
            args.push(o.depth.codec().into());
            args.push(out_path.display().to_string());
        }
        OutFormat::Null => unreachable!("handled above"),
    }
    Ok(args)
}

/// Plain-language description of what will happen to the samples, in order. The
/// GUI shows this so the loss chain is never something you have to reconstruct
/// from three dropdowns.
pub fn loss_chain(cfg: &RenderCfg, info: &AudioInfo, mask: u32) -> Vec<String> {
    let mut v = Vec::new();
    let (stage, _) = input_stage(cfg, info, mask);
    v.push(format!(
        "1. decode {} {} ch {} Hz -> f64 (exact widening)",
        info.codec_name, info.channels, info.sample_rate
    ));
    match stage {
        InputStage::Passthrough => {
            v.push("2. input stage: off (source values reach the DSP unchanged)".into())
        }
        InputStage::Noop => v.push(format!("2. input stage: {}", stage.label())),
        other => v.push(format!(
            "2. input stage: {} -> {} Hz, dither {}, precision {} bits",
            other.label(),
            cfg.resample.rate,
            cfg.resample.dither.label(),
            cfg.resample.precision as u32
        )),
    }
    if cfg.ir.enable {
        v.push(format!(
            "3. IR convolution (afir, mono, {}): {}",
            AFIR_PRECISION,
            cfg.ir
                .file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<unset>".into())
        ));
    } else {
        v.push("3. IR convolution: off".into());
    }
    v.push(format!(
        "4. downmix mask {mask:#06x} in {} -> stereo (bit-exact port, {})",
        crate::mask::format_order(&order_for_mask(mask)),
        f64::PRECISION_NAME
    ));
    // 5. the true-peak stage. Printed whether or not it is limiting, because the meter
    // runs either way and a dry run that hides the one stage that measures the file
    // would leave `--dry-run` unable to explain the number in the report.
    let la = cfg.limiter;
    let work_rate = effective_input_rate(cfg, info);
    v.push(if !la.enable {
        format!(
            "5. true-peak: meter at {}x ({}), limiter off",
            cfg.true_peak.n(),
            cfg.true_peak.chain()
        )
    } else {
        format!(
            "5. true-peak: limit to {:.2} dBTP at {}x ({}), lookahead {:.2} ms -> {:.2} ms \
             effective, +{} samples latency, aims {:.2} dB low ({}x meter)",
            la.ceiling_db,
            la.ratio.n(),
            la.ratio.chain(),
            la.lookahead_ms,
            la.effective_lookahead_ms(work_rate),
            la.latency_samples(work_rate),
            la.guard_db(),
            cfg.true_peak.n(),
        )
    });
    let o = &cfg.out;
    let rate = match o.resample {
        Some(r) if r != effective_input_rate(cfg, info) => {
            format!("soxr -> {r} Hz, dither {}", o.dither.label())
        }
        _ => "no resample".to_string(),
    };
    v.push(format!(
        "6. encode {} {}{}",
        o.format.label(),
        o.depth.label(),
        if rate == "no resample" {
            String::new()
        } else {
            format!(", {rate}")
        },
    ));
    v
}

/// Rate the DSP sees, given the config and the source.
pub fn effective_input_rate(cfg: &RenderCfg, info: &AudioInfo) -> u32 {
    if cfg.resample.enable {
        cfg.resample.rate
    } else {
        info.sample_rate
    }
}

/// Best-effort inverse of [`crate::mask::mask_of`], for the summary line only.
fn order_for_mask(mask: u32) -> Vec<crate::mask::Slot> {
    use crate::mask::Slot;
    Slot::ALL
        .into_iter()
        .filter(|s| mask & s.bit() != 0)
        .collect()
}

/// Marker trait-ish helper so `loss_chain` prints the same word the DSP reports.
trait PrecisionName {
    const PRECISION_NAME: &'static str;
}
impl PrecisionName for f64 {
    const PRECISION_NAME: &'static str = "float64";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::Slot;
    use crate::probe::Caps;

    fn info() -> AudioInfo {
        AudioInfo {
            stream_index: 0,
            codec_name: "truehd".into(),
            profile: Some("Dolby TrueHD + Dolby Atmos".into()),
            sample_rate: 48000,
            channels: 8,
            layout: Some("7.1".into()),
            bits: Some(24),
            duration: 240.0,
        }
    }

    #[test]
    fn defaults_emit_no_resampler_at_all() {
        let cfg = RenderCfg::with_defaults();
        let (stage, steps) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Passthrough);
        // only the layout declaration, no aresample anywhere
        assert_eq!(steps, vec!["aformat=channel_layouts=0x63f".to_string()]);
        assert!(!steps.join(",").contains("aresample"));
    }

    #[test]
    fn default_output_is_flac24_with_no_dither_and_no_resample() {
        let cfg = RenderCfg::with_defaults();
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        assert_eq!(cfg.out.depth, BitDepth::S24);
        let j = args.join(" ");
        assert!(j.contains("-c:a flac"), "{j}");
        assert!(j.contains("-sample_fmt s32"), "{j}");
        assert!(!j.contains("aresample"), "{j}");
        assert!(!j.contains("dither"), "{j}");
    }

    #[test]
    fn ir_filter_string_matches_the_mpv_graph() {
        let s = afir_filter(&IrCfg::default());
        assert_eq!(
            s,
            "afir=irfmt=mono:irnorm=-1:wet=1:minp=4096:maxp=4096:precision=double"
        );
    }

    #[test]
    fn ir_graph_shape() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.ir.enable = true;
        cfg.ir.file = Some(PathBuf::from("/x/ir.wav"));
        let (args, rate, ch) = input_args(
            &cfg,
            &info(),
            crate::dsp::MASK_7_1,
            Path::new("/x/i.mka"),
            None,
        )
        .unwrap();
        assert_eq!((rate, ch), (48000, 8));
        let j = args.join(" ");
        assert!(j.contains("-filter_complex"), "{j}");
        assert!(j.contains("[0:a:0]"), "{j}");
        assert!(
            j.contains("[1:a:0]aformat=sample_fmts=dbl:channel_layouts=mono[ir]"),
            "{j}"
        );
        assert!(j.contains("[am][ir]afir=irfmt=mono"), "{j}");
        assert!(j.contains("-map [o]"), "{j}");
    }

    #[test]
    fn resample_on_adds_soxr_once_with_dither_spelling() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.resample.enable = true;
        cfg.resample.rate = 44100;
        cfg.resample.dither = Dither::LowShibata;
        let (stage, steps) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Soxr);
        let joined = steps.join(",");
        assert_eq!(
            joined,
            "aformat=channel_layouts=0x63f,aresample=44100:resampler=soxr:precision=33:dither_method=low_shibata"
        );
        assert_eq!(joined.matches("aresample").count(), 1);
    }

    #[test]
    fn enabling_resample_at_the_source_rate_is_reported_as_a_noop() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.resample.enable = true;
        cfg.resample.rate = 48000;
        let (stage, _) = input_stage(&cfg, &info(), crate::dsp::MASK_7_1);
        assert_eq!(stage, InputStage::Noop);
    }

    /// A capability set whose ffmpeg has been measured writing lossless 24-bit
    /// FLAC, i.e. the state a machine with a working ffmpeg reports.
    fn caps_flac24() -> Caps {
        Caps {
            flac24: true,
            ..Caps::default()
        }
    }

    #[test]
    fn flac_rejects_float_and_unverified_24bit() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 8 };
        cfg.out.depth = BitDepth::F64;
        assert!(cfg.out.validate(&caps_flac24()).is_err());
        // The default depth is 24-bit, and it is refused on a build where the
        // probe could not prove 24-bit FLAC is lossless — a file labelled 24-bit
        // that quietly dropped bits is worse than an error.
        cfg.out.depth = BitDepth::S24;
        assert!(cfg.out.validate(&Caps::default()).is_err());
        cfg.out.validate(&caps_flac24()).unwrap();
    }

    #[test]
    fn flac_refuses_32_bit_instead_of_writing_a_24_bit_file() {
        // ffmpeg labels what it writes bps=24 and throws the low byte of every
        // 32-bit sample away, so the combination has to be refused by name.
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.format = OutFormat::Flac { level: 8 };
        cfg.out.depth = BitDepth::S32;
        let e = cfg.out.validate(&caps_flac24()).unwrap_err().to_string();
        assert!(e.contains("32-bit FLAC"), "{e}");
        assert!(e.contains("--depth 24"), "{e}");
    }

    #[test]
    fn wav_allows_all_depths() {
        for d in BitDepth::ALL {
            let mut cfg = RenderCfg::with_defaults();
            cfg.out.format = OutFormat::Wav;
            cfg.out.depth = d;
            cfg.out.validate(&Caps::default()).unwrap();
        }
    }

    #[test]
    fn output_resample_uses_soxr_with_precision() {
        let mut cfg = RenderCfg::with_defaults();
        cfg.out.resample = Some(96000);
        let args = output_args(
            &cfg,
            48000,
            Path::new("/tmp/o.flac"),
            Path::new("/tmp/i.mka"),
            &caps_flac24(),
        )
        .unwrap();
        let j = args.join(" ");
        assert!(
            j.contains("aresample=96000:resampler=soxr:precision=33:dither_method=none"),
            "{j}"
        );
    }

    #[test]
    fn pump_declares_rate_and_layout_for_raw_stdin() {
        let j = encoder_pump_args(48000).join(" ");
        assert!(j.contains("-ar 48000"), "{j}");
        assert!(j.contains("-ch_layout stereo"), "{j}");
        assert!(j.contains("pipe:0"), "{j}");
    }

    #[test]
    fn loss_chain_says_off_when_off() {
        let lines = loss_chain(&RenderCfg::with_defaults(), &info(), crate::dsp::MASK_7_1);
        assert_eq!(lines.len(), 6);
        assert!(lines[1].contains("off"), "{}", lines[1]);
        assert!(lines[2].contains("off"), "{}", lines[2]);
        assert!(lines[3].contains("bit-exact port"), "{}", lines[3]);
        assert!(
            lines[3].contains("FL FR FC LFE BL BR SL SR"),
            "{}",
            lines[3]
        );
        assert!(lines[4].contains("limiter off"), "{}", lines[4]);
        assert!(lines[4].contains("meter at 4x"), "{}", lines[4]);
        assert!(lines[5].contains("FLAC"), "{}", lines[5]);
    }

    #[test]
    fn order_for_mask_roundtrips() {
        use crate::mask::mask_of;
        assert_eq!(
            order_for_mask(mask_of(&[
                Slot::Fl,
                Slot::Fr,
                Slot::Fc,
                Slot::Lfe,
                Slot::Bl,
                Slot::Br,
                Slot::Sl,
                Slot::Sr
            ])),
            vec![
                Slot::Fl,
                Slot::Fr,
                Slot::Fc,
                Slot::Lfe,
                Slot::Bl,
                Slot::Br,
                Slot::Sl,
                Slot::Sr
            ]
        );
    }
}
