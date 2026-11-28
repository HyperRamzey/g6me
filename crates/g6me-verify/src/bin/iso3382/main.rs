//! `iso3382` — ISO 3382-1 metric table for a measured-RIR dataset directory.
//!
//! ```text
//! iso3382 <dataset-dir> [--list] [--limit N | --all] [--band HZ]
//!                      [--no-align] [--max-seconds S] [--json | --csv]
//! ```
//!
//! Discovers OpenAIR, dEchorate, or MeshRIR layouts (or a plain tree of WAVs),
//! decodes every entry it can, and prints T20, T30, EDT, C80, D50, LF, and IACC
//! for the broadband response and for octave bands 31.5 Hz – 16 kHz wherever the
//! sample rate and response length support the band.
//!
//! Exit status: 0 when every selected entry was analyzed, 1 when any entry could
//! not be analyzed (for example a dEchorate HDF5 group, which this harness
//! enumerates but does not decode), 2 for a usage error. Notes and per-entry
//! errors go to stderr so `--json`/`--csv` stdout stays parseable.

mod filterbank;
mod loaders;
mod metrics;
mod npy;
#[cfg(test)]
mod testutil;
mod wavread;

use loaders::{DatasetEntry, EntrySource};
use metrics::{Analysis, BandRow, Metrics};
use serde_json::json;
use std::fmt::Display;
use std::path::PathBuf;

const DEFAULT_LIMIT: usize = 25;

struct Args {
    root: PathBuf,
    limit: Option<usize>,
    list: bool,
    json: bool,
    csv: bool,
    align: bool,
    band: Option<usize>,
    max_seconds: Option<f64>,
    help: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            limit: Some(DEFAULT_LIMIT),
            list: false,
            json: false,
            csv: false,
            align: true,
            band: None,
            max_seconds: None,
            help: false,
        }
    }
}

struct RunStats {
    analyzed: usize,
    errors: usize,
}

fn usage() -> &'static str {
    "usage: iso3382 <dataset-dir> [--list] [--limit N | --all] [--band HZ]\n\
     \x20                 [--no-align] [--max-seconds S] [--json | --csv]\n\
     \n\
     Reads OpenAIR, dEchorate, and MeshRIR directory layouts (read-only), plus a\n\
     generic tree of WAV files. Prints T20, T30, EDT, C80, D50, LF, and IACC for\n\
     the broadband response and octave bands 31.5 Hz to 16 kHz.\n\
     \n\
     --list            enumerate entries and exit\n\
     --limit N         analyze at most N entries (default 25)\n\
     --all             analyze every entry\n\
     --band HZ         print only the octave band at HZ (one of 31.5 … 16000)\n\
     --no-align        keep sample 0 as t=0 instead of aligning to the onset\n\
     --max-seconds S   truncate each response to S seconds before analysis\n\
     --json / --csv    machine-readable output (stderr carries notes/errors)"
}

fn value_at<T: std::str::FromStr>(
    argv: &[String],
    i: &mut usize,
    flag: &str,
) -> Result<T, String>
where
    T::Err: Display,
{
    let raw = argv
        .get(*i + 1)
        .ok_or_else(|| format!("{flag} needs a value"))?;
    *i += 1;
    raw.parse().map_err(|e| format!("{flag}: {e}"))
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args::default();
    let mut root: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--list" => args.list = true,
            "--json" => args.json = true,
            "--csv" => args.csv = true,
            "--no-align" => args.align = false,
            "--all" => args.limit = None,
            "--help" | "-h" => args.help = true,
            "--limit" => {
                let n = value_at::<usize>(argv, &mut i, "--limit")?;
                if n == 0 {
                    return Err("--limit must be at least 1 (use --all for every entry)".into());
                }
                args.limit = Some(n);
            }
            "--band" => {
                let hz = value_at::<f64>(argv, &mut i, "--band")?;
                let index = filterbank::OCTAVE_CENTERS_HZ
                    .iter()
                    .position(|c| (c - hz).abs() < 1e-6)
                    .ok_or_else(|| {
                        format!(
                            "--band {hz} is not an octave centre; choose one of {:?}",
                            filterbank::OCTAVE_CENTERS_HZ
                        )
                    })?;
                args.band = Some(index);
            }
            "--max-seconds" => {
                let s = value_at::<f64>(argv, &mut i, "--max-seconds")?;
                if !s.is_finite() || s <= 0.0 {
                    return Err("--max-seconds must be a positive number".into());
                }
                args.max_seconds = Some(s);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option {other} (see --help)"))
            }
            other => {
                if root.is_some() {
                    return Err("exactly one dataset directory may be given".into());
                }
                root = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    if args.json && args.csv {
        return Err("--json and --csv are mutually exclusive".into());
    }
    if args.help {
        return Ok(args);
    }
    args.root = root.ok_or_else(|| "missing dataset directory (see --help)".to_string())?;
    Ok(args)
}

fn fmt_opt(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.decimals$}"),
        _ => "-".to_string(),
    }
}

fn fmt_iacc(metrics: &Metrics) -> String {
    match metrics.iacc {
        Some(i) if i.value.is_finite() => format!("{:.3}", i.value),
        _ => "-".to_string(),
    }
}

fn metrics_row(label: &str, m: &Metrics) -> String {
    format!(
        "   {label:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}",
        fmt_opt(m.t20_s, 3),
        fmt_opt(m.t30_s, 3),
        fmt_opt(m.edt_s, 3),
        fmt_opt(m.c80_db, 1),
        fmt_opt(m.d50, 3),
        fmt_opt(m.lf, 3),
        fmt_iacc(m),
    )
}

fn band_rows(analysis: &Analysis, band: Option<usize>) -> impl Iterator<Item = &BandRow> {
    analysis
        .bands
        .iter()
        .enumerate()
        .filter(move |(i, _)| band.is_none_or(|b| b == *i))
        .map(|(_, row)| row)
}

/// Render one entry's text table.
fn render_table(entry: &DatasetEntry, analysis: &Analysis, band: Option<usize>) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n== {} [{}] ==",
        entry.id,
        entry.source.kind_label()
    );
    let _ = writeln!(
        out,
        "   {} Hz, {}, {:.3} s, onset sample {}",
        analysis.sample_rate,
        analysis.layout.label(),
        analysis.duration_s,
        analysis.onset_sample
    );
    let _ = writeln!(
        out,
        "   {:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}",
        "band", "T20 s", "T30 s", "EDT s", "C80 dB", "D50", "LF", "IACC"
    );
    let _ = writeln!(out, "   {}", "-".repeat(73));
    out.push_str(&metrics_row("broadband", &analysis.broadband));
    out.push('\n');
    let mut unsupported = Vec::new();
    for row in band_rows(analysis, band) {
        if row.supported {
            out.push_str(&metrics_row(&metrics::band_label(row.band_hz), &row.metrics));
        } else {
            out.push_str(&metrics_row(
                &format!("{}*", metrics::band_label(row.band_hz)),
                &row.metrics,
            ));
            unsupported.push(metrics::band_label(row.band_hz));
        }
        out.push('\n');
    }
    if !unsupported.is_empty() {
        let _ = writeln!(
            out,
            "   * not representable at this sample rate/length: {}",
            unsupported.join(", ")
        );
    }
    out
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_value(value: Option<f64>, decimals: usize) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.decimals$}"),
        _ => String::new(),
    }
}

fn metrics_json(m: &Metrics) -> serde_json::Value {
    json!({
        "t20_s": m.t20_s,
        "t30_s": m.t30_s,
        "edt_s": m.edt_s,
        "c80_db": m.c80_db,
        "d50": m.d50,
        "lf": m.lf,
        "iacc": m.iacc.map(|i| json!({
            "value": i.value,
            "lag_samples": i.lag_samples,
            "lag_ms": i.lag_ms,
        })),
    })
}

fn entry_json(entry: &DatasetEntry, analysis: &Analysis) -> serde_json::Value {
    json!({
        "id": entry.id,
        "source": entry.source.kind_label(),
        "metadata": entry.metadata.iter()
            .map(|(k, v)| json!({ "key": k, "value": v }))
            .collect::<Vec<_>>(),
        "sample_rate": analysis.sample_rate,
        "layout": analysis.layout.label(),
        "duration_s": analysis.duration_s,
        "onset_sample": analysis.onset_sample,
        "broadband": metrics_json(&analysis.broadband),
        "bands": analysis.bands.iter().map(|b| json!({
            "band_hz": b.band_hz,
            "supported": b.supported,
            "metrics": metrics_json(&b.metrics),
        })).collect::<Vec<_>>(),
    })
}

fn print_csv(entries: &[(&DatasetEntry, Analysis)], band: Option<usize>) {
    println!(
        "entry,dataset_entry_kind,band,supported,t20_s,t30_s,edt_s,c80_db,d50,lf,iacc,iacc_lag_ms"
    );
    for (entry, analysis) in entries {
        let rows = std::iter::once(("broadband".to_string(), true, &analysis.broadband)).chain(
            band_rows(analysis, band).map(|b| (metrics::band_label(b.band_hz), b.supported, &b.metrics)),
        );
        for (label, supported, m) in rows {
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                csv_escape(&entry.id),
                entry.source.kind_label(),
                csv_escape(&label),
                supported,
                csv_value(m.t20_s, 6),
                csv_value(m.t30_s, 6),
                csv_value(m.edt_s, 6),
                csv_value(m.c80_db, 3),
                csv_value(m.d50, 4),
                csv_value(m.lf, 4),
                csv_value(m.iacc.map(|i| i.value), 4),
                csv_value(m.iacc.map(|i| i.lag_ms), 3),
            );
        }
    }
}

fn analyze_entry(
    entry: &DatasetEntry,
    args: &Args,
) -> Result<Analysis, String> {
    let mut rir = loaders::read_entry(entry)?;
    if let Some(seconds) = args.max_seconds {
        let frames = (seconds * rir.sample_rate as f64).round() as usize;
        let frames = frames.max(64);
        for channel in &mut rir.channels {
            channel.truncate(frames.min(channel.len()));
        }
    }
    metrics::analyze(&rir, args.align).map_err(|e| format!("{}: {e}", entry.id))
}

fn run(args: &Args) -> Result<RunStats, String> {
    let dataset = loaders::load_dataset(&args.root)?;
    for note in &dataset.notes {
        eprintln!("note: {note}");
    }
    if args.list {
        for entry in &dataset.entries {
            println!(
                "{}\t{}",
                entry.id,
                match &entry.source {
                    EntrySource::Wav { .. } => "wav",
                    EntrySource::Npy { .. } => "npy",
                    EntrySource::Hdf5Group { .. } => "hdf5 (not decoded)",
                }
            );
        }
        eprintln!(
            "{} entries ({})",
            dataset.entries.len(),
            dataset.kind.label()
        );
        return Ok(RunStats {
            analyzed: 0,
            errors: 0,
        });
    }

    let selected: Vec<&DatasetEntry> = dataset
        .entries
        .iter()
        .take(args.limit.unwrap_or(usize::MAX))
        .collect();
    if selected.is_empty() {
        return Err(format!(
            "{}: no analyzable entries found ({})",
            dataset.root.display(),
            dataset.kind.label()
        ));
    }

    let mut results: Vec<(&DatasetEntry, Analysis)> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for entry in &selected {
        match analyze_entry(entry, args) {
            Ok(analysis) => results.push((entry, analysis)),
            Err(e) => errors.push(e),
        }
    }
    for error in &errors {
        eprintln!("error: {error}");
    }

    if args.json {
        let entries: Vec<serde_json::Value> = results
            .iter()
            .map(|(entry, analysis)| entry_json(entry, analysis))
            .collect();
        let doc = json!({
            "dataset": dataset.kind.label(),
            "root": dataset.root.display().to_string(),
            "notes": dataset.notes,
            "entries": entries,
            "errors": errors,
        });
        let text = serde_json::to_string_pretty(&doc)
            .map_err(|e| format!("cannot serialize JSON: {e}"))?;
        println!("{text}");
    } else if args.csv {
        print_csv(&results, args.band);
    } else {
        for (entry, analysis) in &results {
            print!("{}", render_table(entry, analysis, args.band));
        }
        eprintln!(
            "\n{} analyzed, {} errors ({} entries in dataset)",
            results.len(),
            errors.len(),
            dataset.entries.len()
        );
    }

    Ok(RunStats {
        analyzed: results.len(),
        errors: errors.len(),
    })
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("iso3382: {e}");
            std::process::exit(2);
        }
    };
    if args.help {
        println!("{}", usage());
        return;
    }
    match run(&args) {
        Ok(stats) => {
            if !args.list && (stats.errors > 0 || stats.analyzed == 0) {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("iso3382: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{geometric_rir, write_text, write_wav_f32, TempDir};

    fn args_for(root: &std::path::Path) -> Args {
        Args {
            root: root.to_path_buf(),
            ..Args::default()
        }
    }

    #[test]
    fn parse_args_accepts_flags_and_one_directory() {
        let argv: Vec<String> = [
            "/data/openair",
            "--limit",
            "3",
            "--band",
            "1000",
            "--no-align",
            "--max-seconds",
            "2.5",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        let a = parse_args(&argv).expect("parse");
        assert_eq!(a.root, PathBuf::from("/data/openair"));
        assert_eq!(a.limit, Some(3));
        assert_eq!(a.band, Some(5));
        assert!(!a.align);
        assert_eq!(a.max_seconds, Some(2.5));
    }

    #[test]
    fn parse_args_rejects_unknown_band_and_conflicting_outputs() {
        let bad_band: Vec<String> = ["d", "--band", "999"]
            .iter()
            .map(ToString::to_string)
            .collect();
        assert!(parse_args(&bad_band).is_err());
        let conflict: Vec<String> = ["d", "--json", "--csv"]
            .iter()
            .map(ToString::to_string)
            .collect();
        assert!(parse_args(&conflict).is_err());
        let two_dirs: Vec<String> = ["a", "b"].iter().map(ToString::to_string).collect();
        assert!(parse_args(&two_dirs).is_err());
        assert!(parse_args(&[]).is_err());
    }

    #[test]
    fn table_lists_broadband_and_every_octave_band() {
        let rir = metrics::Rir {
            sample_rate: 48_000,
            channels: vec![geometric_rir(0.5, 48_000.0, 1.5)],
        };
        let analysis = metrics::analyze(&rir, true).expect("analysis");
        let entry = DatasetEntry {
            id: "synthetic.wav".into(),
            source: EntrySource::Wav {
                path: PathBuf::from("synthetic.wav"),
            },
            metadata: Vec::new(),
            sample_rate: None,
        };
        let table = render_table(&entry, &analysis, None);
        assert!(table.contains("broadband"), "{table}");
        assert!(table.contains("31.5 Hz"), "{table}");
        assert!(table.contains("16 kHz"), "{table}");
        assert_eq!(table.matches(" Hz ").count() + table.matches(" kHz ").count(), 10);
    }

    #[test]
    fn cli_analyzes_a_synthetic_openair_tree() {
        let dir = TempDir::new("cli-openair");
        let wav = dir.path().join("wav/venue__stereo__a.wav");
        let left = geometric_rir(0.5, 48_000.0, 1.2);
        write_wav_f32(&wav, 48_000, &[left.clone(), left]);
        write_text(
            &dir.path().join("metadata.csv"),
            "file,rt60\nwav/venue__stereo__a.wav,0.50\n",
        );
        let args = args_for(dir.path());
        let stats = run(&args).expect("run cli on fixture");
        assert_eq!(stats.analyzed, 1);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn cli_reports_hdf5_entries_as_errors_and_keeps_wav_entries() {
        let dir = TempDir::new("cli-dechorate");
        write_text(
            &dir.path().join("dEchorate_database.csv"),
            ",filename,src_id,src_type,src_signal,room_code,mic_id\n\
             0,s,1,chirp,chirp,010000,1\n\
             1,s,2,chirp,chirp,010000,5\n",
        );
        write_wav_f32(
            &dir.path().join("010000/rir/1/1.wav"),
            48_000,
            &[geometric_rir(0.4, 48_000.0, 0.8)],
        );
        let args = args_for(dir.path());
        let stats = run(&args).expect("run cli on dEchorate fixture");
        assert_eq!(stats.analyzed, 1);
        assert_eq!(stats.errors, 1);
    }
}
