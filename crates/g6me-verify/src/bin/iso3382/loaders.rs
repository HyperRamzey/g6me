//! Read-only loaders for measured-RIR dataset directory layouts.
//!
//! Three layouts are recognised, in this order:
//!
//! * **MeshRIR** — a session directory containing `pos_mic.npy`, `pos_src.npy`
//!   and `ir_<mic>.npy` files. Each `ir_<mic>.npy` is a C-order array
//!   `(num_sources, ir_length)`; every row becomes one entry. The dataset's
//!   documented format is `.npy` for Python and `.mat` for MATLAB; `.mat` files
//!   are enumerated in a note but not decoded (no MAT reader in this
//!   dependency-free harness). Officially 48 kHz, 32768-sample IRs
//!   (<https://www.sh01.org/MeshRIR/>), so `.npy` entries default to 48 kHz.
//! * **dEchorate** — the official annotation CSV `dEchorate_database.csv`
//!   (`room_code`, `src_id`, `mic_id`, …) plus either actual HDF5 group
//!   `<room_code>/rir/<src>/<mic>` (the v3 dataset is a single 83 GB HDF5 file)
//!   or an exported tree `<root>/<room_code>/rir/<src>/<mic>.wav`. HDF5 payloads
//!   are enumerated from the CSV but not decoded: this harness has no HDF5
//!   dependency, and pretending otherwise would mean guessing at the file. The
//!   exported WAV tree is fully read. Room codes are the six wall states in
//!   floor, ceiling, west, south, east, north order; source ids 1–9 and mic ids
//!   1–31 follow the dataset constants, so the `99` silence/noise rows are
//!   skipped.
//! * **OpenAIR** — the library's WAV naming convention
//!   `<venue>__<format>__<ir-tag>.wav` (format `mono`, `stereo`, or `b-format`)
//!   as distributed in the audEERING `openair` package, with an optional
//!   per-file RT60 table (CSV with a file column and an RT60 column) and optional
//!   `*.txt` sidecars. Files are discovered under a `wav/` tree or by the
//!   double-underscore name. OpenAIR's site is suspended, so this loader never
//!   touches the network and works entirely from a local mirror.
//!
//! Anything else is a **generic** tree of WAV files, one entry each.
//!
//! All discovery is read-only and skips symlinks (a symlinked directory could
//! otherwise walk outside the dataset or loop).

use crate::metrics::Rir;
use crate::{npy, wavread};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    OpenAir,
    DEchorate,
    MeshRir,
    Generic,
}

impl DatasetKind {
    pub fn label(self) -> &'static str {
        match self {
            DatasetKind::OpenAir => "openair",
            DatasetKind::DEchorate => "dechorate",
            DatasetKind::MeshRir => "meshrir",
            DatasetKind::Generic => "generic",
        }
    }
}

/// Where one entry's samples come from.
#[derive(Clone, Debug)]
pub enum EntrySource {
    Wav {
        path: PathBuf,
    },
    Npy {
        path: PathBuf,
        row: usize,
        rows: usize,
    },
    /// Enumerated but not decodable without an HDF5 reader.
    Hdf5Group {
        path: PathBuf,
        group: String,
    },
}

impl EntrySource {
    pub fn kind_label(&self) -> &'static str {
        match self {
            EntrySource::Wav { .. } => "wav",
            EntrySource::Npy { .. } => "npy",
            EntrySource::Hdf5Group { .. } => "hdf5",
        }
    }
}

#[derive(Clone, Debug)]
pub struct DatasetEntry {
    pub id: String,
    pub source: EntrySource,
    pub metadata: Vec<(String, String)>,
    pub sample_rate: Option<u32>,
}

impl DatasetEntry {
    /// Test-only convenience for asserting parsed metadata.
    #[cfg(test)]
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub kind: DatasetKind,
    pub root: PathBuf,
    pub entries: Vec<DatasetEntry>,
    pub notes: Vec<String>,
}

/// A parsed OpenAIR file name: `<venue>__<format>__<tag>.wav`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAirName {
    pub venue: String,
    pub format: String,
    pub tag: String,
}

pub fn parse_openair_name(stem: &str) -> Option<OpenAirName> {
    let mut parts = stem.split("__");
    let venue = parts.next()?;
    let format = parts.next()?;
    let tag = parts.next()?;
    if venue.is_empty() || format.is_empty() {
        return None;
    }
    // The tag itself may contain `__`; rejoin anything left.
    let mut tag = tag.to_string();
    for rest in parts {
        tag.push_str("__");
        tag.push_str(rest);
    }
    Some(OpenAirName {
        venue: venue.to_string(),
        format: format.to_ascii_lowercase(),
        tag,
    })
}

/// The six wall states of a dEchorate room code, in dataset order.
pub const DECHORATE_WALLS: [&str; 6] = [
    "floor", "ceiling", "west", "south", "east", "north",
];

/// Parse a dEchorate room code such as `010000` into per-wall states.
/// Digits are `0` (absorbing), `1` (reflective), `2` (mixed panel state);
/// the ordering is ground truth from the dataset constants, the digit meanings
/// are read off the CSV flags and reported verbatim rather than interpreted.
pub fn parse_dechorate_room_code(code: &str) -> Option<Vec<(&'static str, char)>> {
    let mut chars = code.chars();
    let mut out = Vec::with_capacity(6);
    for wall in DECHORATE_WALLS {
        let c = chars.next()?;
        if !matches!(c, '0' | '1' | '2') {
            return None;
        }
        out.push((wall, c));
    }
    if chars.next().is_some() {
        return None;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Discovery helpers
// ---------------------------------------------------------------------------

fn walk(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("{}: cannot read dir: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let path = entry.path();
            let ft = entry
                .file_type()
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn rel_id(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn read_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Split one CSV line, honouring double-quoted fields and `""` escapes.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted => {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            }
            '"' if field.is_empty() => quoted = true,
            ',' if !quoted => out.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    out.push(field);
    out
}

fn csv_rows(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(|l| l.trim_end_matches('\r'))
        .filter(|l| !l.trim().is_empty())
        .map(parse_csv_line)
        .collect()
}

fn column(header: &[String], names: &[&str]) -> Option<usize> {
    header.iter().position(|h| {
        let h = h.trim().trim_start_matches('\u{feff}');
        names.iter().any(|n| h.eq_ignore_ascii_case(n))
    })
}

fn parse_id(raw: &str) -> Option<u32> {
    let v: f64 = raw.trim().parse().ok()?;
    if v.is_finite() && v >= 0.0 && v.fract() == 0.0 {
        Some(v as u32)
    } else {
        None
    }
}

fn key_values(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let split = line.find([':', '=']).map(|i| (i, line[i..].chars().next().unwrap()));
        let Some((i, sep)) = split else { continue };
        let key = line[..i].trim();
        let value = line[i + sep.len_utf8()..].trim();
        if !key.is_empty() && !value.is_empty() && key.len() <= 64 {
            out.push((key.to_string(), value.to_string()));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Detection and dispatch
// ---------------------------------------------------------------------------

/// Detect the dataset layout under `root` and enumerate its entries.
pub fn load_dataset(root: &Path) -> Result<Dataset> {
    if !root.is_dir() {
        return Err(format!("{}: not a directory", root.display()));
    }
    let files = walk(root)?;
    let has_meshrir = files
        .iter()
        .any(|p| file_name(p).eq_ignore_ascii_case("pos_mic.npy"));
    if has_meshrir {
        return load_meshrir(root, &files);
    }
    let has_dechorate = files.iter().any(|p| {
        file_name(p).eq_ignore_ascii_case("dechorate_database.csv")
            || matches!(extension(p).as_deref(), Some("hdf5" | "h5"))
    }) || files.iter().any(|p| dechorate_key_from_path(p).is_some());
    if has_dechorate {
        return load_dechorate(root, &files);
    }
    let openair = files.iter().any(|p| {
        extension(p).as_deref() == Some("wav")
            && (parse_openair_name(&file_stem(p)).is_some()
                || p.components()
                    .any(|c| c.as_os_str().eq_ignore_ascii_case("wav")))
    });
    if openair {
        return load_openair(root, &files);
    }
    load_generic(root, &files)
}

// ---------------------------------------------------------------------------
// MeshRIR
// ---------------------------------------------------------------------------

fn load_meshrir(root: &Path, files: &[PathBuf]) -> Result<Dataset> {
    let mut dirs: Vec<PathBuf> = files
        .iter()
        .filter(|p| file_name(p).eq_ignore_ascii_case("pos_mic.npy"))
        .filter_map(|p| p.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    dirs.dedup();

    let mut entries = Vec::new();
    let mut notes = vec![
        "MeshRIR: .npy arrays decoded (C order, float32/float64); .mat files are not decoded"
            .to_string(),
    ];
    for dir in &dirs {
        let mut ir_files: Vec<&PathBuf> = files
            .iter()
            .filter(|p| {
                p.parent() == Some(dir.as_path())
                    && extension(p).as_deref() == Some("npy")
                    && file_stem(p).starts_with("ir_")
            })
            .collect();
        ir_files.sort();
        if ir_files.is_empty() {
            notes.push(format!("{}: pos_mic.npy without ir_*.npy", dir.display()));
            continue;
        }
        let sample_rate = meshrir_sample_rate(dir, files);
        for path in ir_files {
            let shape = npy::npy_shape(path)?;
            let (rows, len) = match shape.as_slice() {
                [rows, len] => (*rows, *len),
                [len] => (1, *len),
                other => {
                    notes.push(format!(
                        "{}: expected a 1-D or 2-D IR array, found shape {other:?}",
                        path.display()
                    ));
                    continue;
                }
            };
            let stem = file_stem(path);
            for row in 0..rows {
                entries.push(DatasetEntry {
                    id: format!("{}/{}[src{row}]", rel_id(root, dir), stem),
                    source: EntrySource::Npy {
                        path: path.clone(),
                        row,
                        rows,
                    },
                    metadata: vec![
                        ("layout".into(), "meshrir".into()),
                        ("mic_index".into(), stem.trim_start_matches("ir_").into()),
                        ("source_index".into(), row.to_string()),
                        ("ir_length".into(), len.to_string()),
                    ],
                    sample_rate,
                });
            }
        }
    }
    Ok(Dataset {
        kind: DatasetKind::MeshRir,
        root: root.to_path_buf(),
        entries,
        notes,
    })
}

/// The documented MeshRIR rate is 48 kHz; a JSON sidecar may carry it.
fn meshrir_sample_rate(dir: &Path, files: &[PathBuf]) -> Option<u32> {
    for path in files.iter().filter(|p| {
        p.parent() == Some(dir) && extension(p).as_deref() == Some("json")
    }) {
        let Some(text) = read_text(path) else { continue };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        if let Some(rate) = json_number(&value, &["sample_rate", "samplerate", "fs", "Fs"]) {
            if rate > 0.0 && rate <= 768_000.0 && rate.fract() == 0.0 {
                return Some(rate as u32);
            }
        }
    }
    Some(48_000)
}

fn json_number(value: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if keys.iter().any(|want| k.eq_ignore_ascii_case(want)) {
                    if let Some(n) = v.as_f64() {
                        return Some(n);
                    }
                }
            }
            map.values().find_map(|v| json_number(v, keys))
        }
        serde_json::Value::Array(items) => items.iter().find_map(|v| json_number(v, keys)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// dEchorate
// ---------------------------------------------------------------------------

/// If `path` lies in an exported dEchorate tree, return
/// `(room_code, source_id, mic_id)` from the path components.
fn dechorate_key_from_path(path: &Path) -> Option<(String, u32, u32)> {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    // The deepest matching component is the room code; an earlier six-digit
    // directory above the dataset root must not shadow it.
    let code_idx = (0..parts.len().saturating_sub(3))
        .rev()
        .find(|&i| parts[i].len() == 6 && parts[i].chars().all(|c| matches!(c, '0' | '1' | '2')))?;
    let mut rest = parts[code_idx + 1..].iter();
    if rest.clone().next().is_some_and(|p| p.eq_ignore_ascii_case("rir")) {
        rest.next();
    }
    let src = parse_id(rest.next()?)?;
    let mic_name = rest.next()?;
    let mic = parse_id(Path::new(mic_name).file_stem()?.to_str()?)?;
    Some((parts[code_idx].clone(), src, mic))
}

fn exported_dechorate_wav(
    files: &[PathBuf],
    code: &str,
    src: u32,
    mic: u32,
) -> Option<PathBuf> {
    files
        .iter()
        .find(|p| {
            extension(p).as_deref() == Some("wav")
                && dechorate_key_from_path(p) == Some((code.to_string(), src, mic))
        })
        .cloned()
}

fn load_dechorate(root: &Path, files: &[PathBuf]) -> Result<Dataset> {
    let hdf5_path = files
        .iter()
        .find(|p| matches!(extension(p).as_deref(), Some("hdf5" | "h5")))
        .cloned()
        .unwrap_or_else(|| root.join("dechorate.hdf5"));
    let csv_path = files.iter().find(|p| {
        file_name(p).eq_ignore_ascii_case("dechorate_database.csv")
    });
    let mut entries: BTreeMap<(String, u32, u32), DatasetEntry> = BTreeMap::new();
    let mut notes = vec![
        "dEchorate: HDF5 payloads are enumerated from the official CSV but not decoded (no HDF5 \
         dependency); exported WAVs at <room>/rir/<src>/<mic>.wav are read"
            .to_string(),
    ];

    if let Some(csv) = csv_path {
        let text = read_text(csv)
            .ok_or_else(|| format!("{}: cannot read annotation CSV", csv.display()))?;
        let rows = csv_rows(&text);
        if rows.is_empty() {
            return Err(format!("{}: empty annotation CSV", csv.display()));
        }
        let header = &rows[0];
        let code_col = column(header, &["room_code"])
            .ok_or_else(|| format!("{}: no room_code column", csv.display()))?;
        let src_col = column(header, &["src_id"])
            .ok_or_else(|| format!("{}: no src_id column", csv.display()))?;
        let mic_col = column(header, &["mic_id"])
            .ok_or_else(|| format!("{}: no mic_id column", csv.display()))?;
        let signal_col = column(header, &["src_signal"]);
        for row in &rows[1..] {
            let Some(code) = row.get(code_col).map(|s| s.trim()) else {
                continue;
            };
            let Some(walls) = parse_dechorate_room_code(code) else {
                continue;
            };
            let Some(src) = row.get(src_col).and_then(|s| parse_id(s)) else {
                continue;
            };
            let Some(mic) = row.get(mic_col).and_then(|s| parse_id(s)) else {
                continue;
            };
            // Constants: source ids 1..=9, microphone ids 1..=31; id 99 marks
            // silence/noise recordings that have no room response.
            if !(1..=9).contains(&src) || !(1..=31).contains(&mic) {
                continue;
            }
            let key = (code.to_string(), src, mic);
            if entries.contains_key(&key) {
                continue;
            }
            let mut metadata = vec![
                ("layout".into(), "dechorate".into()),
                ("room_code".into(), code.to_string()),
                ("src_id".into(), src.to_string()),
                ("mic_id".into(), mic.to_string()),
            ];
            for (wall, state) in &walls {
                metadata.push((format!("wall_{wall}"), state.to_string()));
            }
            if let Some(signal) = signal_col.and_then(|c| row.get(c)) {
                if !signal.trim().is_empty() {
                    metadata.push(("src_signal".into(), signal.trim().to_string()));
                }
            }
            let source = match exported_dechorate_wav(files, code, src, mic) {
                Some(path) => EntrySource::Wav { path },
                None => EntrySource::Hdf5Group {
                    path: hdf5_path.clone(),
                    group: format!("{code}/rir/{src}/{mic}"),
                },
            };
            entries.insert(
                key,
                DatasetEntry {
                    id: format!("{code}/rir/{src}/{mic}"),
                    source,
                    metadata,
                    sample_rate: Some(48_000),
                },
            );
        }
    } else {
        // No CSV: enumerate an exported tree directly.
        for path in files {
            let Some((code, src, mic)) = dechorate_key_from_path(path) else {
                continue;
            };
            if extension(path).as_deref() != Some("wav") {
                continue;
            }
            let key = (code.clone(), src, mic);
            entries.entry(key).or_insert_with(|| DatasetEntry {
                id: format!("{code}/rir/{src}/{mic}"),
                source: EntrySource::Wav { path: path.clone() },
                metadata: vec![
                    ("layout".into(), "dechorate".into()),
                    ("room_code".into(), code),
                    ("src_id".into(), src.to_string()),
                    ("mic_id".into(), mic.to_string()),
                ],
                sample_rate: Some(48_000),
            });
        }
    }
    if entries.is_empty() {
        notes.push("dEchorate layout detected but no room/source/mic entries were found".into());
    }
    Ok(Dataset {
        kind: DatasetKind::DEchorate,
        root: root.to_path_buf(),
        entries: entries.into_values().collect(),
        notes,
    })
}

// ---------------------------------------------------------------------------
// OpenAIR
// ---------------------------------------------------------------------------

/// RT60 values keyed by relative path and by bare file name.
fn openair_rt60_table(files: &[PathBuf]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for path in files.iter().filter(|p| extension(p).as_deref() == Some("csv")) {
        let Some(text) = read_text(path) else { continue };
        let rows = csv_rows(&text);
        if rows.len() < 2 {
            continue;
        }
        let header = &rows[0];
        let Some(file_col) = column(header, &["file", "filename", "path", "wav"]) else {
            continue;
        };
        let Some(rt_col) = column(
            header,
            &["rt60", "reverberation-time", "reverberation_time", "rt"],
        ) else {
            continue;
        };
        for row in &rows[1..] {
            let (Some(file), Some(rt)) = (row.get(file_col), row.get(rt_col)) else {
                continue;
            };
            let file = file.trim();
            let rt = rt.trim();
            if file.is_empty() || rt.is_empty() {
                continue;
            }
            map.entry(file.to_string()).or_insert_with(|| rt.to_string());
            if let Some(name) = Path::new(file).file_name() {
                map.entry(name.to_string_lossy().into_owned())
                    .or_insert_with(|| rt.to_string());
            }
        }
    }
    map
}

fn load_openair(root: &Path, files: &[PathBuf]) -> Result<Dataset> {
    let rt60 = openair_rt60_table(files);
    let mut entries = Vec::new();
    for path in files.iter().filter(|p| extension(p).as_deref() == Some("wav")) {
        let id = rel_id(root, path);
        let stem = file_stem(path);
        let mut metadata = vec![("layout".into(), "openair".into())];
        if let Some(name) = parse_openair_name(&stem) {
            metadata.push(("venue".into(), name.venue));
            metadata.push(("format".into(), name.format));
            metadata.push(("ir_tag".into(), name.tag));
        }
        let table_key = rt60
            .get(&id)
            .or_else(|| rt60.get(&file_name(path)))
            .or_else(|| rt60.get(&stem));
        if let Some(value) = table_key {
            metadata.push(("rt60_500hz_s".into(), value.clone()));
        }
        for sidecar in [path.with_extension("txt"), path.with_file_name("info.txt")] {
            if sidecar == *path {
                continue;
            }
            if let Some(text) = read_text(&sidecar) {
                for (k, v) in key_values(&text) {
                    if !metadata.iter().any(|(existing, _)| existing == &k) {
                        metadata.push((k, v));
                    }
                }
            }
        }
        entries.push(DatasetEntry {
            id,
            source: EntrySource::Wav { path: path.clone() },
            metadata,
            sample_rate: None,
        });
    }
    Ok(Dataset {
        kind: DatasetKind::OpenAir,
        root: root.to_path_buf(),
        entries,
        notes: vec![
            "OpenAIR: discovered locally by *.wav + '<venue>__<format>__<tag>' naming or a wav/ \
             tree; RT60 read from any CSV carrying file + RT60 columns; no network access"
                .to_string(),
        ],
    })
}

// ---------------------------------------------------------------------------
// Generic + reading
// ---------------------------------------------------------------------------

fn load_generic(root: &Path, files: &[PathBuf]) -> Result<Dataset> {
    let mut entries = Vec::new();
    let mut notes = Vec::new();
    for path in files.iter().filter(|p| extension(p).as_deref() == Some("wav")) {
        entries.push(DatasetEntry {
            id: rel_id(root, path),
            source: EntrySource::Wav { path: path.clone() },
            metadata: vec![("layout".into(), "generic".into())],
            sample_rate: None,
        });
    }
    if entries.is_empty() {
        notes.push("no WAV, MeshRIR .npy, or dEchorate CSV/HDF5 files found".into());
    }
    Ok(Dataset {
        kind: DatasetKind::Generic,
        root: root.to_path_buf(),
        entries,
        notes,
    })
}

/// Read one entry's samples. Errors are explicit: an enumerated HDF5 group is
/// reported as not decodable rather than skipped silently.
pub fn read_entry(entry: &DatasetEntry) -> Result<Rir> {
    match &entry.source {
        EntrySource::Wav { path } => wavread::read_wav(path),
        EntrySource::Npy { path, row, rows } => {
            let array = npy::read_npy(path)?;
            let len = match array.shape.as_slice() {
                [_, len] => *len,
                [len] => *len,
                other => {
                    return Err(format!(
                        "{}: expected a 1-D or 2-D IR array, found shape {other:?}",
                        path.display()
                    ))
                }
            };
            if *row >= *rows || *rows == 0 || len == 0 {
                return Err(format!("{}: row {row} of {rows} is out of range", path.display()));
            }
            let start = row
                .checked_mul(len)
                .ok_or_else(|| format!("{}: row offset overflow", path.display()))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| format!("{}: row end overflow", path.display()))?;
            let data = array
                .data
                .get(start..end)
                .ok_or_else(|| format!("{}: truncated array payload", path.display()))?
                .to_vec();
            Ok(Rir {
                sample_rate: entry.sample_rate.unwrap_or(48_000),
                channels: vec![data],
            })
        }
        EntrySource::Hdf5Group { path, group } => Err(format!(
            "HDF5 group '{group}' in {} is not decoded by this harness; export the group to a \
             float WAV under <root>/<room_code>/rir/<src>/<mic>.wav (or read it with h5py)",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        write_npy_f64, write_text, write_wav_f32, TempDir,
    };

    const DECHORATE_CSV: &str = "\
,filename,src_id,src_type,src_signal,room_code,mic_id
0,session-a,1.0,chirp,chirp,010000,1.0
1,session-a,2.0,chirp,chirp,010000,5.0
2,session-a,99.0,silence,silence,010000,3.0
";

    fn openair_fixture(root: &Path) {
        let wav = root.join("wav/venue__stereo__tag_one.wav");
        write_wav_f32(&wav, 48_000, &[vec![0.5; 64], vec![-0.5; 64]]);
        write_text(
            &root.join("metadata.csv"),
            "file,reverberation-time\nwav/venue__stereo__tag_one.wav,1.40\n",
        );
        write_text(&wav.with_extension("txt"), "RT60: 1.35\nSource: dodecahedron\n");
    }

    #[test]
    fn openair_layout_name_and_metadata_are_loaded() {
        let dir = TempDir::new("openair");
        openair_fixture(dir.path());
        let dataset = load_dataset(dir.path()).expect("load openair fixture");
        assert_eq!(dataset.kind, DatasetKind::OpenAir);
        assert_eq!(dataset.entries.len(), 1);
        let entry = &dataset.entries[0];
        assert_eq!(entry.id, "wav/venue__stereo__tag_one.wav");
        assert_eq!(entry.metadata_value("venue"), Some("venue"));
        assert_eq!(entry.metadata_value("format"), Some("stereo"));
        assert_eq!(entry.metadata_value("ir_tag"), Some("tag_one"));
        // CSV table wins over the sidecar value of the same key.
        assert_eq!(entry.metadata_value("rt60_500hz_s"), Some("1.40"));
        assert_eq!(entry.metadata_value("Source"), Some("dodecahedron"));
        let rir = read_entry(entry).expect("read OpenAIR wav");
        assert_eq!(rir.sample_rate, 48_000);
        assert_eq!(rir.channels.len(), 2);
    }

    #[test]
    fn openair_name_parser_keeps_extra_double_underscores() {
        let name = parse_openair_name("1st_baptist_nashville__stereo__far__wide").unwrap();
        assert_eq!(name.venue, "1st_baptist_nashville");
        assert_eq!(name.format, "stereo");
        assert_eq!(name.tag, "far__wide");
        assert!(parse_openair_name("plain_name").is_none());
    }

    #[test]
    fn dechorate_csv_enumerates_groups_and_reads_exported_wavs() {
        let dir = TempDir::new("dechorate");
        write_text(&dir.path().join("dEchorate_database.csv"), DECHORATE_CSV);
        write_wav_f32(
            &dir.path().join("010000/rir/1/1.wav"),
            48_000,
            &[vec![0.25; 128]],
        );
        let dataset = load_dataset(dir.path()).expect("load dechorate fixture");
        assert_eq!(dataset.kind, DatasetKind::DEchorate);
        // The 99 silence row is not a room response and is skipped.
        assert_eq!(dataset.entries.len(), 2);
        let first = dataset
            .entries
            .iter()
            .find(|e| e.id == "010000/rir/1/1")
            .expect("exported entry");
        assert!(matches!(first.source, EntrySource::Wav { .. }));
        assert_eq!(first.metadata_value("wall_ceiling"), Some("1"));
        assert_eq!(first.metadata_value("wall_floor"), Some("0"));
        let rir = read_entry(first).expect("read exported dEchorate wav");
        assert_eq!(rir.frames(), 128);

        let missing = dataset
            .entries
            .iter()
            .find(|e| e.id == "010000/rir/2/5")
            .expect("HDF5-enumerated entry");
        assert!(matches!(missing.source, EntrySource::Hdf5Group { .. }));
        let err = read_entry(missing).unwrap_err();
        assert!(err.contains("HDF5 group '010000/rir/2/5'"), "{err}");
    }

    #[test]
    fn dechorate_room_code_parsing_is_exact() {
        let walls = parse_dechorate_room_code("010000").unwrap();
        assert_eq!(walls.len(), 6);
        assert_eq!(walls[0], ("floor", '0'));
        assert_eq!(walls[1], ("ceiling", '1'));
        assert_eq!(walls[5], ("north", '0'));
        assert!(parse_dechorate_room_code("01000").is_none());
        assert!(parse_dechorate_room_code("0100000").is_none());
        assert!(parse_dechorate_room_code("01000x").is_none());
    }

    #[test]
    fn meshrir_layout_enumerates_each_source_row() {
        let dir = TempDir::new("meshrir");
        let session = dir.path().join("S1-M3969_npy");
        write_npy_f64(&session.join("pos_mic.npy"), &[2, 3], &[0.0; 6], false);
        write_npy_f64(&session.join("pos_src.npy"), &[1, 3], &[1.0, 2.0, 3.0], false);
        let mic0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mic1 = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8];
        write_npy_f64(&session.join("ir_0.npy"), &[2, 4], &mic0, false);
        write_npy_f64(&session.join("ir_1.npy"), &[2, 4], &mic1, false);
        write_text(&session.join("metadata.json"), "{\"sample_rate\": 48000}");

        let dataset = load_dataset(dir.path()).expect("load meshrir fixture");
        assert_eq!(dataset.kind, DatasetKind::MeshRir);
        assert_eq!(dataset.entries.len(), 4, "2 mics x 2 sources");
        let entry = dataset
            .entries
            .iter()
            .find(|e| e.id.ends_with("ir_1[src1]"))
            .expect("second source of second mic");
        assert_eq!(entry.sample_rate, Some(48_000));
        let rir = read_entry(entry).expect("read npy row");
        assert_eq!(rir.sample_rate, 48_000);
        assert_eq!(rir.channels[0], mic1[4..8].to_vec());
    }

    #[test]
    fn generic_tree_falls_back_to_wavs() {
        let dir = TempDir::new("generic");
        write_wav_f32(&dir.path().join("a.wav"), 44_100, &[vec![0.0; 32]]);
        write_wav_f32(&dir.path().join("sub/b.wav"), 48_000, &[vec![0.0; 32]]);
        let dataset = load_dataset(dir.path()).expect("load generic fixture");
        assert_eq!(dataset.kind, DatasetKind::Generic);
        assert_eq!(dataset.entries.len(), 2);
        assert!(dataset.entries.iter().all(|e| matches!(e.source, EntrySource::Wav { .. })));
    }

    #[test]
    fn empty_directory_is_an_empty_generic_dataset() {
        let dir = TempDir::new("empty");
        let dataset = load_dataset(dir.path()).expect("empty dir");
        assert_eq!(dataset.kind, DatasetKind::Generic);
        assert!(dataset.entries.is_empty());
        assert!(!dataset.notes.is_empty());
    }

    #[test]
    fn csv_parser_handles_quotes_and_commas() {
        let row = parse_csv_line("\"a,b\",plain,\"quote \"\"x\"\"\"");
        assert_eq!(row, vec!["a,b", "plain", "quote \"x\""]);
    }
}
