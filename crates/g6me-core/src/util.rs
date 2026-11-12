//! Small shared helpers.

use std::path::Path;

/// True when both paths refer to the same file on disk.
///
/// Comparing canonicalized paths is the portable part; on Unix we also compare
/// (dev, inode) because a file can be reached through several names, and
/// overwriting the input with a partially written output is the one mistake this
/// tool must never make.
pub fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        // One side does not exist yet (the normal case for the output): fall
        // back to a lexical comparison of the absolute paths.
        _ => abs(a) == abs(b),
    }
}

fn abs(p: &Path) -> std::path::PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(p)
    }
}

/// Turn a byte count into something a status line can show.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} {}", UNITS[0])
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Seconds as `h:mm:ss` / `m:ss`.
pub fn human_time(s: f64) -> String {
    if !s.is_finite() || s <= 0.0 {
        return "--:--".to_string();
    }
    let t = s.round() as u64;
    if t >= 3600 {
        format!("{}:{:02}:{:02}", t / 3600, (t % 3600) / 60, t % 60)
    } else {
        format!("{}:{:02}", t / 60, t % 60)
    }
}

/// A path as an ffmpeg/ffprobe argument.
///
/// A bare relative name is read as an option when it starts with `-`, and as a protocol
/// (`Live:` in `Live:Tokyo.flac`) when it has a colon, so relative paths get an explicit
/// `./` in front.
pub fn ffmpeg_arg(p: &Path) -> String {
    if p.is_relative() && !p.as_os_str().is_empty() {
        Path::new(".").join(p).display().to_string()
    } else {
        p.display().to_string()
    }
}

/// Safe filename for an output derived from an input.
/// Logical CPUs available to this process.
///
/// The ceiling the GUI's threads slider and the CLI's `--threads` both use, so
/// the two frontends cannot disagree about what the machine can run.
pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// Highest sensible worker count: one file per logical CPU.
///
/// Beyond that the workers only compete for the same cores, and each already
/// runs three threads plus two ffmpeg children.
pub fn max_threads() -> usize {
    cpu_count()
}

/// Clamp a requested worker count into `1..=max_threads()`.
pub fn clamp_threads(n: usize) -> usize {
    n.clamp(1, max_threads())
}

pub fn replace_extension(p: &Path, ext: &str) -> std::path::PathBuf {
    let mut name = p
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    if let Some(dot) = name.rfind('.') {
        // A dot at position 0 is the name itself, not an extension. Truncating there turns
        // ".profile" into "" and the output into ".flac": a hidden file with no stem, which
        // the filesystem happily accepts and nobody finds in a listing.
        if dot > 0 {
            name.truncate(dot);
        }
    }
    let mut out = p.parent().unwrap_or(Path::new("")).to_path_buf();
    out.push(format!("{name}.{ext}"));
    out
}

#[cfg(test)]
mod ffmpeg_arg_tests {
    use super::*;

    #[test]
    fn relative_names_cannot_read_as_options_or_protocols() {
        let a = ffmpeg_arg(Path::new("-dash.flac"));
        assert!(!a.starts_with('-'), "{a}");
        let a = ffmpeg_arg(Path::new("Live:Tokyo.flac"));
        assert!(a.starts_with('.'), "{a}");
        let abs = std::env::temp_dir().join("x.flac");
        assert_eq!(ffmpeg_arg(&abs), abs.display().to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn same_file_detects_identity() {
        let d = std::env::temp_dir().join(format!("g6me-util-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let a = d.join("a.bin");
        std::fs::write(&a, b"x").unwrap();
        assert!(same_file(&a, &PathBuf::from(&a)));
        assert!(same_file(&a, &d.join(".").join("a.bin")));
        assert!(!same_file(&a, &d.join("b.flac")));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn human_formatters() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_time(75.0), "1:15");
        assert_eq!(human_time(3661.0), "1:01:01");
        assert_eq!(human_time(0.0), "--:--");
    }

    #[test]
    fn extension_swap() {
        assert_eq!(
            replace_extension(Path::new("/x/y/z.mka"), "flac"),
            PathBuf::from("/x/y/z.flac")
        );
        assert_eq!(
            replace_extension(Path::new("noext"), "wav"),
            PathBuf::from("noext.wav")
        );
    }
}
