//! Test-only fixtures: tiny synthetic datasets under a private temporary
//! directory. No real OpenAIR/dEchorate/MeshRIR data is downloaded or required
//! (OpenAIR's site is suspended; the harness must not block on the network).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A unique temporary directory, removed on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "g6me-iso3382-{}-{}-{n}",
            tag,
            std::process::id()
        ));
        // A stale directory from a crashed earlier run is not this test's data.
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn ensure_parent(path: &Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture parent");
    }
}

pub fn write_raw(path: &Path, bytes: &[u8]) {
    ensure_parent(path);
    std::fs::write(path, bytes).expect("write fixture");
}

pub fn write_text(path: &Path, text: &str) {
    write_raw(path, text.as_bytes());
}

/// Write a WAV through the workspace writer so both size fields are patched;
/// `wav::header` alone emits zero sizes and is not a complete file.
fn write_wav(path: &Path, rate: u32, channels: &[Vec<f64>], bits: u32, float: bool, payload: Vec<u8>) {
    ensure_parent(path);
    let mut w = g6me_core::wav::WavWriter::create(path, rate, channels.len() as u16, bits, float, None)
        .expect("create fixture WAV");
    w.write(&payload).expect("write fixture WAV");
    w.finish().expect("finish fixture WAV");
}

/// Write interleaved 32-bit float WAV.
pub fn write_wav_f32(path: &Path, rate: u32, channels: &[Vec<f64>]) {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut payload = Vec::new();
    for frame in 0..frames {
        for ch in channels {
            payload.extend_from_slice(&(ch[frame] as f32).to_le_bytes());
        }
    }
    write_wav(path, rate, channels, 32, true, payload);
}

/// Write interleaved 16-bit PCM WAV, rounding half away from zero.
pub fn write_wav_i16(path: &Path, rate: u32, channels: &[Vec<f64>]) {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut payload = Vec::new();
    for frame in 0..frames {
        for ch in channels {
            let v = (ch[frame].clamp(-1.0, 1.0) * 32768.0).round() as i32;
            let v = v.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    write_wav(path, rate, channels, 16, false, payload);
}

/// Write interleaved 24-bit PCM WAV.
pub fn write_wav_i24(path: &Path, rate: u32, channels: &[Vec<f64>]) {
    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut payload = Vec::new();
    for frame in 0..frames {
        for ch in channels {
            let v = (ch[frame].clamp(-1.0, 1.0) * 8_388_608.0).round() as i32;
            let v = v.clamp(-8_388_608, 8_388_607) as u32;
            payload.extend_from_slice(&v.to_le_bytes()[..3]);
        }
    }
    write_wav(path, rate, channels, 24, false, payload);
}

fn npy_bytes(descr: &str, shape: &[usize], fortran: bool, payload: &[u8]) -> Vec<u8> {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    let shape_text = if dims.len() == 1 {
        format!("{},", dims[0])
    } else {
        dims.join(", ")
    };
    let fortran = if fortran { "True" } else { "False" };
    let mut header = format!(
        "{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': ({shape_text}), }}"
    );
    // NumPy terminates the header with a newline and pads so the header ends on
    // a 64-byte boundary. Purely cosmetic for this parser, but realistic.
    header.push('\n');
    let padded = header.len().div_ceil(64) * 64 - 10;
    while header.len() < padded {
        header.insert(header.len() - 1, ' ');
    }
    let mut out = Vec::with_capacity(10 + header.len() + payload.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn write_npy_f64(path: &Path, shape: &[usize], data: &[f64], fortran: bool) {
    let mut payload = Vec::with_capacity(data.len() * 8);
    for v in data {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    write_raw(path, &npy_bytes("<f8", shape, fortran, &payload));
}

pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f64], big_endian: bool) {
    let mut payload = Vec::with_capacity(data.len() * 4);
    for v in data {
        let b = if big_endian {
            (*v as f32).to_be_bytes()
        } else {
            (*v as f32).to_le_bytes()
        };
        payload.extend_from_slice(&b);
    }
    let descr = if big_endian { ">f4" } else { "<f4" };
    // `npy_bytes` does not care about byte order; the payload already has it.
    write_raw(path, &npy_bytes(descr, shape, false, &payload));
}

/// A decaying response with a closed-form envelope, for metric tests.
pub fn geometric_rir(t60_s: f64, sample_rate: f64, seconds: f64) -> Vec<f64> {
    let n = (seconds * sample_rate).round() as usize;
    let a = 10f64.powf(-3.0 / (t60_s * sample_rate));
    let mut v = Vec::with_capacity(n);
    let mut cur = 1.0;
    for _ in 0..n {
        v.push(cur);
        cur *= a;
    }
    v
}
