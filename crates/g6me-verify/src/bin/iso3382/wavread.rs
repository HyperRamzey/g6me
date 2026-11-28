//! WAV decoding for the ISO 3382 harness.
//!
//! The header parser is the workspace's own `g6me_core::wav::read_info_file`
//! (it already understands RIFF, `WAVE_FORMAT_EXTENSIBLE`, and the channel
//! mask); this module only adds the sample decoder the verifier did not need.
//! Supported encodings: unsigned 8-bit PCM, signed 16/24/32-bit PCM, and
//! IEEE-754 float 32/64. Everything else is rejected by name rather than
//! silently misread.

use crate::metrics::Rir;
use g6me_core::wav::{read_info_file, WavInfo};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

/// Refuse to allocate a sample buffer larger than this from an untrusted
/// header. A claimed 4 GiB data chunk is a corrupt file, not a long RIR.
const MAX_DATA_BYTES: u64 = 2 << 30;

/// Read a WAV file as de-interleaved f64 channels.
pub fn read_wav(path: &Path) -> Result<Rir> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("{}: cannot open: {e}", path.display()))?;
    let (info, offset) =
        read_info_file(&mut file).map_err(|e| format!("{}: bad WAV header: {e}", path.display()))?;
    let bytes = read_data(&mut file, &info, offset, path)?;
    let samples = decode(&bytes, info.bits, info.float)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    deinterleave(samples, info.channels as usize)
        .map(|channels| Rir {
            sample_rate: info.rate,
            channels,
        })
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn read_data(
    file: &mut std::fs::File,
    info: &WavInfo,
    offset: u64,
    path: &Path,
) -> Result<Vec<u8>> {
    if info.data_len > MAX_DATA_BYTES {
        return Err(format!(
            "{}: data chunk claims {} bytes, above the {} byte cap",
            path.display(),
            info.data_len,
            MAX_DATA_BYTES
        ));
    }
    let len = usize::try_from(info.data_len)
        .map_err(|_| format!("{}: data chunk does not fit in memory", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("{}: seek failed: {e}", path.display()))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)
        .map_err(|e| format!("{}: truncated sample data: {e}", path.display()))?;
    Ok(bytes)
}

/// Decode raw sample bytes to f64 in [-1, 1) for integer PCM (0 dBFS = 1.0).
fn decode(bytes: &[u8], bits: u32, float: bool) -> Result<Vec<f64>> {
    match (float, bits) {
        (false, 8) => Ok(bytes.iter().map(|b| (*b as f64 - 128.0) / 128.0).collect()),
        (false, 16) => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes(*c) as f64 / 32768.0)
            .collect()),
        (false, 24) => Ok(bytes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| {
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8; // sign-extend
                v as f64 / 8_388_608.0
            })
            .collect()),
        (false, 32) => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f64 / 2_147_483_648.0)
            .collect()),
        (true, 32) => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect()),
        (true, 64) => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()),
        (float, bits) => Err(format!(
            "unsupported sample encoding: {}{bits}-bit",
            if float { "float " } else { "PCM " }
        )),
    }
}

fn deinterleave(samples: Vec<f64>, channels: usize) -> Result<Vec<Vec<f64>>> {
    if channels == 0 {
        return Err("zero channels".into());
    }
    let frames = samples.len() / channels;
    if frames == 0 {
        return Err("no complete audio frames".into());
    }
    let mut out: Vec<Vec<f64>> = (0..channels).map(|_| Vec::with_capacity(frames)).collect();
    for frame in samples.chunks_exact(channels) {
        for (dst, s) in out.iter_mut().zip(frame.iter()) {
            dst.push(*s);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{write_wav_f32, write_wav_i16, write_wav_i24, TempDir};

    #[test]
    fn float32_round_trips_exactly() {
        let dir = TempDir::new("wav-f32");
        let path = dir.path().join("stereo.wav");
        let left: Vec<f64> = (0..64).map(|i| i as f64 / 64.0 - 0.5).collect();
        let right: Vec<f64> = left.iter().map(|v| -v).collect();
        write_wav_f32(&path, 48_000, &[left.clone(), right.clone()]);
        let rir = read_wav(&path).expect("read float32 WAV");
        assert_eq!(rir.sample_rate, 48_000);
        assert_eq!(rir.channels.len(), 2);
        assert_eq!(rir.frames(), 64);
        // f32 can represent these dyadic values exactly.
        assert_eq!(rir.channels[0], left);
        assert_eq!(rir.channels[1], right);
    }

    #[test]
    fn int16_and_int24_normalize_by_the_full_scale_power_of_two() {
        let dir = TempDir::new("wav-int");
        let p16 = dir.path().join("mono16.wav");
        write_wav_i16(
            &p16,
            44_100,
            &[vec![0.5, -0.5, 0.0, 32_767.0 / 32_768.0]],
        );
        let rir = read_wav(&p16).expect("read PCM16");
        assert_eq!(rir.sample_rate, 44_100);
        assert!((rir.channels[0][0] - 0.5).abs() < 1e-15);
        assert!((rir.channels[0][1] + 0.5).abs() < 1e-15);
        assert_eq!(rir.channels[0][2], 0.0);

        let p24 = dir.path().join("mono24.wav");
        write_wav_i24(
            &p24,
            48_000,
            &[vec![0.5, -0.25, 8_388_607.0 / 8_388_608.0]],
        );
        let rir = read_wav(&p24).expect("read PCM24");
        assert!((rir.channels[0][0] - 0.5).abs() < 1e-15, "{:?}", rir.channels[0]);
        assert!((rir.channels[0][1] + 0.25).abs() < 1e-15);
        // Negative values must be sign-extended, not read as large positives.
        assert!(rir.channels[0][1] < 0.0);
    }

    #[test]
    fn decoding_rejects_unknown_encodings() {
        assert!(decode(&[0u8; 8], 64, false).is_err());
        assert!(decode(&[0u8; 8], 12, true).is_err());
    }
}
