//! Minimal RIFF/WAVE reader and writer.
//!
//! ffmpeg handles every real container; this exists for the two places where
//! spawning a process for a handful of bytes is silly: the tests build and check
//! their own WAVs, and the live monitor wraps a PCM stream in a WAV header when
//! asked to record a file.
//!
//! Multi-channel files are written as `WAVE_FORMAT_EXTENSIBLE` with an explicit
//! channel mask, because that mask is exactly the field the C reference and
//! ffmpeg both read to learn the channel *order*. A plain PCM header carries no
//! order at all, which is how 7.1 files end up with their surrounds in the front
//! slots.

use crate::error::{Error, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

const SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];
const SUBTYPE_FLOAT: [u8; 16] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Standard channel order for a mask-free file, most-common-first.
pub fn guess_mask(channels: u16) -> u32 {
    const BITS: [u32; 18] = [
        0x1,    // FL
        0x2,    // FR
        0x4,    // FC
        0x8,    // LFE
        0x10,   // BL
        0x20,   // BR
        0x200,  // SL
        0x400,  // SR
        0x40,   // FLC
        0x80,   // FRC
        0x100,  // BC
        0x800,  // FLC(SLc?)
        0x1000, // FRC
        0x2000, // LTR
        0x4000, // LTR
        0x8000, // RSL
        0x10000, 0x20000,
    ];
    BITS.iter()
        .take(channels.min(BITS.len() as u16) as usize)
        .fold(0, |acc, bit| acc | bit)
}

/// Classic 44-byte header when the file is 1–2 channels and needs no mask,
/// otherwise the 58-byte extensible header.
pub fn header(rate: u32, channels: u16, bits: u32, float: bool, mask: Option<u32>) -> Vec<u8> {
    let block = channels as u32 * bits / 8;
    let extensible = channels > 2 || mask.is_some_and(|m| m != guess_mask(channels));
    let mut h = Vec::with_capacity(58);
    let tag = if float {
        WAVE_FORMAT_IEEE_FLOAT
    } else {
        WAVE_FORMAT_PCM
    };
    let fmt_tag = if extensible {
        WAVE_FORMAT_EXTENSIBLE
    } else {
        tag
    };
    let fmt_len = if extensible { 40u32 } else { 16u32 };
    // RIFF header: size is patched by the writer, so zero it here.
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&0u32.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&fmt_len.to_le_bytes());
    h.extend_from_slice(&fmt_tag.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * block).to_le_bytes());
    h.extend_from_slice(&(block as u16).to_le_bytes());
    h.extend_from_slice(&(bits as u16).to_le_bytes());
    if extensible {
        // wValidBitsPerSample: the container and the payload have the same width
        // here, because ffmpeg hands us exactly the precision we asked for.
        h.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        h.extend_from_slice(&(bits as u16).to_le_bytes()); // wValidBitsPerSample
        h.extend_from_slice(&(mask.unwrap_or_else(|| guess_mask(channels))).to_le_bytes());
        h.extend_from_slice(if float { &SUBTYPE_FLOAT } else { &SUBTYPE_PCM });
    }
    h.extend_from_slice(b"data");
    h.extend_from_slice(&0u32.to_le_bytes());
    h
}

/// Streaming WAV writer: header first, sizes patched on [`finish`].
pub struct WavWriter {
    f: std::fs::File,
    /// The length of the header that was actually written. Not a constant: a classic
    /// 2-channel header is 44 bytes and a `WAVE_FORMAT_EXTENSIBLE` one is 68, and both
    /// size fields patched at the end are measured from it.
    hdr_len: u64,
    data_len: u64,
}

impl WavWriter {
    pub fn create(
        path: &Path,
        rate: u32,
        channels: u16,
        bits: u32,
        float: bool,
        mask: Option<u32>,
    ) -> Result<Self> {
        let hdr = header(rate, channels, bits, float, mask);
        // Recorded rather than recomputed at the end: the header that exists and the
        // header guessed from the channel count are different lengths as soon as a custom
        // mask is involved, and the two size fields live at the end of whichever one was
        // actually written.
        let hdr_len = hdr.len() as u64;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&hdr)?;
        Ok(Self {
            f,
            hdr_len,
            data_len: 0,
        })
    }

    pub fn write(&mut self, pcm: &[u8]) -> Result<()> {
        self.f.write_all(pcm)?;
        self.data_len += pcm.len() as u64;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64> {
        let total = self.data_len;
        // RIFF chunk size = file size - 8; data chunk size = payload. Both measured from
        // `hdr_len`, because a hard-coded 36 (the 44-byte classic header minus 8) writes a
        // RIFF size 24 bytes short onto every 68-byte extensible file — which is every
        // multichannel WAV this program writes. A strict parser honours that size and
        // drops the last 24 bytes of audio, so the file plays and is quietly truncated.
        self.f.seek(SeekFrom::Start(4))?;
        self.f
            .write_all(&((total + self.hdr_len - 8) as u32).to_le_bytes())?;
        // The header ends with "data" followed by its 4-byte size, so the size
        // field is the last 4 bytes of the header — writing 8 bytes back would
        // land on the "data" tag itself and ffmpeg would reject the file.
        self.f.seek(SeekFrom::Start(self.hdr_len - 4))?;
        self.f.write_all(&(total as u32).to_le_bytes())?;
        self.f.flush()?;
        Ok(total)
    }
}

/// Convenience used by the tests: an s32 WAV with a standard mask.
pub fn write_pcm_s32(path: &Path, rate: u32, channels: u16, pcm: &[u8]) -> Result<()> {
    let mut w = WavWriter::create(path, rate, channels, 32, false, None)?;
    w.write(pcm)?;
    w.finish()?;
    Ok(())
}

/// What a WAV header claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavInfo {
    pub rate: u32,
    pub channels: u16,
    pub bits: u32,
    pub float: bool,
    pub mask: Option<u32>,
    pub data_len: u64,
}

impl WavInfo {
    pub fn frames(&self) -> u64 {
        let b = (self.channels as u64) * (self.bits as u64 / 8);
        // A header claiming zero bytes per frame is corrupt, not infinite.
        self.data_len.checked_div(b).unwrap_or(0)
    }
}

/// Parse a WAV header and report where the samples start.
pub fn read_info(path: &Path) -> Result<(WavInfo, u64)> {
    read_info_file(&mut std::fs::File::open(path)?)
}

pub fn read_info_file(f: &mut std::fs::File) -> Result<(WavInfo, u64)> {
    let mut riff = [0u8; 12];
    f.read_exact(&mut riff).map_err(short)?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(Error::msg("not a RIFF/WAVE file"));
    }
    let mut info = WavInfo {
        rate: 0,
        channels: 0,
        bits: 0,
        float: false,
        mask: None,
        data_len: 0,
    };
    loop {
        let mut ch = [0u8; 8];
        match f.read_exact(&mut ch) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(Error::msg("WAV has no data chunk"))
            }
            Err(e) => return Err(Error::Io(e)),
        }
        // The chunk header is 4 bytes of id followed by 4 bytes of size.
        let id = &ch[0..4];
        let size = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]) as u64;
        if id == b"fmt " {
            // Cap checked before the allocation, not after. A chunk header can claim any
            // 32-bit size, and `vec![0u8; size]` on the strength of it means a 44-byte
            // file can ask for 4 GiB of RAM before anything reads it and notices. The
            // largest fmt chunk that exists in the wild is 40 bytes.
            if size > 1024 {
                return Err(Error::msg("absurd WAV fmt chunk size"));
            }
            let mut b = vec![0u8; size as usize];
            f.read_exact(&mut b).map_err(short)?;
            if b.len() < 16 {
                return Err(Error::msg("short WAV fmt chunk"));
            }
            let tag = u16::from_le_bytes([b[0], b[1]]);
            info.channels = u16::from_le_bytes([b[2], b[3]]);
            info.rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            info.bits = u16::from_le_bytes([b[14], b[15]]) as u32;
            info.float = tag == WAVE_FORMAT_IEEE_FLOAT;
            if tag == WAVE_FORMAT_EXTENSIBLE && b.len() >= 40 {
                // cbSize is at 16, so dwChannelMask follows the 2-byte valid-bits
                // field at 20, and the 16-byte sub-format GUID starts at 24.
                info.mask = Some(u32::from_le_bytes([b[20], b[21], b[22], b[23]]));
                info.float = b[24] == 0x03; // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
            } else if info.channels > 2 {
                info.mask = Some(guess_mask(info.channels));
            }
        } else if id == b"data" {
            // Samples with no format in front of them are not a stream. Returned as a
            // parsed header it says 0 channels at 0 Hz and 0 bits, which every caller
            // believes until it divides.
            if info.channels == 0 || info.rate == 0 || info.bits == 0 {
                return Err(Error::msg("WAV data chunk before any fmt chunk"));
            }
            let off = f.stream_position()?;
            info.data_len = size;
            return Ok((info, off));
        } else {
            f.seek(SeekFrom::Current(size as i64 + (size & 1) as i64))?;
        }
    }
}

fn short(e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        Error::msg("file is shorter than its WAV header")
    } else {
        Error::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("g6me-wav-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    /// The declared sizes of an 8-channel file, read back as bytes rather than through
    /// this module's own reader: a reader that trusts the header cannot catch a header
    /// that lies, and `finish()` used to patch 36 bytes of header onto a 68-byte one.
    #[test]
    fn multichannel_wav_declares_its_real_sizes() {
        let p = tmp("8ch-sizes.wav");
        let pcm = vec![0x5au8; 1024 * 8 * 4];
        write_pcm_s32(&p, 48000, 8, &pcm).unwrap();
        let b = std::fs::read(&p).unwrap();
        let len = b.len() as u64;
        assert_eq!(len, 68 + pcm.len() as u64, "extensible header is 68 bytes");
        assert_eq!(&b[60..64], b"data", "data chunk must sit at 60");
        let riff = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64;
        let data = u32::from_le_bytes([b[64], b[65], b[66], b[67]]) as u64;
        assert_eq!(
            riff,
            len - 8,
            "RIFF size is the file minus its first 8 bytes"
        );
        assert_eq!(data, pcm.len() as u64, "data size is the payload");
    }

    /// Two channels with a non-default mask is extensible too, and the mask is not passed
    /// to the patch-up pass: `finish()` rebuilt the header with `mask: None`, got a 44-byte
    /// layout, and wrote the data size over the SubFormat GUID while leaving the real data
    /// size at zero.
    #[test]
    fn a_custom_mask_on_two_channels_keeps_its_guid_and_sizes() {
        let p = tmp("2ch-mask.wav");
        let pcm = vec![0x11u8; 512 * 2 * 2];
        let mut w = WavWriter::create(&p, 48000, 2, 16, false, Some(0x4)).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[60..64], b"data", "header stayed extensible");
        assert_eq!(
            &b[44..50],
            &SUBTYPE_PCM[0..6],
            "the SubFormat GUID must survive the size patch"
        );
        let riff = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64;
        assert_eq!(riff, b.len() as u64 - 8);
        let data = u32::from_le_bytes([b[64], b[65], b[66], b[67]]) as u64;
        assert_eq!(data, pcm.len() as u64, "the real data size, not zero");
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(info.mask, Some(0x4));
        assert_eq!(info.frames(), 512);
        assert_eq!(off as usize + pcm.len(), b.len());
    }

    /// A `data` chunk before any `fmt ` describes nothing. Returned as a valid file it
    /// becomes a stream with 0 channels and 0 Hz, and the frame count is a division that
    /// only a `checked_div` keeps out of trouble.
    #[test]
    fn data_before_fmt_is_refused_not_trusted() {
        let p = tmp("data-first.wav");
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&36u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"data");
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, &b).unwrap();
        let e = read_info(&p).unwrap_err();
        assert!(
            e.to_string().contains("fmt"),
            "the error has to name what is missing: {e}"
        );
    }

    /// A 4 GiB `fmt ` size in a 44-byte file must be refused by looking at the number.
    /// Allocating first means a corrupt or hostile header asks for 4 GiB of RAM before
    /// anything notices the file is shorter than its own chunk.
    #[test]
    fn an_absurd_fmt_size_is_refused_before_it_is_allocated() {
        let p = tmp("huge-fmt.wav");
        let mut b: Vec<u8> = b"RIFF".to_vec();
        b.extend_from_slice(&400u32.to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        b.extend_from_slice(&[0u8; 16]);
        std::fs::write(&p, &b).unwrap();
        let e = read_info(&p).unwrap_err();
        assert!(e.to_string().contains("fmt"), "got: {e}");
    }

    #[test]
    fn roundtrip_8ch_s32_carries_the_7_1_mask() {
        let p = tmp("8ch.wav");
        let pcm = vec![0u8; 48000 * 8 * 4];
        write_pcm_s32(&p, 48000, 8, &pcm).unwrap();
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(info.channels, 8);
        assert_eq!(info.rate, 48000);
        assert_eq!(info.bits, 32);
        assert_eq!(info.mask, Some(0x63f), "7.1 mask");
        // 12 (RIFF) + 8 + 40 (fmt with 40-byte extensible body) + 8 (data header)
        assert_eq!(off as usize, 68, "extensible header is 68 bytes");
        assert_eq!(info.frames(), 48000);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn stereo_stays_classic_44_bytes() {
        let p = tmp("st.wav");
        write_pcm_s32(&p, 44100, 2, &vec![0u8; 44100 * 2 * 4]).unwrap();
        let (info, off) = read_info(&p).unwrap();
        assert_eq!(off as usize, 44);
        assert_eq!(info.mask, None);
        assert_eq!(info.frames(), 44100);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn patched_sizes_are_right() {
        let p = tmp("sizes.wav");
        let pcm = vec![7u8; 1000];
        let mut w = WavWriter::create(&p, 48000, 2, 16, false, None).unwrap();
        w.write(&pcm).unwrap();
        w.finish().unwrap();
        let len = std::fs::metadata(&p).unwrap().len();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(len, 44 + 1000);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as u64, len - 8);
        assert_eq!(
            u32::from_le_bytes([b[40], b[41], b[42], b[43]]) as u64,
            1000,
            "data chunk size"
        );
        assert_eq!(&b[44..], &pcm[..]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_non_wav() {
        let p = tmp("bad.wav");
        std::fs::write(&p, b"not a wav at all, really").unwrap();
        assert!(read_info(&p).is_err());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn f64_float_header_roundtrip() {
        let p = tmp("f64.wav");
        let h = header(48000, 8, 64, true, Some(0x63f));
        assert_eq!(h.len(), 68);
        assert_eq!(u16::from_le_bytes([h[20], h[21]]), WAVE_FORMAT_EXTENSIBLE);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        drop(f);
        let (info, _) = read_info(&p).unwrap();
        assert!(info.float);
        assert_eq!(info.bits, 64);
        assert_eq!(info.mask, Some(0x63f));
        std::fs::remove_file(&p).ok();
    }
}
