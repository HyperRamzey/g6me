//! Minimal read-only NumPy `.npy` decoder, sufficient for the MeshRIR layout
//! (`ir_<mic>.npy`, `pos_mic.npy`, `pos_src.npy`).
//!
//! Supports NPY format versions 1.0–3.0, C-order arrays of `float64`/`float32`
//! in either byte order. Fortran order and every other dtype are rejected with
//! an explicit error; guessing at them would silently transpose a spatial field
//! or reinterpret integers as pressures.

use std::path::Path;

type Result<T> = std::result::Result<T, String>;

/// A decoded array flattened in C order.
#[derive(Clone, Debug)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    F64,
    F32,
}

/// Read and decode an `.npy` file.
pub fn read_npy(path: &Path) -> Result<NpyArray> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_npy(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Read only the header and report the logical shape.
pub fn npy_shape(path: &Path) -> Result<Vec<usize>> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut prefix = [0u8; 12];
    use std::io::{Read, Seek, SeekFrom};
    file.read_exact(&mut prefix)
        .map_err(|e| format!("{}: short npy header: {e}", path.display()))?;
    let header = header_len(&prefix).map_err(|e| format!("{}: {e}", path.display()))?;
    // The 12-byte peek covers two bytes of the header itself for version 1
    // (which starts at offset 10); rewind so the header slice is aligned.
    let start: u64 = if prefix[6] == 1 { 10 } else { 12 };
    file.seek(SeekFrom::Start(start))
        .map_err(|e| format!("{}: seek failed: {e}", path.display()))?;
    let mut rest = vec![0u8; header];
    file.read_exact(&mut rest)
        .map_err(|e| format!("{}: short npy header: {e}", path.display()))?;
    let text = std::str::from_utf8(&rest)
        .map_err(|_| format!("{}: npy header is not UTF-8", path.display()))?;
    parse_shape(text).map_err(|e| format!("{}: {e}", path.display()))
}

fn header_len(prefix: &[u8]) -> Result<usize> {
    if prefix.len() < 10 || &prefix[0..6] != b"\x93NUMPY" {
        return Err("not a NumPy .npy file (bad magic)".into());
    }
    let major = prefix[6];
    match major {
        1 => Ok(u16::from_le_bytes([prefix[8], prefix[9]]) as usize),
        2 | 3 => {
            if prefix.len() < 12 {
                return Err("short version-2/3 npy prefix".into());
            }
            Ok(u32::from_le_bytes([prefix[8], prefix[9], prefix[10], prefix[11]]) as usize)
        }
        v => Err(format!("unsupported npy format version {v}.x")),
    }
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    let header_len = header_len(bytes)?;
    let start: usize = if bytes[6] == 1 { 10 } else { 12 };
    let end = start
        .checked_add(header_len)
        .ok_or_else(|| "npy header length overflow".to_string())?;
    if end > bytes.len() {
        return Err("npy header extends past end of file".into());
    }
    let text = std::str::from_utf8(&bytes[start..end])
        .map_err(|_| "npy header is not UTF-8".to_string())?;

    let (dtype, fortran) = parse_descr(text)?;
    if fortran {
        return Err("Fortran-order npy arrays are not supported".into());
    }
    let shape = parse_shape(text)?;
    let count = shape
        .iter()
        .try_fold(1usize, |acc, d| acc.checked_mul(*d))
        .ok_or_else(|| "npy shape overflows usize".to_string())?;
    let itemsize = match dtype {
        Dtype::F64 => 8,
        Dtype::F32 => 4,
    };
    let want = count
        .checked_mul(itemsize)
        .ok_or_else(|| "npy data size overflows usize".to_string())?;
    let payload = &bytes[end..];
    if payload.len() != want {
        return Err(format!(
            "npy payload is {} bytes, shape {:?} with {itemsize}-byte items needs {want}",
            payload.len(),
            shape
        ));
    }
    let big = parse_endianness(text)?;
    let data = match itemsize {
        8 => payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| {
                if big {
                    f64::from_be_bytes(*c)
                } else {
                    f64::from_le_bytes(*c)
                }
            })
            .collect(),
        4 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                if big {
                    f32::from_be_bytes(*c) as f64
                } else {
                    f32::from_le_bytes(*c) as f64
                }
            })
            .collect(),
        _ => unreachable!(),
    };
    Ok(NpyArray { shape, data })
}

fn field_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(key)?;
    let after = &text[at + key.len()..];
    let colon = after.find(':')?;
    Some(after[colon + 1..].trim_start())
}

fn parse_descr(text: &str) -> Result<(Dtype, bool)> {
    let raw = field_value(text, "'descr'")
        .or_else(|| field_value(text, "\"descr\""))
        .ok_or_else(|| "npy header has no 'descr'".to_string())?;
    let quote = raw
        .chars()
        .next()
        .filter(|c| *c == '\'' || *c == '"')
        .ok_or_else(|| "malformed npy 'descr'".to_string())?;
    let rest = &raw[1..];
    let close = rest
        .find(quote)
        .ok_or_else(|| "unterminated npy 'descr'".to_string())?;
    let descr = &rest[..close];

    let fortran_raw = field_value(text, "'fortran_order'")
        .or_else(|| field_value(text, "\"fortran_order\""))
        .ok_or_else(|| "npy header has no 'fortran_order'".to_string())?;
    let fortran = fortran_raw.starts_with("True");

    let (order, code) = descr
        .split_at_checked(1)
        .ok_or_else(|| format!("malformed npy dtype '{descr}'"))?;
    let dtype = match code {
        "f8" => Dtype::F64,
        "f4" => Dtype::F32,
        other => {
            return Err(format!(
                "unsupported npy dtype '{other}'; only float64 ('f8') and float32 ('f4') are decoded"
            ))
        }
    };
    if !matches!(order, "<" | ">" | "=" | "|") {
        return Err(format!("unsupported npy byte-order marker '{order}'"));
    }
    Ok((dtype, fortran))
}

fn parse_endianness(text: &str) -> Result<bool> {
    let raw = field_value(text, "'descr'")
        .or_else(|| field_value(text, "\"descr\""))
        .ok_or_else(|| "npy header has no 'descr'".to_string())?;
    // `raw` starts at the opening quote; the order marker is the next char.
    let order = raw.chars().nth(1).unwrap_or('|');
    match order {
        '>' => Ok(true),
        '<' | '=' | '|' => Ok(false),
        other => Err(format!("unsupported npy byte-order marker '{other}'")),
    }
}

fn parse_shape(text: &str) -> Result<Vec<usize>> {
    let raw = field_value(text, "'shape'")
        .or_else(|| field_value(text, "\"shape\""))
        .ok_or_else(|| "npy header has no 'shape'".to_string())?;
    let open = raw
        .find('(')
        .ok_or_else(|| "malformed npy 'shape' (no opening parenthesis)".to_string())?;
    let close = raw
        .find(')')
        .ok_or_else(|| "malformed npy 'shape' (no closing parenthesis)".to_string())?;
    if close < open {
        return Err("malformed npy 'shape'".into());
    }
    let mut shape = Vec::new();
    for token in raw[open + 1..close].split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let dim: usize = token
            .parse()
            .map_err(|_| format!("malformed npy shape dimension '{token}'"))?;
        shape.push(dim);
    }
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{write_npy_f32, write_npy_f64, write_raw, TempDir};

    fn sample_data() -> Vec<f64> {
        vec![0.0, 0.5, -0.5, 1.25, -2.5, 0.125, 7.0, -8.0]
    }

    #[test]
    fn float64_round_trips_in_c_order() {
        let dir = TempDir::new("npy-f64");
        let path = dir.path().join("ir_0.npy");
        write_npy_f64(&path, &[2, 4], &sample_data(), false);
        let a = read_npy(&path).expect("read npy");
        assert_eq!(a.shape, vec![2, 4]);
        assert_eq!(a.data, sample_data());
        assert_eq!(npy_shape(&path).unwrap(), vec![2, 4]);
    }

    #[test]
    fn float32_and_big_endian_are_decoded() {
        let dir = TempDir::new("npy-f32");
        let path = dir.path().join("big.npy");
        let values = vec![0.5f64, -0.25, 1.0];
        write_npy_f32(&path, &[3], &values, true);
        let a = read_npy(&path).expect("read big-endian f32 npy");
        assert_eq!(a.shape, vec![3]);
        for (got, want) in a.data.iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-7);
        }
    }

    #[test]
    fn fortran_order_is_rejected() {
        let dir = TempDir::new("npy-fortran");
        let path = dir.path().join("f.npy");
        write_npy_f64(&path, &[2, 2], &[1.0, 2.0, 3.0, 4.0], true);
        let err = read_npy(&path).unwrap_err();
        assert!(err.contains("Fortran-order"), "{err}");
    }

    #[test]
    fn truncated_and_mislabelled_arrays_are_rejected() {
        let dir = TempDir::new("npy-bad");
        let path = dir.path().join("short.npy");
        write_npy_f64(&path, &[2, 4], &[1.0, 2.0], false);
        let err = read_npy(&path).unwrap_err();
        assert!(err.contains("payload is"), "{err}");

        let path = dir.path().join("magic.npy");
        write_raw(&path, b"not a numpy file");
        let err = read_npy(&path).unwrap_err();
        assert!(err.contains("magic"), "{err}");
    }

    #[test]
    fn shape_only_read_matches_full_read() {
        let dir = TempDir::new("npy-shape");
        let path = dir.path().join("shape.npy");
        // Non-zero payload: a header slice that overruns into the data by even
        // two bytes is then invalid UTF-8, which zero payload would hide.
        write_npy_f64(&path, &[3, 2, 2], &[0.1; 12], false);
        assert_eq!(npy_shape(&path).unwrap(), vec![3, 2, 2]);
    }
}
