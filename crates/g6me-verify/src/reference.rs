//! The compiled reference binaries, and the contract for driving them.
//!
//! `build.rs` compiles the vendored C port in several numerical configurations
//! and hands the paths over as `cargo::rustc-env` entries. The contract, taken
//! from `g6ct_cli.c` (which this crate does not modify):
//!
//! ```text
//! g6ct_cli <in.raw> <out2.raw> [mask] [channels]
//! ```
//!
//! * **in**: interleaved `float32`, `channels` per frame — always f32, even in the
//!   `-DG6CT_DOUBLE` build, which widens on read.
//! * **out**: interleaved `float32`, 2 channels — the f64 build narrows on write.
//! * **mask**: parsed with base 0, so `0x63f` works. Default `G6CT_MASK_7_1`.
//! * **channels**: 8 (slots `FL FR FC LFE BL BR SL SR` → `0 1 2 3 4 5 9 10`) or 12.
//! * the reference processes **1024 frames per call**, which is why chunk-size
//!   invariance is worth asserting rather than assuming.
use std::path::PathBuf;
use std::process::Command;

/// 8-channel input maps onto these WAVEFORMATEXTENSIBLE slot positions.
pub const SLOTS_8: [usize; 8] = [0, 1, 2, 3, 4, 5, 9, 10];
/// 12-channel Atmos bed maps 1:1.
pub const SLOTS_12: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
/// The mask the reference defaults to, and the app's stock 7.1 input.
pub const MASK_7_1: u32 = 0x063f;
/// Frames per call inside `g6ct_cli.c`.
pub const REF_BLOCK: usize = 1024;

/// One built reference configuration.
#[derive(Clone, Debug)]
pub struct RefBuild {
    /// Short name, e.g. `contract_off/f64`.
    pub name: String,
    /// `f32` (float state) or `f64` (`-DG6CT_DOUBLE`).
    pub width: &'static str,
    /// Which numerical contract this build represents.
    pub variant: String,
    pub path: PathBuf,
    /// The exact flags it was built with, so a report can be reproduced.
    pub flags: String,
}

/// Every reference build that `build.rs` managed to produce.
pub fn builds() -> Vec<RefBuild> {
    // (env key, variant label, width) as emitted by build.rs.
    const SPECS: &[(&str, &str, &str)] = &[
        ("CONTRACT_OFF", "f32", "f32"),
        ("CONTRACT_OFF", "f64", "f64"),
        ("NATIVE", "f32", "f32"),
        ("NATIVE", "f64", "f64"),
        ("O3_NATIVE", "f32", "f32"),
        ("O3_NATIVE", "f64", "f64"),
        ("CONTRACT_FAST", "f32", "f32"),
        ("CONTRACT_FAST", "f64", "f64"),
    ];
    let mut out = Vec::new();
    for (variant, width, _) in SPECS {
        let key = format!("G6ME_REF_{variant}_{width}");
        let Some(p) = option_env_str(&key) else {
            continue;
        };
        let flags = option_env_str(&format!("{key}_FLAGS")).unwrap_or_default();
        out.push(RefBuild {
            name: format!("{}/{width}", variant.to_lowercase()),
            width,
            variant: (*variant).to_lowercase(),
            path: PathBuf::from(p),
            flags,
        });
    }
    out
}

/// `env!()` at a computed name is not possible, and `std::env::var` would read the
/// *runtime* environment rather than the build-time one, so go through `option_env!`
/// for each key we might have emitted.
fn option_env_str(key: &str) -> Option<String> {
    // The set is small and closed, so a match is honest and avoids unsafe tricks.
    macro_rules! probe {
        ($($k:literal),* $(,)?) => {
            match key {
                $($k => option_env!($k).map(|s| s.to_string()),)*
                _ => None,
            }
        };
    }
    probe!(
        "G6ME_REF_CONTRACT_OFF_f32",
        "G6ME_REF_CONTRACT_OFF_f64",
        "G6ME_REF_NATIVE_f32",
        "G6ME_REF_NATIVE_f64",
        "G6ME_REF_O3_NATIVE_f32",
        "G6ME_REF_O3_NATIVE_f64",
        "G6ME_REF_CONTRACT_FAST_f32",
        "G6ME_REF_CONTRACT_FAST_f64",
    )
}

pub fn cc() -> String {
    option_env!("G6ME_VERIFY_CC").unwrap_or("none").to_string()
}

pub fn cc_version() -> String {
    option_env!("G6ME_VERIFY_CC_VERSION")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Run one reference build over a raw interleaved-f32 file.
pub fn run(
    b: &RefBuild,
    input: &PathBuf,
    output: &PathBuf,
    mask: u32,
    channels: u16,
) -> Result<Vec<u8>, String> {
    let o = Command::new(&b.path)
        .arg(input)
        .arg(output)
        .arg(format!("{mask:#x}"))
        .arg(channels.to_string())
        .output()
        .map_err(|e| format!("could not run {}: {e}", b.path.display()))?;
    if !o.status.success() {
        return Err(format!(
            "reference exited {}: {}",
            o.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    std::fs::read(output).map_err(|e| format!("reading {}: {e}", output.display()))
}
