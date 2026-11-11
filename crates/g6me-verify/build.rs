//! Build the reference C port into several numerical configurations so the
//! verifier can compare the Rust DSP against real, independent ground truth.
//!
//! The sources are the vendored copies of `g6ct.c` / `g6ct_cli.c`, byte-identical
//! to the port they came from (`vendor/g6ct/UPSTREAM.md`). Nothing here is
//! compiled into the shipping app — this is a test harness, and the app never
//! links the C code.
//!
//! Why several builds rather than one:
//!
//! * the **contract** build (`-ffp-contract=off -fno-fast-math`) is the one the
//!   upstream port declares as its numerical contract, and the one the f32 build
//!   was proven bit-identical to the vendor DLL with;
//! * the **native** builds prove the result does not depend on `-march` or `-O3`;
//! * the **contract=fast** build is adversarial: it is what a careless build of
//!   the same C looks like. If the compiler contracts mul+add into FMA there, the
//!   diff against the contract build is the direct evidence that
//!   `-ffp-contract=off` is load-bearing and not decoration.
//!
//! Every f64 variant is the `-DG6CT_DOUBLE` build, which is what the Rust DSP
//! (`f64` internal state, per the project's invariant) is expected to reproduce
//! bit for bit.
//!
//! No C compiler, or a compiler that rejects the flags, degrades to "no reference
//! built" and the verifier reports those checks as skipped instead of failing the
//! build.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One reference configuration to build.
struct Variant {
    /// Short name, uppercased into the emitted env var.
    name: &'static str,
    /// Extra compiler flags. Flags the compiler does not accept are skipped
    /// (see `accepts`), so an MSVC or clang build still gets the canonical one.
    flags: &'static [&'static str],
    /// Optional flags that may be dropped if unsupported.
    optional: &'static [&'static str],
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "contract_off",
        flags: &["-O2", "-std=c11"],
        optional: &["-ffp-contract=off", "-fno-fast-math"],
    },
    Variant {
        name: "native",
        flags: &["-O2", "-std=c11", "-march=native"],
        optional: &["-ffp-contract=off", "-fno-fast-math"],
    },
    Variant {
        name: "o3_native",
        flags: &["-O3", "-std=c11", "-march=native"],
        optional: &["-ffp-contract=off", "-fno-fast-math"],
    },
    Variant {
        name: "contract_fast",
        flags: &["-O3", "-std=c11", "-march=native"],
        optional: &["-ffp-contract=fast"],
    },
];

fn main() {
    println!("cargo::rerun-if-changed=../../vendor/g6ct/g6ct.c");
    println!("cargo::rerun-if-changed=../../vendor/g6ct/g6ct_cli.c");
    println!("cargo::rerun-if-changed=../../vendor/g6ct/g6ct.h");
    println!("cargo::rerun-if-changed=build.rs");

    let vendor = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/g6ct")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../../vendor/g6ct"));
    let cli_c = vendor.join("g6ct_cli.c");
    let dsp_c = vendor.join("g6ct.c");
    if !cli_c.is_file() || !dsp_c.is_file() {
        note(&format!(
            "no vendored C sources under {} — verification will be skipped",
            vendor.display()
        ));
        return;
    }

    let Some(cc) = find_compiler() else {
        note("no C compiler found (set $CC) — reference comparisons will be skipped");
        return;
    };
    println!("cargo::rustc-env=G6ME_VERIFY_CC={}", cc.display());
    if let Some(v) = compiler_version(&cc) {
        println!("cargo::rustc-env=G6ME_VERIFY_CC_VERSION={v}");
    }

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let mut built = 0;
    for v in VARIANTS {
        for wide in [false, true] {
            let tag = if wide { "f64" } else { "f32" };
            let bin = out.join(format!("g6ct_ref_{}_{}{}", v.name, tag, exe()));
            let mut flags: Vec<String> = v.flags.iter().map(|s| (*s).to_string()).collect();
            // An unsupported optional flag (MSVC has no -ffp-contract) must not
            // lose the whole variant: keep the build, drop the flag.
            for f in v.optional {
                if accepts(&cc, &vendor, f) {
                    flags.push((*f).to_string());
                }
            }
            if wide {
                flags.push("-DG6CT_DOUBLE".to_string());
            }
            match compile(&cc, &vendor, &flags, &bin) {
                Ok(()) => {
                    built += 1;
                    let key = v.name.to_uppercase();
                    println!(
                        "cargo::rustc-env=G6ME_REF_{}_{}={}",
                        key,
                        tag,
                        bin.display()
                    );
                    println!(
                        "cargo::rustc-env=G6ME_REF_{}_{}_FLAGS={}",
                        key,
                        tag,
                        flags.join(" ")
                    );
                }
                Err(e) => note(&format!("reference build {} {tag} skipped: {e}", v.name)),
            }
        }
    }
    println!("cargo::rustc-env=G6ME_VERIFY_REF_BUILT={built}");
    if built == 0 {
        note("the C compiler produced no reference binaries — see the errors above");
    }
}

fn note(msg: &str) {
    println!("cargo::warning={msg}");
}

fn exe() -> &'static str {
    if cfg!(windows) {
        ".exe"
    } else {
        ""
    }
}

/// `$CC`, then the usual suspects. `cc`/`clang`/`gcc` cover Arch and CI; MSVC's
/// `cl` is only reachable from a developer prompt, and that is fine.
fn find_compiler() -> Option<PathBuf> {
    let cands: Vec<PathBuf> = std::env::var_os("CC")
        .map(|c| c.into())
        .into_iter()
        .chain(
            ["cc", "clang", "gcc", "cl"]
                .iter()
                .map(|n| PathBuf::from(*n)),
        )
        .collect();
    cands.into_iter().find(|c| compiles_ok(c))
}

/// A compiler counts only if it can actually build these two files: a stray
/// `cc` wrapper (a distcc shim, a stale alternatives entry) would otherwise be
/// picked and then fail every variant.
fn compiles_ok(cc: &Path) -> bool {
    let dir = match std::env::temp_dir().canonicalize() {
        Ok(d) => d,
        Err(_) => return false,
    };
    let dir = dir.join(format!("g6me-cc-probe-{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    let src = dir.join("probe.c");
    let ok = std::fs::write(&src, "int main(void){return 0;}\n")
        .and_then(|()| {
            Command::new(cc)
                .arg("-std=c11")
                .arg(&src)
                .arg("-o")
                .arg(dir.join(if cfg!(windows) { "probe.exe" } else { "probe" }))
                .output()
        })
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

fn compiler_version(cc: &Path) -> Option<String> {
    let out = Command::new(cc).arg("-v").arg("--version").output().ok()?;
    let txt = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    txt.lines().next().map(|l| l.trim().to_string())
}

/// Does this compiler accept `flag` for these sources? Probed, never assumed:
/// `-ffp-contract=` and `-march=native` are GNU/clang-only spellings, and clang
/// warns (but accepts) some combinations.
fn accepts(cc: &Path, vendor: &Path, flag: &str) -> bool {
    let dir = match std::env::temp_dir().canonicalize() {
        Ok(d) => d,
        Err(_) => return false,
    };
    let dir = dir.join(format!(
        "g6me-flag-probe-{}-{}",
        std::process::id(),
        flag.replace(['-', '=', '.'], "_")
    ));
    let _ = std::fs::create_dir_all(&dir);
    let ok = Command::new(cc)
        .arg("-std=c11")
        .arg("-c")
        .arg(flag)
        .arg(vendor.join("g6ct.c"))
        .arg("-o")
        .arg(dir.join("probe.o"))
        .output()
        .map(|o| {
            // A warning is fine; an error (or "unknown argument") is not.
            let err = String::from_utf8_lossy(&o.stderr).to_lowercase();
            o.status.success()
                && !(err.contains("unknown argument")
                    || err.contains("unsupported")
                    || err.contains("unrecognized"))
        })
        .unwrap_or(false);
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

fn compile(cc: &Path, vendor: &Path, flags: &[String], out_bin: &Path) -> Result<(), String> {
    let mut cmd = Command::new(cc);
    cmd.args(flags)
        .arg("-Wall")
        .arg("-Wextra")
        .arg(vendor.join("g6ct.c"))
        .arg(vendor.join("g6ct_cli.c"))
        .arg("-o")
        .arg(out_bin);
    if !cfg!(windows) {
        cmd.arg("-lm");
    }
    // Deterministic output path; the object files land next to it and are
    // discarded, since the verifier only ever runs the binaries.
    let o = cmd.output().map_err(|e| format!("{}: {e}", cc.display()))?;
    if !o.status.success() {
        let err = String::from_utf8_lossy(&o.stderr);
        return Err(err.lines().take(6).collect::<Vec<_>>().join(" / "));
    }
    Ok(())
}
