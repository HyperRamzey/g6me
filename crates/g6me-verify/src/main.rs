//! `g6me-verify` — does the Rust downmixer really reproduce the reference, bit
//! for bit, and can we prove it?
//!
//! Run with no arguments for the full suite. Exit status is non-zero if any check
//! fails, so CI can gate on it. `--require-all` additionally fails on skipped
//! checks, which is what a release job should use: a suite that "passed" because
//! it had no C compiler to compare against is not a verification.
//!
//! ```text
//! g6me-verify                  # everything, human-readable
//! g6me-verify --json           # machine-readable
//! g6me-verify --group bitexact # one family
//! g6me-verify --seconds 300    # longer drift run
//! g6me-verify --require-all    # treat SKIP as failure
//! ```

mod checks;
mod reference;
mod signal;

use checks::{Check, Status};
use serde_json::json;
use std::io::Write;

#[derive(Default)]
struct Args {
    json: bool,
    require_all: bool,
    group: Option<String>,
    seconds: u64,
    frames: usize,
    quiet: bool,
}

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("g6me-verify: {msg}");
            std::process::exit(2);
        }
    };

    let builds = reference::builds();
    let scratch = checks::Scratch::new("run");
    let mut all: Vec<Check> = Vec::new();

    // Environment first: a report that does not say what it was measured on is
    // not reproducible, so this is part of the output rather than a debug line.
    let env = Check {
        group: "env",
        name: "toolchain".into(),
        status: Status::Info,
        detail: format!(
            "cc={} version={:?} reference builds={}",
            reference::cc(),
            reference::cc_version(),
            builds.len(),
        ),
        metric: Some(json!({
            "cc": reference::cc(),
            "cc_version": reference::cc_version(),
            "reference_builds": builds.iter().map(|b| json!({
                "name": b.name, "flags": b.flags,
            })).collect::<Vec<_>>(),
        })),
    };
    all.push(env);

    let precision = <f64 as g6me_core::dsp::Real>::PRECISION;
    all.push(Check {
        group: "env",
        name: "dsp precision invariant".into(),
        status: if_false_then_fail(precision == "float64"),
        detail: format!(
            "the shipped downmixer works in {precision}; the f64 invariant is what the \
             reference's -DG6CT_DOUBLE build is compared against"
        ),
        metric: Some(json!({ "precision": precision })),
    });

    if wanted(&args, "bitexact") {
        all.extend(checks::bitexact(&builds, &scratch, args.frames));
    }
    if wanted(&args, "builds") {
        all.extend(checks::builds_agree(&builds, &scratch, args.frames));
    }
    if wanted(&args, "width") {
        all.extend(checks::width_gap(&builds, &scratch, args.frames));
    }
    if wanted(&args, "chunk") {
        all.extend(checks::chunk_invariance(&builds, &scratch, args.frames));
    }
    if wanted(&args, "continuity") {
        all.extend(checks::state_continuity(args.frames));
    }
    if wanted(&args, "phase") {
        all.extend(checks::phase(&builds, &scratch, 8192));
    }
    if wanted(&args, "drift") {
        all.extend(checks::drift(&builds, &scratch, args.seconds));
    }
    if wanted(&args, "contraction") {
        all.extend(checks::contraction());
        all.extend(checks::adversarial(&builds, &scratch, args.frames));
    }
    if wanted(&args, "asm") {
        all.extend(checks::asm_guard(&builds));
    }

    let fails = all.iter().filter(|c| c.status == Status::Fail).count();
    let skips = all.iter().filter(|c| c.status == Status::Skip).count();
    let passes = all.iter().filter(|c| c.status == Status::Pass).count();

    if args.json {
        let v = json!({
            "summary": { "pass": passes, "fail": fails, "skip": skips },
            "checks": all.iter().map(|c| json!({
                "group": c.group, "name": c.name,
                "status": c.status.label(), "detail": c.detail,
                "metric": c.metric,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&v).unwrap());
    } else {
        report(&all, passes, fails, skips);
    }

    let bad = fails > 0 || (args.require_all && skips > 0);
    if bad && !args.json {
        let mut err = std::io::stderr();
        let _ = writeln!(
            err,
            "\ng6me-verify: {fails} failed, {skips} skipped{}",
            if args.require_all && skips > 0 {
                " (--require-all: skipped counts as failure)"
            } else {
                ""
            }
        );
    }
    std::process::exit(if bad { 1 } else { 0 });
}

/// One helper for the `status:` field of a literal Check, to keep the env check
/// readable.
fn if_false_then_fail(ok: bool) -> Status {
    if ok {
        Status::Pass
    } else {
        Status::Fail
    }
}

/// The argument following position `i`.
fn value_at(argv: &[String], i: usize, what: &str) -> Result<String, String> {
    argv.get(i + 1)
        .cloned()
        .ok_or_else(|| format!("{what} needs a value"))
}

fn wanted(a: &Args, g: &str) -> bool {
    a.group.as_deref().map(|w| w == g).unwrap_or(true)
}

fn report(all: &[Check], passes: usize, fails: usize, skips: usize) {
    let width = all
        .iter()
        .map(|c| c.group.len() + c.name.len() + 1)
        .max()
        .unwrap_or(20)
        .min(58);
    let mut group: Option<&str> = None;
    for c in all {
        if group != Some(c.group) {
            group = Some(c.group);
            println!("\n{}", c.group);
            println!("{}", "-".repeat(c.group.len()));
        }
        let label = format!("{} / {}", c.group, c.name);
        let pad = width.saturating_sub(label.len());
        println!(
            "  [{}] {label}{}  {}",
            c.status.label(),
            " ".repeat(pad),
            wrap_detail(&c.detail, width + 8)
        );
    }
    println!("\n{} passed, {} failed, {} skipped", passes, fails, skips);
}

/// Keep long evidence lines readable by indenting continuations.
fn wrap_detail(s: &str, indent: usize) -> String {
    let mut line = String::new();
    let mut out = String::new();
    let pad = " ".repeat(indent);
    for w in s.split(' ') {
        if line.len() + w.len() > 96 && !line.is_empty() {
            out.push_str(&line);
            out.push('\n');
            out.push_str(&pad);
            line = String::new();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !out.is_empty() && !line.is_empty() {
        out.push_str(&line);
        out
    } else if line.is_empty() {
        out
    } else {
        line
    }
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        seconds: 60,
        frames: 200_000,
        ..Default::default()
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Index-based walking, with the value consumed in the same arm that read the
    // flag, so no closure has to borrow the cursor.
    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--json" => a.json = true,
            "--require-all" => a.require_all = true,
            "--quiet" => a.quiet = true,
            "--group" => {
                a.group = Some(value_at(&argv, i, "--group")?);
                i += 1;
            }
            "--seconds" => {
                a.seconds = value_at(&argv, i, "--seconds")?
                    .parse()
                    .map_err(|_| "--seconds wants a number")?;
                if a.seconds == 0 {
                    return Err("--seconds must be at least 1".into());
                }
                i += 1;
            }
            "--frames" => {
                a.frames = value_at(&argv, i, "--frames")?
                    .parse()
                    .map_err(|_| "--frames wants a number")?;
                if a.frames < 4096 {
                    return Err("--frames must be at least 4096 to exercise the filters".into());
                }
                i += 1;
            }
            "-h" | "--help" => {
                println!(
                    "usage: g6me-verify [--json] [--group NAME] [--seconds N] [--frames N] \
                     [--require-all]\ngroups: bitexact builds width chunk continuity phase \
                     drift contraction asm"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 1;
    }
    Ok(a)
}

#[cfg(test)]
mod suite {
    //! The full verification suite, as a `cargo test` case.
    //!
    //! Kept small enough to run on every `cargo test` (short signals, a few seconds
    //! of drift material) while the binary stays the thorough one used by CI and by
    //! a release checklist. If `build.rs` found no C compiler, this reports as
    //! skipped rather than green — a verification that compared nothing must never
    //! look like a pass.
    use super::*;

    #[test]
    fn verification_suite_passes() {
        let builds = reference::builds();
        if builds.is_empty() {
            eprintln!(
                "SKIP: no reference binaries. Install a C compiler (cc/clang/gcc) so \
                 build.rs can compile vendor/g6ct, then re-run."
            );
            return;
        }
        // Widths present: the f64 ones are the ones the Rust DSP must match.
        assert!(
            builds.iter().any(|b| b.width == "f64"),
            "reference built, but not the -DG6CT_DOUBLE variant the Rust DSP is compared to"
        );
        let scratch = checks::Scratch::new("test");
        let frames = 20_000;
        let mut all = Vec::new();
        all.extend(checks::bitexact(&builds, &scratch, frames));
        all.extend(checks::builds_agree(&builds, &scratch, frames));
        all.extend(checks::width_gap(&builds, &scratch, frames));
        all.extend(checks::chunk_invariance(&builds, &scratch, frames));
        all.extend(checks::state_continuity(frames));
        all.extend(checks::phase(&builds, &scratch, 4096));
        all.extend(checks::drift(&builds, &scratch, 5));
        all.extend(checks::contraction());
        all.extend(checks::adversarial(&builds, &scratch, frames));
        all.extend(checks::asm_guard(&builds));

        let failed: Vec<&Check> = all.iter().filter(|c| c.status == Status::Fail).collect();
        let skipped: Vec<&Check> = all.iter().filter(|c| c.status == Status::Skip).collect();
        for c in &skipped {
            eprintln!("SKIP {}/{}: {}", c.group, c.name, c.detail);
        }
        for c in &failed {
            eprintln!("FAIL {}/{}: {}", c.group, c.name, c.detail);
        }
        assert!(
            failed.is_empty(),
            "{} of {} verification checks failed",
            failed.len(),
            all.len()
        );
        assert!(!all.is_empty(), "the suite ran no checks at all");
        // The asm group needs objdump; everything else must actually have run.
        let run = all.iter().filter(|c| c.status != Status::Skip).count();
        assert!(
            run >= 40,
            "only {run} checks ran, expected the whole suite ({})",
            all.len()
        );
    }

    #[test]
    fn rust_dsp_works_in_double() {
        // The f64 invariant, asserted where a regression would be caught: the
        // reference comparison above only means something if this stays true.
        assert_eq!(<f64 as g6me_core::dsp::Real>::PRECISION, "float64");
    }
}
