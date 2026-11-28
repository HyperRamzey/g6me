//! Process-wide CUDA stage accounting for one render.
//!
//! Every offload point records whether its stage ran on the GPU or on the CPU
//! fallback, and why. The CLI prints the summary once at the end of a render, so a
//! CUDA banner can never stand in front of a run that silently did all its work on
//! the CPU. Recording is always compiled; without the `cuda` feature nothing is
//! ever recorded and the summary reports zero stages.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Default, Clone)]
struct StageCount {
    gpu_calls: usize,
    cpu_calls: usize,
    /// First non-empty detail: a GPU precision/geometry tag, or the first fallback
    /// reason (kept so a repeated failure does not produce a wall of identical text).
    detail: String,
}

static STAGES: OnceLock<Mutex<BTreeMap<&'static str, StageCount>>> = OnceLock::new();
static INIT_ERROR: OnceLock<String> = OnceLock::new();

fn stages() -> &'static Mutex<BTreeMap<&'static str, StageCount>> {
    STAGES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn lock() -> std::sync::MutexGuard<'static, BTreeMap<&'static str, StageCount>> {
    stages()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Record one successful GPU dispatch of `stage`.
pub fn record_gpu(stage: &'static str, detail: impl Into<String>) {
    let mut map = lock();
    let entry = map.entry(stage).or_default();
    entry.gpu_calls += 1;
    let detail = detail.into();
    if entry.detail.is_empty() && !detail.is_empty() {
        entry.detail = detail;
    }
}

/// Record one CPU fallback of `stage`, with the reason it could not run on the GPU.
pub fn record_cpu_fallback(stage: &'static str, reason: impl Into<String>) {
    let mut map = lock();
    let entry = map.entry(stage).or_default();
    entry.cpu_calls += 1;
    let reason = reason.into();
    if entry.detail.is_empty() && !reason.is_empty() {
        entry.detail = reason;
    }
}

/// Record the first runtime-initialization failure. The cached `None` is returned to
/// every later stage, so the error is reported once instead of being swallowed by
/// each of them.
pub fn record_init_failure(reason: &str) {
    let _ = INIT_ERROR.set(reason.to_string());
}

/// Snapshot of the process-wide report.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Summary {
    pub gpu_stages: usize,
    pub cpu_fallback_stages: usize,
    pub lines: Vec<String>,
    pub init_error: Option<String>,
}

fn line(stage: &str, count: &StageCount) -> String {
    let calls = if count.detail.is_empty() {
        String::new()
    } else {
        format!("; {}", count.detail)
    };
    match (count.gpu_calls, count.cpu_calls) {
        (gpu, 0) => format!("stage {stage}: GPU ({gpu} call(s){calls})"),
        (0, cpu) => format!("stage {stage}: CPU fallback ({cpu} call(s){calls})"),
        (gpu, cpu) => {
            format!("stage {stage}: GPU ({gpu} call(s)) + CPU fallback ({cpu} call(s){calls})")
        }
    }
}

/// Read the accumulated report without printing it.
pub fn summary() -> Summary {
    let map = lock();
    let mut out = Summary {
        init_error: INIT_ERROR.get().cloned(),
        ..Summary::default()
    };
    for (stage, count) in map.iter() {
        if count.gpu_calls > 0 {
            out.gpu_stages += 1;
        }
        if count.cpu_calls > 0 {
            out.cpu_fallback_stages += 1;
        }
        out.lines.push(line(stage, count));
    }
    out
}

/// Print the report once, at the end of a render. Silent unless `--cuda` was asked
/// for. Lines are prefixed with `[cuda]` so a caller (and the verifier's end-to-end
/// check) can separate them from ordinary progress output.
pub fn print_summary(requested: bool) {
    if !requested {
        return;
    }
    let report = summary();
    if let Some(err) = &report.init_error {
        eprintln!("[cuda] runtime init failed: {err}");
    }
    for line in &report.lines {
        eprintln!("[cuda] {line}");
    }
    if report.gpu_stages == 0 && report.cpu_fallback_stages == 0 {
        eprintln!("[cuda] GPU stages run: 0 (this configuration has no GPU work)");
    } else {
        eprintln!(
            "[cuda] GPU stages: {}; CPU-fallback stages: {}",
            report.gpu_stages, report.cpu_fallback_stages
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_gpu_and_fallback_with_reasons() {
        record_gpu("unit-test-stage", "precision int8");
        record_cpu_fallback("unit-test-fallback", "runtime unavailable");
        let report = summary();
        assert!(report
            .lines
            .iter()
            .any(|l| l == "stage unit-test-stage: GPU (1 call(s); precision int8)"));
        assert!(report.lines.iter().any(
            |l| l == "stage unit-test-fallback: CPU fallback (1 call(s); runtime unavailable)"
        ));
        assert!(report.gpu_stages >= 1 && report.cpu_fallback_stages >= 1);
    }
}
