//! Deliberately FMA-contracted Rust binary used as the `asm` group's sabotage
//! control. It is never shipped: `g6me-verify` only disassembles it and requires
//! the scanner to find at least one fused multiply-add in it. Without that
//! positive control, a green "0 FMA" result could mean the mnemonic matcher never
//! matches anything (the C counter-example only proves the C compiler side).
//!
//! The kernel mirrors the shape of the `stn` downmix arithmetic (a scaled
//! multiply-add) but is compiled with the `fma` target feature so `mul_add`
//! lowers to a hardware FMA instruction instead of a libm call.

use std::hint::black_box;

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "fma")]
unsafe fn stn_kernel_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(never)]
fn stn_kernel_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(target_arch = "x86_64")]
fn run(a: f64, b: f64, c: f64) -> f64 {
    // Only executed by hand; do not SIGILL on a pre-FMA CPU. The instruction is
    // still emitted in the binary, which is what the disassembly audit checks.
    if std::arch::is_x86_feature_detected!("fma") {
        unsafe { stn_kernel_fma(a, b, c) }
    } else {
        a * b + c
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn run(a: f64, b: f64, c: f64) -> f64 {
    stn_kernel_fma(a, b, c)
}

fn main() {
    let a = black_box(1.0 + f64::EPSILON);
    let b = black_box(1.0 + f64::EPSILON);
    let c = black_box(-(1.0 + 2.0 * f64::EPSILON));
    println!("{:.17e}", run(a, b, c));
}
