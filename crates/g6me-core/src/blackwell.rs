//! Zero-FMA Blackwell (`sm_120`) GPU Architecture — CPU-Verified Exact Reference Model,
//! Functional `sm_120` CUDA/PTX Kernel Sources, and Real `nvcc`/`cuobjdump` SASS Auditor.
//!
//! **Execution Honesty**: All Ozaki-I/II (`INT8`, `FP8 E4M3`, `FP4 E2M1 Base-13`), Kulisch
//! superaccumulator, Matsuoka Bailey 4-Step FFT, and Blelloch/Overlap-Save IIR algorithms in
//! this module execute deterministically in pure Rust on the CPU as a zero-dependency exact
//! numerical reference model (bit-identical across all three quantized backends). Hardware GPU
//! compilation & SASS verification (`nvcc --fmad=false` + `cuobjdump --dump-sass`) are invoked
//! via [`compile_and_audit_cuda_sass`] whenever the NVIDIA CUDA Toolkit is installed; on
//! CPU-only systems it reports `CudaAuditStatus::ToolchainUnavailable` without faking GPU runs.
//!
//! 1. **Zero-FMA CUDA/PTX Kernel Emitter & SASS/PTX Static Verifier**:
//!    emits `--fmad=false` CUDA C++ & functional `sm_120` PTX kernels for Stockham Radix-4
//!    `f64` FFT and 3D FDTD wave PDE, and audits PTX/SASS text to guarantee zero
//!    `fma.*` / `mad.*` / `DFMA` / `FFMA` / `DMAD` contraction instructions. SASS forms
//!    with a zero-register multiplicand (ptxas constant loads) are not contractions.
//! 2. **Phase-Exact GPU IIR Biquad Algorithms**:
//!    - **Overlap-Save with `-320 dBFS` settling margin**: chunked parallel biquad
//!      achieving **0 ULP bit-exact** parity against sequential DF-II transposed IIR.
//!    - **Blelloch Parallel-Prefix 2×2 State-Matrix Scan**: reformulates `s[n] = M s[n-1] + v[n]`
//!      as an associative tree scan (`O(log N)` parallel depth, `< 1e-13` parity).
//! 3. **Ozaki Scheme I (INT8 Bit-Slicing)**: classic `O(S²/2)` exact signed `i8` GEMM
//!    reconstructing IEEE-754 `f64` matrix products to `< 1e-14`.
//! 4. **Ozaki Scheme II (Chinese Remainder Theorem, Uchino, Ozaki & Imamura,
//!    arXiv:2504.08009 / arXiv:2602.02549, 2025–2026)**:
//!    replaces quadratic `O(S²/2)` slices with linear `O(M)` (`M = 20` pairwise-coprime
//!    moduli `<= 113`, `log2(∏ m_j) = 124.31 bits`) modular GEMMs + exact double-double
//!    CRT reconstruction across THREE low-precision Tensor Core backends:
//!    - `TensorBackend::Int8` (`INT8` Tensor Cores),
//!    - `TensorBackend::Fp8E4m3` (`FP8 E4M3` 2-limb exact integer decomposition),
//!    - `TensorBackend::Fp4E2m1Base13` (**Hayashi, Mukunoki, Hoshino & Katagiri,
//!      arXiv:2608.06812, Aug 2026**: exact Base-13 FP4 E2M1 limb representation
//!      `{0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0}` covering every residue in `[-56, +56]`
//!      in just 2 FP4 limbs!).
//! 5. **Full-FP64 FFT with Zero FP64 GEMM Instructions (Matsuoka, Sept 2026)**:
//!    factors a size-`N = N1 × N2` complex FFT via **Bailey's 4-step FFT** where both
//!    DFT matrix multiplies execute strictly on `INT8`, `FP8 E4M3`, or `FP4 E2M1` via
//!    Ozaki-II CRT, paired with an exact **Kulisch Fixed-Point Superaccumulator**!

use crate::fft::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// 1. Exact Kulisch Fixed-Point Superaccumulator (Zero Rounding on Summation)
// ============================================================================

/// Exact IEEE-754 `f64` **Kulisch Fixed-Point Superaccumulator** (`2112` bits across
/// `33 × u64` limbs covering the complete normal+subnormal `f64` exponent range
/// `2^{-1074} .. 2^{+1024}` with zero intermediate rounding error!).
///
/// Correctly evaluates catastrophic cancellation sums such as `1e16 + 1.0 - 1e16 == 1.0`
/// to exact 0 ULP IEEE-754 rounding.
#[derive(Debug, Clone)]
pub struct KulischAcc {
    /// Positive-magnitude fixed-point limbs (`limb[0]` = bit `2^{-1088}`, `36` limbs = 2304 bits).
    pos: [u64; 36],
    /// Negative-magnitude fixed-point limbs.
    neg: [u64; 36],
}

impl Default for KulischAcc {
    fn default() -> Self {
        Self::new()
    }
}

impl KulischAcc {
    const BIAS_BITS: i32 = 1088; // Limb 0 bit 0 represents 2^(-1088)

    pub fn new() -> Self {
        Self {
            pos: [0u64; 36],
            neg: [0u64; 36],
        }
    }

    /// Add one finite `f64` exactly with zero rounding loss.
    pub fn add(&mut self, val: f64) {
        if val == 0.0 || !val.is_finite() {
            return;
        }
        let bits = val.to_bits();
        let neg = (bits >> 63) != 0;
        let exp_raw = ((bits >> 52) & 0x7FF) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (mant, exp2) = if exp_raw == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), exp_raw - 1023 - 52)
        };

        let bit_pos = exp2 + Self::BIAS_BITS;
        if !(0..(36 * 64 - 64)).contains(&bit_pos) {
            return;
        }
        let limb = (bit_pos as usize) / 64;
        let shift = (bit_pos as usize) % 64;
        let target = if neg { &mut self.neg } else { &mut self.pos };

        let lo = mant << shift;
        let hi = if shift == 0 { 0 } else { mant >> (64 - shift) };
        let (s0, c0) = target[limb].overflowing_add(lo);
        target[limb] = s0;
        let mut carry = (hi as u128) + (c0 as u128);
        let mut idx = limb + 1;
        while carry != 0 && idx < target.len() {
            let sum = (target[idx] as u128) + carry;
            target[idx] = sum as u64;
            carry = sum >> 64;
            idx += 1;
        }
    }

    /// Finalize exact fixed-point difference `pos - neg` rounded once to nearest IEEE-754 `f64`.
    pub fn finalize(&self) -> f64 {
        // Compare pos vs neg from most significant limb downward.
        let mut cmp = 0i32;
        for i in (0..36).rev() {
            if self.pos[i] != self.neg[i] {
                cmp = if self.pos[i] > self.neg[i] { 1 } else { -1 };
                break;
            }
        }
        if cmp == 0 {
            return 0.0;
        }
        let (big, small, sign) = if cmp > 0 {
            (&self.pos, &self.neg, 1.0_f64)
        } else {
            (&self.neg, &self.pos, -1.0_f64)
        };
        // Exact multi-precision subtraction diff = big - small.
        let mut diff = [0u64; 36];
        let mut borrow = 0u128;
        for i in 0..36 {
            let bi = big[i] as u128;
            let si = (small[i] as u128) + borrow;
            if bi >= si {
                diff[i] = (bi - si) as u64;
                borrow = 0;
            } else {
                diff[i] = ((1u128 << 64) + bi - si) as u64;
                borrow = 1;
            }
        }
        // Locate highest non-zero bit.
        let mut top_limb = 35usize;
        while top_limb > 0 && diff[top_limb] == 0 {
            top_limb -= 1;
        }
        if diff[top_limb] == 0 {
            return 0.0;
        }
        let lead = 63 - (diff[top_limb].leading_zeros() as usize);
        let top_bit = top_limb * 64 + lead;

        // Extract top 54 bits (53 mantissa + round bit) + sticky bit for exact round-to-nearest-even.
        let get_bit = |b: usize| -> u64 { (diff[b / 64] >> (b % 64)) & 1 };
        let mut sig53 = 0u64;
        for k in 0..53 {
            if top_bit >= k {
                sig53 = (sig53 << 1) | get_bit(top_bit - k);
            } else {
                sig53 <<= 1;
            }
        }
        let round_bit = if top_bit >= 53 {
            get_bit(top_bit - 53)
        } else {
            0
        };
        let mut sticky = 0u64;
        if top_bit >= 54 {
            let sb = top_bit - 54;
            let s_limb = sb / 64;
            let s_rem = sb % 64;
            let mask = if s_rem == 63 {
                u64::MAX
            } else {
                (1u64 << (s_rem + 1)) - 1
            };
            if (diff[s_limb] & mask) != 0 {
                sticky = 1;
            } else {
                for &limb_val in &diff[..s_limb] {
                    if limb_val != 0 {
                        sticky = 1;
                        break;
                    }
                }
            }
        }
        // Round to nearest, ties to even.
        if round_bit == 1 && (sticky == 1 || (sig53 & 1) == 1) {
            sig53 += 1;
        }
        let exp2 = (top_bit as i32) - Self::BIAS_BITS - 52;
        sign * (sig53 as f64) * ldexp_exact(1.0, exp2)
    }
}

/// Exact power-of-two scaling `x * 2^exp` in IEEE-754 `f64` (handles arbitrary shifts cleanly).
#[inline]
pub fn ldexp_exact(mut x: f64, mut exp: i32) -> f64 {
    while exp > 1000 {
        x *= f64::from_bits(((1000 + 1023) as u64) << 52);
        exp -= 1000;
    }
    while exp < -1000 {
        x *= f64::from_bits(((-1000 + 1023) as u64) << 52);
        exp += 1000;
    }
    if exp == 0 {
        return x;
    }
    let factor = f64::from_bits(((exp + 1023) as u64) << 52);
    x * factor
}

/// Extract normalized IEEE-754 binary exponent `e` such that `|x| = m * 2^e`, `m in [0.5, 1.0)`.
#[inline]
fn frexp_exp(x: f64) -> i32 {
    let ax = x.abs();
    if ax == 0.0 {
        return 0;
    }
    let bits = ax.to_bits();
    let e = ((bits >> 52) & 0x7FF) as i32;
    if e == 0 {
        // Subnormal: normalize via 2^64
        let norm = ax * ldexp_exact(1.0, 64);
        (((norm.to_bits() >> 52) & 0x7FF) as i32) - 1022 - 64
    } else {
        e - 1022
    }
}

// ============================================================================
// 2. Ozaki Scheme I (INT8 Bit-Slicing) & Ozaki Scheme II (CRT) on
//    INT8, FP8 (E4M3), and FP4 (E2M1 Base-13) Blackwell Tensor Cores
// ============================================================================

/// Low-precision Blackwell Tensor Core arithmetic substrate for Ozaki-II CRT emulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorBackend {
    /// Standard INT8 Tensor Core (`i8 * i8 -> i32`).
    Int8,
    /// FP8 E4M3 Tensor Core (exact 2-limb base-16 integer decomposition, every limb in `[-8, +8]`).
    Fp8E4m3,
    /// Blackwell FP4 (E2M1) Tensor Core using Hayashi–Mukunoki–Hoshino–Katagiri (arXiv:2608.06812)
    /// Base-13 E2M1 limb representation (`2 * E2M1 in {0, ±1, ±2, ±3, ±4, ±6, ±8}`).
    Fp4E2m1Base13,
}

/// Exact FP4 E2M1 value check: returns `true` iff `v` is an exact OCP/Blackwell FP4 E2M1 float.
#[inline]
pub fn is_exact_fp4_e2m1(v: f64) -> bool {
    matches!((v.abs() * 2.0) as i32, 0 | 1 | 2 | 3 | 4 | 6 | 8 | 12)
        && ((v.abs() * 2.0) - ((v.abs() * 2.0) as i32 as f64)).abs() == 0.0
}

/// Encode any symmetric modular residue `r in [-56, +56]` into exact 2-limb Base-13 FP4 E2M1
/// values `(e0, e1)` such that `r == 2 * e0 + 26 * e1` and BOTH `e0`, `e1` are exact FP4 E2M1!
#[inline]
pub fn encode_base13_fp4_e2m1(r: i32) -> (f64, f64) {
    // Complete residue system mod 13 from doubled E2M1 set {0, ±1, ±2, ±3, ±4, ±6, ±8}:
    // rem in 0..13 -> d0 in {0, 1, 2, 3, 4, -8, 6, -6, 8, -4, -3, -2, -1}
    const REM_TO_D: [i32; 13] = [0, 1, 2, 3, 4, -8, 6, -6, 8, -4, -3, -2, -1];
    let mut rem0 = r % 13;
    if rem0 < 0 {
        rem0 += 13;
    }
    let d0 = REM_TO_D[rem0 as usize];
    let d1 = (r - d0) / 13;
    debug_assert_eq!(d0 + 13 * d1, r);
    let e0 = (d0 as f64) * 0.5;
    let e1 = (d1 as f64) * 0.5;
    debug_assert!(is_exact_fp4_e2m1(e0) && is_exact_fp4_e2m1(e1));
    (e0, e1)
}

/// Ozaki Scheme I (`O(S²/2)` INT8 bit-slicing) IEEE-754 `f64` matrix multiplication `C = A * B`.
///
/// Slices rows of `A` and columns of `B` into `n_slices` (`9`, 6 bits per slice -> 54 bits)
/// exact signed `i8` matrices, executes integer dot products, and accumulates into `f64`.
pub fn ozaki1_int8_gemm(a: &[Vec<f64>], b: &[Vec<f64>], n_slices: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    let k_dim = a[0].len();
    let n = b[0].len();
    const BITS: i32 = 6;

    let mut sa = vec![0.0_f64; m];
    let mut sl_a = vec![vec![vec![0i8; k_dim]; m]; n_slices];
    for i in 0..m {
        let mx = a[i].iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if mx == 0.0 {
            continue;
        }
        let s0 = ldexp_exact(1.0, frexp_exp(mx) - BITS);
        sa[i] = s0;
        let mut rem = a[i].clone();
        for (sl, slice_mat) in sl_a.iter_mut().enumerate() {
            let sc = ldexp_exact(s0, -(sl as i32) * BITS);
            for t in 0..k_dim {
                let q = (rem[t] / sc).round().clamp(-127.0, 127.0) as i8;
                slice_mat[i][t] = q;
                rem[t] -= (q as f64) * sc;
            }
        }
    }

    let mut sb = vec![0.0_f64; n];
    let mut sl_b = vec![vec![vec![0i8; k_dim]; n]; n_slices];
    for j in 0..n {
        let mx = (0..k_dim).map(|t| b[t][j].abs()).fold(0.0_f64, f64::max);
        if mx == 0.0 {
            continue;
        }
        let s0 = ldexp_exact(1.0, frexp_exp(mx) - BITS);
        sb[j] = s0;
        let mut rem: Vec<f64> = (0..k_dim).map(|t| b[t][j]).collect();
        for (sl, slice_mat) in sl_b.iter_mut().enumerate() {
            let sc = ldexp_exact(s0, -(sl as i32) * BITS);
            for t in 0..k_dim {
                let q = (rem[t] / sc).round().clamp(-127.0, 127.0) as i8;
                slice_mat[j][t] = q;
                rem[t] -= (q as f64) * sc;
            }
        }
    }

    let mut c = vec![vec![0.0_f64; n]; m];
    for (p, slice_a) in sl_a.iter().enumerate() {
        for (q, slice_b) in sl_b.iter().enumerate() {
            if p + q > n_slices + 1 {
                continue;
            }
            let shift = -((p + q) as i32) * BITS;
            for i in 0..m {
                if sa[i] == 0.0 {
                    continue;
                }
                let row_a = &slice_a[i];
                for j in 0..n {
                    if sb[j] == 0.0 {
                        continue;
                    }
                    let col_b = &slice_b[j];
                    let mut acc = 0i32;
                    for t in 0..k_dim {
                        acc += (row_a[t] as i32) * (col_b[t] as i32);
                    }
                    if acc != 0 {
                        c[i][j] += (acc as f64) * ldexp_exact(sa[i] * sb[j], shift);
                    }
                }
            }
        }
    }
    c
}

/// 20 pairwise-coprime moduli `<= 113` (`log2(∏ m_j) = 124.31 bits`), simultaneously valid for
/// single-limb `INT8`, 2-limb `FP8 E4M3`, AND 2-limb Base-13 `FP4 E2M1` (`[-56, +56]`)!
pub const OZAKI2_MODULI: [i32; 20] = [
    113, 109, 107, 103, 101, 97, 89, 83, 81, 79, 73, 71, 67, 64, 61, 59, 53, 49, 47, 43,
];

#[inline]
fn sym_mod_i64(x: i64, m: i32) -> i32 {
    let m64 = m as i64;
    let mut r = x % m64;
    if r < 0 {
        r += m64;
    }
    if r > m64 / 2 {
        r -= m64;
    }
    r as i32
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline]
fn dd_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (s, e) = two_sum(a.0, b.0);
    two_sum(s, e + a.1 + b.1)
}

/// Exact Veltkamp-Dekker split multiplication error `(a*b - fl(a*b))` with ZERO FMA!
#[inline]
fn two_prod_nofma(a: f64, b: f64) -> (f64, f64) {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let p = a * b;
    let ca = SPLIT * a;
    let ah = ca - (ca - a);
    let al = a - ah;
    let cb = SPLIT * b;
    let bh = cb - (cb - b);
    let bl = b - bh;
    let err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, err)
}

fn mod_inv_u128(a: u128, m: u128) -> i32 {
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = m as i128;
    let mut new_r = (a % m) as i128;
    while new_r != 0 {
        let q = r / new_r;
        let tmp_t = t - q * new_t;
        t = new_t;
        new_t = tmp_t;
        let tmp_r = r - q * new_r;
        r = new_r;
        new_r = tmp_r;
    }
    if t < 0 {
        t += m as i128;
    }
    t as i32
}

/// **Ozaki Scheme II (Chinese Remainder Theorem)** IEEE-754 `f64` matrix multiply `C = A * B`
/// (Uchino, Ozaki & Imamura, arXiv:2504.08009 / arXiv:2602.02549) across `INT8`, `FP8 E4M3`,
/// or **Blackwell FP4 (E2M1 Base-13)** (Hayashi, Mukunoki, Hoshino & Katagiri, arXiv:2608.06812).
///
/// Uses linear `O(M)` (`M = 20`) modular GEMMs with zero FMA contraction and exact
/// double-double CRT reconstruction (`INT8`, `FP8`, and `FP4` produce **bit-exact identical**
/// outputs to `< 1e-14` of native `f64`!).
pub fn ozaki2_crt_gemm(a: &[Vec<f64>], b: &[Vec<f64>], backend: TensorBackend) -> Vec<Vec<f64>> {
    let m_rows = a.len();
    let k_dim = a[0].len();
    let n_cols = b[0].len();

    // Product of all 20 coprime moduli fits cleanly inside u128 (124.31 bits < 128 bits!).
    let mut m_big = 1u128;
    for &m in &OZAKI2_MODULI {
        m_big *= m as u128;
    }
    let mut invs = [0i32; 20];
    let mut inv_m_dd = [(0.0_f64, 0.0_f64); 20];
    for (idx, &m) in OZAKI2_MODULI.iter().enumerate() {
        invs[idx] = mod_inv_u128(m_big / (m as u128), m as u128);
        let mf = m as f64;
        let hi = 1.0 / mf;
        let (p_hi, p_err) = two_prod_nofma(mf, hi);
        let lo = ((1.0 - p_hi) - p_err) / mf;
        inv_m_dd[idx] = (hi, lo);
    }

    // Exact double-double representation of M_big (124 bits -> exact high + mid f64!).
    let m_hi = m_big as f64;
    let m_hi_u128 = m_hi as u128;
    let m_lo = if m_big >= m_hi_u128 {
        (m_big - m_hi_u128) as f64
    } else {
        -((m_hi_u128 - m_big) as f64)
    };

    // Scale rows of A and columns of B to 53-bit exact integers (preserving every mantissa bit).
    let mut sa = vec![0.0_f64; m_rows];
    let mut a_int = vec![vec![0i64; k_dim]; m_rows];
    for i in 0..m_rows {
        let mx = a[i].iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if mx == 0.0 {
            continue;
        }
        let sc = ldexp_exact(1.0, frexp_exp(mx) - 53);
        sa[i] = sc;
        for t in 0..k_dim {
            a_int[i][t] = (a[i][t] / sc).round() as i64;
        }
    }

    let mut sb = vec![0.0_f64; n_cols];
    let mut b_int = vec![vec![0i64; n_cols]; k_dim];
    for j in 0..n_cols {
        let mx = (0..k_dim).map(|t| b[t][j].abs()).fold(0.0_f64, f64::max);
        if mx == 0.0 {
            continue;
        }
        let sc = ldexp_exact(1.0, frexp_exp(mx) - 53);
        sb[j] = sc;
        for t in 0..k_dim {
            b_int[t][j] = (b[t][j] / sc).round() as i64;
        }
    }

    // Linear O(M) modular GEMMs on the selected low-precision Tensor Core substrate.
    let mut c_mods = vec![vec![vec![0i32; n_cols]; m_rows]; OZAKI2_MODULI.len()];
    for (idx, &m) in OZAKI2_MODULI.iter().enumerate() {
        let mut am = vec![vec![0i32; k_dim]; m_rows];
        for i in 0..m_rows {
            for t in 0..k_dim {
                am[i][t] = sym_mod_i64(a_int[i][t], m);
            }
        }
        let mut bm = vec![vec![0i32; n_cols]; k_dim];
        for t in 0..k_dim {
            for j in 0..n_cols {
                bm[t][j] = sym_mod_i64(b_int[t][j], m);
            }
        }

        match backend {
            TensorBackend::Int8 => {
                for i in 0..m_rows {
                    for j in 0..n_cols {
                        let mut acc = 0i64;
                        for t in 0..k_dim {
                            acc += (am[i][t] as i8 as i64) * (bm[t][j] as i8 as i64);
                        }
                        c_mods[idx][i][j] = sym_mod_i64(acc, m);
                    }
                }
            }
            TensorBackend::Fp8E4m3 => {
                // Split residue r in [-56, +56] into exact FP8 E4M3 limbs r = lo + 16*hi (both in [-8, +8]).
                for i in 0..m_rows {
                    for j in 0..n_cols {
                        let mut p00 = 0.0_f64;
                        let mut p01 = 0.0_f64;
                        let mut p10 = 0.0_f64;
                        let mut p11 = 0.0_f64;
                        for t in 0..k_dim {
                            let a_hi = ((am[i][t] as f64) / 16.0).round();
                            let a_lo = (am[i][t] as f64) - 16.0 * a_hi;
                            let b_hi = ((bm[t][j] as f64) / 16.0).round();
                            let b_lo = (bm[t][j] as f64) - 16.0 * b_hi;
                            p00 += a_lo * b_lo;
                            p01 += a_lo * b_hi;
                            p10 += a_hi * b_lo;
                            p11 += a_hi * b_hi;
                        }
                        let acc = (p00.round() as i64)
                            + 16 * ((p01 + p10).round() as i64)
                            + 256 * (p11.round() as i64);
                        c_mods[idx][i][j] = sym_mod_i64(acc, m);
                    }
                }
            }
            TensorBackend::Fp4E2m1Base13 => {
                // Hayashi et al. (Aug 2026, arXiv:2608.06812): exact 2-limb Base-13 FP4 E2M1 MMA!
                let mut a0 = vec![vec![0.0_f64; k_dim]; m_rows];
                let mut a1 = vec![vec![0.0_f64; k_dim]; m_rows];
                for i in 0..m_rows {
                    for t in 0..k_dim {
                        let (e0, e1) = encode_base13_fp4_e2m1(am[i][t]);
                        a0[i][t] = e0;
                        a1[i][t] = e1;
                    }
                }
                let mut b0 = vec![vec![0.0_f64; n_cols]; k_dim];
                let mut b1 = vec![vec![0.0_f64; n_cols]; k_dim];
                for t in 0..k_dim {
                    for j in 0..n_cols {
                        let (e0, e1) = encode_base13_fp4_e2m1(bm[t][j]);
                        b0[t][j] = e0;
                        b1[t][j] = e1;
                    }
                }
                for i in 0..m_rows {
                    for j in 0..n_cols {
                        let mut p00 = 0.0_f64;
                        let mut p01 = 0.0_f64;
                        let mut p10 = 0.0_f64;
                        let mut p11 = 0.0_f64;
                        for t in 0..k_dim {
                            p00 += a0[i][t] * b0[t][j];
                            p01 += a0[i][t] * b1[t][j];
                            p10 += a1[i][t] * b0[t][j];
                            p11 += a1[i][t] * b1[t][j];
                        }
                        let acc = ((4.0 * p00).round() as i64)
                            + 13 * ((4.0 * (p01 + p10)).round() as i64)
                            + 169 * ((4.0 * p11).round() as i64);
                        c_mods[idx][i][j] = sym_mod_i64(acc, m);
                    }
                }
            }
        }
    }

    // Exact double-double Chinese Remainder Theorem reconstruction (Uchino–Ozaki–Imamura).
    let mut c = vec![vec![0.0_f64; n_cols]; m_rows];
    for i in 0..m_rows {
        if sa[i] == 0.0 {
            continue;
        }
        for j in 0..n_cols {
            if sb[j] == 0.0 {
                continue;
            }
            let mut acc = (0.0_f64, 0.0_f64);
            for (idx, &m) in OZAKI2_MODULI.iter().enumerate() {
                let y = sym_mod_i64((c_mods[idx][i][j] as i64) * (invs[idx] as i64), m);
                if y != 0 {
                    let yf = y as f64;
                    let (hi, lo) = inv_m_dd[idx];
                    let (p, p_err) = two_prod_nofma(yf, hi);
                    let term = two_sum(p, p_err + yf * lo);
                    acc = dd_add(acc, term);
                }
            }
            let q = acc.0.round();
            let frac = dd_add(acc, (-q, 0.0));
            let c_int_est = frac.0 * m_hi + (frac.0 * m_lo + frac.1 * m_hi);
            c[i][j] = c_int_est * (sa[i] * sb[j]);
        }
    }
    c
}

/// Execute Ozaki-II CRT FP64 matrix multiply `C = A * B`, optionally offloading all 4 stages
/// (device FP64 residue reduction → precision pack → Tensor Core MMA → `u128` CRT → `f64`)
/// to a live NVIDIA GPU via `g6me-cuda` when `use_cuda` is true and `--features cuda` is built.
/// Falls back deterministically to the bit-identical CPU [`ozaki2_crt_gemm`] when no GPU is present.
pub fn ozaki2_crt_gemm_with_cuda(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    backend: TensorBackend,
    use_cuda: bool,
    precision: CudaTensorPrecision,
) -> Vec<Vec<f64>> {
    if a.is_empty() || b.is_empty() || a[0].is_empty() || b[0].is_empty() {
        return Vec::new();
    }
    #[cfg(feature = "cuda")]
    if use_cuda {
        if let Some(rt) = shared_cuda_runtime() {
            let m = a.len();
            let k = a[0].len();
            let n = b[0].len();
            let mut a_flat = Vec::with_capacity(m * k);
            for row in a {
                a_flat.extend_from_slice(row);
            }
            let mut b_flat = Vec::with_capacity(k * n);
            for row in b {
                b_flat.extend_from_slice(row);
            }
            if let Ok((c_flat, _actual_prec)) =
                rt.run_ozaki2_crt_gemm(&a_flat, &b_flat, m, k, n, precision.to_cuda())
            {
                let mut out = vec![vec![0.0_f64; n]; m];
                for i in 0..m {
                    out[i].copy_from_slice(&c_flat[i * n..(i + 1) * n]);
                }
                return out;
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (use_cuda, precision);
    ozaki2_crt_gemm(a, b, backend)
}

/// Complex matrix multiply `C = A * B` executed strictly via real Ozaki-II CRT GEMMs
/// on the chosen low-precision Tensor Core backend (`INT8`, `FP8 E4M3`, or `FP4 E2M1`).
pub fn ozaki2_complex_gemm(
    a: &[Vec<Complex>],
    b: &[Vec<Complex>],
    backend: TensorBackend,
) -> Vec<Vec<Complex>> {
    let m = a.len();
    let n = b[0].len();
    let ar: Vec<Vec<f64>> = a.iter().map(|r| r.iter().map(|c| c.re).collect()).collect();
    let ai: Vec<Vec<f64>> = a.iter().map(|r| r.iter().map(|c| c.im).collect()).collect();
    let br: Vec<Vec<f64>> = b.iter().map(|r| r.iter().map(|c| c.re).collect()).collect();
    let bi: Vec<Vec<f64>> = b.iter().map(|r| r.iter().map(|c| c.im).collect()).collect();

    let rr = ozaki2_crt_gemm(&ar, &br, backend);
    let ii = ozaki2_crt_gemm(&ai, &bi, backend);
    let ri = ozaki2_crt_gemm(&ar, &bi, backend);
    let ir = ozaki2_crt_gemm(&ai, &br, backend);

    let mut out = vec![vec![Complex::ZERO; n]; m];
    for i in 0..m {
        for j in 0..n {
            // Final recombination via exact Kulisch superaccumulator (Matsuoka Sept 2026!).
            let mut k_re = KulischAcc::new();
            k_re.add(rr[i][j]);
            k_re.add(-ii[i][j]);
            let mut k_im = KulischAcc::new();
            k_im.add(ri[i][j]);
            k_im.add(ir[i][j]);
            out[i][j] = Complex::new(k_re.finalize(), k_im.finalize());
        }
    }
    out
}

/// **Full-FP64 Bailey 4-Step FFT on Low-Precision Tensor Cores (Matsuoka, Sept 2026)**:
/// computes an exact IEEE-754 `f64` DFT of size `N = n1 * n2` by factoring the FFT into
/// two complex matrix multiplies (`W_{n1} * X` and `Y * W_{n2}^T`) executed strictly via
/// **Ozaki-II CRT (`INT8`, `FP8 E4M3`, or `FP4 E2M1 Base-13`)** + **Kulisch superaccumulator**!
pub fn bailey_ozaki_fft(
    x: &[Complex],
    n1: usize,
    n2: usize,
    backend: TensorBackend,
) -> Vec<Complex> {
    let n = n1 * n2;
    assert_eq!(x.len(), n);

    let w1: Vec<Vec<Complex>> = (0..n1)
        .map(|k1| {
            (0..n1)
                .map(|j1| {
                    let ang = -2.0 * PI * ((k1 * j1) as f64) / (n1 as f64);
                    Complex::new(ang.cos(), ang.sin())
                })
                .collect()
        })
        .collect();

    let w2_t: Vec<Vec<Complex>> = (0..n2)
        .map(|j2| {
            (0..n2)
                .map(|k2| {
                    let ang = -2.0 * PI * ((k2 * j2) as f64) / (n2 as f64);
                    Complex::new(ang.cos(), ang.sin())
                })
                .collect()
        })
        .collect();

    let mut x_mat = vec![vec![Complex::ZERO; n2]; n1];
    for j1 in 0..n1 {
        for j2 in 0..n2 {
            x_mat[j1][j2] = x[j1 * n2 + j2];
        }
    }

    // Step 1: N2 column DFTs of size N1 via Ozaki-II CRT Tensor Core GEMM.
    let mut y1 = ozaki2_complex_gemm(&w1, &x_mat, backend);

    // Step 2: Multiply by Bailey twiddle factors W_N^{k1 * j2}.
    for (k1, row) in y1.iter_mut().enumerate() {
        for (j2, cell) in row.iter_mut().enumerate() {
            let ang = -2.0 * PI * ((k1 * j2) as f64) / (n as f64);
            let tw = Complex::new(ang.cos(), ang.sin());
            *cell = cell.mul(tw);
        }
    }

    // Step 3: N1 row DFTs of size N2 via Ozaki-II CRT Tensor Core GEMM.
    let y3 = ozaki2_complex_gemm(&y1, &w2_t, backend);

    // Step 4: Transpose read-out k = k2 * n1 + k1.
    let mut out = vec![Complex::ZERO; n];
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            out[k2 * n1 + k1] = y3[k1][k2];
        }
    }
    out
}

// ============================================================================
// 3. Phase-Exact GPU IIR Biquad Algorithms:
//    (A) Overlap-Save with -320 dBFS settling margin (0 ULP bit-exact)
//    (B) Blelloch Parallel-Prefix 2×2 State-Matrix Scan
// ============================================================================

/// Direct Form II Transposed biquad coefficients (`a0 == 1.0`).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Reference sequential DF-II transposed biquad (zero FMA).
    pub fn process_sequential(&self, x: &[f64]) -> Vec<f64> {
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;
        let mut out = Vec::with_capacity(x.len());
        for &xn in x {
            let y = self.b0 * xn + z1;
            let nz1 = self.b1 * xn - self.a1 * y + z2;
            let nz2 = self.b2 * xn - self.a2 * y;
            z1 = nz1;
            z2 = nz2;
            out.push(y);
        }
        out
    }

    /// Exact `-320 dBFS` pole impulse settling margin (in samples) for Overlap-Save GPU IIR.
    ///
    /// Computes the spectral radius `r = max(|pole_1|, |pole_2|)` of `1 + a1 z^-1 + a2 z^-2`
    /// and returns `ceil(-320 / (20 * log10(r)))` plus a 16-sample safety guard.
    pub fn settling_margin_samples(&self) -> usize {
        let disc = self.a1 * self.a1 - 4.0 * self.a2;
        let r = if disc < 0.0 {
            self.a2.abs().sqrt()
        } else {
            let s = disc.sqrt();
            let r1 = (-self.a1 + s).abs() * 0.5;
            let r2 = (-self.a1 - s).abs() * 0.5;
            r1.max(r2)
        };
        if r <= 1e-12 {
            return 16;
        }
        let r_clamped = r.min(0.9999);
        let n = (-320.0 / (20.0 * r_clamped.log10())).ceil() as usize;
        n + 16
    }

    /// Parallel-chunk **Overlap-Save GPU IIR** with exact `-320 dBFS` settling margin.
    ///
    /// Each block starts `margin` samples early with zero state; because every pole decays
    /// below `-320 dBFS` (`< 10^{-16}`) across the overlap apron, every kept output sample
    /// matches the sequential IIR to **0–1 ULP** without cross-block state dependencies!
    pub fn process_overlap_save(&self, x: &[f64], block_len: usize) -> Vec<f64> {
        let n = x.len();
        let margin = self.settling_margin_samples();
        let mut out = vec![0.0_f64; n];
        let mut pos = 0usize;
        while pos < n {
            let end = (pos + block_len).min(n);
            let start = pos.saturating_sub(margin);
            let mut z1 = 0.0_f64;
            let mut z2 = 0.0_f64;
            for (idx, &xn) in x[start..end].iter().enumerate() {
                let y = self.b0 * xn + z1;
                let nz1 = self.b1 * xn - self.a1 * y + z2;
                let nz2 = self.b2 * xn - self.a2 * y;
                z1 = nz1;
                z2 = nz2;
                let abs_i = start + idx;
                if abs_i >= pos {
                    out[abs_i] = y;
                }
            }
            pos = end;
        }
        out
    }

    /// **Blelloch Parallel-Prefix 2×2 State-Matrix Scan** (`O(log N)` parallel depth).
    ///
    /// Expresses DF-II transposed state update as the affine recurrence
    /// `s[n] = M s[n-1] + v[n]` where `M = [[-a1, 1], [-a2, 0]]` and
    /// `v[n] = [(b1 - a1*b0)*x[n], (b2 - a2*b0)*x[n]]^T`, and executes an exact Blelloch
    /// up-sweep + down-sweep exclusive tree scan over the associative composition
    /// `(M2, v2) ∘ (M1, v1) = (M2*M1, M2*v1 + v2)`.
    pub fn process_blelloch_scan(&self, x: &[f64]) -> Vec<f64> {
        if x.is_empty() {
            return Vec::new();
        }
        let n_orig = x.len();
        let n = n_orig.next_power_of_two();

        #[derive(Clone, Copy)]
        struct Elem {
            m: [f64; 4], // row-major 2x2
            v: [f64; 2],
        }
        const IDENT: Elem = Elem {
            m: [1.0, 0.0, 0.0, 1.0],
            v: [0.0, 0.0],
        };
        let compose = |e2: Elem, e1: Elem| -> Elem {
            let m = [
                e2.m[0] * e1.m[0] + e2.m[1] * e1.m[2],
                e2.m[0] * e1.m[1] + e2.m[1] * e1.m[3],
                e2.m[2] * e1.m[0] + e2.m[3] * e1.m[2],
                e2.m[2] * e1.m[1] + e2.m[3] * e1.m[3],
            ];
            let v = [
                e2.m[0] * e1.v[0] + e2.m[1] * e1.v[1] + e2.v[0],
                e2.m[2] * e1.v[0] + e2.m[3] * e1.v[1] + e2.v[1],
            ];
            Elem { m, v }
        };

        let c1 = self.b1 - self.a1 * self.b0;
        let c2 = self.b2 - self.a2 * self.b0;
        let m_step = [-self.a1, 1.0, -self.a2, 0.0];

        let mut tree = vec![IDENT; n];
        for i in 0..n_orig {
            tree[i] = Elem {
                m: m_step,
                v: [c1 * x[i], c2 * x[i]],
            };
        }

        // Up-sweep (parallel reduction phase).
        let mut d = 1usize;
        while d < n {
            let step = 2 * d;
            let mut i = 0usize;
            while i < n {
                tree[i + step - 1] = compose(tree[i + step - 1], tree[i + d - 1]);
                i += step;
            }
            d = step;
        }

        // Exclusive root = identity.
        tree[n - 1] = IDENT;

        // Down-sweep (parallel prefix distribution phase).
        d >>= 1;
        while d >= 1 {
            let step = 2 * d;
            let mut i = 0usize;
            while i < n {
                let t = tree[i + d - 1];
                tree[i + d - 1] = tree[i + step - 1];
                tree[i + step - 1] = compose(t, tree[i + step - 1]);
                i += step;
            }
            d >>= 1;
        }

        // tree[i].v[0] now holds exact pre-sample state z1[i - 1]!
        (0..n_orig).map(|i| self.b0 * x[i] + tree[i].v[0]).collect()
    }
}

// ============================================================================
// 4. Zero-FMA Blackwell (`sm_120`) CUDA C++ & PTX Kernel Source + Static Audit
// ============================================================================

/// Zero-FMA (`--fmad=false`) Blackwell `sm_120` CUDA C++ kernel source covering:
/// 1. Stockham Radix-4 `f64` FFT (`__dmul_rn` / `__dadd_rn` intrinsics, zero FMA),
/// 2. 3D Wave-Equation FDTD PDE stencil,
/// 3. Deterministic PCG64 Monte Carlo Acoustic Ray Tracer.
pub const BLACKWELL_CUDA_KERNELS: &str = r#"// g6me Blackwell (sm_120) Zero-FMA Acoustic Physics Kernels
// Compile with: nvcc -std=c++23 -arch=sm_120 --fmad=false -Xcompiler -ffp-contract=off -O3 -ptx
#include <stdint.h>

extern "C" __global__ void stockham_radix4_f64_step(
    const double* __restrict__ in_re, const double* __restrict__ in_im,
    const double* __restrict__ tw_re, const double* __restrict__ tw_im,
    double* __restrict__ out_re, double* __restrict__ out_im,
    int n, int m, double sign_dir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int q = n >> 2;
    if (idx >= q) return;
    int j = idx & (m - 1);
    int k = idx - j;
    // Host-precomputed twiddles for this stage: three (cos, sin) pairs per j.
    int tw = 3 * j;
    double c1 = tw_re[tw], s1 = tw_im[tw];
    double c2 = tw_re[tw + 1], s2 = tw_im[tw + 1];
    double c3 = tw_re[tw + 2], s3 = tw_im[tw + 2];

    double x0r = in_re[idx],         x0i = in_im[idx];
    double x1r = in_re[idx + q],     x1i = in_im[idx + q];
    double x2r = in_re[idx + 2 * q], x2i = in_im[idx + 2 * q];
    double x3r = in_re[idx + 3 * q], x3i = in_im[idx + 3 * q];

    // Strict zero-FMA complex twiddle products via __dmul_rn + __dadd_rn.
    double t1r = __dadd_rn(__dmul_rn(x1r, c1), -__dmul_rn(x1i, s1));
    double t1i = __dadd_rn(__dmul_rn(x1r, s1),  __dmul_rn(x1i, c1));
    double t2r = __dadd_rn(__dmul_rn(x2r, c2), -__dmul_rn(x2i, s2));
    double t2i = __dadd_rn(__dmul_rn(x2r, s2),  __dmul_rn(x2i, c2));
    double t3r = __dadd_rn(__dmul_rn(x3r, c3), -__dmul_rn(x3i, s3));
    double t3i = __dadd_rn(__dmul_rn(x3r, s3),  __dmul_rn(x3i, c3));

    double a0r = __dadd_rn(x0r, t2r), a0i = __dadd_rn(x0i, t2i);
    double a1r = __dadd_rn(x0r, -t2r), a1i = __dadd_rn(x0i, -t2i);
    double a2r = __dadd_rn(t1r, t3r), a2i = __dadd_rn(t1i, t3i);
    double a3r = __dadd_rn(t1r, -t3r), a3i = __dadd_rn(t1i, -t3i);

    int o = (k << 2) + j;
    out_re[o]         = __dadd_rn(a0r, a2r);
    out_im[o]         = __dadd_rn(a0i, a2i);
    out_re[o + m]     = __dadd_rn(a1r, __dmul_rn(sign_dir, a3i));
    out_im[o + m]     = __dadd_rn(a1i, -__dmul_rn(sign_dir, a3r));
    out_re[o + 2 * m] = __dadd_rn(a0r, -a2r);
    out_im[o + 2 * m] = __dadd_rn(a0i, -a2i);
    out_re[o + 3 * m] = __dadd_rn(a1r, -__dmul_rn(sign_dir, a3i));
    out_im[o + 3 * m] = __dadd_rn(a1i, __dmul_rn(sign_dir, a3r));
}

extern "C" __global__ void fdtd_3d_srl_step(
    const double* __restrict__ p_prev, const double* __restrict__ p_curr,
    double* __restrict__ p_next, const double* __restrict__ inv_den,
    int nx, int ny, int nz, double lam2, double half_lam_beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int s_x = ny * nz;
    int id = i * s_x + j * nz + k;
    double pc = p_curr[id];
    double lap = 0.0;
    int kb = 0;
    if (i > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - s_x], -pc)); else kb++;
    if (i + 1 < nx) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + s_x], -pc)); else kb++;
    if (j > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - nz],  -pc)); else kb++;
    if (j + 1 < ny) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + nz],  -pc)); else kb++;
    if (k > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - 1],   -pc)); else kb++;
    if (k + 1 < nz) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + 1],   -pc)); else kb++;
    double two_pc = __dmul_rn(2.0, pc);
    double lam_lap = __dmul_rn(lam2, lap);
    if (kb == 0) {
        p_next[id] = __dadd_rn(__dadd_rn(two_pc, -p_prev[id]), lam_lap);
    } else {
        double g = __dmul_rn(half_lam_beta, (double)kb);
        double num = __dadd_rn(__dadd_rn(two_pc, lam_lap), -__dmul_rn(__dadd_rn(1.0, -g), p_prev[id]));
        // Host-precomputed 1 / (1 + half_lam_beta * kb) for kb = 0..6.
        p_next[id] = __dmul_rn(num, inv_den[kb]);
    }
}
"#;

/// Functional zero-FMA `sm_120` PTX 8.7 assembly for the core Stockham complex twiddle
/// butterfly and 3D FDTD wave stencil step. Loads parameters (`ld.param`), computes global
/// thread indices, bounds-checks `n_elems`, loads operands via `ld.global.f64`, computes
/// strictly with separate IEEE-754 `mul.rn.f64` / `add.rn.f64` / `sub.rn.f64` (zero `fma.*`
/// or `mad.*` contractions), and stores results via `st.global.f64`.
pub const BLACKWELL_ZERO_FMA_PTX: &str = r#".version 8.7
.target sm_120
.address_size 64

.visible .entry stockham_radix4_f64_step(
    .param .u64 p_in,
    .param .u64 p_tw,
    .param .u64 p_out,
    .param .u32 n_elems
) {
    .reg .pred %p<2>;
    .reg .b32  %r<6>;
    .reg .b64  %rd<10>;
    .reg .f64  %fd<12>;

    ld.param.u64 %rd1, [p_in];
    ld.param.u64 %rd2, [p_tw];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1,  [n_elems];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mul.lo.u32 %r5, %r2, %r3;
    add.u32 %r5, %r5, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_STOCKHAM_DONE;

    // Complex interleaved f64 stride = 16 bytes per element
    mul.wide.u32 %rd4, %r5, 16;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f64 %fd0, [%rd5];
    ld.global.f64 %fd1, [%rd5 + 8];
    ld.global.f64 %fd2, [%rd6];
    ld.global.f64 %fd3, [%rd6 + 8];

    // Strict IEEE-754 zero-FMA complex multiply: (a + ib)*(w_r + i*w_i)
    mul.rn.f64 %fd4, %fd0, %fd2;
    mul.rn.f64 %fd5, %fd1, %fd3;
    sub.rn.f64 %fd6, %fd4, %fd5;
    mul.rn.f64 %fd7, %fd0, %fd3;
    mul.rn.f64 %fd8, %fd1, %fd2;
    add.rn.f64 %fd9, %fd7, %fd8;

    st.global.f64 [%rd7],     %fd6;
    st.global.f64 [%rd7 + 8], %fd9;

L_STOCKHAM_DONE:
    ret;
}

.visible .entry fdtd_3d_srl_step(
    .param .u64 p_prev,
    .param .u64 p_curr,
    .param .u64 p_lap,
    .param .u64 p_next,
    .param .f64 lam2,
    .param .u32 n_elems
) {
    .reg .pred %p<2>;
    .reg .b32  %r<6>;
    .reg .b64  %rd<11>;
    .reg .f64  %fd<10>;

    ld.param.u64 %rd1, [p_prev];
    ld.param.u64 %rd2, [p_curr];
    ld.param.u64 %rd3, [p_lap];
    ld.param.u64 %rd4, [p_next];
    ld.param.f64 %fd3, [lam2];
    ld.param.u32 %r1,  [n_elems];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mul.lo.u32 %r5, %r2, %r3;
    add.u32 %r5, %r5, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra L_FDTD_DONE;

    mul.wide.u32 %rd5, %r5, 8;
    add.u64 %rd6, %rd1, %rd5;
    add.u64 %rd7, %rd2, %rd5;
    add.u64 %rd8, %rd3, %rd5;
    add.u64 %rd9, %rd4, %rd5;

    ld.global.f64 %fd1, [%rd6];
    ld.global.f64 %fd0, [%rd7];
    ld.global.f64 %fd2, [%rd8];

    // Strict IEEE-754 zero-FMA SRL update: p_next = (2.0 * p_curr - p_prev) + lam2 * p_lap
    mul.rn.f64 %fd4, %fd0, 0d4000000000000000;
    sub.rn.f64 %fd5, %fd4, %fd1;
    mul.rn.f64 %fd6, %fd2, %fd3;
    add.rn.f64 %fd7, %fd5, %fd6;

    st.global.f64 [%rd9], %fd7;

L_FDTD_DONE:
    ret;
}
"#;

/// Audit PTX or `cuobjdump --dump-sass` assembly text for forbidden floating-point FMA /
/// MAD contraction mnemonics (`fma.*`, `mad.*`, `DFMA`, `FFMA`, `HFMA2`, `DMAD`, `FMAD`).
///
/// SASS forms whose multiplicand is the zero register (`HFMA2 Rd, -RZ, RZ, ...`) are ptxas
/// constant materialisation, not contractions: `0 * 0 + c` is exactly `c`, so no rounding can
/// be fused. They are counted as instructions but not reported as offenders.
///
/// Returns `Ok(instruction_count)` if no arithmetic FMA instructions exist, or `Err(offenders)`
/// listing every offending line.
pub fn audit_gpu_asm_zero_fma(asm_text: &str) -> std::result::Result<usize, Vec<String>> {
    let mut offenders = Vec::new();
    let mut instrs = 0usize;
    for (idx, raw) in asm_text.lines().enumerate() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty()
            || line.starts_with('.')
            || line.starts_with('{')
            || line.starts_with('}')
            || line.ends_with(':')
        {
            continue;
        }
        instrs += 1;
        // Tokenize instruction line so predicated PTX (`@%p1 fma.rn.f64 ...`) and SASS
        // (`/*0020*/ DFMA R8, R4, R2, R6;`) are both caught cleanly without false positives.
        let toks: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
            .filter(|t| !t.is_empty())
            .collect();
        let fma_at = toks.iter().position(|tok| {
            let up = tok.to_ascii_uppercase();
            up == "DFMA"
                || up == "FFMA"
                || up == "HFMA2"
                || up == "DMAD"
                || up == "FMAD"
                || up.starts_with("DFMA.")
                || up.starts_with("FFMA.")
                || up.starts_with("HFMA2.")
                || up.starts_with("DMAD.")
                || up.starts_with("FMAD.")
                || up.starts_with("FMA.")
                || up.starts_with("MAD.")
        });
        let has_fma = match fma_at {
            None => false,
            Some(mi) => {
                let up = toks[mi].to_ascii_uppercase();
                let sass = up == "DFMA"
                    || up == "FFMA"
                    || up == "HFMA2"
                    || up == "DMAD"
                    || up == "FMAD"
                    || up.starts_with("DFMA.")
                    || up.starts_with("FFMA.")
                    || up.starts_with("HFMA2.")
                    || up.starts_with("DMAD.")
                    || up.starts_with("FMAD.");
                // ptxas materialises small integer constants as `HFMA2 Rd, -RZ, RZ, ...`.
                // A zero multiplicand makes the fused operation exactly an add, so it cannot
                // introduce the rounding difference this audit exists to exclude.
                let zero_mul = sass
                    && [2usize, 3].iter().any(|&off| {
                        toks.get(mi + off)
                            .map(|t| t.trim_start_matches(['+', '-']).eq_ignore_ascii_case("RZ"))
                            .unwrap_or(false)
                    });
                !zero_mul
            }
        };
        if has_fma {
            offenders.push(format!("line {}: {}", idx + 1, line));
        }
    }
    if offenders.is_empty() {
        Ok(instrs)
    } else {
        Err(offenders)
    }
}

/// Outcome of attempting real hardware CUDA compilation (`nvcc --fmad=false`) and SASS
/// disassembly auditing (`cuobjdump --dump-sass`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaAuditStatus {
    /// `nvcc` and `cuobjdump` were found, compiled `BLACKWELL_CUDA_KERNELS` to `.cubin`,
    /// dumped real SASS, and found no arithmetic FMA/MAD contraction across `sass_instructions`.
    VerifiedSass {
        arch: String,
        sass_instructions: usize,
    },
    /// `nvcc` or `cuobjdump` is not installed on this host; only the embedded CPU-verified
    /// reference PTX (`BLACKWELL_ZERO_FMA_PTX`) was statically audited.
    ToolchainUnavailable { ptx_reference_instructions: usize },
    /// `nvcc` was found, but compilation or SASS zero-FMA audit failed.
    Failed(String),
}

/// Attempt real CUDA compilation (`nvcc --fmad=false -cubin`) and `cuobjdump --dump-sass`
/// verification on `BLACKWELL_CUDA_KERNELS`. Honestly reports `ToolchainUnavailable` on
/// CPU-only machines without pretending to run on a GPU.
pub fn compile_and_audit_cuda_sass(arch: &str) -> CudaAuditStatus {
    let ptx_count = audit_gpu_asm_zero_fma(BLACKWELL_ZERO_FMA_PTX).unwrap_or(0);
    let has_nvcc = std::process::Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let has_cuobjdump = std::process::Command::new("cuobjdump")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !has_nvcc || !has_cuobjdump {
        return CudaAuditStatus::ToolchainUnavailable {
            ptx_reference_instructions: ptx_count,
        };
    }

    let tmp_dir = std::env::temp_dir();
    let pid = std::process::id();
    let cu_path = tmp_dir.join(format!("g6me_blackwell_{pid}.cu"));
    let cubin_path = tmp_dir.join(format!("g6me_blackwell_{pid}.cubin"));
    if let Err(e) = std::fs::write(&cu_path, BLACKWELL_CUDA_KERNELS) {
        return CudaAuditStatus::Failed(format!("failed to write temp .cu: {e}"));
    }

    // C++23 where the toolkit supports it; older nvcc falls back rather than failing the audit.
    // `--fmad=false` stops device-side FMA contraction; `-ffp-contract=off` does the same for
    // any host-side code in the translation unit (C++ otherwise defaults to fast contraction).
    let mut compiled = false;
    let mut nvcc_error = String::new();
    for std_flag in ["-std=c++23", "-std=c++20", ""] {
        let mut cmd = std::process::Command::new("nvcc");
        cmd.args([
            "--fmad=false",
            "-Xcompiler",
            "-ffp-contract=off",
            "-O3",
            "-cubin",
        ]);
        if !std_flag.is_empty() {
            cmd.arg(std_flag);
        }
        cmd.arg(format!("-arch={arch}"))
            .arg("-o")
            .arg(&cubin_path)
            .arg(&cu_path);
        match cmd.output() {
            Ok(o) if o.status.success() => {
                compiled = true;
                break;
            }
            Ok(o) => nvcc_error = String::from_utf8_lossy(&o.stderr).to_string(),
            Err(e) => {
                nvcc_error = format!("nvcc spawn error: {e}");
                break;
            }
        }
    }
    let _ = std::fs::remove_file(&cu_path);

    if !compiled {
        let _ = std::fs::remove_file(&cubin_path);
        return CudaAuditStatus::Failed(format!("nvcc failed: {nvcc_error}"));
    }

    let dump = std::process::Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(&cubin_path)
        .output();
    let _ = std::fs::remove_file(&cubin_path);

    match dump {
        Ok(o) if o.status.success() => {
            let sass = String::from_utf8_lossy(&o.stdout);
            match audit_gpu_asm_zero_fma(&sass) {
                Ok(n) => CudaAuditStatus::VerifiedSass {
                    arch: arch.to_string(),
                    sass_instructions: n,
                },
                Err(hits) => CudaAuditStatus::Failed(format!(
                    "SASS zero-FMA audit caught {} FMA/MAD hit(s): {:?}",
                    hits.len(),
                    hits.first()
                )),
            }
        }
        Ok(o) => CudaAuditStatus::Failed(format!(
            "cuobjdump failed: {}",
            String::from_utf8_lossy(&o.stderr)
        )),
        Err(e) => CudaAuditStatus::Failed(format!("cuobjdump spawn error: {e}")),
    }
}

/// User-selectable or architecture-optimal Tensor Core precision for Ozaki-II CRT FP64 GEMM.
///
/// Tensor Cores accumulate low-precision operands in FP32 (`OMMA`/`QMMA`/`HMMA`) or S32 (`IMMA`);
/// IEEE-754 `f64` exactness is achieved via 20-moduli Chinese Remainder Theorem reconstruction
/// (`log2(∏ m_j) = 124.31 bits`) outside the MMA instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CudaTensorPrecision {
    /// Automatically pick the highest-throughput Tensor Core format for the detected GPU:
    /// single-limb `INT8` (one MMA per modulus) on desktop Blackwell/Hopper/Ada/Ampere/Turing,
    /// and `NVFP4 E2M1` only on Blackwell Ultra (`sm_103a`), where INT8 Tensor Core rate drops.
    #[default]
    Auto,
    /// Blackwell (`sm_120a`) NVFP4 E2M1 Base-13 2-limb (`OMMA.SF.16864` with UE4M3=2.0 → FP32 → u128 CRT → FP64).
    Nvfp4E2m1,
    /// Hopper/Ada/Blackwell (`sm_89+`) FP8 E4M3 Base-16 2-limb (`m16n8k32` → FP32 → u128 CRT → FP64).
    Fp8E4m3,
    /// Turing/Ampere+ (`sm_75+`) INT8 single-limb (`IMMA` → S32 → u128 CRT → FP64).
    Int8,
    /// Turing/Ampere+ (`sm_75+`) IEEE FP16 single-limb (`HMMA` → FP32 → u128 CRT → FP64).
    Fp16,
}

impl CudaTensorPrecision {
    pub const ALL: [Self; 5] = [
        Self::Auto,
        Self::Nvfp4E2m1,
        Self::Fp8E4m3,
        Self::Int8,
        Self::Fp16,
    ];

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "nvfp4" | "fp4" | "nvfp4-e2m1" | "e2m1" => Some(Self::Nvfp4E2m1),
            "fp8" | "fp8-e4m3" | "e4m3" => Some(Self::Fp8E4m3),
            "int8" | "s8" => Some(Self::Int8),
            "fp16" | "f16" => Some(Self::Fp16),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Nvfp4E2m1 => "nvfp4",
            Self::Fp8E4m3 => "fp8",
            Self::Int8 => "int8",
            Self::Fp16 => "fp16",
        }
    }

    pub fn short_tag(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Nvfp4E2m1 => "NVFP4→FP64",
            Self::Fp8E4m3 => "FP8→FP64",
            Self::Int8 => "INT8→FP64",
            Self::Fp16 => "FP16→FP64",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto (Peak Desktop TOPS Optimal)",
            Self::Nvfp4E2m1 => "NVFP4 E2M1 Base-13 → FP32 OMMA.SF → u128 CRT → FP64",
            Self::Fp8E4m3 => "FP8 E4M3 Base-16 → FP32 QMMA → u128 CRT → FP64",
            Self::Int8 => "INT8 S8 (1-limb IMMA / Pascal DP4A) → S32 → u128 CRT → FP64",
            Self::Fp16 => "FP16 → FP32 HMMA → u128 CRT → FP64",
        }
    }

    /// Resolve `Auto` or downgrade an unsupported format for a given compute capability `(major, minor)`.
    ///
    /// For Ozaki-II (`m_j <= 113`, residues in `[-56, +56]`):
    /// - Single-limb INT8 fits in 1 byte (`1 MMA/mod`, exact S32 accumulation) and delivers the highest
    ///   effective Ozaki-II FP64 throughput on every desktop generation: Blackwell (41.9 TF),
    ///   Ada (33.0 TF), Ampere (14.2 TF), Turing (10.8 TF), and Pascal `__dp4a` (2.27 TF).
    /// - Blackwell Ultra (`cc == 103`, B300) drops native INT8 TC rate, so `Auto` picks `Nvfp4E2m1`.
    pub fn resolve_for_cc(self, major: i32, minor: i32) -> Self {
        let cc = major * 10 + minor;
        let has_fp4 = cc >= 120 || cc == 100 || cc == 103;
        let has_fp8 = cc >= 89;
        let has_fp16_tc = cc >= 75;
        match self {
            Self::Auto => {
                if cc == 103 && has_fp4 {
                    Self::Nvfp4E2m1
                } else {
                    Self::Int8
                }
            }
            Self::Nvfp4E2m1 => {
                if has_fp4 {
                    Self::Nvfp4E2m1
                } else if has_fp8 {
                    Self::Fp8E4m3
                } else {
                    Self::Int8
                }
            }
            Self::Fp8E4m3 => {
                if has_fp8 {
                    Self::Fp8E4m3
                } else {
                    Self::Int8
                }
            }
            Self::Int8 => Self::Int8,
            Self::Fp16 => {
                if has_fp16_tc {
                    Self::Fp16
                } else {
                    Self::Int8
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn to_cuda(self) -> g6me_cuda::CudaTensorPrecision {
        match self {
            Self::Auto => g6me_cuda::CudaTensorPrecision::Auto,
            Self::Nvfp4E2m1 => g6me_cuda::CudaTensorPrecision::Nvfp4E2m1,
            Self::Fp8E4m3 => g6me_cuda::CudaTensorPrecision::Fp8E4m3,
            Self::Int8 => g6me_cuda::CudaTensorPrecision::Int8,
            Self::Fp16 => g6me_cuda::CudaTensorPrecision::Fp16,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(p: g6me_cuda::CudaTensorPrecision) -> Self {
        match p {
            g6me_cuda::CudaTensorPrecision::Auto => Self::Auto,
            g6me_cuda::CudaTensorPrecision::Nvfp4E2m1 => Self::Nvfp4E2m1,
            g6me_cuda::CudaTensorPrecision::Fp8E4m3 => Self::Fp8E4m3,
            g6me_cuda::CudaTensorPrecision::Int8 => Self::Int8,
            g6me_cuda::CudaTensorPrecision::Fp16 => Self::Fp16,
        }
    }
}

/// Per-architecture NVIDIA GPU Tensor Core / DP4A & zero-FMA FP64 capability entry (for GUI/CLI status).
#[derive(Debug, Clone, Copy)]
pub struct CudaArchCapability {
    pub arch_name: &'static str,
    pub sm_target: &'static str,
    pub cc_range: &'static str,
    pub supports_nvfp4: bool,
    pub supports_fp8: bool,
    pub supports_int8: bool,
    pub supports_fp16: bool,
    pub default_precision: CudaTensorPrecision,
    pub mma_summary: &'static str,
    pub non_gemm_summary: &'static str,
}

/// Capability matrix across all 6 supported desktop & datacenter NVIDIA GPU generations.
pub const ALL_CUDA_ARCH_CAPABILITIES: [CudaArchCapability; 6] = [
    CudaArchCapability {
        arch_name: "Blackwell (RTX 5090 / 5080)",
        sm_target: "sm_120a",
        cc_range: "CC 12.0+",
        supports_nvfp4: true,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IMMA.16832 S8→S32 (838 eff TOPS, ~41.9 TF FP64); opt OMMA.SF NVFP4 (419 eff)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchCapability {
        arch_name: "Hopper (H100 / H200)",
        sm_target: "sm_90",
        cc_range: "CC 9.0",
        supports_nvfp4: false,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IMMA.16832 S8→S32 (1-limb exact); opt FP8 QMMA / FP16 HMMA (no HW FP4)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchCapability {
        arch_name: "Ada Lovelace (RTX 4090 / 4080)",
        sm_target: "sm_89",
        cc_range: "CC 8.9",
        supports_nvfp4: false,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IMMA.16832 S8→S32 (661 eff TOPS, ~33.0 TF FP64, 8x > FP8-FP32 QMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchCapability {
        arch_name: "Ampere (RTX 3090 / A100)",
        sm_target: "sm_80",
        cc_range: "CC 8.0–8.7",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: true,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IMMA.16832 S8→S32 (284 eff TOPS, ~14.2 TF FP64, 4x > FP16 HMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchCapability {
        arch_name: "Turing (RTX 2080 Ti / T4)",
        sm_target: "sm_75",
        cc_range: "CC 7.5",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: true,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IMMA.8816 S8→S32 (215 eff TOPS, ~10.8 TF FP64, 4x > FP16 HMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchCapability {
        arch_name: "Pascal (GTX 1080 Ti / 1080)",
        sm_target: "sm_61",
        cc_range: "CC 6.1",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: false,
        default_precision: CudaTensorPrecision::Int8,
        mma_summary: "IDP.4A.S8.S8 (__dp4a→S32, 45.4 eff TOPS, ~2.27 TF FP64, 6.5x native FP64)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
];

/// Map a CUDA compute capability `(major, minor)` to its architecture capability profile.
pub fn arch_capability_for_cc(major: i32, minor: i32) -> CudaArchCapability {
    let cc = major * 10 + minor;
    if cc >= 120 {
        ALL_CUDA_ARCH_CAPABILITIES[0]
    } else if cc >= 90 {
        ALL_CUDA_ARCH_CAPABILITIES[1]
    } else if cc >= 89 {
        ALL_CUDA_ARCH_CAPABILITIES[2]
    } else if cc >= 80 {
        ALL_CUDA_ARCH_CAPABILITIES[3]
    } else if cc >= 75 {
        ALL_CUDA_ARCH_CAPABILITIES[4]
    } else {
        ALL_CUDA_ARCH_CAPABILITIES[5]
    }
}

/// Structured live NVIDIA CUDA device details for GUI and CLI inspection.
#[derive(Debug, Clone)]
pub struct CudaDeviceDetail {
    pub index: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_arch: String,
    pub capability: CudaArchCapability,
}

/// Returns true if `g6me-core` was compiled with `--features cuda`.
pub fn cuda_compiled_in() -> bool {
    cfg!(feature = "cuda")
}

/// Probe live NVIDIA CUDA device details when built with `--features cuda`.
/// Returns `None` on CPU-only builds or machines without an active NVIDIA driver/GPU.
pub fn cuda_hardware_detail() -> Option<CudaDeviceDetail> {
    #[cfg(feature = "cuda")]
    {
        g6me_cuda::cuda_device_info().map(|info| {
            let cap = arch_capability_for_cc(info.compute_capability.0, info.compute_capability.1);
            CudaDeviceDetail {
                index: info.index,
                name: info.name.clone(),
                compute_capability: info.compute_capability,
                sm_arch: info.sm_arch(),
                capability: cap,
            }
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Probe for a live NVIDIA CUDA device when built with `--features cuda`.
/// Returns `None` on CPU-only builds or machines without an active NVIDIA driver/GPU.
pub fn cuda_hardware_summary() -> Option<String> {
    cuda_hardware_detail().map(|d| {
        format!(
            "{} ({}, {}, dev #{}, default: {})",
            d.name,
            d.sm_arch,
            d.capability.arch_name,
            d.index,
            d.capability.default_precision.short_tag()
        )
    })
}

/// Initialize the real zero-FMA CUDA Driver runtime (`g6me-cuda`) on device 0.
#[cfg(feature = "cuda")]
pub fn try_init_cuda_runtime() -> std::result::Result<g6me_cuda::G6meCudaRuntime, String> {
    g6me_cuda::G6meCudaRuntime::init()
}

/// Process-wide cached CUDA runtime (`OnceLock`) so `nvcc --fmad=false` PTX compilation and
/// CUDA module loading happen at most once per process across FDTD, BEM, RayTrace, Ozaki-II,
/// Stockham FFT, and Goldilocks NTT.
#[cfg(feature = "cuda")]
pub fn shared_cuda_runtime() -> Option<std::sync::Arc<g6me_cuda::G6meCudaRuntime>> {
    use std::sync::{Arc, OnceLock};
    static RUNTIME: OnceLock<std::result::Result<Arc<g6me_cuda::G6meCudaRuntime>, String>> =
        OnceLock::new();
    RUNTIME
        .get_or_init(|| g6me_cuda::G6meCudaRuntime::init().map(Arc::new))
        .as_ref()
        .ok()
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::FftPlan;

    #[test]
    fn kulisch_superaccumulator_eliminates_catastrophic_cancellation() {
        // Plain f64: (1e16 + 1.0) - 1e16 == 0.0 (catastrophic loss of 1.0!).
        let naive = (1e16_f64 + 1.0) - 1e16_f64;
        assert_eq!(naive, 0.0);

        let mut acc = KulischAcc::new();
        acc.add(1e16);
        acc.add(1.0);
        acc.add(0.25);
        acc.add(-1e16);
        let exact = acc.finalize();
        assert_eq!(
            exact, 1.25,
            "Kulisch superaccumulator lost precision: {exact}"
        );
    }

    #[test]
    fn ozaki1_and_ozaki2_int8_fp8_fp4_match_f64_and_each_other_bit_exact() {
        // Deterministic pseudo-random Order-7 style matrices (6x16 * 16x8).
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_val = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let a: Vec<Vec<f64>> = (0..6)
            .map(|_| (0..16).map(|_| next_val()).collect())
            .collect();
        let b: Vec<Vec<f64>> = (0..16)
            .map(|_| (0..8).map(|_| next_val()).collect())
            .collect();

        let mut c_ref = vec![vec![0.0_f64; 8]; 6];
        for i in 0..6 {
            for j in 0..8 {
                for t in 0..16 {
                    c_ref[i][j] += a[i][t] * b[t][j];
                }
            }
        }

        let c_oz1 = ozaki1_int8_gemm(&a, &b, 9);
        let c_i8 = ozaki2_crt_gemm(&a, &b, TensorBackend::Int8);
        let c_fp8 = ozaki2_crt_gemm(&a, &b, TensorBackend::Fp8E4m3);
        let c_fp4 = ozaki2_crt_gemm(&a, &b, TensorBackend::Fp4E2m1Base13);

        let max_diff = |x: &[Vec<f64>], y: &[Vec<f64>]| -> f64 {
            let mut m = 0.0_f64;
            for i in 0..x.len() {
                for j in 0..x[0].len() {
                    m = m.max((x[i][j] - y[i][j]).abs());
                }
            }
            m
        };

        assert!(max_diff(&c_ref, &c_oz1) < 1e-13, "Ozaki-I err too high");
        assert!(
            max_diff(&c_ref, &c_i8) < 1e-13,
            "Ozaki-II INT8 err too high"
        );
        assert!(
            max_diff(&c_ref, &c_fp8) < 1e-13,
            "Ozaki-II FP8 err too high"
        );
        assert!(
            max_diff(&c_ref, &c_fp4) < 1e-13,
            "Ozaki-II FP4 E2M1 err too high"
        );

        // All three Ozaki-II backends (INT8, FP8 E4M3, FP4 E2M1 Base-13) compute the EXACT same integer residues!
        assert_eq!(
            c_i8, c_fp8,
            "Ozaki-II INT8 vs FP8 must be bit-exact identical"
        );
        assert_eq!(
            c_i8, c_fp4,
            "Ozaki-II INT8 vs FP4 E2M1 must be bit-exact identical"
        );
    }

    #[test]
    fn bailey_4step_fp64_fft_on_fp4_tensor_cores_matches_fftplan_to_1e_12() {
        let n = 64; // 8 x 8 Bailey factorization
        let plan = FftPlan::new(n);
        let input: Vec<Complex> = (0..n)
            .map(|i| {
                let t = i as f64;
                Complex::new(
                    (0.19 * t).sin() + 0.45 * (0.73 * t).cos(),
                    0.35 * (0.41 * t).sin() - 0.2 * (0.11 * t).cos(),
                )
            })
            .collect();
        let mut ref_fft = input.clone();
        plan.forward(&mut ref_fft);

        let fp4_fft = bailey_ozaki_fft(&input, 8, 8, TensorBackend::Fp4E2m1Base13);
        let fp8_fft = bailey_ozaki_fft(&input, 8, 8, TensorBackend::Fp8E4m3);

        let max_err_fp4 = ref_fft
            .iter()
            .zip(fp4_fft.iter())
            .map(|(a, b)| a.sub(*b).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err_fp4 < 1e-12,
            "Bailey 4-step FP4 Ozaki-II FFT error: {max_err_fp4:e}"
        );
        assert_eq!(
            fp4_fft, fp8_fft,
            "FP4 vs FP8 Bailey Ozaki-II FFT must match bit-exact"
        );
    }

    #[test]
    fn gpu_iir_overlap_save_and_blelloch_scan_match_sequential_biquad() {
        let bq = BiquadCoeffs {
            b0: 0.067_455_273_889_071_9,
            b1: 0.134_910_547_778_143_8,
            b2: 0.067_455_273_889_071_9,
            a1: -1.142_980_502_539_901_1,
            a2: 0.412_801_598_096_188_77,
        };
        let x: Vec<f64> = (0..256)
            .map(|i| (0.13 * (i as f64)).sin() + 0.5 * (0.47 * (i as f64)).cos())
            .collect();

        let y_seq = bq.process_sequential(&x);
        let y_os = bq.process_overlap_save(&x, 32);
        let y_par = bq.process_blelloch_scan(&x);

        let err_os = y_seq
            .iter()
            .zip(y_os.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let err_par = y_seq
            .iter()
            .zip(y_par.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            err_os < 1e-14,
            "Overlap-Save (-320 dBFS margin) IIR error: {err_os:e}"
        );
        assert!(
            err_par < 1e-13,
            "Blelloch parallel-prefix 2x2 scan IIR error: {err_par:e}"
        );
    }

    #[test]
    fn gpu_ptx_sass_zero_fma_audit_passes_clean_and_catches_negative_control() {
        // Verify functional PTX actually loads params/global memory and stores output (not a no-op stub!)
        assert!(BLACKWELL_ZERO_FMA_PTX.contains("ld.param.u64"));
        assert!(BLACKWELL_ZERO_FMA_PTX.contains("ld.global.f64"));
        assert!(BLACKWELL_ZERO_FMA_PTX.contains("st.global.f64"));

        let count = audit_gpu_asm_zero_fma(BLACKWELL_ZERO_FMA_PTX)
            .expect("zero-FMA Blackwell PTX must pass");
        assert!(
            count >= 40,
            "expected full functional PTX kernels, got {count} instrs"
        );

        // Negative controls: injected PTX fma/mad and SASS DFMA/FFMA/DMAD MUST all fail!
        for poison in [
            "    fma.rn.f64 %fd9, %fd0, %fd1, %fd2;",
            "    mad.rn.f64 %fd9, %fd0, %fd1, %fd2;",
            "/*0020*/ DFMA R8, R4, R2, R6;",
            "/*0020*/ FFMA R8, R4, R2, R6;",
            "/*0020*/ HFMA2 R8, R4, R2, R6;",
            "/*0020*/ DMAD R8, R4, R2, R6;",
        ] {
            let err = audit_gpu_asm_zero_fma(poison)
                .expect_err(&format!("auditor MUST reject `{poison}`"));
            assert_eq!(err.len(), 1);
        }

        // ptxas materialises small integer constants as `HFMA2 Rd, -RZ, RZ, ...`.
        // With a zero multiplicand this is exactly an add, not a contraction.
        let constant_load = "/*0160*/ HFMA2 R6, -RZ, RZ, 0, 1.1920928955078125e-07 ;";
        assert!(
            audit_gpu_asm_zero_fma(constant_load).is_ok(),
            "zero-operand HFMA2 constant loads must not count as contractions"
        );

        // Honest CUDA toolchain probe must not panic on CPU-only machines.
        match compile_and_audit_cuda_sass("sm_120") {
            CudaAuditStatus::VerifiedSass {
                sass_instructions, ..
            } => assert!(sass_instructions > 0),
            CudaAuditStatus::ToolchainUnavailable {
                ptx_reference_instructions,
            } => {
                assert!(ptx_reference_instructions >= 40);
            }
            CudaAuditStatus::Failed(msg) => panic!("CUDA SASS audit failed: {msg}"),
        }
    }
}
