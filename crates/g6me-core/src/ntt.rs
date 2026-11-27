//! Exact Finite-Field Number-Theoretic Transform (NTT) FIR Convolution (`--exact-ntt`).
//!
//! Maps 24-bit signed integer PCM (`[-2^23, 2^23 - 1]`) and fixed-point FIR kernels
//! (`Q28` quantized `i64`) into the 64-bit **Goldilocks prime field**
//! `Z / pZ` with `p = 2^64 - 2^32 + 1 = 18_446_744_069_414_584_321` (Pollard 1971;
//! Agarwal & Burrus 1974), executing linear convolution via radix-2 Cooley–Tukey NTT
//! with **strictly 0 ULPs of intermediate floating-point rounding error** — mathematically
//! bit-identical on x86, ARM, RISC-V, and GPU targets without any floating-point flags.
//!
//! Because `p - 1 = 2^32 * (2^32 - 1)`, the Goldilocks field possesses exact primitive
//! `2^k`-th roots of unity up to `N = 2^32`, with primitive generator `g = 7` and
//! `2^32`-th root `w_32 = 7^(2^32 - 1) mod p = 1_753_635_133_440_165_772`.

/// The 64-bit Goldilocks prime `p = 2^64 - 2^32 + 1`.
pub const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

/// Primitive generator of `(Z / pZ)^*`.
pub const GOLDILOCKS_G: u64 = 7;

/// Primitive `2^32`-th root of unity in `Z / pZ`: `7^((p - 1) / 2^32) mod p`.
pub const GOLDILOCKS_W32: u64 = 1_753_635_133_440_165_772;

/// Fixed-point fractional bits for quantized FIR kernels (`Q28`, scale `2^28`).
///
/// With 24-bit PCM (`|x| <= 2^23`) and normalized FIR kernel L1-norm `<= 8.0` (`<= 2^31` in Q28),
/// the worst-case linear convolution sum is `<= 2^54`, leaving `512x` headroom below `p / 2`.
pub const NTT_KERNEL_QBITS: u32 = 28;

/// Fixed-point PCM integer scale (`2^23`, exact 24-bit signed audio range).
pub const NTT_PCM_SCALE: f64 = 8_388_608.0; // 2^23

/// Fixed-point FIR kernel scale (`2^28`).
pub const NTT_KERNEL_SCALE: f64 = 268_435_456.0; // 2^28

/// Exact modular addition `(a + b) mod p` for canonical `a, b < p`.
#[inline]
pub fn mod_add(a: u64, b: u64) -> u64 {
    let s = (a as u128) + (b as u128);
    let p = GOLDILOCKS_P as u128;
    if s >= p {
        (s - p) as u64
    } else {
        s as u64
    }
}

/// Exact modular subtraction `(a - b) mod p` for canonical `a, b < p`.
#[inline]
pub fn mod_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        GOLDILOCKS_P - (b - a)
    }
}

/// Fast exact Goldilocks modular reduction of a 128-bit product `x = a * b` mod `p`.
///
/// Uses the exact identities `2^64 ≡ 2^32 - 1 (mod p)` and `2^96 ≡ -1 (mod p)`:
/// writing `x = lo + mid * 2^64 + hi * 2^96` with `lo < 2^64`, `mid < 2^32`, `hi < 2^32`,
/// we get `x ≡ lo - hi + mid * (2^32 - 1) (mod p)`.
#[inline]
pub fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi_all = (x >> 64) as u64;
    let mid = hi_all & 0xFFFF_FFFF;
    let hi = hi_all >> 32;

    // Step 1: (lo - hi) mod p
    let (mut t, borrow) = lo.overflowing_sub(hi);
    if borrow {
        // Underflow by 2^64 ≡ 2^32 - 1 (mod p) -> subtract (2^32 - 1)
        t = t.wrapping_sub(0xFFFF_FFFF);
    }

    // Step 2: add mid * (2^32 - 1) = (mid << 32) - mid
    let mid_term = (mid << 32) - mid;
    let (mut res, carry) = t.overflowing_add(mid_term);
    if carry {
        // Overflow by 2^64 ≡ 2^32 - 1 (mod p)
        let (r2, carry2) = res.overflowing_add(0xFFFF_FFFF);
        res = if carry2 {
            r2.wrapping_add(0xFFFF_FFFF)
        } else {
            r2
        };
    }

    if res >= GOLDILOCKS_P {
        res -= GOLDILOCKS_P;
    }
    res
}

/// Exact modular multiplication `(a * b) mod p`.
#[inline]
pub fn mod_mul(a: u64, b: u64) -> u64 {
    reduce128((a as u128) * (b as u128))
}

/// Exact modular exponentiation `base^exp mod p` via binary square-and-multiply.
pub fn mod_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    base %= GOLDILOCKS_P;
    while exp > 0 {
        if (exp & 1) == 1 {
            acc = mod_mul(acc, base);
        }
        base = mod_mul(base, base);
        exp >>= 1;
    }
    acc
}

/// Exact multiplicative inverse `a^(-1) mod p` via Fermat's Little Theorem (`a^(p-2) mod p`).
pub fn mod_inv(a: u64) -> u64 {
    debug_assert!(
        !a.is_multiple_of(GOLDILOCKS_P),
        "cannot invert zero in Goldilocks field"
    );
    mod_pow(a, GOLDILOCKS_P - 2)
}

/// Map a signed `i64` into its canonical Goldilocks field representative `[0, p - 1]`.
#[inline]
pub fn i64_to_field(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) % GOLDILOCKS_P
    } else {
        let mag = (-(v as i128)) as u64 % GOLDILOCKS_P;
        if mag == 0 {
            0
        } else {
            GOLDILOCKS_P - mag
        }
    }
}

/// Map a canonical Goldilocks field element `[0, p - 1]` back to a balanced signed `i128`
/// in `[-(p-1)/2, +(p-1)/2]`.
#[inline]
pub fn field_to_i128(v: u64) -> i128 {
    let half = GOLDILOCKS_P / 2;
    if v <= half {
        v as i128
    } else {
        (v as i128) - (GOLDILOCKS_P as i128)
    }
}

/// Precomputed radix-2 Cooley–Tukey Number-Theoretic Transform plan over `Z / pZ`.
#[derive(Debug, Clone)]
pub struct NttPlan {
    n: usize,
    bitrev: Vec<usize>,
    twiddles_fwd: Vec<Vec<u64>>,
    twiddles_inv: Vec<Vec<u64>>,
    inv_n: u64,
}

impl NttPlan {
    /// Construct an exact NTT plan for power-of-two length `n` (`1 <= n <= 2^28`).
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 1,
            "NTT length must be a power of two, got {n}"
        );
        let log_n = n.trailing_zeros();
        assert!(log_n <= 32, "Goldilocks 2-adicity is 32, got 2^{log_n}");

        let mut bitrev = vec![0usize; n];
        if log_n > 0 {
            for (i, slot) in bitrev.iter_mut().enumerate() {
                *slot = i.reverse_bits() >> (usize::BITS - log_n);
            }
        }

        // Primitive n-th root of unity: w_n = w_32^(2^(32 - log_n)) mod p
        let w_n = mod_pow(GOLDILOCKS_W32, 1u64 << (32 - log_n));
        let w_n_inv = mod_inv(w_n);

        let mut twiddles_fwd = Vec::with_capacity(log_n as usize);
        let mut twiddles_inv = Vec::with_capacity(log_n as usize);
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step_exp = (n / len) as u64;
            let step_fwd = mod_pow(w_n, step_exp);
            let step_inv = mod_pow(w_n_inv, step_exp);

            let mut tf = Vec::with_capacity(half);
            let mut ti = Vec::with_capacity(half);
            let mut wf = 1u64;
            let mut wi = 1u64;
            for _ in 0..half {
                tf.push(wf);
                ti.push(wi);
                wf = mod_mul(wf, step_fwd);
                wi = mod_mul(wi, step_inv);
            }
            twiddles_fwd.push(tf);
            twiddles_inv.push(ti);
            len <<= 1;
        }

        Self {
            n,
            bitrev,
            twiddles_fwd,
            twiddles_inv,
            inv_n: mod_inv(n as u64),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn twiddles_fwd(&self) -> &[Vec<u64>] {
        &self.twiddles_fwd
    }

    pub fn twiddles_inv(&self) -> &[Vec<u64>] {
        &self.twiddles_inv
    }

    pub fn inv_n(&self) -> u64 {
        self.inv_n
    }

    /// Execute in-place forward Number-Theoretic Transform over `Z / pZ`.
    pub fn forward(&self, buf: &mut [u64]) {
        self.transform(buf, false);
    }

    /// Execute in-place inverse Number-Theoretic Transform over `Z / pZ` (including `n^-1`).
    pub fn inverse(&self, buf: &mut [u64]) {
        self.transform(buf, true);
        for v in buf.iter_mut() {
            *v = mod_mul(*v, self.inv_n);
        }
    }

    fn transform(&self, buf: &mut [u64], inverse: bool) {
        assert_eq!(buf.len(), self.n);
        if self.n <= 1 {
            return;
        }
        for (i, &j) in self.bitrev.iter().enumerate() {
            if i < j {
                buf.swap(i, j);
            }
        }
        let stages = if inverse {
            &self.twiddles_inv
        } else {
            &self.twiddles_fwd
        };
        for (stage_idx, twiddles) in stages.iter().enumerate() {
            let half = 1usize << stage_idx;
            let len = half << 1;
            let mut start = 0usize;
            while start < self.n {
                for (k, &w) in twiddles.iter().enumerate() {
                    let u = buf[start + k];
                    let v = mod_mul(buf[start + k + half], w);
                    buf[start + k] = mod_add(u, v);
                    buf[start + k + half] = mod_sub(u, v);
                }
                start += len;
            }
        }
    }
}

/// Exact integer linear convolution `y = x * h` via Goldilocks NTT (`0` rounding error).
///
/// Requires `sum_k |x[k]| * max|h|` to stay below `p / 2 ≈ 9.22e18`.
pub fn exact_int_convolve(x: &[i64], h: &[i64]) -> Vec<i128> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let out_len = x.len() + h.len() - 1;
    let n = out_len.next_power_of_two();
    let plan = NttPlan::new(n);

    let mut fx = vec![0u64; n];
    let mut fh = vec![0u64; n];
    for (i, &v) in x.iter().enumerate() {
        fx[i] = i64_to_field(v);
    }
    for (i, &v) in h.iter().enumerate() {
        fh[i] = i64_to_field(v);
    }

    plan.forward(&mut fx);
    plan.forward(&mut fh);
    for i in 0..n {
        fx[i] = mod_mul(fx[i], fh[i]);
    }
    plan.inverse(&mut fx);

    fx[..out_len].iter().map(|&v| field_to_i128(v)).collect()
}

/// Streaming Overlap-Add Exact Finite-Field NTT FIR Convolver for 24-bit fixed-point PCM
/// (`--exact-ntt`).
///
/// Quantizes the FIR impulse response once to `Q28` exact signed integers, converts each
/// incoming audio sample to exact 24-bit signed integer (`round(x * 2^23)`), executes
/// partitioned Overlap-Add convolution strictly inside the 64-bit Goldilocks prime field
/// `Z / pZ` (0 ULP intermediate error), and rescales the exact integer sum back to `f64`.
#[derive(Debug, Clone)]
pub struct NttOverlapAddConvolver {
    block_size: usize,
    ir_len: usize,
    plan: NttPlan,
    ir_ntt: Vec<u64>,
    in_buf: Vec<i64>,
    in_pos: usize,
    overlap: Vec<i128>,
    out_buf: Vec<i128>,
    out_pos: usize,
    out_ready: usize,
}

impl NttOverlapAddConvolver {
    /// Create a streaming exact Goldilocks NTT convolver from an `f64` FIR impulse response.
    pub fn new(ir: &[f64]) -> Self {
        let ir_len = ir.len().max(1);
        let block_size = ir_len.next_power_of_two().clamp(256, 4096);
        let fft_n = (block_size + ir_len - 1).next_power_of_two();
        let plan = NttPlan::new(fft_n);

        let mut ir_ntt = vec![0u64; fft_n];
        for (i, &v) in ir.iter().enumerate() {
            let q = (v * NTT_KERNEL_SCALE).round() as i64;
            ir_ntt[i] = i64_to_field(q);
        }
        plan.forward(&mut ir_ntt);

        Self {
            block_size,
            ir_len,
            plan,
            ir_ntt,
            in_buf: vec![0i64; block_size],
            in_pos: 0,
            overlap: vec![0i128; ir_len.saturating_sub(1)],
            out_buf: vec![0i128; block_size],
            out_pos: 0,
            out_ready: 0,
        }
    }

    /// Convolve an entire slice of `f64` audio samples (mapped to 24-bit exact integer PCM)
    /// through the Goldilocks prime field NTT convolver, returning `x.len()` samples.
    pub fn convolve_slice(&self, x: &[f64]) -> Vec<f64> {
        if x.is_empty() {
            return Vec::new();
        }
        let inv_scale = 1.0 / (NTT_PCM_SCALE * NTT_KERNEL_SCALE);
        let n = self.plan.len();
        let mut out = vec![0i128; x.len() + self.ir_len - 1];
        let mut blk = vec![0u64; n];

        let mut offset = 0usize;
        while offset < x.len() {
            let take = (x.len() - offset).min(self.block_size);
            blk.fill(0);
            for i in 0..take {
                let q = (x[offset + i] * NTT_PCM_SCALE)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0) as i64;
                blk[i] = i64_to_field(q);
            }
            self.plan.forward(&mut blk);
            for (b, &kern) in blk.iter_mut().zip(self.ir_ntt.iter()) {
                *b = mod_mul(*b, kern);
            }
            self.plan.inverse(&mut blk);
            let conv_len = take + self.ir_len - 1;
            for i in 0..conv_len {
                out[offset + i] += field_to_i128(blk[i]);
            }
            offset += take;
        }

        out[..x.len()]
            .iter()
            .map(|&v| (v as f64) * inv_scale)
            .collect()
    }

    /// Convolve an entire slice of `f64` audio samples through the exact 64-bit Goldilocks
    /// prime field NTT convolver, optionally offloading NTT butterfly stages & pointwise
    /// field multiplication to a live NVIDIA GPU via `g6me-cuda` when `use_cuda` is true.
    /// Falls back bit-identically (0 ULP) to [`Self::convolve_slice`] on CPU-only hosts.
    pub fn convolve_slice_with_cuda(&self, x: &[f64], use_cuda: bool) -> Vec<f64> {
        #[cfg(feature = "cuda")]
        if use_cuda && !x.is_empty() {
            if let Some(rt) = crate::blackwell::shared_cuda_runtime() {
                let inv_scale = 1.0 / (NTT_PCM_SCALE * NTT_KERNEL_SCALE);
                let n = self.plan.len();
                let mut out = vec![0i128; x.len() + self.ir_len - 1];
                let mut blk = vec![0u64; n];
                let mut offset = 0usize;
                let mut gpu_ok = true;
                while offset < x.len() {
                    let take = (x.len() - offset).min(self.block_size);
                    blk.fill(0);
                    for i in 0..take {
                        let q = (x[offset + i] * NTT_PCM_SCALE)
                            .round()
                            .clamp(-8_388_608.0, 8_388_607.0)
                            as i64;
                        blk[i] = i64_to_field(q);
                    }
                    match rt.run_goldilocks_ntt_convolve(
                        &blk,
                        &self.ir_ntt,
                        self.plan.twiddles_fwd(),
                        self.plan.twiddles_inv(),
                        self.plan.inv_n(),
                    ) {
                        Ok(res) => {
                            let conv_len = take + self.ir_len - 1;
                            for i in 0..conv_len {
                                out[offset + i] += field_to_i128(res[i]);
                            }
                        }
                        Err(_) => {
                            gpu_ok = false;
                            break;
                        }
                    }
                    offset += take;
                }
                if gpu_ok {
                    return out[..x.len()]
                        .iter()
                        .map(|&v| (v as f64) * inv_scale)
                        .collect();
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = use_cuda;
        self.convolve_slice(x)
    }

    /// Process a streaming block of `f64` samples in place with exact block-size-invariant
    /// Goldilocks NTT arithmetic (zero floating-point rounding inside the convolution).
    ///
    /// Note: introduces `block_size` samples of deterministic streaming framing latency,
    /// matching `self.latency()`.
    pub fn process_sample(&mut self, sample: f64) -> f64 {
        let q = (sample * NTT_PCM_SCALE)
            .round()
            .clamp(-8_388_608.0, 8_388_607.0) as i64;
        self.in_buf[self.in_pos] = q;
        self.in_pos += 1;

        let out_val = if self.out_pos < self.out_ready {
            let v = self.out_buf[self.out_pos];
            self.out_pos += 1;
            (v as f64) / (NTT_PCM_SCALE * NTT_KERNEL_SCALE)
        } else {
            0.0
        };

        if self.in_pos == self.block_size {
            self.flush_block();
        }
        out_val
    }

    pub fn latency(&self) -> usize {
        self.block_size
    }

    pub fn reset(&mut self) {
        self.in_buf.fill(0);
        self.in_pos = 0;
        self.overlap.fill(0);
        self.out_buf.fill(0);
        self.out_pos = 0;
        self.out_ready = 0;
    }

    fn flush_block(&mut self) {
        let n = self.plan.len();
        let mut blk = vec![0u64; n];
        for (i, &v) in self.in_buf.iter().enumerate() {
            blk[i] = i64_to_field(v);
        }
        self.plan.forward(&mut blk);
        for (b, &kern) in blk.iter_mut().zip(self.ir_ntt.iter()) {
            *b = mod_mul(*b, kern);
        }
        self.plan.inverse(&mut blk);

        let ov_len = self.overlap.len();
        for (i, &b) in blk.iter().take(self.block_size).enumerate() {
            let mut val = field_to_i128(b);
            if i < ov_len {
                val += self.overlap[i];
            }
            self.out_buf[i] = val;
        }
        for i in 0..ov_len {
            let tail = field_to_i128(blk[self.block_size + i]);
            let carry = if self.block_size + i < ov_len {
                self.overlap[self.block_size + i]
            } else {
                0
            };
            self.overlap[i] = tail + carry;
        }
        self.in_pos = 0;
        self.out_pos = 0;
        self.out_ready = self.block_size;
    }
}

/// Zero-floating-point Blackwell (`sm_120`) CUDA kernel source for exact 64-bit Goldilocks
/// Prime Field NTT butterfly execution (`p = 2^64 - 2^32 + 1`, pure integer ALU).
pub const BLACKWELL_GOLDILOCKS_NTT_CUDA: &str = r#"
// g6me Exact 64-bit Goldilocks Prime Field NTT Kernel for Blackwell (sm_120)
// Zero floating-point instructions anywhere — 0.000000000000000000 ULP error!
#include <stdint.h>

static constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

__device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    unsigned __int128 s = (unsigned __int128)a + (unsigned __int128)b;
    return (s >= GOLDILOCKS_P) ? (uint64_t)(s - GOLDILOCKS_P) : (uint64_t)s;
}

__device__ __forceinline__ uint64_t gl_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (GOLDILOCKS_P - (b - a));
}

__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi_all = (uint64_t)(prod >> 64);
    uint64_t mid = hi_all & 0xFFFFFFFFULL;
    uint64_t hi  = hi_all >> 32;
    uint64_t t = lo - hi;
    if (lo < hi) t -= 0xFFFFFFFFULL;
    uint64_t mid_term = (mid << 32) - mid;
    uint64_t res = t + mid_term;
    if (res < t) {
        uint64_t r2 = res + 0xFFFFFFFFULL;
        res = (r2 < res) ? (r2 + 0xFFFFFFFFULL) : r2;
    }
    if (res >= GOLDILOCKS_P) res -= GOLDILOCKS_P;
    return res;
}

extern "C" __global__ void g6me_goldilocks_ntt_stage(
    uint64_t* __restrict__ buf,
    const uint64_t* __restrict__ twiddles,
    uint32_t n,
    uint32_t half
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (n >> 1)) return;
    uint32_t grp = tid / half;
    uint32_t k   = tid % half;
    uint32_t idx = grp * (half << 1) + k;
    uint64_t u = buf[idx];
    uint64_t v = gl_mul(buf[idx + half], twiddles[k]);
    buf[idx]        = gl_add(u, v);
    buf[idx + half] = gl_sub(u, v);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goldilocks_field_identities_and_primitive_root_exact() {
        // 2^64 ≡ 2^32 - 1 (mod p), 2^96 ≡ -1 (mod p)
        assert_eq!(reduce128(1u128 << 64), (1u64 << 32) - 1);
        assert_eq!(reduce128(1u128 << 96), GOLDILOCKS_P - 1);

        // Edge case: (p-1) * (p-1) = 1 mod p
        assert_eq!(mod_mul(GOLDILOCKS_P - 1, GOLDILOCKS_P - 1), 1);
        // Arbitrary large multiplication checked against u128 exact remainder
        let a = GOLDILOCKS_P - 123_456_789;
        let b = GOLDILOCKS_P - 987_654_321;
        let exact = (((a as u128) * (b as u128)) % (GOLDILOCKS_P as u128)) as u64;
        assert_eq!(mod_mul(a, b), exact);

        // Primitive 2^32-th root check
        assert_eq!(mod_pow(GOLDILOCKS_W32, 1u64 << 31), GOLDILOCKS_P - 1);
        assert_eq!(mod_pow(GOLDILOCKS_W32, 1u64 << 32), 1);

        // Multiplicative inverse check
        let inv7 = mod_inv(7);
        assert_eq!(mod_mul(7, inv7), 1);
    }

    #[test]
    fn ntt_forward_inverse_roundtrip_zero_error() {
        let n = 1024;
        let plan = NttPlan::new(n);
        let mut orig = Vec::with_capacity(n);
        for i in 0..n {
            let signed = ((i as i64 * 1_047_293) % 16_777_216) - 8_388_608;
            orig.push(i64_to_field(signed));
        }
        let mut work = orig.clone();
        plan.forward(&mut work);
        plan.inverse(&mut work);
        assert_eq!(work, orig, "NTT roundtrip must be bit-exact (0 ULP error)");
    }

    #[test]
    fn exact_int_convolve_matches_naive_integer_convolution_zero_error() {
        // 24-bit PCM samples × Q28 signed FIR taps
        let x: Vec<i64> = (0..317)
            .map(|i| (((i as i64 * 73_856_093) ^ (i as i64 * 19_349_663)) % 16_777_215) - 8_388_607)
            .collect();
        let h: Vec<i64> = (0..129)
            .map(|i| {
                let s = ((i as f64) * 0.17).sin() * ((i as f64) * -0.03).exp();
                (s * NTT_KERNEL_SCALE).round() as i64
            })
            .collect();

        let ntt_y = exact_int_convolve(&x, &h);
        let mut direct_y = vec![0i128; x.len() + h.len() - 1];
        for (i, &xv) in x.iter().enumerate() {
            for (j, &hv) in h.iter().enumerate() {
                direct_y[i + j] += (xv as i128) * (hv as i128);
            }
        }
        assert_eq!(
            ntt_y, direct_y,
            "Goldilocks NTT linear convolution must equal exact i128 direct convolution bit-for-bit!"
        );
    }

    #[test]
    fn overlap_add_ntt_convolver_matches_direct_convolution_and_streaming() {
        let ir: Vec<f64> = (0..96)
            .map(|i| {
                let t = i as f64;
                ((-t * 0.06).exp()) * (t * 0.23).cos() * 0.25
            })
            .collect();
        let x: Vec<f64> = (0..900)
            .map(|i| 0.7 * ((i as f64) * 0.07).sin() + 0.2 * ((i as f64) * 0.31).cos())
            .collect();

        let conv = NttOverlapAddConvolver::new(&ir);
        let y_slice = conv.convolve_slice(&x);

        // Compare against exact quantized integer direct convolution (0 ULP diff!)
        let x_q: Vec<i64> = x
            .iter()
            .map(|&v| (v * NTT_PCM_SCALE).round() as i64)
            .collect();
        let h_q: Vec<i64> = ir
            .iter()
            .map(|&v| (v * NTT_KERNEL_SCALE).round() as i64)
            .collect();
        let direct = exact_int_convolve(&x_q, &h_q);
        let inv_scale = 1.0 / (NTT_PCM_SCALE * NTT_KERNEL_SCALE);
        for i in 0..x.len() {
            let expected = (direct[i] as f64) * inv_scale;
            assert_eq!(
                y_slice[i].to_bits(),
                expected.to_bits(),
                "sample {i}: slice NTT != exact direct integer convolution"
            );
        }

        // Verify streaming sample-by-sample matches slice output after `latency()` frames!
        let mut stream = NttOverlapAddConvolver::new(&ir);
        let lat = stream.latency();
        let mut y_stream = Vec::with_capacity(x.len());
        for &s in &x {
            y_stream.push(stream.process_sample(s));
        }
        for _ in 0..lat {
            y_stream.push(stream.process_sample(0.0));
        }
        for i in 0..x.len() {
            assert_eq!(
                y_stream[i + lat].to_bits(),
                y_slice[i].to_bits(),
                "sample {i}: streaming NTT != slice NTT"
            );
        }
    }
}
