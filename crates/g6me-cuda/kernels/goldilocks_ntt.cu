// Exact 64-bit Goldilocks Prime (p = 2^64 - 2^32 + 1) NTT Butterfly & Pointwise Multiply.
// Zero floating-point operations -> zero FMA drift by construction (exact integer algebra).

#include <stdint.h>

static constexpr uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;
static constexpr uint64_t GOLDILOCKS_EPS = 0x00000000FFFFFFFFULL; // 2^32 - 1

__device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    bool carry = sum < a;
    if (carry || sum >= GOLDILOCKS_P) {
        sum -= GOLDILOCKS_P;
    }
    return sum;
}

__device__ __forceinline__ uint64_t gl_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + (GOLDILOCKS_P - b));
}

__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    // Reduce 128-bit (hi * 2^64 + lo) mod (2^64 - 2^32 + 1):
    // 2^64 === 2^32 - 1 (mod p), 2^96 === -1 (mod p).
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;
    uint64_t hi_hi = hi >> 32;
    uint64_t t1 = lo;
    uint64_t sub = hi_hi;
    uint64_t diff = (t1 >= sub) ? (t1 - sub) : (t1 - sub - GOLDILOCKS_EPS);
    uint64_t add_term = hi_lo * GOLDILOCKS_EPS;
    uint64_t res = diff + add_term;
    if (res < diff) {
        res += GOLDILOCKS_EPS;
    }
    if (res >= GOLDILOCKS_P) {
        res -= GOLDILOCKS_P;
    }
    return res;
}

extern "C" __global__ void g6me_goldilocks_ntt_stage(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    int n,
    int half_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n >> 1;
    if (tid >= total_pairs) return;

    int block = tid / half_len;
    int j = tid - block * half_len;
    int i0 = block * (half_len << 1) + j;
    int i1 = i0 + half_len;

    uint64_t u = data[i0];
    uint64_t v = gl_mul(data[i1], twiddles[j]);
    data[i0] = gl_add(u, v);
    data[i1] = gl_sub(u, v);
}

extern "C" __global__ void g6me_goldilocks_pointwise_mac(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ acc,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = gl_add(acc[i], gl_mul(a[i], b[i]));
}

extern "C" __global__ void g6me_goldilocks_bitrev(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int n,
    int log_n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = (log_n > 0) ? (__brev((uint32_t)i) >> (32 - log_n)) : 0u;
    dst[j] = src[i];
}

extern "C" __global__ void g6me_goldilocks_pointwise_mul(
    uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl_mul(a[i], b[i]);
}

extern "C" __global__ void g6me_goldilocks_scale(
    uint64_t* __restrict__ data,
    uint64_t inv_n,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = gl_mul(data[i], inv_n);
}

// ---------------------------------------------------------------------------
// Fused single-launch transform (docs/CUDA_PERF_PLAN.md section 1).
//
// One thread block holds the whole 2^log_n transform in dynamic shared memory
// (`2^log_n * 8` bytes, so 64 KiB at N = 8192). The host must opt the function
// into the dynamic shared-memory limit
// (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`) before the first launch.
//
// Operation order is exactly the multi-launch path's:
//   * bit reversal is a pure permutation (`dst[brev(i)] = src[i]`),
//   * stage `s` in ascending order uses the stage-`s` twiddle table (length
//     `1 << s`) at offset `(1 << s) - 1` in the concatenated table,
//   * each butterfly is `u = x[i0]`, `v = gl_mul(x[i1], tw[k])`,
//     `x[i0] = gl_add(u, v)`, `x[i1] = gl_sub(u, v)`.
// Every butterfly is independent within a stage, so shared-memory execution
// produces the identical canonical field element as the global-memory version.
// This file contains no floating-point instruction of any kind.
//
// `blockIdx.x` selects the independent transform in a contiguous
// `[batch][2^log_n]` buffer; `channel_mask` lets the caller skip inactive
// rows without changing the launch shape. `has_scale != 0` multiplies the
// result by `scale` (the inverse NTT's `n^-1 mod p`).
extern "C" __global__ void g6me_goldilocks_ntt_fused_inplace(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    int log_n,
    uint64_t scale,
    int has_scale,
    unsigned int channel_mask
) {
    if (channel_mask != 0xFFFFFFFFu && ((channel_mask >> blockIdx.x) & 1u) == 0u) return;
    const int n = 1 << log_n;
    uint64_t* x = data + (size_t)blockIdx.x * (size_t)n;
    extern __shared__ uint64_t smem[];
    const int tid = threadIdx.x;
    const int nthr = (int)blockDim.x;

    for (int i = tid; i < n; i += nthr) {
        uint32_t j = (log_n > 0) ? (__brev((uint32_t)i) >> (32 - log_n)) : 0u;
        smem[j] = x[i];
    }
    __syncthreads();

    for (int s = 0; s < log_n; ++s) {
        const int half = 1 << s;
        const uint64_t* tw = twiddles + (half - 1);
        for (int t = tid; t < (n >> 1); t += nthr) {
            const int grp = t >> s;
            const int k = t & (half - 1);
            const int i0 = (grp << (s + 1)) + k;
            const int i1 = i0 + half;
            const uint64_t u = smem[i0];
            const uint64_t v = gl_mul(smem[i1], tw[k]);
            smem[i0] = gl_add(u, v);
            smem[i1] = gl_sub(u, v);
        }
        __syncthreads();
    }

    if (has_scale) {
        for (int i = tid; i < n; i += nthr) smem[i] = gl_mul(smem[i], scale);
    }
    for (int i = tid; i < n; i += nthr) x[i] = smem[i];
}

// ---------------------------------------------------------------------------
// Device-resident FDL-UPOLA accumulation (docs/CUDA_PERF_PLAN.md section 1).
//
// `acc[0][k]` / `acc[1][k]` receive the exact field sum over the requested
// partitions and active channels, for one transform bin `k` per thread. The
// modular sum is exact and associative, so this is bit-identical to the CPU
// loop in `BinauralPlus::push_completed_upola_block` regardless of the subject
// order; a row of zeros contributes `gl_mul(0, k) = 0` exactly as the CPU's
// zero-row skip does.
//
// Partition 0 is the just-completed row stored at `head`; partitions
// `p >= 1` walk back in time through `[fdl_slots][slots][n]`.
extern "C" __global__ void g6me_goldilocks_upola_accumulate(
    const uint64_t* __restrict__ fdl,
    const uint64_t* __restrict__ kern_l,
    const uint64_t* __restrict__ kern_r,
    uint64_t* __restrict__ acc,
    int n,
    int slots,
    int fdl_slots,
    int n_parts,
    int p_start,
    int p_end,
    int head,
    unsigned int channel_mask
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    (void)n_parts;
    uint64_t al = 0;
    uint64_t ar = 0;
    for (int p = p_start; p < p_end; ++p) {
        const int slot = (p == 0) ? head : (head + fdl_slots - (p - 1)) % fdl_slots;
        for (int ch = 0; ch < slots; ++ch) {
            if (((channel_mask >> ch) & 1u) == 0u) continue;
            const size_t xoff = ((size_t)(slot * slots + ch)) * (size_t)n + (size_t)k;
            const size_t koff = ((size_t)(p * slots + ch)) * (size_t)n + (size_t)k;
            const uint64_t xv = fdl[xoff];
            al = gl_add(al, gl_mul(xv, kern_l[koff]));
            ar = gl_add(ar, gl_mul(xv, kern_r[koff]));
        }
    }
    acc[k] = al;
    acc[n + k] = ar;
}

// Partition-0 accumulation for the sub-block path, where the transform of the
// current hop lives in `[slots][n]` scratch instead of the delay line.
extern "C" __global__ void g6me_goldilocks_upola_mac_scratch(
    const uint64_t* __restrict__ scratch,
    const uint64_t* __restrict__ kern_l,
    const uint64_t* __restrict__ kern_r,
    uint64_t* __restrict__ acc,
    int n,
    int slots,
    unsigned int channel_mask
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint64_t al = 0;
    uint64_t ar = 0;
    for (int ch = 0; ch < slots; ++ch) {
        if (((channel_mask >> ch) & 1u) == 0u) continue;
        const size_t off = (size_t)ch * (size_t)n + (size_t)k;
        const uint64_t xv = scratch[off];
        al = gl_add(al, gl_mul(xv, kern_l[off]));
        ar = gl_add(ar, gl_mul(xv, kern_r[off]));
    }
    acc[k] = al;
    acc[n + k] = ar;
}

