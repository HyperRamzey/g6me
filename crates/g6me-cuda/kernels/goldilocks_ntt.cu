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

