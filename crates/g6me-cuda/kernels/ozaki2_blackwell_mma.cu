// Zero-FMA Multi-Architecture Ozaki-II CRT FP64-on-Tensor-Core Pipeline.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
//
// Full live device-side pipeline (zero CPU emulation, zero FMA contractions):
//   Stage 1: FP64 per-row / per-col IEEE-754 exponent extraction & 53-bit integer
//            modular reduction into 20 pairwise-coprime symmetric residues r in [-56, +56].
//   Stage 2: Device-side precision packing:
//            - Blackwell (sm_120a): Base-13 2-limb NVFP4 E2M1 nibbles + UE4M3 scale (2.0)
//            - Ada/Hopper/Blackwell (sm_89+): Base-16 2-limb FP8 E4M3 bytes
//            - Pascal..Blackwell (sm_61+): single-limb signed INT8 (4x4 / 32x32 tiled) & IEEE FP16 (sm_75+)
//   Stage 3: Shared-memory tiled hardware MMA / DP4A residue GEMM across all 20 moduli (blockIdx.z = mod):
//            - sm_120a: 4-warp 32x32 tiled mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4...ue4m3 (OMMA.SF)
//            - sm_89+:  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (QMMA)
//            - sm_80+:  4-warp 32x32 tiled mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (IMMA, desktop optimal!)
//            - sm_75:   4-warp 32x32 tiled mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 (IMMA, desktop optimal!)
//            - sm_61:   128-thread 32x32 tiled __dp4a (IDP.4A.S8.S8 -> s32, 4x FP32 / 128x FP64 on Pascal!)
//            - sm_80+:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (HMMA)
//            - sm_75:   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 (HMMA)
//            FP32/S32 accumulators are flushed to exact int64_t well before overflow (zero rounding
//            for arbitrary K), recombined in exact int64_t, and reduced mod m_k.
//   Stage 4: Exact 128-bit integer CRT reconstruction (unsigned __int128 mod M_BIG, 0 ULP
//            intermediate error, 0 FMA, 0 division) -> IEEE-754 f64 via __dmul_rn/__dadd_rn.

#include <stdint.h>

static constexpr int OZAKI2_NMOD = 20;

static constexpr uint64_t OZAKI2_M_BIG_LO  = 0x6377639e4e1aeac0ULL;
static constexpr uint64_t OZAKI2_M_BIG_HI  = 0x13dab74a538b2a1fULL;
static constexpr uint64_t OZAKI2_M_HALF_LO = 0xb1bbb1cf270d7560ULL;
static constexpr uint64_t OZAKI2_M_HALF_HI = 0x09ed5ba529c5950fULL;

// UE4M3 = 2.0 (0x40) packed 4x into a 32-bit scale register so hardware NVFP4 MMA computes
// (2.0 * e_a) * (2.0 * e_b) = d_a * d_b directly as exact integers in FP32.
static constexpr uint32_t NVFP4_UE4M3_SCALE_2_0_4X = 0x40404040u;

__constant__ int32_t OZAKI2_MODULI[20] = {
    113, 109, 107, 103, 101, 97, 89, 83, 81, 79,
     73,  71,  67,  64,  61, 59, 53, 49, 47, 43
};

__constant__ uint32_t OZAKI2_CRT_INV[20] = {
    21u, 87u,  6u, 59u, 22u, 87u, 41u, 69u, 38u,  9u,
    13u, 34u, 65u,  3u,  5u, 16u,  8u, 33u, 18u,  7u
};

__constant__ uint64_t OZAKI2_MDIV_LO[20] = {
    0xe36dccc67f8ef6c0ULL, 0xb5c1aeb64bdf5dc0ULL, 0x97a897a8f4c47040ULL, 0xea98c55112280740ULL,
    0xf44fc19eb744b3c0ULL, 0xbc68289d57e5e2c0ULL, 0x1820fb5dd2dae8c0ULL, 0xba4235a24af71240ULL,
    0x600af4967f624ec0ULL, 0xf10e7928e3d2f940ULL, 0xb0b47c1ab76d14c0ULL, 0x0166a3a816bbdf40ULL,
    0x9e242b8445f11e40ULL, 0x7d8ddd8e79386babULL, 0xdbdc2fcc09ac81c0ULL, 0x2d136125651ed440ULL,
    0x979cd193e47e17c0ULL, 0xa936ae70f1eba6c0ULL, 0x021dc63463b45140ULL, 0x854a37e5ea00a040ULL
};

__constant__ uint64_t OZAKI2_MDIV_HI[20] = {
    0x002cfad35917e2ebULL, 0x002ea16352f7e1daULL, 0x002f8084488e75a0ULL, 0x003158c512358b97ULL,
    0x003252ec756bd553ULL, 0x0034662da1d9d8d8ULL, 0x00391bf273feafefULL, 0x003d3ccf8bcd3e31ULL,
    0x003ebfe4792152b1ULL, 0x00405692c37ce69bULL, 0x0045a051acfae485ULL, 0x004796699c380b69ULL,
    0x004bdc86d73791d2ULL, 0x004f6add294e2ca8ULL, 0x005352be12280d47ULL, 0x005625d187eca9efULL,
    0x005fe67a2837c200ULL, 0x0067ba9906edf12fULL, 0x006c2494a52e8911ULL, 0x007633e3f609309bULL
};

// Hayashi et al. (arXiv:2608.06812) Base-13 E2M1 nibble tables indexed by (r + 56), r in [-56, +56].
// r = (2*e0) + 13*(2*e1), where e0, e1 are exact FP4 E2M1 values in {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4}.
__constant__ uint8_t BASE13_E2M1_LO[113] = {
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4,0xe,0x5,0xd,0x6,
    0xc,0xb,0xa,0x9,0x0,0x1,0x2,0x3,0x4
};

__constant__ uint8_t BASE13_E2M1_HI[113] = {
    0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xb,0xc,0xb,0xc,
    0xb,0xb,0xb,0xb,0xb,0xb,0xb,0xb,0xb,0xa,0xb,0xa,0xb,
    0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0xa,0x9,0xa,0x9,0xa,
    0x9,0x9,0x9,0x9,0x9,0x9,0x9,0x9,0x9,0x0,0x9,0x0,0x9,
    0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x0,0x1,0x0,
    0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x2,0x1,0x2,0x1,
    0x2,0x2,0x2,0x2,0x2,0x2,0x2,0x2,0x2,0x3,0x2,0x3,0x2,
    0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x4,0x3,
    0x4,0x4,0x4,0x4,0x4,0x4,0x4,0x4,0x4
};

// Base-16 FP8 E4M3 byte tables indexed by (r + 56), r = lo + 16*hi with lo in [-8,+8], hi in [-4,+4].
__constant__ uint8_t BASE16_FP8_LO[113] = {
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0,0xce,0xcc,0xca,0xc8,0xc4,0xc0,0xb8,0x00,0x38,0x40,0x44,0x48,0x4a,0x4c,0x4e,
    0xd0
};

__constant__ uint8_t BASE16_FP8_HI[113] = {
    0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,0xc4,
    0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,0xc0,
    0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,0xb8,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,0x38,
    0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,0x40,
    0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,0x44,
    0x48
};

// Exact IEEE binary16 bit patterns for integers r in [-56, +56], indexed by (r + 56).
__constant__ uint16_t FP16_RESIDUE_LUT[113] = {
    0xd300,0xd2e0,0xd2c0,0xd2a0,0xd280,0xd260,0xd240,0xd220,0xd200,0xd1e0,0xd1c0,0xd1a0,0xd180,
    0xd160,0xd140,0xd120,0xd100,0xd0e0,0xd0c0,0xd0a0,0xd080,0xd060,0xd040,0xd020,0xd000,0xcfc0,
    0xcf80,0xcf40,0xcf00,0xcec0,0xce80,0xce40,0xce00,0xcdc0,0xcd80,0xcd40,0xcd00,0xccc0,0xcc80,
    0xcc40,0xcc00,0xcb80,0xcb00,0xca80,0xca00,0xc980,0xc900,0xc880,0xc800,0xc700,0xc600,0xc500,
    0xc400,0xc200,0xc000,0xbc00,0x0000,0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600,0x4700,0x4800,
    0x4880,0x4900,0x4980,0x4a00,0x4a80,0x4b00,0x4b80,0x4c00,0x4c40,0x4c80,0x4cc0,0x4d00,0x4d40,
    0x4d80,0x4dc0,0x4e00,0x4e40,0x4e80,0x4ec0,0x4f00,0x4f40,0x4f80,0x4fc0,0x5000,0x5020,0x5040,
    0x5060,0x5080,0x50a0,0x50c0,0x50e0,0x5100,0x5120,0x5140,0x5160,0x5180,0x51a0,0x51c0,0x51e0,
    0x5200,0x5220,0x5240,0x5260,0x5280,0x52a0,0x52c0,0x52e0,0x5300
};

// ============================================================================
// Zero-FMA IEEE-754 power-of-two scaling & symmetric modular reduction helpers
// ============================================================================

__device__ __forceinline__ int32_t frexp_exp_f64(double x) {
    uint64_t bits = (uint64_t)__double_as_longlong(x) & 0x7FFFFFFFFFFFFFFFULL;
    if (bits == 0ULL) return 0;
    int32_t raw_e = (int32_t)(bits >> 52);
    if (raw_e != 0) {
        return raw_e - 1022; // matches frexp(): x in [0.5, 1.0) * 2^exp
    }
    // Subnormal: normalize by 2^54 (zero FMA)
    double norm = __dmul_rn(__longlong_as_double((long long)bits), 18014398509481984.0);
    uint64_t nbits = (uint64_t)__double_as_longlong(norm) & 0x7FFFFFFFFFFFFFFFULL;
    return ((int32_t)(nbits >> 52) - 1022) - 54;
}

__device__ __forceinline__ double pow2_f64(int32_t exp) {
    // Valid for exp in [-1022, 1023]
    uint64_t bits = (uint64_t)(exp + 1023) << 52;
    return __longlong_as_double((long long)bits);
}

__device__ __forceinline__ double ldexp_zero_fma(double x, int32_t exp) {
    if (x == 0.0 || exp == 0) return x;
    while (exp > 1000) {
        x = __dmul_rn(x, pow2_f64(1000));
        exp -= 1000;
    }
    while (exp < -1000) {
        x = __dmul_rn(x, pow2_f64(-1000));
        exp += 1000;
    }
    return __dmul_rn(x, pow2_f64(exp));
}

__device__ __forceinline__ int8_t sym_mod_i64(int64_t v, int32_t m) {
    int64_t r = v % (int64_t)m;
    if (r < 0) r += (int64_t)m;
    if (r > (int64_t)(m >> 1)) r -= (int64_t)m;
    return (int8_t)r;
}

__device__ __forceinline__ int32_t pos_mod_i64(int64_t v, int32_t m) {
    int64_t r = v % (int64_t)m;
    if (r < 0) r += (int64_t)m;
    return (int32_t)r;
}

// ============================================================================
// Stage 1: Device-side FP64 Scaling & 20-Moduli Symmetric Reduction
// ============================================================================

// Computes per-row exponent exp_a[row] and reduces row-major A[M * K] into
// a_sym[mod_idx * M * K + row * K + k] in [-56, +56].
extern "C" __global__ void g6me_ozaki2_scale_reduce_a_f64(
    const double* __restrict__ a_f64, // [M * K] row-major
    int32_t* __restrict__ exp_a,      // [M] (0x7fffffff marks zero row)
    int8_t* __restrict__ a_sym,       // [20 * M * K]
    int m,
    int k_dim,
    int shift_bits
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    const double* a_row = a_f64 + (size_t)row * (size_t)k_dim;
    double mx = 0.0;
    for (int k = 0; k < k_dim; ++k) {
        double v = fabs(a_row[k]);
        if (v > mx) mx = v;
    }

    size_t mat_stride = (size_t)m * (size_t)k_dim;
    size_t row_off = (size_t)row * (size_t)k_dim;
    if (mx == 0.0) {
        exp_a[row] = 0x7fffffff;
        for (int mi = 0; mi < OZAKI2_NMOD; ++mi) {
            int8_t* dst = a_sym + (size_t)mi * mat_stride + row_off;
            for (int k = 0; k < k_dim; ++k) dst[k] = 0;
        }
        return;
    }

    int32_t ea = frexp_exp_f64(mx);
    exp_a[row] = ea;
    int32_t scale_exp = shift_bits - ea;

    for (int k = 0; k < k_dim; ++k) {
        double scaled = ldexp_zero_fma(a_row[k], scale_exp);
        int64_t q = llrint(scaled);
        for (int mi = 0; mi < OZAKI2_NMOD; ++mi) {
            a_sym[(size_t)mi * mat_stride + row_off + (size_t)k] = sym_mod_i64(q, OZAKI2_MODULI[mi]);
        }
    }
}

// Computes per-col exponent exp_b[col] and reduces row-major B[K * N] into
// TRANSPOSED b_sym[mod_idx * N * K + col * K + k] in [-56, +56] (ready for .row.col MMA).
extern "C" __global__ void g6me_ozaki2_scale_reduce_b_f64(
    const double* __restrict__ b_f64, // [K * N] row-major
    int32_t* __restrict__ exp_b,      // [N] (0x7fffffff marks zero col)
    int8_t* __restrict__ b_sym,       // [20 * N * K] transposed [N, K] per modulus
    int k_dim,
    int n,
    int shift_bits
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    double mx = 0.0;
    for (int k = 0; k < k_dim; ++k) {
        double v = fabs(b_f64[(size_t)k * (size_t)n + (size_t)col]);
        if (v > mx) mx = v;
    }

    size_t mat_stride = (size_t)n * (size_t)k_dim;
    size_t col_off = (size_t)col * (size_t)k_dim;
    if (mx == 0.0) {
        exp_b[col] = 0x7fffffff;
        for (int mi = 0; mi < OZAKI2_NMOD; ++mi) {
            int8_t* dst = b_sym + (size_t)mi * mat_stride + col_off;
            for (int k = 0; k < k_dim; ++k) dst[k] = 0;
        }
        return;
    }

    int32_t eb = frexp_exp_f64(mx);
    exp_b[col] = eb;
    int32_t scale_exp = shift_bits - eb;

    for (int k = 0; k < k_dim; ++k) {
        double scaled = ldexp_zero_fma(b_f64[(size_t)k * (size_t)n + (size_t)col], scale_exp);
        int64_t q = llrint(scaled);
        for (int mi = 0; mi < OZAKI2_NMOD; ++mi) {
            b_sym[(size_t)mi * mat_stride + col_off + (size_t)k] = sym_mod_i64(q, OZAKI2_MODULI[mi]);
        }
    }
}

// ============================================================================
// Stage 2: Device-side Precision Packing Kernels (Batched across all 20 moduli)
// ============================================================================

// Packs symmetric residues [20 * rows * K] into Base-13 2-limb NVFP4 E2M1 uint32 words
// (8 nibbles per uint32) with padded shape [20 * rows_pad * (k_pad / 8)], plus UE4M3
// scale factors (2.0 = 0x40404040u per 64-element K-tile).
extern "C" __global__ void g6me_ozaki2_pack_nvfp4_base13(
    const int8_t* __restrict__ sym, // [20 * rows * k_dim]
    uint32_t* __restrict__ lo_fp4,  // [20 * rows_pad * (k_pad / 8)]
    uint32_t* __restrict__ hi_fp4,  // [20 * rows_pad * (k_pad / 8)]
    uint32_t* __restrict__ sf_buf,  // [20 * rows_pad * (k_pad / 64)]
    int rows,
    int k_dim,
    int rows_pad,
    int k_pad
) {
    int k_word = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. (k_pad / 8) - 1
    int row    = blockIdx.y * blockDim.y + threadIdx.y; // 0 .. rows_pad - 1
    int mi     = blockIdx.z;                            // 0 .. 19
    int k_words = k_pad >> 3;
    if (row >= rows_pad || k_word >= k_words || mi >= OZAKI2_NMOD) return;

    uint32_t w_lo = 0u;
    uint32_t w_hi = 0u;
    if (row < rows) {
        const int8_t* src_row = sym + ((size_t)mi * (size_t)rows + (size_t)row) * (size_t)k_dim;
        int k_base = k_word << 3;
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            int k = k_base + s;
            if (k < k_dim) {
                int idx = (int)src_row[k] + 56;
                uint32_t n0 = (uint32_t)BASE13_E2M1_LO[idx];
                uint32_t n1 = (uint32_t)BASE13_E2M1_HI[idx];
                w_lo |= (n0 << (s * 4));
                w_hi |= (n1 << (s * 4));
            }
        }
    }
    size_t dst_idx = ((size_t)mi * (size_t)rows_pad + (size_t)row) * (size_t)k_words + (size_t)k_word;
    lo_fp4[dst_idx] = w_lo;
    hi_fp4[dst_idx] = w_hi;

    if ((k_word & 7) == 0 && sf_buf != 0) {
        int k_tiles = k_pad >> 6;
        size_t sf_idx = ((size_t)mi * (size_t)rows_pad + (size_t)row) * (size_t)k_tiles + (size_t)(k_word >> 3);
        sf_buf[sf_idx] = NVFP4_UE4M3_SCALE_2_0_4X;
    }
}

// Packs symmetric residues [20 * rows * K] into Base-16 2-limb FP8 E4M3 uint32 words
// (4 bytes per uint32) with padded shape [20 * rows_pad * (k_pad / 4)].
extern "C" __global__ void g6me_ozaki2_pack_fp8_base16(
    const int8_t* __restrict__ sym,
    uint32_t* __restrict__ lo_fp8,
    uint32_t* __restrict__ hi_fp8,
    int rows,
    int k_dim,
    int rows_pad,
    int k_pad
) {
    int k_word = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. (k_pad / 4) - 1
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int mi     = blockIdx.z;
    int k_words = k_pad >> 2;
    if (row >= rows_pad || k_word >= k_words || mi >= OZAKI2_NMOD) return;

    uint32_t w_lo = 0u;
    uint32_t w_hi = 0u;
    if (row < rows) {
        const int8_t* src_row = sym + ((size_t)mi * (size_t)rows + (size_t)row) * (size_t)k_dim;
        int k_base = k_word << 2;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int k = k_base + s;
            if (k < k_dim) {
                int idx = (int)src_row[k] + 56;
                w_lo |= ((uint32_t)BASE16_FP8_LO[idx] << (s * 8));
                w_hi |= ((uint32_t)BASE16_FP8_HI[idx] << (s * 8));
            }
        }
    }
    size_t dst_idx = ((size_t)mi * (size_t)rows_pad + (size_t)row) * (size_t)k_words + (size_t)k_word;
    lo_fp8[dst_idx] = w_lo;
    hi_fp8[dst_idx] = w_hi;
}

// Packs symmetric residues [20 * rows * K] into signed INT8 uint32 words (4 bytes/word).
extern "C" __global__ void g6me_ozaki2_pack_int8(
    const int8_t* __restrict__ sym,
    uint32_t* __restrict__ out_i8,
    int rows,
    int k_dim,
    int rows_pad,
    int k_pad
) {
    int k_word = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int mi     = blockIdx.z;
    int k_words = k_pad >> 2;
    if (row >= rows_pad || k_word >= k_words || mi >= OZAKI2_NMOD) return;

    uint32_t w = 0u;
    if (row < rows) {
        const int8_t* src_row = sym + ((size_t)mi * (size_t)rows + (size_t)row) * (size_t)k_dim;
        int k_base = k_word << 2;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int k = k_base + s;
            if (k < k_dim) {
                uint32_t byte_val = (uint32_t)(uint8_t)src_row[k];
                w |= (byte_val << (s * 8));
            }
        }
    }
    size_t dst_idx = ((size_t)mi * (size_t)rows_pad + (size_t)row) * (size_t)k_words + (size_t)k_word;
    out_i8[dst_idx] = w;
}

// Packs symmetric residues [20 * rows * K] into exact IEEE FP16 uint32 words (2 halfs/word).
extern "C" __global__ void g6me_ozaki2_pack_fp16(
    const int8_t* __restrict__ sym,
    uint32_t* __restrict__ out_f16,
    int rows,
    int k_dim,
    int rows_pad,
    int k_pad
) {
    int k_word = blockIdx.x * blockDim.x + threadIdx.x; // 0 .. (k_pad / 2) - 1
    int row    = blockIdx.y * blockDim.y + threadIdx.y;
    int mi     = blockIdx.z;
    int k_words = k_pad >> 1;
    if (row >= rows_pad || k_word >= k_words || mi >= OZAKI2_NMOD) return;

    uint32_t w = 0u;
    if (row < rows) {
        const int8_t* src_row = sym + ((size_t)mi * (size_t)rows + (size_t)row) * (size_t)k_dim;
        int k_base = k_word << 1;
        if (k_base < k_dim) {
            w |= (uint32_t)FP16_RESIDUE_LUT[(int)src_row[k_base] + 56];
        }
        if (k_base + 1 < k_dim) {
            w |= ((uint32_t)FP16_RESIDUE_LUT[(int)src_row[k_base + 1] + 56] << 16);
        }
    }
    size_t dst_idx = ((size_t)mi * (size_t)rows_pad + (size_t)row) * (size_t)k_words + (size_t)k_word;
    out_f16[dst_idx] = w;
}

// ============================================================================
// Stage 3: Hardware Tensor Core MMA Wrappers & Live Residue GEMM Kernels
// ============================================================================

// --- 3A. Blackwell sm_120a NVFP4 E2M1 Block-Scaled MMA (m16n8k64, scale_vec::4X, ue4m3) ---
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_SPECIFIC__))
__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0)
    );
}
#endif

// Launch: blockDim = (128, 1, 1) [4 warps per 32x32 shared-memory CTA tile],
//         gridDim  = (ceil(n/32), ceil(m/32), 20).
extern "C" __global__ void g6me_ozaki2_nvfp4_mma_residue_gemm(
    const uint32_t* __restrict__ a_lo, // [20 * m_pad * (k_pad / 8)]
    const uint32_t* __restrict__ a_hi,
    const uint32_t* __restrict__ b_lo, // [20 * n_pad * (k_pad / 8)] (transposed B)
    const uint32_t* __restrict__ b_hi,
    const uint32_t* __restrict__ sfa_buf,
    const uint32_t* __restrict__ sfb_buf,
    int32_t* __restrict__ c_mod,       // [20 * m * n]
    int m,
    int n,
    int m_pad,
    int n_pad,
    int k_pad
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_SPECIFIC__))
    int mi = blockIdx.z;
    if (mi >= OZAKI2_NMOD) return;
    int tile_m = blockIdx.y * 32;
    int tile_n = blockIdx.x * 32;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;   // 0..3 (2x2 warps covering 32x32)
    int lane = tid & 31;
    int group_id = lane >> 2; // 0..7
    int tidg = lane & 3;      // 0..3
    int wm = (warp_id >> 1) * 16; // 0 or 16
    int wn = (warp_id & 1) * 16;  // 0 or 16

    int k_words = k_pad >> 3; // uint32 words per row (8 nibbles/word)
    int k_tiles = k_pad >> 6; // 64 elements = 8 uint32 words per K-tile
    int32_t modulus = OZAKI2_MODULI[mi];

    const uint32_t* a0_base = a_lo + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* a1_base = a_hi + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* b0_base = b_lo + (size_t)mi * (size_t)n_pad * (size_t)k_words;
    const uint32_t* b1_base = b_hi + (size_t)mi * (size_t)n_pad * (size_t)k_words;
    const uint32_t* sfa_base = sfa_buf + (size_t)mi * (size_t)m_pad * (size_t)k_tiles;
    const uint32_t* sfb_base = sfb_buf + (size_t)mi * (size_t)n_pad * (size_t)k_tiles;

    // Bank-conflict-free shared-memory tiles for 32x64 K-slices (only ~4.8 KB per CTA)
    __shared__ uint32_t sm_a0[32][9];
    __shared__ uint32_t sm_a1[32][9];
    __shared__ uint32_t sm_b0[32][9];
    __shared__ uint32_t sm_b1[32][9];
    __shared__ uint32_t sm_sfa[32];
    __shared__ uint32_t sm_sfb[32];

    int64_t t00[2][4] = {{0,0,0,0},{0,0,0,0}};
    int64_t t01[2][4] = {{0,0,0,0},{0,0,0,0}};
    int64_t t10[2][4] = {{0,0,0,0},{0,0,0,0}};
    int64_t t11[2][4] = {{0,0,0,0},{0,0,0,0}};

    // Flush FP32 MMA accumulators into int64 every 256 K-tiles (K = 16,384; max |sum| <= 1,048,576 << 2^24).
    for (int kt_outer = 0; kt_outer < k_tiles; kt_outer += 256) {
        int kt_end = kt_outer + 256;
        if (kt_end > k_tiles) kt_end = k_tiles;

        float p00[2][4] = {{0.0f,0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f,0.0f}};
        float p01[2][4] = {{0.0f,0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f,0.0f}};
        float p10[2][4] = {{0.0f,0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f,0.0f}};
        float p11[2][4] = {{0.0f,0.0f,0.0f,0.0f},{0.0f,0.0f,0.0f,0.0f}};

        for (int kt = kt_outer; kt < kt_end; ++kt) {
            // Cooperative 128-thread coalesced tile load: 256 uint32 words per matrix
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int idx = tid + u * 128;
                int r_s = idx >> 3; // 0..31
                int w_s = idx & 7;  // 0..7
                size_t a_off = (size_t)(tile_m + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s);
                size_t b_off = (size_t)(tile_n + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s);
                sm_a0[r_s][w_s] = a0_base[a_off];
                sm_a1[r_s][w_s] = a1_base[a_off];
                sm_b0[r_s][w_s] = b0_base[b_off];
                sm_b1[r_s][w_s] = b1_base[b_off];
            }
            if (tid < 32) {
                sm_sfa[tid] = sfa_base[(size_t)(tile_m + tid) * (size_t)k_tiles + (size_t)kt];
                sm_sfb[tid] = sfb_base[(size_t)(tile_n + tid) * (size_t)k_tiles + (size_t)kt];
            }
            __syncthreads();

            // Load A fragments once and reuse across both 8-column N-subtiles
            int ar0 = wm + group_id;
            int ar1 = ar0 + 8;
            uint32_t a0_r0 = sm_a0[ar0][tidg];
            uint32_t a0_r1 = sm_a0[ar1][tidg];
            uint32_t a0_r2 = sm_a0[ar0][tidg + 4];
            uint32_t a0_r3 = sm_a0[ar1][tidg + 4];
            uint32_t a1_r0 = sm_a1[ar0][tidg];
            uint32_t a1_r1 = sm_a1[ar1][tidg];
            uint32_t a1_r2 = sm_a1[ar0][tidg + 4];
            uint32_t a1_r3 = sm_a1[ar1][tidg + 4];
            uint32_t sfa   = sm_sfa[ar0];

            #pragma unroll
            for (int ns = 0; ns < 2; ++ns) {
                int bc = wn + ns * 8 + group_id;
                uint32_t b0_r0 = sm_b0[bc][tidg];
                uint32_t b0_r1 = sm_b0[bc][tidg + 4];
                uint32_t b1_r0 = sm_b1[bc][tidg];
                uint32_t b1_r1 = sm_b1[bc][tidg + 4];
                uint32_t sfb   = sm_sfb[bc];

                mma_nvfp4_m16n8k64(p00[ns][0], p00[ns][1], p00[ns][2], p00[ns][3],
                                   a0_r0, a0_r1, a0_r2, a0_r3, b0_r0, b0_r1,
                                   p00[ns][0], p00[ns][1], p00[ns][2], p00[ns][3], sfa, sfb);
                mma_nvfp4_m16n8k64(p01[ns][0], p01[ns][1], p01[ns][2], p01[ns][3],
                                   a0_r0, a0_r1, a0_r2, a0_r3, b1_r0, b1_r1,
                                   p01[ns][0], p01[ns][1], p01[ns][2], p01[ns][3], sfa, sfb);
                mma_nvfp4_m16n8k64(p10[ns][0], p10[ns][1], p10[ns][2], p10[ns][3],
                                   a1_r0, a1_r1, a1_r2, a1_r3, b0_r0, b0_r1,
                                   p10[ns][0], p10[ns][1], p10[ns][2], p10[ns][3], sfa, sfb);
                mma_nvfp4_m16n8k64(p11[ns][0], p11[ns][1], p11[ns][2], p11[ns][3],
                                   a1_r0, a1_r1, a1_r2, a1_r3, b1_r0, b1_r1,
                                   p11[ns][0], p11[ns][1], p11[ns][2], p11[ns][3], sfa, sfb);
            }
            __syncthreads();
        }

        #pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                t00[ns][v] += llrintf(p00[ns][v]);
                t01[ns][v] += llrintf(p01[ns][v]);
                t10[ns][v] += llrintf(p10[ns][v]);
                t11[ns][v] += llrintf(p11[ns][v]);
            }
        }
    }

    int32_t* c_plane = c_mod + (size_t)mi * (size_t)m * (size_t)n;
    #pragma unroll
    for (int ns = 0; ns < 2; ++ns) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            int out_r = tile_m + wm + group_id + 8 * (v >> 1);
            int out_c = tile_n + wn + ns * 8 + 2 * tidg + (v & 1);
            if (out_r < m && out_c < n) {
                int64_t total = t00[ns][v] + 13LL * (t01[ns][v] + t10[ns][v]) + 169LL * t11[ns][v];
                c_plane[(size_t)out_r * (size_t)n + (size_t)out_c] = pos_mod_i64(total, modulus);
            }
        }
    }
#else
    (void)a_lo; (void)a_hi; (void)b_lo; (void)b_hi;
    (void)sfa_buf; (void)sfb_buf; (void)c_mod;
    (void)m; (void)n; (void)m_pad; (void)n_pad; (void)k_pad;
#endif
}

// --- 3B. Ada / Hopper / Blackwell (sm_89+) FP8 E4M3 Base-16 MMA (m16n8k32) ---
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
__device__ __forceinline__ void mma_fp8_m16n8k32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}
#endif

extern "C" __global__ void g6me_ozaki2_fp8_mma_residue_gemm(
    const uint32_t* __restrict__ a_lo, // [20 * m_pad * (k_pad / 4)]
    const uint32_t* __restrict__ a_hi,
    const uint32_t* __restrict__ b_lo, // [20 * n_pad * (k_pad / 4)]
    const uint32_t* __restrict__ b_hi,
    int32_t* __restrict__ c_mod,       // [20 * m * n]
    int m,
    int n,
    int m_pad,
    int n_pad,
    int k_pad
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    int mi = blockIdx.z;
    if (mi >= OZAKI2_NMOD) return;
    int tile_m = blockIdx.y * 16;
    int tile_n = blockIdx.x * 8;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tidg = lane & 3;

    int k_words = k_pad >> 2; // 4 bytes per uint32
    int k_tiles = k_pad >> 5; // 32 elements = 8 uint32 words per K-tile
    int32_t modulus = OZAKI2_MODULI[mi];

    const uint32_t* a0_base = a_lo + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* a1_base = a_hi + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* b0_base = b_lo + (size_t)mi * (size_t)n_pad * (size_t)k_words;
    const uint32_t* b1_base = b_hi + (size_t)mi * (size_t)n_pad * (size_t)k_words;

    int row0 = tile_m + group_id;
    int row1 = row0 + 8;
    int col_b = tile_n + group_id;

    int64_t t00[4] = {0, 0, 0, 0};
    int64_t t01[4] = {0, 0, 0, 0};
    int64_t t10[4] = {0, 0, 0, 0};
    int64_t t11[4] = {0, 0, 0, 0};

    // Flush every 512 K-tiles (K = 16,384; max |sum| <= 1,048,576 << 2^24).
    for (int kt_outer = 0; kt_outer < k_tiles; kt_outer += 512) {
        int kt_end = kt_outer + 512;
        if (kt_end > k_tiles) kt_end = k_tiles;

        float p00[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float p01[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float p10[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float p11[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int kt = kt_outer; kt < kt_end; ++kt) {
            int kw0 = kt * 8 + tidg;
            int kw1 = kw0 + 4;
            uint32_t a0_r0 = a0_base[(size_t)row0 * (size_t)k_words + (size_t)kw0];
            uint32_t a0_r1 = a0_base[(size_t)row1 * (size_t)k_words + (size_t)kw0];
            uint32_t a0_r2 = a0_base[(size_t)row0 * (size_t)k_words + (size_t)kw1];
            uint32_t a0_r3 = a0_base[(size_t)row1 * (size_t)k_words + (size_t)kw1];

            uint32_t a1_r0 = a1_base[(size_t)row0 * (size_t)k_words + (size_t)kw0];
            uint32_t a1_r1 = a1_base[(size_t)row1 * (size_t)k_words + (size_t)kw0];
            uint32_t a1_r2 = a1_base[(size_t)row0 * (size_t)k_words + (size_t)kw1];
            uint32_t a1_r3 = a1_base[(size_t)row1 * (size_t)k_words + (size_t)kw1];

            uint32_t b0_r0 = b0_base[(size_t)col_b * (size_t)k_words + (size_t)kw0];
            uint32_t b0_r1 = b0_base[(size_t)col_b * (size_t)k_words + (size_t)kw1];

            uint32_t b1_r0 = b1_base[(size_t)col_b * (size_t)k_words + (size_t)kw0];
            uint32_t b1_r1 = b1_base[(size_t)col_b * (size_t)k_words + (size_t)kw1];

            mma_fp8_m16n8k32(p00[0], p00[1], p00[2], p00[3],
                             a0_r0, a0_r1, a0_r2, a0_r3, b0_r0, b0_r1,
                             p00[0], p00[1], p00[2], p00[3]);
            mma_fp8_m16n8k32(p01[0], p01[1], p01[2], p01[3],
                             a0_r0, a0_r1, a0_r2, a0_r3, b1_r0, b1_r1,
                             p01[0], p01[1], p01[2], p01[3]);
            mma_fp8_m16n8k32(p10[0], p10[1], p10[2], p10[3],
                             a1_r0, a1_r1, a1_r2, a1_r3, b0_r0, b0_r1,
                             p10[0], p10[1], p10[2], p10[3]);
            mma_fp8_m16n8k32(p11[0], p11[1], p11[2], p11[3],
                             a1_r0, a1_r1, a1_r2, a1_r3, b1_r0, b1_r1,
                             p11[0], p11[1], p11[2], p11[3]);
        }

        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            t00[v] += llrintf(p00[v]);
            t01[v] += llrintf(p01[v]);
            t10[v] += llrintf(p10[v]);
            t11[v] += llrintf(p11[v]);
        }
    }

    int32_t* c_plane = c_mod + (size_t)mi * (size_t)m * (size_t)n;
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int out_r = tile_m + group_id + 8 * (v >> 1);
        int out_c = tile_n + 2 * tidg + (v & 1);
        if (out_r < m && out_c < n) {
            int64_t total = t00[v] + 16LL * (t01[v] + t10[v]) + 256LL * t11[v];
            c_plane[(size_t)out_r * (size_t)n + (size_t)out_c] = pos_mod_i64(total, modulus);
        }
    }
#else
    (void)a_lo; (void)a_hi; (void)b_lo; (void)b_hi;
    (void)c_mod; (void)m; (void)n; (void)m_pad; (void)n_pad; (void)k_pad;
#endif
}

// --- 3C. Ampere+ (sm_80+: m16n8k32) & Turing (sm_75: m8n8k16) INT8 Tensor Core MMA ---
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void mma_int8_m16n8k32(
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
__device__ __forceinline__ void mma_int8_m8n8k16(
    int32_t& d0, int32_t& d1,
    uint32_t a0, uint32_t b0,
    int32_t c0, int32_t c1
) {
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(b0), "r"(c0), "r"(c1)
    );
}
#endif

// Launch: blockDim = (128, 1, 1) [4 warps / 128 threads per 32x32 shared-memory CTA tile],
//         gridDim  = (ceil(n/32), ceil(m/32), 20).
// Supports ALL desktop generations from Pascal sm_61 (__dp4a / IDP.4A.S8.S8) through
// Turing sm_75 (IMMA.8816), Ampere sm_80/86, Ada sm_89, Hopper sm_90, and Blackwell sm_120a (IMMA.16832).
extern "C" __global__ void g6me_ozaki2_int8_mma_residue_gemm(
    const uint32_t* __restrict__ a_i8, // [20 * m_pad * (k_pad / 4)]
    const uint32_t* __restrict__ b_i8, // [20 * n_pad * (k_pad / 4)]
    int32_t* __restrict__ c_mod,       // [20 * m * n]
    int m,
    int n,
    int m_pad,
    int n_pad,
    int k_pad
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    int mi = blockIdx.z;
    if (mi >= OZAKI2_NMOD) return;
    int tile_m = blockIdx.y * 32;
    int tile_n = blockIdx.x * 32;
    int tid = threadIdx.x;

    int k_words = k_pad >> 2;
    int k_tiles = k_pad >> 5; // 32 INT8 elements = 8 uint32 words per K-tile
    int32_t modulus = OZAKI2_MODULI[mi];

    const uint32_t* a_base = a_i8 + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* b_base = b_i8 + (size_t)mi * (size_t)n_pad * (size_t)k_words;

    // Bank-conflict-free shared-memory tiles: 32 rows x 8 uint32 words (only 2.3 KB per CTA)
    __shared__ uint32_t sm_a[32][9];
    __shared__ uint32_t sm_b[32][9];

#if (__CUDA_ARCH__ >= 750)
    // Tensor Core path (Turing sm_75 IMMA.8816, Ampere/Ada/Hopper/Blackwell sm_80+ IMMA.16832):
    // 4 warps arranged 2x2 over 32x32; each warp computes 16x16 (2 N-subtiles of 8).
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int group_id = lane >> 2;
    int tidg = lane & 3;
    int wm = (warp_id >> 1) * 16;
    int wn = (warp_id & 1) * 16;

    int64_t tot[2][4] = {{0,0,0,0},{0,0,0,0}};

    // Flush s32 accumulator into int64 every 1024 K-tiles (K = 32,768; max ~1.03e8 << 2^31).
    for (int kt_outer = 0; kt_outer < k_tiles; kt_outer += 1024) {
        int kt_end = kt_outer + 1024;
        if (kt_end > k_tiles) kt_end = k_tiles;

        int32_t acc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int kt = kt_outer; kt < kt_end; ++kt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int idx = tid + u * 128;
                int r_s = idx >> 3;
                int w_s = idx & 7;
                sm_a[r_s][w_s] = a_base[(size_t)(tile_m + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s)];
                sm_b[r_s][w_s] = b_base[(size_t)(tile_n + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s)];
            }
            __syncthreads();

            int ar0 = wm + group_id;
            int ar1 = ar0 + 8;
            uint32_t a0 = sm_a[ar0][tidg];
            uint32_t a1 = sm_a[ar1][tidg];
            uint32_t a2 = sm_a[ar0][tidg + 4];
            uint32_t a3 = sm_a[ar1][tidg + 4];

            #pragma unroll
            for (int ns = 0; ns < 2; ++ns) {
                int bc = wn + ns * 8 + group_id;
                uint32_t b0 = sm_b[bc][tidg];
                uint32_t b1 = sm_b[bc][tidg + 4];
#if (__CUDA_ARCH__ >= 800)
                mma_int8_m16n8k32(acc[ns][0], acc[ns][1], acc[ns][2], acc[ns][3],
                                  a0, a1, a2, a3, b0, b1,
                                  acc[ns][0], acc[ns][1], acc[ns][2], acc[ns][3]);
#else
                // Turing sm_75: 2x along M (ar0, ar1) x 2x along K (kw0, kw1) using m8n8k16
                mma_int8_m8n8k16(acc[ns][0], acc[ns][1], a0, b0, acc[ns][0], acc[ns][1]);
                mma_int8_m8n8k16(acc[ns][0], acc[ns][1], a2, b1, acc[ns][0], acc[ns][1]);
                mma_int8_m8n8k16(acc[ns][2], acc[ns][3], a1, b0, acc[ns][2], acc[ns][3]);
                mma_int8_m8n8k16(acc[ns][2], acc[ns][3], a3, b1, acc[ns][2], acc[ns][3]);
#endif
            }
            __syncthreads();
        }
        #pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) tot[ns][v] += (int64_t)acc[ns][v];
        }
    }

    int32_t* c_plane = c_mod + (size_t)mi * (size_t)m * (size_t)n;
    #pragma unroll
    for (int ns = 0; ns < 2; ++ns) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            int out_r = tile_m + wm + group_id + 8 * (v >> 1);
            int out_c = tile_n + wn + ns * 8 + 2 * tidg + (v & 1);
            if (out_r < m && out_c < n) {
                c_plane[(size_t)out_r * (size_t)n + (size_t)out_c] = pos_mod_i64(tot[ns][v], modulus);
            }
        }
    }
#else
    // Pascal sm_61 hardware IDP.4A.S8.S8 (__dp4a) path:
    // 128 threads arranged 16x8; each thread computes a 2x4 register-blocked micro-tile (8 s32 accumulators).
    int ty = tid >> 3; // 0..15 -> rows 2*ty, 2*ty+1
    int tx = tid & 7;  // 0..7  -> cols 4*tx + 0..3
    int64_t tot[2][4] = {{0,0,0,0},{0,0,0,0}};

    for (int kt_outer = 0; kt_outer < k_tiles; kt_outer += 1024) {
        int kt_end = kt_outer + 1024;
        if (kt_end > k_tiles) kt_end = k_tiles;

        int32_t acc[2][4] = {{0,0,0,0},{0,0,0,0}};
        for (int kt = kt_outer; kt < kt_end; ++kt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int idx = tid + u * 128;
                int r_s = idx >> 3;
                int w_s = idx & 7;
                sm_a[r_s][w_s] = a_base[(size_t)(tile_m + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s)];
                sm_b[r_s][w_s] = b_base[(size_t)(tile_n + r_s) * (size_t)k_words + (size_t)(kt * 8 + w_s)];
            }
            __syncthreads();

            #pragma unroll
            for (int w_s = 0; w_s < 8; ++w_s) {
                int a0 = (int)sm_a[2 * ty][w_s];
                int a1 = (int)sm_a[2 * ty + 1][w_s];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int bv = (int)sm_b[4 * tx + j][w_s];
                    acc[0][j] = __dp4a(a0, bv, acc[0][j]);
                    acc[1][j] = __dp4a(a1, bv, acc[1][j]);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) tot[i][j] += (int64_t)acc[i][j];
        }
    }

    int32_t* c_plane = c_mod + (size_t)mi * (size_t)m * (size_t)n;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int out_r = tile_m + 2 * ty + i;
        if (out_r < m) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int out_c = tile_n + 4 * tx + j;
                if (out_c < n) {
                    c_plane[(size_t)out_r * (size_t)n + (size_t)out_c] = pos_mod_i64(tot[i][j], modulus);
                }
            }
        }
    }
#endif
#else
    (void)a_i8; (void)b_i8; (void)c_mod;
    (void)m; (void)n; (void)m_pad; (void)n_pad; (void)k_pad;
#endif
}

// --- 3D. Ampere+ (sm_80+: m16n8k16) & Turing (sm_75: m16n8k8) FP16 Tensor Core MMA ---
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void mma_fp16_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
__device__ __forceinline__ void mma_fp16_m16n8k8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1,
    uint32_t b0,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1),
          "r"(b0),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}
#endif

extern "C" __global__ void g6me_ozaki2_fp16_mma_residue_gemm(
    const uint32_t* __restrict__ a_f16, // [20 * m_pad * (k_pad / 2)]
    const uint32_t* __restrict__ b_f16, // [20 * n_pad * (k_pad / 2)]
    int32_t* __restrict__ c_mod,        // [20 * m * n]
    int m,
    int n,
    int m_pad,
    int n_pad,
    int k_pad
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    int mi = blockIdx.z;
    if (mi >= OZAKI2_NMOD) return;
    int tile_m = blockIdx.y * 16;
    int tile_n = blockIdx.x * 8;
    int lane = threadIdx.x & 31;
    int group_id = lane >> 2;
    int tidg = lane & 3;

    int k_words = k_pad >> 1; // 2 halfs per uint32
    int k_tiles = k_pad >> 4; // 16 elements = 8 uint32 words per tile
    int32_t modulus = OZAKI2_MODULI[mi];

    const uint32_t* a_base = a_f16 + (size_t)mi * (size_t)m_pad * (size_t)k_words;
    const uint32_t* b_base = b_f16 + (size_t)mi * (size_t)n_pad * (size_t)k_words;

    int row0 = tile_m + group_id;
    int row1 = row0 + 8;
    int col_b = tile_n + group_id;

    int64_t tot[4] = {0, 0, 0, 0};

    // Flush FP32 accumulator into int64 every 128 K-tiles (K = 2,048; max 2048*3136 = 6.42e6 < 2^24).
    for (int kt_outer = 0; kt_outer < k_tiles; kt_outer += 128) {
        int kt_end = kt_outer + 128;
        if (kt_end > k_tiles) kt_end = k_tiles;

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int kt = kt_outer; kt < kt_end; ++kt) {
            int kw0 = kt * 8 + tidg;
            int kw1 = kw0 + 4;
            uint32_t a0 = a_base[(size_t)row0 * (size_t)k_words + (size_t)kw0];
            uint32_t a1 = a_base[(size_t)row1 * (size_t)k_words + (size_t)kw0];
            uint32_t a2 = a_base[(size_t)row0 * (size_t)k_words + (size_t)kw1];
            uint32_t a3 = a_base[(size_t)row1 * (size_t)k_words + (size_t)kw1];
            uint32_t b0 = b_base[(size_t)col_b * (size_t)k_words + (size_t)kw0];
            uint32_t b1 = b_base[(size_t)col_b * (size_t)k_words + (size_t)kw1];
#if (__CUDA_ARCH__ >= 800)
            mma_fp16_m16n8k16(acc[0], acc[1], acc[2], acc[3],
                              a0, a1, a2, a3, b0, b1,
                              acc[0], acc[1], acc[2], acc[3]);
#else
            mma_fp16_m16n8k8(acc[0], acc[1], acc[2], acc[3],
                             a0, a1, b0, acc[0], acc[1], acc[2], acc[3]);
            mma_fp16_m16n8k8(acc[0], acc[1], acc[2], acc[3],
                             a2, a3, b1, acc[0], acc[1], acc[2], acc[3]);
#endif
        }
        #pragma unroll
        for (int v = 0; v < 4; ++v) tot[v] += llrintf(acc[v]);
    }

    int32_t* c_plane = c_mod + (size_t)mi * (size_t)m * (size_t)n;
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int out_r = tile_m + group_id + 8 * (v >> 1);
        int out_c = tile_n + 2 * tidg + (v & 1);
        if (out_r < m && out_c < n) {
            c_plane[(size_t)out_r * (size_t)n + (size_t)out_c] = pos_mod_i64(tot[v], modulus);
        }
    }
#else
    (void)a_f16; (void)b_f16; (void)c_mod;
    (void)m; (void)n; (void)m_pad; (void)n_pad; (void)k_pad;
#endif
}

// ============================================================================
// Stage 4: Exact 128-Bit Integer CRT Reconstruction -> IEEE-754 FP64 (Zero-FMA)
// ============================================================================

extern "C" __global__ void g6me_ozaki2_crt_reconstruct_u128_f64(
    const int32_t* __restrict__ c_mod, // [20 * m * n] residues in [0, m_k - 1]
    const int32_t* __restrict__ exp_a, // [m] (0x7fffffff marks zero row)
    const int32_t* __restrict__ exp_b, // [n] (0x7fffffff marks zero col)
    double* __restrict__ out_f64,      // [m * n]
    int m,
    int n,
    int shift_bits
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    size_t idx = (size_t)row * (size_t)n + (size_t)col;
    int32_t ea = exp_a[row];
    int32_t eb = exp_b[col];
    if (ea == 0x7fffffff || eb == 0x7fffffff) {
        out_f64[idx] = 0.0;
        return;
    }

    const unsigned __int128 m_big  = ((unsigned __int128)OZAKI2_M_BIG_HI << 64) | (unsigned __int128)OZAKI2_M_BIG_LO;
    const unsigned __int128 m_half = ((unsigned __int128)OZAKI2_M_HALF_HI << 64) | (unsigned __int128)OZAKI2_M_HALF_LO;
    size_t plane_stride = (size_t)m * (size_t)n;

    // Exact 128-bit integer CRT accumulation: term < M_BIG < 2^125, acc + term < 2*M_BIG < 2^126 << 2^128.
    unsigned __int128 acc = 0;
    #pragma unroll
    for (int mi = 0; mi < OZAKI2_NMOD; ++mi) {
        uint32_t r = (uint32_t)c_mod[(size_t)mi * plane_stride + idx];
        uint32_t mod = (uint32_t)OZAKI2_MODULI[mi];
        uint32_t t = (r * OZAKI2_CRT_INV[mi]) % mod;
        unsigned __int128 m_div = ((unsigned __int128)OZAKI2_MDIV_HI[mi] << 64) | (unsigned __int128)OZAKI2_MDIV_LO[mi];
        unsigned __int128 term = (unsigned __int128)t * m_div;
        acc += term;
        if (acc >= m_big) acc -= m_big;
    }

    bool neg = acc > m_half;
    unsigned __int128 mag = neg ? (m_big - acc) : acc;
    uint64_t hi = (uint64_t)(mag >> 64);
    uint64_t lo = (uint64_t)mag;

    // Exact zero-FMA conversion of 125-bit unsigned integer to IEEE-754 f64
    // (2^64 = 18446744073709551616.0 is exact in f64).
    double val = __dadd_rn(__dmul_rn((double)hi, 18446744073709551616.0), (double)lo);
    if (neg) val = -val;

    out_f64[idx] = ldexp_zero_fma(val, ea + eb - 2 * shift_bits);
}
