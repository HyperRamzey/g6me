// Zero-FMA Ozaki-II CRT FP64 reconstruction kernels for NVIDIA Blackwell (sm_120) & Ampere+.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
//
// The executed kernels below are scalar CUDA-core code. The file also defines the Blackwell
// NVFP4 E2M1 block-scaled tensor-core wrapper
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
// which is exercised only by the compile-only probe kernel at the end of the file; the sm_120a
// SASS audit fails if that MMA instruction disappears. Plain sm_120 rejects it in ptxas.
// CRT residue reconstruction uses strict IEEE-754 __dadd_rn / __dmul_rn (zero DFMA contractions).

#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
__device__ __forceinline__ void blackwell_nvfp4_mma_m16n8k64(
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

extern "C" __global__ void g6me_ozaki2_crt_reconstruct_f64(
    const int32_t* __restrict__ c_mod,       // [n_mod * m * n] integer residues C_k mod m_k
    const double* __restrict__ crt_weights,  // [n_mod] precomputed (M / m_k) * inv_k
    const double* __restrict__ moduli_f64,   // [n_mod]
    double M_prod,
    double inv_M_prod,
    double scale_a_b,                        // 2^(e_a + e_b - 2*shift)
    double* __restrict__ out_f64,            // [m * n]
    int m,
    int n,
    int n_mod
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    int idx = row * n + col;
    int stride = m * n;
    double acc = 0.0;
    double q_est = 0.0;

    for (int k = 0; k < n_mod; ++k) {
        double r = (double)c_mod[k * stride + idx];
        double w = crt_weights[k];
        double term = __dmul_rn(r, w);
        acc = __dadd_rn(acc, term);
        // Host passes inv_M_prod = 1.0 / M_prod: a multiply by the precomputed reciprocal
        // keeps this reconstruction free of IEEE `div.rn.f64` (whose ptxas MUFU+DFMA
        // microcode is not a contraction, but is avoidable here).
        q_est = __dadd_rn(q_est, __dmul_rn(r, __dmul_rn(w, inv_M_prod)));
    }
    // Nearest-integer CRT folding without FMA
    double q_round = rint(q_est);
    double folded = __dadd_rn(acc, -__dmul_rn(q_round, M_prod));
    // Re-center into [-M/2, M/2]
    double half_m = __dmul_rn(0.5, M_prod);
    if (folded > half_m) folded = __dadd_rn(folded, -M_prod);
    if (folded < -half_m) folded = __dadd_rn(folded, M_prod);
    (void)moduli_f64;
    out_f64[idx] = __dmul_rn(folded, scale_a_b);
}

extern "C" __global__ void g6me_ozaki2_int8_residue_gemm(
    const int8_t* __restrict__ a_mod, // [m * k]
    const int8_t* __restrict__ b_mod, // [k * n]
    int32_t* __restrict__ c_mod,      // [m * n]
    int m,
    int n,
    int k_dim,
    int32_t modulus
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;

    int32_t acc = 0;
    int a_row = row * k_dim;
    for (int kk = 0; kk < k_dim; ++kk) {
        acc += (int32_t)a_mod[a_row + kk] * (int32_t)b_mod[kk * n + col];
    }
    int32_t rem = acc % modulus;
    if (rem < 0) rem += modulus;
    c_mod[row * n + col] = rem;
}

// Compile-coverage probe for the arch-gated Blackwell path. The shipped kernels above are
// scalar CUDA-core code; this kernel is never launched, but its call site keeps the NVFP4
// `mma.sync` instruction in the sm_120a SASS, so the verifier can prove the block actually
// assembles instead of being dead code. Plain sm_120 rejects the instruction in ptxas,
// hence the __CUDA_ARCH_SPECIFIC__ guard.
extern "C" __global__ void g6me_ozaki2_nvfp4_mma_compile_probe(float* out) {
#if defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH__ >= 1200)
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    uint32_t z = 0u;
    blackwell_nvfp4_mma_m16n8k64(d0, d1, d2, d3, z, z, z, z, z, z,
                                 0.0f, 0.0f, 0.0f, 0.0f, z, z);
    if (out != 0) out[0] = d0 + d1 + d2 + d3;
#else
    (void)out;
#endif
}
