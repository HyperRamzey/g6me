//! Optional zero-FMA CUDA / Multi-Architecture Tensor Core GPU backend and real
//! `nvcc --fmad=false` + `cuobjdump --dump-sass` hardware SASS auditor for G6ME.
//!
//! - Uses `cudarc` with dynamic loading (`libcuda.so.1` loaded at runtime via `dlopen`) so binaries
//!   built with `--features cuda` still run cleanly on CPU-only machines and fall back gracefully.
//! - Embeds and loads ALL 6 real CUDA C++ kernels into [`G6meCudaRuntime`]:
//!   1. `fdtd_3d.cu` — zero-FMA 3D SRL FDTD acoustic wave solver (`f64`),
//!   2. `raytrace_hoa.cu` — zero-FMA deterministic PCG32 stochastic ray tracer (`f64`),
//!   3. `stockham_fft_f64.cu` — zero-FMA Stockham radix-2 complex FFT + MAC (`f64`),
//!   4. `goldilocks_ntt.cu` — exact 64-bit Goldilocks prime NTT convolution (pure integer, 0 ULP),
//!   5. `bem_matvec.cu` — zero-FMA complex Helmholtz BEM matvec (`f64`),
//!   6. `ozaki2_blackwell_mma.cu` — full 4-stage device-side Ozaki-II CRT FP64-on-Tensor-Core
//!      pipeline supporting:
//!      - Blackwell (`sm_120a`): NVFP4 E2M1 Base-13 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` → `u128` CRT → `f64`,
//!      - Hopper/Ada (`sm_90`/`sm_89`): FP8 E4M3 Base-16 `mma.m16n8k32.f32.e4m3` → `u128` CRT → `f64`,
//!      - Ampere (`sm_80`/`sm_86`): INT8 `IMMA.16832` & FP16 `HMMA.16816` → `u128` CRT → `f64`,
//!      - Turing (`sm_75`): INT8 `IMMA.8816` & FP16 `HMMA.1688` → `u128` CRT → `f64`.
//! - Provides real `nvcc --fmad=false` PTX + `cuobjdump --dump-sass` verification over all 6
//!   kernels AND across all 5 GPU architectures (`sm_120a`, `sm_90`, `sm_89`, `sm_80`, `sm_75`),
//!   plus a mandatory `--fmad=true` sabotage negative control.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Embedded zero-FMA 3D SRL FDTD acoustic wave kernel (`fdtd_3d.cu`).
pub const KERNEL_FDTD_3D_CU: &str = include_str!("../kernels/fdtd_3d.cu");
/// Embedded zero-FMA stochastic ray-tracing HOA kernel (`raytrace_hoa.cu`).
pub const KERNEL_RAYTRACE_HOA_CU: &str = include_str!("../kernels/raytrace_hoa.cu");
/// Embedded zero-FMA FP64 Stockham FFT + complex MAC kernel (`stockham_fft_f64.cu`).
pub const KERNEL_STOCKHAM_FFT_CU: &str = include_str!("../kernels/stockham_fft_f64.cu");
/// Embedded exact 64-bit Goldilocks prime NTT kernel (`goldilocks_ntt.cu`).
pub const KERNEL_GOLDILOCKS_NTT_CU: &str = include_str!("../kernels/goldilocks_ntt.cu");
/// Embedded zero-FMA Helmholtz BEM complex matvec kernel (`bem_matvec.cu`).
pub const KERNEL_BEM_MATVEC_CU: &str = include_str!("../kernels/bem_matvec.cu");
/// Embedded zero-FMA Multi-Arch Ozaki-II CRT Tensor Core kernel (`ozaki2_blackwell_mma.cu`).
pub const KERNEL_OZAKI2_MMA_CU: &str = include_str!("../kernels/ozaki2_blackwell_mma.cu");

/// Number of pairwise-coprime Ozaki-II CRT moduli (`<= 113`, `log2(∏ m_k) = 124.31 bits`).
pub const OZAKI2_NMOD: usize = 20;

/// User-selectable or architecture-resolved Tensor Core precision for Ozaki-II FP64 GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaTensorPrecision {
    /// Automatically pick the highest-throughput precision supported by the active GPU arch:
    /// every arch from `sm_61` through `sm_120a` → `INT8` (single-limb IMMA/DP4A, faster than
    /// the 2-limb NVFP4/FP8 paths). NVFP4 is never selected automatically: the
    /// `mma.sync … kind::mxf4nvf4` path used here exists only on `sm_120a` (consumer Blackwell),
    /// while B200/B300 (`sm_100a`/`sm_103a`) would need a separate `tcgen05` FP4 kernel.
    #[default]
    Auto,
    /// Blackwell (`sm_120a`) NVFP4 E2M1 Base-13 2-limb (`OMMA.SF.16864` with UE4M3=2.0 → FP32 → `u128` CRT → FP64).
    Nvfp4E2m1,
    /// Ada/Hopper/Blackwell (`sm_89+`) FP8 E4M3 Base-16 2-limb (`m16n8k32` → FP32 → `u128` CRT → FP64).
    Fp8E4m3,
    /// Pascal (`sm_61` `__dp4a` / `IDP.4A.S8.S8`) & Turing..Blackwell (`sm_75+` `IMMA`)
    /// INT8 single-limb (`1 MMA/mod` → exact `S32` → `u128` CRT → FP64; fastest on desktop GPUs).
    Int8,
    /// Turing/Ampere+ (`sm_75+`) IEEE FP16 single-limb (`m16n8k16`/`m16n8k8` → FP32 → `u128` CRT → FP64).
    Fp16,
}

impl CudaTensorPrecision {
    /// Resolve `Auto` (or gracefully downgrade an unsupported manual choice) to the highest-throughput
    /// bit-exact Ozaki-II hardware path for `(major, minor)` and compiled target `arch`.
    ///
    /// **Desktop TOPS analysis (`M = 20` moduli, residues in `[-56, +56]`)**:
    /// - Single-limb `INT8` (`1 MMA/mod`, exact `S32` accum):
    ///   - RTX 5090 (`sm_120a`): `838 TOPS / 1 = 838 eff TOPS` (~41.9 TFLOPS FP64 peak, **2× faster** than 2-limb NVFP4 `1676/4 = 419 eff TOPS`).
    ///   - RTX 4090 (`sm_89`): `661 TOPS / 1 = 661 eff TOPS` (~33.0 TFLOPS FP64 peak, **8× faster** than 2-limb FP8-FP32 `330/4 = 83 eff TOPS`).
    ///   - RTX 3090 (`sm_86`): `284 TOPS / 1 = 284 eff TOPS` (~14.2 TFLOPS FP64 peak, **4× faster** than FP16-FP32).
    ///   - RTX 2080 Ti (`sm_75`): `215 TOPS / 1 = 215 eff TOPS` (~10.8 TFLOPS FP64 peak, **4× faster** than FP16-FP32).
    ///   - GTX 1080 Ti (`sm_61`): `45.4 TOPS` via `IDP.4A.S8.S8` (`__dp4a`) (~2.27 TFLOPS FP64 peak, **6.5× native FP64**).
    /// - `Auto` always resolves to `Int8`; NVFP4 is an explicit opt-in on `sm_120a` only.
    /// - Explicit `Nvfp4E2m1` / `Fp8E4m3` / `Fp16` selections are honored where the deployed
    ///   kernel actually has a body: NVFP4 requires `cc == 120` **and** an arch-specific (`'a'`)
    ///   compile target; B200/B300 (`sm_100a`/`sm_103a`) lack that `mma.sync` form and downgrade
    ///   to FP8. This mirrors the `__CUDA_ARCH__ >= 1200` guards in
    ///   `ozaki2_blackwell_mma.cu`, so a selection can never resolve to an empty kernel.
    pub fn resolve_for_arch(self, major: i32, minor: i32, compiled_arch: &str) -> Self {
        let cc = major * 10 + minor;
        let has_nvfp4 = cc == 120 && compiled_arch.ends_with('a');
        let has_fp8 = cc >= 89;
        let has_fp16_tc = cc >= 75;
        match self {
            Self::Auto => Self::Int8,
            Self::Nvfp4E2m1 => {
                if has_nvfp4 {
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
}

/// Per-architecture Tensor Core / DP4A & FP64 offload capability specification.
#[derive(Debug, Clone, Copy)]
pub struct CudaArchPlan {
    pub arch_name: &'static str,
    pub sm_target: &'static str,
    pub cc_range: &'static str,
    pub supports_nvfp4: bool,
    pub supports_fp8: bool,
    pub supports_int8: bool,
    pub supports_fp16: bool,
    pub default_ozaki_precision: CudaTensorPrecision,
    pub mma_opcode_summary: &'static str,
    pub non_gemm_summary: &'static str,
}

/// Complete capability matrix across supported desktop & datacenter NVIDIA GPU architectures.
pub const ALL_CUDA_ARCH_PLANS: [CudaArchPlan; 7] = [
    CudaArchPlan {
        arch_name: "Blackwell (RTX 5090 / 5080)",
        sm_target: "sm_120a",
        cc_range: "CC 12.0+",
        supports_nvfp4: true,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary:
            "IMMA.16832 S8→S32 (838 eff TOPS, ~41.9 TF FP64); opt OMMA.SF NVFP4 (419 eff)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Blackwell datacenter (B200 / B300)",
        sm_target: "sm_100 / sm_103",
        cc_range: "CC 10.0–10.3",
        supports_nvfp4: false,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary:
            "IMMA.16832 S8→S32 (exact 1-limb); opt FP8 QMMA / FP16 HMMA (FP4 needs tcgen05, not implemented)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Hopper (H100 / H200)",
        sm_target: "sm_90",
        cc_range: "CC 9.0",
        supports_nvfp4: false,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary:
            "IMMA.16832 S8→S32 (1-limb exact); opt FP8 QMMA / FP16 HMMA (no HW FP4)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Ada Lovelace (RTX 4090 / 4080)",
        sm_target: "sm_89",
        cc_range: "CC 8.9",
        supports_nvfp4: false,
        supports_fp8: true,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary: "IMMA.16832 S8→S32 (661 eff TOPS, ~33.0 TF FP64, 8x > FP8-FP32 QMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Ampere (RTX 3090 / A100)",
        sm_target: "sm_80",
        cc_range: "CC 8.0–8.7",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary: "IMMA.16832 S8→S32 (284 eff TOPS, ~14.2 TF FP64, 4x > FP16 HMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Turing (RTX 2080 Ti / T4)",
        sm_target: "sm_75",
        cc_range: "CC 7.5",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: true,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary: "IMMA.8816 S8→S32 (215 eff TOPS, ~10.8 TF FP64, 4x > FP16 HMMA)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
    CudaArchPlan {
        arch_name: "Pascal (GTX 1080 Ti / 1080)",
        sm_target: "sm_61",
        cc_range: "CC 6.1",
        supports_nvfp4: false,
        supports_fp8: false,
        supports_int8: true,
        supports_fp16: false,
        default_ozaki_precision: CudaTensorPrecision::Int8,
        mma_opcode_summary:
            "IDP.4A.S8.S8 (__dp4a→S32, 45.4 eff TOPS, ~2.27 TF FP64, 6.5x native FP64)",
        non_gemm_summary: "FDTD/BEM/RayTrace/Stockham: Zero-FMA FP64; NTT: Exact u64",
    },
];

/// Map a detected CUDA compute capability `(major, minor)` to its [`CudaArchPlan`].
pub fn arch_capability_plan(major: i32, minor: i32) -> CudaArchPlan {
    let cc = major * 10 + minor;
    if cc >= 120 {
        ALL_CUDA_ARCH_PLANS[0]
    } else if cc >= 100 {
        ALL_CUDA_ARCH_PLANS[1]
    } else if cc >= 90 {
        ALL_CUDA_ARCH_PLANS[2]
    } else if cc >= 89 {
        ALL_CUDA_ARCH_PLANS[3]
    } else if cc >= 80 {
        ALL_CUDA_ARCH_PLANS[4]
    } else if cc >= 75 {
        ALL_CUDA_ARCH_PLANS[5]
    } else {
        ALL_CUDA_ARCH_PLANS[6]
    }
}

/// A production CUDA kernel source paired with the hardware-SASS audit policy it requires.
#[derive(Debug, Clone, Copy)]
pub struct CudaKernelSource {
    /// Logical kernel name (also the embedded `.cu` basename).
    pub name: &'static str,
    /// Full embedded CUDA C++ source.
    pub source: &'static str,
    /// True only for kernels whose source calls IEEE correctly-rounded `div.rn.f64` /
    /// `sqrt.rn.f64`. ptxas lowers those single IEEE operations to `MUFU.RCP64H` /
    /// `MUFU.RSQ64H` plus a DFMA Newton-Raphson refinement (and DFMA slow paths); those
    /// DFMA instructions are the hardware implementation of one correctly-rounded operation
    /// — bit-identical to CPU `f64` division/square root — not contractions of the kernel's
    /// own arithmetic.
    pub ieee_div_sqrt: bool,
}

/// All 6 production CUDA kernels paired with their logical names and audit policy.
pub const ALL_CUDA_KERNELS: [CudaKernelSource; 6] = [
    CudaKernelSource {
        name: "fdtd_3d",
        source: KERNEL_FDTD_3D_CU,
        ieee_div_sqrt: false,
    },
    CudaKernelSource {
        name: "raytrace_hoa",
        source: KERNEL_RAYTRACE_HOA_CU,
        ieee_div_sqrt: true,
    },
    CudaKernelSource {
        name: "stockham_fft_f64",
        source: KERNEL_STOCKHAM_FFT_CU,
        ieee_div_sqrt: false,
    },
    CudaKernelSource {
        name: "goldilocks_ntt",
        source: KERNEL_GOLDILOCKS_NTT_CU,
        ieee_div_sqrt: false,
    },
    CudaKernelSource {
        name: "bem_matvec",
        source: KERNEL_BEM_MATVEC_CU,
        ieee_div_sqrt: false,
    },
    CudaKernelSource {
        name: "ozaki2_blackwell_mma",
        source: KERNEL_OZAKI2_MMA_CU,
        ieee_div_sqrt: false,
    },
];

/// Detected NVIDIA CUDA device summary.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub index: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub is_blackwell: bool,
}

impl CudaDeviceInfo {
    /// Target `sm_XY` (or `sm_120a` for Blackwell NVFP4 MMA) architecture string for `nvcc`.
    pub fn sm_arch(&self) -> String {
        if self.compute_capability.0 == 12 && self.compute_capability.1 == 0 {
            "sm_120a".to_string()
        } else {
            format!(
                "sm_{}{}",
                self.compute_capability.0, self.compute_capability.1
            )
        }
    }

    /// Capability plan for this device's compute capability.
    pub fn arch_plan(&self) -> CudaArchPlan {
        arch_capability_plan(self.compute_capability.0, self.compute_capability.1)
    }
}

/// Probe CUDA device 0 dynamically via `libcuda.so.1` / `nvcuda.dll`.
/// Returns `None` cleanly on machines without an NVIDIA GPU or driver.
pub fn cuda_device_info() -> Option<CudaDeviceInfo> {
    let count = CudaContext::device_count().ok()?;
    if count <= 0 {
        return None;
    }
    let ctx = CudaContext::new(0).ok()?;
    let name = ctx.name().unwrap_or_else(|_| "NVIDIA CUDA GPU".to_string());
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(8);
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    Some(CudaDeviceInfo {
        index: 0,
        name,
        compute_capability: (major, minor),
        is_blackwell: major >= 10,
    })
}

/// Per-stage Stockham twiddles, device-resident (`(tw_re, tw_im)` per radix-2 stage).
type FftTwiddleStages = Vec<(CudaSlice<f64>, CudaSlice<f64>)>;
/// Per-stage Goldilocks NTT twiddles, device-resident (one `u64` table per radix-2 stage).
type NttTwiddleStages = Vec<CudaSlice<u64>>;

/// Live `cudarc` GPU runtime with ALL 6 dynamically compiled zero-FMA PTX modules loaded.
pub struct G6meCudaRuntime {
    pub info: CudaDeviceInfo,
    pub compiled_arch: String,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    fdtd_mod: Arc<CudaModule>,
    bem_mod: Arc<CudaModule>,
    ozaki2_mod: Arc<CudaModule>,
    raytrace_mod: Arc<CudaModule>,
    fft_mod: Arc<CudaModule>,
    ntt_mod: Arc<CudaModule>,
    /// Device-resident Stockham twiddle stages keyed by `(n, inverse)`. Uploaded once per
    /// transform length/direction instead of once per block (the previous per-stage htod
    /// uploads were the dominant redundant host->device traffic in the streaming UPOLA loop).
    fft_twiddle_cache: Mutex<HashMap<(usize, bool), FftTwiddleStages>>,
    /// Device-resident Goldilocks NTT twiddle stages keyed by `(n, inverse, fingerprint)`.
    ntt_twiddle_cache: Mutex<HashMap<(usize, bool, u64), NttTwiddleStages>>,
    /// Concatenated per-stage twiddle tables for the fused single-launch transform,
    /// keyed like `ntt_twiddle_cache`. Stage `s` starts at element `(1 << s) - 1`.
    ntt_fused_twiddle_cache: Mutex<HashMap<(usize, bool, u64), CudaSlice<u64>>>,
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` for this device. The fused
    /// NTT needs `n * 8` dynamic shared bytes, which exceeds the 48 KiB default on every
    /// target that can hold a 4096/8192-point transform.
    max_dynamic_shared: usize,
}

/// Threads per block for the fused single-block Goldilocks NTT. One block owns the whole
/// transform, so this only trades per-stage loop trips against `__syncthreads` overhead.
const NTT_FUSED_THREADS: u32 = 1024;

#[inline]
fn align_up(v: usize, align: usize) -> usize {
    v.div_ceil(align) * align
}

/// Fixed-point scaling exponent for the Ozaki-II Stage-1 limb decomposition.
///
/// Full 53-bit `f64` mantissa precision holds for `k_dim <= 131072` (2^17); above that the
/// exponent shrinks as `(123 - log2(k)) / 2` (52 bits at 2^18, 49 at 2^24, floored at 40) so
/// the 123-bit CRT product still cannot overflow. Within those bounds the result is exact for
/// the quantized limbs; the quantization itself, not the CRT, is what limits accuracy.
///
/// MUST stay identical to `g6me_core::blackwell::ozaki2_shift_bits`: the CPU reference and
/// the live GPU stage-1 quantization must produce identical limbs for bit-exact parity.
#[inline]
pub fn ozaki2_shift_bits(k_dim: usize) -> i32 {
    if k_dim <= 131_072 {
        53
    } else {
        let log2_k = (k_dim as f64).log2().ceil() as i32;
        ((123 - log2_k) / 2).clamp(40, 53)
    }
}

/// Cheap content fingerprint for one direction's Goldilocks NTT twiddle tables.
///
/// `NttPlan` tables are deterministic functions of `(n, direction)`, so the fingerprint is a
/// defensive guard rather than a strict identity: it catches a caller passing a table of the
/// right length but different values.
fn ntt_twiddle_key(n: usize, inverse: bool, twiddles: &[Vec<u64>]) -> (usize, bool, u64) {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ (n as u64);
    if inverse {
        h ^= 0xA5A5_A5A5_A5A5_A5A5;
    }
    for stage in twiddles {
        h = h.rotate_left(7) ^ (stage.len() as u64).wrapping_mul(0x0000_0100_0000_01B3);
        h = h.rotate_left(7) ^ stage.first().copied().unwrap_or(0);
        h = h.rotate_left(7) ^ stage.last().copied().unwrap_or(0);
    }
    (n, inverse, h)
}

/// `(ray_delays, ray_energies, ray_dirs)` from [`G6meCudaRuntime::run_raytrace_hoa_pass`].
pub type RayTraceHoaOutput = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Host-built, x-independent FMM interaction plan for the zero-FMA BEM operator.
///
/// The caller (G6ME core) builds this once per `(solver, k, adjoint)` from the same
/// octree, admissibility test and coefficient formulas used by the CPU FMM matvec.
/// [`G6meCudaRuntime::bem_fmm_operator`] uploads it to the device exactly once; every
/// subsequent [`CudaBemOperator::apply`] only ships the `2n` vector components and runs
/// the kernels. All arrays are flat and C-order so they map directly to device buffers.
#[derive(Debug, Clone, Default)]
pub struct BemFmmPlan {
    /// Number of BEM elements.
    pub n: usize,
    /// Number of octree nodes (pre-order ids; every child id > its parent id).
    pub node_count: usize,
    /// Deepest octree level index (root level is 0).
    pub max_level: usize,
    /// Exact CPU solid-angle diagonal `A_ii`.
    pub diag: Vec<f64>,
    /// Per-source `-area` (the CPU `w = x[j] * -area` left factor).
    pub src_scale: Vec<f64>,
    /// Per-source outward normal, x components (`[n]`; `src_ny` / `src_nz` likewise).
    pub src_nx: Vec<f64>,
    pub src_ny: Vec<f64>,
    pub src_nz: Vec<f64>,
    /// Leaf node id containing each source element.
    pub src_leaf: Vec<i32>,
    /// Flattened octree child lists (`[node_count + 1]` offsets).
    pub child_start: Vec<i32>,
    pub children: Vec<i32>,
    /// Flattened leaf source lists (`[node_count + 1]` offsets), stored in CPU order.
    pub leaf_start: Vec<i32>,
    pub leaf_src: Vec<i32>,
    /// Node ids grouped by depth (`[max_level + 2]` offsets).
    pub level_offsets: Vec<i32>,
    pub level_nodes: Vec<i32>,
    /// Forward per-target interaction list in CPU DFS order (`[n + 1]` offsets).
    pub fwd_start: Vec<i32>,
    /// 0 = near element, 1 = far octree node.
    pub fwd_kind: Vec<i32>,
    pub fwd_index: Vec<i32>,
    /// Near: exact complex coefficient. Far: exact complex cluster scalar.
    pub fwd_a: Vec<f64>,
    pub fwd_b: Vec<f64>,
    /// Far only: `node.center - x_target` components.
    pub fwd_dx: Vec<f64>,
    pub fwd_dy: Vec<f64>,
    pub fwd_dz: Vec<f64>,
    /// Adjoint near list, bucketed per source in ascending target order (`[n + 1]`).
    pub adj_src_start: Vec<i32>,
    pub adj_src_i: Vec<i32>,
    pub adj_src_cr: Vec<f64>,
    pub adj_src_ci: Vec<f64>,
    /// Adjoint far list, bucketed per octree node in ascending target order.
    pub adj_node_start: Vec<i32>,
    pub adj_node_i: Vec<i32>,
    pub adj_node_sr: Vec<f64>,
    pub adj_node_si: Vec<f64>,
    pub adj_node_dx: Vec<f64>,
    pub adj_node_dy: Vec<f64>,
    pub adj_node_dz: Vec<f64>,
}

/// Cached zero-FMA BEM operator for one `(solver, k, adjoint)` solve.
///
/// Construction uploads the full dense matrix (explicit path) or the FMM interaction
/// plan once. [`Self::apply`] performs no `n^2` host work and re-uploads no coefficient
/// data: it ships `x` (`2n` doubles), launches the kernels, and reads back `y`.
pub struct CudaBemOperator {
    stream: Arc<CudaStream>,
    n: usize,
    d_xr: CudaSlice<f64>,
    d_xi: CudaSlice<f64>,
    d_yr: CudaSlice<f64>,
    d_yi: CudaSlice<f64>,
    kind: BemOperatorKind,
}

enum BemOperatorKind {
    Dense {
        d_mat_re: CudaSlice<f64>,
        d_mat_im: CudaSlice<f64>,
        func: CudaFunction,
    },
    FmmForward(Box<BemFmmForwardDevice>),
    FmmAdjoint(Box<BemFmmAdjointDevice>),
}

struct BemFmmForwardDevice {
    /// `(begin, count)` into `d_level_nodes` for each depth, root first.
    level_ranges: Vec<(i32, i32)>,
    d_src_scale: CudaSlice<f64>,
    d_src_nx: CudaSlice<f64>,
    d_src_ny: CudaSlice<f64>,
    d_src_nz: CudaSlice<f64>,
    d_child_start: CudaSlice<i32>,
    d_children: CudaSlice<i32>,
    d_leaf_start: CudaSlice<i32>,
    d_leaf_src: CudaSlice<i32>,
    d_level_nodes: CudaSlice<i32>,
    d_diag: CudaSlice<f64>,
    d_mom: CudaSlice<f64>,
    d_f_start: CudaSlice<i32>,
    d_f_kind: CudaSlice<i32>,
    d_f_index: CudaSlice<i32>,
    d_f_a: CudaSlice<f64>,
    d_f_b: CudaSlice<f64>,
    d_f_dx: CudaSlice<f64>,
    d_f_dy: CudaSlice<f64>,
    d_f_dz: CudaSlice<f64>,
    fn_upward: CudaFunction,
    fn_forward: CudaFunction,
}

struct BemFmmAdjointDevice {
    node_count: usize,
    level_ranges: Vec<(i32, i32)>,
    d_src_scale: CudaSlice<f64>,
    d_src_nx: CudaSlice<f64>,
    d_src_ny: CudaSlice<f64>,
    d_src_nz: CudaSlice<f64>,
    d_src_leaf: CudaSlice<i32>,
    d_child_start: CudaSlice<i32>,
    d_children: CudaSlice<i32>,
    d_level_nodes: CudaSlice<i32>,
    d_diag: CudaSlice<f64>,
    d_a_src_start: CudaSlice<i32>,
    d_a_src_i: CudaSlice<i32>,
    d_a_src_cr: CudaSlice<f64>,
    d_a_src_ci: CudaSlice<f64>,
    d_a_node_start: CudaSlice<i32>,
    d_a_node_i: CudaSlice<i32>,
    d_a_node_sr: CudaSlice<f64>,
    d_a_node_si: CudaSlice<f64>,
    d_a_node_dx: CudaSlice<f64>,
    d_a_node_dy: CudaSlice<f64>,
    d_a_node_dz: CudaSlice<f64>,
    d_grad: CudaSlice<f64>,
    fn_nodes: CudaFunction,
    fn_push: CudaFunction,
    fn_sources: CudaFunction,
}

/// `(x_re, x_im, y_re, y_im)` device buffers reused across every [`CudaBemOperator::apply`].
type BemIoBuffers = (
    CudaSlice<f64>,
    CudaSlice<f64>,
    CudaSlice<f64>,
    CudaSlice<f64>,
);

#[inline]
fn bem_grid(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count.max(1) as u32).div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl G6meCudaRuntime {
    /// Initialize CUDA device 0 and compile + load all 6 zero-FMA PTX modules via `nvcc --fmad=false`.
    pub fn init() -> Result<Self, String> {
        let info = cuda_device_info().ok_or_else(|| "no CUDA GPU detected".to_string())?;
        let ctx = CudaContext::new(info.index)
            .map_err(|e| format!("failed to create CUDA context: {e}"))?;
        let stream = ctx.default_stream();
        let arch = pick_supported_nvcc_arch(Some(&info.sm_arch()))?;
        // `pick_supported_nvcc_arch` falls back to whatever the toolkit accepts, which can be
        // newer than the live device (e.g. CUDA 13 dropping Pascal and selecting sm_120a for a
        // CC 6.1 card). PTX is forward-compatible only toward newer GPUs, and architecture-
        // specific `sm_XYa` targets are not forward-compatible at all, so reject a target the
        // device cannot execute instead of loading it and silently falling back to CPU.
        let (maj, min) = info.compute_capability;
        if let Some((amaj, amin)) = parse_sm_arch_cc(&arch) {
            let runnable = if arch.ends_with('a') {
                amaj == maj && amin == min
            } else {
                amaj < maj || (amaj == maj && amin <= min)
            };
            if !runnable {
                return Err(format!(
                    "nvcc selected {arch} but the live device is CC {maj}.{min}; \
                     the installed toolkit cannot target this GPU"
                ));
            }
        }

        let load = |name: &str, src: &str| -> Result<Arc<CudaModule>, String> {
            let ptx = compile_cu_to_ptx(name, src, &arch, false)?;
            ctx.load_module(Ptx::from_src(ptx))
                .map_err(|e| format!("failed to load {name} PTX module ({arch}): {e}"))
        };

        let fdtd_mod = load("fdtd_3d", KERNEL_FDTD_3D_CU)?;
        let bem_mod = load("bem_matvec", KERNEL_BEM_MATVEC_CU)?;
        let ozaki2_mod = load("ozaki2_blackwell_mma", KERNEL_OZAKI2_MMA_CU)?;
        let raytrace_mod = load("raytrace_hoa", KERNEL_RAYTRACE_HOA_CU)?;
        let fft_mod = load("stockham_fft_f64", KERNEL_STOCKHAM_FFT_CU)?;
        let ntt_mod = load("goldilocks_ntt", KERNEL_GOLDILOCKS_NTT_CU)?;
        // Dynamic shared-memory opt-in is a per-device hard limit (99 KiB on cc 12.0,
        // 64 KiB on sm_75). Query it instead of assuming the 48 KiB default.
        let max_dynamic_shared = ctx
            .attribute(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map(|v| v.max(0) as usize)
            .unwrap_or(48 * 1024);

        Ok(Self {
            info,
            compiled_arch: arch,
            ctx,
            stream,
            fdtd_mod,
            bem_mod,
            ozaki2_mod,
            raytrace_mod,
            fft_mod,
            ntt_mod,
            fft_twiddle_cache: Mutex::new(HashMap::new()),
            ntt_twiddle_cache: Mutex::new(HashMap::new()),
            ntt_fused_twiddle_cache: Mutex::new(HashMap::new()),
            max_dynamic_shared,
        })
    }

    /// Upload the Stockham twiddle stages for `(n, inverse)` exactly once per process.
    ///
    /// The host-side twiddle expressions are unchanged from the previous per-call path
    /// (`ang = sign * 2*pi*q / (2m)`, `cos`/`sin`), so caching cannot change numerics; it only
    /// removes `2 * log2(n)` redundant htod copies per block.
    fn ensure_fft_twiddles(&self, n: usize, inverse: bool) -> Result<(), String> {
        let key = (n, inverse);
        let mut guard = self
            .fft_twiddle_cache
            .lock()
            .map_err(|_| "fft twiddle cache poisoned".to_string())?;
        if guard.contains_key(&key) {
            return Ok(());
        }
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let half_n = n / 2;
        let mut stages: FftTwiddleStages = Vec::with_capacity(n.trailing_zeros() as usize);
        let mut m = half_n;
        while m >= 1 {
            let mut tw_re = Vec::with_capacity(m);
            let mut tw_im = Vec::with_capacity(m);
            for q in 0..m {
                let ang = sign * 2.0 * PI * (q as f64) / ((2 * m) as f64);
                tw_re.push(ang.cos());
                tw_im.push(ang.sin());
            }
            let d_re = self
                .stream
                .clone_htod(&tw_re)
                .map_err(|e| format!("htod fft tw_re: {e}"))?;
            let d_im = self
                .stream
                .clone_htod(&tw_im)
                .map_err(|e| format!("htod fft tw_im: {e}"))?;
            stages.push((d_re, d_im));
            m >>= 1;
        }
        guard.insert(key, stages);
        Ok(())
    }

    /// Upload the Goldilocks NTT twiddle stages exactly once per `(n, direction, table)`.
    ///
    /// The tables come from [`crate::G6meCudaRuntime`]-external `NttPlan`s and are already
    /// precomputed on the host; the fingerprint guards against silently reusing a cache entry
    /// for a different table of the same length.
    fn ensure_ntt_twiddles(
        &self,
        n: usize,
        inverse: bool,
        twiddles: &[Vec<u64>],
    ) -> Result<(usize, bool, u64), String> {
        let key = ntt_twiddle_key(n, inverse, twiddles);
        let mut guard = self
            .ntt_twiddle_cache
            .lock()
            .map_err(|_| "ntt twiddle cache poisoned".to_string())?;
        if let std::collections::hash_map::Entry::Vacant(entry) = guard.entry(key) {
            let mut stages: NttTwiddleStages = Vec::with_capacity(twiddles.len());
            for tw in twiddles {
                let d_tw = self
                    .stream
                    .clone_htod(tw.as_slice())
                    .map_err(|e| format!("htod ntt tw: {e}"))?;
                stages.push(d_tw);
            }
            entry.insert(stages);
        }
        Ok(key)
    }

    /// Concatenate one direction's per-stage twiddle tables into the single device buffer the
    /// fused kernel indexes (`stage s` at offset `(1 << s) - 1`), uploaded once per key.
    ///
    /// The concatenation is a pure layout change: every `u64` is copied verbatim from the
    /// `NttPlan` tables the multi-launch path uses, so the fused kernel reads exactly the
    /// same twiddle values.
    fn ensure_fused_ntt_twiddles(
        &self,
        n: usize,
        inverse: bool,
        twiddles: &[Vec<u64>],
    ) -> Result<(usize, bool, u64), String> {
        let log_n = n.trailing_zeros() as usize;
        if twiddles.len() != log_n {
            return Err("Goldilocks NTT twiddle stage count mismatch".into());
        }
        let mut flat = Vec::with_capacity(n.saturating_sub(1));
        for (stage, tw) in twiddles.iter().enumerate() {
            if tw.len() != (1usize << stage) {
                return Err(format!(
                    "Goldilocks NTT stage {stage} twiddle length {} != {}",
                    tw.len(),
                    1usize << stage
                ));
            }
            flat.extend_from_slice(tw);
        }
        if flat.len() != n - 1 {
            return Err(format!(
                "Goldilocks NTT twiddle concatenation length {} != {}",
                flat.len(),
                n - 1
            ));
        }
        let key = ntt_twiddle_key(n, inverse, twiddles);
        let mut guard = self
            .ntt_fused_twiddle_cache
            .lock()
            .map_err(|_| "ntt fused twiddle cache poisoned".to_string())?;
        if let std::collections::hash_map::Entry::Vacant(entry) = guard.entry(key) {
            let d_tw = self
                .stream
                .clone_htod(&flat)
                .map_err(|e| format!("htod fused ntt tw: {e}"))?;
            entry.insert(d_tw);
        }
        Ok(key)
    }

    /// Resolve which Tensor Core precision will execute on this GPU for a user request.
    pub fn active_precision(&self, requested: CudaTensorPrecision) -> CudaTensorPrecision {
        requested.resolve_for_arch(
            self.info.compute_capability.0,
            self.info.compute_capability.1,
            &self.compiled_arch,
        )
    }

    /// Execute the whole 3D SRL FDTD simulation on the GPU.
    /// Returns `out[receiver][step]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_fdtd_3d_simulation(
        &self,
        p_prev: &[f64],
        p_curr: &[f64],
        inv_den: &[f64; 7],
        nx: usize,
        ny: usize,
        nz: usize,
        lam2: f64,
        half_lam_beta: f64,
        rec_ids: &[usize],
        steps: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        let n = nx * ny * nz;
        if p_prev.len() != n || p_curr.len() != n {
            return Err("FDTD buffer size mismatch".into());
        }
        for &rid in rec_ids {
            if rid >= n {
                return Err(format!("receiver id {rid} is outside the {n}-voxel grid"));
            }
        }
        let n_rec = rec_ids.len();
        let ids_i32: Vec<i32> = rec_ids.iter().map(|&r| r as i32).collect();

        let mut d_prev = self
            .stream
            .clone_htod(p_prev)
            .map_err(|e| format!("htod p_prev: {e}"))?;
        let mut d_curr = self
            .stream
            .clone_htod(p_curr)
            .map_err(|e| format!("htod p_curr: {e}"))?;
        let mut d_next = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc p_next: {e}"))?;
        let d_inv = self
            .stream
            .clone_htod(inv_den.as_slice())
            .map_err(|e| format!("htod inv_den: {e}"))?;
        let d_rec = if n_rec == 0 {
            self.stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| format!("alloc rec_ids: {e}"))?
        } else {
            self.stream
                .clone_htod(&ids_i32)
                .map_err(|e| format!("htod rec_ids: {e}"))?
        };
        let mut d_out = self
            .stream
            .alloc_zeros::<f64>((steps * n_rec).max(1))
            .map_err(|e| format!("alloc rec_out: {e}"))?;

        let fn_step: CudaFunction = self
            .fdtd_mod
            .load_function("g6me_fdtd_3d_srl_step")
            .map_err(|e| format!("load g6me_fdtd_3d_srl_step: {e}"))?;
        let fn_rec: CudaFunction = self
            .fdtd_mod
            .load_function("g6me_fdtd_record_receivers")
            .map_err(|e| format!("load g6me_fdtd_record_receivers: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (
                (nx as u32).div_ceil(8),
                (ny as u32).div_ceil(8),
                (nz as u32).div_ceil(4),
            ),
            block_dim: (8, 8, 4),
            shared_mem_bytes: 0,
        };
        let rec_cfg = LaunchConfig {
            grid_dim: ((n_rec.max(1) as u32).div_ceil(32), 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let n_rec_i = n_rec as i32;
        for step in 0..steps {
            if n_rec > 0 {
                let step_i = step as i32;
                let mut rec_b = self.stream.launch_builder(&fn_rec);
                rec_b
                    .arg(&d_curr)
                    .arg(&d_rec)
                    .arg(&mut d_out)
                    .arg(&n_rec_i)
                    .arg(&step_i);
                unsafe { rec_b.launch(rec_cfg) }
                    .map_err(|e| format!("launch fdtd_record step {step}: {e}"))?;
            }
            let mut builder = self.stream.launch_builder(&fn_step);
            builder
                .arg(&d_prev)
                .arg(&d_curr)
                .arg(&mut d_next)
                .arg(&d_inv)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i)
                .arg(&lam2)
                .arg(&half_lam_beta);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("launch fdtd_3d step {step}: {e}"))?;
            std::mem::swap(&mut d_prev, &mut d_curr);
            std::mem::swap(&mut d_curr, &mut d_next);
        }

        if n_rec == 0 || steps == 0 {
            return Ok(vec![vec![0.0; steps]; n_rec]);
        }
        let host = self
            .stream
            .clone_dtoh(&d_out)
            .map_err(|e| format!("dtoh rec_out: {e}"))?;
        let mut out = vec![vec![0.0_f64; steps]; n_rec];
        for step in 0..steps {
            for (r, row) in out.iter_mut().enumerate() {
                row[step] = host[step * n_rec + r];
            }
        }
        Ok(out)
    }

    /// Execute a zero-FMA complex Helmholtz BEM matrix-vector product `y = A * x` on the GPU.
    ///
    /// One-shot convenience wrapper around [`Self::bem_dense_operator`]; the cached
    /// operator path used by GMRES uploads the matrix once for the whole solve.
    pub fn run_bem_matvec(
        &self,
        mat_re: &[f64],
        mat_im: &[f64],
        x_re: &[f64],
        x_im: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = x_re.len();
        let mut op = self.bem_dense_operator(mat_re, mat_im, n)?;
        op.apply(x_re, x_im)
    }

    /// Build a cached dense complex BEM operator: uploads the `n x n` row-major matrix
    /// exactly once, then [`CudaBemOperator::apply`] only ships/receives `2n` doubles.
    pub fn bem_dense_operator(
        &self,
        mat_re: &[f64],
        mat_im: &[f64],
        n: usize,
    ) -> Result<CudaBemOperator, String> {
        if n == 0 || mat_re.len() != n * n || mat_im.len() != n * n {
            return Err("BEM matvec dimension mismatch".into());
        }
        let func = self
            .bem_mod
            .load_function("g6me_bem_complex_matvec")
            .map_err(|e| format!("load g6me_bem_complex_matvec: {e}"))?;
        let d_mat_re = self
            .stream
            .clone_htod(mat_re)
            .map_err(|e| format!("htod mat_re: {e}"))?;
        let d_mat_im = self
            .stream
            .clone_htod(mat_im)
            .map_err(|e| format!("htod mat_im: {e}"))?;
        let (d_xr, d_xi, d_yr, d_yi) = self.bem_io_buffers(n)?;
        Ok(CudaBemOperator {
            stream: self.stream.clone(),
            n,
            d_xr,
            d_xi,
            d_yr,
            d_yi,
            kind: BemOperatorKind::Dense {
                d_mat_re,
                d_mat_im,
                func,
            },
        })
    }

    /// Build a cached zero-FMA FMM BEM operator for one `(solver, k, adjoint)` solve.
    /// All interaction coefficients are uploaded once; `apply` does no host `O(n^2)` work.
    pub fn bem_fmm_operator(
        &self,
        plan: &BemFmmPlan,
        adjoint: bool,
    ) -> Result<CudaBemOperator, String> {
        if plan.n == 0
            || plan.diag.len() != plan.n
            || plan.src_scale.len() != plan.n
            || plan.src_nx.len() != plan.n
            || plan.src_ny.len() != plan.n
            || plan.src_nz.len() != plan.n
            || plan.src_leaf.len() != plan.n
        {
            return Err("FMM BEM plan dimension mismatch".into());
        }
        if plan.child_start.len() != plan.node_count + 1
            || plan.leaf_start.len() != plan.node_count + 1
            || plan.level_offsets.len() != plan.max_level + 2
        {
            return Err("FMM BEM plan octree layout mismatch".into());
        }
        let (d_xr, d_xi, d_yr, d_yi) = self.bem_io_buffers(plan.n)?;
        let up = |v: &[f64], what: &str| -> Result<CudaSlice<f64>, String> {
            self.stream
                .clone_htod(v)
                .map_err(|e| format!("htod {what}: {e}"))
        };
        let upi = |v: &[i32], what: &str| -> Result<CudaSlice<i32>, String> {
            self.stream
                .clone_htod(v)
                .map_err(|e| format!("htod {what}: {e}"))
        };
        let level_ranges: Vec<(i32, i32)> = (0..plan.max_level + 1)
            .map(|l| {
                (
                    plan.level_offsets[l],
                    plan.level_offsets[l + 1] - plan.level_offsets[l],
                )
            })
            .collect();

        let kind = if adjoint {
            if plan.adj_src_start.len() != plan.n + 1
                || plan.adj_node_start.len() != plan.node_count + 1
            {
                return Err("FMM BEM adjoint plan layout mismatch".into());
            }
            let fn_nodes = self
                .bem_mod
                .load_function("g6me_bem_fmm_adjoint_nodes")
                .map_err(|e| format!("load g6me_bem_fmm_adjoint_nodes: {e}"))?;
            let fn_push = self
                .bem_mod
                .load_function("g6me_bem_fmm_adjoint_push")
                .map_err(|e| format!("load g6me_bem_fmm_adjoint_push: {e}"))?;
            let fn_sources = self
                .bem_mod
                .load_function("g6me_bem_fmm_adjoint_sources")
                .map_err(|e| format!("load g6me_bem_fmm_adjoint_sources: {e}"))?;
            BemOperatorKind::FmmAdjoint(Box::new(BemFmmAdjointDevice {
                node_count: plan.node_count,
                level_ranges,
                d_src_scale: up(&plan.src_scale, "src_scale")?,
                d_src_nx: up(&plan.src_nx, "src_nx")?,
                d_src_ny: up(&plan.src_ny, "src_ny")?,
                d_src_nz: up(&plan.src_nz, "src_nz")?,
                d_src_leaf: upi(&plan.src_leaf, "src_leaf")?,
                d_child_start: upi(&plan.child_start, "child_start")?,
                d_children: upi(&plan.children, "children")?,
                d_level_nodes: upi(&plan.level_nodes, "level_nodes")?,
                d_diag: up(&plan.diag, "diag")?,
                d_a_src_start: upi(&plan.adj_src_start, "adj_src_start")?,
                d_a_src_i: upi(&plan.adj_src_i, "adj_src_i")?,
                d_a_src_cr: up(&plan.adj_src_cr, "adj_src_cr")?,
                d_a_src_ci: up(&plan.adj_src_ci, "adj_src_ci")?,
                d_a_node_start: upi(&plan.adj_node_start, "adj_node_start")?,
                d_a_node_i: upi(&plan.adj_node_i, "adj_node_i")?,
                d_a_node_sr: up(&plan.adj_node_sr, "adj_node_sr")?,
                d_a_node_si: up(&plan.adj_node_si, "adj_node_si")?,
                d_a_node_dx: up(&plan.adj_node_dx, "adj_node_dx")?,
                d_a_node_dy: up(&plan.adj_node_dy, "adj_node_dy")?,
                d_a_node_dz: up(&plan.adj_node_dz, "adj_node_dz")?,
                d_grad: self
                    .stream
                    .alloc_zeros::<f64>(6 * plan.node_count)
                    .map_err(|e| format!("alloc adjoint grad: {e}"))?,
                fn_nodes,
                fn_push,
                fn_sources,
            }))
        } else {
            if plan.fwd_start.len() != plan.n + 1 {
                return Err("FMM BEM forward plan layout mismatch".into());
            }
            let fn_upward = self
                .bem_mod
                .load_function("g6me_bem_fmm_upward")
                .map_err(|e| format!("load g6me_bem_fmm_upward: {e}"))?;
            let fn_forward = self
                .bem_mod
                .load_function("g6me_bem_fmm_forward")
                .map_err(|e| format!("load g6me_bem_fmm_forward: {e}"))?;
            BemOperatorKind::FmmForward(Box::new(BemFmmForwardDevice {
                level_ranges,
                d_src_scale: up(&plan.src_scale, "src_scale")?,
                d_src_nx: up(&plan.src_nx, "src_nx")?,
                d_src_ny: up(&plan.src_ny, "src_ny")?,
                d_src_nz: up(&plan.src_nz, "src_nz")?,
                d_child_start: upi(&plan.child_start, "child_start")?,
                d_children: upi(&plan.children, "children")?,
                d_leaf_start: upi(&plan.leaf_start, "leaf_start")?,
                d_leaf_src: upi(&plan.leaf_src, "leaf_src")?,
                d_level_nodes: upi(&plan.level_nodes, "level_nodes")?,
                d_diag: up(&plan.diag, "diag")?,
                d_mom: self
                    .stream
                    .alloc_zeros::<f64>(6 * plan.node_count)
                    .map_err(|e| format!("alloc fmm moments: {e}"))?,
                d_f_start: upi(&plan.fwd_start, "fwd_start")?,
                d_f_kind: upi(&plan.fwd_kind, "fwd_kind")?,
                d_f_index: upi(&plan.fwd_index, "fwd_index")?,
                d_f_a: up(&plan.fwd_a, "fwd_a")?,
                d_f_b: up(&plan.fwd_b, "fwd_b")?,
                d_f_dx: up(&plan.fwd_dx, "fwd_dx")?,
                d_f_dy: up(&plan.fwd_dy, "fwd_dy")?,
                d_f_dz: up(&plan.fwd_dz, "fwd_dz")?,
                fn_upward,
                fn_forward,
            }))
        };

        Ok(CudaBemOperator {
            stream: self.stream.clone(),
            n: plan.n,
            d_xr,
            d_xi,
            d_yr,
            d_yi,
            kind,
        })
    }

    /// Assemble the exact closed-surface solid-angle diagonal on the GPU with a
    /// zero-FMA kernel (host precomputes nothing that depends on target/source pairs).
    /// Returns `None`-style errors so callers can fall back bit-identically to the CPU
    /// `par_map_indexed` assembly.
    pub fn bem_solid_angle_diag(
        &self,
        centroids: &[f64],
        normals: &[f64],
        areas: &[f64],
    ) -> Result<Vec<f64>, String> {
        let n = areas.len();
        if n == 0 || centroids.len() != 3 * n || normals.len() != 3 * n {
            return Err("BEM solid-angle dimension mismatch".into());
        }
        let d_cen = self
            .stream
            .clone_htod(centroids)
            .map_err(|e| format!("htod centroids: {e}"))?;
        let d_nrm = self
            .stream
            .clone_htod(normals)
            .map_err(|e| format!("htod normals: {e}"))?;
        let d_area = self
            .stream
            .clone_htod(areas)
            .map_err(|e| format!("htod areas: {e}"))?;
        let mut d_diag = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc solid-angle diag: {e}"))?;
        let func = self
            .bem_mod
            .load_function("g6me_bem_solid_angle_diag")
            .map_err(|e| format!("load g6me_bem_solid_angle_diag: {e}"))?;
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&d_cen)
            .arg(&d_nrm)
            .arg(&d_area)
            .arg(&mut d_diag)
            .arg(&n_i);
        unsafe { builder.launch(bem_grid(n)) }
            .map_err(|e| format!("launch solid-angle diag: {e}"))?;
        self.stream
            .clone_dtoh(&d_diag)
            .map_err(|e| format!("dtoh solid-angle diag: {e}"))
    }

    fn bem_io_buffers(&self, n: usize) -> Result<BemIoBuffers, String> {
        let _ = &self.ctx;
        let alloc = |what: &str| {
            self.stream
                .alloc_zeros::<f64>(n)
                .map_err(|e| format!("alloc {what}: {e}"))
        };
        Ok((
            alloc("bem x_re")?,
            alloc("bem x_im")?,
            alloc("bem y_re")?,
            alloc("bem y_im")?,
        ))
    }

    /// Execute the full 4-stage device-side Ozaki-II CRT FP64-on-Tensor-Core GEMM `C = A * B`
    /// on the GPU:
    /// 1. Device FP64 exponent extraction + 20-moduli symmetric residue reduction,
    /// 2. Device precision packing (`NVFP4 Base-13` / `FP8 Base-16` / `INT8` / `FP16`),
    /// 3. Hardware Tensor Core MMA residue GEMM across all 20 moduli in parallel (`blockIdx.z`),
    /// 4. Exact 128-bit integer CRT reconstruction → zero-FMA IEEE-754 `f64`.
    ///
    /// Returns `(c_row_major, actual_precision_used)`.
    pub fn run_ozaki2_crt_gemm(
        &self,
        a_row_major: &[f64],
        b_row_major: &[f64],
        m: usize,
        k: usize,
        n: usize,
        requested_precision: CudaTensorPrecision,
    ) -> Result<(Vec<f64>, CudaTensorPrecision), String> {
        if a_row_major.len() != m * k || b_row_major.len() != k * n {
            return Err("Ozaki-II GEMM dimension mismatch".into());
        }
        let prec = self.active_precision(requested_precision);
        if m == 0 || n == 0 || k == 0 {
            return Ok((vec![0.0; m * n], prec));
        }

        let shift_bits = ozaki2_shift_bits(k);
        let m_i = m as i32;
        let k_i = k as i32;
        let n_i = n as i32;

        // Upload A and B once; Stages 1–4 stay strictly device-resident.
        let d_a = self
            .stream
            .clone_htod(a_row_major)
            .map_err(|e| format!("htod A: {e}"))?;
        let d_b = self
            .stream
            .clone_htod(b_row_major)
            .map_err(|e| format!("htod B: {e}"))?;
        let mut d_exp_a = self
            .stream
            .alloc_zeros::<i32>(m)
            .map_err(|e| format!("alloc exp_a: {e}"))?;
        let mut d_exp_b = self
            .stream
            .alloc_zeros::<i32>(n)
            .map_err(|e| format!("alloc exp_b: {e}"))?;
        let mut d_a_sym = self
            .stream
            .alloc_zeros::<i8>(OZAKI2_NMOD * m * k)
            .map_err(|e| format!("alloc a_sym: {e}"))?;
        let mut d_b_sym = self
            .stream
            .alloc_zeros::<i8>(OZAKI2_NMOD * n * k)
            .map_err(|e| format!("alloc b_sym: {e}"))?;

        // --- Stage 1: Device FP64 exponent extraction + 20-moduli symmetric reduction ---
        let fn_red_a = self
            .ozaki2_mod
            .load_function("g6me_ozaki2_scale_reduce_a_f64")
            .map_err(|e| format!("load scale_reduce_a: {e}"))?;
        let fn_red_b = self
            .ozaki2_mod
            .load_function("g6me_ozaki2_scale_reduce_b_f64")
            .map_err(|e| format!("load scale_reduce_b: {e}"))?;
        {
            let cfg_a = LaunchConfig {
                grid_dim: ((m as u32).div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&fn_red_a);
            b.arg(&d_a)
                .arg(&mut d_exp_a)
                .arg(&mut d_a_sym)
                .arg(&m_i)
                .arg(&k_i)
                .arg(&shift_bits);
            unsafe { b.launch(cfg_a) }.map_err(|e| format!("launch scale_reduce_a: {e}"))?;
        }
        {
            let cfg_b = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&fn_red_b);
            b.arg(&d_b)
                .arg(&mut d_exp_b)
                .arg(&mut d_b_sym)
                .arg(&k_i)
                .arg(&n_i)
                .arg(&shift_bits);
            unsafe { b.launch(cfg_b) }.map_err(|e| format!("launch scale_reduce_b: {e}"))?;
        }

        let mut d_c_mod = self
            .stream
            .alloc_zeros::<i32>(OZAKI2_NMOD * m * n)
            .map_err(|e| format!("alloc c_mod: {e}"))?;

        let m_pad = align_up(m, 32);
        let n_pad = align_up(n, 32);
        let m_pad_i = m_pad as i32;
        let n_pad_i = n_pad as i32;
        // 4-warp / 128-thread 32x32 shared-memory CTA config for NVFP4 & INT8 (and Pascal DP4A)
        let mma_cfg_32x32 = LaunchConfig {
            grid_dim: ((n_pad / 32) as u32, (m_pad / 32) as u32, OZAKI2_NMOD as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // 1-warp 16x8 CTA config for optional FP8 / FP16 paths
        let mma_cfg = LaunchConfig {
            grid_dim: ((n_pad / 8) as u32, (m_pad / 16) as u32, OZAKI2_NMOD as u32),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        // --- Stage 2 & 3: Precision Pack + Hardware Tensor Core MMA ---
        match prec {
            CudaTensorPrecision::Nvfp4E2m1 => {
                let k_pad = align_up(k, 64);
                let k_pad_i = k_pad as i32;
                let k_words = k_pad / 8;
                let k_tiles = k_pad / 64;

                let mut d_a_lo = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_lo fp4: {e}"))?;
                let mut d_a_hi = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_hi fp4: {e}"))?;
                let mut d_sfa = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_tiles)
                    .map_err(|e| format!("alloc sfa: {e}"))?;

                let mut d_b_lo = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_lo fp4: {e}"))?;
                let mut d_b_hi = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_hi fp4: {e}"))?;
                let mut d_sfb = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_tiles)
                    .map_err(|e| format!("alloc sfb: {e}"))?;

                let fn_pack = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_pack_nvfp4_base13")
                    .map_err(|e| format!("load pack_nvfp4: {e}"))?;
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (m_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_a_sym)
                        .arg(&mut d_a_lo)
                        .arg(&mut d_a_hi)
                        .arg(&mut d_sfa)
                        .arg(&m_i)
                        .arg(&k_i)
                        .arg(&m_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_nvfp4 A: {e}"))?;
                }
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (n_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_b_sym)
                        .arg(&mut d_b_lo)
                        .arg(&mut d_b_hi)
                        .arg(&mut d_sfb)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&n_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_nvfp4 B: {e}"))?;
                }

                let fn_mma = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_nvfp4_mma_residue_gemm")
                    .map_err(|e| format!("load nvfp4_mma: {e}"))?;
                let mut b = self.stream.launch_builder(&fn_mma);
                b.arg(&d_a_lo)
                    .arg(&d_a_hi)
                    .arg(&d_b_lo)
                    .arg(&d_b_hi)
                    .arg(&d_sfa)
                    .arg(&d_sfb)
                    .arg(&mut d_c_mod)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&m_pad_i)
                    .arg(&n_pad_i)
                    .arg(&k_pad_i);
                unsafe { b.launch(mma_cfg_32x32) }.map_err(|e| format!("launch nvfp4_mma: {e}"))?;
            }
            CudaTensorPrecision::Fp8E4m3 => {
                let k_pad = align_up(k, 32);
                let k_pad_i = k_pad as i32;
                let k_words = k_pad / 4;

                let mut d_a_lo = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_lo fp8: {e}"))?;
                let mut d_a_hi = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_hi fp8: {e}"))?;
                let mut d_b_lo = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_lo fp8: {e}"))?;
                let mut d_b_hi = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_hi fp8: {e}"))?;

                let fn_pack = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_pack_fp8_base16")
                    .map_err(|e| format!("load pack_fp8: {e}"))?;
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (m_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_a_sym)
                        .arg(&mut d_a_lo)
                        .arg(&mut d_a_hi)
                        .arg(&m_i)
                        .arg(&k_i)
                        .arg(&m_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_fp8 A: {e}"))?;
                }
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (n_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_b_sym)
                        .arg(&mut d_b_lo)
                        .arg(&mut d_b_hi)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&n_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_fp8 B: {e}"))?;
                }

                let fn_mma = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_fp8_mma_residue_gemm")
                    .map_err(|e| format!("load fp8_mma: {e}"))?;
                let mut b = self.stream.launch_builder(&fn_mma);
                b.arg(&d_a_lo)
                    .arg(&d_a_hi)
                    .arg(&d_b_lo)
                    .arg(&d_b_hi)
                    .arg(&mut d_c_mod)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&m_pad_i)
                    .arg(&n_pad_i)
                    .arg(&k_pad_i);
                unsafe { b.launch(mma_cfg) }.map_err(|e| format!("launch fp8_mma: {e}"))?;
            }
            CudaTensorPrecision::Int8 | CudaTensorPrecision::Auto => {
                let k_pad = align_up(k, 32);
                let k_pad_i = k_pad as i32;
                let k_words = k_pad / 4;

                let mut d_a_i8 = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_i8: {e}"))?;
                let mut d_b_i8 = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_i8: {e}"))?;

                let fn_pack = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_pack_int8")
                    .map_err(|e| format!("load pack_int8: {e}"))?;
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (m_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_a_sym)
                        .arg(&mut d_a_i8)
                        .arg(&m_i)
                        .arg(&k_i)
                        .arg(&m_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_int8 A: {e}"))?;
                }
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (n_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_b_sym)
                        .arg(&mut d_b_i8)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&n_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_int8 B: {e}"))?;
                }

                let fn_mma = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_int8_mma_residue_gemm")
                    .map_err(|e| format!("load int8_mma: {e}"))?;
                let mut b = self.stream.launch_builder(&fn_mma);
                b.arg(&d_a_i8)
                    .arg(&d_b_i8)
                    .arg(&mut d_c_mod)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&m_pad_i)
                    .arg(&n_pad_i)
                    .arg(&k_pad_i);
                unsafe { b.launch(mma_cfg_32x32) }.map_err(|e| format!("launch int8_mma: {e}"))?;
            }
            CudaTensorPrecision::Fp16 => {
                let k_pad = align_up(k, 16);
                let k_pad_i = k_pad as i32;
                let k_words = k_pad / 2;

                let mut d_a_f16 = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * m_pad * k_words)
                    .map_err(|e| format!("alloc a_f16: {e}"))?;
                let mut d_b_f16 = self
                    .stream
                    .alloc_zeros::<u32>(OZAKI2_NMOD * n_pad * k_words)
                    .map_err(|e| format!("alloc b_f16: {e}"))?;

                let fn_pack = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_pack_fp16")
                    .map_err(|e| format!("load pack_fp16: {e}"))?;
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (m_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_a_sym)
                        .arg(&mut d_a_f16)
                        .arg(&m_i)
                        .arg(&k_i)
                        .arg(&m_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_fp16 A: {e}"))?;
                }
                {
                    let cfg = LaunchConfig {
                        grid_dim: (
                            (k_words as u32).div_ceil(16),
                            (n_pad as u32).div_ceil(16),
                            OZAKI2_NMOD as u32,
                        ),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = self.stream.launch_builder(&fn_pack);
                    b.arg(&d_b_sym)
                        .arg(&mut d_b_f16)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&n_pad_i)
                        .arg(&k_pad_i);
                    unsafe { b.launch(cfg) }.map_err(|e| format!("launch pack_fp16 B: {e}"))?;
                }

                let fn_mma = self
                    .ozaki2_mod
                    .load_function("g6me_ozaki2_fp16_mma_residue_gemm")
                    .map_err(|e| format!("load fp16_mma: {e}"))?;
                let mut b = self.stream.launch_builder(&fn_mma);
                b.arg(&d_a_f16)
                    .arg(&d_b_f16)
                    .arg(&mut d_c_mod)
                    .arg(&m_i)
                    .arg(&n_i)
                    .arg(&m_pad_i)
                    .arg(&n_pad_i)
                    .arg(&k_pad_i);
                unsafe { b.launch(mma_cfg) }.map_err(|e| format!("launch fp16_mma: {e}"))?;
            }
        }

        // --- Stage 4: Exact 128-bit integer CRT reconstruction → zero-FMA IEEE-754 f64 ---
        let mut d_out = self
            .stream
            .alloc_zeros::<f64>(m * n)
            .map_err(|e| format!("alloc out_f64: {e}"))?;
        let fn_crt = self
            .ozaki2_mod
            .load_function("g6me_ozaki2_crt_reconstruct_u128_f64")
            .map_err(|e| format!("load crt_reconstruct_u128: {e}"))?;
        let crt_cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(&fn_crt);
        b.arg(&d_c_mod)
            .arg(&d_exp_a)
            .arg(&d_exp_b)
            .arg(&mut d_out)
            .arg(&m_i)
            .arg(&n_i)
            .arg(&shift_bits);
        unsafe { b.launch(crt_cfg) }.map_err(|e| format!("launch crt_reconstruct_u128: {e}"))?;

        let out = self
            .stream
            .clone_dtoh(&d_out)
            .map_err(|e| format!("dtoh out_f64: {e}"))?;
        Ok((out, prec))
    }

    /// Execute the zero-FMA stochastic PCG32 acoustic ray-tracing pass on the GPU,
    /// specular-only (`wall_scatter` = 0). Prefer [`Self::run_raytrace_hoa_pass_with_scattering`]
    /// when the room materials carry ISO-17497 scattering.
    ///
    /// Returns `(ray_delays, ray_energies, ray_dirs)` of lengths `n_rays * max_bounces`
    /// (and `* 3`). Wall ordering is `[+x, -x, +y, -y, -z, +z]`, matching `RoomGeometry`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_raytrace_hoa_pass(
        &self,
        room_dims: [f64; 3],
        src_pos: [f64; 3],
        lst_pos: [f64; 3],
        wall_refl: [f64; 6],
        n_rays: usize,
        max_bounces: usize,
        c_air: f64,
        det_radius: f64,
        base_seed: u64,
    ) -> Result<RayTraceHoaOutput, String> {
        self.run_raytrace_hoa_pass_with_scattering(
            room_dims,
            src_pos,
            lst_pos,
            wall_refl,
            [0.0; 6],
            n_rays,
            max_bounces,
            c_air,
            det_radius,
            base_seed,
        )
    }

    /// Execute the zero-FMA stochastic PCG32 acoustic ray-tracing pass with per-wall
    /// ISO-17497 scattering coefficients.
    ///
    /// `wall_refl` is the mid-band energy reflection coefficient `1 - alpha_1k` and
    /// `wall_scatter` the Lambertian scattering probability, both in `RoomGeometry` wall
    /// ordering `[+x, -x, +y, -y, -z, +z]`. Each scattered ray samples the cosine-weighted
    /// hemisphere with a Marsaglia unit-disk rejection (no device `sin`/`cos`), so the
    /// trajectory stream is bit-identical to the CPU tracer in `g6me-core/src/room.rs`.
    ///
    /// Returns `(ray_delays, ray_energies, ray_dirs)` of lengths `n_rays * max_bounces`
    /// (and `* 3`); `ray_delays == -1.0` / `ray_energies == 0.0` marks a bounce with no
    /// receiver hit.
    #[allow(clippy::too_many_arguments)]
    pub fn run_raytrace_hoa_pass_with_scattering(
        &self,
        room_dims: [f64; 3],
        src_pos: [f64; 3],
        lst_pos: [f64; 3],
        wall_refl: [f64; 6],
        wall_scatter: [f64; 6],
        n_rays: usize,
        max_bounces: usize,
        c_air: f64,
        det_radius: f64,
        base_seed: u64,
    ) -> Result<RayTraceHoaOutput, String> {
        let total = n_rays * max_bounces;
        if total == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }
        let d_dims = self
            .stream
            .clone_htod(&room_dims)
            .map_err(|e| format!("htod room_dims: {e}"))?;
        let d_src = self
            .stream
            .clone_htod(&src_pos)
            .map_err(|e| format!("htod src_pos: {e}"))?;
        let d_lst = self
            .stream
            .clone_htod(&lst_pos)
            .map_err(|e| format!("htod lst_pos: {e}"))?;
        let d_refl = self
            .stream
            .clone_htod(&wall_refl)
            .map_err(|e| format!("htod wall_refl: {e}"))?;
        let d_scatter = self
            .stream
            .clone_htod(&wall_scatter)
            .map_err(|e| format!("htod wall_scatter: {e}"))?;
        let mut d_delays = self
            .stream
            .alloc_zeros::<f64>(total)
            .map_err(|e| format!("alloc ray_delays: {e}"))?;
        let mut d_energies = self
            .stream
            .alloc_zeros::<f64>(total)
            .map_err(|e| format!("alloc ray_energies: {e}"))?;
        let mut d_dirs = self
            .stream
            .alloc_zeros::<f64>(total * 3)
            .map_err(|e| format!("alloc ray_dirs: {e}"))?;

        let func = self
            .raytrace_mod
            .load_function("g6me_raytrace_hoa_pass")
            .map_err(|e| format!("load g6me_raytrace_hoa_pass: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((n_rays as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rays_i = n_rays as i32;
        let max_bounces_i = max_bounces as i32;
        let mut b = self.stream.launch_builder(&func);
        b.arg(&d_dims)
            .arg(&d_src)
            .arg(&d_lst)
            .arg(&d_refl)
            .arg(&d_scatter)
            .arg(&mut d_delays)
            .arg(&mut d_energies)
            .arg(&mut d_dirs)
            .arg(&n_rays_i)
            .arg(&max_bounces_i)
            .arg(&c_air)
            .arg(&det_radius)
            .arg(&base_seed);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch raytrace_hoa_pass: {e}"))?;

        let delays = self
            .stream
            .clone_dtoh(&d_delays)
            .map_err(|e| format!("dtoh ray_delays: {e}"))?;
        let energies = self
            .stream
            .clone_dtoh(&d_energies)
            .map_err(|e| format!("dtoh ray_energies: {e}"))?;
        let dirs = self
            .stream
            .clone_dtoh(&d_dirs)
            .map_err(|e| format!("dtoh ray_dirs: {e}"))?;
        Ok((delays, energies, dirs))
    }

    /// Execute a zero-FMA FP64 Stockham Radix-2 complex FFT (or inverse FFT with `1/N` scaling)
    /// completely on the GPU.
    ///
    /// Steady-state transfer pattern: `htod re` + `htod im` + `dtoh re` + `dtoh im`.
    /// Twiddles are uploaded once per `(n, inverse)` and reused from device memory for every
    /// subsequent block instead of being recomputed and copied on every call.
    pub fn run_stockham_fft_f64(
        &self,
        re: &[f64],
        im: &[f64],
        inverse: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = re.len();
        if im.len() != n || !n.is_power_of_two() || n == 0 {
            return Err("Stockham FFT requires equal power-of-two re/im lengths".into());
        }
        if n == 1 {
            return Ok((re.to_vec(), im.to_vec()));
        }
        let (d_re, d_im) = self.run_stockham_core(re, Some(im), inverse)?;
        let out_re = self
            .stream
            .clone_dtoh(&d_re)
            .map_err(|e| format!("dtoh fft re: {e}"))?;
        let out_im = self
            .stream
            .clone_dtoh(&d_im)
            .map_err(|e| format!("dtoh fft im: {e}"))?;
        Ok((out_re, out_im))
    }

    /// Forward real-input FFT (`im` is implicitly zero): the imaginary input is produced by a
    /// device `memset`, so only the real block crosses the host->device bus.
    ///
    /// Returns the full complex transform (the caller keeps `n/2 + 1` bins).
    /// Transfer pattern: `htod re` + `dtoh re` + `dtoh im`.
    pub fn run_stockham_fft_f64_real_input(
        &self,
        re: &[f64],
        inverse: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = re.len();
        if !n.is_power_of_two() || n == 0 {
            return Err("Stockham FFT requires a power-of-two length".into());
        }
        if n == 1 {
            return Ok((re.to_vec(), vec![0.0]));
        }
        let (d_re, d_im) = self.run_stockham_core(re, None, inverse)?;
        let out_re = self
            .stream
            .clone_dtoh(&d_re)
            .map_err(|e| format!("dtoh fft re: {e}"))?;
        let out_im = self
            .stream
            .clone_dtoh(&d_im)
            .map_err(|e| format!("dtoh fft im: {e}"))?;
        Ok((out_re, out_im))
    }

    /// Complex-input FFT returning only the real output. Used by the half-spectrum `irfft`
    /// reconstruction, which discards the imaginary part; skipping that `dtoh` halves the
    /// output traffic while leaving the kernel arithmetic identical to
    /// [`Self::run_stockham_fft_f64`].
    ///
    /// Transfer pattern: `htod re` + `htod im` + `dtoh re`.
    pub fn run_stockham_fft_f64_real_output(
        &self,
        re: &[f64],
        im: &[f64],
        inverse: bool,
    ) -> Result<Vec<f64>, String> {
        let n = re.len();
        if im.len() != n || !n.is_power_of_two() || n == 0 {
            return Err("Stockham FFT requires equal power-of-two re/im lengths".into());
        }
        if n == 1 {
            return Ok(re.to_vec());
        }
        let (d_re, _d_im) = self.run_stockham_core(re, Some(im), inverse)?;
        self.stream
            .clone_dtoh(&d_re)
            .map_err(|e| format!("dtoh fft re: {e}"))
    }

    /// Shared Stockham executor: upload the (optional) imaginary input, run `log2(n)` radix-2
    /// stages with device-cached twiddles, apply the `1/N` inverse scale, and return the
    /// device-resident result.
    fn run_stockham_core(
        &self,
        re: &[f64],
        im: Option<&[f64]>,
        inverse: bool,
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>), String> {
        let n = re.len();
        debug_assert!(n >= 2 && n.is_power_of_two());
        let mut d_in_re = self
            .stream
            .clone_htod(re)
            .map_err(|e| format!("htod fft re: {e}"))?;
        let mut d_in_im = match im {
            Some(im) => self
                .stream
                .clone_htod(im)
                .map_err(|e| format!("htod fft im: {e}"))?,
            // Real-input forward transform: zero the imaginary lane on the device instead of
            // uploading an `n`-element zero vector.
            None => self
                .stream
                .alloc_zeros::<f64>(n)
                .map_err(|e| format!("alloc fft im zeros: {e}"))?,
        };
        let mut d_out_re = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc fft out_re: {e}"))?;
        let mut d_out_im = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc fft out_im: {e}"))?;

        let fn_stage = self
            .fft_mod
            .load_function("g6me_stockham_radix2_f64_stage")
            .map_err(|e| format!("load stockham stage: {e}"))?;

        let half_n = n / 2;
        let half_n_i = half_n as i32;
        let cfg = LaunchConfig {
            grid_dim: ((half_n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        self.ensure_fft_twiddles(n, inverse)?;
        let guard = self
            .fft_twiddle_cache
            .lock()
            .map_err(|_| "fft twiddle cache poisoned".to_string())?;
        let stages = guard
            .get(&(n, inverse))
            .ok_or_else(|| "fft twiddle cache entry missing".to_string())?;
        for (stage_idx, (tw_re, tw_im)) in stages.iter().enumerate() {
            let s_i = (1usize << stage_idx) as i32;
            let m_i = (half_n >> stage_idx) as i32;
            let mut b = self.stream.launch_builder(&fn_stage);
            b.arg(&d_in_re)
                .arg(&d_in_im)
                .arg(tw_re)
                .arg(tw_im)
                .arg(&mut d_out_re)
                .arg(&mut d_out_im)
                .arg(&half_n_i)
                .arg(&s_i)
                .arg(&m_i);
            unsafe { b.launch(cfg) }
                .map_err(|e| format!("launch stockham stage {stage_idx}: {e}"))?;
            std::mem::swap(&mut d_in_re, &mut d_out_re);
            std::mem::swap(&mut d_in_im, &mut d_out_im);
        }
        drop(guard);

        if inverse {
            let fn_scale = self
                .fft_mod
                .load_function("g6me_stockham_scale_f64")
                .map_err(|e| format!("load stockham scale: {e}"))?;
            let scale_cfg = LaunchConfig {
                grid_dim: ((n as u32).div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let inv_n = 1.0 / (n as f64);
            let n_i = n as i32;
            let mut b = self.stream.launch_builder(&fn_scale);
            b.arg(&mut d_in_re).arg(&mut d_in_im).arg(&inv_n).arg(&n_i);
            unsafe { b.launch(scale_cfg) }.map_err(|e| format!("launch stockham scale: {e}"))?;
        }
        Ok((d_in_re, d_in_im))
    }

    /// Execute a zero-FMA complex spectrum multiply-accumulate `acc += a * b` on the GPU.
    pub fn run_complex_mac_f64(
        &self,
        a_re: &[f64],
        a_im: &[f64],
        b_re: &[f64],
        b_im: &[f64],
        acc_re: &[f64],
        acc_im: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = a_re.len();
        if a_im.len() != n
            || b_re.len() != n
            || b_im.len() != n
            || acc_re.len() != n
            || acc_im.len() != n
        {
            return Err("complex MAC buffer length mismatch".into());
        }
        let d_ar = self
            .stream
            .clone_htod(a_re)
            .map_err(|e| format!("htod a_re: {e}"))?;
        let d_ai = self
            .stream
            .clone_htod(a_im)
            .map_err(|e| format!("htod a_im: {e}"))?;
        let d_br = self
            .stream
            .clone_htod(b_re)
            .map_err(|e| format!("htod b_re: {e}"))?;
        let d_bi = self
            .stream
            .clone_htod(b_im)
            .map_err(|e| format!("htod b_im: {e}"))?;
        let mut d_acc_r = self
            .stream
            .clone_htod(acc_re)
            .map_err(|e| format!("htod acc_re: {e}"))?;
        let mut d_acc_i = self
            .stream
            .clone_htod(acc_im)
            .map_err(|e| format!("htod acc_im: {e}"))?;

        let func = self
            .fft_mod
            .load_function("g6me_complex_mac_f64")
            .map_err(|e| format!("load complex_mac_f64: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&func);
        b.arg(&d_ar)
            .arg(&d_ai)
            .arg(&d_br)
            .arg(&d_bi)
            .arg(&mut d_acc_r)
            .arg(&mut d_acc_i)
            .arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch complex_mac_f64: {e}"))?;

        let out_r = self
            .stream
            .clone_dtoh(&d_acc_r)
            .map_err(|e| format!("dtoh acc_re: {e}"))?;
        let out_i = self
            .stream
            .clone_dtoh(&d_acc_i)
            .map_err(|e| format!("dtoh acc_im: {e}"))?;
        Ok((out_r, out_i))
    }

    /// Execute exact 64-bit Goldilocks prime NTT block convolution on the GPU:
    /// `INTT(NTT(a_field) .* ir_ntt_domain)` with strictly 0 floating-point operations (0 ULP).
    pub fn run_goldilocks_ntt_convolve(
        &self,
        a_field: &[u64],
        ir_ntt_domain: &[u64],
        twiddles_fwd: &[Vec<u64>],
        twiddles_inv: &[Vec<u64>],
        inv_n: u64,
    ) -> Result<Vec<u64>, String> {
        let n = a_field.len();
        if ir_ntt_domain.len() != n || !n.is_power_of_two() || n == 0 {
            return Err("Goldilocks NTT requires equal power-of-two lengths".into());
        }
        let log_n = n.trailing_zeros() as i32;
        if twiddles_fwd.len() != log_n as usize || twiddles_inv.len() != log_n as usize {
            return Err("Goldilocks NTT twiddle stage count mismatch".into());
        }
        let n_i = n as i32;
        let d_src = self
            .stream
            .clone_htod(a_field)
            .map_err(|e| format!("htod ntt a: {e}"))?;
        let d_ir = self
            .stream
            .clone_htod(ir_ntt_domain)
            .map_err(|e| format!("htod ntt ir: {e}"))?;
        let mut d_buf = self
            .stream
            .alloc_zeros::<u64>(n)
            .map_err(|e| format!("alloc ntt buf: {e}"))?;
        let mut d_tmp = self
            .stream
            .alloc_zeros::<u64>(n)
            .map_err(|e| format!("alloc ntt tmp: {e}"))?;

        let fn_bitrev = self
            .ntt_mod
            .load_function("g6me_goldilocks_bitrev")
            .map_err(|e| format!("load goldilocks_bitrev: {e}"))?;
        let fn_stage = self
            .ntt_mod
            .load_function("g6me_goldilocks_ntt_stage")
            .map_err(|e| format!("load goldilocks_ntt_stage: {e}"))?;
        let fn_mul = self
            .ntt_mod
            .load_function("g6me_goldilocks_pointwise_mul")
            .map_err(|e| format!("load goldilocks_pointwise_mul: {e}"))?;
        let fn_scale = self
            .ntt_mod
            .load_function("g6me_goldilocks_scale")
            .map_err(|e| format!("load goldilocks_scale: {e}"))?;

        let cfg_n = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_half = LaunchConfig {
            grid_dim: (((n / 2).max(1) as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        // Forward NTT: bitrev(d_src -> d_buf) + log2(N) butterfly stages
        {
            let mut b = self.stream.launch_builder(&fn_bitrev);
            b.arg(&d_src).arg(&mut d_buf).arg(&n_i).arg(&log_n);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt bitrev fwd: {e}"))?;
        }
        // Device-cached forward twiddles: uploaded once per (n, table) instead of per block.
        self.ensure_ntt_twiddles(n, false, twiddles_fwd)?;
        {
            let guard = self
                .ntt_twiddle_cache
                .lock()
                .map_err(|_| "ntt twiddle cache poisoned".to_string())?;
            let stages = guard
                .get(&ntt_twiddle_key(n, false, twiddles_fwd))
                .ok_or_else(|| "ntt twiddle cache entry missing".to_string())?;
            for (stage_idx, d_tw) in stages.iter().enumerate() {
                let half_len = (1usize << stage_idx) as i32;
                let mut b = self.stream.launch_builder(&fn_stage);
                b.arg(&mut d_buf).arg(d_tw).arg(&n_i).arg(&half_len);
                unsafe { b.launch(cfg_half) }
                    .map_err(|e| format!("launch ntt fwd stage {stage_idx}: {e}"))?;
            }
        }

        // Pointwise multiply in NTT domain
        {
            let mut b = self.stream.launch_builder(&fn_mul);
            b.arg(&mut d_buf).arg(&d_ir).arg(&n_i);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt pointwise_mul: {e}"))?;
        }

        // Inverse NTT: bitrev(d_buf -> d_tmp) + log2(N) inverse stages + scale by inv_n
        {
            let mut b = self.stream.launch_builder(&fn_bitrev);
            b.arg(&d_buf).arg(&mut d_tmp).arg(&n_i).arg(&log_n);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt bitrev inv: {e}"))?;
        }
        // Device-cached inverse twiddles: uploaded once per (n, table) instead of per block.
        self.ensure_ntt_twiddles(n, true, twiddles_inv)?;
        {
            let guard = self
                .ntt_twiddle_cache
                .lock()
                .map_err(|_| "ntt twiddle cache poisoned".to_string())?;
            let stages = guard
                .get(&ntt_twiddle_key(n, true, twiddles_inv))
                .ok_or_else(|| "ntt twiddle cache entry missing".to_string())?;
            for (stage_idx, d_tw) in stages.iter().enumerate() {
                let half_len = (1usize << stage_idx) as i32;
                let mut b = self.stream.launch_builder(&fn_stage);
                b.arg(&mut d_tmp).arg(d_tw).arg(&n_i).arg(&half_len);
                unsafe { b.launch(cfg_half) }
                    .map_err(|e| format!("launch ntt inv stage {stage_idx}: {e}"))?;
            }
        }
        {
            let mut b = self.stream.launch_builder(&fn_scale);
            b.arg(&mut d_tmp).arg(&inv_n).arg(&n_i);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt scale: {e}"))?;
        }

        self.stream
            .clone_dtoh(&d_tmp)
            .map_err(|e| format!("dtoh ntt out: {e}"))
    }

    /// Execute a single forward or inverse 64-bit Goldilocks prime NTT on the GPU
    /// (passing `scale = Some(inv_n)` for inverse NTT), strictly 0 FP operations (0 ULP).
    ///
    /// This is the fused single-launch path from `docs/CUDA_PERF_PLAN.md` section 1: when
    /// the whole transform fits one block's opt-in dynamic shared memory, the bit reversal,
    /// all `log2(n)` butterfly stages and the inverse scale run in one kernel launch with one
    /// H2D and one D2H. Above the shared-memory limit (or if the driver rejects the opt-in)
    /// it falls back to the same batched multi-launch sequence as before; both paths compute
    /// the identical canonical field elements.
    pub fn run_goldilocks_ntt_transform(
        &self,
        data: &[u64],
        twiddles: &[Vec<u64>],
        scale: Option<u64>,
    ) -> Result<Vec<u64>, String> {
        let n = data.len();
        if !n.is_power_of_two() || n == 0 {
            return Err("Goldilocks NTT requires power-of-two length".into());
        }
        if n == 1 {
            return Ok(data.to_vec());
        }
        if n.saturating_mul(std::mem::size_of::<u64>()) <= self.max_dynamic_shared {
            match self.run_goldilocks_ntt_transform_fused(data, twiddles, scale) {
                Ok(out) => return Ok(out),
                Err(fused_err) => {
                    // The plan's documented fallback: if the driver refuses the dynamic
                    // shared-memory opt-in or the launch, use the multi-launch sequence.
                    return self
                        .run_goldilocks_ntt_transform_legacy(data, twiddles, scale)
                        .map_err(|legacy_err| {
                            format!(
                                "fused NTT failed ({fused_err}); legacy NTT failed ({legacy_err})"
                            )
                        });
                }
            }
        }
        self.run_goldilocks_ntt_transform_legacy(data, twiddles, scale)
    }

    /// One-launch forward/inverse NTT: bit reversal, all stages and the optional inverse
    /// scale inside a single block's dynamic shared memory (`n * 8` bytes).
    fn run_goldilocks_ntt_transform_fused(
        &self,
        data: &[u64],
        twiddles: &[Vec<u64>],
        scale: Option<u64>,
    ) -> Result<Vec<u64>, String> {
        let n = data.len();
        let log_n = n.trailing_zeros() as i32;
        if twiddles.len() != log_n as usize {
            return Err("Goldilocks NTT twiddle stage count mismatch".into());
        }
        let shared_mem_bytes = std::mem::size_of_val(data) as u32;
        if shared_mem_bytes as usize > self.max_dynamic_shared {
            return Err(format!(
                "fused NTT needs {shared_mem_bytes} B dynamic shared memory, device allows {} B",
                self.max_dynamic_shared
            ));
        }
        let inverse = scale.is_some();
        let key = self.ensure_fused_ntt_twiddles(n, inverse, twiddles)?;
        let mut d_data = self
            .stream
            .clone_htod(data)
            .map_err(|e| format!("htod fused ntt data: {e}"))?;
        let func = self
            .ntt_mod
            .load_function("g6me_goldilocks_ntt_fused_inplace")
            .map_err(|e| format!("load goldilocks_ntt_fused_inplace: {e}"))?;
        func.set_attribute(
            sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            shared_mem_bytes as i32,
        )
        .map_err(|e| format!("set fused ntt max dynamic shared ({shared_mem_bytes} B): {e}"))?;
        let inv_n = scale.unwrap_or(0);
        let has_scale: i32 = i32::from(scale.is_some());
        let all_channels: u32 = u32::MAX;
        {
            let guard = self
                .ntt_fused_twiddle_cache
                .lock()
                .map_err(|_| "ntt fused twiddle cache poisoned".to_string())?;
            let d_tw = guard
                .get(&key)
                .ok_or_else(|| "fused ntt twiddle cache entry missing".to_string())?;
            let mut b = self.stream.launch_builder(&func);
            b.arg(&mut d_data)
                .arg(d_tw)
                .arg(&log_n)
                .arg(&inv_n)
                .arg(&has_scale)
                .arg(&all_channels);
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (NTT_FUSED_THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe { b.launch(cfg) }.map_err(|e| format!("launch fused ntt: {e}"))?;
        }
        self.stream
            .clone_dtoh(&d_data)
            .map_err(|e| format!("dtoh fused ntt out: {e}"))
    }

    /// Original multi-launch transform (bit-reversal kernel, one launch per butterfly stage,
    /// optional scale kernel). Kept as the above-limit fallback and as the before/after
    /// measurement reference for the fused path.
    fn run_goldilocks_ntt_transform_legacy(
        &self,
        data: &[u64],
        twiddles: &[Vec<u64>],
        scale: Option<u64>,
    ) -> Result<Vec<u64>, String> {
        let n = data.len();
        if !n.is_power_of_two() || n == 0 {
            return Err("Goldilocks NTT requires power-of-two length".into());
        }
        if n == 1 {
            return Ok(data.to_vec());
        }
        let log_n = n.trailing_zeros() as i32;
        if twiddles.len() != log_n as usize {
            return Err("Goldilocks NTT twiddle stage count mismatch".into());
        }
        let n_i = n as i32;
        let d_src = self
            .stream
            .clone_htod(data)
            .map_err(|e| format!("htod ntt data: {e}"))?;
        let mut d_buf = self
            .stream
            .alloc_zeros::<u64>(n)
            .map_err(|e| format!("alloc ntt buf: {e}"))?;

        let fn_bitrev = self
            .ntt_mod
            .load_function("g6me_goldilocks_bitrev")
            .map_err(|e| format!("load goldilocks_bitrev: {e}"))?;
        let fn_stage = self
            .ntt_mod
            .load_function("g6me_goldilocks_ntt_stage")
            .map_err(|e| format!("load goldilocks_ntt_stage: {e}"))?;

        let cfg_n = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_half = LaunchConfig {
            grid_dim: (((n / 2).max(1) as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        {
            let mut b = self.stream.launch_builder(&fn_bitrev);
            b.arg(&d_src).arg(&mut d_buf).arg(&n_i).arg(&log_n);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt bitrev: {e}"))?;
        }
        // Device-cached twiddles (`scale.is_some()` == inverse direction). Steady-state transfer
        // pattern per call: `htod data` + `dtoh out` only.
        let inverse = scale.is_some();
        self.ensure_ntt_twiddles(n, inverse, twiddles)?;
        {
            let guard = self
                .ntt_twiddle_cache
                .lock()
                .map_err(|_| "ntt twiddle cache poisoned".to_string())?;
            let stages = guard
                .get(&ntt_twiddle_key(n, inverse, twiddles))
                .ok_or_else(|| "ntt twiddle cache entry missing".to_string())?;
            for (stage_idx, d_tw) in stages.iter().enumerate() {
                let half_len = (1usize << stage_idx) as i32;
                let mut b = self.stream.launch_builder(&fn_stage);
                b.arg(&mut d_buf).arg(d_tw).arg(&n_i).arg(&half_len);
                unsafe { b.launch(cfg_half) }
                    .map_err(|e| format!("launch ntt stage {stage_idx}: {e}"))?;
            }
        }
        if let Some(inv_n) = scale {
            let fn_scale = self
                .ntt_mod
                .load_function("g6me_goldilocks_scale")
                .map_err(|e| format!("load goldilocks_scale: {e}"))?;
            let mut b = self.stream.launch_builder(&fn_scale);
            b.arg(&mut d_buf).arg(&inv_n).arg(&n_i);
            unsafe { b.launch(cfg_n) }.map_err(|e| format!("launch ntt scale: {e}"))?;
        }

        self.stream
            .clone_dtoh(&d_buf)
            .map_err(|e| format!("dtoh ntt out: {e}"))
    }

    /// Build the device-resident FDL-UPOLA state of `docs/CUDA_PERF_PLAN.md` section 1.
    ///
    /// Everything that is a pure function of the preset is uploaded exactly once:
    /// * `kern_l` / `kern_r` — flat `[n_parts][slots][n]` NTT-domain kernels,
    /// * the concatenated forward/inverse twiddle stages,
    /// * the frequency-domain delay line `[n_parts - 1][slots][n]`,
    /// * both ear accumulators (`[2][n]`) and the per-channel hop scratch (`[slots][n]`).
    ///
    /// A streaming block then runs forward transforms, pointwise accumulation and both
    /// inverse transforms entirely on the device; only the inverse-NTT result is read back.
    #[allow(clippy::too_many_arguments)]
    pub fn create_upola_state(
        &self,
        n: usize,
        slots: usize,
        n_parts: usize,
        inv_n: u64,
        kern_l: &[u64],
        kern_r: &[u64],
        twiddles_fwd: &[Vec<u64>],
        twiddles_inv: &[Vec<u64>],
    ) -> Result<CudaUpolaState, String> {
        if !n.is_power_of_two() || n < 2 {
            return Err(format!("UPOLA NTT length {n} is not a power of two >= 2"));
        }
        if slots == 0 || n_parts == 0 {
            return Err("UPOLA state requires slots > 0 and n_parts > 0".into());
        }
        let log_n = n.trailing_zeros() as i32;
        let shared_mem_bytes = (n * std::mem::size_of::<u64>()) as u32;
        if shared_mem_bytes as usize > self.max_dynamic_shared {
            return Err(format!(
                "UPOLA NTT N={n} needs {shared_mem_bytes} B dynamic shared memory, device allows {} B",
                self.max_dynamic_shared
            ));
        }
        let elem = n_parts
            .checked_mul(slots)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| "UPOLA kernel table length overflow".to_string())?;
        if kern_l.len() != elem || kern_r.len() != elem {
            return Err(format!(
                "UPOLA kernel tables must be {elem} words ([{n_parts}][{slots}][{n}]), got {} and {}",
                kern_l.len(),
                kern_r.len()
            ));
        }
        let fdl_slots = n_parts - 1;
        let fdl_alloc = (fdl_slots * slots * n).max(1);
        let acc_alloc = 2 * n;

        // Concatenate and upload the twiddle stages (the same bytes the legacy path reads).
        let mut tw_fwd_host = Vec::with_capacity(n - 1);
        let mut tw_inv_host = Vec::with_capacity(n - 1);
        for (stage, (fwd, inv)) in twiddles_fwd.iter().zip(twiddles_inv).enumerate() {
            let want = 1usize << stage;
            if fwd.len() != want || inv.len() != want {
                return Err(format!(
                    "UPOLA twiddle stage {stage} length ({}, {}) != {want}",
                    fwd.len(),
                    inv.len()
                ));
            }
            tw_fwd_host.extend_from_slice(fwd);
            tw_inv_host.extend_from_slice(inv);
        }
        if twiddles_fwd.len() != log_n as usize
            || twiddles_inv.len() != log_n as usize
            || tw_fwd_host.len() != n - 1
            || tw_inv_host.len() != n - 1
        {
            return Err("UPOLA twiddle stage count/length mismatch".into());
        }

        let d_fdl = self
            .stream
            .alloc_zeros::<u64>(fdl_alloc)
            .map_err(|e| format!("alloc upola fdl: {e}"))?;
        let d_kern_l = self
            .stream
            .clone_htod(kern_l)
            .map_err(|e| format!("htod upola kern_l: {e}"))?;
        let d_kern_r = self
            .stream
            .clone_htod(kern_r)
            .map_err(|e| format!("htod upola kern_r: {e}"))?;
        let d_acc = self
            .stream
            .alloc_zeros::<u64>(acc_alloc)
            .map_err(|e| format!("alloc upola acc: {e}"))?;
        let d_scratch = self
            .stream
            .alloc_zeros::<u64>(slots * n)
            .map_err(|e| format!("alloc upola scratch: {e}"))?;
        let d_tw_fwd = self
            .stream
            .clone_htod(&tw_fwd_host)
            .map_err(|e| format!("htod upola tw_fwd: {e}"))?;
        let d_tw_inv = self
            .stream
            .clone_htod(&tw_inv_host)
            .map_err(|e| format!("htod upola tw_inv: {e}"))?;

        let fn_fused = self
            .ntt_mod
            .load_function("g6me_goldilocks_ntt_fused_inplace")
            .map_err(|e| format!("load goldilocks_ntt_fused_inplace: {e}"))?;
        fn_fused
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_mem_bytes as i32,
            )
            .map_err(|e| {
                format!("set upola fused max dynamic shared ({shared_mem_bytes} B): {e}")
            })?;
        let fn_accum = self
            .ntt_mod
            .load_function("g6me_goldilocks_upola_accumulate")
            .map_err(|e| format!("load goldilocks_upola_accumulate: {e}"))?;
        let fn_mac_row = self
            .ntt_mod
            .load_function("g6me_goldilocks_upola_mac_scratch")
            .map_err(|e| format!("load goldilocks_upola_mac_scratch: {e}"))?;

        Ok(CudaUpolaState {
            stream: self.stream.clone(),
            n,
            log_n,
            slots,
            n_parts,
            fdl_slots,
            inv_n,
            d_fdl,
            d_kern_l,
            d_kern_r,
            d_acc,
            d_scratch,
            d_tw_fwd,
            d_tw_inv,
            fn_fused,
            fn_accum,
            fn_mac_row,
        })
    }
}

/// Device-resident FDL-UPOLA state for the exact Goldilocks NTT engine.
///
/// The delay line, NTT-domain kernels, both ear accumulators, the per-hop scratch row and
/// the twiddle stages all live in VRAM, so a streaming block costs only one H2D per active
/// channel and one D2H for the final inverse transform, instead of the per-transform host
/// round trip of [`G6meCudaRuntime::run_goldilocks_ntt_transform`]. Every operation is exact
/// integer modular arithmetic in arrival order, so results are bit-identical to the CPU
/// `BinauralPlus` UPOLA loops.
pub struct CudaUpolaState {
    stream: Arc<CudaStream>,
    n: usize,
    log_n: i32,
    slots: usize,
    n_parts: usize,
    fdl_slots: usize,
    inv_n: u64,
    /// `[fdl_slots][slots][n]` frequency-domain delay line.
    d_fdl: CudaSlice<u64>,
    /// `[n_parts][slots][n]` NTT-domain left/right kernels.
    d_kern_l: CudaSlice<u64>,
    d_kern_r: CudaSlice<u64>,
    /// `[2][n]` — left ear at `0..n`, right ear at `n..2n`.
    d_acc: CudaSlice<u64>,
    /// `[slots][n]` transform scratch for sub-block / single-partition hops.
    d_scratch: CudaSlice<u64>,
    d_tw_fwd: CudaSlice<u64>,
    d_tw_inv: CudaSlice<u64>,
    fn_fused: CudaFunction,
    fn_accum: CudaFunction,
    fn_mac_row: CudaFunction,
}

impl std::fmt::Debug for CudaUpolaState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaUpolaState")
            .field("n", &self.n)
            .field("slots", &self.slots)
            .field("n_parts", &self.n_parts)
            .field("fdl_slots", &self.fdl_slots)
            .finish_non_exhaustive()
    }
}

impl CudaUpolaState {
    /// Transform length (`N_FFT`).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of uniform partitions.
    pub fn n_parts(&self) -> usize {
        self.n_parts
    }

    /// Number of channel slots.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Zero the whole delay line and accumulator bank (stream reset / preset reload).
    pub fn reset(&mut self) -> Result<(), String> {
        self.stream
            .memset_zeros(&mut self.d_fdl)
            .map_err(|e| format!("memset upola fdl: {e}"))?;
        self.stream
            .memset_zeros(&mut self.d_acc)
            .map_err(|e| format!("memset upola acc: {e}"))
    }

    fn check_fdl_row(&self, head: usize) -> Result<(), String> {
        if self.fdl_slots == 0 || head >= self.fdl_slots {
            return Err(format!(
                "UPOLA head {head} out of range for {} delay-line slots",
                self.fdl_slots
            ));
        }
        Ok(())
    }

    /// Zero one delay-line row (`[slots][n]`), the exact analogue of the CPU
    /// `ntt_fdl[head][ch].fill(0)` for silent or absent channels.
    pub fn zero_fdl_row(&mut self, head: usize) -> Result<(), String> {
        self.check_fdl_row(head)?;
        let begin = head * self.slots * self.n;
        let end = begin + self.slots * self.n;
        let mut row = self.d_fdl.slice_mut(begin..end);
        self.stream
            .memset_zeros(&mut row)
            .map_err(|e| format!("memset upola fdl row: {e}"))
    }

    /// Upload one quantized time-domain block (exactly `n` words, zero padded) into the
    /// delay line row for channel `ch`.
    pub fn upload_fdl_row(&mut self, head: usize, ch: usize, data: &[u64]) -> Result<(), String> {
        self.check_fdl_row(head)?;
        if ch >= self.slots || data.len() != self.n {
            return Err(format!(
                "UPOLA fdl upload ch={ch} len={} (slots={}, n={})",
                data.len(),
                self.slots,
                self.n
            ));
        }
        let begin = (head * self.slots + ch) * self.n;
        let mut dst = self.d_fdl.slice_mut(begin..begin + self.n);
        self.stream
            .memcpy_htod(data, &mut dst)
            .map_err(|e| format!("htod upola fdl row: {e}"))
    }

    /// Upload one quantized hop into the per-channel scratch row (sub-block path).
    pub fn upload_scratch_row(&mut self, ch: usize, data: &[u64]) -> Result<(), String> {
        if ch >= self.slots || data.len() != self.n {
            return Err(format!(
                "UPOLA scratch upload ch={ch} len={} (slots={}, n={})",
                data.len(),
                self.slots,
                self.n
            ));
        }
        let begin = ch * self.n;
        let mut dst = self.d_scratch.slice_mut(begin..begin + self.n);
        self.stream
            .memcpy_htod(data, &mut dst)
            .map_err(|e| format!("htod upola scratch row: {e}"))
    }

    /// Forward-transform every channel row of the scratch bank whose mask bit is set.
    pub fn forward_scratch(&mut self, channel_mask: u32) -> Result<(), String> {
        let log_n = self.log_n;
        let shared_mem_bytes = (self.n * std::mem::size_of::<u64>()) as u32;
        let zero_scale = 0u64;
        let no_scale = 0i32;
        let batch = self.slots.max(1) as u32;
        let mut builder = self.stream.launch_builder(&self.fn_fused);
        builder
            .arg(&mut self.d_scratch)
            .arg(&self.d_tw_fwd)
            .arg(&log_n)
            .arg(&zero_scale)
            .arg(&no_scale)
            .arg(&channel_mask);
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (NTT_FUSED_THREADS, 1, 1),
            shared_mem_bytes,
        };
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch upola forward scratch: {e}"))?;
        Ok(())
    }

    /// Forward-transform every channel row of delay-line row `head` whose mask bit is set.
    pub fn forward_fdl_row(&mut self, head: usize, channel_mask: u32) -> Result<(), String> {
        self.check_fdl_row(head)?;
        let log_n = self.log_n;
        let shared_mem_bytes = (self.n * std::mem::size_of::<u64>()) as u32;
        let zero_scale = 0u64;
        let no_scale = 0i32;
        let begin = head * self.slots * self.n;
        let end = begin + self.slots * self.n;
        let batch = self.slots.max(1) as u32;
        let mut row = self.d_fdl.slice_mut(begin..end);
        let mut builder = self.stream.launch_builder(&self.fn_fused);
        builder
            .arg(&mut row)
            .arg(&self.d_tw_fwd)
            .arg(&log_n)
            .arg(&zero_scale)
            .arg(&no_scale)
            .arg(&channel_mask);
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (NTT_FUSED_THREADS, 1, 1),
            shared_mem_bytes,
        };
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch upola forward fdl: {e}"))?;
        Ok(())
    }

    /// Exact tail accumulation from the delay line into both ear accumulators.
    ///
    /// `p_start = 0, p_end = 1` reproduces the CPU partition-0 accumulate from the completed
    /// row at `head`; `p_start = 1, p_end = n_parts` reproduces the tail sum over the
    /// `n_parts - 1` stored rows, walking back in time exactly as the CPU loop does.
    pub fn accumulate_fdl(
        &mut self,
        head: usize,
        p_start: usize,
        p_end: usize,
        channel_mask: u32,
    ) -> Result<(), String> {
        self.check_fdl_row(head)?;
        if p_start > p_end || p_end > self.n_parts {
            return Err(format!(
                "UPOLA accumulate range {p_start}..{p_end} outside 0..{}",
                self.n_parts
            ));
        }
        let n_i = self.n as i32;
        let slots_i = self.slots as i32;
        let fdl_i = self.fdl_slots as i32;
        let n_parts_i = self.n_parts as i32;
        let ps_i = p_start as i32;
        let pe_i = p_end as i32;
        let head_i = head as i32;
        let cfg = LaunchConfig {
            grid_dim: ((self.n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.fn_accum);
        builder
            .arg(&self.d_fdl)
            .arg(&self.d_kern_l)
            .arg(&self.d_kern_r)
            .arg(&mut self.d_acc)
            .arg(&n_i)
            .arg(&slots_i)
            .arg(&fdl_i)
            .arg(&n_parts_i)
            .arg(&ps_i)
            .arg(&pe_i)
            .arg(&head_i)
            .arg(&channel_mask);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch upola accumulate: {e}"))?;
        Ok(())
    }

    /// Partition-0 accumulation from the scratch bank (sub-block and single-partition hops).
    pub fn mac_scratch_partition0(&mut self, channel_mask: u32) -> Result<(), String> {
        let n_i = self.n as i32;
        let slots_i = self.slots as i32;
        let cfg = LaunchConfig {
            grid_dim: ((self.n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.stream.launch_builder(&self.fn_mac_row);
        builder
            .arg(&self.d_scratch)
            .arg(&self.d_kern_l)
            .arg(&self.d_kern_r)
            .arg(&mut self.d_acc)
            .arg(&n_i)
            .arg(&slots_i)
            .arg(&channel_mask);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch upola mac scratch: {e}"))?;
        Ok(())
    }

    /// Inverse-transform both ear accumulators in one launch (one block per ear) and return
    /// `2 * n` host words: left ear at `0..n`, right ear at `n..2n`.
    pub fn inverse_acc_to_host(&mut self) -> Result<Vec<u64>, String> {
        let log_n = self.log_n;
        let inv_n = self.inv_n;
        let has_scale = 1i32;
        let all_channels = u32::MAX;
        let shared_mem_bytes = (self.n * std::mem::size_of::<u64>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (2, 1, 1),
            block_dim: (NTT_FUSED_THREADS, 1, 1),
            shared_mem_bytes,
        };
        let mut builder = self.stream.launch_builder(&self.fn_fused);
        builder
            .arg(&mut self.d_acc)
            .arg(&self.d_tw_inv)
            .arg(&log_n)
            .arg(&inv_n)
            .arg(&has_scale)
            .arg(&all_channels);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch upola inverse: {e}"))?;
        self.stream
            .clone_dtoh(&self.d_acc)
            .map_err(|e| format!("dtoh upola inverse: {e}"))
    }
}

impl CudaBemOperator {
    /// Number of BEM elements this operator acts on.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Apply the cached zero-FMA BEM operator: `y = A x` (or `A^T x`), where the
    /// orientation was fixed at construction. Only `x` is uploaded per call; all
    /// coefficients and interaction lists stay resident on the device.
    pub fn apply(&mut self, x_re: &[f64], x_im: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        if x_re.len() != self.n || x_im.len() != self.n {
            return Err("BEM operator input dimension mismatch".into());
        }
        let stream = self.stream.clone();
        stream
            .memcpy_htod(x_re, &mut self.d_xr)
            .map_err(|e| format!("htod x_re: {e}"))?;
        stream
            .memcpy_htod(x_im, &mut self.d_xi)
            .map_err(|e| format!("htod x_im: {e}"))?;

        match &mut self.kind {
            BemOperatorKind::Dense {
                d_mat_re,
                d_mat_im,
                func,
            } => {
                let n_i = self.n as i32;
                let mut builder = stream.launch_builder(func);
                builder
                    .arg(d_mat_re)
                    .arg(d_mat_im)
                    .arg(&self.d_xr)
                    .arg(&self.d_xi)
                    .arg(&mut self.d_yr)
                    .arg(&mut self.d_yi)
                    .arg(&n_i);
                unsafe { builder.launch(bem_grid(self.n)) }
                    .map_err(|e| format!("launch bem_matvec: {e}"))?;
            }
            BemOperatorKind::FmmForward(dev) => {
                dev.apply(
                    &stream,
                    &self.d_xr,
                    &self.d_xi,
                    &mut self.d_yr,
                    &mut self.d_yi,
                    self.n,
                )?;
            }
            BemOperatorKind::FmmAdjoint(dev) => {
                dev.apply(
                    &stream,
                    &self.d_xr,
                    &self.d_xi,
                    &mut self.d_yr,
                    &mut self.d_yi,
                    self.n,
                )?;
            }
        }

        let yr = stream
            .clone_dtoh(&self.d_yr)
            .map_err(|e| format!("dtoh y_re: {e}"))?;
        let yi = stream
            .clone_dtoh(&self.d_yi)
            .map_err(|e| format!("dtoh y_im: {e}"))?;
        Ok((yr, yi))
    }
}

impl BemFmmForwardDevice {
    fn apply(
        &mut self,
        stream: &Arc<CudaStream>,
        xr: &CudaSlice<f64>,
        xi: &CudaSlice<f64>,
        yr: &mut CudaSlice<f64>,
        yi: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), String> {
        // Upward pass: deepest level first, so every child moment is final before
        // its parent reads it. One launch per depth level, deterministic per node.
        for &(begin, count) in self.level_ranges.iter().rev() {
            let mut b = stream.launch_builder(&self.fn_upward);
            b.arg(xr)
                .arg(xi)
                .arg(&self.d_src_scale)
                .arg(&self.d_src_nx)
                .arg(&self.d_src_ny)
                .arg(&self.d_src_nz)
                .arg(&self.d_child_start)
                .arg(&self.d_children)
                .arg(&self.d_leaf_start)
                .arg(&self.d_leaf_src)
                .arg(&self.d_level_nodes)
                .arg(&begin)
                .arg(&count)
                .arg(&mut self.d_mom);
            unsafe { b.launch(bem_grid(count as usize)) }
                .map_err(|e| format!("launch fmm upward: {e}"))?;
        }
        let n_i = n as i32;
        let mut b = stream.launch_builder(&self.fn_forward);
        b.arg(xr)
            .arg(xi)
            .arg(&self.d_diag)
            .arg(&self.d_f_start)
            .arg(&self.d_f_kind)
            .arg(&self.d_f_index)
            .arg(&self.d_f_a)
            .arg(&self.d_f_b)
            .arg(&self.d_f_dx)
            .arg(&self.d_f_dy)
            .arg(&self.d_f_dz)
            .arg(&self.d_mom)
            .arg(yr)
            .arg(yi)
            .arg(&n_i);
        unsafe { b.launch(bem_grid(n)) }
            .map(|_| ())
            .map_err(|e| format!("launch fmm forward: {e}"))
    }
}

impl BemFmmAdjointDevice {
    fn apply(
        &mut self,
        stream: &Arc<CudaStream>,
        xr: &CudaSlice<f64>,
        xi: &CudaSlice<f64>,
        yr: &mut CudaSlice<f64>,
        yi: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), String> {
        // Direct far-field adjoint gradients per node (ascending target order).
        let nc_i = self.node_count as i32;
        let mut b = stream.launch_builder(&self.fn_nodes);
        b.arg(xr)
            .arg(xi)
            .arg(&self.d_a_node_start)
            .arg(&self.d_a_node_i)
            .arg(&self.d_a_node_sr)
            .arg(&self.d_a_node_si)
            .arg(&self.d_a_node_dx)
            .arg(&self.d_a_node_dy)
            .arg(&self.d_a_node_dz)
            .arg(&mut self.d_grad)
            .arg(&nc_i);
        unsafe { b.launch(bem_grid(self.node_count)) }
            .map_err(|e| format!("launch fmm adjoint nodes: {e}"))?;

        // Downward push: root level first, then each deeper level. Each child has a
        // unique parent, so the level-ordered launches are race-free and deterministic.
        let push_levels = self.level_ranges.len().saturating_sub(1);
        for &(begin, count) in self.level_ranges.iter().take(push_levels) {
            let mut b = stream.launch_builder(&self.fn_push);
            b.arg(&self.d_child_start)
                .arg(&self.d_children)
                .arg(&self.d_level_nodes)
                .arg(&begin)
                .arg(&count)
                .arg(&mut self.d_grad);
            unsafe { b.launch(bem_grid(count as usize)) }
                .map_err(|e| format!("launch fmm adjoint push: {e}"))?;
        }

        // Per-source diagonal + near + leaf L2P accumulation.
        let n_i = n as i32;
        let mut b = stream.launch_builder(&self.fn_sources);
        b.arg(xr)
            .arg(xi)
            .arg(&self.d_diag)
            .arg(&self.d_a_src_start)
            .arg(&self.d_a_src_i)
            .arg(&self.d_a_src_cr)
            .arg(&self.d_a_src_ci)
            .arg(&self.d_src_scale)
            .arg(&self.d_src_nx)
            .arg(&self.d_src_ny)
            .arg(&self.d_src_nz)
            .arg(&self.d_src_leaf)
            .arg(&self.d_grad)
            .arg(yr)
            .arg(yi)
            .arg(&n_i);
        unsafe { b.launch(bem_grid(n)) }
            .map(|_| ())
            .map_err(|e| format!("launch fmm adjoint sources: {e}"))
    }
}

/// Result of compiling and auditing a single CUDA kernel's hardware SASS.
#[derive(Debug, Clone)]
pub struct KernelSassAudit {
    pub name: String,
    pub arch: String,
    pub sass_instructions: usize,
    /// Distinct hardware Tensor Core MMA opcodes detected in the SASS (e.g. `OMMA.SF...`,
    /// `QMMA...`, `IMMA...`, `HMMA...`). Empty for scalar FP64 / integer kernels.
    pub mma_opcodes: Vec<String>,
}

/// Extract distinct hardware Tensor Core MMA opcode mnemonics from `cuobjdump --dump-sass`.
pub fn extract_sass_mma_opcodes(sass_text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for raw in sass_text.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        for tok in line.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
            let up = tok.to_ascii_uppercase();
            let is_mma = up.starts_with("OMMA")
                || up.starts_with("QMMA")
                || up.starts_with("IMMA")
                || up.starts_with("HMMA")
                || up.starts_with("BMMA")
                || up.starts_with("DMMA")
                || up.starts_with("GMMA")
                || up.starts_with("IDP.4A")
                || up.starts_with("UTCOMMA")
                || up.starts_with("UTCQMMA");
            if is_mma && !found.iter().any(|x| x == &up) {
                found.push(up);
            }
        }
    }
    found.sort();
    found
}

/// Audit nvcc-generated PTX for contracted multiply-add instructions.
pub fn audit_ptx_zero_fma(ptx: &str) -> Result<usize, Vec<String>> {
    let mut offenders = Vec::new();
    let mut instrs = 0usize;
    for (idx, raw) in ptx.lines().enumerate() {
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
        let mut toks = line.split_whitespace();
        let mut op = toks.next().unwrap_or("");
        if op.starts_with('@') {
            op = toks.next().unwrap_or("");
        }
        let op = op.to_ascii_lowercase();
        if op.starts_with("fma.rn.") || op.starts_with("mad.rn.") {
            offenders.push(format!("line {}: {}", idx + 1, line));
        }
    }
    if offenders.is_empty() {
        Ok(instrs)
    } else {
        Err(offenders)
    }
}

/// Audit `cuobjdump --dump-sass` text for forbidden floating-point FMA / MAD contractions.
/// Excludes ptxas zero-multiplicand constant materialisation (`HFMA2 Rd, -RZ, RZ, ...`).
pub fn audit_sass_zero_fma(asm_text: &str) -> Result<usize, Vec<String>> {
    audit_sass_zero_fma_policy(asm_text, false)
}

/// SASS audit with an explicit IEEE-division/square-root allowance.
pub fn audit_sass_zero_fma_policy(
    asm_text: &str,
    allow_ieee_dfma: bool,
) -> Result<usize, Vec<String>> {
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
                let is_dfma = up == "DFMA" || up.starts_with("DFMA.");
                let zero_mul = sass
                    && [2usize, 3].iter().any(|&off| {
                        toks.get(mi + off)
                            .map(|t| t.trim_start_matches(['+', '-']).eq_ignore_ascii_case("RZ"))
                            .unwrap_or(false)
                    });
                let ieee_microcode = allow_ieee_dfma && is_dfma;
                !(zero_mul || ieee_microcode)
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

/// Check whether `nvcc` and `cuobjdump` are both installed in `PATH`.
pub fn nvcc_toolchain_available() -> bool {
    let ok = |bin: &str| {
        Command::new(bin)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    ok("nvcc") && ok("cuobjdump")
}

/// Parse the compute capability encoded in an `sm_XY`/`sm_XYZ[a]` target string.
/// `sm_61` → `(6, 1)`, `sm_90` → `(9, 0)`, `sm_120a` → `(12, 0)`, `sm_103a` → `(10, 3)`.
pub fn parse_sm_arch_cc(arch: &str) -> Option<(i32, i32)> {
    let tail = arch.strip_prefix("sm_")?;
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.len() < 2 {
        return None;
    }
    let split = digits.len() - 1;
    Some((digits[..split].parse().ok()?, digits[split..].parse().ok()?))
}

/// Choose the highest `sm_XX` architecture supported by the installed `nvcc`,
/// preferring `preferred` (or `sm_120a` Blackwell on CUDA 12.8+, down to `sm_75`).
pub fn pick_supported_nvcc_arch(preferred: Option<&str>) -> Result<String, String> {
    if !nvcc_toolchain_available() {
        return Err("nvcc/cuobjdump toolchain not installed".into());
    }
    let probe_src = "extern \"C\" __global__ void g6me_probe(double* p) { p[0] = 1.0; }\n";
    let (cu_path, cubin_path) = temp_cuda_paths("arch_probe");
    std::fs::write(&cu_path, probe_src).map_err(|e| format!("write probe .cu: {e}"))?;

    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = preferred {
        candidates.push(p.to_string());
    }
    for arch in [
        "sm_120a", "sm_120", "sm_90", "sm_89", "sm_86", "sm_80", "sm_75", "sm_61",
    ] {
        if !candidates.iter().any(|c| c == arch) {
            candidates.push(arch.to_string());
        }
    }

    let mut chosen = None;
    for arch in &candidates {
        let ok = Command::new("nvcc")
            .args(["--fmad=false", "-O3", "-cubin"])
            .arg(format!("-arch={arch}"))
            .arg("-o")
            .arg(&cubin_path)
            .arg(&cu_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            chosen = Some(arch.clone());
            break;
        }
    }
    let _ = std::fs::remove_file(&cu_path);
    let _ = std::fs::remove_file(&cubin_path);
    chosen.ok_or_else(|| "nvcc could not compile for any supported sm_75..sm_120 target".into())
}

fn temp_cuda_paths(tag: &str) -> (PathBuf, PathBuf) {
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let seq = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    (
        tmp.join(format!("g6me_cuda_{tag}_{pid}_{seq}.cu")),
        tmp.join(format!("g6me_cuda_{tag}_{pid}_{seq}.out")),
    )
}

fn compile_cu_to_ptx(
    tag: &str,
    src: &str,
    arch: &str,
    fmad_enabled: bool,
) -> Result<String, String> {
    let (cu_path, ptx_path) = temp_cuda_paths(tag);
    std::fs::write(&cu_path, src).map_err(|e| format!("write {tag}.cu: {e}"))?;
    let fmad_flag = if fmad_enabled {
        "--fmad=true"
    } else {
        "--fmad=false"
    };
    let mut cmd = Command::new("nvcc");
    cmd.arg(fmad_flag);
    if !fmad_enabled {
        cmd.args(["-Xcompiler", "-ffp-contract=off"]);
    }
    let out = cmd
        .args(["-O3", "-ptx", "-Wno-deprecated-gpu-targets"])
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&ptx_path)
        .arg(&cu_path)
        .output()
        .map_err(|e| format!("spawn nvcc for {tag}: {e}"))?;
    let _ = std::fs::remove_file(&cu_path);
    if !out.status.success() {
        let _ = std::fs::remove_file(&ptx_path);
        return Err(format!(
            "nvcc -ptx failed for {tag} ({arch}): {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let ptx = std::fs::read_to_string(&ptx_path).map_err(|e| format!("read {tag}.ptx: {e}"))?;
    let _ = std::fs::remove_file(&ptx_path);
    Ok(ptx)
}

fn compile_and_dump_sass(
    tag: &str,
    src: &str,
    arch: &str,
    fmad_enabled: bool,
) -> Result<String, String> {
    let (cu_path, cubin_path) = temp_cuda_paths(tag);
    std::fs::write(&cu_path, src).map_err(|e| format!("write {tag}.cu: {e}"))?;
    let fmad_flag = if fmad_enabled {
        "--fmad=true"
    } else {
        "--fmad=false"
    };
    let mut cmd = Command::new("nvcc");
    cmd.arg(fmad_flag);
    if !fmad_enabled {
        cmd.args(["-Xcompiler", "-ffp-contract=off"]);
    }
    cmd.args(["-O3", "-cubin", "-Wno-deprecated-gpu-targets"])
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&cubin_path)
        .arg(&cu_path);
    let out = cmd
        .output()
        .map_err(|e| format!("spawn nvcc for {tag}: {e}"))?;
    let _ = std::fs::remove_file(&cu_path);
    if !out.status.success() {
        let _ = std::fs::remove_file(&cubin_path);
        return Err(format!(
            "nvcc -cubin ({fmad_flag}) failed for {tag} ({arch}): {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let dump = Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(&cubin_path)
        .output()
        .map_err(|e| format!("spawn cuobjdump for {tag}: {e}"))?;
    let _ = std::fs::remove_file(&cubin_path);
    if !dump.status.success() {
        return Err(format!(
            "cuobjdump --dump-sass failed for {tag}: {}",
            String::from_utf8_lossy(&dump.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&dump.stdout).into_owned())
}

/// Compile ALL 6 embedded `.cu` kernels with `nvcc --fmad=false`, then verify both
/// zero-FMA PTX and zero-FMA hardware SASS (plus live MMA opcodes on Blackwell).
pub fn audit_all_cuda_kernels_sass(
    preferred_arch: Option<&str>,
) -> Result<Vec<KernelSassAudit>, String> {
    let arch = pick_supported_nvcc_arch(preferred_arch)?;
    let mut results = Vec::with_capacity(ALL_CUDA_KERNELS.len());
    for kernel in ALL_CUDA_KERNELS {
        let name = kernel.name;
        let ptx = compile_cu_to_ptx(name, kernel.source, &arch, false)?;
        // Every kernel must emit zero `fma.rn` in its PTX, including the IEEE-div/sqrt
        // kernel: `--fmad=false` is nvcc's only contraction point, so a clean PTX proves
        // the source arithmetic is unfused. `ieee_div_sqrt` relaxes the SASS layer only,
        // for ptxas' MUFU+DFMA microcode expansion of one correctly-rounded operation.
        let ptx_instrs = audit_ptx_zero_fma(&ptx).map_err(|offenders| {
            format!(
                "CUDA kernel `{name}` ({arch}) PTX contains {} contracted fma.rn instruction(s) \
                 with --fmad=false: {}",
                offenders.len(),
                offenders
                    .iter()
                    .take(4)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        })?;
        if ptx_instrs == 0 {
            return Err(format!("CUDA kernel `{name}` produced 0 PTX instructions"));
        }

        let sass = compile_and_dump_sass(name, kernel.source, &arch, false)?;
        let count =
            audit_sass_zero_fma_policy(&sass, kernel.ieee_div_sqrt).map_err(|offenders| {
                format!(
                    "CUDA kernel `{name}` ({arch}) emitted {} forbidden FMA/MAD SASS \
                     instruction(s): {}",
                    offenders.len(),
                    offenders
                        .iter()
                        .take(4)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" | ")
                )
            })?;
        if count == 0 {
            return Err(format!("CUDA kernel `{name}` produced 0 SASS instructions"));
        }
        let mma_opcodes = extract_sass_mma_opcodes(&sass);
        let arch_num: u32 = arch
            .trim_start_matches("sm_")
            .trim_end_matches('a')
            .parse()
            .unwrap_or(0);
        if name == "ozaki2_blackwell_mma" && arch.ends_with('a') && arch_num >= 120 {
            let has_nvfp4_omma = mma_opcodes
                .iter()
                .any(|op| op.starts_with("OMMA") && op.contains("E2M1"));
            if !has_nvfp4_omma {
                return Err(format!(
                    "CUDA kernel `{name}` ({arch}) contains no NVFP4 OMMA.SF E2M1 instruction in \
                     SASS (found: {mma_opcodes:?}): the Blackwell NVFP4 path is dead code"
                ));
            }
        }
        results.push(KernelSassAudit {
            name: name.to_string(),
            arch: arch.clone(),
            sass_instructions: count,
            mma_opcodes,
        });
    }
    Ok(results)
}

/// Compile and audit `ozaki2_blackwell_mma.cu` across ALL supported GPU architectures
/// (`sm_120a`, `sm_90`, `sm_89`, `sm_80`, `sm_75`) that the installed `nvcc` accepts,
/// verifying zero-FMA PTX/SASS and confirming each target emits its expected hardware
/// Tensor Core MMA instructions (NVFP4 `OMMA.SF` on Blackwell, FP8 `QMMA` on Ada,
/// `IMMA` + `HMMA` on Ampere/Turing).
pub fn audit_multi_arch_ozaki2_sass() -> Result<Vec<KernelSassAudit>, String> {
    if !nvcc_toolchain_available() {
        return Err("nvcc/cuobjdump toolchain not installed".into());
    }
    let targets = ["sm_120a", "sm_90", "sm_89", "sm_80", "sm_75", "sm_61"];
    let mut results = Vec::new();
    for arch in targets {
        let ptx = match compile_cu_to_ptx("ozaki2_multi", KERNEL_OZAKI2_MMA_CU, arch, false) {
            Ok(p) => p,
            Err(e) => {
                // Skip newer/older targets if the installed nvcc rejects the arch flag.
                if e.contains("Unsupported gpu architecture") {
                    continue;
                }
                return Err(e);
            }
        };
        audit_ptx_zero_fma(&ptx).map_err(|offenders| {
            format!(
                "ozaki2_blackwell_mma ({arch}) PTX had {} fma.rn offenders: {}",
                offenders.len(),
                offenders.first().cloned().unwrap_or_default()
            )
        })?;

        let sass = compile_and_dump_sass("ozaki2_multi", KERNEL_OZAKI2_MMA_CU, arch, false)?;
        let count = audit_sass_zero_fma(&sass).map_err(|offenders| {
            format!(
                "ozaki2_blackwell_mma ({arch}) SASS had {} FMA offenders: {}",
                offenders.len(),
                offenders.first().cloned().unwrap_or_default()
            )
        })?;
        let mma_opcodes = extract_sass_mma_opcodes(&sass);

        match arch {
            "sm_120a" => {
                if !mma_opcodes
                    .iter()
                    .any(|op| op.starts_with("OMMA") && op.contains("E2M1"))
                {
                    return Err(format!(
                        "sm_120a missing NVFP4 OMMA.SF E2M1 opcode (got {mma_opcodes:?})"
                    ));
                }
                if !mma_opcodes.iter().any(|op| op.starts_with("QMMA")) {
                    return Err(format!(
                        "sm_120a missing FP8 QMMA opcode (got {mma_opcodes:?})"
                    ));
                }
                if !mma_opcodes.iter().any(|op| op.starts_with("IMMA")) {
                    return Err(format!(
                        "sm_120a missing INT8 IMMA opcode (got {mma_opcodes:?})"
                    ));
                }
            }
            "sm_89" => {
                if !mma_opcodes
                    .iter()
                    .any(|op| op.starts_with("QMMA") && op.contains("E4M3"))
                {
                    return Err(format!(
                        "sm_89 missing FP8 QMMA E4M3 opcode (got {mma_opcodes:?})"
                    ));
                }
                if !mma_opcodes.iter().any(|op| op.starts_with("IMMA")) {
                    return Err(format!(
                        "sm_89 missing INT8 IMMA opcode (got {mma_opcodes:?})"
                    ));
                }
            }
            "sm_90" | "sm_80" | "sm_75" => {
                let has_imma = mma_opcodes.iter().any(|op| op.starts_with("IMMA"));
                let has_hmma = mma_opcodes.iter().any(|op| op.starts_with("HMMA"));
                if !(has_imma && has_hmma) {
                    return Err(format!(
                        "{arch} missing IMMA/HMMA opcodes (got {mma_opcodes:?})"
                    ));
                }
            }
            "sm_61" if !mma_opcodes.iter().any(|op| op.starts_with("IDP.4A")) => {
                return Err(format!(
                    "sm_61 Pascal missing IDP.4A.S8.S8 (__dp4a) opcode (got {mma_opcodes:?})"
                ));
            }
            _ => {}
        }

        results.push(KernelSassAudit {
            name: "ozaki2_blackwell_mma".to_string(),
            arch: arch.to_string(),
            sass_instructions: count,
            mma_opcodes,
        });
    }
    if results.is_empty() {
        return Err("nvcc could not compile ozaki2_blackwell_mma for any sm_75..sm_120a".into());
    }
    Ok(results)
}

/// Sabotage negative control (AGENTS.md Sabotage Rule): compiles a contractible double-precision
/// multiply-add kernel with `nvcc --fmad=true` and verifies that BOTH auditors catch it.
pub fn verify_sass_auditor_catches_fmad_sabotage(
    preferred_arch: Option<&str>,
) -> Result<usize, String> {
    let arch = pick_supported_nvcc_arch(preferred_arch)?;
    let sabotage_src = r#"
extern "C" __global__ void g6me_fma_sabotage_probe(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    double* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i] + c[i];
    }
}
"#;
    let ptx = compile_cu_to_ptx("sabotage_fma_ptx", sabotage_src, &arch, true)?;
    if audit_ptx_zero_fma(&ptx).is_ok() {
        return Err(format!(
            "SABOTAGE FAILED: nvcc --fmad=true ({arch}) produced PTX without `fma.rn`, \
             but the PTX auditor missed the contraction!"
        ));
    }
    let sass = compile_and_dump_sass("sabotage_fma", sabotage_src, &arch, true)?;
    match audit_sass_zero_fma(&sass) {
        Ok(n) => Err(format!(
            "SABOTAGE FAILED: nvcc --fmad=true ({arch}) produced {n} SASS instructions, but auditor missed DFMA!"
        )),
        Err(offenders) => Ok(offenders.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_six_cuda_kernels_pass_real_nvcc_fmad_false_and_sabotage_catches_dfma() {
        if !nvcc_toolchain_available() {
            let clean = "/*0010*/ DMUL R4, R2, R6;\n/*0020*/ DADD R8, R4, R10;\n";
            assert_eq!(audit_sass_zero_fma(clean).expect("clean SASS"), 2);
            let dirty = "/*0010*/ DFMA R8, R2, R6, R10;\n";
            assert!(audit_sass_zero_fma(dirty).is_err());
            assert!(audit_sass_zero_fma_policy(dirty, true).is_ok());
            assert!(audit_sass_zero_fma_policy("/*0010*/ FFMA R8, R2, R6, R10;\n", true).is_err());
            let clean_ptx = "mul.rn.f64 %rd1, %rd2, %rd3;\nadd.rn.f64 %rd4, %rd1, %rd5;\n";
            assert_eq!(audit_ptx_zero_fma(clean_ptx).expect("clean PTX"), 2);
            let dirty_ptx = "fma.rn.f64 %rd1, %rd2, %rd3, %rd4;\n";
            assert!(audit_ptx_zero_fma(dirty_ptx).is_err());
            let predicated = "@%p1 fma.rn.f64 %rd1, %rd2, %rd3, %rd4;\n";
            assert!(audit_ptx_zero_fma(predicated).is_err());
            return;
        }
        let audits = audit_all_cuda_kernels_sass(None).expect(
            "all 6 CUDA kernels must compile with --fmad=false, emit 0 fma.rn PTX and 0 \
             contracted SASS FMA",
        );
        assert_eq!(audits.len(), 6);
        for a in &audits {
            assert!(
                a.sass_instructions > 5,
                "kernel {} produced suspiciously few SASS instructions: {}",
                a.name,
                a.sass_instructions
            );
        }
        let caught = verify_sass_auditor_catches_fmad_sabotage(None)
            .expect("Sabotage check: nvcc --fmad=true MUST trigger DFMA detection");
        assert!(caught >= 1);
    }

    #[test]
    fn multi_arch_ozaki2_sass_emits_expected_tensor_core_opcodes_and_zero_fma() {
        if !nvcc_toolchain_available() {
            return;
        }
        let audits = audit_multi_arch_ozaki2_sass()
            .expect("multi-arch Ozaki-II SASS audit across sm_120a/sm_90/sm_89/sm_80/sm_75");
        assert!(
            audits.len() >= 4,
            "expected at least 4 compiled architectures, got {}",
            audits.len()
        );
        for a in &audits {
            assert!(
                !a.mma_opcodes.is_empty(),
                "arch {} must emit live hardware Tensor Core MMA opcodes",
                a.arch
            );
        }
    }

    #[test]
    fn arch_precision_resolution_matches_hardware_capabilities() {
        // Desktop Blackwell sm_120a: Auto -> INT8 IMMA (2x faster 1-limb vs 2-limb NVFP4), manual NVFP4 honored
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(12, 0, "sm_120a"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Nvfp4E2m1.resolve_for_arch(12, 0, "sm_120a"),
            CudaTensorPrecision::Nvfp4E2m1
        );
        // Blackwell Ultra sm_103a (B300): no sm_120a-style OMMA FP4 body exists, so Auto is
        // INT8 and an explicit NVFP4 request downgrades to FP8 instead of an empty kernel.
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(10, 3, "sm_103a"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Nvfp4E2m1.resolve_for_arch(10, 3, "sm_103a"),
            CudaTensorPrecision::Fp8E4m3
        );
        // Plain sm_120 (without 'a'): NVFP4 MMA requires sm_120a, gracefully falls back to FP8
        assert_eq!(
            CudaTensorPrecision::Nvfp4E2m1.resolve_for_arch(12, 0, "sm_120"),
            CudaTensorPrecision::Fp8E4m3
        );
        // Hopper & Ada: Auto -> INT8 IMMA (8x faster than 2-limb FP8-FP32 on Ada), manual FP8 honored
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(9, 0, "sm_90"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(8, 9, "sm_89"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Nvfp4E2m1.resolve_for_arch(8, 9, "sm_89"),
            CudaTensorPrecision::Fp8E4m3
        );
        // Ampere & Turing: Auto -> INT8, manual NVFP4/FP8 downgrade to INT8, FP16 preserved
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(8, 6, "sm_86"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Fp8E4m3.resolve_for_arch(7, 5, "sm_75"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Fp16.resolve_for_arch(7, 5, "sm_75"),
            CudaTensorPrecision::Fp16
        );
        // Pascal sm_61: Auto -> INT8 (IDP.4A.S8.S8 __dp4a), manual FP16/FP8/NVFP4 downgrade to INT8
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(6, 1, "sm_61"),
            CudaTensorPrecision::Int8
        );
        assert_eq!(
            CudaTensorPrecision::Fp16.resolve_for_arch(6, 1, "sm_61"),
            CudaTensorPrecision::Int8
        );
    }

    /// Bit-exact CPU simulation of the exact `ozaki2_blackwell_mma.cu` device pipeline:
    /// Stage 1 (frexp + 20-moduli) -> Stage 2 (E2M1 Base-13 nibble packing + UE4M3=2.0 scale) ->
    /// Stage 3 (warp `m16n8k64` fragment addressing) -> Stage 4 (`u128` modular CRT -> `f64`),
    /// plus a Sabotage Rule check proving wrong UE4M3 scale factor fails immediately.
    #[test]
    fn device_nvfp4_base13_warp_fragment_and_u128_crt_bit_simulation_plus_sabotage() {
        const MODULI: [i32; 20] = [
            113, 109, 107, 103, 101, 97, 89, 83, 81, 79, 73, 71, 67, 64, 61, 59, 53, 49, 47, 43,
        ];
        const CRT_INV: [u32; 20] = [
            21, 87, 6, 59, 22, 87, 41, 69, 38, 9, 13, 34, 65, 3, 5, 16, 8, 33, 18, 7,
        ];
        const M_BIG: u128 = (0x13dab74a538b2a1f_u128 << 64) | 0x6377639e4e1aeac0_u128;
        const M_HALF: u128 = M_BIG >> 1;

        fn e2m1_decode(nib: u8) -> f32 {
            const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let v = MAG[(nib & 7) as usize];
            if (nib & 8) != 0 {
                -v
            } else {
                v
            }
        }
        fn e2m1_encode(d: i32) -> u8 {
            // d = 2 * e2m1 in {0, +-1, +-2, +-3, +-4, +-6, +-8}
            let sign = if d < 0 { 8u8 } else { 0u8 };
            let mag = match d.abs() {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 3,
                4 => 4,
                6 => 5,
                8 => 6,
                _ => panic!("not representable in Base-13 E2M1: {d}"),
            };
            sign | mag
        }
        fn base13_encode(r: i32) -> (u8, u8) {
            const REM_TO_D: [i32; 13] = [0, 1, 2, 3, 4, -8, 6, -6, 8, -4, -3, -2, -1];
            let rem = r.rem_euclid(13) as usize;
            let d0 = REM_TO_D[rem];
            let d1 = (r - d0) / 13;
            (e2m1_encode(d0), e2m1_encode(d1))
        }
        fn ue4m3_decode(byte: u8) -> f32 {
            let exp = ((byte >> 3) & 0xF) as i32;
            let man = (byte & 0x7) as f32;
            if exp == 0 {
                (man / 8.0) * 2.0_f32.powi(-6)
            } else {
                (1.0 + man / 8.0) * 2.0_f32.powi(exp - 7)
            }
        }
        fn frexp_exp(x: f64) -> i32 {
            let bits = x.abs().to_bits();
            if bits == 0 {
                return 0;
            }
            ((bits >> 52) as i32) - 1022
        }

        let run_sim = |sf_byte: u8| -> f64 {
            let (m, k, n) = (19usize, 73usize, 11usize);
            let mut a = vec![0.0_f64; m * k];
            let mut b = vec![0.0_f64; k * n];
            for i in 0..m {
                let row_scale = 10.0_f64.powi(((i % 5) as i32) - 2);
                for t in 0..k {
                    a[i * k + t] = row_scale * ((i * 7 + t * 3) as f64 * 0.37).sin();
                }
            }
            for t in 0..k {
                for j in 0..n {
                    let col_scale = 10.0_f64.powi(((j % 4) as i32) - 1);
                    b[t * n + j] = col_scale * ((t * 5 + j * 11) as f64 * 0.29).cos();
                }
            }

            // Stage 1: exponent extraction + 20-moduli symmetric reduction
            let shift = 53i32;
            let mut exp_a = vec![0i32; m];
            let mut exp_b = vec![0i32; n];
            let mut a_sym = vec![0i8; 20 * m * k];
            let mut b_sym = vec![0i8; 20 * n * k];
            for i in 0..m {
                let mx = (0..k).map(|t| a[i * k + t].abs()).fold(0.0, f64::max);
                let ea = frexp_exp(mx);
                exp_a[i] = ea;
                let fac = 2.0_f64.powi(shift - ea);
                for t in 0..k {
                    let q = (a[i * k + t] * fac).round() as i64;
                    for mi in 0..20 {
                        let md = MODULI[mi] as i64;
                        let mut r = q.rem_euclid(md);
                        if r > md / 2 {
                            r -= md;
                        }
                        a_sym[mi * m * k + i * k + t] = r as i8;
                    }
                }
            }
            for j in 0..n {
                let mx = (0..k).map(|t| b[t * n + j].abs()).fold(0.0, f64::max);
                let eb = frexp_exp(mx);
                exp_b[j] = eb;
                let fac = 2.0_f64.powi(shift - eb);
                for t in 0..k {
                    let q = (b[t * n + j] * fac).round() as i64;
                    for mi in 0..20 {
                        let md = MODULI[mi] as i64;
                        let mut r = q.rem_euclid(md);
                        if r > md / 2 {
                            r -= md;
                        }
                        b_sym[mi * n * k + j * k + t] = r as i8;
                    }
                }
            }

            // Stage 2: Base-13 nibble packing into padded uint32 words
            let m_pad = align_up(m, 16);
            let n_pad = align_up(n, 8);
            let k_pad = align_up(k, 64);
            let k_words = k_pad / 8;
            let k_tiles = k_pad / 64;
            let sf_val = ue4m3_decode(sf_byte);

            let pack = |sym: &[i8], rows: usize, rows_pad: usize| -> (Vec<u32>, Vec<u32>) {
                let mut lo = vec![0u32; 20 * rows_pad * k_words];
                let mut hi = vec![0u32; 20 * rows_pad * k_words];
                for mi in 0..20 {
                    for r in 0..rows {
                        for kw in 0..k_words {
                            let mut wl = 0u32;
                            let mut wh = 0u32;
                            for s in 0..8 {
                                let kk = kw * 8 + s;
                                if kk < k {
                                    let (n0, n1) =
                                        base13_encode(sym[mi * rows * k + r * k + kk] as i32);
                                    wl |= (n0 as u32) << (s * 4);
                                    wh |= (n1 as u32) << (s * 4);
                                }
                            }
                            let idx = (mi * rows_pad + r) * k_words + kw;
                            lo[idx] = wl;
                            hi[idx] = wh;
                        }
                    }
                }
                (lo, hi)
            };
            let (a_lo, a_hi) = pack(&a_sym, m, m_pad);
            let (b_lo, b_hi) = pack(&b_sym, n, n_pad);

            // Stage 3: Warp-exact m16n8k64 fragment addressing
            let mut c_mod = vec![0i32; 20 * m * n];
            for mi in 0..20 {
                let md = MODULI[mi] as i64;
                for bm in 0..(m_pad / 16) {
                    for bn in 0..(n_pad / 8) {
                        let tm = bm * 16;
                        let tn = bn * 8;
                        // Simulate all 32 warp lanes computing their 4 fragment outputs
                        for lane in 0..32usize {
                            let gid = lane >> 2;
                            let tidg = lane & 3;
                            for v in 0..4usize {
                                let out_r = tm + gid + 8 * (v >> 1);
                                let out_c = tn + 2 * tidg + (v & 1);
                                if out_r >= m || out_c >= n {
                                    continue;
                                }
                                let (mut p00, mut p01, mut p10, mut p11) =
                                    (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
                                for kt in 0..k_tiles {
                                    for sub in 0..8usize {
                                        let kw = kt * 8 + sub;
                                        let a0w = a_lo[(mi * m_pad + out_r) * k_words + kw];
                                        let a1w = a_hi[(mi * m_pad + out_r) * k_words + kw];
                                        let b0w = b_lo[(mi * n_pad + out_c) * k_words + kw];
                                        let b1w = b_hi[(mi * n_pad + out_c) * k_words + kw];
                                        for s in 0..8 {
                                            let a0e = e2m1_decode(((a0w >> (s * 4)) & 0xF) as u8)
                                                * sf_val;
                                            let a1e = e2m1_decode(((a1w >> (s * 4)) & 0xF) as u8)
                                                * sf_val;
                                            let b0e = e2m1_decode(((b0w >> (s * 4)) & 0xF) as u8)
                                                * sf_val;
                                            let b1e = e2m1_decode(((b1w >> (s * 4)) & 0xF) as u8)
                                                * sf_val;
                                            p00 += a0e * b0e;
                                            p01 += a0e * b1e;
                                            p10 += a1e * b0e;
                                            p11 += a1e * b1e;
                                        }
                                    }
                                }
                                let total = (p00.round() as i64)
                                    + 13 * ((p01.round() as i64) + (p10.round() as i64))
                                    + 169 * (p11.round() as i64);
                                c_mod[mi * m * n + out_r * n + out_c] = total.rem_euclid(md) as i32;
                            }
                        }
                    }
                }
            }

            // Stage 4: Exact u128 CRT reconstruction -> f64
            let mut max_rel = 0.0_f64;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0u128;
                    for mi in 0..20 {
                        let r = c_mod[mi * m * n + i * n + j] as u32;
                        let md = MODULI[mi] as u32;
                        let t = ((r as u64 * CRT_INV[mi] as u64) % md as u64) as u128;
                        let m_div = M_BIG / (md as u128);
                        acc += t * m_div;
                        if acc >= M_BIG {
                            acc -= M_BIG;
                        }
                    }
                    let neg = acc > M_HALF;
                    let mag = if neg { M_BIG - acc } else { acc };
                    let hi = (mag >> 64) as u64;
                    let lo = mag as u64;
                    let mut val = (hi as f64) * 18_446_744_073_709_551_616.0 + (lo as f64);
                    if neg {
                        val = -val;
                    }
                    let got = val * 2.0_f64.powi(exp_a[i] + exp_b[j] - 2 * shift);
                    let mut ref_v = 0.0_f64;
                    let mut ref_abs = 0.0_f64;
                    for t in 0..k {
                        let p = a[i * k + t] * b[t * n + j];
                        ref_v += p;
                        ref_abs += p.abs();
                    }
                    let rel = (got - ref_v).abs() / ref_abs.max(1e-12);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
            }
            max_rel
        };

        // Correct UE4M3 scale = 2.0 (0x40): achieves full IEEE-754 f64 accuracy!
        let err_ok = run_sim(0x40);
        assert!(
            err_ok < 1e-14,
            "NVFP4 Base-13 + u128 CRT simulation must achieve < 1e-14 backward error, got {err_ok:e}"
        );

        // Sabotage Rule: wrong UE4M3 scale = 1.0 (0x38) breaks exact integer scaling!
        let err_sabotaged = run_sim(0x38);
        assert!(
            err_sabotaged > 0.05,
            "Sabotage check: wrong UE4M3 scale factor MUST fail parity (got {err_sabotaged:e})"
        );
    }

    // ---------------------------------------------------------------------------------
    // Fused single-launch NTT + device-resident UPOLA (docs/CUDA_PERF_PLAN.md section 1).
    // These are test-only CPU references: they exist to prove the live GPU integer path
    // is bit-exact, not to run in production.
    // ---------------------------------------------------------------------------------

    const GL_P: u64 = 0xFFFF_FFFF_0000_0001;

    fn gl_add_ref(a: u64, b: u64) -> u64 {
        ((a as u128 + b as u128) % GL_P as u128) as u64
    }

    fn gl_sub_ref(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            GL_P - (b - a)
        }
    }

    fn gl_mul_ref(a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % GL_P as u128) as u64
    }

    fn gl_pow_ref(mut base: u64, mut exp: u64) -> u64 {
        let mut acc = 1u64;
        base %= GL_P;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = gl_mul_ref(acc, base);
            }
            base = gl_mul_ref(base, base);
            exp >>= 1;
        }
        acc
    }

    /// Test-only replica of `g6me_core::ntt::NttPlan` tables (`W32 = 7^((p-1)/2^32)`).
    fn glm_plan(n: usize) -> (Vec<Vec<u64>>, Vec<Vec<u64>>, u64) {
        let log_n = n.trailing_zeros();
        let w32 = 1_753_635_133_440_165_772u64;
        let w_n = gl_pow_ref(w32, 1u64 << (32 - log_n));
        let w_n_inv = gl_pow_ref(w_n, GL_P - 2);
        let mut fwd = Vec::new();
        let mut inv = Vec::new();
        let mut len = 2usize;
        while len <= n {
            let half = len / 2;
            let step_f = gl_pow_ref(w_n, (n / len) as u64);
            let step_i = gl_pow_ref(w_n_inv, (n / len) as u64);
            let mut tf = Vec::with_capacity(half);
            let mut ti = Vec::with_capacity(half);
            let (mut wf, mut wi) = (1u64, 1u64);
            for _ in 0..half {
                tf.push(wf);
                ti.push(wi);
                wf = gl_mul_ref(wf, step_f);
                wi = gl_mul_ref(wi, step_i);
            }
            fwd.push(tf);
            inv.push(ti);
            len <<= 1;
        }
        (fwd, inv, gl_pow_ref(n as u64, GL_P - 2))
    }

    /// Test-only CPU reference transform with the same operation order as `NttPlan`.
    fn glm_ntt_ref(buf: &mut [u64], twiddles: &[Vec<u64>], scale: Option<u64>) {
        let n = buf.len();
        let log_n = n.trailing_zeros();
        if log_n > 0 {
            for i in 0..n {
                let j = i.reverse_bits() >> (usize::BITS - log_n);
                if i < j {
                    buf.swap(i, j);
                }
            }
        }
        for (stage, tw) in twiddles.iter().enumerate() {
            let half = 1usize << stage;
            let len = half << 1;
            let mut start = 0usize;
            while start < n {
                for (k, &w) in tw.iter().enumerate() {
                    let u = buf[start + k];
                    let v = gl_mul_ref(buf[start + k + half], w);
                    buf[start + k] = gl_add_ref(u, v);
                    buf[start + k + half] = gl_sub_ref(u, v);
                }
                start += len;
            }
        }
        if let Some(inv) = scale {
            for v in buf.iter_mut() {
                *v = gl_mul_ref(*v, inv);
            }
        }
    }

    fn test_field_vec(seed: u64, len: usize) -> Vec<u64> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s % GL_P
            })
            .collect()
    }

    /// Live gate: the fused single-launch transform must equal both the multi-launch path
    /// and the CPU reference bit-for-bit (forward and inverse), and the device-resident
    /// UPOLA state must reproduce a scalar CPU simulation of the same FDL-UPOLA loop.
    #[test]
    fn fused_ntt_and_device_upola_are_bit_exact_against_cpu_reference() {
        let Ok(rt) = G6meCudaRuntime::init() else {
            // No CUDA GPU / driver / nvcc on this host: the parity gate cannot run here.
            return;
        };

        // ---- fused transform vs multi-launch vs CPU, both directions --------------
        for n in [256usize, 4096, 8192] {
            let (fwd, inv, inv_n) = glm_plan(n);
            let input = test_field_vec(0x1234_5678_9ABC_DEF0 ^ n as u64, n);

            let mut cpu_f = input.clone();
            glm_ntt_ref(&mut cpu_f, &fwd, None);
            let legacy_f = rt
                .run_goldilocks_ntt_transform_legacy(&input, &fwd, None)
                .expect("legacy forward NTT");
            let fused_f = rt
                .run_goldilocks_ntt_transform(&input, &fwd, None)
                .expect("fused forward NTT");
            assert_eq!(
                legacy_f, cpu_f,
                "legacy forward NTT differs from CPU at n={n}"
            );
            assert_eq!(
                fused_f, cpu_f,
                "fused forward NTT differs from CPU at n={n}"
            );

            let mut cpu_i = cpu_f.clone();
            glm_ntt_ref(&mut cpu_i, &inv, Some(inv_n));
            let fused_i = rt
                .run_goldilocks_ntt_transform(&cpu_f, &inv, Some(inv_n))
                .expect("fused inverse NTT");
            assert_eq!(
                fused_i, cpu_i,
                "fused inverse NTT differs from CPU at n={n}"
            );
            assert_eq!(
                fused_i, input,
                "inverse(forward(x)) must return x bit-for-bit at n={n}"
            );
        }

        // ---- device-resident UPOLA vs a scalar CPU simulation ---------------------
        let n = 512usize;
        let slots = 4usize;
        let n_parts = 3usize;
        let fdl_slots = n_parts - 1;
        let (fwd, inv, inv_n) = glm_plan(n);
        let kern_l = test_field_vec(0xDEAD_BEEF_CAFE_0001, n_parts * slots * n);
        let kern_r = test_field_vec(0xDEAD_BEEF_CAFE_0002, n_parts * slots * n);
        let mut state = rt
            .create_upola_state(n, slots, n_parts, inv_n, &kern_l, &kern_r, &fwd, &inv)
            .expect("create CudaUpolaState");
        assert_eq!(state.n(), n);
        assert_eq!(state.n_parts(), n_parts);

        // CPU simulation mirrors `BinauralPlus`: FDL row per head, partition-0 from the
        // just-completed row, partitions 1..n_parts walking back in time.
        let mut cpu_fdl = vec![0u64; fdl_slots * slots * n];
        let active_mask: u32 = (1 << 0) | (1 << 1) | (1 << 3); // ch 2 inactive
        let mut inputs: Vec<Vec<u64>> = Vec::new();
        for blk in 0..5u64 {
            inputs.push(test_field_vec(0xB10C_0000 + blk, n));
        }

        for (blk, input) in inputs.iter().enumerate() {
            let head = blk % fdl_slots;
            // CPU side: zero row, transform active channels into it.
            let row = &mut cpu_fdl[head * slots * n..(head + 1) * slots * n];
            row.fill(0);
            let mut transformed = vec![vec![0u64; n]; slots];
            for ch in 0..slots {
                if (active_mask >> ch) & 1 == 0 {
                    continue;
                }
                let mut x = input.clone();
                glm_ntt_ref(&mut x, &fwd, None);
                transformed[ch].copy_from_slice(&x);
                row[ch * n..(ch + 1) * n].copy_from_slice(&x);
            }

            // Device side: the same upload + one batched fused forward.
            state.zero_fdl_row(head).expect("zero fdl row");
            for ch in 0..slots {
                if (active_mask >> ch) & 1 == 0 {
                    continue;
                }
                state
                    .upload_fdl_row(head, ch, input)
                    .expect("upload fdl row");
            }
            state
                .forward_fdl_row(head, active_mask)
                .expect("fused fdl forward");

            // Partition 0 (completed row) → both ears.
            let mut cpu_acc = vec![0u64; 2 * n];
            for (ch, x) in transformed.iter().enumerate() {
                if (active_mask >> ch) & 1 == 0 {
                    continue;
                }
                for k in 0..n {
                    let off = ch * n + k;
                    cpu_acc[k] = gl_add_ref(cpu_acc[k], gl_mul_ref(x[k], kern_l[off]));
                    cpu_acc[n + k] = gl_add_ref(cpu_acc[n + k], gl_mul_ref(x[k], kern_r[off]));
                }
            }
            glm_ntt_ref(&mut cpu_acc[..n], &inv, Some(inv_n));
            glm_ntt_ref(&mut cpu_acc[n..], &inv, Some(inv_n));
            state
                .accumulate_fdl(head, 0, 1, active_mask)
                .expect("accum p0");
            let gpu_acc = state.inverse_acc_to_host().expect("inverse acc");
            assert_eq!(
                gpu_acc, cpu_acc,
                "UPOLA partition-0 accumulation differs at block {blk}"
            );

            // Tail partitions p = 1..n_parts from the stored rows.
            let mut cpu_tail = vec![0u64; 2 * n];
            for p in 1..n_parts {
                let slot = (head + fdl_slots - (p - 1)) % fdl_slots;
                for ch in 0..slots {
                    if (active_mask >> ch) & 1 == 0 {
                        continue;
                    }
                    for k in 0..n {
                        let xv = cpu_fdl[(slot * slots + ch) * n + k];
                        let off = (p * slots + ch) * n + k;
                        cpu_tail[k] = gl_add_ref(cpu_tail[k], gl_mul_ref(xv, kern_l[off]));
                        cpu_tail[n + k] = gl_add_ref(cpu_tail[n + k], gl_mul_ref(xv, kern_r[off]));
                    }
                }
            }
            glm_ntt_ref(&mut cpu_tail[..n], &inv, Some(inv_n));
            glm_ntt_ref(&mut cpu_tail[n..], &inv, Some(inv_n));
            state
                .accumulate_fdl(head, 1, n_parts, active_mask)
                .expect("accum tail");
            let gpu_tail = state.inverse_acc_to_host().expect("inverse tail");
            assert_eq!(
                gpu_tail, cpu_tail,
                "UPOLA tail accumulation differs at block {blk}"
            );
        }

        // Scratch (sub-block / single-partition) path: partition-0 from [slots][n] scratch.
        let mut scratch_ref = vec![vec![0u64; n]; slots];
        for (ch, slot_ref) in scratch_ref.iter_mut().enumerate() {
            if (active_mask >> ch) & 1 == 0 {
                continue;
            }
            let mut x = inputs[0].clone();
            glm_ntt_ref(&mut x, &fwd, None);
            slot_ref.copy_from_slice(&x);
            state
                .upload_scratch_row(ch, &inputs[0])
                .expect("upload scratch row");
        }
        state
            .forward_scratch(active_mask)
            .expect("fused scratch forward");
        let mut cpu_scratch = vec![0u64; 2 * n];
        for (ch, x) in scratch_ref.iter().enumerate() {
            if (active_mask >> ch) & 1 == 0 {
                continue;
            }
            for k in 0..n {
                let off = ch * n + k;
                cpu_scratch[k] = gl_add_ref(cpu_scratch[k], gl_mul_ref(x[k], kern_l[off]));
                cpu_scratch[n + k] = gl_add_ref(cpu_scratch[n + k], gl_mul_ref(x[k], kern_r[off]));
            }
        }
        glm_ntt_ref(&mut cpu_scratch[..n], &inv, Some(inv_n));
        glm_ntt_ref(&mut cpu_scratch[n..], &inv, Some(inv_n));
        state
            .mac_scratch_partition0(active_mask)
            .expect("scratch partition-0 mac");
        let gpu_scratch = state.inverse_acc_to_host().expect("inverse scratch");
        assert_eq!(
            gpu_scratch, cpu_scratch,
            "UPOLA scratch partition-0 differs from CPU"
        );

        state.reset().expect("reset upola state");
    }

    /// Before/after measurement on the same host (run with `--ignored --nocapture`).
    /// Prints the per-transform wall time for the multi-launch path vs the fused path and
    /// the per-block time for the stateless UPOLA loop vs the VRAM-resident state.
    #[test]
    #[ignore = "live GPU benchmark; run with --ignored --nocapture"]
    fn bench_fused_ntt_and_upola_residency() {
        let Ok(rt) = G6meCudaRuntime::init() else {
            eprintln!("bench skipped: no CUDA GPU/driver/nvcc on this host");
            return;
        };
        use std::time::Instant;

        let n = 8192usize;
        let (fwd, inv, inv_n) = glm_plan(n);
        let input = test_field_vec(0x5EED_0000_0000_0001, n);
        let iters = 50usize;

        // Warm-up + correctness sanity.
        let expect = rt
            .run_goldilocks_ntt_transform(&input, &fwd, None)
            .expect("fused forward");
        assert_eq!(
            rt.run_goldilocks_ntt_transform_legacy(&input, &fwd, None)
                .expect("legacy forward"),
            expect
        );

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = rt
                .run_goldilocks_ntt_transform_legacy(&input, &fwd, None)
                .expect("legacy forward");
            std::hint::black_box(out);
        }
        let legacy_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let out = rt
                .run_goldilocks_ntt_transform(&input, &fwd, None)
                .expect("fused forward");
            std::hint::black_box(out);
        }
        let fused_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Device-resident UPOLA block vs the stateless per-buffer loop.
        let slots = 6usize;
        let n_parts = 8usize;
        let fdl_slots = n_parts - 1;
        let kern_l = test_field_vec(0xA5A5_0001, n_parts * slots * n);
        let kern_r = test_field_vec(0xA5A5_0002, n_parts * slots * n);
        let mut state = rt
            .create_upola_state(n, slots, n_parts, inv_n, &kern_l, &kern_r, &fwd, &inv)
            .expect("create CudaUpolaState");
        let mask: u32 = 0b11_1111; // six active channels
        let blocks = 20usize;

        // Legacy block: per-channel forward, host pointwise accumulation, two inverses.
        let host_in: Vec<Vec<u64>> = (0..slots)
            .map(|ch| test_field_vec(0xBEEF_0000 + ch as u64, n))
            .collect();
        let t0 = Instant::now();
        for _ in 0..blocks {
            let mut acc = vec![0u64; 2 * n];
            for (ch, input) in host_in.iter().enumerate() {
                let x = rt
                    .run_goldilocks_ntt_transform_legacy(input, &fwd, None)
                    .expect("legacy forward");
                for k in 0..n {
                    let off = ch * n + k;
                    acc[k] = gl_add_ref(acc[k], gl_mul_ref(x[k], kern_l[off]));
                    acc[n + k] = gl_add_ref(acc[n + k], gl_mul_ref(x[k], kern_r[off]));
                }
            }
            let l = rt
                .run_goldilocks_ntt_transform_legacy(&acc[..n], &inv, Some(inv_n))
                .expect("legacy inverse l");
            let r = rt
                .run_goldilocks_ntt_transform_legacy(&acc[n..], &inv, Some(inv_n))
                .expect("legacy inverse r");
            std::hint::black_box((l, r));
        }
        let legacy_block_us = t0.elapsed().as_secs_f64() * 1e6 / blocks as f64;

        // Device-resident block: uploads + fused forwards + accumulate + batched inverse.
        let t0 = Instant::now();
        for blk in 0..blocks {
            let head = blk % fdl_slots;
            state.zero_fdl_row(head).expect("zero row");
            for (ch, input) in host_in.iter().enumerate() {
                state.upload_fdl_row(head, ch, input).expect("upload row");
            }
            state.forward_fdl_row(head, mask).expect("forward row");
            state.accumulate_fdl(head, 0, 1, mask).expect("accum p0");
            let _ = state.inverse_acc_to_host().expect("inverse p0");
            state
                .accumulate_fdl(head, 1, n_parts, mask)
                .expect("accum tail");
            let _ = state.inverse_acc_to_host().expect("inverse tail");
        }
        let resident_block_us = t0.elapsed().as_secs_f64() * 1e6 / blocks as f64;

        eprintln!(
            "[bench] N={n} forward transform: multi-launch {legacy_us:.1} us, fused {fused_us:.1} us, \
             speedup {:.2}x",
            legacy_us / fused_us
        );
        eprintln!(
            "[bench] one {slots}-channel {n_parts}-partition UPOLA block: stateless {legacy_block_us:.1} us, \
             VRAM-resident {resident_block_us:.1} us, speedup {:.2}x",
            legacy_block_us / resident_block_us
        );
        eprintln!(
            "[bench] kernel launches per block: stateless ~{} ({} transforms x 14-15 stages), \
             resident 4 (+{} htod +1 memset)",
            (slots + 2) * 14 + 2,
            slots + 2,
            slots
        );

        // Small-N sweep: a single 8192-point transform is the fused kernel's worst case
        // (one block, 2^13 butterflies per stage); smaller transforms trade launch count
        // against block occupancy differently.
        for &sn in &[256usize, 1024, 2048, 4096] {
            let (sfwd, _, _) = glm_plan(sn);
            let sinput = test_field_vec(0x5EED_0000_0000_0001, sn);
            let s_expect = rt
                .run_goldilocks_ntt_transform_legacy(&sinput, &sfwd, None)
                .expect("legacy sweep");
            assert_eq!(
                rt.run_goldilocks_ntt_transform(&sinput, &sfwd, None)
                    .expect("fused sweep"),
                s_expect
            );
            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(
                    rt.run_goldilocks_ntt_transform_legacy(&sinput, &sfwd, None)
                        .expect("legacy sweep"),
                );
            }
            let l_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            let t0 = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(
                    rt.run_goldilocks_ntt_transform(&sinput, &sfwd, None)
                        .expect("fused sweep"),
                );
            }
            let f_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            eprintln!(
                "[bench] N={sn} forward transform: multi-launch {l_us:.1} us, fused {f_us:.1} us, \
                 speedup {:.2}x",
                l_us / f_us
            );
        }
        assert!(fused_us.is_finite() && resident_block_us.is_finite());
    }
}
