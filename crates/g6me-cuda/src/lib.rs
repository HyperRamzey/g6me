//! Optional zero-FMA CUDA / Multi-Architecture Tensor Core GPU backend and real
//! `nvcc --fmad=false` + `cuobjdump --dump-sass` hardware SASS auditor for G6ME.
//!
//! - Uses `cudarc` with dynamic loading (`libcuda.so.1` loaded at runtime via `dlopen`) so binaries
//!   built with `--features cuda` still run cleanly on CPU-only machines and fall back gracefully.
//! - Embeds and loads ALL 6 real CUDA C++ kernels into [`G6meCudaRuntime`]:
//!   1. `fdtd_3d.cu` — zero-FMA 3D SRL FDTD acoustic wave solver (`f64`),
//!   2. `raytrace_hoa.cu` — zero-FMA deterministic PCG64 stochastic ray tracer (`f64`),
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
    /// `sm_120a` → `NVFP4 E2M1`, `sm_89+` → `FP8 E4M3`, `sm_75+` → `INT8`.
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
    /// - Blackwell Ultra (`cc == 103`, B300) drops native INT8 TC rate, so `Auto` picks `Nvfp4E2m1` there.
    /// - Explicit `Nvfp4E2m1` / `Fp8E4m3` / `Fp16` selections remain honored whenever hardware supports them.
    pub fn resolve_for_arch(self, major: i32, minor: i32, compiled_arch: &str) -> Self {
        let cc = major * 10 + minor;
        let has_nvfp4 = (cc >= 120 || cc == 100 || cc == 103) && compiled_arch.ends_with('a');
        let has_fp8 = cc >= 89;
        let has_fp16_tc = cc >= 75;
        match self {
            Self::Auto => {
                if cc == 103 && has_nvfp4 {
                    Self::Nvfp4E2m1
                } else {
                    Self::Int8
                }
            }
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
pub const ALL_CUDA_ARCH_PLANS: [CudaArchPlan; 6] = [
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
    } else if cc >= 90 {
        ALL_CUDA_ARCH_PLANS[1]
    } else if cc >= 89 {
        ALL_CUDA_ARCH_PLANS[2]
    } else if cc >= 80 {
        ALL_CUDA_ARCH_PLANS[3]
    } else if cc >= 75 {
        ALL_CUDA_ARCH_PLANS[4]
    } else {
        ALL_CUDA_ARCH_PLANS[5]
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
        is_blackwell: major >= 12,
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
}

#[inline]
fn align_up(v: usize, align: usize) -> usize {
    v.div_ceil(align) * align
}

/// Fixed-point scaling exponent for the Ozaki-II Stage-1 limb decomposition.
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

impl G6meCudaRuntime {
    /// Initialize CUDA device 0 and compile + load all 6 zero-FMA PTX modules via `nvcc --fmad=false`.
    pub fn init() -> Result<Self, String> {
        let info = cuda_device_info().ok_or_else(|| "no CUDA GPU detected".to_string())?;
        let ctx = CudaContext::new(info.index)
            .map_err(|e| format!("failed to create CUDA context: {e}"))?;
        let stream = ctx.default_stream();
        let arch = pick_supported_nvcc_arch(Some(&info.sm_arch()))?;

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
    pub fn run_bem_matvec(
        &self,
        mat_re: &[f64],
        mat_im: &[f64],
        x_re: &[f64],
        x_im: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = x_re.len();
        if x_im.len() != n || mat_re.len() != n * n || mat_im.len() != n * n {
            return Err("BEM matvec dimension mismatch".into());
        }
        let d_mr = self
            .stream
            .clone_htod(mat_re)
            .map_err(|e| format!("htod mat_re: {e}"))?;
        let d_mi = self
            .stream
            .clone_htod(mat_im)
            .map_err(|e| format!("htod mat_im: {e}"))?;
        let d_xr = self
            .stream
            .clone_htod(x_re)
            .map_err(|e| format!("htod x_re: {e}"))?;
        let d_xi = self
            .stream
            .clone_htod(x_im)
            .map_err(|e| format!("htod x_im: {e}"))?;
        let mut d_yr = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc y_re: {e}"))?;
        let mut d_yi = self
            .stream
            .alloc_zeros::<f64>(n)
            .map_err(|e| format!("alloc y_im: {e}"))?;

        let func: CudaFunction = self
            .bem_mod
            .load_function("g6me_bem_complex_matvec")
            .map_err(|e| format!("load g6me_bem_complex_matvec: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let mut builder = self.stream.launch_builder(&func);
        builder
            .arg(&d_mr)
            .arg(&d_mi)
            .arg(&d_xr)
            .arg(&d_xi)
            .arg(&mut d_yr)
            .arg(&mut d_yi)
            .arg(&n_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch bem_matvec: {e}"))?;

        let yr = self
            .stream
            .clone_dtoh(&d_yr)
            .map_err(|e| format!("dtoh y_re: {e}"))?;
        let yi = self
            .stream
            .clone_dtoh(&d_yi)
            .map_err(|e| format!("dtoh y_im: {e}"))?;
        let _ = &self.ctx;
        Ok((yr, yi))
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

    /// Execute the zero-FMA stochastic PCG64 acoustic ray-tracing pass on the GPU.
    /// Returns `(ray_delays, ray_energies, ray_dirs)` of lengths `n_rays * max_bounces` (and `* 3`).
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
        // Blackwell Ultra sm_103a (B300): Auto -> NVFP4 OMMA.SF
        assert_eq!(
            CudaTensorPrecision::Auto.resolve_for_arch(10, 3, "sm_103a"),
            CudaTensorPrecision::Nvfp4E2m1
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
}
