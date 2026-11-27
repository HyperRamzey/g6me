//! Optional zero-FMA CUDA / Blackwell (`sm_120`) GPU backend and real `nvcc --fmad=false` +
//! `cuobjdump --dump-sass` hardware SASS auditor for G6ME.
//!
//! - Uses `cudarc` with dynamic loading (`libcuda.so.1` loaded at runtime via `dlopen`) so binaries
//!   built with `--features cuda` still run cleanly on CPU-only machines and fall back gracefully.
//! - Embeds 6 real CUDA C++ kernels (`fdtd_3d.cu`, `raytrace_hoa.cu`, `stockham_fft_f64.cu`,
//!   `goldilocks_ntt.cu`, `bem_matvec.cu`, `ozaki2_blackwell_mma.cu`).
//! - Provides real `nvcc --fmad=false` PTX + `cuobjdump --dump-sass` verification over all 6
//!   kernels: zero contracted `fma.rn` in the PTX and zero contracted FMA in the hardware SASS,
//!   plus a mandatory `--fmad=true` sabotage negative control confirming both auditors catch
//!   contraction when `--fmad=false` is omitted.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

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
/// Embedded zero-FMA Blackwell NVFP4 / INT8 Ozaki-II CRT kernel (`ozaki2_blackwell_mma.cu`).
pub const KERNEL_OZAKI2_MMA_CU: &str = include_str!("../kernels/ozaki2_blackwell_mma.cu");

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
    /// own arithmetic. Every kernel must still emit zero `fma.rn` in its nvcc PTX, and every
    /// non-DFMA fused SASS form (`FFMA`/`HFMA2`/`DMAD`/`FMAD`) remains forbidden here too.
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
        // 1/c, 1/n_rays and the per-bounce ray/box hit parameters are IEEE divisions.
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
    /// Target `sm_XY` architecture string for `nvcc`.
    pub fn sm_arch(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
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

/// Live `cudarc` GPU runtime with dynamically compiled zero-FMA PTX modules.
pub struct G6meCudaRuntime {
    pub info: CudaDeviceInfo,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    fdtd_mod: Arc<CudaModule>,
    bem_mod: Arc<CudaModule>,
}

impl G6meCudaRuntime {
    /// Initialize CUDA device 0 and compile the zero-FMA PTX modules via `nvcc --fmad=false`.
    pub fn init() -> Result<Self, String> {
        let info = cuda_device_info().ok_or_else(|| "no CUDA GPU detected".to_string())?;
        let ctx = CudaContext::new(info.index)
            .map_err(|e| format!("failed to create CUDA context: {e}"))?;
        let stream = ctx.default_stream();
        let arch = pick_supported_nvcc_arch(Some(&info.sm_arch()))?;
        let fdtd_ptx = compile_cu_to_ptx("fdtd_3d", KERNEL_FDTD_3D_CU, &arch, false)?;
        let bem_ptx = compile_cu_to_ptx("bem_matvec", KERNEL_BEM_MATVEC_CU, &arch, false)?;
        let fdtd_mod = ctx
            .load_module(Ptx::from_src(fdtd_ptx))
            .map_err(|e| format!("failed to load fdtd_3d PTX module: {e}"))?;
        let bem_mod = ctx
            .load_module(Ptx::from_src(bem_ptx))
            .map_err(|e| format!("failed to load bem_matvec PTX module: {e}"))?;
        Ok(Self {
            info,
            ctx,
            stream,
            fdtd_mod,
            bem_mod,
        })
    }

    /// Execute one zero-FMA 3D SRL FDTD acoustic wave step on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn run_fdtd_3d_step(
        &self,
        p_prev: &[f64],
        p_curr: &[f64],
        inv_den: &[f64; 7],
        nx: usize,
        ny: usize,
        nz: usize,
        lam2: f64,
        half_lam_beta: f64,
    ) -> Result<Vec<f64>, String> {
        let n = nx * ny * nz;
        if p_prev.len() != n || p_curr.len() != n {
            return Err("FDTD buffer size mismatch".into());
        }
        let d_prev = self
            .stream
            .clone_htod(p_prev)
            .map_err(|e| format!("htod p_prev: {e}"))?;
        let d_curr = self
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

        let func: CudaFunction = self
            .fdtd_mod
            .load_function("g6me_fdtd_3d_srl_step")
            .map_err(|e| format!("load g6me_fdtd_3d_srl_step: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (
                (nx as u32).div_ceil(8),
                (ny as u32).div_ceil(8),
                (nz as u32).div_ceil(4),
            ),
            block_dim: (8, 8, 4),
            shared_mem_bytes: 0,
        };
        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let mut builder = self.stream.launch_builder(&func);
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
        unsafe { builder.launch(cfg) }.map_err(|e| format!("launch fdtd_3d: {e}"))?;

        self.stream
            .clone_dtoh(&d_next)
            .map_err(|e| format!("dtoh p_next: {e}"))
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
}

/// Result of compiling and auditing a single CUDA kernel's hardware SASS.
#[derive(Debug, Clone)]
pub struct KernelSassAudit {
    pub name: String,
    pub arch: String,
    pub sass_instructions: usize,
}

/// Audit nvcc-generated PTX for contracted multiply-add instructions.
///
/// `--fmad=false` is nvcc's only contraction point: a source-level `a*b + c` is emitted as
/// `mul.rn` + `add.rn`, so an `fma.rn.*` instruction in the PTX means either an explicit
/// source-level FMA intrinsic or an inlined libdevice polynomial (e.g. `cos`/`sin`). Both
/// are forbidden in the zero-FMA kernels. Returns the PTX instruction count on success.
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
///
/// Set `allow_ieee_dfma` only for kernels that call IEEE `div.rn.f64`/`sqrt.rn.f64`, and only
/// after [`audit_ptx_zero_fma`] has proven their PTX contains no contracted `fma.rn`. ptxas
/// implements each correctly-rounded operation as `MUFU.RCP64H`/`MUFU.RSQ64H` + DFMA Newton
/// refinement plus DFMA slow paths; permitting exactly `DFMA` (and its rounding-mode variants)
/// leaves those microcoded IEEE operations intact while `FFMA`/`HFMA2`/`DMAD`/`FMAD` and every
/// zero-multiplicand rule remain enforced.
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
/// preferring `preferred` (or `sm_120` Blackwell on CUDA 12.8+, down to `sm_75`).
pub fn pick_supported_nvcc_arch(preferred: Option<&str>) -> Result<String, String> {
    if !nvcc_toolchain_available() {
        return Err("nvcc/cuobjdump toolchain not installed".into());
    }
    let probe_src = "extern \"C\" __global__ void g6me_probe(double* p) { p[0] = 1.0; }\n";
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let cu_path = tmp.join(format!("g6me_arch_probe_{pid}.cu"));
    let cubin_path = tmp.join(format!("g6me_arch_probe_{pid}.cubin"));
    std::fs::write(&cu_path, probe_src).map_err(|e| format!("write probe .cu: {e}"))?;

    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = preferred {
        candidates.push(p.to_string());
    }
    for arch in ["sm_120", "sm_90", "sm_89", "sm_86", "sm_80", "sm_75"] {
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
    (
        tmp.join(format!("g6me_cuda_{tag}_{pid}.cu")),
        tmp.join(format!("g6me_cuda_{tag}_{pid}.out")),
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
        .args(["-O3", "-ptx"])
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
    cmd.args(["-O3", "-cubin"])
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

/// Compile ALL 6 embedded `.cu` kernels with `nvcc --fmad=false`, then verify two layers:
///
/// 1. **PTX contraction audit** — the nvcc PTX must contain zero `fma.rn` instructions. This is
///    exact: `--fmad=false` is the only point where nvcc can contract `a*b + c`, so a clean PTX
///    proves the kernel's arithmetic is unfused.
/// 2. **Hardware SASS audit** — `cuobjdump --dump-sass` must show zero contracted FMA forms.
///    Kernels marked [`CudaKernelSource::ieee_div_sqrt`] call IEEE correctly-rounded
///    `div.rn.f64`/`sqrt.rn.f64`, which ptxas expands into `MUFU` + `DFMA` microcode; those
///    `DFMA` implement one IEEE operation and are allowed, every other fused form is not.
pub fn audit_all_cuda_kernels_sass(
    preferred_arch: Option<&str>,
) -> Result<Vec<KernelSassAudit>, String> {
    let arch = pick_supported_nvcc_arch(preferred_arch)?;
    let mut results = Vec::with_capacity(ALL_CUDA_KERNELS.len());
    for kernel in ALL_CUDA_KERNELS {
        let name = kernel.name;
        let ptx = compile_cu_to_ptx(name, kernel.source, &arch, false)?;
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
        results.push(KernelSassAudit {
            name: name.to_string(),
            arch: arch.clone(),
            sass_instructions: count,
        });
    }
    Ok(results)
}

/// Sabotage negative control (AGENTS.md Sabotage Rule): compiles a contractible double-precision
/// multiply-add kernel with `nvcc --fmad=true` and verifies that BOTH auditors catch it — the PTX
/// must contain the contracted `fma.rn.f64`, and the SASS dumped with `cuobjdump --dump-sass`
/// must contain a `DFMA`. Returns the number of caught SASS `DFMA` offenders.
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
            // CPU-only CI without nvcc: still verify both auditors discriminate synthetic input.
            let clean = "/*0010*/ DMUL R4, R2, R6;\n/*0020*/ DADD R8, R4, R10;\n";
            assert_eq!(audit_sass_zero_fma(clean).expect("clean SASS"), 2);
            let dirty = "/*0010*/ DFMA R8, R2, R6, R10;\n";
            assert!(audit_sass_zero_fma(dirty).is_err());
            // The IEEE div/sqrt allowance must still reject FFMA and accept DFMA.
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
}
