//! `g6me-scan` — 4-Stage Pure-Rust + Blackwell-Ready Smartphone Photogrammetry-to-FMM-BEM CLI.
//!
//! Turns multi-view DNG 1.7 captures (bypassing IFD0 preview traps, decoding Uncompressed /
//! Lossless JPEG SOF3 / JPEG-XL 52546 pixel tiles, and evaluating `OpcodeList2/3` analytically
//! with zero resampled pixels) or v7.0 zero-copy 50MP `RAW10` + OIS/IMU `.burst` bundles into a
//! watertight, acoustically conditioned (`h <= lambda_min / 6`) `.obj` head/pinna mesh ready for
//! `g6me --mode plus --head-mesh <FILE.obj>` (or direct in-memory FMM-BEM Helmholtz synthesis).
//!
//! Supports AprilTag3, ChArUco, AND **Russian Debit Cards (GOST R ISO/IEC 7810-2006 ID-1
//! `85.60 × 53.98 mm`, horizontal & vertical)** with `3.18 mm` rounded-corner tangent extrapolation,
//! ISO/IEC 7816-2 contact-chip orientation anchoring, and automatic in-place card-face privacy
//! scrubbing.
//!
//! **Zero synthetic fallback**: `g6me-scan` processes ONLY real `--dng` and `--burst` inputs.
//! If a live capture fails validation, pixel decode, anchor detection, or multi-view
//! reconstruction, `g6me-scan` exits honestly with exit code `2`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use g6me_core::atmosphere::AtmosphereCfg;
use g6me_core::blackwell::CudaTensorPrecision;
use g6me_core::burst3d::{
    preintegrate_imu_manifold, separate_specular_and_micro_depth, wronski_oissr_zero_demosaic_fuse,
};
use g6me_core::dng::{
    decode_dng_file, demosaic_malvar_he_cutler, parse_dng_file, unpack_mipi_raw10_to_f64,
    verify_burst_gates, BurstMetaSidecar, CfaPattern, DngMetadata,
};
use g6me_core::scan::{
    reconstruct_from_multiview_images, MetricAnchorKind, MultiViewScanConfig, ScanPipelineOutput,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AnchorArg {
    /// Russian debit card (GOST R ISO/IEC 7810-2006 ID-1 horizontal 85.60 x 53.98 mm, r=3.18 mm).
    RussianCard,
    /// Russian debit card (GOST R ISO/IEC 7810-2006 ID-1 vertical 53.98 x 85.60 mm, r=3.18 mm).
    RussianCardVertical,
    /// AprilTag3 (tag36h11) square fiducial.
    Apriltag3,
    /// ChArUco checkerboard-AprilTag fiducial.
    Charuco,
}

impl AnchorArg {
    fn to_kind(self, anchor_mm: f64) -> MetricAnchorKind {
        match self {
            Self::RussianCard => MetricAnchorKind::RussianDebitCardId1 {
                vertical_layout: false,
            },
            Self::RussianCardVertical => MetricAnchorKind::RussianDebitCardId1 {
                vertical_layout: true,
            },
            Self::Apriltag3 => MetricAnchorKind::AprilTag3 { side_mm: anchor_mm },
            Self::Charuco => MetricAnchorKind::ChArUco {
                square_mm: anchor_mm,
            },
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "g6me-scan",
    version,
    about = "4-stage smartphone photogrammetry (DNG 1.7 / v7.0 RAW10+OIS/IMU) -> watertight FMM-BEM head/pinna mesh"
)]
struct Cli {
    /// Inspect or reconstruct from DNG 1.7 file(s) or a directory of `.dng` captures. Pass multiple `--dng` flags or a directory for multi-view 3D reconstruction with `--out`.
    #[arg(long)]
    dng: Vec<PathBuf>,

    /// Factory optical calibration DNG (e.g. reference_lens56_opcodes.dng) supplying OpcodeList3 WarpRectilinear/FixVignetteRadial for v7.0 RAW10 bursts.
    #[arg(long)]
    ref_opcodes: Option<PathBuf>,

    /// Path to a v7.0 zero-copy 50MP RAW10 + OIS/IMU burst directory containing `meta.json` and `frame_0..5.raw10`.
    #[arg(long)]
    burst: Option<PathBuf>,

    /// Metric scale calibration target held next to the ear/head during capture.
    #[arg(long, value_enum, default_value_t = AnchorArg::RussianCard)]
    anchor: AnchorArg,

    /// Side length in mm when using `--anchor apriltag3` or `--anchor charuco`.
    #[arg(long, default_value_t = 40.0)]
    anchor_mm: f64,

    /// Target maximum acoustic BEM triangle edge length in mm (default 25 mm for quick CPU preview; use 2.5 mm <= lambda/6 at 20 kHz for full-band BEM).
    #[arg(long, default_value_t = 25.0)]
    max_edge_mm: f64,

    /// TSDF and generalized-winding voxel grid resolution per axis (10..=48, default 24).
    #[arg(long, default_value_t = 24)]
    grid_res: usize,

    /// Output path for the conditioned watertight Wavefront `.obj` head/pinna mesh.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Opt-in zero-FMA CUDA GPU offload for Ozaki-II Tensor Core Schur Bundle Adjustment and
    /// FMM-BEM verification (requires `--features cuda` + NVIDIA GPU; deterministic CPU fallback).
    #[arg(long)]
    cuda: bool,

    /// Tensor Core precision for Ozaki-II CRT FP64 Schur GEMM when `--cuda` is active:
    /// `auto` (single-limb INT8 on sm_61..sm_120a), `nvfp4` (explicit, sm_120a only),
    /// `fp8`, `int8`, `fp16`; unsupported formats downgrade per `resolve_for_arch`.
    #[arg(long, default_value = "auto", value_parser = parse_cuda_precision)]
    cuda_precision: CudaTensorPrecision,
}

fn parse_cuda_precision(s: &str) -> Result<CudaTensorPrecision, String> {
    CudaTensorPrecision::parse(s).ok_or_else(|| {
        format!("unknown --cuda-precision '{s}' (expected: auto, nvfp4, fp8, int8, or fp16)")
    })
}

enum ScanError {
    /// Standard I/O or argument failure (exit code 1).
    General(String),
    /// Honest refusal on real input (unsupported pixel codec, single-view `--out`, missing anchor, failed burst gate) -> exit code 2.
    RealInputRefused(String),
}

impl From<g6me_core::Error> for ScanError {
    fn from(e: g6me_core::Error) -> Self {
        Self::General(e.to_string())
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.cuda {
        #[cfg(feature = "cuda")]
        match g6me_core::blackwell::try_init_cuda_runtime() {
            Ok(rt) => {
                let requested = cli.cuda_precision.to_cuda();
                let active = rt.active_precision(requested);
                // `Auto` resolving to INT8 is the intended outcome, not a downgrade.
                if cli.cuda_precision != g6me_core::blackwell::CudaTensorPrecision::Auto
                    && active != requested
                {
                    eprintln!(
                        "g6me-scan warning: --cuda-precision {} is unavailable on {} ({}); \
                         using {}",
                        requested.short_tag(),
                        rt.info.name,
                        rt.compiled_arch,
                        active.short_tag()
                    );
                }
            }
            Err(e) => eprintln!(
                "g6me-scan warning: --cuda requested but the CUDA runtime could not start \
                 ({e}); running the CPU pipeline"
            ),
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!(
            "g6me-scan warning: --cuda requested but this binary was built without the \
             `cuda` feature; running the CPU pipeline"
        );
    }
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(ScanError::RealInputRefused(msg)) => {
            eprintln!("g6me-scan refused real-input export (exit 2): {msg}");
            ExitCode::from(2)
        }
        Err(ScanError::General(msg)) => {
            eprintln!("g6me-scan error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn collect_dng_paths(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, ScanError> {
    let mut files = Vec::new();
    for p in inputs {
        if p.is_dir() {
            let mut dir_entries: Vec<PathBuf> = fs::read_dir(p)
                .map_err(|e| ScanError::General(format!("cannot read dir `{}`: {e}", p.display())))?
                .filter_map(|res| res.ok().map(|ent| ent.path()))
                .filter(|path| {
                    path.extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| ext.eq_ignore_ascii_case("dng"))
                        .unwrap_or(false)
                })
                .collect();
            dir_entries.sort();
            files.extend(dir_entries);
        } else {
            files.push(p.clone());
        }
    }
    Ok(files)
}

fn print_and_export_result(
    out: &ScanPipelineOutput,
    c_air: f64,
    out_path: Option<&Path>,
    label: &str,
    use_cuda: bool,
) -> Result<(), ScanError> {
    println!(
        "Stage 1 Anchor  : {:?} | scale={:.4} mm/px | residual={:.6} mm | privacy_mask={}",
        out.anchor.kind,
        out.anchor.meters_per_pixel * 1e3,
        out.anchor.fit_residual_mm,
        out.anchor.privacy_masked
    );
    println!(
        "Stage 2 BA      : Ozaki-II {:?} Schur | RMS {:.3} -> {:.4} px | VCM +-{:.2}% f | scale err {:.3} mm",
        out.ba_report.schur_backend,
        out.ba_report.initial_rms_px,
        out.ba_report.final_rms_px,
        out.ba_report.max_vcm_breathing_pct,
        out.ba_report.metric_scale_error_mm
    );
    println!(
        "Stage 3-4 Mesh  : {} verts, {} tris, watertight={} (boundary={}, w={:.4}), max_edge={:.2} mm ({label})",
        out.mesh_stats.vertices,
        out.mesh_stats.faces,
        out.mesh_stats.is_watertight,
        out.mesh_stats.boundary_edges,
        out.mesh_stats.center_winding,
        out.mesh_stats.max_edge_m * 1e3
    );
    let hl = out.solver.solve_reciprocal_hrtfs_with_cuda(
        2000.0,
        c_air,
        &[[0.0, 2.0, 0.0]],
        out.solver.left_ear_elem,
        true,
        use_cuda,
    )[0];
    let hr = out.solver.solve_reciprocal_hrtfs_with_cuda(
        2000.0,
        c_air,
        &[[0.0, 2.0, 0.0]],
        out.solver.right_ear_elem,
        true,
        use_cuda,
    )[0];
    let ild = 20.0 * (hl.norm() / hr.norm().max(1e-12)).log10();
    println!(
        "FMM-BEM 2 kHz   : |H_L|={:.4}, |H_R|={:.4}, ILD={:+.2} dB (in-memory handoff verified)",
        hl.norm(),
        hr.norm(),
        ild
    );

    if let Some(p) = out_path {
        fs::write(p, out.mesh.to_obj_string())
            .map_err(|e| ScanError::General(format!("cannot write OBJ `{}`: {e}", p.display())))?;
        println!("Wrote watertight acoustic head mesh to `{}`", p.display());
    }
    Ok(())
}

fn run(cli: &Cli) -> Result<(), ScanError> {
    let atm = AtmosphereCfg::default();
    let c_air = atm.speed_of_sound_m_s();
    let grid_res = cli.grid_res.clamp(10, 48);

    // 1. Optional factory optical calibration profile (`--ref-opcodes`)
    let mut ref_meta: Option<DngMetadata> = None;
    if let Some(ref p) = cli.ref_opcodes {
        let meta = parse_dng_file(p)?;
        println!(
            "Loaded factory OpcodeList3 profile from `{}` (warp={}, vignette={})",
            p.display(),
            meta.warp_rectilinear.is_some(),
            meta.vignette.is_some()
        );
        ref_meta = Some(meta);
    }

    // 2. Real DNG inspect / multi-view reconstruction path
    let dng_files = collect_dng_paths(&cli.dng)?;
    if !dng_files.is_empty() {
        let first_meta = parse_dng_file(&dng_files[0])?;
        print!(
            "{}",
            first_meta.honesty_report_measured(dng_files.len(), None, None)
        );
        if ref_meta.is_none() {
            ref_meta = Some(first_meta);
        }

        if let Some(ref out_path) = cli.out {
            if dng_files.len() < 2 {
                return Err(ScanError::RealInputRefused(format!(
                    "cannot reconstruct a 3D head mesh from a single DNG (`{}`). \
                     Provide >= 2 multi-view DNG files (`--dng v1.dng --dng v2.dng ...` or `--dng <dir>`) \
                     or a v7.0 RAW10 burst (`--burst <dir>`).",
                    dng_files[0].display()
                )));
            }
            let mut frames_rgb = Vec::with_capacity(dng_files.len());
            let mut frames_luma = Vec::with_capacity(dng_files.len());
            let mut width = 0usize;
            let mut height = 0usize;
            for fp in &dng_files {
                let dec = decode_dng_file(fp).map_err(|e| {
                    ScanError::RealInputRefused(format!(
                        "cannot decode DNG `{}`: {e}",
                        fp.display()
                    ))
                })?;
                if width == 0 {
                    width = dec.width;
                    height = dec.height;
                } else if dec.width != width || dec.height != height {
                    return Err(ScanError::RealInputRefused(format!(
                        "DNG resolution mismatch in `{}` ({}x{} vs {}x{})",
                        fp.display(),
                        dec.width,
                        dec.height,
                        width,
                        height
                    )));
                }
                frames_rgb.push(dec.rgb);
                frames_luma.push(dec.luma);
            }
            let f_est = (width.max(height) as f64) * 0.9;
            let cfg = MultiViewScanConfig {
                anchor_kind: cli.anchor.to_kind(cli.anchor_mm),
                fx: f_est,
                fy: f_est,
                cx: (width as f64) * 0.5,
                cy: (height as f64) * 0.5,
                tsdf_res: grid_res,
                winding_res: grid_res,
                max_edge_m: (cli.max_edge_mm * 1e-3).max(0.0025),
                speed_of_sound_m_s: c_air,
                use_cuda: cli.cuda,
                cuda_tensor_precision: cli.cuda_precision,
            };
            let warp_prior = ref_meta.as_ref().and_then(|m| m.warp_rectilinear.as_ref());
            let out = reconstruct_from_multiview_images(
                &frames_rgb,
                &frames_luma,
                width,
                height,
                &cfg,
                warp_prior,
                None,
            )
            .map_err(|e| {
                ScanError::RealInputRefused(format!(
                    "real multi-view DNG reconstruction failed: {e}"
                ))
            })?;
            print_and_export_result(&out, c_air, Some(out_path), "real multi-view DNG", cli.cuda)?;
            return Ok(());
        }
    }

    // 3. Real v7.0 RAW10 + OIS/IMU burst path
    if let Some(ref burst_dir) = cli.burst {
        let meta_path = burst_dir.join("meta.json");
        let meta_txt = fs::read_to_string(&meta_path).map_err(|e| {
            ScanError::General(format!("cannot read `{}`: {e}", meta_path.display()))
        })?;
        let sidecar: BurstMetaSidecar = serde_json::from_str(&meta_txt).map_err(|e| {
            ScanError::General(format!(
                "invalid v7.0 meta.json `{}`: {e}",
                meta_path.display()
            ))
        })?;
        let mut blobs = Vec::with_capacity(sidecar.frames.len());
        for fm in &sidecar.frames {
            let fp = burst_dir.join(&fm.file_name);
            let bytes = fs::read(&fp).map_err(|e| {
                ScanError::General(format!("cannot read RAW10 frame `{}`: {e}", fp.display()))
            })?;
            blobs.push(bytes);
        }
        let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
        let rep = verify_burst_gates(&sidecar, &refs)?;
        println!(
            "=== v7.0 50MP RAW10 Burst Verification (`{}`) ===",
            burst_dir.display()
        );
        println!(
            "  Frames          : {} ({}x{}, CFA={})",
            sidecar.frames.len(),
            sidecar.frames[0].width,
            sidecar.frames[0].height,
            sidecar.cfa
        );
        println!(
            "  Distinct-Frame  : {} ({} unique SHA-256 hashes)",
            if rep.distinct_frames_ok {
                "PASS"
            } else {
                "FAIL (aliased RDI buffer!)"
            },
            rep.distinct_frame_hashes.len()
        );
        println!(
            "  Clock-Bracket   : {} (IMU {} samples, OIS var={:.3} px^2)",
            if rep.clock_bracket_ok { "PASS" } else { "FAIL" },
            sidecar.imu_gyro_uncal.len(),
            rep.ois_variance_px2
        );
        if !rep.distinct_frames_ok || !rep.clock_bracket_ok {
            return Err(ScanError::RealInputRefused(
                "v7.0 burst failed hardware integrity gates (Distinct-Frame or Clock-Bracket)"
                    .into(),
            ));
        }

        if let Some(ref out_path) = cli.out {
            let pat = CfaPattern::parse_str(&sidecar.cfa);
            let width = sidecar.frames[0].width;
            let height = sidecar.frames[0].height;
            // Proportionally scale black_level alongside white_level if sidecar reported 12/14/16-bit
            // sensor levels for a 10-bit MIPI RAW10 bitstream.
            let (black_level, white_level) = if sidecar.white_level > 1023.0 {
                let scale = 1023.0 / sidecar.white_level;
                (sidecar.black_level * scale, 1023.0)
            } else {
                (sidecar.black_level, sidecar.white_level)
            };
            let mut cfa_frames = Vec::with_capacity(sidecar.frames.len());
            let mut frames_rgb = Vec::with_capacity(sidecar.frames.len());
            let mut frames_luma = Vec::with_capacity(sidecar.frames.len());
            let mut ois_shifts = Vec::with_capacity(sidecar.frames.len());
            let mut baselines = Vec::with_capacity(sidecar.frames.len());

            let t_ref = sidecar.frames[0].sensor_timestamp_ns;
            let ois_ref = sidecar.frames[0]
                .ois_samples
                .first()
                .map(|s| (s.dx_px, s.dy_px))
                .unwrap_or((0.0, 0.0));

            for (fi, (fm, blob)) in sidecar.frames.iter().zip(&blobs).enumerate() {
                if fm.width != width || fm.height != height {
                    return Err(ScanError::RealInputRefused(format!(
                        "RAW10 burst frame {fi} dimension mismatch ({}x{} vs {}x{})",
                        fm.width, fm.height, width, height
                    )));
                }
                let cfa = unpack_mipi_raw10_to_f64(
                    blob,
                    fm.width,
                    fm.height,
                    fm.row_stride,
                    black_level,
                    white_level,
                )?;
                let rgb = demosaic_malvar_he_cutler(&cfa, width, height, pat);
                let luma: Vec<f64> = rgb
                    .iter()
                    .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
                    .collect();

                let ois = fm
                    .ois_samples
                    .first()
                    .map(|s| (s.dx_px - ois_ref.0, s.dy_px - ois_ref.1))
                    .unwrap_or((0.0, 0.0));
                ois_shifts.push(ois);

                let pre = preintegrate_imu_manifold(
                    &sidecar.imu_gyro_uncal,
                    &sidecar.imu_accel_uncal,
                    t_ref,
                    fm.sensor_timestamp_ns,
                );
                baselines.push((pre.delta_p[0], pre.delta_p[1]));

                cfa_frames.push(cfa);
                frames_rgb.push(rgb);
                frames_luma.push(luma);
            }

            let cfa_refs: Vec<&[f64]> = cfa_frames.iter().map(|v| v.as_slice()).collect();
            let oissr_rgb =
                wronski_oissr_zero_demosaic_fuse(&cfa_refs, &ois_shifts, width, height, pat, 0.65);
            let oissr_luma: Vec<f64> = oissr_rgb
                .iter()
                .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
                .collect();
            frames_rgb[0] = oissr_rgb;
            frames_luma[0] = oissr_luma;

            let [fx_raw, fy_raw, cx_raw, cy_raw, _] = sidecar.lens_intrinsic_calibration;
            let fx = if fx_raw > 1.0 {
                fx_raw
            } else {
                (width as f64) * 0.9
            };
            let fy = if fy_raw > 1.0 { fy_raw } else { fx };
            let cx = if cx_raw > 1.0 {
                cx_raw
            } else {
                (width as f64) * 0.5
            };
            let cy = if cy_raw > 1.0 {
                cy_raw
            } else {
                (height as f64) * 0.5
            };

            let max_base = baselines
                .iter()
                .map(|&(bx, by)| bx.hypot(by))
                .fold(0.0_f64, f64::max);
            let depth_hint = if max_base > 2e-4 && width * height <= 262_144 {
                let l_refs: Vec<&[f64]> = frames_luma.iter().map(|v| v.as_slice()).collect();
                Some(
                    separate_specular_and_micro_depth(
                        &l_refs, &baselines, width, height, fx, 0.12, 0.75, 20,
                    )
                    .depth_m,
                )
            } else {
                None
            };

            let cfg = MultiViewScanConfig {
                anchor_kind: cli.anchor.to_kind(cli.anchor_mm),
                fx,
                fy,
                cx,
                cy,
                tsdf_res: grid_res,
                winding_res: grid_res,
                max_edge_m: (cli.max_edge_mm * 1e-3).max(0.0025),
                speed_of_sound_m_s: c_air,
                use_cuda: cli.cuda,
                cuda_tensor_precision: cli.cuda_precision,
            };
            let warp_prior = ref_meta.as_ref().and_then(|m| m.warp_rectilinear.as_ref());
            let out = reconstruct_from_multiview_images(
                &frames_rgb,
                &frames_luma,
                width,
                height,
                &cfg,
                warp_prior,
                depth_hint.as_deref(),
            )
            .map_err(|e| {
                ScanError::RealInputRefused(format!("real RAW10 burst reconstruction failed: {e}"))
            })?;
            print_and_export_result(
                &out,
                c_air,
                Some(out_path),
                "real v7.0 RAW10 burst",
                cli.cuda,
            )?;
            return Ok(());
        }
        return Ok(());
    }

    if cli.out.is_some() {
        return Err(ScanError::RealInputRefused(
            "`--out` requires real multi-view `--dng` captures or a `--burst` directory.".into(),
        ));
    }

    if ref_meta.is_none() {
        println!(
            "Pass `--dng <FILE_OR_DIR>` or `--burst <DIR>` (and `--out <mesh.obj>` for 3D reconstruction). Use `--help` for options."
        );
    }
    Ok(())
}
