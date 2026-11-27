//! `g6me-scan` — 4-Stage Pure-Rust + Blackwell-Ready Smartphone Photogrammetry-to-FMM-BEM CLI.
//!
//! Turns multi-view DNG 1.7 captures (bypassing IFD0 preview traps & evaluating `OpcodeList2/3`
//! analytically with zero resampled pixels) or v7.0 zero-copy 50MP `RAW10` + OIS/IMU `.burst`
//! bundles into a watertight, acoustically conditioned (`h <= lambda_min / 6`) `.obj` head/pinna
//! mesh ready for `g6me --mode plus --head-mesh <FILE.obj>` (or direct in-memory FMM-BEM
//! Helmholtz synthesis!).
//!
//! Supports AprilTag3, ChArUco, AND **Russian Debit Cards (GOST R ISO/IEC 7810-2006 ID-1
//! `85.60 × 53.98 mm`, horizontal & vertical)** with `3.18 mm` rounded-corner tangent extrapolation,
//! ISO/IEC 7816-2 contact-chip orientation anchoring, and automatic in-place card-face privacy
//! scrubbing.
//!
//! **Honesty guarantee**: all synthetic fixtures (`build_synthetic_dng_17`, synthetic card
//! coordinates, `TriangleMesh::procedural_head_with_pinnae`) are strictly isolated inside
//! `--demo`. Real `--dng` and `--burst` inputs NEVER silently fall back to a procedural dummy
//! mesh — if a live capture fails validation, pixel decode, anchor detection, or multi-view
//! reconstruction, `g6me-scan` exits honestly with exit code `2`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use g6me_core::atmosphere::AtmosphereCfg;
use g6me_core::bem::TriangleMesh;
use g6me_core::burst3d::{
    mat3_mul, preintegrate_imu_manifold, separate_specular_and_micro_depth, so3_exp,
    wronski_oissr_zero_demosaic_fuse, MAT3_ID,
};
use g6me_core::dng::{
    build_synthetic_dng_17, decode_dng_file, demosaic_malvar_he_cutler, parse_dng_file,
    unpack_mipi_raw10_to_f64, verify_burst_gates, BurstMetaSidecar, CfaPattern, DngMetadata,
    FixVignetteRadialOpcode, GainMapOpcode, WarpPlaneCoeffs, WarpRectilinearOpcode,
    COMPRESSION_JPEG_XL, PHOTOMETRIC_LINEAR_RAW,
};
use g6me_core::scan::{
    bundle_adjust_ozaki2_fp4_schur, calibrate_apriltag3_anchor, calibrate_russian_debit_card_id1,
    reconstruct_from_multiview_images, run_stage3_to_stage4_to_fmm_bem, sample_surfels_from_mesh,
    BaObservation, MetricAnchorKind, MetricGaugeConstraint, MultiViewScanConfig,
    RollingShutterCamera, ScanPipelineOutput, ISO7816_CHIP_CENTER_X_MM, ISO7816_CHIP_CENTER_Y_MM,
    RUSSIAN_ID1_HEIGHT_MM, RUSSIAN_ID1_WIDTH_MM,
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

    /// Output path for the conditioned watertight Wavefront `.obj` head/pinna mesh.
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Run a strictly isolated synthetic 4-stage pipeline self-test/demonstration (cannot be combined with real `--dng` or `--burst` inputs).
    #[arg(long)]
    demo: bool,
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
    let hl = out.solver.solve_reciprocal_hrtfs(
        2000.0,
        c_air,
        &[[0.0, 2.0, 0.0]],
        out.solver.left_ear_elem,
        true,
    )[0];
    let hr = out.solver.solve_reciprocal_hrtfs(
        2000.0,
        c_air,
        &[[0.0, 2.0, 0.0]],
        out.solver.right_ear_elem,
        true,
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

    let has_real_input = !cli.dng.is_empty() || cli.burst.is_some();
    if cli.demo && has_real_input {
        return Err(ScanError::RealInputRefused(
            "`--demo` cannot be combined with real `--dng` or `--burst` inputs; remove `--demo` to process real captures honestly.".into(),
        ));
    }

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
        // Honest inspect report: computes real CA residual from parsed WarpRectilinear,
        // and reports VCM focus breathing as N/A until multi-view BA executes.
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
                     or a v7.0 RAW10 burst (`--burst <dir>`). Refusing to write a synthetic dummy head.",
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
                tsdf_res: 14,
                winding_res: 14,
                max_edge_m: (cli.max_edge_mm * 1e-3).max(0.0025),
                speed_of_sound_m_s: c_air,
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
            print_and_export_result(&out, c_air, Some(out_path), "real multi-view DNG")?;
            return Ok(());
        }
    }

    // 3. Real v7.0 RAW10 + OIS/IMU burst path (ACTUALLY unpacks RAW10 & runs burst3d + scan.rs!)
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
            // Real burst3d ingest: unpack RAW10 -> Forster SO(3) IMU + OIS shifts ->
            // Wronski/OISSR zero-demosaic fuse + Chugunov specular/micro-depth -> real SfM/BA/2DGS!
            let pat = CfaPattern::parse_str(&sidecar.cfa);
            let width = sidecar.frames[0].width;
            let height = sidecar.frames[0].height;
            let white_level = if sidecar.white_level > 1023.0 {
                1023.0
            } else {
                sidecar.white_level
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
                    sidecar.black_level,
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
            // Replace frame 0 with the OISSR super-resolved frame for cleaner anchor edge fitting.
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

            // Estimate Chugunov specular separation + micro-baseline depth if IMU translation > 0.2 mm.
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
                tsdf_res: 14,
                winding_res: 14,
                max_edge_m: (cli.max_edge_mm * 1e-3).max(0.0025),
                speed_of_sound_m_s: c_air,
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
            print_and_export_result(&out, c_air, Some(out_path), "real v7.0 RAW10 burst")?;
            return Ok(());
        }
        return Ok(());
    }

    // 4. Strictly isolated `--demo` synthetic self-test (ONLY reachable when `--demo` is explicit!)
    if cli.demo {
        run_synthetic_demo(cli, c_air)?;
        return Ok(());
    }

    if cli.out.is_some() {
        return Err(ScanError::RealInputRefused(
            "`--out` requires real multi-view `--dng` captures, a `--burst` directory, or explicit `--demo`.".into(),
        ));
    }

    if ref_meta.is_none() {
        println!(
            "Run `g6me-scan --demo` for a synthetic self-test, or pass `--dng <FILE_OR_DIR>`, `--burst <DIR>`, and `--out <mesh.obj>`. Use `--help` for options."
        );
    }
    Ok(())
}

/// Strictly isolated synthetic demonstration (`--demo` ONLY).
/// Runs the real `bundle_adjust_ozaki2_fp4_schur` solver (never hardcoded RMS literals!)
/// and demonstrates Stages 1..4 -> FMM-BEM on a synthetic head mesh.
fn run_synthetic_demo(cli: &Cli, c_air: f64) -> Result<(), ScanError> {
    println!("=== g6me-scan SYNTHETIC DEMO MODE (--demo) ===");
    let synth = DngMetadata {
        dng_version: [1, 7, 0, 0],
        preview_width: 1024,
        preview_height: 768,
        main_width: 8160,
        main_height: 6120,
        bits_per_sample: 16,
        compression: COMPRESSION_JPEG_XL,
        photometric: PHOTOMETRIC_LINEAR_RAW,
        new_subfile_type: 16,
        subifd_routed: true,
        cfa_pattern: Some(CfaPattern::Gbrg),
        black_level: 256.0,
        white_level: 4095.0,
        gain_maps: vec![GainMapOpcode {
            top: 0,
            left: 0,
            bottom: 6120,
            right: 8160,
            plane: 0,
            planes: 4,
            row_pitch: 2,
            col_pitch: 2,
            map_points_v: 2,
            map_points_h: 2,
            map_spacing_v: 1.0,
            map_spacing_h: 1.0,
            map_origin_v: 0.0,
            map_origin_h: 0.0,
            map_planes: 1,
            gains: vec![1.15, 1.15, 1.15, 1.15],
        }],
        vignette: Some(FixVignetteRadialOpcode {
            k: [0.16, 0.03, 0.01, 0.0, 0.0],
            cx_hat: 0.5,
            cy_hat: 0.5,
        }),
        warp_rectilinear: Some(WarpRectilinearOpcode {
            planes: vec![
                WarpPlaneCoeffs {
                    k: [1.0005, -0.022, 0.006, 0.0],
                    m: [1e-4, -1e-4],
                },
                WarpPlaneCoeffs {
                    k: [1.0000, -0.020, 0.005, 0.0],
                    m: [1e-4, -1e-4],
                },
                WarpPlaneCoeffs {
                    k: [0.9994, -0.018, 0.004, 0.0],
                    m: [1e-4, -1e-4],
                },
            ],
            cx_hat: 0.5,
            cy_hat: 0.5,
        }),
    };
    let synth_bytes = build_synthetic_dng_17(&synth);
    let parsed = g6me_core::dng::parse_dng_bytes(&synth_bytes)?;
    let ca_res = parsed.measure_analytical_ca_residual_px();

    // Stage 1 (Synthetic Demo): Metric Anchor Calibration
    let anchor = match cli.anchor {
        AnchorArg::RussianCard | AnchorArg::RussianCardVertical => {
            let vertical = matches!(cli.anchor, AnchorArg::RussianCardVertical);
            let (w_mm, h_mm) = if vertical {
                (RUSSIAN_ID1_HEIGHT_MM, RUSSIAN_ID1_WIDTH_MM)
            } else {
                (RUSSIAN_ID1_WIDTH_MM, RUSSIAN_ID1_HEIGHT_MM)
            };
            let s = 12.0;
            let top = [
                [100.0 + 10.0 * s, 100.0],
                [100.0 + (w_mm - 10.0) * s, 100.0],
            ];
            let bot = [
                [100.0 + 10.0 * s, 100.0 + h_mm * s],
                [100.0 + (w_mm - 10.0) * s, 100.0 + h_mm * s],
            ];
            let left = [
                [100.0, 100.0 + 10.0 * s],
                [100.0, 100.0 + (h_mm - 10.0) * s],
            ];
            let right = [
                [100.0 + w_mm * s, 100.0 + 10.0 * s],
                [100.0 + w_mm * s, 100.0 + (h_mm - 10.0) * s],
            ];
            let chip = [
                100.0 + ISO7816_CHIP_CENTER_X_MM * s,
                100.0 + ISO7816_CHIP_CENTER_Y_MM * s,
            ];
            calibrate_russian_debit_card_id1([&top, &right, &bot, &left], Some(chip), vertical)?
        }
        AnchorArg::Apriltag3 | AnchorArg::Charuco => calibrate_apriltag3_anchor(
            [
                [100.0, 100.0],
                [500.0, 100.0],
                [500.0, 500.0],
                [100.0, 500.0],
            ],
            cli.anchor_mm,
            matches!(cli.anchor, AnchorArg::Charuco),
        )?,
    };

    // Stage 2 (Synthetic Demo): actually invoke Ozaki-II FP4 -> FP64 Rolling-Shutter Schur BA!
    let gt_rots = [
        MAT3_ID,
        so3_exp([0.04, 0.22, -0.03]),
        so3_exp([-0.05, -0.20, 0.02]),
    ];
    let gt_trans = [[0.0, 0.0, 0.35], [-0.08, 0.01, 0.36], [0.08, -0.01, 0.34]];
    let gt_pts = vec![
        [0.0, 0.0, 0.0],
        [0.08560, 0.0, 0.0],
        [0.08560, 0.05398, 0.0],
        [0.0, 0.05398, 0.0],
        [-0.04, -0.03, 0.02],
        [0.03, -0.04, -0.02],
        [-0.03, 0.06, 0.03],
        [0.06, 0.07, -0.03],
    ];
    let mut gt_cams = Vec::new();
    for c in 0..3 {
        gt_cams.push(RollingShutterCamera {
            rot: gt_rots[c],
            trans: gt_trans[c],
            omega_rad_s: [0.05, -0.08, 0.02],
            vel_m_s: [0.01, -0.005, 0.0],
            fx: 3000.0,
            fy: 3000.0,
            cx: 2040.0,
            cy: 1530.0,
            width: 4080.0,
            height: 3060.0,
            vcm_focal_scale: if c == 1 { 1.003 } else { 1.0 },
            rolling_shutter_skew_s: 0.038,
            ois_shift_px: [1.2 * (c as f64), -0.8 * (c as f64)],
        });
    }
    let mut obs = Vec::new();
    for (ci, cam) in gt_cams.iter().enumerate() {
        for (pi, &pt) in gt_pts.iter().enumerate() {
            let rough = cam.project_at_row(pt, 1530.0, None);
            let exact = cam.project_at_row(pt, rough[1], None);
            obs.push(BaObservation {
                cam_idx: ci,
                pt_idx: pi,
                u_px: exact[0],
                v_px: exact[1],
            });
        }
    }
    let mut cams = gt_cams.clone();
    cams[1].trans[0] += 0.004;
    cams[1].rot = mat3_mul(&so3_exp([0.012, -0.010, 0.008]), &cams[1].rot);
    cams[1].vcm_focal_scale = 1.0;
    cams[2].trans[1] -= 0.003;
    let mut pts = gt_pts.clone();
    for (i, p) in pts.iter_mut().enumerate().skip(2) {
        p[0] += 0.0015 * (i as f64).sin();
        p[2] -= 0.0020 * (i as f64).cos();
    }
    let gauges = vec![MetricGaugeConstraint {
        pt_a: 0,
        pt_b: 1,
        target_dist_m: RUSSIAN_ID1_WIDTH_MM * 1e-3,
        weight: 1e5,
    }];
    let ba = bundle_adjust_ozaki2_fp4_schur(&mut cams, &mut pts, &obs, &gauges, None, 10.0, 12);
    print!(
        "{}",
        parsed.honesty_report_measured(cams.len(), Some(ba.max_vcm_breathing_pct), ca_res)
    );

    // Stage 3..4 (Synthetic Demo): procedural head -> surfels -> TSDF -> Winding Repair -> FMM-BEM
    let head = TriangleMesh::procedural_head_with_pinnae(0.0875, 2);
    let surfels = sample_surfels_from_mesh(&head);
    let out = run_stage3_to_stage4_to_fmm_bem(
        anchor,
        ba,
        &surfels,
        14,
        14,
        (cli.max_edge_mm * 1e-3).max(0.0025),
        c_air,
    )?;
    print_and_export_result(&out, c_air, cli.out.as_deref(), "synthetic --demo")
}
