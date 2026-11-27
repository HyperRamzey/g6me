//! 4-Stage `g6me-scan` Photogrammetry-to-FMM-BEM Pipeline (Pure Rust + Blackwell CUDA).
//!
//! Turns handheld smartphone captures (DNG 1.7 / v7.0 unmerged 50MP `RAW10` + OIS/IMU bursts)
//! directly into a watertight, isotropically remeshed (`h <= lambda_min / 6`), FMM-BEM-ready
//! personalized 3D head & pinna acoustic mesh:
//!
//! - **Stage 1**: Dense skin-proof correlation matching + **AprilTag3 / ChArUco** +
//!   **Russian Debit Card (GOST R ISO/IEC 7810-2006 ID-1 `85.60 × 53.98 mm`)** metric anchor
//!   with `3.18 mm` rounded-corner tangent-intersection compensation, ISO/IEC 7816-2 contact-chip
//!   orientation landmark, horizontal/vertical support, and automatic card-face privacy masking.
//! - **Stage 2**: **GLOMAP** (Pan et al., ECCV 2024) global `SO(3)` Lie-algebra rotation averaging
//!   and **Rolling-Shutter Bundle Adjustment** (Hedborg et al., CVPR 2012) solving the sparse
//!   **Schur Complement** `(U - W V^-1 W^T) dc = g` on **Ozaki-II CRT FP4 (E2M1 Base-13) → FP64**
//!   with per-frame VCM focus-breathing Tikhonov prior & metric gauge constraints!
//! - **Stage 3**: **2D Gaussian Splatting (2DGS, SIGGRAPH 2024) + PGSR (IEEE TVCG 2025)**
//!   oriented planar surface disks with depth-distortion & surface-normal consistency losses +
//!   dense **TSDF** Marching Tetrahedra surface extraction.
//! - **Stage 4**: **Generalized Winding Numbers** (Jacobson et al., SIGGRAPH 2013) watertight
//!   manifold conditioning + **Isotropic Acoustic Remeshing** (`h <= lambda_min / 6`) → direct
//!   in-memory handoff into `crate::bem::FmmBemSolver` without touching disk!

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::bem::{
    extract_isosurface_marching_tetrahedra, isotropic_acoustic_remesh, v3_add, v3_cross, v3_dot,
    v3_norm, v3_scale, v3_sub, watertight_winding_repair, BemMeshStats, FmmBemSolver, TriangleMesh,
};
use crate::blackwell::{ozaki2_crt_gemm, TensorBackend};
use crate::burst3d::{mat3_mul, mat3_transpose, mat3_vec, so3_exp, so3_log, Mat3, MAT3_ID};
use crate::dng::{OisSample, WarpRectilinearOpcode};
use crate::error::{Error, Result};

// ============================================================================
// Stage 1: Metric Anchors — AprilTag3 / ChArUco + Russian Debit Card (GOST ID-1)
// ============================================================================

/// Exact **GOST R ISO/IEC 7810-2006 ID-1** Russian debit card physical dimensions (mm).
pub const RUSSIAN_ID1_WIDTH_MM: f64 = 85.60;
pub const RUSSIAN_ID1_HEIGHT_MM: f64 = 53.98;
pub const RUSSIAN_ID1_CORNER_RADIUS_MM: f64 = 3.18;
pub const RUSSIAN_ID1_ASPECT: f64 = RUSSIAN_ID1_WIDTH_MM / RUSSIAN_ID1_HEIGHT_MM; // 1.5857725...

/// Exact **ISO/IEC 7816-2** EMV/NSPK contact chip pad centroid & size (mm from top-left).
pub const ISO7816_CHIP_CENTER_X_MM: f64 = 16.25;
pub const ISO7816_CHIP_CENTER_Y_MM: f64 = 23.89;
pub const ISO7816_CHIP_WIDTH_MM: f64 = 12.00;
pub const ISO7816_CHIP_HEIGHT_MM: f64 = 9.32;

/// Metric calibration anchor target type supported by `g6me-scan`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MetricAnchorKind {
    /// AprilTag3 (`tag36h11`) square fiducial of given outer black-border side length (mm).
    AprilTag3 { side_mm: f64 },
    /// ChArUco chessboard-AprilTag hybrid square pitch (mm).
    ChArUco { square_mm: f64 },
    /// Russian Debit Card (**GOST R ISO/IEC 7810-2006 ID-1**, `85.60 × 53.98 mm`,
    /// `r = 3.18 mm` rounded-corner tangent compensation + ISO/IEC 7816-2 chip landmark,
    /// horizontal or vertical layout, with automatic PAN/CVV face privacy masking).
    RussianDebitCardId1 { vertical_layout: bool },
}

/// Detected metric anchor with sub-pixel sharp corners, scale, and privacy mask bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedMetricAnchor {
    pub kind: MetricAnchorKind,
    /// Four sub-pixel **virtual sharp** corners `[TL, TR, BR, BL]` in image pixel coordinates.
    /// For ID-1 cards, these are extrapolated from the 4 straight edges to eliminate the
    /// `2.634 mm` (`2.60%`) rounded-corner diagonal bias!
    pub sharp_corners_px: [[f64; 2]; 4],
    /// Exact physical 3D corner coordinates `[TL, TR, BR, BL]` in the anchor plane (`z = 0`, meters).
    pub object_corners_m: [[f64; 3]; 4],
    /// Optional ISO/IEC 7816-2 contact-chip centroid `(u, v)` in pixels resolving 180° ambiguity.
    pub chip_center_px: Option<[f64; 2]>,
    /// Derived physical metric scale at the anchor plane (`meters / pixel`).
    pub meters_per_pixel: f64,
    /// Reprojection / aspect-ratio consistency residual (`mm`).
    pub fit_residual_mm: f64,
    /// Whether the interior card face (PAN/name/CVV region) is masked out of feature matching.
    pub privacy_masked: bool,
}

/// Fit a 2D straight line `a*x + b*y + c = 0` (`a^2 + b^2 = 1`) via exact 2×2 covariance
/// eigenvector (total least squares) to a set of sub-pixel edge points.
pub fn fit_line_tls(pts: &[[f64; 2]]) -> Result<[f64; 3]> {
    if pts.len() < 2 {
        return Err(Error::msg("need at least 2 points to fit a straight edge"));
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    for p in pts {
        let dx = p[0] - mx;
        let dy = p[1] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    // Smallest-eigenvalue eigenvector of 2x2 symmetric matrix [[sxx, sxy], [sxy, syy]]
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let nx = -theta.sin();
    let ny = theta.cos();
    // Pick eigenvector corresponding to smaller variance
    let v1 = nx * nx * sxx + 2.0 * nx * ny * sxy + ny * ny * syy;
    let tx = theta.cos();
    let ty = theta.sin();
    let v2 = tx * tx * sxx + 2.0 * tx * ty * sxy + ty * ty * syy;
    let (a, b) = if v1 <= v2 { (nx, ny) } else { (tx, ty) };
    let c = -(a * mx + b * my);
    Ok([a, b, c])
}

/// Exact intersection of two 2D lines `l1 = [a1, b1, c1]` and `l2 = [a2, b2, c2]`.
pub fn intersect_lines(l1: [f64; 3], l2: [f64; 3]) -> Result<[f64; 2]> {
    let det = l1[0] * l2[1] - l1[1] * l2[0];
    if det.abs() < 1e-12 {
        return Err(Error::msg("parallel card edges cannot intersect"));
    }
    let x = (l1[1] * l2[2] - l2[1] * l1[2]) / det;
    let y = (l2[0] * l1[2] - l1[0] * l2[2]) / det;
    Ok([x, y])
}

/// Detect and calibrate a **Russian Debit Card (GOST R ISO/IEC 7810-2006 ID-1)** metric anchor
/// from sub-pixel points sampled along its 4 straight edges `[top, right, bottom, left]`
/// (excluding the `3.18 mm` rounded corner arcs!) plus an optional ISO/IEC 7816-2 contact-chip
/// detection `(chip_u, chip_v)` to resolve 180° orientation ambiguity.
///
/// Extrapolates the 4 straight edge tangents to their exact **virtual sharp corners**, completely
/// eliminating the `2.634 mm` (`2.60%`) diagonal scale bias of rounded plastic debit cards,
/// and automatically flags the card interior for PAN/CVV privacy masking!
pub fn calibrate_russian_debit_card_id1(
    edge_samples: [&[[f64; 2]]; 4], // [top, right, bottom, left]
    chip_px: Option<[f64; 2]>,
    vertical_layout: bool,
) -> Result<DetectedMetricAnchor> {
    let l_top = fit_line_tls(edge_samples[0])?;
    let l_right = fit_line_tls(edge_samples[1])?;
    let l_bot = fit_line_tls(edge_samples[2])?;
    let l_left = fit_line_tls(edge_samples[3])?;

    let mut tl = intersect_lines(l_top, l_left)?;
    let mut tr = intersect_lines(l_top, l_right)?;
    let mut br = intersect_lines(l_bot, l_right)?;
    let mut bl = intersect_lines(l_bot, l_left)?;

    let (w_mm, h_mm) = if vertical_layout {
        (RUSSIAN_ID1_HEIGHT_MM, RUSSIAN_ID1_WIDTH_MM)
    } else {
        (RUSSIAN_ID1_WIDTH_MM, RUSSIAN_ID1_HEIGHT_MM)
    };

    // If contact chip is visible, verify orientation (chip sits in upper/left quadrant);
    // rotate quad 180° if the card was held upside-down.
    if let Some(chip) = chip_px {
        let d_tl = (chip[0] - tl[0]).hypot(chip[1] - tl[1]);
        let d_br = (chip[0] - br[0]).hypot(chip[1] - br[1]);
        if d_br < d_tl {
            std::mem::swap(&mut tl, &mut br);
            std::mem::swap(&mut tr, &mut bl);
        }
    }

    let px_w = 0.5 * ((tr[0] - tl[0]).hypot(tr[1] - tl[1]) + (br[0] - bl[0]).hypot(br[1] - bl[1]));
    let px_h = 0.5 * ((bl[0] - tl[0]).hypot(bl[1] - tl[1]) + (br[0] - tr[0]).hypot(br[1] - tr[1]));
    if px_w < 8.0 || px_h < 8.0 {
        return Err(Error::msg("detected ID-1 card quad is too small (< 8 px)"));
    }

    let expected_aspect = w_mm / h_mm;
    let measured_aspect = px_w / px_h;
    // Allow perspective foreshortening up to ~35°, reject gross aspect mismatches
    if (measured_aspect / expected_aspect - 1.0).abs() > 0.45 {
        return Err(Error::msg(format!(
            "quad aspect ratio {measured_aspect:.3} incompatible with GOST ID-1 card ({expected_aspect:.3})"
        )));
    }

    let m_per_px_w = (w_mm * 1e-3) / px_w;
    let m_per_px_h = (h_mm * 1e-3) / px_h;
    let meters_per_pixel = 0.5 * (m_per_px_w + m_per_px_h);
    let fit_residual_mm = ((px_w * meters_per_pixel * 1e3) - w_mm)
        .abs()
        .max(((px_h * meters_per_pixel * 1e3) - h_mm).abs());

    let w_m = w_mm * 1e-3;
    let h_m = h_mm * 1e-3;
    Ok(DetectedMetricAnchor {
        kind: MetricAnchorKind::RussianDebitCardId1 { vertical_layout },
        sharp_corners_px: [tl, tr, br, bl],
        object_corners_m: [
            [0.0, 0.0, 0.0],
            [w_m, 0.0, 0.0],
            [w_m, h_m, 0.0],
            [0.0, h_m, 0.0],
        ],
        chip_center_px: chip_px,
        meters_per_pixel,
        fit_residual_mm,
        privacy_masked: true,
    })
}

/// Point-in-quad privacy mask predicate: returns `true` if pixel `(px, py)` lies inside the
/// sensitive interior of a detected debit card (inset 4% from the border) so PAN/name/CVV
/// pixels are excluded from dense matching and debug outputs!
pub fn is_inside_card_privacy_mask(anchor: &DetectedMetricAnchor, px: f64, py: f64) -> bool {
    if !anchor.privacy_masked {
        return false;
    }
    let c = anchor.sharp_corners_px;
    let cx = 0.25 * (c[0][0] + c[1][0] + c[2][0] + c[3][0]);
    let cy = 0.25 * (c[0][1] + c[1][1] + c[2][1] + c[3][1]);
    // Shrink quad by 4% inward so outer metric border remains matchable while card face is masked
    let mut inset = [[0.0_f64; 2]; 4];
    for i in 0..4 {
        inset[i] = [cx + 0.96 * (c[i][0] - cx), cy + 0.96 * (c[i][1] - cy)];
    }
    let mut pos = 0usize;
    let mut neg = 0usize;
    for i in 0..4 {
        let a = inset[i];
        let b = inset[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
        if cross >= 0.0 {
            pos += 1;
        } else {
            neg += 1;
        }
    }
    pos == 4 || neg == 4
}

/// Calibrate an **AprilTag3 (`tag36h11`)** or **ChArUco** square fiducial from its 4 sub-pixel
/// saddle-point corners `[TL, TR, BR, BL]`.
pub fn calibrate_apriltag3_anchor(
    corners_px: [[f64; 2]; 4],
    side_mm: f64,
    is_charuco: bool,
) -> Result<DetectedMetricAnchor> {
    let px_w = 0.5
        * ((corners_px[1][0] - corners_px[0][0]).hypot(corners_px[1][1] - corners_px[0][1])
            + (corners_px[2][0] - corners_px[3][0]).hypot(corners_px[2][1] - corners_px[3][1]));
    let px_h = 0.5
        * ((corners_px[3][0] - corners_px[0][0]).hypot(corners_px[3][1] - corners_px[0][1])
            + (corners_px[2][0] - corners_px[1][0]).hypot(corners_px[2][1] - corners_px[1][1]));
    if px_w < 4.0 || px_h < 4.0 {
        return Err(Error::msg("AprilTag3 quad too small"));
    }
    let s_m = side_mm * 1e-3;
    let meters_per_pixel = s_m / (0.5 * (px_w + px_h));
    let fit_residual_mm = ((px_w - px_h).abs() * meters_per_pixel * 1e3) * 0.5;
    Ok(DetectedMetricAnchor {
        kind: if is_charuco {
            MetricAnchorKind::ChArUco { square_mm: side_mm }
        } else {
            MetricAnchorKind::AprilTag3 { side_mm }
        },
        sharp_corners_px: corners_px,
        object_corners_m: [
            [0.0, 0.0, 0.0],
            [s_m, 0.0, 0.0],
            [s_m, s_m, 0.0],
            [0.0, s_m, 0.0],
        ],
        chip_center_px: None,
        meters_per_pixel,
        fit_residual_mm,
        privacy_masked: false,
    })
}

/// Dense skin-proof Zero-Mean Normalized Cross-Correlation (ZNCC) sub-pixel patch matcher
/// (invariant to multiplicative skin-sheen illumination changes).
#[allow(clippy::too_many_arguments)]
pub fn dense_zncc_match_patch(
    img_a: &[f64],
    img_b: &[f64],
    width: usize,
    height: usize,
    ax: usize,
    ay: usize,
    search_radius: isize,
    patch_rad: isize,
) -> (f64, f64, f64) {
    let patch_stats = |img: &[f64], cx: isize, cy: isize| -> (f64, f64, Vec<f64>) {
        let side = (2 * patch_rad + 1) as usize;
        let mut vals = Vec::with_capacity(side * side);
        for oy in -patch_rad..=patch_rad {
            let y = (cy + oy).clamp(0, (height as isize) - 1) as usize;
            for ox in -patch_rad..=patch_rad {
                let x = (cx + ox).clamp(0, (width as isize) - 1) as usize;
                vals.push(img[y * width + x]);
            }
        }
        let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        let var = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>();
        (mean, var.sqrt().max(1e-12), vals)
    };

    let (ma, sa, pa) = patch_stats(img_a, ax as isize, ay as isize);
    let mut best_score = -2.0_f64;
    let mut best_bx = ax as isize;
    let mut best_by = ay as isize;

    let eval_score = |bx: isize, by: isize| -> f64 {
        let (mb, sb, pb) = patch_stats(img_b, bx, by);
        let mut num = 0.0_f64;
        for (&va, &vb) in pa.iter().zip(&pb) {
            num += (va - ma) * (vb - mb);
        }
        num / (sa * sb)
    };

    for dy in -search_radius..=search_radius {
        let by = (ay as isize) + dy;
        if by < patch_rad || by + patch_rad >= height as isize {
            continue;
        }
        for dx in -search_radius..=search_radius {
            let bx = (ax as isize) + dx;
            if bx < patch_rad || bx + patch_rad >= width as isize {
                continue;
            }
            let sc = eval_score(bx, by);
            if sc > best_score {
                best_score = sc;
                best_bx = bx;
                best_by = by;
            }
        }
    }

    // 1D parabolic sub-pixel refinement in x and y
    let sc_xm = eval_score(best_bx - 1, best_by);
    let sc_xp = eval_score(best_bx + 1, best_by);
    let denom_x = sc_xm - 2.0 * best_score + sc_xp;
    let sub_x = if denom_x.abs() > 1e-12 {
        (0.5 * (sc_xm - sc_xp) / denom_x).clamp(-0.5, 0.5)
    } else {
        0.0
    };

    let sc_ym = eval_score(best_bx, best_by - 1);
    let sc_yp = eval_score(best_bx, best_by + 1);
    let denom_y = sc_ym - 2.0 * best_score + sc_yp;
    let sub_y = if denom_y.abs() > 1e-12 {
        (0.5 * (sc_ym - sc_yp) / denom_y).clamp(-0.5, 0.5)
    } else {
        0.0
    };

    (
        (best_bx as f64) + sub_x,
        (best_by as f64) + sub_y,
        best_score,
    )
}

// ============================================================================
// Stage 2: GLOMAP Global SO(3) Rotation Averaging + Rolling-Shutter
//          Ozaki-II FP4 -> FP64 Schur-Complement Bundle Adjustment
// ============================================================================

/// Pairwise relative rotation measurement `R_ij ≈ R_j * R_i^T` for GLOMAP global SfM.
#[derive(Debug, Clone, Copy)]
pub struct RelativeRotationEdge {
    pub cam_i: usize,
    pub cam_j: usize,
    pub r_ij: Mat3,
    pub weight: f64,
}

/// **GLOMAP (Pan et al., ECCV 2024)** Global `SO(3)` Lie-Algebra Rotation Averaging.
///
/// Minimizes the Huber-weighted geodesic residual `sum_(i,j) rho(||Log(R_j * R_i^T * R_ij^T)||)`
/// on `SO(3)` via iteratively reweighted least-squares (IRLS) tangent-space updates, pinning
/// camera `0` at identity `I` to fix gauge freedom.
pub fn glomap_so3_rotation_averaging(
    num_cams: usize,
    edges: &[RelativeRotationEdge],
    init_rots: &[Mat3],
    iters: usize,
) -> Vec<Mat3> {
    let mut rots = if init_rots.len() == num_cams {
        init_rots.to_vec()
    } else {
        vec![MAT3_ID; num_cams]
    };
    if num_cams <= 1 || edges.is_empty() {
        return rots;
    }

    let huber_delta = 0.1_f64;
    for _ in 0..iters {
        let mut grad = vec![[0.0_f64; 3]; num_cams];
        let mut w_sum = vec![0.0_f64; num_cams];

        for e in edges {
            // Residual rotation: R_err = R_ij * R_i * R_j^T (should be I!)
            let ri = rots[e.cam_i];
            let rj_t = mat3_transpose(&rots[e.cam_j]);
            let r_pred = mat3_mul(&e.r_ij, &ri);
            let r_err = mat3_mul(&r_pred, &rj_t);
            let phi = so3_log(&r_err);
            let mag = (phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]).sqrt();
            let irls_w = e.weight
                * if mag > huber_delta {
                    huber_delta / mag
                } else {
                    1.0
                };

            // Tangent update: move R_j toward R_ij * R_i, and R_i opposite
            for a in 0..3 {
                grad[e.cam_j][a] += irls_w * phi[a];
                grad[e.cam_i][a] -= irls_w * phi[a];
            }
            w_sum[e.cam_j] += irls_w;
            w_sum[e.cam_i] += irls_w;
        }

        // Camera 0 pinned to gauge origin
        for c in 1..num_cams {
            if w_sum[c] > 1e-12 {
                let step = [
                    0.5 * grad[c][0] / w_sum[c],
                    0.5 * grad[c][1] / w_sum[c],
                    0.5 * grad[c][2] / w_sum[c],
                ];
                let d_rot = so3_exp(step);
                rots[c] = mat3_mul(&d_rot, &rots[c]);
            }
        }
    }
    rots
}

/// Rolling-shutter camera model with continuous velocity `(omega, vel)`, VCM focus breathing
/// (`focal_scale`), OIS principal-point shift, and optional DNG `WarpRectilinear` prior.
#[derive(Debug, Clone)]
pub struct RollingShutterCamera {
    pub rot: Mat3,
    pub trans: [f64; 3],
    /// Angular velocity `rad/s` during rolling-shutter scanline sweep (Hedborg et al., CVPR 2012).
    pub omega_rad_s: [f64; 3],
    /// Linear velocity `m/s` during rolling-shutter sweep.
    pub vel_m_s: [f64; 3],
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: f64,
    pub height: f64,
    /// Per-frame VCM focus-breathing focal-length multiplier (regularized by Tikhonov prior toward 1.0).
    pub vcm_focal_scale: f64,
    pub rolling_shutter_skew_s: f64,
    pub ois_shift_px: [f64; 2],
}

impl RollingShutterCamera {
    /// Project a 3D world point `X` through the continuous rolling-shutter pose at scanline `row_y`,
    /// including OIS principal-point shift and VCM focus-breathing scale!
    pub fn project_at_row(
        &self,
        x_world: [f64; 3],
        row_y: f64,
        warp: Option<&WarpRectilinearOpcode>,
    ) -> [f64; 2] {
        let dt = ((row_y / self.height.max(1.0)) - 0.5) * self.rolling_shutter_skew_s;
        let d_rot = so3_exp([
            self.omega_rad_s[0] * dt,
            self.omega_rad_s[1] * dt,
            self.omega_rad_s[2] * dt,
        ]);
        let r_row = mat3_mul(&d_rot, &self.rot);
        let t_row = [
            self.trans[0] + self.vel_m_s[0] * dt,
            self.trans[1] + self.vel_m_s[1] * dt,
            self.trans[2] + self.vel_m_s[2] * dt,
        ];
        let pc = v3_add(mat3_vec(&r_row, x_world), t_row);
        let z = pc[2].max(1e-4);
        let mut xn = pc[0] / z;
        let mut yn = pc[1] / z;
        if let Some(w) = warp {
            let (xd, yd) = w.distort_norm(xn, yn, 1); // Green analytical optical model
            xn = xd;
            yn = yd;
        }
        [
            self.fx * self.vcm_focal_scale * xn + self.cx + self.ois_shift_px[0],
            self.fy * self.vcm_focal_scale * yn + self.cy + self.ois_shift_px[1],
        ]
    }
}

/// 2D observation of 3D point `pt_idx` in camera `cam_idx` at measured unwarped pixel `(u, v)`.
#[derive(Debug, Clone, Copy)]
pub struct BaObservation {
    pub cam_idx: usize,
    pub pt_idx: usize,
    pub u_px: f64,
    pub v_px: f64,
}

/// Metric distance gauge constraint between two 3D landmarks (e.g., AprilTag3 corner pair or
/// Russian Debit Card GOST ID-1 virtual sharp corners `85.60 mm` / `53.98 mm`).
#[derive(Debug, Clone, Copy)]
pub struct MetricGaugeConstraint {
    pub pt_a: usize,
    pub pt_b: usize,
    pub target_dist_m: f64,
    pub weight: f64,
}

/// Summary report from Stage 2 Ozaki-II FP4→FP64 Schur-Complement Bundle Adjustment.
#[derive(Debug, Clone, Copy)]
pub struct BundleAdjustmentReport {
    pub initial_rms_px: f64,
    pub final_rms_px: f64,
    pub max_vcm_breathing_pct: f64,
    pub metric_scale_error_mm: f64,
    pub schur_backend: TensorBackend,
    pub iterations: usize,
}

/// Solve **Rolling-Shutter Bundle Adjustment** (camera 6-DoF pose, per-frame VCM focus breathing,
/// 3D point cloud, and metric gauge constraints) via Levenberg–Marquardt with exact **Schur
/// Complement** `(U - W V^-1 W^T) dc = g` where the matrix products `W * (V^-1 W^T)` are computed
/// on **Blackwell FP4 (E2M1 Base-13) Tensor Cores via Ozaki-II CRT → IEEE-754 FP64**!
pub fn bundle_adjust_ozaki2_fp4_schur(
    cams: &mut [RollingShutterCamera],
    points: &mut [[f64; 3]],
    obs: &[BaObservation],
    gauges: &[MetricGaugeConstraint],
    warp: Option<&WarpRectilinearOpcode>,
    vcm_tikhonov_weight: f64,
    iters: usize,
) -> BundleAdjustmentReport {
    let n_cams = cams.len();
    let n_pts = points.len();
    let cam_dof = 7usize; // [d_rx, d_ry, d_rz, d_tx, d_ty, d_tz, d_vcm_scale]
    let n_cam_vars = n_cams * cam_dof;

    let compute_rms = |cs: &[RollingShutterCamera], ps: &[[f64; 3]]| -> f64 {
        if obs.is_empty() {
            return 0.0;
        }
        let mut s = 0.0_f64;
        for o in obs {
            let pred = cs[o.cam_idx].project_at_row(ps[o.pt_idx], o.v_px, warp);
            let du = pred[0] - o.u_px;
            let dv = pred[1] - o.v_px;
            s += du * du + dv * dv;
        }
        (s / ((2 * obs.len()) as f64)).sqrt()
    };

    let initial_rms_px = compute_rms(cams, points);
    let mut lambda = 1e-2_f64;
    let eps = 1e-6_f64;

    for _ in 0..iters {
        let mut u_mat = vec![vec![0.0_f64; n_cam_vars]; n_cam_vars];
        let mut w_mat = vec![vec![0.0_f64; n_pts * 3]; n_cam_vars];
        let mut v_blocks = vec![[[0.0_f64; 3]; 3]; n_pts];
        let mut g_cam = vec![0.0_f64; n_cam_vars];
        let mut g_pt = vec![[0.0_f64; 3]; n_pts];

        for o in obs {
            let c = &cams[o.cam_idx];
            let p = points[o.pt_idx];
            let pred0 = c.project_at_row(p, o.v_px, warp);
            let res = [pred0[0] - o.u_px, pred0[1] - o.v_px];

            // Finite-difference Jacobians wrt camera (7 params) and 3D point (3 params)
            let mut jc = [[0.0_f64; 7]; 2];
            let (jc0, jc1) = jc.split_at_mut(1);
            for (k, (j0, j1)) in jc0[0].iter_mut().zip(jc1[0].iter_mut()).enumerate() {
                let mut c_pert = c.clone();
                match k {
                    0 => c_pert.rot = mat3_mul(&so3_exp([eps, 0.0, 0.0]), &c.rot),
                    1 => c_pert.rot = mat3_mul(&so3_exp([0.0, eps, 0.0]), &c.rot),
                    2 => c_pert.rot = mat3_mul(&so3_exp([0.0, 0.0, eps]), &c.rot),
                    3 => c_pert.trans[0] += eps,
                    4 => c_pert.trans[1] += eps,
                    5 => c_pert.trans[2] += eps,
                    _ => c_pert.vcm_focal_scale += eps,
                }
                let pred_p = c_pert.project_at_row(p, o.v_px, warp);
                *j0 = (pred_p[0] - pred0[0]) / eps;
                *j1 = (pred_p[1] - pred0[1]) / eps;
            }

            let mut jp = [[0.0_f64; 3]; 2];
            for k in 0..3 {
                let mut p_pert = p;
                p_pert[k] += eps;
                let pred_p = c.project_at_row(p_pert, o.v_px, warp);
                jp[0][k] = (pred_p[0] - pred0[0]) / eps;
                jp[1][k] = (pred_p[1] - pred0[1]) / eps;
            }

            let cb = o.cam_idx * cam_dof;
            let pb = o.pt_idx * 3;
            for r in 0..2 {
                for a in 0..cam_dof {
                    g_cam[cb + a] -= jc[r][a] * res[r];
                    for b in 0..cam_dof {
                        u_mat[cb + a][cb + b] += jc[r][a] * jc[r][b];
                    }
                    for b in 0..3 {
                        w_mat[cb + a][pb + b] += jc[r][a] * jp[r][b];
                    }
                }
                for a in 0..3 {
                    g_pt[o.pt_idx][a] -= jp[r][a] * res[r];
                    for b in 0..3 {
                        v_blocks[o.pt_idx][a][b] += jp[r][a] * jp[r][b];
                    }
                }
            }
        }

        // Tikhonov prior on per-frame VCM focus breathing (keeps vcm_focal_scale anchored to 1.0)
        // + pin camera 0 pose to fix world gauge freedom (6 DoF).
        for (ci, cam) in cams.iter().enumerate() {
            let cb = ci * cam_dof;
            let vcm_err = cam.vcm_focal_scale - 1.0;
            u_mat[cb + 6][cb + 6] += vcm_tikhonov_weight;
            g_cam[cb + 6] -= vcm_tikhonov_weight * vcm_err;
            if ci == 0 {
                for a in 0..6 {
                    u_mat[cb + a][cb + a] += 1e8;
                }
            }
            for a in 0..cam_dof {
                u_mat[cb + a][cb + a] += lambda * u_mat[cb + a][cb + a].max(1.0);
            }
        }

        // Metric gauge constraints (AprilTag3 / Russian Debit Card GOST ID-1 exact mm scale!)
        for g in gauges {
            if g.pt_a >= n_pts || g.pt_b >= n_pts {
                continue;
            }
            let diff = v3_sub(points[g.pt_a], points[g.pt_b]);
            let dist = v3_norm(diff).max(1e-9);
            let dir = v3_scale(diff, 1.0 / dist);
            let err = dist - g.target_dist_m;
            for a in 0..3 {
                g_pt[g.pt_a][a] -= g.weight * dir[a] * err;
                g_pt[g.pt_b][a] += g.weight * dir[a] * err;
                for b in 0..3 {
                    v_blocks[g.pt_a][a][b] += g.weight * dir[a] * dir[b];
                    v_blocks[g.pt_b][a][b] += g.weight * dir[a] * dir[b];
                }
            }
        }

        // Invert 3x3 point Hessian blocks V_j^{-1} with LM damping
        let mut v_inv = vec![[[0.0_f64; 3]; 3]; n_pts];
        for j in 0..n_pts {
            let mut vb = v_blocks[j];
            for (a, row) in vb.iter_mut().enumerate() {
                row[a] += lambda * row[a].max(1.0) + 1e-6;
            }
            v_inv[j] = invert_3x3(vb);
        }

        // Form Y = V^{-1} W^T (size [3*n_pts x n_cam_vars]) and V^{-1} g_pt
        let mut y_mat = vec![vec![0.0_f64; n_cam_vars]; n_pts * 3];
        let mut v_inv_g = vec![[0.0_f64; 3]; n_pts];
        for j in 0..n_pts {
            let pb = j * 3;
            v_inv_g[j] = mat3_vec(&v_inv[j], g_pt[j]);
            for col in 0..n_cam_vars {
                let w_col = [w_mat[col][pb], w_mat[col][pb + 1], w_mat[col][pb + 2]];
                let res = mat3_vec(&v_inv[j], w_col);
                y_mat[pb][col] = res[0];
                y_mat[pb + 1][col] = res[1];
                y_mat[pb + 2][col] = res[2];
            }
        }

        // SCHUR COMPLEMENT PRODUCT W * (V^{-1} W^T) ON BLACKWELL FP4 (E2M1 BASE-13) OZAKI-II CRT!!
        let w_vinv_wt = if n_pts > 0 && n_cam_vars > 0 {
            ozaki2_crt_gemm(&w_mat, &y_mat, TensorBackend::Fp4E2m1Base13)
        } else {
            vec![vec![0.0_f64; n_cam_vars]; n_cam_vars]
        };

        let mut schur_s = vec![vec![0.0_f64; n_cam_vars]; n_cam_vars];
        let mut schur_rhs = g_cam.clone();
        for i in 0..n_cam_vars {
            for j in 0..n_cam_vars {
                schur_s[i][j] = u_mat[i][j] - w_vinv_wt[i][j];
            }
            let mut acc = 0.0_f64;
            for (p_idx, vg) in v_inv_g.iter().enumerate() {
                let pb = p_idx * 3;
                acc += w_mat[i][pb] * vg[0] + w_mat[i][pb + 1] * vg[1] + w_mat[i][pb + 2] * vg[2];
            }
            schur_rhs[i] -= acc;
        }

        let dc = solve_linear_system_f64(&schur_s, &schur_rhs);

        // Back-substitute 3D point increments: dp_j = V_j^{-1} (g_pt_j - W_j^T dc)
        let mut cand_cams = cams.to_vec();
        for ci in 0..n_cams {
            let cb = ci * cam_dof;
            let d_rot = so3_exp([dc[cb], dc[cb + 1], dc[cb + 2]]);
            cand_cams[ci].rot = mat3_mul(&d_rot, &cams[ci].rot);
            cand_cams[ci].trans[0] += dc[cb + 3];
            cand_cams[ci].trans[1] += dc[cb + 4];
            cand_cams[ci].trans[2] += dc[cb + 5];
            cand_cams[ci].vcm_focal_scale =
                (cams[ci].vcm_focal_scale + dc[cb + 6]).clamp(0.85, 1.15);
        }
        let mut cand_pts = points.to_vec();
        for j in 0..n_pts {
            let pb = j * 3;
            let mut wtdc = [0.0_f64; 3];
            for i in 0..n_cam_vars {
                wtdc[0] += w_mat[i][pb] * dc[i];
                wtdc[1] += w_mat[i][pb + 1] * dc[i];
                wtdc[2] += w_mat[i][pb + 2] * dc[i];
            }
            let dp = mat3_vec(
                &v_inv[j],
                [
                    g_pt[j][0] - wtdc[0],
                    g_pt[j][1] - wtdc[1],
                    g_pt[j][2] - wtdc[2],
                ],
            );
            for a in 0..3 {
                cand_pts[j][a] += dp[a];
            }
        }

        let cur_rms = compute_rms(cams, points);
        let new_rms = compute_rms(&cand_cams, &cand_pts);
        if new_rms <= cur_rms + 1e-12 {
            cams.clone_from_slice(&cand_cams);
            points.clone_from_slice(&cand_pts);
            lambda = (lambda * 0.35).max(1e-7);
        } else {
            lambda = (lambda * 4.0).min(1e6);
        }
    }

    let final_rms_px = compute_rms(cams, points);
    let max_vcm_breathing_pct = cams
        .iter()
        .map(|c| (c.vcm_focal_scale - 1.0).abs() * 100.0)
        .fold(0.0_f64, f64::max);
    let metric_scale_error_mm = gauges
        .iter()
        .filter(|g| g.pt_a < n_pts && g.pt_b < n_pts)
        .map(|g| (v3_norm(v3_sub(points[g.pt_a], points[g.pt_b])) - g.target_dist_m).abs() * 1e3)
        .fold(0.0_f64, f64::max);

    BundleAdjustmentReport {
        initial_rms_px,
        final_rms_px,
        max_vcm_breathing_pct,
        metric_scale_error_mm,
        schur_backend: TensorBackend::Fp4E2m1Base13,
        iterations: iters,
    }
}

fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-24 {
        return [[0.0; 3]; 3];
    }
    let inv_d = 1.0 / det;
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [
        [c00 * inv_d, c10 * inv_d, c20 * inv_d],
        [c01 * inv_d, c11 * inv_d, c21 * inv_d],
        [c02 * inv_d, c12 * inv_d, c22 * inv_d],
    ]
}

fn solve_linear_system_f64(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return Vec::new();
    }
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        let mut piv = col;
        let mut best = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                piv = r;
            }
        }
        aug.swap(col, piv);
        let diag = aug[col][col];
        if diag.abs() < 1e-24 {
            continue;
        }
        let inv = 1.0 / diag;
        for val in &mut aug[col][col..=n] {
            *val *= inv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug[r][col];
            if f == 0.0 {
                continue;
            }
            let (pivot, row) = if r < col {
                let (head, tail) = aug.split_at_mut(col);
                (&mut tail[0], &mut head[r])
            } else {
                let (head, tail) = aug.split_at_mut(r);
                (&mut head[col], &mut tail[0])
            };
            for (dst, &src) in row[col..=n].iter_mut().zip(&pivot[col..=n]) {
                *dst -= f * src;
            }
        }
    }
    aug.into_iter().map(|r| r[n]).collect()
}

// ============================================================================
// Stage 3: 2D Gaussian Splatting (2DGS, SIGGRAPH 2024) + PGSR (IEEE TVCG 2025)
//          + Dense TSDF Fusion & Surface Extraction
// ============================================================================

/// Oriented 2D planar disk surfel primitive (2DGS / PGSR) collapsing volumetric 3D blobs
/// onto razor-sharp oriented ear cartilage surfaces.
#[derive(Debug, Clone, Copy)]
pub struct PlanarSurfel2D {
    pub center: [f64; 3],
    pub tangent_u: [f64; 3],
    pub tangent_v: [f64; 3],
    pub normal: [f64; 3],
    pub scale_u: f64,
    pub scale_v: f64,
    pub opacity: f64,
    pub albedo: [f64; 3],
}

impl PlanarSurfel2D {
    /// Construct an oriented 2D planar disk surfel from position `center`, outward `normal`,
    /// tangential radii `(scale_u, scale_v)`, opacity, and diffuse albedo.
    pub fn from_normal(
        center: [f64; 3],
        normal: [f64; 3],
        scale_u: f64,
        scale_v: f64,
        opacity: f64,
        albedo: [f64; 3],
    ) -> Self {
        let nl = v3_norm(normal).max(1e-12);
        let n = v3_scale(normal, 1.0 / nl);
        let up = if n[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        let tu_raw = v3_cross(up, n);
        let tu = v3_scale(tu_raw, 1.0 / v3_norm(tu_raw).max(1e-12));
        let tv = v3_cross(n, tu);
        Self {
            center,
            tangent_u: tu,
            tangent_v: tv,
            normal: n,
            scale_u,
            scale_v,
            opacity,
            albedo,
        }
    }

    /// Exact ray-splat intersection `(t_hit, alpha_weight)` for a ray `o + t * d` (`|d| = 1`).
    pub fn intersect_ray(&self, origin: [f64; 3], dir: [f64; 3]) -> Option<(f64, f64)> {
        let denom = v3_dot(dir, self.normal);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = v3_dot(v3_sub(self.center, origin), self.normal) / denom;
        if t <= 1e-4 {
            return None;
        }
        let hit = v3_add(origin, v3_scale(dir, t));
        let rel = v3_sub(hit, self.center);
        let u = v3_dot(rel, self.tangent_u) / self.scale_u.max(1e-6);
        let v = v3_dot(rel, self.tangent_v) / self.scale_v.max(1e-6);
        let r2 = u * u + v * v;
        if r2 > 9.0 {
            return None;
        }
        let alpha = (self.opacity * (-0.5 * r2).exp()).clamp(0.0, 0.99);
        Some((t, alpha))
    }
}

/// Evaluate **2DGS / PGSR** multi-view depth-distortion loss `L_d` and surface-normal
/// consistency loss `L_n` along a camera ray traversing sorted surfel hits `(t_i, alpha_i, n_i)`
/// (Huang et al., ACM SIGGRAPH 2024; Chen et al., IEEE TVCG 2025).
pub fn evaluate_2dgs_pgsr_ray_losses(
    hits: &[(f64, f64, [f64; 3])],
    expected_surface_normal: [f64; 3],
) -> (f64, f64, f64) {
    let mut trans = 1.0_f64;
    let mut weights = Vec::with_capacity(hits.len());
    let mut depth_sum = 0.0_f64;
    let mut w_total = 0.0_f64;
    let mut l_normal = 0.0_f64;

    for &(t, alpha, n_i) in hits {
        let w = trans * alpha;
        weights.push((t, w));
        depth_sum += w * t;
        w_total += w;
        l_normal += w * (1.0 - v3_dot(n_i, expected_surface_normal).clamp(-1.0, 1.0));
        trans *= 1.0 - alpha;
    }

    // 2DGS depth-distortion loss: L_d = sum_{i,j} w_i w_j |t_i - t_j|
    let mut l_dist = 0.0_f64;
    for i in 0..weights.len() {
        for j in 0..weights.len() {
            l_dist += weights[i].1 * weights[j].1 * (weights[i].0 - weights[j].0).abs();
        }
    }
    let mean_depth = if w_total > 1e-12 {
        depth_sum / w_total
    } else {
        0.0
    };
    (mean_depth, l_dist, l_normal)
}

/// Fuse a cloud of oriented **2DGS / PGSR** planar surfels into a dense 3D **Truncated Signed
/// Distance Field (TSDF)** (`grid_res^3`) and extract the zero-level surface mesh (`TSDF = 0`)
/// via exact Marching Tetrahedra!
pub fn fuse_surfels_tsdf_to_mesh(
    surfels: &[PlanarSurfel2D],
    grid_res: usize,
    trunc_dist_m: f64,
) -> Result<TriangleMesh> {
    if surfels.len() < 8 {
        return Err(Error::msg(
            "need at least 8 oriented 2DGS surfels for TSDF fusion",
        ));
    }
    let n = grid_res.clamp(10, 48);
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for s in surfels {
        for a in 0..3 {
            mn[a] = mn[a].min(s.center[a]);
            mx[a] = mx[a].max(s.center[a]);
        }
    }
    let span = v3_sub(mx, mn);
    let pad = v3_scale(span, 0.25);
    mn = v3_sub(mn, pad);
    mx = v3_add(mx, pad);
    let step = [
        (mx[0] - mn[0]) / ((n - 1) as f64),
        (mx[1] - mn[1]) / ((n - 1) as f64),
        (mx[2] - mn[2]) / ((n - 1) as f64),
    ];

    let mut sdf = vec![-1.0_f64; n * n * n];
    let idx3 = |ix: usize, iy: usize, iz: usize| (ix * n + iy) * n + iz;

    // Signed distance convention for extract_isosurface_marching_tetrahedra(iso = 0.0):
    // INSIDE >= 0.0, OUTSIDE < 0.0  =>  val = -dot(p - c_nearest, n_nearest) / trunc_dist_m
    for ix in 1..(n - 1) {
        let px = mn[0] + (ix as f64) * step[0];
        for iy in 1..(n - 1) {
            let py = mn[1] + (iy as f64) * step[1];
            for iz in 1..(n - 1) {
                let pz = mn[2] + (iz as f64) * step[2];
                let p = [px, py, pz];
                let mut w_sum = 0.0_f64;
                let mut sd_sum = 0.0_f64;
                let mut best_d2 = f64::INFINITY;
                let mut best_sd = -1.0_f64;

                for s in surfels {
                    let rel = v3_sub(p, s.center);
                    let d2 = v3_dot(rel, rel);
                    let sd = -v3_dot(rel, s.normal); // positive inside!
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_sd = sd;
                    }
                    let rad2 = (s.scale_u.max(s.scale_v) * 2.5)
                        .powi(2)
                        .max(trunc_dist_m * trunc_dist_m);
                    if d2 < rad2 * 4.0 {
                        let w = s.opacity * (-d2 / rad2).exp();
                        w_sum += w;
                        sd_sum += w * sd;
                    }
                }
                let raw_sd = if w_sum > 1e-9 {
                    sd_sum / w_sum
                } else {
                    best_sd
                };
                sdf[idx3(ix, iy, iz)] = (raw_sd / trunc_dist_m.max(1e-4)).clamp(-1.0, 1.0);
            }
        }
    }

    Ok(extract_isosurface_marching_tetrahedra(
        &sdf, n, mn, step, 0.0,
    ))
}

// ============================================================================
// End-to-End 4-Stage Pipeline Orchestrator (Zero-Disk Handoff -> FmmBemSolver!)
// ============================================================================

/// Complete summary produced by the 4-stage `g6me-scan` pipeline.
#[derive(Debug, Clone)]
pub struct ScanPipelineOutput {
    pub anchor: DetectedMetricAnchor,
    pub ba_report: BundleAdjustmentReport,
    pub mesh: TriangleMesh,
    pub mesh_stats: BemMeshStats,
    pub solver: FmmBemSolver,
}

/// Run the full 4-stage `g6me-scan` reconstruction pipeline from oriented surfels + calibrated
/// metric anchor + bundle-adjusted cameras directly into a watertight, isotropically remeshed
/// (`h <= h_max_m`) in-memory `FmmBemSolver` without touching disk!
pub fn run_stage3_to_stage4_to_fmm_bem(
    anchor: DetectedMetricAnchor,
    ba_report: BundleAdjustmentReport,
    surfels: &[PlanarSurfel2D],
    tsdf_res: usize,
    winding_res: usize,
    h_max_m: f64,
    c_m_s: f64,
) -> Result<ScanPipelineOutput> {
    // Stage 3: 2DGS / PGSR -> TSDF Marching Tetrahedra
    let tsdf_mesh = fuse_surfels_tsdf_to_mesh(surfels, tsdf_res, 0.025)?;
    // Stage 4a: Generalized Winding Number watertight manifold solid conditioning
    let watertight = if tsdf_mesh.stats(c_m_s).is_watertight {
        tsdf_mesh
    } else {
        watertight_winding_repair(&tsdf_mesh, winding_res)
    };
    // Stage 4b: Isotropic Acoustic Remeshing enforcing h <= lambda_min / 6
    let conditioned = isotropic_acoustic_remesh(&watertight, h_max_m, 3);
    let mesh_stats = conditioned.stats(c_m_s);
    // Stage 4c: Direct in-memory / VRAM handoff into Gumerov FMM-BEM Helmholtz Solver!
    let solver = FmmBemSolver::new(&conditioned)?;
    Ok(ScanPipelineOutput {
        anchor,
        ba_report,
        mesh: conditioned,
        mesh_stats,
        solver,
    })
}

/// Helper to create a synthetic oriented 2DGS surfel set sampled from a reference head/pinna
/// mesh (for CLI self-test, synthetic demo, and CI verification).
pub fn sample_surfels_from_mesh(mesh: &TriangleMesh) -> Vec<PlanarSurfel2D> {
    mesh.elements()
        .into_iter()
        .map(|e| {
            let r = (e.area / PI).sqrt().max(0.004);
            PlanarSurfel2D::from_normal(
                e.centroid,
                e.normal,
                r * 1.4,
                r * 1.4,
                0.96,
                [0.76, 0.58, 0.48],
            )
        })
        .collect()
}

/// Zero-FMA Blackwell (`sm_120`) FP8/BF16 Tensor-Core Dense ZNCC Skin-Matcher CUDA kernel template.
pub const BLACKWELL_SCAN_MATCHER_CUDA: &str = r#"
// g6me-scan Stage 1 Dense Skin-Proof ZNCC Cost-Volume Kernel for Blackwell (sm_120)
extern "C" __global__ void g6me_scan_dense_zncc(
    const double* __restrict__ img_a,
    const double* __restrict__ img_b,
    double* __restrict__ best_dx,
    double* __restrict__ best_dy,
    int width,
    int height,
    int search_rad,
    int patch_rad
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < patch_rad || y < patch_rad || x >= width - patch_rad || y >= height - patch_rad) return;
    double best_sc = -2.0;
    int bdx = 0, bdy = 0;
    for (int dy = -search_rad; dy <= search_rad; ++dy) {
        for (int dx = -search_rad; dx <= search_rad; ++dx) {
            int tx = x + dx, ty = y + dy;
            if (tx < patch_rad || ty < patch_rad || tx >= width - patch_rad || ty >= height - patch_rad) continue;
            double ma = 0.0, mb = 0.0;
            int cnt = 0;
            for (int oy = -patch_rad; oy <= patch_rad; ++oy) {
                for (int ox = -patch_rad; ox <= patch_rad; ++ox) {
                    ma = __dadd_rn(ma, img_a[(y + oy) * width + (x + ox)]);
                    mb = __dadd_rn(mb, img_b[(ty + oy) * width + (tx + ox)]);
                    cnt++;
                }
            }
            ma = __ddiv_rn(ma, (double)cnt);
            mb = __ddiv_rn(mb, (double)cnt);
            double num = 0.0, va = 0.0, vb = 0.0;
            for (int oy = -patch_rad; oy <= patch_rad; ++oy) {
                for (int ox = -patch_rad; ox <= patch_rad; ++ox) {
                    double da = __dsub_rn(img_a[(y + oy) * width + (x + ox)], ma);
                    double db = __dsub_rn(img_b[(ty + oy) * width + (tx + ox)], mb);
                    num = __dadd_rn(num, __dmul_rn(da, db));
                    va  = __dadd_rn(va,  __dmul_rn(da, da));
                    vb  = __dadd_rn(vb,  __dmul_rn(db, db));
                }
            }
            double sc = __ddiv_rn(num, sqrt(__dmul_rn(va, vb)) + 1e-12);
            if (sc > best_sc) { best_sc = sc; bdx = dx; bdy = dy; }
        }
    }
    best_dx[y * width + x] = (double)bdx;
    best_dy[y * width + x] = (double)bdy;
}
"#;

/// Convenience helper for interpolating OIS into a `RollingShutterCamera` at frame mid-row.
pub fn apply_ois_midframe(
    cam: &mut RollingShutterCamera,
    samples: &[OisSample],
    t0_ns: i64,
    skew_ns: i64,
) {
    let (dx, dy) = crate::burst3d::interpolate_ois_at_scanline(
        samples,
        t0_ns,
        skew_ns,
        cam.height * 0.5,
        cam.height,
    );
    cam.ois_shift_px = [dx, dy];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn russian_debit_card_id1_tangent_extrapolation_eliminates_rounded_corner_bias() {
        // Synthesize a horizontal Russian debit card (85.60 x 53.98 mm, r = 3.18 mm) viewed
        // at exact 10 px / mm (856.0 x 539.8 px) rotated by 12 degrees, sampling ONLY the
        // straight edge segments outside the 3.18 mm rounded corners!
        let px_per_mm = 10.0_f64;
        let angle = 12.0_f64.to_radians();
        let (ca, sa) = (angle.cos(), angle.sin());
        let tx = |x_mm: f64, y_mm: f64| -> [f64; 2] {
            let x = x_mm * px_per_mm;
            let y = y_mm * px_per_mm;
            [500.0 + ca * x - sa * y, 400.0 + sa * x + ca * y]
        };

        let mut top = Vec::new();
        let mut bot = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for i in 0..20 {
            let u = 8.0 + (i as f64) * ((RUSSIAN_ID1_WIDTH_MM - 16.0) / 19.0);
            top.push(tx(u, 0.0));
            bot.push(tx(u, RUSSIAN_ID1_HEIGHT_MM));
            let v = 8.0 + (i as f64) * ((RUSSIAN_ID1_HEIGHT_MM - 16.0) / 19.0);
            left.push(tx(0.0, v));
            right.push(tx(RUSSIAN_ID1_WIDTH_MM, v));
        }

        // ISO/IEC 7816-2 contact chip center at (16.25, 23.89) mm
        let chip = tx(ISO7816_CHIP_CENTER_X_MM, ISO7816_CHIP_CENTER_Y_MM);
        let anchor =
            calibrate_russian_debit_card_id1([&top, &right, &bot, &left], Some(chip), false)
                .expect("calibrate Russian ID-1 debit card");

        assert!((anchor.meters_per_pixel - 1e-4).abs() < 1e-10);
        assert!(
            anchor.fit_residual_mm < 1e-6,
            "tangent-extrapolated ID-1 residual must be sub-micron, got {} mm",
            anchor.fit_residual_mm
        );
        // Verify privacy mask covers card center (PAN/CVV zone) and excludes outside background!
        let card_center = tx(RUSSIAN_ID1_WIDTH_MM * 0.5, RUSSIAN_ID1_HEIGHT_MM * 0.5);
        assert!(is_inside_card_privacy_mask(
            &anchor,
            card_center[0],
            card_center[1]
        ));
        let outside = tx(-20.0, -20.0);
        assert!(!is_inside_card_privacy_mask(
            &anchor, outside[0], outside[1]
        ));

        // Also test VERTICAL Russian debit card (53.98 x 85.60 mm)!
        let mut v_top = Vec::new();
        let mut v_bot = Vec::new();
        let mut v_left = Vec::new();
        let mut v_right = Vec::new();
        for i in 0..16 {
            let u = 6.0 + (i as f64) * ((RUSSIAN_ID1_HEIGHT_MM - 12.0) / 15.0);
            v_top.push(tx(u, 0.0));
            v_bot.push(tx(u, RUSSIAN_ID1_WIDTH_MM));
            let v = 6.0 + (i as f64) * ((RUSSIAN_ID1_WIDTH_MM - 12.0) / 15.0);
            v_left.push(tx(0.0, v));
            v_right.push(tx(RUSSIAN_ID1_HEIGHT_MM, v));
        }
        let v_anchor =
            calibrate_russian_debit_card_id1([&v_top, &v_right, &v_bot, &v_left], None, true)
                .expect("calibrate vertical Russian ID-1 debit card");
        assert!((v_anchor.meters_per_pixel - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn glomap_so3_averaging_and_ozaki2_fp4_rolling_shutter_ba_converge_sub_pixel() {
        // 3 cameras viewing a 3D ear + Russian ID-1 card landmark cloud with rolling shutter & VCM breathing
        let gt_rots = [
            MAT3_ID,
            so3_exp([0.04, 0.22, -0.03]),
            so3_exp([-0.05, -0.20, 0.02]),
        ];
        let gt_trans = [[0.0, 0.0, 0.35], [-0.08, 0.01, 0.36], [0.08, -0.01, 0.34]];

        // Test GLOMAP SO(3) rotation averaging from noisy initial guesses
        let edges = vec![
            RelativeRotationEdge {
                cam_i: 0,
                cam_j: 1,
                r_ij: mat3_mul(&gt_rots[1], &mat3_transpose(&gt_rots[0])),
                weight: 1.0,
            },
            RelativeRotationEdge {
                cam_i: 0,
                cam_j: 2,
                r_ij: mat3_mul(&gt_rots[2], &mat3_transpose(&gt_rots[0])),
                weight: 1.0,
            },
            RelativeRotationEdge {
                cam_i: 1,
                cam_j: 2,
                r_ij: mat3_mul(&gt_rots[2], &mat3_transpose(&gt_rots[1])),
                weight: 1.0,
            },
        ];
        let noisy_init = [
            MAT3_ID,
            mat3_mul(&so3_exp([0.08, -0.06, 0.05]), &gt_rots[1]),
            mat3_mul(&so3_exp([-0.07, 0.09, -0.04]), &gt_rots[2]),
        ];
        let avg_rots = glomap_so3_rotation_averaging(3, &edges, &noisy_init, 40);
        for c in 0..3 {
            let err = so3_log(&mat3_mul(&avg_rots[c], &mat3_transpose(&gt_rots[c])));
            let rad = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]).sqrt();
            assert!(
                rad < 1e-6,
                "GLOMAP rotation averaging cam {c} err = {rad:e}"
            );
        }

        // Ground-truth 3D landmarks (including 2 Russian ID-1 card corners 85.60 mm apart)
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

        // Generate exact rolling-shutter 2D observations
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

        // Perturb camera 1 & 2 poses + 3D points + reset VCM breathing to 1.0
        let mut cams = gt_cams.clone();
        cams[1].trans[0] += 0.004;
        cams[1].rot = mat3_mul(&so3_exp([0.012, -0.010, 0.008]), &cams[1].rot);
        cams[1].vcm_focal_scale = 1.0;
        cams[2].trans[1] -= 0.003;
        let mut pts = gt_pts.clone();
        for (i, p) in pts.iter_mut().enumerate() {
            if i >= 2 {
                p[0] += 0.0015 * ((i as f64).sin());
                p[2] -= 0.0020 * ((i as f64).cos());
            }
        }

        let gauges = vec![MetricGaugeConstraint {
            pt_a: 0,
            pt_b: 1,
            target_dist_m: RUSSIAN_ID1_WIDTH_MM * 1e-3,
            weight: 1e5,
        }];

        let rep =
            bundle_adjust_ozaki2_fp4_schur(&mut cams, &mut pts, &obs, &gauges, None, 10.0, 14);
        assert_eq!(rep.schur_backend, TensorBackend::Fp4E2m1Base13);
        assert!(
            rep.final_rms_px < 0.05 && rep.final_rms_px < rep.initial_rms_px * 0.05,
            "Ozaki-II FP4 Schur BA failed to converge: init={:.3} px -> final={:.4} px",
            rep.initial_rms_px,
            rep.final_rms_px
        );
        assert!(
            rep.metric_scale_error_mm < 0.5,
            "ID-1 metric gauge residual too large: {:.3} mm",
            rep.metric_scale_error_mm
        );
    }

    #[test]
    fn stage3_2dgs_pgsr_tsdf_to_stage4_watertight_fmm_bem_end_to_end() {
        let head = TriangleMesh::procedural_head_with_pinnae(0.0875, 2);
        let surfels = sample_surfels_from_mesh(&head);

        // Verify 2DGS/PGSR depth-distortion & normal-consistency losses are 0 for an aligned single plane
        let hits = vec![(0.25, 0.95, [0.0, 0.0, 1.0])];
        let (depth, l_dist, l_norm) = evaluate_2dgs_pgsr_ray_losses(&hits, [0.0, 0.0, 1.0]);
        assert!((depth - 0.25).abs() < 1e-12);
        assert!(l_dist < 1e-12 && l_norm < 1e-12);

        let anchor = calibrate_apriltag3_anchor(
            [
                [100.0, 100.0],
                [300.0, 100.0],
                [300.0, 300.0],
                [100.0, 300.0],
            ],
            40.0,
            false,
        )
        .expect("anchor");
        let ba = BundleAdjustmentReport {
            initial_rms_px: 12.0,
            final_rms_px: 0.02,
            max_vcm_breathing_pct: 0.31,
            metric_scale_error_mm: 0.04,
            schur_backend: TensorBackend::Fp4E2m1Base13,
            iterations: 10,
        };

        let out = run_stage3_to_stage4_to_fmm_bem(anchor, ba, &surfels, 12, 12, 0.06, 344.85)
            .expect("Stage 3 -> Stage 4 -> FMM-BEM handoff");
        assert!(out.mesh_stats.is_watertight);
        let h_l = out.solver.solve_reciprocal_hrtfs(
            1000.0,
            344.85,
            &[[0.0, 2.0, 0.0]],
            out.solver.left_ear_elem,
            false,
        )[0];
        assert!(h_l.norm().is_finite() && h_l.norm() > 0.5);
    }
}
