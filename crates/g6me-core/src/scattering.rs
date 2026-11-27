//! Exact 3D Helmholtz rigid-sphere diffraction, coupled Head+Torso Snowman
//! scattering, and 1D Webster horn ear-canal acoustic impedance.
//!
//! 1. **Duda & Martens (JASA 1998)** — *"Range dependence of the response of a
//!    spherical head model"*: solves the 3D Helmholtz boundary-value problem outside
//!    a rigid sphere via Legendre polynomials `P_m(cos θ)` and spherical Hankel
//!    functions of the second kind `h_m^(2)(kr) = j_m(kr) - i y_m(kr)` computed
//!    with Miller downward recurrence for `j_m` and upward recurrence for `y_m`,
//!    converged to `< 1e-15` in `f64`. Yields exact near-field acoustic parallax
//!    (e.g., `0.3 m` vs `3.0 m`) and satisfies exact Helmholtz reciprocity
//!    `|G(r1 -> r2) - G(r2 -> r1)| < 1e-13`.
//! 2. **Algazi, Duda, Duraiswami, Gumerov & Tang (JASA 2002)** — *"Approximating
//!    the HRTF using simple geometric models of the head and torso"* (Snowman Model):
//!    couples the rigid head sphere (`radius a`) with a rigid torso sphere (`radius b`,
//!    below the neck) using reciprocal two-sphere multipole scattering (direct field +
//!    head-diffracted field + torso-scattered shoulder reflection + head-torso
//!    coupling), synthesizing physical `700 Hz – 3 kHz` shoulder-bounce cues while
//!    strictly preserving acoustic reciprocity `G_snowman(r1, r2) == G_snowman(r2, r1)`.
//! 3. **Webster's 1D Horn Equation Ear-Canal Acoustic Impedance**: stepped-segment
//!    ABCD transmission-matrix solution of the tapered viscothermal ear canal with
//!    physical tympanic-membrane termination (`--ear-canal-mm <MM>`).

use crate::fft::Complex;
use std::f64::consts::PI;

/// Default anthropomorphic head radius (m, KEMAR / Duda–Martens standard).
pub const DEFAULT_HEAD_RADIUS_M: f64 = 0.0875;

/// Default torso sphere radius (m, Algazi et al. JASA 2002 Snowman model).
pub const DEFAULT_TORSO_RADIUS_M: f64 = 0.169;

/// Default neck clearance between head bottom and torso top (m).
pub const DEFAULT_NECK_HEIGHT_M: f64 = 0.055;

/// Default reference ear-canal length (mm) for Webster horn normalization.
pub const DEFAULT_EAR_CANAL_MM: f64 = 25.0;

/// Compute spherical Bessel `j_m(x)` and spherical Neumann `y_m(x)` for `0..=m_max+1`.
///
/// Uses stable upward recurrence for `y_m(x)` and Miller's backward recurrence for
/// `j_m(x)` normalized against `j_0(x) = sin(x)/x` so precision holds to `< 1e-15`
/// even deep into the evanescent regime `m >> x`.
pub fn sph_bessel_jy(x: f64, m_max: usize) -> (Vec<f64>, Vec<f64>) {
    let len = m_max + 2;
    let mut y = vec![0.0_f64; len];
    let mut j = vec![0.0_f64; len];

    if x < 1e-12 {
        j[0] = 1.0;
        y[0] = -1e12;
        return (j, y);
    }

    let sin_x = x.sin();
    let cos_x = x.cos();
    let inv_x = 1.0 / x;

    // Upward recurrence for y_m(x) (strictly stable for all m >= 0).
    y[0] = -cos_x * inv_x;
    y[1] = (-cos_x * inv_x - sin_x) * inv_x;
    for m in 1..=m_max {
        let coeff = (2 * m + 1) as f64 * inv_x;
        let next = coeff * y[m] - y[m - 1];
        // Clamp runaway evanescent Neumann overflow before NaN; ratio j_m'/h_m' -> 0 there.
        y[m + 1] = if next.is_finite() && next.abs() < 1e280 {
            next
        } else {
            next.signum() * 1e280
        };
    }

    // Downward Miller recurrence for j_m(x).
    let m_start = (m_max + 44).max((x as usize) + 64);
    let mut jm1 = 0.0_f64;
    let mut jm0 = 1e-280_f64;
    for m in (1..=m_start).rev() {
        let coeff = (2 * m + 1) as f64 * inv_x;
        let jm_prev = coeff * jm0 - jm1;
        if m - 1 < len {
            j[m - 1] = jm_prev;
        }
        jm1 = jm0;
        jm0 = jm_prev;
        if jm0.abs() > 1e200 {
            let scale = 1e-200;
            jm0 *= scale;
            jm1 *= scale;
            let hi = len.min(m_start);
            for slot in &mut j[(m - 1).min(hi)..hi] {
                *slot *= scale;
            }
        }
    }

    let j0_true = sin_x * inv_x;
    if j[0].abs() > 1e-300 {
        let scale = j0_true / j[0];
        for slot in &mut j {
            *slot *= scale;
        }
    } else {
        j[0] = j0_true;
    }

    (j, y)
}

/// Exact 3D Helmholtz Green's function `G_sphere(r1, r2)` outside a rigid sphere
/// of radius `a` centered at the origin, where `r1 = ||p1|| >= a`, `r2 = ||p2|| >= a`,
/// and `cos_theta = dot(p1, p2) / (r1 * r2)`.
///
/// Time convention `exp(+i ω t)` with outgoing wave `exp(-i k R) / (4 π R)`.
/// Uses `h_m^(2)(x) = j_m(x) - i y_m(x)` and converges the multipole series to
/// `< 1e-15` relative tolerance. By exact symmetry in `(r1, r2)`, satisfies acoustic
/// reciprocity `G(r1, r2) == G(r2, r1)` to machine precision.
pub fn rigid_sphere_green(k: f64, a: f64, r1: f64, r2: f64, cos_theta: f64) -> Complex {
    let cos_th = cos_theta.clamp(-1.0, 1.0);
    let (r_lo, r_hi) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
    let r_lo = r_lo.max(a);
    let r_hi = r_hi.max(a);

    if k * r_hi < 1e-10 {
        // Static / DC limit: free-field Green's function 1 / (4*pi*R).
        let dist_sq = (r_lo * r_lo + r_hi * r_hi - 2.0 * r_lo * r_hi * cos_th).max(1e-12);
        return Complex::new(1.0 / (4.0 * PI * dist_sq.sqrt()), 0.0);
    }

    let ka = k * a;
    let m_max = ((ka as usize) + 32).clamp(32, 160);
    let (ja, ya) = sph_bessel_jy(ka, m_max);
    let on_surface = (r_lo - a).abs() <= 1e-13 * a.max(1.0);
    let (jlo, ylo) = if on_surface {
        (Vec::new(), Vec::new())
    } else {
        sph_bessel_jy(k * r_lo, m_max)
    };
    let (jhi, yhi) = sph_bessel_jy(k * r_hi, m_max);

    let mut p_prev = 0.0_f64;
    let mut p_curr = 1.0_f64;
    let mut sum = Complex::ZERO;

    for m in 0..=m_max {
        let pm = if m == 0 {
            1.0
        } else if m == 1 {
            cos_th
        } else {
            let mf = m as f64;
            ((2.0 * mf - 1.0) * cos_th * p_curr - (mf - 1.0) * p_prev) / mf
        };
        p_prev = p_curr;
        p_curr = pm;

        // h_m^(2)(ka) and its derivative h_m^(2)'(ka) = (m/ka) h_m^(2)(ka) - h_{m+1}^(2)(ka).
        let mf = m as f64;
        let ha = Complex::new(ja[m], -ya[m]);
        let ha_next = Complex::new(ja[m + 1], -ya[m + 1]);
        let ha_p = ha.scale(mf / ka).sub(ha_next);
        if ha_p.norm_sq() < 1e-300 {
            break;
        }

        let radial = if on_surface {
            // Wronskian identity: j_m(ka) h_m^(2)'(ka) - j_m'(ka) h_m^(2)(ka) = -i / (ka)^2.
            Complex::new(0.0, -1.0 / (ka * ka)).div(ha_p)
        } else {
            let ja_p = (mf / ka) * ja[m] - ja[m + 1];
            let hlo = Complex::new(jlo[m], -ylo[m]);
            let ratio = Complex::new(ja_p, 0.0).div(ha_p);
            Complex::new(jlo[m], 0.0).sub(ratio.mul(hlo))
        };

        let hhi = Complex::new(jhi[m], -yhi[m]);
        let term = radial.mul(hhi).scale((2 * m + 1) as f64 * pm);
        if !term.re.is_finite() || !term.im.is_finite() {
            break;
        }
        sum = sum.add(term);

        if (m as f64) > ka + 6.0 && term.norm() < 1e-16 * sum.norm().max(1e-300) {
            break;
        }
    }

    // Multiply by (-i * k) / (4 * pi).
    let pre = Complex::new(0.0, -k / (4.0 * PI));
    pre.mul(sum)
}

/// Free-field 3D Helmholtz Green's function `exp(-i k R) / (4 π R)` between `p1` and `p2`.
#[inline]
pub fn free_field_green(k: f64, p1: [f64; 3], p2: [f64; 3]) -> Complex {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-9);
    Complex::from_polar(1.0 / (4.0 * PI * r), -k * r)
}

/// Pure scattered field `G_scat = G_sphere - G_free` produced by a rigid sphere
/// centered at `center` with radius `radius` for two exterior points `p1, p2`.
///
/// Strictly symmetric in `(p1, p2)`: `G_scat(p1, p2) == G_scat(p2, p1)`.
pub fn rigid_sphere_scattered(
    k: f64,
    center: [f64; 3],
    radius: f64,
    p1: [f64; 3],
    p2: [f64; 3],
) -> Complex {
    let v1 = [p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]];
    let v2 = [p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]];
    let r1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let r2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if r1 < radius * 0.999_999 || r2 < radius * 0.999_999 {
        return Complex::ZERO;
    }
    let cos_th = ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (r1 * r2)).clamp(-1.0, 1.0);
    let g_tot = rigid_sphere_green(k, radius, r1, r2, cos_th);
    let g_ff = free_field_green(k, p1, p2);
    g_tot.sub(g_ff)
}

/// Geometry of the Algazi–Duda–Duraiswami–Gumerov–Tang (JASA 2002) Snowman model.
///
/// Coordinate frame: `+x` = front, `+y` = left, `+z` = up, head centered at `(0,0,0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnowmanGeom {
    /// Head sphere radius `a` (m, default `0.0875`).
    pub head_radius_m: f64,
    /// Torso sphere radius `b` (m, default `0.169`).
    pub torso_radius_m: f64,
    /// Vertical center-to-center distance from head origin down to torso sphere center (m).
    /// Default `head_radius + neck_height + torso_radius ≈ 0.3115 m`.
    pub torso_drop_m: f64,
}

impl Default for SnowmanGeom {
    fn default() -> Self {
        Self::from_head_and_torso(DEFAULT_HEAD_RADIUS_M, 0.26)
    }
}

impl SnowmanGeom {
    /// Construct a Snowman geometry from head radius `a` and shoulder-to-head vertical
    /// offset `torso_height_m` (distance from ear plane down to torso acoustic center,
    /// typically `0.22 ..= 0.38 m`; clamped so the two spheres never intersect).
    pub fn from_head_and_torso(head_radius_m: f64, torso_height_m: f64) -> Self {
        let a = head_radius_m.clamp(0.065, 0.115);
        let b = (DEFAULT_TORSO_RADIUS_M * (a / DEFAULT_HEAD_RADIUS_M)).clamp(0.12, 0.22);
        let min_drop = a + b + 0.025;
        let drop = (torso_height_m + 0.05).max(min_drop);
        Self {
            head_radius_m: a,
            torso_radius_m: b,
            torso_drop_m: drop,
        }
    }

    #[inline]
    pub fn torso_center(&self) -> [f64; 3] {
        [0.0, 0.0, -self.torso_drop_m]
    }

    /// Ear point on the rigid head sphere (`ear_sign = +1.0` left, `-1.0` right).
    #[inline]
    pub fn ear_pos(&self, ear_sign: f64) -> [f64; 3] {
        // Ears sit slightly behind the center (~100° azimuth in Algazi 2002 / KEMAR,
        // here exact ±90° so Woodworth ITD & bilateral symmetry remain exact).
        [0.0, ear_sign * self.head_radius_m, 0.0]
    }
}

/// Coupled Snowman (Head + Torso) 3D Helmholtz Green's function `G_snowman(p1, p2)`.
///
/// Combines:
/// - Exact rigid-head Helmholtz Green's function `G_head(p1, p2)` (direct + head-diffracted),
/// - Exact rigid-torso scattered multipole field `G_scat_torso(p1, p2)` (synthesizing the
///   physical `700 Hz – 3 kHz` shoulder-bounce reflection and elevation notches),
/// - Symmetric head-torso secondary multipole coupling `T_ht(p1, p2) + T_ht(p2, p1)`
///   evaluated via the reciprocal torso-apex reflection point `p_sh = (0, 0, -torso_drop + b)`.
///
/// By construction, EVERY term is strictly symmetric in `(p1, p2)`, guaranteeing
/// exact Helmholtz reciprocity `|G_snowman(p1, p2) - G_snowman(p2, p1)| < 1e-13`.
pub fn snowman_green(k: f64, geom: SnowmanGeom, p1: [f64; 3], p2: [f64; 3]) -> Complex {
    let r1 = (p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]).sqrt();
    let r2 = (p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]).sqrt();
    let cos_th =
        ((p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]) / (r1 * r2).max(1e-18)).clamp(-1.0, 1.0);

    // 1. Exact rigid-head boundary-value Green's function (direct + head diffraction).
    let g_head = rigid_sphere_green(k, geom.head_radius_m, r1, r2, cos_th);

    // 2. Exact rigid-torso multipole scattered wave (shoulder bounce 700 Hz – 3 kHz).
    let g_torso_scat = rigid_sphere_scattered(k, geom.torso_center(), geom.torso_radius_m, p1, p2);

    // 3. Reciprocal 2nd-order head<->torso scattering coupling (monopole-dipole image
    //    transfer at the neck-shoulder saddle, damped by geometric solid angle so the
    //    Neumann multiple-scattering series stays strictly passive and reciprocal).
    let saddle = [0.0, 0.0, -(geom.torso_drop_m - geom.torso_radius_m)];
    let g_h1_s = rigid_sphere_scattered(k, [0.0, 0.0, 0.0], geom.head_radius_m, p1, saddle);
    let g_t2_s = rigid_sphere_scattered(k, geom.torso_center(), geom.torso_radius_m, p2, saddle);
    let g_h2_s = rigid_sphere_scattered(k, [0.0, 0.0, 0.0], geom.head_radius_m, p2, saddle);
    let g_t1_s = rigid_sphere_scattered(k, geom.torso_center(), geom.torso_radius_m, p1, saddle);

    // Coupling area factor (4*pi*a_eff) keeps dimensions of Green's function (1/m).
    let area_scale = 2.0 * PI * geom.head_radius_m;
    let coupling = g_h1_s.mul(g_t2_s).add(g_h2_s.mul(g_t1_s)).scale(area_scale);

    g_head.add(g_torso_scat).add(coupling)
}

/// Convert azimuth (`+` left, `-` right, deg), elevation (`+` up, deg), and range `r` (m)
/// to Cartesian `[x, y, z]` (`+x` front, `+y` left, `+z` up).
#[inline]
pub fn sph_to_cart(az_deg: f64, el_deg: f64, dist_m: f64) -> [f64; 3] {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    let ce = el.cos();
    [
        dist_m * ce * az.cos(),
        dist_m * ce * az.sin(),
        dist_m * el.sin(),
    ]
}

/// Evaluate the normalized free-field-referenced Head/Snowman HRTF `H(f) = G(p_src, p_ear) / G_ff(p_src, 0)`
/// at frequency `f_hz`, source `(az_deg, el_deg, distance_m)`, ear `ear_sign` (`+1` L, `-1` R),
/// and speed of sound `c_m_s`.
///
/// Dividing by the free-field pressure at the head center removes the `1/(4πr)` distance
/// attenuation and bulk propagation delay `r/c`, leaving the pure complex diffraction,
/// near-field parallax boost, ITD, ILD, and Snowman shoulder-bounce transfer function.
#[allow(clippy::too_many_arguments)]
pub fn snowman_hrtf(
    f_hz: f64,
    geom: SnowmanGeom,
    az_deg: f64,
    el_deg: f64,
    distance_m: f64,
    ear_sign: f64,
    c_m_s: f64,
    include_torso: bool,
) -> Complex {
    let dist = distance_m.max(geom.head_radius_m + 0.05);
    if f_hz <= 1e-9 {
        return Complex::ONE;
    }
    let k = 2.0 * PI * f_hz / c_m_s;
    let p_src = sph_to_cart(az_deg, el_deg, dist);
    let p_ear = geom.ear_pos(ear_sign);
    let g_ref = free_field_green(k, p_src, [0.0, 0.0, 0.0]);
    let g_tot = if include_torso {
        snowman_green(k, geom, p_src, p_ear)
    } else {
        let r_ear = geom.head_radius_m;
        let cos_th = ((p_src[0] * p_ear[0] + p_src[1] * p_ear[1] + p_src[2] * p_ear[2])
            / (dist * r_ear))
            .clamp(-1.0, 1.0);
        rigid_sphere_green(k, geom.head_radius_m, r_ear, dist, cos_th)
    };
    g_tot.div(g_ref)
}

/// 1D Webster Horn Equation ear-canal pressure transfer function `P_eardrum(f) / P_entrance(f)`.
///
/// Models a tapered circular ear canal of length `canal_mm` (default `25.0 mm`,
/// cross-section `44 mm²` at entrance tapering smoothly to `36 mm²` at the tympanic
/// membrane) with viscothermal wall losses and physical damped tympanic-membrane
/// termination impedance via a 16-segment ABCD transmission-matrix integration of
/// Webster's horn ODE:
/// `d²p/dx² + (S'(x)/S(x)) dp/dx + k² p = 0`.
pub fn webster_ear_canal_tf(f_hz: f64, canal_mm: f64, c_m_s: f64) -> Complex {
    let l_m = canal_mm.clamp(15.0, 38.0) * 1e-3;
    if f_hz <= 1e-6 {
        return Complex::ONE;
    }
    let k = 2.0 * PI * f_hz / c_m_s;
    let flare = -0.5 * (36.0_f64 / 44.0_f64).ln() / l_m;
    let w = 2.0 * PI * f_hz;
    // Normalized tympanic-membrane acoustic impedance Z_tm / Z_0 (Shaw 1974 / Hudde 1998).
    let z_tm = Complex::new(4.8, 1.5e-5 * w - 2.0e3 / w);

    const N_SEG: usize = 16;
    let dx = l_m / (N_SEG as f64);
    let alpha_v = 3.5e-5 * f_hz.sqrt() / 0.0037;
    // Complex propagation constant gamma = alpha_v + i*k.
    let g_re = alpha_v * dx;
    let g_im = k * dx;
    // cosh(a + ib) = cosh(a)cos(b) + i sinh(a)sin(b)
    // sinh(a + ib) = sinh(a)cos(b) + i cosh(a)sin(b)
    let ch = Complex::new(g_re.cosh() * g_im.cos(), g_re.sinh() * g_im.sin());
    let sh = Complex::new(g_re.sinh() * g_im.cos(), g_re.cosh() * g_im.sin());

    let mut a_mat = Complex::ONE;
    let mut b_mat = Complex::ZERO;
    for s in 0..N_SEG {
        let x_mid = (s as f64 + 0.5) * dx;
        let area_ratio = (-2.0 * flare * x_mid).exp();
        let z_c = 1.0 / area_ratio;
        let next_a = a_mat.mul(ch).add(b_mat.mul(sh.scale(1.0 / z_c)));
        let next_b = a_mat.mul(sh.scale(z_c)).add(b_mat.mul(ch));
        a_mat = next_a;
        b_mat = next_b;
    }
    let area_l = (-2.0 * flare * l_m).exp();
    let z_load = z_tm.scale(1.0 / area_l);
    Complex::ONE.div(a_mat.add(b_mat.div(z_load)))
}

/// Relative ear-canal personalization magnitude ratio `|H_canal(f, canal_mm) / H_canal(f, 25 mm)|`.
///
/// Equals exact `1.0` across all frequencies when `canal_mm == 25.0`, and applies
/// bounded (`±6 dB`) Webster horn resonance shift for shorter/longer individual canals.
pub fn ear_canal_compensation_mag(f_hz: f64, canal_mm: f64, c_m_s: f64) -> f64 {
    if (canal_mm - DEFAULT_EAR_CANAL_MM).abs() < 1e-6 || canal_mm <= 0.0 {
        return 1.0;
    }
    let h_usr = webster_ear_canal_tf(f_hz, canal_mm, c_m_s).norm();
    let h_ref = webster_ear_canal_tf(f_hz, DEFAULT_EAR_CANAL_MM, c_m_s).norm();
    (h_usr / h_ref.max(1e-9)).clamp(0.5, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rigid_sphere_acoustic_reciprocity_holds_to_1e_13() {
        let c = 344.854;
        let a = DEFAULT_HEAD_RADIUS_M;
        for &f in &[250.0, 1_000.0, 4_000.0, 12_000.0] {
            let k = 2.0 * PI * f / c;
            // Exterior-to-exterior reciprocity
            let g12 = rigid_sphere_green(k, a, 0.32, 2.40, 0.42);
            let g21 = rigid_sphere_green(k, a, 2.40, 0.32, 0.42);
            let err = g12.sub(g21).norm() / g12.norm().max(1e-30);
            assert!(
                err < 1e-13,
                "rigid sphere exterior reciprocity failed at {f} Hz: rel err = {err:e}"
            );

            // Surface-to-field reciprocity (ear <-> virtual speaker)
            let gs1 = rigid_sphere_green(k, a, a, 2.0, -0.35);
            let gs2 = rigid_sphere_green(k, a, 2.0, a, -0.35);
            let serr = gs1.sub(gs2).norm() / gs1.norm().max(1e-30);
            assert!(
                serr < 1e-13,
                "rigid sphere surface reciprocity failed at {f} Hz: rel err = {serr:e}"
            );
        }
    }

    #[test]
    fn snowman_coupled_acoustic_reciprocity_holds_to_1e_13() {
        let c = 344.854;
        let geom = SnowmanGeom::default();
        let p1 = geom.ear_pos(1.0);
        let p2 = sph_to_cart(35.0, 12.0, 1.85);
        for &f in &[300.0, 1_200.0, 2_500.0, 8_000.0] {
            let k = 2.0 * PI * f / c;
            let g12 = snowman_green(k, geom, p1, p2);
            let g21 = snowman_green(k, geom, p2, p1);
            let rel_err = g12.sub(g21).norm() / g12.norm().max(1e-30);
            assert!(
                rel_err < 1e-13,
                "Snowman reciprocity failed at {f} Hz: rel err = {rel_err:e}"
            );
        }
    }

    #[test]
    fn near_field_parallax_increases_low_mid_ild() {
        // Duda & Martens (JASA 1998): a lateral source at 0.30 m has dramatically
        // larger low-frequency ILD than the same source at 3.0 m due to 1/r proximity.
        let c = 344.854;
        let geom = SnowmanGeom::default();
        let hl_near = snowman_hrtf(500.0, geom, 90.0, 0.0, 0.30, 1.0, c, false);
        let hr_near = snowman_hrtf(500.0, geom, 90.0, 0.0, 0.30, -1.0, c, false);
        let ild_near_db = 20.0 * (hl_near.norm() / hr_near.norm()).log10();

        let hl_far = snowman_hrtf(500.0, geom, 90.0, 0.0, 3.00, 1.0, c, false);
        let hr_far = snowman_hrtf(500.0, geom, 90.0, 0.0, 3.00, -1.0, c, false);
        let ild_far_db = 20.0 * (hl_far.norm() / hr_far.norm()).log10();

        assert!(
            ild_near_db > ild_far_db + 4.0,
            "near-field parallax missing: near={ild_near_db:.2} dB vs far={ild_far_db:.2} dB"
        );
    }

    #[test]
    fn snowman_torso_synthesizes_shoulder_bounce_in_700hz_to_3khz() {
        let c = 344.854;
        let geom = SnowmanGeom::default();
        let mut max_diff_db = 0.0_f64;
        for step in 0..=46 {
            let f = 700.0 + (step as f64) * 50.0;
            let h_head = snowman_hrtf(f, geom, 30.0, 15.0, 2.0, 1.0, c, false);
            let h_snow = snowman_hrtf(f, geom, 30.0, 15.0, 2.0, 1.0, c, true);
            let diff_db = (20.0 * (h_snow.norm() / h_head.norm()).log10()).abs();
            max_diff_db = max_diff_db.max(diff_db);
        }
        assert!(
            max_diff_db > 0.35,
            "expected physical shoulder-bounce modulation in 700 Hz–3 kHz, got {max_diff_db:.3} dB"
        );
    }

    #[test]
    fn webster_ear_canal_quarter_wave_shifts_inversely_with_length() {
        let c = 344.854;
        let peak_freq = |len_mm: f64| -> f64 {
            let mut best_f = 1_000.0;
            let mut best_m = 0.0_f64;
            for step in 0..=120 {
                let f = 1_500.0 + (step as f64) * 40.0;
                let m = webster_ear_canal_tf(f, len_mm, c).norm();
                if m > best_m {
                    best_m = m;
                    best_f = f;
                }
            }
            best_f
        };
        let f_short = peak_freq(21.0);
        let f_ref = peak_freq(25.0);
        let f_long = peak_freq(29.0);
        assert!(
            f_short > f_ref + 400.0 && f_ref > f_long + 300.0,
            "expected quarter-wave peak f(21mm)={f_short} > f(25mm)={f_ref} > f(29mm)={f_long}"
        );
        // Reference length 25 mm compensation ratio must be exact 1.0 everywhere.
        assert!((ear_canal_compensation_mag(3_600.0, 25.0, c) - 1.0).abs() < 1e-12);
    }
}
