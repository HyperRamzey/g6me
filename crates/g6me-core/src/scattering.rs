//! Exact 3D Helmholtz rigid-sphere diffraction, coupled Head+Torso Snowman
//! scattering, and 1D Webster horn ear-canal acoustic impedance.
//!
//! 1. **Duda & Martens (JASA 1998)** — *"Range dependence of the response of a
//!    spherical head model"*: solves the 3D Helmholtz boundary-value problem outside
//!    a rigid sphere via Legendre polynomials `P_m(cos θ)` and spherical Hankel
//!    functions of the second kind `h_m^(2)(kr) = j_m(kr) - i y_m(kr)`. The Hankel
//!    sequence is evaluated in the scaled form `H_m(z) = h_m(z) / A_m(ka)`
//!    (`A_m(x) = (2m-1)!!/x^(m+1)`) so the evanescent tail does not overflow `f64`,
//!    and the radial products `j_m(k r) A_m(ka)` are built by a scaled downward
//!    Miller recurrence so the `f64` underflow of `j_m` (`m ≳ 122` at 125 Hz) does
//!    not truncate the series. Yields exact near-field acoustic parallax
//!    (e.g., `0.3 m` vs `3.0 m`) and satisfies exact Helmholtz reciprocity
//!    `|G(r1 -> r2) - G(r2 -> r1)| < 1e-13`.
//! 2. **Head + torso two-sphere scattering in the snowman geometry of Algazi, Duda,
//!    Duraiswami, Gumerov & Tang (JASA 2002)**: couples the rigid head sphere (`radius a`)
//!    with a rigid torso sphere (`radius b`, below the neck) using a reciprocal
//!    **single-scattering approximation** (direct field + head-diffracted field +
//!    torso-scattered field + one symmetric head<->torso saddle coupling).
//!    This is *not* Algazi's multipole re-expansion of the coupled two-sphere
//!    boundary-value problem, which requires the Gumerov–Duraiswami translation and
//!    re-expansion coefficients; it reproduces the `700 Hz – 3 kHz` shoulder-bounce band
//!    qualitatively while preserving exact reciprocity `G(r1, r2) == G(r2, r1)`.
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
/// even deep into the evanescent regime `m >> x`. The `y` output saturates at
/// `±1e280` once the true value leaves the `f64` range (`m` around 150 at `x ~ 1.6`);
/// callers that need that tail use the scaled Hankel sequence instead.
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

/// Scaled spherical Hankel function of the second kind,
/// `H_m(z) = h_m^(2)(z) / A_m(x0)` with `A_m(x) = (2m-1)!! / x^(m+1)`.
///
/// In the evanescent tail `h_m^(2)(z) ~ -i (2m-1)!! / z^(m+1)`, which overflows
/// `f64` around `m ~ 150` for low `ka`; dividing by `A_m(x0)` keeps both `H_m` and
/// the forward recurrence
/// `H_{m+1} = (x0/z) H_m - x0^2 H_{m-1} / ((2m+1)(2m-1))`
/// bounded, so the adaptive near-field cutoff can be honoured without saturation.
/// The recurrence selects the dominant (Hankel) solution; the omitted
/// `j_m / A_m(x0)` solution decays like `(x0 z)^m / ((2m+1)!! (2m-1)!!)`.
fn sph_hankel2_scaled(x0: f64, z: f64, m_max: usize) -> Vec<Complex> {
    let mut h = vec![Complex::ZERO; m_max + 2];
    // h_0^(2)(z) = i e^{-i z} / z, so H_0 = i (x0/z) e^{-i z}.
    h[0] = Complex::from_polar(x0 / z, 0.5 * PI - z);
    // h_1^(2)(z) = (sin z / z^2 - cos z / z) + i (cos z / z^2 + sin z / z);
    // H_1 = h_1^(2)(z) * x0^2.
    let (sz, cz) = (z.sin(), z.cos());
    let inv_z = 1.0 / z;
    let h1_re = sz * inv_z * inv_z - cz * inv_z;
    let h1_im = cz * inv_z * inv_z + sz * inv_z;
    h[1] = Complex::new(h1_re * x0 * x0, h1_im * x0 * x0);
    let x2 = x0 * x0;
    for m in 1..=m_max {
        let mf = m as f64;
        let c = x2 / ((2.0 * mf + 1.0) * (2.0 * mf - 1.0));
        h[m + 1] = h[m].scale(x0 / z).sub(h[m - 1].scale(c));
    }
    h
}

/// Number of multipole terms required by the exterior rigid-sphere radial series.
///
/// `k_r_lo` is `k * r_lo`. Two independent bounds set the count:
///
/// * `j_m(k r_lo)` turns from oscillatory to evanescent at `m ~ k r_lo`; past that
///   point the radial product decays factorially. Summing through the turning point
///   plus an Airy-transition margin of `12 (k r_lo)^(1/3)` terms (minimum 32) reaches
///   the `f64` floor. Checked against 50-digit `mpmath` over the audio grid
///   (`k r_lo <= 440`): when the upper clamp is not binding, the neglected tail is
///   below `1e-14` relative.
/// * For `m > k r_hi` both radial functions are evanescent, so the tail decays
///   geometrically like `(r_lo/r_hi)^m` and `38/ln(r_hi/r_lo)` terms push it below
///   `1e-16`. This bound governs the near-surface geometry where the two radii are
///   close; on the surface (`r_lo = a`) it is the only active bound when
///   `k r_hi < k r_lo + 32`.
///
/// The `64..=512` clamp bounds the pathological near-equal-radius geometry, where
/// the series converges only conditionally and more terms keep improving it.
fn radial_series_m_max(k_r_lo: f64, r_lo: f64, r_hi: f64) -> usize {
    let x = k_r_lo.max(0.0);
    let airy_margin = (12.0 * x.cbrt()).ceil() + 4.0;
    let turning = (x.ceil() as usize).saturating_add(airy_margin.max(32.0) as usize);
    let rho = (r_hi / r_lo).max(1.0 + 1e-12);
    let geometric = ((38.0 / rho.ln().max(1e-6)).ceil() as usize).saturating_add(8);
    turning.max(geometric).clamp(64, 512)
}

/// Scaled spherical Bessel product `J_m = j_m(y) * A_m(x)` with
/// `A_m(x) = (2m-1)!! / x^(m+1)`, evaluated directly by a scaled downward Miller
/// recurrence and normalized against the exact `J_0 = sin(y) / (x y)`.
///
/// For `m > y`, `j_m(y) ~ y^m/(2m+1)!!` underflows `f64` around `m = 122` at
/// `y = ka = 0.2`, while `A_m(x)` leaves the range far earlier; their product is
/// `~ 1/((2m+1)x)` when `x = y` and must not be truncated. The product satisfies
/// `J_{m+1} = ((2m+1)^2/(x y)) J_m - ((2m+1)(2m-1)/x^2) J_{m-1}`, whose minimal
/// solution is `j_m(y) A_m(x)`. The recurrence is integrated downward from
/// `m >> max(x, y)`, rescaling the sweep whenever it approaches the `f64` limits,
/// so neither factor is ever formed on its own.
fn scaled_j_times_a(x: f64, y: f64, m_max: usize) -> Vec<f64> {
    let len = m_max + 2;
    let mut c = vec![0.0_f64; len];
    if x <= 0.0 || y <= 0.0 || !x.is_finite() || !y.is_finite() {
        return c;
    }
    let m_start = (m_max + 44).max((y as usize).saturating_add(64));
    let inv_x = 1.0 / x;
    let inv_xy = 1.0 / (x * y);
    let mut cm1 = 0.0_f64; // c[m + 1]
    let mut cm0 = 1e-280_f64; // c[m]
    for m in (1..=m_start).rev() {
        let mf = m as f64;
        let a = (2.0 * mf + 1.0) * (2.0 * mf + 1.0) * inv_xy;
        let b = (2.0 * mf + 1.0) * (2.0 * mf - 1.0) * inv_x * inv_x;
        let c_prev = (a * cm0 - cm1) / b;
        if m - 1 < len {
            c[m - 1] = c_prev;
        }
        cm1 = cm0;
        cm0 = c_prev;
        // Going down the minimal solution crosses many decades (it grows like
        // (y/x)^m); keep the sweep inside f64 and carry the factor into the terms
        // already stored.
        let rescale = if cm0.abs() > 1e200 {
            1e-200
        } else if cm0 != 0.0 && cm0.abs() < 1e-200 {
            1e200
        } else {
            1.0
        };
        if rescale != 1.0 {
            cm0 *= rescale;
            cm1 *= rescale;
            let hi = len.min(m_start);
            for slot in &mut c[(m - 1).min(hi)..hi] {
                *slot *= rescale;
            }
        }
    }
    // Exact anchors: J_0 = j_0(y) A_0(x) = sin(y)/(x y) and
    // J_1 = j_1(y) A_1(x) = (sin y/y^2 - cos y/y)/x^2.
    let (anchor, index) = if c[0].abs() > 1e-300 {
        (y.sin() / (x * y), 0)
    } else if c[1].abs() > 1e-300 {
        ((y.sin() / (y * y) - y.cos() / y) / (x * x), 1)
    } else {
        return c;
    };
    let scale = anchor / c[index];
    for slot in &mut c {
        *slot *= scale;
    }
    c
}

/// Exact 3D Helmholtz Green's function `G_sphere(r1, r2)` outside a rigid sphere
/// of radius `a` centered at the origin, where `r1 = ||p1|| >= a`, `r2 = ||p2|| >= a`,
/// and `cos_theta = dot(p1, p2) / (r1 * r2)`.
///
/// Time convention `exp(+i ω t)` with outgoing wave `exp(-i k R) / (4 π R)`.
/// Uses `h_m^(2)(x) = j_m(x) - i y_m(x)` and converges the multipole series to the
/// `f64` floor (`~1e-16` relative in the near field, `≲ 1e-13` in the worst measured
/// far-field geometry). By exact symmetry in `(r1, r2)`, satisfies acoustic
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
    let k_r_lo = k * r_lo;
    // The cutoff is set by the series' two convergence scales, not by r_lo/a (which
    // is 1 on the surface and used to clamp every on-surface call to 512 terms):
    // the j_m(k r_lo) turning point at m ~ k r_lo, and the geometric tail beyond
    // m > k r_hi. See `radial_series_m_max`.
    let m_max = radial_series_m_max(k_r_lo, r_lo, r_hi);
    let on_surface = (r_lo - a).abs() <= 1e-13 * a.max(1.0);

    // Work with the scaled Hankel sequence (see `sph_hankel2_scaled`). The unscaled
    // y_m leaves the f64 range in exactly the high-m terms the adaptive cutoff needs
    // for a near-surface observation point; the scaled recurrence is bounded.
    let hx = sph_hankel2_scaled(ka, ka, m_max);
    let hhi = sph_hankel2_scaled(ka, k * r_hi, m_max);
    let hlo = if on_surface {
        hx.clone()
    } else {
        sph_hankel2_scaled(ka, k_r_lo, m_max)
    };

    // Scaled radial products for the off-surface bracket:
    // jlo_a[m] = j_m(k r_lo) A_m(ka) and ja_a[m] = j_m(ka) A_m(ka).
    // Forming the products directly keeps them representable where the individual
    // j_m and A_m factors underflow/overflow f64 (see `scaled_j_times_a`).
    let (jlo_a, ja_a) = if on_surface {
        (Vec::new(), Vec::new())
    } else {
        (
            scaled_j_times_a(ka, k_r_lo, m_max),
            scaled_j_times_a(ka, ka, m_max + 1),
        )
    };

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

        let mf = m as f64;
        // D_m = h_m^(2)'(ka) / A_m(ka) = (m/ka) H_m(ka) - ((2m+1)/ka) H_{m+1}(ka).
        let d = hx[m]
            .scale(mf / ka)
            .sub(hx[m + 1].scale((2.0 * mf + 1.0) / ka));
        if d.norm_sq() < 1e-300 {
            break;
        }

        let term = if on_surface {
            // Wronskian identity: j_m(ka) h_m^(2)'(ka) - j_m'(ka) h_m^(2)(ka) = -i / (ka)^2.
            // radial = -i / (ka^2 h_m^(2)'(ka)); with h_m(k r_hi) = A_m(ka) H_m(k r_hi),
            // term = -i (2m+1) P_m H_m(k r_hi) / (ka^2 D_m).
            let t = hhi[m].div(d);
            Complex::new(t.im, -t.re).scale((2.0 * mf + 1.0) * pm / (ka * ka))
        } else {
            // bracket = j_m(k r_lo) - (j_m'(ka)/h_m^(2)'(ka)) h_m^(2)(k r_lo).
            // With h_m(z) = A_m(ka) H_m(z), bracket * h_m(k r_hi) becomes
            //   (j_m(k r_lo) A_m(ka)) H_m(k r_hi)
            //   - (j_m'(ka) A_m(ka)) H_m(k r_lo) H_m(k r_hi) / D_m,
            // and j_m'(ka) A_m(ka) = (m/ka) j_m A_m(ka) - (ka/(2m+1)) j_{m+1} A_{m+1}(ka).
            let q = ja_a[m] * (mf / ka) - ja_a[m + 1] * (ka / (2.0 * mf + 1.0));
            let t1 = hhi[m].scale(jlo_a[m]);
            let t2 = hlo[m].mul(hhi[m]).div(d).scale(q);
            t1.sub(t2).scale((2.0 * mf + 1.0) * pm)
        };

        if !term.re.is_finite() || !term.im.is_finite() {
            break;
        }
        sum = sum.add(term);

        if mf > k_r_lo + 6.0 && term.norm() < 1e-16 * sum.norm().max(1e-300) {
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

/// Coupled Snowman (Head + Torso) two-sphere Green's function `G_snowman(p1, p2)`.
///
/// This is a reciprocal **single-scattering approximation** to the two-sphere
/// boundary-value problem, not Algazi et al.'s multipole re-expansion. It combines:
/// - Exact rigid-head Helmholtz Green's function `G_head(p1, p2)` (direct + head-diffracted),
/// - Exact rigid-torso scattered field `G_scat_torso(p1, p2)` (the `700 Hz – 3 kHz`
///   shoulder-bounce band and elevation notches),
/// - One symmetric head-torso coupling term evaluated at the torso-apex saddle point
///   `p_sh = (0, 0, -torso_drop + b)`.
///
/// By construction, EVERY term is strictly symmetric in `(p1, p2)`, guaranteeing
/// exact Helmholtz reciprocity `|G_snowman(p1, p2) - G_snowman(p2, p1)| < 1e-13`.
/// Multiple head<->torso scattering events are neglected.
pub fn snowman_green(k: f64, geom: SnowmanGeom, p1: [f64; 3], p2: [f64; 3]) -> Complex {
    let r1 = (p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2]).sqrt();
    let r2 = (p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]).sqrt();
    let cos_th =
        ((p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]) / (r1 * r2).max(1e-18)).clamp(-1.0, 1.0);

    // 1. Exact rigid-head boundary-value Green's function (direct + head diffraction).
    let g_head = rigid_sphere_green(k, geom.head_radius_m, r1, r2, cos_th);

    // 2. Exact single-sphere rigid-torso scattered wave (shoulder bounce 700 Hz – 3 kHz).
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
    fn near_field_series_matches_high_precision_reference() {
        // 60-digit mpmath evaluation of the exterior rigid-sphere series at rho = 1.25,
        // theta = pi. The old m_max = ka + 32 cutoff gave 1.4e-3 / 6.8e-4 relative error
        // here, so a 1e-9 tolerance discriminates the adaptive cutoff.
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        for (f, re, im) in [
            (
                1_000.0_f64,
                -0.385_783_444_255_766_2,
                0.097_099_711_997_860_93,
            ),
            (
                4_000.0_f64,
                0.163_898_508_173_518_33,
                -0.268_133_591_841_351_1,
            ),
        ] {
            let k = 2.0 * PI * f / c;
            let r = 1.25 * a;
            let g = rigid_sphere_green(k, a, r, a, -1.0);
            let g_ref = Complex::from_polar(1.0 / (4.0 * PI * r), -k * r);
            let h = g.div(g_ref);
            let expect = Complex::new(re, im);
            let rel = h.sub(expect).norm() / expect.norm();
            assert!(rel < 1e-9, "near-field series error at {f} Hz: {rel:e}");
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

    #[test]
    fn on_surface_term_count_tracks_convergence_not_a_fixed_cap() {
        // On the surface r_lo = a, so the angular tail decays like (a/r_hi)^m and the
        // j_m turning point sits at m ~ ka. A 50-digit mpmath evaluation of the same
        // series converges at m = 12..29 for r_hi = 2..10 m at 125 Hz..4 kHz and at
        // m = 51 for 12 kHz. The old cutoff used r_lo/a, which is 1 on the surface,
        // so it always clamped every on-surface call to 512 terms.
        let a = DEFAULT_HEAD_RADIUS_M;
        let c = 344.854_f64;
        for &f in &[125.0_f64, 250.0, 1_000.0, 4_000.0, 12_000.0] {
            let ka = 2.0 * PI * f / c * a;
            for &r in &[2.0_f64, 5.0, 10.0] {
                let m = radial_series_m_max(ka, a, r);
                assert!(
                    m <= 64,
                    "{f} Hz, r = {r} m: m_max = {m}, expected <= 64 (series converges by m <= 51)"
                );
                assert!(
                    m < 512,
                    "on-surface term count must not hit the old 512 cap"
                );
            }
        }
        // At 20 kHz ka = 31.9 and the Airy margin needs m = 70 (mpmath).
        let ka20 = 2.0 * PI * 20_000.0 / c * a;
        for &r in &[2.0_f64, 5.0, 10.0] {
            let m = radial_series_m_max(ka20, a, r);
            assert!(m <= 96, "20 kHz, r = {r} m: m_max = {m}, expected <= 96");
            assert!(m < 512);
        }
    }

    #[test]
    fn on_surface_green_matches_high_precision_at_2_5_10_m() {
        // 60-digit mpmath references for h = G(k, a, R, a, cos = -1) / G_ref with
        // G_ref = exp(-ikR)/(4 pi R). The reduced on-surface term count (64 at 12 kHz
        // instead of the old 512) must still reproduce the exact series.
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        for (f, r, re, im) in [
            (
                1_000.0_f64,
                2.0_f64,
                -0.860_222_333_354_848_5,
                -0.581_185_213_017_809_9,
            ),
            (
                1_000.0,
                5.0,
                -0.878_606_183_258_691_1,
                -0.626_259_515_404_759_5,
            ),
            (
                1_000.0,
                10.0,
                -0.884_771_748_019_710_6,
                -0.641_832_909_205_254_4,
            ),
            (
                12_000.0,
                2.0,
                0.775_704_110_988_236_8,
                0.061_228_895_176_753_19,
            ),
            (
                12_000.0,
                5.0,
                0.771_470_714_070_063_1,
                0.285_229_114_772_421_24,
            ),
            (
                12_000.0,
                10.0,
                0.756_004_431_013_975_2,
                0.361_363_389_316_898_65,
            ),
        ] {
            let k = 2.0 * PI * f / c;
            let g = rigid_sphere_green(k, a, r, a, -1.0);
            let g_ref = Complex::from_polar(1.0 / (4.0 * PI * r), -k * r);
            let h = g.div(g_ref);
            let expect = Complex::new(re, im);
            let rel = h.sub(expect).norm() / expect.norm();
            assert!(rel < 1e-11, "on-surface {f} Hz / {r} m: rel err = {rel:e}");
        }
    }

    #[test]
    fn scaled_radial_product_matches_high_precision_beyond_underflow() {
        // j_m(ka) underflows f64 at m = 122 (125 Hz) and m = 170 (1 kHz), yet the
        // series needs J_m = j_m(y) A_m(ka), which stays O(1e-2 .. 1e165).
        // 60-digit mpmath references computed at the same f64 arguments.
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        let f = 125.0_f64;
        let k = 2.0 * PI * f / c;
        let (x, y) = (k * a, k * 0.12);
        let j = scaled_j_times_a(x, y, 256);
        for (m, expect) in [
            (122usize, 1.112_846_165_785_190_9e15_f64),
            (150, 6.279_083_570_643_619e18),
            (200, 3.404_013_862_133_299e25),
            (256, 1.278_574_060_856_14e33),
        ] {
            let rel = (j[m] - expect).abs() / expect.abs();
            assert!(rel < 1e-12, "J_{m}(k r_lo) at 125 Hz: rel err = {rel:e}");
        }
        // On-axis J_m = j_m(ka) A_m(ka) approaches 1/((2m+1)ka); it is finite and
        // non-vanishing exactly where sph_bessel_jy returns 0.0.
        let b = scaled_j_times_a(x, x, 320);
        for (m, expect) in [
            (122usize, 0.020_480_301_055_708_826_f64),
            (256, 0.009_781_449_650_789_74),
        ] {
            let rel = (b[m] - expect).abs() / expect;
            assert!(rel < 1e-12, "J_{m}(ka) at 125 Hz: rel err = {rel:e}");
        }
        let f = 1_000.0_f64;
        let k = 2.0 * PI * f / c;
        let (x, y) = (k * a, k * 0.32);
        let j = scaled_j_times_a(x, y, 300);
        for (m, expect) in [
            (170usize, 9.490_668_948_778_331e92_f64),
            (220, 1.065_536_672_550_816_3e121),
            (256, 1.727_145_679_904_831_5e141),
            (300, 8.890_140_655_000_878e165),
        ] {
            let rel = (j[m] - expect).abs() / expect;
            assert!(rel < 1e-12, "J_{m}(k r_lo) at 1 kHz: rel err = {rel:e}");
        }
        let b = scaled_j_times_a(x, x, 256);
        for (m, expect) in [
            (170usize, 0.001_832_668_499_358_581_8_f64),
            (256, 0.001_219_714_895_554_450_3),
        ] {
            let rel = (b[m] - expect).abs() / expect;
            assert!(rel < 1e-12, "J_{m}(ka) at 1 kHz: rel err = {rel:e}");
        }
    }

    #[test]
    fn off_surface_series_not_truncated_by_bessel_underflow() {
        // ka = 0.199 (125 Hz): the old j_m(ka) table underflows to 0.0 at m = 122 and
        // the `ja[m] == 0.0` break truncated the series there, giving 8.8e-7 relative
        // error against this 60-digit mpmath reference. The true tail (r_lo/r_hi)^m is
        // still ~1e-7 at m = 122 for r_hi/r_lo = 1.083.
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        let k = 2.0 * PI * 125.0 / c;
        let g = rigid_sphere_green(k, a, 0.12, 0.130, 0.5);
        let expect = Complex::new(0.633_350_135_522_297_5, -0.176_624_808_478_865_33);
        let rel = g.sub(expect).norm() / expect.norm();
        assert!(rel < 1e-9, "underflow-truncated tail: rel err = {rel:e}");
    }

    #[test]
    fn off_surface_green_matches_high_precision_across_audio_range() {
        // 60-digit mpmath references. The old `ka + 32` floor truncated at m = 38-51
        // while the j_m(k r_lo) turning point at k r_lo = 33-98 needs m = 76-130,
        // giving 1.3e-2 (4 kHz) and 1.0 (12 kHz) relative error before the fix.
        // The 1 kHz case is well conditioned: old and new agree to ~1 ulp (2.3e-16).
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        for (f, r1, r2, ct, re, im, tol) in [
            (
                1_000.0_f64,
                0.32_f64,
                2.40_f64,
                0.42_f64,
                -0.027_186_332_267_177_59,
                0.020_911_099_599_164_61,
                1e-12_f64,
            ),
            (
                4_000.0,
                0.45,
                2.10,
                0.35,
                0.040_555_112_122_563_707,
                -0.011_463_277_787_819_156,
                1e-12,
            ),
            (
                12_000.0,
                0.32,
                2.40,
                0.42,
                -0.037_407_002_327_584_32,
                0.000_144_537_839_732_818_85,
                1e-11,
            ),
        ] {
            let k = 2.0 * PI * f / c;
            let g = rigid_sphere_green(k, a, r1, r2, ct);
            let expect = Complex::new(re, im);
            let rel = g.sub(expect).norm() / expect.norm();
            assert!(
                rel < tol,
                "{f} Hz, r = {r1}/{r2} m: rel err = {rel:e} (tol {tol:e})"
            );
        }
    }

    #[test]
    fn green_function_is_finite_across_audio_grid() {
        // Guard the scaled recurrences against overflow/NaN over the full parameter
        // space used by the HRTF path and the verifier checks.
        let c = 344.854_f64;
        let a = DEFAULT_HEAD_RADIUS_M;
        for &f in &[20.0_f64, 125.0, 1_000.0, 4_000.0, 12_000.0, 20_000.0] {
            let k = 2.0 * PI * f / c;
            for &r1 in &[a, 1.2 * a, 0.12_f64, 0.32, 0.45, 2.4, 10.0] {
                for &r2 in &[a, 1.2 * a, 0.12_f64, 0.32, 0.45, 2.4, 10.0] {
                    for &ct in &[-1.0_f64, -0.3, 0.0, 0.7, 1.0] {
                        let g = rigid_sphere_green(k, a, r1, r2, ct);
                        assert!(
                            g.re.is_finite() && g.im.is_finite(),
                            "non-finite G at f = {f} Hz, r = {r1}/{r2} m, cos = {ct}"
                        );
                    }
                }
            }
        }
    }
}
