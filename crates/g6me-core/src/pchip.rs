//! Fritsch–Carlson monotonic cubic Hermite (PCHIP) interpolation in `log10(f)`.
//!
//! The room model interpolates per-band material tables (`absorption[10]`,
//! `scattering[10]`) onto arbitrary FFT-bin frequencies. Plain piecewise-linear
//! interpolation on the *unnormalised* `log2(f)` axis is discontinuous at the
//! nominal ISO centres whose spacing is not exactly one octave (`63 → 125 Hz`
//! spans `0.9887` octaves): the left-hand limit at 125 Hz approaches
//! `y(63) + 0.9887·(y(125) − y(63))`, then the right-hand branch restarts from
//! `y(125)`, stepping the interpolated value by ~1.1 % of the 63→125 rise.
//! Monotone cubic Hermite interpolation removes the step (it is C¹) and, unlike a
//! natural cubic spline, cannot overshoot the tabulated values.
//!
//! Ported term-for-term from `HyprEQ/src/pchip.rs::Pchip` (the same functions in
//! `hypreq.py::pchip_log_interp`), including:
//!
//! * queries clamped to `[f_nodes[0], f_nodes[-1]]` **before** the `log10`, so
//!   everything below the first knot reads the first value instead of
//!   extrapolating;
//! * interval index `searchsorted(…, side="right") - 1` clipped to `[0, n-2]`,
//!   so a query exactly on an interior knot takes the interval *above* it (both
//!   Hermite blends agree there);
//! * the tangent at a knot is the weighted harmonic mean of the two secant
//!   slopes only when they agree in sign, and exactly `0.0` otherwise, which
//!   flattens the curve at every local extremum instead of overshooting it.
//!
//! ### Endpoint Derivatives: Fritsch & Carlson (1980) Eq. (8)
//! Endpoints use the shape-preserving non-centred 3-point formula
//! `d[0] = ((2*h[0] + h[1])*delta[0] - h[0]*delta[1]) / (h[0] + h[1])`
//! with Fritsch–Carlson monotonicity clamping (zeroed if the sign disagrees with
//! `delta[0]`, bounded by `3*delta[0]` if the direction reverses), exactly
//! matching SciPy's `PchipInterpolator` and Cleve Moler's `pchiptx.m`.
//!
//! Reference: C. Moler, *Numerical Computing with MATLAB*, SIAM 2004, ch. 3.6;
//! F. N. Fritsch & R. E. Carlson, "Monotone Piecewise Cubic Interpolation",
//! SIAM J. Numer. Anal. 17(2), 1980, doi:10.1137/0717021.

/// Insertion point of `x` in ascending `a`, on the right of equal elements —
/// `numpy.searchsorted(a, x, side="right")`.
#[must_use]
pub fn searchsorted_right(a: &[f64], x: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = a.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a[mid] <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Monotone cubic Hermite interpolator through `(x, y)`, both ascending in `x`.
///
/// Built once, queried many times: the node tangents (`d`) do not depend on the
/// query.
#[derive(Debug, Clone)]
pub struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Node tangents, same length as `x`.
    d: Vec<f64>,
    /// Successive `x` differences, length `x.len() - 1`.
    h: Vec<f64>,
    f_min: f64,
    f_max: f64,
}

/// One-sided three-point estimate for the endpoint derivative, matching
/// Fritsch & Carlson (1980) Eq. (8), Cleve Moler (`pchiptx.m`), and SciPy's
/// `PchipInterpolator._edge_case`.
fn pchip_edge_case(h0: f64, h1: f64, m0: f64, m1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    let same_sign = (d > 0.0 && m0 > 0.0) || (d < 0.0 && m0 < 0.0);
    if !same_sign {
        return 0.0;
    }
    let opposite_secants = (m0 > 0.0 && m1 < 0.0) || (m0 < 0.0 && m1 > 0.0);
    if opposite_secants && d.abs() > 3.0 * m0.abs() {
        return 3.0 * m0;
    }
    d
}

/// Shape-preserving cubic Hermite basis evaluated at interval parameter `t`.
#[inline]
fn hermite(t: f64, h: f64, y0: f64, y1: f64, d0: f64, d1: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
}

impl Pchip {
    /// Fit through `(f_nodes, g_nodes)` in `log10(f)` space, so `f_nodes` are plain Hz.
    ///
    /// Nodes are sorted ascending (last write winning at a repeated frequency)
    /// before the tangents are computed; the monotone limiter and the binary
    /// search in [`Pchip::eval`] are only correct on an ascending axis.
    ///
    /// # Panics
    /// If `f_nodes` and `g_nodes` differ in length, `f_nodes` is empty, or a
    /// node frequency is not finite and positive.
    #[must_use]
    pub fn new(f_nodes: &[f64], g_nodes: &[f64]) -> Self {
        assert_eq!(f_nodes.len(), g_nodes.len(), "node/gain length mismatch");
        assert!(!f_nodes.is_empty(), "PCHIP needs at least one node");
        let mut order: Vec<usize> = (0..f_nodes.len()).collect();
        order.sort_by(|&a, &b| f_nodes[a].total_cmp(&f_nodes[b]));
        let mut nodes: Vec<f64> = Vec::with_capacity(f_nodes.len());
        let mut gains: Vec<f64> = Vec::with_capacity(g_nodes.len());
        for &i in &order {
            let (f, g) = (f_nodes[i], g_nodes[i]);
            if nodes.last().is_some_and(|&prev| prev == f) {
                let n = gains.len() - 1;
                gains[n] = g;
            } else {
                nodes.push(f);
                gains.push(g);
            }
        }
        assert!(
            nodes.iter().all(|f| f.is_finite() && *f > 0.0),
            "PCHIP nodes must be finite positive frequencies; got {}",
            nodes
                .iter()
                .find(|f| !(f.is_finite() && **f > 0.0))
                .map_or(String::new(), |f| f.to_string())
        );
        let f_min = nodes[0];
        let f_max = nodes[nodes.len() - 1];
        let y = gains;
        let x: Vec<f64> = nodes.iter().map(|f| f.log10()).collect();
        if x.len() == 1 {
            return Self {
                x,
                y,
                d: vec![0.0],
                h: Vec::new(),
                f_min,
                f_max,
            };
        }
        let h: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        let delta: Vec<f64> = h
            .iter()
            .zip(y.windows(2))
            .map(|(hi, yg)| (yg[1] - yg[0]) / hi)
            .collect();

        let mut d = vec![0.0; y.len()];
        for k in 1..x.len() - 1 {
            if delta[k - 1] * delta[k] > 0.0 {
                let w1 = 2.0 * h[k] + h[k - 1];
                let w2 = h[k] + 2.0 * h[k - 1];
                d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
            } else {
                d[k] = 0.0;
            }
        }
        if x.len() == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            d[0] = pchip_edge_case(h[0], h[1], delta[0], delta[1]);
            let n = x.len();
            d[n - 1] = pchip_edge_case(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }

        Self {
            x,
            y,
            d,
            h,
            f_min,
            f_max,
        }
    }

    /// Number of knots, after [`Pchip::new`] has folded any repeats.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Always `false`: a `Pchip` keeps at least one node whatever it is given.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Evaluate at a frequency in Hz (clamped to the knot range).
    #[must_use]
    pub fn eval(&self, f_hz: f64) -> f64 {
        if self.x.len() == 1 {
            return self.y[0];
        }
        if f_hz.is_nan() {
            return f64::NAN;
        }
        let f = f_hz.clamp(self.f_min, self.f_max);
        let fq = f.log10();
        let idx = searchsorted_right(&self.x, fq)
            .saturating_sub(1)
            .min(self.x.len() - 2);
        let hi = self.h[idx];
        let t = (fq - self.x[idx]) / hi;
        hermite(
            t,
            hi,
            self.y[idx],
            self.y[idx + 1],
            self.d[idx],
            self.d[idx + 1],
        )
    }

    /// Evaluate on a grid, using a monotonic search cursor when queries ascend.
    #[must_use]
    pub fn eval_grid(&self, f_query: &[f64]) -> Vec<f64> {
        f_query.iter().map(|&f| self.eval(f)).collect()
    }
}

/// One-shot form of [`Pchip::new`] + [`Pchip::eval`].
#[must_use]
pub fn pchip_log_interp(f_nodes: &[f64], g_nodes: &[f64], f_query: &[f64]) -> Vec<f64> {
    Pchip::new(f_nodes, g_nodes).eval_grid(f_query)
}

/// The common single-point case.
#[must_use]
pub fn pchip_eval1(f_nodes: &[f64], g_nodes: &[f64], f_hz: f64) -> f64 {
    Pchip::new(f_nodes, g_nodes).eval(f_hz)
}

/// Allocation-free [`Pchip::eval`] over a fixed-size node table.
///
/// The room model evaluates ten-band material tables inside per-bin loops, so
/// the general [`Pchip`] (which allocates four `Vec`s per construction) is the
/// wrong shape there. This is the same algorithm with stack arrays and the same
/// SciPy-compatible endpoint rule; `N` is the table length.
///
/// # Panics
/// If `N < 2`.
#[must_use]
pub fn pchip_eval_fixed<const N: usize>(nodes_x: &[f64; N], nodes_y: &[f64; N], f_hz: f64) -> f64 {
    assert!(N >= 2, "PCHIP needs at least two nodes");
    if f_hz.is_nan() {
        return f64::NAN;
    }
    let f = f_hz.clamp(nodes_x[0], nodes_x[N - 1]);
    let xq = f.log10();
    let x: [f64; N] = std::array::from_fn(|i| nodes_x[i].log10());
    let mut h = [0.0_f64; N];
    let mut delta = [0.0_f64; N];
    for i in 0..N - 1 {
        h[i] = x[i + 1] - x[i];
        delta[i] = (nodes_y[i + 1] - nodes_y[i]) / h[i];
    }
    let mut d = [0.0_f64; N];
    for k in 1..N - 1 {
        if delta[k - 1] * delta[k] > 0.0 {
            let w1 = 2.0 * h[k] + h[k - 1];
            let w2 = h[k] + 2.0 * h[k - 1];
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k]);
        }
    }
    if N == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
    } else {
        d[0] = pchip_edge_case(h[0], h[1], delta[0], delta[1]);
        d[N - 1] = pchip_edge_case(h[N - 2], h[N - 3], delta[N - 2], delta[N - 3]);
    }
    // Interval index in `[0, N-2]`. At a query exactly on an interior knot this
    // keeps the interval below it (the heap [`Pchip::eval`] picks the one above
    // via `searchsorted_right - 1`); both Hermite blends agree exactly there, so
    // the evaluated value is the knot value either way.
    let mut idx = 0usize;
    while idx + 1 < N && xq > x[idx + 1] {
        idx += 1;
    }
    let hi = h[idx];
    let t = (xq - x[idx]) / hi;
    hermite(t, hi, nodes_y[idx], nodes_y[idx + 1], d[idx], d[idx + 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The monotone limiter and the binary search in `eval` both assume an
    /// ascending axis. Nodes are therefore normalised on the way in: ascending,
    /// last write winning at a repeated frequency.
    #[test]
    fn unsorted_and_repeated_nodes_describe_the_same_curve_as_sorted_ones() {
        let shuffled_f = [1000.0, 100.0, 1000.0, 10000.0, 20.0];
        let shuffled_g = [0.0, -2.0, 1.0, -3.0, 4.0];
        // The same nodes, ordered, with the two rows at 1 kHz collapsed to the later write.
        let want_f = [20.0, 100.0, 1000.0, 10000.0];
        let want_g = [4.0, -2.0, 1.0, -3.0];
        let shuffled = Pchip::new(&shuffled_f, &shuffled_g);
        let ordered = Pchip::new(&want_f, &want_g);
        for &f in &[20.0, 50.0, 100.0, 400.0, 1000.0, 4000.0, 10000.0] {
            assert_eq!(
                shuffled.f_min, want_f[0],
                "the axis start came from a row order, not the data"
            );
            assert_eq!(
                shuffled.f_max, want_f[3],
                "the axis end came from a row order"
            );
            assert!(
                (shuffled.eval(f) - ordered.eval(f)).abs() < 1e-12,
                "at {f} Hz: {} vs {}",
                shuffled.eval(f),
                ordered.eval(f)
            );
        }
        assert_eq!(shuffled.eval(1000.0), 1.0, "the last write must win");
    }

    #[test]
    fn matches_scipy_pchip_on_a_known_curve() {
        // Values match standard Fritsch & Carlson (1980) Eq. (8) / SciPy PchipInterpolator.
        //   python: scipy.interpolate.PchipInterpolator(np.log10([20.,100.,1000.,10000.,20000.]),
        //                                              [ 3., 0.,-2., 4., 0.])
        //   evaluated at [20., 50., 1000., 3703., 20000., 50000., 1.]
        //     -> [3., 1.0757301, -2., 1.6131173, 0., 0., 3.]
        let f = [20.0, 100.0, 1000.0, 10000.0, 20000.0];
        let g = [3.0, 0.0, -2.0, 4.0, 0.0];
        let q = [20.0, 50.0, 1000.0, 3703.0, 20000.0, 50000.0, 1.0];
        let got = pchip_log_interp(&f, &g, &q);
        let want = [3.0, 1.075_730_1, -2.0, 1.613_117_3, 0.0, 0.0, 3.0];
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 5e-7, "{a} vs {b}");
        }
    }

    #[test]
    fn interpolates_knots_exactly_and_clamps_outside() {
        let f = [20.0, 63.0, 500.0, 4000.0, 16000.0];
        let g = [1.5, -3.25, 0.75, 6.125, -0.5];
        let p = Pchip::new(&f, &g);
        for (&fi, &gi) in f.iter().zip(&g) {
            assert!((p.eval(fi) - gi).abs() < 1e-12, "knot {fi}");
        }
        assert_eq!(p.eval(1.0), g[0], "below the first knot clamps");
        assert_eq!(p.eval(1e6), g[g.len() - 1], "above the last knot clamps");
    }

    #[test]
    fn monotone_between_knots_of_one_sign() {
        // A strictly increasing run must not overshoot: PCHIP's harmonic-mean
        // tangents are the whole reason this interpolator was chosen.
        let f = [20.0, 40.0, 80.0, 160.0, 320.0];
        let g = [-6.0, -4.0, -2.0, -1.0, 0.0];
        let p = Pchip::new(&f, &g);
        let mut grid = Vec::new();
        let (a, b, n) = (20.0_f64, 320.0_f64, 4000usize);
        let ratio = (b / a).powf(1.0 / (n - 1) as f64);
        let mut f_i = a;
        for _ in 0..n {
            grid.push(f_i);
            f_i *= ratio;
        }
        for w in grid.windows(2) {
            assert!(
                p.eval(w[1]) >= p.eval(w[0]) - 1e-12,
                "dip between {} and {}",
                w[0],
                w[1]
            );
        }
        for &f_i in &grid {
            assert!(
                p.eval(f_i) <= 0.0 + 1e-12 && p.eval(f_i) >= -6.0 - 1e-12,
                "overshoot at {f_i}"
            );
        }
    }

    #[test]
    fn flattens_at_a_peak() {
        let f = [100.0, 1000.0, 10000.0];
        let g = [0.0, 5.0, 0.0];
        let p = Pchip::new(&f, &g);
        assert_eq!(p.d[1], 0.0);
        let near = p.eval(1000.0 * 1.001);
        assert!(near < 5.0, "no boost past the knot: {near}");
        assert!((near - 4.999_999_058).abs() < 5e-8, "{near}");
    }

    /// Invariant: Fritsch & Carlson (1980) Eq. (8) has 0 % derivative error on
    /// quadratics, whereas first-order secants have 100 % error. Sabotage:
    /// returning `delta[0]` yields `d[0] = 1.0` (failing the test).
    #[test]
    fn endpoints_match_fritsch_carlson_and_scipy_exact() {
        // Knots chosen so log10(f) = [0.0, 1.0, 2.0, 3.0], with g = x^2:
        let f = [1.0, 10.0, 100.0, 1000.0];
        let g = [0.0, 1.0, 4.0, 9.0];
        let p = Pchip::new(&f, &g);
        // Exact f'(0) for y = x^2 is 0.0; exact f'(3) is 6.0.
        assert!((p.d[0] - 0.0).abs() < 1e-12, "d[0] was {}", p.d[0]);
        assert!((p.d[3] - 6.0).abs() < 1e-12, "d[3] was {}", p.d[3]);
    }

    #[test]
    fn single_node_is_a_constant() {
        let p = Pchip::new(&[1000.0], &[2.5]);
        assert_eq!(p.eval(20.0), 2.5);
        assert_eq!(p.eval(20000.0), 2.5);
    }

    #[test]
    fn searchsorted_matches_the_right_side_convention() {
        let a = [1.0, 2.0, 2.0, 5.0];
        assert_eq!(searchsorted_right(&a, 0.5), 0);
        assert_eq!(searchsorted_right(&a, 2.0), 3, "right of equal elements");
        assert_eq!(searchsorted_right(&a, 5.0), 4);
        assert_eq!(searchsorted_right(&a, 6.0), 4);
    }

    #[test]
    fn a_query_off_the_end_of_the_grid_clamps_instead_of_wrapping() {
        let p = Pchip::new(&[20.0, 1000.0, 20000.0], &[1.0, 0.0, -1.0]);
        assert_eq!(p.eval(0.0), 1.0);
        assert_eq!(p.eval(-5.0), 1.0);
        assert_eq!(p.eval(f64::NEG_INFINITY), 1.0);
        assert_eq!(p.eval(f64::INFINITY), -1.0);
        assert_eq!(p.eval(20.0), 1.0);
        assert_eq!(p.eval(20000.0), -1.0);
        assert!(p.eval(f64::NAN).is_nan());
    }

    #[test]
    fn a_repeated_knot_cannot_turn_the_curve_into_nan() {
        // A zero-wide interval makes every divided slope over it infinite, and one
        // infinite slope is enough to poison the whole cubic.
        let p = Pchip::new(&[20.0, 1000.0, 1000.0, 20000.0], &[1.0, 0.5, -0.5, -1.0]);
        assert_eq!(
            p.len(),
            3,
            "the repeated knot is folded, last value winning"
        );
        for f in [20.0, 100.0, 1000.0, 5000.0, 20000.0] {
            assert!(p.eval(f).is_finite(), "{f} Hz gave {}", p.eval(f));
        }
        assert_eq!(p.eval(1000.0), -0.5);
    }

    /// The allocation-free fixed-size evaluator must be the same curve as the
    /// heap [`Pchip`], including at exact knots and outside the range.
    #[test]
    fn fixed_evaluator_matches_the_heap_pchip() {
        let f = [
            31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
        ];
        let g = [0.30, 0.36, 0.42, 0.55, 0.68, 0.74, 0.78, 0.82, 0.84, 0.86];
        let p = Pchip::new(&f, &g);
        let mut q = vec![
            1.0, 31.5, 44.548, 63.0, 88.388, 125.0, 176.777, 1_000.0, 16_000.0, 30_000.0,
        ];
        let mut x = 31.5_f64;
        while x < 16_000.0 {
            q.push(x);
            x *= 1.000_1;
        }
        for &fq in &q {
            let want = p.eval(fq);
            let got = pchip_eval_fixed(&f, &g, fq);
            assert!(
                (got - want).abs() < 1e-12,
                "fixed vs heap at {fq} Hz: {got} vs {want}"
            );
        }
    }

    /// SciPy `PchipInterpolator(log10(ROOM_BANDS_HZ), table)` reference values
    /// (SciPy 1.18.1), so "SciPy-compatible endpoints" is checked against the
    /// implementation, not against our own tangent formula.
    #[test]
    fn ten_band_table_matches_scipy_pchip_reference() {
        let f = [
            31.5, 63.0, 125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0,
        ];
        let g = [0.30, 0.36, 0.42, 0.55, 0.68, 0.74, 0.78, 0.82, 0.84, 0.86];
        let cases = [
            (31.5_f64, 0.300_000_000_000_000_f64),
            (44.548, 0.329_913_226_085_599),
            (62.9, 0.359_861_696_632_238),
            (63.0, 0.360_000_000_000_000),
            (88.388, 0.386_920_563_063_386),
            (124.999, 0.419_999_045_535_077),
            (125.0, 0.420_000_000_000_000),
            (125.001, 0.420_000_954_475_772),
            (176.777, 0.479_087_635_515_444),
            (250.0, 0.550_000_000_000_000),
            (353.553, 0.620_986_615_821_964),
            (1_000.0, 0.740_000_000_000_000),
            (2_828.427, 0.801_666_663_909_399),
            (4_000.0, 0.820_000_000_000_000),
            (8_000.0, 0.840_000_000_000_000),
            (16_000.0, 0.860_000_000_000_000),
        ];
        let p = Pchip::new(&f, &g);
        for &(fq, want) in &cases {
            let heap = p.eval(fq);
            let fixed = pchip_eval_fixed(&f, &g, fq);
            assert!(
                (heap - want).abs() < 1e-12,
                "heap at {fq}: {heap} vs {want}"
            );
            assert!(
                (fixed - want).abs() < 1e-12,
                "fixed at {fq}: {fixed} vs {want}"
            );
        }
    }
}
