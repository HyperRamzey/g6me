// Zero-FMA Complex Helmholtz FMM-BEM kernels.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
// Strictly uses IEEE-754 __dadd_rn / __dsub_rn / __dmul_rn (zero DFMA/DMAD contractions).
// Device code performs no libm sin/cos, no sqrt and no IEEE division: every transcendental
// or reciprocal coefficient is precomputed on the host once per `(solver, k, adjoint)` and
// the solid-angle assembly uses a bit-hack + Newton reciprocal square root built only from
// the ordered multiply/add intrinsics, so `ptxas` has no MUFU+DFMA expansion to emit.

extern "C" __global__ void g6me_bem_complex_matvec(
    const double* __restrict__ mat_re, // [n * n]
    const double* __restrict__ mat_im, // [n * n]
    const double* __restrict__ x_re,   // [n]
    const double* __restrict__ x_im,   // [n]
    double* __restrict__ y_re,         // [n]
    double* __restrict__ y_im,         // [n]
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double sum_re = 0.0;
    double sum_im = 0.0;
    int base = row * n;
    for (int col = 0; col < n; ++col) {
        double ar = mat_re[base + col];
        double ai = mat_im[base + col];
        double xr = x_re[col];
        double xi = x_im[col];
        double pr = __dsub_rn(__dmul_rn(ar, xr), __dmul_rn(ai, xi));
        double pi = __dadd_rn(__dmul_rn(ar, xi), __dmul_rn(ai, xr));
        sum_re = __dadd_rn(sum_re, pr);
        sum_im = __dadd_rn(sum_im, pi);
    }
    y_re[row] = sum_re;
    y_im[row] = sum_im;
}

// ---------------------------------------------------------------------------
// FP64 reciprocal square root from ordered multiply/add only.
//
// `1/sqrt(a)` with the classic double bit-hack seed (~3.4% relative error)
// followed by five Newton steps (`y <- y * (1.5 - 0.5*a*y*y)`), quadratic
// convergence drives the error to the double rounding floor. No division, no
// sqrt and therefore no ptxas MUFU+DFMA expansion; `r = a * rsqrt(a)`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double g6me_bem_rsqrt(double a) {
    double y = __longlong_as_double(
        0x5FE6EB50C7B537A9LL - (__double_as_longlong(a) >> 1));
#pragma unroll
    for (int it = 0; it < 5; ++it) {
        double yy = __dmul_rn(y, y);
        double t = __dmul_rn(__dmul_rn(-0.5, a), yy);
        t = __dadd_rn(1.5, t);
        y = __dmul_rn(y, t);
    }
    return y;
}

// ---------------------------------------------------------------------------
// Exact closed-surface static solid-angle diagonal
// `A_ii = 1 + sum_{j != i} (-((y_j - x_i) . n_j) / (4 pi r^3)) * area_j`.
// One thread per row, `j` ascending exactly like the CPU `par_map_indexed`
// reference, so the accumulation order (and every addend) matches.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_solid_angle_diag(
    const double* __restrict__ cen,  // [3n] interleaved centroids
    const double* __restrict__ nrm,  // [3n] interleaved outward normals
    const double* __restrict__ area, // [n]
    double* __restrict__ diag,       // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double px = cen[3 * i + 0];
    double py = cen[3 * i + 1];
    double pz = cen[3 * i + 2];
    const double four_pi = 12.566370614359172953850573533118;
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        int j3 = 3 * j;
        double dx = __dsub_rn(cen[j3 + 0], px);
        double dy = __dsub_rn(cen[j3 + 1], py);
        double dz = __dsub_rn(cen[j3 + 2], pz);
        double r2 = __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)),
                              __dmul_rn(dz, dz));
        double r = __dmul_rn(r2, g6me_bem_rsqrt(r2));
        r = (r > 1e-12) ? r : 1e-12;
        double q = __dmul_rn(__dmul_rn(__dmul_rn(four_pi, r), r), r);
        double dot = __dadd_rn(__dadd_rn(__dmul_rn(dx, nrm[j3 + 0]),
                                         __dmul_rn(dy, nrm[j3 + 1])),
                               __dmul_rn(dz, nrm[j3 + 2]));
        double rsq = g6me_bem_rsqrt(q);
        double inv_q = __dmul_rn(rsq, rsq);
        double term = __dmul_rn(__dmul_rn(-dot, inv_q), area[j]);
        sum = __dadd_rn(sum, term);
    }
    diag[i] = __dadd_rn(1.0, sum);
}

// ---------------------------------------------------------------------------
// FMM upward pass (P2M / M2M), one kernel launch per octree depth level,
// deepest level first. One thread per node; a leaf sums its source list in
// stored order and an internal node sums its children in stored order, exactly
// reproducing the CPU `matvec_fmm` moment recursion.
// `mom` layout: [6*node + 2*axis + {0:re, 1:im}] for axis = 0,1,2.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_fmm_upward(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const double* __restrict__ src_scale, // [n] = -area
    const double* __restrict__ src_nx,    // [n]
    const double* __restrict__ src_ny,
    const double* __restrict__ src_nz,
    const int* __restrict__ child_start,  // [node_count + 1]
    const int* __restrict__ children,     // [children.len()]
    const int* __restrict__ leaf_start,   // [node_count + 1]
    const int* __restrict__ leaf_src,     // [leaf_src.len()]
    const int* __restrict__ level_nodes,  // [node_count]
    int level_begin,
    int level_count,
    double* __restrict__ mom
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= level_count) return;
    int nid = level_nodes[level_begin + t];
    double m0r = 0.0, m0i = 0.0, m1r = 0.0, m1i = 0.0, m2r = 0.0, m2i = 0.0;
    int cs = child_start[nid];
    int ce = child_start[nid + 1];
    if (cs == ce) {
        for (int e = leaf_start[nid]; e < leaf_start[nid + 1]; ++e) {
            int j = leaf_src[e];
            double wr = __dmul_rn(x_re[j], src_scale[j]);
            double wi = __dmul_rn(x_im[j], src_scale[j]);
            m0r = __dadd_rn(m0r, __dmul_rn(wr, src_nx[j]));
            m0i = __dadd_rn(m0i, __dmul_rn(wi, src_nx[j]));
            m1r = __dadd_rn(m1r, __dmul_rn(wr, src_ny[j]));
            m1i = __dadd_rn(m1i, __dmul_rn(wi, src_ny[j]));
            m2r = __dadd_rn(m2r, __dmul_rn(wr, src_nz[j]));
            m2i = __dadd_rn(m2i, __dmul_rn(wi, src_nz[j]));
        }
    } else {
        for (int c = cs; c < ce; ++c) {
            const double* m = mom + 6 * children[c];
            m0r = __dadd_rn(m0r, m[0]);
            m0i = __dadd_rn(m0i, m[1]);
            m1r = __dadd_rn(m1r, m[2]);
            m1i = __dadd_rn(m1i, m[3]);
            m2r = __dadd_rn(m2r, m[4]);
            m2i = __dadd_rn(m2i, m[5]);
        }
    }
    double* out = mom + 6 * nid;
    out[0] = m0r;
    out[1] = m0i;
    out[2] = m1r;
    out[3] = m1i;
    out[4] = m2r;
    out[5] = m2i;
}

// ---------------------------------------------------------------------------
// FMM forward M2P + near-field pass: one thread per target element, walking a
// single per-target interaction list in CPU DFS order. Entries are tagged
// 0 = near (index is a source element, a/b are the exact host coefficient) and
// 1 = far (index is an octree node, a/b are the exact host cluster scalar and
// d = node.center - x_target). The dot product and complex products replicate
// the CPU association order.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_fmm_forward(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const double* __restrict__ diag,
    const int* __restrict__ entry_start, // [n + 1]
    const int* __restrict__ entry_kind,  // [entries]
    const int* __restrict__ entry_index, // [entries]
    const double* __restrict__ ea,       // [entries] near coeff.re / far scalar.re
    const double* __restrict__ eb,       // [entries] near coeff.im / far scalar.im
    const double* __restrict__ edx,      // [entries] far d.x
    const double* __restrict__ edy,
    const double* __restrict__ edz,
    const double* __restrict__ mom,
    double* __restrict__ y_re,
    double* __restrict__ y_im,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double acc_r = __dmul_rn(x_re[i], diag[i]);
    double acc_i = __dmul_rn(x_im[i], diag[i]);
    for (int e = entry_start[i]; e < entry_start[i + 1]; ++e) {
        int kind = entry_kind[e];
        int idx = entry_index[e];
        if (kind == 0) {
            double cr = ea[e];
            double ci = eb[e];
            double pr = __dsub_rn(__dmul_rn(cr, x_re[idx]), __dmul_rn(ci, x_im[idx]));
            double pi = __dadd_rn(__dmul_rn(cr, x_im[idx]), __dmul_rn(ci, x_re[idx]));
            acc_r = __dadd_rn(acc_r, pr);
            acc_i = __dadd_rn(acc_i, pi);
        } else {
            const double* m = mom + 6 * idx;
            double dr = __dadd_rn(__dmul_rn(m[0], edx[e]), __dmul_rn(m[2], edy[e]));
            dr = __dadd_rn(dr, __dmul_rn(m[4], edz[e]));
            double di = __dadd_rn(__dmul_rn(m[1], edx[e]), __dmul_rn(m[3], edy[e]));
            di = __dadd_rn(di, __dmul_rn(m[5], edz[e]));
            double sr = ea[e];
            double si = eb[e];
            double pr = __dsub_rn(__dmul_rn(sr, dr), __dmul_rn(si, di));
            double pi = __dadd_rn(__dmul_rn(sr, di), __dmul_rn(si, dr));
            acc_r = __dadd_rn(acc_r, pr);
            acc_i = __dadd_rn(acc_i, pi);
        }
    }
    y_re[i] = acc_r;
    y_im[i] = acc_i;
}

// ---------------------------------------------------------------------------
// FMM exact-adjoint local field accumulation: for every octree node, sum the
// far-field adjoint gradient contributions of its incoming targets in ascending
// target order (the order the CPU `matvec_fmm_adjoint` outer loop accumulates
// them in). `grad` layout matches `mom`.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_fmm_adjoint_nodes(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const int* __restrict__ entry_start, // [node_count + 1]
    const int* __restrict__ entry_index, // [entries] target element i
    const double* __restrict__ ea,       // [entries] scalar.re
    const double* __restrict__ eb,       // [entries] scalar.im
    const double* __restrict__ edx,      // [entries] d.x
    const double* __restrict__ edy,
    const double* __restrict__ edz,
    double* __restrict__ grad,
    int node_count
) {
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= node_count) return;
    double g0r = 0.0, g0i = 0.0, g1r = 0.0, g1i = 0.0, g2r = 0.0, g2i = 0.0;
    for (int e = entry_start[nid]; e < entry_start[nid + 1]; ++e) {
        int i = entry_index[e];
        double sr = ea[e];
        double si = eb[e];
        double tr = __dsub_rn(__dmul_rn(sr, x_re[i]), __dmul_rn(si, x_im[i]));
        double ti = __dadd_rn(__dmul_rn(sr, x_im[i]), __dmul_rn(si, x_re[i]));
        g0r = __dadd_rn(g0r, __dmul_rn(tr, edx[e]));
        g0i = __dadd_rn(g0i, __dmul_rn(ti, edx[e]));
        g1r = __dadd_rn(g1r, __dmul_rn(tr, edy[e]));
        g1i = __dadd_rn(g1i, __dmul_rn(ti, edy[e]));
        g2r = __dadd_rn(g2r, __dmul_rn(tr, edz[e]));
        g2i = __dadd_rn(g2i, __dmul_rn(ti, edz[e]));
    }
    double* out = grad + 6 * nid;
    out[0] = g0r;
    out[1] = g0i;
    out[2] = g1r;
    out[3] = g1i;
    out[4] = g2r;
    out[5] = g2i;
}

// ---------------------------------------------------------------------------
// FMM exact-adjoint downward pass (L2L / L2P push), one kernel launch per level,
// root first: each parent adds its local field to every child. Children have a
// unique parent, so the reduction stays deterministic with no atomics.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_fmm_adjoint_push(
    const int* __restrict__ child_start,
    const int* __restrict__ children,
    const int* __restrict__ level_nodes,
    int level_begin,
    int level_count,
    double* __restrict__ grad
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= level_count) return;
    int nid = level_nodes[level_begin + t];
    const double* g = grad + 6 * nid;
    double g0r = g[0], g0i = g[1], g1r = g[2], g1i = g[3], g2r = g[4], g2i = g[5];
    for (int c = child_start[nid]; c < child_start[nid + 1]; ++c) {
        double* gc = grad + 6 * children[c];
        gc[0] = __dadd_rn(gc[0], g0r);
        gc[1] = __dadd_rn(gc[1], g0i);
        gc[2] = __dadd_rn(gc[2], g1r);
        gc[3] = __dadd_rn(gc[3], g1i);
        gc[4] = __dadd_rn(gc[4], g2r);
        gc[5] = __dadd_rn(gc[5], g2i);
    }
}

// ---------------------------------------------------------------------------
// FMM exact-adjoint source pass: one thread per source element j. Starts from
// the diagonal term, adds near-field coefficients in ascending target order,
// then the leaf L2P push `((g0*n0 + g1*n1) + g2*n2) * (-area_j)` — the exact
// arithmetic sequence of the CPU `matvec_fmm_adjoint`.
// ---------------------------------------------------------------------------
extern "C" __global__ void g6me_bem_fmm_adjoint_sources(
    const double* __restrict__ x_re,
    const double* __restrict__ x_im,
    const double* __restrict__ diag,
    const int* __restrict__ entry_start, // [n + 1]
    const int* __restrict__ entry_index, // [entries] target element i
    const double* __restrict__ ea,       // [entries] coeff.re
    const double* __restrict__ eb,       // [entries] coeff.im
    const double* __restrict__ src_scale, // [n] = -area
    const double* __restrict__ src_nx,
    const double* __restrict__ src_ny,
    const double* __restrict__ src_nz,
    const int* __restrict__ src_leaf,    // [n]
    const double* __restrict__ grad,
    double* __restrict__ y_re,
    double* __restrict__ y_im,
    int n
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double acc_r = __dmul_rn(x_re[j], diag[j]);
    double acc_i = __dmul_rn(x_im[j], diag[j]);
    for (int e = entry_start[j]; e < entry_start[j + 1]; ++e) {
        int i = entry_index[e];
        double cr = ea[e];
        double ci = eb[e];
        double pr = __dsub_rn(__dmul_rn(cr, x_re[i]), __dmul_rn(ci, x_im[i]));
        double pi = __dadd_rn(__dmul_rn(cr, x_im[i]), __dmul_rn(ci, x_re[i]));
        acc_r = __dadd_rn(acc_r, pr);
        acc_i = __dadd_rn(acc_i, pi);
    }
    const double* g = grad + 6 * src_leaf[j];
    double pr = __dadd_rn(__dmul_rn(g[0], src_nx[j]), __dmul_rn(g[2], src_ny[j]));
    pr = __dadd_rn(pr, __dmul_rn(g[4], src_nz[j]));
    double pi = __dadd_rn(__dmul_rn(g[1], src_nx[j]), __dmul_rn(g[3], src_ny[j]));
    pi = __dadd_rn(pi, __dmul_rn(g[5], src_nz[j]));
    acc_r = __dadd_rn(acc_r, __dmul_rn(pr, src_scale[j]));
    acc_i = __dadd_rn(acc_i, __dmul_rn(pi, src_scale[j]));
    y_re[j] = acc_r;
    y_im[j] = acc_i;
}
