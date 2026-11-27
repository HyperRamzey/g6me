// Zero-FMA 3D Standard Rectilinear (SRL) FDTD Acoustic Wave Solver Kernel for G6ME.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
// Strictly uses IEEE-754 __dadd_rn / __dmul_rn (zero DFMA/DMAD contractions).

extern "C" __global__ void g6me_fdtd_3d_srl_step(
    const double* __restrict__ p_prev,
    const double* __restrict__ p_curr,
    double* __restrict__ p_next,
    const double* __restrict__ inv_den,
    int nx,
    int ny,
    int nz,
    double lam2,
    double half_lam_beta
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;

    int s_x = ny * nz;
    int id = i * s_x + j * nz + k;
    double pc = p_curr[id];
    double lap = 0.0;
    int kb = 0;

    if (i > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - s_x], -pc)); else kb++;
    if (i + 1 < nx) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + s_x], -pc)); else kb++;
    if (j > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - nz],  -pc)); else kb++;
    if (j + 1 < ny) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + nz],  -pc)); else kb++;
    if (k > 0)      lap = __dadd_rn(lap, __dadd_rn(p_curr[id - 1],   -pc)); else kb++;
    if (k + 1 < nz) lap = __dadd_rn(lap, __dadd_rn(p_curr[id + 1],   -pc)); else kb++;

    double two_pc = __dmul_rn(2.0, pc);
    double lam_lap = __dmul_rn(lam2, lap);
    if (kb == 0) {
        p_next[id] = __dadd_rn(__dadd_rn(two_pc, -p_prev[id]), lam_lap);
    } else {
        double g = __dmul_rn(half_lam_beta, (double)kb);
        double num = __dadd_rn(
            __dadd_rn(two_pc, lam_lap),
            -__dmul_rn(__dadd_rn(1.0, -g), p_prev[id])
        );
        // Host-precomputed 1.0 / (1.0 + half_lam_beta * kb) for kb = 0..6 avoids FP division.
        p_next[id] = __dmul_rn(num, inv_den[kb]);
    }
}

extern "C" __global__ void g6me_fdtd_record_receivers(
    const double* __restrict__ p_curr,
    const int* __restrict__ rec_ids,
    double* __restrict__ rec_out,
    int n_rec,
    int step
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rec) return;
    rec_out[step * n_rec + r] = p_curr[rec_ids[r]];
}

