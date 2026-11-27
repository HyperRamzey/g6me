// Zero-FMA FP64 Stockham Radix-2 Complex FFT & Pointwise Spectrum Multiply Kernels.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
// Strictly uses IEEE-754 __dadd_rn / __dsub_rn / __dmul_rn (zero DFMA/DMAD contractions).

extern "C" __global__ void g6me_stockham_radix2_f64_stage(
    const double* __restrict__ in_re,
    const double* __restrict__ in_im,
    const double* __restrict__ tw_re,
    const double* __restrict__ tw_im,
    double* __restrict__ out_re,
    double* __restrict__ out_im,
    int half_n,
    int s,
    int m
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    int q = tid / s;
    int p = tid - q * s;
    int in_a = p + s * q;
    int in_b = in_a + s * m;
    int out_a = p + s * (2 * q);
    int out_b = out_a + s;

    double ar = in_re[in_a];
    double ai = in_im[in_a];
    double br = in_re[in_b];
    double bi = in_im[in_b];

    double wr = tw_re[q];
    double wi = tw_im[q];

    // Strict zero-FMA complex butterfly: u = a + b, v = (a - b) * w
    double ur = __dadd_rn(ar, br);
    double ui = __dadd_rn(ai, bi);
    double dr = __dsub_rn(ar, br);
    double di = __dsub_rn(ai, bi);

    double vr = __dsub_rn(__dmul_rn(dr, wr), __dmul_rn(di, wi));
    double vi = __dadd_rn(__dmul_rn(dr, wi), __dmul_rn(di, wr));

    out_re[out_a] = ur;
    out_im[out_a] = ui;
    out_re[out_b] = vr;
    out_im[out_b] = vi;
}

extern "C" __global__ void g6me_complex_mac_f64(
    const double* __restrict__ a_re,
    const double* __restrict__ a_im,
    const double* __restrict__ b_re,
    const double* __restrict__ b_im,
    double* __restrict__ acc_re,
    double* __restrict__ acc_im,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ar = a_re[i], ai = a_im[i];
    double br = b_re[i], bi = b_im[i];
    double pr = __dsub_rn(__dmul_rn(ar, br), __dmul_rn(ai, bi));
    double pi = __dadd_rn(__dmul_rn(ar, bi), __dmul_rn(ai, br));
    acc_re[i] = __dadd_rn(acc_re[i], pr);
    acc_im[i] = __dadd_rn(acc_im[i], pi);
}
