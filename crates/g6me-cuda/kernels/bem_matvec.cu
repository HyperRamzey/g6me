// Zero-FMA Complex Helmholtz FMM-BEM Dense/Near-Field Matrix-Vector Product Kernel.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
// Strictly uses IEEE-754 __dadd_rn / __dsub_rn / __dmul_rn (zero DFMA/DMAD contractions).

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
