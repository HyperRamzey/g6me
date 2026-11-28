// Zero-FMA Stochastic Monte Carlo Ray-Tracing -> Order-7 HOA Energy Histogram Kernel.
// Compiled with: nvcc --fmad=false -Xcompiler -ffp-contract=off -O3
// Deterministic per-ray PCG64 stream with IEEE-754 __dadd_rn / __dmul_rn. No libm
// transcendentals: the uniform-sphere launch and the cosine-weighted Lambertian wall
// scatter both use Marsaglia (1972) rejection on the unit disk, so the PTX carries zero
// fma.rn. IEEE `sqrt.rn.f64` / `div.rn.f64` remain; ptxas lowers those to
// MUFU + DFMA microcode, which is the hardware implementation of one correctly-rounded
// operation (bit-identical to CPU f64), not a contraction of this kernel's arithmetic.
//
// The CPU tracer in `g6me-core/src/room.rs` (`RayPcg64`, `trace_single_ray`) mirrors every
// floating-point operation, PCG64 draw and branch below line for line, and both sides feed
// one shared host accumulator (`accumulate_hoa_hit`), so CPU and GPU produce bit-identical
// (delay, energy, direction) records and therefore bit-identical HOA histograms.
//
// Wall indexing is the CPU `RoomGeometry` ordering:
//   0 = +x (front), 1 = -x (back), 2 = +y (left), 3 = -y (right), 4 = -z (floor),
//   5 = +z (ceiling)

#include <stdint.h>

__device__ __forceinline__ uint64_t pcg64_next(uint64_t* state, uint64_t inc) {
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (uint64_t)((xorshifted >> rot) | (xorshifted << ((-rot) & 31u)));
}

__device__ __forceinline__ double pcg64_uniform(uint64_t* state, uint64_t inc) {
    uint64_t hi = pcg64_next(state, inc);
    uint64_t lo = pcg64_next(state, inc);
    uint64_t bits = ((hi & 0x1FFFFFULL) << 32) | lo;
    return __dmul_rn((double)bits, 1.1102230246251565e-16); // 2^-53
}

extern "C" __global__ void g6me_raytrace_hoa_pass(
    const double* __restrict__ room_dims,    // [Lx, Ly, Lz]
    const double* __restrict__ src_pos,      // [sx, sy, sz]
    const double* __restrict__ lst_pos,      // [rx, ry, rz]
    const double* __restrict__ wall_refl,    // [6] energy reflection coefficient per wall
    const double* __restrict__ wall_scatter, // [6] ISO-17497 scattering coefficient per wall
    double* __restrict__ ray_delays,         // [n_rays * max_bounces] arrival time (s), -1 = none
    double* __restrict__ ray_energies,       // [n_rays * max_bounces] ray energy, 0 = none
    double* __restrict__ ray_dirs,           // [n_rays * max_bounces * 3] arrival dir (=-ray dir)
    int n_rays,
    int max_bounces,
    double c_air,
    double det_radius,
    uint64_t base_seed
) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= n_rays) return;

    uint64_t inc = ((uint64_t)ray_id << 1) | 1ULL;
    uint64_t state = base_seed ^ ((uint64_t)(ray_id + 1) * 0x9E3779B97F4A7C15ULL);
    (void)pcg64_next(&state, inc);

    // Uniform sphere launch direction via Marsaglia (1972) rejection on the unit disk.
    // Each ray owns an independent PCG64 stream, so the variable draw count stays
    // deterministic per ray.
    double ux, uy, s;
    do {
        ux = __dadd_rn(__dmul_rn(2.0, pcg64_uniform(&state, inc)), -1.0);
        uy = __dadd_rn(__dmul_rn(2.0, pcg64_uniform(&state, inc)), -1.0);
        s = __dadd_rn(__dmul_rn(ux, ux), __dmul_rn(uy, uy));
    } while (s >= 1.0 || s == 0.0);
    double f = __dmul_rn(2.0, sqrt(__dadd_rn(1.0, -s)));
    double dx = __dmul_rn(ux, f);
    double dy = __dmul_rn(uy, f);
    double dz = __dadd_rn(1.0, -__dmul_rn(2.0, s));

    double px = src_pos[0], py = src_pos[1], pz = src_pos[2];
    double lx = room_dims[0], ly = room_dims[1], lz = room_dims[2];
    double rx = lst_pos[0], ry = lst_pos[1], rz = lst_pos[2];
    double det_r2 = __dmul_rn(det_radius, det_radius);

    double energy = __ddiv_rn(1.0, (double)n_rays);
    double energy_floor = __dmul_rn(1e-7, energy);
    double total_dist = 0.0;

    for (int b = 0; b < max_bounces; ++b) {
        int out_idx = ray_id * max_bounces + b;
        ray_delays[out_idx] = -1.0;
        ray_energies[out_idx] = 0.0;

        // Intersect ray with 6 axis-aligned shoebox walls
        double tx = (dx > 1e-12) ? __ddiv_rn(__dadd_rn(lx, -px), dx)
                  : (dx < -1e-12) ? __ddiv_rn(-px, dx) : 1e30;
        double ty = (dy > 1e-12) ? __ddiv_rn(__dadd_rn(ly, -py), dy)
                  : (dy < -1e-12) ? __ddiv_rn(-py, dy) : 1e30;
        double tz = (dz > 1e-12) ? __ddiv_rn(__dadd_rn(lz, -pz), dz)
                  : (dz < -1e-12) ? __ddiv_rn(-pz, dz) : 1e30;

        double t_hit = tx;
        int wall = (dx > 0.0) ? 0 : 1;
        if (ty < t_hit) { t_hit = ty; wall = (dy > 0.0) ? 2 : 3; }
        if (tz < t_hit) { t_hit = tz; wall = (dz > 0.0) ? 5 : 4; }
        if (t_hit <= 1e-9 || t_hit > 1e20) break;

        // Check receiver sphere intersection along segment [0, t_hit]
        double ocx = __dadd_rn(rx, -px);
        double ocy = __dadd_rn(ry, -py);
        double ocz = __dadd_rn(rz, -pz);
        double proj = __dadd_rn(__dadd_rn(__dmul_rn(ocx, dx), __dmul_rn(ocy, dy)), __dmul_rn(ocz, dz));
        if (proj > 0.0 && proj < t_hit) {
            double cx = __dadd_rn(ocx, -__dmul_rn(proj, dx));
            double cy = __dadd_rn(ocy, -__dmul_rn(proj, dy));
            double cz = __dadd_rn(ocz, -__dmul_rn(proj, dz));
            double d2 = __dadd_rn(__dadd_rn(__dmul_rn(cx, cx), __dmul_rn(cy, cy)), __dmul_rn(cz, cz));
            if (d2 <= det_r2) {
                ray_delays[out_idx] = __ddiv_rn(__dadd_rn(total_dist, proj), c_air);
                ray_energies[out_idx] = energy;
                ray_dirs[out_idx * 3 + 0] = -dx;
                ray_dirs[out_idx * 3 + 1] = -dy;
                ray_dirs[out_idx * 3 + 2] = -dz;
            }
        }

        // Advance to wall (clamped exactly like the CPU tracer) and apply mid-band absorption
        px = fmin(fmax(__dadd_rn(px, __dmul_rn(t_hit, dx)), 1e-6), __dadd_rn(lx, -1e-6));
        py = fmin(fmax(__dadd_rn(py, __dmul_rn(t_hit, dy)), 1e-6), __dadd_rn(ly, -1e-6));
        pz = fmin(fmax(__dadd_rn(pz, __dmul_rn(t_hit, dz)), 1e-6), __dadd_rn(lz, -1e-6));
        total_dist = __dadd_rn(total_dist, t_hit);
        energy = __dmul_rn(energy, wall_refl[wall]);

        // Specular reflection (sign flip), then the shared energy-floor break
        if (wall == 0 || wall == 1) dx = -dx;
        else if (wall == 2 || wall == 3) dy = -dy;
        else dz = -dz;

        if (energy < energy_floor) break;

        // ISO-17497 Lambertian diffuse scattering blend. Cosine-weighted hemisphere via
        // Marsaglia unit-disk rejection plus the Lambert equal-area lift z = sqrt(1 - s):
        // no libm sin/cos, so this branch is bit-reproducible on the CPU.
        if (pcg64_uniform(&state, inc) < wall_scatter[wall]) {
            double sx, sy, s2;
            do {
                sx = __dadd_rn(__dmul_rn(2.0, pcg64_uniform(&state, inc)), -1.0);
                sy = __dadd_rn(__dmul_rn(2.0, pcg64_uniform(&state, inc)), -1.0);
                s2 = __dadd_rn(__dmul_rn(sx, sx), __dmul_rn(sy, sy));
            } while (s2 >= 1.0 || s2 == 0.0);
            double sz = sqrt(__dadd_rn(1.0, -s2));
            if (wall == 0)      { dx = __dmul_rn(-1.0, sz); dy = sx; dz = sy; }
            else if (wall == 1) { dx = sz;                 dy = sx; dz = sy; }
            else if (wall == 2) { dx = sx; dy = __dmul_rn(-1.0, sz); dz = sy; }
            else if (wall == 3) { dx = sx; dy = sz;                 dz = sy; }
            else if (wall == 4) { dx = sx; dy = sy; dz = sz; }
            else                { dx = sx; dy = sy; dz = __dmul_rn(-1.0, sz); }
        }
    }
}
