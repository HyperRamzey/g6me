# CUDA performance plan — five sketches, grounded in the current tree

Status: design sketch (uncommitted). Targets the `sm_120a` Blackwell worker that already
runs the production offload. Every claim below cites the code it would change.

Guiding constraints, unchanged from the correctness round:

- The zero-FMA contract (`--fmad=false`, `__dmul_rn`/`__dadd_rn`/`__dsub_rn`) is not negotiable
  in kernels whose bit-output reaches the verifier.
- A speed change may alter rounding **only** where the current gate is already tolerance-based
  (GMRES residuals, FFT parity `1.958e-14`). Where the gate is bit-exact (Ozaki-II CRT, NTT,
  raytrace HOA, CLI render), the transformed kernel must reproduce the same operation order.
- Every item lands with a before/after measurement on the same machine and an explicit parity
  check; "faster" without a gate is not a deliverable.

### Native peers, not emulation

The CPU and GPU paths are peers: each implements the same acoustic/mathematical stage natively
on its own hardware. Neither is a software model of the other's instruction stream.

- No production CPU fallback may be a hand-written reproduction of a CUDA kernel's tiling,
  reduction order, or memory layout. The fallback is a real CPU algorithm; equivalence is
  checked with a documented tolerance.
- Test-only oracles may mirror the other side bit-for-bit (e.g. `trace_single_ray` vs
  `raytrace_hoa.cu`); they must not be on a production path.
- Bit-exactness between the two native paths is a **design goal**, achieved by specification:
  identical algorithm, identical operation order where the paper fixes it, strict IEEE
  rounding (zero-FMA, no fast-math, same precision policy), same summation order. Integer-exact
  stages (Ozaki-II CRT, Goldilocks NTT, integer histogram sums) are bit-identical by
  requirement. Float stages are driven to the smallest reproducible difference; where the
  hardware genuinely cannot match (transcendentals, tensor-core internal accumulation), measure
  the residual and gate at that bound — do not claim bit-identity you do not have.
- Every algorithm and numerical claim is taken from the **actual paper or standard**, fetched
  and cited by DOI/arXiv/standard number plus equation or section. Secondary web sources
  (blogs, summaries, dataset cards) are discovery aids only, never the authority.
- Preserving a *GPU* kernel's operation order across a *GPU* optimization is a separate axis:
  that keeps the optimizer honest against an existing self-consistency gate and is not CPU
  emulation. It may be dropped only when the gate is replaced by a tolerance gate with a
  recorded bound.

---

## 0. Baseline to measure first

| # | Path | Where | Metric |
|---|------|-------|--------|
| B1 | One UPOLA block, NTT engine (`N_FFT=8192`, `HOP=6144`) | `crates/g6me-core/src/plus.rs:2406-2493` | wall µs/block, kernel count |
| B2 | One UPOLA block, FFT engine | `plus.rs:2437-2493` | wall µs/block, kernel count |
| B3 | FMM-BEM GMRES, n=512/2k/8k, one frequency | `bem.rs:1745` | s/solve, GPU idle fraction |
| B4 | FDTD 3D, 256³, 1000 steps | `g6me-cuda/src/lib.rs:741` | steps/s, achieved GB/s |
| B5 | Ozaki-II CRT GEMM, m=n=k=2048 | `g6me-cuda/src/lib.rs:1137` | TFLOP/s effective, % of INT8 peak |
| B6 | Full `g6me --cuda` control-room render | CLI | wall time breakdown by stage |
| B7 | Ozaki-II vs cuBLAS emulated-FP64 GEMM | probe below | GFLOPS + max backward error |

Measurement harness: a `g6me-bench` subcommand (or `#[ignore]` benches in `g6me-cuda`) that
prints the six numbers with `cudaEvent` timing and a launch count from `CUDA_LAUNCH_BLOCKING`
off. These numbers are the acceptance baseline for every item below.

---

## 1. Fused single-launch NTT + VRAM-resident UPOLA (highest ratio, lowest risk)

**Current state.** `run_goldilocks_ntt_transform` (`lib.rs:2135`) is a one-shot wrapper:
H2D input, allocate two device buffers, bit-reversal kernel, 13 butterfly-stage launches,
(D2H) — ~17 launches and a host round-trip **per transform, per channel, per block**
(`plus.rs:2519`, `2523-2531`). At `HOP=6144` and 48 kHz that is ~7.8 blocks/s/channel; the
launch + transfer overhead alone is comparable to the arithmetic.

**Why it fuses.** `N = 8192`, Goldilocks `u64`: 8192 × 8 B = **64 KiB**, below the 99 KiB
per-block shared-memory limit of cc 12.0. A single 256–512 thread block can hold the whole
transform in shared memory: load → bitrev in shared → 13 stages with `__syncthreads()` →
optional inverse scale → store. One launch, one H2D, one D2H.

Implementation detail that decides feasibility: 64 KiB exceeds the 48 KiB *static* shared-memory
limit, so the kernel must declare `extern __shared__` and the host must raise
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` to 65536 before the first launch.
`cudarc` 0.19.10 exposes this on the safe path (`CudaFunction::set_attribute`,
`driver/safe/core.rs:2543`), so no raw driver binding is needed. If `set_attribute` fails on the
driver in use, the fallback is the same batched-multi-launch design as §2.

**Bit-exactness.** The GPU stages already mirror `NttPlan::forward` operation-for-operation
(`ntt.rs`); moving them into one block keeps the same per-element butterfly order and the same
`u128`-modular arithmetic, so the fused kernel must reproduce the current 0-ULP parity — the
existing check (`g6me-verify`: direct NTT transform parity) is the gate, extended to length
8192 and to the inverse direction.

**UPOLA residency.** `push_completed_upola_block` (`plus.rs:2497`) keeps the whole FDL and both
accumbents in `Vec<u64>` on the host, then ships each 8192-word buffer down and back. Add a
`CudaUpolaState` that owns:
- `ntt_fdl[head][ch]` (device, `n_parts × SLOTS × 8192 u64`),
- `ntt_acc_l/r` (device),
- the IR partitions in NTT domain (device, uploaded once when a preset loads),
- twiddle stages (already device-cached, `lib.rs:2073`).

The block callback then launches: `forward × active_channels`, `pointwise-mul` folded into the
accumulate kernel, `inverse` per ear — all device-resident, no host round-trip until the OLA
tail is read out. Target: **×10–20 fewer launches per block**, NTT transforms 2–4× faster.

**Stretch (behind a flag, separate parity study).** Real-input FFT at 8192 can be computed as a
4096-point complex FFT (64 KiB) plus the standard unpacking pass. That *would* fit one block,
but the CPU `FftPlan::rfft` (`fft.rs:226`) is a full 8192-point complex transform, so the result
would change the numerical contract. Do not adopt without also changing the CPU to the packed
algorithm and re-baselining every golden.

---

## 2. Batched FFT pipeline + CUDA Graphs (for the 128 KiB problem)

**Current state.** Same one-shot pattern as NTT, but the transform is out-of-place Stockham with
per-stage global kernels (`stockham_fft_f64.cu`); 8192 complex `f64` = **128 KiB**, which cannot
be resident in one block (99 KiB cap) and cudarc 0.19.10 has no cluster/`cuLaunchKernelEx`
entry point (checked: `driver/safe` has none, only `sys` types). A single-launch fused FFT is
therefore not available on this stack; do not plan around it.

**What is available.**
1. **Batch across blocks/channels.** The per-stage kernels are element-wise over `N`. Give them a
   `batch` dimension and run all channels (and, where the algorithm allows, several consecutive
   blocks) in one launch per stage. 13 stages × 1 launch instead of 13 × 8 transforms.
2. **CUDA Graph the whole per-block sequence.** `cudarc::driver::safe::CudaGraph`
   (`begin_capture`/`end_capture`/`launch`) exists and is proven in the crate's own tests.
   Capture the fixed sequence (H2D staging → stage kernels → IFFT → OLA accumulate) once per
   preset and replay per block. Graph replay overhead is a few µs for the whole sequence.
3. **Double-buffer host↔device** with two pinned staging buffers so the next block's H2D
   overlaps the current block's compute.

**Gate.** FFT parity stays at the current tolerance (`~2e-14` max abs vs CPU); the HOA histogram
check stays bit-exact because it does not consume the FFT path today. Log achieved
transforms/s against `cuFFT` as an external reference — no cuFFT dependency in-tree, just a
number in the report.

**Note on bit-exactness.** The reason FFT parity is `1.958e-14` and NTT is 0 ULP is algorithmic:
CPU `fft.rs` is bitrev + DIT, GPU `stockham_fft_f64.cu` is Stockham. Unifying the algorithms
(a shared-memory DIT kernel matching CPU order) would make the FFT bit-exact too; it fits at
`N ≤ 4096` and at 8192 only via the real-input halving trick, which changes the CPU contract.
Recorded as a known, deliberate divergence, not a bug.

---

## 3. GPU-resident GMRES + frequency/RHS batching

**Current state.** `solve_gmres_with_chief_cuda` (`bem.rs:1745`) builds the cached operator once
(good), but every iteration: packs `x_re|im` on the host, H2D, launches the matvec, D2H, unpacks,
then runs MGS on the host (`bem.rs:1880-1908`, O(n·m) complex ops per iteration). The Arnoldi
basis `v_basis` is a `Vec<Vec<Complex>>` — n×m host memory touched every iteration.
`solve_reciprocal_hrtfs_with_cuda` (`bem.rs:2013`) is called **once per frequency**, each call
re-solving from scratch and re-uploading nothing (operator is per-k), so a 24-band HRTF sweep is
24 serial solves with a mostly idle GPU.

**Part A — keep the Krylov state device-resident.** New `CudaGmresState` owning `x`, `r`, `w`,
`V` (max 33 × n complex), the FMM/dense operator handle, and a small host-visible scalar slot
for `β`/`h_{ij}` reductions. Per iteration only scalars cross PCIe. MGS becomes a batched
dot-product + axpy kernel pair. The Krylov recurrence is unchanged (MGS + Givens), so the
residual trajectory should match the current solve to within the same tolerance the verifier
already accepts (`GPU FMM-BEM reciprocal solve parity`, n=512).

**Part B — batched multi-frequency solve.** Add
`solve_reciprocal_hrtfs_band_batch(&self, freqs: &[f64], ...)` that:
- builds one operator per frequency (memory permitting; cap by free VRAM with a fallback to
  chunks),
- assigns each in-flight solve to its own stream (2–4 streams), interleaving matvecs so the
  Tensor/CUDA cores stay busy while one solve runs its scalar reductions,
- restarts are independent, so no cross-stream synchronization until a band completes.

**Part C — sketched/CA-GMRES (only if Parts A+B measure insufficient).** CA-GMRES
(Hoemmen) orthogonalizes against a basis of `s` Krylov vectors selected by a cheap residual
sketch, cutting global syncs per iteration by ~s. It changes the basis and therefore the
rounding path — acceptable only under the existing tolerance gate, and it must be benchmarked
against the *new* A+B implementation, not the current one.

**Gate.** Reciprocal-solve parity vs CPU FMM at the existing tolerance; residual ≤ `tol` for
every band; wall-clock per 24-band sweep reported before/after.

---

## 4. AN5D temporal-blocked FDTD (2.5D wavefront)

**Current state.** `run_fdtd_3d_simulation` (`lib.rs:741-864`) launches one kernel per step
(plus one receiver kernel), with a full grid read/write per step. At 256³ that is ~134 MB of
traffic per step per array touch — memory-bound, and 1000 steps means 1000 launch overheads
plus zero temporal reuse in cache.

**AN5D design (arXiv:2001.01473).** Process a time-block of `Tb=4` steps and a space-block of
`Sz=8–16` z-slices in one kernel:
- grid walks a 2.5D wavefront: block `(x, y, t)` consumes `z`-slices produced by its
  predecessor in the same time block,
- each thread keeps the front `Tb+2` planes in registers/shared, so a cell's stencil values are
  reused across `Tb` time steps instead of round-tripping to global,
- arithmetic order per cell is byte-identical to the current 7-point stencil
  (`g6me_fdtd_3d_srl_step`), so the CPU/GPU parity stays exact; the win is traffic, not math.

Expected 2–4× on the 3D path at `Tb=4`, scaling with stencil order and block size. Risk: the
wavefront dependency is easy to get wrong at domain boundaries and with the SRL damping profile
(`inv_den` varies per voxel) — the existing CPU/GPU simulation test plus a new `Tb=1` vs `Tb=4`
bit-compare on a 64³ grid is the sabotage check (per project rule 1).

**Complement: CUDA Graph the step loop.** Independent of AN5D and shippable first: capture the
`steps`-length launch sequence (receiver kernel + step kernel with buffer ping-pong) and replay.
Even without temporal blocking, this removes per-step launch cost — meaningful for small grids
where launch ≈ kernel time. Buffer swapping currently alternates three device buffers
(`lib.rs:846-847`); the graph must encode the even/odd parity explicitly (two captured graphs,
or `HOP`-period unrolling), which is the one subtle part.

---

## 5. Fused, cp.async-pipelined Ozaki-II MMA

**Current state.** `run_ozaki2_crt_gemm` (`lib.rs:1137`) issues: scale-reduce A, scale-reduce B,
precision pack (one or more launches), INT8/FP4 MMA over 20 moduli, then CRT reconstruction —
roughly 6–10 launches per GEMM, each reading/writing global. The MMA kernel is bandwidth-bound on
the 20-modulus `i8` operands (d_a_sym/d_b_sym are `20 × m × k`), and the pack kernels are
element-wise passes over the same memory.

**Design.**
1. **Fuse the prologue.** One kernel: exponent extraction + symmetric reduction for A *and* B
   (they are independent element-wise maps), writing packed operands directly in the layout the
   MMA kernel consumes. Removes 2–3 launches and one full read/write of the operands.
2. **cp.async double-buffering in the MMA kernel.** Stage A/B tiles into shared memory with
   `cp.async` (16 B per thread), two stages, `__pipeline_commit/wait_prior`. The current kernel
   reads operands from global inside the k-loop; at 2048³ the shared-memory tile pipeline is the
   standard 1.5–2× win for INT8 tensor cores.
3. **Fold the CRT epilogue.** CRT reconstruct already reads `c_mod` per output; keep it as the
   epilogue of the MMA kernel when the 20-modulus cross-block reduction can be done with a
   single CTA per output tile (true when the moduli are the grid's z-dimension, as in
   `mma_cfg_32x32`, `lib.rs:1235`). Otherwise keep a separate CRT launch and graph the pair.
4. **CUDA Graph the GEMM sequence** for the bundle-adjustment inner loop
   (`scan.rs:720`, called once per BA iteration with changing values but fixed shapes) — capture
   per shape, replay per iteration. Values change through the *same* buffers, so no graph
   re-instantiation is needed.

**Gate.** The exact-CRT property is the product: after every change,
`ozaki2_crt_reconstructs_exact_integer_limb_product` must still hold, the
`Ozaki-II INT8 vs FP8 / FP4 ... bit-exact identical` assertions must still hold, and the live
`Live GPU Ozaki-II CRT GEMM parity` check must keep `max backward err < 1e-13` (current live
gate) with no regression in the reported value. This is the one item where a "faster" result
with any bit drift is a failure, not a trade-off. Report TFLOP/s and % of peak alongside.

**Optional — RandNLA preconditioning for BA.** Woodbury/randomized preconditioners can cut BA
iterations, but the current Schur path is already exact-Ozaki and the BA test asserts
sub-pixel convergence in ≤14 iterations. Only pursue if B5 shows the GEMM dominating the scan
pipeline after items 1–4; otherwise it is complexity without a measured target.

---

## 6. CUDA Tile (`ct::scaled_mma` / cuTile Rust) backend for Ozaki-II Stage 3 — opt-in

**Prerequisites verified on this machine (not assumed):** `nvcc` V13.5.37, driver
620.73.04, RTX 5070 at CC 12.0 (sm_120), and `/opt/cuda/bin/tileiras` present. `cutile` 0.5.0
is on crates.io (2026-11-05); its compatibility matrix lists block-scaled MMA as
Tile IR 13.3+, target `sm_100+`, and `sm_120` as a supported Blackwell target. PDL is 13.4+,
`sm_90+`, and needs a host runtime with `cuLaunchKernelEx` — cuTile's own runtime has it;
`cudarc` 0.19.10 does not. GPUs below `sm_80` are unsupported by Tile IR, so this backend can
never replace the multi-arch SASS audit (`sm_75..sm_120a`); the hand-written
`ozaki2_blackwell_mma.cu` stays the portable fallback and the audit target.

**Why it is safe for the 0-ULP contract (numbers checked, not asserted).** The Stage-2 NVFP4
pack stores Base-13 limbs (`e = d/2`, `d ∈ {0,±1,±2,±3,±4,±6,±8}`) with a UE4M3 scale of 2.0
per operand (`ozaki2_blackwell_mma.cu:33`), so the tensor-core operands are the *integers*
`d0,d1` with `|d| ≤ 8`, and each limb accumulator sums integer products `≤ 64` over K.
At the shipped maximum `K = 131_072`: `64·K = 2^23 < 2^24`, so every FP32 partial sum is an
exact integer and any K-loop reordering tileiras performs is bit-neutral. The INT8 path is
weaker-bounded: `56²·K = 411,041,792 < 2^31`. The CRT budget itself holds with `log2` headroom
`0.311` bits (`M = ∏ m_j = 2^124.311`, max K from the integer bound `162,646`).

**The one structural constraint that preserves exactness:** the four limb accumulators
(`p00, p01, p10, p11`) must stay separate until the integer combination
`4·p00 + 52·(p01+p10) + 676·p11`; folding them into one FP32 accumulator is safe only while
the weighted total stays below `2^24`, and the bound for the combined total is `28,224·K`,
which exceeds it for `K ≥ 595`. The CPU reference and the current kernel both keep them
separate (`blackwell.rs:591-594`, `ozaki2_blackwell_mma.cu:547-556`); a tile kernel must too.

**Unconfirmed, spike before planning around it:** cuTile's DSL documents FP4 `mmaf_scaled`
explicitly but only calls out `i4` as an unsupported high-level integer MMA operand — i8
`mma` support is not stated either way. The INT8 Stage-3 kernel therefore stays hand-written
unless a hello-world i8 GEMM on this GPU proves otherwise. NVFP4/FP8 `mmaf_scaled` is the
documented path and the one worth pursuing.

**Integration shape.** `cutile = "0.5"` as an **optional** dependency of `g6me-cuda` behind a
`cutile` feature; dispatch inside `run_ozaki2_crt_gemm` only when
`prec == Nvfp4E2m1 && cfg!(feature = "cutile")` and the live device is `sm_100+`. Never in the
default build, never in `--require-all` on a machine without a 13.5 toolkit. Two CUDA binding
stacks (`cudarc` + `cuda-core`) will coexist; the spike must prove they share the primary
context and that cutile's streams synchronize with `cudarc`'s (or that the Ozaki-II Stage-3
call is a clean hand-off with explicit sync on both sides).

**Spike order (cheap, before any integration):**
1. `cargo add cutile` in a scratch crate, run the stock GEMM example on the 5070 — proves
   `tileiras` JIT + sm_120 + driver 620 works at all.
2. An NVFP4 `mmaf_scaled` residue GEMM for `K = 8192`, four separate accumulators, compared
   bit-for-bit against `ozaki2_crt_gemm`'s limb path — proves the exactness argument on
   hardware, not on paper.
3. Only then wire it as the optional Stage-3 backend and benchmark against B5.

**Caveat on the third-party benchmark in the proposal** (LuisaCompute #285, "97.6 vs 131.2 TF
at 1024³, ~75%"): unverified, single-source, and on a different GPU. The acceptance target is
our own B5 before/after on this RTX 5070; the spike either beats the current `32×32` CTA kernel
by a measured margin or the backend is not merged.

---

## 7. Blackwell micro-architecture extras — verified locally, ranked by what is actually reachable

All local facts below were queried from this machine (CUDA 13.5.37, driver 620.73.04, RTX 5070
CC 12.0), not assumed:

| device attribute | value |
|---|---|
| L2 cache | 50,331,648 B (48 MiB) |
| shared memory per SM | 102,400 B (100 KiB) |
| max opt-in shared per block | 101,376 B (99 KiB) |
| max access-policy window | 134,217,728 B (128 MiB) |
| cluster launch support | 1 (supported) |
| registers / threads / blocks per SM | 65,536 / 1,536 / 24 |

**7.1 256-bit loads/stores (`double4_32a`, `ulonglong4_32a`).** Present in
`/opt/cuda/include/vector_types.h:455,487` with `__builtin_align__(32)`. Zero arithmetic change,
so the zero-FMA contract is untouched. Apply to: `goldilocks_ntt.cu` (4 u64 per load/store),
`stockham_fft_f64.cu` (2 complex f64), `fdtd_3d.cu` (4 voxels along z). This is a contained,
measurable win and should ride along with §1/§2/§4 rather than be its own agent.
Caveat: the compiler emits LDG.256 only when the pointer is provably 32-B aligned and the
target is Blackwell; keep the scalar tail path for the last `N % 4` elements.

**7.2 Thread-block clusters + DSMEM.** `cudaDevAttrClusterLaunch = 1` on this GPU, so the
hardware path exists. With 99 KiB opt-in per block, an 8-block cluster pools ≈792 KiB of
distributed shared memory — enough for a 32,768-point `u64` NTT (256 KiB) or even a 16,384-point
complex-`f64` FFT (256 KiB), on-chip in one launch. **The blocker is purely the launch API:**
`cudarc` 0.19.10's safe layer has no `cuLaunchKernelEx`/cluster-dimension attribute (only the
`sys` types), and cuTile's runtime does not expose cluster launches to foreign kernels either.
Reaching it means a hand-written raw-driver launch shim with `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`
and `cudaFuncAttributeNonPortableClusterSizeAllowed`. Verdict: **stretch**, only if §2's batched
FFT + graphs still leaves the FFT dominating. Note the proposal's 16k/32k FFT sizes are not in
the tree — UPOLA is pinned at `N_FFT = 8192` (`plus.rs:90`); a larger size is a product change,
not just a kernel change. For the shipped 8192 `f64` FFT (128 KiB), a 2-block cluster would make
a single-launch version possible; that is the realistic use, if the raw launch shim is built.

**7.3 PDL (programmatic dependent launch).** `cudaTriggerProgrammaticLaunchCompletion` /
`cudaGridDependencySynchronize` exist in `cuda_device_runtime_api.h:448`; Tile IR 13.4+ supports
PDL with `sm_90+`, and cuTile's runtime has the `cuLaunchKernelEx` path. For the hand-written
kernels, `cudarc` again has no launch-attribute surface. Where it would pay: the Ozaki-II chain
(`scale_reduce_a/b → pack → mma → crt`) and FMM-BEM level launches, hiding the drain latency of
each grid. Verdict: **defer** until §5 lands; the fused prologue already removes two of the
dependent launches PDL would overlap, so the remaining gain is launch-latency-only.

**7.4 Device-side conditioning (`CU_GRAPH_COND_TYPE_WHILE`).** The enum and
`CUgraphConditionalHandle` exist (`cuda.h:315,2025`). Putting the GMRES convergence loop into a
graph-conditional body is possible in principle, but the residual norm is produced by an Arnoldi
step whose Givens rotations run where the condition must be evaluated; this turns a 60-line solver
into a graph-construction problem with device-updatable handles, and the tolerance gate would need
a device `while` with a host-visible completion value. It also locks the solver to graphs, which
breaks the CPU fallback symmetry. Verdict: **not planned**; the device-resident GMRES of §3
removes the host round-trips that motivate it.

**7.5 3D TMA + L2 persistence.** TMA is real on this GPU but needs `cuTensorMapEncodeTiled`,
mbarrier setup, and a descriptor lifetime manager; `cudarc` exposes no tensor-map API. AN5D (§4)
already attacks the same traffic problem with ordinary loads and keeps the stencil arithmetic
bit-identical; TMA would be a second rewrite for a smaller residual gain. L2 persistence
(`cuStreamSetAttribute` + `CU_ACCESS_PROPERTY_PERSISTING`) is not in `cudarc` either; the
twiddle tables are 64 KiB–1 MiB and the fused/device-resident designs already keep them hot
without a policy window. Verdict: **defer both**; revisit only if profiling shows twiddle
re-fetch or FDTD boundary divergence dominating after §1–§4.

**7.6 Pinned host memory + dual-stream overlap.** `cudarc` does expose it:
`CudaContext::alloc_pinned::<T>()` → `PinnedHostSlice` (`driver/safe/core.rs:1456-1486`), and
`CudaStream` supports async copies. §1 and §2 already remove most per-block host traffic by
keeping UPOLA state on device; pinning still helps the remaining staged paths (scan I/O, WAV
ingest) and the double-buffered H2D/D2H in §2. Cheap, low risk; fold into §2's implementation.

---

## 8. Benchmark probe against cuBLAS emulated FP64 (reference only, never the engine)

CUDA 13.5's cuBLAS exposes fixed-point FP64 emulation:
`CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT = 79` (`/opt/cuda/include/cublas_api.h:216`), and
`cudarc` 0.19.10's `cublas` feature already binds the same enum
(`src/cublas/sys/mod.rs:80`). That makes a fair in-tree probe possible without adding a new
dependency: a `g6me-bench` (or `#[ignore]` test) that runs the same `m×k×n` GEMM through both
engines on the same host inputs.

**What the probe reports, per shape (512², 2048², 8192³):**
- our `run_ozaki2_crt_gemm` (INT8 and NVFP4): wall time, GFLOPS, and max backward error vs the
  exact CPU `u128` CRT reference (expected 0.0 for the limb path);
- cuBLAS with compute type 79: wall time, GFLOPS, and max backward error vs the same reference;
- and, as the honesty anchor, a sequential FP64 CPU dot-product reference on a sampled subset
  to show what "exact" means for these inputs.

**Rules for using it.** cuBLAS is closed-source and cannot be audited for hidden DFMA or for its
internal reduction order, so it is never a pipeline dependency and never a correctness oracle.
It is an external speed reference and a sanity check on our own numbers. Expect it to be
*near*-exact rather than bit-identical to our CRT: report accuracy and speed separately, and do
not claim parity without the measured backward error on the same inputs. If cuBLAS beats us by a
large margin, that is input to §5/§6, not a reason to replace the audited engine.

The probe lives behind an optional `cublas-probe` feature in `g6me-cuda` (cudarc's `cublas`
feature is already available) and is excluded from the default verifier and CI matrix, exactly
like the cuTile path.

---

## Ordering and agents

| Order | Item | Why first | Agent |
|-------|------|-----------|-------|
| 1 | §1 fused NTT + UPOLA residency | biggest constant-factor win, bit-exact gate already exists | implementation agent 1 |
| 2 | §2 batched FFT + graph | same file family as §1, shared staging buffers | implementation agent 1 (after §1) |
| 3 | §4 FDTD graph, then AN5D `Tb=4` | independent subsystem, clean bit-compare gate | implementation agent 2 |
| 4 | §5 Ozaki prologue fusion + cp.async | feeds the scan pipeline; 0.0 gate is strict | implementation agent 3 |
| 5 | §3 GMRES device residency + band batching | tolerance gate; do after A+B baselines exist | speed agent (own stream) |

Each agent must: take the B-number baseline before editing, keep changes behind the existing
`use_cuda` dispatch so CPU fallback is untouched, add a covering test (sabotage-checked per
project rule 1), run `cargo test` + `cargo clippy --features cuda -- -D warnings` + the full
verifier with a GPU, and report the before/after numbers with the exact commands used.

## What this plan does **not** claim

- No single-launch 8192 complex-`f64` FFT: 128 KiB > 99 KiB per block, and cudarc exposes no
  cluster launch. The honest deliverable is batching + graphs (§2).
- No change to the zero-FMA or exact-CRT contracts anywhere; the only tolerance-level gates are
  the ones that were tolerance-level before (§2 FFT, §3 GMRES).
- No cuFFT/cuBLAS dependency: every kernel stays in-tree and auditable by `cuobjdump`.
