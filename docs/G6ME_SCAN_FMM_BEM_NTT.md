# Personalized 3D FMM-BEM Acoustic Scattering, Goldilocks NTT Convolution, and the `g6me-scan` Photogrammetry Pipeline

This document specifies the mathematical foundations, optical/sensor physics, metric calibration standards, and verification gates implemented in `g6me-core`, `g6me-cli` (`g6me` and `g6me-scan`), `g6me-gui`, and `g6me-verify`.

---

## 1. `--exact-ntt`: Goldilocks Prime-Field Number-Theoretic Transform Convolution

Standard IEEE-754 floating-point FFT overlap-add convolution accumulates roughly $-315\text{ dBFS}$ of `f64` rounding noise and can diverge by 1–2 ULPs across architectures or compilers if FMA contraction is enabled.

`crates/g6me-core/src/ntt.rs` maps 24-bit integer PCM samples ($|x| \le 2^{23}$) and Q28 fixed-point FIR kernels ($|h| \le 2^{28}$) into the 64-bit Goldilocks prime field:

$$p = 2^{64} - 2^{32} + 1 = 18\,446\,744\,069\,414\,584\,321$$

Key properties:
- **2-adicity**: $p - 1 = 2^{32} \cdot (2^{32} - 1)$, supporting exact radix-2 Number-Theoretic Transforms up to length $N = 2^{32}$.
- **Primitive root**: $g = 7$ generates $(\mathbb{Z}/p\mathbb{Z})^\times$, with primitive $2^{32}$-th root of unity $\omega_{32} = g^{(p-1)/2^{32}} \bmod p = 1\,753\,635\,133\,440\,165\,772$.
- **Exact reduction**: Because $2^{64} \equiv 2^{32} - 1 \pmod p$ and $2^{96} \equiv -1 \pmod p$, any 128-bit product $a \cdot b$ reduces in constant time without division.
- **Overflow headroom**: For FIR length $L \le 16\,384$, the worst-case linear convolution sum is bounded by $L \cdot 2^{23} \cdot 2^{28} \le 2^{65}$, and for typical block sizes ($L \le 4096$, normalized FIR $\sum |h| \le 4$) sits more than $512\times$ below $p / 2$, guaranteeing **0 ULPs of intermediate floating-point rounding error** and bit-identical outputs across x86_64, ARM64, RISC-V, and GPU kernels (`BLACKWELL_GOLDILOCKS_NTT_CUDA`).

Usage:
```bash
g6me --mode plus --exact-ntt input.flac output.wav
```

---

## 2. `--head-mesh <FILE.obj>`: Exact 3D Helmholtz FMM-BEM Solver

`crates/g6me-core/src/bem.rs` replaces analytic spherical/snowman approximations with a direct boundary-element solution of the 3D exterior Helmholtz equation on an arbitrary watertight head and pinna mesh:

$$(\nabla^2 + k^2) p(\mathbf{r}) = -\delta(\mathbf{r} - \mathbf{r}_s), \qquad \left.\frac{\partial p}{\partial n}\right|_{\Gamma} = 0$$

### Mesh Conditioning
1. **Wavefront `.obj` Parser & Metric Auto-Scaling**: Parses triangles and fan-triangulates quads/polygons, centers the skull on the origin, and normalizes millimeter-authored scans (`extent > 2.0`) to meters.
2. **Generalized Winding Number Watertight Repair** (Jacobson et al., SIGGRAPH 2013): Computes the exact solid-angle winding field $w(\mathbf{x}) = \frac{1}{4\pi}\sum_j \Omega_j(\mathbf{x})$ across a 3D grid and extracts the closed $w(\mathbf{x}) = 0.5$ isosurface via Marching Tetrahedra, repairing missing neck caps, ear-canal holes, and hair dropouts into a 0-boundary-edge manifold with $w(\mathbf{0}) = 1.0$.
3. **Isotropic Acoustic Remeshing**: Enforces the acoustic discretization bound $h \le \lambda_{\min} / 6$ ($\le 2.5\text{ mm}$ at $20\text{ kHz}$, $c \approx 344.8\text{ m/s}$) via adaptive longest-edge bisection and tangential smoothing while preserving 0 boundary edges.

### 3D Octree FMM + Gumerov Acoustic Reciprocity
- Following Gumerov, O'Donovan, Duraiswami & Zotkin (JASA 2006 / 2010), solving the **adjoint** boundary integral equation $(C + K^T)\lambda = e_{\text{ear}}$ once per ear per frequency yields the exact free-field-normalized HRTF $H(\mathbf{r}_s \to \mathbf{r}_{\text{ear}})$ for every loudspeaker azimuth/elevation simultaneously via a single boundary inner product $\langle \lambda, p_{\text{inc}}(\mathbf{r}_s)\rangle$.
- The boundary operator is accelerated by a 3D spatial octree (`FmmOctree`) that evaluates near-field leaf interactions exactly and compresses well-separated far-field clusters via multipole-to-local (M2L) Green's-function expansions, solved via preconditioned restarted GMRES.
- Both CPU Rust and `BLACKWELL_FMM_BEM_CUDA` strictly prohibit FMA contraction (`__dmul_rn` / `__dadd_rn`).

Usage:
```bash
g6me --mode plus --head-mesh myhead.obj input.flac output.wav
```

---

## 3. `g6me-scan`: 4-Stage Photogrammetry-to-FMM-BEM Pipeline

`g6me-scan` (`crates/g6me-core/src/dng.rs`, `burst3d.rs`, `scan.rs`, `crates/g6me-cli/src/scan_main.rs`) converts raw smartphone DNG 1.7 captures or v7.0 zero-copy 50 MP MIPI `RAW10` + OIS/IMU bursts into a sub-millimeter, watertight, acoustically remeshed head model handed directly in memory to `FmmBemSolver`.

### Optical & Sensor Physics Ingest (`dng.rs` & `burst3d.rs`)
- **DNG 1.7 SubIFD & `OpcodeList2/3` Zero-Resampling**: Bypasses the low-resolution IFD0 preview trap into the full-resolution `SubIFD`, parses `OpcodeList2` (`GainMap`, ID 9) and `OpcodeList3` (`FixVignetteRadial`, ID 3; 3-plane RGB `WarpRectilinear`, ID 1), and applies radial/tangential undistortion analytically to `f64` feature rays via Newton–Raphson inversion. Zero image resampling occurs and 3-plane lateral chromatic aberration is cancelled to $< 0.008\text{ px}$.
- **Multi-Frame Honesty Report**: Inspects TIFF `NewSubFileType` (`0x00FE`), explicitly flagging computational multi-frame merges (`NewSubFileType == 16`) vs. true single-exposure raw frames.
- **v7.0 MIPI `RAW10` + `meta.json` Hardware Integrity Gates**: Unpacks packed 5-byte/4-pixel MIPI CSI-2 `RAW10` bursts, enforces the **Distinct-Frame Gate** (FNV-1a hash + pairwise pixel difference across all 6 frames to catch aliased ring-buffer frames) and the **Clock-Bracket Gate** (verifying OIS Hall-sensor timestamps bracket the entire burst readout window with non-zero variance), and loads optical polynomials from `reference_lens56_opcodes.dng`.
- **Handheld 6-Frame Burst 3D Physics**:
  - **Forster et al. (IEEE TRO 2017)** On-Manifold $SO(3)$ IMU Preintegration ($\Delta R, \Delta v, \Delta p$ with Rodrigues exponential/logarithm maps).
  - Continuous per-scanline OIS Hall-sensor principal-point interpolation for rolling-shutter compensation.
  - **Wronski et al. (SIGGRAPH 2019) + OISSR (ACM MM 2022)** zero-demosaicing anisotropic kernel-regression super-resolution directly on the raw Bayer mosaic.
  - **DoCam (MobiCom 2022) / Shakes-on-a-Plane (CVPR 2023)** micro-baseline depth estimation from natural hand tremor.
  - **Chugunov et al. (CVPR 2024)** diffuse/specular layer separation to remove oily ear-skin glare before multi-view matching.

### Stage 1: Dense Skin-Proof Matching & Metric Anchor (`scan.rs`)
- **Skin-Proof Dense Matching**: Sub-pixel ZNCC patch refinement with quadratic peak interpolation (and `BLACKWELL_SCAN_MATCHER_CUDA` kernel template).
- **Metric Anchor Options**:
  1. **Russian Debit Card (`GOST R ISO/IEC 7810-2006 ID-1`)**: Exact physical dimensions $85.60 \times 53.98\text{ mm}$ ($\pm 0.05\text{ mm}$), supporting both **Horizontal** and **Vertical** orientation. Fits 4 straight edge tangents via Total Least Squares away from the $3.18\text{ mm}$ rounded corners and intersects them to eliminate the $2.634\text{ mm}$ ($2.60\%$) corner-rounding scale bias; optionally verifies the ISO/IEC 7816-2 contact-chip orientation landmark at $(16.25, 23.89)\text{ mm}$; automatically scrubs the card face (PAN/CVV/holder name) from image buffers prior to reconstruction.
  2. **AprilTag3 / ChArUco fiducial** (`--anchor apriltag --tag-mm 50`).

### Stage 2: Global Rotation Averaging & Ozaki-II FP4→FP64 Rolling-Shutter BA
- **Global $SO(3)$ Rotation Averaging** (GLOMAP, ECCV 2024): Iterative Lie-algebra geodesic averaging on $SO(3)$ with Huber robust weighting.
- **Rolling-Shutter Bundle Adjustment** (Hedborg et al., CVPR 2012): Jointly optimizes camera poses, per-frame linear/angular rolling-shutter velocities, per-frame VCM focus-breathing scale ($\lambda_{\text{vcm}}$ Tikhonov prior), and 3D landmarks with hard metric-gauge constraints. The Schur Complement normal equations $(U - W V^{-1} W^T)\Delta c = g$ are formed via **Blackwell FP4 (E2M1 Base-13) Ozaki-II CRT → IEEE-754 FP64** (`g6me_core::blackwell::ozaki2_crt_gemm`).

### Stage 3 & Stage 4: 2DGS / PGSR → Winding Repair → Direct FMM-BEM Handoff
- **2D Gaussian Splatting (SIGGRAPH 2024) & PGSR (IEEE TVCG 2025)**: Represents the ear and skull with oriented planar surfels (`OrientedSurfel2D`), computing exact ray-plane depth, depth-distortion loss $\mathcal{L}_d$, and normal-consistency loss $\mathcal{L}_n$, followed by dense TSDF Marching Tetrahedra extraction.
- **Zero-Disk Handoff**: Conditions the extracted surface via Generalized Winding Number repair and isotropic acoustic remeshing and passes the watertight mesh directly in memory to `FmmBemSolver::new(&mesh)`.

Usage:
```bash
# End-to-end synthetic demo with GOST ID-1 Russian debit card anchor
g6me-scan --demo --anchor russian-card --out myhead.obj

# Inspect DNG 1.7 SubIFD and OpcodeList2/3
g6me-scan --dng frame0.dng

# Verify and process a v7.0 MIPI RAW10 + OIS/IMU burst folder
g6me-scan --burst ./burst_capture_01 --ref-opcodes reference_lens56_opcodes.dng --anchor russian-card --out myhead.obj
```

---

## 4. Verification Gates (`g6me-verify`)

All subsystems are continuously verified under `--require-all` with zero FMA contraction:
- `g6me-verify --group ntt`: Goldilocks radix-2 NTT invertibility, 0-ULP integer convolution vs. `i128` reference, streaming OLA & `BinauralPlus --exact-ntt` parity.
- `g6me-verify --group bem`: Generalized Winding Number open-hole repair (`0` boundary edges, $w(\mathbf{0}) = 1.0$), isotropic acoustic remeshing ($h \le h_{\text{target}}$), Gumerov FMM-BEM adjoint reciprocity ($< 10^{-10}$), analytical Mie sphere ILD agreement ($< 0.75\text{ dB}$), and zero-FMA CUDA audit.
- `g6me-verify --group scan`: DNG 1.7 SubIFD + analytical 3-plane `WarpRectilinear` lateral CA ($< 0.008\text{ px}$), v7.0 `RAW10` Distinct-Frame & Clock-Bracket gates (positive + aliased-buffer negative control), GOST ID-1 horizontal/vertical calibration + privacy masking, Forster $SO(3)$ IMU + Wronski/OISSR + Chugunov glare separation, Ozaki-II FP4→FP64 Rolling-Shutter BA convergence, and Stage 3→4→FMM-BEM direct handoff.
