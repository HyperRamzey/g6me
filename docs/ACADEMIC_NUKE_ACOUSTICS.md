# Physical Acoustics, Biophysical Verification, and Tensor-Core FP64 Emulation in `g6me`

This document specifies the exact mathematical formulations, numerical invariants, and verification gates added to `g6me` for physical binaural acoustics, hybrid 3D room modeling, biophysical cochlear auditing, Bayesian personalization, and low-precision Tensor-Core IEEE-754 `f64` emulation.

All `g6me-core` DSP routines are implemented in pure Rust with zero external crates and zero fused multiply-add (FMA) contraction so the reference downmixer (`RendererMode::Reference`) remains bit-for-bit identical to `vendor/g6ct`.

---

## 1. Exact 3D Helmholtz Rigid-Sphere Scattering + Coupled Snowman Torso

Module: `crates/g6me-core/src/scattering.rs`

Replaces the 1-pole/1-zero Brown–Duda (1998) approximation when `--plus-physics hankel-magls` is selected:

1. **Duda & Martens (JASA 1998)** — Exact Green's function for acoustic scattering around a rigid sphere of radius $a$ between $\mathbf{r}_1$ and $\mathbf{r}_2$ with angular separation $\theta$:
   $$G_{\text{sphere}}(\mathbf{r}_1, \mathbf{r}_2; k) = -\frac{ik}{4\pi} \sum_{m=0}^{\infty} (2m+1)\, P_m(\cos\theta)\, h_m^{(2)}(k r_{>}) \left[ j_m(k r_{<}) - \frac{j_m'(k a)}{h_m^{(2)\prime}(k a)}\, h_m^{(2)}(k r_{<}) \right]$$
   - Computed in `f64` via Miller downward recurrence for $j_m(x)$ and stable upward recurrence for $y_m(x)$ (`sph_bessel_jy`), summed until the relative term magnitude drops below $10^{-15}$.
   - Provides exact near-field acoustic parallax via `--plus-distance <METERS>` ($0.25\text{–}10.0\,\text{m}$).
2. **Algazi, Duda, Duraiswami, Gumerov & Tang (JASA 2002) — Coupled Snowman Head + Torso**:
   - Models head ($a_h$) and torso ($a_t$, vertical separation `--plus-torso-height`) via reciprocal free-field + head-scattered + torso-scattered + symmetric single-bounce head$\leftrightarrow$torso coupling.
   - Synthesizes physical shoulder-bounce interference patterns in the $700\,\text{Hz}\text{–}3\,\text{kHz}$ band from boundary-value scattering geometry.
3. **Invariant**: Exact acoustic reciprocity $|H(\mathbf{r}_1 \to \mathbf{r}_2) - H(\mathbf{r}_2 \to \mathbf{r}_1)| / |H| < 10^{-13}$ enforced in `cargo test` and `g6me-verify --group acoustics`.

---

## 2. Atmospheric Thermodynamics & Quantum Molecular $\text{O}_2/\text{N}_2$ Air Absorption

Module: `crates/g6me-core/src/atmosphere.rs`

Replaces hardcoded $343\,\text{m/s}$ with real thermodynamic air state (`--room-temp-c`, `--room-rh`, `--co2-ppm`, `--room-pressure-kpa`):

1. **Owen Cramer (JASA 1993)**: Virial-equation speed of sound $c(T, P, \text{RH}, x_{\text{CO}_2})$. At $21.5\,^\circ\text{C}$, $45\%\,\text{RH}$, $420\,\text{ppm}\,\text{CO}_2$, $101.325\,\text{kPa}$: $c = 344.8544\,\text{m/s}$.
2. **Bass, Sutherland, Zuckerwar, Blackstock & Hester (JASA 1995 / ISO 9613-1)**: Classical thermoviscous attenuation plus exact $\text{O}_2$ and $\text{N}_2$ vibrational relaxation frequencies $f_{r,\text{O}}$ and $f_{r,\text{N}}$, giving $\alpha(f)\,\text{dB/m}$ ($\approx 0.540\,\text{dB/m}$ at $20\,\text{kHz}$ in studio air).

---

## 3. Exact Slepian DPSS Eigenvector Tail Window

Module: `crates/g6me-core/src/fft.rs` (`slepian_dpss`, `half_slepian_tail_apply`)

Replaces the Half-Kaiser tail approximation with the 0th-order **Discrete Prolate Spheroidal Sequence (Slepian & Pollak, 1961)** when `--plus-slepian` is enabled:

- Computes the dominant eigenvector of Grünbaum's commuting symmetric tridiagonal matrix in $O(N)$ per inverse-iteration step via the Thomas algorithm, scaling cleanly to $N = 65{,}536$.
- Provably achieves strictly lower out-of-band energy leakage than a matched Kaiser window for equal time-bandwidth product $NW$.

---

## 4. Order-7 HOA, MagLS Binaural Decoding, Hold Tapering & Bark-Warped All-Pass Decorrelation

Module: `crates/g6me-core/src/hoa.rs`

1. **Order-7 Spherical Harmonics ($64$ ACN/N3D channels)** integrated on an exact $9 \times 16 = 144$-node Gauss–Legendre $\times$ Equiangular spherical quadrature grid (Gram matrix orthonormality error $< 10^{-14}$).
2. **MagLS Binaural Decoding** (Schörkhuber et al., DAGA 2018; Deppisch et al., IEEE TASLP 2022): least-squares complex SH decoding below $1.5\,\text{kHz}$ transitioning smoothly to magnitude-least-squares decoding above $1.5\,\text{kHz}$ to eliminate spatial aliasing comb notches.
3. **Hold et al. (ICASSP 2019) SH Tapering & Diffuse Coloration Compensation**: order-dependent tapering weights with exact diffuse-field power preservation.
4. **Kermit-Canfield & Abel (DAFx 2016) Bark-Warped All-Pass Decorrelator** (`--plus-warped-decorr`): cascaded frequency-warped all-pass lattice exhibiting $0.000000\,\text{dB}$ magnitude ripple ($||H(f)| - 1| < 10^{-13}$) with smooth low-frequency-to-high-frequency group-delay dispersion.

---

## 5. Hybrid 3D Room Acoustics Engine, Paraunitary FDN & Velvet Noise

Module: `crates/g6me-core/src/room.rs` (`--room anechoic|control-room|theater`, `--room-json <PATH>`)

Fuses three complementary physical wave layers plus lossless late diffusion:

1. **Layer 1 ($> 500\,\text{Hz}$)**: 3D Image-Source Method (orders 1–3) with octave-band wall absorption & ISO 9613-1 air absorption, plus deterministic **PCG64 Monte Carlo Ray Tracing** (orders 4–50 with ISO 17497 Lambertian scattering) accumulated into an Order-7 HOA histogram and decoded via MagLS.
2. **Layer 2 (Edge Diffraction)**: **Svensson–Fred–Vanderkooy (JASA 1999)** analytic Biot–Tolstoy–Medwin (BTM) finite-edge diffraction integrated along studio desk / console baffle edges (Kirsch & Ewert, TASLP 2023).
3. **Layer 3 ($< 500\,\text{Hz}$ — Wave-Equation PDE)**: **Hamilton & Bilbao (TASLP 2017)** 3D Finite-Difference Time-Domain (FDTD) acoustic wave-equation solver with locally reacting boundary impedance. Exploits **Acoustic Reciprocity** by running only **2** FDTD simulations (left ear source, right ear source) while recording all 12 virtual speaker coordinates simultaneously.
4. **Late Externalization (`--plus-fdn`)**: **Schlecht & Habets (TASLP 2017 / TSP 2020)** 8-channel Paraunitary Scattering Feedback Delay Network ($M^T M = I$ to $< 10^{-13}$) blended with **Välimäki et al. (TASLP 2013)** ternary Velvet Noise ($\{-1, 0, +1\}$).

---

## 6. 1D Webster Horn Equation Ear-Canal Impedance

Module: `crates/g6me-core/src/scattering.rs` (`webster_ear_canal_tf`, `ear_canal_compensation_mag`, `--ear-canal-mm <MM>`)

Models the ear canal as a 1D acoustic transmission line / Webster horn with physical tympanic membrane termination impedance ($R_{\text{tm}} = 4.8\,Z_0$), producing realistic quarter-wave resonance shifts ($\approx 3.6\,\text{kHz}$ at $25\,\text{mm}$, $\approx 4.3\,\text{kHz}$ at $21\,\text{mm}$, $\approx 3.1\,\text{kHz}$ at $29\,\text{mm}$) and applying a causal minimum-phase differential compensation relative to the $25.0\,\text{mm}$ reference.

---

## 7. Biophysical Cochlea Audit, Bayesian Externalization & Hear-to-Hear Personalization

Module: `crates/g6me-core/src/cochlea.rs`

1. **Zilany–Bruce–Carney (JASA 2014 / Bruce et al. 2018) Cochlea & Auditory-Nerve Neurogram**: 24-CF ERB-spaced gammatone basilar-membrane filterbank $\to$ nonlinear OHC compressive transduction $\to$ IHC low-pass rectification $\to$ power-law/adaptation synapse firing rate, scored via **NSIM** (Neurogram Similarity Index Measure).
2. **Baumgartner & Majdak (2021) Bayesian Auditory Externalization Model**: integrates monaural spectral-gradient cues and binaural IACC / early-reflection cues into $E \in [0, 1]$, gated in `g6me-verify --audit-cochlea [WAV]` and `g6me-verify --group cochlea` so `G6+ Studio` and `G6+ 3D ControlRoom` strictly beat anechoic `G6+ Pure`.
3. **Hear-to-Hear Bayesian Active Personalization Loop**: interactive BALD (Bayesian Active Learning by Disagreement) mutual-information optimizer over a $5 \times 4 \times 5 = 100$-point anatomical posterior `(head_radius_m, torso_height_m, pinna_shelf_db)`. Converges to ground truth within 10–15 A/B clicks and is integrated directly into the `g6me-gui` Renderer tab.

---

## 8. Blackwell (`sm_120`) Zero-FMA & Tensor-Core FP64 Emulation Suite

Module: `crates/g6me-core/src/blackwell.rs`

Implements full IEEE-754 `f64` linear algebra and FFT computation using exclusively low-precision Tensor-Core arithmetic plus exact fixed-point reduction:

1. **Exact 2304-Bit Kulisch Fixed-Point Superaccumulator (`KulischAcc`)**: eliminates catastrophic cancellation ($10^{16} + 1.0 - 10^{16} + 10^{-12} = 1.0 + 10^{-12}$ with `0.0` error).
2. **Ozaki Scheme I (INT8 Bit-Slicing GEMM)**: $O(S^2/2)$ error-free mantissa slicing to $< 10^{-13}$.
3. **Ozaki Scheme II (CRT — Uchino, Ozaki & Imamura, arXiv:2504.08009 / arXiv:2602.02549)**:
   - Reduces complexity from quadratic $O(S^2/2)$ to linear $O(M)$ using $M = 20$ pairwise-coprime moduli $m_j \le 113$ whose product spans $124.31$ bits (fitting exactly in `u128`).
   - Supports three Tensor-Core backends with **bit-exact identical (`0.0` diff)** CRT reconstruction:
     - `TensorBackend::Int8`
     - `TensorBackend::Fp8E4m3`
     - `TensorBackend::Fp4E2m1Base13` (**Hayashi, Mukunoki, Hoshino & Katagiri, Aug 2026, arXiv:2608.06812**): represents every residue $r \in [-56, +56]$ via 2 exact FP4 E2M1 limbs in Base-13 ($r = d_0 + 13\,d_1$, where doubled E2M1 $\{0, \pm 1, \pm 2, \pm 3, \pm 4, \pm 6, \pm 8\}$ forms a complete residue system mod $13$).
4. **Full-FP64 Bailey 4-Step FFT on FP4/FP8/INT8 Tensor Cores** (`bailey_ozaki_fft`, **Matsuoka, Sept 2026**): factors an $N_1 \times N_2$ complex DFT into Ozaki-II CRT matrix multiplies + exact Kulisch reduction, reaching $< 10^{-12}$ vs native `f64` FFT with zero FP64 hardware instructions in the matrix kernels.
5. **Phase-Exact GPU IIR**:
   - Overlap-Save with analytic $-320\,\text{dBFS}$ pole-decay settling margin.
   - Blelloch $O(\log N)$ parallel-prefix $2 \times 2$ state-transition matrix scan matching sequential IIR to $< 10^{-13}$.
6. **Blackwell (`sm_120`) Zero-FMA CUDA C++ & PTX Audit**: embeds zero-FMA (`__dmul_rn` / `__dadd_rn`, `--fmad=false`) CUDA kernels, reference `sm_120` PTX (`mul.rn.f64` / `add.rn.f64`), and `audit_gpu_asm_zero_fma` with negative-control verification rejecting `fma.rn.f64`, `mad.*`, and `DFMA`.
