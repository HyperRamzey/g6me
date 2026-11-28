# G6ME Lean 4 formalization

A core-Lean-4 formalization of the exact integer and rational mathematics in
`crates/g6me-core/`.  No Mathlib, no axioms, no `sorry`.

- Toolchain: pinned in `lean-toolchain` (Lean 4.27.0 / Lake 5.0.0). The CI
  `lean` job installs exactly that toolchain with elan and runs the gate, so
  the claims on this page are what CI enforces, not local observations.
- Build: `cd lean && lake build`
- Purity gate: `./check_axioms.sh` (forbidden-token scan, full build, theorem
  coverage against the EXPECTED registry, and `#print axioms` for every
  registered theorem).
- The gate permits only `propext` and `Quot.sound`.  It rejects
  `Classical.choice`, `sorryAx`, `Lean.ofReduceBool` and
  `Lean.trustCompiler`.  Current result: 266 public theorems checked, axioms
  limited to `propext` and `Quot.sound`.

The Rust tree is untouched.  Everything here lives under `lean/`.

## Why there is no `ring` / `nlinarith`

Mathlib is not available.  `G6ME.Basic` therefore provides two custom
normalisers:

- `ring_nat` / `ring_int`: distribute products and normalise with `simp`,
  then finish the resulting linear identity with `omega`.
- `ring_int3`: a two-phase variant that first distributes, then normalises
  monomials modulo associativity/commutativity, then calls `omega`.  Needed
  for degree-3 and degree-4 monomials.

`grind` was tested and rejected: it proves the polynomial identities, but
`#print axioms` reports `[propext, Classical.choice, Quot.sound]`, which the
purity gate forbids.

Two core `Nat` lemmas that this development needs are themselves proved with
`Classical.choice`: `Nat.mul_lt_mul_left` and `Nat.pow_lt_pow_right`.  They are
therefore not used; `G6ME.Basic` proves the constructive replacements
`mul_lt_mul_left_of_pos` and `two_pow_lt_two_pow`, and the Kulisch module uses
those.

## Layout

| File | Rust source | Content |
|---|---|---|
| `G6ME/Basic.lean` | — | tactics, modular-congruence lemmas, constructive Nat facts |
| `G6ME/Goldilocks.lean` | `ntt.rs` | Goldilocks modulus and the 128-bit reduction |
| `G6ME/GoldilocksNtt.lean` | `ntt.rs` | field ops, signed conversions, butterfly, size-2 NTT |
| `G6ME/BlackwellFp4.lean` | `blackwell.rs` | Base-13 FP4 E2M1 encoding, FP8 split |
| `G6ME/BlackwellCrt.lean` | `blackwell.rs` | 20-modulus CRT |
| `G6ME/BlackwellKulisch.lean` | `blackwell.rs` | Kulisch limb accumulator |
| `G6ME/Dng.lean` | `dng.rs` | MIPI RAW10 and 16-bit packing |
| `G6ME/Stn.lean` | `stn.rs` | STN mask unity sum |
| `G6ME/Fdn.lean` | `room.rs` | 8×8 FDN paraunitary matrix |
| `G6ME/Limiter.lean` | `limiter.rs` | ceiling/gain bounds, ADAA2 passband |
| `G6ME/Hoa.lean` | `hoa.rs` | Legendre recurrence, ACN layout, taper energy |
| `G6ME/Burst3d.lean` | `burst3d.rs` | Rodrigues SO(3) algebra |
| `G6ME/Upmix.lean` | `upmix.rs` | 2×2 Hermitian PCA/APES algebra |

## Theorem inventory

### `G6ME.Basic` — shared infrastructure (no Rust counterpart)

`meq_refl`, `meq_symm`, `meq_trans`, `cancel_add_mod`, `MEq.add`, `MEq.mul`,
`mod_add_mul_right`, `mod_add_mul_left`, `mod_add_mul_right'`,
`mod_add_self_left`, `mod_add_self_right` are congruence lemmas over
`a % n = b % n`.  `mul_lt_mul_left_of_pos` and `two_pow_lt_two_pow` are the
constructive replacements described above.

### `G6ME.Goldilocks` — `ntt.rs:63` `reduce128`, `ntt.rs:37-142`

Constants and decomposition:

| Theorem | Rust | Statement |
|---|---|---|
| `P_eq` | `ntt.rs:24` `GOLDILOCKS_P` | `p = 2^64 - 2^32 + 1` |
| `K_eq`, `two64_eq`, `two96_eq`, `two128_eq` | `ntt.rs:63-95` | `K = 2^32-1`, `2^64 = p+K`, `2^96+1 = (2^32+1)p` |
| `loOf_lt`, `hiAllOf_lt`, `midOf_lt`, `hiOf_lt` | `ntt.rs:64-70` | limb ranges of the 128-bit split |
| `tOf_lt`, `mid_mul_K_lt`, `sOf_lt`, `sOf_mod`, `resOf_lt` | `ntt.rs:71-93` | intermediate bounds and the `s - 2^64` wrap |
| `key_aux`, `reduce_key_identity` | `ntt.rs:63-95` | the exact integer identity behind the reduction |
| `sOf_meq`, `resOf_meq_s` | `ntt.rs:71-93` | congruence of the folded value to `x` |
| `reduce128_spec` | `ntt.rs:63` `reduce128` | `x < 2^128 -> reduce128 x = x % p` |

Not proved: primality of `p` (the field axioms are therefore not derived, only
the arithmetic identities are).  The theorem is an unconditional equality of
`Nat` values, so it implies the Rust reduction is exact for every `u128`.

### `G6ME.Goldilocks` (continued) — `GoldilocksNtt.lean`, `ntt.rs:37-142, 236-268`

| Theorem | Rust | Statement |
|---|---|---|
| `P_lt_two64`, `P_lt_two128` | — | size facts used by the proofs |
| `modAdd_spec`, `modAdd_lt` | `ntt.rs:37` `mod_add` | equals `(a+b) % p`, and stays canonical |
| `modSub_spec`, `modSub_lt` | `ntt.rs:49` `mod_sub` | equals `(a+(p-b)) % p`, and stays canonical |
| `mul_lt_two128`, `modMul_spec`, `modMul_lt`, `modMul_one` | `ntt.rs:97` `mod_mul` | equals `(a*b) % p` via `reduce128`; `modMul a 1 = a` |
| `i64ToField_lt` | `ntt.rs:126` `i64_to_field` | result is canonical in `[0,p)` |
| `fieldToI128_range`, `fieldToI128_i64ToField` | `ntt.rs:142` `field_to_i128` | balanced representative in `[-(p-1)/2,(p-1)/2]`, and roundtrip on that range |
| `inv2_lt`, `inv2_mul_two`, `two_mul_inv2_mod` | `ntt.rs:230-233` inverse scaling | `2 * inv2 = 1 (mod p)` |
| `butterfly_inverse_fst`, `butterfly_inverse_snd` | `ntt.rs:262-268` | the two butterfly outputs recover `u` and `v` after `n^{-1}` scaling |
| `ntt2_roundtrip` | `ntt.rs:236` `transform` + `ntt.rs:230` `inverse` | forward-then-inverse is the identity for the size-2 transform |

`butterfly u v w` is a literal transcription of `ntt.rs:262-268`
(`t = mod_mul(v,w)`, outputs `mod_add(u,t)`, `mod_sub(u,t)`); `ntt2` is the
single-butterfly size-2 plan with twiddle `1`, `intt2` is the same butterfly
followed by `mod_mul(·, inv2)`.

Not proved: correctness of the general `N` Cooley–Tukey transform (bit-reversal
permutation, stage structure, twiddle tables and roots of unity), the
`exact_int_convolve` pipeline (`ntt.rs:271`), and the no-overflow argument for
the convolution coefficient bound.  Only the size-2 instance is proved.

### `G6ME.BlackwellFp4` — `blackwell.rs:257,265,438-620`

| Theorem | Rust | Statement |
|---|---|---|
| `remToD_residues` | `blackwell.rs:265` | the 13-entry table is a complete residue system mod 13 |
| `remToD_correct`, `remToD_mem`, `remToD_mem_getD` | `blackwell.rs:265` | the selected entry is congruent to `r % 13` and is a doubled E2M1 value |
| `isD_residue_injective` | `blackwell.rs:265` | a doubled E2M1 value is determined by its residue mod 13 |
| `encodeD_reconstruct` | `blackwell.rs:265` `encode_base13_fp4_e2m1` | `r = d0 + 13*d1` on `[-56,56]`, the range used by `ozaki2_crt_gemm` |
| `encodeD_mem` | `blackwell.rs:265` | both limbs are in `{0,±1,±2,±3,±4,±6,±8}` |
| `encodeD_unique` | `blackwell.rs:265` | the two-limb representation is unique on the supported range |
| `base13_product_identity` | `blackwell.rs:438-620` | `(d0+13d1)(e0+13e1)` expands in base 13 |
| `base13_gemm_identity` | `blackwell.rs:438-620` | the exact integer content of the `Fp4E2m1Base13` GEMM branch |
| `fp8_product_identity` | `blackwell.rs:438-620` | the `lo + 16*hi` two-limb product identity |
| `fp8_split_exists` | `blackwell.rs:438-620` | every `r` in `[-56,56]` splits as `lo + 16*hi` with the stated limb bounds |

The Lean limbs are the *doubled* values; the Rust function returns `d0/2`,
`d1/2`.  Not proved: `ldexp_exact` (`blackwell.rs:204`), any `f64` rounding,
the tensor-core execution itself, and the Ozaki-II double-double
reconstruction (`blackwell.rs:438-620`).

### `G6ME.Crt` — `blackwell.rs:365,411,438`

| Theorem | Rust | Statement |
|---|---|---|
| `moduli_length`, `M_eq`, `M_pos`, `M_lt_two128` | `blackwell.rs:365` `OZAKI2_MODULI` | the 20-element list, its product `M = 26390970204770983741947274447420058304`, and `M < 2^128` |
| `moduli_pairwise` | `blackwell.rs:365` | the 20 moduli are pairwise coprime (kernel-checked) |
| `mAt_pos_of_lt`, `mAt_dvd_M_of_lt`, `mAt_mem` | `blackwell.rs:365` | list-index facts for the moduli |
| `inv_check_of_lt`, `cross_dvd_of_ne` | `blackwell.rs:411` `mod_inv_u128` | `inv_i * (M/m_i) = 1 (mod m_i)`, and `m_j | M/m_i` for `i ≠ j` |
| `crtTerm_mod`, `crtX_spec` | `blackwell.rs:438-620` | the CRT sum has the prescribed residues |
| `crt_exists` | `blackwell.rs:438-620` | existence of `x < M` with `x % m_i = r_i` |
| `crt_unique` | `blackwell.rs:438-620` | uniqueness below `M` |

Not proved: the floating-point double-double evaluation of `S/M`, the
`sym_mod_i64` conversion (`blackwell.rs:370`), and the final fp64 rounding.

### `G6ME.Kulisch` — `blackwell.rs:54,68,78,115`

| Theorem | Rust | Statement |
|---|---|---|
| `B_eq`, `B_pos` | `blackwell.rs:54-60` | limb base `2^64` |
| `limbVal_zeros`, `length_zeros` | `blackwell.rs:70-76` `new` | the 36 zero limbs represent 0 |
| `length_addCarry`, `limbVal_addCarry` | `blackwell.rs:78-112` `add` | carry propagation preserves the value when the sum fits |
| `length_addAt`, `limbVal_addAt` | `blackwell.rs:78-112` | insertion at word `i` represents `old + v*B^i` |
| `scaled_mant` | `blackwell.rs:92` | `mant * 2^(pos%64) * B^(pos/64) = mant * 2^pos` |
| `Acc.add_exact` | `blackwell.rs:78` `KulischAcc::add` | one insertion increases the accumulator value by exactly `mant * 2^bitpos` |
| `bitpos_range` | `blackwell.rs:68` `BIAS_BITS = 1088` | a finite `f64` has `exp2` in `[-1074,971]`, so `bit_pos` is in `[14,2059]` |
| `fp64_term_fits` | `blackwell.rs:54-60` | a 53-bit mantissa at `bitpos <= 2059` fits in 36 limbs |
| `kulisch_single_add_exact` | `blackwell.rs:78` | one finite-`f64` term is accumulated exactly from empty |

Not proved: accumulation of a whole sequence of terms (the induction over
`Acc.add` calls is not carried out; only the per-insertion invariant is
proved), `finalize` (`blackwell.rs:115`) and its final rounding of
`pos - neg` to `f64`, and the `is_finite`/zero fast path.

### `G6ME.Dng` — `dng.rs:2475,2503,1629`

| Theorem | Rust | Statement |
|---|---|---|
| `raw10_group_layout`, `unpackGroup_cons`, `base4_field0..3`, `div4_mul4_add_mod4`, `unpackGroup_packed` | `dng.rs:2475` `pack_mipi_raw10` | the five-byte group layout, `p / 4` in bytes 0..3, the four 2-bit LSB fields |
| `raw10_group_roundtrip`, `raw10_group_roundtrip_append` | `dng.rs:2475/2503` | unpack-after-pack is the identity for values `< 1024` |
| `raw10_group_lsb_fields` | `dng.rs:2475` | the LSB byte is `p0%4 + 4*(p1%4) + 16*(p2%4) + 64*(p3%4)` |
| `packGroup_length`, `raw10_group_byte_bounds`, `raw10_group_unpack_bounds` | `dng.rs:2475/2503` | five bytes per group, every byte in `[0,255]` |
| `packRow_eq_flatMap`, `packRow_length`, `raw10_row_roundtrip(_unbounded)`, `raw10_row_byte_bounds` | `dng.rs:2475/2503` | row-level packing and roundtrip |
| `packFrame_eq_flatMap`, `padRow_length`, `take_packRow`, `drop_padRow`, `raw10_unpack_ignores_padding`, `raw10_frame_roundtrip`, `packFrame_length`, `raw10_frame_byte_bounds` | `dng.rs:2475/2503` | frame-level packing, stride padding ignored by the decoder, roundtrip and bounds |
| `pack16_le_roundtrip`, `pack16_be_roundtrip`, `pack16_le_byte_bounds`, `pack16_be_byte_bounds`, `unpack16_le_bounds`, `unpack16_be_bounds` | `dng.rs:1629` `unpack_uncompressed_block` (`bps = 16`) | little- and big-endian 16-bit sample packing |

Not proved: the `f64` black-level/white-level normalisation
(`((p as f64) - black) * inv_span`), the `& 0x3FF` masks (the identity on the
`< 1024` input domain), the 8-bit and packed 10/12/14-bit branches, lossless
JPEG (`dng.rs:1316/1496`) and every other codec in the file.

### `G6ME.Stn` — `stn.rs:52,72`

| Theorem | Rust | Statement |
|---|---|---|
| `stn_masks_nonneg_residual` | `stn.rs:72` `fuzzy_masks` | `fs + ft <= D`: the signal and transient supports are disjoint |
| `stn_masks_unity`, `stn_masks_unity_full` | `stn.rs:72` | `fs + ft + (D - fs - ft) = D` |
| `maskTriple_sum`, `noiseTriple_sum` | `stn.rs:72` | the scaled triple and the `denom <= 1e-30` fast-path triple sum to `D` |
| `fuzzy_masks_unity`, `fuzzy_masks_zero_denom` | `stn.rs:72` | both branches of the model sum to the common denominator |

The model is explicitly rational: `x/D` stands for the `f64` value; `D <= 2*c`
is the Rust `beta_l >= 0.5` clamp; the zero-support hypotheses abstract
`fuzzy_sin2` (`stn.rs:52`) returning exactly `0.0` at or below `beta_l`.
Not proved: the `sin^2` transition between the thresholds, the Hann COLA
window (`stn.rs:111`), and the STN filter itself.

### `G6ME.Fdn` — `room.rs:321` `ScatteringFdn::paraunitary_matrix`

| Theorem | Rust | Statement |
|---|---|---|
| `vRaw_norm_sq` | `room.rs:321` | `v_raw · v_raw = 10041` (the `sqrt` denominator, squared) |
| `hadamard_gram` | `room.rs:321` | `H8ᵀ H8 = 8 I` |
| `householder_gram` | `room.rs:321` | `R'ᵀ R' = 10041² I` |
| `fdn_gram` | `room.rs:321` | `U'ᵀ U' = 8 * 10041² I`, i.e. `Uᵀ U = I` after dividing by `(10041*sqrt 8)²` |
| `hadamard_transpose_mul`, `householder_symmetric` | `room.rs:321` | `H8 H8ᵀ = 8 I`, `R'` symmetric |

All three Gram identities are kernel computations on explicit integer
matrices (`decide`).  Not proved: the `f64` division by `sqrt 8` and by
`||v_raw||`, i.e. no statement is made about floating-point rounding.

### `G6ME.Limiter` — `limiter.rs:83,86,236,249,261,283,427`

| Theorem | Rust | Statement |
|---|---|---|
| `ceiling_in_range` | `limiter.rs:83,86,427` | the clamped ceiling is in `[MIN_CEILING_DB, MAX_CEILING_DB]` |
| `gain_num_le`, `gain_bounds` | `limiter.rs:283` `tick` | the convex blend `w*target + (D-w)*prev` over `D` lies in `[0,D]` |
| `gain_full_weight`, `gain_zero_weight` | `limiter.rs:283` | the blend endpoints are the target and the previous gain |
| `limiter_output_bound`, `limiter_output_cross` | `limiter.rs:283` | a gain capped at `ceiling/peak` keeps the output at or below the ceiling |
| `adaa2_delta_passband`, `adaa2_passband`, `adaa2_passband_delta_zero` | `limiter.rs:236,249,261,283` | inside `[-t0,t0]` the residual and its antiderivatives are zero and `tick` returns the delayed sample |
| `adaa2_div_guard`, `adaa2_div_guard_cur`, `adaa2_div_guard_prev` | `limiter.rs:236-283` | `|a-b| > EPS > 0` implies `a-b ≠ 0`, so the ADAA2 divisions are guarded |

Decibels are modelled as whole `Int` dB.  Not proved: `f64` rounding, the
fractional-dB value, the saturated `sqrt`/`asinh` branch of `delta`/`d1`/`d2`
(it is an arbitrary function `sat`/`slow` in the model), attack/release
coefficients, lookahead alignment, and the true-peak meter.

### `G6ME.Hoa` — `hoa.rs:97,144,176`

| Theorem | Rust | Statement |
|---|---|---|
| `legLoop_at_one`, `legLoop_at_minus_one` | `hoa.rs:97` `assoc_legendre` (`m = 0`) | the literal integer-division loop gives `P_l(1)=1` and `P_l(-1)=(-1)^l` |
| `legLoop_two_two`, `legLoop_recurrence_fails_at_two` | `hoa.rs:97` | at `l=2, x=2` the loop truncates `11/2` to `5`; the cross-multiplied recurrence is therefore **false** in general |
| `legLoop_recurrence_iff`, `legLoop_recurrence` | `hoa.rs:97` | the recurrence holds exactly when the numerator is divisible by `l+2` |
| `legendre_recurrence` | `hoa.rs:97` | the recurrence in the exact rational model `LegVal` (cross-multiplied, no division) |
| `legendre_at_one`, `legendre_at_minus_one` | `hoa.rs:97` | the endpoint values in the rational model |
| `hoa_channel_count`, `hoa_order7_channels` | `hoa.rs:144` `eval_sh_order7` | `sum_{n=0}^{7}(2n+1) = 64` |
| `hoa_acn_pairs_length`, `hoa_acn_index_range`, `hoa_acn_index_injective`, `hoa_acn_index_range_mem`, `hoa_acn_index_inj` | `hoa.rs:144` | the 64 ACN pairs, the flat index `n*n+n+m`, range `[0,64)` and injectivity |
| `taper_unit_energy_eq`, `taper_foldl_scaled`, `taper_scaled_energy`, `taper_energy_identity` | `hoa.rs:176` `hold_sh_taper_weights` | if `comp² * e_raw = e_unit` then the tapered degree-weighted energy is `e_unit` |

Not proved: orthonormality of the spherical harmonics (an integral identity
requiring real analysis and trigonometry), the N3D normalisation values in
`eval_sh_order7`, and every `sqrt`/`sin`/`cos` value.  The literal `Int`
division loop is *not* a Legendre polynomial at every integer point; that
discrepancy is proved, not hidden.

### `G6ME.Burst3d` — `burst3d.rs:52,57,77`

| Theorem | Rust | Statement |
|---|---|---|
| `K_skew` | `burst3d.rs:52` `skew` | `Kᵀ = -K` |
| `K_sq` | `burst3d.rs:57` `so3_exp` | `K² = w wᵀ - θ² I` |
| `K_cube` | `burst3d.rs:57` | `K³ = -θ² K` |
| `K_mul_w`, `K_ww_zero`, `ww_K_zero`, `ww_ww` | `burst3d.rs:57` | the axis is in the kernel; `(w wᵀ)² = θ² w wᵀ` |
| `R_sub_transpose` | `burst3d.rs:77` `so3_log` | `R - Rᵀ = 2 a K`, the matrix the log reads |
| `R_mul_w`, `R_zero` | `burst3d.rs:57` | `R w = w`; `R = I` when `w = 0` |
| `log_roundtrip`, `log_roundtrip_w0/w1/w2` | `burst3d.rs:77` | if `scale * a = 1`, the scaled doubled log vector is `2 w` (componentwise) |
| `R_gram_conditional` | `burst3d.rs:57` `so3_exp` | `Rᵀ R = I` entrywise, from the two scalar side conditions `2b - a² - b²θ² = 0` and `(1-bθ²)² + a²θ² = 1` |

**Conditionally proved:** `R_gram_conditional` proves `Rᵀ R = I` from the
two scalar side conditions `2b - a² - b²θ² = 0` and `(1-bθ²)² + a²θ² = 1`.
The expansion behind it is

`Rᵀ R = ((1-bθ²)² + a²θ²) I + (2b - a² - b²θ²) w wᵀ`,

so those conditions are exactly what is needed; with `a` and `b` free — as
they are in this integer model — the unconditional statement is false.  The
Rust computes `a = sin θ / θ` and `b = (1 - cos θ) / θ²`; the side
conditions are then the trigonometric identity `sin² + cos² = 1`, which
needs real analysis, so the trigonometric instantiation is not proved here.
`det R = 1` likewise depends on the trigonometric definitions.  Also not
modelled: `f64` rounding, `sqrt`, `sin`, `cos`, `acos`, the `clamp` calls,
and the `|φ| < 1e-7` Taylor branch of `so3_exp`/`so3_log` (whose
coefficients do not satisfy the exact orthogonality relation).

### `G6ME.Upmix` — `upmix.rs:149` `pae_decompose_bin`

| Theorem | Rust | Statement |
|---|---|---|
| `mul_comm`, `mul_assoc`, `conj_mul`, `conj_conj`, `conj_neg`, `mul_neg_left` | `upmix.rs:149` | the complex-number ring laws used by the bin processing |
| `normSq_conj`, `normSq_neg`, `normSq_mul`, `normSq_add`, `normSq_scale`, `normSq_conj_mul`, `normSq_neg_mul` | `upmix.rs:149` | `|z|²` identities, including multiplicativity |
| `dot_eq_re_mul_conj`, `v2_ip_zero`, `v2_norm` | `upmix.rs:149` | the constructed second eigenvector is orthogonal to the first and has the same norm |
| `cross_eq`, `dot_neg_left`, `cross_neg` | `upmix.rs:149` | the two cross terms cancel |
| `parseval_specific`, `parseval_unit` | `upmix.rs:149` | `|proj₁|² + |proj₂|² = (|p|²+|q|²)(|x0|²+|x1|²)`, and the unit-vector corollary |
| `normSq_rotPlus`, `normSq_rotMinus` | `upmix.rs:149` | the APES quadrature rotations preserve the norm |
| `primary_energy`, `rescale_energy`, `ambient_rescale_conservation`, `zero_branch_energy` | `upmix.rs:149` | primary energy is `w_p²|proj|²`; if `s² e_cur = e_in - e_p` the rescaled ambient energy makes the total exactly `e_in` |

Not modelled: the `f64` EVD (`disc = sqrt((r11-r22)² + 4|r12|²)`, `λ1`,
`λ2`), the clamping of `p_frac` and `centerness`, the phase division
`r12 / |r12|`, and the ambient cross-term before the final rescale.  Those are
`f64`/`sqrt`/`clamp` computations, not polynomial identities.

## Consolidated list of what is NOT proved

1. **No floating-point semantics anywhere.**  Lean's `Float` is not IEEE-754,
   so no theorem in this development claims anything about `f64` rounding,
   NaN, infinities or denormals.  The `f64` code is modelled by exact
   `Int`/`Nat`/rational abstractions (`x/D`, cleared denominators, doubled
   values), and each module header says exactly which affine/rounding steps
   are omitted.
2. **Goldilocks:** primality of `p`; general-`N` NTT correctness; root-of-unity
   and twiddle theory; `exact_int_convolve` end to end.
3. **Blackwell:** `ldexp_exact`; tensor-core execution; Ozaki-II double-double
   reconstruction (`ozaki2_crt_gemm`); Kulisch multi-term accumulation and the
   final `finalize` rounding.
4. **DNG:** black/white-level normalisation; `& 0x3FF`; 8-bit and packed
   10/12/14-bit paths; lossless JPEG.
5. **STN / Limiter / HOA:** `sin²`, `sqrt`, `asinh`, `clamp`, and all
   fractional-dB / real-trigonometry values.
6. **Burst3d:** the trigonometric instantiation of `RᵀR = I`
   (`a = sin θ / θ`, `b = (1 - cos θ) / θ²`) and `det R = 1` (see above);
   Taylor branch.
7. **Upmix:** the `f64` eigenvalue solve, `sqrt`, `clamp`, phase division.
8. **PCHIP:** not formalized because PCHIP does not exist in this repository;
   it belongs to HyprEQ, a different project, and was deliberately omitted.

## Purity

`check_axioms.sh` enforces the gate in four phases.  (1) The source-token scan
strips nested block comments and line comments first, because the module
headers name the forbidden tokens in prose.  (2) `lake build`.  (3) A coverage
check enumerates every public `theorem`/`lemma` declared at column 0 in
`G6ME/*.lean` and `G6ME.lean`, resolves its namespace, and requires the result
to equal the EXPECTED registry in the script exactly: a declared theorem that
is missing from EXPECTED fails, and an EXPECTED name with no declaration
fails.  Private and protected declarations are not matched (Lean mangles
private names, so `#print axioms` cannot address them); the sources currently
contain none.  (4) The `#print axioms` probe is generated from EXPECTED itself,
so the registry is the single source of truth, and the script prints every
axiom line it checks and fails unless every line contains only `propext` and
`Quot.sound`.  The registry currently covers 266 theorems, matching the
source enumeration exactly.
