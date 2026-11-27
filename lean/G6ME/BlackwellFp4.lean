/-
  G6ME.BlackwellFp4 — exact Base-13 FP4 (E2M1) limb encoding.

  Models `crates/g6me-core/src/blackwell.rs`:

  * `is_exact_fp4_e2m1`
  * `encode_base13_fp4_e2m1`
  * the `TensorBackend::Fp4E2m1Base13` branch of `ozaki2_crt_gemm`
    (the 2-limb product recombination)

  The floating-point values of E2M1 are `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
  We work with the *doubled* magnitudes `D = {0, ±1, ±2, ±3, ±4, ±6, ±8}`
  which is exactly the set the Rust code calls a complete residue system
  modulo 13.  `encodeD` returns the doubled limbs `(d0, d1)`; the Rust function
  returns `(d0/2, d1/2)`.  No `Float` reasoning is used.
-/

import G6ME.Basic

set_option maxRecDepth 10000

namespace G6ME.BlackwellFp4

/-- The complete residue system modulo 13 used by the Rust encoder. -/
def remToD : List Int := [0, 1, 2, 3, 4, -8, 6, -6, 8, -4, -3, -2, -1]

/-- Membership in the doubled E2M1 magnitude set `{0, ±1, ±2, ±3, ±4, ±6, ±8}`. -/
def isD (d : Int) : Bool :=
  d == 0 || d == 1 || d == -1 || d == 2 || d == -2 || d == 3 || d == -3 ||
  d == 4 || d == -4 || d == 6 || d == -6 || d == 8 || d == -8

/-- `encode_base13_fp4_e2m1` with limbs kept doubled.
`encodeD r = (d0, d1)` satisfies `r = d0 + 13*d1`; the Rust values are
`d0/2` and `d1/2`. -/
def encodeD (r : Int) : Int × Int :=
  let rem := r % 13
  let d0 := remToD.getD rem.toNat 0
  let d1 := (r - d0) / 13
  (d0, d1)

/-- The doubled E2M1 set is a complete residue system modulo 13: the table
entries reduce to `0, 1, ..., 12` in order. -/
theorem remToD_residues :
    (List.range 13).all
      (fun k => (remToD.getD k 0) % 13 == (k : Int)) = true := by
  decide

theorem all_range {n : Nat} {p : Nat → Bool}
    (h : (List.range n).all p = true) : ∀ i, i < n → p i = true := by
  intro i hi
  exact (List.all_eq_true.mp h) i (List.mem_range.mpr hi)

/-- Every residue `r % 13` selects a table entry congruent to it. -/
theorem remToD_correct (r : Int) (h0 : 0 ≤ r % 13) (_h1 : r % 13 < 13) :
    (remToD.getD (r % 13).toNat 0) % 13 = r % 13 := by
  have hk : (r % 13).toNat < 13 := by
    have : ((r % 13).toNat : Int) = r % 13 := Int.toNat_of_nonneg h0
    omega
  have hmem := all_range remToD_residues (r % 13).toNat hk
  have hcast : ((r % 13).toNat : Int) = r % 13 := Int.toNat_of_nonneg h0
  rw [hcast] at hmem
  exact beq_iff_eq.mp hmem

/-- The table stays inside the doubled E2M1 set. -/
theorem remToD_mem :
    (List.range 13).all (fun k => isD (remToD.getD k 0)) = true := by
  decide

theorem remToD_mem_getD (k : Nat) (hk : k < 13) : isD (remToD.getD k 0) = true :=
  all_range remToD_mem k hk

/-- Membership in `D` implies the residue is one of the 13 distinct values:
the canonical representative of `d % 13` is unique inside `D`. -/
theorem isD_residue_injective : ∀ a ∈ remToD, ∀ b ∈ remToD,
    a % 13 = b % 13 → a = b := by
  decide

/-- The encoder reconstructs the residue exactly, for the full supported
range `[-56, 56]` of `ozaki2_crt_gemm`. -/
theorem encodeD_reconstruct (r : Int) (hr0 : -56 ≤ r) (hr1 : r ≤ 56) :
    (encodeD r).1 + 13 * (encodeD r).2 = r := by
  -- finite check over the 113 supported residues, then transport along the range
  have hcheck :
      (List.range 113).all
        (fun i => (encodeD ((i : Int) - 56)).1 + 13 * (encodeD ((i : Int) - 56)).2
          == ((i : Int) - 56)) = true := by
    decide
  have hlt : (r + 56).toNat < 113 := by
    have hcast : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  have hmem := all_range hcheck (r + 56).toNat hlt
  have hcast : (((r + 56).toNat : Int) - 56) = r := by
    have h2 : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  rw [hcast] at hmem
  exact beq_iff_eq.mp hmem

/-- Both encoded limbs lie in the doubled E2M1 set. -/
theorem encodeD_mem (r : Int) (hr0 : -56 ≤ r) (hr1 : r ≤ 56) :
    isD (encodeD r).1 = true ∧ isD (encodeD r).2 = true := by
  have hcheck :
      (List.range 113).all
        (fun i => isD (encodeD ((i : Int) - 56)).1
          && isD (encodeD ((i : Int) - 56)).2) = true := by
    decide
  have hlt : (r + 56).toNat < 113 := by
    have hcast : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  have hmem := all_range hcheck (r + 56).toNat hlt
  have hcast : (((r + 56).toNat : Int) - 56) = r := by
    have h2 : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  rw [hcast] at hmem
  simpa using hmem

/-- The Base-13 encoding is unique: any two representations of `r` by doubled
E2M1 limbs coincide. -/
theorem encodeD_unique (r d0 d1 : Int) (hr0 : -56 ≤ r) (hr1 : r ≤ 56)
    (hd0 : isD d0 = true) (_hd1 : isD d1 = true) (hrep : d0 + 13 * d1 = r) :
    (d0, d1) = encodeD r := by
  have henc_rec := encodeD_reconstruct r hr0 hr1
  have henc_mem := encodeD_mem r hr0 hr1
  -- both d0 and the encoded first limb are congruent to r modulo 13
  have hd0_mem : d0 ∈ remToD := by
    -- `isD` is the same disjunction as list membership
    simp [isD, remToD] at hd0 ⊢
    omega
  have he0_mem : (encodeD r).1 ∈ remToD := by
    have hthis : isD (encodeD r).1 = true := henc_mem.1
    simp [isD, remToD] at hthis ⊢
    omega
  have hcong : d0 % 13 = (encodeD r).1 % 13 := by
    have h13d1 : (13 * d1) % 13 = 0 :=
      Int.emod_eq_zero_of_dvd (Int.dvd_mul_right 13 d1)
    have h13e1 : (13 * (encodeD r).2) % 13 = 0 :=
      Int.emod_eq_zero_of_dvd (Int.dvd_mul_right 13 (encodeD r).2)
    have h1 : r % 13 = d0 % 13 := by
      rw [← hrep]
      rw [Int.add_emod]
      rw [h13d1]
      rw [Int.add_zero]
      rw [Int.emod_emod]
    have h2 : r % 13 = (encodeD r).1 % 13 := by
      conv => lhs; rw [← henc_rec]
      rw [Int.add_emod]
      rw [h13e1]
      rw [Int.add_zero]
      rw [Int.emod_emod]
    rw [← h1, ← h2]
  have h0 : d0 = (encodeD r).1 := isD_residue_injective d0 hd0_mem (encodeD r).1 he0_mem hcong
  have h1 : d1 = (encodeD r).2 := by
    rw [h0] at hrep
    have hmul : 13 * d1 = 13 * (encodeD r).2 := by omega
    exact Int.eq_of_mul_eq_mul_left (by decide : (13 : Int) ≠ 0) hmul
  rw [h0, h1]

/-- The doubled-limb product identity: writing `a = d0 + 13 d1` and
`b = e0 + 13 e1`, the product expands in base 13 exactly. -/
theorem base13_product_identity (d0 d1 e0 e1 : Int) :
    (d0 + 13 * d1) * (e0 + 13 * e1)
      = d0 * e0 + 13 * (d0 * e1 + d1 * e0) + 169 * (d1 * e1) := by
  ring_int

/-- The exact integer content of the `Fp4E2m1Base13` GEMM branch:
with the Rust variables `a = 2*a0 + 26*a1`, `b = 2*b0 + 26*b1`,
the accumulated expression `4*p00 + 13*(4*(p01+p10)) + 169*(4*p11)`
equals `a*b`. -/
theorem base13_gemm_identity (a0 a1 b0 b1 : Int) :
    (2 * a0 + 26 * a1) * (2 * b0 + 26 * b1)
      = 4 * (a0 * b0) + 13 * (4 * (a0 * b1 + a1 * b0)) + 169 * (4 * (a1 * b1)) := by
  ring_int

/-- FP8 E4M3 two-limb exact-integer decomposition identity: every integer
`r = lo + 16*hi` with `|lo|, |hi| <= 8` is an exact sum of two E4M3-representable
integers, and the product expands exactly.  (The bound hypotheses document the
range in which the Rust rounding produces such limbs; no `Float` semantics are
claimed.) -/
theorem fp8_product_identity (lo hi lo' hi' : Int) :
    (lo + 16 * hi) * (lo' + 16 * hi') =
      lo * lo' + 16 * (lo * hi' + hi * lo') + 256 * (hi * hi') := by
  ring_int

/-- Range fact behind the FP8 split: for `|r| <= 56` the nearest-integer
`hi` to `r/16` satisfies `|hi| <= 4` and the residual `r - 16*hi` lies in
`[-8, 8]`.  Stated as an existence theorem over the integers (the Rust
`f64::round` is the source of `hi`, and is not modelled). -/
theorem fp8_split_exists (r : Int) (hr0 : -56 ≤ r) (hr1 : r ≤ 56) :
    ∃ hi lo : Int, -4 ≤ hi ∧ hi ≤ 4 ∧ -8 ≤ lo ∧ lo ≤ 8 ∧ r = lo + 16 * hi := by
  -- explicit nearest-integer split, checked over the finite supported range
  have hcheck :
      (List.range 113).all
        (fun i =>
          let r : Int := (i : Int) - 56
          let hi := (r + 8) / 16
          let lo := r - 16 * hi
          decide (-4 ≤ hi ∧ hi ≤ 4 ∧ -8 ≤ lo ∧ lo ≤ 8)) = true := by
    decide
  have hlt : (r + 56).toNat < 113 := by
    have hcast : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  have hmem := all_range hcheck (r + 56).toNat hlt
  have hcast : (((r + 56).toNat : Int) - 56) = r := by
    have h2 : ((r + 56).toNat : Int) = r + 56 := Int.toNat_of_nonneg (by omega)
    omega
  rw [hcast] at hmem
  have hmem' : -4 ≤ (r + 8) / 16 ∧ (r + 8) / 16 ≤ 4 ∧
      -8 ≤ r - 16 * ((r + 8) / 16) ∧ r - 16 * ((r + 8) / 16) ≤ 8 := by
    simpa using hmem
  exact ⟨(r + 8) / 16, r - 16 * ((r + 8) / 16), hmem'.1, hmem'.2.1,
    hmem'.2.2.1, hmem'.2.2.2, by omega⟩

end G6ME.BlackwellFp4
