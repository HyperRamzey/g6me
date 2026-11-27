/-
  G6ME.GoldilocksNtt — field operations, conversions, and a concrete NTT.

  Models `crates/g6me-core/src/ntt.rs`:

  * `mod_add`, `mod_sub`, `mod_mul` (via `reduce128`)
  * `i64_to_field`, `field_to_i128`
  * the radix-2 butterfly of `NttPlan::transform`
  * the size-2 forward/inverse NTT pipeline (`ntt2` / `intt2`), the concrete
    case of `exact_int_convolve` in which the transform is a single butterfly

  The size-2 roundtrip is proved outright.  General-`N` Cooley–Tukey
  correctness would require a theory of roots of unity and bit-reversal
  permutations; that is not attempted here and is stated as not proved in the
  README.
-/

import G6ME.Goldilocks

set_option maxRecDepth 10000

namespace G6ME.Goldilocks

/-! ## The three field operations -/

/-- `ntt.rs::mod_add`. -/
def modAdd (a b : Nat) : Nat :=
  if a + b ≥ P then a + b - P else a + b

/-- `ntt.rs::mod_sub`. -/
def modSub (a b : Nat) : Nat :=
  if a ≥ b then a - b else P - (b - a)

/-- `ntt.rs::mod_mul`. -/
def modMul (a b : Nat) : Nat := reduce128 (a * b)

theorem P_lt_two64 : P < 2 ^ 64 := by decide

theorem P_lt_two128 : P < 2 ^ 128 := by decide

theorem modAdd_spec {a b : Nat} (ha : a < P) (hb : b < P) :
    modAdd a b = (a + b) % P := by
  unfold modAdd
  by_cases h : a + b ≥ P
  · rw [if_pos h]
    have hsub : a + b - P < P := by omega
    rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt hsub]
  · rw [if_neg h]
    exact (Nat.mod_eq_of_lt (by omega)).symm

theorem modAdd_lt {a b : Nat} (ha : a < P) (hb : b < P) : modAdd a b < P := by
  rw [modAdd_spec ha hb]
  exact Nat.mod_lt _ P_pos

theorem modSub_spec {a b : Nat} (ha : a < P) (hb : b < P) :
    modSub a b = (a + (P - b)) % P := by
  unfold modSub
  by_cases h : a ≥ b
  · rw [if_pos h]
    have hb_le : b ≤ a := by omega
    have hsum : a + (P - b) ≥ P := by omega
    rw [Nat.mod_eq_sub_mod hsum]
    have hsub : a + (P - b) - P = a - b := by omega
    rw [hsub, Nat.mod_eq_of_lt (by omega)]
  · rw [if_neg h]
    have hlt : a < b := by omega
    have hsum : a + (P - b) < P := by omega
    rw [Nat.mod_eq_of_lt hsum]
    omega

theorem modSub_lt {a b : Nat} (ha : a < P) (hb : b < P) : modSub a b < P := by
  rw [modSub_spec ha hb]
  exact Nat.mod_lt _ P_pos

theorem mul_lt_two128 {a b : Nat} (ha : a < P) (hb : b < P) : a * b < 2 ^ 128 := by
  have h1 : a * b < P * P := Nat.mul_lt_mul'' ha hb
  have h2 : P * P < 2 ^ 128 := by decide
  omega

theorem modMul_spec {a b : Nat} (ha : a < P) (hb : b < P) :
    modMul a b = (a * b) % P := by
  unfold modMul
  exact reduce128_spec (a * b) (mul_lt_two128 ha hb)

theorem modMul_lt {a b : Nat} (ha : a < P) (hb : b < P) : modMul a b < P := by
  rw [modMul_spec ha hb]
  exact Nat.mod_lt _ P_pos

theorem modMul_one {a : Nat} (ha : a < P) : modMul a 1 = a := by
  unfold modMul
  rw [Nat.mul_one, reduce128_spec a (Nat.lt_trans ha P_lt_two128),
    Nat.mod_eq_of_lt ha]

/-! ## Signed conversions -/

/-- `ntt.rs::i64_to_field`, expressed with `Int.emod` (identical to the Rust
branching for every `i64`; the result is the canonical residue in `[0, p)`). -/
def i64ToField (v : Int) : Nat := (v % (P : Int)).toNat

theorem i64ToField_lt (v : Int) : i64ToField v < P := by
  unfold i64ToField
  have h0 : 0 ≤ v % (P : Int) := Int.emod_nonneg v (by decide)
  have h1 : v % (P : Int) < (P : Int) := Int.emod_lt_of_pos v (by decide)
  have hcast : ((v % (P : Int)).toNat : Int) = v % (P : Int) := Int.toNat_of_nonneg h0
  omega

/-- `ntt.rs::field_to_i128`: balanced representative in `[-(p-1)/2, (p-1)/2]`. -/
def fieldToI128 (v : Nat) : Int :=
  if v ≤ P / 2 then (v : Int) else (v : Int) - (P : Int)

theorem fieldToI128_range (v : Nat) (hv : v < P) :
    -(((P / 2 : Nat)) : Int) ≤ fieldToI128 v ∧
      fieldToI128 v ≤ ((P / 2 : Nat) : Int) := by
  unfold fieldToI128
  have hP2 : P = 2 * (P / 2) + 1 := by decide
  by_cases h : v ≤ P / 2
  · rw [if_pos h]
    constructor
    · omega
    · exact Int.ofNat_le.mpr h
  · rw [if_neg h]
    have hge : (P / 2 : Nat) + 1 ≤ v := by omega
    constructor
    · omega
    · omega

/-- The balanced `i128` conversion inverts `i64_to_field` on the balanced range. -/
theorem fieldToI128_i64ToField (v : Int)
    (hlo : -(((P / 2 : Nat)) : Int) ≤ v) (hhi : v ≤ ((P / 2 : Nat) : Int)) :
    fieldToI128 (i64ToField v) = v := by
  have hP2 : P = 2 * (P / 2) + 1 := by decide
  have hbound : v < (P : Int) := by omega
  have hlow : -(P : Int) < v := by omega
  have h0 : 0 ≤ v % (P : Int) := Int.emod_nonneg v (by decide)
  have h1 : v % (P : Int) < (P : Int) := Int.emod_lt_of_pos v (by decide)
  have hcast : (((v % (P : Int)).toNat : Int)) = v % (P : Int) := Int.toNat_of_nonneg h0
  unfold fieldToI128 i64ToField
  by_cases hv : 0 ≤ v
  · have hr : v % (P : Int) = v := Int.emod_eq_of_lt hv hbound
    rw [hr]
    have htoNat : v.toNat ≤ P / 2 := by
      have : (v.toNat : Int) = v := Int.toNat_of_nonneg hv
      omega
    rw [if_pos htoNat]
    exact Int.toNat_of_nonneg hv
  · have hvneg : v < 0 := by omega
    -- for `-p < v < 0`, `v % p = v + p`
    have hr : v % (P : Int) = v + (P : Int) := by
      have hadd : 0 ≤ v + (P : Int) := by omega
      have haddlt : v + (P : Int) < (P : Int) := by omega
      have hmod : (v + (P : Int)) % (P : Int) = v + (P : Int) :=
        Int.emod_eq_of_lt hadd haddlt
      have hshift : (v + (P : Int)) % (P : Int) = v % (P : Int) := by
        rw [Int.add_emod, Int.emod_self, Int.add_zero, Int.emod_emod]
      rw [← hshift]
      exact hmod
    rw [hr]
    have hge : ¬ ((v + (P : Int)).toNat ≤ P / 2) := by
      have hcast' : (((v + (P : Int)).toNat : Int)) = v + (P : Int) :=
        Int.toNat_of_nonneg (by omega)
      omega
    rw [if_neg hge]
    have hcast' : (((v + (P : Int)).toNat : Int)) = v + (P : Int) :=
      Int.toNat_of_nonneg (by omega)
    omega

/-! ## The radix-2 butterfly and the size-2 NTT -/

/-- One `NttPlan::transform` butterfly:
`(u + v*w, u - v*w)` in the field. -/
def butterfly (u v w : Nat) : Nat × Nat :=
  let t := modMul v w
  (modAdd u t, modSub u t)

/-- Size-2 forward NTT: a single butterfly with twiddle `w = 1`. -/
def ntt2 (x0 x1 : Nat) : Nat × Nat := butterfly x0 x1 1

/-- `2^{-1} mod p`, the size-2 inverse scale `inv_n`. -/
def inv2 : Nat := 9223372034707292161

theorem inv2_lt : inv2 < P := by decide

theorem inv2_mul_two : (2 * inv2) % P = 1 := by decide

/-- Scaling by `inv2` after doubling returns the input. -/
theorem two_mul_inv2_mod (z : Nat) (hz : z < P) : (2 * z * inv2) % P = z := by
  rw [show 2 * z * inv2 = z * (2 * inv2) by ring_nat]
  rw [Nat.mul_mod, inv2_mul_two, Nat.mul_one, Nat.mod_mod, Nat.mod_eq_of_lt hz]

/-- Size-2 inverse NTT: the same butterfly (twiddle `1`) followed by the
`n^{-1}` scaling, exactly as `NttPlan::inverse` does. -/
def intt2 (y : Nat × Nat) : Nat × Nat :=
  let b := butterfly y.1 y.2 1
  (modMul b.1 inv2, modMul b.2 inv2)

theorem meq_mod (a : Nat) : MEq P (a % P) a := by
  unfold MEq
  rw [Nat.mod_mod]

theorem butterfly_inverse_fst (u v : Nat) (hu : u < P) (hv : v < P) :
    modMul (modAdd (modAdd u v) (modSub u v)) inv2 = u := by
  have hs : modAdd u v < P := modAdd_lt hu hv
  have hd : modSub u v < P := modSub_lt hu hv
  rw [modMul_spec (modAdd_lt hs hd) inv2_lt, modAdd_spec hs hd,
    modSub_spec hu hv]
  rw [modAdd_spec hu hv]
  have hstep : ((((u + v) % P + (u + (P - v)) % P) % P) * inv2) % P
      = (((u + v) + (u + (P - v))) * inv2) % P := by
    rw [Nat.mul_mod, Nat.mod_mod, ← Nat.add_mod, ← Nat.mul_mod]
  rw [hstep]
  have harith : (u + v) + (u + (P - v)) = 2 * u + P := by omega
  rw [harith]
  have hsplit : (2 * u + P) * inv2 = (2 * u) * inv2 + P * inv2 := by ring_nat
  rw [hsplit, mod_add_mul_left]
  exact two_mul_inv2_mod u hu

/-- The second component of the inverse butterfly, in the general form needed
for `modSub`'s Nat subtraction: any `s ≡ u+v` and `d ≡ u+(p-v)` work. -/
theorem butterfly_inverse_snd_aux (u v s d : Nat) (_hu : u < P) (hv : v < P)
    (hslt : s < P) (hd : d < P) (hs : MEq P s (u + v))
    (hdd : MEq P d (u + (P - v))) :
    modMul (modSub s d) inv2 = v := by
  rw [modMul_spec (modSub_lt hslt hd) inv2_lt, modSub_spec hslt hd]
  -- `((s + (P-d)) % P * inv2) % P = v`
  have hmeq : MEq P (s + (P - d)) (2 * v) := by
    apply cancel_add_mod P (s + (P - d)) (2 * v) d P_pos
    -- both sides plus `d` reduce to `s % P`
    have hL : ((s + (P - d)) + d) % P = s % P := by
      have hPd : s + (P - d) + d = s + P := by
        have hd' : P - d + d = P := Nat.sub_add_cancel (Nat.le_of_lt hd)
        omega
      rw [hPd, mod_add_self_right]
    have hR : (2 * v + d) % P = s % P := by
      have h2d : (2 * v + d) % P = (2 * v + (u + (P - v))) % P := by
        rw [Nat.add_mod (2 * v) d P, Nat.add_mod (2 * v) (u + (P - v)) P, hdd]
      rw [h2d]
      have harith : 2 * v + (u + (P - v)) = (u + v) + P := by omega
      rw [harith, Nat.add_comm (u + v) P, mod_add_self_left]
      exact hs.symm
    rw [hL, hR]
  have hmul : ((s + (P - d)) * inv2) % P = v := by
    have h := MEq.mul hmeq (meq_refl P inv2)
    unfold MEq at h
    rw [two_mul_inv2_mod v hv] at h
    exact h
  have hinner : ((s + (P - d)) % P * inv2) % P = ((s + (P - d)) * inv2) % P := by
    rw [Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod]
  rw [hinner]
  exact hmul

theorem butterfly_inverse_snd (u v : Nat) (hu : u < P) (hv : v < P) :
    modMul (modSub (modAdd u v) (modSub u v)) inv2 = v := by
  have hs : modAdd u v < P := modAdd_lt hu hv
  have hd : modSub u v < P := modSub_lt hu hv
  exact butterfly_inverse_snd_aux u v (modAdd u v) (modSub u v) hu hv hs hd
    (by rw [modAdd_spec hu hv]; exact meq_mod (u + v))
    (by rw [modSub_spec hu hv]; exact meq_mod (u + (P - v)))

/-- **Size-2 NTT roundtrip**: forward then inverse (including the `n^{-1}`
scaling) is the identity on canonical field elements. -/
theorem ntt2_roundtrip (x0 x1 : Nat) (h0 : x0 < P) (h1 : x1 < P) :
    intt2 (ntt2 x0 x1) = (x0, x1) := by
  have ht : modMul x1 1 = x1 := modMul_one h1
  have hb : ntt2 x0 x1 = (modAdd x0 x1, modSub x0 x1) := by
    unfold ntt2 butterfly
    rw [ht]
  rw [hb]
  unfold intt2 butterfly
  dsimp only
  rw [modMul_one (modSub_lt h0 h1)]
  exact Prod.ext (butterfly_inverse_fst x0 x1 h0 h1)
    (butterfly_inverse_snd x0 x1 h0 h1)

end G6ME.Goldilocks
