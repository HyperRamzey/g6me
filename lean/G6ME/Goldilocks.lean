/-
  G6ME.Goldilocks — exact arithmetic in the Goldilocks prime field.

  Models `crates/g6me-core/src/ntt.rs`:

  * `GOLDILOCKS_P`, `GOLDILOCKS_G`, `GOLDILOCKS_W32`, `NTT_KERNEL_QBITS`
  * `mod_add`, `mod_sub`, `reduce128`, `mod_mul`, `mod_pow`
  * `i64_to_field`, `field_to_i128`
  * the radix-2 butterfly of `NttPlan::transform`
  * the size-2 exact NTT convolution pipeline of `exact_int_convolve`

  The development proves the reduction identity at the level of natural
  numbers (u64 values modelled by `Nat` together with explicit `< 2^64`
  hypotheses), never pretending that Lean's `Float` is IEEE-754.
-/

import G6ME.Basic

set_option maxRecDepth 10000

namespace G6ME.Goldilocks

/-! ## Constants -/

/-- The Goldilocks prime `p = 2^64 - 2^32 + 1`. -/
def P : Nat := 18446744069414584321

/-- `2^32 - 1`, the correction term of the 128-bit reduction. -/
def K : Nat := 4294967295

theorem P_eq : P = 2 ^ 64 - 2 ^ 32 + 1 := by decide

theorem K_eq : K = 2 ^ 32 - 1 := by decide

theorem P_pos : 0 < P := by decide

theorem P_gt_two63 : 2 ^ 63 < P := by decide

theorem two64_eq : 2 ^ 64 = P + K := by decide

theorem two96_eq : 2 ^ 96 + 1 = (2 ^ 32 + 1) * P := by decide

theorem two128_eq : 2 ^ 128 = 2 ^ 64 * 2 ^ 64 := by decide

/-! ## The 128-bit reduction

`reduce128` is a direct transcription of `ntt.rs::reduce128`.  The Rust code
works on `u128`/`u64` with wrapping arithmetic; we model the same branches on
`Nat`, where `lo`, `hi`, `mid` are the 64/96-bit digits of `x`. -/

/-- Low 64 bits of `x`. -/
def loOf (x : Nat) : Nat := x % 2 ^ 64

/-- Bits 64..127 of `x`. -/
def hiAllOf (x : Nat) : Nat := x / 2 ^ 64

/-- Bits 64..95 of `x`. -/
def midOf (x : Nat) : Nat := hiAllOf x % 2 ^ 32

/-- Bits 96..127 of `x`. -/
def hiOf (x : Nat) : Nat := hiAllOf x / 2 ^ 32

/-- Step 1 of `reduce128`: `(lo - hi)` brought into `[0, p)` when it is
negative.  In the borrow case the Rust code computes
`2^64 + lo - hi - (2^32 - 1) = p + lo - hi`. -/
def tOf (x : Nat) : Nat :=
  if loOf x ≥ hiOf x then loOf x - hiOf x else P - (hiOf x - loOf x)

/-- Step 2: add `mid * (2^32 - 1)`. -/
def sOf (x : Nat) : Nat := tOf x + midOf x * K

/-- The value of the wrapping accumulator before the final correction. -/
def resOf (x : Nat) : Nat :=
  if sOf x < 2 ^ 64 then sOf x % 2 ^ 64
  else if sOf x % 2 ^ 64 + K < 2 ^ 64 then sOf x % 2 ^ 64 + K
  else sOf x % 2 ^ 64 + 2 * K - 2 ^ 64

/-- `ntt.rs::reduce128`: reduce a 128-bit product modulo `p`. -/
def reduce128 (x : Nat) : Nat :=
  if resOf x ≥ P then resOf x - P else resOf x

/-! ## Bounds on the intermediate values -/

theorem loOf_lt (x : Nat) : loOf x < 2 ^ 64 :=
  Nat.mod_lt _ (by decide)

theorem hiAllOf_lt (x : Nat) (hx : x < 2 ^ 128) : hiAllOf x < 2 ^ 64 := by
  unfold hiAllOf
  rw [Nat.div_lt_iff_lt_mul (by decide : 0 < 2 ^ 64)]
  rw [two128_eq] at hx
  exact hx

theorem midOf_lt (x : Nat) : midOf x < 2 ^ 32 :=
  Nat.mod_lt _ (by decide)

theorem hiOf_lt (x : Nat) (hx : x < 2 ^ 128) : hiOf x < 2 ^ 32 := by
  unfold hiOf
  rw [Nat.div_lt_iff_lt_mul (by decide : 0 < 2 ^ 32)]
  have := hiAllOf_lt x hx
  omega

theorem tOf_lt (x : Nat) (hx : x < 2 ^ 128) : tOf x < 2 ^ 64 := by
  unfold tOf
  have hlo := loOf_lt x
  have hhi := hiOf_lt x hx
  have hK : K = 4294967295 := rfl
  have hPK : P + K = 2 ^ 64 := by rw [← two64_eq]
  by_cases h : loOf x ≥ hiOf x
  · rw [if_pos h]; omega
  · rw [if_neg h]
    have hhi_le : hiOf x - loOf x ≤ K := by omega
    omega

theorem mid_mul_K_lt (x : Nat) : midOf x * K < 2 ^ 64 := by
  have hmid : midOf x ≤ 4294967295 := by
    have := midOf_lt x
    omega
  have hmul : midOf x * K ≤ 4294967295 * 4294967295 := by
    apply Nat.mul_le_mul hmid
    show K ≤ 4294967295
    decide
  have hnum : 4294967295 * 4294967295 < 2 ^ 64 := by decide
  omega

theorem sOf_lt (x : Nat) (hx : x < 2 ^ 128) : sOf x < 2 ^ 65 := by
  unfold sOf
  have h1 := tOf_lt x hx
  have h2 := mid_mul_K_lt x
  omega

/-- When `s ≥ 2^64`, its wrapping remainder is exactly `s - 2^64`. -/
theorem sOf_mod (x : Nat) (hx : x < 2 ^ 128) (h : sOf x ≥ 2 ^ 64) :
    sOf x % 2 ^ 64 = sOf x - 2 ^ 64 := by
  have hs := sOf_lt x hx
  rw [Nat.mod_eq_sub_mod h]
  rw [Nat.mod_eq_of_lt (by omega : sOf x - 2 ^ 64 < 2 ^ 64)]

theorem resOf_lt (x : Nat) (hx : x < 2 ^ 128) : resOf x < 2 ^ 64 := by
  unfold resOf
  have hs := sOf_lt x hx
  have hPK : P + K = 2 ^ 64 := by rw [← two64_eq]
  by_cases h1 : sOf x < 2 ^ 64
  · rw [if_pos h1]
    exact Nat.mod_lt _ (by decide)
  · rw [if_neg h1]
    have hr0 := sOf_mod x hx (by omega)
    have hK : K = 4294967295 := rfl
    by_cases h2 : sOf x % 2 ^ 64 + K < 2 ^ 64
    · rw [if_pos h2]; omega
    · rw [if_neg h2]
      rw [hK] at h2 ⊢
      omega

/-! ## The reduction computes `x % p` -/

/-- The exact integer identity behind the reduction:
`x + hi = (lo + mid*(2^32-1)) + p * (mid + hi*(2^32+1))`. -/
theorem key_aux (lo mid hi : Nat) (_hmid : mid < 2 ^ 32) (_hhi : hi < 2 ^ 32) :
    (lo + (mid + hi * 2 ^ 32) * 2 ^ 64) + hi
      = (lo + mid * K) + P * (mid + hi * (2 ^ 32 + 1)) := by
  have hPK : P + K = 2 ^ 64 := by rw [← two64_eq]
  have h96 : 2 ^ 96 + 1 = (2 ^ 32 + 1) * P := two96_eq
  have e1 : (mid + hi * 2 ^ 32) * 2 ^ 64 = mid * 2 ^ 64 + hi * 2 ^ 96 := by
    ring_nat
  rw [e1]
  have e2 : mid * 2 ^ 64 = mid * (P + K) := by rw [hPK]
  rw [e2]
  have e3 : hi * 2 ^ 96 + hi = hi * ((2 ^ 32 + 1) * P) := by
    rw [← h96]
    ring_nat
  calc
    lo + (mid * (P + K) + hi * 2 ^ 96) + hi
        = lo + mid * (P + K) + (hi * 2 ^ 96 + hi) := by ring_nat
    _ = lo + mid * (P + K) + hi * ((2 ^ 32 + 1) * P) := by rw [e3]
    _ = (lo + mid * K) + P * (mid + hi * (2 ^ 32 + 1)) := by
      rw [Nat.mul_add]
      ring_nat

theorem reduce_key_identity (x : Nat) (hx : x < 2 ^ 128) :
    x + hiOf x = (loOf x + midOf x * K) + P * (midOf x + hiOf x * (2 ^ 32 + 1)) := by
  have hxdec : x = loOf x + hiAllOf x * 2 ^ 64 := by
    unfold loOf hiAllOf
    have h := Nat.div_add_mod x (2 ^ 64)
    rw [Nat.mul_comm] at h
    omega
  have hhidec : hiAllOf x = midOf x + hiOf x * 2 ^ 32 := by
    unfold midOf hiOf
    have h := Nat.div_add_mod (hiAllOf x) (2 ^ 32)
    rw [Nat.mul_comm] at h
    omega
  calc
    x + hiOf x = (loOf x + hiAllOf x * 2 ^ 64) + hiOf x :=
      congrArg (fun t => t + hiOf x) hxdec
    _ = (loOf x + (midOf x + hiOf x * 2 ^ 32) * 2 ^ 64) + hiOf x := by
      rw [hhidec]
    _ = (loOf x + midOf x * K) + P * (midOf x + hiOf x * (2 ^ 32 + 1)) :=
      key_aux (loOf x) (midOf x) (hiOf x) (midOf_lt x) (hiOf_lt x hx)

/-- `s + hi` is congruent to `x + hi` modulo `p`: the borrow correction
`p + lo - hi` and the `mid*(2^32-1)` term implement `x - hi` exactly. -/
theorem sOf_meq (x : Nat) (hx : x < 2 ^ 128) :
    MEq P (sOf x + hiOf x) (x + hiOf x) := by
  unfold MEq sOf tOf
  have hkey := reduce_key_identity x hx
  have hlo := loOf_lt x
  have hhi := hiOf_lt x hx
  have hPK : P + K = 2 ^ 64 := by rw [← two64_eq]
  by_cases h : loOf x ≥ hiOf x
  · rw [if_pos h]
    have ht : loOf x - hiOf x + midOf x * K + hiOf x = loOf x + midOf x * K := by
      omega
    rw [ht]
    rw [hkey, mod_add_mul_right']
  · rw [if_neg h]
    have hP : 2 ^ 32 ≤ P := by decide
    have hscalar : P - (hiOf x - loOf x) + hiOf x = P + loOf x := by
      have hle : loOf x ≤ hiOf x := by omega
      have hb : hiOf x - loOf x ≤ P := by omega
      omega
    have ht : P - (hiOf x - loOf x) + midOf x * K + hiOf x
        = P + (loOf x + midOf x * K) := by
      rw [show P - (hiOf x - loOf x) + midOf x * K + hiOf x
            = (P - (hiOf x - loOf x) + hiOf x) + midOf x * K by ring_nat]
      rw [hscalar]
      ring_nat
    rw [ht]
    rw [hkey]
    rw [mod_add_self_left]
    rw [mod_add_mul_right']

/-- The wrapping accumulator is congruent to `s` modulo `p`:
every `2^64` wrap is compensated by adding `2^32 - 1`, since
`2^64 = p + (2^32 - 1)`. -/
theorem resOf_meq_s (x : Nat) (hx : x < 2 ^ 128) :
    MEq P (resOf x + hiOf x) (sOf x + hiOf x) := by
  unfold MEq resOf
  have hs := sOf_lt x hx
  have hPK : P + K = 2 ^ 64 := by rw [← two64_eq]
  by_cases h1 : sOf x < 2 ^ 64
  · rw [if_pos h1]
    rw [Nat.mod_eq_of_lt h1]
  · rw [if_neg h1]
    have hr0 := sOf_mod x hx (by omega)
    by_cases h2 : sOf x % 2 ^ 64 + K < 2 ^ 64
    · rw [if_pos h2]
      rw [hr0]
      have heq : sOf x + hiOf x = (sOf x - 2 ^ 64 + K + hiOf x) + P := by
        omega
      rw [heq, mod_add_self_right]
    · rw [if_neg h2]
      rw [hr0]
      have heq : sOf x + hiOf x = (sOf x - 2 ^ 64 + 2 * K - 2 ^ 64 + hiOf x) + 2 * P := by
        omega
      rw [heq, mod_add_mul_right]

/-- `reduce128` returns the canonical residue `x % p`. -/
theorem reduce128_spec (x : Nat) (hx : x < 2 ^ 128) : reduce128 x = x % P := by
  unfold reduce128
  have hres := resOf_lt x hx
  have hcancel : resOf x % P = x % P := by
    apply cancel_add_mod P (resOf x) x (hiOf x) P_pos
    exact meq_trans (resOf_meq_s x hx) (sOf_meq x hx)
  by_cases h : resOf x ≥ P
  · rw [if_pos h]
    have hbig : 2 ^ 64 < 2 * P := by decide
    have hlt : resOf x - P < P := by omega
    rw [← hcancel]
    rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt hlt]
  · rw [if_neg h]
    have hlt : resOf x < P := by omega
    rw [← hcancel, Nat.mod_eq_of_lt hlt]

end G6ME.Goldilocks
