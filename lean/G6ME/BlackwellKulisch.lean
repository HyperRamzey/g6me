/-
  G6ME.BlackwellKulisch — exact limb-level Kulisch superaccumulator.

  Models `crates/g6me-core/src/blackwell.rs::KulischAcc`:

  * two 36-limb `u64` fixed-point accumulators (`pos`, `neg`),
  * `add`, which decomposes a value into `mant * 2^bit_pos` and adds it into
    the chosen accumulator with 64-bit word carries,
  * `finalize`, whose exact part is the integer difference `pos - neg`.

  Proven here: the limb array after a carry-propagating insertion represents
  exactly `old_value + mant * 2^bit_pos` (given the insertion fits in 36 limbs),
  so the accumulator is exact with zero rounding.  The final single rounding of
  `pos - neg` to IEEE-754 `f64` is expressible only with a `Float` model and is
  **not** proven here; `finalize`'s comparison/subtraction is represented by the
  integer-valued `Acc.value`.
-/

import G6ME.Basic

set_option exponentiation.threshold 3000

namespace G6ME.Kulisch

/-! ## Word parameters -/

/-- Word base: one `u64` limb. -/
def B : Nat := 2 ^ 64

theorem B_eq : B = 18446744073709551616 := by decide

theorem B_pos : 0 < B := by decide

/-- The accumulator width in limbs. -/
def nLimbs : Nat := 36

/-- The accumulator capacity, `2^2304`. -/
def width : Nat := B ^ nLimbs

/-- The Rust `KulischAcc::BIAS_BITS`. -/
def biasBits : Nat := 1088

/-- `limb[0]` has weight `B^0 = 1`; `limb[i]` has weight `B^i`. -/
def limbVal : List Nat → Nat
  | [] => 0
  | x :: xs => x + B * limbVal xs

/-- A list of `n` zero limbs. -/
def zeros : Nat → List Nat
  | 0 => []
  | n + 1 => 0 :: zeros n

theorem limbVal_zeros (n : Nat) : limbVal (zeros n) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [zeros, limbVal, ih]

theorem length_zeros (n : Nat) : (zeros n).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [zeros, ih]

/-! ## Carry propagation

`addCarry l c` adds the carry word `c` at the least significant limb and
propagates the carry.  If `limbVal l + c` fits in `l.length` words, the value
is preserved exactly; a carry out of the top would be lost. -/

def addCarry : List Nat → Nat → List Nat
  | [], _ => []
  | x :: xs, c => (x + c) % B :: addCarry xs ((x + c) / B)

theorem length_addCarry (l : List Nat) (c : Nat) :
    (addCarry l c).length = l.length := by
  induction l generalizing c with
  | nil => rfl
  | cons x xs ih => simp [addCarry, ih]

/-- Limb-level carry propagation is exact when the sum fits. -/
theorem limbVal_addCarry (l : List Nat) (c : Nat)
    (h : limbVal l + c < B ^ l.length) :
    limbVal (addCarry l c) = limbVal l + c := by
  induction l generalizing c with
  | nil =>
      simp [addCarry, limbVal] at h ⊢
      omega
  | cons x xs ih =>
      have hdecomp : x + c = (x + c) % B + B * ((x + c) / B) :=
        (Nat.mod_add_div (x + c) B).symm
      have hlen : (x :: xs).length = xs.length + 1 := List.length_cons
      simp only [limbVal] at h
      rw [hlen, Nat.pow_succ] at h
      have hq : limbVal xs + (x + c) / B < B ^ xs.length := by
        have h1 : (x + c) % B + B * (limbVal xs + (x + c) / B)
            < B ^ xs.length * B := by
          have hx : x + B * limbVal xs + c
              = (x + c) % B + B * (limbVal xs + (x + c) / B) := by
            calc x + B * limbVal xs + c
                = (x + c) + B * limbVal xs := by ring_nat
              _ = ((x + c) % B + B * ((x + c) / B)) + B * limbVal xs := by
                    conv => lhs; rw [hdecomp]
              _ = (x + c) % B + (B * limbVal xs + B * ((x + c) / B)) := by
                    ring_nat
              _ = (x + c) % B + B * (limbVal xs + (x + c) / B) := by
                    rw [Nat.mul_add]
          rw [hx] at h
          exact h
        have h2 : B * (limbVal xs + (x + c) / B) < B ^ xs.length * B := by
          omega
        rw [Nat.mul_comm (B ^ xs.length) B] at h2
        exact mul_lt_mul_left_of_pos B_pos h2
      simp only [addCarry, limbVal]
      rw [ih ((x + c) / B) hq]
      calc (x + c) % B + B * (limbVal xs + (x + c) / B)
          = (x + c) % B + (B * limbVal xs + B * ((x + c) / B)) := by
              rw [Nat.mul_add]
        _ = ((x + c) % B + B * ((x + c) / B)) + B * limbVal xs := by ring_nat
        _ = (x + c) + B * limbVal xs := by rw [← hdecomp]
        _ = x + B * limbVal xs + c := by ring_nat

/-! ## Insertion at an arbitrary limb -/

/-- `addAt l i v` adds the word `v` at limb `i`, propagating carries. -/
def addAt : List Nat → Nat → Nat → List Nat
  | [], _, _ => []
  | x :: xs, 0, v => (x + v) % B :: addCarry xs ((x + v) / B)
  | x :: xs, Nat.succ i, v => x :: addAt xs i v

theorem length_addAt (l : List Nat) (i v : Nat) (hi : i < l.length) :
    (addAt l i v).length = l.length := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
      cases i with
      | zero => simp [addAt, length_addCarry]
      | succ i =>
          have hi' : i < xs.length := by
            simp only [List.length_cons] at hi
            omega
          simp [addAt, ih i hi']

/-- Limb-level insertion at word `i` represents `old + v * B^i` exactly. -/
theorem limbVal_addAt (l : List Nat) (i v : Nat) (hi : i < l.length)
    (h : limbVal l + v * B ^ i < B ^ l.length) :
    limbVal (addAt l i v) = limbVal l + v * B ^ i := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
      cases i with
      | zero =>
          have hdecomp : x + v = (x + v) % B + B * ((x + v) / B) :=
            (Nat.mod_add_div (x + v) B).symm
          have hlen : (x :: xs).length = xs.length + 1 := List.length_cons
          have hB : B ^ (xs.length + 1) = B ^ xs.length * B := by
            rw [show xs.length + 1 = Nat.succ xs.length by omega, Nat.pow_succ]
          simp only [limbVal, Nat.pow_zero, Nat.mul_one] at h ⊢
          rw [hlen, hB] at h
          have hq : limbVal xs + (x + v) / B < B ^ xs.length := by
            have h1 : (x + v) % B + B * (limbVal xs + (x + v) / B)
                < B ^ xs.length * B := by
              have hx : x + B * limbVal xs + v
                  = (x + v) % B + B * (limbVal xs + (x + v) / B) := by
                calc x + B * limbVal xs + v
                    = (x + v) + B * limbVal xs := by ring_nat
                  _ = ((x + v) % B + B * ((x + v) / B)) + B * limbVal xs := by
                        conv => lhs; rw [hdecomp]
                  _ = (x + v) % B + (B * limbVal xs + B * ((x + v) / B)) := by
                        ring_nat
                  _ = (x + v) % B + B * (limbVal xs + (x + v) / B) := by
                        rw [Nat.mul_add]
              rw [hx] at h
              exact h
            have h2 : B * (limbVal xs + (x + v) / B) < B ^ xs.length * B := by
              omega
            rw [Nat.mul_comm (B ^ xs.length) B] at h2
            exact mul_lt_mul_left_of_pos B_pos h2
          simp only [addAt, limbVal]
          rw [limbVal_addCarry xs ((x + v) / B) hq]
          calc (x + v) % B + B * (limbVal xs + (x + v) / B)
              = (x + v) % B + (B * limbVal xs + B * ((x + v) / B)) := by
                  rw [Nat.mul_add]
            _ = ((x + v) % B + B * ((x + v) / B)) + B * limbVal xs := by ring_nat
            _ = (x + v) + B * limbVal xs := by rw [← hdecomp]
            _ = x + B * limbVal xs + v := by ring_nat
      | succ i =>
          have hi' : i < xs.length := by
            simp only [List.length_cons] at hi
            omega
          have hlen : (x :: xs).length = xs.length + 1 := List.length_cons
          have hB : B ^ (xs.length + 1) = B ^ xs.length * B := by
            rw [show xs.length + 1 = Nat.succ xs.length by omega, Nat.pow_succ]
          have hv : v * B ^ (i + 1) = B * (v * B ^ i) := by
            rw [show i + 1 = Nat.succ i by omega, Nat.pow_succ]
            ring_nat
          simp only [limbVal] at h
          rw [hlen, hB, hv] at h
          have hV : limbVal xs + v * B ^ i < B ^ xs.length := by
            have h2 : B * (limbVal xs + v * B ^ i) < B ^ xs.length * B := by
              have hx : x + B * (limbVal xs + v * B ^ i)
                  = x + B * limbVal xs + B * (v * B ^ i) := by ring_nat
              rw [← hx] at h
              omega
            rw [Nat.mul_comm (B ^ xs.length) B] at h2
            exact mul_lt_mul_left_of_pos B_pos h2
          simp only [addAt, limbVal]
          rw [ih i hi' hV]
          rw [Nat.mul_add, hv]
          ring_nat

/-- The identity `mant * 2^(bitpos % 64) * B^(bitpos / 64) = mant * 2^bitpos`
that makes an insertion at word `bitpos/64` equal to a scaled mantissa. -/
theorem scaled_mant (mant bitpos : Nat) :
    mant * 2 ^ (bitpos % 64) * B ^ (bitpos / 64) = mant * 2 ^ bitpos := by
  have hsplit : bitpos = 64 * (bitpos / 64) + bitpos % 64 :=
    (Nat.div_add_mod bitpos 64).symm
  calc mant * 2 ^ (bitpos % 64) * B ^ (bitpos / 64)
      = mant * 2 ^ (bitpos % 64) * (2 ^ 64) ^ (bitpos / 64) := by rw [B_eq]
    _ = mant * 2 ^ (bitpos % 64) * 2 ^ (64 * (bitpos / 64)) := by rw [Nat.pow_mul]
    _ = mant * (2 ^ (64 * (bitpos / 64)) * 2 ^ (bitpos % 64)) := by ring_nat
    _ = mant * 2 ^ (64 * (bitpos / 64) + bitpos % 64) := by rw [← Nat.pow_add]
    _ = mant * 2 ^ bitpos := by rw [← hsplit]

/-! ## The signed accumulator -/

/-- `pos`/`neg` limb arrays, as in `KulischAcc`. -/
structure Acc where
  pos : List Nat
  neg : List Nat

/-- The accumulator initialised to 36 zero limbs on each side. -/
def Acc.empty : Acc := ⟨zeros nLimbs, zeros nLimbs⟩

/-- Insert `mant * 2^bitpos` into the positive or negative side, as `add` does. -/
def Acc.add (a : Acc) (neg : Bool) (mant bitpos : Nat) : Acc :=
  if neg then
    { a with neg := addAt a.neg (bitpos / 64) (mant * 2 ^ (bitpos % 64)) }
  else
    { a with pos := addAt a.pos (bitpos / 64) (mant * 2 ^ (bitpos % 64)) }

/-- Exact integer value represented by the pair of limb arrays. -/
def Acc.value (a : Acc) : Int := (limbVal a.pos : Int) - (limbVal a.neg : Int)

/-- **Exactness of one limb insertion**: if the selected side has room for the
new term, its represented value increases by exactly `mant * 2^bitpos`. -/
theorem Acc.add_exact (a : Acc) (neg : Bool) (mant bitpos : Nat)
    (hidx : bitpos / 64 < (if neg then a.neg.length else a.pos.length))
    (hbound : (if neg then limbVal a.neg else limbVal a.pos) + mant * 2 ^ bitpos
      < B ^ (if neg then a.neg.length else a.pos.length)) :
    (a.add neg mant bitpos).value
      = a.value + (if neg then -((mant * 2 ^ bitpos : Nat) : Int)
                   else ((mant * 2 ^ bitpos : Nat) : Int)) := by
  have hidx' : bitpos / 64 < (if neg then a.neg else a.pos).length := by
    cases neg <;> simpa using hidx
  have hbound' : limbVal (if neg then a.neg else a.pos)
      + (mant * 2 ^ (bitpos % 64)) * B ^ (bitpos / 64)
      < B ^ (if neg then a.neg else a.pos).length := by
    rw [scaled_mant]
    cases neg <;> simpa using hbound
  have hval := limbVal_addAt (if neg then a.neg else a.pos) (bitpos / 64)
    (mant * 2 ^ (bitpos % 64)) hidx' hbound'
  rw [scaled_mant] at hval
  cases neg
  · simp only [Bool.false_eq_true, if_false, Acc.add, Acc.value] at hval ⊢
    rw [hval, Int.natCast_add]
    ring_int
  · simp only [if_true, Acc.add, Acc.value] at hval ⊢
    rw [hval, Int.natCast_add]
    ring_int

/-- Length preservation of the positive side. -/
theorem Acc.add_pos_length (a : Acc) (neg : Bool) (mant bitpos : Nat)
    (hidx : bitpos / 64 < (if neg then a.neg.length else a.pos.length)) :
    (a.add neg mant bitpos).pos.length = a.pos.length := by
  cases neg
  · simp only [Bool.false_eq_true, if_false, Acc.add, if_false]
    exact length_addAt _ _ _ hidx
  · simp only [if_true, Acc.add]

/-- Length preservation of the negative side. -/
theorem Acc.add_neg_length (a : Acc) (neg : Bool) (mant bitpos : Nat)
    (hidx : bitpos / 64 < (if neg then a.neg.length else a.pos.length)) :
    (a.add neg mant bitpos).neg.length = a.neg.length := by
  cases neg
  · simp only [Bool.false_eq_true, if_false, Acc.add]
  · simp only [if_true, Acc.add, if_true]
    exact length_addAt _ _ _ hidx

/-! ## The `f64` range facts -/

/-- For a finite `f64`, `exp2` lies in `[-1074, 971]`; the Rust bit position
`bit_pos = exp2 + 1088` therefore lies in `[14, 2059]`, inside the accepted
range `[0, 2240)`. -/
theorem bitpos_range (exp2 : Int) (hlo : -1074 ≤ exp2) (hhi : exp2 ≤ 971) :
    0 ≤ exp2 + 1088 ∧ exp2 + 1088 < 2240 := by
  constructor <;> omega

/-- A value with 53-bit mantissa at bit position `≤ 2059` fits in the 36-limb
`2^2304` accumulator. -/
theorem fp64_term_fits (mant bitpos : Nat) (hm : mant < 2 ^ 53) (hb : bitpos ≤ 2059) :
    mant * 2 ^ bitpos < width := by
  have h1 : mant * 2 ^ bitpos ≤ 2 ^ 53 * 2 ^ 2059 :=
    Nat.mul_le_mul (Nat.le_of_lt hm)
      (Nat.pow_le_pow_right (by decide : 0 < 2) hb)
  have h2 : 2 ^ 53 * 2 ^ 2059 = 2 ^ 2112 := by rw [← Nat.pow_add]
  have h3 : 2 ^ 2112 < 2 ^ 2304 := two_pow_lt_two_pow (by decide)
  have h4 : (2 ^ 2304 : Nat) = width := by
    unfold width nLimbs B
    rw [← Nat.pow_mul]
  calc mant * 2 ^ bitpos ≤ 2 ^ 53 * 2 ^ 2059 := h1
    _ = 2 ^ 2112 := h2
    _ < 2 ^ 2304 := h3
    _ = width := h4

/-- **A single add of one finite-`f64` mantissa is exact**: an empty
accumulator holds precisely `mant * 2^bitpos` in its positive array. -/
theorem kulisch_single_add_exact (mant bitpos : Nat) (hm : mant < 2 ^ 53)
    (hb : bitpos ≤ 2059) :
    (Acc.empty.add false mant bitpos).value = ((mant * 2 ^ bitpos : Nat) : Int) := by
  have hidx : bitpos / 64 < (if false then Acc.empty.neg.length else Acc.empty.pos.length) := by
    simp only [Bool.false_eq_true, if_false]
    unfold Acc.empty nLimbs
    rw [length_zeros]
    rw [Nat.div_lt_iff_lt_mul (by decide : 0 < 64)]
    omega
  have hbound : (if false then limbVal Acc.empty.neg else limbVal Acc.empty.pos)
      + mant * 2 ^ bitpos < B ^ (if false then Acc.empty.neg.length else Acc.empty.pos.length) := by
    simp only [Bool.false_eq_true, if_false]
    unfold Acc.empty nLimbs
    rw [limbVal_zeros, length_zeros]
    have hf := fp64_term_fits mant bitpos hm hb
    unfold width nLimbs at hf
    omega
  have h := Acc.add_exact Acc.empty false mant bitpos hidx hbound
  simpa [Acc.empty, Acc.value, limbVal_zeros] using h

end G6ME.Kulisch
