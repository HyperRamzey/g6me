/-
  G6ME.Hoa — order-7 spherical-harmonic combinatorial and algebraic facts.

  Models parts of `crates/g6me-core/src/hoa.rs`:

  * `FACT` (`0! ..= 14!`) as exact `Nat` factorials;
  * the unnormalised associated Legendre recurrence used by
    `assoc_legendre` in the `m = 0` case;
  * the order-7 ACN channel count (`HOA_CH = 64`) and the ACN flat index
    `n * n + n + m` on the 64 pairs with `|m| <= n <= 7`;
  * the conditional diffuse-field energy identity behind
    `hold_sh_taper_weights` (`comp^2 * e_raw = e_unit` implies the tapered
    degree-weighted sum equals `e_unit`).

  Everything here is core Lean 4: no Mathlib, no analysis.  In particular the
  **orthonormality of the spherical harmonics is not proved** — that is an
  identity of integrals of products of trigonometric functions over the
  sphere and needs real analysis, which this development deliberately does
  not model.  Likewise the `sqrt`/`sin`/`cos` values that appear in
  `eval_sh_order7` and `hold_sh_taper_weights` are represented by exact
  integer/rational surrogates (`comp^2` as an integer `cs`, ratios of
  factorials as `Nat`), never by Lean `Float`.

  ## The two Legendre models

  The Rust loop (`assoc_legendre` with `m = 0`) is transcribed literally as
  `legLoop : Nat -> Int -> Int`, including the integer division by `l`.  Its
  endpoint values are exact: `legLoop l 1 = 1` and
  `legLoop l (-1) = (-1)^l`.  But the literal cross-multiplied recurrence

    `(l+2) * leg (l+2) x = (2*(l+2)-1) * x * leg (l+1) x
                            - ((l+2)-1) * leg l x`

  **cannot hold for an `Int`-valued `leg` at every `Int` point**: at
  `l = 2, x = 2` the right-hand side is `11`, so the equation would demand
  `2 * leg 2 2 = 11`, which has no integer solution.  `legLoop` truncates
  toward zero there (`11 / 2 = 5`, proved as `legLoop_two_two`), and the
  recurrence therefore only holds when `l+2` divides the numerator; that
  exact statement is `legLoop_recurrence_iff`.

  The headline theorems below use an **exact rational model** `leg`, whose
  values are integer numerator/denominator pairs with cross-multiplied
  equality `LegEq`.  The division is never performed: `LegVal.divNat` appends
  the divisor to the denominator, so the recurrence is an exact identity for
  all `l` and `x` (`legendre_recurrence`), and the two endpoint theorems
  `legendre_at_one` / `legendre_at_minus_one` hold exactly (as rationals).
-/

import G6ME.Basic

set_option maxRecDepth 100000

namespace G6ME.Hoa

/-! ## Factorials

`FACT` in `hoa.rs` is an `f64` table; the entries are exactly the integers
`0! ..= 14!`, so the faithful exact model is the structural `Nat` factorial. -/

/-- Structural factorial `fact n = n!`. -/
def fact : Nat -> Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

/-- Every entry of the Rust `FACT` table equals `fact n`. -/
theorem fact_table :
    (List.range 15).map fact =
      [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800,
       479001600, 6227020800, 87178291200] := by
  decide

/-! ## Powers of `-1` and a small numeral-normalising integer tactic -/

theorem int_mul_two (a : Int) : a * 2 = a + a := by
  rw [show (2 : Int) = 1 + 1 from rfl]
  ring_int

/-- `ring_int` with numeral factors `2` expanded to repeated addition.  Core
`omega` abstracts nonlinear monomials as atoms, so it cannot relate
`K * (e * 2)` to `2 * (K * e)` unless the numerals are first normalised. -/
macro "ring_int2" : tactic =>
  `(tactic|
    (simp only [Int.sub_eq_add_neg, Int.mul_add, Int.add_mul, Int.mul_neg,
        Int.neg_mul, Int.neg_neg, Int.neg_add, Int.mul_assoc, Int.mul_comm,
        Int.mul_left_comm, Int.add_assoc, Int.add_comm, Int.add_left_comm,
        Int.mul_one, Int.one_mul, Int.two_mul, int_mul_two]
     <;> omega))

theorem neg_one_pow_succ (n : Nat) : (-1 : Int) ^ (n + 1) = -((-1 : Int) ^ n) := by
  rw [Int.pow_succ]
  simp

theorem neg_one_pow_succ_succ (n : Nat) : (-1 : Int) ^ (n + 2) = (-1 : Int) ^ n := by
  rw [Int.pow_succ, Int.pow_succ]
  simp

/-! ## The Rust loop with integer division

`assoc_legendre l 0 x` evaluates `P_0 = 1`, `P_1 = x`, and then

  `P_l = ((2l-1) * x * P_{l-1} - (l-1) * P_{l-2}) / l`

for `l >= 2`.  `legLoop` is that loop over `Int`, with Lean's Euclidean
division.  It differs from the real Legendre polynomial exactly where the
division is inexact; see the module docstring. -/

/-- Literal transcription of the Rust `m = 0` associated-Legendre loop. -/
def legLoop : Nat -> Int -> Int
  | 0, _ => 1
  | 1, x => x
  | (l + 2), x =>
      ((2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
        - (((l + 2 : Nat) : Int) - 1) * legLoop l x) / ((l + 2 : Nat) : Int)

/-- Defining equation of `legLoop` at `l + 2`. -/
theorem legLoop_succ_succ (l : Nat) (x : Int) :
    legLoop (l + 2) x =
      ((2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
        - (((l + 2 : Nat) : Int) - 1) * legLoop l x) / ((l + 2 : Nat) : Int) := rfl

/-- `P_l(1) = 1` for the integer-division loop (the divisions are exact at
`x = 1`). -/
theorem legLoop_at_one : ∀ l : Nat, legLoop l 1 = 1
  | 0 => rfl
  | 1 => rfl
  | l + 2 => by
      have ih1 : legLoop (l + 1) 1 = 1 := legLoop_at_one (l + 1)
      have ih2 : legLoop l 1 = 1 := legLoop_at_one l
      have hk : (((l + 2 : Nat) : Int)) ≠ 0 := by
        intro h
        have h2 : (l + 2 : Nat) = 0 := Int.ofNat_eq_zero.mp h
        omega
      rw [legLoop_succ_succ, ih1, ih2]
      have hn : (2 * ((l + 2 : Nat) : Int) - 1) * 1 * 1
          - (((l + 2 : Nat) : Int) - 1) * 1 = ((l + 2 : Nat) : Int) := by ring_int
      rw [hn]
      exact Int.ediv_self hk

/-- `P_l(-1) = (-1)^l` for the integer-division loop (again the divisions are
exact at this point). -/
theorem legLoop_at_minus_one : ∀ l : Nat, legLoop l (-1) = (-1 : Int) ^ l
  | 0 => rfl
  | 1 => rfl
  | l + 2 => by
      have ih1 : legLoop (l + 1) (-1) = (-1 : Int) ^ (l + 1) := legLoop_at_minus_one (l + 1)
      have ih2 : legLoop l (-1) = (-1 : Int) ^ l := legLoop_at_minus_one l
      have hp1 : (-1 : Int) ^ (l + 1) = -((-1 : Int) ^ l) := neg_one_pow_succ l
      have hp2 : (-1 : Int) ^ (l + 2) = (-1 : Int) ^ l := neg_one_pow_succ_succ l
      have hk : (((l + 2 : Nat) : Int)) ≠ 0 := by
        intro h
        have h2 : (l + 2 : Nat) = 0 := Int.ofNat_eq_zero.mp h
        omega
      rw [legLoop_succ_succ, ih1, ih2, hp1, hp2]
      have hn : (2 * ((l + 2 : Nat) : Int) - 1) * (-1) * -((-1 : Int) ^ l)
          - (((l + 2 : Nat) : Int) - 1) * ((-1 : Int) ^ l)
          = ((l + 2 : Nat) : Int) * ((-1 : Int) ^ l) := by ring_int2
      rw [hn, Int.mul_comm]
      exact Int.mul_ediv_cancel ((-1 : Int) ^ l) hk

/-- The exact status of the cross-multiplied recurrence for the Rust loop:
the equation holds **iff** the numerator is divisible by `l + 2`.  (For
general integer `x` it is not; see `legLoop_recurrence_fails_at_two`.) -/
theorem legLoop_recurrence_iff (l : Nat) (x : Int) :
    ((l + 2 : Nat) : Int) * legLoop (l + 2) x =
        (2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
          - (((l + 2 : Nat) : Int) - 1) * legLoop l x
      ↔ (((l + 2 : Nat) : Int) ∣
          (2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
            - (((l + 2 : Nat) : Int) - 1) * legLoop l x) := by
  rw [legLoop_succ_succ]
  constructor
  · intro h
    exact ⟨((2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
        - (((l + 2 : Nat) : Int) - 1) * legLoop l x) / ((l + 2 : Nat) : Int), h.symm⟩
  · intro h
    calc ((l + 2 : Nat) : Int) *
          (((2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
            - (((l + 2 : Nat) : Int) - 1) * legLoop l x) / ((l + 2 : Nat) : Int))
        = (((2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
            - (((l + 2 : Nat) : Int) - 1) * legLoop l x) / ((l + 2 : Nat) : Int))
            * ((l + 2 : Nat) : Int) := Int.mul_comm _ _
      _ = (2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
            - (((l + 2 : Nat) : Int) - 1) * legLoop l x := Int.ediv_mul_cancel h

/-- The recurrence for the Rust loop whenever the integer division is exact. -/
theorem legLoop_recurrence {l : Nat} {x : Int}
    (h : (((l + 2 : Nat) : Int) ∣
          (2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
            - (((l + 2 : Nat) : Int) - 1) * legLoop l x)) :
    ((l + 2 : Nat) : Int) * legLoop (l + 2) x =
        (2 * ((l + 2 : Nat) : Int) - 1) * x * legLoop (l + 1) x
          - (((l + 2 : Nat) : Int) - 1) * legLoop l x :=
  (legLoop_recurrence_iff l x).mpr h

/-- Concrete value showing the truncation: at `l = 2, x = 2` the loop computes
`(3*2*2 - 1) / 2 = 11 / 2 = 5`. -/
theorem legLoop_two_two : legLoop 2 2 = 5 := by decide

/-- The literal cross-multiplied recurrence is false for the integer-division
loop at `l = 2, x = 2`: `2 * 5 = 10`, not `11`.  This is why the headline
recurrence below is stated in the exact rational model `leg`. -/
theorem legLoop_recurrence_fails_at_two :
    ¬ ((2 : Int) * legLoop 2 2
        = (2 * (2 : Int) - 1) * 2 * legLoop 1 2
            - ((2 : Int) - 1) * legLoop 0 2) := by
  decide

/-! ## Exact rational model for the headline theorems

`LegVal` stores an integer numerator over an integer denominator; no
normalisation or division is ever performed.  `LegEq a b` is the
cross-multiplied equality `a.num * b.den = b.num * a.den`, i.e. equality of
the represented rationals.  The denominator is positive in every value that
`leg` constructs (it only ever gets multiplied by positive `Nat`s), but the
algebra below does not need that fact. -/

/-- Integer numerator over an integer denominator. -/
structure LegVal where
  num : Int
  den : Int

/-- Cross-multiplied equality of two represented rationals. -/
def LegEq (a b : LegVal) : Prop := a.num * b.den = b.num * a.den

/-- A rational with denominator one. -/
def LegVal.ofInt (z : Int) : LegVal := ⟨z, 1⟩

def LegVal.mul (a b : LegVal) : LegVal :=
  ⟨a.num * b.num, a.den * b.den⟩

def LegVal.add (a b : LegVal) : LegVal :=
  ⟨a.num * b.den + b.num * a.den, a.den * b.den⟩

def LegVal.neg (a : LegVal) : LegVal := ⟨-a.num, a.den⟩

/-- Division by a positive natural `k`, made exact by multiplying the
denominator: the represented value is unchanged and no integer division is
performed. -/
def LegVal.divNat (a : LegVal) (k : Nat) : LegVal :=
  ⟨a.num, a.den * (k : Int)⟩

/-- The exact rational Legendre sequence: the same three-term numerator
recurrence as the Rust loop, with the division by `l + 2` kept in the
denominator instead of being rounded to an integer. -/
def leg : Nat -> Int -> LegVal
  | 0, _ => LegVal.ofInt 1
  | 1, x => LegVal.ofInt x
  | (l + 2), x =>
      LegVal.divNat
        (LegVal.add
          (LegVal.mul (LegVal.mul (LegVal.ofInt (2 * ((l + 2 : Nat) : Int) - 1))
            (LegVal.ofInt x)) (leg (l + 1) x))
          (LegVal.neg (LegVal.mul (LegVal.ofInt (((l + 2 : Nat) : Int) - 1)) (leg l x))))
        (l + 2)

/-- Defining equation of `leg` at `l + 2`. -/
theorem leg_succ_succ (l : Nat) (x : Int) :
    leg (l + 2) x =
      LegVal.divNat
        (LegVal.add
          (LegVal.mul (LegVal.mul (LegVal.ofInt (2 * ((l + 2 : Nat) : Int) - 1))
            (LegVal.ofInt x)) (leg (l + 1) x))
          (LegVal.neg (LegVal.mul (LegVal.ofInt (((l + 2 : Nat) : Int) - 1)) (leg l x))))
        (l + 2) := rfl

/-- Multiplying a rational by `k` after dividing it by `k` is the identity in
the represented-rational sense (cross-multiplied, so no cancellation by
`k` is needed). -/
theorem LegEq.mul_ofNat_divNat (k : Nat) (a : LegVal) :
    LegEq (LegVal.mul (LegVal.ofInt (k : Int)) (LegVal.divNat a k)) a := by
  simp only [LegVal.mul, LegVal.divNat, LegVal.ofInt, LegEq, Int.one_mul]
  ring_int

/-- The cross-multiplied Legendre recurrence, exact for all `l` and `x`:

  `(l+2) * P_{l+2}(x) = (2*(l+2)-1) * x * P_{l+1}(x) - ((l+2)-1) * P_l(x)`.

The scalars and the subtraction are operations on the represented rationals;
`LegEq` is their equality. -/
theorem legendre_recurrence (l : Nat) (x : Int) :
    LegEq
      (LegVal.mul (LegVal.ofInt ((l + 2 : Nat) : Int)) (leg (l + 2) x))
      (LegVal.add
        (LegVal.mul (LegVal.mul (LegVal.ofInt (2 * ((l + 2 : Nat) : Int) - 1))
          (LegVal.ofInt x)) (leg (l + 1) x))
        (LegVal.neg (LegVal.mul (LegVal.ofInt (((l + 2 : Nat) : Int) - 1)) (leg l x)))) := by
  rw [leg_succ_succ]
  exact LegEq.mul_ofNat_divNat (l + 2) _

/-- A small factorisation lemma: `a*c - b*c = (a-b)*c` over `Int`. -/
theorem int_sub_mul (a b c : Int) : a * c - b * c = (a - b) * c := by ring_int

/-- Numerator identity for the `x = 1` endpoint: if the two previous
denominators are equal to their numerators, the new numerator is the new
denominator. -/
theorem frac_at_one_num (A B K D1 D0 : Int) (hK : A - B = K) :
    A * D1 * D0 + (-(B * D0)) * D1 = D1 * D0 * K := by
  have e1 : A * D1 * D0 = A * (D1 * D0) := by rw [Int.mul_assoc]
  have e2 : (-(B * D0)) * D1 = -(B * (D1 * D0)) := by
    rw [Int.neg_mul, Int.mul_assoc, Int.mul_comm D0 D1]
  rw [e1, e2, ← Int.sub_eq_add_neg, int_sub_mul, hK]
  exact Int.mul_comm K (D1 * D0)

/-- Numerator identity for the `x = -1` endpoint, with the sign `e = (-1)^l`
tracked as an integer. -/
theorem frac_at_minus_one_num (A B K e D1 D0 : Int) (hK : A - B = K) :
    A * (-1) * (-e * D1) * D0 + (-(B * (e * D0))) * D1 = e * (D1 * D0 * K) := by
  have e1 : A * (-1) * (-e * D1) * D0 = A * (e * (D1 * D0)) := by
    simp only [Int.mul_neg, Int.neg_neg, Int.mul_comm, Int.mul_left_comm, Int.mul_one]
  have e2 : (-(B * (e * D0))) * D1 = -(B * (e * (D1 * D0))) := by
    rw [Int.neg_mul]
    simp only [Int.mul_comm, Int.mul_left_comm]
  rw [e1, e2, ← Int.sub_eq_add_neg, int_sub_mul, hK]
  simp only [Int.mul_comm, Int.mul_left_comm]

/-- `P_l(1) = 1` in the exact rational model. -/
theorem legendre_at_one : ∀ l : Nat, LegEq (leg l 1) (LegVal.ofInt 1)
  | 0 => by simp [leg, LegVal.ofInt, LegEq]
  | 1 => by simp [leg, LegVal.ofInt, LegEq]
  | l + 2 => by
      have ih1 : LegEq (leg (l + 1) 1) (LegVal.ofInt 1) := legendre_at_one (l + 1)
      have ih2 : LegEq (leg l 1) (LegVal.ofInt 1) := legendre_at_one l
      have h1 : (leg (l + 1) 1).num = (leg (l + 1) 1).den := by
        simpa [LegEq, LegVal.ofInt] using ih1
      have h2 : (leg l 1).num = (leg l 1).den := by
        simpa [LegEq, LegVal.ofInt] using ih2
      have hK : (2 * ((l + 2 : Nat) : Int) - 1) - (((l + 2 : Nat) : Int) - 1)
          = ((l + 2 : Nat) : Int) := by omega
      rw [leg_succ_succ]
      simp only [LegVal.mul, LegVal.add, LegVal.neg, LegVal.divNat, LegVal.ofInt, LegEq]
      rw [h1, h2]
      simp only [Int.mul_one, Int.one_mul]
      exact frac_at_one_num _ _ _ _ _ hK

/-- `P_l(-1) = (-1)^l` in the exact rational model. -/
theorem legendre_at_minus_one :
    ∀ l : Nat, LegEq (leg l (-1)) (LegVal.ofInt ((-1 : Int) ^ l))
  | 0 => by simp [leg, LegVal.ofInt, LegEq]
  | 1 => by simp [leg, LegVal.ofInt, LegEq]
  | l + 2 => by
      have ih1 : LegEq (leg (l + 1) (-1)) (LegVal.ofInt ((-1 : Int) ^ (l + 1))) :=
        legendre_at_minus_one (l + 1)
      have ih2 : LegEq (leg l (-1)) (LegVal.ofInt ((-1 : Int) ^ l)) :=
        legendre_at_minus_one l
      have h1 : (leg (l + 1) (-1)).num = (-1 : Int) ^ (l + 1) * (leg (l + 1) (-1)).den := by
        simpa [LegEq, LegVal.ofInt] using ih1
      have h2 : (leg l (-1)).num = (-1 : Int) ^ l * (leg l (-1)).den := by
        simpa [LegEq, LegVal.ofInt] using ih2
      have hp1 : (-1 : Int) ^ (l + 1) = -((-1 : Int) ^ l) := neg_one_pow_succ l
      have hp2 : (-1 : Int) ^ (l + 2) = (-1 : Int) ^ l := neg_one_pow_succ_succ l
      have hK : (2 * ((l + 2 : Nat) : Int) - 1) - (((l + 2 : Nat) : Int) - 1)
          = ((l + 2 : Nat) : Int) := by omega
      rw [leg_succ_succ]
      simp only [LegVal.mul, LegVal.add, LegVal.neg, LegVal.divNat, LegVal.ofInt, LegEq]
      rw [h1, h2, hp1]
      simp only [Int.mul_one, Int.one_mul]
      rw [hp2]
      exact frac_at_minus_one_num _ _ _ _ _ _ hK

/-! ## Order-7 channel count -/

/-- `HOA_ORDER` from `hoa.rs`. -/
def HOA_ORDER : Nat := 7

/-- `HOA_CH = (HOA_ORDER + 1)^2` from `hoa.rs`. -/
def HOA_CH : Nat := (HOA_ORDER + 1) * (HOA_ORDER + 1)

/-- The Rust loop's channel count `sum_{n=0}^{7} (2n+1)`. -/
theorem hoa_channel_count :
    (List.range 8).foldl (fun acc n => acc + (2 * n + 1)) 0 = 64 := by
  decide

/-- `HOA_CH = 64`: order 7 has 64 ACN channels. -/
theorem hoa_order7_channels : G6ME.Hoa.HOA_CH = 64 := by
  decide

/-! ## ACN flat index

`eval_sh_order7` lays channels out in ACN order: for each order `n` it walks
`m = -n .. n`, so the flat index of `(n, m)` is
`sum_{k<n} (2k+1) + (n + m) = n*n + n + m`.  `acnPairs` enumerates the 64
pairs `(n, m)` with `|m| <= n <= 7` by writing `m = k - n` for
`k = 0 .. 2n`.  Range and injectivity are finite computations, discharged by
`decide` (with `maxRecDepth` raised for the 64 x 64 injectivity check). -/

/-- ACN flat index: `n*n + n + m`. -/
def acnIndex (n : Nat) (m : Int) : Int :=
  (n : Int) * (n : Int) + (n : Int) + m

/-- The 64 ACN pairs `(n, m)` with `|m| <= n <= 7`, in ACN order. -/
def acnPairs : List (Nat × Int) :=
  (List.range 8).flatMap fun n =>
    (List.range (2 * n + 1)).map fun k => (n, (Int.ofNat k) - (Int.ofNat n))

/-- All ordered pairs of ACN channels, used for the injectivity check. -/
def acnPairProduct : List ((Nat × Int) × (Nat × Int)) :=
  acnPairs.flatMap fun p => acnPairs.map fun q => (p, q)

/-- The enumeration has exactly 64 entries. -/
theorem hoa_acn_pairs_length : acnPairs.length = 64 := by
  decide

/-- Every ACN flat index lands in `[0, 64)`. -/
theorem hoa_acn_index_range :
    acnPairs.all (fun p => decide (0 <= acnIndex p.1 p.2 ∧ acnIndex p.1 p.2 < 64)) = true := by
  decide

/-- The ACN flat index is injective on the 64 `(n, m)` pairs. -/
theorem hoa_acn_index_injective :
    acnPairProduct.all (fun pq =>
      decide (acnIndex pq.1.1 pq.1.2 = acnIndex pq.2.1 pq.2.2 → pq.1 = pq.2)) = true := by
  decide

/-- Propositional form of the range statement. -/
theorem hoa_acn_index_range_mem (p : Nat × Int) (hp : p ∈ acnPairs) :
    0 <= acnIndex p.1 p.2 ∧ acnIndex p.1 p.2 < 64 :=
  of_decide_eq_true (List.all_eq_true.mp hoa_acn_index_range p hp)

/-- Propositional form of the injectivity statement. -/
theorem hoa_acn_index_inj (p q : Nat × Int)
    (hp : p ∈ acnPairs) (hq : q ∈ acnPairs)
    (h : acnIndex p.1 p.2 = acnIndex q.1 q.2) : p = q := by
  have hmem : (p, q) ∈ acnPairProduct := by
    rw [acnPairProduct, List.mem_flatMap]
    exact ⟨p, hp, by
      rw [List.mem_map]
      exact ⟨q, hq, rfl⟩⟩
  exact of_decide_eq_true (List.all_eq_true.mp hoa_acn_index_injective (p, q) hmem) h

/-! ## Diffuse-field taper energy

`hold_sh_taper_weights` scales the raw order weights by a common
`comp = sqrt(e_unit / e_raw)`, where `e_unit = sum_n (2n+1)` and
`e_raw = sum_n (2n+1) w_n^2`.  We model the square of the compression factor
by an integer `cs`; then `(comp * w_n)^2` is `cs * w_n^2`, and the tapered
degree-weighted energy is `cs` times the raw energy.  The conditional theorem
below is exactly the diffuse-field energy preservation claim.  The square
root, the real weights and the `max(1e-12, ...)` guard are not modelled. -/

/-- Degree multiplicity `2n+1`. -/
def degreeCount (n : Nat) : Nat := 2 * n + 1

/-- The eight orders `0 ..= 7`. -/
def taperIndices : List Nat := [0, 1, 2, 3, 4, 5, 6, 7]

/-- Raw degree-weighted energy `sum_n (2n+1) w_n^2`. -/
def taperRawEnergy (w : Nat -> Nat) : Nat :=
  taperIndices.foldl (fun acc n => acc + degreeCount n * w n * w n) 0

/-- Tapered degree-weighted energy, with `cs` representing `comp^2`. -/
def taperScaledEnergy (cs : Nat) (w : Nat -> Nat) : Nat :=
  taperIndices.foldl (fun acc n => acc + degreeCount n * (cs * w n * w n)) 0

/-- Unit energy `sum_n (2n+1)`; it equals the channel count 64. -/
def taperUnitEnergy : Nat :=
  taperIndices.foldl (fun acc n => acc + degreeCount n) 0

/-- The diffuse-field unit energy for order 7 is the 64-channel count. -/
theorem taper_unit_energy_eq : taperUnitEnergy = 64 := by
  decide

/-- The scaling identity for an arbitrary accumulator: scaling every weight
square by `cs` scales the whole fold by `cs`. -/
theorem taper_foldl_scaled (c w : Nat -> Nat) (cs : Nat) :
    ∀ (l : List Nat) (accA accB : Nat), accB = cs * accA ->
      l.foldl (fun acc n => acc + c n * (cs * w n * w n)) accB
        = cs * l.foldl (fun acc n => acc + c n * w n * w n) accA
  | [], accA, accB, h => by simpa using h
  | x :: xs, accA, accB, h => by
      rw [List.foldl_cons, List.foldl_cons]
      apply taper_foldl_scaled c w cs xs
        (accA + c x * w x * w x)
        (accB + c x * (cs * w x * w x))
      rw [h]
      ring_nat

/-- `cs` times the raw energy equals the tapered energy sum. -/
theorem taper_scaled_energy (cs : Nat) (w : Nat -> Nat) :
    taperScaledEnergy cs w = cs * taperRawEnergy w := by
  unfold taperScaledEnergy taperRawEnergy
  exact taper_foldl_scaled degreeCount w cs taperIndices 0 0 (by rw [Nat.mul_zero])

/-- Conditional diffuse-field energy preservation: if the squared compression
factor `cs` satisfies `cs * e_raw = e_unit`, then the tapered degree-weighted
sum equals `e_unit`.  This is the algebraic content of
`hold_sh_taper_weights`; the square root that produces `comp` is not
modelled. -/
theorem taper_energy_identity {cs eUnit : Nat} {w : Nat -> Nat}
    (h : cs * taperRawEnergy w = eUnit) :
    taperScaledEnergy cs w = eUnit := by
  rw [taper_scaled_energy, h]

/-! ## What is deliberately not proved

The orthonormality of the real spherical harmonics — the integrals
`∫ Y_i Y_j dΩ = δ_ij`, equivalently the trigonometric/product-to-sum
identities behind `eval_sh_order7`'s N3D normalisation — is **not** proved
here.  It requires real analysis and trigonometric integrals, which are
outside this core-only development.  Likewise nothing here evaluates
`sin`, `cos`, `sqrt`, `asinh`, or the `f64` factorials; the results are the
exact combinatorial and algebraic skeletons listed above. -/

end G6ME.Hoa
