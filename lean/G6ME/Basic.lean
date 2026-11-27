/-
  G6ME.Basic — shared infrastructure for the G6ME Lean formalization.

  Everything here is pure Lean 4 core: no Mathlib, no axioms beyond `propext`
  and `Quot.sound`.  The `ring_nat` / `ring_int` tactics expand products and
  then discharge the resulting linear identity with `omega`; they are used
  throughout for the polynomial identities of the DSP code.
-/

namespace G6ME

/-! ## A small polynomial normaliser over `Nat` and `Int`.

`simp` with the distributive/associativity/commutativity laws puts a
polynomial expression into a sum of monomials; `omega` then closes the
identity, treating each monomial as an atom.  This is the workhorse for all
algebraic identities in this development (the core distribution has no
`ring` tactic). -/

macro "ring_nat" : tactic =>
  `(tactic|
    (simp only [Nat.mul_add, Nat.add_mul, Nat.mul_assoc, Nat.mul_comm,
        Nat.mul_left_comm, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm,
        Nat.mul_one, Nat.one_mul, Nat.mul_zero, Nat.zero_mul,
        Nat.add_zero, Nat.zero_add]
     <;> omega))

macro "ring_int" : tactic =>
  `(tactic|
    (simp only [Int.sub_eq_add_neg, Int.mul_add, Int.add_mul, Int.mul_neg,
        Int.neg_mul, Int.neg_neg, Int.neg_add, Int.mul_assoc, Int.mul_comm,
        Int.mul_left_comm, Int.add_assoc, Int.add_comm, Int.add_left_comm,
        Int.mul_one, Int.one_mul, Int.mul_zero, Int.zero_mul,
        Int.add_zero, Int.zero_add, Int.neg_zero,
        Int.reduceMul, Int.reduceAdd, Int.reduceNeg]
     <;> omega))

/-- A two-phase variant for identities with monomials of degree three or
more: distribute first, then normalise the monomials modulo associativity
and commutativity, then discharge the resulting linear identity.  The
single-pass `ring_int` can leave differently-associated monomials that
`omega` treats as distinct atoms. -/
macro "ring_int3" : tactic =>
  `(tactic|
    ((try simp only [Int.sub_eq_add_neg, Int.mul_add, Int.add_mul, Int.mul_neg,
        Int.neg_mul, Int.neg_neg, Int.neg_add, Int.neg_mul_neg])
     <;> (try simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm,
        Int.mul_one, Int.mul_zero, Int.zero_mul,
        Int.add_assoc, Int.add_comm, Int.add_left_comm,
        Int.add_zero, Int.zero_add, Int.reduceMul, Int.reduceAdd, Int.reduceNeg])
     <;> omega))

/-! ## Two constructive arithmetic facts

Core's `Nat.mul_lt_mul_left` and `Nat.pow_lt_pow_right` are themselves proved
with `Classical.choice` (verified with `#print axioms`), which this
development forbids.  The two lemmas below give the same content with
constructive proofs, using only lemmas whose axiom lists are empty. -/

/-- Strict cancellation of a positive left factor. -/
theorem mul_lt_mul_left_of_pos {a b c : Nat} (_hc : 0 < c) (h : c * a < c * b) :
    a < b := by
  rcases Nat.lt_or_ge a b with hlt | hge
  · exact hlt
  · have hle : c * b ≤ c * a := Nat.mul_le_mul_left c hge
    omega

/-- Strict monotonicity of `2 ^ ·`, constructive. -/
theorem two_pow_lt_two_pow {a b : Nat} (h : a < b) : 2 ^ a < 2 ^ b := by
  induction b with
  | zero => omega
  | succ b ih =>
      rcases Nat.lt_succ_iff_lt_or_eq.mp h with hlt | heq
      · have h1 : 2 ^ a < 2 ^ b := ih hlt
        have h2 : 2 ^ b < 2 ^ Nat.succ b := by
          rw [Nat.pow_succ]
          have hp : 0 < 2 ^ b := Nat.pow_pos (by decide)
          omega
        omega
      · subst heq
        rw [Nat.pow_succ]
        have hp : 0 < 2 ^ a := Nat.pow_pos (by decide)
        omega

/-! ## Congruence modulo a natural number

The core library has no `Nat.ModEq`, so we define the equivalence relation
`a ≡ b (mod n)` as equality of remainders and prove the few closure and
cancellation lemmas the Goldilocks development needs. -/

/-- `MEq n a b` means `a` and `b` have the same remainder modulo `n`. -/
def MEq (n a b : Nat) : Prop := a % n = b % n

theorem meq_refl (n a : Nat) : MEq n a a := rfl

theorem meq_symm {n a b : Nat} (h : MEq n a b) : MEq n b a := h.symm

theorem meq_trans {n a b c : Nat} (h1 : MEq n a b) (h2 : MEq n b c) : MEq n a c :=
  h1.trans h2

/-- Adding the same value on the right cancels in a congruence. -/
theorem cancel_add_mod (n a b c : Nat) (hn : 0 < n)
    (h : (a + c) % n = (b + c) % n) : a % n = b % n := by
  have ha := Nat.mod_lt a hn
  have hb := Nat.mod_lt b hn
  have hc := Nat.mod_lt c hn
  rw [Nat.add_mod a c n, Nat.add_mod b c n] at h
  by_cases h1 : a % n + c % n < n
  · by_cases h2 : b % n + c % n < n
    · rw [Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt h2] at h
      omega
    · rw [Nat.mod_eq_of_lt h1] at h
      rw [Nat.mod_eq_sub_mod (by omega : b % n + c % n ≥ n),
        Nat.mod_eq_of_lt (by omega : b % n + c % n - n < n)] at h
      omega
  · by_cases h2 : b % n + c % n < n
    · rw [Nat.mod_eq_of_lt h2] at h
      rw [Nat.mod_eq_sub_mod (by omega : a % n + c % n ≥ n),
        Nat.mod_eq_of_lt (by omega : a % n + c % n - n < n)] at h
      omega
    · rw [Nat.mod_eq_sub_mod (by omega : a % n + c % n ≥ n),
        Nat.mod_eq_of_lt (by omega : a % n + c % n - n < n)] at h
      rw [Nat.mod_eq_sub_mod (by omega : b % n + c % n ≥ n),
        Nat.mod_eq_of_lt (by omega : b % n + c % n - n < n)] at h
      omega

/-- Congruence is compatible with addition. -/
theorem MEq.add {n a b c d : Nat} (h1 : MEq n a b) (h2 : MEq n c d) :
    MEq n (a + c) (b + d) := by
  unfold MEq at *
  rw [Nat.add_mod a c n, Nat.add_mod b d n, h1, h2]

/-- Congruence is compatible with multiplication. -/
theorem MEq.mul {n a b c d : Nat} (h1 : MEq n a b) (h2 : MEq n c d) :
    MEq n (a * c) (b * d) := by
  unfold MEq at *
  rw [Nat.mul_mod a c n, Nat.mul_mod b d n, h1, h2]

/-- A multiple of the modulus may be dropped. -/
theorem mod_add_mul_right (n a b : Nat) : (a + b * n) % n = a % n :=
  Nat.add_mul_mod_self_right a b n

/-- A multiple of the modulus may be dropped (modulus on the left). -/
theorem mod_add_mul_left (n a b : Nat) : (a + n * b) % n = a % n :=
  Nat.add_mul_mod_self_left a n b

/-- A multiple of the modulus may be dropped when the modulus is the left
factor of the product. -/
theorem mod_add_mul_right' (n a b : Nat) : (a + n * b) % n = a % n := by
  rw [Nat.mul_comm, Nat.add_mul_mod_self_right]

/-- The modulus itself may be dropped on the left. -/
theorem mod_add_self_left (n a : Nat) : (n + a) % n = a % n := by
  rw [Nat.add_comm]
  have h : a + n = a + n * 1 := by rw [Nat.mul_one]
  rw [h, Nat.add_mul_mod_self_left]

/-- The modulus itself may be dropped on the right. -/
theorem mod_add_self_right (n a : Nat) : (a + n) % n = a % n := by
  have h : a + n = a + 1 * n := by rw [Nat.one_mul]
  rw [h, Nat.add_mul_mod_self_right]

end G6ME
