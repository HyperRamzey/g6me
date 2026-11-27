/-
  G6ME.BlackwellCrt — the 20-modulus Chinese Remainder Theorem of Ozaki-II.

  Models `crates/g6me-core/src/blackwell.rs`:

  * `OZAKI2_MODULI` (20 pairwise-coprime moduli, product < 2^128)
  * `mod_inv_u128` / the `invs` table of `ozaki2_crt_gemm`
  * the residue-to-integer reconstruction step (its exact integer content)

  For residues `r_i` with `r_i < m_i`, the CRT sum
  `S = Σ r_i * inv_i * (M / m_i)` satisfies `S ≡ r_i (mod m_i)`, so
  `x = S % M` is the unique `0 ≤ x < M` with `x % m_i = r_i` for all `i`.

  The floating-point double-double evaluation of `S/M` in the Rust code is
  not modelled: the theorems here are the exact integer CRT statement.
-/

import G6ME.Basic

set_option maxRecDepth 10000

namespace G6ME.Crt

/-! ## Generic list lemmas -/

/-- Adding a constant on the right may be reduced modulo `n` on either side. -/
theorem add_mod_right_cancel (a b n : Nat) : (a + b) % n = (a % n + b) % n := by
  by_cases hn : n = 0
  · subst n; simp
  · calc (a + b) % n = (a % n + b % n) % n := Nat.add_mod a b n
      _ = ((a % n) % n + b % n) % n := by rw [Nat.mod_mod]
      _ = (a % n + b) % n := (Nat.add_mod (a % n) b n).symm

/-- Folding an additive sum keeps the same remainder as folding with a
reduction after every step. -/
theorem foldl_add_mod (l : List Nat) (f : Nat → Nat) (n init : Nat) :
    (l.foldl (fun acc i => acc + f i) init) % n
      = l.foldl (fun acc i => (acc + f i) % n) (init % n) := by
  induction l generalizing init with
  | nil => rfl
  | cons a t ih =>
      simp only [List.foldl_cons]
      rw [ih (init + f a)]
      rw [add_mod_right_cancel init (f a) n]

/-- Pointwise congruent summands give congruent additive folds. -/
theorem foldl_add_congr (l : List Nat) (f g : Nat → Nat) (n init init' : Nat)
    (hinit : init % n = init' % n)
    (h : ∀ i ∈ l, f i % n = g i % n) :
    (l.foldl (fun acc i => acc + f i) init) % n
      = (l.foldl (fun acc i => acc + g i) init') % n := by
  induction l generalizing init init' with
  | nil => exact hinit
  | cons a t ih =>
      simp only [List.foldl_cons]
      apply ih
      · rw [Nat.add_mod init (f a) n, Nat.add_mod init' (g a) n,
          hinit, h a (by simp)]
      · intro i hi
        exact h i (by simp [hi])

/-- The indicator fold over `List.range n` selects exactly the term at `j`. -/
theorem foldl_range_ite (n j c : Nat) :
    (List.range n).foldl (fun acc i => acc + (if i = j then c else 0)) 0
      = if j < n then c else 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [ih]
      by_cases hjn : j = n
      · subst hjn
        simp
      · have hnj : ¬ n = j := fun h => hjn h.symm
        by_cases hlt : j < n
        · have hlt' : j < Nat.succ n := Nat.lt_succ_of_lt hlt
          simp [hlt, hlt', hnj]
        · have hge : ¬ j < Nat.succ n := by omega
          simp [hlt, hge, hnj]

/-- Folding a product from an arbitrary initial value. -/
theorem foldl_mul_eq (l : List Nat) (b : Nat) :
    l.foldl (fun acc m => acc * m) b
      = b * l.foldl (fun acc m => acc * m) 1 := by
  induction l generalizing b with
  | nil => simp
  | cons a t ih =>
      simp only [List.foldl_cons]
      rw [ih (b * a)]
      simp only [Nat.one_mul]
      rw [ih a]
      ring_nat

/-- A number coprime to every element of a list is coprime to their product. -/
theorem foldl_coprime (l : List Nat) (a b : Nat) (hb : Nat.Coprime a b)
    (h : ∀ m ∈ l, Nat.Coprime a m) :
    Nat.Coprime a (l.foldl (fun acc m => acc * m) b) := by
  induction l generalizing b with
  | nil => exact hb
  | cons x t ih =>
      simp only [List.foldl_cons]
      apply ih (b := b * x)
      · exact Nat.Coprime.mul_right hb (h x (by simp))
      · intro m hm
        exact h m (by simp [hm])

/-- If every element of a pairwise-coprime list divides `d`, then so does the
product of the list. -/
theorem foldl_mul_dvd (l : List Nat) (d : Nat)
    (hcop : l.Pairwise Nat.Coprime) (hdvd : ∀ m ∈ l, m ∣ d) :
    l.foldl (fun acc m => acc * m) 1 ∣ d := by
  induction l with
  | nil => exact Nat.one_dvd d
  | cons a t ih =>
      rw [List.pairwise_cons] at hcop
      obtain ⟨ha, hcop_t⟩ := hcop
      have hstep : (a :: t).foldl (fun acc m => acc * m) 1
          = a * t.foldl (fun acc m => acc * m) 1 := by
        simp only [List.foldl_cons, Nat.one_mul]
        exact foldl_mul_eq t a
      rw [hstep]
      exact Nat.Coprime.mul_dvd_of_dvd_of_dvd
        (foldl_coprime t a 1 (Nat.coprime_one_right a) ha)
        (hdvd a (by simp))
        (ih hcop_t (fun m hm => hdvd m (by simp [hm])))

/-- If `a ≤ b` and `a`, `b` have the same remainder modulo `n > 0`, then
`n` divides `b - a`. -/
theorem dvd_sub_of_mod_eq {a b n : Nat} (hab : a ≤ b) (hn : 0 < n)
    (h : a % n = b % n) : n ∣ b - a := by
  have hb : a + (b - a) = b := Nat.add_sub_of_le hab
  have h' : (a + (b - a)) % n = a % n := by rw [hb]; exact h.symm
  rw [add_mod_right_cancel, Nat.add_mod, Nat.mod_mod] at h'
  have hd : (b - a) % n = 0 := by
    have ha := Nat.mod_lt a hn
    have hdlt := Nat.mod_lt (b - a) hn
    by_cases hlt : a % n + (b - a) % n < n
    · rw [Nat.mod_eq_of_lt hlt] at h'
      omega
    · rw [Nat.mod_eq_sub_mod (by omega : a % n + (b - a) % n ≥ n),
        Nat.mod_eq_of_lt (by omega : a % n + (b - a) % n - n < n)] at h'
      omega
  exact Nat.dvd_of_mod_eq_zero hd

/-! ## The concrete moduli and product -/

/-- `OZAKI2_MODULI`: 20 pairwise-coprime moduli, all `≤ 113`. -/
def moduli : List Nat :=
  [113, 109, 107, 103, 101, 97, 89, 83, 81, 79, 73, 71, 67, 64, 61, 59, 53, 49, 47, 43]

/-- The `i`-th modulus (out of range reads as `0`). -/
def mAt (i : Nat) : Nat := moduli.getD i 0

/-- `M = ∏ m_j`, the CRT modulus, computed by the same left fold as the Rust
loop over `OZAKI2_MODULI`. -/
def M : Nat := moduli.foldl (fun acc m => acc * m) 1

/-- The modular inverses `inv_j = (M / m_j)^{-1} mod m_j`, exactly the values
`mod_inv_u128` produces for `a = M / m_j`. -/
def invs : List Nat :=
  [21, 87, 6, 59, 22, 87, 41, 69, 38, 9, 13, 34, 65, 3, 5, 16, 8, 33, 18, 7]

/-- The `i`-th inverse (out of range reads as `0`). -/
def invAt (i : Nat) : Nat := invs.getD i 0

/-- The `i`-th CRT summand `r_i * inv_i * (M / m_i)`. -/
def crtTerm (rs : List Nat) (i : Nat) : Nat :=
  rs.getD i 0 * invAt i * (M / mAt i)

/-- The CRT sum `S = Σ_i r_i * inv_i * (M / m_i)`. -/
def crtSum (rs : List Nat) : Nat :=
  (List.range 20).foldl (fun acc i => acc + crtTerm rs i) 0

/-- The reconstructed representative `x = S % M`. -/
def crtX (rs : List Nat) : Nat := crtSum rs % M

/-! ## Concrete checks

All of these are kernel computations on the literal constants above. -/

theorem moduli_length : moduli.length = 20 := by decide

/-- The product of the 20 moduli is the 125-bit number the Rust code stores in
a `u128`. -/
theorem M_eq : M = 26390970204770983741947274447420058304 := by decide

theorem M_pos : 0 < M := by decide

/-- `log2(M) < 128`: the product fits the `u128` used by `ozaki2_crt_gemm`. -/
theorem M_lt_two128 : M < 2 ^ 128 := by decide

/-- The moduli are pairwise coprime. -/
theorem moduli_pairwise : List.Pairwise Nat.Coprime moduli := by decide

theorem mAt_pos_check : (List.range 20).all (fun i => decide (0 < mAt i)) = true := by
  decide

theorem mAt_dvd_M_check : (List.range 20).all (fun i => M % mAt i == 0) = true := by
  decide

/-- `inv_i * (M / m_i) ≡ 1 (mod m_i)`, the defining property of the inverse
table produced by `mod_inv_u128`. -/
theorem inv_check :
    (List.range 20).all (fun i => (invAt i * (M / mAt i)) % mAt i == 1) = true := by
  decide

/-- For `i ≠ j`, `M / m_i ≡ 0 (mod m_j)`: the other CRT summands vanish. -/
theorem cross_dvd_check :
    (List.range 20).all (fun i =>
      (List.range 20).all (fun j =>
        if i = j then true else (M / mAt i) % mAt j == 0)) = true := by
  decide

/-! ## Extracting the checks -/

theorem all_range {n : Nat} {p : Nat → Bool}
    (h : (List.range n).all p = true) : ∀ i, i < n → p i = true := by
  intro i hi
  exact (List.all_eq_true.mp h) i (List.mem_range.mpr hi)

theorem mAt_pos_of_lt {i : Nat} (hi : i < 20) : 0 < mAt i :=
  of_decide_eq_true (all_range mAt_pos_check i hi)

theorem mAt_dvd_M_of_lt {i : Nat} (hi : i < 20) : mAt i ∣ M :=
  Nat.dvd_of_mod_eq_zero (beq_iff_eq.mp (all_range mAt_dvd_M_check i hi))

theorem inv_check_of_lt {i : Nat} (hi : i < 20) :
    (invAt i * (M / mAt i)) % mAt i = 1 :=
  beq_iff_eq.mp (all_range inv_check i hi)

theorem cross_dvd_of_ne {i j : Nat} (hi : i < 20) (hj : j < 20) (hij : i ≠ j) :
    (M / mAt i) % mAt j = 0 := by
  have h1 := all_range cross_dvd_check i hi
  have h2 := all_range h1 j hj
  rw [if_neg hij] at h2
  exact beq_iff_eq.mp h2

/-- `getD` at an in-range index is the corresponding list element. -/
theorem getD_eq_getElem {l : List Nat} {i d : Nat} (h : i < l.length) :
    l.getD i d = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

theorem mAt_eq_getElem (i : Nat) (hi : i < moduli.length) : mAt i = moduli[i] :=
  getD_eq_getElem hi

theorem mAt_mem {i : Nat} (hi : i < 20) : mAt i ∈ moduli := by
  have hi' : i < moduli.length := by rw [moduli_length]; exact hi
  rw [mAt_eq_getElem i hi']
  exact List.getElem_mem hi'

/-! ## The CRT existence theorem -/

/-- Each CRT summand reduces to `r_j` at `j` and to `0` at every other modulus. -/
theorem crtTerm_mod (rs : List Nat) (j : Nat) (hj : j < 20)
    (hbound : rs.getD j 0 < mAt j) :
    ∀ i ∈ List.range 20,
      crtTerm rs i % mAt j = (if i = j then rs.getD j 0 else 0) % mAt j := by
  intro i hi
  have hi20 : i < 20 := List.mem_range.mp hi
  by_cases hij : i = j
  · have hterm : crtTerm rs i % mAt j = rs.getD j 0 := by
      rw [hij]
      have hx1 : (invAt j * (M / mAt j)) % mAt j = 1 := inv_check_of_lt hj
      unfold crtTerm
      rw [Nat.mul_assoc, Nat.mul_mod, hx1, Nat.mul_one, Nat.mod_mod,
        Nat.mod_eq_of_lt hbound]
    rw [hterm, if_pos hij]
    exact (Nat.mod_eq_of_lt hbound).symm
  · have h0 : (M / mAt i) % mAt j = 0 := cross_dvd_of_ne hi20 hj hij
    have hx0 : (invAt i * (M / mAt i)) % mAt j = 0 := by
      rw [Nat.mul_mod, h0]
      simp
    have hterm : crtTerm rs i % mAt j = 0 := by
      unfold crtTerm
      rw [Nat.mul_assoc, Nat.mul_mod, hx0]
      simp
    rw [if_neg hij, hterm]
    simp

/-- `crtX rs` has the prescribed residues. -/
theorem crtX_spec (rs : List Nat) (j : Nat) (hj : j < 20)
    (hbound : rs.getD j 0 < mAt j) :
    crtX rs % mAt j = rs.getD j 0 := by
  have hdvd : mAt j ∣ M := mAt_dvd_M_of_lt hj
  have hx : crtX rs % mAt j = crtSum rs % mAt j := by
    unfold crtX
    exact Nat.mod_mod_of_dvd (crtSum rs) hdvd
  have hcongr := foldl_add_congr (List.range 20) (crtTerm rs)
    (fun i => if i = j then rs.getD j 0 else 0) (mAt j) 0 0
    (by simp) (crtTerm_mod rs j hj hbound)
  have hred : crtSum rs % mAt j
      = (List.range 20).foldl
          (fun acc i => acc + (if i = j then rs.getD j 0 else 0)) 0 % mAt j := by
    unfold crtSum
    exact hcongr
  have hg : (List.range 20).foldl
      (fun acc i => acc + (if i = j then rs.getD j 0 else 0)) 0 = rs.getD j 0 := by
    rw [foldl_range_ite]
    simp [hj]
  calc crtX rs % mAt j = crtSum rs % mAt j := hx
    _ = (List.range 20).foldl
          (fun acc i => acc + (if i = j then rs.getD j 0 else 0)) 0 % mAt j := hred
    _ = rs.getD j 0 % mAt j := by rw [hg]
    _ = rs.getD j 0 := Nat.mod_eq_of_lt (hbound)

/-- **CRT existence**: for any residue list of length 20 respecting the moduli,
there is a unique `x < M` with `x % m_i = r_i`. -/
theorem crt_exists (rs : List Nat) (_hlen : rs.length = 20)
    (hbound : ∀ i, i < 20 → rs.getD i 0 < mAt i) :
    ∃ x, x < M ∧ ∀ i, i < 20 → x % mAt i = rs.getD i 0 :=
  ⟨crtX rs, Nat.mod_lt _ M_pos, fun i hi => crtX_spec rs i hi (hbound i hi)⟩

/-! ## Uniqueness -/

/-- **CRT uniqueness** (the `x ≤ y` case). -/
theorem crt_unique_le (x y : Nat) (hxy : x ≤ y) (_hx : x < M) (hy : y < M)
    (h : ∀ i, i < 20 → x % mAt i = y % mAt i) : x = y := by
  have hd : ∀ i, i < 20 → mAt i ∣ y - x := by
    intro i hi
    exact dvd_sub_of_mod_eq hxy (mAt_pos_of_lt hi) (h i hi)
  have hMdvd : M ∣ y - x := by
    unfold M
    apply foldl_mul_dvd
    · exact moduli_pairwise
    · intro m hm
      rcases List.mem_iff_getElem.mp hm with ⟨i, hiLen, hget⟩
      have hi20 : i < 20 := by rw [← moduli_length]; exact hiLen
      have hmEq : m = mAt i := by
        rw [← hget]
        exact (mAt_eq_getElem i hiLen).symm
      rw [hmEq]
      exact hd i hi20
  have hlt : y - x < M := by omega
  have h0 : y - x = 0 := by
    have hmod := Nat.mod_eq_zero_of_dvd hMdvd
    rwa [Nat.mod_eq_of_lt hlt] at hmod
  omega

/-- **CRT uniqueness**: two representatives below `M` with the same residues
are equal. -/
theorem crt_unique (x y : Nat) (hx : x < M) (hy : y < M)
    (h : ∀ i, i < 20 → x % mAt i = y % mAt i) : x = y := by
  by_cases hxy : x ≤ y
  · exact crt_unique_le x y hxy hx hy h
  · have hyx : y ≤ x := by omega
    exact (crt_unique_le y x hyx hy hx (fun i hi => (h i hi).symm)).symm

end G6ME.Crt
