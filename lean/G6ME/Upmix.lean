/-
  G6ME.Upmix — the exact algebra of the 2×2 Hermitian PCA/ASE/APES stage in
  `crates/g6me-core/src/upmix.rs::pae_decompose_bin`.

  Modelled over `Int` (a rational abstraction; no `f64` rounding, `sqrt`,
  `hypot`, `clamp`, or eigenvalue `disc` computation):

  * complex arithmetic (`Cx`), the norm form `|z|² = re² + im²`, and the
    multiplicativity `|z w|² = |z|² |w|²`,
  * the constructed second eigenvector `v₂ = (-conj v₁_1, conj v₁_0)` is
    orthogonal to `v₁` and has the same norm,
  * the exact Parseval identity for that basis:
    `|⟨v₁,x⟩|² + |⟨v₂,x⟩|² = (|v₁_0|²+|v₁_1|²)(|x_0|²+|x_1|²)`.
    The Rust normalises by `hypot(a,b)`, so `|v₁|² = 1` and the projections
    conserve `|xl|² + |xr|²` exactly; we prove the unnormalised form and the
    normalised corollary,
  * the quadrature rotation used by the APES ambient path preserves the norm,
  * the primary path's energy `|p_l|² + |p_r|² = w_p² |proj₁|²`,
  * the final ambient rescale: if the squared scale `s²` is chosen so that
    `s² e_a_cur = e_in - e_p` (the Rust `s = sqrt(target / cur)`), then the
    total output energy is exactly the input energy.

  NOT modelled: the floating-point EVD (`disc = sqrt((r11-r22)² + 4|r12|²)`,
  `λ1`, `λ2`), the clamping of `p_frac` and `centerness`, the phase division
  `r12 / |r12|`, and the ambient cross-term before the final rescale.  Those
  are `f64`/`sqrt`/`clamp` computations, not polynomial identities.
-/

import G6ME.Basic

set_option maxHeartbeats 1000000

namespace G6ME.Upmix

/-! ## Complex numbers as integer pairs -/

structure Cx where
  re : Int
  im : Int

def Cx.zero : Cx := ⟨0, 0⟩

def Cx.add (z w : Cx) : Cx := ⟨z.re + w.re, z.im + w.im⟩

def Cx.neg (z : Cx) : Cx := ⟨-z.re, -z.im⟩

def Cx.conj (z : Cx) : Cx := ⟨z.re, -z.im⟩

def Cx.mul (z w : Cx) : Cx :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

def Cx.scale (k : Int) (z : Cx) : Cx := ⟨k * z.re, k * z.im⟩

/-- `Complex::norm_sq`. -/
def Cx.normSq (z : Cx) : Int := z.re * z.re + z.im * z.im

/-- Real inner product `Re(z conj w)`. -/
def dot (z w : Cx) : Int := z.re * w.re + z.im * w.im

/-! ## Ring laws -/

theorem mul_comm (z w : Cx) : Cx.mul z w = Cx.mul w z := by
  cases z with | mk zr zi =>
  cases w with | mk wr wi =>
  simp only [Cx.mul]
  rw [Cx.mk.injEq]
  constructor
  · rw [Int.mul_comm zr wr, Int.mul_comm zi wi]
  · rw [Int.mul_comm zr wi, Int.mul_comm zi wr, Int.add_comm]

theorem mul_assoc (x y z : Cx) :
    Cx.mul (Cx.mul x y) z = Cx.mul x (Cx.mul y z) := by
  cases x with | mk xr xi =>
  cases y with | mk yr yi =>
  cases z with | mk zr zi =>
  simp only [Cx.mul]
  rw [Cx.mk.injEq]
  constructor <;> ring_int3

theorem conj_mul (z w : Cx) :
    Cx.conj (Cx.mul z w) = Cx.mul (Cx.conj z) (Cx.conj w) := by
  cases z with | mk zr zi =>
  cases w with | mk wr wi =>
  simp only [Cx.mul, Cx.conj]
  rw [Cx.mk.injEq]
  constructor <;> ring_int3

theorem conj_conj (z : Cx) : Cx.conj (Cx.conj z) = z := by
  cases z with | mk zr zi => simp [Cx.conj]

theorem conj_neg (z : Cx) : Cx.conj (Cx.neg z) = Cx.neg (Cx.conj z) := by
  cases z with | mk zr zi => simp [Cx.conj, Cx.neg]

theorem mul_neg_left (z w : Cx) : Cx.mul (Cx.neg z) w = Cx.neg (Cx.mul z w) := by
  cases z with | mk zr zi => cases w with | mk wr wi =>
  simp [Cx.mul, Cx.neg]
  constructor <;> ring_int3

/-! ## Norm identities -/

theorem normSq_conj (z : Cx) : Cx.normSq (Cx.conj z) = Cx.normSq z := by
  cases z with | mk zr zi =>
  simp [Cx.normSq, Cx.conj, Int.neg_mul_neg]

theorem normSq_neg (z : Cx) : Cx.normSq (Cx.neg z) = Cx.normSq z := by
  cases z with | mk zr zi =>
  simp [Cx.normSq, Cx.neg, Int.neg_mul_neg]

/-- `|z w|² = |z|² |w|²`. -/
theorem normSq_mul (z w : Cx) :
    Cx.normSq (Cx.mul z w) = Cx.normSq z * Cx.normSq w := by
  cases z with | mk zr zi =>
  cases w with | mk wr wi =>
  simp [Cx.normSq, Cx.mul]
  ring_int3

/-- `|z + w|² = |z|² + |w|² + 2 Re(z conj w)`. -/
theorem normSq_add (z w : Cx) :
    Cx.normSq (Cx.add z w)
      = Cx.normSq z + Cx.normSq w + dot z w + dot z w := by
  cases z with | mk zr zi =>
  cases w with | mk wr wi =>
  simp [Cx.normSq, Cx.add, dot]
  ring_int3

theorem normSq_scale (k : Int) (z : Cx) :
    Cx.normSq (Cx.scale k z) = k * k * Cx.normSq z := by
  cases z with | mk zr zi =>
  simp [Cx.normSq, Cx.scale]
  ring_int3

theorem normSq_conj_mul (p x : Cx) :
    Cx.normSq (Cx.mul (Cx.conj p) x) = Cx.normSq p * Cx.normSq x := by
  rw [normSq_mul, normSq_conj]

theorem normSq_neg_mul (p x : Cx) :
    Cx.normSq (Cx.mul (Cx.neg p) x) = Cx.normSq p * Cx.normSq x := by
  rw [normSq_mul, normSq_neg]

theorem dot_eq_re_mul_conj (z w : Cx) :
    dot z w = (Cx.mul z (Cx.conj w)).re := by
  cases z with | mk zr zi =>
  cases w with | mk wr wi =>
  simp [dot, Cx.mul, Cx.conj, Int.mul_neg]

/-! ## The second eigenvector `v₂ = (-conj v₁_1, conj v₁_0)` -/

/-- The Rust construction of the orthogonal eigenvector, for
`v₁ = (p, q)`: `v₂ = (neg (conj q), conj p)`. -/
def v2Of (p q : Cx) : Cx × Cx := (Cx.neg (Cx.conj q), Cx.conj p)

def ip (u v : Cx × Cx) : Cx :=
  Cx.add (Cx.mul (Cx.conj u.1) v.1) (Cx.mul (Cx.conj u.2) v.2)

def n2 (u : Cx × Cx) : Int := Cx.normSq u.1 + Cx.normSq u.2

theorem v2_ip_zero (p q : Cx) : ip (p, q) (v2Of p q) = Cx.zero := by
  simp [ip, v2Of, Cx.zero, Cx.add, Cx.mul, Cx.conj, Cx.neg]
  constructor <;> ring_int3

theorem v2_norm (p q : Cx) : n2 (v2Of p q) = n2 (p, q) := by
  simp [n2, v2Of, normSq_neg, normSq_conj, Int.add_comm]

/-! ## Parseval for `{v₁, v₂}`

`proj1 = conj(p) x0 + conj(q) x1`,
`proj2 = conj(v2_0) x0 + conj(v2_1) x1 = (-q) x0 + p x1`. -/

/-- The cross terms `Re((conj p x0) conj(conj q x1))` and
`Re((-q x0) conj(p x1))` are negatives of each other. -/
theorem cross_eq (p q x0 x1 : Cx) :
    dot (Cx.mul (Cx.conj p) x0) (Cx.mul (Cx.conj q) x1)
      = dot (Cx.mul q x0) (Cx.mul p x1) := by
  rw [dot_eq_re_mul_conj, dot_eq_re_mul_conj, conj_mul, conj_mul, conj_conj]
  have h : Cx.mul (Cx.mul (Cx.conj p) x0) (Cx.mul q (Cx.conj x1))
      = Cx.mul (Cx.mul q x0) (Cx.mul (Cx.conj p) (Cx.conj x1)) := by
    calc Cx.mul (Cx.mul (Cx.conj p) x0) (Cx.mul q (Cx.conj x1))
        = Cx.mul (Cx.conj p) (Cx.mul x0 (Cx.mul q (Cx.conj x1))) := by
            rw [mul_assoc]
      _ = Cx.mul (Cx.conj p) (Cx.mul (Cx.mul x0 q) (Cx.conj x1)) := by
            rw [← mul_assoc x0 q (Cx.conj x1)]
      _ = Cx.mul (Cx.conj p) (Cx.mul (Cx.mul q x0) (Cx.conj x1)) := by
            rw [mul_comm x0 q]
      _ = Cx.mul (Cx.mul (Cx.conj p) (Cx.mul q x0)) (Cx.conj x1) := by
            rw [← mul_assoc (Cx.conj p) (Cx.mul q x0) (Cx.conj x1)]
      _ = Cx.mul (Cx.mul (Cx.mul q x0) (Cx.conj p)) (Cx.conj x1) := by
            rw [mul_comm (Cx.conj p) (Cx.mul q x0)]
      _ = Cx.mul (Cx.mul q x0) (Cx.mul (Cx.conj p) (Cx.conj x1)) := by
            rw [mul_assoc (Cx.mul q x0) (Cx.conj p) (Cx.conj x1)]
  rw [h]

theorem dot_neg_left (z w : Cx) : dot (Cx.neg z) w = -dot z w := by
  cases z with | mk zr zi => cases w with | mk wr wi =>
  simp [dot, Cx.neg]
  ring_int3

theorem cross_neg (p q x0 x1 : Cx) :
    dot (Cx.mul (Cx.conj p) x0) (Cx.mul (Cx.conj q) x1)
      = - dot (Cx.mul (Cx.neg q) x0) (Cx.mul p x1) := by
  rw [mul_neg_left, dot_neg_left, Int.neg_neg, cross_eq]

/-- **Parseval, unnormalised**: the two projections onto the constructed
orthogonal basis satisfy `|proj1|² + |proj2|² = (|p|²+|q|²)(|x0|²+|x1|²)`. -/
theorem parseval_specific (p q x0 x1 : Cx) :
    Cx.normSq (Cx.add (Cx.mul (Cx.conj p) x0) (Cx.mul (Cx.conj q) x1))
      + Cx.normSq (Cx.add (Cx.mul (Cx.neg q) x0) (Cx.mul p x1))
      = (Cx.normSq p + Cx.normSq q) * (Cx.normSq x0 + Cx.normSq x1) := by
  rw [normSq_add, normSq_add, normSq_conj_mul, normSq_conj_mul,
    normSq_neg_mul, normSq_mul]
  rw [cross_neg]
  ring_int3

/-- **Parseval, normalised**: when `v₁` is a unit vector
(`hypot(a,b)` normalisation in the Rust), the projections conserve the
input energy `|x0|² + |x1|²`. -/
theorem parseval_unit (p q x0 x1 : Cx)
    (h : Cx.normSq p + Cx.normSq q = 1) :
    Cx.normSq (Cx.add (Cx.mul (Cx.conj p) x0) (Cx.mul (Cx.conj q) x1))
      + Cx.normSq (Cx.add (Cx.mul (Cx.neg q) x0) (Cx.mul p x1))
      = Cx.normSq x0 + Cx.normSq x1 := by
  rw [parseval_specific, h, Int.one_mul]

/-! ## Quadrature rotation and the final energy arithmetic -/

/-- `+j z = (-Im z, Re z)`, the APES `rot_r1_l`. -/
def rotPlus (z : Cx) : Cx := ⟨-z.im, z.re⟩

/-- `-j z = (Im z, -Re z)`, the APES `rot_r1_r`. -/
def rotMinus (z : Cx) : Cx := ⟨z.im, -z.re⟩

theorem normSq_rotPlus (z : Cx) : Cx.normSq (rotPlus z) = Cx.normSq z := by
  cases z with | mk zr zi =>
  simp [Cx.normSq, rotPlus, Int.neg_mul_neg, Int.add_comm]

theorem normSq_rotMinus (z : Cx) : Cx.normSq (rotMinus z) = Cx.normSq z := by
  cases z with | mk zr zi =>
  simp [Cx.normSq, rotMinus, Int.neg_mul_neg, Int.add_comm]

/-- Primary-path energy: scaling both components of `proj₁ v₁` by `w_p`
gives `w_p² |proj₁|²` when `v₁` is unit. -/
theorem primary_energy (wp : Int) (proj v0 v1 : Cx)
    (h : Cx.normSq v0 + Cx.normSq v1 = 1) :
    Cx.normSq (Cx.scale wp (Cx.mul proj v0))
      + Cx.normSq (Cx.scale wp (Cx.mul proj v1))
      = wp * wp * Cx.normSq proj := by
  rw [normSq_scale, normSq_scale, normSq_mul, normSq_mul,
    ← Int.mul_add, ← Int.mul_add, h, Int.mul_one]

/-- The final APES rescale: if the squared scale `s²` satisfies
`s² (|a_l|² + |a_r|²) = target` (the Rust `s = sqrt(target / cur)`), the
scaled pair has exactly the target energy. -/
theorem rescale_energy (a b : Cx) (s target : Int)
    (h : s * s * (Cx.normSq a + Cx.normSq b) = target) :
    Cx.normSq (Cx.scale s a) + Cx.normSq (Cx.scale s b) = target := by
  rw [normSq_scale, normSq_scale, ← Int.mul_add, h]

/-- **Total per-bin energy conservation**: primary energy plus the rescaled
ambient energy equals the input energy. -/
theorem ambient_rescale_conservation (a_l a_r : Cx) (e_in e_p s : Int)
    (h : s * s * (Cx.normSq a_l + Cx.normSq a_r) = e_in - e_p) :
    e_p + (Cx.normSq (Cx.scale s a_l) + Cx.normSq (Cx.scale s a_r)) = e_in := by
  have hres := rescale_energy a_l a_r s (e_in - e_p) h
  omega

/-- The zero-input branch of `pae_decompose_bin` returns zero energy. -/
theorem zero_branch_energy (p_l p_r a_l a_r : Cx)
    (h1 : p_l = Cx.zero) (h2 : p_r = Cx.zero)
    (h3 : a_l = Cx.zero) (h4 : a_r = Cx.zero) :
    Cx.normSq p_l + Cx.normSq p_r + Cx.normSq a_l + Cx.normSq a_r = 0 := by
  rw [h1, h2, h3, h4]
  simp [Cx.zero, Cx.normSq]

end G6ME.Upmix
