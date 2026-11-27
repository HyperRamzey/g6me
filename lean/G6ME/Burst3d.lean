/-
  G6ME.Burst3d — the exact algebra of the Rodrigues `SO(3)` exponential and
  logarithm in `crates/g6me-core/src/burst3d.rs`.

  Modelled exactly, over `Int`:

  * `skew` (`K`),
  * the algebraic identities `Kᵀ = -K`, `K² = w wᵀ - θ² I`, `K³ = -θ² K`,
  * the Rodrigues matrix `R = I + a K + b K²` written in the explicit form
    `R = (1 - b θ²) I + a K + b w wᵀ` used by `so3_exp`,
  * the skew part `R - Rᵀ = 2 a K` from which `so3_log` reads its rotation
    vector,
  * the axis invariance `R w = w`,
  * the logarithm roundtrip: if the scale factor satisfies `scale * a = 1`
    (the exact branch `θ / sin θ`, where `a = sin θ / θ`), then the scaled
    doubled log vector recovers `w`.

  NOT modelled: `f64` rounding, `sqrt`, `sin`, `cos`, `acos`, the `clamp`
  calls, and the `|φ| < 1e-7` Taylor branch of `so3_exp` / `so3_log` (that
  branch sets `a = 1 - θ²/6`, `b = 1/2 - θ²/24`, which does not satisfy the
  exact orthogonality relation).

  The orthogonality identity `Rᵀ R = I` is **not a polynomial identity of
  this model** in the free parameters `a` and `b`: expanding with the lemmas
  below gives

    `Rᵀ R = ((1-bθ²)² + a²θ²) I + (2b - a² - b²θ²) w wᵀ`.

  The conditional form is proved below as `R_gram_conditional`: if the two
  scalar side conditions `2b - a² - b²θ² = 0` and `(1-bθ²)² + a²θ² = 1`
  hold, then `Rᵀ R = I` entrywise.  The Rust obtains `a = sin θ / θ` and
  `b = (1 - cos θ) / θ²`, for which those conditions are the trigonometric
  identity `sin²θ + cos²θ = 1`; establishing that instantiation needs a real
  `sin`/`cos`/`sqrt` development, which this core-only formalization
  deliberately excludes, so the trigonometric instantiation and `det R = 1`
  are still not proved here.
-/

import G6ME.Basic

set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

namespace G6ME.Burst3d

/-- A 3×3 integer matrix, indexed over all `Nat` pairs; only the `3×3` block
is meaningful, and every definition below vanishes outside it. -/
abbrev Mat3 := Nat → Nat → Int

/-- `3×3` case split: every `i < 3` is `0`, `1`, or `2`. -/
theorem nat3_cases {i : Nat} (h : i < 3) : i = 0 ∨ i = 1 ∨ i = 2 := by omega

/-- Kronecker delta. -/
def delta (i j : Nat) : Int := if i = j then 1 else 0

/-- The `i`-th coordinate of `(w0, w1, w2)`, zero outside the block. -/
def wget (w0 w1 w2 : Int) (i : Nat) : Int :=
  if i = 0 then w0 else if i = 1 then w1 else if i = 2 then w2 else 0

/-- `θ² = w0² + w1² + w2²`. -/
def theta2 (w0 w1 w2 : Int) : Int := w0 * w0 + w1 * w1 + w2 * w2

/-- `burst3d.rs::skew`. -/
def K (w0 w1 w2 : Int) : Mat3 :=
  fun i j =>
    if i = 0 then (if j = 1 then -w2 else if j = 2 then w1 else 0)
    else if i = 1 then (if j = 0 then w2 else if j = 2 then -w0 else 0)
    else if i = 2 then (if j = 0 then -w1 else if j = 1 then w0 else 0)
    else 0

/-- `burst3d.rs::mat3_mul`. -/
def mul (a b : Mat3) : Mat3 :=
  fun i j => a i 0 * b 0 j + a i 1 * b 1 j + a i 2 * b 2 j

/-- `burst3d.rs::mat3_transpose`. -/
def transpose (a : Mat3) : Mat3 := fun i j => a j i

/-- The outer product `w wᵀ`. -/
def ww (w0 w1 w2 : Int) : Mat3 :=
  fun i j => wget w0 w1 w2 i * wget w0 w1 w2 j

/-- `burst3d.rs::mat3_vec`. -/
def mulVec (m : Mat3) (v : Nat → Int) (i : Nat) : Int :=
  m i 0 * v 0 + m i 1 * v 1 + m i 2 * v 2

/-- The Rodrigues matrix in the explicit form used by `so3_exp`:
`R = (1 - b θ²) I + a K + b w wᵀ`, where the Rust code has
`a = sin θ / θ` and `b = (1 - cos θ) / θ²`. -/
def R (a b w0 w1 w2 : Int) : Mat3 :=
  fun i j =>
    (1 - b * theta2 w0 w1 w2) * delta i j
      + a * K w0 w1 w2 i j
      + b * wget w0 w1 w2 i * wget w0 w1 w2 j

/-! ## The Lie-algebra identities -/

/-- `K` is skew-symmetric: `Kᵀ = -K`. -/
theorem K_skew (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    transpose (K w0 w1 w2) i j = -K w0 w1 w2 i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [transpose, K]

/-- `K² = w wᵀ - θ² I`. -/
theorem K_sq (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (K w0 w1 w2) (K w0 w1 w2) i j
      = ww w0 w1 w2 i j - theta2 w0 w1 w2 * delta i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [mul, K, ww, wget, theta2, delta] <;> ring_int

/-- `K³ = -θ² K`, the characteristic relation of a skew matrix in 3D. -/
theorem K_cube (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (mul (K w0 w1 w2) (K w0 w1 w2)) (K w0 w1 w2) i j
      = -theta2 w0 w1 w2 * K w0 w1 w2 i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [mul, K, wget, theta2] <;> ring_int

/-- `K w = 0`: the rotation axis is the kernel of `K`, hence of `K²`. -/
theorem K_mul_w (w0 w1 w2 : Int) (i : Nat) (hi : i < 3) :
    mulVec (K w0 w1 w2) (wget w0 w1 w2) i = 0 := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  simp [mulVec, K, wget] <;> ring_int

/-- `K (w wᵀ) = 0`. -/
theorem K_ww_zero (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (K w0 w1 w2) (ww w0 w1 w2) i j = 0 := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [mul, K, ww, wget] <;> ring_int

/-- `(w wᵀ) K = 0`. -/
theorem ww_K_zero (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (ww w0 w1 w2) (K w0 w1 w2) i j = 0 := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [mul, K, ww, wget] <;> ring_int

/-- `(w wᵀ)² = θ² (w wᵀ)`. -/
theorem ww_ww (w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (ww w0 w1 w2) (ww w0 w1 w2) i j
      = theta2 w0 w1 w2 * ww w0 w1 w2 i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [mul, ww, wget, theta2] <;> ring_int

/-! ## The exponential's skew part and the logarithm -/

/-- `R - Rᵀ = 2 a K`.  This is exactly the matrix from which `so3_log`
reads `v = ½ (R₂₁ - R₁₂, R₀₂ - R₂₀, R₁₀ - R₀₁)`; the factor `2` is the
`0.5` of the Rust code, kept doubled so the identity stays integral. -/
theorem R_sub_transpose (a b w0 w1 w2 : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    R a b w0 w1 w2 i j - transpose (R a b w0 w1 w2) i j
      = a * K w0 w1 w2 i j + a * K w0 w1 w2 i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [R, transpose, K, wget, theta2, delta] <;> ring_int

/-- The axis is fixed: `R w = w`. -/
theorem R_mul_w (a b w0 w1 w2 : Int) (i : Nat) (hi : i < 3) :
    mulVec (R a b w0 w1 w2) (wget w0 w1 w2) i = wget w0 w1 w2 i := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  simp [mulVec, R, K, wget, theta2, delta] <;> ring_int

/-- `R = I` when `w = 0`. -/
theorem R_zero (a b : Int) (i j : Nat) (hi : i < 3) (hj : j < 3) :
    R a b 0 0 0 i j = delta i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;>
  rcases nat3_cases hj with rfl | rfl | rfl <;>
  simp [R, K, wget, theta2, delta]

/-- **Logarithm roundtrip, doubled.**  If the `so3_log` scale factor
inverts `a` (`scale * a = 1`, true on the exact branch where
`a = sin θ / θ` and `scale = θ / sin θ`), then the scaled doubled log
vector is `2 w`.  Rust's `v` is the half of our `R i j - R j i`, so its
scaled value is `w`; the halving is not representable in `Int` and is left
as the factor `2` on the right-hand side. -/
theorem log_roundtrip (a b w0 w1 w2 scale : Int) (h : scale * a = 1)
    (i j : Nat) (hi : i < 3) (hj : j < 3) :
    scale * (R a b w0 w1 w2 i j - R a b w0 w1 w2 j i)
      = K w0 w1 w2 i j + K w0 w1 w2 i j := by
  have hk : R a b w0 w1 w2 i j - R a b w0 w1 w2 j i
      = a * K w0 w1 w2 i j + a * K w0 w1 w2 i j := by
    simpa [transpose] using R_sub_transpose a b w0 w1 w2 i j hi hj
  rw [hk]
  calc scale * (a * K w0 w1 w2 i j + a * K w0 w1 w2 i j)
      = (scale * a) * (K w0 w1 w2 i j + K w0 w1 w2 i j) := by ring_int
    _ = 1 * (K w0 w1 w2 i j + K w0 w1 w2 i j) := by rw [h]
    _ = K w0 w1 w2 i j + K w0 w1 w2 i j := by ring_int

/-- First component of the doubled rotation vector, `v₀ = ½ (R₂₁ - R₁₂)`. -/
theorem log_roundtrip_w0 (a b w0 w1 w2 scale : Int) (h : scale * a = 1) :
    scale * (R a b w0 w1 w2 2 1 - R a b w0 w1 w2 1 2) = w0 + w0 := by
  have hmain := log_roundtrip a b w0 w1 w2 scale h 2 1 (by decide) (by decide)
  simpa [K] using hmain

/-- Second component of the doubled rotation vector, `v₁ = ½ (R₀₂ - R₂₀)`. -/
theorem log_roundtrip_w1 (a b w0 w1 w2 scale : Int) (h : scale * a = 1) :
    scale * (R a b w0 w1 w2 0 2 - R a b w0 w1 w2 2 0) = w1 + w1 := by
  have hmain := log_roundtrip a b w0 w1 w2 scale h 0 2 (by decide) (by decide)
  simpa [K] using hmain

/-- Third component of the doubled rotation vector, `v₂ = ½ (R₁₀ - R₀₁)`. -/
theorem log_roundtrip_w2 (a b w0 w1 w2 scale : Int) (h : scale * a = 1) :
    scale * (R a b w0 w1 w2 1 0 - R a b w0 w1 w2 0 1) = w2 + w2 := by
  have hmain := log_roundtrip a b w0 w1 w2 scale h 1 0 (by decide) (by decide)
  simpa [K] using hmain

/-! ## Orthogonality from the two scalar side conditions

`R` is `c I + a K + b w wᵀ` with `c = 1 - b θ²`.  The Gram product is
expanded with the matrix combinators below, the Lie-algebra identities
`K² = w wᵀ - θ² I`, `K (w wᵀ) = 0`, `(w wᵀ) K = 0`, `(w wᵀ)² = θ² w wᵀ`
collapse the nine terms, and the scalar identity `gram_scalar` collects them
into `((1-bθ²)² + a²θ²) I + (2b - a² - b²θ²) w wᵀ`. -/

/-- Entrywise scalar multiplication of matrices. -/
def smul (c : Int) (A : Mat3) : Mat3 := fun i j => c * A i j

/-- Entrywise sum of matrices. -/
def addM (A B : Mat3) : Mat3 := fun i j => A i j + B i j

/-- The identity matrix on the block (`delta` everywhere). -/
def idM : Mat3 := fun i j => delta i j

/-- Multiplication is additive in its left argument. -/
theorem mul_add_left (A B C : Mat3) (i j : Nat) :
    mul (addM A B) C i j = mul A C i j + mul B C i j := by
  simp only [mul, addM]
  ring_int

/-- Multiplication is additive in its right argument. -/
theorem mul_add_right (A B C : Mat3) (i j : Nat) :
    mul A (addM B C) i j = mul A B i j + mul A C i j := by
  simp only [mul, addM]
  ring_int

/-- A scalar on the left factors out of a product. -/
theorem mul_smul_left (c : Int) (A B : Mat3) (i j : Nat) :
    mul (smul c A) B i j = c * mul A B i j := by
  simp only [mul, smul]
  ring_int

/-- A scalar on the right factors out of a product. -/
theorem mul_smul_right (c : Int) (A B : Mat3) (i j : Nat) :
    mul A (smul c B) i j = c * mul A B i j := by
  simp only [mul, smul]
  ring_int

/-- Two scalars factor out of a product. -/
theorem mul_smul_smul (x y : Int) (A B : Mat3) (i j : Nat) :
    mul (smul x A) (smul y B) i j = x * (y * mul A B i j) := by
  rw [mul_smul_left, mul_smul_right]

/-- `I` is a left identity. -/
theorem mul_idM_left (A : Mat3) (i j : Nat) (hi : i < 3) :
    mul idM A i j = A i j := by
  rcases nat3_cases hi with rfl | rfl | rfl <;> simp [mul, idM, delta]

/-- `I` is a right identity. -/
theorem mul_idM_right (A : Mat3) (i j : Nat) (hj : j < 3) :
    mul A idM i j = A i j := by
  rcases nat3_cases hj with rfl | rfl | rfl <;> simp [mul, idM, delta]

/-- `I · I = I`. -/
theorem mul_idM_idM (i j : Nat) (hi : i < 3) (_hj : j < 3) :
    mul idM idM i j = delta i j := by
  rw [mul_idM_left idM i j hi]
  rfl

/-- `delta` is symmetric. -/
theorem delta_comm (i j : Nat) : delta j i = delta i j := by
  by_cases h : i = j
  · subst h
    simp [delta]
  · have h' : ¬ j = i := fun hh => h hh.symm
    simp [delta, h, h']

/-- `K` is skew-symmetric without the block bounds (both sides vanish
outside the block). -/
theorem K_skew_all (w0 w1 w2 : Int) (i j : Nat) :
    K w0 w1 w2 j i = -K w0 w1 w2 i j := by
  by_cases hi : i < 3
  · by_cases hj : j < 3
    · rcases nat3_cases hi with rfl | rfl | rfl <;>
        rcases nat3_cases hj with rfl | rfl | rfl <;> simp [K]
    · have hj0 : j ≠ 0 := by omega
      have hj1 : j ≠ 1 := by omega
      have hj2 : j ≠ 2 := by omega
      rcases nat3_cases hi with rfl | rfl | rfl <;> simp [K, hj0, hj1, hj2]
  · have hi0 : i ≠ 0 := by omega
    have hi1 : i ≠ 1 := by omega
    have hi2 : i ≠ 2 := by omega
    by_cases hj : j < 3
    · rcases nat3_cases hj with rfl | rfl | rfl <;> simp [K, hi0, hi1, hi2]
    · have hj0 : j ≠ 0 := by omega
      have hj1 : j ≠ 1 := by omega
      have hj2 : j ≠ 2 := by omega
      simp [K, hi0, hi1, hi2, hj0, hj1, hj2]

/-- `R` as `c I + a K + b w wᵀ` with `c = 1 - b θ²`. -/
theorem R_as_addM (a b w0 w1 w2 : Int) :
    R a b w0 w1 w2
      = addM (addM (smul (1 - b * theta2 w0 w1 w2) idM)
          (smul a (K w0 w1 w2))) (smul b (ww w0 w1 w2)) := by
  funext i j
  simp only [R, addM, smul, idM, ww]
  ring_int

/-- `Rᵀ` as `c I - a K + b w wᵀ`, using `Kᵀ = -K`. -/
theorem transpose_R_as_addM (a b w0 w1 w2 : Int) :
    transpose (R a b w0 w1 w2)
      = addM (addM (smul (1 - b * theta2 w0 w1 w2) idM)
          (smul (-a) (K w0 w1 w2))) (smul b (ww w0 w1 w2)) := by
  funext i j
  simp only [transpose, R, addM, smul, idM, ww]
  rw [delta_comm j i, K_skew_all w0 w1 w2 i j]
  ring_int

/-- The nine-term bilinear expansion of a product of two three-term matrix
combinations. -/
theorem mul_addM_addM (x y z u v w : Int) (A B C D E F : Mat3) (i j : Nat) :
    mul (addM (addM (smul x A) (smul y B)) (smul z C))
        (addM (addM (smul u D) (smul v E)) (smul w F)) i j
      = x * (u * mul A D i j) + x * (v * mul A E i j) + x * (w * mul A F i j)
      + y * (u * mul B D i j) + y * (v * mul B E i j) + y * (w * mul B F i j)
      + z * (u * mul C D i j) + z * (v * mul C E i j) + z * (w * mul C F i j) := by
  simp only [mul_add_left, mul_add_right, mul_smul_smul]
  ring_int

/-- The scalar core of the expansion.  With `c = 1 - b θ²`, `t = θ²`,
`d = delta i j`, `w = (w wᵀ) i j` and `k = K i j`, the nine expanded terms
collapse to `(c² + a² t) d + (b c + b c - a² + b² t) w`. -/
theorem gram_scalar (a b c t d w k : Int) :
    c * (c * d) + c * (a * k) + c * (b * w)
      + (-a) * (c * k) + (-a) * (a * (w - t * d)) + (-a) * (b * 0)
      + b * (c * w) + b * (a * 0) + b * (b * (t * w))
    = (c * c + a * a * t) * d + (b * c + b * c - a * a + b * b * t) * w := by
  ring_int3

/-- **Conditional orthogonality.**  If the two scalar side conditions
`2b - a² - b²θ² = 0` and `(1-bθ²)² + a²θ² = 1` hold, then `Rᵀ R = I`
entrywise.  The conditions are exactly the trigonometric identity
`sin²θ + cos²θ = 1` at `a = sin θ / θ`, `b = (1 - cos θ) / θ²`; that
instantiation is not formalized here (see the module header). -/
theorem R_gram_conditional (a b w0 w1 w2 : Int)
    (h1 : 2 * b - a * a - b * b * theta2 w0 w1 w2 = 0)
    (h2 : (1 - b * theta2 w0 w1 w2) * (1 - b * theta2 w0 w1 w2)
        + a * a * theta2 w0 w1 w2 = 1)
    (i j : Nat) (hi : i < 3) (hj : j < 3) :
    mul (transpose (R a b w0 w1 w2)) (R a b w0 w1 w2) i j = delta i j := by
  have hW : b * (1 - b * theta2 w0 w1 w2) + b * (1 - b * theta2 w0 w1 w2)
      - a * a + b * b * theta2 w0 w1 w2 = 0 := by
    have hb : b * (1 - b * theta2 w0 w1 w2)
        = b - b * (b * theta2 w0 w1 w2) := by
      rw [Int.sub_eq_add_neg, Int.mul_add, Int.mul_one, Int.mul_neg]
      rfl
    rw [hb]
    rw [Int.mul_assoc b b (theta2 w0 w1 w2)]
    have h1' : 2 * b - a * a - b * (b * theta2 w0 w1 w2) = 0 := by
      rw [← Int.mul_assoc b b (theta2 w0 w1 w2)]
      exact h1
    omega
  rw [transpose_R_as_addM, R_as_addM, mul_addM_addM]
  rw [mul_idM_idM i j hi hj,
    mul_idM_left (K w0 w1 w2) i j hi,
    mul_idM_left (ww w0 w1 w2) i j hi,
    mul_idM_right (K w0 w1 w2) i j hj,
    K_sq w0 w1 w2 i j hi hj,
    K_ww_zero w0 w1 w2 i j hi hj,
    mul_idM_right (ww w0 w1 w2) i j hj,
    ww_K_zero w0 w1 w2 i j hi hj,
    ww_ww w0 w1 w2 i j hi hj]
  rw [gram_scalar a b (1 - b * theta2 w0 w1 w2) (theta2 w0 w1 w2)
    (delta i j) (ww w0 w1 w2 i j) (K w0 w1 w2 i j)]
  rw [h2, hW]
  ring_int

end G6ME.Burst3d
