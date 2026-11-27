/-
  G6ME.Stn — unity-summing sinusoidal-transient-noise (STN) masks.

  Models `crates/g6me-core/src/stn.rs` (`fuzzy_sin2` and `fuzzy_masks`) in an
  explicitly integer/rational abstraction with a common denominator `D > 0`.

  The Rust code operates on `f64`.  Nothing here pretends Lean's `Float` is
  IEEE-754.  Instead every mask value is a scaled natural number `x` standing
  for the rational `x / D`:

  * the anisotropy ratio is `r_s = a / D` with `0 <= a <= D`, and the
    complementary ratio is `r_t = 1 - r_s = (D - a) / D`;
  * the lower fuzzy threshold is `beta_l = c / D` with `2 * c >= D`, which is
    exactly the Rust clamp `beta_l >= 0.5`;
  * `fs` and `ft` are the scaled memberships `D * S_s` and `D * S_t`.  The
    hypotheses `a <= c -> fs = 0` and `D - a <= c -> ft = 0` abstract the fact
    that `fuzzy_sin2` returns exactly `0.0` at or below `beta_l`.

  The `sin^2` transition itself needs real trigonometry and is **not**
  modelled: nothing below constrains the mask between the thresholds.  What is
  modelled and proved is the support-disjointness argument from the Rust
  doc comment (which is what guarantees `S_n >= 0`), the resulting unity sum,
  and the `denom <= 1e-30` fast path that returns the pure-noise triple.

  The Rust fast path is `return (0.0, 0.0, 1.0)`; scaled by `D` that is the
  triple `(0, 0, D)` (`noiseTriple` below), whose components sum to `D`.
-/

import G6ME.Basic

namespace G6ME.Stn

/-! ## Scaled mask triples -/

/-- A triple of mask weights in the common-denominator representation. -/
structure MaskTriple where
  /-- Scaled sine mask `D * S_s`. -/
  signal : Nat
  /-- Scaled transient mask `D * S_t`. -/
  transient : Nat
  /-- Scaled noise mask `D * S_n`. -/
  noise : Nat

/-- Sum of the three scaled mask weights. -/
def MaskTriple.sum (m : MaskTriple) : Nat :=
  m.signal + m.transient + m.noise

/-! ## Support disjointness and the unity sum

`fuzzy_masks` computes `r_s = m_h / (m_h + m_v)` and `r_t = 1 - r_s`, so the
two ratios add to one.  With `beta_l >= 0.5` the two memberships cannot both
be positive: two positive values would force `r_s > beta_l` and
`1 - r_s > beta_l`, hence `1 = r_s + (1 - r_s) > 2 * beta_l >= 1`, a
contradiction.  That is the proof of `stn_masks_nonneg_residual`, written out
in the scaled integer model. -/

/-- The signal and transient supports are disjoint, so their scaled
memberships leave at most the whole denominator for themselves:
`fs + ft <= D`.  Equivalently the scaled residual `D - fs - ft` is
non-negative, which is what makes `S_n = max(0, 1 - S_s - S_t)` exactly
`1 - S_s - S_t` (the `max` never bites). -/
theorem stn_masks_nonneg_residual
    {D a c fs ft : Nat}
    (_hD : 0 < D)
    (haD : a <= D)
    (hbeta : D <= 2 * c)
    (hfs_le : fs <= D)
    (hft_le : ft <= D)
    (hfs_zero : a <= c -> fs = 0)
    (hft_zero : D - a <= c -> ft = 0) :
    fs + ft <= D := by
  suffices h : ¬ D < fs + ft by omega
  intro hlt
  -- A positive `fs` forces `a > c`, and a positive `ft` forces `D - a > c`.
  have hfs_ne : fs ≠ 0 := by
    intro h0
    rw [h0, Nat.zero_add] at hlt
    omega
  have ha_gt : c < a := by
    suffices hle : ¬ a <= c by omega
    intro hac
    exact hfs_ne (hfs_zero hac)
  have hft_ne : ft ≠ 0 := by
    intro h0
    rw [h0, Nat.add_zero] at hlt
    omega
  have hb_gt : c < D - a := by
    suffices hle : ¬ D - a <= c by omega
    intro hbc
    exact hft_ne (hft_zero hbc)
  -- Then `D = a + (D - a) > 2 * c >= D`, a contradiction.
  have hsum : a + (D - a) = D := Nat.add_sub_of_le haD
  omega

/-- The unity sum: the scaled noise residual is exactly `D - fs - ft`, and
the three scaled weights sum to the common denominator `D`. -/
theorem stn_masks_unity {D fs ft : Nat} (h : fs + ft <= D) :
    fs + ft + (D - fs - ft) = D := by
  rw [Nat.sub_sub]
  exact Nat.add_sub_of_le h

/-- The same unity sum under the full STN hypotheses (the non-negativity of
the residual is discharged by `stn_masks_nonneg_residual`). -/
theorem stn_masks_unity_full
    {D a c fs ft : Nat}
    (hD : 0 < D)
    (haD : a <= D)
    (hbeta : D <= 2 * c)
    (hfs_le : fs <= D)
    (hft_le : ft <= D)
    (hfs_zero : a <= c -> fs = 0)
    (hft_zero : D - a <= c -> ft = 0) :
    fs + ft + (D - fs - ft) = D :=
  stn_masks_unity
    (stn_masks_nonneg_residual hD haD hbeta hfs_le hft_le hfs_zero hft_zero)

/-! ## The triple form -/

/-- The `fuzzy_masks` triple `(S_s, S_t, 1 - S_s - S_t)`, scaled by `D`. -/
def maskTriple (D fs ft : Nat) : MaskTriple :=
  ⟨fs, ft, D - fs - ft⟩

/-- The scaled triple sums to `D` whenever the residual is non-negative. -/
theorem maskTriple_sum {D fs ft : Nat} (h : fs + ft <= D) :
    (maskTriple D fs ft).sum = D :=
  stn_masks_unity h

/-- The pure-noise triple returned by the `denom <= 1e-30` fast path,
scaled by `D`: the image of the Rust `(0.0, 0.0, 1.0)`. -/
def noiseTriple (D : Nat) : MaskTriple :=
  ⟨0, 0, D⟩

/-- The fast-path triple still sums to the common denominator `D`. -/
theorem noiseTriple_sum (D : Nat) : (noiseTriple D).sum = D := by
  simp [noiseTriple, MaskTriple.sum]

/-- The branch structure of `fuzzy_masks`: `degenerate` represents the
`denom <= 1e-30` guard (a `Nat` denominator of zero in this abstraction).
On the degenerate branch the function returns `(0, 0, 1)`; otherwise it
returns `(S_s, S_t, 1 - S_s - S_t)`. -/
def fuzzyMasksModel (D : Nat) (degenerate : Bool) (fs ft : Nat) : MaskTriple :=
  if degenerate then noiseTriple D else maskTriple D fs ft

/-- Both branches of `fuzzy_masks` produce a triple that sums to `D`: the
denominator guard returns the pure-noise triple and the normal branch sums by
the disjoint-support argument. -/
theorem fuzzy_masks_unity {D fs ft : Nat}
    (_hD : 0 < D) (h : fs + ft <= D) (degenerate : Bool) :
    (fuzzyMasksModel D degenerate fs ft).sum = D := by
  cases degenerate
  · exact maskTriple_sum h
  · exact noiseTriple_sum D

/-- Specialisation of the guard: when the denominator is degenerate the
returned pure-noise triple sums to `D`. -/
theorem fuzzy_masks_zero_denom (D : Nat) :
    (fuzzyMasksModel D true 0 0).sum = D :=
  noiseTriple_sum D

end G6ME.Stn
