/-
  G6ME.Limiter — integer/rational gain and clamp bounds.

  Models parts of `crates/g6me-core/src/limiter.rs` in exact integer
  arithmetic.  The Rust implementation is `f64` DSP; the theorems below are
  stated over `Int`/`Nat` and are explicitly an **abstraction** of that code:
  gains and membership values are scaled integers over a common positive
  denominator, and no theorem here claims anything about IEEE-754 rounding,
  denormals or NaN.  What is captured is the order structure that the audio
  guarantees rest on:

  * the true-peak ceiling is clamped to `[MIN_CEILING_DB, MAX_CEILING_DB]`;
  * the per-sample gain is a convex blend of the target and previous gains;
  * a gain capped at `ceiling / peak` cannot push the sample above the
    ceiling;
  * the residual ADAA2 clipper is exactly transparent inside its passband and
    its divisions are guarded by a nonzero denominator.
-/

import G6ME.Basic

namespace G6ME.Limiter

/-! ## Ceiling clamp

The Rust constants are `MIN_CEILING_DB = -24.0` and `MAX_CEILING_DB = 0.0`.
We model decibels as `Int` (whole dB), which is the right abstraction for the
interval proof; the fractional dB value itself is not used. -/

/-- `MIN_CEILING_DB` from `limiter.rs`. -/
def MIN_CEILING_DB : Int := -24

/-- `MAX_CEILING_DB` from `limiter.rs`. -/
def MAX_CEILING_DB : Int := 0

/-- The `[MIN_CEILING_DB, MAX_CEILING_DB]` clamp applied to the configured
ceiling.  (`Cfg::validate` rejects out-of-range values; this clamp is the
shaping that produces a usable in-range ceiling.) -/
def clampCeiling (x : Int) : Int :=
  if x < MIN_CEILING_DB then MIN_CEILING_DB
  else if MAX_CEILING_DB < x then MAX_CEILING_DB
  else x

/-- The clamped ceiling always lies in `[MIN_CEILING_DB, MAX_CEILING_DB]`. -/
theorem ceiling_in_range (x : Int) :
    MIN_CEILING_DB <= clampCeiling x ∧ clampCeiling x <= MAX_CEILING_DB := by
  unfold clampCeiling MIN_CEILING_DB MAX_CEILING_DB
  by_cases h1 : x < -24
  · rw [if_pos h1]
    constructor <;> omega
  · rw [if_neg h1]
    by_cases h2 : (0 : Int) < x
    · rw [if_pos h2]
      constructor <;> omega
    · rw [if_neg h2]
      constructor <;> omega

/-! ## The per-sample gain as a convex blend

`gainNum D w target prev = w * target + (D - w) * prev` is the numerator of
the convex combination

  `g = (w / D) * target + (1 - w / D) * prev`

over the common denominator `D`.  With all three scaled quantities in
`[0, D]`, dividing the numerator by `D` keeps the normalised gain in
`[0, 1]`; the theorem `gain_bounds` states the scaled form `0 <= g <= D`. -/

/-- Numerator of the convex blend of `target` and `prev` with scaled weight
`w`, over the common denominator `D`. -/
def gainNum (D w target prev : Nat) : Nat :=
  w * target + (D - w) * prev

/-- The applied per-sample gain, scaled by `D`. -/
def gain (D w target prev : Nat) : Nat :=
  gainNum D w target prev / D

/-- The blend numerator never exceeds `D * D`: this is the cross-multiplied
form of "the normalised blend is at most one". -/
theorem gain_num_le {D w target prev : Nat}
    (_hD : 0 < D) (hw : w <= D) (ht : target <= D) (hp : prev <= D) :
    gainNum D w target prev <= D * D := by
  unfold gainNum
  have h1 : w * target <= w * D := Nat.mul_le_mul_left w ht
  have h2 : (D - w) * prev <= (D - w) * D := Nat.mul_le_mul_left (D - w) hp
  have h3 : w + (D - w) = D := Nat.add_sub_of_le hw
  calc w * target + (D - w) * prev
      <= w * D + (D - w) * D := Nat.add_le_add h1 h2
    _ = (w + (D - w)) * D := (Nat.add_mul w (D - w) D).symm
    _ = D * D := by rw [h3]

/-- The scaled gain lies in `[0, D]`, i.e. the normalised gain lies in
`[0, 1]`. -/
theorem gain_bounds {D w target prev : Nat}
    (hD : 0 < D) (hw : w <= D) (ht : target <= D) (hp : prev <= D) :
    0 <= gain D w target prev ∧ gain D w target prev <= D := by
  constructor
  · exact Nat.zero_le _
  · unfold gain
    exact Nat.div_le_of_le_mul (gain_num_le hD hw ht hp)

/-- At full weight the blend is exactly the target. -/
theorem gain_full_weight {D target prev : Nat}
    (hD : 0 < D) (_ht : target <= D) :
    gain D D target prev = target := by
  unfold gain gainNum
  rw [Nat.sub_self, Nat.zero_mul, Nat.add_zero, Nat.mul_div_cancel_left target hD]

/-- At zero weight the blend is exactly the previous gain. -/
theorem gain_zero_weight {D target prev : Nat}
    (hD : 0 < D) (_hp : prev <= D) :
    gain D 0 target prev = prev := by
  unfold gain gainNum
  rw [Nat.zero_mul, Nat.zero_add, Nat.sub_zero, Nat.mul_div_cancel_left prev hD]

/-- If the applied gain is capped at `ceiling / peak`, the resulting output
magnitude `gain * peak` stays at or below the ceiling.  `peak` is the sample
magnitude whose gain demand is `ceiling / peak`; `peak > 0` is the Rust
assumption that the division is meaningful. -/
theorem limiter_output_bound {g peak ceiling : Nat} (_hpeak : 0 < peak) :
    min g (ceiling / peak) * peak <= ceiling := by
  calc min g (ceiling / peak) * peak
      <= (ceiling / peak) * peak :=
        Nat.mul_le_mul_right peak (Nat.min_le_right g (ceiling / peak))
    _ <= ceiling := Nat.div_mul_le_self ceiling peak

/-- The same statement in the cross-multiplied form used by the side chain:
if `g * peak <= ceiling` then the output magnitude is under the ceiling. -/
theorem limiter_output_cross {g peak ceiling : Nat}
    (h : g * peak <= ceiling) : g * peak <= ceiling := h

/-! ## ADAA2 residual clipper

`AdaaClipper::tick` has an exact passband fast path: when all three stored
samples lie in `[-t0, t0]` it returns the delayed sample `x1`, because the
residual `delta` (and its first two antiderivatives `d1`, `d2`) are exactly
zero there.  We model `delta` on the passband as `0`; its saturated branch
involves `sqrt` and `asinh` in `f64` and is *not* modelled.  The `tick` slow
branch is likewise represented by an arbitrary function `slow`.

The only real-valued fact used below is the **division guard**: the Rust code
divides by `x0 - x1` only when `|x0 - x1| > EPS`; with `EPS > 0` that
denominator is nonzero.  This is proved over `Int`. -/

/-- Integer absolute value, the model of `f64::abs` on the values that reach
the passband comparison. -/
def iabs (x : Int) : Int := if x < 0 then -x else x

theorem iabs_nonneg (x : Int) : 0 <= iabs x := by
  unfold iabs
  split <;> omega

theorem iabs_zero : iabs 0 = 0 := by simp [iabs]

/-- The residual `delta` restricted to the passband: `0` whenever
`|x| <= t0`.  The out-of-band branch is left abstract as `sat`. -/
def delta (sat : Int -> Int) (t0 x : Int) : Int :=
  if iabs x <= t0 then 0 else sat x

/-- On the passband, `delta` is exactly zero. -/
theorem adaa2_delta_passband {sat : Int -> Int} {t0 x : Int}
    (h : iabs x <= t0) : delta sat t0 x = 0 := by
  unfold delta
  rw [if_pos h]

/-- The `tick` fast path: if `|x0|`, `|x1|` and `|x2|` are all at most `t0`,
the returned sample is exactly `x1` (the one-sample-delayed input).  The slow
(ADAA2) branch is the arbitrary function `slow`. -/
def tick (slow : Int -> Int -> Int -> Int) (t0 x0 x1 x2 : Int) : Int :=
  if iabs x0 <= t0 ∧ iabs x1 <= t0 ∧ iabs x2 <= t0 then x1 else slow x0 x1 x2

/-- Inside the passband the ADAA2 fast path is the identity delayed by one
sample: `y[n] = x[n-1]`, with zero passband droop. -/
theorem adaa2_passband {slow : Int -> Int -> Int -> Int} {t0 x0 x1 x2 : Int}
    (h0 : iabs x0 <= t0) (h1 : iabs x1 <= t0) (h2 : iabs x2 <= t0) :
    tick slow t0 x0 x1 x2 = x1 := by
  unfold tick
  rw [if_pos ⟨h0, h1, h2⟩]

/-- The residual is zero at each of the three samples when the fast path
triggers. -/
theorem adaa2_passband_delta_zero {sat : Int -> Int} {t0 x0 x1 x2 : Int}
    (h0 : iabs x0 <= t0) (h1 : iabs x1 <= t0) (h2 : iabs x2 <= t0) :
    delta sat t0 x0 = 0 ∧ delta sat t0 x1 = 0 ∧ delta sat t0 x2 = 0 :=
  ⟨adaa2_delta_passband h0, adaa2_delta_passband h1, adaa2_delta_passband h2⟩

/-- The division guard behind the ADAA2 slopes: if `|a - b| > EPS > 0` then
`a - b ≠ 0`, so the Rust divisions by `x0 - x1`, `x1 - x2` and `x0 - x2`
cannot divide by zero. -/
theorem adaa2_div_guard {a b EPS : Int}
    (hEPS : 0 < EPS) (h : EPS < iabs (a - b)) : a - b ≠ 0 := by
  intro hz
  rw [hz, iabs_zero] at h
  omega

/-- Division-guard instance for the current pair `(x0, x1)`. -/
theorem adaa2_div_guard_cur {x0 x1 EPS : Int} (hEPS : 0 < EPS)
    (h : EPS < iabs (x0 - x1)) : x0 - x1 ≠ 0 :=
  adaa2_div_guard hEPS h

/-- Division-guard instance for the previous pair `(x1, x2)`. -/
theorem adaa2_div_guard_prev {x1 x2 EPS : Int} (hEPS : 0 < EPS)
    (h : EPS < iabs (x1 - x2)) : x1 - x2 ≠ 0 :=
  adaa2_div_guard hEPS h

end G6ME.Limiter
