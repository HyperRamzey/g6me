/-
  G6ME.Fdn — exact orthogonality of the 8×8 scattering feedback matrix.

  Models `crates/g6me-core/src/room.rs::ScatteringFdn::paraunitary_matrix`:

  * `h[i][j] = (-1)^popcount(i & j) / sqrt 8`  (normalised Walsh–Hadamard)
  * `v = v_raw / ||v_raw||` with
    `v_raw = [42, -31, 53, 19, -44, 27, -35, 16] / 100`
  * `r[i][j] = δ_ij - 2 v_i v_j`               (Householder reflection)
  * `u = r * h`

  The Rust code uses `f64`; the exact mathematical content is a rational
  identity: `uᵀ u = I` because `hᵀh = I` and `rᵀr = I`.  To keep the proof in
  exact integer arithmetic we clear the two irrational normalisations:

  * `10041 = v_raw · v_raw` (with `v_raw` scaled by 100), so
    `R' = 10041·I - 2·v_raw·v_rawᵀ = 10041 · r` is an integer matrix;
  * `H8` is the ±1 Hadamard matrix, `H8ᵀ H8 = 8 I`, so `h = H8 / sqrt 8`;
  * `U' = R' · H8`, and `U = U' / (10041 · sqrt 8)`.

  The theorems below prove, by kernel computation on concrete integer
  matrices, `H8ᵀ H8 = 8 I`, `R'ᵀ R' = 10041² I`, and `U'ᵀ U' = 8·10041² I`.
  Dividing the last identity by `(10041·sqrt 8)² = 8·10041²` gives exactly
  the Rust claim `uᵀ u = I`.  No floating-point semantics are assumed.
-/

import G6ME.Basic

namespace G6ME.Fdn

/-- An 8×8 integer matrix as a list of rows. -/
abbrev M8 := List (List Int)

/-- Number of set bits of `n`, for `n < 256` (the only inputs used). -/
def popc8 (n : Nat) : Nat :=
  (List.range 8).foldl (fun acc k => acc + (n / 2 ^ k) % 2) 0

/-- Matrix product. -/
def mmul (a b : M8) : M8 :=
  a.map fun row =>
    (List.range 8).map fun j =>
      (List.range 8).foldl
        (fun acc k => acc + row.getD k 0 * (b.getD k []).getD j 0) 0

/-- Matrix transpose. -/
def mtrans (a : M8) : M8 :=
  (List.range 8).map fun j => (List.range 8).map fun i => (a.getD i []).getD j 0

/-- The identity matrix scaled by `c`. -/
def scaledId (c : Int) : M8 :=
  (List.range 8).map fun i => (List.range 8).map fun j => if i = j then c else 0

/-- The ±1 Walsh–Hadamard matrix `H8` (unnormalised). -/
def H8 : M8 :=
  (List.range 8).map fun i =>
    (List.range 8).map fun j => if popc8 (i &&& j) % 2 = 0 then (1 : Int) else -1

/-- The code's `v_raw` vector, scaled by 100 (so it is integral). -/
def vRaw : List Int := [42, -31, 53, 19, -44, 27, -35, 16]

/-- The cleared-denominator Householder matrix `R' = 10041·I - 2·v_raw·v_rawᵀ`. -/
def Rprime : M8 :=
  (List.range 8).map fun i =>
    (List.range 8).map fun j =>
      (if i = j then (10041 : Int) else 0) - 2 * vRaw.getD i 0 * vRaw.getD j 0

/-- The cleared-denominator code matrix `U' = R' · H8`. -/
def Uprime : M8 := mmul Rprime H8

/-- `v_raw · v_raw = 10041`, the exact squared norm the Rust code computes
with `f64::sqrt`. -/
theorem vRaw_norm_sq :
    (List.range 8).foldl (fun acc i => acc + vRaw.getD i 0 * vRaw.getD i 0) 0
      = (10041 : Int) := by
  decide

/-- `H8ᵀ H8 = 8 I`.  This is the unnormalised Hadamard orthogonality; the
Rust `h = H8 / sqrt 8` therefore satisfies `hᵀ h = I`. -/
theorem hadamard_gram : mmul (mtrans H8) H8 = scaledId 8 := by
  decide

/-- `R'ᵀ R' = 10041² I`, i.e. the Householder reflection is orthogonal. -/
theorem householder_gram : mmul (mtrans Rprime) Rprime = scaledId (10041 * 10041) := by
  decide

/-- `U'ᵀ U' = 8 · 10041² I`.  Since `U = U'/(10041·sqrt 8)`, this is exactly
`Uᵀ U = I` for the matrix `ScatteringFdn::paraunitary_matrix` constructs. -/
theorem fdn_gram : mmul (mtrans Uprime) Uprime = scaledId (8 * 10041 * 10041) := by
  decide

/-- The transpose of `H8` is its own inverse up to the scalar 8, stated in the
form used by the composition argument. -/
theorem hadamard_transpose_mul : mmul H8 (mtrans H8) = scaledId 8 := by
  decide

/-- `R'` is symmetric, as a Householder reflection must be. -/
theorem householder_symmetric : mtrans Rprime = Rprime := by
  decide

end G6ME.Fdn
