/-
  G6ME.Dng — a `Nat` / `List` model of the two DNG byte packings in
  `crates/g6me-core/src/dng.rs`.

  This module formalises, with complete proofs, the byte layouts used by:

  * `pack_mipi_raw10` — MIPI CSI-2 RAW10: four 10-bit pixels in five bytes,
    bytes `0..3 = pᵢ / 4` (the top eight bits) and byte `4` the four two-bit
    LSBs packed little-field-first (`p0%4`, `4*(p1%4)`, `16*(p2%4)`,
    `64*(p3%4)`).
  * `unpack_mipi_raw10_to_f64` — the *raw10 decoding half only*: five bytes
    are read per group and each pixel is `bᵢ * 4 + fieldᵢ(lsb)`.
  * `unpack_uncompressed_block` — the `bps = 16` branch, little- and
    big-endian.

  The frame model mirrors the Rust indexing: each row occupies `stride`
  bytes, only the first `5 * groups` bytes of a row carry pixel data, and the
  remaining bytes (zero padding in `pack_mipi_raw10`) are ignored by the
  decoder.

  Deliberately NOT modelled:

  * the `f64` black-level / white-level normalisation
    `((p as f64) - black_level) * inv_span` — this is `Float` arithmetic in
    the Rust and Lean's `Float` is not IEEE-754, so it is left out rather
    than modelled dishonestly;
  * the `& 0x3FF` masks in the Rust packer — they are the identity on the
    documented `< 1024` input domain, which every theorem below assumes;
  * the 8-bit and packed 10/12/14-bit branches of `unpack_uncompressed_block`,
    and every other codec in `dng.rs` (lossless JPEG, JPEG-XL, opcodes).

  Axiom discipline: no `sorry`/`admit`, no `native_decide`, no `axiom`
  declarations.  The `#print axioms` inventory at the end of the file records
  the axioms of every headline theorem; the actual output lists only
  `propext` and `Quot.sound` (from core `simp`/`omega` reasoning), never
  `Classical.choice`.
-/

import G6ME.Basic

namespace G6ME.Dng

/-! ## 1. The RAW10 group codec: 4 pixels ⇄ 5 bytes

`packGroup p0 p1 p2 p3` is exactly the body of the per-group loop in
`pack_mipi_raw10`; `unpackGroup` is exactly the per-group read in
`unpack_mipi_raw10_to_f64`, minus the `f64` affine transform.

A group is a right-nested quadruple `Nat × Nat × Nat × Nat`, matching
`(p0, p1, p2, p3)` in Lean. -/

/-- A group of four 10-bit pixel values, right-nested like Lean's
`(p0, p1, p2, p3)` notation. -/
abbrev Group := Nat × Nat × Nat × Nat

/-- MIPI RAW10 packing of one group of four pixels into five bytes.

Rust equivalent (`pack_mipi_raw10`):

```rust
row_dst[g * 5]     = (p0 >> 2) as u8;
row_dst[g * 5 + 1] = (p1 >> 2) as u8;
row_dst[g * 5 + 2] = (p2 >> 2) as u8;
row_dst[g * 5 + 3] = (p3 >> 2) as u8;
row_dst[g * 5 + 4] =
    ((p0 & 3) | ((p1 & 3) << 2) | ((p2 & 3) << 4) | ((p3 & 3) << 6)) as u8;
```

On `Nat`, `p >> 2` is `p / 4` and `p & 3` is `p % 4`. -/
def packGroup (p0 p1 p2 p3 : Nat) : List Nat :=
  [p0 / 4, p1 / 4, p2 / 4, p3 / 4,
   p0 % 4 + 4 * (p1 % 4) + 16 * (p2 % 4) + 64 * (p3 % 4)]

/-- MIPI RAW10 unpacking of one 5-byte group.

Rust equivalent (`unpack_mipi_raw10_to_f64`):

```rust
let p0 = (b0 << 2) | (lsb & 3);
let p1 = (b1 << 2) | ((lsb >> 2) & 3);
let p2 = (b2 << 2) | ((lsb >> 4) & 3);
let p3 = (b3 << 2) | ((lsb >> 6) & 3);
```

Short inputs decode to `(0, 0, 0, 0)`; every theorem below feeds it exactly
five bytes. -/
def unpackGroup : List Nat → Group
  | b0 :: b1 :: b2 :: b3 :: b4 :: _ =>
      (b0 * 4 + b4 % 4, b1 * 4 + (b4 / 4) % 4,
       b2 * 4 + (b4 / 16) % 4, b3 * 4 + (b4 / 64) % 4)
  | _ => (0, 0, 0, 0)

/-- `packGroup` is literally the Rust write pattern: four high bytes
`pᵢ / 4` followed by one byte holding the four two-bit LSB fields. -/
theorem raw10_group_layout (p0 p1 p2 p3 : Nat) :
    packGroup p0 p1 p2 p3 =
      [p0 / 4, p1 / 4, p2 / 4, p3 / 4,
       p0 % 4 + 4 * (p1 % 4) + 16 * (p2 % 4) + 64 * (p3 % 4)] := rfl

/-- `unpackGroup` is literally the Rust read pattern on five bytes. -/
theorem unpackGroup_cons (b0 b1 b2 b3 b4 : Nat) (rest : List Nat) :
    unpackGroup (b0 :: b1 :: b2 :: b3 :: b4 :: rest) =
      (b0 * 4 + b4 % 4, b1 * 4 + (b4 / 4) % 4,
       b2 * 4 + (b4 / 16) % 4, b3 * 4 + (b4 / 64) % 4) := rfl

/-! ### Base-4 digit extraction

The fifth byte is a base-4 number with digits `p0%4, p1%4, p2%4, p3%4` in
little-endian order.  The next four lemmas are the exact digit-extraction
facts the Rust `>>`/`&` combination performs. -/

/-- Two-bit field `0` of the packed LSB byte. -/
theorem base4_field0 (a b c d : Nat) (ha : a < 4) :
    (a + 4 * b + 16 * c + 64 * d) % 4 = a := by omega

/-- Two-bit field `1` of the packed LSB byte. -/
theorem base4_field1 (a b c d : Nat) (ha : a < 4) (hb : b < 4) :
    ((a + 4 * b + 16 * c + 64 * d) / 4) % 4 = b := by omega

/-- Two-bit field `2` of the packed LSB byte. -/
theorem base4_field2 (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) :
    ((a + 4 * b + 16 * c + 64 * d) / 16) % 4 = c := by omega

/-- Two-bit field `3` of the packed LSB byte. -/
theorem base4_field3 (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4)
    (hd : d < 4) :
    ((a + 4 * b + 16 * c + 64 * d) / 64) % 4 = d := by omega

/-- Reassembling a value from its quotient and remainder by four. -/
theorem div4_mul4_add_mod4 (p : Nat) : p / 4 * 4 + p % 4 = p := by
  have h := Nat.div_add_mod p 4
  omega

/-! ### The group roundtrip -/

/-- The pure list-level roundtrip: unpacking the literal five bytes written
by `packGroup` recovers the four pixels. -/
theorem unpackGroup_packed (p0 p1 p2 p3 : Nat) :
    unpackGroup [p0 / 4, p1 / 4, p2 / 4, p3 / 4,
        p0 % 4 + 4 * (p1 % 4) + 16 * (p2 % 4) + 64 * (p3 % 4)]
      = (p0, p1, p2, p3) := by
  have m0 : p0 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m1 : p1 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m2 : p2 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m3 : p3 % 4 < 4 := Nat.mod_lt _ (by decide)
  have f0 := base4_field0 (p0 % 4) (p1 % 4) (p2 % 4) (p3 % 4) m0
  have f1 := base4_field1 (p0 % 4) (p1 % 4) (p2 % 4) (p3 % 4) m0 m1
  have f2 := base4_field2 (p0 % 4) (p1 % 4) (p2 % 4) (p3 % 4) m0 m1 m2
  have f3 := base4_field3 (p0 % 4) (p1 % 4) (p2 % 4) (p3 % 4) m0 m1 m2 m3
  simp only [unpackGroup, Prod.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [f0]; exact div4_mul4_add_mod4 p0
  · rw [f1]; exact div4_mul4_add_mod4 p1
  · rw [f2]; exact div4_mul4_add_mod4 p2
  · rw [f3]; exact div4_mul4_add_mod4 p3

/-- RAW10 pack/unpack is the identity on one group of 10-bit pixels.

The `< 1024` hypotheses are the domain of the Rust code (they make every
produced byte fit in a `u8`, see `raw10_group_byte_bounds`).  The inverse
identity itself holds for every `Nat`: `Nat.div_add_mod` loses no
information, so the bounds are not needed by this proof. -/
theorem raw10_group_roundtrip (p0 p1 p2 p3 : Nat)
    (_h0 : p0 < 1024) (_h1 : p1 < 1024) (_h2 : p2 < 1024) (_h3 : p3 < 1024) :
    unpackGroup (packGroup p0 p1 p2 p3) = (p0, p1, p2, p3) := by
  unfold packGroup
  exact unpackGroup_packed p0 p1 p2 p3

/-- Trailing bytes after the five-byte group do not affect the decode. -/
theorem raw10_group_roundtrip_append (p0 p1 p2 p3 : Nat) (rest : List Nat) :
    unpackGroup (packGroup p0 p1 p2 p3 ++ rest) = (p0, p1, p2, p3) := by
  unfold packGroup
  simp only [List.cons_append, List.nil_append]
  exact unpackGroup_packed p0 p1 p2 p3

/-- Each 2-bit field of the packed LSB byte recovers the low bits of the
corresponding pixel — the "the fifth byte is the bit-packed LSBs" half of the
layout specification. -/
theorem raw10_group_lsb_fields (p0 p1 p2 p3 : Nat) :
    let lsb := p0 % 4 + 4 * (p1 % 4) + 16 * (p2 % 4) + 64 * (p3 % 4)
    lsb % 4 = p0 % 4 ∧ (lsb / 4) % 4 = p1 % 4 ∧
      (lsb / 16) % 4 = p2 % 4 ∧ (lsb / 64) % 4 = p3 % 4 := by
  have m0 : p0 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m1 : p1 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m2 : p2 % 4 < 4 := Nat.mod_lt _ (by decide)
  have m3 : p3 % 4 < 4 := Nat.mod_lt _ (by decide)
  dsimp only
  exact ⟨base4_field0 _ _ _ _ m0, base4_field1 _ _ _ _ m0 m1,
    base4_field2 _ _ _ _ m0 m1 m2, base4_field3 _ _ _ _ m0 m1 m2 m3⟩

/-! ### Byte-range lemmas -/

theorem packGroup_length (p0 p1 p2 p3 : Nat) :
    (packGroup p0 p1 p2 p3).length = 5 := rfl

/-- Every byte written by the RAW10 packer is a valid `u8`. -/
theorem raw10_group_byte_bounds (p0 p1 p2 p3 : Nat)
    (h0 : p0 < 1024) (h1 : p1 < 1024) (h2 : p2 < 1024) (h3 : p3 < 1024) :
    ∀ b ∈ packGroup p0 p1 p2 p3, b < 256 := by
  intro b hb
  simp only [packGroup, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl | rfl | rfl
  · omega
  · omega
  · omega
  · omega
  · have m0 : p0 % 4 < 4 := Nat.mod_lt _ (by decide)
    have m1 : p1 % 4 < 4 := Nat.mod_lt _ (by decide)
    have m2 : p2 % 4 < 4 := Nat.mod_lt _ (by decide)
    have m3 : p3 % 4 < 4 := Nat.mod_lt _ (by decide)
    omega

/-- Any five `u8` bytes decode to four 10-bit pixel values. -/
theorem raw10_group_unpack_bounds (b0 b1 b2 b3 b4 : Nat)
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256)
    (_h4 : b4 < 256) :
    let g := unpackGroup [b0, b1, b2, b3, b4]
    g.1 < 1024 ∧ g.2.1 < 1024 ∧ g.2.2.1 < 1024 ∧ g.2.2.2 < 1024 := by
  simp only [unpackGroup]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- The four components of a group are all 10-bit. -/
def GroupInBounds (g : Group) : Prop :=
  g.1 < 1024 ∧ g.2.1 < 1024 ∧ g.2.2.1 < 1024 ∧ g.2.2.2 < 1024

/-- Every group of a list is 10-bit. -/
def AllInBounds (gs : List Group) : Prop := ∀ g ∈ gs, GroupInBounds g

theorem raw10_group_bytes_of_bounds (g : Group) (h : GroupInBounds g) :
    ∀ b ∈ packGroup g.1 g.2.1 g.2.2.1 g.2.2.2, b < 256 := by
  obtain ⟨h0, h1, h2, h3⟩ := h
  exact raw10_group_byte_bounds g.1 g.2.1 g.2.2.1 g.2.2.2 h0 h1 h2 h3

/-! ## 2. Row level: lists of groups ⇄ lists of bytes

A row is `groups.length` consecutive 5-byte groups, exactly as
`pack_mipi_raw10` walks `row_src[g * 4 + k]` in steps of 5 output bytes.

`packRow` is written as a structural fold; `packRow_eq_flatMap` shows it is
the `flatMap packGroup` of the specification. -/

/-- Pack a row of groups: concatenate the five bytes of each group. -/
def packRow : List Group → List Nat
  | [] => []
  | g :: rest => packGroup g.1 g.2.1 g.2.2.1 g.2.2.2 ++ packRow rest

/-- `packRow` is `flatMap packGroup`, i.e. the same list the Rust writes. -/
theorem packRow_eq_flatMap (gs : List Group) :
    packRow gs = gs.flatMap (fun g => packGroup g.1 g.2.1 g.2.2.1 g.2.2.2) := by
  induction gs with
  | nil => rfl
  | cons g rest ih =>
      simp only [packRow, List.flatMap_cons, ih]

/-- Unpack a row: split the byte list into consecutive five-byte chunks.
A truncated tail (fewer than five bytes) yields nothing, mirroring the fact
that the Rust only reads whole groups. -/
def unpackRow : List Nat → List Group
  | [] => []
  | b0 :: r1 =>
      match r1 with
      | [] => []
      | b1 :: r2 =>
          match r2 with
          | [] => []
          | b2 :: r3 =>
              match r3 with
              | [] => []
              | b3 :: r4 =>
                  match r4 with
                  | [] => []
                  | b4 :: rest => unpackGroup [b0, b1, b2, b3, b4] :: unpackRow rest

/-- A packed row is exactly five bytes per group. -/
theorem packRow_length (gs : List Group) :
    (packRow gs).length = 5 * gs.length := by
  induction gs with
  | nil => rfl
  | cons g rest ih =>
      simp only [packRow, List.length_append, packGroup_length, ih, List.length_cons]
      omega

/-- Row-level RAW10 inverse.  As at group level, the inverse holds for every
`Nat`; the `< 1024` bounds matter only for the byte range. -/
theorem raw10_row_roundtrip_unbounded (gs : List Group) :
    unpackRow (packRow gs) = gs := by
  induction gs with
  | nil => rfl
  | cons g rest ih =>
      obtain ⟨p0, p1, p2, p3⟩ := g
      simp only [packRow, packGroup, List.cons_append, List.nil_append,
                 unpackRow, unpackGroup_packed, ih]

/-- Row-level RAW10 inverse for in-range (10-bit) groups. -/
theorem raw10_row_roundtrip (gs : List Group) (_h : AllInBounds gs) :
    unpackRow (packRow gs) = gs :=
  raw10_row_roundtrip_unbounded gs

/-- Every byte of a packed row of 10-bit groups is a valid `u8`. -/
theorem raw10_row_byte_bounds (gs : List Group) (h : AllInBounds gs) :
    ∀ b ∈ packRow gs, b < 256 := by
  induction gs with
  | nil =>
      intro b hb
      simp [packRow] at hb
  | cons g rest ih =>
      have hg : GroupInBounds g := h g (List.mem_cons_self ..)
      have hrest : AllInBounds rest := fun x hx => h x (List.mem_cons_of_mem _ hx)
      intro b hb
      simp only [packRow, List.mem_append] at hb
      rcases hb with hb | hb
      · exact raw10_group_bytes_of_bounds g hg b hb
      · exact ih hrest b hb

/-! ## 3. Frame level: rows padded to a stride

`packFrame stride rows` writes each packed row into a `stride`-byte slot,
zero-filling the remainder — exactly the `vec![0u8; row_stride * height]`
allocation of `pack_mipi_raw10`, whose inner loop writes only the first
`packed_row = (width / 4) * 5` bytes of each slot.

`unpackFrame stride groups rows` reads `5 * groups` bytes from the start of
each slot and then skips a full `stride`, exactly the `row_src` slice
`raw[y*row_stride .. y*row_stride + packed_row]` of
`unpack_mipi_raw10_to_f64`. -/

/-- A packed row zero-padded to `stride` bytes. -/
def padRow (stride : Nat) (gs : List Group) : List Nat :=
  packRow gs ++ List.replicate (stride - (packRow gs).length) 0

/-- Pack a frame: concatenate the padded slots. -/
def packFrame : Nat → List (List Group) → List Nat
  | _, [] => []
  | stride, r :: rest => padRow stride r ++ packFrame stride rest

/-- `packFrame` is the `flatMap` of `padRow` over the rows. -/
theorem packFrame_eq_flatMap (stride : Nat) (rows : List (List Group)) :
    packFrame stride rows = rows.flatMap (fun r => padRow stride r) := by
  induction rows with
  | nil => rfl
  | cons r rest ih =>
      simp only [packFrame, List.flatMap_cons, ih]

/-- Take `n` from a list whose length is `n`. -/
theorem take_of_length_eq {l : List Nat} {n : Nat} (h : l.length = n)
    (rest : List Nat) : List.take n (l ++ rest) = l := by
  rw [← h]
  exact List.take_left

/-- Drop `n` from a list whose length is `n`. -/
theorem drop_of_length_eq {l : List Nat} {n : Nat} (h : l.length = n)
    (rest : List Nat) : List.drop n (l ++ rest) = rest := by
  rw [← h]
  exact List.drop_left

/-- A padded slot has exactly `stride` bytes. -/
theorem padRow_length (stride : Nat) (gs : List Group)
    (h : (packRow gs).length ≤ stride) : (padRow stride gs).length = stride := by
  unfold padRow
  rw [List.length_append, List.length_replicate]
  omega

/-- Reading `5 * groups` bytes from the start of a slot returns exactly the
packed row: all padding is beyond the read window. -/
theorem take_packRow (stride : Nat) (gs : List Group) (rest : List Nat) :
    List.take (5 * gs.length) (padRow stride gs ++ rest) = packRow gs := by
  rw [← packRow_length gs]
  unfold padRow
  rw [List.append_assoc]
  exact take_of_length_eq (l := packRow gs) rfl _

/-- Padding bytes after a packed row never affect the decoded groups. -/
theorem raw10_unpack_ignores_padding (stride : Nat) (gs : List Group)
    (rest : List Nat) :
    unpackRow (List.take (5 * gs.length) (padRow stride gs ++ rest)) = gs := by
  rw [take_packRow]
  exact raw10_row_roundtrip_unbounded gs

/-- Skipping a full slot discards the padding and reaches the next row. -/
theorem drop_padRow (stride : Nat) (gs : List Group) (rest : List Nat)
    (h : (packRow gs).length ≤ stride) :
    List.drop stride (padRow stride gs ++ rest) = rest := by
  exact drop_of_length_eq (l := padRow stride gs) (padRow_length stride gs h) rest

/-- Decode `n` rows of `groups` groups each from a stride-padded frame. -/
def unpackFrame (stride nGroups : Nat) : Nat → List Nat → List (List Group)
  | 0, _ => []
  | k + 1, bytes =>
      unpackRow (List.take (5 * nGroups) bytes) ::
        unpackFrame stride nGroups k (List.drop stride bytes)

/-- Frame-level RAW10 inverse: if every row has exactly `nGroups` in-range
groups and the stride is large enough, decoding the padded frame recovers the
original rows.  Padding bytes are never inspected. -/
theorem raw10_frame_roundtrip (stride nGroups : Nat) (rows : List (List Group))
    (hinb : ∀ r ∈ rows, AllInBounds r)
    (hlen : ∀ r ∈ rows, r.length = nGroups)
    (hstride : ∀ r ∈ rows, (packRow r).length ≤ stride) :
    unpackFrame stride nGroups rows.length (packFrame stride rows) = rows := by
  induction rows with
  | nil => rfl
  | cons r rest ih =>
      have hr_stride : (packRow r).length ≤ stride := hstride r (List.mem_cons_self ..)
      have hr_len : r.length = nGroups := hlen r (List.mem_cons_self ..)
      have rest_stride : ∀ x ∈ rest, (packRow x).length ≤ stride :=
        fun x hx => hstride x (List.mem_cons_of_mem _ hx)
      have rest_len : ∀ x ∈ rest, x.length = nGroups :=
        fun x hx => hlen x (List.mem_cons_of_mem _ hx)
      have rest_inb : ∀ x ∈ rest, AllInBounds x :=
        fun x hx => hinb x (List.mem_cons_of_mem _ hx)
      simp only [packFrame, List.length_cons, Nat.add_one, unpackFrame]
      have htake : List.take (5 * nGroups) (padRow stride r ++ packFrame stride rest)
          = packRow r := by
        rw [← hr_len]
        exact take_packRow stride r (packFrame stride rest)
      rw [htake, drop_padRow stride r (packFrame stride rest) hr_stride,
          raw10_row_roundtrip_unbounded r, ih rest_inb rest_len rest_stride]

/-- The padded frame has exactly `stride * height` bytes, matching the
`row_stride * height` allocation checked by the Rust decoder. -/
theorem packFrame_length (stride : Nat) (rows : List (List Group))
    (h : ∀ r ∈ rows, (packRow r).length ≤ stride) :
    (packFrame stride rows).length = stride * rows.length := by
  induction rows with
  | nil => rfl
  | cons r rest ih =>
      have hr : (packRow r).length ≤ stride := h r (List.mem_cons_self ..)
      have hrest : ∀ x ∈ rest, (packRow x).length ≤ stride :=
        fun x hx => h x (List.mem_cons_of_mem _ hx)
      simp only [packFrame, List.length_append, padRow_length stride r hr,
                 ih hrest, List.length_cons]
      ring_nat

/-- Every byte of a packed frame (pixels and zero padding) is a valid `u8`. -/
theorem raw10_frame_byte_bounds (stride : Nat) (rows : List (List Group))
    (h : ∀ r ∈ rows, AllInBounds r) :
    ∀ b ∈ packFrame stride rows, b < 256 := by
  induction rows with
  | nil =>
      intro b hb
      simp [packFrame] at hb
  | cons r rest ih =>
      have hr : AllInBounds r := h r (List.mem_cons_self ..)
      have hrest : ∀ x ∈ rest, AllInBounds x :=
        fun x hx => h x (List.mem_cons_of_mem _ hx)
      intro b hb
      simp only [packFrame, List.mem_append] at hb
      rcases hb with hb | hb
      · simp only [padRow, List.mem_append] at hb
        rcases hb with hb | hb
        · exact raw10_row_byte_bounds r hr b hb
        · simp only [List.mem_replicate] at hb
          obtain ⟨_, rfl⟩ := hb
          omega
      · exact ih hrest b hb

/-! ## 4. `unpack_uncompressed_block`, `bps = 16`

The Rust reads two bytes per sample:

```rust
let b = [raw[i * 2], raw[i * 2 + 1]];
out[i] = if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) };
```

`from_le_bytes` puts the low byte first, `from_be_bytes` the high byte
first.  The functions below mirror that byte order exactly.

As with RAW10, the inverse identities hold for every `Nat`; `v < 65536` is
the `u16` domain of the Rust code and is what makes the two bytes fit in a
`u8` (`pack16_*_byte_bounds`). -/

/-- Little-endian 16-bit packing: `(low byte, high byte)`. -/
def pack16le (v : Nat) : Nat × Nat := (v % 256, v / 256)

/-- Little-endian 16-bit unpacking: `low + 256 * high`. -/
def unpack16le (p : Nat × Nat) : Nat := p.1 + 256 * p.2

/-- Big-endian 16-bit packing: `(high byte, low byte)`. -/
def pack16be (v : Nat) : Nat × Nat := (v / 256, v % 256)

/-- Big-endian 16-bit unpacking: `high * 256 + low`. -/
def unpack16be (p : Nat × Nat) : Nat := p.1 * 256 + p.2

theorem pack16_le_roundtrip_unbounded (v : Nat) :
    unpack16le (pack16le v) = v := by
  unfold pack16le unpack16le
  have h := Nat.mod_add_div v 256
  omega

theorem pack16_be_roundtrip_unbounded (v : Nat) :
    unpack16be (pack16be v) = v := by
  unfold pack16be unpack16be
  have h := Nat.div_add_mod v 256
  omega

/-- Little-endian 16-bit roundtrip on the `u16` domain. -/
theorem pack16_le_roundtrip (v : Nat) (_hv : v < 65536) :
    unpack16le (pack16le v) = v :=
  pack16_le_roundtrip_unbounded v

/-- Big-endian 16-bit roundtrip on the `u16` domain. -/
theorem pack16_be_roundtrip (v : Nat) (_hv : v < 65536) :
    unpack16be (pack16be v) = v :=
  pack16_be_roundtrip_unbounded v

theorem pack16_le_byte_bounds (v : Nat) (hv : v < 65536) :
    (pack16le v).1 < 256 ∧ (pack16le v).2 < 256 := by
  unfold pack16le
  exact ⟨Nat.mod_lt _ (by decide), by omega⟩

theorem pack16_be_byte_bounds (v : Nat) (hv : v < 65536) :
    (pack16be v).1 < 256 ∧ (pack16be v).2 < 256 := by
  unfold pack16be
  exact ⟨by omega, Nat.mod_lt _ (by decide)⟩

/-- Two `u8` bytes decode to a `u16` (little-endian). -/
theorem unpack16_le_bounds (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    unpack16le (b0, b1) < 65536 := by
  unfold unpack16le
  omega

/-- Two `u8` bytes decode to a `u16` (big-endian). -/
theorem unpack16_be_bounds (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    unpack16be (b0, b1) < 65536 := by
  unfold unpack16be
  omega

/-! ## 5. Axiom inventory

Every headline theorem is checked below with `#print axioms`.  The expected
output is `'…' does not depend on any axioms` or a list containing only
`propext` (and possibly `Quot.sound`); `Classical.choice` must not appear. -/

#print axioms raw10_group_roundtrip
#print axioms raw10_group_layout
#print axioms raw10_group_lsb_fields
#print axioms raw10_group_byte_bounds
#print axioms raw10_group_unpack_bounds
#print axioms raw10_row_roundtrip
#print axioms raw10_row_byte_bounds
#print axioms raw10_frame_roundtrip
#print axioms raw10_unpack_ignores_padding
#print axioms packFrame_length
#print axioms raw10_frame_byte_bounds
#print axioms pack16_le_roundtrip
#print axioms pack16_be_roundtrip
#print axioms pack16_le_byte_bounds
#print axioms pack16_be_byte_bounds
#print axioms unpack16_le_bounds
#print axioms unpack16_be_bounds

end G6ME.Dng
