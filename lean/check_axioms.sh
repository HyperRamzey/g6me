#!/usr/bin/env bash
# check_axioms.sh — the purity gate for the G6ME Lean development.
#
# Three checks, all of which must pass:
#   1. no forbidden source token occurs in code (comments are stripped first,
#      because the module headers legitimately *name* the forbidden tokens in
#      prose);
#   2. `lake build` succeeds;
#   3. `#print axioms` for every headline theorem prints only `propext` and
#      `Quot.sound` — never `Classical.choice`, `sorryAx`,
#      `Lean.ofReduceBool` or `Lean.trustCompiler`.
#
# Every command is bounded with coreutils `timeout`.
# Exit status is 0 only if all three checks pass.

set -uo pipefail
cd "$(dirname "$0")"

fail=0
probe="_axiom_check.lean"
out="_axiom_check.out"
trap 'rm -f "$probe" "$out"' EXIT

echo "== 1/3  forbidden source tokens (block and line comments stripped) =="
if ! python3 - <<'PY'
import pathlib
import re
import sys

TOKENS = [
    ("sorry",             r"\bsorry\b"),
    ("admit",             r"\badmit\b"),
    ("native_decide",     r"\bnative_decide\b"),
    ("bv_decide",         r"\bbv_decide\b"),
    ("Lean.ofReduceBool", r"\bLean\.ofReduceBool\b"),
    ("Lean.trustCompiler",r"\bLean\.trustCompiler\b"),
    ("Classical.choice",  r"\bClassical\.choice\b"),
]


def strip_comments(src: str) -> str:
    """Remove nested /- ... -/ blocks and -- line comments, keeping newlines."""
    out = []
    i, n, depth = 0, len(src), 0
    while i < n:
        if src.startswith("/-", i):
            depth += 1
            i += 2
            while i < n and depth:
                if src.startswith("/-", i):
                    depth += 1
                    i += 2
                elif src.startswith("-/", i):
                    depth -= 1
                    i += 2
                else:
                    if src[i] == "\n":
                        out.append("\n")
                    i += 1
            continue
        if src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        out.append(src[i])
        i += 1
    return "".join(out)


files = sorted(pathlib.Path("G6ME").glob("*.lean")) + [pathlib.Path("G6ME.lean")]
bad = False
for path in files:
    code = strip_comments(path.read_text(encoding="utf-8"))
    for lineno, line in enumerate(code.splitlines(), 1):
        for name, pattern in TOKENS:
            if re.search(pattern, line):
                print(f"FORBIDDEN {name}: {path}:{lineno}: {line.strip()}")
                bad = True
if bad:
    print("result: FAIL (forbidden token in code)")
    sys.exit(1)
print(f"result: OK ({len(files)} files, no forbidden token outside comments)")
PY
then
  fail=1
fi

echo
echo "== 2/3  lake build =="
if timeout 600 lake build; then
  echo "result: OK (lake build exit 0)"
else
  echo "result: FAIL (lake build failed)"
  fail=1
fi

echo
echo "== 3/3  #print axioms for the headline theorems =="
cat > "$probe" <<'LEAN'
import G6ME

#print axioms G6ME.mul_lt_mul_left_of_pos
#print axioms G6ME.two_pow_lt_two_pow

#print axioms G6ME.Goldilocks.reduce128_spec
#print axioms G6ME.Goldilocks.modAdd_spec
#print axioms G6ME.Goldilocks.modSub_spec
#print axioms G6ME.Goldilocks.modMul_spec
#print axioms G6ME.Goldilocks.i64ToField_lt
#print axioms G6ME.Goldilocks.fieldToI128_i64ToField
#print axioms G6ME.Goldilocks.ntt2_roundtrip

#print axioms G6ME.BlackwellFp4.encodeD_reconstruct
#print axioms G6ME.BlackwellFp4.encodeD_unique
#print axioms G6ME.BlackwellFp4.base13_product_identity
#print axioms G6ME.BlackwellFp4.fp8_product_identity
#print axioms G6ME.BlackwellFp4.fp8_split_exists

#print axioms G6ME.Crt.crt_exists
#print axioms G6ME.Crt.crt_unique
#print axioms G6ME.Crt.moduli_pairwise

#print axioms G6ME.Kulisch.limbVal_addCarry
#print axioms G6ME.Kulisch.limbVal_addAt
#print axioms G6ME.Kulisch.scaled_mant
#print axioms G6ME.Kulisch.Acc.add_exact
#print axioms G6ME.Kulisch.fp64_term_fits
#print axioms G6ME.Kulisch.kulisch_single_add_exact

#print axioms G6ME.Dng.raw10_group_roundtrip
#print axioms G6ME.Dng.raw10_row_roundtrip
#print axioms G6ME.Dng.raw10_frame_roundtrip
#print axioms G6ME.Dng.raw10_unpack_ignores_padding
#print axioms G6ME.Dng.pack16_le_roundtrip
#print axioms G6ME.Dng.pack16_be_roundtrip
#print axioms G6ME.Dng.packFrame_length

#print axioms G6ME.Stn.stn_masks_nonneg_residual
#print axioms G6ME.Stn.stn_masks_unity
#print axioms G6ME.Stn.fuzzy_masks_unity

#print axioms G6ME.Fdn.hadamard_gram
#print axioms G6ME.Fdn.householder_gram
#print axioms G6ME.Fdn.fdn_gram

#print axioms G6ME.Limiter.ceiling_in_range
#print axioms G6ME.Limiter.gain_bounds
#print axioms G6ME.Limiter.limiter_output_bound
#print axioms G6ME.Limiter.adaa2_passband
#print axioms G6ME.Limiter.adaa2_div_guard

#print axioms G6ME.Hoa.legLoop_at_one
#print axioms G6ME.Hoa.legLoop_at_minus_one
#print axioms G6ME.Hoa.legLoop_recurrence_iff
#print axioms G6ME.Hoa.legendre_recurrence
#print axioms G6ME.Hoa.hoa_order7_channels
#print axioms G6ME.Hoa.hoa_acn_index_injective
#print axioms G6ME.Hoa.taper_energy_identity

#print axioms G6ME.Burst3d.K_skew
#print axioms G6ME.Burst3d.K_sq
#print axioms G6ME.Burst3d.K_cube
#print axioms G6ME.Burst3d.R_gram_conditional
#print axioms G6ME.Burst3d.R_sub_transpose
#print axioms G6ME.Burst3d.R_mul_w
#print axioms G6ME.Burst3d.log_roundtrip

#print axioms G6ME.Upmix.parseval_specific
#print axioms G6ME.Upmix.parseval_unit
#print axioms G6ME.Upmix.primary_energy
#print axioms G6ME.Upmix.ambient_rescale_conservation
LEAN

if ! timeout 300 lake env lean "$probe" > "$out" 2>&1; then
  cat "$out"
  echo "result: FAIL (axiom probe did not elaborate)"
  fail=1
else
  if ! python3 - "$out" <<'PY'
import re
import sys

EXPECTED = [
    "G6ME.mul_lt_mul_left_of_pos",
    "G6ME.two_pow_lt_two_pow",
    "G6ME.Goldilocks.reduce128_spec",
    "G6ME.Goldilocks.modAdd_spec",
    "G6ME.Goldilocks.modSub_spec",
    "G6ME.Goldilocks.modMul_spec",
    "G6ME.Goldilocks.i64ToField_lt",
    "G6ME.Goldilocks.fieldToI128_i64ToField",
    "G6ME.Goldilocks.ntt2_roundtrip",
    "G6ME.BlackwellFp4.encodeD_reconstruct",
    "G6ME.BlackwellFp4.encodeD_unique",
    "G6ME.BlackwellFp4.base13_product_identity",
    "G6ME.BlackwellFp4.fp8_product_identity",
    "G6ME.BlackwellFp4.fp8_split_exists",
    "G6ME.Crt.crt_exists",
    "G6ME.Crt.crt_unique",
    "G6ME.Crt.moduli_pairwise",
    "G6ME.Kulisch.limbVal_addCarry",
    "G6ME.Kulisch.limbVal_addAt",
    "G6ME.Kulisch.scaled_mant",
    "G6ME.Kulisch.Acc.add_exact",
    "G6ME.Kulisch.fp64_term_fits",
    "G6ME.Kulisch.kulisch_single_add_exact",
    "G6ME.Dng.raw10_group_roundtrip",
    "G6ME.Dng.raw10_row_roundtrip",
    "G6ME.Dng.raw10_frame_roundtrip",
    "G6ME.Dng.raw10_unpack_ignores_padding",
    "G6ME.Dng.pack16_le_roundtrip",
    "G6ME.Dng.pack16_be_roundtrip",
    "G6ME.Dng.packFrame_length",
    "G6ME.Stn.stn_masks_nonneg_residual",
    "G6ME.Stn.stn_masks_unity",
    "G6ME.Stn.fuzzy_masks_unity",
    "G6ME.Fdn.hadamard_gram",
    "G6ME.Fdn.householder_gram",
    "G6ME.Fdn.fdn_gram",
    "G6ME.Limiter.ceiling_in_range",
    "G6ME.Limiter.gain_bounds",
    "G6ME.Limiter.limiter_output_bound",
    "G6ME.Limiter.adaa2_passband",
    "G6ME.Limiter.adaa2_div_guard",
    "G6ME.Hoa.legLoop_at_one",
    "G6ME.Hoa.legLoop_at_minus_one",
    "G6ME.Hoa.legLoop_recurrence_iff",
    "G6ME.Hoa.legendre_recurrence",
    "G6ME.Hoa.hoa_order7_channels",
    "G6ME.Hoa.hoa_acn_index_injective",
    "G6ME.Hoa.taper_energy_identity",
    "G6ME.Burst3d.K_skew",
    "G6ME.Burst3d.K_sq",
    "G6ME.Burst3d.K_cube",
    "G6ME.Burst3d.R_gram_conditional",
    "G6ME.Burst3d.R_sub_transpose",
    "G6ME.Burst3d.R_mul_w",
    "G6ME.Burst3d.log_roundtrip",
    "G6ME.Upmix.parseval_specific",
    "G6ME.Upmix.parseval_unit",
    "G6ME.Upmix.primary_energy",
    "G6ME.Upmix.ambient_rescale_conservation",
]

FORBIDDEN = {"Classical.choice", "sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"}
ALLOWED = {"propext", "Quot.sound"}

line_re = re.compile(
    r"^'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)$"
)

seen = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        m = line_re.match(line)
        if not m:
            continue
        name = m.group(1)
        ax = m.group(2)
        axioms = [] if ax is None else [a.strip() for a in ax.split(",") if a.strip()]
        seen[name] = axioms
        print(line)

bad = False
missing = [n for n in EXPECTED if n not in seen]
if missing:
    print("MISSING axiom lines for: " + ", ".join(missing))
    bad = True
for name, axioms in seen.items():
    for ax in axioms:
        if ax in FORBIDDEN:
            print(f"FORBIDDEN axiom {ax} on {name}")
            bad = True
        elif ax not in ALLOWED:
            print(f"UNEXPECTED axiom {ax} on {name}")
            bad = True

if bad:
    print("result: FAIL (axiom check)")
    sys.exit(1)
print(f"result: OK ({len(seen)} theorem(s) checked; "
      f"axioms limited to {sorted(set(a for xs in seen.values() for a in xs)) or ['none']})")
PY
  then
    fail=1
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "PURITY GATE: PASS"
else
  echo "PURITY GATE: FAIL"
fi
exit "$fail"
