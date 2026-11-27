#!/usr/bin/env bash
# check_axioms.sh — the purity gate for the G6ME Lean development.
#
# Four checks, all of which must pass:
#   1. no forbidden source token occurs in code (comments are stripped first,
#      because the module headers legitimately *name* the forbidden tokens in
#      prose);
#   2. `lake build` succeeds;
#   3. theorem coverage: every public `theorem`/`lemma` declared in the
#      sources appears in the EXPECTED registry below, and every EXPECTED
#      name is declared in the sources.  This is the port of HyprEQ's
#      `scripts/lean-verify.sh` coverage invariant: the audit may not silently
#      omit a shipped theorem, and the registry may not name a theorem that
#      does not exist;
#   4. `#print axioms` for every EXPECTED theorem prints only `propext` and
#      `Quot.sound` — never `Classical.choice`, `sorryAx`,
#      `Lean.ofReduceBool` or `Lean.trustCompiler`.
#
# The audit probe is generated from the EXPECTED registry, so the registry is
# the single source of truth: a declaration missing from it fails check 3
# before any axiom line is printed.
#
# Private declarations: enumeration anchors on `theorem `/`lemma ` at column 0
# (optionally preceded on the same line by `@[...]` attributes), exactly the
# `^theorem ` anchor HyprEQ uses.  `private theorem`/`private lemma` and
# `protected theorem` lines do not match and are therefore outside the audit;
# Lean mangles private names, so `#print axioms` cannot address them by source
# name.  The sources currently contain no private or protected declarations.
#
# Every command is bounded with coreutils `timeout`.
# Exit status is 0 only if all four checks pass.

set -uo pipefail
cd "$(dirname "$0")"

fail=0
probe="_axiom_check.lean"
out="_axiom_check.out"
expected_file="_axiom_expected.txt"
trap 'rm -f "$probe" "$out" "$expected_file"' EXIT

echo "== 1/4  forbidden source tokens (block and line comments stripped) =="
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
echo "== 2/4  lake build =="
if timeout 600 lake build; then
  echo "result: OK (lake build exit 0)"
else
  echo "result: FAIL (lake build failed)"
  fail=1
fi

echo
echo "== 3/4  theorem coverage (sources vs EXPECTED registry) =="
if ! python3 - "$expected_file" "$probe" <<'PY'
import collections
import pathlib
import re
import sys

# The EXPECTED registry.  Every public theorem/lemma declared in the sources
# must appear here, and every name here must be a real declaration.  The
# #print axioms probe is generated from this list.
EXPECTED = [
    "G6ME.BlackwellFp4.all_range",
    "G6ME.BlackwellFp4.base13_gemm_identity",
    "G6ME.BlackwellFp4.base13_product_identity",
    "G6ME.BlackwellFp4.encodeD_mem",
    "G6ME.BlackwellFp4.encodeD_reconstruct",
    "G6ME.BlackwellFp4.encodeD_unique",
    "G6ME.BlackwellFp4.fp8_product_identity",
    "G6ME.BlackwellFp4.fp8_split_exists",
    "G6ME.BlackwellFp4.isD_residue_injective",
    "G6ME.BlackwellFp4.remToD_correct",
    "G6ME.BlackwellFp4.remToD_mem",
    "G6ME.BlackwellFp4.remToD_mem_getD",
    "G6ME.BlackwellFp4.remToD_residues",
    "G6ME.Burst3d.K_cube",
    "G6ME.Burst3d.K_mul_w",
    "G6ME.Burst3d.K_skew",
    "G6ME.Burst3d.K_skew_all",
    "G6ME.Burst3d.K_sq",
    "G6ME.Burst3d.K_ww_zero",
    "G6ME.Burst3d.R_as_addM",
    "G6ME.Burst3d.R_gram_conditional",
    "G6ME.Burst3d.R_mul_w",
    "G6ME.Burst3d.R_sub_transpose",
    "G6ME.Burst3d.R_zero",
    "G6ME.Burst3d.delta_comm",
    "G6ME.Burst3d.gram_scalar",
    "G6ME.Burst3d.log_roundtrip",
    "G6ME.Burst3d.log_roundtrip_w0",
    "G6ME.Burst3d.log_roundtrip_w1",
    "G6ME.Burst3d.log_roundtrip_w2",
    "G6ME.Burst3d.mul_addM_addM",
    "G6ME.Burst3d.mul_add_left",
    "G6ME.Burst3d.mul_add_right",
    "G6ME.Burst3d.mul_idM_idM",
    "G6ME.Burst3d.mul_idM_left",
    "G6ME.Burst3d.mul_idM_right",
    "G6ME.Burst3d.mul_smul_left",
    "G6ME.Burst3d.mul_smul_right",
    "G6ME.Burst3d.mul_smul_smul",
    "G6ME.Burst3d.nat3_cases",
    "G6ME.Burst3d.transpose_R_as_addM",
    "G6ME.Burst3d.ww_K_zero",
    "G6ME.Burst3d.ww_ww",
    "G6ME.Crt.M_eq",
    "G6ME.Crt.M_lt_two128",
    "G6ME.Crt.M_pos",
    "G6ME.Crt.add_mod_right_cancel",
    "G6ME.Crt.all_range",
    "G6ME.Crt.cross_dvd_check",
    "G6ME.Crt.cross_dvd_of_ne",
    "G6ME.Crt.crtTerm_mod",
    "G6ME.Crt.crtX_spec",
    "G6ME.Crt.crt_exists",
    "G6ME.Crt.crt_unique",
    "G6ME.Crt.crt_unique_le",
    "G6ME.Crt.dvd_sub_of_mod_eq",
    "G6ME.Crt.foldl_add_congr",
    "G6ME.Crt.foldl_add_mod",
    "G6ME.Crt.foldl_coprime",
    "G6ME.Crt.foldl_mul_dvd",
    "G6ME.Crt.foldl_mul_eq",
    "G6ME.Crt.foldl_range_ite",
    "G6ME.Crt.getD_eq_getElem",
    "G6ME.Crt.inv_check",
    "G6ME.Crt.inv_check_of_lt",
    "G6ME.Crt.mAt_dvd_M_check",
    "G6ME.Crt.mAt_dvd_M_of_lt",
    "G6ME.Crt.mAt_eq_getElem",
    "G6ME.Crt.mAt_mem",
    "G6ME.Crt.mAt_pos_check",
    "G6ME.Crt.mAt_pos_of_lt",
    "G6ME.Crt.moduli_length",
    "G6ME.Crt.moduli_pairwise",
    "G6ME.Dng.base4_field0",
    "G6ME.Dng.base4_field1",
    "G6ME.Dng.base4_field2",
    "G6ME.Dng.base4_field3",
    "G6ME.Dng.div4_mul4_add_mod4",
    "G6ME.Dng.drop_of_length_eq",
    "G6ME.Dng.drop_padRow",
    "G6ME.Dng.pack16_be_byte_bounds",
    "G6ME.Dng.pack16_be_roundtrip",
    "G6ME.Dng.pack16_be_roundtrip_unbounded",
    "G6ME.Dng.pack16_le_byte_bounds",
    "G6ME.Dng.pack16_le_roundtrip",
    "G6ME.Dng.pack16_le_roundtrip_unbounded",
    "G6ME.Dng.packFrame_eq_flatMap",
    "G6ME.Dng.packFrame_length",
    "G6ME.Dng.packGroup_length",
    "G6ME.Dng.packRow_eq_flatMap",
    "G6ME.Dng.packRow_length",
    "G6ME.Dng.padRow_length",
    "G6ME.Dng.raw10_frame_byte_bounds",
    "G6ME.Dng.raw10_frame_roundtrip",
    "G6ME.Dng.raw10_group_byte_bounds",
    "G6ME.Dng.raw10_group_bytes_of_bounds",
    "G6ME.Dng.raw10_group_layout",
    "G6ME.Dng.raw10_group_lsb_fields",
    "G6ME.Dng.raw10_group_roundtrip",
    "G6ME.Dng.raw10_group_roundtrip_append",
    "G6ME.Dng.raw10_group_unpack_bounds",
    "G6ME.Dng.raw10_row_byte_bounds",
    "G6ME.Dng.raw10_row_roundtrip",
    "G6ME.Dng.raw10_row_roundtrip_unbounded",
    "G6ME.Dng.raw10_unpack_ignores_padding",
    "G6ME.Dng.take_of_length_eq",
    "G6ME.Dng.take_packRow",
    "G6ME.Dng.unpack16_be_bounds",
    "G6ME.Dng.unpack16_le_bounds",
    "G6ME.Dng.unpackGroup_cons",
    "G6ME.Dng.unpackGroup_packed",
    "G6ME.Fdn.fdn_gram",
    "G6ME.Fdn.hadamard_gram",
    "G6ME.Fdn.hadamard_transpose_mul",
    "G6ME.Fdn.householder_gram",
    "G6ME.Fdn.householder_symmetric",
    "G6ME.Fdn.vRaw_norm_sq",
    "G6ME.Goldilocks.K_eq",
    "G6ME.Goldilocks.P_eq",
    "G6ME.Goldilocks.P_gt_two63",
    "G6ME.Goldilocks.P_lt_two128",
    "G6ME.Goldilocks.P_lt_two64",
    "G6ME.Goldilocks.P_pos",
    "G6ME.Goldilocks.butterfly_inverse_fst",
    "G6ME.Goldilocks.butterfly_inverse_snd",
    "G6ME.Goldilocks.butterfly_inverse_snd_aux",
    "G6ME.Goldilocks.fieldToI128_i64ToField",
    "G6ME.Goldilocks.fieldToI128_range",
    "G6ME.Goldilocks.hiAllOf_lt",
    "G6ME.Goldilocks.hiOf_lt",
    "G6ME.Goldilocks.i64ToField_lt",
    "G6ME.Goldilocks.inv2_lt",
    "G6ME.Goldilocks.inv2_mul_two",
    "G6ME.Goldilocks.key_aux",
    "G6ME.Goldilocks.loOf_lt",
    "G6ME.Goldilocks.meq_mod",
    "G6ME.Goldilocks.midOf_lt",
    "G6ME.Goldilocks.mid_mul_K_lt",
    "G6ME.Goldilocks.modAdd_lt",
    "G6ME.Goldilocks.modAdd_spec",
    "G6ME.Goldilocks.modMul_lt",
    "G6ME.Goldilocks.modMul_one",
    "G6ME.Goldilocks.modMul_spec",
    "G6ME.Goldilocks.modSub_lt",
    "G6ME.Goldilocks.modSub_spec",
    "G6ME.Goldilocks.mul_lt_two128",
    "G6ME.Goldilocks.ntt2_roundtrip",
    "G6ME.Goldilocks.reduce128_spec",
    "G6ME.Goldilocks.reduce_key_identity",
    "G6ME.Goldilocks.resOf_lt",
    "G6ME.Goldilocks.resOf_meq_s",
    "G6ME.Goldilocks.sOf_lt",
    "G6ME.Goldilocks.sOf_meq",
    "G6ME.Goldilocks.sOf_mod",
    "G6ME.Goldilocks.tOf_lt",
    "G6ME.Goldilocks.two128_eq",
    "G6ME.Goldilocks.two64_eq",
    "G6ME.Goldilocks.two96_eq",
    "G6ME.Goldilocks.two_mul_inv2_mod",
    "G6ME.Hoa.LegEq.mul_ofNat_divNat",
    "G6ME.Hoa.fact_table",
    "G6ME.Hoa.frac_at_minus_one_num",
    "G6ME.Hoa.frac_at_one_num",
    "G6ME.Hoa.hoa_acn_index_inj",
    "G6ME.Hoa.hoa_acn_index_injective",
    "G6ME.Hoa.hoa_acn_index_range",
    "G6ME.Hoa.hoa_acn_index_range_mem",
    "G6ME.Hoa.hoa_acn_pairs_length",
    "G6ME.Hoa.hoa_channel_count",
    "G6ME.Hoa.hoa_order7_channels",
    "G6ME.Hoa.int_mul_two",
    "G6ME.Hoa.int_sub_mul",
    "G6ME.Hoa.legLoop_at_minus_one",
    "G6ME.Hoa.legLoop_at_one",
    "G6ME.Hoa.legLoop_recurrence",
    "G6ME.Hoa.legLoop_recurrence_fails_at_two",
    "G6ME.Hoa.legLoop_recurrence_iff",
    "G6ME.Hoa.legLoop_succ_succ",
    "G6ME.Hoa.legLoop_two_two",
    "G6ME.Hoa.leg_succ_succ",
    "G6ME.Hoa.legendre_at_minus_one",
    "G6ME.Hoa.legendre_at_one",
    "G6ME.Hoa.legendre_recurrence",
    "G6ME.Hoa.neg_one_pow_succ",
    "G6ME.Hoa.neg_one_pow_succ_succ",
    "G6ME.Hoa.taper_energy_identity",
    "G6ME.Hoa.taper_foldl_scaled",
    "G6ME.Hoa.taper_scaled_energy",
    "G6ME.Hoa.taper_unit_energy_eq",
    "G6ME.Kulisch.Acc.add_exact",
    "G6ME.Kulisch.Acc.add_neg_length",
    "G6ME.Kulisch.Acc.add_pos_length",
    "G6ME.Kulisch.B_eq",
    "G6ME.Kulisch.B_pos",
    "G6ME.Kulisch.bitpos_range",
    "G6ME.Kulisch.fp64_term_fits",
    "G6ME.Kulisch.kulisch_single_add_exact",
    "G6ME.Kulisch.length_addAt",
    "G6ME.Kulisch.length_addCarry",
    "G6ME.Kulisch.length_zeros",
    "G6ME.Kulisch.limbVal_addAt",
    "G6ME.Kulisch.limbVal_addCarry",
    "G6ME.Kulisch.limbVal_zeros",
    "G6ME.Kulisch.scaled_mant",
    "G6ME.Limiter.adaa2_delta_passband",
    "G6ME.Limiter.adaa2_div_guard",
    "G6ME.Limiter.adaa2_div_guard_cur",
    "G6ME.Limiter.adaa2_div_guard_prev",
    "G6ME.Limiter.adaa2_passband",
    "G6ME.Limiter.adaa2_passband_delta_zero",
    "G6ME.Limiter.ceiling_in_range",
    "G6ME.Limiter.gain_bounds",
    "G6ME.Limiter.gain_full_weight",
    "G6ME.Limiter.gain_num_le",
    "G6ME.Limiter.gain_zero_weight",
    "G6ME.Limiter.iabs_nonneg",
    "G6ME.Limiter.iabs_zero",
    "G6ME.Limiter.limiter_output_bound",
    "G6ME.Limiter.limiter_output_cross",
    "G6ME.MEq.add",
    "G6ME.MEq.mul",
    "G6ME.Stn.fuzzy_masks_unity",
    "G6ME.Stn.fuzzy_masks_zero_denom",
    "G6ME.Stn.maskTriple_sum",
    "G6ME.Stn.noiseTriple_sum",
    "G6ME.Stn.stn_masks_nonneg_residual",
    "G6ME.Stn.stn_masks_unity",
    "G6ME.Stn.stn_masks_unity_full",
    "G6ME.Upmix.ambient_rescale_conservation",
    "G6ME.Upmix.conj_conj",
    "G6ME.Upmix.conj_mul",
    "G6ME.Upmix.conj_neg",
    "G6ME.Upmix.cross_eq",
    "G6ME.Upmix.cross_neg",
    "G6ME.Upmix.dot_eq_re_mul_conj",
    "G6ME.Upmix.dot_neg_left",
    "G6ME.Upmix.mul_assoc",
    "G6ME.Upmix.mul_comm",
    "G6ME.Upmix.mul_neg_left",
    "G6ME.Upmix.normSq_add",
    "G6ME.Upmix.normSq_conj",
    "G6ME.Upmix.normSq_conj_mul",
    "G6ME.Upmix.normSq_mul",
    "G6ME.Upmix.normSq_neg",
    "G6ME.Upmix.normSq_neg_mul",
    "G6ME.Upmix.normSq_rotMinus",
    "G6ME.Upmix.normSq_rotPlus",
    "G6ME.Upmix.normSq_scale",
    "G6ME.Upmix.parseval_specific",
    "G6ME.Upmix.parseval_unit",
    "G6ME.Upmix.primary_energy",
    "G6ME.Upmix.rescale_energy",
    "G6ME.Upmix.v2_ip_zero",
    "G6ME.Upmix.v2_norm",
    "G6ME.Upmix.zero_branch_energy",
    "G6ME.cancel_add_mod",
    "G6ME.meq_refl",
    "G6ME.meq_symm",
    "G6ME.meq_trans",
    "G6ME.mod_add_mul_left",
    "G6ME.mod_add_mul_right",
    "G6ME.mod_add_mul_right'",
    "G6ME.mod_add_self_left",
    "G6ME.mod_add_self_right",
    "G6ME.mul_lt_mul_left_of_pos",
    "G6ME.two_pow_lt_two_pow",
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


DECL = re.compile(r"^(?:@\[[^\]]*\]\s*)*(theorem|lemma)\s+(\S+)")
NS = re.compile(r"^namespace\s+(\S+)\s*$")
SEC = re.compile(r"^section(?:\s+(\S+))?\s*$")
END = re.compile(r"^end(?:\s+(\S+))?\s*$")

files = sorted(pathlib.Path("G6ME").glob("*.lean"))
umbrella = pathlib.Path("G6ME.lean")
if umbrella.exists():
    files.append(umbrella)

found = []
for path in files:
    stack = []  # (kind, name); kind is "ns" or "sec"
    code = strip_comments(path.read_text(encoding="utf-8"))
    for line in code.splitlines():
        m = NS.match(line)
        if m:
            stack.append(("ns", m.group(1)))
            continue
        m = SEC.match(line)
        if m:
            stack.append(("sec", m.group(1)))
            continue
        m = END.match(line)
        if m:
            name = m.group(1)
            if not stack:
                continue
            if name is None or stack[-1][1] == name:
                stack.pop()
            continue
        m = DECL.match(line)
        if m:
            ns = [n for kind, n in stack if kind == "ns" and n]
            found.append(".".join(ns + [m.group(2)]))

found_set = set(found)
expected_set = set(EXPECTED)
dups_found = sorted(n for n, c in collections.Counter(found).items() if c > 1)
dups_expected = sorted(n for n, c in collections.Counter(EXPECTED).items() if c > 1)
missing = sorted(found_set - expected_set)
stale = sorted(expected_set - found_set)

bad = False
if dups_found:
    print("DUPLICATE declaration(s) in the sources:")
    for n in dups_found:
        print("  " + n)
    bad = True
if dups_expected:
    print("DUPLICATE EXPECTED entr(ies):")
    for n in dups_expected:
        print("  " + n)
    bad = True
if missing:
    print("UNCOVERED theorem(s) declared in the sources but absent from EXPECTED:")
    for n in missing:
        print("  " + n)
    bad = True
if stale:
    print("STALE EXPECTED name(s) with no declaration in the sources:")
    for n in stale:
        print("  " + n)
    bad = True
if bad:
    print("result: FAIL (coverage gap)")
    sys.exit(1)

expected_sorted = sorted(expected_set)
pathlib.Path(sys.argv[1]).write_text("\n".join(expected_sorted) + "\n", encoding="utf-8")
probe = ["import G6ME", ""]
probe += [f"#print axioms {n}" for n in expected_sorted]
pathlib.Path(sys.argv[2]).write_text("\n".join(probe) + "\n", encoding="utf-8")
print(f"result: OK ({len(expected_sorted)} public theorem(s); sources match EXPECTED exactly)")
PY
then
  fail=1
fi

echo
echo "== 4/4  #print axioms for every EXPECTED theorem =="
if [ "$fail" -ne 0 ]; then
  echo "result: SKIPPED (an earlier check failed)"
else
  if ! timeout 300 lake env lean "$probe" > "$out" 2>&1; then
    cat "$out"
    echo "result: FAIL (axiom probe did not elaborate)"
    fail=1
  else
    if ! python3 - "$out" "$expected_file" <<'PY'
import re
import sys

with open(sys.argv[2], encoding="utf-8") as fh:
    expected = [ln.strip() for ln in fh if ln.strip()]
expected_set = set(expected)

FORBIDDEN = {"Classical.choice", "sorryAx", "Lean.ofReduceBool", "Lean.trustCompiler"}
ALLOWED = {"propext", "Quot.sound"}

# Greedy name group: theorem names may themselves contain an apostrophe
# (`mod_add_mul_right'`), which Lean prints as `'...'' depends on axioms: ...`.
line_re = re.compile(
    r"^'(.*)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)$"
)

seen = {}
bad_lines = []
dups = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            bad_lines.append(line)
            continue
        name = m.group(1)
        if name in seen:
            dups.append(name)
        ax = m.group(2)
        axioms = [] if ax is None else [a.strip() for a in ax.split(",") if a.strip()]
        seen[name] = axioms
        print(line)

bad = False
if bad_lines:
    print("UNEXPECTED audit output:")
    for line in bad_lines:
        print("  " + line)
    bad = True
if dups:
    print("DUPLICATE audit line(s): " + ", ".join(sorted(set(dups))))
    bad = True
missing = [n for n in expected if n not in seen]
unexpected = sorted(set(seen) - expected_set)
if missing:
    print("MISSING axiom lines for: " + ", ".join(missing))
    bad = True
if unexpected:
    print("UNEXPECTED audited theorem(s): " + ", ".join(unexpected))
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
all_ax = sorted({a for xs in seen.values() for a in xs})
print(f"result: OK ({len(seen)} theorem(s) checked; "
      f"axioms limited to {all_ax or ['none']})")
PY
    then
      fail=1
    fi
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "PURITY GATE: PASS"
else
  echo "PURITY GATE: FAIL"
fi
exit "$fail"
