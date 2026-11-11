#!/usr/bin/env bash
# Build the optimized, *verified* flavour of G6ME.
#
# Why a script instead of a profile in Cargo.toml: `-C target-cpu` is a codegen
# switch, not a release setting, and putting it in Cargo.toml would also apply it
# to build scripts and to anyone's local debugging build. It goes here, explicitly,
# with the evidence attached.
#
# The flavour is x86-64-v3 on purpose:
#   * it is a named microarchitecture level (AVX2 + FMA + BMI2 + SSE4.2, i.e.
#     Haswell/Excavator and newer), so the resulting binary runs on any such CPU;
#   * `-C target-cpu=native` bakes in the *build machine* — on a Zen 3 box it
#     produces a znver3 binary that SIGILLs elsewhere, and Arch's
#     /etc/makepkg.conf.d/rust.conf injects exactly that flag into every package
#     build. Never use `native` for something you intend to hand to another machine.
#
# Bit-exactness is the whole point of this project, so the optimized build is not
# trusted: it is checked. `target-cpu=x86-64-v3` makes FMA *available*, and an FMA
# in the recursive biquad path would change the filters' state and drift the phase.
# g6me-verify disassembles the shipped DSP and fails on any FMA mnemonic, and also
# builds a deliberate -ffp-contract=fast reference that it requires to diverge, so
# the guard is known to be able to fail.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3 -C llvm-args=-unroll-threshold=1000"
export CARGO_TARGET_DIR=target-v3
# Thin LTO and one codegen unit: whole-program inlining without the link-time
# memory cost of `fat`, matching what Arch's rust.conf exports.
LTO=(--config 'profile.release.lto="thin"' --config 'profile.release.codegen-units=1')

echo "== building x86-64-v3 (RUSTFLAGS=$RUSTFLAGS)"
cargo build --release --locked "${LTO[@]}" --workspace

echo
echo "== the numerical contract, against reference binaries compiled from vendor/g6ct"
"$CARGO_TARGET_DIR/release/g6me-verify" --require-all

echo
echo "== proof that the flags really landed, and really did not fuse"
for b in target/release/g6me target-v3/release/g6me; do
  objdump -d "$b" | grep -E "^\s+[0-9a-f]+:" >/tmp/g6me-dis.txt || true
  printf '%-28s ins=%-7s fma=%-3s ymm=%-5s v-opcodes=%s\n' "$b" \
    "$(wc -l </tmp/g6me-dis.txt)" \
    "$(grep -cE 'vfmadd|vfmsub|vfnmadd|vfnmsub' /tmp/g6me-dis.txt)" \
    "$(grep -c '%ymm' /tmp/g6me-dis.txt)" \
    "$(grep -coE '\bv[a-z0-9]+' /tmp/g6me-dis.txt)"
  rm -f /tmp/g6me-dis.txt
done

echo
echo "== byte-for-byte identity: same inputs through both builds"
# Render with each build and compare SHA-256 of the outputs. Three paths, because
# they exercise different code: the plain matrix, soxr resample + depth + dither,
# and the affine IR convolution.
: "${G6ME_TEST_INPUT:?set G6ME_TEST_INPUT to a multichannel file or folder}"
# The IR path is whoever's machine this runs on; override or leave unset to skip.
G6ME_IR=${G6ME_IR:-$HOME/.config/mpv/ir.wav}
declare -A CFG=([plain]="" [soxr]="--rate 48000 --depth s16 --dither low_shibata")
[[ -f $G6ME_IR ]] && CFG[ir]="--ir $G6ME_IR"

for name in plain soxr ir; do
  [[ -v CFG[$name] ]] || continue
  flags=${CFG[$name]}
  for flavor in "" "-v3"; do
    "target${flavor}/release/g6me" -y -o "/tmp/g6me-opt-$name$flavor" $flags "$G6ME_TEST_INPUT" >/dev/null
  done
  a=$(find "/tmp/g6me-opt-$name" -type f | sort | xargs -r sha256sum | cut -d' ' -f1)
  b=$(find "/tmp/g6me-opt-$name-v3" -type f | sort | xargs -r sha256sum | cut -d' ' -f1)
  if [[ $a == $b ]]; then
    echo "  $name: identical ($(printf '%s\n' "$a" | wc -l) file(s))"
  else
    echo "  $name: DIFFER — the optimized build is not bit-exact, do not ship it" >&2
    exit 1
  fi
done
