#!/usr/bin/env bash
# Build the optimized x86-64-v3 G6ME flavour and install the binaries.
#
# The optimization set is exactly the one scripts/build-optimized.sh proves
# bit-exact: x86-64-v3 (AVX2/FMA/BMI2, never `native`), thin LTO, one codegen
# unit, O3 and the wider unroll threshold. `--locked` keeps Cargo.lock honest.
#
# Before replacing anything, the numerical contract is run through
# `g6me-verify --require-all`. Existing binaries are kept as
# `<name>.pre-<YYYYmmdd_HHMMSS>.bak`, so an install is reversible.
#
# Usage: scripts/build-install.sh [--skip-verify] [--prefix DIR]
#   G6ME_INSTALL_DIR overrides the default prefix (~/.local/bin).
#   CARGO_TARGET_DIR overrides the default build directory (target-v3).
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

prefix="${G6ME_INSTALL_DIR:-$HOME/.local/bin}"
verify=1
while (($#)); do
  case "$1" in
    --skip-verify) verify=0; shift ;;
    --prefix)
      prefix=${2:?--prefix needs a directory}
      shift 2
      ;;
    --prefix=*) prefix=${1#*=}; shift ;;
    -h | --help)
      awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n1)
: "${version:?could not read the workspace version from Cargo.toml}"

export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3 -C llvm-args=-unroll-threshold=1000"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target-v3}"
lto=(--config 'profile.release.lto="thin"' --config 'profile.release.codegen-units=1')

echo "== building G6ME $version (x86-64-v3, thin LTO, $CARGO_TARGET_DIR)"
cargo build --release --locked "${lto[@]}" --workspace --bins

if ((verify)); then
  echo
  echo "== numerical contract (g6me-verify --require-all)"
  "$CARGO_TARGET_DIR/release/g6me-verify" --require-all
fi

echo
echo "== installing into $prefix"
install -d "$prefix"
stamp=$(date +%Y%m%d_%H%M%S)
for bin in g6me g6me-scan g6me-gui; do
  src="$CARGO_TARGET_DIR/release/$bin"
  if [[ ! -x $src ]]; then
    echo "missing built binary: $src" >&2
    exit 1
  fi
  if [[ -e $prefix/$bin ]]; then
    cp -a "$prefix/$bin" "$prefix/$bin.pre-$stamp.bak"
  fi
  install -m 0755 "$src" "$prefix/$bin"
done

echo "installed: $("$prefix/g6me" --version)"
case ":$PATH:" in
  *":$prefix:"*) ;;
  *) echo "warning: $prefix is not on PATH" >&2 ;;
esac
