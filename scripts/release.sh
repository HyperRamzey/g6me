#!/usr/bin/env bash
# Build distributable G6ME artefacts: a Linux AppImage, a Linux tarball, and a
# Windows zip — all with the same, explicitly-pinned optimization set that
# scripts/build-optimized.sh proves bit-exact.
#
# Pinning matters more here than anywhere else in the repo. A release binary is the
# one build someone else's CPU has to run and someone else's archive has to stay
# identical to, so:
#
#   * `-C target-cpu=x86-64-v3`, never `native`. Arch's
#     /etc/makepkg.conf.d/rust.conf injects `native` into every Rust package build,
#     which would flavour these for the build machine and SIGILL on the next one;
#     RUSTFLAGS is set here so it wins over that file.
#   * thin LTO + one codegen unit + O3 + the unroller, the same set the verifier was
#     run against. v3 makes FMA reachable, and one FMA in the recursive biquad path
#     changes filter state, so the FMA guard below is re-run over *these* binaries,
#     Linux and Windows alike.
#   * `--locked`, so the resolve in Cargo.lock is the resolve that was tested.
#
# What is NOT bundled, on purpose:
#   * ffmpeg — a separate GPL-covered program, huge, and present on any desktop that
#     plays audio. G6ME finds it on PATH; G6ME_FFMPEG / G6ME_FFPROBE override.
#   * OpenGL / Vulkan driver libraries — bundling a vendor GL against the target's
#     driver is the classic broken-AppImage story, so those stay with the system.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

# `--remap-path-prefix` because a Rust release binary keeps the source paths of every
# panic message in .rodata, and `strip --strip-all` does not remove them. Without this,
# the same commit compiled in /home/runner/work/g6me/g6me and in a checkout on /mnt
# are different bytes for a reason that has nothing to do with the program.
export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3 -C llvm-args=-unroll-threshold=1000"
RUSTFLAGS="$RUSTFLAGS --remap-path-prefix=$here=/build --remap-path-prefix=${HOME:-/root}/.cargo=/cargo"
# Symbols stay in while the verifier runs — its disassembly guard reads them — and are
# stripped from the copies that actually ship, below.
CARGO_CFG=(--config 'profile.release.lto="thin"' --config 'profile.release.codegen-units=1'
  --config 'profile.release.strip="none"')
V=$(cargo metadata --no-deps --format-version 1 |
  jq -r '.packages[] | select(.name=="g6me-gui") | .version')
dist="$here/dist"
rm -rf "$dist"
mkdir -p "$dist"

# The driver stack stays outside the bundle; everything else gets copied in.
EXCLUDES=()
for l in 'libGL*' 'libEGL*' 'libGLES*' 'libvulkan*' 'libdrm*' 'libgbm*' 'libxshmfence*' \
  'libwayland-client*' 'libwayland-cursor*' 'libX11*' 'libXext*' 'libpthread*' 'libc.*' \
  'libm.*' 'libdl*' 'librt*' 'ld-linux*' 'libresolv*' 'libdbus*' 'libudev*'; do
  EXCLUDES+=(--exclude-library "$l")
done

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- linux ---------
say "linux x86-64-v3"
CARGO_TARGET_DIR=target-rel cargo build --release --locked "${CARGO_CFG[@]}" --workspace
L=target-rel/release

say "the numerical contract, for exactly these flags"
"$L/g6me-verify" --require-all | tail -3

# What ships is stripped; what was verified keeps its symbols. Strip first, then
# assemble, so nothing ends up packing an unstripped binary by accident.
say "stage the shipping binaries (stripped)"
bin="$dist/bin"
mkdir -p "$bin"
install -m755 "$L/g6me" "$L/g6me-gui" "$bin/"
strip --strip-all "$bin/g6me" "$bin/g6me-gui"

say "AppImage"
appdir="$dist/G6ME-$V.AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" \
  "$appdir/usr/share/icons/hicolor/scalable/apps" "$appdir/usr/share/doc/g6me"
install -m755 "$bin/g6me" "$bin/g6me-gui" "$appdir/usr/bin/"
cp packaging/linux/g6me.desktop "$appdir/usr/share/applications/"
cp packaging/linux/g6me.svg "$appdir/usr/share/icons/hicolor/scalable/apps/"
cp packaging/linux/AppRun "$appdir/AppRun"
chmod +x "$appdir/AppRun"
cp README.md LICENSE "$appdir/usr/share/doc/g6me/"
ln -sf usr/share/icons/hicolor/scalable/apps/g6me.svg "$appdir/.DirIcon"

# Cached outside dist, which is only ever the thing you upload.
tool=${LINUXDEPLOY:-$here/.g6me-cache/linuxdeploy}
if [[ ! -x $tool ]]; then
  curl -sL -o "$tool" --create-dirs \
    https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
  chmod +x "$tool"
fi
# Run with dist as the working directory: linuxdeploy writes the AppImage to the CWD and
# names it after the desktop file's Name= ("G6 Media Encoder" ->
# G6_Media_Encoder-<ver>-x86_64.AppImage), so globbing for a guessed name is how an
# AppImage gets quietly left behind in the source tree.
(
  cd "$dist"
  LINUXDEPLOY_OUTPUT_VERSION=$V APPIMAGE_EXTRACT_AND_RUN=1 NO_STRIP=true \
    "$tool" --appdir "$appdir" "${EXCLUDES[@]}" \
    --desktop-file "$here/packaging/linux/g6me.desktop" \
    --icon-file "$here/packaging/linux/g6me.svg" \
    --output appimage >linuxdeploy.log 2>&1
)
if ! compgen -G "$dist/*.AppImage" >/dev/null; then
  echo "linuxdeploy produced no AppImage; last lines of its log:" >&2
  tail -20 "$dist/linuxdeploy.log" >&2
  exit 1
fi
printf '  %s\n' "$dist"/*.AppImage

say "linux tarball (for a headless box or a server, no FUSE needed)"
# Fixed mtime, ownership and order, `gzip -n` to drop the timestamp gzip would embed:
# an archive that differs per build machine cannot be checked by a user who downloads
# it twice from two mirrors, which is the entire point of publishing sha256sums.
tarball() {
  local dir=$1 out=$2; shift 2
  tar --sort=name --format=gnu --owner=0 --group=0 --numeric-owner \
    --mtime='UTC 1970-01-01' -C "$dir" -cf - "$@" | gzip -n >"$out"
}
tarball "$bin" "$dist/g6me-$V-linux-x86_64.tar.gz" g6me g6me-gui
tarball "$here/packaging/linux" "$dist/g6me-$V-linux-x86_64-desktop.tar.gz" g6me.desktop g6me.svg

# -------------------------------------------------------------- windows ---------
say "windows x86_64-pc-windows-gnu, same flags"
CARGO_TARGET_DIR=target-win cargo build --release --locked "${CARGO_CFG[@]}" \
  --target x86_64-pc-windows-gnu --workspace
W=target-win/x86_64-pc-windows-gnu/release
for e in g6me.exe g6me-gui.exe; do
  [[ -f $W/$e ]] || { echo "missing $W/$e" >&2; exit 1; }
  # Stripped by the mingw binutils, not the host's — the wrong `strip` on a PE either
  # fails or quietly does nothing.
  x86_64-w64-mingw32-strip --strip-all "$W/$e" 2>/dev/null || true
done

cat >"$dist/README-WINDOWS.txt" <<EOF
G6 Media Encoder $V — Windows

  g6me.exe       command line:  g6me -o out -j 4 folder\
  g6me-gui.exe   the window

Needs ffmpeg on PATH (or set G6ME_FFMPEG / G6ME_FFPROBE). It is not bundled: it is a
separate GPL program, and shipping it here would oblige this archive to carry its
sources too. Any recent build works; 24-bit FLAC output additionally needs the
reference flac.exe. Test what you have with:

  g6me.exe --caps

Built for x86-64-v3 (AVX2 and later: Haswell/Excavator 2013+). On an older CPU use a
generic build instead — cargo build --release --locked.
EOF

# `zip` stores each file's mtime, so the same exes zipped an hour apart are different
# bytes. python's zipfile with a fixed timestamp is deterministic and always present.
python3 - "$W" "$dist" "$V" <<'PY'
import pathlib, sys, zipfile
w, d, v = map(pathlib.Path, sys.argv[1:4])
with zipfile.ZipFile(d / f"g6me-{v}-windows-x86_64.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for n in ("g6me.exe", "g6me-gui.exe"):
        info = zipfile.ZipInfo(n, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, (w / n).read_bytes())
    info = zipfile.ZipInfo("README-WINDOWS.txt", date_time=(1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, (d / "README-WINDOWS.txt").read_bytes())
PY

# ------------------------------------------------------------- the guard --------
say "FMA guard over the artefacts themselves"
# The verifier already scanned the DSP it was built from; this scans the binaries
# that actually ship, including the PE. An FMA here would mean the release build
# drifted from what was verified.
fma_scan() {
  local f=$1 ins fma
  # A stripped binary disassembles fine, but a scan that decoded nothing would also
  # report zero FMA and look like a pass, so the instruction count is asserted too.
  ins=$(objdump -d --no-show-raw-insn "$f" 2>/dev/null | grep -cE '^ ' || true)
  fma=$(objdump -d --no-show-raw-insn "$f" 2>/dev/null |
    grep -cE '\b(vfmadd|vfmsub|vfnmadd|vfnmsub)[a-z0-9]*[ \t]' || true)
  printf '  %-46s ins=%-7s fma=%s\n' "$(basename "$f")" "${ins:-0}" "${fma:-0}"
  ((ins > 1000)) || { echo "  could not disassemble $(basename "$f") — the guard is not running" >&2; exit 1; }
  ((fma == 0)) || { echo "  FMA in a shipping binary — do not release" >&2; exit 1; }
}
fma_scan "$appdir/usr/bin/g6me"
fma_scan "$appdir/usr/bin/g6me-gui"
fma_scan "$W/g6me.exe"
fma_scan "$W/g6me-gui.exe"

say "artefacts"
# Staging trees are not deliverables; drop them so `dist` is exactly what gets uploaded.
rm -rf "$bin" "$appdir"
(
  cd "$dist"
  find . -maxdepth 1 -type f ! -name sha256sums.txt ! -name linuxdeploy.log ! -name files.txt -printf '%P\n' |
    sort >files.txt
  # coreutils' sha256sum has no --files-from, and a glob would skip nothing but also
  # choke on the staging directories, so feed it the list.
  xargs -a files.txt -d '\n' sha256sum -- >sha256sums.txt
  while read -r f; do
    printf '  %8s  %s\n' "$(du -h --apparent-size -- "$f" | cut -f1)" "$f"
  done <files.txt
  rm -f files.txt
)
printf '\n  (cd dist && sha256sum -c sha256sums.txt)\n'
