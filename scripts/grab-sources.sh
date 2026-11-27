#!/usr/bin/env bash
# Copy the project's sources to the clipboard, each one annotated with its path.
#
#   scripts/grab-sources.sh                    # code + docs + build files (default)
#   scripts/grab-sources.sh --stat             # show what would be copied, copy nothing
#   scripts/grab-sources.sh --code             # skip the prose (*.md)
#   scripts/grab-sources.sh --docs             # prose only
#   scripts/grab-sources.sh --data             # also include Cargo.lock and JSON fixtures
#   scripts/grab-sources.sh 'crates/g6me-core/*' 'docs/*.md'   # restrict to path patterns
#   scripts/grab-sources.sh --numbers          # prefix every line with its line number
#   scripts/grab-sources.sh --out bundle.txt   # write a file instead of the clipboard
#
# Selection is from `git ls-files`, so nothing untracked and nothing under target/
# is ever included — `git add` a new file before bundling it. Binaries and files over
# 2 MB are skipped with a note on stderr. Order is stable: the workspace manifest,
# then each crate's manifest and sources, the vendored g6ct reference, scripts,
# packaging, CI, docs, and prose, so two runs of the same tree paste in the same order.
set -euo pipefail

cd "$(dirname "$0")/.."

max_bytes=$((2 * 1024 * 1024))
out=""
to_clipboard=1
numbers=0
stat_only=0
mode="all"
data=0
patterns=()

while (($#)); do
    case "$1" in
        --stat) stat_only=1 ;;
        --code) mode="code" ;;
        --docs) mode="docs" ;;
        --data) data=1 ;;
        --numbers) numbers=1 ;;
        --out) shift; out="${1:?--out needs a path}"; to_clipboard=0 ;;
        -h | --help)
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            exit 2
            ;;
        *) patterns+=("$1") ;;
    esac
    shift
done

# Extension sets. `c`/`h` are the vendored g6ct reference the verifier compares
# against; `mjs` is the docs preview server; `yml`/`yaml` are the CI and release
# workflows; `desktop` is the launcher. `json` and `lock` are data, so they stay out
# until --data asks for them — with the one exception below.
code_exts=(rs c h sh mjs toml yml yaml desktop)
doc_exts=(md)
data_exts=(json lock)
# Files with no extension that are still part of the tree a reviewer needs.
name_allowlist=(LICENSE)
# Executables without an extension. AppRun is the AppImage entry point: shell, not prose.
code_names=(AppRun)
# JSON at these paths is configuration, not measurement data. package.json drives the
# docs preview server, so a reviewer needs it beside the code that serves it.
config_prefixes=(package.json)

# What a path must match to be selected, given the mode. Built once.
wanted=()
case "$mode" in
    code)
        for e in "${code_exts[@]}"; do wanted+=("*.$e"); done
        ;;
    docs)
        for e in "${doc_exts[@]}"; do wanted+=("*.$e"); done
        ;;
    *)
        for e in "${code_exts[@]}" "${doc_exts[@]}"; do wanted+=("*.$e"); done
        ;;
esac
if ((data)); then
    for e in "${data_exts[@]}"; do wanted+=("*.$e"); done
fi

# Order groups explicitly rather than alphabetically: a reader pastes this into a
# review, and "workspace manifest, core crate, then the crates that consume it" reads
# better than "AGENTS.md, Cargo.lock, Cargo.toml, ...". Core comes first because it is
# the DSP the other crates exist to drive.
order_groups=(
    "Cargo.toml"
    "crates/g6me-core/Cargo.toml"
    "crates/g6me-core/src/"
    "crates/g6me-cli/Cargo.toml"
    "crates/g6me-cli/src/"
    "crates/g6me-gui/Cargo.toml"
    "crates/g6me-gui/src/"
    "crates/g6me-gui/testdata/"
    "crates/g6me-verify/Cargo.toml"
    "crates/g6me-verify/build.rs"
    "crates/g6me-verify/src/"
    "vendor/"
    "scripts/"
    "packaging/"
    ".github/"
    "docs/"
    "package.json"
    "README.md"
    "AGENTS.md"
    "Cargo.lock"
    "LICENSE"
)

mapfile -t tracked < <(git ls-files)

in_group() {
    # Is $1 in the same order group as $2? Exact file, or under the directory.
    local path="$1" group="$2"
    case "$group" in
        */) [[ "${path#"$group"}" != "$path" ]] ;;
        *) [[ "$path" == "$group" ]] ;;
    esac
}
is_config() {
    # Is $1 configuration that ships with the code rather than measurement data?
    local path="$1" p
    for p in "${config_prefixes[@]}"; do
        [[ "$path" == "$p"* ]] && return 0
    done
    return 1
}
matches_ext() {
    local path="$1" e
    for e in "${wanted[@]}"; do
        # shellcheck disable=SC2053  # right-hand side is a pattern on purpose
        [[ "$path" == $e ]] && return 0
    done
    if [[ "$mode" != "docs" ]] && is_config "$path"; then
        return 0
    fi
    local base="${path##*/}"
    if [[ "$mode" != "docs" ]]; then
        for allowed in "${code_names[@]}"; do
            [[ "$base" == "$allowed" ]] && return 0
        done
    fi
    if [[ "$mode" != "code" ]]; then
        for allowed in "${name_allowlist[@]}"; do
            [[ "$base" == "$allowed" ]] && return 0
        done
    fi
    return 1
}

declare -a picked=()
for group in "${order_groups[@]}"; do
    for f in "${tracked[@]}"; do
        in_group "$f" "$group" || continue
        matches_ext "$f" || continue
        picked+=("$f")
    done
done
# Anything tracked that no group claims still belongs in the bundle; dropping it
# silently would hand a reviewer a tree that looks complete and is not.
for f in "${tracked[@]}"; do
    claimed=0
    for group in "${order_groups[@]}"; do
        if in_group "$f" "$group"; then
            claimed=1
            break
        fi
    done
    ((claimed)) && continue
    matches_ext "$f" && picked+=("$f")
done

# Optional pattern restriction, matched with the shell's own globbing so
# `crates/g6me-core/*` behaves the way it looks.
if ((${#patterns[@]})); then
    filtered=()
    for f in "${picked[@]}"; do
        for p in "${patterns[@]}"; do
            # shellcheck disable=SC2053
            [[ "$f" == $p ]] && filtered+=("$f") && break
        done
    done
    picked=("${filtered[@]}")
fi

# Drop unreadable, binary and oversized files, saying so on stderr.
final=()
skipped=0
for f in "${picked[@]}"; do
    if [[ ! -r "$f" ]]; then
        echo "skip (unreadable): $f" >&2
        skipped=$((skipped + 1))
        continue
    fi
    size=$(wc -c <"$f")
    if ((size > max_bytes)); then
        echo "skip (over 2 MB): $f ($size bytes)" >&2
        skipped=$((skipped + 1))
        continue
    fi
    if ! grep -qI '' "$f" 2>/dev/null; then
        echo "skip (binary): $f" >&2
        skipped=$((skipped + 1))
        continue
    fi
    final+=("$f")
done

if ((${#final[@]} == 0)); then
    echo "nothing selected" >&2
    exit 1
fi

total=0
for f in "${final[@]}"; do
    total=$((total + $(wc -c <"$f")))
done

{
    echo "selected: ${#final[@]} file(s), ${total} bytes"
    echo "branch  : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "tree    : DIRTY - the bundle contains working-tree contents, not HEAD"
    else
        echo "tree    : clean"
    fi
    ((skipped)) && echo "skipped : $skipped"
} >&2

if ((stat_only)); then
    printf '%s\n' "${final[@]}" >&2
    exit 0
fi

emit() {
    local f="$1"
    local lines size rel
    lines=$(wc -l <"$f")
    size=$(wc -c <"$f")
    rel="$f"
    printf '\n'
    printf '=%.0s' {1..78}
    printf '\nFILE: %s  (%s lines, %s bytes)\n' "$rel" "$lines" "$size"
    printf '=%.0s' {1..78}
    printf '\n'
    if ((numbers)); then
        nl -ba "$f"
    else
        cat "$f"
    fi
    # A trailing newline per file keeps the last line of one file from running into
    # the banner of the next when this is pasted into a fixed-width viewer.
    printf '\n'
}

header() {
    printf '=%.0s' {1..78}
    printf '\ng6me sources  (%s @ %s%s)\n' \
        "$(git rev-parse --abbrev-ref HEAD)" \
        "$(git rev-parse --short HEAD)" \
        "$([[ -n "$(git status --porcelain)" ]] && echo ', working tree')"
    printf '%s file(s), %s bytes, generated %s\n' \
        "${#final[@]}" "$total" "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'Each block is one file, introduced by a FILE: line. No file was edited to\n'
    printf 'produce this; the tree is what it is.\n'
    printf '=%.0s' {1..78}
    printf '\n'
}

if [[ -n "$out" ]]; then
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } >"$out"
    echo "wrote $out" >&2
    exit 0
fi

if ((to_clipboard)); then
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } | if command -v wl-copy >/dev/null 2>&1 && [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        wl-copy
    elif command -v xclip >/dev/null 2>&1; then
        xclip -selection clipboard
    else
        echo "no wl-copy or xclip found; writing sources-bundle.txt instead" >&2
        cat >sources-bundle.txt
    fi
else
    {
        header
        for f in "${final[@]}"; do emit "$f"; done
    } | cat
fi
