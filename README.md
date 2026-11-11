# G6ME — G6 Media Encoder

Batch renderer for the **G6 Creative MatrixEncoder** binaural downmixer: takes
7.1 / 5.1 / Dolby-Atmos-12 multichannel audio and writes the same 2-channel
sum-and-biquad matrix the original plugin produces, as FLAC or WAV, from a
command line or a GUI, on Linux and Windows.

It exists because the plugin itself is a Windows VST2 insert. Real libraries are
files on a disk, and you want the whole library converted overnight with the same
numbers — including `.mka` containers holding Dolby TrueHD + Atmos, which the
plugin can only play in real time through a host.

```
multichannel file ──ffmpeg decode──► f64 samples ──► Dsp (f64) ──► encoder ──► FLAC/WAV
   .mka .wav .flac .mkv .mp4 …        (optional soxr      (the exact        (optional 24-bit
                                       resample, IR)       plugin matrix)    via `flac`)
```

## What is guaranteed

| | |
|---|---|
| **Bit-exact to the reference** | The DSP is a port of `g6ct.c` (vendored under `vendor/g6ct/`) and is compared against binaries compiled from that exact C source by `g6me-verify`: 98 checks, all passing. |
| **f64 internals** | The downmixer runs in `f64` — the same precision as the reference's `-DG6CT_DOUBLE` build and mpv's `afir … precision=double`. Not configurable, on purpose: state precision is not a taste parameter. |
| **No FMA, no fast-math, ever** | Contraction changes IIR state, and IIR state is phase. `g6me-verify` disassembles the shipped DSP and the reference builds and fails if a fused multiply-add appears; it also builds a deliberate `-ffp-contract=fast` counter-example and requires that it *does* diverge, so the guard is known to have teeth. |
| **Resampleless by default** | Nothing touches the sample rate, bit depth or timing unless you ask. The default path is decode → f64 → downmix → encode. |
| **Deterministic** | Same input, same flags, same bytes. Repeat a run with `--sha256` and compare. |
| **No half-written output** | A cancelled or failed file leaves no playable partial file behind. |
| **DSP is single-threaded per file** | One contiguous block sequence through the filters. Parallelism is *between* files; decode / DSP / encode overlap per file through bounded queues, so the memory ceiling is explicit (`--queue`). |

What is *not* guaranteed: that a downmix that clips on your headphones is
pleasant. The matrix is the plugin's, gains and all.

## Requirements

* **ffmpeg** with `libsoxr`, `afir` and `lavfi` — checked at runtime, not assumed.
  Developed and tested against `n9.0.2`. Without libsoxr the program refuses to
  resample rather than silently picking another resampler; the default
  (no resample) path still works.
* **`flac`** from the reference `flac-tools` build only if you want **24-bit FLAC** —
  ffmpeg's own FLAC encoder emits 16- or 32-bit, so 24-bit goes through the
  reference encoder. Other depths and WAV need nothing extra.
* **Rust 1.87+** to build (developed on 1.99). A C compiler only for
  `g6me-verify`, which builds the reference binaries to compare against; without a
  compiler those checks are skipped rather than failing the build.
* No audio device, ever, for the GUI. It has no cpal/ALSA/PipeWire/Pulse/JACK
  dependency at all and is structurally unable to open a stream — meters come from
  the samples being rendered.

```sh
# capability check: what can this ffmpeg actually do, and is 24-bit FLAC available
cargo run -p g6me-cli -- --caps
```

## Build

```sh
cargo build --release            # g6me (CLI) and g6me-gui in target/release/
cargo test --workspace           # 133 tests
cargo run -p g6me-verify         # the bit-exactness evidence (98 checks)
```

## Command line

```sh
# a whole library, FLAC level 8, four files at a time
g6me -o /music/binaural -j 4 /music/atmos/

# one album to 16-bit FLAC, resampled first, with the headphone calibration IR
g6me -o out --rate 48000 --depth s16 --dither low_shibata \
     --ir ~/.config/mpv/ir.wav album.mka

# exactly what would run, without running it
g6me --dry-run -o out --rate 44100 concert.mka
```

Inputs are files **and folders**, as many as you like; folders are scanned
recursively and the tree is mirrored under `-o`. Without `-o`, each output lands
next to its input. Existing outputs are skipped, not overwritten (`-y` to force —
and `--force` will still never truncate an input file).

The switches that matter most, and why:

| flag | why it exists |
|---|---|
| `-r, --rate` | Rate conversion happens **before** the downmix, through libsoxr, and only when you pass it. Doing it afterwards would make the resampler part of the matrix's phase response. |
| `--depth`, `--dither` | Depth reduction is a separate, explicit decision from rate conversion. `--dither low_shibata \| triangular \| none`. |
| `--ir` | Mono headphone-calibration impulse convolved into every input channel first, exactly the way mpv's `afir` does (`irnorm=-1`, `precision=double`), so the calibration you already use for listening is the one applied here. |
| `--order FL,FR,FC,LFE,BL,BR,SL,SR` | Some containers mislabel their channel layout. This overrides it instead of guessing. |
| `--queue` | The memory bound: `queue × block × channels × 8` bytes per file. |
| `--sha256` | Prints a hash per output, so "deterministic" is something you can check rather than believe. |
| `--monitor`, `--device` | Play the result through an output device instead of writing a file (CLI only; the GUI has no audio path at all). |

Full help: `g6me --help`. `--json` emits one object per finished file for scripts.

## GUI

```sh
cargo run -p g6me-gui
```

Wayland, X11 and Windows (eframe/egui). Same controls as the CLI — format, depth,
level, resample, dither, IR, channel order, threads, block, queue, force, dry run,
SHA-256 — plus:

* **Queue with real multi-selection.** Click, Ctrl-click to add or remove, Shift-click
  for a range, double-click to commit. Selected count is shown before you commit.
* **Drive and volume shortcuts**, read from the mount table (Linux), the drive letters
  (Windows) or `/Volumes` (macOS), filtered to things worth visiting — no `/proc`,
  no `tmpfs`, no squashfs snaps, and `/run/media/<you>/…` does appear.
* **Per-file meters.** Every file being rendered gets its own card: the plugin's eight
  input slots plus the two outputs, sample peak, peak-hold, decay, with the same
  0.30 s fall-off the VST2 meter used. Cards keep their last frame after the file
  finishes, so you can see what clipped after the batch is done, and clicking a card
  clears its holds.
* **Dry run before you commit**: source codec and profile, detected layout, the channel
  order and mask actually chosen, the input stage selected, the human-readable chain,
  and the exact ffmpeg command lines.
* **Four themes**: system, dark, light, and a true `#000000` OLED black.

## Verification

`g6me-verify` is the reason this project is trustworthy rather than plausible. It
compiles the vendored C reference in eight configurations (`contract_off`,
`native`, `o3_native`, plus a `-ffp-contract=fast` counter-example, in f32 and f64)
and runs:

* `bitexact` — Rust f64 vs each canonical C f64 build, over 8 signals and 4 input
  configurations (7.1, 5.1, Atmos-12, front-only — a mask selects which filters
  exist at all, so one mask would prove little).
* `builds` — the canonical C builds agree with each other; any disagreement means the
  build flags are load-bearing, which is a failure, not a footnote.
* `chunk` — different block sizes, identical bytes.
* `continuity` — filter state across block boundaries is load-bearing: resetting it
  mid-file *must* change the output (it does, in ~6.6k of 200k frames).
* `phase`, `drift` — per-channel phase comparison and 60 s of continuous running.
* `contraction` — the counter-example build *must* diverge, and `asm` — the shipped
  DSP and the canonical reference builds must contain no FMA.

```
98 passed, 0 failed, 0 skipped
```

Run it through `cargo run -p g6me-verify` (or `cargo test -p g6me-verify`) rather
than a stale binary in `target/` — an out-of-date verifier is worse than none.

## Optimized builds (x86-64-v3)

The default release build is generic x86-64. The optimized flavour is **explicit**, never
`native`:

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3 -C llvm-args=-unroll-threshold=1000"
# + lto=thin, codegen-units=1
scripts/build-optimized.sh   # builds it, runs the verifier, and compares output bytes
```

x86-64-v3 is a named microarchitecture level (AVX2 + FMA + BMI2 + SSE4.2 — Haswell and
Excavator onwards), so the binary runs on any such CPU. `-C target-cpu=native` flavours
the output for the build machine and SIGILLs on the next one, so it is not acceptable for
anything you hand to someone else — and Arch's `/etc/makepkg.conf.d/rust.conf` injects
exactly that flag into *every* Rust package build, which means a `PKGBUILD` has to
override it:

```sh
build() {
  export RUSTFLAGS="-C target-cpu=x86-64-v3 -C opt-level=3"   # not -C target-cpu=native
  export CARGO_TARGET_DIR=target-v3
  cargo build --release --locked --workspace
}
```

v3 makes FMA *available*, and a single FMA inside a recursive biquad changes the filter
state and drifts phase, so the optimized build is checked rather than trusted. Measured
here on a Ryzen 7 5700X3D:

| build | instructions | FMA | `%ymm` refs | verifier |
|---|---|---|---|---|
| generic (default) | 294,080 | **0** | 123 | 98/98 |
| x86-64-v3 + thin LTO + O3 + unroll | 313,249 | **0** | 4,515 | 98/98 |

For scale, the deliberate `-ffp-contract=fast` reference build contains 32 FMA
instructions and fails the same comparison, so "no FMA found" means something.

End to end the two builds write **byte-identical** FLAC — SHA-256 compared — for the
plain matrix, for `--rate 48000 --depth s16 --dither low_shibata`, and for the IR
convolution.

What it actually buys: about 12 % off the DSP's own time (150 ms → 132 ms on the 60 s
verification case). Total throughput did not move measurably, because ffmpeg decode and
encode dominate a real render — so build v3 when you are CPU-bound on known hardware, and
do not make it the default you ship.

## Layout

```
crates/g6me-core     DSP port, pipeline (ffmpeg ⇄ DSP ⇄ encoder), batch engine,
                     capabilities probe, meters, WAV I/O, monitoring (CLI-side)
crates/g6me-cli      `g6me`
crates/g6me-gui      `g6me-gui` — eframe, no audio backend
crates/g6me-verify   bit-exactness harness; builds vendor/g6ct in 8 configurations
vendor/g6ct          the C reference, byte-identical to upstream (see UPSTREAM.md)
```

## Known gaps

* **True-peak metering is not in yet.** The meters show sample peak. Inter-sample
  peak (EBU Tech 3341 / BS.1770-4) is planned next; sample peak alone under-reports
  up to ~3 dB for material near full scale, which matters exactly when you are
  closest to clipping.
* **24-bit FLAC depends on the reference `flac` CLI** being installed. It is detected
  at runtime; without it the GUI says so and the CLI refuses, rather than quietly
  writing 32-bit.
* **No macOS CI**, although the mount-list code has a macOS path and eframe runs
  there.
* `--monitor` (CLI only) is deliberately unverified on Windows.

## Provenance and license

MIT — see [LICENSE](LICENSE). The DSP is a port of the G6 Creative MatrixEncoder
analysis; the C reference under `vendor/g6ct/` is kept byte-identical to its source
(`vendor/g6ct/UPSTREAM.md` records which) so that "we match the reference" stays a
checkable claim rather than a memory.
