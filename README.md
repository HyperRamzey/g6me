# G6ME — G6 Media Encoder

> **Disclaimer:** This is an independent research and interoperability project. It is not
> affiliated with, endorsed by, or sponsored by Creative Technology Ltd. Sound Blaster and
> Sound BlasterX are trademarks of Creative Technology Ltd. No proprietary firmware or
> driver binaries are distributed in this repository.

Batch renderer for the **G6 Creative MatrixEncoder** binaural downmixer: takes
7.1 / 5.1 / Dolby-Atmos-12 (overheads are nulled, use DME\EE first) multichannel audio and writes the same 2-channel
sum-and-biquad matrix the original plugin produces, as FLAC or WAV, from a
command line or a GUI, on Linux and Windows.

It exists because the plugin itself is a Windows VST2 insert. Real libraries are
files on a disk, and you want the whole library converted overnight with the same
numbers — including `.mka` containers holding Dolby TrueHD + Atmos, which the
plugin can only play in real time through a host.

```
multichannel file ──ffmpeg decode──► f64 samples ──► Dsp (f64) ──► encoder ──► FLAC/WAV
   .mka .wav .flac .mkv .mp4 …        (optional soxr      (the exact        (24-bit FLAC,
                                       resample, IR)       plugin matrix)    or any depth in WAV)
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
* Nothing else for output. ffmpeg's own FLAC encoder writes genuine 16- and 24-bit
  FLAC — `g6me --caps` proves it on your build by encoding a known 24-bit block and
  decoding it back — and FLAC carries integers only, so 32-bit and float go to WAV.
  Compression levels are ffmpeg's own scale, 0 (fastest) to 12 (smallest).
* **Rust 1.87+** to build (developed on 1.99). A C compiler only for
  `g6me-verify`, which builds the reference binaries to compare against; without a
  compiler those checks are skipped rather than failing the build.
* No audio device, ever, for the GUI. It has no cpal/ALSA/PipeWire/Pulse/JACK
  dependency at all and is structurally unable to open a stream — meters come from
  the samples being rendered.

```sh
# capability check: what can this ffmpeg actually do, incl. the 24-bit FLAC round-trip
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
recursively and the tree is mirrored under `-o`. Give it **two or more** folders and each
one keeps its own folder name under `-o`, because mirroring both against their own root puts
`AlbumA/01.mka` and `AlbumB/01.mka` on the same `out/01.flac` — and that collision is silent,
since the second file is skipped as already rendered. A folder reached twice, or a symlink
pointing back at an ancestor, is visited once. Without `-o`, each output lands
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

## True peak

Every render reports the true peak of what it wrote, in dBTP, **with the ratio it was
measured at** — `tp +0.23 dBTP @x4`. A dBTP number without a ratio is not comparable to
anyone else's number, so the ratio is part of the measurement rather than a footnote.
Alongside it: per-channel true peak, sample peak and RMS, all taken after the limiter and
before the encoder, so they describe the signal the file is made of.

**How it is measured.** ITU-R BS.1770-5 Annex 2 order-48, 4-phase, 12-tap polyphase FIR for
4x — the Recommendation's own printed coefficients, in canonical oldest→newest order, with
none of the 12.04 dB pre-attenuation (every stage here is f64). BS.1770 tabulates 4x only,
so 2x/8x/16x/32x are designed in this repo: centred Kaiser-windowed polyphase banks, 51
taps per phase, 80 dB stopband, a stated band edge at 0.45·fs, each phase DC-normalised,
plus parabolic refinement of the crest between grid points and a floor at the sample peak
(no interpolating lowpass can read above its own transition band at Nyquist). Accuracy
gates in `tp.rs` compare against an **exact periodic FFT reconstruction**, and every offered
ratio must land inside its stated error.

What that reference is *not*: ffmpeg's `soxr`. Measured on a single tone, soxr at 4x reports
−17.6 dB at 21.6 kHz and −33.6 dB at 23.52 kHz — it guts the top of the band, so using it as
ground truth for 16x/32x made our filters look like they over-read by half a dB when it was
the reference under-reading. Cross-checks with it are still shown, labelled as such.

**Limits, stated.** The printed 4x bank reads up to about 0.6 dB low on a crest landing
between grid points, which is why the limiter defaults to 8x while the meter stays at 4x: a
ceiling cannot be held by a meter that low. Content above 0.45·fs carries a separate
bandwidth caveat (`band_caveat_db`) rather than being folded into the number above it.

**The limiter** is the last stage before ffmpeg receives bytes — before any quantisation or
clipping — so it is the real ceiling on the delivered file, not an intention. Two stages,
both on the true-peak side chain:

* **Transient.** Keeps a lookahead schedule: it sees a peak before the sample grid does and
  ramps the gain in ahead of it. Returns in 25 ms. Linked 100 % by default — a peak in one
  channel is still a peak in the file.
* **Release.** A peak-hold window with a 60 ms attack and a 900 ms release: it answers level
  rather than isolated crests. Linked 75 % by default, because a gain move lasting a third of
  a second is long enough to pull a stereo image about.

Emission multiplies the two, and the transient stage's demand is computed against the peak
as the release stage left it, so they never spend the same dB twice. The ceiling is aimed
`guard_db` low — the part the meter cannot see plus what the gain slew can still overshoot —
and the latency it adds is appended to the file and reported, never hidden.

**One setpoint, two effects.** `--setpoint -3` trims the whole file by 3 dB *and* caps it at
−3 dBTP. A constant gain does the level work at no cost and bends nothing; the limiter is
left with only the peaks that survive the trim. `--gain` and `--ceiling` set the two halves
separately, and `--setpoint` refuses to be combined with either, because a command line with
two opinions about one level is not a command line. On `810.mka`, whose true peak is
+0.23 dBTP:

| | max reduction | limiter engaged | delivered TP |
|---|---|---|---|
| `--ceiling -3` alone | 3.85 dB | 99.6 % of frames | −3.61 dBTP |
| `--setpoint -3` | 0.85 dB | 17.1 % of frames | −3.62 dBTP |

That table is the argument. A ceiling asked to do level work holds the file by riding gain
through nearly all of it — and that 99.6 % is the processor genuinely holding a reduction of
0.75 dB through the file, not a stuck counter; the same file trimmed by a setpoint is touched
on 17.1 % of its frames instead. (`engaged_frames` used to latch: a one-pole release
converges to within 2.1e-11 dB of unity and then stops in IEEE 754, so `gain < 1.0` stayed
true for the rest of the file after the first peak. The follower is now told to arrive, and
the counters ask for a reduction of at least 0.001 dB. Both numbers above are post-fix
measurements; the 87 % this table first claimed for the setpoint row was the latch, not the
limiter.) ffmpeg's own peak meter agrees on the delivered file (−3.4 dBFS against our
−3.62 dBTP @x4, both under the −3 ceiling).

**Defaults**: limiter on, −3 dBTP, 2 ms lookahead, 60/900 ms, 100 %/75 % linking, 8x on a 4x
meter. On is a deliberate change of default behaviour — a file leaving here above 0 dBTP
clips somewhere downstream, and a ceiling that nothing enforces is a claim. The off path is
the contract, and it is proven rather than asserted: `--no-limit` on `810.mka` still writes
`0c3d30cdaff5dffc5db24360b353bca9d1a8393a664a1e13a085540434a2eb9a`.

**Two hashes, two scopes**, both from `--sha256`: `sha256` is the sample stream (container
independent, so FLAC and WAV of the same render agree), `file_sha256` is the file as
`sha256sum` sees it. The old field is neither of the two and does not survive a container
change.

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
than a stale binary in `target/` — an out-of-date verifier is worse than none. Every
check also runs per-group (`--group bitexact` …), so a failure can be reproduced
without waiting for the whole suite.

Two claims beyond the check itself are worth stating, because both are cheap to assert
falsely:

* **The verifier has teeth.** Changing one coefficient in `dsp.rs` by one ULP turns 33
  `bitexact` checks red with `max |d| = 2.98e-8` — the kind of difference no listening
  test would ever find.
* **Cross-platform, not just cross-build.** The `x86_64-pc-windows-gnu` binaries
  produced by the release script render a TrueHD/Atmos `.mka` to a FLAC with the same
  SHA-256 as the Linux build — checked by running the Windows `g6me.exe` under Wine
  against a *different* ffmpeg build (`n9.0.2-48-gd4070a14ce` vs `n9.0.2`), so the
  agreement covers the decoder and encoder choices too, not just the DSP.

## Releases

Each tag publishes, from one commit and one `Cargo.lock`:

| file | what it is |
|---|---|
| `G6_Media_Encoder-<ver>-x86_64.AppImage` | Linux, one file, no install. Runs on Wayland and X11; contains `g6me-gui` and `g6me` (`G6ME_BIN=g6me ./G6_Media_Encoder-*.AppImage --caps` runs the CLI). |
| `g6me-<ver>-linux-x86_64.tar.gz` | the same two binaries unpacked, for a headless box or a server where FUSE is not available. |
| `g6me-<ver>-linux-x86_64-desktop.tar.gz` | `.desktop` entry and icon, if you want the AppImage in an application menu. |
| `g6me-<ver>-windows-x86_64.zip` | `g6me.exe` and `g6me-gui.exe`, built for x86-64-v3. |
| `sha256sums.txt` | checksums of all of the above. |

**ffmpeg is not bundled** — it is a separate GPL program and any desktop that plays
audio already has it; put it on `PATH`, or point `G6ME_FFMPEG` / `G6ME_FFPROBE` at a
specific build. `g6me --caps` says what your ffmpeg can actually do, including a
measured 24-bit FLAC round-trip rather than a guess from the version string. The
binaries are built for
**x86-64-v3** (AVX2, i.e. Haswell/Excavator 2013 and later); on older hardware build
from source without `--target-cpu`.

### Are those artefacts reproducible?

Within one toolchain, yes — measured, not asserted. The same commit built from two
different checkout paths now yields **byte-identical** files for all five, and
`sha256sums.txt` is the evidence. Four things were in the way, found in this order:

- a Rust release binary keeps the source path of every panic message in `.rodata`, and
  `strip --strip-all` does not remove it → `--remap-path-prefix`
- `tar` and `zip` record mtimes and ownership, `gzip` records a timestamp → fixed
  mtime/owner/order, `gzip -n`, zip written with a fixed stamp
- the AppImage's squashfs records the build time → `SOURCE_DATE_EPOCH=0`
- GNU `ld` writes the link time into the PE `TimeDateStamp`, and GNU `strip` then writes
  the *current* clock there even after you asked `ld` not to → both it and the image
  checksum covering it are zeroed after stripping (0 means "not computed", which an
  application may say). The exe still loads and still renders a bit-identical FLAC,
  checked under Wine.

Across **different distributions**, no — and it is worth being exact about why: the final
link uses the distro's `ld` and CRT objects (for the Windows target, its mingw binutils),
so Arch and GitHub's Ubuntu runner produce different bytes from identical source.
Same distro plus same tag gives the same file; that is the honest claim, and closing the
rest means a pinned container rather than another compiler flag.

What *is* guaranteed across machines is the thing that matters here: the **audio**.
`810.mka` rendered by the Windows binary under Wine against a BtbN ffmpeg, and by the
Linux binary against a distro ffmpeg, produced SHA-256
`0c3d30cdaff5dffc5db24360b353bca9d1a8393a664a1e13a085540434a2eb9a` both times — the same
file, from different builds, on different platforms.

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
packaging/linux      desktop entry, icon, AppRun for the AppImage
scripts/             build-optimized.sh (verified v3 build), release.sh (artefacts)
```

## Known gaps

* **The headline dBTP is the stream; the file's own number is `delivered`.** The
  true-peak stage sees the signal before the encoder, and things still stand between
  that point and the bytes: output resampling and output dither run inside ffmpeg,
  downstream of the meter, and an encoder clamps to its own sample format. So every
  render now decodes the finished file and reports it as `delivered` next to the
  metered numbers, with a note when the two disagree by more than 0.05 dB. On real
  material: a hot 7.1 render metered **+0.233 dBTP** arrived as an s16 WAV at
  **+0.084 dBTP** with sample peaks parked at −0.0003 dBFS — the encoder clamped, and
  the report says so instead of quoting a crest the file cannot hold. On the default
  lossless path the two agree to four decimal places. The cost is one extra decode
  pass per file: an 8 s render becomes about 14 s. (`-r/--rate` is input-side, before
  the downmix, so it is not what this is about; the output-side resampler is.)
* **DC is metered, and removed only on request.** Every render reports `dc_offset`
  per channel — in full-scale units, because an offset carries a sign and is compared
  against a quantisation step — for the stream and again for the finished file decoded
  back. The downmixer's own contribution is measured rather than assumed: its all-pass
  cascade has `H(1) = (b0+b1+1)/(1+a1+a2) = 1` exactly, since each section's numerator is
  its denominator reversed, so it carries the vendor module's anti-denormal bias to the
  output at unity — a settled `+4 * EPSILON` on the left (+4.0e-10, −188 dBFS, 0.0034 of
  one 24-bit step) and exactly zero on the right, where the front and surround branches
  difference out. That is left alone deliberately: erasing it would mean editing the
  bit-exact port for an offset four orders of magnitude below one LSB.

  What is worth catching arrives from outside, and on real material it arrives in
  quantity. `810.mka` measures **+1.89e-4 / +1.80e-4** at the output — −74 dBFS, about
  1590 LSBs of 24-bit — and ffmpeg's `astats` reads +1.94e-4 / +1.76e-4 on the source's
  own front pair, so the offset belongs to the recording, not to the port. The 7.1 matrix
  is why it matters: eight channels sum into two, carrying input DC at **+5.179** on the
  left and **−0.351** on the right, so a quarter of DC in arrives as 1.29 out, which is
  level as far as a limiter is concerned — it will spend gain reduction holding an offset
  that never clips, and report a true peak no transient caused.

  `--dc-block` inserts a 5 Hz first-order blocker (`H(z) = g(1-z⁻¹)/(1-Rz⁻¹)`, exactly
  zero at DC, exactly unity at Nyquist) before the trim and the limiter: on that file the
  reported mean falls from 1.89e-4 to 4.9e-9. It is opt-in and the default path stays
  bit-exact, so turning it on changes every hash. What it leaves behind is the blocker's
  own settling spread over the file — which is the reason a DC figure is only meaningful
  with the window it was measured over, and why the tests here drain the limiter's
  lookahead and skip the filter's transient before they claim a number.
* **FLAC stops at 24 bits.** That is the container and the encoder, not a choice:
  ffmpeg's FLAC encoder writes `bps=24` and, fed 32-bit PCM, drops the low byte of
  every sample. G6ME therefore refuses `--format flac --depth 32` instead of shipping
  a file labelled 24-bit that looks 32-bit, and sends 32-bit int and float to WAV
  (`pcm_s32le` / `pcm_f32le`, both genuine). If an ffmpeg ever failed the 24-bit
  round-trip probe, 24-bit FLAC is refused on that machine too and `--caps` says why.
* **No macOS CI**, although the mount-list code has a macOS path and eframe runs
  there.
* `--monitor` (CLI only) is deliberately unverified on Windows.

## Provenance and license

MIT — see [LICENSE](LICENSE). The DSP is a port of the G6 Creative MatrixEncoder
analysis; the C reference under `vendor/g6ct/` is kept byte-identical to its source
(`vendor/g6ct/UPSTREAM.md` records which) so that "we match the reference" stays a
checkable claim rather than a memory.
