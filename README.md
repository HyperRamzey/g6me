# G6ME — G6 Media Encoder

> **Disclaimer:** This is an independent research and interoperability project. It is not
> affiliated with, endorsed by, or sponsored by Creative Technology Ltd. Sound Blaster and
> Sound BlasterX are trademarks of Creative Technology Ltd. No proprietary firmware or
> driver binaries are distributed in this repository.

Batch renderer for the **G6 Creative MatrixEncoder**, and for **G6+**, a universal
non-personalized binaural renderer that ships alongside it. Give it 7.1 / 5.1 /
Dolby-Atmos-12 (overheads are nulled for the reference engine — use DME\EE first)
and it writes a 2-channel binaural file as FLAC or WAV, from a command line or a
GUI, on Linux and Windows.

It exists because the plugin itself is a Windows VST2 insert. Real libraries are
files on a disk, and you want the whole library converted overnight with the same
numbers — including `.mka` containers holding Dolby TrueHD + Atmos, which the
plugin can only play in real time through a host.

```text
multichannel file ──ffmpeg decode──► f64 samples ──► renderer (f64) ──► trim / DC / limiter ──► FLAC/WAV
 .mka .wav .flac .mkv .mp4 …        (optional soxr      Reference: exact        (the delivered      24-bit FLAC,
                                     resample, IR)       plugin matrix        ceiling, measured    or any depth
                                                         Plus: G6+ binaural   and reported)       in WAV)
```

## Roadmap — 2026-11-28 (in progress)

A verification-first program toward full CPU/GPU feature parity and paper-literal
conformance. Every item lands with a sabotage test that fails on the unfixed or wrong
implementation. The working checklist is `docs/handoffs/roadmap-temp-2026-11-28.md`.

**Doctrine.** Production CPU and GPU implement the same feature natively on different
hardware: same algorithm and operation order, strict IEEE, zero FMA. Integer stages are
bit-identical; float stages are gated at the smallest measured difference. A CUDA stage is
never "missing" and never satisfied by a silent host fallback; a GPU-only feature gets a
bit-exact CPU counterpart. A paper's name appears only on an implementation of that paper.

**Wave 1 — correctness (complete).** Room late-tail per-band RT60 (envelope floor,
single ISO 9613-1 air absorption, PCHIP cross-band interpolation), Bruce–Erfani–Zilany
(BEZ2018a) auditory periphery (tail padding held at resting drive, 31-sample decimator
alignment, SPL calibration against the C reference, integer redock clock; PLA/chirp gains
confirmed correct), ISO 3382-1 LF denominator + IEC 61260 class-1 filterbank (energy
0.99963, band-sum error 1e-11), Baumgartner–Majdak 7-cue externalization, DNG float/GainMap
edge cases, remesh budget reporting wired through `g6me-scan` (`--remesh-max-faces`,
`target_met` warning), spherical near-field HRTF series (r_hi cutoff, scaled Miller J_m),
UPOLA single-inverse and rollback atomicity, and the CUDA E2E stage gate.

**Wave 1.5 — integration (in progress).** Head-mesh/room junction: one shared 1 ms causal
pre-delay (the old `n_fft/8` delay put direct sound 42–85 ms behind the room), ear-canal
entrance instead of helix rim, phase-unwrapped anchor interpolation, and room reflections
decoded through the BEM head rather than the snowman. Plus: verifier gate requiring every
exercised CUDA stage, SOFA AES69 / PLY I/O, adaptive scan reconstruction (card detector,
dense burst depth, high-resolution TSDF), and optimal-Q Ozaki-II.

**Wave 2 — measured room validation.** Lundeby cross-point truncation and noise handling,
Tier-A shoebox presets in CLI/GUI, OpenAIR/dEchorate gates, passive frequency-dependent FDTD
boundaries (Kowalczyk–van Walstijn; Bilbao–Hamilton), colorless FDN matrices (Dal Santo et
al.) and per-band GFDN absorption (Das et al. 2025).

**Wave 3 — binaural and personalisation.** MagLS phase continuation (Hold, Meyer-Kahlen &
Pulkki 2023), bilateral MagLS, diffuse-field equalisation with regularised complex inversion,
reverberant externalisation (Li et al. 2020) and full-sphere localisation (Barumerli et al.
2023), SOFA AES69 I/O with Mesh2HRTF/NumCalc interop, minimum-phase HRIR synthesis from
HyprEQ's Hilbert-cepstral generator, and Møller et al. (2007) phase-matched stereo for
per-ear HPEQ.

**Wave 4 — compute.** ETH Zürich optimal-Q Ozaki-II modulus schemes (FP4 59 / FP8 30 GEMMs,
INT8 moduli ≤ 239), CUDA perf plan §§2–5 (batched FFT + CUDA Graph, FDTD temporal blocking,
GPU-resident GMRES).

**Wave 5 — scan completeness.** Tiled 12–50 MP depth, configurable remesh budget, optional
AprilTag3 (tag36h11) payload decoding.

No new equalisation features are introduced: HyprEQ primitives (PCHIP, complex inversion,
IEC 61260, Neumaier summation, minimum-phase FIR) are ported only where a paper's algorithm
requires them.

## Two engines, one pipeline

| | **Reference** (default) | **Plus** (`--mode plus`) |
|---|---|---|
| What it is | A bit-exact port of the G6 `CMatrixEncoderEfxMod` matrix: the same all-pass cascade, the same gains, the same `f64` state. | A universal, non-personalized binaural renderer: physical head shadow and ITD, causal bass-mono, pinna shadow, diffuse-field equalization, BS.1116 lateral reflections, DEE-safe decorrelation. |
| Sounds like | Direct and punchy; sharper surround phase contrast, no HRTF or room shaping. | Speakers in front of you: stable out-of-head width, natural interaural time and level, neutral 1-ERB tonality that stacks cleanly under headphone EQ. |
| The claim | Bit-identical to binaries compiled from the vendored C reference (`g6me-verify`, 129 checks). | Designed, not cloned: every invariant it promises is a test in `plus.rs` (flat diffuse field, Woodworth ITD, bass coherence, symmetry, no pre-ringing, chunk-size invariance). |
| Verified arithmetic | `f64`, no FMA, no fast-math — proven by disassembly. | Same `f64` contract, zero FMA, deterministic. |
| Unchanged by it | Every existing command line, GUI preset and verifier contract. | Opt-in. The default path never enters the plus code. |

You choose per render, in the CLI with `--mode`, in the GUI with the **Renderer** tab.
Both engines run through the same decoder, meters, limiter, encoder and batch
engine, so the two paths cannot disagree about a file's level, its format or its
provenance.

## What is guaranteed — Reference engine

| | |
|---|---|
| **Bit-exact to the reference** | The DSP is a port of `g6ct.c` (vendored under `vendor/g6ct/`) and is compared against binaries compiled from that exact C source by `g6me-verify`: 129 checks, all passing. |
| **f64 internals** | The downmixer runs in `f64` — the same precision as the reference's `-DG6CT_DOUBLE` build and mpv's `afir … precision=double`. Not configurable, on purpose: state precision is not a taste parameter. |
| **No FMA, no fast-math, ever** | Contraction changes IIR state, and IIR state is phase. `g6me-verify` disassembles the shipped DSP and the reference builds and fails if a fused multiply-add appears; it also builds a deliberate `-ffp-contract=fast` counter-example and requires that it *does* diverge, so the guard is known to have teeth. |
| **Resampleless by default** | Nothing touches the sample rate, bit depth or timing unless you ask. The default path is decode → f64 → downmix → encode. |
| **Deterministic** | Same input, same flags, same bytes. Repeat a run with `--sha256` and compare. |
| **A render that does not finish leaves nothing** | Output is written to a sibling of the destination and renamed into place as the last step, so a cancelled or failed render never creates the name you asked for. Writing the destination directly meant every failure had to be undone by a delete, and a delete can lose a race with the operating system: a cancelled batch on Windows once left a 46-byte FLAC — a header, no audio — in an album directory that looked fine on Linux. |
| **No half-written output** | A cancelled or failed file leaves no playable partial file behind. |
| **DSP is single-threaded per file** | One contiguous block sequence through the filters. Parallelism is *between* files; decode / DSP / encode overlap per file through bounded queues, so the memory ceiling is explicit (`--queue`). |

What is *not* guaranteed: that a downmix that clips on your headphones is
pleasant. The matrix is the plugin's, gains and all.

## G6+ — the Plus renderer

`--mode plus` (GUI: **Renderer → Plus**) replaces the hardware matrix with a
binaural renderer designed for 48 kHz surround beds, with no ear scans, no
measured HRTF and no per-headphone profile. It is built from published acoustics
rather than sampled from a head:

1. **Brown–Duda (1998) spherical head shadow** — frequency-dependent diffraction
   ILD for an 8.75 cm head at 343 m/s, converted to strictly causal minimum phase
   by Hilbert-cepstral folding.
2. **Kulkarni–Isabelle–Colburn (1999) excess Woodworth ITD** — measures the ITD the
   shadow filter already carries and adds only the remaining time-of-flight, so
   total ITD matches Woodworth (FL ≈ 261 µs, SL ≈ 607 µs, BL ≈ 322 µs) without
   counting the same delay twice.
3. **Causal LR4 bass-mono crossover** at 120 Hz by default — keeps sub-bass
   phase-coherent across closed-backs, open-backs, planars and electrostatics,
   with zero pre-ringing.
4. **Universal front–back pinna shadow** — a gentle rear HF shelf and a subtle
   1.2 kHz Blauert directional band, with no sharp individualized notches.
5. **DEE-safe side/back decorrelation** — a first-order causal all-pass on
   `BL`/`BR` above the bass-mono frequency, so folding `.4` Atmos heights into
   `SL+BL`/`SR+BR` cannot phase-lock into a coherent `+4.4 dB` pileup. Individual
   magnitude coloration is 0.00 dB and ITD is not disturbed.
6. **Common 1-ERB diffuse-field equalization** (Zaunschirm, Schörkhuber & Höldrich,
   JASA 2018) on the direct path — the 7.0 ring is flat to within ≤ ±0.20 dB
   (1-ERB Glasberg–Moore excitation) independent of lateral reflections, then
   normalized to the reference's uncorrelated per-ear diffuse power (`3.500`,
   +5.44 dB). This is why a headphone EQ curve measured on top of G6+ stays valid.
7. **ITU-R BS.1116 lateral early reflections** — smooth alternating polarity,
   ≥ 8.5 ms, nothing inside 6 ms, with a Hassager-smoothed diffuse tail. The
   `externalize` control moves the image out of the head without the comb-filter
   bounce that a bare reflection produces.
8. **Adaptive 1-ERB phantom-center de-comb** (Shirley–Kendrick–Churchill, JAES
   2007) — measures the crosstalk comb deficit of each frontal pair above 450 Hz
   and fills it in minimum phase, so centered vocals and snare stay open.
9. **Bauer / Linkwitz / Völk–Fastl coherent LF trim** — a user-tunable 340 Hz
   first-order causal shelf on full-range channels, compensating the +3…+4 dB of
   in-phase low-frequency summation that correlated music and surround mixes can
   build up against the reference. Discrete `LFE` is left untouched unless asked.
10. **LFE and streaming overlap-add** — optional 120 Hz 4th-order Butterworth LPF
    on `.1`, aligned to the causal pre-delay and summed mono at `1/√2` into both
    ears. Synthesis is `N_IR = 2048`, `N_FFT = 8192`, `HOP = 6144`, deterministic
    `f64` with no FMA; kernels scale with the sample rate so the impulse stays
    ~42.7 ms at 96, 192 and 384 kHz too.
11. **Hankel–MagLS physical scattering** (`--plus-physics hankel-magls`) — exact 3D
    spherical-Hankel head scattering with a coupled snowman torso and order-7 HOA
    MagLS, alongside the fast Brown–Duda default. Head radius, torso spacing, pinna
    shelf and Webster ear-canal length are explicit controls.
12. **Hybrid 3D room** (`--room anechoic|control-room|theater`, `--room-quality preview|production`, `--room-json`, `--cuda`, `--cuda-precision`) —
    FDTD + image-source + boundary-transmission + parallel deterministic PCG32 ray-traced late field,
    convolved via zero-latency UPOLA (341.3 ms Control Room, 682.7 ms Theater Stage at 48 kHz) with
    SHA-256 kernel caching, and temperature, humidity, CO₂ and pressure driving Cramer (1993) speed of
    sound and ISO 9613-1 air absorption. A JSON room file sets dimensions, wall absorption, listener
    position and desk baffles.
13. **Exact convolution** (`--exact-ntt`) — the binaural FIR runs in the 64-bit
    Goldilocks field (`p = 2^64 − 2^32 + 1`) by NTT, so intermediate rounding is zero
    ULP rather than merely small.
14. **2.0 → 7.1 PAE upmix** (`--upmix`) — He PCA+ASE primary/ambient extraction with
    APES and Kraft–Zölzer decorrelators; centre lock, surround and rear weights, LR4
    LFE extraction and a decorrelation blend are separate controls.
15. **STN virtual bass and micro-detail** (`--vbass`, `--detail`) — zero-IMD
    psychoacoustic sub-bass synthesis (Moliner–Välimäki / Mu–Gan–Tan /
    Oo–Gan–Hawksford) and a Sines+Transients-only presence enhancer that leaves noise
    untouched.
16. **Perceptual unmasking** (`--unmask`) — Moore 2016 / ISO 532-2 auditory-model
    dynamic EQ as a minimum-phase FIR: recover masked ERB bands, tame dominant
    maskers, with an equal-information spectral tilt.
17. **Punchy limiter** (`--limit-style punchy`, `--limit-clip-db`) — optional
    Signalsmith C¹ two-stage boxcar lookahead with a log-domain slow stage and an
    ADAA2 C² soft clipper, while `transparent` remains the bit-exact default.

Three further toggles replace the default DSP primitives: `--plus-slepian` (exact
Slepian DPSS Heisenberg-optimal tail window), `--plus-fdn` (orthogonal-scattering
FDN after the Schlecht–Habets lossless prototype + ternary velvet-noise late
externalization tail) and `--plus-warped-decorr`
(Kermit–Canfield & Abel Bark-warped all-pass surround decorrelation).

### Profiles

| profile | width | focus | surround | externalize | bass mono | LF trim | de-comb |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `pure` | 1.00 | 0.70 | 1.00 | 0.00 | 120 Hz | 0.0 dB | 0.00 |
| `studio` (default) | 1.00 | 0.65 | 1.00 | 0.45 | 120 Hz | −1.5 dB | 0.55 |
| `cinema` | 1.18 | 0.55 | 1.20 | 0.75 | 110 Hz | −2.1 dB | 0.75 |

`pure` is the anechoic speaker sphere: no room, strict 0 dB LF balance, no
de-comb. `studio` is a balanced nearfield control room. `cinema` is a wider
theatrical stage with richer surrounds and a deeper room. Move any slider and the
GUI reports the profile as **Custom**; the CLI can override any single value
without leaving its profile.

Live controls, CLI and GUI alike: stage width (0.50–1.50), center focus
(0.00–1.00), surround depth (0.50–1.50), externalization (0.00–1.00), bass-mono
crossover (0–250 Hz), LF balance (−6.0…+3.0 dB), phantom de-comb (0.00–1.00),
discrete LFE level (−12…+12 dB), LFE low-pass on/off, and whether the LF shelf
also applies to `LFE`. The physical model adds the Brown–Duda/Hankel–MagLS choice,
speaker distance (0.25–10 m), head radius (65–115 mm), snowman torso spacing
(200–380 mm), pinna shelf (−4…+4 dB) and Webster ear-canal length (15–38 mm); the
room panel selects the anechoic/control-room/theater presets or a JSON room and
sets temperature, humidity, CO₂ and pressure. The mastering switches bring the PAE
upmix, STN virtual bass, micro-detail, perceptual unmasking, exact Goldilocks NTT
convolution and the transparent/punchy limiter into the same render.

### What Plus changes about timing

G6+ is a causal FIR renderer: a 1 ms pre-delay is built into the kernel, blocks
are convolved and overlap-added without buffering a block, and at end of file the
remaining 2047-sample impulse tail is flushed through the limiter and encoded
rather than discarded. At 48 kHz that is +42.7 ms of file length; at 96/192/384 kHz
the kernel scales and the tail roughly doubles/quadruples. The output is
deterministic and the file length is the honest consequence of the filter, not a
mystery.

## Requirements

* **ffmpeg** with `libsoxr`, `afir` and `lavfi` — checked at runtime, not assumed.
  Developed and tested against `n9.0.2`. Without libsoxr the program refuses to
  resample rather than silently picking another resampler; the default
  (no resample) path still works.
* Nothing else for output. ffmpeg's own FLAC encoder writes genuine 16- and 24-bit
  FLAC — `g6me --caps` proves it on your build by encoding a known 24-bit block and
  decoding it back — and FLAC carries integers only, so 32-bit and float go to WAV.
  Compression levels are ffmpeg's own scale, 0 (fastest) to 12 (smallest).
* **Rust 1.100+** to build. A C compiler only for
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
cargo build --release            # g6me, g6me-scan and g6me-gui in target/release/
cargo test --workspace           # 374 tests (289 core, 25 CLI, 43 GUI, 4 CUDA, 13 harness)
cargo run -p g6me-verify         # the bit-exactness evidence (129 checks)
scripts/build-install.sh         # optimized x86-64-v3 build, verified, installed
```

## Command line

```sh
# the reference matrix, a whole library, FLAC level 8, four files at a time
g6me -o /music/binaural -j 4 /music/atmos/

# G6+ Studio, the universal binaural renderer
g6me --mode plus -o /music/binaural /music/atmos/

# G6+ Cinema, wider stage and deeper surrounds, with the headphone calibration IR
g6me --mode plus --plus-profile cinema --ir ~/.config/mpv/ir.wav album.mka

# G6+ tuned by hand: wider front, dry center, +6 dB discrete LFE, no LFE low-pass
g6me --mode plus --plus-width 1.25 --plus-focus 0.80 \
     --plus-lfe-gain 6 --no-lfe-lpf out.mka

# one album to 16-bit FLAC, resampled first, with the reference matrix
g6me -o out --rate 48000 --depth s16 --dither low_shibata album.mka

# exactly what would run, without running it
g6me --dry-run -o out --mode plus --rate 44100 concert.mka
```

Inputs are files **and folders**, as many as you like; folders are scanned
recursively and the tree is mirrored under `-o`. Give it **two or more** folders and each
one keeps its own folder name under `-o`, because mirroring both against their own root puts
`AlbumA/01.mka` and `AlbumB/01.mka` on the same `out/01.flac` — and that collision is silent,
since the second file is skipped as already rendered. A folder reached twice, or a symlink
pointing back at an ancestor, is visited once. Without `-o`, each output lands
next to its input. Existing outputs are skipped, not overwritten (`-y` to force —
and `--force` will still never truncate an input file).

Any container ffmpeg can decode is accepted at render time; the folder scanner
offers `mka mkv webm m4a m4b mp4 mov m4v flac wav wave rf64 aiff aif aifc afc w64
caf wma ape mlp thd tta wv tak oga ogg opus spx ac3 eac3 dts dtshd dff dsf aac mp3
mp2 m4r amr 3gp 3g2 voc au snd raw`, and skips obvious non-audio sidecars. Multi-channel
streams may be 5.1, 7.1 or Atmos-12; a stream with no front pair is refused rather
than folded.

### The switches that matter most, and why

| flag | why it exists |
|---|---|
| `--mode reference \| plus` | Which renderer. `reference` is the default and the bit-exact path; `plus` selects G6+. |
| `--plus-profile pure \| studio \| cinema` | G6+ acoustic preset. `studio` is the default. |
| `--plus-width`, `--plus-focus`, `--plus-surround`, `--plus-externalize` | G6+ soundfield geometry: stage width, center dryness, surround depth, lateral early reflections. |
| `--plus-bass-mono`, `--plus-bass-trim`, `--plus-decomb` | G6+ low end and center: LR4 bass-mono crossover, 340 Hz coherent-LF shelf, phantom-center fill. |
| `--plus-lfe-gain`, `--plus-lfe-lf-trim`, `--no-lfe-lpf` | G6+ `.1` handling: discrete LFE level, whether the LF shelf reaches LFE, and the 120 Hz LFE low-pass. |
| `-r, --rate` | Rate conversion happens **before** the renderer, through libsoxr, and only when you pass it. Doing it afterwards would make the resampler part of the matrix's phase response. |
| `--depth`, `--dither` | Depth reduction is a separate, explicit decision from rate conversion. `--dither wannamaker \| low_shibata \| triangular \| none`, plus the full soxr dither set (`wannamaker` runs an in-app 9-tap psychoacoustic noise shaper for 16/24-bit output). |
| `--ir` | Mono headphone-calibration impulse convolved into every input channel first, exactly the way mpv's `afir` does (`irnorm=-1`, `precision=double`), so the calibration you already use for listening is the one applied here. |
| `--order FL,FR,FC,LFE,BL,BR,SL,SR` | Some containers mislabel their channel layout. This overrides it instead of guessing. |
| `--queue` | The memory bound: `queue × block × channels × 8` bytes per file. |
| `--sha256` | Prints a hash per output, so "deterministic" is something you can check rather than believe. |
| `--monitor`, `--device` | Play the result through an output device instead of writing a file (CLI only; the GUI has no audio path at all). |
| `--upmix`, `--upmix-center`, `--upmix-surround`, `--upmix-rear`, `--upmix-no-lfe`, `--upmix-lfe-hz`, `--upmix-decorr` | 2.0 → 7.1 Primary-Ambient Extraction before either renderer. |
| `--vbass`, `--vbass-cutoff`, `--vbass-amount`, `--vbass-gain`, `--vbass-decay`, `--vbass-punch` | Zero-IMD psychoacoustic sub-bass synthesis. |
| `--detail`, `--detail-intensity`, `--detail-bias`, `--detail-transients`, `--detail-tilt` | Sines+Transients micro-detail and presence enhancement. |
| `--unmask`, `--unmask-recover`, `--unmask-tame`, `--unmask-max-db`, `--unmask-bias`, `--unmask-brighten` | Moore 2016 / ISO 532-2 perceptual unmasking dynamic EQ. |
| `--plus-physics`, `--plus-distance`, `--plus-head-radius`, `--plus-torso-height`, `--plus-pinna-shelf`, `--ear-canal-mm` | Physical head/torso scattering model and near-field parallax. |
| `--room`, `--room-quality`, `--room-json`, `--room-temp-c`, `--room-rh`, `--co2-ppm`, `--room-pressure-kpa`, `--cuda`, `--cuda-precision` | Hybrid 3D room (`preview`: 500 Hz control-room / 275 Hz theater FDTD + 2,048 rays; `production`: 670 Hz control-room / 437 Hz theater FDTD + 50k rays), ISO 9613-1/Cramer air absorption, and optional `--features cuda` offload of FDTD, ray tracing, BEM matvec, Stockham FFT, Goldilocks NTT and Ozaki-II CRT GEMM (`auto|nvfp4|fp8|int8|fp16`). |
| `--head-mesh` | OBJ head/pinna scan for CHIEF-regularized FMM-BEM personalised HRTF synthesis. |
| `--exact-ntt` | Goldilocks-field NTT convolution with zero intermediate rounding. |
| `--plus-slepian`, `--plus-fdn`, `--plus-warped-decorr` | Exact Slepian window, scattering FDN late tail, Bark-warped surround decorrelation. |
| `--limit-style`, `--limit-clip-db` | Transparent or punchy limiter, plus ADAA2 soft-clipper drive. |
| `--assess` | Non-intrusive Klippel + King–Agerkvist electro-mechanical-thermal transducer assessment (`none`, `headphone-dynamic`, `headphone-planar`, `studio-monitor`). |

Full help: `g6me --help`. `--json` emits one object per finished file for scripts,
including the renderer, the true peak with its ratio, the limiter report, the
delivered file measurement, DC offset and the full BS.1770-5 loudness block.

## Acoustic scan — `g6me-scan`

`g6me-scan` is the measurement side of the project: it turns a smartphone capture
into a watertight, FMM-BEM-ready head/pinna mesh and hands it to `--head-mesh`.
Four stages, all in memory:

1. **DNG 1.7 ingest** — uncompressed, Lossless-JPEG SOF3 (ITU-T T.81, predictors
   1–7), and JPEG-XL (compression tag `52546`, via `jxl-oxide`, including
   Row/Column interleave) strips and tiles decode natively; v7.0 zero-copy 50 MP
   MIPI RAW10 bursts with OIS/IMU sidecars are unpacked; OpcodeList3
   WarpRectilinear and FixVignetteRadial are applied, and `--dng` prints the
   Honesty Report.
2. **Metric anchor** — Russian ID-1 card (horizontal or vertical, GOST R ISO/IEC
   7810-2006), or a square fiducial board (AprilTag3 tag36h11 / ChArUco-style outer
   quad; the detector fits the quad only and decodes no tag ID). The detected anchor
   fixes metres per pixel, so the reconstruction is metric rather than scale-free.
3. **Surfel photometric refinement (2DGS/PGSR-inspired) → TSDF** — oriented surfels
   are refined by coordinate descent on multi-view photometric variance (not the
   papers' differentiable rasterizer), fused into a TSDF, and extracted with
   crack-free marching tetrahedra; bundle adjustment runs in Ozaki-II FP4→FP64.
4. **Watertight acoustic conditioning** — generalized winding numbers (Jacobson et
   al., SIGGRAPH 2013) close the surface, isotropic remeshing enforces
   `h <= lambda_min / 6`, and the mesh goes directly into the CHIEF-regularized
   FMM-BEM Helmholtz solver without touching disk.

```sh
# inspect a DNG and print the Honesty Report
g6me-scan --dng capture.dng

# metric multi-view reconstruction from a RAW10 burst
g6me-scan --burst burst/ --anchor russian-card --max-edge-mm 2.5 -o head.obj
```

`--head-mesh head.obj` then uses that mesh for personalised FMM-BEM HRTF synthesis
in the renderer. The optional `crates/g6me-cuda` crate (`--features cuda`) ships 6
standalone `.cu` kernels (`fdtd_3d.cu`, `raytrace_hoa.cu`, `stockham_fft_f64.cu`,
`goldilocks_ntt.cu`, `bem_matvec.cu`, `ozaki2_blackwell_mma.cu`) compiled with
`nvcc --fmad=false`. `g6me-verify --group cuda` audits every kernel's PTX for 0
contracted `fma.rn` and its `cuobjdump --dump-sass` hardware SASS for 0 fused
`FFMA`/`HFMA2`/`DMAD`/`FMAD` (the IEEE `div.rn`/`sqrt.rn` MUFU+DFMA microcode in
`raytrace_hoa` is one correctly-rounded operation, not a contraction), runs a
mandatory `--fmad=true` sabotage negative control, and recompiles the Ozaki-II
kernel for every architecture the installed toolkit accepts (`sm_120a`, `sm_90`,
`sm_89`, `sm_80`, `sm_75` here, plus `sm_61` where nvcc still targets Pascal),
requiring the Tensor Core / DP4A opcode each target should emit (`OMMA.SF` NVFP4,
`QMMA` FP8, `IMMA`, `HMMA`, `IDP.4A`). On a live NVIDIA GPU the same group runs CPU-vs-GPU
parity for all six kernels — 3D FDTD at toy and production grid sizes, Ozaki-II CRT
GEMM across every supported precision (bit-exact, integer CRT reconstruction asserted
per cell), Stockham FFT, bit-exact Goldilocks NTT, BEM complex matvec — plus
production-path parity: the full `raytrace_hoa` HOA histogram with Lambertian wall
scattering is bit-identical bin for bin through the direct kernel entry point (not the
fallback wrapper), and the FMM-BEM reciprocal HRTF solve through
`FmmBemSolver::new_with_cuda(_, true)` uses the GPU solid-angle diagonal and matches the
CPU solver within 1e-10. A full `g6me --mode plus --room control-room --cuda` render is
**tolerance-level, not bit-identical**: GPU room synthesis changes the BRIR (measured up
to 1.4e-4 in the rendered output, from the different FDTD summation order), while the
UPOLA Stockham FFT stage agrees to ~2e-14 and the exact-NTT convolution path is
bit-identical. Each stage has two native implementations — a Rust CPU algorithm and a
standalone CUDA kernel written for the device — rather than one emulating the other, so
float parity is gated at tolerance (FDTD `max |Δp|` 1e-13, BEM matvec 1e-13, FMM-BEM
1e-10, Stockham FFT 1e-11, end-to-end render 1e-3) and bit-exact gates are kept only for
the integer-exact Goldilocks NTT and Ozaki-II CRT paths. The raytrace histogram is the
current exception: its CPU fallback is a line-for-line mirror of the kernel, so the
bin-for-bin equality is by construction, not independent agreement; reclassification to
a native CPU tracer (whose gate then becomes tolerance-level) is pending. Every run now
prints one `[cuda] stage …` line per stage, so a CPU fallback cannot hide behind the
CUDA banner. On a host with no GPU the
group reports that instead of failing. `--cuda` offloads all six paths when an NVIDIA GPU is present; the FDTD
grids stay resident in VRAM for the whole run (one upload, one download), and every
dispatch falls back deterministically to the pure-Rust CPU solver. `--cuda-precision
auto|nvfp4|fp8|int8|fp16` selects the Ozaki-II Tensor Core format (`auto` picks
single-limb INT8 everywhere the kernel supports, `sm_61..sm_120a`; explicit NVFP4 is
available only on `sm_120a` and otherwise downgrades to FP8/INT8). If the driver or
CUDA toolkit cannot start, `--cuda` warns once on stderr and runs the CPU pipeline
instead of failing silently. In the default `reference` mode there are no GPU stages at
all: `--cuda` now says so explicitly, and the run ends with `[cuda] GPU stages run: 0`
rather than leaving a CUDA banner to imply an offload that never happened.

Design notes: `docs/G6ME_SCAN_FMM_BEM_NTT.md` (scan, FMM-BEM, NTT),
`docs/ACADEMIC_NUKE_ACOUSTICS.md` (room, CUDA, noise shaping, transducer, and perceptual models), and
`docs/PATENT_CLEAN_MASTERING_SUITE.md` (upmix, virtual bass, detail, unmasking).

## GUI

```sh
cargo run -p g6me-gui
```

Wayland, X11 and Windows (eframe/egui). Native OS file and folder pickers through
Win32 COM, Cocoa, or the XDG Desktop Portal on Linux (no GTK C dependency), with
the built-in egui browser available when built `--no-default-features`. Files and
folders can also be dragged onto the window, and every path bar accepts a pasted
path.

The window is the render, arranged the way a mixer expects: queue left, controls
centre, meters and history along the bottom. It never opens an audio device.

* **Renderer tab.** Choose **Reference (Bit-Exact G6)** or **Plus (G6+ Binaural)**.
  In Plus mode, pick `pure` / `studio` / `cinema` and adjust width, center focus,
  surround depth, externalization, bass-mono, LF balance, phantom de-comb, LFE
  level and LFE filtering, with a live polar soundfield radar beside them and a
  readout of the resulting FL ITD, center DRR, LF balance, LFE level, de-comb
  amount and the DFEQ tolerance. The physical model exposes Brown–Duda vs
  Hankel–MagLS, distance, head radius, torso spacing, pinna shelf and ear-canal
  length; the room panel selects anechoic/control-room/theater or a JSON room and
  sets temperature, humidity, CO₂ and pressure.
* **Mastering panel.** The 2.0 → 7.1 PAE upmix, STN virtual bass, micro-detail,
  perceptual unmasking, exact Goldilocks NTT convolution and the
  transparent/punchy limiter are bound to the same core types as the CLI flags.
* **Output, True peak, Resample, IR calibration, Advanced tabs.** Same controls as
  the CLI — format, depth, level, resample, dither, IR, channel order, threads,
  block, queue, metadata, force, SHA-256, block DC — bound to the same core types,
  so the two front ends cannot drift apart.
* **Mastering presets** on the True peak tab: `Transparent (-3 dBTP)`,
  `Streaming (-1 dBTP)`, `Headroom Setpoint (-3 dB)`, `Bit-Exact (Limiter Off)`.
  One click sets ceiling, trim, DC blocking, lookahead and linking together.
* **Queue with real multi-selection.** Click, Ctrl-click to add or remove, Shift-click
  for a range, double-click to commit. Selected count is shown before you commit.
  A filename filter, status chips with live counts (`all`, `queued`, `done`,
  `failed`, `skipped`), per-row detail and removal, and an audio-only filter on the
  file browser.
* **Drive and volume shortcuts**, read from the mount table (Linux), the drive letters
  (Windows) or `/Volumes` (macOS), filtered to things worth visiting — no `/proc`,
  no `tmpfs`, no squashfs snaps, and `/run/media/<you>/…` does appear. `/boot` and `/tmp`
  are filtered by path rather than by filesystem type, because a machine is free to format
  them as ordinary vfat and ext4 — which is exactly when they stop looking like plumbing
  and start taking buttons. A path that is not listed is still typeable.
* **Sort the picker by name, modified or created.** Newest first, folders always first
  (the list is also how you walk a tree), ties broken by name, and an entry the filesystem
  would not date at the end rather than the start. "Created" is birth time where the
  filesystem records one; where it does not, the modified date is used and the picker says
  so above the list instead of showing a column of identical values.
* **Per-file meters.** Every file being rendered gets its own card: the renderer's eight
  input slots plus the two outputs, sample peak, peak-hold, decay, with the same
  0.30 s fall-off the VST2 meter used, plus a live EBU Tech 3205-E stereo phase
  correlation meter beside the output bars. Cards keep their last frame after the
  file finishes, so you can see what clipped after the batch is done, and clicking a
  card clears its holds.
* **Finished-file history.** One retained line per file with true peak and ratio,
  sample peak and RMS, integrated LUFS, momentary/short-term maxima, LRA, PLR and
  phase correlation, graded against a selectable delivery target. Copy it as text,
  CSV or JSON — the record is of the bytes written, not of the renderer's
  intention.
* **Dry run before you commit**: source codec and profile, detected layout, the channel
  order and mask actually chosen, the input stage selected, the renderer and its
  settings, the human-readable chain, and the exact ffmpeg command lines.
* **Four themes**: system, dark, light, and a true `#000000` OLED black.
* **Keyboard**: Ctrl+Enter starts, Esc cancels, Ctrl+D runs the dry run.

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
taps per phase, 80 dB stopband, a stated band edge at 0.45·fs (un-renormalised per phase),
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

* **Transient.** The lookahead stage: it sees a peak before the sample grid does and ramps
  the gain in ahead of it, so the crest is already ducked by the time the sample that caused
  it is emitted. Returns in 25 ms. Linked 100 % by default — peaks are ducked in both
  channels together, because a peak in one channel is still a peak in the file.
* **Level.** A peak-hold window with a 60 ms attack and a 900 ms release, riding under the
  transient stage: it answers level rather than isolated crests. **The window slides:** the
  peak a frame measured leaves the maximum exactly one release window after it went in, so a
  crest is let go 900 ms after it. It used to be a counter that restarted the hold at its own
  boundary, which held a peak for anywhere between 900 and 1800 ms depending on where inside
  the window it happened to land — and, at that same boundary, threw away the rest of the
  window's content and could let go early out from under a passage still decaying. The hold is
  now the maximum of exactly the last `release` ms of measured peaks, per channel, which is
  what the sentence above has always claimed; the test asserts it frame by frame against a
  rescan of the same ring. Linked 75 % by default —
  level is shared three-quarters, not wholly, because a gain move lasting a third of a second
  is long enough to pull a stereo image about.

Emission multiplies the two, and the transient stage's demand is computed against the peak
as the level stage left it, so they never spend the same dB twice.

Every number the limiter report prints — `engaged_frames`, `max_gain_reduction_db`,
`release_frames`, `release_reduction_db`, `transient_reduction_db` — is measured at
**emission**, on frames that were delivered, and never on the frame currently being fed.
During the pre-roll those two cursors are `lookahead` frames apart and the fed frames are
still the delay line's own silence: a counter that is not gated on delivery books reductions
the listener never heard — with the gate removed, a fixture with a 54-frame lookahead reported
21 frames engaged before a single delivered sample existed. A file shorter
than the lookahead now reports zero engaged frames, and the
transient column no longer fills in the scheduling pass that cannot see the output cursor.

**Where the gain aims.** The gain targets `guard_db` *below* the requested ceiling: what the
side chain's filter cannot see, plus what the gain slew can still overshoot on the way down.
At the default 8x that is 0.02 dB + 0.02 dB = **0.04 dB**, so a −3.00 dBTP ceiling measures
−3.04 dBTP and a −1.00 dBTP ceiling measures −1.04 dBTP — the ceiling is where it was asked
to be, minus a guard that is stated rather than assumed. Ask for the meter's 4x instead and
the same expression is 0.62 dB (0.60 + 0.02), and the same −3.00 ceiling lands near −3.62:
half a dB of level handed back on every file, which is what the 4x bank's 0.60 dB of
under-read on a crest between grid points costs. That is why the limiter's ratio and the
meter's are different constants, and why a `--limit-ratio 4` render prints a `guard_db` of
0.62 rather than quietly keeping the old number.

**Measured, every ratio, both ends of the lookahead.** Same file, `--ceiling -1`, and the
ceiling checked three ways: what the limiter says it left, what this program measures after
re-decoding the finished file, and what ffmpeg's own `ebur128=peak=true` reads on the file.
The last column is an independent implementation of the standard's detector, and it prints
one decimal, so it agrees or doesn't at 0.1 dB.

| ratio | `guard_db` | lookahead asked → honoured | limiter post | file re-decoded | ffmpeg |
|:-:|:-:|:-:|:-:|:-:|:-:|
| x2 | 0.32 | 0.20 → 0.60 ms | −1.320 | −1.300 | −1.3 |
| x4 | 0.62 | 0.20 → 0.21 ms | −1.620 | −1.620 | −1.5 |
| x8 | 0.04 | 0.20 → 0.60 ms | −1.040 | −1.021 | −1.0 |
| x16 | 0.04 | 0.20 → 0.60 ms | −1.040 | −1.022 | −1.0 |
| x32 | 0.04 | 0.20 → 0.60 ms | −1.040 | −1.022 | −1.0 |

and every one of them identical at `--lookahead 2`, which is the default. Two things in that
table are worth reading rather than nodding at. The honoured lookahead is not always what was
asked for: 0.2 ms is below the interpolator's own delay at x2/x8/x16/x32, so the limiter says
so and runs at 0.60 ms, while x4 — whose bank is the shortest — really can act in 0.21 ms, and
both numbers are in the report as `requested_lookahead_ms` and `effective_lookahead_ms`. And
ffmpeg reads the x4 file 0.12 dB *hotter* than this program's own x4 measurement of the same
bytes does; it is under the ceiling either way, but it is the reason a reading is quoted with
its ratio and never treated as the last word.

**Latency.** Lookahead plus the interpolator's own delay, prepended as leading zeros at
the start of the file (with the held tail flushed at the end) and reported, never hidden. At
the defaults that is 96 samples of lookahead — **2.00 ms
at 48 kHz, honoured as asked** — plus 26 samples of filter delay: **122 samples, 2.542 ms**,
so a rendered file is 122 frames longer than its source and `810.mka` goes from 4210520
frames to 4210642. A `--lookahead` shorter than the side chain's floor is raised to the floor
and says so, `requested_lookahead_ms` beside `effective_lookahead_ms`.

**One setpoint, two effects.** `--setpoint -3` trims the whole file by 3 dB *and* caps it at
−3 dBTP. A constant gain does the level work at no cost and bends nothing; the limiter is
left with only the peaks that survive the trim. `--gain` and `--ceiling` set the two halves
separately, and `--setpoint` refuses to be combined with either, because a command line with
two opinions about one level is not a command line. On `810.mka`, whose true peak is
+0.23 dBTP:

| | max reduction | limiter engaged | delivered TP |
|---|---|---|---|
| `--ceiling -3` alone | 3.30 dB | 98.2 % of frames | −3.04 dBTP |
| `--setpoint -3` | 0.28 dB | 3.3 % of frames | −3.04 dBTP |

That table is the argument. A ceiling asked to do level work holds the file by riding gain
through nearly all of it — and that 98.2 % is the processor genuinely holding a reduction of
0.48 dB through the file, not a stuck counter; the same file trimmed by a setpoint is touched
on 3.3 % of its frames instead. Both rows land at the same −3.04 because the aim is the same;
what differs is how much of the file had to be held there to get it. (`engaged_frames` used
to latch: a one-pole release
converges to within 2.1e-11 dB of unity and then stops in IEEE 754, so `gain < 1.0` stayed
true for the rest of the file after the first peak. The follower is now told to arrive, and
the counters ask for a reduction of at least 0.001 dB. Both numbers above are post-fix
measurements at the default 8x; the 87 % this table first claimed for the setpoint row was
the latch, not the limiter, and the 17.1 % before it was measured at 4x.

Re-measured here on 2026-11-20 against a binary built from the same tree with the pre-sliding
hold restored (same flags, same file, same ffmpeg, the only difference the two source files):
**on real content the port moves the report by hundredths of a dB and leaves the ceiling
claim exactly where it was.** The ceiling
row engaged on the same 4 134 988 frames of 4 210 642 — not the same percentage, the same
integer — and the setpoint row moved by 38 frames (140 148 → 140 186, +0.0009 pp). What moved
is where the slow stage's deepest reduction lands: 3.2803 → 3.3039 dB of total reduction
(+0.024 dB) and 0.5688 → 0.4751 dB of slow-stage reduction (−0.094 dB), because a hold that
slides is bound by the loudest of the last 900 ms at *every* frame instead of at a boundary
that restarted under it, so the frame it binds is a different one. Delivered true peak moved
by 4.1e-5 dB and the delivered sample peak by 4e-6 dB: the ceiling it holds is the same file.
Outside cross-check:
ffmpeg's own `astats` reads −3.040000 dBFS sample peak on the delivered file against our
−3.0397 dBTP @x8 — a sample peak just under a true peak, which is the shape of a file held
by true peak rather than by samples.)

**Defaults**: limiter on, −3 dBTP, 2 ms lookahead, 60/900 ms, 100 %/75 % linking, 8x side
chain on a 4x meter. The 8x only became true in the 0.3 line: both front ends fell back to
`Ratio::default()`, which is the *meter's* 4x, so the side chain ran a step shallower than
this document, the CLI's own help text and `Cfg::default` each claimed, `guard_db` was 0.62
not 0.04, and every limited file sat about 0.58 dB below the ceiling it was set to. The
fallback is `limiter::DEFAULT_RATIO` now, with a test in each front end asserting it.
On is a deliberate change of default behaviour — a file leaving here above 0 dBTP
clips somewhere downstream, and a ceiling that nothing enforces is a claim. The off path is
the contract, and it is proven rather than asserted: `--no-limit` on `810.mka` still writes
`0c3d30cdaff5dffc5db24360b353bca9d1a8393a664a1e13a085540434a2eb9a`.

**Two hashes, two scopes**, both from `--sha256`: `sha256` is the sample stream (container
independent, so FLAC and WAV of the same render agree), `file_sha256` is the file as
`sha256sum` sees it. The old field is neither of the two and does not survive a container
change.

## Loudness, range, PLR and stereo phase

Every render also produces a **BS.1770-5 loudness report**, measured on the stereo
stream handed to the encoder (which on the default lossless path is the finished file's
sample stream; the decoded-back file is reported separately as `delivered`, as with the
true peak):

* **Integrated loudness** (`integrated_lufs`), ITU-R BS.1770-5 / EBU R128: the two-stage
  K-weighting pre-filter, 400 ms gating blocks with 75 % overlap, a −70 LKFS absolute gate
  and a −10 LU relative gate. The biquads are designed from Brecht De Man's (AES 2018)
  continuous-time prototype, which reproduces the Recommendation's 48 kHz tables to
  `< 1e-15` and keeps 44.1/88.2/96/192 kHz inside EBU Tech 3341's ±0.1 LU.
* **Momentary and short-term maxima** (400 ms / 3 s), **Loudness Range** (`P95 − P10` over
  gated 3 s blocks, EBU Tech 3342), and **PLR** (`true peak − integrated loudness`,
  AES TD1008).
* **Stereo phase correlation** (`rho`, EBU Tech 3205-E): `+1.0` is identical mono, `0.0`
  is decorrelated stereo, negative is inverted polarity. This is the number that catches a
  mislabelled surround channel or an all-pass stage that has drifted, before it reaches a
  headphone.

Fields that are mathematically undefined — integrated loudness for a clip under 400 ms,
LRA under 3 s, silence — are reported as `null` rather than as a fabricated number, so the
JSON stays RFC 8259 clean and a downstream script can tell "quiet" from "unmeasured".

**Delivery targets.** The GUI grades the finished-file history against one of four targets,
and the badge is the answer rather than a colour:

| target | loudness | tolerance | max true peak |
|---|:-:|:-:|:-:|
| Streaming | −14 LUFS | ±2.0 LU | −1.0 dBTP |
| EBU R128 | −23 LUFS | ±0.5 LU | −1.0 dBTP |
| ATSC A/85 | −24 LUFS | ±2.0 LU | −2.0 dBTP |
| Cinema | −24 LUFS | ±3.0 LU | −1.0 dBTP |

The badge reads `PASS`, `+1.2 LU` / `−1.2 LU`, `OVER TP`, or `—` when the file is too short
to measure. The selected target travels with the copied CSV/JSON, so a delivery note records
which standard it was graded against. The CLI prints a compact `I −14.2 LUFS · LRA 6.1 LU ·
PLR 12.4 dB` on the completion line, and the full block in `--json`.

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
129 passed, 0 failed, 0 skipped
```

Run it through `cargo run -p g6me-verify` (or `cargo test -p g6me-verify`) rather
than a stale binary in `target/` — an out-of-date verifier is worse than none. Every
check also runs per-group (`--group bitexact` …), so a failure can be reproduced
without waiting for the whole suite. The count above assumes a sibling `g6me` binary
exists to disassemble, as in the release archive; a bare `cargo run -p g6me-verify`
builds only the verifier, finds no sibling, and reports 128. With `--features cuda`
the suite grows to 159 checks: the six-kernel PTX/SASS audit,
the multi-architecture Ozaki-II opcode audit, and the live GPU-vs-CPU parity group
(20 checks) — all six kernels plus a direct Goldilocks NTT butterfly comparison, the
Stockham FFT inverse round-trip, a 65,610-voxel × 128-step production FDTD grid, the
bit-exact raytrace histogram through the direct kernel entry point, the FMM-BEM
reciprocal solve with the GPU solid-angle diagonal asserted, and an end-to-end
`g6me --mode plus --room control-room --cuda` render with zero CPU-fallback stages.

The workspace test suite covers what the verifier does not: 375 tests across the
core (290, plus two opt-in full-resolution true-peak tests ignored by default), CLI
(25), GUI (43), CUDA (4) and harness (13). With `--features cuda` the core grows to
300 — the live CPU/GPU parity tests for dense and FMM BEM, raytrace scattering and
Ozaki-II become active — for 385 workspace tests. The G6+ invariants live there —
a flat 1-ERB diffuse field across all three profiles, measured ITD against
Woodworth, sub-bass coherence, exact L/R mirror symmetry, a clean kernel tail with
no pre-ringing, chunk-size-invariant overlap-add streaming, stereo bypass, LFE gain
and filtering, and high-sample-rate behaviour — along with the limiter's sliding
hold, the loudness gates, the correlation meter and the queue filter. The acoustic
scan is covered end to end in `scan.rs` (synthetic ID-1 card detection, privacy
scrub, multi-view bundle adjustment, winding-number repair, FMM-BEM handoff), the
DNG decoder is tested round-trip against both uncompressed and Lossless-JPEG SOF3
strips, and `blackwell.rs` checks the zero-FMA PTX/SASS audit, including real
`nvcc --fmad=false` + `cuobjdump` verification when the CUDA Toolkit is installed.

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
file, from different builds, on different platforms. (That hash is from the reference
engine; G6+ is deterministic too, but its kernel depends on the sample rate and the
profile, as documented above.)

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
here on a Ryzen 7 5700X3D with the current dependency set (the JPEG-XL decoder is linked
into `g6me-scan`):

| build | instructions | contracted FMA (DSP) | `%ymm` refs | verifier |
|---|---|---|---|---|
| generic (default) | 492,394 | **0** | 123 | 129/129 |
| x86-64-v3 + thin LTO + O3 + unroll | 535,742 | **0** | 13,983 | 129/129 |

The FMA column is what `g6me-verify` enforces: it disassembles the `g6me*` symbols and
fails on any FMA mnemonic. A whole-binary `objdump` additionally finds two `vfmaddsd`
instructions inside `compiler_builtins::fma` — the implementation of an explicitly
requested fused multiply-add, not a contraction of source `a*b+c`, and outside the DSP.

For scale, the deliberate `-ffp-contract=fast` reference build contains 32 FMA
instructions and fails the same comparison, so "no FMA found" means something.

End to end the two builds write **byte-identical** FLAC — SHA-256 compared — for the
plain matrix, for `--rate 48000 --depth s16 --dither low_shibata`, and for the IR
convolution. G6+ is deterministic under the same flags, but its kernels are a function
of the sample rate and profile, so compare G6+ output against the same G6+ settings, not
against the reference engine.

What it actually buys: about 12 % off the DSP's own time (150 ms → 132 ms on the 60 s
verification case). Total throughput did not move measurably, because ffmpeg decode and
encode dominate a real render — so build v3 when you are CPU-bound on known hardware, and
do not make it the default you ship.

## Layout

```
crates/g6me-core     DSP port, G6+ renderer, FFT/NTT, room (UPOLA + FDTD + raytracing),
                     upmix, mastering stages, Wannamaker noise shaping, Klippel
                     transducer assessment, acoustic scan (DNG/LJPEG/JXL, metric
                     anchors, TSDF, winding repair, CHIEF FMM-BEM), pipeline,
                     batch engine, capabilities probe, meters, loudness, WAV I/O
crates/g6me-cuda     optional `--features cuda` zero-FMA `.cu` kernels, `nvcc` +
                     `cuobjdump --dump-sass` auditor, and `cudarc` offload of all
                     six kernels with deterministic CPU fallback
crates/g6me-cli      `g6me` and `g6me-scan`
crates/g6me-gui      `g6me-gui` — eframe, no audio backend
crates/g6me-verify   bit-exactness, zero-FMA ASM/SASS, acoustics, mastering, and scan harness
vendor/g6ct          the C reference, byte-identical to upstream (see UPSTREAM.md)
packaging/linux      desktop entry, icon, AppRun for the AppImage
scripts/             build-optimized.sh (verified v3 build), build-install.sh
                     (build + verify + install), release.sh (artefacts)
docs/                acoustic scan, FMM-BEM/NTT and mastering design notes
```

## Known gaps

* **A hard kill can orphan a staging file.** Renders are written to a hidden sibling
  (`.g6me-<pid>-<n>-<name>`) and renamed into place, so `SIGKILL`, a power cut or a crash
  mid-encode can leave one behind. It is inert — never queued as input, never mistaken for
  your file, and safe to delete — but the guarantee is about *your* filenames, not about a
  spotless temp directory.

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
  the renderer, so it is not what this is about; the output-side resampler is.)

* **DC is metered, and removed only on request.** Every render reports `dc_offset`
  per channel — in full-scale units, because an offset carries a sign and is compared
  against a quantisation step — for the stream and again for the finished file decoded
  back. It joins the one-line completion summary only when some channel passes 1e-3 of full
  scale (−60 dBFS), because a number on every line is a number nobody reads: `810.mka` at
  1.9e-4 is a real, measured, immaterial offset and stays in the full report, while a file
  carrying 0.02 prints `dc +2.00e-2 +2.00e-2` where a batch log can be grepped for it. The downmixer's own contribution is measured rather than assumed: its all-pass
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

  `--dc-block` — or the "block DC" box on the panel's True peak tab, which is deliberately
  above the limiter's own controls, since an offset ships in the file whether or not
  anything is holding peaks — inserts a 5 Hz first-order blocker (`H(z) = g(1-z⁻¹)/(1-Rz⁻¹)`, exactly
  zero at DC, exactly unity at Nyquist) before the trim and the limiter: on that file the
  reported mean falls from 1.89e-4 to 4.9e-9. It is opt-in and the default path stays
  bit-exact, so turning it on changes every hash. What it leaves behind is the blocker's
  own settling spread over the file — which is the reason a DC figure is only meaningful
  with the window it was measured over, and why the tests here drain the limiter's
  lookahead and skip the filter's transient before they claim a number.

* **G6+ is a universal renderer, not your HRTF.** It is deliberately non-individualized:
  the head is a sphere, the pinna is a broad shadow rather than a measured notch, and the
  image is built from published models. It will not place a source at the same elevation or
  with the same front-back certainty as a measured HRTF for your own ears. What it does
  promise is that the result is neutral in tonality, stable across headphones, and free of
  the DEE height-folding pileup that a naive sum produces — and that stacking your own
  headphone EQ on top changes the tone, not the geometry.

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
checkable claim rather than a memory. G6+ is an original implementation of published
acoustics (Brown–Duda, Woodworth, Kulkarni–Isabelle–Colburn, Zaunschirm, ITU-R BS.1116,
Shirley–Kendrick, Bauer–Linkwitz) on top of that same `f64` pipeline.
