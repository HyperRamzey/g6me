# Room presets people actually use, and what it takes to render them 20 Hz – 10 kHz+

Research date: 2026-11-28. Status: research + implementation plan. No engine code changed by this document.

## 0. Verdict

**Yes, a useful preset library is implementable — but not by scaling the current renderer unchanged.**

- The low band (20 Hz to the FDTD cutoff) is already wave-based and cheap. A 90 m³ control room's
  first mode is 28.6 Hz; a 15,000 m³ hall's first mode is 3.5 Hz. FDTD at 500–670 Hz resolves both.
- The high band (cutoff to 10 kHz+) **must** be geometric. Wave-based FDTD at 10 kHz needs
  `dx = 343/(10000·6) = 5.72 mm`, i.e. 88.6 billion cells / 709 GB per f64 field for the Musikverein.
  That is not a tuning problem; it is out of reach.
- The current geometric path is **single-band**: it applies `absorption[3]` (1 kHz) on every bounce
  (`room.rs:1267`, `room.rs:1376`). Until that becomes per-band, no preset can honestly claim
  10 kHz spectral accuracy, regardless of geometry.
- Measured validation exists from **31.25 Hz to 16 kHz** (OpenAIR), **100 Hz to 14 kHz** (dEchorate),
  and dense spatial grids at 48 kHz (MeshRIR). No public dataset gives a calibrated 20 Hz oracle;
  20–31.5 Hz must be validated against modal theory and the 31.25 Hz band.

So the deliverable is: a **shoebox preset library** (the engine's geometry model) with per-band
targets, plus a **10-band renderer upgrade** and a **measured-data validation harness**. Non-shoebox
rooms (vineyard halls, horseshoe opera houses, Gothic cathedrals) need a geometry rewrite and are
explicitly out of scope for this phase.

## 1. What the engine supports today (verified against source)

| Item | Current state | Location |
|---|---|---|
| Geometry | Shoebox only: `dims_m[]` + 6 walls; validated 1.5–60 m | `room.rs:200-212`, `308-328` |
| Presets | Anechoic, Control Room (6×5×3 m), Theater Stage (12×9.5×5.2 m) | `room.rs:26-37`, `215-296` |
| Wall absorption | 6 octave bands: 125, 250, 500, 1k, 2k, 4k Hz | `room.rs:166-190` |
| Scattering | ISO 17497 coefficient, frequency-flat | `room.rs:171-172` |
| Target RT | One scalar `target_rt60_s` | `room.rs:207-208` |
| FDTD | 6 PPW, Preview 500 Hz / 65k voxels; Production 670 Hz / 260k voxels | `room.rs:723-744` |
| Ray tracing | Monte Carlo, per-wall `absorption[3]` only | `room.rs:1262-1288`, `1376-1378` |
| ISM | Orders 1–3 with `reflectance(f)` (band-interpolated) | `room.rs:176-190` |
| FDN | 8×8 paraunitary, broadband gain | `room.rs:349-360` |

Consequence: low-frequency output is already physically modelled; high-frequency output is
"1 kHz absorption applied at all frequencies", i.e. the right decay shape only by accident.
The Theater Stage misses the global Production target: at `dx = 0.1316 m` the 6-PPW limit is
`343/(0.1316·6) = 434 Hz`, not 670 Hz. `docs/ACADEMIC_NUKE_ACOUSTICS.md` now states the
correct 437 Hz budget figure; `RoomQuality::Production` still carries the global 670 Hz label.

## 2. The ten bands and where each one comes from

Proposed band set (ISO 266 preferred octaves plus the two extra OpenAIR bands):

`31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz`

20 Hz is not an octave centre; it is a **lower band edge**. "20 Hz coverage" means the render
chain must not high-pass above 20 Hz and the FDTD grid must include the room's lowest modes.
For the reference rooms that is already true: 3.5 Hz (Musikverein length), 4.5 Hz (Boston),
14.3 Hz (Theater Stage), 28.6 Hz (Control Room).

Per-band method and confidence:

| Band | Method | Validation source | Confidence |
|---|---|---|---|
| 20–31.5 Hz | FDTD (3-D, 6 PPW) | modal theory + OpenAIR 31.25 Hz RT | Medium: no calibrated 20 Hz IR oracle |
| 31.5–125 Hz | FDTD | OpenAIR RT @ 31.25/62.5/125 Hz | High for RT, medium for fine structure |
| 125–500 Hz | FDTD to cutoff, ISM/ray above | ISO 3382-1 + all measured sets | High |
| 500 Hz–4 kHz | ISM (orders 1–3) + per-band ray | ISO 3382-1, OpenAIR, dEchorate | High after B1 |
| 4–16 kHz | per-band ray + air absorption | OpenAIR RT @ 8k/16k, 96 kHz files | Medium-high after B1 |
| 16–20 kHz+ | ray + air absorption | 96 kHz OpenAIR subset only | Low: no RT band above 16 kHz |

The FDTD/geometric crossover stays at the current 500–670 Hz. It is a **model handoff**, not a
sample-exact splice: match the diffuse-field energy and the band-limited RT across the crossover,
then gate on measured ISO 3382 metrics, not on waveform identity.

## 3. Measured datasets worth implementing against

| Dataset | Content | Rate | Geometry / coverage | License | Use |
|---|---|---|---|---|---|
| **OpenAIR** | 341 files, IR + BRIR + B-format; RT60 given at 31.25, 62.5, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz | 44.1 / 48 / 96 kHz | 46+ environments: halls, churches, rooms, outdoors | CC BY 4.0 | Primary full-band oracle (31.5 Hz–16 kHz) |
| **dEchorate** | 1800 annotated RIRs, 6 arrays × 5 mics, 6 sources, 11 wall configs; early-echo and image-source annotations | 48 kHz / 32-bit | 6 × 6 × 2.4 m cuboid, RT60 ≈ 0.6 s, ESS 100 Hz–14 kHz | Check Zenodo record 5562386 | Early-reflection / geometry / echo-label validation |
| **MeshRIR** | 4410 RIRs on dense 3-D and 2-D grids | 48 kHz, 32768 samples | 7.0 × 6.4 × 2.7 m, RT60 0.38 s / 0.19 s | CC BY 4.0 | Spatial-field and receiver-grid validation |
| **AcousticRooms** | 300k+ **simulated** RIRs, 260 rooms, 332 materials | 22.05 kHz processed, 32 kHz raw | 20–1000 m³, 10 room types | CC BY 4.0 | Geometry + material corpus only; not an acoustic oracle |
| **3D-MARCo** | 71 simultaneous mics, performances + RIRs | to verify from record | St Paul's Hall, Huddersfield | Open access | Concert-hall spatial validation (metadata fetch timed out; verify before use) |
| **BUT Reverb DB** | 1300+ RIRs, 8 rooms | to verify | small/medium rooms | research use | Speech-room RT calibration |
| **ACE Challenge** | 7 rooms, 1/2/3/5/8/32 ch | to verify | small/medium/large | CC BY-ND | Cross-check only (no derivatives) |
| **Arni** | 132,037 RIRs, 5342 panel configurations | to verify | Aalto variable-acoustics lab | check record | Calibrating Sabine/Eyring inversion |
| **Aachen IR DB** | 344 binaural RIRs, 5 environments incl. church | to verify | dummy head | research use | Binaural church check |
| **RWCP** | 143 multichannel RIRs, 14 rooms | to verify | rooms | research use | Legacy cross-check |

Two hard facts about availability:

- `openairlib.net` currently returns **"Account Suspended"**; the site is not usable. The dataset
  itself is still distributed (audEERING `openair` package lists 341 files, CC BY 4.0; the
  University of York record also exists). Mirror the needed subset locally before building gates.
- AcousticRooms is generated by Treble in **hybrid mode: wave < 710 Hz, image-source order 4 with
  50k rays, stochastic ray tracing with 5000 rays**, and its released RIRs are 22.05 kHz
  (Nyquist 11.025 kHz). It cannot validate 16 kHz content and is not measured ground truth.

## 4. Recommended preset library

Tier A geometry and materials are implementable **today** (shoebox, fits the 60 m validator,
measured targets exist). The hall low band is not: see the voxel-cap note below. Tier B needs
the per-band renderer. Tier C needs non-shoebox geometry and is not scheduled.

### Tier A — shoebox rooms with measured targets

`T` = occupied mid-frequency RT, `EDT` in s, `G` in dB, `C80` in dB. Values are Beranek's
compiled measurements as tabulated by akuTEK (Table 2); geometry from the cited building sources.

| Preset | Dimensions L×W×H (m) | V (m³) | T | EDT | G | C80 | Character |
|---|---|---|---|---|---|---|---|
| **Control Room (existing)** | 6.0 × 5.0 × 3.0 | 90 | 0.25 target | — | — | — | BS.1116 / EBU Tech 3276 |
| **Theater Stage (existing)** | 12.0 × 9.5 × 5.2 | 593 | 0.42 target | — | — | — | scoring / mix stage |
| **Small studio live room** | 8.0 × 6.0 × 4.0 | 192 | 0.35–0.55 design | — | — | — | calibrated to BUT/ACE small rooms |
| **Lecture theatre** | 15.0 × 10.0 × 6.0 | 900 | 0.7–0.9 design | — | — | — | speech, ACE/BUT cross-check |
| **Cinema** | 20.0 × 12.0 × 8.0 | 1920 | 0.4–0.6 design | — | — | — | dubbing stage |
| **Vienna Musikverein, Grosser Saal** | 48.8 × 19.1 × 17.75 | 15,000 | 2.0 | 2.0 | 4 | −1 | shoebox reference; LF 0.17, 1−IACC 0.64, ITDG 12 ms |
| **Boston Symphony Hall** | 38.1 × 22.9 × 18.6 | 18,750 | 1.9 | 1.9 | 2 | 0 | shoebox; LF 0.22, 1−IACC 0.65, ITDG 15 ms |
| **Amsterdam Concertgebouw, Grote Zaal** | 44 × 28 × 17 | 18,780 | 2.0 (Beranek) / 2.2 occupied (hall) | 1.9 | 3 | −1 | shoebox; LF 0.18, ITDG 21 ms |
| **Carnegie Hall** | L to verify; W 26; H/W 0.9 | 24,270 | 1.8 | 1.7 | 1 | 0 | shoebox; add only after length verified |

Notes:

- Concertgebouw's RT is reported as 2.0 s occupied by Beranek and 2.2 s occupied / 2.8 s empty by
  the hall; keep both as a tolerance band, not a single number.
- The Musikverein box volume from raw dimensions is 16,544 m³; the quoted 15,000 m³ is the
  acoustic volume. Store the quoted volume as metadata, not as `L·W·H`.
- Theater Stage must either raise its voxel cap to ~963k cells for 670 Hz
  (`12/0.0853 × 9.5/0.0853 × 5.2/0.0853`) or stop advertising 670 Hz. Today it does neither.
- Hall-scale FDTD is not free. At the current 260k-voxel Production cap, the Musikverein grid
  coarsens to `dx = (16,544/260,000)^(1/3) = 0.399 m`, i.e. a 143 Hz wave band — not 500–670 Hz.
  Reaching 500 Hz needs ~11.2M cells (0.09 GB per f64 field); 670 Hz needs ~26.8M cells
  (0.21 GB per field). Both fit the RTX 5070's 12 GB, but they are 40–100× the current
  per-render voxel budget. Decide per preset whether the low band is wave-based to 500 Hz or
  handed to the geometric path earlier, and record that decision in the preset metadata.

### Tier B — calibrated, not yet measured

- **Stone church / cathedral nave**, rectangular: 30 × 15 × 12 m, target RT 3–6 s, per-band
  rise below 125 Hz; validate against OpenAIR cathedral/church entries.
- **Scoring stage**, 30 × 18 × 12 m: RT 1.0–1.4 s; no public measured IR set found, so mark the
  preset "design target" until a dataset is located.
- **Chamber hall**, 20 × 12 × 9 m: RT 1.4–1.8 s; OpenAIR chamber/small-hall entries.

### Tier C — not implementable without a geometry rewrite

Berlin Philharmonie (vineyard, H/W 0.3), Bayreuth Festspielhaus (fan), horseshoe opera houses,
Gothic cathedrals, and any room with balconies, coupled volumes, or non-planar walls. The
`RoomGeometry` type cannot represent them, and both the FDTD voxelizer and the ray tracer assume
six axis-aligned planes.

## 5. What "20 Hz – 10 kHz+" requires in code

Ordered by dependency. Each is independently testable.

### B0. Ten-band wall model (small)

Replace `absorption: [f64; 6]` with a band table. Keep backward compatibility:

```rust
pub struct WallMaterial {
    /// Energy absorption at [31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k] Hz.
    pub absorption: [f64; 10],
    pub scattering: [f64; 10], // per-band ISO 17497, not one scalar
}
```

Serde: accept 6-element arrays by log-interpolating to 10 (or by holding the two edge bands,
which is what the current `reflectance()` already does at `room.rs:179-189`). Existing preset
JSON must keep loading unchanged.

### B1. Per-band geometric transport (the critical path)

- Replace the scalar ray energy with a 10-band vector. On each bounce multiply band `b` by
  `1 - alpha_b`; on each segment multiply by `exp(-m_b(f) · d)` using ISO 9613-1.
- Same for ISM taps: `reflectance(f)` already exists; extend the band table to 10.
- Keep HOA histograms per band, or render directional HOA in 3 perceptual bands and energy in 10.
  Per-band HOA multiplies the histogram memory by the band count; measure before committing.
- CUDA `raytrace_hoa.cu` currently takes `wall_absorption[6]` and uses index 3
  (`room.rs:1366-1378`). Change signature to `[10]` per wall and accumulate per band.
- Gate: CPU vs GPU parity per band, bit-exact under the existing ascending-`ray_id` reduction.
  A sabotage run must show the 8k band diverging when its absorption column is perturbed.

Without B1, every "10 kHz" claim is false. This is the one non-negotiable change.

### B2. FDTD low-band boundary (medium)

The current single `bass_trap_admittance: f64` cannot produce a measured per-band RT curve.
Options, in order of rigor:

1. Frequency-dependent real impedance boundary (IIR/admittance filter) per wall — correct,
   research-grade, hardest to keep bit-exact.
2. Three band-limited FDTD runs (e.g. 31.5–125, 125–500, 500–cutoff) with band-specific
   admittance, summed after rendering — simpler, triples cost, still physically defensible.
3. Keep one broadband run and fit only the 125–500 Hz average — honest only if the preset
   metadata says the low-band RT is a fit, not a prediction.

Choose (1) for the reference library if the schedule allows; otherwise (2) with an explicit
per-band metadata flag.

### B3. Render chain audit for 20 Hz

- Confirm no default high-pass above 20 Hz in the UPOLA/BRIR path. The pipeline is 8192-point
  FFT at the render rate (`plus.rs:90-92`), so 48 kHz gives a 24 kHz Nyquist; 20 Hz is 3.4 bins
  in an 8192 FFT — fine, but any DC/rumble filter must be documented and disabled for room IRs.
- Source directivity: a dodecahedron is not omnidirectional below ~50 Hz. Preset metadata must
  state the assumed source (omni point vs. real loudspeaker) or the 20–50 Hz level is meaningless.

### B4. Calibration pipeline (medium)

For each preset and band:

1. Compute the room's total absorption from the measured target `T_b` (Sabine or Eyring,
   Eyring for alpha > 0.2).
2. Distribute absorption over the six surfaces using material classes (front/side/back/floor/
   ceiling), then solve a constrained least-squares fit to the target `T_b`, `EDT_b`, `C80_b`,
   `D50_b` and `G_b`.
3. Store the fitted `alpha[10]` and the **fit residual per band** in the preset. A preset with a
   residual above tolerance is marked "approximate" and may not carry a measured-room name.

The Arni dataset (5342 panel configurations with measured RT) is the right calibration corpus for
step 1: it directly tests whether the Sabine/Eyring inversion recovers known absorption.

### B5. Validation harness (medium)

New verifier checks, all against downloaded measured IRs:

| Gate | Metric | Tolerance (proposed) |
|---|---|---|
| RT per band | T20/T30, 31.5 Hz–16 kHz | ±5% or ±1 JND, whichever is larger |
| Early decay | EDT per band | ±10% |
| Clarity | C80 at 500/1k/2k | ±1 dB |
| Definition | D50 at 500/1k/2k | ±0.05 |
| Spatial | LF, 1−IACC | ±0.05 |
| Strength | G (relative to a calibrated source) | ±1 dB |
| Early echoes | dEchorate echo times | ±1 sample at 48 kHz for first-order |

Every gate must be shown to fail under a deliberate perturbation (scale one band's alpha by 10%,
or swap two wall materials) before it is trusted. A gate that passes both the correct and the
sabotaged preset is not a gate.

## 6. Preset metadata schema

Extend the room JSON (backward compatible, all new fields optional):

```jsonc
{
  "name": "Vienna Musikverein, Grosser Saal",
  "provenance": "Beranek 2004 via akuTEK Table 2; geometry Wikipedia",
  "dims_m": [48.8, 19.1, 17.75],
  "acoustic_volume_m3": 15000.0,
  "bands_hz": [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
  "target_rt60_s": [3.2, 2.9, 2.4, 2.2, 2.1, 2.0, 1.9, 1.7, 1.3, 0.9],
  "target_edt_s":  [3.1, 2.8, 2.4, 2.2, 2.1, 2.0, 1.9, 1.7, 1.3, 0.9],
  "target_c80_db": [null, null, null, null, null, -1.0, -1.0, -1.0, null, null],
  "target_g_db": 4.0,
  "target_lf": 0.17,
  "target_1_minus_iacc": 0.64,
  "target_itdg_ms": 12.0,
  "measured_ref": "openair:<venue-id>",
  "fit_residual_db": [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.6]
}
```

The RT curve above is a **placeholder shape**, not measured data. The real per-band curve must be
filled from OpenAIR's RT60 bands for the specific venue or from Beranek's octave tables before the
preset is enabled. Do not ship a preset whose `measured_ref` does not resolve.

## 7. Answer to "rooms people use"

The practical library, in priority order:

1. **Control room** — already present; the one room type where the standard (EBU Tech 3276 /
   ITU-R BS.1116: 0.2–0.4 s, `T = 0.25·(V/100)^(1/3)` averaged 200 Hz–4 kHz, +0.3 s allowed at
   63 Hz) is more important than any single measured hall.
2. **Studio live room / vocal booth** — high commercial use; calibrate to BUT/ACE small rooms.
3. **Theater / cinema dubbing stage** — existing Theater Stage; fix the 437 vs 670 Hz gap.
4. **Lecture theatre** — speech clarity; ACE/BUT.
5. **Shoebox concert halls** — Musikverein, Boston, Concertgebouw; the only iconic halls the
   current geometry can represent exactly.
6. **Rectangular church / cathedral nave** — OpenAIR church/cathedral RT curves.
7. **Chamber hall** — OpenAIR small halls, Brahms Saal geometry once its RT is sourced.

Not recommended until geometry work: Berlin Philharmonie, Bayreuth, opera houses, Gothic
cathedrals. Naming a preset after one of these while rendering a shoebox would violate the
paper/spec-literal rule.

## 8. Implementation order

| Step | Work | Depends on | Rough size |
|---|---|---|---|
| 1 | B0 ten-band material + serde compat | — | S |
| 2 | B1 per-band ray/ISM energy + CUDA parity | B0 | M |
| 3 | B3 render-chain 20 Hz audit | — | S |
| 4 | Preset schema + Musikverein/Boston/Concertgebouw geometry & targets | B0 | S |
| 5 | B5 harness: OpenAIR subset (RT/EDT/C80) | B1, B4 | M |
| 6 | B4 inverse fit from target RT; Arni calibration test | B0 | M |
| 7 | B2 frequency-dependent FDTD boundary | B4 | L |
| 8 | Tier B presets + dEchorate/MeshRIR spatial gates | B5 | M |
| 9 | Tier C geometry (polygon extrusion, BVH, voxelizer) | — | XL |

## 9. What this document does not claim

- It does not claim the current renderer already covers 20 Hz–10 kHz. It covers 20 Hz to
  437–670 Hz wave-based, then broadband geometric with 1 kHz absorption.
- It does not claim any measured dataset is an exact oracle at 20 Hz. OpenAIR's lowest RT band is
  31.25 Hz; below that, modal theory is the reference.
- It does not claim AcousticRooms is ground truth; it is a simulated corpus, useful for geometry
  and materials only.
- It does not claim the Tier A RT values are octave-band curves. They are mid-frequency single
  numbers; the per-band curves still have to be sourced per venue.

## 10. Sources

- OpenAIR dataset card (341 files, bands, rates, CC BY 4.0): https://audeering.github.io/datasets/datasets/openair.html
- OpenAIR library record (University of York): https://pure.york.ac.uk/portal/en/datasets/openair-the-open-acoustic-impulse-response-library/
- IR dataset index: https://github.com/moisestohias/IR-datasets (fork of Graphi07/room-impulse-responses)
- dEchorate paper: https://arxiv.org/abs/2104.13168
- MeshRIR: https://sh01k.github.io/MeshRIR/
- AcousticRooms: https://github.com/facebookresearch/AcousticRooms
- Concert-hall parameter table (Beranek data): https://akutek.info/concert_hall_acoustics_files/parameters.htm
- Musikverein geometry: https://en.wikipedia.org/wiki/Musikverein
- Concertgebouw geometry and RT: https://en.wikipedia.org/wiki/Concertgebouw,_Amsterdam
- Boston Symphony Hall geometry: https://en.wikipedia.org/wiki/Symphony_Hall_(Boston)
- EBU Tech 3276 / ITU-R BS.1116 RT guidance (secondary summary): https://acusticax.com/en/optimize-the-acoustics-and-calibration-of-your-home-theater/
- pyroomacoustics material/band API: https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.parameters.html
