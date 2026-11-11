# Vendored C reference

These files are the reference implementation that the Rust port is checked
against, byte for byte. They are compiled by `crates/g6ct-verify/build.rs`
(the float64 build with renamed symbols, so both precisions can be linked into
one test binary) and are **never** linked into the shipped app.

| file | role |
|---|---|
| `g6ct.c`, `g6ct.h` | the bit-exact C11 port of `CMatrixEncoderEfxMod` (`KSUSBAPO64.dll`) |
| `g6ct_cli.c` | raw-interleaved-float CLI renderer, used as the file-level oracle |

## Provenance

```
KSUSBAPO64.dll  CMatrixEncoderEfxMod        (USB Audio APO, AMD64)
      |  reverse engineered (see below)
      v
g6-re/port/g6ct.c   float32 build: bit-identical to the DLL under Wine
      |  copied here unchanged
      v
g6ct-standalone/vendor/g6ct/                this directory
```

Obtained from `g6-re/port/` (this repo's sibling project) at the commit named in
`docs/VERIFICATION.md`. The DLL replica status is established in
`g6-re/apo-harness/verification_dll_wine.txt`.

## Numerical contract of the C reference

From `g6-re/port/Makefile` — these flags are load-bearing, not tuning knobs:

```
-ffp-contract=off -fno-fast-math      # FMA contraction would break bit-exactness
```

The Rust port reproduces that contract by language default: Rust compiles
`+ - *` on `f32`/`f64` with `fp-contract=off`, and `crates/g6ct-verify`
asserts it in the generated assembly on top of comparing outputs bit for bit.

## License

MIT, (c) 2026 HyperRamzey — see `../../LICENSE`.

Reverse-engineering notes, the Wine A/B harness that proves the float32 build is
sample-identical to the shipping driver, and the LADSPA/VST2 deliveries all live
in the companion repository: <https://github.com/HyperRamzey/g6-re>.
