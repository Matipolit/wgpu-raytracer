Current application defaults: three bounces, roulette enabled. Roulette begins
before bounce four, so it has no effect at the default three-bounce cap.
Historical experiments and training below use an explicit eight-bounce budget.

# Sky importance sampling experiment

Run the experimental path explicitly:

```sh
cargo run --release --locked -- --sponza --bounces 8 --russian-roulette --sky-importance
```

Sky sampling is off by default. It is currently applied to glTF PBR surfaces;
legacy sphere materials retain their existing sampling. The sky remains the
existing analytic model, including its mirrored lower-hemisphere behavior.

## Implementation

- A CPU-built 64×32 equal-solid-angle distribution samples analytic sky luminance.
  Five percent uniform probability supplies full support. Stored PDFs are derived
  from the rounded CDF increments so sampling and evaluated probabilities agree.
- At each eligible mesh vertex, one sky direction is sampled, the diffuse/GGX
  BSDF is evaluated, and an any-hit visibility query tests for blockers.
- Power-heuristic multiple importance sampling combines these sky samples with
  existing BSDF continuation samples. Camera-visible sky is unweighted; escaped
  mesh continuation rays get the complementary MIS weight. Mesh emission and
  legacy sphere continuations retain their original treatment.
- No NEE sample is taken at the last allowed surface bounce: it would add an
  environment connection beyond the original eight-bounce integration budget.
- Shadow traversal returns at the first accepted surface. It preserves alpha
  cutouts, single/double-sided flags, parallel-ray handling and the same range
  limits as ordinary traversal, and avoids resolving normals or shading.
- The distribution adds 16 KiB to the existing sky storage buffer, without a new
  binding. It updates when sky parameters change, not on every camera change.

The sky-enabled path shares a loaded PBR surface between its light evaluation
and continuation sampling. The original BSDF implementation remains the disabled
path; a GPU regression checks that continuation weights match.

## Measured outcome: not a clear win

Intel Iris Xe TGL GT2 / Vulkan, 640×400. Both variants use roulette; both denoisers
are disabled. The same two courtyard views and seed offsets 10,000 / 20,000 are
compared against existing independent-seed 512-spp references. Reference images
still contain noise. Two views are not a universal performance evaluation.

| View | Additional cost per sample | Display MSE change at similar time | PSNR change by seed | Compressed-radiance MSE change |
|---|---:|---:|---|---:|
| 1 | +42% | −2.4% (better) | +0.106 / +0.107 dB | +5.5% (worse) |
| 19 | +46% | +10.2% (worse) | −0.356 / −0.486 dB | +15.1% (worse) |

Baseline: 128 spp with roulette. Sky-enabled: 91 spp in view 1 and 88 spp in
view 19. Final warmed GPU budgets differ by 0.3–2.8%. These are roughly 9–10
seconds per image. First-frame setup/JIT time, scene loading, readback and disk
I/O are excluded. Images include the first sample. Runs are sequential, baseline
then candidate, so clock/temperature drift remains a measurement limitation.
A pilot estimates candidate spp and one correction is allowed when >5% off.
Display metrics use float sRGB after tone mapping, before 8-bit quantization;
compressed radiance uses x/(1+x). Changes average two seeds per view.

The small gain in one display metric is not enough to recommend enabling this
by default, particularly since radiance error worsened in both views. Sky
brightness does not account for visibility through Sponza's openings. Portal- or
visibility-guided sampling would address that mismatch more directly, but is
not implemented here. The results do not show that all environment importance
sampling is unhelpful; they assess this implementation and these views.

## Reproduce

```sh
cargo build --release --locked
training/.venv/bin/python performance/sky/experiment.py
python3 performance/sky/report.py
cargo test --locked -- --include-ignored
```

NumPy and the existing overnight manifest plus reference arrays are required.
Generated PNGs, raw float arrays, per-render timing reports, `summary.json` and
`sky-results.html` are in `results/` (gitignored). The console log is preserved
in `experiment.log`. Tests cover PDF normalization/full support, GPU sampling
consistency, unoccluded MIS energy preservation, BSDF continuation equivalence,
and any-hit versus closest-hit visibility (including cutouts and Sponza).

Final validation: all 23 tests passed with GPU tests enabled. Formatting and
whitespace checks passed. A deterministic sky-disabled render with roulette
matched the pre-change executable's raw radiance byte-for-byte.

## Updated 2048-spp RR reference metrics

The current reports use the eight-bounce 2048-spp references. Original 512-spp
metrics above remain historical; their JSON is archived in the results directory.

| View | Display MSE change | PSNR gain by seed |
|---|---:|---|
| 1 | -3.1% | +0.100 dB / +0.170 dB |
| 19 | +11.2% | -0.387 dB / -0.532 dB |

# Sun disk (default since 2026-11-17)

The analytic sky now has a sun disk (angular radius 0.2666°, `--no-sun` turns it off,
`--night` dims the atmosphere to moonlight and removes it). Its colour follows the
atmosphere's direct-beam transmittance (Rayleigh + turbidity-dependent aerosol optical
depth, Kasten–Young air mass); its brightness is scaled so that at 45° elevation and
turbidity 4 it gives 4× the sky's horizontal illuminance.

Mesh PBR surfaces sample the disk at every bounce (uniform cone, one shadow ray) and
combine it with BSDF hits on the disk by the power heuristic; the last surface's sample
has weight 1 because no continuation is traced from it. Sky importance sampling, when
enabled, uses the atmosphere alone. A GPU test checks that the light-sampled and
BSDF-sampled parts add up to the direct sun integral.

Cost on Iris Xe, `--fastest`, 640×400, median GPU frame time, two interleaved runs:

| Scene | `--no-sun` | sun | `--night` |
|---|---:|---:|---:|
| Sponza atrium | 17.9 ms | 24.8 ms (+38%) | 18.5 ms |
| Bistro street | 28.0 ms | 38.7 ms (+38%) | 28.0 ms |

Skipping the shadow ray at the last bounce saved only about 1%, so every bounce samples
the sun.

# Lamps: emissive triangles sampled directly (2026-11-17)

Emissive scene triangles form a lamp table (alias method over emitted luminance ×
area, packed into the scene buffer). Each mesh surface takes one light sample: the
sun with probability 0.75 when both exist, otherwise a lamp. The lamp is chosen by
resampled importance sampling (the first stage of ReSTIR DI) among
`LAMP_CANDIDATES` = 8 candidates scored by luminance × max(n·ω, 0) / source pdf; MIS
against BSDF hits uses the source pdf on both sides, so the weights still sum to one.
GPU tests: lamp sampling matches BSDF hits on emitters (5%), MIS/RIS preserve energy.

`--night` now dims the atmosphere to about 1e-4 of daylight (was 2e-3, which looked
like dusk and drowned the lamps).

Bistro at night, Iris Xe, `--fastest`, 640×400, 16 spp vs a 2048-spp reference
(display PSNR), median GPU time per sample (`RT_LAMP_CANDIDATES`):

| Lamp sampling | ms/spp | PSNR |
|---|---:|---:|
| none (BSDF hits only) | 28.3 | 13.87 dB |
| 1 candidate | 52.8 | 15.48 dB |
| 4 | 62.0 | 16.41 dB |
| 8 (default) | 71.3 | 16.76 dB |
| 16 | 88.2 | 17.11 dB |
| 32 | 127.5 | 17.37 dB |

Each candidate costs about 2.3 ms here (three random vertex reads); the shadow rays
about 24 ms. At equal time on this GPU plain path tracing is still competitive by
mean error; the gain is in fireflies and in what spatial/temporal reuse (ReSTIR)
can build on. Daytime Bistro: 38.8 → 44.8 ms/spp (+16%).

RTX 5070 Ti, `--hw-rt`, 960×600, fixed exposure, vs an 8192-spp reference (the
pre-lamp build patched to the same night sky):

| Lamp sampling | ms/spp | 16 spp | 64 spp |
|---|---:|---:|---:|
| none | 2.51 | 19.19 dB | 20.45 dB |
| 1 candidate | 5.29 | 20.85 dB | 23.77 dB |
| 8 (default) | 6.42 | 22.22 dB | 26.18 dB |
| 32 | 10.8 | 22.92 dB | 27.16 dB |

Without lamp sampling the error is dominated by fireflies and barely falls with more
samples (+1.3 dB for 4×); with it the error falls as expected. At equal time (64 spp
at 8 candidates vs ~164 spp without) lamp sampling wins by roughly 1.7 dB here, and
the gap grows with the sample count.
