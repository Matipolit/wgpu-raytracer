# GGX visible-normal sampling experiment

Opt-in:

```sh
cargo run --release --locked -- --sponza --ggx-vndf
```

The renderer's three-bounce default and roulette setting are unchanged. VNDF
changes how glTF PBR specular directions are sampled, not the BRDF, lighting,
geometry, bounce count or denoising. Legacy sphere materials are unchanged.

## Implementation and correctness

The implementation follows Eric Heitz's isotropic projected-hemisphere method:
[Sampling the GGX Distribution of Visible Normals, JCGT 7(4), 2018](https://jcgt.org/published/0007/04/01/).
It is standard VNDF sampling, not the report's later bounded-VNDF variant.

The normal is sampled in a local shading frame after stretching the view by
alpha = roughness². Reflection uses that sampled normal. The specular direction
PDF changes from `D(h) * NdotH / (4 * VdotH)` to
`D(h) * G1(view) / (4 * NdotV)`. Both are combined with the same cosine-diffuse
PDF and the existing diffuse/specular selection probability. Sky MIS evaluation
also uses the matching PDF when the two experimental flags are combined.
Below-surface samples remain zero contribution; retrying them without changing
the PDF would alter the estimator. Views below/at the shading-normal horizon
fall back to the original sampler and PDF to preserve existing behavior.

The GPU validation samples 18 combinations of view cosine (1, 0.2, 0.05),
roughness (0.2, 0.5, 1), and metallic factor (0, 1), with 131,072 samples per
sampler per combination. It compares expected constant-environment contribution,
variance, zero-contribution counts and sampled-normal validity. Observed means
agreed within approximately 0.4%; this is statistical evidence on those cases,
not a proof covering every surface configuration. The underlying BRDF's energy
behavior is deliberately preserved rather than changed by this experiment.

At cos(view)=0.05, roughness=0.5, metallic=1, measured variance fell from 0.572997
to 0.048345 (91.6% lower); zero-contribution samples fell from 36.0% to 1.15%.
This controlled white-environment diagnostic must not be interpreted as a
Sponza-wide speed or variance improvement.

## Rendering comparison

Intel Iris Xe TGL GT2 / Vulkan, 640×400, three bounces. Roulette is enabled in
both variants but does not activate before bounce four. Both denoisers and sky
importance sampling are disabled. Two clear courtyard views (IDs 1 and 19),
three independent evaluation seed offsets (10,000, 20,000, 30,000).

New independent **2048-spp, three-bounce NDF references** use seed offset
2,000,000. The previous eight-bounce references are not used: they integrate a
different set of light paths. High-sample references still contain noise.
Reference files live alongside this experiment and do not replace the earlier
eight-bounce references.

Baseline uses 128 spp. A short VNDF pilot estimates the candidate spp that fits
the warmed GPU budget; one correction is allowed for a >5% mismatch. Totals
exclude the first frame, loading/JIT, readback and disk I/O. Images include the
first sample. Runs are sequential, baseline then candidate; GPU clock and
thermal drift remain limitations. Two views are a limited scene evaluation.

Metrics are float display-space sRGB MSE/PSNR after the renderer's tone curve,
and compressed-radiance MSE using x/(1+x). Timings and per-seed budgets are
retained in the generated summary. Lower error per sample is not enough:
VNDF must repay its overhead at comparable render time.

## Reproduce

```sh
cargo build --release --locked
training/.venv/bin/python performance/vndf/experiment.py
python3 performance/vndf/report.py
cargo test --locked vndf_ -- --ignored --nocapture
```

The experiment needs NumPy and the overnight dataset's camera manifest. PNGs,
raw radiance, reference arrays, timings, the summary, validation measurements
and the interactive report are generated in `results/` (gitignored). Completed
reference renders are reused. `experiment.log` and `gpu-validation.log` preserve
this run's measurements.

## Measured result

| View | Mean cost/sample | Display MSE change at similar time | Compressed-radiance MSE change | PSNR changes by seed |
|---|---:|---:|---:|---|
| 1 | +4.2% | +5.1% (worse) | +9.2% | −0.327 / −0.293 / −0.026 dB |
| 19 | +4.9% | +2.9% (worse) | +4.7% | +0.016 / −0.303 / −0.076 dB |

Baseline uses 128 spp; VNDF uses 120/120/129 spp in view 1 and 124/115/121 spp
in view 19. Final warmed GPU budgets differ by up to 4.8%. These small image-error
differences are comparable in scale to the budget/clock variation, so they do not
establish a universal regression. They do show **no demonstrated practical win**
for enabling this implementation across these Sponza views. VNDF remains opt-in.
The strong controlled grazing-specular result is not representative of the whole
scene, where diffuse transport and texture/visibility noise also contribute.

Validation: ten CPU tests passed, plus the new 18-condition GPU VNDF test, the
existing sky-MIS energy/continuation check and compute-versus-fragment regression.
With VNDF disabled, a raw Sponza smoke image matched the pre-change executable
byte-for-byte. Formatting and whitespace checks passed. The full unrelated GPU
suite was not rerun for this sampler-only change.
