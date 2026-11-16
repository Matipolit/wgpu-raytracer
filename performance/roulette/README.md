Current application defaults: three bounces, roulette enabled. Roulette begins
before bounce four, so it has no effect at the default three-bounce cap.
Historical experiments and training below use an explicit eight-bounce budget.

# Russian roulette experiment

Run the eight-bounce comparison setting with:

```sh
cargo run --release --locked -- --sponza --bounces 8 --russian-roulette
```

Before bounce four, and each later bounce, the tracer computes survival
probability `clamp(max(throughput.rgb), 0.05, 0.95)`. A failed trial terminates
the path. A surviving path divides throughput by that probability. This keeps
the expected radiance contribution unchanged relative to the existing integrator,
which still has an eight-bounce maximum. Contributions already collected are
retained. The same implementation covers meshes and spheres. Use `--no-russian-roulette` to disable roulette.

## Experiment

Intel Iris Xe TGL GT2 / Vulkan, 640×400, no neural or bilateral denoising.
Two clear Sponza courtyard views (overnight dataset IDs 1 and 19), with two
sample-offset seeds each (10,000 and 20,000). Existing 512-spp fixed-depth
references use different frame seeds (1–512). Reference images retain noise;
these are comparative measurements on two views, not universal performance or
convergence guarantees. The two views were also part of the earlier training
set, but no learned model participates in this experiment.

Fixed-depth renders use 64 spp. A short RR pilot predicts the sample count that
fits the fixed render's warmed GPU budget. One correction is made if the first
RR attempt misses by more than 5%. Reported final RR budgets were 1.0–4.2% longer
than fixed; setup/JIT, readback, disk I/O and the first sample's GPU time are
excluded from budget matching. Each image includes that first sample. Runs are
sequential, fixed followed by RR; temperatures/clocks can drift. The repeated
seeds are a small robustness check, not a formal confidence interval.

Display metrics use float sRGB after the renderer's exposure/tone curve, before
8-bit quantization. Compressed-radiance MSE is also recorded in the raw summary.
Lower MSE and higher PSNR are better.

| View | Mean per-sample speedup | Mean display MSE reduction at similar time | PSNR gain by seed |
|---|---:|---:|---|
| 1 | 2.47× | 44.8% | +2.58 / +2.58 dB |
| 19 | 2.43× | 44.3% | +2.46 / +2.62 dB |

Fixed: 64 spp. Matched-budget RR: 160/160 spp for view 1 and 156/161 spp for
view 19. This is about 11.5–12.6 seconds of warmed GPU work per image.

At **the same 64 spp**, first-seed RR renders took 4.58 s / 4.37 s, but PSNR
was 0.49 / 0.64 dB worse than fixed depth. RR trades noisier individual samples
for more samples per second. It is not a replacement for denoising or better
light sampling. Tiny light sources and rare bright paths can still cause fireflies.

## Reproduce

```sh
cargo build --release --locked
training/.venv/bin/python performance/roulette/experiment.py
training/.venv/bin/python performance/roulette/report.py
cargo test --locked
cargo test --locked roulette_preserves_expected_throughput -- --ignored
cargo test --locked compute_matches_fragment_reference_through_accumulation_and_resize -- --ignored
```

Requires the existing overnight `manifest.json`, its two raw 512-spp references,
and NumPy. Captures, per-render timing JSON, `summary.json` and the visual report
are generated in `results/` (gitignored). `experiment.log` and `report.log` preserve
this run's console output.

Additional offline controls: `--sample-offset N` offsets initial frame seeds;
`--timing-output FILE.json` writes warmed GPU totals, median and p95 (requires
timestamps and at least two samples). Omitting the RR flag preserves the old path.

Validation: nine default CPU tests passed; the new GPU expectation test passed
for zero, dim, ordinary and >1 throughput, including the survival clamps. The
existing GPU compute-versus-fragment regression passed. A separate deterministic
render with RR disabled matched the pre-change executable byte-for-byte.

## Updated 2048-spp RR reference metrics

The current reports use the eight-bounce 2048-spp references. Original 512-spp
metrics above remain historical; their JSON is archived in the results directory.

| View | Display MSE change | PSNR gain by seed |
|---|---:|---|
| 1 | -48.6% | +2.900 dB / +2.883 dB |
| 19 | -47.6% | +2.725 dB / +2.891 dB |
