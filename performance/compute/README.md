# Fragment → compute migration

Measured 2026-11-15 on Intel Iris Xe (TGL GT2), Vulkan, Mesa 26.2.7-arch1.1.
Both versions use wgpu 30.0.2, egui 0.37 with bundled fonts, and release builds.
The fragment executable was saved before this migration, after the font and
Wayland shutdown fixes.

## Implementation

- Path tracing and accumulation run in an 8×8 compute workgroup per tile. Every
  invocation owns one pixel; dispatch dimensions round up with bounds checks.
- Bilateral denoising runs in a separate 8×8 compute pass. Workgroups load the
  requested pixel neighborhood into shared memory, including the halo, then
  synchronize before filtering. Threads outside the viewport still participate
  in the load and barrier. Radii above the UI's maximum of 5 use a global-memory
  fallback with the same filter equation.
- The compute passes use separate writable and read-only bindings. Presentation
  reads the completed filtered buffer in a small render pass that tone-maps it
  and draws egui.
- Once sampling finishes, the filtered result is cached. Subsequent redraws skip
  compute dispatches and redundant uniform uploads. Settings changes and resizing
  reset the result; both buffers follow the viewport size.
- Sampling, random seeds, bounce limits, material/lighting code, and float32
  radiance precision are preserved. No resolution or quality reduction is used.

## Verification

All 10 release tests pass, including the previous resize, denoiser, font, and egui
compositing regressions. The native close test exits cleanly on three consecutive
runs. The frozen pre-compute shader in
`src/raytracer/tests/fragment_reference.wgsl` is used only in tests; production
has no fragment-tracing fallback.

The image comparison runs both implementations on the same GPU with identical
seeds, comparing every output pixel. It covers 13×9, 63×47, and 128×96 viewports;
denoising disabled, radius 0, and radii 1, 3, 5, and 6; and nine successive frames
through accumulation, completion, and resizing. All 162 comparisons differ by
at most one 8-bit channel step (1/255), allowing stage/compiler rounding.

## Timing method

Three runs per executable alternate order (before/after, after/before,
before/after). Each of four scenarios warms for 24 frames and measures 96.
Values below are the median of three per-run medians, in milliseconds.

The fixed scene, default fly camera, 8 bounces, 3×3 filter, and UI are identical.
Accumulating uses 1 sample per pixel per frame with a 512-sample target.
Converged uses 8 samples per frame with a 128-sample target, reached before
warmup ends. GPU timestamps surround rendering, including presentation and UI.
CPU timing includes encoding and submission; total wall time includes GPU
completion and the small timestamp readback. There is one frame in flight, with
no window, vsync, compositor, or input dispatch in the timing path. UI status
updates at 10 Hz while idle. Separate medians are not additive.

## Final results

| Scenario | GPU, fragment → compute | GPU change | Total wall, fragment → compute | CPU, fragment → compute |
|---|---:|---:|---:|---:|
| 800 × 600, accumulating | 4.848 → 4.808 | -0.8% | 5.667 → 5.601 | 0.456 → 0.441 |
| 800 × 600, converged | 2.212 → 0.940 | -57.5% | 3.094 → 1.940 | 0.348 → 0.311 |
| 2560 × 1600, accumulating | 40.368 → 41.333 | +2.4% | 41.277 → 42.366 | 0.478 → 0.467 |
| 2560 × 1600, converged | 6.371 → 2.684 | -57.9% | 7.186 → 3.427 | 0.431 → 0.386 |

Fullscreen accumulation shows a small regression: GPU time rises about 2.4%
(0.964 ms), and serial total frame time rises about 2.6%. Small-window accumulation
is approximately unchanged/slightly faster in these runs. Compute is therefore
not an across-the-board tracing speedup on this machine.

After convergence, GPU frame time is about 58% lower at both resolutions;
fullscreen serial total frame time falls about 52%. Most of this gain comes
from caching the unchanged filtered image. Both implementations stop launching
new paths at the sample target, but the fragment baseline re-filters on every
display frame. This is not evidence that ray traversal itself became 58% faster;
a fragment renderer could also be extended to cache its filtered output.

Raw medians and p95 values are in [results/](results/), aggregated values in
[comparison.json](results/comparison.json), and the interactive comparison in
[comparison.html](comparison.html). These measurements describe this GPU and the
default settings; they do not establish behavior on other GPUs, with different
filter radii/sample budgets, continuous interaction, or pipelined presentation.
GPU clocks and other desktop activity also affect small differences.

## Memory and tuning

The separate filtered float32 RGB buffer adds 12 bytes per pixel: **46.875 MiB**
at 2560×1600. Accumulated and filtered radiance together use 93.75 MiB there,
excluding textures, UI, scene buffers, and the swapchain.

Single-run 8×4 and 16×4 tracing workgroup trials did not show a clear gain, so
8×8 remains the default. Their raw data is in `tuning-8x4.json` and
`tuning-16x4.json`. `initial.json` and `initial-results/` preserve the initial
compute measurements before the final redundant-upload optimization. Only
`results/` describes the final implementation.

## Reproduce

```sh
cargo test --release --locked -- --include-ignored
cargo build --release --locked
python3 tests/native_close.py
./target/release/weekend-raytracer-wgpu --benchmark
python3 performance/compare.py /path/to/fragment-baseline \
  target/release/weekend-raytracer-wgpu --rounds 3 --output performance/compute/results
```

The original baseline for this session is
`/tmp/compute-migration-yuj3z11z/fragment-baseline`; it is not committed. Save a
release executable built with the benchmark before future changes to compare
against it. Timestamp query support is required only for benchmarking.
