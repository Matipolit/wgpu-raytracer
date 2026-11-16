# Iris Xe non-neural renderer optimization

Measured on Intel Iris Xe TGL GT2, Vulkan, release build. This work applies the
attached architecture proposal's separation of traversal and surface resolution
within the existing path tracer. It preserves the integrator, scene, sampling,
8-bounce limit, resolution, texture filtering, alpha cutouts and bilateral denoiser.
It does not implement the proposal's new engine, caches, screen-space tracing,
wavefront queues, temporal reuse or hardware ray tracing.

## Retained changes

- Traversal carries a 16-byte triangle candidate (distance, barycentrics, index).
  It interpolates normals/tangents and samples normal maps only for the final
  closest hit, instead of every successively closer triangle. Alpha masks still
  run during traversal, preserving visibility through cutouts.
- Reciprocal ray direction and parallel-axis masks are computed once per BVH
  query. Vectorized slab tests retain explicit handling of parallel rays and
  origins on box boundaries, avoiding infinity/NaN boundary failures.
- Static 48-byte traversal records store triangle origin/edges, determinant
  threshold, material and culling/opacity flags. Opaque candidate tests avoid
  dependent vertex and material fetches and repeated edge/length calculations.
- A selected child already passed its box test, so it is not tested again on
  entry. Deferred siblings are retested against the updated closest distance.

Sponza geometry storage increases from 20.6 to 32.6 MiB (+12.0 MiB). Its 256 MiB
texture buffer is unchanged. Sphere-only scenes allocate no triangle records.

## Experiments rejected

Lane-interleaved shared-memory traversal stacks measured 304–313 ms/sample at
640×400, versus 189–197 ms with private stacks. The shared implementation was
removed. `shared-tuning.json` preserves its measurements.

8×4 was slightly faster in one view and slower in the other; 8×2, 16×2 and 4×8
were also tested. The default stays 8×8. `packed-tuning.json` and
`skip-tuning.json` record exploratory single-run results, not final comparisons.
`RAYTRACER_WORKGROUP=8x4` remains available for repeatable tuning experiments.

## Measurement and correctness

`final-640.json` interleaves baseline/candidate order across two runs per case.
Each run renders 24 deterministic samples at 640×400. It reports the median GPU
time across frames 2–24, then the median across runs. These timings include
tracing, filtering and tone-mapped presentation; loading, pipeline creation,
image readback and disk I/O are excluded. Baseline/candidate raw float32 radiance
is compared byte-for-byte for every case. `final-720.json` repeats the Sponza
comparison at 1280×720 with 16 samples and one run per binary.

The original executable is `/tmp/raytracer-baseline`. Binary fingerprints are
in `environment.json`; `/tmp` artifacts are temporary. `optimization.patch`
records the renderer changes against the working tree as it existed at the
start of this optimization (the starting tree already had uncommitted work).

Reproduce against a saved baseline executable:

```sh
cargo build --release --locked
python3 performance/laptop/compare.py /path/to/baseline \
  target/release/weekend-raytracer-wgpu --output /tmp/comparison.json
python3 performance/laptop/measure.py --output /tmp/workgroups.json \
  --workgroups 8x8 8x4 8x2 16x2 4x8 --cases sponza sponza-reverse
cargo test --locked -- --include-ignored
```

GPU tests cover BVH versus brute force (including Sponza), parallel/inside rays,
alpha cutouts, texture sampling, normal maps, emission, sphere lights,
accumulation, resizing, display composition and denoiser controls. Raw identity
on the benchmark views is additional regression evidence, not proof that all
floating-point intersections on all possible scenes are bit-identical.

## Results

| Scene | Resolution | Before (ms) | After (ms) | Speedup | Raw identical |
|---|---|---:|---:|---:|---|
| sponza | 640x400 | 447.2 | 185.2 | 2.42× | yes |
| sponza-reverse | 640x400 | 470.0 | 193.7 | 2.43× | yes |
| sponza | 1280x720 | 1558.1 | 647.8 | 2.41× | yes |
| sponza-reverse | 1280x720 | 1649.0 | 746.4 | 2.21× | yes |

At 1280×720 this remains around 0.65–0.75 seconds per sample, so these changes
do not make full-resolution Sponza interactive. They establish a faster exact
tracer before larger changes to sampling, temporal reconstruction or scheduling.

A longer sphere-only control (1280×720, 256 samples, four interleaved runs)
measured 7.684 ms before and 7.685 ms after: effectively
unchanged, with byte-identical raw output. See `sphere-control.json`. The much
shorter 24-frame sphere observations in `final-640.json` varied with GPU ramp-up
and should not be interpreted as a stable regression.

Final validation: `cargo test --locked -- --include-ignored` passed all 19 tests
(including all 10 GPU tests); `python3 tests/native_close.py` passed three
consecutive native window-close checks. `cargo fmt --check` and
`git diff --check` also passed.
