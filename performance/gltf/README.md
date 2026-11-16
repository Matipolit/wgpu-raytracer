# glTF/BVH validation and performance

Measured on Intel Iris Xe TGL GT2, Vulkan, Mesa 26.2.7-arch1.1. Release builds,
24 warmup frames and 96 measured frames per case; three alternating runs of each
executable. Raw runs, [comparison JSON](comparison.json),
[chart](comparison.html) and [binary hashes/commands](manifest.json) are retained.
The before executable was saved before the glTF/BVH work in this session.

Active sphere tracing GPU time (median of run medians):

| Resolution | Before | After | Change |
| --- | ---: | ---: | ---: |
| 800 × 600 | 4.788 ms | 4.117 ms | −14.0% |
| 2560 × 1600 | 40.056 ms | 32.871 ms | −17.9% |

No active-frame regression in this workload. The final code specializes known
sphere/light counts, replaces the incorrect full-scene light PDF with the actual
sphere-light solid-angle mixture, samples the previously excluded final light,
and terminates zero-throughput paths. These are correctness and efficiency
changes, so the old and new lighting/noise are different: the timing result is
not a fixed-image speedup or a claim that mesh BVH traversal accelerates spheres.

Completed-display medians varied strongly between runs: at 800 × 600, before
0.150–1.832 ms and after 0.179–1.834 ms. We do not claim a converged-display
speedup from these results. The chart shows the stable active workload only.

Sponza has no previous mesh renderer baseline. The [saved 256-spp render](sponza.png)
and [render log](sponza-render.log) record its absolute performance and memory.
At 640 × 400, eight bounces, the warmed median was **331.6 ms per sample/frame**;
256 spp took **85.6 s** wall time (excluding asset import and GPU setup). Import
was 0.82 s and BVH construction 0.356 s in that run. This is an absolute timing
on this integrated GPU, not an equivalence to the much lighter sphere scene.
The scene has 262,266 triangles, 192,496 vertices, 65 textures, a 157,907-node BVH
(depth 27), approximately 20.6 MiB of geometry/BVH storage and 256 MiB of packed
texture pixels. The asset GLB is deliberately ignored by Git; see
[provenance and download instructions](../../assets/sponza/README.md).

Validation: all 19 CPU/GPU tests passed, including BVH/brute-force differential
checks on Sponza, transformed fixtures, cutouts, sRGB filtering, sampler wrapping,
analytic sphere-light PDFs, font coverage, accumulation/resizing, and compute vs
fragment parity with the common light corrections. The native close path passed
three consecutive sphere-scene runs and three consecutive Sponza runs.
`cargo fmt --check` and `git diff --check` passed. Repository-wide Clippy is still
not clean under the crate's pedantic lint settings (including existing rendering
code and test-fixture casts/float comparisons); the new import and offline-render
modules have no Clippy errors.

See [loading design and supported subset](../gltf-loading.md).
