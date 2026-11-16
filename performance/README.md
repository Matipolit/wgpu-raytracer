# ImGui → egui performance comparison

This is the earlier UI migration report. See [compute/README.md](compute/README.md)
for the subsequent compute migration and its current measurements.

Measured 2026-11-15 on Intel Iris Xe (TGL GT2), Vulkan, Mesa 26.2.7-arch1.1.
Both builds use wgpu 30.0.2, release optimization, the same scene, default fly
camera, 8 bounces, and the same 3×3 denoiser. The ray-tracing shader is unchanged
by this UI migration. These are the refreshed measurements with bundled fonts
enabled, after fixing the missing-glyph and Wayland shutdown bugs. Earlier
measurements without usable egui fonts are preserved in `pre-font-results/`
and `uncached-results/` for diagnosis, not as the current comparison.

## Method

The ImGui executable was built and saved before replacing its dependencies.
`src/benchmark.rs` was added first, with the original parameter controls extracted
from the application; its timing/scenario loop was retained during migration.
Both versions build and draw their real parameter panel. No window, compositor,
vsync, pixel readback, or input-event dispatch is included.

Each case warms for 24 frames, then measures 96. There are three runs per build,
alternating order (before/after, after/before, before/after). GPU times use hardware
timestamps around accumulation, display, and UI drawing. CPU frame times run
through queue submission. UI CPU time includes layout, tessellation, uploads, and
draw encoding. Total wall time includes CPU work, GPU completion, and the small
timestamp readback. One frame is in flight, so these measure serial latency,
not fully pipelined windowed throughput. Separate medians are not additive.

Accumulating uses 1 sample per frame and a 512-sample limit. Converged uses
8 samples per frame and a 128-sample limit, reached before warmup ends. Render
progress and FPS labels are active; there is no pointer/keyboard interaction.
Egui reuses painted geometry while idle, updating status at 10 Hz; input events,
resize/DPI/focus changes, and egui repaint deadlines trigger updates sooner.
Scene rendering and accumulation continue every frame.

## Results

Values are the median of three per-run medians, in milliseconds. The chart is in
[comparison.html](comparison.html); raw data and p95 values are in
[results/](results/), with aggregated values in [comparison.json](results/comparison.json).

| Scenario | GPU, ImGui → egui | GPU change | Total wall, ImGui → egui | UI CPU, ImGui → egui |
|---|---:|---:|---:|---:|
| 800 × 600, accumulating | 4.832 → 4.847 | +0.3% | 5.802 → 5.706 | 0.119 → 0.010 |
| 800 × 600, converged | 2.269 → 2.352 | +3.7% | 3.142 → 3.207 | 0.115 → 0.010 |
| 2560 × 1600, accumulating | 40.446 → 40.371 | -0.2% | 41.627 → 41.237 | 0.118 → 0.010 |
| 2560 × 1600, converged | 6.348 → 6.368 | +0.3% | 7.280 → 7.213 | 0.127 → 0.011 |

GPU medians changed by −0.2% to +3.7%; the largest absolute increase was
0.083 ms in the small-window converged case. At fullscreen, GPU differences were
−0.2% during accumulation and +0.3% after convergence. Median total wall time
changed by −1.7% to +2.1%, and median CPU frame time fell by 13–22%.
GPU clocks and desktop activity vary on this integrated GPU; small differences
are not evidence that the ray tracer itself became faster.

This does **not** establish that every egui update is faster. The initial port
rebuilt the whole UI each frame, adding about 0.4 ms of CPU work; those measurements
are preserved in [uncached-results/](uncached-results/). The final implementation
avoids redundant work, but individual rebuild frames still have a higher UI CPU
cost than ImGui. See final raw p95 values. Continuous interaction, text editing,
other hardware, and other DPI settings are not covered by these performance
numbers. Median gains do not imply a worst-case or mean-latency improvement.

## Reproduce

Requires a GPU supporting timestamp queries. Normal application use does not
require that feature.

```sh
cargo build --release --locked
./target/release/weekend-raytracer-wgpu --benchmark
python3 performance/compare.py /path/to/baseline-binary \
  target/release/weekend-raytracer-wgpu --rounds 3 --output performance/results
```

The comparison script saves every run and checks that the adapter, driver,
scenario list, and sample counts match. The original ImGui executable for this
session was saved outside the repository under `/tmp/egui-migration-3zyi5tf_/imgui-baseline`.
Keep an executable built with this benchmark before future changes to compare
against it. Egui 0.37 requires Rust 1.95 or newer; the old ImGui dependencies and
git pin have been removed.
