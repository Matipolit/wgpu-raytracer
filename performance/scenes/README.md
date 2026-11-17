# Multi-scene traversal optimization — in-progress report

## Current state (2026-11-17)

The sections below are the 2026-11-16 report; this summary supersedes its timings and
"current best configuration".

- **Fastest tracer:** `--fastest` = `--sah-stop --skip-final-scatter --quad-bvh --sbvh`, leaf
  size 1, 32-entry shared-memory traversal stack (exact). About 1.87x geomean over the
  previous best (`--sah-stop --skip-final-scatter`) on the eight untextured views, 2.3-2.9x
  on Sponza and Bistro.
- **No loop guards:** shaders are compiled without naga's per-loop iteration guards
  (1.04x geomean, bit-identical output). `--loop-guards` restores them.
- **Auto-tuner** (`src/raytracer/tuner.rs`), on by default:
  - *Per frame:* GPU timestamps choose between trace variants: texture mips from ray cones
    (first hit only or the whole path) and paths starting from a rasterized visibility buffer
    (`src/raytracer/visibility.rs`). 1.15x geomean over fixed settings (hairball 1.88x,
    textured Bistro 1.37x).
  - *At startup:* traversal stack and workgroup shape are measured once per GPU, driver and
    scene, and cached in `~/.cache/wgpu-raytracer/tuning.txt`.
  - Controls: `--no-auto-tune`, `--no-calibrate`, `--recalibrate`, `--tuner-log`;
    `--ray-cones`, `--cones-primary-only`, `--no-ray-cones`, `--visibility-primary`,
    `--no-visibility-primary` force a setting.
- **Rasterized visibility buffer:** `--visibility` fills the G-buffer by rasterization
  (1.7-3.7x faster than tracing first hits at 720p; 99.9% of pixels bit-identical).
  GPU occlusion culling exists but is opt-in (`RT_VIS_OCCLUSION=1`); it culled little here.
- **Temporal neural denoiser:** `--temporal-model MODEL.wgsl` from
  `training/temporal/export.py`. 10.7 ms at 720p on Iris Xe (was 27.8 ms), matches PyTorch
  to 3e-5. Training, capture and evaluation scripts are in `training/temporal/`.
- **Rejected or opt-in (measured):** 8-wide and 8-bit quantized BVHs, wavefront tracing,
  blue-noise path sampling (`--blue-noise-path`), sky importance sampling (off by default),
  raw-Vulkan rewrite (naga's output measured as good as optimized SPIR-V; the only WGSL cost
  was the loop guards).


Status as of 2026-11-16. All work is **uncommitted** in the working tree; the
fastest configuration still relies on experiment environment variables (see
[Current best configuration](#current-best-configuration)).

## Summary

- The benchmark now covers **7 scenes / 8 views** (interiors, urban exterior,
  pathological hairball, voxel city, small-object chess set) instead of Sponza only.
- Current best: **1.39× faster** (geomean) than the previous best
  `--sah-stop --skip-final-scatter`, and **1.50×** faster than the default settings.
  Every view improved by 1.31×–1.53×.
- Main wins, in order of size: a 4-wide BVH whose children sit in one node and
  are tested together, a child-pair node layout, keeping the traversal stack
  in registers, and hardware-sampled BC-compressed textures.
- Two pieces of infrastructure were also added on request: **auto-exposure** and a
  **hardware texturing architecture**. The second also lifts the 2 GiB
  texture limit; textured Bistro (3.4 GB of decoded texels) now loads in 574 MiB.

## Current timings

Intel Iris Xe (Raptor Lake-P, 96 EU), Mesa 26.0.8 ANV / Vulkan. 640×400, 3 bounces,
roulette on, denoiser off, fixed exposure. Median warmed GPU frame time (trace + filter
+ display) over 32 samples, median of 3 interleaved repetitions. Run-to-run noise
is about ±1.5%.

| View | Triangles | Default (ms) | Previous best (ms) | **Current (ms)** | vs previous | vs default | Rays/s previous → current |
|---|---:|---:|---:|---:|---:|---:|---:|
| sponza-atrium | 262 k | 57.0 | 53.2 | **39.2** | 1.36× | 1.45× | 12.8 → 17.3 M |
| sponza-gallery | 262 k | 51.4 | 47.8 | **35.4** | 1.35× | 1.45× | 14.6 → 19.7 M |
| bistro-street (untextured) | 2.83 M | 95.3 | 89.3 | **65.7** | 1.36× | 1.45× | 7.4 → 10.0 M |
| hairball | 2.85 M | 37.6 | 34.2 | **24.8** | 1.38× | 1.51× | 10.5 → 14.5 M |
| fireplace-room | 143 k | 20.9 | 19.3 | **13.8** | 1.40× | 1.51× | 34.8 → 48.6 M |
| living-room | 581 k | 23.0 | 22.6 | **15.6** | 1.45× | 1.47× | 31.6 → 45.7 M |
| rungholt-aerial | 5.82 M | 15.7 | 14.2 | **10.8** | 1.31× | 1.45× | 35.2 → 46.3 M |
| chess | 1.50 M | 11.7 | 10.6 | **6.9** | 1.53× | 1.70× | 42.1 → 64.3 M |
| **geomean** | | | | | **1.39×** | **1.50×** | |

Measurement caveat: until the launcher fix below, imported models without a preset view
inherited the interactive camera's depth of field (aperture 0.8, focus 10.8 units).
So every view except Sponza above was rendered with depth of field, identically for all
configurations, which means the speedups are unaffected. Re-measured with a pinhole camera
(current best), times are within ±2% except chess, which drops 7.3 → **6.1 ms**:
its tiny board was heavily blurred.

Textured Bistro (new texture path only; the old one cannot load it):
**66.4 ms** with the current best, 98.2 ms with `--sah-stop --skip-final-scatter`.

Raw data: [results/in-progress-report.json](results/in-progress-report.json),
[results/in-progress-bistro-textured.json](results/in-progress-bistro-textured.json).
"Rays/s" uses the measured rays per pixel from the traversal counters (table below).

## What was achieved

### Kept changes (each measured against the step before it)

| Step | Geomean | Exact? | What and why |
|---|---:|---|---|
| Child-pair BVH2 records (`--pair-bvh`) | 1.114× vs previous best | yes (ties only) | Both children's boxes live in the parent's 64-byte record, so each step reads **one cache line instead of two**. Traversal was found to be memory-latency bound (below). |
| 4-wide BVH (`--quad-bvh`) | 1.218× vs previous best | yes (ties only) | 128-byte records with SoA child boxes, all four slab tests together, and a 5-comparator **register sorting network**. This halves dependent node fetches per ray (Sponza 62 → 31 nodes/ray). The repo's earlier 4-wide result (0.57×) came from its implementation: insertion sort through private arrays, which Mesa puts in scratch memory. |
| Leaf triangle count in child reference | 1.040× vs quad | yes | The reference encodes `first \| (count−1)<<26`. This removes a dependent "last triangle" flag load per triangle tested. |
| Module-scope stack | 1.034× vs quad | yes | WGSL zero-initializes local arrays. Mesa turned the 64-entry stack init into **64 scratch-memory writes per ray**; module scope pays that once per pixel. Superseded by the register stack. |
| 16-entry register stack | 1.072× vs the two above | approximate | At ≤16 entries Mesa keeps the stack in registers (0 spills). A full stack drops further pushes. That changed ≤0.33% of pixels, as isolated single-sample speckles (visually checked on Bistro). |
| Hardware textures (now the only path) | 1.046× (7 views) | appearance-equivalent | See [Texturing architecture](#texturing-architecture). Biggest gain on alpha-tested Sponza (1.13×). Block-averaged radiance matches the old path within 0.3%. |

The single-run ladder multiplies to slightly more than the consolidated measurement, which is expected from separate runs. The consolidated table above is the authoritative number.

### Diagnosis that guided the work

- `INTEL_DEBUG=cs` showed the tracing kernel compiles to **SIMD8, ~5.8k instructions**, with the
  traversal stack in **scratch memory** and zero-initialized per ray.
- Traversal counters (`RT_STATS=1`) at the previous best:

  | View | Rays/pixel | BVH2 nodes/ray | BVH4 nodes/ray | Triangles/ray |
  |---|---:|---:|---:|---:|
  | sponza-atrium | 2.65 | 62.1 | 31.0 | 14.7 |
  | sponza-gallery | 2.73 | 61.7 | 30.5 | 16.9 |
  | bistro-street | 2.57 | 76.4 | 37.0 | 32.0 |
  | hairball | 1.41 | 51.3 | 26.3 | 27.3 |
  | fireplace-room | 2.62 | 25.3 | 11.4 | 5.0 |
  | living-room | 2.79 | 24.3 | 11.2 | 6.0 |
  | rungholt-aerial | 1.96 | 27.3 | 16.3 | 2.8 |
  | chess | 1.74 | 17.5 | 9.7 | 2.8 |

- Napkin math for Sponza: ~42 M node visits/frame × ~80 instructions ≈ 3–4 ms of ALU, in a 50 ms
  frame. With incoherent rays, each binary node visit touched two cache lines (~5 GB of line traffic per
  frame). So the kernel is bound by **dependent memory latency and bandwidth**, not arithmetic. Every kept change reduces cache lines touched or dependent fetches per ray.

### Rejected experiments

| Experiment | Result | Lesson |
|---|---:|---|
| 32-entry stack (binary BVH) | 0.982× | Still spills; size alone does not help unless it reaches registers. |
| Cull popped 4-wide entries by stored distance | 1.003× (chess 0.90×) | The extra 64-entry distance array doubles scratch zero-init. It also changed hits through box/triangle float disagreement. |
| 8-bit quantized 4-wide nodes (64 B) | 0.969× | Halves bytes, but decoding 24 bytes per node costs more than it saves. It helped only Sponza atrium (+11%). |
| 12-entry register stack | 1.020× | Overflows often: 5.9% of Bistro pixels changed. 16 is the sweet spot. |
| SAH traversal cost 0.5 / 2.0 | 1.012× / 0.969× vs cost 1 | 0.5 is mixed per scene (−2% to +6%); inconclusive. |
| Ray-cone mip selection (`--ray-cones`, hardware) | 0.986× | Cone bookkeeping costs more than the texture cache saves at 640×400. |
| Stochastic texture filtering (`--stochastic-all`, hardware) | 0.986× | Hardware bilinear is already free. |

## New infrastructure

- **Scene suite**: `performance/scenes/scenes.json` (views) and `scripts/convert_scene.py`
  (Blender OBJ/FBX → GLB; maps BLEND materials to MASK; optional uniform scale).
  Bistro is scaled by 0.01 to fit the tracer's fixed `[0.001, 1000]` ray interval.
- **Harness**: `performance/scenes/bench.py` interleaves configurations per scene with rotated
  order. It reports per-view medians, geomean speedup, raw-identical status, and the share of
  pixels visibly changed (>2/255 after tone mapping, same seeds).
- **Traversal counters**: `RT_STATS=1` outputs nodes/triangles/rays per pixel instead of radiance.
- **Auto-exposure**: a single-workgroup pass computes the log-average luminance over a fixed
  64×64 grid of 4×4 blocks of the filtered image. It is on by default in the app, with a UI
  toggle and compensation slider. Offline renders use it unless `--exposure` is given;
  `--exposure-compensation` scales it.

### Texturing architecture

Textures previously lived in one storage buffer as uncompressed RGBA8 without mips.
The shader did its own wrap, 4-tap bilinear filtering and `pow()` sRGB decode, and
the whole set had to fit one 2 GiB storage binding. Now (`src/raytracer/gpu_textures.rs`):

- A `binding_array<texture_2d<f32>>` plus 18 wrap/filter samplers, indexed per lane.
  This requires Vulkan descriptor indexing (`TEXTURE_BINDING_ARRAY` + non-uniform indexing).
- One GPU texture per image per use: sRGB colour, linear data, or normal map.
  Full mip chains are filtered in linear light.
- BC compression via `texpresso` (new dependency): BC1 for colour and data, BC3
  when a MASK material reads alpha, and BC5 for normal maps (the shader rebuilds z).
  It falls back to RGBA8 for non-multiple-of-4 images or devices without BC.
- Index 0 is a white placeholder, and empty slots point at it. Using an out-of-range
  "none" index hung the GPU on Mesa ANV, even on paths that never executed (found via a
  test device-loss).
- Memory: chess 528 MiB → 141 MiB, textured Bistro 3.4 GB → 574 MiB (both including mips).
  Encoding takes 0.2–7 s at load.

All 27 CPU/GPU tests pass. The texture-filter test's tolerance was widened to 1e-3
because hardware sRGB decoding is permitted to be approximate.

## Trying it interactively

```sh
scripts/try_scene.sh                       # list scenes
scripts/try_scene.sh bistro-street-textured            # fastest configuration
scripts/try_scene.sh sponza-atrium previous            # --sah-stop --skip-final-scatter
scripts/try_scene.sh chess default --bounces 8         # extra flags pass through
```

The app now accepts `--camera x,y,z,dx,dy,dz,fov` (previously offline-only) and
`--move-speed`. WASD speed defaults to 1/15 of the scene's diagonal per second.
Imported models without a preset view start without depth of field.

## Current best configuration

```sh
cargo build --release --locked
RT_QUAD_STACK=16 RT_QUAD_LEAF_COUNT=1 target/release/weekend-raytracer-wgpu \
    --sponza --sah-stop --skip-final-scatter --quad-bvh
```

`RT_*` variables are experiment switches: `RT_QUAD_STACK` sets the stack size,
`RT_QUAD_LEAF_COUNT` the leaf encoding, `RT_PRIVATE_STACK` the module-scope stack,
`RT_QUAD_CULL` / `RT_QUAD_Q8` the rejected variants, `RT_SAH_CT` the SAH cost and
`RT_STATS` the counters. They should be turned into flags or defaults
before committing (first item below).

Reproduce the table:

```sh
F=--sah-stop,--skip-final-scatter
python3 performance/scenes/bench.py --out /tmp/report --reps 3 \
  --scenes sponza-atrium,sponza-gallery,bistro-street,hairball,fireplace-room,living-room,rungholt-aerial,chess \
  --config prev-best=BASELINE_BINARY:$F --config current=WRAPPER_SETTING_RT_VARS:$F,--quad-bvh
```

Scene sources: McGuire Computer Graphics Archive (`casual-effects.com/g3d/data10/research/model/{bistro,hairball,living_room,fireplace_room,rungholt}`;
Bistro textures are the separate `BuildingTextures`, `OtherTextures`, `PropTextures` downloads)
and Khronos glTF-Sample-Assets `ABeautifulGame`. Convert with
`blender --background --factory-startup --python scripts/convert_scene.py -- IN.obj OUT.glb [SCALE]`.
The Bistro MTL needs its `..\` paths changed to `../`.

## Not yet tried / next steps

Ranked by expected payoff for the cost. Estimates are guesses grounded in the counters above.

1. **Productize the current best** (small). Make the 4-wide BVH, leaf counts and register stack
   the defaults or real flags, remove dead experiment paths, and commit.
2. **Spatial-split BVH (SBVH)** (builder only, medium). Sponza and Bistro have long, overlapping
   architectural triangles. Literature gains are 10–30% on such scenes. It costs memory
   (duplicated references) and build time.
3. **SAH-aware 4-wide collapse** (small–medium). The 4-wide tree currently collapses the binary tree greedily by
   largest area. An SAH-optimal collapse, plus leaf-size/cost tuning for 4-wide, should cut node
   visits further. Expect 5–10%.
4. **Exact register stack** (small). Drop the farthest entry instead of the nearest, or bound tree
   depth at build time, so the 16-entry stack never loses geometry. Correctness more than speed.
5. **Triangle test cleanup** (small). Hoist the per-triangle `length(ray.direction)`, defer the
   division, and consider a precomputed (Woop-style) triangle transform. Bistro and hairball test
   27–33 triangles/ray. Expect 3–8% there.
6. **Split traversal from shading (wavefront)** (large). The monolithic 5.8k-instruction
   kernel forces SIMD8. A trace-only kernel could compile wider, keep lanes busy (no
   shading divergence) and enable ray sorting for coherence. Potentially the largest remaining
   lever, but a pass/buffer redesign.
7. **Persistent threads / dynamic ray fetch** (medium). Lanes idle when their path ends early
   (sky hits); fetching new work keeps SIMD lanes full. Pairs well with 6.
8. **8-wide compressed nodes (CWBVH-style)** (large). 4-wide quantization lost on decode cost, so
   only worth revisiting with a cheaper decode or after 6.
9. **Workgroup shape retune** for the new kernel (trivial). 8×8 was chosen for the old kernel.
10. **Exact texture follow-ups** (optional). BC7 for higher-quality colour (needs an ISPC
    encoder), and a cooked texture cache to avoid re-encoding at every launch.

Not pursued, because they are outside "raw tracer first": temporal accumulation and reuse,
upscaling, and denoiser changes.
