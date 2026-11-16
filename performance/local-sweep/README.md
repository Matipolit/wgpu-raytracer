# Combined local optimization report

Measured 20 independent rendering variants, a cooked-scene cache, a sparse spatial filter, and 2 selected combinations. Application defaults remain three bounces, roulette on, every new experiment off.

## Main finding

The strongest confirmed exact-output combination was SAH leaf-cost stopping plus final-hit shading elimination: 1.098× per-sample speedup and 8.9% lower equal-time display error in these tests. It matched baseline raw pixels in both fixed-count checks. Adding packed attributes to that combination was less effective. Defaults were left unchanged.

The warm scene cache reduced startup wall time by 3.33×; it does not accelerate steady-state rays. Sparse spatial filtering was slower and increased image error. Small single-feature differences remain inconclusive.

## Results

| Technique | GPU speedup | Raw identical at 32 spp | Equal-time display MSE | Type |
|---|---:|---|---:|---|
| SAH leaf-cost stopping | 1.093× | yes | not separately tested | geometry |
| Packed shading attributes | 1.066× | no | -3.4% | precision |
| Stochastic all-map filtering | 1.057× | no | -1.6% | approximation |
| 32-bin SAH | 1.045× | no | -2.4% | geometry |
| Skip final-hit shading/scatter | 1.043× | yes | not separately tested | exact |
| Stochastic color filtering | 1.012× | no | +0.3% | sampling |
| Packed ray directions | 1.010× | no | +1.1% | precision |
| Spatial blue-noise / R2 jitter | 1.007× | no | -1.5% | sampling |
| Material opacity classification | 1.003× | yes | not separately tested | exact |
| FP16 path radiance/throughput | 1.000× | no | -1.0% | precision |
| Diffuse alpha-area approximation | 0.999× | no | +0.1% | approximation |
| Conservative software opacity cells | 0.991× | yes | not separately tested | geometry |
| Cap alpha texture tests at four | 0.987× | no | +0.9% | approximation |
| Ray-cone texture mip selection | 0.976× | no | +2.1% | filtering |
| Bounded GGX VNDF | 0.917× | no | +0.0% | sampling |
| Compressed eight-wide BVH | 0.798× | yes | not separately tested | geometry |
| BVH leaf size 8 | 0.797× | yes | not separately tested | geometry |
| BVH leaf size 16 | 0.625× | yes | not separately tested | geometry |
| Eight-wide BVH | 0.586× | yes | not separately tested | geometry |
| Four-wide BVH | 0.567× | yes | not separately tested | geometry |

Speedup >1 is faster. Negative MSE change is better. Screen speedups are geometric means across two views, normalized against interleaved baseline controls. They are screening measurements, not formal confidence intervals. Exact same-count pixels do not imply equal-time quality was measured separately.

## Combinations

### Combined exact candidates

Measured 1.098× per-sample speedup; equal-time display MSE -8.9%, compressed-radiance MSE -9.0%. Maximum budget mismatch 1.4%.

```sh
cargo run --release --locked -- --sponza --sah-stop --skip-final-scatter
```

### Combined quality candidates

Measured 1.059× per-sample speedup; equal-time display MSE -5.3%, compressed-radiance MSE -5.1%. Maximum budget mismatch 1.1%.

```sh
cargo run --release --locked -- --sponza --sah-stop --skip-final-scatter --packed-shading
```

## Startup and spatial filter

Warm cooked-scene cache: median process wall time 1.711 s → 0.514 s (3.33×), three alternating trials per mode, 96×64/8 spp. The cache saves loading/build work, not steady-state tracing work. Source path/metadata, relevant options and importer source fingerprint key the cache; checksum validates payload. It is limited to the supplied self-contained Sponza GLB.

| View | Baseline filter p95 | Sparse filter p95 | Filter speedup | Filtered display MSE change |
|---|---:|---:|---:|---:|
| 1 | 3.374 ms | 4.840 ms | 0.70× | +18.9% |
| 19 | 3.646 ms | 4.050 ms | 0.90× | +21.7% |

Filter timings isolate 100 warmed dispatches. Filter-only speedup is not total-render speedup. The sparse kernel uses current-frame radiance only; it reads no history.

## Method

Intel Iris Xe TGL GT2 / Vulkan. All rendered comparisons use native 640×400, three bounces, roulette enabled (inactive before bounce four), sky importance and prior VNDF off. No upscaling, history reprojection, lighting cache, raster primary visibility or queue/pass redesign is introduced. Original progressive accumulation is retained. Denoising is off except in the separate filter experiment.

References are independent 2048-spp, three-bounce NDF renders from the VNDF experiment (seed offset 2,000,000). Screening uses 32 spp/seed 10,000 in two courtyard views (1 and 19), with baseline controls every four variants and linear interpolation of baseline GPU cost in time. Each screen point is one process run. Raw radiance identity is checked against same-seed baseline captures. The pre-sweep binary is also checked when available.

Precision/sampling/appearance variants and any geometry variant with non-identical raw output receive two seeds × two views at matched warmed GPU budgets. Baseline is 64 spp; candidate count is predicted from screening and corrected up to twice to target ±3%. Actual mismatches are retained, not hidden. Candidate counts and flags are in every record. Timing excludes first frame, imports/JIT, readback and disk writes. Sorted mean GPU totals divided by warmed sample count determine budgets. Sequential runs still permit clock/temperature drift. Small differences are inconclusive.

Display MSE uses float sRGB after the renderer tone curve; compressed-radiance MSE uses x/(1+x). References retain Monte Carlo noise. Changes to filtering or alpha visibility can change the target image; a lower scalar error does not establish physical equivalence. Results from two views do not establish universal wins.

## Implementation distinctions

- Four/eight-wide nodes are a local collapse of the current binary SAH tree. The compressed variant stores outward-rounded half bounds and packed metadata, 128 bytes per eight-child node. This is not the report’s specialized 80-byte CWBVH/bitmask traversal implementation. Overflow falls back to a full triangle query rather than dropping hits. Tied distances select the lowest current triangle index.
- Packed shading vertices use 32 bytes instead of 64: positions remain float32; normals/tangents use octahedral snorm16 and UVs use half. Quantization is a quality tradeoff, including UVs used by alpha.
- FP16 uses native f16 live radiance/throughput state, with float32 shading and final accumulation. It does not prove a particular register allocation or validate large HDR ranges outside the tested scenes. Packed directions carry an octahedral word between mesh bounces, reconstructing for traversal.
- Stochastic color filtering samples one bilinear-weighted texel for color/emission maps; all-map mode also changes normal/roughness and alpha filtering. Nonlinear material evaluation means matching expected texel values does not guarantee matching the original BRDF.
- Ray cones choose texture mip levels using a conservative footprint and a roughness-based spread heuristic. Color mips average in linear light; odd dimensions use area weights. Alpha tests stay at base level. This changes texture prefiltering, not output resolution. It is a heuristic prototype, not a full ray-differential implementation.
- Material opacity classification can only remove a MASK test if every texel passes conservatively. Sponza had no such materials. Software opacity cells use a 4×4 barycentric grid with unknown/opaque/transparent states; classification scans a conservative texel rectangle, limits work, and falls back to exact alpha for unknown cells. Small classified coverage can make this slower.
- Diffuse alpha-area approximation shrinks triangle coverage by material mean opacity for diffuse secondary rays; other rays retain alpha. The four-test cap treats further unknown alpha hits as opaque. Both can change visibility and are never considered exact winners.
- Blue-noise primary jitter uses original toroidal void-and-cluster spatial rank masks with an R2 frame-index shift. No prior rendered pixels are read. This is not the report’s filter-trained FAST/STBN sequence.
- Bounded VNDF uses the 2023 bounded spherical-cap sampler and its matching mixture/MIS PDF; it is separate from the prior 2018 VNDF test.
- Final-hit skip removes unused normal/shading/BSDF work at the bounce cap while retaining emission/UVs and the three path RNG draws, preserving subsequent samples in a batch.

## Scope of the supplied report

| Report idea | Treatment |
|---|---|
| Compact hit/ray/shading state | Existing 16-byte candidates plus packed attributes, oct directions and native FP16 trials |
| Wide/compressed software BVH | Four/eight-wide and compressed prototype measured; specialized 80-byte CWBVH not claimed |
| Static BVH cooking | Leaf-size, bin-count, SAH stopping and scene-cache trials |
| Tier-C opacity alternatives | Exact classification/cells, approximate area shrink and alpha-test cap measured |
| Texture stochastic filtering / ray cones | Color-only, all-map and mip-footprint trials |
| Noise sequences / bounded GGX | Spatial blue-noise alternative and bounded VNDF measured |
| Spatial denoising | Current-frame sparse bilateral trial |
| Shared-memory stack / classic VNDF / sky sampling | Already tried earlier; historical results are not mixed into current three-bounce numbers |
| Temporal stages, reservoirs/history, radiance caches | Excluded by requested history/rearchitecture constraints |
| Screen-space trace chain, material-sorted wavefront queues, primary resolve | Require new pass/buffer architecture; excluded |
| Half-resolution rays/upscalers | Excluded by resolution/reconstruction constraint |
| Hardware RT/SER/hardware opacity micromaps/DLSS | Not applicable to this Iris Xe compute path |
| Dynamic H-PLOC, skinning, streaming, far-field HLOD, light-list infrastructure | Current static Sponza scene lacks these systems; infrastructure work rather than a local optimization |
| Watertight intersections / firefly clamps | Robustness or biased image-quality changes, not a reduction of current tracing work; not presented as speed wins |
| Native 16-bit integer counters | WGSL does not provide the report’s native u16 shader-counter representation |

## Reproduce

```sh
cargo build --release --locked
training/.venv/bin/python -m pip install -r performance/local-sweep/requirements.txt
training/.venv/bin/python -u performance/local-sweep/run.py
python3 performance/local-sweep/report.py
```

Completed captures are reused only when binary fingerprint, parameters and output files match. Every render has a record, raw radiance, PNG, timing and log under results/. See combined.json, combined.csv, screen.json, quality.json, combinations.json, startup.json and filters.json. No model API is used by the runner.

## Validation

All 27 default CPU/GPU tests passed after the sweep, including the Sponza brute-force comparison. Additional grid/parallel/inside-ray tests passed for four-wide, eight-wide and compressed eight-wide traversal; software-opacity cutout/visibility checks and bounded-VNDF energy checks also passed. Fixed-count validation confirmed raw identity for the recommended two-flag combination in both views. The pre-sweep control matched the unmodified baseline pixels. Scene-cache startup captures were also byte-identical across modes.

## Sources

- Supplied Zet architecture draft (local attachment), especially sections 5–7, 10–13.
- [Eto and Tokuyoshi, Bounded VNDF Sampling for Smith–GGX Reflections](https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf). The implemented sampler/PDF follows the bounded spherical-cap construction.
- [Heitz, GGX visible normals](https://jcgt.org/published/0007/04/01/) for the previous VNDF baseline and terminology.

## Per-view quality and switches

| Technique | View 1 GPU speedup | View 19 GPU speedup | View 1 MSE change | View 19 MSE change | Flags |
|---|---:|---:|---:|---:|---|
| Four-wide BVH | 0.565× | 0.569× | not separately tested | not separately tested | `--bvh4` |
| Eight-wide BVH | 0.604× | 0.568× | not separately tested | not separately tested | `--bvh8` |
| Compressed eight-wide BVH | 0.812× | 0.783× | not separately tested | not separately tested | `--quantized-bvh` |
| BVH leaf size 8 | 0.786× | 0.808× | not separately tested | not separately tested | `--bvh-leaf 8` |
| BVH leaf size 16 | 0.609× | 0.641× | not separately tested | not separately tested | `--bvh-leaf 16` |
| 32-bin SAH | 1.056× | 1.035× | -2.6% | -2.2% | `--sah32` |
| SAH leaf-cost stopping | 1.107× | 1.079× | not separately tested | not separately tested | `--sah-stop` |
| Packed shading attributes | 1.055× | 1.078× | -1.8% | -4.9% | `--packed-shading` |
| Stochastic color filtering | 1.008× | 1.016× | +2.9% | -2.2% | `--stochastic-color` |
| Stochastic all-map filtering | 1.059× | 1.056× | +1.0% | -4.1% | `--stochastic-all` |
| FP16 path radiance/throughput | 0.995× | 1.004× | +0.1% | -2.2% | `--fp16-state` |
| Packed ray directions | 1.005× | 1.015× | +2.2% | -0.0% | `--packed-directions` |
| Skip final-hit shading/scatter | 1.038× | 1.047× | not separately tested | not separately tested | `--skip-final-scatter` |
| Material opacity classification | 1.002× | 1.004× | not separately tested | not separately tested | `--classify-opacity` |
| Conservative software opacity cells | 1.000× | 0.983× | not separately tested | not separately tested | `--opacity-micromap --scene-cache` |
| Diffuse alpha-area approximation | 0.995× | 1.002× | +0.2% | +0.1% | `--approximate-alpha` |
| Cap alpha texture tests at four | 0.997× | 0.977× | +0.9% | +0.8% | `--alpha-cap4` |
| Spatial blue-noise / R2 jitter | 1.010× | 1.004× | -0.6% | -2.4% | `--blue-noise-jitter` |
| Bounded GGX VNDF | 0.915× | 0.918× | +3.6% | -3.6% | `--bounded-vndf` |
| Ray-cone texture mip selection | 0.954× | 0.998× | +4.7% | -0.4% | `--ray-cones --scene-cache` |