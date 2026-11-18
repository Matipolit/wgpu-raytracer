# Frame-time pass, 2026-11-18

## Inputs
target: the real-time renderer's frame (compute path tracer, ReSTIR DI + GI, temporal
denoiser, TAAU), `src/raytracer/*.wgsl` and its host code.
workload: Sponza atrium and night Bistro, Low preset (render scale 0.5, td-half16, no
bounce reuse) and Medium (0.67, td2-half24), 2 bounces, still camera, 241 frames.
primary metric: GPU ms per frame (median over the run, `RT_PASS_TIMES=1`), on the
Steam Deck at 1280x800. The Iris Xe laptop is the second target; its timings are only
usable when nobody is working on it.
out of bounds: image changes beyond noise (checked with
`training/reference/raw_frames.py` against an unbiased reference) unless stated.
budget: stop at < 2% per change.

## Baseline (00bdfa7)
command (on the Deck, in ~/wgpu-raytracer-test):
`RT_PASS_TIMES=1 target/release/weekend-raytracer-wgpu SCENE --size 1280x800 --fastest
--bounces 2 --samples 241 --no-calibrate --render-output /tmp/x.png PRESET`
(harness: one line per run, interleaved A/B; calibration on the Deck picks the default
kernel in 43 of 49 cached cases, so `--no-calibrate` is representative.)
machine: Steam Deck (AMD Van Gogh, RADV, Mesa 26.2-devel), SteamOS 3.9.
metric (ms; runs repeat to 0.1-0.5 ms):

| scene, preset | total | raster | lamps | trace | bounce reuse | denoiser | upscale | bloom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sponza Low | 19.05 | 0.28 | | 14.40 | | 2.45 | 1.30 | 0.52 |
| Sponza Medium | 34.81 | 0.30 | | 19.72 | 7.52 | 4.89 | 1.63 | 0.54 |
| night Bistro Low | 45.53 | 1.07 | 5.37 | 34.72 | | 2.46 | 1.30 | 0.51 |
| night Bistro Medium | 79.46 | 1.25 | 9.48 | 47.91 | 13.77 | 4.82 | 1.52 | 0.52 |

## Data inventory (hot path, per frame)
Low renders 640x400 (256k pixels), Medium 857x536 (459k).
- image: 12 B per pixel; G-buffer: 48 B (+16 B emission) per pixel, written by the
  trace pass, read by the bounce reuse and the denoiser.
- lamp reservoirs: 24 B per slot; two sets per pixel and one per 2x2 block.
- bounce records: 48 B per pixel, written by every traced pixel; final bounce
  reservoirs: 32 B per pixel, two sets.
- lamps: night Bistro has thousands of emissive triangles. A lamp point needed the
  triangle (16 B), three vertices (64 B each, scattered) and the material: five cache
  lines. Per pixel the lamp reuse merges 7 reservoirs, the lamp pass draws 4
  candidates per block, bounce vertices 2: about 50 lines per pixel.
- BVH: 4-wide, tuned in earlier passes.

## Napkin math
- Sponza Low trace: 14.4 ms / 256k pixels = 56 ns per pixel for about 2 rays plus
  shading, about 15-20 ns per ray on the Deck. Night Bistro (2.8 M triangles, alpha
  tested foliage): about 45 ns per ray.
- Bounce reuse, Sponza Medium: 7.5 ms / 459k = 16 ns per pixel for about 200 B read
  and written and a quarter of a ray: memory bound.

## Hotspots (ranked, Deck, ablations)
1. trace pass, 60-75% of the frame. Sun shadow rays: 3.7 of 14.4 ms (Sponza Low).
   Lamp reservoir merge with 6 neighbours: 3.5 of 34.6 ms (night Low).
2. bounce reuse, 17-22% at Medium. Neighbours: 2.2 ms of 7.6 (Sponza); a visibility
   ray for every temporal pick would add 1.5 ms (it is traced every 4th frame).
3. lamp pass, 12% in night Bistro: the fresh winner's shadow ray is 4.2 of 5.4 ms.
4. denoiser 2.5 (Low) / 4.8 ms (Medium), upscale 1.3-1.5, bloom 0.5, raster 0.3-1.2.

## Changes

### 0. The last vertex had no light (bug)  [FIXED]
`--fastest` includes SKIP_FINAL_SCATTER, which returned the path's last hit without a
normal; the light sample there then evaluated to zero. Without bounce reuse (the Low
preset) the bounce brought back only sky and emission: the median 8x8 block of Sponza
was at 4% of its brightness (mean 0.86). The last vertex is now lit with its geometric
normal (FLAT_LAST; mean 1.001, median block 1.001 against full shading normals), and
the bounce reuse sample's secondary point uses the same resolve (Medium unchanged:
0.993 vs 0.992).
cost of the fix without reuse: Sponza Low 19.05 -> 25.40 ms, night Low 45.43 -> 64.50.
with bounce reuse (per 2x2 block): Sponza 19.12, night 45.14 ms. **Low now uses bounce
reuse**: the old frame time, correct light, less noise.
files: mesh.wgsl, raytracer.wgsl, mod.rs, launcher.rs

### 1. Lamp records  [KEPT]
hypothesis: a lamp point costs five scattered cache lines; one 64-byte record per lamp
(corner + material, edge + weight, edge + triangle, unit normal) makes it one.
predicted: part of the 3.5 ms merge cost and of the lamp pass.
result: night Medium 79.75 -> 77.99 ms (-2.2%: lamps 9.45 -> 8.47, trace 48.16 ->
46.97); night Low (no reuse) 64.50 -> 62.48 (-3.1%). Image: 16-frame mean per block
1.000 of before (10th-90th percentile 0.97-1.03, the noise).
files: mesh.rs (lamp_table), raytracer.wgsl (lampAt, sampleLampPoint)

### 2. Bounce-reuse neighbours only for short histories  [KEPT]
hypothesis: with a long history (M >= 8) the 4 neighbours (48 B scattered reads each,
and a visibility ray when one wins) add nothing.
result: Sponza Medium 33.94 -> 32.23 ms (-5.0%: reuse 7.53 -> 5.71); night Medium
77.86 -> 76.00 (-2.4%); Low with reuse: Sponza 18.89 -> 18.49, night 44.56 -> 44.39.
noise (laptop, 640x360, against a reference), always -> short histories only:
Sponza per pixel 0.41 -> 0.35, per block 0.070 -> 0.056, block error 0.068 -> 0.051;
day Bistro 0.40 -> 0.32, 0.078 -> 0.061, block error 0.042 -> 0.042.
cost: day Bistro's mean brightness 0.987 -> 0.955. Temporal reuse alone is about 4%
dark there (0.947 with no neighbours); the neighbours' bright bias was hiding it.
Not the material's specular terms (tried: no change). Open.
files: raytracer.wgsl (GI_SPATIAL_BELOW), mod.rs

### 3. Fewer lamp neighbours  [REVERTED]
6 -> 3 neighbours: night Low 44.39 -> 43.01 ms (-3.1%), but noise per block 0.130 ->
0.157 (+21%). Not worth it.

### 4. The material's F0 / F90 in the bounce reuse  [REVERTED]
No effect on day Bistro's brightness (0.955 either way).

### 5. Coarse material at the last vertex  [REVERTED]
hypothesis: the last vertex's material is read from random texels of full-size
textures; a 16 x 16 mip would stay in the cache.
result: Sponza Medium 31.91 -> 31.94 ms, night 75.90 -> 75.83, day 81.82 -> 82.07: no
effect (image unchanged too). Those texture reads are not a bottleneck.

## Breakdown after changes 0-2 (Deck, Medium, ms)
- Sponza (31.9): trace 18.9 = primary shading 4.2 + primary sun rays 4.4 + bounce ray
  and resolve 5.2 + light sample at the bounce vertex 5.1 (a quarter of the pixels
  trace a bounce: 45 ns each for the ray, 44 ns for the light sample); reuse 5.7;
  denoiser 4.8; upscale 1.5. The bounce costs 15.5 ms in total, 48% of the frame.
- night Bistro (76.0): trace 47.1, of which the bounce 21.1 (183 ns per bounce);
  reuse 12.1 (the temporal picks' visibility rays, every 4th frame, about 3 ms);
  lamps 8.5; denoiser 4.8. The bounce costs 32.7 ms, 43% of the frame, for about 1%
  of this scene's light.
- the lamp reservoir merge with 6 neighbours is down to 0.4 ms (it was 3.5).

## Final (this pass)
Deck, 1280x800, baseline 00bdfa7 -> now:
- Sponza Medium 33.9 -> 31.9 ms (-5.9%); night Bistro Medium 79.8 -> 76.0 (-4.8%).
- Low: 19.05 -> 18.5 ms (Sponza) and 45.4 -> 44.4 (night), now with the bounce light
  it was missing.
kept: lamp records; bounce-reuse neighbours for short histories; the last-vertex fix
with bounce reuse in Low.
reverted: fewer lamp neighbours (noise), coarse last-vertex material (no effect),
material F0/F90 in the reuse (no effect).
next steps (payoff / cost):
1. Sun shadow rays from a shadow map where it is unambiguous (a min/max depth
   pyramid from the sun; rays only near shadow edges): primary sun rays are 4.4 ms
   and bounce-vertex light samples 5.1 ms of Sponza Medium's 31.9. Up to -15% in sun
   scenes; a new raster pass and classification, a day or two.
2. Fewer bounce owners (one per 2x4 or 3x3 pixels instead of 2x2): the bounce is
   43-48% of the frame; halving the owners is about -13 to -15%, at a slower response
   of bounce light to change. Needs a motion-quality check.
3. Night Bistro's reuse pass: 11.6 ms without neighbours against 5.2 in Sponza for the
   same pixel count; only 3 ms are explained (recheck rays). Unexplained.
