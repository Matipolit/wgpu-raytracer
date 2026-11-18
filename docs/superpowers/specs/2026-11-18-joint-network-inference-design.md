# Joint denoise + upscale network in the renderer (proposal)

Status: proposal for review, not started. Written 2026-11-18 while the v3 tiers train
(scripts/overnight_v3.sh); decisions marked **(yours)** need an answer before building.

## Why

The v3 overnight run trains the joint network (training/upscale/model.py) in five
tiers, tiny (8 channels, 2 layers) to ultra (64 channels, 4 layers). Today the renderer
can only run the older temporal denoiser (render resolution) followed by the
non-learned TAAU upscaler (upscale.wgsl). The joint network replaces both: one pass
from 1-spp render-resolution samples to the output image. Until it runs in the
renderer, the trained tiers are only numbers in v3-summary.txt.

Goals: run any tier in the live app and in offline renders. The launcher picks the
largest tier that fits a frame-time budget on the current GPU. Use the fastest math
each GPU has. Output matches PyTorch (parity test, as training/temporal/parity.py does
for the temporal denoiser).

Non-goals: new training, a new architecture, native Vulkan in this step.

Anti-goal: a slower Steam Deck Low/Medium than today's temporal denoiser + TAAU
(Medium, Sponza: 4.9 + 1.5 ms).

## What the network needs at runtime

Per output pixel (model.py `step`):
1. **This frame's samples:** a Gaussian gather of the 4x4 render pixels around the
   output pixel, at their jittered positions, demodulated by the low albedo. The
   renderer already has this: upscale.wgsl gathers the same way, and the render-res
   G-buffer has the low albedo.
2. **Output-resolution guides:** albedo, distance, normal, roughness, motion (output
   pixels), metallic, kind. **The renderer has none.** Its G-buffer is at render
   resolution.
3. **History:** at output resolution, warped by the output motion and validated
   against the previous output guides (distance, normal, kind), as the temporal
   denoiser does.
4. **Network:** 18 features, a 2x2 average pool, LAYERS 3x3 convs at half output
   resolution, a nearest-neighbour upsample, then a 1x1 head of 28 outputs at full
   resolution: a temporal blend factor and three 9-tap softmax kernels at dilations
   1, 2, 4. The result is remodulated by the output albedo.

Steps 3 and 4 have the same structure as the temporal denoiser that
training/temporal/export.py already generates WGSL for, at render resolution. That
generator, including the f16 path from 2026-11-18 (91d851b), carries over almost
unchanged.

## Approaches

1. **WGSL kernels generated per tier, f16 where available (recommended).** A new
   training/upscale/export.py emits the gather/features, pool, conv, blend and filter
   kernels for a checkpoint, as the temporal exporter does, with f16 variants. The
   output-resolution guides come from a new pass (below). This runs on every GPU we
   have, and the f16 path reuses the work already measured on the Deck and Xe.
   Large/ultra on the RTX run at f16 CUDA-core rate, not on tensor cores.
2. **Approach 1, plus cooperative-matrix convs for large/ultra on the RTX.** Reach
   tensor cores through the native-Vulkan escape hatch in
   2026-11-18-native-vulkan-rt-design.md: `VK_KHR_cooperative_matrix` compute
   pipelines recorded into wgpu's command encoder. A second step after approach 1
   works; it only pays where convs dominate (large/ultra at 1440p and up).
3. **int8 convs with `dot4I8Packed`.** WGSL exposes packed int8 dot products, a 4x
   rate on RDNA2 and Xe (DP4A). This needs quantization-aware fine-tuning and
   calibration, and risks quality. A later experiment for the Deck's tiny/small
   tiers, not part of the first step.

## Design (approach 1)

### Output-resolution guide pass
- Reuse the visibility raster (visibility.rs/wgsl: triangle id + barycentrics per
  pixel) at **output** resolution, with the output jitter. It is a raster with no
  shading: 1.2 ms at 0.67 render scale on the Deck, so roughly 2.7 ms at full output
  resolution. To be measured.
- A resolve kernel turns each visibility texel into the 12 guides the network reads:
  albedo (base colour texture, ray-cone or mip-0 lookup), distance, shading normal,
  roughness, metallic, kind, and motion (current minus previous camera projection of
  the hit point; moving objects use their previous transforms, as the tracer does).
  It reuses mesh.wgsl's hit resolution (`meshResolveLocal`) and pbrSurface code.
- Alternative **(yours)**: skip the pass and feed render-resolution guides,
  nearest-sampled. Cheaper, but the network was trained with true full-resolution
  albedo; texture detail would come out blurred and it would need retraining.

### Inference kernels (training/upscale/export.py)
- `csJnFeatures` (output res): gather plus demodulation (as upscale.wgsl), history
  warp and validation, the 18 features packed as f16, and the warp record.
- `csJnPool`, `csJnConv{1..LAYERS}` (half res): generated as in
  training/temporal/export.py, with weights in a uniform buffer and f16 variants.
  Tiers with more than about 32 channels exceed the 64 KiB uniform limit, so they
  take weights from a read-only storage buffer.
- `csJnBlend`, `csJnFilter{0,1,2}` (output res): head rows plus softmax taps. The last
  one remodulates, writes history and guides, and writes the output image.
- The pipeline replaces the temporal denoiser + upscaler stages when
  `--joint-model FILE.wgsl` is given. Display, bloom and exposure read the output
  image as they read the upscaler's today.

### Tier choice
- Export all five tiers. At startup (next to the auto-tuner's calibration) time each
  tier at the target resolution and keep the largest under a budget: the launcher's
  frame-time target minus the measured trace time. Cache the result with the
  calibration key.
- Launcher presets map to budgets, not tiers **(yours: budgets per preset)**.

### Verification
- Parity: an extended training/temporal/parity.py runs PyTorch `step` on a captured
  sequence and compares with the renderer's output frame by frame. Target: 1 frame
  ≥ 50 dB; the f16 temporal denoiser measured 66 dB.
- Speed: RT_PASS_TIMES on the Deck (Low/Medium), Xe and the RTX for each tier.
- Quality: evaluate.py numbers for the chosen tiers vs today's temporal denoiser + TAAU
  on the v3-eval set, at the same scale.

## Open questions (yours)

1. Output-resolution guide pass (recommended) or render-resolution guides with
   retraining?
2. Frame-time budgets per launcher preset (e.g. Low 33 ms, Medium 25 ms, High 16 ms
   on the target device), or pick tiers by hand per preset?
3. Should the Deck and the laptop keep the current temporal denoiser + TAAU as a
   fallback preset until the joint network beats it there?

## Left to the builder

Kernel workgroup shapes, buffer packing, how the exporter shares code with the
temporal exporter, and where the resolve kernel lives (visibility.wgsl or a new file).
