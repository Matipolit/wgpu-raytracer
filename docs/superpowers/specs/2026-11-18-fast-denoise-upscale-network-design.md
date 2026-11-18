# Fast denoise + upscale network (design)

Status: for review. Written 2026-11-18. Replaces the plan in
2026-11-18-joint-network-inference-design.md: that network works in the renderer
(`--joint-model`), but it is too slow and too unstable to use.

## Why

The v3 joint network does its per-pixel work at output resolution: sample gather,
history warp, temporal blend and three a-trous filter levels. That is six passes over
every output pixel, and the passes are limited by memory traffic. On the laptop
(Iris Xe, 1280x720 output) it adds 16-22 ms per frame at any tier and render scale.
Today's temporal denoiser + TAAU adds 3.6 ms (Low) to 7.4 ms (Medium). Its videos also
flicker (measured 2026-11-18):
- The network kept a fixed history weight (alpha about 0.8-0.87), so its memory is
  5-8 frames. It was trained on 16-frame clips from a cold start, with no history-age
  input.
- 7-14% of pixels drop their history every frame, mostly from the normal check (dot
  >= 0.8 on normal-mapped shading normals under sub-pixel jitter).

## Goal

One learned denoise + upscale stage that the user can run on the laptop.
- **Budget** (Iris Xe, 1280x720 output): 5 ms target, 6-8 ms acceptable, for
  everything after the path tracer. That includes the output-resolution albedo and
  excludes bloom.
- **Image:** at least as good as td + TAAU at the same budget, in PSNR on the
  v3-style evaluation set. Stable: long-video flicker at or below TAAU's.
- **Integration:** a launcher denoiser choice, like the temporal models. The Deck and
  the RTX use the same path (bigger tiers where the budget allows).

Non-goals: tensor-core or native-Vulkan inference; a network for the 1.0 render scale
(High keeps td + TAAU or no upscaling).

Anti-goals: a stage that is faster but flickers more than TAAU; a network that only
works at one render scale.

## Approaches

1. **Render-resolution temporal network + one output-resolution pass (chosen).** The
   learned temporal denoise runs at render resolution, as td-half16 does today: 2.4 ms
   at 0.5, 4.9 ms at 0.67. One output-resolution pass then:
   - gathers the denoised, demodulated render-resolution samples at their jittered
     positions;
   - blends them into an output-resolution history with a few learned per-pixel
     weights (a small head over cheap features);
   - remodulates by the output-resolution albedo.
   That pass costs about TAAU's 1.1-1.5 ms plus about 0.85 ms of raster for the albedo
   and edge guides. Estimated totals: about 5 ms at 0.5, 6.5-7.5 ms at 0.67 (f16
   convs, 12-16 channels).
2. **Today's joint network with fewer, fused output-resolution passes.** Even with one
   output pass, its features (the 18-channel gather and warp) and head stay at output
   resolution. The floor is about 8-10 ms on the laptop. Rejected: over budget at
   both scales.
3. **Keep td + TAAU and only retrain td for stability.** Fastest (3.6 / 7.4 ms), but
   TAAU blurs texture detail it never sees: the albedo is upsampled from render
   resolution. Kept as the fallback and as the baseline to beat.

## Design (approach 1)

### Network
- **Stage A, render resolution:** the temporal denoiser's structure (training/temporal
  model.py): demodulated 1-spp input, reprojected history, features, a 2x2 pool, 2-3
  convs of 12-16 channels at half render resolution (f16), a blend and 2-3 a-trous
  levels. New inputs: history age (frames since the last reset, capped at 64, as
  log2(age)/6) and a geometric-normal history check. Output: demodulated radiance at
  render resolution plus its history.
- **Stage B, output resolution, one pass:**
  - Gather the stage A output for the output pixel: 4x4 render pixels, Gaussian by
    jittered distance, the same gather as upscale.wgsl and model.py.
  - Reproject the output history (demodulated rgb, 8 bytes, packed f16) by the
    output-resolution motion.
  - Validate the history by distance and geometric normal.
  - A tiny learned head (one hidden layer of 8, inputs: gather confidence,
    nearest-sample distance, history age and validity, luminance difference) predicts
    the blend weight and a neighbourhood clamp strength.
  - Remodulate by the output albedo; write the image and the history.
  All weights fit in registers; no intermediate buffers.
- **Output albedo and guides:** the output-resolution visibility raster (already the
  single raster in joint mode) and an albedo fetch inside stage B. No guide buffer is
  written; stage B reads the raster directly.

### Training
- Data: as v3, with 64-frame sequences, references only for the last 16 frames (the
  first 48 warm the history), and `--di-shared` off.
- Loss: per-frame L1 in log space plus the existing temporal term on the late frames
  (truncated backpropagation through the last 16), at all three render scales.
- Tiers: 12 and 16 channels at 2 and 3 convs; tier choice by measured time on the
  target GPU, as the launcher already does by preset.

### Verification
- Parity: WGSL vs PyTorch test vectors (training/upscale/parity_vector.py pattern),
  both stages, both precisions.
- Speed: RT_PASS_TIMES on the laptop at 0.5 and 0.67, 1280x720, on AC power,
  interleaved against td + TAAU.
- Quality: PSNR vs td + TAAU (not raw TAAU) on the evaluation set, plus the long-video
  history-trust and flicker measurement (the 2026-11-18 diagnostic) on 300-frame
  videos.

## Calls made in writing this (correct me)
- **Training is end to end:** one model, an output-resolution loss, truncated
  backpropagation through both histories. The capture format already holds
  render-resolution inputs and full-resolution references, so there is no separate
  stage A run.
- **Quality bar:** at the same preset (Low vs td-half16 + TAAU, Medium vs td2-half24 +
  TAAU) within the time budget, not "equal milliseconds".
- **Stability gate:** the 2026-11-18 video diagnostic (history weight, reset rate,
  motion-compensated frame-to-frame change) becomes a committed script. The gate is
  frame-to-frame change no higher than td + TAAU's on the same 300-frame videos.
- **Data:** recaptured at the current commit with the runtime defaults (ReSTIR DI+GI,
  GI blocks, no `--di-shared`), so inputs match what the app renders, including
  ReSTIR's remaining darkening.
- **`--joint-model` stays** until the new network ships; then the v3 joint networks go,
  and stage B reuses joint mode's single output-resolution raster.

## Decisions (yours)
1. Which preset must reach 5 ms: Low (0.5, likely) or Medium (0.67, likely 6.5-7.5 ms)?
2. Should stage B also run after today's td (no retraining of stage A) as a first step?
   It would show the output-pass gain alone within a day, before an overnight run.

## Left to the builder
Exact feature lists, kernel fusion details, buffer packing, export code sharing with
the temporal exporter, the history-age encoding.
