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

## Research and measurements (2026-11-18)

A literature review (NRD, SVGF/A-SVGF, Thomas et al. HPG 2022, Hasselgren et al. EG
2020, Isik et al. 2021, DLSS-RR / FSR2 / XeSS / AMD Ray Regeneration docs, Arm NSS,
QRISP, NSRD, "Frozen History" 2026) and measurements on still captures found these
faults in v3, largest first:
1. **Remodulation by one jittered albedo sample**, never accumulated (measured). The
   final `x * albedo` applies a per-frame jittered albedo: texture and geometry edges
   flicker at full amplitude after the network. On a still 192x120 Sponza capture,
   this alone flickers 0.087 log units per frame, about 9x TAAU's total on the
   video; 30-37% of pixels change albedo between frames. Fix: accumulate in
   modulated (final colour) space, or keep a separate albedo history.
2. **Short, hard-reset history.** SVGF/NRD keep a per-pixel history length, blend at
   1/N up to about 60 frames, and shrink the length on doubtful matches instead of
   resetting. NRD feeds back the unfiltered accumulation, not the spatially filtered
   output, and reprojects with Catmull-Rom (bilinear blurs every frame). We reset
   7-14% of pixels per frame and validate with jittered depth and normals.
3. **Reference noise in the temporal loss** (measured). Still-camera 1024-spp
   references change 0.018 (Sponza) / 0.014 (fireplace) log units per frame, 3% and
   0.6% of the signal, and 7.7% at night: about twice TAAU's whole flicker. The
   temporal term cannot see flicker below that. Fix: denoise the references, weight
   the loss toward late frames.
4. **ReSTIR's temporally correlated noise** (AMD Ray Regeneration): blobs lasting
   several frames survive a short history. Fix: randomize ReSTIR's temporal reuse.
5. **Specular divided by base-colour albedo.** Production denoisers split diffuse and
   specular, or divide by a pre-integrated specular term.
6. **No exposure normalization** of inputs (DLSS/FSR2/XeSS all take exposure).

Costs in the literature: XeSS (DP4a) is about 4.3-4.9 ms at 1080p output on an Iris
Xe-class GPU, upscaling an already clean image. U-Net denoisers (Thomas et al.:
5.2 ms on an RTX 3070 with TensorRT int8) would take over 50 ms here. Only a tiny
per-pixel network driving fixed filters fits.

## Goal

One learned denoise + upscale stage that the user can run on the laptop.
- **Budget** (Iris Xe), for everything after the path tracer, including the
  output-resolution albedo and excluding bloom:
  - stretch target: 5 ms at 1920x1080 output, XeSS's cost, but also denoising
    1-spp input;
  - realistic: 6-8 ms at 1080p; first milestone 5 ms at 1280x720.
  - Today's pieces scaled to 1080p (render 960x540): td-half16 about 6 ms, a TAAU
    pass 2.5-3.5 ms, the raster about 1.9 ms, so about 10-11 ms. The stretch target
    needs every piece halved (f16, fused kernels, a leaner network).
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
  log2(age)/6) and a soft geometric-normal / plane-distance history check. The
  history is the unfiltered temporal accumulation (the a-trous levels filter a copy
  for output). Inputs exposure-normalized. Specular either split (separate signal) or
  divided by a pre-integrated specular term instead of the base colour. Output:
  demodulated radiance at render resolution plus its history.
- **Stage B, output resolution, one pass** (accumulates in final-colour space, so
  jittered albedo and edges are anti-aliased over time):
  - Gather the stage A output (demodulated) for the output pixel: 4x4 render pixels,
    Gaussian by jittered distance, the same gather as upscale.wgsl and model.py.
  - Remodulate this frame's gathered value by this frame's (jittered) output albedo.
  - Reproject the output history (final colour, packed f16, plus a history length)
    with Catmull-Rom.
  - Validate softly: plane distance and geometric normal on unjittered data; a
    doubtful match shrinks the history length instead of resetting it.
  - A tiny learned head (one hidden layer of 8, inputs: gather confidence,
    nearest-sample distance, history length and confidence, luminance difference)
    predicts the blend weight (around 1/N) and a neighbourhood clamp strength.
  - Write the image and the history.
  All weights fit in registers; no intermediate buffers.
- **Output albedo and guides:** the output-resolution visibility raster (already the
  single raster in joint mode) and an albedo fetch inside stage B. No guide buffer is
  written; stage B reads the raster directly.

### Training
- Data: as v3, with 64-frame sequences, references only for the last 16 frames (the
  first 48 warm the history), and `--di-shared` off. References denoised offline
  (Open Image Denoise at full quality) so the temporal target is clean. ReSTIR's
  temporal reuse randomized.
- Loss: per-frame L1 in exposure-normalized log space, plus the temporal term on the
  late frames (truncated backpropagation through the last 16), at all three render
  scales. The recurrence stays in f32.
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

## Order of work (agreed 2026-11-18)
1. **No retraining:** stage B as a non-learned pass (TAAU blend with history length,
   soft unjittered checks, Catmull-Rom) after today's td, remodulating by the
   output-resolution albedo and accumulating in final colour. Plus an in-renderer
   evaluation: final frames of a moving-camera sequence and their motion-compensated
   frame-to-frame change, so td + TAAU, the joint network and stage B are compared on
   what the app shows.
2. **Stage A network fixes and the learned stage B head**, with the new training data.
3. **Performance toward the 1080p stretch target:** f16, fused kernels, inline albedo
   resolve.

## Left to the builder
Exact feature lists, kernel fusion details, buffer packing, export code sharing with
the temporal exporter, the history-age encoding.
