# ReSTIR DI/GI on weak GPUs (2026-11-18)

Measurements behind the ReSTIR defaults. Frame times are GPU ms per frame from
`RT_PASS_TIMES=1` offline renders (241 frames, 1280x800 output, `--fastest`, 2 bounces,
td2-half24 denoiser) on the Steam Deck (RADV, wave32). Medium = 0.67 render scale.
Quality comes from captures (`training/temporal/capture.py`, 320x200, 8 frames, motion
0.03-0.3) compared with 4-bounce references: PSNR after exposure normalisation and
x/(1+x) per pixel and on 4x4 averages, and the mean brightness ratio.

## Lamp table after SBVH builds (bug, 9415ba0)

The SBVH triangle array repeats split triangles once per leaf. The lamp table counted
every repeat, so a lamp split k times was chosen k times as often as its pdf said: its
light counted k times in light sampling. Captures use `--sbvh`, so references and
inputs were both too bright on split lamps. ReSTIR amplified it: 1-spp inputs were
1.8-2x the reference. Fixed by listing each emitter once. **Data captured with `--sbvh`
before this commit has over-bright lamp lighting.**

## Visibility applied once (329fb5e)

Bitterli et al.'s "visibility reuse" stored occluded winners with W = 0 and their full
M. A zero-W history of M = 80 then dilutes the next frame's 4 fresh candidates to 4/84
of their weight, which darkens penumbras. Reservoirs now resample the visibility-free
target; the shadow ray only gates the shaded sample.

| scene | brightness (plain / old / new) | 4x4 PSNR old -> new |
|---|---|---|
| night Bistro | 0.996 / 0.968 / 0.986 | 27.41 -> 27.43 dB |
| fireplace room | 0.785 / 0.704 / 0.756 | 22.79 -> 24.01 dB |

The DI first pass no longer traces a shadow ray; the frame time is unchanged.
Superseded on 2026-11-18 (visibility-aware lamp reuse below): without any visibility
the reservoirs keep proposing hidden lamps. The old switch is gone.

## GI bounce per 2x2 block (9642570)

With `--fastest`, the last vertex is not shaded, so ReSTIR GI keeps the secondary
vertex shaded. That adds a light sample and a long shadow ray at the secondary hit of
every pixel (+41 ms on night Bistro Medium). Now one pixel per 2x2 block (rotating)
traces the bounce; the other three end at the primary surface and get their indirect
light from reuse (8 spatial taps). The trace grid is remapped into four
quarter-resolution planes, so a wave holds owners only or none.

| Medium | per-pixel GI | blocks | blocks + plane remap | no GI |
|---|---:|---:|---:|---:|
| night Bistro | 135.3 ms | 94.5 ms | 81.7 ms | 75.9 ms |
| Sponza | 51.1 ms | 37.6 ms | 31.8 ms | 30.0 ms |

Sponza 4x4 PSNR: 25.8 dB per-pixel GI, 24.6 dB blocks, 20.8 dB without GI. Night
Bistro gains nothing measurable from GI in this view (indirect is about 5% of the
light). `--restir-full` or RT_GI_FULL keep per-pixel GI (launcher High).

## GI temporal visibility (fa8c712)

A pixel's own reprojected GI history was tested when it was chosen, so it is re-tested
only every 4th frame (staggered over 2x2 pixels): a moving occluder leaks for at most
4 frames. GI pass on night Bistro: 16.9 -> 11.6 ms; Sponza: 8.1 -> 6.9 ms. Quality is
within 0.05 dB (Sponza, living room). Storing occluded GI picks with their weight
instead of W = 0 (RT_GI_VISIBILITY_REUSE=0) made no measurable difference.

## f16 denoiser convs (91d851b)

`export.py --f16` writes FILE.f16.wgsl: hidden layers typed vec4<f16>, conv weights
in a second uniform as f16 row pairs. The renderer picks it with SHADER_F16 +
SHADER_I16 (naga lowers the f16 bitcasts through 16-bit integers). One frame matches
f32 at 66 dB. td2-half24: Deck 5.41 -> 4.89 ms, Iris Xe 1.60 -> 1.42 ms. The gain is
smaller than packed math's 2x; the f32 head and filter kernels are a large share.

## Shared DI (opt-in, 3cbffad)

The per-pixel DI shadow ray is 26.8 of the 54 ms trace pass on night Bistro Medium.
`--di-shared` runs the spatial merge and the shadow ray once per 2x2 block; pixels on
a similar surface shade the block's sample. Night Bistro Medium: 75.7 -> 60.1 ms.
Quality: 4x4 PSNR -0.6 dB (Bistro) / -0.5 dB (fireplace), and the noise is visibly
2x2-correlated, which the current denoisers were not trained on, so it is not the
default. Trusting last frame's DI visibility instead (as GI does) saved only 1 ms:
the winner usually comes from a neighbour's reservoir.

## Raw-frame noise and bias (2026-11-18, 00bdfa7)

Method: raw 1-spp frames as the real-time renderer traces them (ReSTIR DI + GI, 2
bounces, no denoiser or upscaler, 960x540, still camera). `--render-output x.png
--raw-frames FILE` keeps every 32nd frame; `training/reference/raw_frames.py
--reference DIR` gives the noise of one frame around the mean of 16 (relative to an
unbiased reference, `--reference-output` with 2 bounces), the share of lit pixels
that got no light in a frame, and how far the 16-frame mean is from the reference per
8x8 block (bias plus leftover noise). Noise and bias runs on the RTX PC (they match
the laptop's to 1%); frame times on the Steam Deck (repeat to 0.5 ms).

Four changes, all defaults now:
1. **Bounded visible normals** (`--ndf-sampling` goes back). Sampling the normal
   distribution reflects many specular samples below the surface; the path dies.
2. **Null bounce samples** (RT_GI_NULL=0). A pixel whose bounce sample failed
   recorded nothing and skipped temporal and spatial reuse for the frame. It now
   counts as a sample that found nothing and the pixel still reuses.
3. **Unusable bounce samples dropped** (RT_GI_DROP=0). A neighbour's sample outside
   the Jacobian's range was counted as a sample without light, and a pick found
   hidden was kept with weight 0 and its full history length. Both dilute the good
   samples. Now the neighbour is skipped and the hidden pick forgotten.
4. **Visibility-aware lamp reuse** (RT_RESTIR_VIS=0). The winner of the fresh
   candidates gets a shadow ray (hidden: the candidates enter the history as samples
   that found nothing), and a final winner hidden at the shaded pixel is forgotten.
   Before, reservoirs resampled a target without visibility and kept proposing bright
   lamps behind walls: with direct light only, 72% of lit night Bistro pixels got no
   light in a frame.

| scene | | noise per pixel | noise per 8x8 block | no light | brightness | block error |
|---|---|---:|---:|---:|---:|---:|
| Sponza | before | 1.14 | 0.174 | 39.5% | | |
| Sponza | 1 | 0.63 | 0.098 | 19.4% | 0.99 | |
| Sponza | 1-2 | 0.40 | 0.062 | 9.2% | 0.987 | 0.080 |
| Sponza | all | 0.42 | 0.068 | 9.0% | 1.002 | 0.056 |
| Bistro day | before | 1.04 | 0.158 | 29.5% | | |
| Bistro day | 1 | 0.59 | 0.094 | 15.7% | | |
| Bistro day | 1-2 | 0.36 | 0.057 | 10.3% | 0.877 | 0.141 |
| Bistro day | all | 0.39 | 0.071 | 5.5% | 0.996 | 0.035 |
| Bistro night | 1-2 | 1.33 | 0.306 | 10.8% | 0.981 | 0.083 |
| Bistro night | all | 0.36 | 0.120 | 5.7% | 0.970 | 0.073 |
| fireplace room | 1-2 | 0.75 | 0.166 | 4.0% | 0.943 | 0.058 |
| fireplace room | all | 0.29 | 0.109 | 1.7% | 0.945 | 0.060 |

(Rows "before" and "1" normalize noise by the frames' own mean, the others by the
reference; brightness leaves out the brightest 1% of blocks.)

- Day Bistro was 12% too dark, up to 50% on facades and foliage: change 3. The block
  error of 0.035 is at the noise floor of the unbiased path tracer's 16-frame mean.
- Night Bistro's noise is direct lamp light: direct only 1.33, with the bounce 1.35.
  Lamp candidates barely matter (16 / 4: 2-4% less noise for 3-4.5 ms on the Deck;
  the default stays 4 / 2, `--lamp-candidates N,M`).
- Variants measured and dropped: fresh candidates only every 2nd / 4th / 8th frame
  (block noise 0.120 / 0.134 / 0.150 and a growing bright bias, little time saved);
  keeping part of a hidden lamp winner's history (more bias); not counting temporal
  bounce samples that cannot light the pixel (fixes day Bistro, 12% too bright in
  Sponza's shade); a normal check on the temporal bounce reservoir (no effect).

Frame time, Steam Deck, 1280x800 (ms; cumulative):

| scene, preset | before | 1 | 1-2 | 1-3 | all |
|---|---:|---:|---:|---:|---:|
| Sponza Low | 17.7 | 19.0 | 19.1 | 19.1 | 19.1 |
| Sponza Medium | 29.9 | 33.7 | 34.6 | 34.0 | 34.8 |
| night Bistro Low | 43.2 | 44.2 | 44.2 | 44.2 | 45.5 |
| night Bistro Medium | 75.9 | 78.1 | 79.4 | 79.0 | 79.5 |

- Change 1 costs the most (Sponza Medium +13%): fewer paths die at the first hit, so
  more bounce rays are traced. Changes 2 and 3 cost up to 1 ms.
- Change 4: the lamp pass grows by 4-7.5 ms and the trace pass shrinks by 3-7 ms
  (the shaded lamp is usually near and visible), +0.5 to +3% in total.
- Laptop (Iris Xe, 1280x720, two quiet runs each): Sponza Medium 33-35 -> 39-40 ms,
  night Bistro Low 42-43 -> 43-46 ms, Medium 73-83 -> 78 ms. Its timings while in
  use are unusable (the lamp pass alone read 4.4 or 18 ms).

Finished image (td2-half24 + TAAU, 1280x720 at 0.67, last of 96 still frames against
a finished reference, exposure from the log-average luminance), before -> all:

| scene | median error per pixel | median brightness | PSNR |
|---|---|---|---|
| Sponza | 0.191 -> 0.160 | 0.92 -> 1.00 | 20.85 -> 21.06 dB |
| Bistro day | 0.201 -> 0.130 | 0.84 -> 0.99 | 23.19 -> 24.13 dB |
| Bistro night | 0.175 -> 0.141 | 0.92 -> 0.98 | 14.55 -> 14.49 dB |
| fireplace room | 0.101 -> 0.097 | 0.97 -> 0.99 | 14.76 -> 16.11 dB |

Flicker (stability evaluation, still / slow / fast camera) is unchanged or slightly
lower in all four scenes. The denoisers were trained on the old noise.

The lamp and bounce reuse passes are now rebuilt with the calibrated kernel
configuration; they kept the default stack before.

## Not adopted

Alpha-test shortcuts (`--opacity-micromap`, `--approximate-alpha`, `--alpha-cap4`)
change nothing on night Bistro (76 +- 1 ms): the cost is traversal itself.
