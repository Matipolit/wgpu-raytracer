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
`RT_RESTIR_VISIBILITY_REUSE=1` restores the old behaviour.

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

## Bounded visible normals by default; lamp candidates (2026-11-18)

Raw 1-spp frames as the real-time renderer traces them (ReSTIR DI + GI, 2 bounces, no
denoiser or upscaler, 960x540, still camera, Iris Xe): `--render-output x.png
--raw-frames FILE` keeps every 32nd frame, `training/reference/raw_frames.py`
measures the noise of one frame around the mean of 16. Repeated runs agree to 1%.

| scene | setting | noise per pixel (median) | noise per 8x8 block | pixels with no light | ms |
|---|---|---:|---:|---:|---:|
| Sponza | normal distribution (before) | 1.14 | 0.174 | 39.5% | 65.8 |
| Sponza | bounded visible normals | 0.64 | 0.099 | 19.4% | 66.8 |
| Bistro day | before | 1.04 | 0.158 | 29.5% | 208 |
| Bistro day | bounded visible normals | 0.59 | 0.094 | 15.7% | 203 |
| Bistro night | before | 1.73 | 0.356 | 32.7% | 188 |
| Bistro night | bounded visible normals | 1.67 | 0.347 | 17.4% | 184 |
| Bistro night | 16 / 4 lamp candidates | 1.70 | 0.341 | 31.1% | 186 |
| Bistro night | 32 / 8 lamp candidates | 1.69 | 0.334 | | 189 |
| Bistro night | 16 / 4 candidates + bounded | 1.64 | 0.334 | | 188 |

- **Bounded visible normals are now the default** (`--ndf-sampling` goes back).
  Sampling the normal distribution reflects many specular samples below the surface;
  such a pixel then gets no bounce and no ReSTIR GI that frame. Mean brightness is
  unchanged (within 1%). td2 + TAAU on still Sponza is unchanged (30.36 -> 30.40 dB,
  flicker 0.0010): the denoiser was hiding this noise already.
- **Lamp candidates stay at 4 / 2** (`--lamp-candidates N,M` or RT_LAMP_CANDIDATES):
  16 / 4 lowers night Bistro's noise 2-4%, 32 / 8 2-6%. Frame times were within
  run-to-run noise (179-188 ms for the same setting); the laptop was in use.
- Still 16-19% of lit pixels get no light in a frame with ReSTIR GI on.

## Not adopted

Alpha-test shortcuts (`--opacity-micromap`, `--approximate-alpha`, `--alpha-cap4`)
change nothing on night Bistro (76 +- 1 ms): the cost is traversal itself.
