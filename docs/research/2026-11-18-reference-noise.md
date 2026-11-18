# Why 2048 spp is still noisy, and whether the path tracer is at fault

Written 2026-11-18. Measurements with `--reference-output` (docs/reference-frames.md)
and `training/reference/noise.py`; literature by a research agent (sources below).
[measured] is ours, [paper] is stated in a source, [inference] is reasoning.

## Answer

Nothing is broken, but the estimator is inefficient for these two views, and 2048 spp
is far from converged there.
- The estimator is unbiased: sky importance, visible-normal sampling, no Russian
  roulette and 16 or 64 lamp candidates all give the same mean (within 0.3%).
  [measured]
- The noise per sample is very high where light arrives indirectly. The error of a
  pixel after N samples is (relative standard deviation of one sample) / sqrt(N).
  [measured, laptop, 256 spp, 360x240 windows of a 1080p frame]

| view | rel. std of one sample (median pixel) | samples for 1% error | error at 2048 spp |
|---|---:|---:|---:|
| Sponza, sunlit floor | 0.11 | 131 | 0.2% |
| Sponza, shaded arcades (the comparison window) | 4.06 | 165,000 | 9-11% |
| night Bistro (the comparison window) | 3.77 | 142,000 | 7-8% |

- "2048 spp is practically perfect" holds for directly lit surfaces (first row). The
  windows we compared contain none.
- Earlier figures in log units (0.013 for Sponza) looked small because
  log1p(exposure * x) compresses: at mid-grey 0.013 log units is about 9% of the
  pixel's value.

## Where the noise comes from

**Sponza (shaded window).** [measured]
- With 1 bounce the window is black (mean 0.0003 of 0.129): no direct sun reaches it.
  2 bounces give 84% of the light, 3 give 98%.
- Without the sun the mean is 0.025: the sky gives 20% of the light and about 11% of
  the variance. The other 89% of the variance is sunlight that reaches the pixel
  after one bounce off a sunlit surface.
- That path is found by a BSDF-sampled bounce that must land on a sunlit surface. A
  relative standard deviation of 4 corresponds to about 6% of bounce directions
  doing so. [inference] The path tracer has no way to aim the bounce.
- Small levers: sky importance sampling -4% noise, visible-normal sampling -5%,
  bounded visible normals -7%, no Russian roulette -2%, all together about -9%.
  Bounces 3, 4 and 8 give the same relative noise.

**Night Bistro.** [measured]
- Direct light from the lamps alone has a relative standard deviation of 4.8 per
  sample: lamps are drawn from one global table over the whole scene, then the best
  of 4 candidates is kept.
- More candidates help the typical pixel: 4 -> 16 -> 64 candidates give 3.77 -> 2.87
  -> 2.45 (2.4x fewer samples at 64), direct light only 4.77 -> 2.65. The worst
  tenth of pixels barely moves (7.96 -> 7.30): occluded lamps and indirect light.

## What published work does

- Reference sample counts [paper]: Isik et al. 2021 2048 spp (4096 for test images);
  Chaitanya et al. 2017 2000-4000; ReSTIR GI 8192; Moreau et al. 2019 (Bistro,
  direct only) 10,000; Noisebase 6144, as the median of three 2048-spp renders "to
  suppress fireflies", and its training references are denoised with OIDN on top
  because "some noise and fireflies are present".
- Fireflies survive 100,000 spp with a plain mean (G-MoN paper); an 18-hour path
  traced reference "still exhibits residual noise" (Practical Path Guiding). [paper]
- Noise2Noise: training on 64-spp noisy targets reached 32.42 dB against 32.95 dB
  with clean 131k-spp targets. [paper] Noisy targets cost little for a per-frame
  loss; our problem is the temporal term, which compares two noisy targets.
- Many lights [paper]: a light BVH against uniform selection in Bistro (direct, 1
  spp): 8x lower MSE (Moreau et al.); PBRT-v4's light BVH is 2.4x better than power
  sampling. ReSTIR DI found a light BVH "generally under-performs" resampling 32
  power-sampled candidates.
- Path guiding [paper]: 11x lower MSE at equal time in an indirect-lit kitchen
  (Mueller et al. 2017). A GPU guiding paper counts Sponza as direct-light dominated
  and still gains there.
- Owen-scrambled Sobol [paper]: large gains for direct light at low sample counts
  (RMSE 0.020 -> 0.0017 at 16 spp), none for global illumination with sub-pixel
  texture at 64 spp; PBRT-v4 reports about 10% lower MSE on a glossy scene.
- Visible-normal sampling bounds the specular sample weight to [0, 1]; sampling the
  normal distribution does not (Heitz & d'Eon 2014). [paper]
- Not verified first-hand: KPCN and Hasselgren et al. sample counts; no figures found
  for Thomas et al. 2022, OIDN's training data or DLSS Ray Reconstruction.

## What would bring references closer to converged

| change | expected gain | cost |
|---|---|---|
| Path guiding of the first bounce (learned directional histograms per region) | Sponza-type indirect: up to about 10x fewer samples [paper, other scenes] | large: new sampler, pdf, training pass |
| 64 lamp candidates, or a light tree | night: 2.4x fewer samples at the median [measured]; up to 8x for a tree [paper] | candidates: one constant; tree: medium |
| Sky importance + bounded visible normals in reference mode | about 15% fewer samples [measured] | two flags |
| Bounded denoise (finish.py, already built) | typical pixel's noise about 3x lower [measured] | done |
| Median of independent renders for fireflies (Noisebase) | removes dots, slight darkening | small |
| More samples on the RTX PC | error falls with sqrt(samples) | time: 1% at the median pixel needs about 80x the samples of 2048 spp |

No pure Monte Carlo estimator makes 2048 spp noise-free for indirect light: even at a
relative standard deviation of 1 per sample (a very good estimator) the error at
2048 spp is 2.2%. [inference] References for this kind of lighting are made
converged-looking by many more samples, a denoise pass, or both.

## Sources
- Isik et al. 2021: https://www.mustafaisik.net/pdfs/isik2021-anf.pdf
- Noisebase: https://balint.io/noisebase/datasets/sampleset_v1.html ;
  Balint et al. 2023: https://balint.io/nppd/nppd_paper.pdf
- Chaitanya et al. 2017:
  https://research.nvidia.com/sites/default/files/publications/dnn_denoise_author.pdf
- ReSTIR GI: https://d1qx31qr3h6wln.cloudfront.net/publications/ReSTIR%20GI.pdf
- Moreau et al. 2019:
  https://research.nvidia.com/sites/default/files/pubs/2019-07_Dynamic-Many-Light-Sampling//MPC19.pdf
- ReSTIR DI:
  https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf
- GRIS: https://d1qx31qr3h6wln.cloudfront.net/publications/sig22_GRIS.pdf
- Noise2Noise: https://arxiv.org/pdf/1803.04189
- G-MoN: https://diglib.eg.org/server/api/core/bitstreams/667856b1-32c6-4cd6-a288-1b246e526ae1/content
- Practical Path Guiding: https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf
- GPU path guiding: https://arxiv.org/pdf/2405.06997
- PBRT-v4: https://pbr-book.org/4ed/Light_Sources/Light_Sampling ,
  https://pbr-book.org/4ed/Light_Sources/Infinite_Area_Lights ,
  https://pbr-book.org/4ed/Sampling_and_Reconstruction/Sobol_Samplers
- Burley 2020: https://jcgt.org/published/0009/04/01/paper.pdf
- Heitz & d'Eon 2014: https://hal.inria.fr/hal-00996995v2/document
