# Reference frames

Written 2026-11-18. How training and evaluation targets are rendered:
`--reference-output DIR` (src/reference.rs) and `training/reference/finish.py`.
Status: single still frames; not yet wired into `--capture` (see "Not done").

## What was wrong with "N spp at every pixel"

The old references (`--capture --reference-spp 1024`, earlier 2048) traced the same
number of samples at every pixel and stored the plain mean:
- **Still noisy.** At 2048 spp the median pixel of the shaded Sponza window is off by
  about 11% of its value (0.013 log units; log units understate relative error about
  7x at mid-grey), and by 7% in night Bistro. Why, and what would change it:
  docs/research/2026-11-18-reference-noise.md. Still-camera 1024-spp references
  changed 0.6-7.7% of the signal from frame to frame
  (docs/research/2026-11-18-neural-denoise-upscale-literature.md 1.4). The temporal
  loss cannot see flicker smaller than that.
- **No error estimate.** A single mean says nothing about how wrong it is, so noise
  could neither be measured nor targeted.
- **Samples spent evenly.** Sky and directly lit walls converge in tens of samples;
  dark, indirectly lit regions need thousands.
- **Fireflies stay.** One rare bright sample remains a bright dot in the mean.

## Design

1. **Estimator: the plain path tracer, nothing biased.** ReSTIR off, no approximation
   flags, light sampling with MIS at every vertex, Russian roulette. Lamp sampling
   draws 16 candidates at the primary hit and 4 deeper (`RT_LAMP_CANDIDATES`,
   `RT_LAMP_CANDIDATES_DEEP`; 4 and 2 in the real-time renderer): resampling stays
   unbiased with any candidate count, and night Bistro's error fell 11% at equal time.
2. **Eight independent groups of samples.** Every pixel gets a mean per group, so its
   standard error is measured: std(group means) / sqrt(8).
3. **Adaptive sampling by tile (120 px).** After a first pass over the frame, each
   round samples only the tiles above the target error and within 0.6 of the worst
   tile, until every tile is below `--reference-error` or `--reference-seconds` run
   out. A tile's error is the 95th percentile of its pixels' standard error in
   exposure-normalized log space, the space of the training loss:
   exposure * se / (1 + exposure * mean), exposure = 0.18 / mean luminance.
4. **Pixel-integrated guides.** Albedo and normal are averaged over all samples of a
   pixel (shader override `REFERENCE_GUIDES`), not taken from one jittered sample.
5. **Finish (finish.py):**
   - *Firefly rejection*, extreme outliers only: a group above twice the other
     groups' mean plus 12 of their standard deviations is dropped for that pixel.
   - *Bounded denoise.* The two halves of the groups are independent images A and B.
     A is filtered with non-local-means weights measured on B and B with weights from
     A (Rousselle et al. 2012), with distances normalized by the measured variance.
     The filter runs on lighting (image / albedo) with a weight on the normal, and
     the albedo is multiplied back, so texture is not filtered.
   - *The bound:* the result may differ from a pixel's mean by at most 2 standard
     errors. Where the image is converged the filter changes nothing; it can never
     invent or remove more than the measured uncertainty.

Dispatches are 8 samples over one tile: the Intel driver resets a dispatch that runs
longer than about half a second and silently drops its work.

## Measuring a reference without a ground truth

Two independent renders of the same view (old and new) estimate the same image, so in
log space: rms(old - new mean)^2 = error_old^2 + error_new^2, and the new mean's error
is known from its groups. For the finished image, the filter's change c = final - mean
is compared with d = old - mean: the share of noise removed is mean(c * d) /
error_new^2, and what is left of mean(c^2) is error the filter added.

`finish.py --groups A:B` finishes a subset of the groups; two disjoint halves give two
independent finished images, whose difference measures the finished image's noise
directly (at half the budget). Detail kept: the high-pass images (minus a Gaussian
blur, sigma 1.5 px) of two independent renders share only real detail, so
mean(hp(final) * hp(old)) / mean(hp(new mean) * hp(old)) is the share of it the
finish keeps.

## Results (2026-11-18, laptop Iris Xe)

A 600x360 window (x 600, y 360) of a 1920x1080 frame, 4 bounces, old and new given the
same render time. A whole 1080p frame at 2048 spp takes about 35 min (Sponza) and
80 min (night Bistro) on this GPU. Errors are RMS in log units; the 1% of pixels
where the two renders differ most are left out of the cross-check. The laptop was in
use during the runs, so times and samples per second are rough.

| | Sponza atrium | night Bistro |
|---|---:|---:|
| render time, old / new | 234 s / 235 s | 504 s / 500 s |
| samples per pixel, old | 2048 | 2048 |
| samples per pixel, new (range) | 2475 (1792-2560) | 1690 (256-2816) |
| error, old | 0.0131 | 0.0016 |
| error, new plain mean | 0.0138 | 0.0025 |
| error, new finished | 0.0094 | 0.0020 |
| noise at half budget, mean -> finished (RMS) | 0.0199 -> 0.0081 | 0.0047 -> 0.0035 |
| the same for the typical pixel (median) | 0.0084 -> 0.0026 | 0.00075 -> 0.00028 |
| real detail kept | 98.3% | 100.0% |
| brightness, finished vs old | 1.0014 | 1.0005 |
| pixels with a rejected firefly | 0.07% | 0.16% |

- **The finish is what helps.** It cuts a typical pixel's noise about 3x and the RMS
  1.3-2.5x. The RMS is held up by rare bright samples below the firefly threshold.
- **Adaptive sampling did not help at equal time here.** Noise in these windows is
  spread evenly; in night Bistro the tiles that reached the target stopped at 256
  spp where the old method gave them 2048, and the readbacks cost 17% of the samples.
  Its use is a stated error target, not speed.
- **Filtering the image itself loses detail** (first version): 89% of real detail
  kept in Sponza, 0.0127 error. Filtering lighting only: 98.3%, 0.0094.
- The cross-check's split between old and new error is good to about 10% (new mean
  vs old in Sponza should differ by sqrt(2475/2048), not come out equal).

## Not done

- **Not in `--capture` yet**: moving sequences, crops and the frame files still use
  the old references.
- **Bounce count**: the comparison used 4 bounces for both; whether targets should
  match the runtime's 2 bounces or be the full transport is still open.
- **Heavy tails**: rare bright samples dominate what error is left. Candidates: a
  robust per-pixel estimator over more groups, or path regularization.
- **Allocation**: spend by need (samples proportional to a tile's noise) instead of
  stopping tiles at the target while others are far above it; read the guides once
  per round, not per group.
- Low-discrepancy sampling (Owen-scrambled Sobol) for the first dimensions.
- The run on the RTX PC, where the budget per frame can be far larger.
