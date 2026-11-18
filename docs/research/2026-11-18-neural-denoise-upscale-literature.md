# Real-time neural denoising and upscaling: what the literature says about our design

Written 2026-11-18 from a literature review (a research agent read the papers and SDK
docs listed under Sources) and our own measurements. The review compared the v3 joint
network (training/upscale/model.py) and the planned redesign
(docs/superpowers/specs/2026-11-18-fast-denoise-upscale-network-design.md) against
published real-time methods.

[paper] means a source states it; [inference] is the reviewer's reasoning;
[measured] is our own measurement. Two citation corrections from the review: QRISP
("Efficient neural supersampling on a novel gaming dataset") is by Qualcomm, not
Intel; "Interactive Monte Carlo Denoising using Affinity of Neural Features" is by
Isik, Mullia, Fisher, Eisenmann and Gharbi (Adobe).

## 1. Decisions the literature flags as wrong or risky, largest first

### 1.1 Remodulating by a jittered albedo that is never accumulated
- The output is `x * (albedo + eps)` with this frame's albedo, taken from one jittered
  sample per pixel. History validation also uses jittered depth and normals.
- Nothing accumulates the albedo over time, so every texture and geometry edge
  flickers at full amplitude after the network. The network cannot remove it, while
  the 1024-spp references are pixel-integrated. TAAU accumulates the final colour, so
  its edges converge. [inference]
- Thomas et al. 2022 filter and accumulate albedo over time "to remove jagged edges…
  and flickering in the temporal domain". [paper]
- "Frozen History" (2026) lists jittered depth in history tests as a registration
  error: about 103x larger reprojected depth change on a static view. [paper]
- [measured] On a still 192x120 Sponza capture, remodulating a perfectly stable
  demodulated signal by each frame's jittered albedo alone flickers 0.087 log units
  per frame, about 9x TAAU's total flicker on the video. 30-37% of pixels change
  albedo between frames.
- Fix: accumulate in modulated (final colour) space, or keep a separate albedo
  history; validate on pixel-centre or geometric data. (Done in the non-learned detail
  upscaler, 64179ae.)

### 1.2 History too short, enforced as a hard reset
- SVGF and NRD keep a per-pixel history length and blend at 1/N up to a cap. NRD sizes
  the cap as accumulation period x FPS, allowing about 60 frames with white noise when
  given a confidence input. [paper]
- On a failed or doubtful check, NRD shrinks the history length instead of resetting
  it (`historyLength *= lerp(confidence, 1, ...)`). A-SVGF uses temporal gradients the
  same way. [paper, the A-SVGF part from memory]
- We reset 7-14% of pixels to a single sample every frame [measured], mostly from the
  shading-normal check on normal-mapped surfaces under jitter.
- Thomas et al.: bicubic instead of bilinear history warping gave "stronger temporal
  reuse". [paper] We warp bilinearly, which blurs the history every frame, so the
  network learns to trust it less. [inference]
- We blend, filter (a-trous) and feed the filtered result back, so the history is
  re-blurred every frame. SVGF feeds back after the first wavelet level [from memory];
  NRD keeps the pre-filter accumulation as history. [inference]
- Training: Hasselgren et al. 2020 apply the spatial loss only on the last frame of
  5-frame clips; Arm NSS increases the spatial weight over a 16-step window of
  ~100-frame sequences; Frozen History skips the first 2 frames. [paper] We weight
  cold-start frames equally.
- Expected effect: large. This is the likely cause of the constant alpha ~0.85, SVGF's
  fixed 0.8. Isik et al. measured up to 1.5 dB loss from a fixed 0.8 blend. [paper]
- [measured] The v3 large network keeps alpha 0.79-0.87 over all 300 frames of a
  video, an effective memory of 5-8 frames.

### 1.3 ReSTIR makes input noise correlated across frames
- AMD Ray Regeneration: ReSTIR introduces "cross-sample dependencies [it] is not
  designed to handle"; it recommends permutation sampling or randomized temporal
  reuse. NRD reports residual "boiling". [paper]
- Blobs lasting several frames cannot be averaged away by a short history. That fits
  night scenes with many small lamps being the worst. [inference] Expected effect:
  large in those scenes.

### 1.4 Reference noise in the temporal loss
- With independent per-frame references, the reference's frame-to-frame difference
  carries about sqrt(2) sigma_ref of noise. Under L1, the gradient against output
  changes smaller than sigma_ref is weak, so the term barely sees small flicker,
  worst in dark many-light scenes where 1024 spp is still noisy. [inference]
- Others also use independent references (Isik 2048 spp, Hasselgren 1k spp) [paper],
  so it is the size of sigma_ref that matters.
- [measured] Still-camera 1024-spp references change 0.018 (Sponza), 0.014 (fireplace)
  and 0.0003 (night Bistro) log units per frame: 3%, 0.6% and 7.7% of the signal,
  about twice TAAU's total flicker on Sponza.
- Fixes: denoise the references, or compute the temporal term on low-passed
  differences.
- Weighting: Thomas et al. use temporal 0.8 / spatial 0.2. Frozen History shows that
  once inputs are correctly registered, a large temporal weight freezes accumulation
  (-0.7 to -0.9 dB on static frames at 0.25), and warns that std and tPSNR both reward
  freezing. [paper] Gate on PSNR and stability together.

### 1.5 Specular divided by base-colour albedo; surface motion for reflections
- NRD, AMD Ray Regeneration and DLSS-RR all take separate diffuse and specular albedo,
  the specular one pre-integrated (EnvBRDF). [paper] Thomas et al. filter diffuse and
  specular in separate paths; composite = albedo x diffuse + specular. [paper] NSRD
  (CVPR 2024) demodulates by the pre-integrated total BRDF F_beta. [paper]
- Dividing a dielectric highlight by (dark albedo + 0.01) amplifies it up to 100x, then
  texture is put back into it. [inference]
- DLSS-RR asks for specular motion vectors or hit distance; Hasselgren et al. name
  reflections as where temporal reuse fails. [paper]
- Expected effect: medium; it also drags the single shared alpha down. [inference]

### 1.6 No exposure normalization of inputs
- DLSS, FSR2 and XeSS all take exposure. [paper]
- In dark scenes log1p is close to linear, so night-scene errors carry little weight
  in the loss. [inference] Expected effect: medium, concentrated in night scenes.

### 1.7 Minor
- Keep the recurrent blend in f32: Frozen History uses a bf16 trunk with f32
  accumulation [paper]; bf16 rounds away small updates when alpha is near 1. [inference]
- References use 4 bounces, the runtime 2: a bias the network cannot see in its
  inputs. [inference]
- Not problems: our Halton jitter (period 1024) is longer than XeSS's 32-64 [paper];
  our motion vectors are jitter-free on a static camera (checked in code); kernel
  prediction is the established choice for denoising.

## 2. What published methods cost

| Method | Cost | Setup |
|---|---|---|
| Thomas et al. 2022 | 5.19 ms | RTX 3070, 720p to 1440p, TensorRT int8 U-Net + f16 Vulkan filters |
| Hasselgren et al. 2020 | 14 ms (small) / 42 ms | RTX 2080 Ti, 1080p |
| Isik et al. 2021 | 52.7 ms | RTX 2080 Ti, 1920x1024 |
| NSRR (Xiao et al. 2020) | ~13.9 ms | RTX 3090, 1080p (from QRISP's table) |
| QRISP 2023 | 1.08-3.52 ms | RTX 3090 fp16, 1080p |
| XeSS (DP4a path) | ~4.3-4.9 ms | Iris Xe (i7-1370P), 1080p output; Intel's iGPU budget is < 5 ms |
| Arm NSS | ~4 ms | 540p to 1080p on Mali with ~10 TOPS neural accelerators; predicts parameters, not images |
| NRD ReBLUR / ReLAX | 2.55 / 3.25 ms | RTX 4080, 1440p |
| AMD Ray Regeneration | 0.99 ms / 3.74 ms | RX 9070 XT, 540p / 1080p |
| FSR2 | 1.3 ms | RX 590, 1080p |

[inference] Iris Xe is about 2 TFLOPS f32 (about 4 f16), roughly a tenth of an RTX
3070. Thomas-class U-Nets would take well over 50 ms on it, and a clean-input upscaler
(XeSS) alone costs about 2 ms at 720p. So 5 ms is realistic only for a tiny per-pixel
network driving fixed filters: a temporal-denoiser-class stage at render resolution
plus one output pass.

## 3. Recommended direction (from the review)

- Keep the planned split: render-resolution network + one output pass. It is the
  class Arm NSS uses: predict a blend weight alpha, a history clamp strength theta and
  one small kernel, not images.
- **Stage A (render resolution):**
  - keep the unfiltered accumulation as the history;
  - track history length or use a normalized accumulator, with soft confidence
    instead of resets;
  - reproject with Catmull-Rom;
  - validate by plane distance and geometric normal on unjittered data;
  - demodulate by F_beta, or split diffuse and specular if the budget allows.
- **Stage B (output resolution):** accumulate in modulated space, so edges and
  textures get temporal anti-aliasing.
- **Training:** exposure-normalized log inputs; 64-frame sequences with late-frame
  weighting; clean or low-passed targets for the temporal term; f32 recurrence; match
  the runtime ReSTIR settings and bounce count; ReSTIR decorrelation.
- Cheap test before retraining: a static camera with reference lighting times jittered
  albedo, to measure flicker from 1.1 alone. [measured: done, see 1.1]

## Sources
- Thomas et al., Temporally Stable Real-Time Joint Neural Denoising and Supersampling,
  HPG 2022: https://momentsingraphics.de/HPG2022.html
- Yu & Zhao, Frozen History, arXiv 2610.15752 (2026): https://arxiv.org/abs/2610.15752
- Hasselgren et al., Neural Temporal Adaptive Sampling and Denoising, EG 2020:
  https://research.nvidia.com/publication/2020-05_neural-temporal-adaptive-sampling-and-denoising
- Isik et al., Interactive Monte Carlo Denoising using Affinity of Neural Features,
  SIGGRAPH 2021: https://www.mustafaisik.net/anf/
- Mercier et al. (Qualcomm), Efficient neural supersampling on a novel gaming
  dataset, ICCV 2023: https://arxiv.org/abs/2308.01483
- Li et al., Neural Super-Resolution for Real-time Rendering with Radiance
  Demodulation (NSRD), CVPR 2024: https://arxiv.org/abs/2308.06699
- NVIDIA NRD README: https://github.com/NVIDIA-RTX/NRD
- DLSS-RR programming guide:
  https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md
- AMD FSR Ray Regeneration manual: https://gpuopen.com/manuals/fsr_sdk/techniques/denoising/
- FSR2 README: https://github.com/GPUOpen-Effects/FidelityFX-FSR2
- Intel XeSS GDC 2023 slides:
  https://game.intel.com/wp-content/uploads/2024/02/GameDev2023_XeSS_v1.pdf
- Arm Neural Super Sampling: https://huggingface.co/Arm/neural-super-sampling

The reviewer could not re-check SVGF's fixed alpha = 0.2, where SVGF feeds back its
history, or A-SVGF's details (quoted from memory). Meng et al.'s Neural Bilateral
Grid (EGSR 2020) and Xiao et al.'s paper were not read directly; their numbers come
from citing papers.
