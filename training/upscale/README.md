# Joint denoise + upscale

One small network that turns 1-spp frames rendered at a lower resolution (any scale) into
denoised output-resolution frames, using an output-resolution G-buffer and its own history.
`src/raytracer/upscale.wgsl` (`--render-scale S`) is the non-learned baseline it has to beat.

## Run everything on the GPU machine

```sh
git clone -b fast-tracing-and-temporal-denoiser https://github.com/Matipolit/wgpu-raytracer ~/wgpu-raytracer-next
cd ~/wgpu-raytracer-next
scripts/pc_run.sh --from ~/wgpu-raytracer --hours 8 --detach   # tail -f pc-run.log
```

`--from` reuses the old checkout's scenes, captures and Python environment. The script
builds, re-exports and parity-checks last night's temporal denoisers, benchmarks compute
vs hardware ray queries, adds multi-scale inputs to last night's 2048-spp captures (or
captures from scratch), captures new data with moving triangle objects and fast motion,
trains all five quality tiers on both, evaluates them per scale on both evaluation sets and
renders videos. Results: `training/runs/pc-summary.txt`. `--smoke` checks the whole chain in
minutes.

## Quality tiers

One architecture, five sizes (`train.py --tier`), each aimed at a class of GPU. Costs are
multiply-adds of the learned layers per output pixel; the fixed parts (gather, reprojection,
three filter passes) add roughly the same on every tier and dominate on weak GPUs, which
are memory-bound.

| Tier | Channels x 3x3 layers | Weights | MAC / px | GMAC at 1080p | Aimed at |
|---|---|---:|---:|---:|---|
| tiny | 8 x 2, half res | 2.6 k | 1.2 k | 2.5 | old integrated GPUs (UHD 620, Vega 8) |
| small | 16 x 2, half res | 5.9 k | 2.2 k | 4.5 | Iris Xe, Steam Deck |
| medium | 24 x 3, half res | 15.5 k | 4.7 k | 9.8 | Radeon 780M, Arc integrated |
| large | 48 x 3, half res | 51 k | 14.2 k | 29 | RTX 3050/4060 class (tensor cores) |
| ultra | 64 x 4, half res | 124 k | 32 k | 67 | RTX 4070 and up (tensor cores) |

The large and ultra tiers assume matrix hardware: their 3x3 convolutions are 16-channel
matrix multiplies that map onto tensor cores / XMX / WMMA through Vulkan cooperative
matrices (SPIR-V passthrough kernels; WGSL has no matrix types yet). The renderer's
startup calibration can then pick the largest tier that fits the frame budget.

## Moving objects

Training data uses moving triangle objects (`--objects 6`: procedural torus, crate,
cylinder, rock, torus knot, cone with random PBR materials, positions, sizes, velocities
and spin) and fast-motion sequences (`--fast-fraction`: camera x4-6, objects x5-8, or both).
The renderer traces them as rigid instances: each has its own BVH, traversed in object
space after the scene; motion vectors use each object's previous transform.

## Pieces

- **Capture:** `weekend-raytracer-wgpu --capture DIR ... --upscale-capture 0.5,0.67` traces each
  frame's 1-spp input at every scale (shared per-frame jitter) next to the output-resolution
  guides and reference; `--low-only` adds the inputs to an existing capture made with the same
  arguments. Batch version: `training/temporal/capture.py --render-scale S1,S2 [--low-only]`.
- **Model** (`model.py`): per output pixel, this frame's samples gathered by their exact
  jittered positions (demodulated by their own albedo), confidence (weight sum, nearest-sample
  distance), reprojected history and validity, output-resolution guides and the scale; two
  3x3 convs at half resolution, a 1x1 head, a temporal blend and three a-trous filter levels;
  remodulated by the output-resolution albedo.
- **Training** (`train.py`): one random scale per batch, random crops, whole sequences.
- **Evaluation** (`evaluate.py`): PSNR and flicker per scene and scale against the raw splat
  and a replica of the in-renderer upscaler. **Videos**: `upscale_video.py`.

Not yet done: WGSL inference for this network and the output-resolution visibility raster
that feeds it in the renderer.
