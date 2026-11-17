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
captures from scratch), trains a 16- and a 24-channel model, evaluates them per scale and
renders videos. Results: `training/runs/pc-summary.txt`. `--smoke` checks the whole chain in
minutes.

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
