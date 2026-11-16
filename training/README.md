Current application defaults: three bounces, roulette enabled. Roulette begins
before bounce four, so it has no effect at the default three-bounce cap.
Historical experiments and training below use an explicit eight-bounce budget.

Run once from the repository root:

```sh
python3 training/start.py
```

This is a standalone local process: no agent, model API or tokens are needed.
Default: 36 unique Sponza views, 640×400, unfiltered 1 spp / 512 spp pairs,
then up to six hours of CPU training (four threads maximum), then GPU validation
at 1280×720. Capture time is additional to training time. 512 spp is a fixed
reference budget; difficult indirect lighting may still contain noise.

Each pair has PNG previews and little-endian float32 RGB radiance arrays.
Training uses radiance/(1+radiance), preserving HDR information without training
on clipped PNGs. The 3×3 RGB → 4 ReLU → 3 RGB residual network has 127 parameters
(about 0.5 KB float weights), 120 multiply-accumulates per pixel, and a single
compute dispatch. It is spatial only, trained for 1 spp, with no temporal
history or surface guides. Expect limitations on fine geometry and indirect
light; learned quality and the 5 ms budget are measured, not guaranteed.

Camera-level split: 30 training and six validation views. Validation pixels
are fixed and never used for gradient updates. Validation guides checkpoint
selection; this is not an independent test set. The job exports the best
checkpoint and measures 100 warmed isolated filter dispatches with GPU timestamps.
`acceptance.json` passes only if validation MSE beats unfiltered 1 spp and
p95 filter time is at most 5 ms at the stated resolution. It does not silently
activate a failing model. Full rendering and presentation cost is additional.

Progress: `training/runs/overnight.log`, `manifest.json`, `training.json`.
Results: `model.wgsl`, `best.pt`, `checkpoint.pt`, `timing.json`, `acceptance.json`.
Interrupted captures and training resume by rerunning the same command. Stop
with `kill <PID>`; checkpoint writes and completed captures are preserved.
Use a new `--output` directory to change capture settings. Keep the renderer
and scene unchanged when resuming an existing dataset.

Use the result explicitly:

```sh
target/release/weekend-raytracer-wgpu --sponza --bounces 8 --no-russian-roulette --neural-model training/runs/overnight/model.wgsl
```

The existing denoiser toggle controls the neural pass. Without `--neural-model`,
the existing filter is used. This model is trained at 1 spp; performance and
quality at other sample counts/resolutions need separate evaluation.

Options (passed through `training/run.sh`): `--pairs 36 --size 640x400
--benchmark-size 1280x720 --hours 6 --threads 4 --steps 1000000 --output PATH`.
Capture controls also work independently: `--camera x,y,z,dx,dy,dz,fov`,
`--no-denoise`, `--raw-output FILE.f32`. `--filter-report FILE.json` benchmarks
the filter after the specified sample budget is complete.
