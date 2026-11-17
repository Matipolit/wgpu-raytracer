# Sun disk, Bistro at night, launcher menu

## Intent

Today's lighting is an analytic sky with no sun disk, so every scene looks overcast. Before
building ReSTIR (light resampling), the renderer needs test scenarios where light sampling
matters: hard sunlight and a night street lit only by lamps. The user also wants to explore
the current, fastest renderer on the RTX PC through a launcher menu with graphics settings.

Goals: sun disk on by default; a Bistro night preset; a launcher with scene, quality preset
and graphics settings; the newest build running on the PC.
Non-goals: ReSTIR, lamp (emissive triangle) sampling, retraining the denoiser.
Anti-goals: breaking existing command lines and scripts (capture, benchmark, pc_run.sh).

## Approaches

The stage order (scenarios, then light sampling, then ReSTIR DI, then ReSTIR GI) was chosen
earlier. For this stage:

- **Sun disk alone (rejected):** mesh surfaces only find light by chance, so a 0.5° disk
  would produce fireflies everywhere.
- **Sun disk + direct sun sampling (chosen, user approved):** one shadow ray per mesh bounce
  toward a point on the disk, combined with BSDF hits on the disk by multiple importance
  sampling.
- Launcher as a separate tool vs. a menu in the app window: in the app (egui is already
  there). "Launch" restarts the binary with flags, because settings are read from
  command-line flags in dozens of places and many are baked into pipelines.

## Design

1. **Sun disk:** angular radius 0.2666°, part of `skyRadiance` for camera and escaped
   rays. Colour tinted by turbidity and elevation; brightness set so the sun's horizontal
   irradiance is about 4× the sky's on a clear day at 45° (builder's call: calibrated
   numerically on the CPU from the sky model, not absolute units). On by default;
   `--no-sun` restores the old sky. Mesh PBR bounces sample the disk (uniform cone) with a
   shadow ray; BSDF-sampled rays that hit the disk get the complementary MIS weight. Sphere
   scene materials keep their existing sampling.
2. **Bistro at night:** `--night` dims the sky to faint bluish moonlight and turns the sun
   off; lamp materials are the main light. Launcher entry "Bistro (night)".
3. **Launcher:** no arguments → launcher screen (or `--launcher`). Choices:
   - scene;
   - quality preset (Low, Medium, High, Reference);
   - resolution, fullscreen, render scale;
   - denoiser (off, spatial, neural td-half16 / td2-half24 when the files exist);
   - bounces, hardware RT (only if supported), auto-tuner, texture filtering;
   - sun elevation and azimuth.
   "Launch" spawns the same binary with the corresponding flags and exits; the last choices
   are saved to ~/.config/wgpu-raytracer/launcher.txt.
4. **PC:** push, clone on the PC, link the old checkout's assets, copy re-exported denoisers
   to `models/`, build, smoke-run.

Left to the builder: exact preset contents, sun colour formula, file locations.
