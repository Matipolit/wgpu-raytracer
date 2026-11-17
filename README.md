# README

A WebGPU implementation of Peter Shirley's ray tracing books. An interactive renderer which allows you to play around with the camera and renderer settings in real-time.

## Run

Requires Rust 1.95 or newer.

```sh
$ cargo run --release --locked
```

With no arguments this opens the **launcher**: choose a scene (including Bistro at
night), a quality preset (Low, Medium, High, Reference) and graphics settings
(resolution, fullscreen, render scale, denoiser, bounces, texture filtering, hardware
ray tracing, auto-tuner, sun position), then Launch. The launcher restarts the renderer
with the matching flags (printed to the terminal) and remembers the choices in
`~/.config/wgpu-raytracer/launcher.txt`. In the renderer: hold the right mouse button to
look around, WASD to move, Q/E down/up; "Back to launcher" is in the parameter panel.
`--launcher` opens it explicitly; any other arguments skip it.

Mesh scenes have a sun disk (sampled directly on surfaces) at 45° elevation:
`--sun-elevation DEG --sun-azimuth DEG` move it, `--no-sun` restores the overcast-looking
sky, `--night` dims the sky to moonlight so emissive materials light the scene.
Renderer window options: `--size WxH`, `--fullscreen`, `--max-samples N` (progressive
budget), `--no-denoise`. Temporal neural denoisers ship in `models/`
(`--temporal-model models/td2-half24.wgsl`).
With a neural denoiser, directly visible emission (lamps, emissive spheres) bypasses
the network and is added back afterwards, the input is the frame's mean sample, and a
still view converges: raw frames are averaged and the display fades from the denoised
frame to that mean (halfway after 64 still frames). Auto exposure stops adapting at a
log-average luminance of 0.005, so night scenes stay dark.

The renderer uses wgpu 30, winit 0.30, and egui 0.37. `Cargo.lock` records the
complete dependency set.

Path tracing and bilateral denoising run in separate compute passes. The filter
shares pixel neighborhoods within workgroups and caches the completed image.
A small render pass tone-maps that image and draws egui. Accumulation and filtered
radiance use separate full-precision RGB buffers, resized with the viewport.

GPU regression tests cover resizing, accumulation, denoiser controls, and rendering
the textured scene together with egui. They also compare compute output against
the frozen fragment renderer across viewport sizes and filter settings:

```sh
cargo test --release --locked -- --include-ignored
```

The UI bundles its fonts and does not require system font installation. A native
shutdown regression test (requires a graphical session) opens the app, sends its
normal close event after two rendered frames, and checks for a clean exit:

```sh
cargo build --release --locked
python3 tests/native_close.py
```

A repeatable offscreen performance benchmark uses GPU timestamp queries, a fixed
scene/camera, and warmed pipelines at window and fullscreen resolutions:

```sh
cargo run --release --locked -- --benchmark
```

See [performance/compute/README.md](performance/compute/README.md) for the compute
migration comparison, or [performance/README.md](performance/README.md) for the
earlier UI migration measurements.

### Camera and render controls

* Translate the camera using WASD keys.
* Pan the camera by dragging the right mouse button.
* Adjust camera and rendering parameters using UI

## Read

More details about the WebGPU implementation is available in the following series of blog posts.

**[Weekend raytracing with wgpu, part 2](https://nelari.us/post/weekend_raytracing_with_wgpu_2/)**

![blog post part 2](https://nelari.us/img/weekend_raytracing_with_wgpu_2/featured_image.png)
* Adds texture support from "Ray Tracing: The Next Week" and a physically-based sky based on the Hosek-Wilkie model.
* Describes the global texture lookup system implemented in WGSL.

**[Weekend raytracing with wgpu, part 1](https://nelari.us/post/weekend_raytracing_with_wgpu_1/)**

![blog post part 1](https://nelari.us/img/weekend_raytracing_with_wgpu_1/featured_image.png)
* A straightforward fragment shader implementation of the "Ray Tracing In One Weekend" book.
* Notes on porting RNG, recursive rendering function and more to WGSL.

## Asset credits

assets/moon.jpeg
* NASA's Scientific Visualization Studio
* https://svs.gsfc.nasa.gov/4720

assets/earthmap.jpeg
* https://raytracing.github.io/images/earthmap.jpg

assets/sun.jpeg
* https://www.solarsystemscope.com/textures/

## glTF / GLB scenes and Sponza

Download the 2026 PBR/PNG cleanup of classic Sponza (118 MB, pinned checksum):

```sh
python3 scripts/download_sponza.py
cargo run --release --locked -- --sponza
```

Load another static scene, inspect its import statistics, or save an offscreen PNG:

```sh
cargo run --release --locked -- --model path/to/scene.glb
cargo run --release --locked -- --model path/to/scene.gltf --inspect-model
cargo run --release --locked -- --sponza --render-output /tmp/sponza.png --size 640x400 --samples 256
```

`--sponza` selects an interior camera and daylight. Imported scenes support
metallic/roughness PBR maps, normal maps, emission and alpha cutouts. Adjust
exposure in the UI or pass `--exposure 2` when rendering offscreen. Exposure changes
retain accumulated samples. Geometry uses a CPU-built binary SAH BVH traversed
in compute; the original sphere scene remains available without arguments.

This is a static core-glTF loader: compressed mesh/texture extensions, skins,
morphs, blended transparency and vertex colors are not supported. Unsupported
required features report an error. There are no texture mipmaps or explicit
mesh-light/environment importance sampling yet, so Sponza interiors remain noisy
at low sample counts. See [the implementation and validation notes](performance/gltf-loading.md)
and [asset provenance/license](assets/sponza/README.md) for details. Download the
asset before running the complete GPU test suite.

## Overnight neural denoiser

See [training/README.md](training/README.md) for resumable 1/512 spp Sponza capture,
CPU training, WGSL inference and the measured 5 ms acceptance gate.

Measured Iris Xe traversal optimizations and reproduction commands are in
[performance/laptop/README.md](performance/laptop/README.md).

Russian roulette is enabled by default. The default bounce limit is three;
roulette starts before bounce four, so it takes effect when that limit is raised.
Use `--bounces 8` for the historical comparison setting or
`--no-russian-roulette` to disable roulette.
See [performance/roulette/README.md](performance/roulette/README.md) for equal-time
quality comparisons and reproduction commands.

Experimental mesh sky sampling: add `--sky-importance`. Equal-time tests did not
show a consistent gain, so it remains off by default. See
[performance/sky/README.md](performance/sky/README.md).

Experimental GGX visible-normal sampling for mesh PBR materials is available
with `--ggx-vndf`; see [performance/vndf/README.md](performance/vndf/README.md).

The [combined local optimization report](performance/local-sweep/README.md)
compares 20 rendering variants, scene caching, a spatial filter, and two
combinations on Iris Xe. The strongest tested combination is available with
`--sah-stop --skip-final-scatter`; all new experiments remain opt-in.
