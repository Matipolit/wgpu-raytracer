# Native Vulkan ray tracing and tensor-core inference (proposal)

Status: proposal for review (not started). Written 2026-11-18 from measurements on the
RTX 5070 Ti (driver 615.78), the Steam Deck (RADV, Mesa 26.2-dev) and Iris Xe.

## Why

wgpu exposes only inline ray queries in compute shaders. Measured on the RTX at 1080p,
3 bounces, tracing time:

| scene | RT cores (ray queries) | compute BVH |
|---|---:|---:|
| Sponza | 5.5 ms | 13.5 ms |
| Bistro | 28.1 ms | 24.3 ms |
| Bistro, no alpha tests | 26.5 ms | |

Sponza is traversal-bound and the RT cores win 2.4x. Bistro is shading-bound (3.7 M
triangles, 132 materials, 574 MB of textures, divergent threads); the RT cores do not
help and alpha tests are minor (1.5 ms). The tools for that case exist on this driver
but not in wgpu:

- **Shader execution reordering** (`VK_EXT_ray_tracing_invocation_reorder`, also
  `VK_NV_...`): regroups threads by hit material before shading. Khronos' example:
  23% -> 54% coherence, trace time -48%.
- **Ray tracing pipelines** (`VK_KHR_ray_tracing_pipeline`): SER needs them (closest-hit /
  miss shaders and a hit object), and they let the driver schedule shading.
- **Opacity micromaps** (`VK_EXT_opacity_micromap`): alpha-tested foliage resolved inside
  the RT core without any-hit shader calls.
- **Cooperative matrices** (`VK_KHR_cooperative_matrix`, NV extensions): tensor cores from
  compute shaders, for the denoise + upscale network's convolutions (large/ultra tiers).

On the Steam Deck none of this pays: RDNA2 RT units only run box/triangle tests (traversal
stays in shader code, measured 2.3-4.5x slower than our compute BVH) and there is no
matrix hardware. The design keeps wgpu as the portable path and adds a native path only
where the hardware exists.

## Approaches

1. **wgpu-hal escape hatch (recommended).** Keep wgpu for everything; on Vulkan devices
   with the extensions, get the raw `ash` device/queue/buffers through
   `wgpu::Device::as_hal` / `Buffer::as_hal` and record a native ray tracing pipeline
   (and cooperative-matrix compute pipelines) into the same frame via
   `CommandEncoder::as_hal_mut`. Shared resources stay wgpu buffers (G-buffer, image,
   scene data), so the rest of the renderer, the launcher, the auto-tuner and the
   denoiser plumbing are untouched. Shaders: the existing WGSL is translated by naga to
   SPIR-V; the RT stages (raygen, closest-hit, miss, any-hit) are new SPIR-V entry points
   (naga does not emit them: GLSL compiled with `shaderc`/`glslang` at build time, or
   hand-ported). Risk: `as_hal` APIs are unstable across wgpu versions; synchronization
   between wgpu and raw commands must be explicit.
2. **Full port to ash.** Cleanest control, but rewrites ~5k lines of Rust plumbing that
   works today and loses portability (Metal, DX12, the laptop's Vulkan path is fine but
   everything must be redone). Not worth it for two features.
3. **Stay in wgpu, wait.** wgpu has experimental cooperative-matrix work and ray-tracing
   pipeline discussions; timing unknown. Keeps the 2.4x Sponza gain (already available
   through `--hw-rt` + calibration) but no SER/OMM/tensor cores.

## Design (approach 1)

- `src/raytracer/native/` (Vulkan only, behind a cargo feature): device capability probe,
  the RT pipeline (raygen = the current path loop with `traceRayEXT` + `reorderThreadEXT`
  hints keyed by material id; closest-hit resolves attributes; any-hit only for
  non-OMM alpha cases), shader binding table, opacity micromap build for MASK materials
  (4x4 states per triangle from the existing opacity-cell data: `Mesh::opacity_cells`).
- The startup calibration gains a "native RT pipeline" option next to compute / RT
  queries; the auto-tuner keeps choosing per GPU and scene, so the Deck and laptop never
  run it.
- Neural inference: the large/ultra tiers' 3x3 convolutions as 16x16 cooperative-matrix
  tiles (f16 in, f32 accumulate); smaller tiers stay WGSL. A per-tier time budget at
  startup picks the largest tier that fits.
- Training captures use the fastest path the calibration finds, so data generation speeds
  up with the renderer.

## Measure first

Before any porting: a standalone `ash` prototype (one scene, primary + one bounce, SER on
/off, OMM on/off) on Bistro to confirm the gain. Expected from the literature and the
Bistro breakdown: 1.5-2x on shading-bound scenes; if under 1.3x, stop.
