# glTF loading and BVH design

## Dependency decision (2026-11-15)

Use [`gltf` 1.4.1](https://docs.rs/gltf/1.4.1/gltf/) rather than a whole engine or
an OBJ parser. Its [upstream repository](https://github.com/gltf-rs/gltf) is not
archived and had activity in May 2026; the stable release is older (May 2024),
so release recency alone would be misleading. It is an established MIT/Apache
parser with schema validation, indexed/sparse accessor readers, PNG/JPEG import,
GLB and external/data URI buffers, and the two extensions used by this loader.
Its image dependency shares this project's `image` 0.25 dependency.

Also checked [`bvh`](https://github.com/svenstaro/bvh) 0.13 and
[`obvhs`](https://github.com/DGriffin91/obvhs) 0.4. Both are active alternatives;
`obvhs` explicitly targets GPU traversal formats. For this first binary BVH,
a local builder avoids another math/layout abstraction and makes the CPU/WGSL
contract explicit. Revisit an optimized builder/wide BVH if profiling justifies it.

## Data flow

`glTF/GLB -> validated core mesh/material accessors -> world-space static vertices
-> CPU binned SAH BVH -> read-only GPU buffers -> WGSL compute path tracing`.

The selected scene is flattened once; node instances duplicate transformed
vertices. Positions use the world transform, normals its inverse transpose,
and tangents its linear part with determinant sign. Negative determinant
transforms reverse triangle winding. Missing normals use per-face normals;
missing tangents derive a frame from the normal-map UV set.

The BVH uses 16-bin surface-area splits, four-triangle leaves, median fallback
for coincident centroids, and a maximum depth of 48. Nodes are 32 bytes; WGSL
traverses near children first with a 64-entry local stack. Triangle tests use
Möller–Trumbore and interpolate normals and UVs with barycentric weights. Slab
tests handle zero direction components explicitly. The original sphere list
is still intersected linearly, then combined with the nearest mesh hit.

Geometry, BVH, materials and texture descriptors share one u32 storage buffer;
pixels share a second RGBA8 buffer. glTF textures referring to the same image
share its pixel storage while retaining their own sampler descriptors. The
shader decodes sRGB before bilinear filtering for base color and emission;
metallic/roughness and normals remain linear. glTF UVs retain their top-left
image convention; legacy sphere textures keep their existing convention.
The compute stage stays within WebGPU's default eight-storage-buffer limit.
The loader checks the device's per-buffer size limit before mesh upload.

Pipeline overrides remove the mesh/PBR code from sphere-only scenes and sphere
material code from mesh-only scenes. This avoids paying shader register costs
for geometry that is absent. Exposure changes update display uniforms and retain
accumulated samples.

## Supported subset

- Static indexed or unindexed TRIANGLES in the selected default/first scene;
  world transforms, nonuniform/mirrored scales, normals, tangents and UV sets 0/1.
- Metallic/roughness PBR: Lambert diffuse plus GGX NDF sampling, separable Smith
  masking and Schlick Fresnel. Metallic and roughness factors multiply B/G map
  channels; roughness has a 0.045 floor for numerical stability.
- Base-color factor/map, normal map/scale, emission factor/map and
  `KHR_materials_emissive_strength`.
- OPAQUE and MASK materials, alpha cutoff, single/double-sided intersections.
  Alpha tests happen during traversal so rays continue through cutouts.
- Repeat, clamp and mirrored wrapping; nearest or bilinear base-level sampling.
- `KHR_texture_transform` on supported maps, including UV-set overrides.
- PNG/JPEG images (8-bit); GLB images and external/data URI resources.
- First perspective camera, or a fitted default; `--sponza` has an interior view.

Skins, morph targets, vertex colors, BLEND, nontriangle primitives and unsupported
required extensions are rejected with an error. Animation is imported at static
node transforms with a notice. Unsupported optional extensions use their core
fallback with a notice. Draco, meshopt compression, KTX2/BasisU and AVIF are not
implemented: export the PNG/JPEG, uncompressed glTF/GLB variant.

No animation, instancing acceleration (TLAS), texture mipmaps/anisotropy,
normal-map tangent regeneration with MikkTSpace, transmission/clearcoat extensions,
punctual lights or explicit emissive-triangle/environment light sampling yet.
Imported meshes currently receive sky and path-traced emissive illumination;
interior Sponza needs many samples. Baked occlusion maps are not multiplied into
physically traced illumination. Sky/environment importance sampling would be a
useful next step for convergence, independent of BVH traversal speed.

## Validation

`cargo test --release --locked -- --include-ignored` (download Sponza first)
checks CPU transforms, mirrored winding, BVH bounds/coverage, unsupported inputs,
GPU BVH versus brute force, alpha-mask holes, UV orientation, back-face normals,
emissive mesh rendering, sRGB-before-filtering, sampler wrapping, analytic sphere-light solid-angle PDFs (zero,
one, multiple and inside lights), and exposure after convergence. GPU comparisons use
772 synthetic grid rays plus analytic fixture hits, and 512 Sponza rays. Existing
resize, denoiser, font, UI and fragment/compute parity checks remain enabled.
The frozen fragment reference received the same sphere-light correctness fixes
as compute, so it continues to compare execution stages rather than different
lighting algorithms. Sphere lights now sample solid angle, include the final
light, and evaluate the mixture PDF over actual lights instead of intersecting
and counting all scene spheres. Zero-light diffuse scenes use BSDF sampling,
and zero-throughput paths terminate. Shader overrides specialize the known
sphere/light counts, avoiding repeated count-buffer loads in the inner loop.
These estimator changes alter the old sphere scene's noise/brightness and also
remove unnecessary intersection work; timing changes are not a same-image claim.

The offscreen image command records warmed GPU time per sample (excluding the
first upload/driver frame), and can produce a PNG without a graphical session:

```sh
cargo run --release --locked -- --sponza --render-output /tmp/sponza.png --size 640x400 --samples 256
cargo run --release --locked -- --model path/to/scene.glb --render-output /tmp/model.png --exposure 2
```

See `gltf/comparison.json` for the alternating three-round sphere-scene regression
benchmark. Sponza timings and the final image are recorded separately; there was
no preexisting mesh renderer to use as a Sponza performance baseline.
