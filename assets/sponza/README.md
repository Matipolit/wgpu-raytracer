# Sponza test asset

This is the **2026 PNG/PBR cleanup of classic Crytek Sponza**, not Intel's 2022
remake. Geometry: Frank Meinl (Crytek), based on Marko Dabrovic's original;
PBR textures: Alexandre Pestana; glTF conversion and uncompressed texture
pack: Hans-Kristian Arntzen; 2026 cleanup: Ignacio Castaño / Ludicon.

- [2026 cleanup and methodology](https://www.ludicon.com/castano/blog/2026/02/an-updated-sponza-gltf/)
- [Pinned source](https://github.com/ludicon/sponza-gltf/tree/c1da1669da43660b600fe352f3ccfdca6c89cb07)
- [Texture source](https://themaister.net/sponza-gltf-pbr/)
- [Khronos asset/license metadata](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza)

Download: `python scripts/download_sponza.py` from the project root. The script
pins commit `c1da1669da43660b600fe352f3ccfdca6c89cb07` and verifies SHA-256
`258b82df37b703cf383dfc58a64a547814f1b0445cfd9a89bd8867c176921eaf`.
The GLB is 117,889,600 bytes and deliberately ignored by Git.

Run: `cargo run --release --locked -- --sponza`. This selects an interior camera,
daylight sky and exposure suitable for the atrium. Generic `--model` uses the
first perspective glTF camera in the selected scene, or a fitted exterior view.

The scene imports as 262,266 triangles / 192,496 vertices / 25 source materials /
65 textures. Texture pixels use 256 MiB of GPU storage; geometry, material
metadata and BVH use about 20.6 MiB, in addition to rendering buffers.

Retained upstream credit files: [cleanup README](UPSTREAM-README.md),
[original model/PBR credits](ORIGINAL-CREDITS.md), and
[uncompressed texture credits](TEXTURE-CREDITS.md).
[License metadata](LICENSE.md) identifies the model as governed by the
[CRYENGINE agreement](LicenseRef-CRYENGINE-Agreement.txt), while the metadata
itself is CC-BY-4.0. The model does not inherit this raytracer's software license.
