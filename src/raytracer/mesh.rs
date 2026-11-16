//! Static glTF import and a binned SAH binary BVH. All offsets are u32 words.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::cast_sign_loss
)]
use super::{Angle, Camera};
use std::{error::Error, path::Path, time::Instant};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    _pad0: f32,
    pub normal: [f32; 3],
    _pad1: f32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub tangent: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle {
    pub vertices: [u32; 3],
    pub material: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BvhNode {
    pub min: [f32; 3],
    /// Interior: right child (left is this node + 1); leaf: first triangle.
    pub first: u32,
    pub max: [f32; 3],
    /// Zero for interior nodes.
    pub count: u32,
}

#[derive(Default)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub triangles: Vec<Triangle>,
    pub nodes: Vec<BvhNode>,
    pub materials: Vec<[u32; 48]>,
    pub textures: Vec<[u32; 4]>,
    pub pixels: Vec<u32>,
    pub camera: Option<Camera>,
    pub depth: usize,
    pub mip_headers: Vec<[u32; 4]>,
    pub mip_levels: Vec<[u32; 4]>,
    pub opacity_cells: Vec<u32>,
}

type LoadResult<T> = Result<T, Box<dyn Error>>;

pub fn micro_opacity() -> bool {
    option("--opacity-micromap") || std::env::var("RT_MICRO_OPACITY").is_ok()
}
pub fn quantized_bvh() -> bool {
    option("--quantized-bvh") || std::env::var("RT_QUANTIZED_BVH").is_ok()
}
pub fn bvh_width() -> u32 {
    if quantized_bvh() {
        return 8;
    }
    if std::env::args().any(|a| a == "--bvh8") {
        8
    } else if std::env::args().any(|a| a == "--bvh4") {
        4
    } else {
        std::env::var("RAYTRACER_BVH_WIDTH").map_or(2, |s| s.parse().unwrap())
    }
}
fn option(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}
fn pack_half(v: [f32; 2]) -> u32 {
    u32::from(half::f16::from_f32(v[0]).to_bits())
        | (u32::from(half::f16::from_f32(v[1]).to_bits()) << 16)
}
fn pack_normal(v: [f32; 3]) -> u32 {
    let l = v[0].abs() + v[1].abs() + v[2].abs();
    if l == 0.0 {
        return 0x80008000;
    }
    let mut p = if l > 0.0 {
        [v[0] / l, v[1] / l]
    } else {
        [1.0, 0.0]
    };
    if v[2] < 0.0 {
        p = [
            (1.0 - p[1].abs()) * if p[0] >= 0.0 { 1.0 } else { -1.0 },
            (1.0 - p[0].abs()) * if p[1] >= 0.0 { 1.0 } else { -1.0 },
        ];
    }
    let x = (p[0].clamp(-1.0, 1.0) * 32767.0).round() as i16;
    let y = (p[1].clamp(-1.0, 1.0) * 32767.0).round() as i16;
    u32::from(x as u16) | (u32::from(y as u16) << 16)
}
fn wide_node(mesh: &Mesh, index: usize, width: usize, words: &mut Vec<u32>) -> usize {
    let result = words.len() / (width * 8);
    words.resize(words.len() + width * 8, 0);
    let mut frontier = vec![index];
    while frontier.len() < width {
        let Some((slot, _)) = frontier
            .iter()
            .enumerate()
            .filter(|(_, i)| mesh.nodes[**i].count == 0)
            .max_by(|(_, a), (_, b)| {
                let area = |i: usize| {
                    let n = mesh.nodes[i];
                    let e = glm::Vec3::from(n.max) - glm::Vec3::from(n.min);
                    e.x * e.y + e.y * e.z + e.z * e.x
                };
                area(**a).total_cmp(&area(**b))
            })
        else {
            break;
        };
        let parent = frontier.remove(slot);
        frontier.insert(slot, mesh.nodes[parent].first as usize);
        frontier.insert(slot, parent + 1);
    }
    for lane in 0..width {
        let dst = result * width * 8 + lane * 8;
        if lane >= frontier.len() {
            words[dst + 3] = u32::MAX;
            continue;
        }
        let n = mesh.nodes[frontier[lane]];
        let child = if n.count == 0 {
            wide_node(mesh, frontier[lane], width, words) as u32
        } else {
            n.first
        };
        for axis in 0..3 {
            words[dst + axis] = n.min[axis].to_bits();
            words[dst + 4 + axis] = n.max[axis].to_bits();
        }
        words[dst + 3] = child;
        words[dst + 7] = n.count;
    }
    result
}

impl Mesh {
    pub fn load(path: &Path) -> LoadResult<Self> {
        use std::hash::{Hash, Hasher};
        let cache = if option("--scene-cache") && option("--sponza") {
            let metadata = std::fs::metadata(path)?;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            path.canonicalize()?.hash(&mut hash);
            metadata.len().hash(&mut hash);
            metadata
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
                .hash(&mut hash);
            include_str!("mesh.rs").hash(&mut hash);
            crate::argument_value("--bvh-leaf").hash(&mut hash);
            option("--sah32").hash(&mut hash);
            option("--sah-stop").hash(&mut hash);
            option("--ray-cones").hash(&mut hash);
            micro_opacity().hash(&mut hash);
            option("--packed-shading").hash(&mut hash);
            let directory = std::path::Path::new("target/scene-cache");
            std::fs::create_dir_all(directory)?;
            Some(directory.join(format!("{:016x}.bin", hash.finish())))
        } else {
            None
        };
        if let Some(file) = &cache {
            if let Ok(bytes) = std::fs::read(file) {
                if let Some(mesh) = decode_scene_cache(&bytes) {
                    eprintln!("Scene cache hit: {}", file.display());
                    return Ok(mesh);
                }
            }
        }
        let mut mesh = Self::load_uncached(path)?;
        if option("--ray-cones") {
            mesh.build_mips();
        }
        if let Some(file) = cache {
            let temporary = file.with_extension("tmp");
            std::fs::write(&temporary, encode_scene_cache(&mesh))?;
            std::fs::rename(temporary, file)?;
        }
        Ok(mesh)
    }
    fn load_uncached(path: &Path) -> LoadResult<Self> {
        let start = Instant::now();
        // Check required extensions before image import so unsupported codecs get
        // a useful error instead of an obscure missing-image/accessor failure.
        let file = gltf::Gltf::open(path)?;
        for extension in file.document.extensions_required() {
            if !matches!(
                extension,
                "KHR_materials_emissive_strength" | "KHR_texture_transform"
            ) {
                return Err(format!("Unsupported required glTF extension: {extension}. Export uncompressed metallic/roughness glTF 2.0.").into());
            }
        }
        for extension in file.document.extensions_used() {
            if !matches!(
                extension,
                "KHR_materials_emissive_strength" | "KHR_texture_transform"
            ) {
                eprintln!(
                    "glTF: optional {extension} is not implemented; using its core glTF fallback"
                );
            }
        }
        let base = path.parent().unwrap_or(Path::new("."));
        let buffers = gltf::import_buffers(&file.document, Some(base), file.blob)?;
        let images = gltf::import_images(&file.document, Some(base), &buffers)?;
        let document = file.document;
        if document.animations().next().is_some() {
            eprintln!("glTF: animations are not played; importing the static node transforms");
        }
        let mut mesh = Self::default();
        let mut image_descriptors = Vec::new();
        for image in images {
            let offset = u32::try_from(mesh.pixels.len())?;
            let channels = match image.format {
                gltf::image::Format::R8 => 1,
                gltf::image::Format::R8G8 => 2,
                gltf::image::Format::R8G8B8 => 3,
                gltf::image::Format::R8G8B8A8 => 4,
                other => {
                    return Err(format!(
                        "Unsupported glTF image format {other:?}; use 8-bit PNG/JPEG"
                    )
                    .into());
                }
            };
            for pixel in image.pixels.chunks_exact(channels) {
                let rgba = match channels {
                    1 => [pixel[0], pixel[0], pixel[0], 255],
                    2 => [pixel[0], pixel[0], pixel[0], pixel[1]],
                    3 => [pixel[0], pixel[1], pixel[2], 255],
                    _ => [pixel[0], pixel[1], pixel[2], pixel[3]],
                };
                mesh.pixels.push(u32::from_le_bytes(rgba));
            }
            image_descriptors.push([offset, image.width, image.height, 0]);
        }
        for texture in document.textures() {
            let mut desc = image_descriptors[texture.source().index()];
            let sampler = texture.sampler();
            let wrap = |mode| match mode {
                gltf::texture::WrappingMode::Repeat => 0,
                gltf::texture::WrappingMode::ClampToEdge => 1,
                gltf::texture::WrappingMode::MirroredRepeat => 2,
            };
            desc[3] = wrap(sampler.wrap_s())
                | (wrap(sampler.wrap_t()) << 2)
                | (u32::from(sampler.mag_filter() == Some(gltf::texture::MagFilter::Nearest)) << 4);
            mesh.textures.push(desc);
        }
        for material in document.materials() {
            mesh.materials.push(import_material(&material)?);
        }
        let mut default = [0; 48];
        default[..4].fill(1f32.to_bits());
        for i in [7, 8, 9] {
            default[i] = 1f32.to_bits();
        }
        default[10] = 0.5f32.to_bits();
        for i in [16, 24, 32, 40] {
            default[i] = u32::MAX;
        }
        mesh.materials.push(default);
        let scene = document
            .default_scene()
            .or_else(|| document.scenes().next())
            .ok_or("glTF contains no scene")?;
        for node in scene.nodes() {
            mesh.import_node(
                &node,
                &glm::identity(),
                &buffers,
                document.materials().len(),
            )?;
        }
        if mesh.triangles.is_empty() {
            return Err("glTF scene contains no nondegenerate triangles".into());
        }
        let imported = start.elapsed();
        mesh.build_bvh();
        if mesh.camera.is_none() {
            let min = glm::Vec3::from(mesh.nodes[0].min);
            let max = glm::Vec3::from(mesh.nodes[0].max);
            let center = (min + max) * 0.5;
            let extent = max - min;
            // Generic exterior view; --sponza selects a known interior view.
            let eye = center + glm::vec3(0., extent.y * 0.15, glm::length(&extent) * 0.9);
            mesh.camera = Some(Camera {
                eye_pos: eye,
                eye_dir: glm::normalize(&(center - eye)),
                up: glm::vec3(0., 1., 0.),
                vfov: Angle::degrees(60.),
                aperture: 0.,
                focus_distance: glm::distance(&eye, &center),
            });
        }
        eprintln!(
            "glTF: {} triangles, {} vertices, {} materials, {} textures; import {:.2}s, BVH {:.3}s ({} nodes, depth {}); geometry {:.1} MiB, textures {:.1} MiB",
            mesh.triangles.len(),
            mesh.vertices.len(),
            mesh.materials.len() - 1,
            mesh.textures.len(),
            imported.as_secs_f64(),
            start.elapsed().as_secs_f64() - imported.as_secs_f64(),
            mesh.nodes.len(),
            mesh.depth,
            mesh.gpu_words(0, 0).len() as f64 * 4. / 1_048_576.,
            mesh.pixels.len() as f64 * 4. / 1_048_576.
        );
        Ok(mesh)
    }

    fn import_node(
        &mut self,
        node: &gltf::Node<'_>,
        parent: &glm::Mat4,
        buffers: &[gltf::buffer::Data],
        default_material: usize,
    ) -> LoadResult<()> {
        if node.skin().is_some() {
            return Err("Skinned meshes are not supported; export a static mesh".into());
        }
        let transform = parent * glm::Mat4::from_column_slice(&node.transform().matrix().concat());
        let linear = transform.fixed_view::<3, 3>(0, 0).into_owned();
        let inverse = linear.try_inverse().ok_or("Singular glTF node transform")?;
        let normal_matrix = inverse.transpose();
        let sign = linear.determinant().signum();
        if let Some(camera) = node.camera()
            && let gltf::camera::Projection::Perspective(perspective) = camera.projection()
        {
            self.camera.get_or_insert(Camera {
                eye_pos: (transform * glm::vec4(0., 0., 0., 1.)).xyz(),
                eye_dir: glm::normalize(&(linear * glm::vec3(0., 0., -1.))),
                up: glm::normalize(&(linear * glm::vec3(0., 1., 0.))),
                vfov: Angle::radians(perspective.yfov().min(std::f32::consts::FRAC_PI_2)),
                aperture: 0.,
                focus_distance: 1.,
            });
        }
        if let Some(source) = node.mesh() {
            for primitive in source.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    return Err("Only glTF TRIANGLES primitives are supported".into());
                }
                if primitive.morph_targets().next().is_some() {
                    return Err("Morph targets are not supported; export a static mesh".into());
                }
                if primitive
                    .attributes()
                    .any(|(semantic, _)| matches!(semantic, gltf::Semantic::Colors(_)))
                {
                    return Err(
                        "Vertex colors are not yet supported; bake them into base-color textures"
                            .into(),
                    );
                }
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions: Vec<_> = reader
                    .read_positions()
                    .ok_or("Mesh has no positions")?
                    .collect();
                let count = positions.len();
                let normals: Vec<_> = reader
                    .read_normals()
                    .map(Iterator::collect)
                    .unwrap_or_default();
                let uv0: Vec<_> = reader
                    .read_tex_coords(0)
                    .map(|x| x.into_f32().collect())
                    .unwrap_or(vec![[0.; 2]; count]);
                let uv1: Vec<_> = reader
                    .read_tex_coords(1)
                    .map(|x| x.into_f32().collect())
                    .unwrap_or(vec![[0.; 2]; count]);
                let tangents: Vec<_> = reader
                    .read_tangents()
                    .map(Iterator::collect)
                    .unwrap_or(vec![[0.; 4]; count]);
                if (!normals.is_empty() && normals.len() != count)
                    || uv0.len() != count
                    || uv1.len() != count
                    || tangents.len() != count
                {
                    return Err("Mismatched glTF vertex attribute counts".into());
                }
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|x| x.into_u32().collect())
                    .unwrap_or((0..u32::try_from(count)?).collect());
                if !indices.len().is_multiple_of(3) || indices.iter().any(|&i| i as usize >= count)
                {
                    return Err("Invalid triangle indices".into());
                }
                for set in required_uv_sets(&primitive.material())? {
                    if reader.read_tex_coords(set).is_none() {
                        return Err(format!("Material references missing TEXCOORD_{set}").into());
                    }
                }
                let offset = u32::try_from(self.vertices.len())?;
                for (i, position) in positions.into_iter().enumerate() {
                    let p =
                        (transform * glm::vec4(position[0], position[1], position[2], 1.)).xyz();
                    let n = normals
                        .get(i)
                        .map_or(glm::zero(), |n| normal_matrix * glm::Vec3::from(*n));
                    let n = n.try_normalize(1e-12).unwrap_or(glm::zero());
                    let tangent = tangents[i];
                    let t = (linear * glm::vec3(tangent[0], tangent[1], tangent[2]))
                        .try_normalize(1e-12)
                        .unwrap_or(glm::zero());
                    if !p
                        .iter()
                        .chain(n.iter())
                        .chain(uv0[i].iter())
                        .chain(uv1[i].iter())
                        .chain(t.iter())
                        .all(|x| x.is_finite())
                    {
                        return Err("Nonfinite glTF vertex attribute".into());
                    }
                    self.vertices.push(MeshVertex {
                        position: p.into(),
                        _pad0: 0.,
                        normal: n.into(),
                        _pad1: 0.,
                        uv0: uv0[i],
                        uv1: uv1[i],
                        tangent: [t.x, t.y, t.z, tangent[3] * sign],
                    });
                }
                for tri in indices.chunks_exact(3) {
                    let mut vertices = [tri[0] + offset, tri[1] + offset, tri[2] + offset];
                    // Mirrored transforms must preserve the original front face.
                    if sign < 0. {
                        vertices.swap(1, 2);
                    }
                    let [a, b, c] =
                        vertices.map(|i| glm::Vec3::from(self.vertices[i as usize].position));
                    if glm::length2(&glm::cross(&(b - a), &(c - a))) <= 1e-20 {
                        continue;
                    }
                    self.triangles.push(Triangle {
                        vertices,
                        material: u32::try_from(
                            primitive.material().index().unwrap_or(default_material),
                        )?,
                    });
                }
            }
        }
        for child in node.children() {
            self.import_node(&child, &transform, buffers, default_material)?;
        }
        Ok(())
    }

    pub fn gpu_words(&self, sphere_count: u32, light_count: u32) -> Vec<u32> {
        // Header, nodes (8), triangles (4), vertices (16), materials (48), texture descriptors (4).
        let mut words = vec![0; 16];
        words[0] = self.nodes.len() as u32;
        words[1] = self.triangles.len() as u32;
        words[6] = sphere_count;
        words[7] = light_count;
        if bvh_width() > 2 && !self.nodes.is_empty() {
            let mut wide = Vec::new();
            wide_node(self, 0, bvh_width() as usize, &mut wide);
            words[9] = 16;
            if quantized_bvh() {
                for record in wide.chunks_exact(8) {
                    if record[3] == u32::MAX {
                        words.extend([0, 0, 0, 0xffffff]);
                        continue;
                    }
                    assert!(
                        record[3] < 0xffffff && record[7] < 256,
                        "quantized BVH metadata overflow"
                    );
                    let mut values = [0u16; 6];
                    for axis in 0..3 {
                        values[axis] = outward_half(f32::from_bits(record[axis]), false);
                        values[axis + 3] = outward_half(f32::from_bits(record[axis + 4]), true);
                    }
                    for pair in values.chunks_exact(2) {
                        words.push(u32::from(pair[0]) | (u32::from(pair[1]) << 16));
                    }
                    words.push(record[3] | (record[7] << 24));
                }
            } else {
                words.extend(wide);
            }
        } else {
            words.extend_from_slice(bytemuck::cast_slice(&self.nodes));
        }
        words[2] = words.len() as u32;
        words.extend_from_slice(bytemuck::cast_slice(&self.triangles));
        words[3] = words.len() as u32;
        if option("--packed-shading") {
            assert!(
                self.vertices.iter().all(|v| v
                    .uv0
                    .iter()
                    .chain(v.uv1.iter())
                    .all(|x| x.abs() <= 65504.0)),
                "UVs exceed half-float range"
            );
            for v in &self.vertices {
                words.extend(v.position.map(f32::to_bits));
                words.push(pack_normal(v.normal));
                words.push(pack_normal([v.tangent[0], v.tangent[1], v.tangent[2]]));
                words.push(v.tangent[3].to_bits());
                words.push(pack_half(v.uv0));
                words.push(pack_half(v.uv1));
            }
        } else {
            words.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        }
        words[4] = words.len() as u32;
        for material in &self.materials {
            words.extend_from_slice(material);
        }
        words[5] = words.len() as u32;
        words.extend_from_slice(bytemuck::cast_slice(&self.textures));
        // Traversal-only triangle records avoid dependent vertex/material reads.
        words[8] = words.len() as u32;
        let mut opaque = vec![false; self.materials.len()];
        let mut coverage = vec![1.0f32; self.materials.len()];
        if option("--classify-opacity") || option("--approximate-alpha") {
            for (i, m) in self.materials.iter().enumerate() {
                if m[11] & 2 == 0 {
                    continue;
                }
                let factor = f32::from_bits(m[3]);
                let cutoff = f32::from_bits(m[10]);
                coverage[i] = if m[16] == u32::MAX {
                    if factor >= cutoff { 1.0 } else { 0.0 }
                } else {
                    let t = self.textures[m[16] as usize];
                    let data = &self.pixels[t[0] as usize..(t[0] + t[1] * t[2]) as usize];
                    data.iter()
                        .filter(|&&p| factor * ((p >> 24) as f32 / 255.0) >= cutoff)
                        .count() as f32
                        / data.len() as f32
                };
                opaque[i] = if m[16] == u32::MAX {
                    factor >= cutoff + 1e-5
                } else {
                    let t = self.textures[m[16] as usize];
                    self.pixels[t[0] as usize..(t[0] + t[1] * t[2]) as usize]
                        .iter()
                        .all(|p| factor * ((p >> 24) as f32 / 255.0) >= cutoff + 1e-5)
                };
            }
            eprintln!(
                "Opacity classification: {} fully opaque MASK materials",
                opaque.iter().filter(|&&v| v).count()
            );
        }
        for tri in &self.triangles {
            let a = glm::make_vec3(&self.vertices[tri.vertices[0] as usize].position);
            let e1 = glm::make_vec3(&self.vertices[tri.vertices[1] as usize].position) - a;
            let e2 = glm::make_vec3(&self.vertices[tri.vertices[2] as usize].position) - a;
            for value in [
                a.x,
                a.y,
                a.z,
                1e-8 * glm::length(&e1) * glm::length(&e2),
                e1.x,
                e1.y,
                e1.z,
            ] {
                words.push(value.to_bits());
            }
            words.push(tri.material);
            for value in [e2.x, e2.y, e2.z] {
                words.push(value.to_bits());
            }
            words.push(
                self.materials[tri.material as usize][11]
                    & if opaque[tri.material as usize] {
                        !2
                    } else {
                        u32::MAX
                    },
            );
        }
        if !self.opacity_cells.is_empty() {
            words[10] = words.len() as u32;
            words.extend_from_slice(&self.opacity_cells);
        }
        if !self.mip_headers.is_empty() {
            words[12] = words.len() as u32;
            words.extend_from_slice(bytemuck::cast_slice(&self.mip_headers));
            words[15] = words.len() as u32;
            words.extend_from_slice(bytemuck::cast_slice(&self.mip_levels));
        }
        if option("--approximate-alpha") {
            words[13] = words.len() as u32;
            words.extend(coverage.iter().map(|v| v.to_bits()));
        }
        if option("--blue-noise-jitter") {
            words[14] = words.len() as u32;
            for b in include_bytes!("../../assets/sampling/blue-noise-64.bin").chunks_exact(4) {
                words.push(u32::from_le_bytes(b.try_into().unwrap()));
            }
        }
        words
    }

    fn build_opacity_maps(&mut self) {
        fn wrap(x: i32, n: i32, mode: u32) -> usize {
            match mode {
                1 => x.clamp(0, n - 1) as usize,
                2 => {
                    let p = x.rem_euclid(2 * n);
                    if p < n {
                        p as usize
                    } else {
                        (2 * n - 1 - p) as usize
                    }
                }
                _ => x.rem_euclid(n) as usize,
            }
        }
        self.opacity_cells = vec![0; self.triangles.len()];
        let mut known = 0usize;
        let mut tested = 0usize;
        for (index, tri) in self.triangles.iter().enumerate() {
            let m = self.materials[tri.material as usize];
            if m[11] & 2 == 0 {
                continue;
            }
            let factor = f32::from_bits(m[3]);
            let cutoff = f32::from_bits(m[10]);
            let mut uv = [[0.0; 2]; 3];
            for k in 0..3 {
                let v = self.vertices[tri.vertices[k] as usize];
                let mut p = if m[17] == 1 { v.uv1 } else { v.uv0 };
                if option("--packed-shading") {
                    p = p.map(|x| half::f16::from_f32(x).to_f32());
                }
                p = [p[0] * f32::from_bits(m[20]), p[1] * f32::from_bits(m[21])];
                let angle = f32::from_bits(m[22]);
                uv[k] = [
                    f32::from_bits(m[18]) + angle.cos() * p[0] - angle.sin() * p[1],
                    f32::from_bits(m[19]) + angle.sin() * p[0] + angle.cos() * p[1],
                ];
            }
            for row in 0..4 {
                for col in 0..4 {
                    if row + col >= 4 {
                        continue;
                    }
                    tested += 1;
                    let state = if m[16] == u32::MAX {
                        if factor >= cutoff + 1e-5 {
                            1
                        } else if factor < cutoff - 1e-5 {
                            2
                        } else {
                            0
                        }
                    } else {
                        let d = self.textures[m[16] as usize];
                        let mut lo = [f32::INFINITY; 2];
                        let mut hi = [f32::NEG_INFINITY; 2];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let u = (col + dx) as f32 / 4.0;
                                let v = (row + dy) as f32 / 4.0;
                                for axis in 0..2 {
                                    let t = uv[0][axis]
                                        + u * (uv[1][axis] - uv[0][axis])
                                        + v * (uv[2][axis] - uv[0][axis]);
                                    lo[axis] = lo[axis].min(t);
                                    hi[axis] = hi[axis].max(t);
                                }
                            }
                        }
                        let x0 = (lo[0] * d[1] as f32 - 0.5).floor() as i32 - 1;
                        let x1 = (hi[0] * d[1] as f32 - 0.5).floor() as i32 + 2;
                        let y0 = (lo[1] * d[2] as f32 - 0.5).floor() as i32 - 1;
                        let y1 = (hi[1] * d[2] as f32 - 0.5).floor() as i32 + 2;
                        if lo.iter().chain(hi.iter()).any(|x| !x.is_finite())
                            || lo[0].abs().max(hi[0].abs()) * d[1] as f32 > 262144.0
                            || lo[1].abs().max(hi[1].abs()) * d[2] as f32 > 262144.0
                            || i64::from(x1 - x0 + 1) * i64::from(y1 - y0 + 1) > 256
                        {
                            0
                        } else {
                            let mut minimum = f32::INFINITY;
                            let mut maximum = f32::NEG_INFINITY;
                            for y in y0..=y1 {
                                for x in x0..=x1 {
                                    let p = self.pixels[d[0] as usize
                                        + wrap(y, d[2] as i32, (d[3] >> 2) & 3) * d[1] as usize
                                        + wrap(x, d[1] as i32, d[3] & 3)];
                                    let a = factor * ((p >> 24) as f32 / 255.0);
                                    minimum = minimum.min(a);
                                    maximum = maximum.max(a);
                                }
                            }
                            if minimum >= cutoff + 1e-5 {
                                1
                            } else if maximum < cutoff - 1e-5 {
                                2
                            } else {
                                0
                            }
                        }
                    };
                    if state != 0 {
                        known += 1;
                    }
                    self.opacity_cells[index] |= state << ((row * 4 + col) * 2);
                }
            }
        }
        eprintln!("Software opacity cells: {known}/{tested} conservatively classified");
    }

    fn build_mips(&mut self) {
        let mut used = vec![[false; 2]; self.textures.len()];
        for m in &self.materials {
            for (slot, color) in [(16, 1), (32, 1), (24, 0), (40, 0)] {
                if m[slot] != u32::MAX {
                    used[m[slot] as usize][color] = true;
                }
            }
        }
        let linear: [f32; 256] = std::array::from_fn(|i| {
            let x = i as f32 / 255.0;
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        });
        for (index, texture) in self.textures.clone().iter().enumerate() {
            for color in 0..2 {
                let first = self.mip_levels.len() as u32;
                self.mip_levels.push(*texture);
                let mut last = *texture;
                if used[index][color] {
                    while last[1] > 1 || last[2] > 1 {
                        let w = (last[1] / 2).max(1);
                        let h = (last[2] / 2).max(1);
                        let offset = self.pixels.len() as u32;
                        for y in 0..h {
                            for x in 0..w {
                                let mut sum = [0.0f32; 4];
                                let x0 = x as f32 * last[1] as f32 / w as f32;
                                let x1 = (x + 1) as f32 * last[1] as f32 / w as f32;
                                let y0 = y as f32 * last[2] as f32 / h as f32;
                                let y1 = (y + 1) as f32 * last[2] as f32 / h as f32;
                                let area = (x1 - x0) * (y1 - y0);
                                for sy in y0.floor() as u32..y1.ceil() as u32 {
                                    for sx in x0.floor() as u32..x1.ceil() as u32 {
                                        let weight = (((sx + 1) as f32).min(x1)
                                            - (sx as f32).max(x0))
                                            * (((sy + 1) as f32).min(y1) - (sy as f32).max(y0))
                                            / area;
                                        let p = self.pixels[(last[0]
                                            + sy.min(last[2] - 1) * last[1]
                                            + sx.min(last[1] - 1))
                                            as usize];
                                        for c in 0..4 {
                                            let v = ((p >> (c * 8)) & 255) as usize;
                                            sum[c] += weight
                                                * if color == 1 && c < 3 {
                                                    linear[v]
                                                } else {
                                                    v as f32 / 255.0
                                                };
                                        }
                                    }
                                }
                                let mut pixel = 0u32;
                                for c in 0..4 {
                                    let mut v = sum[c];
                                    if color == 1 && c < 3 {
                                        v = if v <= 0.0031308 {
                                            12.92 * v
                                        } else {
                                            1.055 * v.powf(1.0 / 2.4) - 0.055
                                        };
                                    }
                                    pixel |=
                                        ((v.clamp(0.0, 1.0) * 255.0).round() as u32) << (c * 8);
                                }
                                self.pixels.push(pixel);
                            }
                        }
                        last = [offset, w, h, texture[3]];
                        self.mip_levels.push(last);
                    }
                }
                self.mip_headers
                    .push([first, self.mip_levels.len() as u32 - first, 0, 0]);
            }
        }
        eprintln!(
            "Mip chains built: {} descriptors, {} MiB pixels",
            self.mip_levels.len(),
            self.pixels.len() * 4 / 1048576
        );
    }

    pub fn build_bvh(&mut self) {
        self.nodes.clear();
        self.depth = 0;
        if self.triangles.is_empty() {
            return;
        }
        build(
            &mut self.triangles,
            &self.vertices,
            &mut self.nodes,
            0,
            0,
            &mut self.depth,
        );
        if micro_opacity() {
            self.build_opacity_maps();
        }
    }
}

fn required_uv_sets(material: &gltf::Material<'_>) -> LoadResult<Vec<u32>> {
    let pbr = material.pbr_metallic_roughness();
    let mut sets = Vec::new();
    for info in [
        pbr.base_color_texture(),
        pbr.metallic_roughness_texture(),
        material.emissive_texture(),
    ]
    .into_iter()
    .flatten()
    {
        sets.push(
            info.texture_transform()
                .and_then(|t| t.tex_coord())
                .unwrap_or(info.tex_coord()),
        );
    }
    if let Some(info) = material.normal_texture() {
        let set = info
            .extension_value("KHR_texture_transform")
            .and_then(|t| t.get("texCoord"))
            .and_then(gltf::json::Value::as_u64)
            .map_or(info.tex_coord(), |x| x as u32);
        sets.push(set);
    }
    if sets.iter().any(|&set| set > 1) {
        return Err("Only TEXCOORD_0 and TEXCOORD_1 are supported".into());
    }
    Ok(sets)
}

fn import_material(material: &gltf::Material<'_>) -> LoadResult<[u32; 48]> {
    if material.alpha_mode() == gltf::material::AlphaMode::Blend {
        return Err("glTF alphaMode BLEND is not yet supported; use OPAQUE or MASK".into());
    }
    let mut words = [0; 48];
    let pbr = material.pbr_metallic_roughness();
    words[..4].copy_from_slice(&pbr.base_color_factor().map(f32::to_bits));
    let strength = material.emissive_strength().unwrap_or(1.);
    words[4..7].copy_from_slice(&material.emissive_factor().map(|x| (x * strength).to_bits()));
    words[7] = pbr.metallic_factor().to_bits();
    words[8] = pbr.roughness_factor().to_bits();
    words[9] = material
        .normal_texture()
        .map_or(1., |x| x.scale())
        .to_bits();
    words[10] = material.alpha_cutoff().unwrap_or(0.5).to_bits();
    words[11] = u32::from(material.double_sided())
        | (u32::from(material.alpha_mode() == gltf::material::AlphaMode::Mask) << 1);
    for (slot, info) in [
        pbr.base_color_texture(),
        pbr.metallic_roughness_texture(),
        material.emissive_texture(),
    ]
    .into_iter()
    .enumerate()
    {
        let offset = 16 + slot * 8;
        words[offset] = u32::MAX;
        if let Some(info) = info {
            words[offset] = info.texture().index() as u32;
            words[offset + 1] = info.tex_coord();
            words[offset + 4] = 1f32.to_bits();
            words[offset + 5] = 1f32.to_bits();
            if let Some(transform) = info.texture_transform() {
                words[offset + 1] = transform.tex_coord().unwrap_or(info.tex_coord());
                words[offset + 2..offset + 4]
                    .copy_from_slice(&transform.offset().map(f32::to_bits));
                words[offset + 4..offset + 6].copy_from_slice(&transform.scale().map(f32::to_bits));
                words[offset + 6] = transform.rotation().to_bits();
            }
        }
    }
    words[40] = u32::MAX;
    if let Some(info) = material.normal_texture() {
        words[40] = info.texture().index() as u32;
        words[41] = info.tex_coord();
        words[44] = 1f32.to_bits();
        words[45] = 1f32.to_bits();
        if let Some(transform) = info.extension_value("KHR_texture_transform") {
            if let Some(set) = transform
                .get("texCoord")
                .and_then(gltf::json::Value::as_u64)
            {
                words[41] = u32::try_from(set)?;
            }
            for (key, start, count) in [("offset", 42, 2), ("scale", 44, 2), ("rotation", 46, 1)] {
                if let Some(value) = transform.get(key) {
                    for i in 0..count {
                        let v = if count == 1 {
                            value.as_f64()
                        } else {
                            value.get(i).and_then(gltf::json::Value::as_f64)
                        }
                        .ok_or("Invalid normal texture transform")?;
                        let v = v as f32;
                        if !v.is_finite() {
                            return Err("Nonfinite texture transform".into());
                        }
                        words[start + i] = v.to_bits();
                    }
                }
            }
        }
    }
    required_uv_sets(material)?;
    Ok(words)
}

#[derive(Clone, Copy)]
struct Bounds {
    min: glm::Vec3,
    max: glm::Vec3,
}
impl Bounds {
    fn empty() -> Self {
        Self {
            min: glm::vec3(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: glm::vec3(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }
    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.zip_map(&other.min, f32::min),
            max: self.max.zip_map(&other.max, f32::max),
        }
    }
    fn area(self) -> f32 {
        let e = (self.max - self.min).map(|x| x.max(0.));
        2. * (e.x * e.y + e.y * e.z + e.z * e.x)
    }
}
fn triangle_bounds(tri: &Triangle, vertices: &[MeshVertex]) -> Bounds {
    tri.vertices.iter().fold(Bounds::empty(), |b, &v| {
        let p = glm::Vec3::from(vertices[v as usize].position);
        b.union(Bounds { min: p, max: p })
    })
}
fn build(
    triangles: &mut [Triangle],
    vertices: &[MeshVertex],
    nodes: &mut Vec<BvhNode>,
    first: usize,
    depth: usize,
    max_depth: &mut usize,
) -> u32 {
    const BINS: usize = 32;
    static BIN_COUNT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let bin_count = *BIN_COUNT.get_or_init(|| if option("--sah32") { 32 } else { 16 });
    *max_depth = (*max_depth).max(depth);
    let idx = nodes.len();
    let bounds = triangles.iter().fold(Bounds::empty(), |b, t| {
        b.union(triangle_bounds(t, vertices))
    });
    nodes.push(BvhNode {
        min: bounds.min.into(),
        first: first as u32,
        max: bounds.max.into(),
        count: triangles.len() as u32,
    });
    static LEAF: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let leaf_size = *LEAF.get_or_init(|| {
        let n = crate::argument_value("--bvh-leaf").map_or(4, |v| v.parse::<usize>().unwrap());
        assert!(n > 0);
        n
    });
    if triangles.len() <= leaf_size || depth >= 48 {
        return idx as u32;
    }
    let centroid = |t: &Triangle| {
        let b = triangle_bounds(t, vertices);
        (b.min + b.max) * 0.5
    };
    let centroids = triangles.iter().fold(Bounds::empty(), |b, t| {
        let p = centroid(t);
        b.union(Bounds { min: p, max: p })
    });
    let mut best = (f32::INFINITY, 0, 0);
    for axis in 0..3 {
        let extent = centroids.max[axis] - centroids.min[axis];
        if extent <= 1e-9 {
            continue;
        }
        let mut bins = [(Bounds::empty(), 0usize); BINS];
        for t in triangles.iter() {
            let bin = (((centroid(t)[axis] - centroids.min[axis]) / extent * bin_count as f32)
                as usize)
                .min(bin_count - 1);
            bins[bin].0 = bins[bin].0.union(triangle_bounds(t, vertices));
            bins[bin].1 += 1;
        }
        let mut right = [(Bounds::empty(), 0usize); BINS];
        for i in (0..bin_count - 1).rev() {
            right[i] = (
                right[i + 1].0.union(bins[i + 1].0),
                right[i + 1].1 + bins[i + 1].1,
            );
        }
        let mut left = (Bounds::empty(), 0usize);
        for i in 0..bin_count - 1 {
            left = (left.0.union(bins[i].0), left.1 + bins[i].1);
            if left.1 == 0 || right[i].1 == 0 {
                continue;
            }
            let cost = left.0.area() * left.1 as f32 + right[i].0.area() * right[i].1 as f32;
            if cost < best.0 {
                best = (cost, axis, i);
            }
        }
    }
    static STOP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *STOP.get_or_init(|| option("--sah-stop"))
        && triangles.len() <= 32
        && best.0 + bounds.area() >= bounds.area() * triangles.len() as f32
    {
        return idx as u32;
    }
    let mut mid = 0;
    if best.0.is_finite() {
        let axis = best.1;
        let extent = centroids.max[axis] - centroids.min[axis];
        for i in 0..triangles.len() {
            let bin = (((centroid(&triangles[i])[axis] - centroids.min[axis]) / extent
                * bin_count as f32) as usize)
                .min(bin_count - 1);
            if bin <= best.2 {
                triangles.swap(mid, i);
                mid += 1;
            }
        }
    }
    if mid == 0 || mid == triangles.len() {
        mid = triangles.len() / 2;
        let extent = centroids.max - centroids.min;
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };
        triangles
            .select_nth_unstable_by(mid, |a, b| centroid(a)[axis].total_cmp(&centroid(b)[axis]));
    }
    let (left, right) = triangles.split_at_mut(mid);
    build(left, vertices, nodes, first, depth + 1, max_depth);
    let right_idx = build(right, vertices, nodes, first + mid, depth + 1, max_depth);
    nodes[idx].first = right_idx;
    nodes[idx].count = 0;
    idx as u32
}

fn encode_scene_cache(mesh: &Mesh) -> Vec<u8> {
    let mut bytes = b"RTSCENE4".to_vec();
    let camera = mesh.camera.unwrap();
    let values = [
        camera.eye_pos.x,
        camera.eye_pos.y,
        camera.eye_pos.z,
        camera.eye_dir.x,
        camera.eye_dir.y,
        camera.eye_dir.z,
        camera.up.x,
        camera.up.y,
        camera.up.z,
        camera.vfov.as_degrees(),
        camera.aperture,
        camera.focus_distance,
    ];
    bytes.extend(bytemuck::cast_slice::<f32, u8>(&values));
    bytes.extend((mesh.depth as u32).to_le_bytes());
    for data in [
        bytemuck::cast_slice(mesh.vertices.as_slice()),
        bytemuck::cast_slice(mesh.triangles.as_slice()),
        bytemuck::cast_slice(mesh.nodes.as_slice()),
        bytemuck::cast_slice(mesh.materials.as_slice()),
        bytemuck::cast_slice(mesh.textures.as_slice()),
        bytemuck::cast_slice(mesh.pixels.as_slice()),
        bytemuck::cast_slice(mesh.mip_headers.as_slice()),
        bytemuck::cast_slice(mesh.mip_levels.as_slice()),
        bytemuck::cast_slice(mesh.opacity_cells.as_slice()),
    ] {
        bytes.extend((data.len() as u64).to_le_bytes());
        bytes.extend(data);
    }
    let checksum = cache_checksum(&bytes);
    bytes.extend(checksum.to_le_bytes());
    bytes
}
fn cache_checksum(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}
fn decode_scene_cache(mut bytes: &[u8]) -> Option<Mesh> {
    if bytes.len() < 8 {
        return None;
    }
    let (payload, checksum) = bytes.split_at(bytes.len() - 8);
    if cache_checksum(payload) != u64::from_le_bytes(checksum.try_into().ok()?) {
        return None;
    }
    bytes = payload;
    fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        if bytes.len() < n {
            return None;
        }
        let (a, b) = bytes.split_at(n);
        *bytes = b;
        Some(a)
    }
    fn array<T: bytemuck::Pod>(bytes: &mut &[u8]) -> Option<Vec<T>> {
        let size = u64::from_le_bytes(take(bytes, 8)?.try_into().ok()?) as usize;
        if size % std::mem::size_of::<T>() != 0 {
            return None;
        }
        Some(
            take(bytes, size)?
                .chunks_exact(std::mem::size_of::<T>())
                .map(bytemuck::pod_read_unaligned)
                .collect(),
        )
    }
    if take(&mut bytes, 8)? != b"RTSCENE4" {
        return None;
    }
    let c: Vec<f32> = take(&mut bytes, 48)?
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let depth = u32::from_le_bytes(take(&mut bytes, 4)?.try_into().ok()?) as usize;
    let mesh = Mesh {
        vertices: array(&mut bytes)?,
        triangles: array(&mut bytes)?,
        nodes: array(&mut bytes)?,
        materials: array(&mut bytes)?,
        textures: array(&mut bytes)?,
        pixels: array(&mut bytes)?,
        depth,
        mip_headers: array(&mut bytes)?,
        mip_levels: array(&mut bytes)?,
        opacity_cells: array(&mut bytes)?,
        camera: Some(Camera {
            eye_pos: glm::vec3(c[0], c[1], c[2]),
            eye_dir: glm::vec3(c[3], c[4], c[5]),
            up: glm::vec3(c[6], c[7], c[8]),
            vfov: Angle::degrees(c[9]),
            aperture: c[10],
            focus_distance: c[11],
        }),
    };
    if !bytes.is_empty() {
        return None;
    }
    Some(mesh)
}

fn outward_half(value: f32, upper: bool) -> u16 {
    assert!(
        value.is_finite() && value.abs() <= 65504.0,
        "bounds exceed half-float range"
    );
    let h = half::f16::from_f32(value);
    let mut bits = h.to_bits();
    if (upper && h.to_f32() < value) || (!upper && h.to_f32() > value) {
        if upper {
            bits = if bits & 0x8000 == 0 {
                bits + 1
            } else {
                bits - 1
            };
        } else {
            bits = if bits & 0x8000 == 0 {
                bits - 1
            } else {
                bits + 1
            };
        }
    }
    bits
}

#[cfg(test)]
mod local_tests {
    use super::*;
    #[test]
    fn quantized_bounds_are_outward_rounded() {
        for i in -10000..=10000 {
            let x = i as f32 * 0.1234567;
            assert!(half::f16::from_bits(outward_half(x, false)).to_f32() <= x);
            assert!(half::f16::from_bits(outward_half(x, true)).to_f32() >= x);
        }
        for x in [-1e-20, 1e-20, -0.0, 0.0, 65504.0, -65504.0] {
            assert!(half::f16::from_bits(outward_half(x, false)).to_f32() <= x);
            assert!(half::f16::from_bits(outward_half(x, true)).to_f32() >= x);
        }
    }
    #[test]
    fn mipmaps_include_odd_edges_and_decode_srgb_before_averaging() {
        let mut material = [0u32; 48];
        for slot in [16, 24, 32, 40] {
            material[slot] = u32::MAX;
        }
        material[16] = 0;
        material[24] = 0;
        let mut mesh = Mesh {
            textures: vec![[0, 3, 1, 0]],
            pixels: vec![0xff0000ff, 0xff00ff00, 0xffff0000],
            materials: vec![material],
            ..Default::default()
        };
        mesh.build_mips();
        for color in 0..2 {
            let header = mesh.mip_headers[color];
            let level = mesh.mip_levels[(header[0] + header[1] - 1) as usize];
            let pixel = mesh.pixels[level[0] as usize];
            for c in 0..3 {
                let v = ((pixel >> (c * 8)) & 255) as i32;
                let expected = if color == 0 { 85 } else { 156 };
                assert!((v - expected).abs() <= 1);
            }
        }
    }
    #[test]
    fn cooked_scene_roundtrip_and_truncation_rejection() {
        let mesh = Mesh::load_uncached(Path::new("tests/assets/triangles.gltf")).unwrap();
        let bytes = encode_scene_cache(&mesh);
        let decoded = decode_scene_cache(&bytes).unwrap();
        assert_eq!(mesh.gpu_words(0, 0), decoded.gpu_words(0, 0));
        assert_eq!(mesh.pixels, decoded.pixels);
        assert!(decode_scene_cache(&bytes[..bytes.len() - 1]).is_none());
    }
}
