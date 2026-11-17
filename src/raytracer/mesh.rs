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

/// A rigid moving object (see `Mesh::append_object`): its triangles follow the scene's,
/// its own BVH (leaf triangle indices global) is packed after the scene's, and its
/// per-frame transform comes from the renderer's instance buffer.
pub struct MeshObject {
    pub nodes: Vec<BvhNode>,
    /// Object-space bounds.
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub first_triangle: u32,
}

impl MeshVertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            _pad0: 0.,
            normal,
            _pad1: 0.,
            uv0: uv,
            uv1: [0.; 2],
            tangent: [0.; 4],
        }
    }
}

/// Untextured, double-sided metallic/roughness material words (see import_material).
pub fn plain_material(color: [f32; 3], metallic: f32, roughness: f32) -> [u32; 48] {
    let mut words = [0u32; 48];
    words[..4].copy_from_slice(&[color[0], color[1], color[2], 1.].map(f32::to_bits));
    words[7] = metallic.to_bits();
    words[8] = roughness.to_bits();
    words[9] = 1f32.to_bits();
    words[10] = 0.5f32.to_bits();
    words[11] = 1;
    for slot in super::gpu_textures::SLOTS {
        words[slot] = u32::MAX;
    }
    words
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
    pub opacity_cells: Vec<u32>,
    /// Rigid moving objects after the scene's triangles (instance i = object i).
    pub objects: Vec<MeshObject>,
}

type LoadResult<T> = Result<T, Box<dyn Error>>;

pub fn micro_opacity() -> bool {
    option("--opacity-micromap") || std::env::var("RT_MICRO_OPACITY").is_ok()
}
pub fn quantized_bvh() -> bool {
    option("--quantized-bvh") || std::env::var("RT_QUANTIZED_BVH").is_ok()
}
/// Aila-Laine style records: each 64-byte record holds both children's boxes.
pub fn pair_bvh() -> bool {
    option("--pair-bvh")
}
/// Child reference high bit: the low bits are the leaf's first triangle.
const PAIR_LEAF: u32 = 0x8000_0000;
/// Binary BVH -> child-pair records, depth-first so a parent's near child is close.
/// Record words: left min.xyz, left ref, left max.xyz, 0, right min.xyz, right ref, right max.xyz, 0.
fn pair_records(nodes: &[BvhNode]) -> Vec<u32> {
    fn reference(nodes: &[BvhNode], node: usize, words: &mut Vec<u32>) -> u32 {
        if nodes[node].count != 0 {
            assert!(
                nodes[node].first < PAIR_LEAF,
                "too many triangles for pair BVH"
            );
            return PAIR_LEAF | nodes[node].first;
        }
        let record = words.len() / 16;
        words.resize(words.len() + 16, 0);
        for (side, child) in [node + 1, nodes[node].first as usize]
            .into_iter()
            .enumerate()
        {
            let link = reference(nodes, child, words);
            let base = record * 16 + side * 8;
            for axis in 0..3 {
                words[base + axis] = nodes[child].min[axis].to_bits();
                words[base + 4 + axis] = nodes[child].max[axis].to_bits();
            }
            words[base + 3] = link;
        }
        record as u32
    }
    let mut words = Vec::new();
    if nodes[0].count != 0 {
        // A single-leaf tree: the root record's left child is the leaf; the right
        // child is a degenerate box beyond the ray interval that no ray can enter.
        words.resize(16, 0);
        for axis in 0..3 {
            words[axis] = nodes[0].min[axis].to_bits();
            words[4 + axis] = nodes[0].max[axis].to_bits();
            words[8 + axis] = 1e30f32.to_bits();
            words[12 + axis] = 1e30f32.to_bits();
        }
        words[3] = PAIR_LEAF | nodes[0].first;
        words[11] = PAIR_LEAF;
    } else {
        reference(nodes, 0, &mut words);
    }
    words
}

/// Four-wide records (128 bytes): SoA child bounds and four child references.
pub fn quad_bvh() -> bool {
    option("--quad-bvh") || oct_bvh()
}
/// Leaf references carry (count - 1) in bits 26..31 instead of a last-triangle flag.
pub fn quad_leaf_count() -> bool {
    // Default for 4-wide records; RT_QUAD_LEAF_FLAG restores the flag encoding.
    quad_bvh() && std::env::var("RT_QUAD_LEAF_FLAG").is_err()
}
fn leaf_reference(node: &BvhNode) -> u32 {
    if quad_leaf_count() {
        assert!(
            node.first < (1 << 26) - 1 && (1..=32).contains(&node.count),
            "leaf does not fit a quad reference"
        );
        PAIR_LEAF | ((node.count - 1) << 26) | node.first
    } else {
        PAIR_LEAF | node.first
    }
}
/// Eight-wide records (`--oct-bvh`, implies the 4-wide traversal options):
/// 256 bytes, SoA bounds for 8 children, then 8 child references.
/// Hardware ray queries through wgpu acceleration structures (`--hw-rt`).
pub fn hw_rt() -> bool {
    option("--hw-rt")
}
pub fn oct_bvh() -> bool {
    option("--oct-bvh")
}
/// Collapse the binary SAH tree into `WIDTH`-wide SoA records by repeatedly
/// opening the largest-area interior child. Words per record: for each axis,
/// min[WIDTH] then max[WIDTH]; then refs[WIDTH]; padded to 8*WIDTH words.
fn wide_soa_records<const WIDTH: usize>(nodes: &[BvhNode]) -> Vec<u32> {
    fn area(n: &BvhNode) -> f32 {
        let e = [0, 1, 2].map(|a| n.max[a] - n.min[a]);
        e[0] * e[1] + e[1] * e[2] + e[2] * e[0]
    }
    fn record<const WIDTH: usize>(nodes: &[BvhNode], node: usize, words: &mut Vec<u32>) -> u32 {
        let stride = 8 * WIDTH;
        let index = words.len() / stride;
        words.resize(words.len() + stride, 0);
        let mut children = vec![node + 1, nodes[node].first as usize];
        while children.len() < WIDTH {
            let Some(slot) = (0..children.len())
                .filter(|&i| nodes[children[i]].count == 0)
                .max_by(|&a, &b| area(&nodes[children[a]]).total_cmp(&area(&nodes[children[b]])))
            else {
                break;
            };
            let parent = children.remove(slot);
            children.insert(slot, nodes[parent].first as usize);
            children.insert(slot, parent + 1);
        }
        let base = index * stride;
        for lane in 0..WIDTH {
            let Some(&child) = children.get(lane) else {
                for axis in 0..3 {
                    words[base + axis * 2 * WIDTH + lane] = 1e30f32.to_bits();
                    words[base + axis * 2 * WIDTH + WIDTH + lane] = 1e30f32.to_bits();
                }
                words[base + 6 * WIDTH + lane] = PAIR_LEAF;
                continue;
            };
            for axis in 0..3 {
                words[base + axis * 2 * WIDTH + lane] = nodes[child].min[axis].to_bits();
                words[base + axis * 2 * WIDTH + WIDTH + lane] = nodes[child].max[axis].to_bits();
            }
            words[base + 6 * WIDTH + lane] = if nodes[child].count != 0 {
                leaf_reference(&nodes[child])
            } else {
                record::<WIDTH>(nodes, child, words)
            };
        }
        index as u32
    }
    let mut words = Vec::new();
    if nodes[0].count != 0 {
        words.resize(8 * WIDTH, 0);
        for lane in 0..WIDTH {
            for axis in 0..3 {
                let (lo, hi) = if lane == 0 {
                    (nodes[0].min[axis], nodes[0].max[axis])
                } else {
                    (1e30, 1e30)
                };
                words[axis * 2 * WIDTH + lane] = lo.to_bits();
                words[axis * 2 * WIDTH + WIDTH + lane] = hi.to_bits();
            }
            words[6 * WIDTH + lane] = if lane == 0 {
                leaf_reference(&nodes[0])
            } else {
                PAIR_LEAF
            };
        }
    } else {
        record::<WIDTH>(nodes, 0, &mut words);
    }
    words
}

/// Collapse the binary SAH tree: repeatedly open the largest-area interior child.
/// Record words: min.x[4], max.x[4], min.y[4], max.y[4], min.z[4], max.z[4], refs[4], pad[4].
fn quad_records(nodes: &[BvhNode]) -> Vec<u32> {
    fn area(n: &BvhNode) -> f32 {
        let e = [0, 1, 2].map(|a| n.max[a] - n.min[a]);
        e[0] * e[1] + e[1] * e[2] + e[2] * e[0]
    }
    fn record(nodes: &[BvhNode], node: usize, words: &mut Vec<u32>) -> u32 {
        let index = words.len() / 32;
        words.resize(words.len() + 32, 0);
        let mut children = vec![node + 1, nodes[node].first as usize];
        while children.len() < 4 {
            let Some(slot) = (0..children.len())
                .filter(|&i| nodes[children[i]].count == 0)
                .max_by(|&a, &b| area(&nodes[children[a]]).total_cmp(&area(&nodes[children[b]])))
            else {
                break;
            };
            let parent = children.remove(slot);
            children.insert(slot, nodes[parent].first as usize);
            children.insert(slot, parent + 1);
        }
        let base = index * 32;
        for lane in 0..4 {
            let Some(&child) = children.get(lane) else {
                // Empty slot: a degenerate box beyond the ray interval.
                for axis in 0..3 {
                    words[base + axis * 8 + lane] = 1e30f32.to_bits();
                    words[base + axis * 8 + 4 + lane] = 1e30f32.to_bits();
                }
                words[base + 24 + lane] = PAIR_LEAF;
                continue;
            };
            for axis in 0..3 {
                words[base + axis * 8 + lane] = nodes[child].min[axis].to_bits();
                words[base + axis * 8 + 4 + lane] = nodes[child].max[axis].to_bits();
            }
            words[base + 24 + lane] = if nodes[child].count != 0 {
                leaf_reference(&nodes[child])
            } else {
                record(nodes, child, words)
            };
        }
        index as u32
    }
    let mut words = Vec::new();
    if nodes[0].count != 0 {
        words.resize(32, 0);
        for lane in 0..4 {
            for axis in 0..3 {
                let (lo, hi) = if lane == 0 {
                    (nodes[0].min[axis], nodes[0].max[axis])
                } else {
                    (1e30, 1e30)
                };
                words[axis * 8 + lane] = lo.to_bits();
                words[axis * 8 + 4 + lane] = hi.to_bits();
            }
            words[24 + lane] = if lane == 0 {
                leaf_reference(&nodes[0])
            } else {
                PAIR_LEAF
            };
        }
    } else {
        record(nodes, 0, &mut words);
    }
    words
}

/// Quad records quantized to 64 bytes: parent origin, power-of-two scales and
/// outward-rounded 8-bit child bounds (experiment: RT_QUAD_Q8).
pub fn quad_q8() -> bool {
    std::env::var("RT_QUAD_Q8").is_ok()
}
/// Empty child slot reference for quantized quad records.
const QUAD_EMPTY: u32 = u32::MAX;
/// 128-byte float quad records -> 64-byte quantized records (same indices).
/// Words: origin.xyz, packed biased exponents, lo.x hi.x lo.y hi.y lo.z hi.z (4 x u8 each), 0, 0, refs[4].
fn quantize_quad(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(words.len() / 2);
    for record in words.chunks_exact(32) {
        let f = |i: usize| f32::from_bits(record[i]);
        let live: Vec<usize> = (0..4).filter(|&l| f(l) < 1e29).collect();
        let mut q = [0u32; 16];
        let mut exponents = 0u32;
        for axis in 0..3 {
            let lo = live
                .iter()
                .map(|&l| f(axis * 8 + l))
                .fold(f32::INFINITY, f32::min);
            let hi = live
                .iter()
                .map(|&l| f(axis * 8 + 4 + l))
                .fold(f32::NEG_INFINITY, f32::max);
            let (lo, hi) = if live.is_empty() { (0., 0.) } else { (lo, hi) };
            // Smallest power of two with 255 steps spanning the parent extent.
            let mut exponent = ((hi - lo).max(1e-30) / 255.).log2().ceil() as i32;
            let scale = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
            while lo + 255. * scale(exponent) < hi {
                exponent += 1;
            }
            let step = scale(exponent);
            q[axis] = lo.to_bits();
            exponents |= ((exponent + 127) as u32) << (8 * axis);
            for lane in 0..4 {
                let (mut a, mut b) = (0u32, 0u32);
                if live.contains(&lane) {
                    let (cmin, cmax) = (f(axis * 8 + lane), f(axis * 8 + 4 + lane));
                    a = (((cmin - lo) / step).floor().max(0.) as u32).min(255);
                    // Decode exactly as the shader does and widen until conservative.
                    while a > 0 && lo + a as f32 * step > cmin {
                        a -= 1;
                    }
                    b = (((cmax - lo) / step).ceil().max(0.) as u32).min(255);
                    while b < 255 && lo + b as f32 * step < cmax {
                        b += 1;
                    }
                }
                q[4 + axis * 2] |= a << (8 * lane);
                q[5 + axis * 2] |= b << (8 * lane);
            }
        }
        q[3] = exponents;
        for lane in 0..4 {
            q[12 + lane] = if live.contains(&lane) {
                record[24 + lane]
            } else {
                QUAD_EMPTY
            };
        }
        out.extend_from_slice(&q);
    }
    out
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
    crate::has_flag(name)
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
    /// Add a rigid moving object (geometry in object space, one material). Its BVH is
    /// built on its own (same builder and options as the scene) and traversed after the
    /// scene's with the ray in object space. Needs the 4-wide BVH.
    pub fn append_object(
        &mut self,
        vertices: Vec<MeshVertex>,
        triangles: Vec<[u32; 3]>,
        material: [u32; 48],
    ) {
        assert!(quad_bvh(), "moving objects need --quad-bvh (or --fastest)");
        let mut object = Mesh {
            vertices,
            triangles: triangles
                .iter()
                .map(|&vertices| Triangle {
                    vertices,
                    material: 0,
                })
                .collect(),
            materials: vec![material],
            ..Default::default()
        };
        object.build_bvh();
        let (vertex_offset, triangle_offset) =
            (self.vertices.len() as u32, self.triangles.len() as u32);
        let material = self.materials.len() as u32;
        self.vertices.extend(&object.vertices);
        self.triangles
            .extend(object.triangles.iter().map(|t| Triangle {
                vertices: t.vertices.map(|v| v + vertex_offset),
                material,
            }));
        self.materials.push(object.materials[0]);
        let mut nodes = object.nodes;
        for node in nodes.iter_mut().filter(|n| n.count != 0) {
            node.first += triangle_offset;
        }
        let (min, max) = (nodes[0].min, nodes[0].max);
        self.objects.push(MeshObject {
            nodes,
            min,
            max,
            first_triangle: triangle_offset,
        });
    }

    /// Triangles of the static scene (moving objects' triangles follow them).
    pub fn scene_triangle_count(&self) -> usize {
        self.objects
            .first()
            .map_or(self.triangles.len(), |o| o.first_triangle as usize)
    }

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
            option("--sbvh").hash(&mut hash);
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
        let mesh = Self::load_uncached(path)?;
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
        let mut object_roots: Vec<u32> = Vec::new();
        words[0] = self.nodes.len() as u32;
        words[1] = self.triangles.len() as u32;
        words[6] = sphere_count;
        words[7] = light_count;
        if oct_bvh() && !self.nodes.is_empty() {
            words.extend(wide_soa_records::<8>(&self.nodes));
        } else if quad_bvh() && !self.nodes.is_empty() {
            let records = quad_records(&self.nodes);
            if quad_q8() {
                assert!(
                    self.objects.is_empty(),
                    "moving objects do not support RT_QUAD_Q8"
                );
                words.extend(quantize_quad(&records));
            } else {
                words.extend(records);
            }
            // Moving objects' BVHs follow, child record references offset; their roots
            // are listed at header word 15.
            let mut roots = Vec::new();
            for object in &self.objects {
                let base = ((words.len() - 16) / 32) as u32;
                let mut records = quad_records(&object.nodes);
                for record in records.chunks_exact_mut(32) {
                    for reference in &mut record[24..28] {
                        if *reference & PAIR_LEAF == 0 {
                            *reference += base;
                        }
                    }
                }
                roots.push(base);
                words.extend(records);
            }
            object_roots = roots;
        } else if pair_bvh() && !self.nodes.is_empty() {
            words.extend(pair_records(&self.nodes));
        } else if bvh_width() > 2 && !self.nodes.is_empty() {
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
        // GPU materials name sampled GPU textures (see gpu_textures): slot word 0
        // is the GPU texture index plus one (0 = none), word 7 the sampler index.
        let (_, slots) = self.texture_plan();
        for (material, indices) in self.materials.iter().zip(&slots) {
            let mut material = *material;
            for (slot, index) in super::gpu_textures::SLOTS.into_iter().zip(indices) {
                if material[slot] == u32::MAX {
                    material[slot] = 0;
                } else {
                    let flags = self.textures[material[slot] as usize][3];
                    material[slot] = *index + 1;
                    material[slot + 7] = super::gpu_textures::sampler_index(flags);
                }
            }
            words.extend_from_slice(&material);
        }
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
        // Pair traversal walks a leaf's triangles until the one flagged last (bit 2).
        let mut leaf_last = vec![false; self.triangles.len()];
        let object_nodes = self.objects.iter().flat_map(|o| &o.nodes);
        for node in self
            .nodes
            .iter()
            .chain(object_nodes)
            .filter(|n| n.count != 0)
        {
            leaf_last[(node.first + node.count - 1) as usize] = true;
        }
        for (index, tri) in self.triangles.iter().enumerate() {
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
                    }
                    | if leaf_last[index] { 4 } else { 0 },
            );
        }
        if hw_rt() {
            // Triangle index lists for the two BLAS geometries (see hwrt.wgsl).
            let [opaque, shaded] = self.hw_geometry_triangles();
            words[11] = words.len() as u32;
            words.extend(opaque);
            words[5] = words.len() as u32;
            words.extend(shaded);
        }
        if !object_roots.is_empty() {
            words[15] = words.len() as u32;
            words.extend(&object_roots);
        }
        if !self.opacity_cells.is_empty() {
            words[10] = words.len() as u32;
            words.extend_from_slice(&self.opacity_cells);
        }
        if option("--approximate-alpha") {
            words[13] = words.len() as u32;
            words.extend(coverage.iter().map(|v| v.to_bits()));
        }
        if option("--blue-noise-jitter") || option("--blue-noise-path") {
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

    /// Triangles for the hardware BLAS: [opaque double-sided, needs the shader
    /// (alpha-tested or single-sided)], as indices into `triangles`.
    /// Unique triangles for the visibility raster (spatial splits duplicate
    /// references): [double-sided opaque, single-sided opaque, alpha-tested].
    pub fn visibility_triangles(&self) -> [Vec<u32>; 3] {
        let mut seen = std::collections::HashSet::new();
        let mut lists = [Vec::new(), Vec::new(), Vec::new()];
        for (index, tri) in self.triangles[..self.scene_triangle_count()]
            .iter()
            .enumerate()
        {
            if !seen.insert((tri.vertices, tri.material)) {
                continue;
            }
            let flags = self.materials[tri.material as usize][11];
            let list = if flags & 2 != 0 {
                2
            } else if flags & 1 != 0 {
                0
            } else {
                1
            };
            lists[list].push(index as u32);
        }
        lists
    }

    pub fn hw_geometry_triangles(&self) -> [Vec<u32>; 2] {
        let mut lists = [Vec::new(), Vec::new()];
        for (index, tri) in self.triangles[..self.scene_triangle_count()]
            .iter()
            .enumerate()
        {
            let flags = self.materials[tri.material as usize][11];
            lists[usize::from(flags & 3 != 1)].push(index as u32);
        }
        lists
    }

    pub fn build_bvh(&mut self) {
        self.nodes.clear();
        self.depth = 0;
        if self.triangles.is_empty() {
            return;
        }
        if option("--sbvh") {
            let start = Instant::now();
            let before = self.triangles.len();
            let leaf = crate::argument_value("--bvh-leaf")
                .map_or(default_leaf_size(), |v| v.parse::<usize>().unwrap());
            (self.triangles, self.nodes, self.depth) = sbvh::build(
                &self.triangles,
                &self.vertices,
                leaf,
                traversal_cost(),
                option("--sah-stop"),
            );
            eprintln!(
                "SBVH: {} -> {} triangle references (+{:.1}%), {:.2}s",
                before,
                self.triangles.len(),
                100. * (self.triangles.len() as f64 / before as f64 - 1.),
                start.elapsed().as_secs_f64()
            );
        } else {
            build(
                &mut self.triangles,
                &self.vertices,
                &mut self.nodes,
                0,
                0,
                &mut self.depth,
            );
        }
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
        let n = crate::argument_value("--bvh-leaf")
            .map_or(default_leaf_size(), |v| v.parse::<usize>().unwrap());
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
        && best.0 + traversal_cost() * bounds.area() >= bounds.area() * triangles.len() as f32
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

/// Minimum leaf size when `--bvh-leaf` is not given: 1 with `--fastest` (SAH
/// stopping still forms larger leaves where they pay), else the historical 4.
fn default_leaf_size() -> usize {
    if crate::has_flag("--fastest") { 1 } else { 4 }
}

/// SAH node-traversal cost relative to one triangle test (experiment: RT_SAH_CT).
fn traversal_cost() -> f32 {
    static COST: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *COST.get_or_init(|| std::env::var("RT_SAH_CT").map_or(1., |v| v.parse().unwrap()))
}

fn encode_scene_cache(mesh: &Mesh) -> Vec<u8> {
    let mut bytes = b"RTSCENE5".to_vec();
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
    if take(&mut bytes, 8)? != b"RTSCENE5" {
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
        opacity_cells: array(&mut bytes)?,
        objects: Vec::new(),
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
        let base = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        for (srgb, expected) in [(false, 85), (true, 156)] {
            let levels = super::super::gpu_textures::mip_chain(&base, 3, 1, srgb);
            assert_eq!(levels.len(), 2);
            let (w, h, last) = levels.last().unwrap();
            assert_eq!((*w, *h), (1, 1));
            for c in 0..3 {
                assert!((i32::from(last[0][c]) - expected).abs() <= 1);
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

/// Spatial-split BVH (Stich, Friedrich, Dietrich 2009) over triangle references.
/// A reference is a triangle clipped to a box; where the best object split's
/// children overlap noticeably, binned spatial splits are also evaluated and
/// straddling references are split into both children. Leaves list (possibly
/// duplicated) triangles, so the output is an expanded triangle array in leaf order.
mod sbvh {
    use super::{Bounds, BvhNode, MeshVertex, Triangle};

    /// Bin array capacity; the counts used come from `knob` (experiments only).
    const MAX_BINS: usize = 64;
    fn knob(name: &str, default: f32) -> f32 {
        std::env::var(name).map_or(default, |v| v.parse().unwrap())
    }
    /// Extra references allowed, as a fraction of the triangle count.
    const BUDGET: f32 = 0.3;

    #[derive(Clone, Copy)]
    struct Ref {
        tri: u32,
        bounds: Bounds,
    }

    struct Builder<'a> {
        triangles: &'a [Triangle],
        vertices: &'a [MeshVertex],
        nodes: Vec<BvhNode>,
        out: Vec<Triangle>,
        root_area: f32,
        budget: usize,
        depth: usize,
        leaf_size: usize,
        traversal_cost: f32,
        stop: bool,
        object_bins: usize,
        spatial_bins: usize,
        /// Spatial splits are tried only when the object split's children overlap
        /// by more than this fraction of the root's surface area (the paper's alpha).
        overlap: f32,
    }

    fn positions(t: &Triangle, v: &[MeshVertex]) -> [glm::Vec3; 3] {
        t.vertices.map(|i| glm::Vec3::from(v[i as usize].position))
    }
    fn intersect(a: Bounds, b: Bounds) -> Bounds {
        Bounds {
            min: a.min.zip_map(&b.min, f32::max),
            max: a.max.zip_map(&b.max, f32::min),
        }
    }
    fn centroid(b: &Bounds, axis: usize) -> f32 {
        0.5 * (b.min[axis] + b.max[axis])
    }

    /// Bounds of the triangle restricted to lo <= x[axis] <= hi, within `within`.
    fn clip(p: &[glm::Vec3; 3], axis: usize, lo: f32, hi: f32, within: Bounds) -> Bounds {
        let mut b = Bounds::empty();
        let mut add = |q: glm::Vec3| b = b.union(Bounds { min: q, max: q });
        for i in 0..3 {
            let (a, c) = (p[i], p[(i + 1) % 3]);
            if a[axis] >= lo && a[axis] <= hi {
                add(a);
            }
            for plane in [lo, hi] {
                let (x0, x1) = (a[axis], c[axis]);
                if (x0 < plane && x1 > plane) || (x0 > plane && x1 < plane) {
                    let t = (plane - x0) / (x1 - x0);
                    let mut q = a + (c - a) * t;
                    q[axis] = plane;
                    add(q);
                }
            }
        }
        intersect(b, within)
    }

    impl Builder<'_> {
        fn leaf(&mut self, node: usize, refs: &[Ref]) {
            self.nodes[node].first = self.out.len() as u32;
            self.nodes[node].count = refs.len() as u32;
            self.out
                .extend(refs.iter().map(|r| self.triangles[r.tri as usize]));
        }

        fn build(&mut self, refs: Vec<Ref>, depth: usize) -> u32 {
            self.depth = self.depth.max(depth);
            let node = self.nodes.len();
            let bounds = refs.iter().fold(Bounds::empty(), |b, r| b.union(r.bounds));
            self.nodes.push(BvhNode {
                min: bounds.min.into(),
                first: 0,
                max: bounds.max.into(),
                count: 0,
            });
            let n = refs.len();
            if n <= self.leaf_size || depth >= 48 {
                self.leaf(node, &refs);
                return node as u32;
            }
            // Object split: binned SAH over reference centroids.
            let centroids = refs.iter().fold(Bounds::empty(), |b, r| {
                let c = (r.bounds.min + r.bounds.max) * 0.5;
                b.union(Bounds { min: c, max: c })
            });
            let mut best = (
                f32::INFINITY,
                0usize,
                0usize,
                Bounds::empty(),
                Bounds::empty(),
            );
            for axis in 0..3 {
                let extent = centroids.max[axis] - centroids.min[axis];
                if extent <= 1e-9 {
                    continue;
                }
                let object_bins = self.object_bins;
                let bin = |r: &Ref| {
                    (((centroid(&r.bounds, axis) - centroids.min[axis]) / extent
                        * object_bins as f32) as usize)
                        .min(object_bins - 1)
                };
                let mut bins = [(Bounds::empty(), 0usize); MAX_BINS];
                for r in &refs {
                    let b = &mut bins[bin(r)];
                    b.0 = b.0.union(r.bounds);
                    b.1 += 1;
                }
                let mut right = [(Bounds::empty(), 0usize); MAX_BINS];
                for i in (0..object_bins - 1).rev() {
                    right[i] = (
                        right[i + 1].0.union(bins[i + 1].0),
                        right[i + 1].1 + bins[i + 1].1,
                    );
                }
                let mut left = (Bounds::empty(), 0usize);
                for i in 0..object_bins - 1 {
                    left = (left.0.union(bins[i].0), left.1 + bins[i].1);
                    if left.1 == 0 || right[i].1 == 0 {
                        continue;
                    }
                    let cost =
                        left.0.area() * left.1 as f32 + right[i].0.area() * right[i].1 as f32;
                    if cost < best.0 {
                        best = (cost, axis, i, left.0, right[i].0);
                    }
                }
            }
            // Spatial split, only where object-split children overlap.
            let mut spatial: Option<(f32, usize, f32)> = None;
            let overlap = intersect(best.3, best.4);
            let overlaps = (0..3).all(|a| overlap.min[a] <= overlap.max[a])
                && overlap.area() > self.overlap * self.root_area;
            if self.budget > 0 && (overlaps || !best.0.is_finite()) {
                for axis in 0..3 {
                    let (lo, hi) = (bounds.min[axis], bounds.max[axis]);
                    let spatial_bins = self.spatial_bins;
                    let width = (hi - lo) / spatial_bins as f32;
                    if width <= 1e-9 {
                        continue;
                    }
                    let mut bins = [(Bounds::empty(), 0usize, 0usize); MAX_BINS];
                    let index = |x: f32| (((x - lo) / width) as usize).min(spatial_bins - 1);
                    for r in &refs {
                        let (first, last) = (index(r.bounds.min[axis]), index(r.bounds.max[axis]));
                        bins[first].1 += 1;
                        bins[last].2 += 1;
                        if first == last {
                            bins[first].0 = bins[first].0.union(r.bounds);
                            continue;
                        }
                        let p = positions(&self.triangles[r.tri as usize], self.vertices);
                        for b in first..=last {
                            let piece = clip(
                                &p,
                                axis,
                                lo + b as f32 * width,
                                lo + (b + 1) as f32 * width,
                                r.bounds,
                            );
                            bins[b].0 = bins[b].0.union(piece);
                        }
                    }
                    let mut right = [(Bounds::empty(), 0usize); MAX_BINS];
                    for i in (0..spatial_bins - 1).rev() {
                        right[i] = (
                            right[i + 1].0.union(bins[i + 1].0),
                            right[i + 1].1 + bins[i + 1].2,
                        );
                    }
                    let mut left = (Bounds::empty(), 0usize);
                    for i in 0..spatial_bins - 1 {
                        left = (left.0.union(bins[i].0), left.1 + bins[i].1);
                        if left.1 == 0 || right[i].1 == 0 {
                            continue;
                        }
                        let cost =
                            left.0.area() * left.1 as f32 + right[i].0.area() * right[i].1 as f32;
                        if cost < best.0 && spatial.is_none_or(|s| cost < s.0) {
                            spatial = Some((cost, axis, lo + (i + 1) as f32 * width));
                        }
                    }
                }
            }
            let split_cost = spatial.map_or(best.0, |s| s.0);
            let area = bounds.area();
            if self.stop && n <= 32 && split_cost + self.traversal_cost * area >= area * n as f32 {
                self.leaf(node, &refs);
                return node as u32;
            }
            let (mut left, mut right) = (Vec::new(), Vec::new());
            if let Some((_, axis, plane)) = spatial {
                for r in refs {
                    if r.bounds.max[axis] <= plane {
                        left.push(r);
                    } else if r.bounds.min[axis] >= plane {
                        right.push(r);
                    } else {
                        let p = positions(&self.triangles[r.tri as usize], self.vertices);
                        let l = clip(&p, axis, f32::NEG_INFINITY, plane, r.bounds);
                        let h = clip(&p, axis, plane, f32::INFINITY, r.bounds);
                        let valid = |b: &Bounds| (0..3).all(|a| b.min[a] <= b.max[a]);
                        match (valid(&l), valid(&h)) {
                            (true, true) => {
                                self.budget = self.budget.saturating_sub(1);
                                left.push(Ref {
                                    tri: r.tri,
                                    bounds: l,
                                });
                                right.push(Ref {
                                    tri: r.tri,
                                    bounds: h,
                                });
                            }
                            (true, false) => left.push(r),
                            _ => right.push(r),
                        }
                    }
                }
            } else if best.0.is_finite() {
                let (axis, split) = (best.1, best.2);
                let extent = centroids.max[axis] - centroids.min[axis];
                for r in refs {
                    let b = (((centroid(&r.bounds, axis) - centroids.min[axis]) / extent
                        * self.object_bins as f32) as usize)
                        .min(self.object_bins - 1);
                    if b <= split {
                        left.push(r)
                    } else {
                        right.push(r)
                    }
                }
            } else {
                left = refs;
            }
            if left.is_empty() || right.is_empty() {
                // Degenerate: split the list in half along the longest centroid axis.
                let mut all: Vec<Ref> = left.into_iter().chain(right).collect();
                let e = centroids.max - centroids.min;
                let axis = if e.x > e.y && e.x > e.z {
                    0
                } else if e.y > e.z {
                    1
                } else {
                    2
                };
                let mid = all.len() / 2;
                all.select_nth_unstable_by(mid, |a, b| {
                    centroid(&a.bounds, axis).total_cmp(&centroid(&b.bounds, axis))
                });
                right = all.split_off(mid);
                left = all;
            }
            self.build(left, depth + 1);
            let right = self.build(right, depth + 1);
            self.nodes[node].first = right;
            node as u32
        }
    }

    /// Returns (expanded leaf-ordered triangles, nodes, depth).
    pub fn build(
        triangles: &[Triangle],
        vertices: &[MeshVertex],
        leaf_size: usize,
        traversal_cost: f32,
        stop: bool,
    ) -> (Vec<Triangle>, Vec<BvhNode>, usize) {
        let refs: Vec<Ref> = triangles
            .iter()
            .enumerate()
            .map(|(i, t)| Ref {
                tri: i as u32,
                bounds: super::triangle_bounds(t, vertices),
            })
            .collect();
        let root = refs.iter().fold(Bounds::empty(), |b, r| b.union(r.bounds));
        let mut builder = Builder {
            triangles,
            vertices,
            nodes: Vec::with_capacity(triangles.len()),
            out: Vec::with_capacity(triangles.len() * 13 / 10),
            root_area: root.area(),
            budget: (triangles.len() as f32
                * std::env::var("RT_SBVH_BUDGET").map_or(BUDGET, |v| v.parse().unwrap()))
                as usize,
            depth: 0,
            leaf_size,
            traversal_cost,
            stop,
            object_bins: (knob("RT_SBVH_OBJECT_BINS", 16.) as usize).clamp(2, MAX_BINS),
            spatial_bins: (knob("RT_SBVH_SPATIAL_BINS", 32.) as usize).clamp(2, MAX_BINS),
            overlap: knob("RT_SBVH_ALPHA", 1e-5),
        };
        builder.build(refs, 0);
        (builder.out, builder.nodes, builder.depth)
    }
}
