//! Imported glTF images as sampled GPU textures.
//!
//! Every glTF texture becomes one GPU texture per interpretation it is used
//! with (sRGB colour, linear data, tangent-space normals), with a complete mip
//! chain filtered in linear light. When the device supports BC compression,
//! colour and data use BC1, base colour read by an alpha-tested material uses
//! BC3, and normal maps use BC5 (the shader rebuilds z). Hardware samplers do
//! the filtering, wrapping and sRGB decoding. Images whose size is not a
//! multiple of the 4x4 block, or devices without BC support, use RGBA8.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
use super::mesh::Mesh;
use std::num::NonZeroU32;
use wgpu::util::DeviceExt;

/// Wrap S (repeat, clamp, mirror) x wrap T x {linear, nearest}:
/// index = s + 3 t + 9 nearest. Shaders select the nearest twin with `% 9 + 9`.
pub const SAMPLER_COUNT: u32 = 18;
/// Material texture slots (word offsets within a 48-word material record).
pub const SLOTS: [usize; 4] = [16, 24, 32, 40];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Usage {
    /// sRGB colour; alpha unused (BC1).
    Color,
    /// sRGB colour whose alpha drives a MASK cutoff (BC3).
    ColorAlpha,
    /// Linear data such as metallic/roughness (BC1).
    Data,
    /// Tangent-space normal map, x/y only (BC5).
    Normal,
}

/// Features the mesh shader needs: per-lane indexing into texture/sampler arrays.
pub fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
}

/// Features and limits every device that creates a `Raytracer` must request.
pub fn device_requirements(adapter: &wgpu::Adapter) -> (wgpu::Features, wgpu::Limits) {
    let available = adapter.features();
    assert!(
        available.contains(required_features()),
        "GPU must support non-uniformly indexed texture binding arrays (Vulkan descriptor indexing)"
    );
    let limits = adapter.limits();
    let ray_query = if super::mesh::hw_rt() {
        assert!(
            available.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            "--hw-rt needs a GPU with Vulkan ray queries"
        );
        wgpu::Features::EXPERIMENTAL_RAY_QUERY
    } else {
        wgpu::Features::empty()
    };
    (
        required_features() | ray_query | (available & wgpu::Features::TEXTURE_COMPRESSION_BC)
            // Optional: the visibility raster, and frame timing for the auto-tuner.
            | (available & (wgpu::Features::PRIMITIVE_INDEX | wgpu::Features::INDIRECT_FIRST_INSTANCE | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS))
            // RT_SPIRV_TRACE experiments load the trace kernel as raw SPIR-V.
            | if std::env::var("RT_SPIRV_TRACE").is_ok() { wgpu::Features::PASSTHROUGH_SHADERS } else { wgpu::Features::empty() },
        wgpu::Limits {
            max_blas_primitive_count: limits.max_blas_primitive_count,
            max_blas_geometry_count: limits.max_blas_geometry_count,
            max_tlas_instance_count: limits.max_tlas_instance_count,
            max_acceleration_structures_per_shader_stage: limits
                .max_acceleration_structures_per_shader_stage,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            // The wavefront tracer binds a fifth group.
            max_bind_groups: limits.max_bind_groups.min(8),
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_binding_array_elements_per_shader_stage: limits
                .max_binding_array_elements_per_shader_stage,
            max_binding_array_sampler_elements_per_shader_stage: limits
                .max_binding_array_sampler_elements_per_shader_stage,
            ..Default::default()
        },
    )
}

/// Opt-in token for wgpu's experimental features; only `--hw-rt` uses one.
pub fn experimental_features() -> wgpu::ExperimentalFeatures {
    if super::mesh::hw_rt() {
        // SAFETY: ray queries are the only experimental feature requested; the
        // tracer builds and binds acceleration structures as wgpu documents.
        unsafe { wgpu::ExperimentalFeatures::enabled() }
    } else {
        wgpu::ExperimentalFeatures::disabled()
    }
}

/// glTF wrap/filter flags (texture descriptor word 3) -> sampler array index.
pub fn sampler_index(flags: u32) -> u32 {
    (flags & 3) + 3 * ((flags >> 2) & 3) + 9 * ((flags >> 4) & 1)
}

impl Mesh {
    /// GPU textures as (glTF texture, usage), and each material's GPU texture
    /// index per slot in `SLOTS` order (`u32::MAX` when the slot is empty).
    pub fn texture_plan(&self) -> (Vec<(usize, Usage)>, Vec<[u32; 4]>) {
        let mut entries: Vec<(usize, Usage)> = Vec::new();
        let mut slots = Vec::with_capacity(self.materials.len());
        for material in &self.materials {
            let mut indices = [u32::MAX; 4];
            for (k, slot) in SLOTS.into_iter().enumerate() {
                let texture = material[slot];
                if texture == u32::MAX {
                    continue;
                }
                let usage = match slot {
                    16 if material[11] & 2 != 0 => Usage::ColorAlpha,
                    16 | 32 => Usage::Color,
                    24 => Usage::Data,
                    _ => Usage::Normal,
                };
                let srgb = |u: Usage| matches!(u, Usage::Color | Usage::ColorAlpha);
                let existing = entries.iter().position(|&(t, u)| {
                    t == texture as usize && (u == usage || (srgb(u) && srgb(usage)))
                });
                let index = match existing {
                    Some(i) => {
                        if usage == Usage::ColorAlpha {
                            entries[i].1 = Usage::ColorAlpha;
                        }
                        i
                    }
                    None => {
                        entries.push((texture as usize, usage));
                        entries.len() - 1
                    }
                };
                indices[k] = index as u32;
            }
            slots.push(indices);
        }
        (entries, slots)
    }
}

/// Mip chain with wgpu's level sizes (each halving rounds down, minimum 1).
/// Box filtering uses area weights for odd sizes and averages colour in linear light.
pub fn mip_chain(
    base: &[[u8; 4]],
    width: u32,
    height: u32,
    srgb: bool,
) -> Vec<(u32, u32, Vec<[u8; 4]>)> {
    let to_linear: [f32; 256] = std::array::from_fn(|i| {
        let x = i as f32 / 255.0;
        if !srgb {
            x
        } else if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    });
    let mut levels = vec![(width, height, base.to_vec())];
    while let Some((w, h, pixels)) = levels.last().filter(|(w, h, _)| *w > 1 || *h > 1) {
        let (w, h) = (*w, *h);
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            let (y0, y1) = (
                y as f32 * h as f32 / nh as f32,
                (y + 1) as f32 * h as f32 / nh as f32,
            );
            for x in 0..nw {
                let (x0, x1) = (
                    x as f32 * w as f32 / nw as f32,
                    (x + 1) as f32 * w as f32 / nw as f32,
                );
                let area = (x1 - x0) * (y1 - y0);
                let mut sum = [0f32; 4];
                for sy in y0.floor() as u32..(y1.ceil() as u32).min(h) {
                    let wy = ((sy + 1) as f32).min(y1) - (sy as f32).max(y0);
                    for sx in x0.floor() as u32..(x1.ceil() as u32).min(w) {
                        let weight = wy * (((sx + 1) as f32).min(x1) - (sx as f32).max(x0)) / area;
                        let p = pixels[(sy * w + sx) as usize];
                        for c in 0..3 {
                            sum[c] += weight * to_linear[p[c] as usize];
                        }
                        sum[3] += weight * f32::from(p[3]) / 255.0;
                    }
                }
                let mut pixel = [0u8; 4];
                for c in 0..4 {
                    let mut v = sum[c];
                    if srgb && c < 3 {
                        v = if v <= 0.003_130_8 {
                            12.92 * v
                        } else {
                            1.055 * v.powf(1.0 / 2.4) - 0.055
                        };
                    }
                    pixel[c] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                next.push(pixel);
            }
        }
        levels.push((nw, nh, next));
    }
    levels
}

/// One GPU texture's format and tightly packed mip data (layer-major, mip-major).
fn encode(
    mesh: &Mesh,
    texture: usize,
    usage: Usage,
    compress: bool,
) -> (wgpu::TextureFormat, u32, u32, u32, Vec<u8>) {
    let [offset, width, height, _] = mesh.textures[texture];
    let base: Vec<[u8; 4]> = mesh.pixels[offset as usize..(offset + width * height) as usize]
        .iter()
        .map(|p| p.to_le_bytes())
        .collect();
    let srgb = matches!(usage, Usage::Color | Usage::ColorAlpha);
    let alpha = usage == Usage::ColorAlpha && base.iter().any(|p| p[3] != 255);
    let block = compress && width % 4 == 0 && height % 4 == 0;
    use wgpu::TextureFormat as F;
    let (format, codec) = match (block, usage, alpha) {
        (false, Usage::Color | Usage::ColorAlpha, _) => (F::Rgba8UnormSrgb, None),
        (false, _, _) => (F::Rgba8Unorm, None),
        (true, Usage::Normal, _) => (F::Bc5RgUnorm, Some(texpresso::Format::Bc5)),
        (true, Usage::Data, _) => (F::Bc1RgbaUnorm, Some(texpresso::Format::Bc1)),
        (true, _, true) => (F::Bc3RgbaUnormSrgb, Some(texpresso::Format::Bc3)),
        (true, _, false) => (F::Bc1RgbaUnormSrgb, Some(texpresso::Format::Bc1)),
    };
    let levels = mip_chain(&base, width, height, srgb);
    let mut data = Vec::new();
    for (w, h, pixels) in &levels {
        let bytes: &[u8] = bytemuck::cast_slice(pixels);
        match codec {
            None => data.extend_from_slice(bytes),
            Some(codec) => {
                let start = data.len();
                data.resize(start + codec.compressed_size(*w as usize, *h as usize), 0);
                let params = texpresso::Params {
                    algorithm: texpresso::Algorithm::RangeFit,
                    ..Default::default()
                };
                codec.compress(bytes, *w as usize, *h as usize, params, &mut data[start..]);
            }
        }
    }
    (format, width, height, levels.len() as u32, data)
}

/// Sampled textures and samplers for the scene bind group.
pub struct GpuTextures {
    pub views: Vec<wgpu::TextureView>,
    pub samplers: Vec<wgpu::Sampler>,
}

impl GpuTextures {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh) -> Self {
        let start = std::time::Instant::now();
        let (entries, _) = mesh.texture_plan();
        let compress = device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        // Mip filtering is per texture; BC encoding parallelizes inside each level.
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut encoded: Vec<_> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&(texture, usage)) = entries.get(i) else {
                                break;
                            };
                            done.push((i, encode(mesh, texture, usage, compress)));
                        }
                        done
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        encoded.sort_by_key(|(i, _)| *i);
        let mut bytes = 0;
        let mut views: Vec<_> = encoded
            .into_iter()
            .map(|(i, (format, width, height, mips, data))| {
                bytes += data.len() as u64;
                device
                    .create_texture_with_data(
                        queue,
                        &wgpu::TextureDescriptor {
                            label: Some(&format!("glTF texture {}", entries[i].0)),
                            size: wgpu::Extent3d {
                                width,
                                height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: mips,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING,
                            view_formats: &[],
                        },
                        wgpu::util::TextureDataOrder::LayerMajor,
                        &data,
                    )
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();
        // Index 0: a white placeholder that empty material slots name, so no
        // shader path can fetch an out-of-range descriptor (even speculatively,
        // which hangs Mesa's Intel driver).
        views.insert(
            0,
            device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("placeholder texture"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &[255; 4],
                )
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        if !entries.is_empty() {
            eprintln!(
                "Textures: {} GPU textures, {:.1} MiB with mips ({}), {:.2}s",
                entries.len(),
                bytes as f64 / 1_048_576.,
                if compress { "BC" } else { "RGBA8" },
                start.elapsed().as_secs_f64()
            );
        }
        let wraps = [
            wgpu::AddressMode::Repeat,
            wgpu::AddressMode::ClampToEdge,
            wgpu::AddressMode::MirrorRepeat,
        ];
        let samplers = (0..SAMPLER_COUNT)
            .map(|i| {
                let filter = if i >= 9 {
                    wgpu::FilterMode::Nearest
                } else {
                    wgpu::FilterMode::Linear
                };
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("glTF sampler"),
                    address_mode_u: wraps[(i % 3) as usize],
                    address_mode_v: wraps[(i / 3 % 3) as usize],
                    mag_filter: filter,
                    min_filter: filter,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                })
            })
            .collect();
        Self { views, samplers }
    }

    pub fn layout_entries(
        &self,
        visibility: wgpu::ShaderStages,
    ) -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(self.views.len() as u32),
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: NonZeroU32::new(SAMPLER_COUNT),
            },
        ]
    }

    pub fn views(&self) -> Vec<&wgpu::TextureView> {
        self.views.iter().collect()
    }
    pub fn samplers(&self) -> Vec<&wgpu::Sampler> {
        self.samplers.iter().collect()
    }
}
