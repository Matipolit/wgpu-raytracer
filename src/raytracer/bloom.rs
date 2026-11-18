//! Bloom pyramid (bloom.wgsl) and the display settings it shares with the display
//! pass: tonemapper and bloom strength. The pyramid is built from the HDR image the
//! display pass reads; the display blends its half-resolution sum before tonemapping.
use wgpu::util::DeviceExt;

/// Pyramid levels below the image (level 1 is half resolution).
pub const LEVELS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tonemapper {
    /// Troy Sobotka's AgX (Blender's default view transform since 4.0).
    #[default]
    Agx,
    /// ACES filmic (Stephen Hill's RRT + ODT fit).
    Aces,
    /// Khronos PBR Neutral (keeps base colours true, for product shots).
    Neutral,
    /// John Hable's Uncharted 2 curve (the previous default).
    Hable,
}

impl Tonemapper {
    pub const ALL: [Self; 4] = [Self::Agx, Self::Aces, Self::Neutral, Self::Hable];

    pub fn label(self) -> &'static str {
        match self {
            Self::Agx => "AgX",
            Self::Aces => "ACES",
            Self::Neutral => "Khronos PBR Neutral",
            Self::Hable => "Hable (Uncharted 2)",
        }
    }

    /// `--tonemapper agx|aces|neutral|hable`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "agx" => Some(Self::Agx),
            "aces" => Some(Self::Aces),
            "neutral" => Some(Self::Neutral),
            "hable" => Some(Self::Hable),
            _ => None,
        }
    }
}

/// Default bloom strength: the share of the blurred image blended in.
pub const DEFAULT_STRENGTH: f32 = 0.04;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PassUniform {
    dst_width: u32,
    dst_height: u32,
    mode: u32,
    src_width: u32,
    src_height: u32,
    pad: [u32; 3],
}

/// The display pass's settings (raytracer.wgsl `displaySettings`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayUniform {
    tonemapper: u32,
    strength: f32,
    /// Pyramid levels summed (the display divides by it).
    levels: u32,
    pad: u32,
}

/// Level sizes, level 0 being the image: ceil halving, as the textures' mip chain
/// (whose base is level 1) would round down: level sizes come from the base instead.
fn pyramid(size: (u32, u32)) -> Vec<(u32, u32)> {
    // At most LEVELS levels below the image, fewer for tiny images (as many mips as
    // the half-resolution base allows, at least two: the base is at least 2x2).
    let base = (size.0.div_ceil(2).max(2), size.1.div_ceil(2).max(2));
    let mips = (32 - base.0.max(base.1).leading_zeros()).min(LEVELS as u32);
    let mut sizes = vec![size];
    for m in 0..mips.max(2) {
        sizes.push(((base.0 >> m).max(1), (base.1 >> m).max(1)));
    }
    sizes
}

pub struct Bloom {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    display_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    display_uniform: wgpu::Buffer,
    pub display_group: wgpu::BindGroup,
    passes: Vec<(wgpu::BindGroup, (u32, u32))>,
    size: (u32, u32),
    tonemapper: Tonemapper,
    strength: f32,
}

impl Bloom {
    /// Display-pass layout (group 2: bloom texture 10, settings 11, sampler 12).
    pub fn display_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom display layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub fn new(
        device: &wgpu::Device,
        display_layout: wgpu::BindGroupLayout,
        source: &wgpu::Buffer,
        size: (u32, u32),
        tonemapper: Tonemapper,
        strength: f32,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bloom.wgsl").into()),
        });
        let compute = wgpu::ShaderStages::COMPUTE;
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture(2),
                texture(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bloom"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("csBloom"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let display_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display settings"),
            size: std::mem::size_of::<DisplayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (display_group, passes) = Self::build(
            device,
            &layout,
            &display_layout,
            &sampler,
            &display_uniform,
            source,
            size,
        );
        Self {
            pipeline,
            layout,
            display_layout,
            sampler,
            display_uniform,
            display_group,
            passes,
            size,
            tonemapper,
            strength,
        }
    }

    #[allow(clippy::type_complexity)]
    fn build(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        display_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        display_uniform: &wgpu::Buffer,
        source: &wgpu::Buffer,
        size: (u32, u32),
    ) -> (wgpu::BindGroup, Vec<(wgpu::BindGroup, (u32, u32))>) {
        let sizes = pyramid(size);
        let texture = |label, mips: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: sizes[1].0,
                    height: sizes[1].1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };
        // down mip m = level m + 1; up mip m = level m + 1 (levels 1..levels-1).
        let levels = sizes.len() - 1;
        let down = texture("bloom down", levels);
        let up = texture("bloom up", levels - 1);
        let dummy = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("bloom unused input"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let mip = |t: &wgpu::Texture, m: usize| {
            t.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: m as u32,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        // (mode, source size, source view, add view, target view, target size)
        let mut specs = Vec::new();
        specs.push((0, sizes[0], None, None, mip(&down, 0), sizes[1]));
        for l in 2..=levels {
            specs.push((
                1,
                sizes[l - 1],
                Some(mip(&down, l - 2)),
                None,
                mip(&down, l - 1),
                sizes[l],
            ));
        }
        for l in (1..levels).rev() {
            let src = if l + 1 == levels {
                mip(&down, levels - 1)
            } else {
                mip(&up, l)
            };
            specs.push((
                2,
                sizes[l + 1],
                Some(src),
                Some(mip(&down, l - 1)),
                mip(&up, l - 1),
                sizes[l],
            ));
        }
        let passes = specs
            .into_iter()
            .map(|(mode, src, source_view, add_view, target, dst)| {
                let uniform = PassUniform {
                    dst_width: dst.0,
                    dst_height: dst.1,
                    mode,
                    src_width: src.0,
                    src_height: src.1,
                    pad: [0; 3],
                };
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bloom pass"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bloom pass bindings"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: source.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                source_view.as_ref().unwrap_or(&dummy),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(
                                add_view.as_ref().unwrap_or(&dummy),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&target),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                });
                (group, dst)
            })
            .collect();
        let display_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom display bindings"),
            layout: display_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&mip(&up, 0)),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: display_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        (display_group, passes)
    }

    /// New source image or size: rebuild the pyramid and its bindings.
    pub fn resize(&mut self, device: &wgpu::Device, source: &wgpu::Buffer, size: (u32, u32)) {
        (self.display_group, self.passes) = Self::build(
            device,
            &self.layout,
            &self.display_layout,
            &self.sampler,
            &self.display_uniform,
            source,
            size,
        );
        self.size = size;
    }

    pub fn set(&mut self, tonemapper: Tonemapper, strength: f32) {
        self.tonemapper = tonemapper;
        self.strength = strength;
    }

    /// Upload the display settings (cheap; every frame).
    pub fn upload(&self, queue: &wgpu::Queue) {
        let uniform = DisplayUniform {
            tonemapper: Tonemapper::ALL
                .iter()
                .position(|t| *t == self.tonemapper)
                .unwrap_or(0) as u32,
            strength: self.strength,
            levels: pyramid(self.size).len() as u32 - 1,
            pad: 0,
        };
        queue.write_buffer(&self.display_uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Build the pyramid from the current HDR image (skipped when bloom is off).
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.strength <= 0. {
            return;
        }
        for (group, (w, h)) in &self.passes {
            // One pass per level: each level reads the previous one's writes.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bloom"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pyramid_levels_follow_the_half_resolution_mip_chain() {
        let sizes = super::pyramid((1280, 720));
        assert_eq!(sizes[1], (640, 360));
        assert_eq!(sizes[2], (320, 180));
        assert_eq!(sizes[6], (20, 11));
        let small = super::pyramid((16, 16));
        assert_eq!(small.len(), 5);
        assert_eq!(small[4], (1, 1));
        assert_eq!(super::pyramid((1, 1)).len(), 3);
    }
}
