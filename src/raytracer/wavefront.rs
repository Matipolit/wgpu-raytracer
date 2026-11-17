//! Wavefront path tracing (see wavefront.wgsl): per-path ray/state/hit buffers,
//! ping-pong path lists, per-bounce counters and indirect dispatch arguments.
#![allow(clippy::cast_possible_truncation)]

/// Bounce slots in the counter and argument arrays (bounces + 2 must fit).
pub const MAX_STAGES: u32 = 64;
/// Dynamic uniform stride for the per-dispatch bounce index.
const STAGE_STRIDE: u64 = 256;

pub struct Wavefront {
    layouts: [wgpu::BindGroupLayout; 2],
    stage_buffer: wgpu::Buffer,
    args: wgpu::Buffer,
    group: wgpu::BindGroup,
    args_group: wgpu::BindGroup,
    pixels: u32,
    pub generate: wgpu::ComputePipeline,
    pub args_pipeline: wgpu::ComputePipeline,
    pub trace: wgpu::ComputePipeline,
    pub shade: wgpu::ComputePipeline,
    pub accumulate: wgpu::ComputePipeline,
}

impl Wavefront {
    /// Layouts for group 4: (main kernels, argument kernel). The argument buffer
    /// may not be bound as storage while a dispatch reads it indirectly.
    pub fn layouts(device: &wgpu::Device) -> [wgpu::BindGroupLayout; 2] {
        [
            Self::layout(device, &[0, 1, 2, 3, 4]),
            Self::layout(device, &[4, 5]),
        ]
    }
    fn layout(device: &wgpu::Device, storage_bindings: &[u32]) -> wgpu::BindGroupLayout {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wavefront layout"),
            entries: &storage_bindings
                .iter()
                .map(|&b| storage(b))
                .chain([wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                }])
                .collect::<Vec<_>>(),
        })
    }

    pub fn new(
        device: &wgpu::Device,
        layouts: [wgpu::BindGroupLayout; 2],
        pipelines: [wgpu::ComputePipeline; 5],
        pixels: u32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let stages: Vec<u8> = (0..MAX_STAGES)
            .flat_map(|b| {
                let mut entry = vec![0u8; STAGE_STRIDE as usize];
                entry[..4].copy_from_slice(&b.to_le_bytes());
                entry
            })
            .collect();
        let stage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wavefront stages"),
            contents: &stages,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wavefront dispatch arguments"),
            size: u64::from(MAX_STAGES) * 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let [generate, args_pipeline, trace, shade, accumulate] = pipelines;
        let [group, args_group] = Self::bind(device, &layouts, &stage_buffer, &args, pixels);
        Self {
            layouts,
            stage_buffer,
            args,
            group,
            args_group,
            pixels,
            generate,
            args_pipeline,
            trace,
            shade,
            accumulate,
        }
    }

    fn bind(
        device: &wgpu::Device,
        layouts: &[wgpu::BindGroupLayout; 2],
        stages: &wgpu::Buffer,
        args: &wgpu::Buffer,
        pixels: u32,
    ) -> [wgpu::BindGroup; 2] {
        let n = u64::from(pixels.max(1));
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let rays = buffer("wavefront rays", 32 * n);
        let paths = buffer("wavefront paths", 32 * n);
        let hits = buffer("wavefront hits", 16 * n);
        let queues = buffer("wavefront path lists", 8 * n);
        let counts = buffer("wavefront counts", 4 * u64::from(MAX_STAGES));
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let stage = wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: stages,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        };
        [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wavefront bindings"),
                layout: &layouts[0],
                entries: &[
                    entry(0, &rays),
                    entry(1, &paths),
                    entry(2, &hits),
                    entry(3, &queues),
                    entry(4, &counts),
                    stage.clone(),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("wavefront argument bindings"),
                layout: &layouts[1],
                entries: &[entry(4, &counts), entry(5, args), stage],
            }),
        ]
    }

    pub fn resize(&mut self, device: &wgpu::Device, pixels: u32) {
        if pixels != self.pixels {
            [self.group, self.args_group] = Self::bind(
                device,
                &self.layouts,
                &self.stage_buffer,
                &self.args,
                pixels,
            );
            self.pixels = pixels;
        }
    }

    /// Record one sample per pixel. Groups 0-3 must already be bound.
    pub fn record(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        size: (u32, u32),
        tile: [u32; 2],
        bounces: u32,
    ) {
        assert!(
            bounces + 2 <= MAX_STAGES,
            "too many bounces for the wavefront tracer"
        );
        let stage = |b: u32| [(u64::from(b) * STAGE_STRIDE) as u32];
        let tiles = (size.0.div_ceil(tile[0]), size.1.div_ceil(tile[1]));
        pass.set_bind_group(4, &self.group, &stage(0));
        pass.set_pipeline(&self.generate);
        pass.dispatch_workgroups(tiles.0, tiles.1, 1);
        pass.set_bind_group(4, &self.args_group, &stage(0));
        pass.set_pipeline(&self.args_pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        for b in 0..bounces {
            pass.set_bind_group(4, &self.group, &stage(b));
            pass.set_pipeline(&self.trace);
            pass.dispatch_workgroups_indirect(&self.args, u64::from(b) * 12);
            pass.set_pipeline(&self.shade);
            pass.dispatch_workgroups_indirect(&self.args, u64::from(b) * 12);
            pass.set_bind_group(4, &self.args_group, &stage(b + 1));
            pass.set_pipeline(&self.args_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_bind_group(4, &self.group, &stage(bounces));
        pass.set_pipeline(&self.accumulate);
        pass.dispatch_workgroups(tiles.0, tiles.1, 1);
    }
}

pub fn enabled() -> bool {
    crate::has_flag("--wavefront")
}
