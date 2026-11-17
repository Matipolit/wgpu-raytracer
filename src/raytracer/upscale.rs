//! Temporal upscaling (`--render-scale S`, 0 < S < 1): the renderer traces, denoises and
//! keeps its G-buffer at render resolution; this stage turns each frame into an
//! output-resolution image (see upscale.wgsl). It is the non-learned baseline that the
//! joint denoise + upscale network has to beat, and the output-resolution plumbing that
//! network will plug into.
#![allow(clippy::cast_possible_truncation)]
use super::gpu_buffer::UniformBuffer;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    render_size: [u32; 2],
    output_size: [u32; 2],
    jitter: [f32; 2],
    history_valid: u32,
    clamp_history: u32,
    alpha_min: f32,
    alpha_max: f32,
    pad: [u32; 2],
}

/// The render scale from `--render-scale` (1 = no upscaling).
pub fn render_scale() -> f32 {
    crate::argument_value("--render-scale").map_or(1., |v| {
        let s: f32 = v.parse().expect("--render-scale must be a number");
        assert!(s > 0. && s <= 1., "--render-scale must be in (0, 1]");
        s
    })
}

/// Render resolution for an output resolution.
pub fn render_size(output: (u32, u32)) -> (u32, u32) {
    let s = render_scale();
    (
        ((output.0 as f32 * s).round() as u32).max(1),
        ((output.1 as f32 * s).round() as u32).max(1),
    )
}

pub struct Upscaler {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    /// Ping-pong history (rgb, depth) and the two bind groups reading one, writing the other.
    groups: [wgpu::BindGroup; 2],
    output: wgpu::Buffer,
    /// Display bindings for the output image (output-sized frame data).
    pub display: wgpu::BindGroup,
    display_frame: UniformBuffer,
    output_size: (u32, u32),
    render_size: (u32, u32),
    parity: usize,
    history_valid: bool,
    /// Denoised input: clamp history to the neighbourhood and blend faster.
    denoised: bool,
}

impl Upscaler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        image: &wgpu::Buffer,
        gbuffer: &wgpu::Buffer,
        display_layout: &wgpu::BindGroupLayout,
        exposure: &wgpu::Buffer,
        output_size: (u32, u32),
        render_size: (u32, u32),
        denoised: bool,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upscaler"),
            source: wgpu::ShaderSource::Wgsl(include_str!("upscale.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscaler layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upscaler"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("upscaler"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("csUpscale"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upscaler parameters"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let display_frame = UniformBuffer::new(device, 16, 0, Some("upscaled frame data"));
        let (groups, output, display) = Self::bind(
            device,
            &layout,
            &params,
            image,
            gbuffer,
            display_layout,
            &display_frame,
            exposure,
            output_size,
        );
        Self {
            pipeline,
            layout,
            params,
            groups,
            output,
            display,
            display_frame,
            output_size,
            render_size,
            parity: 0,
            history_valid: false,
            denoised,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        image: &wgpu::Buffer,
        gbuffer: &wgpu::Buffer,
        display_layout: &wgpu::BindGroupLayout,
        display_frame: &UniformBuffer,
        exposure: &wgpu::Buffer,
        (width, height): (u32, u32),
    ) -> ([wgpu::BindGroup; 2], wgpu::Buffer, wgpu::BindGroup) {
        let pixels = u64::from(width) * u64::from(height);
        let history: [wgpu::Buffer; 2] = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upscaler history"),
                size: 16 * pixels.max(1),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscaled image"),
            contents: &vec![0u8; 12 * pixels.max(1) as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let groups = std::array::from_fn(|parity| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscaler bindings"),
                layout,
                entries: &[
                    entry(0, params),
                    entry(1, image),
                    entry(2, gbuffer),
                    entry(3, &history[parity]),
                    entry(4, &history[1 - parity]),
                    entry(5, &output),
                ],
            })
        });
        let display = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscaled display bindings"),
            layout: display_layout,
            entries: &[
                display_frame.binding(),
                entry(4, &output),
                entry(6, exposure),
            ],
        });
        (groups, output, display)
    }

    /// Rebind after the render or output size (or the input buffers) changed; history restarts.
    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        image: &wgpu::Buffer,
        gbuffer: &wgpu::Buffer,
        display_layout: &wgpu::BindGroupLayout,
        exposure: &wgpu::Buffer,
        output_size: (u32, u32),
        render_size: (u32, u32),
    ) {
        (self.groups, self.output, self.display) = Self::bind(
            device,
            &self.layout,
            &self.params,
            image,
            gbuffer,
            display_layout,
            &self.display_frame,
            exposure,
            output_size,
        );
        self.output_size = output_size;
        self.render_size = render_size;
        self.history_valid = false;
    }

    /// Upscale this frame; `jitter` is the frame's shared sub-pixel jitter, `exposure_bits`
    /// the display exposure (frame data .w).
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        jitter: [f32; 2],
        frame: u32,
        exposure_bits: u32,
    ) {
        let (alpha_min, alpha_max) = if self.denoised {
            (0.1, 0.4)
        } else {
            (0.04, 0.15)
        };
        let params = Params {
            render_size: [self.render_size.0, self.render_size.1],
            output_size: [self.output_size.0, self.output_size.1],
            jitter,
            history_valid: u32::from(self.history_valid),
            clamp_history: u32::from(self.denoised),
            alpha_min,
            alpha_max,
            pad: [0; 2],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(
            self.display_frame.handle(),
            0,
            bytemuck::cast_slice(&[self.output_size.0, self.output_size.1, frame, exposure_bits]),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("upscale pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[self.parity], &[]);
        pass.dispatch_workgroups(
            self.output_size.0.div_ceil(8),
            self.output_size.1.div_ceil(8),
            1,
        );
        self.parity = 1 - self.parity;
        self.history_valid = true;
    }

    /// The upscaled image, 3 floats per pixel at output resolution (for evaluation).
    #[allow(dead_code)]
    pub fn output(&self) -> &wgpu::Buffer {
        &self.output
    }
}
