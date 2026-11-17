use super::gpu_buffer::{StorageBuffer, UniformBuffer};

/// Separate bindings enforce write -> read transitions between the three passes.
pub(super) struct ImageBuffers {
    pub accumulation: StorageBuffer,
    pub filtered: StorageBuffer,
    /// Automatic exposure state; survives resizes so the image does not flash.
    pub exposure: StorageBuffer,
    /// Primary-hit guides, 48 bytes per pixel (see raytracer.wgsl `gbuffer`).
    pub gbuffer: wgpu::Buffer,
    pub trace_layout: wgpu::BindGroupLayout,
    pub filter_layout: wgpu::BindGroupLayout,
    pub display_layout: wgpu::BindGroupLayout,
    pub trace: wgpu::BindGroup,
    pub filter: wgpu::BindGroup,
    pub display: wgpu::BindGroup,
}

impl ImageBuffers {
    pub fn new(
        device: &wgpu::Device,
        frame: &UniformBuffer,
        bytes: u64,
        auto_exposure: bool,
    ) -> Self {
        let trace_stages = super::tracing_stages();
        let storage = |binding, visibility, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace image layout"),
            entries: &[
                frame.layout(trace_stages),
                storage(1, trace_stages, false),
                storage(7, trace_stages, false),
            ],
        });
        let filter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("filter image layout"),
            entries: &[
                frame.layout(trace_stages),
                storage(2, trace_stages, true),
                storage(3, wgpu::ShaderStages::COMPUTE, false),
                storage(5, wgpu::ShaderStages::COMPUTE, false),
            ],
        });
        let display_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display image layout"),
            entries: &[
                frame.layout(wgpu::ShaderStages::FRAGMENT),
                storage(4, wgpu::ShaderStages::FRAGMENT, true),
                storage(6, wgpu::ShaderStages::FRAGMENT, true),
            ],
        });
        let exposure = StorageBuffer::new_from_bytes(
            device,
            bytemuck::cast_slice(&[0f32, f32::from(u8::from(auto_exposure)), 0., 0.]),
            5,
            Some("exposure state"),
        );
        let (accumulation, filtered, gbuffer, [trace, filter, display]) = Self::allocate(
            device,
            frame,
            &exposure,
            bytes,
            &trace_layout,
            &filter_layout,
            &display_layout,
        );
        Self {
            accumulation,
            filtered,
            exposure,
            gbuffer,
            trace_layout,
            filter_layout,
            display_layout,
            trace,
            filter,
            display,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, frame: &UniformBuffer, bytes: u64) {
        if self.accumulation.handle().size() == bytes {
            return;
        }
        let (accumulation, filtered, gbuffer, [trace, filter, display]) = Self::allocate(
            device,
            frame,
            &self.exposure,
            bytes,
            &self.trace_layout,
            &self.filter_layout,
            &self.display_layout,
        );
        self.accumulation = accumulation;
        self.filtered = filtered;
        self.gbuffer = gbuffer;
        self.trace = trace;
        self.filter = filter;
        self.display = display;
    }

    fn allocate(
        device: &wgpu::Device,
        frame: &UniformBuffer,
        exposure: &StorageBuffer,
        bytes: u64,
        trace_layout: &wgpu::BindGroupLayout,
        filter_layout: &wgpu::BindGroupLayout,
        display_layout: &wgpu::BindGroupLayout,
    ) -> (
        StorageBuffer,
        StorageBuffer,
        wgpu::Buffer,
        [wgpu::BindGroup; 3],
    ) {
        let accumulation = StorageBuffer::new(device, bytes, 1, Some("accumulated radiance"));
        let filtered = StorageBuffer::new(device, bytes, 3, Some("filtered radiance"));
        // `bytes` holds 12 per pixel; the guides need 48.
        let gbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("primary-hit guides"),
            size: bytes * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        fn binding(buffer: &wgpu::Buffer, binding: u32) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let trace = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace image bindings"),
            layout: trace_layout,
            entries: &[
                frame.binding(),
                accumulation.binding(),
                binding(&gbuffer, 7),
            ],
        });
        let filter = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filter image bindings"),
            layout: filter_layout,
            entries: &[
                frame.binding(),
                binding(accumulation.handle(), 2),
                filtered.binding(),
                exposure.binding(),
            ],
        });
        let display = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display image bindings"),
            layout: display_layout,
            entries: &[
                frame.binding(),
                binding(filtered.handle(), 4),
                binding(exposure.handle(), 6),
            ],
        });
        (accumulation, filtered, gbuffer, [trace, filter, display])
    }
}
