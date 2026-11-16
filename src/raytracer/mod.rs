use gpu_buffer::{StorageBuffer, UniformBuffer};
use wgpu::util::DeviceExt;
pub use {angle::Angle, texture::Texture};

use thiserror::Error;

mod angle;
mod gpu_buffer;
mod image_buffers;
pub mod mesh;
mod texture;

#[cfg(test)]
mod tests;

use std::f32::consts::*;

pub(crate) const TRACE_WORKGROUP_SIZE: [u32; 2] = [8, 8];

/// Explicit tuning override for reproducible GPU experiments; cached per process.
pub(crate) fn trace_workgroup_size() -> [u32; 2] {
    static SIZE: std::sync::OnceLock<[u32; 2]> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("RAYTRACER_WORKGROUP").map_or(TRACE_WORKGROUP_SIZE, |value| {
            let (x, y) = value
                .split_once('x')
                .expect("RAYTRACER_WORKGROUP must be WIDTHxHEIGHT");
            let size = [x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap()];
            assert!(
                size[0] > 0
                    && size[1] > 0
                    && size[0] <= 256
                    && size[1] <= 256
                    && size[0] * size[1] <= 256
            );
            size
        })
    })
}

pub struct Raytracer {
    vertex_uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    frame_data_buffer: UniformBuffer,
    images: image_buffers::ImageBuffers,
    camera_buffer: UniformBuffer,
    sampling_parameter_buffer: UniformBuffer,
    hw_sky_state_buffer: StorageBuffer,
    parameter_bind_group: wgpu::BindGroup,
    scene_bind_group: wgpu::BindGroup,
    accumulation_pipeline: wgpu::ComputePipeline,
    denoising_pipeline: wgpu::ComputePipeline,
    filter_dirty: bool,
    display_pipeline: wgpu::RenderPipeline,
    latest_render_params: RenderParams,
    render_progress: RenderProgress,
    frame_number: u32,
}

impl Raytracer {
    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        scene: &Scene,
        render_params: &RenderParams,
    ) -> Result<Self, RenderParamsValidationError> {
        match render_params.validate() {
            Ok(_) => {}
            Err(err) => return Err(err),
        }

        let uniforms = VertexUniforms {
            view_projection_matrix: unit_quad_projection_matrix(),
            model_matrix: glm::identity(),
        };
        let vertex_uniform_buffer = UniformBuffer::new_from_bytes(
            device,
            bytemuck::bytes_of(&uniforms),
            0_u32,
            Some("uniforms"),
        );
        let vertex_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[vertex_uniform_buffer.layout(wgpu::ShaderStages::VERTEX)],
                label: Some("uniforms layout"),
            });
        let vertex_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &vertex_uniform_bind_group_layout,
            entries: &[vertex_uniform_buffer.binding()],
            label: Some("uniforms bind group"),
        });

        let frame_data_buffer =
            UniformBuffer::new(device, 16_u64, 0_u32, Some("frame data buffer"));

        let images = image_buffers::ImageBuffers::new(
            device,
            &frame_data_buffer,
            image_buffer_size(device, render_params.viewport_size)?,
        );
        let camera_buffer = {
            let camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);

            UniformBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&camera),
                0_u32,
                Some("camera buffer"),
            )
        };

        let sampling_parameter_buffer = UniformBuffer::new(
            device,
            std::mem::size_of::<GpuSamplingParams>() as wgpu::BufferAddress, // Updated size
            1_u32,
            Some("sampling parameter buffer"),
        );

        let hw_sky_state_buffer = {
            let sky_state = render_params.sky.to_sky_state()?;

            StorageBuffer::new_from_bytes(
                device,
                bytemuck::bytes_of(&sky_state),
                2_u32,
                Some("sky state buffer"),
            )
        };

        let parameter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    camera_buffer.layout(tracing_stages()),
                    sampling_parameter_buffer.layout(tracing_stages()),
                    hw_sky_state_buffer.layout(tracing_stages(), true),
                ],
                label: Some("parameter layout"),
            });

        let parameter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &parameter_bind_group_layout,
            entries: &[
                camera_buffer.binding(),
                sampling_parameter_buffer.binding(),
                hw_sky_state_buffer.binding(),
            ],
            label: Some("parameter bind group"),
        });

        let (scene_bind_group_layout, scene_bind_group, light_count) = {
            let sphere_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(scene.spheres.as_slice()),
                0_u32,
                Some("scene buffer"),
            );

            let mut global_texture_data: Vec<[f32; 3]> = Vec::new();
            let mut material_data: Vec<GpuMaterial> = Vec::with_capacity(scene.materials.len());

            for material in scene.materials.iter() {
                let gpu_material = match material {
                    Material::Lambertian { albedo } => {
                        GpuMaterial::lambertian(albedo, &mut global_texture_data)
                    }
                    Material::Metal { albedo, fuzz } => {
                        GpuMaterial::metal(albedo, *fuzz, &mut global_texture_data)
                    }
                    Material::Dielectric { refraction_index } => {
                        GpuMaterial::dielectric(*refraction_index)
                    }
                    Material::Checkerboard { odd, even } => {
                        GpuMaterial::checkerboard(odd, even, &mut global_texture_data)
                    }
                    Material::Emissive { emit } => {
                        GpuMaterial::emissive(emit, &mut global_texture_data)
                    }
                };

                material_data.push(gpu_material);
            }

            let material_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(material_data.as_slice()),
                1_u32,
                Some("materials buffer"),
            );

            let texture_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(global_texture_data.as_slice()),
                2_u32,
                Some("textures buffer"),
            );

            let light_indices: Vec<u32> = scene
                .spheres
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    matches!(
                        scene.materials[s.material_idx as usize],
                        Material::Emissive { .. }
                    )
                })
                .map(|(idx, _)| idx as u32)
                .collect();

            let light_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(light_indices.as_slice()),
                3_u32,
                Some("lights buffer"),
            );

            let mesh_words = scene
                .mesh
                .gpu_words(scene.spheres.len() as u32, light_indices.len() as u32);
            for size in [mesh_words.len() * 4, scene.mesh.pixels.len() * 4] {
                if size as u64 > u64::from(device.limits().max_storage_buffer_binding_size) {
                    return Err(RenderParamsValidationError::SceneBufferTooLarge(size));
                }
            }
            let mesh_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(&mesh_words),
                4,
                Some("mesh geometry and BVH"),
            );
            let mesh_texture_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(&scene.mesh.pixels),
                5,
                Some("packed glTF texture pixels"),
            );

            let scene_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        sphere_buffer.layout(tracing_stages(), true),
                        material_buffer.layout(tracing_stages(), true),
                        texture_buffer.layout(tracing_stages(), true),
                        light_buffer.layout(tracing_stages(), true),
                        mesh_buffer.layout(tracing_stages(), true),
                        mesh_texture_buffer.layout(tracing_stages(), true),
                    ],
                    label: Some("scene layout"),
                });
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &[
                    sphere_buffer.binding(),
                    material_buffer.binding(),
                    texture_buffer.binding(),
                    light_buffer.binding(),
                    mesh_buffer.binding(),
                    mesh_texture_buffer.binding(),
                ],
                label: Some("scene bind group"),
            });

            (
                scene_bind_group_layout,
                scene_bind_group,
                light_indices.len() as f64,
            )
        };

        let neural = crate::argument_value("--neural-model")
            .map(|path| std::fs::read_to_string(path).expect("cannot read neural WGSL model"));
        let mut trace_source = include_str!("raytracer.wgsl").to_owned();
        if crate::has_flag("--fp16-state") {
            let start = trace_source.find("fn rayColor(").unwrap();
            let end = start + trace_source[start..].find("\nfn intersection(").unwrap();
            let mut body = trace_source[start..end].to_owned();
            body = body
                .replace("var color = vec3(0f);", "var color = vec3<f16>(0.);")
                .replace(
                    "var throughput = vec3(1f);",
                    "var throughput = vec3<f16>(1.);",
                )
                .replace(
                    "throughput = rouletteThroughput(throughput, rngState);",
                    "throughput = vec3<f16>(rouletteThroughput(vec3<f32>(throughput), rngState));",
                )
                .replace(
                    "throughput *= scatter.throughput;",
                    "throughput *= vec3<f16>(scatter.throughput);",
                )
                .replace("return color;", "return vec3<f32>(color);");
            let mut result = String::new();
            for line in body.lines() {
                if let Some((prefix, expression)) = line.split_once("color += throughput * ") {
                    result.push_str(&format!(
                        "{prefix}color += vec3<f16>(vec3<f32>(throughput) * {});\n",
                        expression.trim_end_matches(';')
                    ));
                } else {
                    result.push_str(line);
                    result.push('\n');
                }
            }
            trace_source.replace_range(start..end, &result);
            trace_source = format!("enable f16;\n{trace_source}");
        }
        let source = format!(
            "{}\n{}\n{}",
            trace_source,
            include_str!("mesh.wgsl"),
            neural.as_deref().unwrap_or("")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raytracer shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                Some(&vertex_uniform_bind_group_layout),
                Some(&images.trace_layout),
                Some(&parameter_bind_group_layout),
                Some(&scene_bind_group_layout),
            ],
            immediate_size: 0,
            label: Some("raytracer layout"),
        });
        let filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    Some(&vertex_uniform_bind_group_layout),
                    Some(&images.filter_layout),
                    Some(&parameter_bind_group_layout),
                ],
                immediate_size: 0,
                label: Some("filter pipeline layout"),
            });
        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    Some(&vertex_uniform_bind_group_layout),
                    Some(&images.display_layout),
                ],
                immediate_size: 0,
                label: Some("display pipeline layout"),
            });
        let create_pipeline =
            |layout: &wgpu::PipelineLayout, entry_point: &str, write_mask, label| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vsMain"),
                        compilation_options: Default::default(),
                        buffers: &[Some(Vertex::desc())],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface_config.format,
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent::REPLACE,
                                alpha: wgpu::BlendComponent::REPLACE,
                            }),
                            write_mask,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        cull_mode: Some(wgpu::Face::Back),
                        // Requires Features::DEPTH_CLAMPING
                        conservative: false,
                        unclipped_depth: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    label: Some(label),
                    // If the pipeline will be used with a multiview render pass, this
                    // indicates how many array layers the attachments will have.
                    multiview_mask: None,
                    cache: None,
                })
            };
        let create_compute =
            |layout: &wgpu::PipelineLayout, entry_point, label, constants: &[(&str, f64)]| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    cache: None,
                })
            };
        let accumulation_pipeline = create_compute(
            &pipeline_layout,
            "csAccumulate",
            "path tracing compute",
            &[
                ("SPHERE_COUNT", scene.spheres.len() as f64),
                ("LIGHT_COUNT", light_count),
                ("TRACE_MESHES", f64::from(!scene.mesh.triangles.is_empty())),
                ("TRACE_SPHERES", f64::from(!scene.spheres.is_empty())),
                (
                    "RUSSIAN_ROULETTE",
                    f64::from(!std::env::args().any(|a| a == "--no-russian-roulette")),
                ),
                (
                    "SKY_IMPORTANCE",
                    f64::from(std::env::args().any(|a| a == "--sky-importance")),
                ),
                (
                    "GGX_VNDF",
                    f64::from(std::env::args().any(|a| a == "--ggx-vndf")),
                ),
                (
                    "GGX_BOUNDED",
                    f64::from(
                        crate::has_flag("--bounded-vndf")
                            || std::env::var("RT_BOUNDED_GGX").is_ok(),
                    ),
                ),
                (
                    "BLUE_NOISE_JITTER",
                    f64::from(crate::has_flag("--blue-noise-jitter")),
                ),
                ("QUANTIZED_BVH", f64::from(mesh::quantized_bvh())),
                (
                    "OPACITY_APPROX",
                    f64::from(crate::has_flag("--approximate-alpha")),
                ),
                ("RAY_CONES", f64::from(crate::has_flag("--ray-cones"))),
                ("ALPHA_CAP", f64::from(crate::has_flag("--alpha-cap4"))),
                ("SOFTWARE_OPACITY", f64::from(mesh::micro_opacity())),
                ("WIDE_BVH", f64::from(mesh::bvh_width())),
                (
                    "PACKED_SHADING",
                    f64::from(crate::has_flag("--packed-shading")),
                ),
                (
                    "STOCHASTIC_TEXTURE",
                    if crate::has_flag("--stochastic-all") {
                        2.0
                    } else if crate::has_flag("--stochastic-color") {
                        1.0
                    } else {
                        0.0
                    },
                ),
                (
                    "PACKED_DIRECTIONS",
                    f64::from(crate::has_flag("--packed-directions")),
                ),
                (
                    "SKIP_FINAL_SCATTER",
                    f64::from(crate::has_flag("--skip-final-scatter")),
                ),
                ("TRACE_WIDTH", f64::from(trace_workgroup_size()[0])),
                ("TRACE_HEIGHT", f64::from(trace_workgroup_size()[1])),
            ],
        );
        let denoising_pipeline = create_compute(
            &filter_pipeline_layout,
            if neural.is_some() {
                "csNeural"
            } else {
                "csDenoise"
            },
            "denoising compute",
            &[(
                "SPARSE_DENOISE",
                f64::from(crate::has_flag("--sparse-denoise")),
            )],
        );
        let display_pipeline = create_pipeline(
            &display_pipeline_layout,
            "fsMain",
            wgpu::ColorWrites::ALL,
            "display pipeline",
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
            label: Some("VertexInput buffer"),
        });

        let render_progress = RenderProgress::new();

        let frame_number = 1_u32
            .checked_add(
                crate::argument_value("--sample-offset")
                    .map_or(0, |v| v.parse::<u32>().expect("invalid sample offset")),
            )
            .expect("sample offset overflow");

        Ok(Self {
            vertex_uniform_bind_group,
            frame_data_buffer,
            images,
            camera_buffer,
            sampling_parameter_buffer,
            hw_sky_state_buffer,
            parameter_bind_group,
            scene_bind_group,
            vertex_buffer,
            accumulation_pipeline,
            denoising_pipeline,
            filter_dirty: true,
            display_pipeline,
            latest_render_params: *render_params,
            render_progress,
            frame_number,
        })
    }

    /// Dispatch tracing, then filtering; presentation only reads the completed result.
    pub fn accumulate_frame(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if !self.filter_dirty
            && self.render_progress.accumulated_samples()
                >= self.latest_render_params.sampling.max_samples_per_pixel
        {
            // The cached image needs no uniform uploads or compute dispatches.
            // Keep random seeds aligned with redraws when rendering resumes.
            self.frame_number = self.frame_number.wrapping_add(1);
            return;
        }
        let params = self.prepare_frame(queue);
        let (width, height) = self.latest_render_params.viewport_size;
        let groups = (width.div_ceil(8), height.div_ceil(8));
        if params.num_samples_per_pixel != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path tracing pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.accumulation_pipeline);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.trace, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.set_bind_group(3, &self.scene_bind_group, &[]);
            pass.dispatch_workgroups(
                width.div_ceil(trace_workgroup_size()[0]),
                height.div_ceil(trace_workgroup_size()[1]),
                1,
            );
            self.filter_dirty = true;
        }
        if self.filter_dirty {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("denoising pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.denoising_pipeline);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.filter, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
            self.filter_dirty = false;
        }
    }

    /// Float32 RGB averages, little endian, row major; no tone mapping or filtering.
    pub fn save_raw(&self, device: &wgpu::Device, queue: &wgpu::Queue, path: &str, samples: u32) {
        let size = self.images.accumulation.handle().size();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(self.images.accumulation.handle(), 0, &buffer, 0, size);
        queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let mut bytes = Vec::with_capacity(size as usize);
        for chunk in data.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().unwrap()) / samples as f32;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    /// Isolated filter GPU time, including warmup; p95 of 100 warmed dispatches.
    pub fn benchmark_filter(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
        assert!(
            device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "GPU timestamps required"
        );
        let query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut times = Vec::new();
        for i in 0..110 {
            self.filter_dirty = true;
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&query, 0);
            self.accumulate_frame(queue, &mut encoder);
            encoder.write_timestamp(&query, 1);
            encoder.resolve_query_set(&query, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
            queue.submit([encoder.finish()]);
            read.slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            {
                let data = read.slice(..).get_mapped_range().unwrap();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                if i >= 10 {
                    times.push(
                        (ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period())
                            / 1e6,
                    );
                }
            }
            read.unmap();
        }
        times.sort_by(f64::total_cmp);
        times[94]
    }

    fn prepare_frame(&mut self, queue: &wgpu::Queue) -> GpuSamplingParams {
        let gpu_sampling_params = self
            .render_progress
            .next_frame(&self.latest_render_params.sampling);
        {
            queue.write_buffer(
                &self.sampling_parameter_buffer.handle(),
                0,
                bytemuck::cast_slice(&[gpu_sampling_params]),
            );
        }

        {
            let viewport_size = self.latest_render_params.viewport_size;
            let frame_number = self.frame_number;
            let frame_data = [
                viewport_size.0,
                viewport_size.1,
                frame_number,
                self.latest_render_params.exposure.to_bits(),
            ];
            queue.write_buffer(
                &self.frame_data_buffer.handle(),
                0,
                bytemuck::cast_slice(&frame_data),
            );
        }

        self.frame_number = self.frame_number.wrapping_add(1);
        gpu_sampling_params
    }

    pub fn render_frame(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.display_pipeline);
        render_pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.images.display, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..VERTICES.len() as u32, 0..1);
    }

    pub fn set_render_params(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_params: &RenderParams,
    ) -> Result<(), RenderParamsValidationError> {
        if *render_params == self.latest_render_params {
            return Ok(());
        }

        match render_params.validate() {
            Ok(_) => {}
            Err(err) => return Err(err),
        }

        // Exposure is a display setting: update it even after convergence,
        // without throwing away the expensive accumulated samples.
        queue.write_buffer(
            self.frame_data_buffer.handle(),
            12,
            bytemuck::bytes_of(&render_params.exposure),
        );
        let mut previous = self.latest_render_params;
        previous.exposure = render_params.exposure;
        if previous == *render_params {
            self.latest_render_params = *render_params;
            return Ok(());
        }
        let buffer_size = image_buffer_size(device, render_params.viewport_size)?;
        let sky_state = if render_params.sky != self.latest_render_params.sky {
            Some(render_params.sky.to_sky_state()?)
        } else {
            None
        };
        self.images
            .resize(device, &self.frame_data_buffer, buffer_size);

        if let Some(sky_state) = sky_state {
            queue.write_buffer(
                &self.hw_sky_state_buffer.handle(),
                0,
                bytemuck::bytes_of(&sky_state),
            );
        }

        {
            let camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);
            queue.write_buffer(&self.camera_buffer.handle(), 0, bytemuck::bytes_of(&camera));
        }

        self.latest_render_params = *render_params;

        self.render_progress.reset();
        self.filter_dirty = true;

        Ok(())
    }

    pub fn progress(&self) -> f32 {
        self.render_progress.accumulated_samples() as f32
            / self.latest_render_params.sampling.max_samples_per_pixel as f32
    }
}

fn image_buffer_size(
    device: &wgpu::Device,
    (width, height): (u32, u32),
) -> Result<u64, RenderParamsValidationError> {
    let limits = device.limits();
    let max_bytes = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size));
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<[f32; 3]>() as u64))
        .filter(|&bytes| bytes <= max_bytes)
        .ok_or(RenderParamsValidationError::ViewportBufferTooLarge {
            width,
            height,
            max_bytes,
        })
}

fn tracing_stages() -> wgpu::ShaderStages {
    let stages = wgpu::ShaderStages::COMPUTE;
    #[cfg(test)]
    let stages = stages | wgpu::ShaderStages::FRAGMENT;
    stages
}

#[derive(Error, Debug)]
pub enum RenderParamsValidationError {
    #[error("exposure must be finite and positive")]
    Exposure,
    #[error("scene buffer needs {0} bytes, exceeding this device's storage-buffer limit")]
    SceneBufferTooLarge(usize),
    #[error(
        "viewport ({width}, {height}) exceeds the accumulation buffer limit of {max_bytes} bytes"
    )]
    ViewportBufferTooLarge {
        width: u32,
        height: u32,
        max_bytes: u64,
    },
    #[error("max_samples_per_pixel ({0}) is not a multiple of num_samples_per_pixel ({1})")]
    MaxSampleCountNotMultiple(u32, u32),
    #[error("viewport_size elements cannot be zero: ({0}, {1})")]
    ViewportSize(u32, u32),
    #[error("vfov must be between 0..=90 degrees")]
    VfovOutOfRange(f32),
    #[error("aperture must be between 0..=1")]
    ApertureOutOfRange(f32),
    #[error("focus_distance must be greater than zero")]
    FocusDistanceOutOfRange(f32),
    #[error(transparent)]
    HwSkyModelValidationError(#[from] hw_skymodel::rgb::Error),
}

#[derive(Default)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub materials: Vec<Material>,
    pub mesh: mesh::Mesh,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    // NOTE: naga memory alignment issue, see discussion at
    // https://github.com/gfx-rs/naga/issues/2000
    // It's safer to just use Vec4 instead of Vec3.
    center: glm::Vec4,  // 0 byte offset
    radius: f32,        // 16 byte offset
    material_idx: u32,  // 20 byte offset
    _padding: [u32; 2], // 24 byte offset, 8 bytes size
}

impl Sphere {
    pub fn new(center: glm::Vec3, radius: f32, material_idx: u32) -> Self {
        Self {
            center: glm::vec3_to_vec4(&center),
            radius,
            material_idx,
            _padding: [0_u32; 2],
        }
    }
}

pub enum Material {
    Lambertian { albedo: Texture },
    Metal { albedo: Texture, fuzz: f32 },
    Dielectric { refraction_index: f32 },
    Checkerboard { even: Texture, odd: Texture },
    Emissive { emit: Texture },
}

#[derive(Clone, Copy, PartialEq)]
pub struct RenderParams {
    pub camera: Camera,
    pub sky: SkyParams,
    pub sampling: SamplingParams,
    pub viewport_size: (u32, u32),
    /// Linear multiplier before tone mapping.
    pub exposure: f32,
}

impl RenderParams {
    fn validate(&self) -> Result<(), RenderParamsValidationError> {
        if !self.exposure.is_finite() || self.exposure <= 0. {
            return Err(RenderParamsValidationError::Exposure);
        }
        if self.sampling.max_samples_per_pixel % self.sampling.num_samples_per_pixel != 0 {
            return Err(RenderParamsValidationError::MaxSampleCountNotMultiple(
                self.sampling.max_samples_per_pixel,
                self.sampling.num_samples_per_pixel,
            ));
        }

        if self.viewport_size.0 == 0_u32 || self.viewport_size.1 == 0_u32 {
            return Err(RenderParamsValidationError::ViewportSize(
                self.viewport_size.0,
                self.viewport_size.1,
            ));
        }

        if !(Angle::degrees(0.0)..=Angle::degrees(90.0)).contains(&self.camera.vfov) {
            return Err(RenderParamsValidationError::VfovOutOfRange(
                self.camera.vfov.as_degrees(),
            ));
        }

        if !(0.0..=1.0).contains(&self.camera.aperture) {
            return Err(RenderParamsValidationError::ApertureOutOfRange(
                self.camera.aperture,
            ));
        }

        if self.camera.focus_distance < 0.0 {
            return Err(RenderParamsValidationError::FocusDistanceOutOfRange(
                self.camera.focus_distance,
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct Camera {
    pub eye_pos: glm::Vec3,
    pub eye_dir: glm::Vec3,
    pub up: glm::Vec3,
    /// Angle must be between 0..=90 degrees.
    pub vfov: Angle,
    /// Aperture must be between 0..=1.
    pub aperture: f32,
    /// Focus distance must be a positive number.
    pub focus_distance: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub struct SkyParams {
    // Azimuth must be between 0..=360 degrees
    pub azimuth_degrees: f32,
    // Inclination must be between 0..=90 degrees
    pub zenith_degrees: f32,
    // Turbidity must be between 1..=10
    pub turbidity: f32,
    // Albedo elements must be between 0..=1
    pub albedo: [f32; 3],
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            azimuth_degrees: 0_f32,
            zenith_degrees: 85_f32,
            turbidity: 4_f32,
            albedo: [1_f32; 3],
        }
    }
}

impl SkyParams {
    fn to_sky_state(self: &SkyParams) -> Result<GpuSkyState, hw_skymodel::rgb::Error> {
        let azimuth = Angle::degrees(self.azimuth_degrees).as_radians();
        let zenith = Angle::degrees(self.zenith_degrees).as_radians();
        let sun_direction = [
            zenith.sin() * azimuth.cos(),
            zenith.cos(),
            zenith.sin() * azimuth.sin(),
            0_f32,
        ];

        let state = hw_skymodel::rgb::SkyState::new(&hw_skymodel::rgb::SkyParams {
            elevation: FRAC_PI_2 - zenith,
            turbidity: self.turbidity,
            albedo: self.albedo,
        })?;

        let (params_data, radiance_data) = state.raw();

        let sampling = sky_sampling_table(&params_data, &radiance_data, sun_direction);
        Ok(GpuSkyState {
            sampling,
            params: params_data,
            radiances: radiance_data,
            _padding: [0_u32, 2],
            sun_direction,
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub max_samples_per_pixel: u32,
    pub num_samples_per_pixel: u32,
    pub num_bounces: u32,
    pub denoising_enabled: bool,
    pub denoising_radius: u32, // 0 means effectively disabled, 1 for 3x3, 2 for 5x5 etc.
    pub denoising_sigma_color: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_samples_per_pixel: 256_u32,
            num_samples_per_pixel: 1_u32,
            num_bounces: 3_u32,
            denoising_enabled: true,
            denoising_radius: 1, // 3x3 kernel
            denoising_sigma_color: 0.15,
        }
    }
}
struct RenderProgress {
    accumulated_samples_per_pixel: u32,
}

impl RenderProgress {
    pub fn new() -> Self {
        Self {
            accumulated_samples_per_pixel: 0_u32,
        }
    }

    pub fn next_frame(&mut self, sampling_params: &SamplingParams) -> GpuSamplingParams {
        let current_accumulated_samples = self.accumulated_samples_per_pixel;
        let next_accumulated_samples =
            sampling_params.num_samples_per_pixel + current_accumulated_samples;

        let denoising_enabled_gpu = if sampling_params.denoising_enabled {
            1_u32
        } else {
            0_u32
        };

        // Initial state or reset
        if current_accumulated_samples == 0_u32 {
            self.accumulated_samples_per_pixel = next_accumulated_samples;
            GpuSamplingParams {
                num_samples_per_pixel: sampling_params.num_samples_per_pixel,
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: next_accumulated_samples,
                clear_accumulated_samples: 1_u32,
                denoising_enabled: denoising_enabled_gpu,
                denoising_radius: sampling_params.denoising_radius,
                denoising_sigma_color: sampling_params.denoising_sigma_color,
                _padding_denoiser: 0,
            }
        }
        // Progressive render
        else if next_accumulated_samples <= sampling_params.max_samples_per_pixel {
            self.accumulated_samples_per_pixel = next_accumulated_samples;
            GpuSamplingParams {
                num_samples_per_pixel: sampling_params.num_samples_per_pixel,
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: next_accumulated_samples,
                clear_accumulated_samples: 0_u32,
                denoising_enabled: denoising_enabled_gpu,
                denoising_radius: sampling_params.denoising_radius,
                denoising_sigma_color: sampling_params.denoising_sigma_color,
                _padding_denoiser: 0,
            }
        }
        // Completed render
        else {
            GpuSamplingParams {
                num_samples_per_pixel: 0_u32, // Stop rendering new samples
                num_bounces: sampling_params.num_bounces,
                accumulated_samples_per_pixel: current_accumulated_samples,
                clear_accumulated_samples: 0_u32,
                denoising_enabled: denoising_enabled_gpu,
                denoising_radius: sampling_params.denoising_radius,
                denoising_sigma_color: sampling_params.denoising_sigma_color,
                _padding_denoiser: 0,
            }
        }
    }

    pub fn reset(&mut self) {
        self.accumulated_samples_per_pixel = 0_u32;
    }

    pub fn accumulated_samples(&self) -> u32 {
        self.accumulated_samples_per_pixel
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCamera {
    eye: glm::Vec3,
    _padding1: f32,
    horizontal: glm::Vec3,
    _padding2: f32,
    vertical: glm::Vec3,
    _padding3: f32,
    u: glm::Vec3,
    _padding4: f32,
    v: glm::Vec3,
    lens_radius: f32,
    lower_left_corner: glm::Vec3,
    _padding5: f32,
}

impl GpuCamera {
    pub fn new(camera: &Camera, viewport_size: (u32, u32)) -> Self {
        let lens_radius = 0.5_f32 * camera.aperture;
        let aspect = viewport_size.0 as f32 / viewport_size.1 as f32;
        let theta = camera.vfov.as_radians();
        let half_height = camera.focus_distance * (0.5_f32 * theta).tan();
        let half_width = aspect * half_height;

        let w = glm::normalize(&camera.eye_dir);
        let v = glm::normalize(&camera.up);
        let u = glm::cross(&w, &v);

        let lower_left_corner =
            camera.eye_pos + camera.focus_distance * w - half_width * u - half_height * v;
        let horizontal = 2_f32 * half_width * u;
        let vertical = 2_f32 * half_height * v;

        Self {
            eye: camera.eye_pos,
            _padding1: 0_f32,
            horizontal,
            _padding2: 0_f32,
            vertical,
            _padding3: 0_f32,
            u,
            _padding4: 0_f32,
            v,
            lens_radius,
            lower_left_corner,
            _padding5: 0_f32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterial {
    id: u32,
    desc1: TextureDescriptor,
    desc2: TextureDescriptor,
    x: f32,
}

impl GpuMaterial {
    pub fn lambertian(albedo: &Texture, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 0_u32,
            desc1: Self::append_to_global_texture_data(albedo, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: 0_f32,
        }
    }

    pub fn metal(albedo: &Texture, fuzz: f32, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 1_u32,
            desc1: Self::append_to_global_texture_data(albedo, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: fuzz,
        }
    }

    pub fn dielectric(refraction_index: f32) -> Self {
        Self {
            id: 2_u32,
            desc1: TextureDescriptor::empty(),
            desc2: TextureDescriptor::empty(),
            x: refraction_index,
        }
    }

    pub fn checkerboard(
        even: &Texture,
        odd: &Texture,
        global_texture_data: &mut Vec<[f32; 3]>,
    ) -> Self {
        Self {
            id: 3_u32,
            desc1: Self::append_to_global_texture_data(even, global_texture_data),
            desc2: Self::append_to_global_texture_data(odd, global_texture_data),
            x: 0_f32,
        }
    }

    pub fn emissive(emit: &Texture, global_texture_data: &mut Vec<[f32; 3]>) -> Self {
        Self {
            id: 4_u32,
            desc1: Self::append_to_global_texture_data(emit, global_texture_data),
            desc2: TextureDescriptor::empty(),
            x: 0_f32,
        }
    }

    fn append_to_global_texture_data(
        texture: &Texture,
        global_texture_data: &mut Vec<[f32; 3]>,
    ) -> TextureDescriptor {
        let dimensions = texture.dimensions();
        let offset = global_texture_data.len() as u32;
        global_texture_data.extend_from_slice(texture.as_slice());
        TextureDescriptor {
            width: dimensions.0,
            height: dimensions.1,
            offset,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureDescriptor {
    width: u32,
    height: u32,
    offset: u32,
}

impl TextureDescriptor {
    pub fn empty() -> Self {
        Self {
            width: 0_u32,
            height: 0_u32,
            offset: 0xffffffff,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSkyState {
    params: [f32; 27],          // 0 byte offset, 108 byte size
    radiances: [f32; 3],        // 108 byte offset, 12 byte size
    _padding: [u32; 2],         // 120 byte offset, 8 byte size
    sun_direction: [f32; 4],    // 128 byte offset, 16 byte size
    sampling: [[f32; 2]; 2048], // CDF and solid-angle PDF, 64x32 equal-area cells
}

// Sample the same analytic sky as WGSL, including its mirrored lower hemisphere.
// A 5% uniform component provides full support even for very dim cells.
fn sky_sampling_table(params: &[f32; 27], radiances: &[f32; 3], sun: [f32; 4]) -> [[f32; 2]; 2048] {
    let mut weights = [0.0_f64; 2048];
    for (i, weight) in weights.iter_mut().enumerate() {
        let y = 1.0 - 2.0 * ((i / 64) as f64 + 0.5) / 32.0;
        let phi = std::f64::consts::TAU * ((i % 64) as f64 + 0.5) / 64.0;
        let r = (1.0 - y * y).sqrt();
        let cg = (r * phi.cos() * f64::from(sun[0])
            + y * f64::from(sun[1])
            + r * phi.sin() * f64::from(sun[2]))
        .clamp(-1.0, 1.0);
        let gamma = cg.acos();
        let mut rgb = [0.0; 3];
        for channel in 0..3 {
            let p: [f64; 9] = std::array::from_fn(|i| f64::from(params[channel * 9 + i]));
            rgb[channel] = (f64::from(radiances[channel])
                * (1.0 + p[0] * (p[1] / (y.abs() + 0.01)).exp())
                * (p[2]
                    + p[3] * (p[4] * gamma).exp()
                    + p[5] * cg * cg
                    + p[6] * (1.0 + cg * cg) / (1.0 + p[8] * p[8] - 2.0 * p[8] * cg).powf(1.5)
                    + p[7] * y.abs().sqrt()))
            .max(0.0);
        }
        *weight = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    }
    let total: f64 = weights.iter().sum();
    let mut cdf = 0.0;
    let mut table = [[0.0; 2]; 2048];
    for (i, &weight) in weights.iter().enumerate() {
        let probability = if total > 0.0 {
            0.95 * weight / total + 0.05 / 2048.0
        } else {
            1.0 / 2048.0
        };
        cdf += probability;
        let rounded = if i == 2047 { 1.0 } else { cdf as f32 };
        let previous = if i == 0 { 0.0 } else { table[i - 1][0] };
        table[i] = [
            rounded,
            ((rounded - previous) * 2048.0 / (4.0 * std::f32::consts::PI)),
        ];
    }
    table[2047][0] = 1.0;
    table
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSamplingParams {
    num_samples_per_pixel: u32,
    num_bounces: u32,
    accumulated_samples_per_pixel: u32,
    clear_accumulated_samples: u32,
    denoising_enabled: u32, // 0 for false, 1 for true
    denoising_radius: u32,
    denoising_sigma_color: f32,
    _padding_denoiser: u32, // To make struct size 32 bytes (8 * 4 bytes)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VertexUniforms {
    view_projection_matrix: glm::Mat4,
    model_matrix: glm::Mat4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // @location(0)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                // @location(1)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: std::mem::size_of::<[f32; 2]>() as u64,
                    shader_location: 1,
                },
            ],
        }
    }
}

fn unit_quad_projection_matrix() -> glm::Mat4 {
    let sw = 0.5_f32;
    let sh = 0.5_f32;

    // Our ortho camera is just centered at (0, 0)

    let left = -sw;
    let right = sw;
    let bottom = -sh;
    let top = sh;

    // DirectX, Metal, wgpu share the same left-handed coordinate system
    // for their normalized device coordinates:
    // https://github.com/gfx-rs/gfx/tree/master/src/backend/dx12
    glm::ortho_lh_zo(left, right, bottom, top, -1_f32, 1_f32)
}

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, 0.5],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5],
        tex_coords: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5],
        tex_coords: [1.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5],
        tex_coords: [0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5],
        tex_coords: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5],
        tex_coords: [1.0, 0.0],
    },
];
