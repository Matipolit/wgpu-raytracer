use gpu_buffer::{StorageBuffer, UniformBuffer};
use wgpu::util::DeviceExt;
pub use {
    angle::Angle,
    bloom::{DEFAULT_STRENGTH as DEFAULT_BLOOM, Tonemapper},
    texture::Texture,
};

use thiserror::Error;

mod angle;
mod bloom;
mod gpu_buffer;
pub mod gpu_textures;
mod hwrt;
mod image_buffers;
pub mod mesh;
pub(crate) mod profiler;
mod temporal;
mod texture;
mod tuner;
pub(crate) mod upscale;
mod visibility;
mod wavefront;

#[cfg(test)]
mod tests;

use std::f32::consts::*;

pub(crate) const TRACE_WORKGROUP_SIZE: [u32; 2] = [8, 8];

/// Explicit tuning override for reproducible GPU experiments; cached per process.
/// GPU instance records (mesh.wgsl meshInstances): world-to-object, object-to-world and
/// previous object-to-world rows, then world bounds; at least one record.
fn instance_words(
    current: &[glm::Mat4],
    previous: &[glm::Mat4],
    bounds: &[([f32; 3], [f32; 3])],
) -> Vec<[f32; 4]> {
    let rows = |m: &glm::Mat4| [0, 1, 2].map(|r| [m[(r, 0)], m[(r, 1)], m[(r, 2)], m[(r, 3)]]);
    let mut words = Vec::new();
    for ((m, p), (lo, hi)) in current.iter().zip(previous).zip(bounds) {
        let inverse = m
            .try_inverse()
            .expect("object transform must be invertible");
        words.extend(rows(&inverse));
        words.extend(rows(m));
        words.extend(rows(p));
        let mut min = glm::vec3(f32::MAX, f32::MAX, f32::MAX);
        let mut max = -min;
        for k in 0..8 {
            let corner = glm::vec4(
                if k & 1 == 0 { lo[0] } else { hi[0] },
                if k & 2 == 0 { lo[1] } else { hi[1] },
                if k & 4 == 0 { lo[2] } else { hi[2] },
                1.,
            );
            let w = m * corner;
            min = glm::min2(&min, &w.xyz());
            max = glm::max2(&max, &w.xyz());
        }
        words.push([min.x, min.y, min.z, 0.]);
        words.push([max.x, max.y, max.z, 0.]);
    }
    if words.is_empty() {
        words.push([0.; 4]);
    }
    words
}

/// Sub-pixel jitter of frame `frame` (Halton 2, 3), identical to raytracer.wgsl
/// `frameJitter`: shared by the tracer (SHARED_JITTER), the visibility raster and the upscaler.
pub(crate) fn frame_jitter(frame: u32) -> [f32; 2] {
    std::array::from_fn(|axis| {
        let base = 2 + axis as u32;
        let (mut i, mut f, mut r) = (frame % 1024 + 1, 1f32, 0f32);
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    })
}

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
    prev_camera_buffer: UniformBuffer,
    /// Camera of the frame most recently traced (becomes the previous camera).
    last_camera: GpuCamera,
    sphere_buffer: wgpu::Buffer,
    sphere_prev_buffer: StorageBuffer,
    /// Moving objects: instance buffer, object-space bounds, last transforms.
    instance_buffer: StorageBuffer,
    object_bounds: Vec<([f32; 3], [f32; 3])>,
    last_object_transforms: Option<Vec<glm::Mat4>>,
    last_sphere_centers: Vec<[f32; 4]>,
    sampling_parameter_buffer: UniformBuffer,
    hw_sky_state_buffer: StorageBuffer,
    parameter_bind_group: wgpu::BindGroup,
    scene_bind_group: wgpu::BindGroup,
    /// Trace pipelines, one per tuner variant.
    trace_pipelines: Vec<wgpu::ComputePipeline>,
    /// Startup-level configuration of the trace kernel and what is needed to rebuild it.
    module_config: tuner::ModuleConfig,
    source_prefix: String,
    source_suffix: String,
    trace_layout: wgpu::PipelineLayout,
    /// ReSTIR's first pass (raytracer.wgsl csRestirInitial), when the scene has lamps.
    restir_pipeline: Option<(wgpu::ComputePipeline, Option<wgpu::ComputePipeline>)>,
    /// Joint network: csOutputGuides and its output-sized frame data.
    output_guides_pipeline: Option<wgpu::ComputePipeline>,
    output_frame: UniformBuffer,
    output_guides_group: Option<wgpu::BindGroup>,
    /// ReSTIR GI reuse (raytracer.wgsl csRestirGi), for mesh scenes.
    gi_pipeline: Option<wgpu::ComputePipeline>,
    bloom: bloom::Bloom,
    profiler: Option<profiler::PassProfiler>,
    trace_constants: Vec<(&'static str, f64)>,
    /// Identifies the scene in the calibration cache.
    scene_key: String,
    tuner: tuner::AutoTuner,
    timer: Option<tuner::FrameTimer>,
    denoising_pipeline: wgpu::ComputePipeline,
    exposure_pipeline: wgpu::ComputePipeline,
    wavefront: Option<wavefront::Wavefront>,
    temporal: Option<temporal::TemporalDenoiser>,
    visibility: Option<visibility::Visibility>,
    /// Temporal upscaler (`--render-scale` below 1) and the output resolution.
    upscaler: Option<upscale::Upscaler>,
    output_size: (u32, u32),
    /// Frame index (frameData.z) of the last traced frame.
    last_frame_index: u32,
    visibility_rastered: bool,
    /// Trace only this (x, y, width, height) rectangle; None traces the whole frame.
    trace_region: Option<(u32, u32, u32, u32)>,
    /// Skip the per-frame filter pass (offline captures read raw radiance only).
    skip_filter: bool,
    filter_dirty: bool,
    display_pipeline: wgpu::RenderPipeline,
    latest_render_params: RenderParams,
    render_progress: RenderProgress,
    frame_number: u32,
    /// ReSTIR frames rendered (the reservoir ping-pong; see `SamplingParams::restir`).
    restir_frames: u32,
    /// GPU time of the most recently measured trace pass (ms; 0 before the first).
    pub last_trace_ms: f64,
}

impl Raytracer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        scene: &Scene,
        render_params: &RenderParams,
    ) -> Result<Self, RenderParamsValidationError> {
        match render_params.validate() {
            Ok(_) => {}
            Err(err) => return Err(err),
        }
        // Upscaling: everything up to the upscaler runs at render resolution; the caller's
        // viewport is the output resolution.
        let output_size = render_params.viewport_size;
        let scaled_params = RenderParams {
            viewport_size: upscale::render_size(output_size),
            ..*render_params
        };
        let render_params = &scaled_params;
        let joint_model = upscale::JointModel::from_args(device);
        let upscaling = upscale::render_scale() < 1. || joint_model.is_some();
        let variants = trace_variants(device, scene);
        // The visibility raster: G-buffer experiments, or a variant starting paths from it.
        let vis_on = device.features().contains(wgpu::Features::PRIMITIVE_INDEX)
            && (crate::has_flag("--visibility") || variants.iter().any(|v| v.visibility_primary));

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

        // ReSTIR history only where ReSTIR can run (the flags, or these parameters).
        let restir_possible = !crate::has_flag("--no-restir")
            && (crate::has_flag("--restir")
                || crate::has_flag("--temporal-model")
                || upscale::joint_requested()
                || render_params.sampling.restir);
        let history = image_buffers::History {
            reservoirs: restir_possible && scene.mesh.lamp_count() > 0,
            gi: restir_possible
                && !scene.mesh.triangles.is_empty()
                && !crate::has_flag("--no-restir-gi"),
        };
        let images = image_buffers::ImageBuffers::new(
            device,
            &frame_data_buffer,
            image_buffer_size(device, render_params.viewport_size, history)?,
            render_params.auto_exposure,
            history,
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

        // Previous-frame camera for motion vectors; starts equal to the current one.
        let initial_camera = GpuCamera::new(&render_params.camera, render_params.viewport_size);
        let prev_camera_buffer = UniformBuffer::new_from_bytes(
            device,
            bytemuck::bytes_of(&initial_camera),
            3_u32,
            Some("previous camera buffer"),
        );

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
                    camera_buffer.layout(scene_stages(vis_on)),
                    sampling_parameter_buffer.layout(scene_stages(vis_on)),
                    hw_sky_state_buffer.layout(scene_stages(vis_on), true),
                    prev_camera_buffer.layout(scene_stages(vis_on)),
                ],
                label: Some("parameter layout"),
            });

        let parameter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &parameter_bind_group_layout,
            entries: &[
                camera_buffer.binding(),
                sampling_parameter_buffer.binding(),
                hw_sky_state_buffer.binding(),
                prev_camera_buffer.binding(),
            ],
            label: Some("parameter bind group"),
        });

        let sphere_prev_centers: Vec<[f32; 4]> = scene
            .spheres
            .iter()
            .map(|s| s.center.into())
            .chain(std::iter::once([0.; 4]))
            .collect();
        let sphere_prev_buffer = StorageBuffer::new_from_bytes(
            device,
            bytemuck::cast_slice(&sphere_prev_centers),
            7_u32,
            Some("previous sphere centres"),
        );
        // Moving objects: 11 vec4 per instance (see mesh.wgsl meshInstances), identity
        // transforms until set_objects.
        let object_bounds: Vec<([f32; 3], [f32; 3])> =
            scene.mesh.objects.iter().map(|o| (o.min, o.max)).collect();
        assert!(
            object_bounds.is_empty() || std::env::var("RT_QUAD_Q8").is_err(),
            "moving objects need the plain 4-wide BVH"
        );
        let identity: Vec<glm::Mat4> = vec![glm::identity(); object_bounds.len()];
        let instance_buffer = StorageBuffer::new_from_bytes(
            device,
            bytemuck::cast_slice(&instance_words(&identity, &identity, &object_bounds)),
            9,
            Some("moving object instances"),
        );
        let (scene_bind_group_layout, scene_bind_group, light_count, sphere_handle) = {
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
            let size = mesh_words.len() * 4;
            if size as u64 > u64::from(device.limits().max_storage_buffer_binding_size) {
                return Err(RenderParamsValidationError::SceneBufferTooLarge(size));
            }
            let mesh_buffer = StorageBuffer::new_from_bytes(
                device,
                bytemuck::cast_slice(&mesh_words),
                4,
                Some("mesh geometry and BVH"),
            );
            let mesh_textures = gpu_textures::GpuTextures::new(device, queue, &scene.mesh);
            let [texture_layout, sampler_layout] =
                mesh_textures.layout_entries(scene_stages(vis_on));
            let texture_views = mesh_textures.views();
            let samplers = mesh_textures.samplers();

            let hw_accel = mesh::hw_rt().then(|| hwrt::HwAccel::new(device, queue, &scene.mesh));
            let scene_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        hwrt::HwAccel::layout_entry(scene_stages(vis_on)),
                        sphere_buffer.layout(scene_stages(vis_on), true),
                        material_buffer.layout(scene_stages(vis_on), true),
                        texture_buffer.layout(scene_stages(vis_on), true),
                        light_buffer.layout(scene_stages(vis_on), true),
                        mesh_buffer.layout(scene_stages(vis_on), true),
                        texture_layout,
                        sampler_layout,
                        sphere_prev_buffer.layout(scene_stages(vis_on), true),
                        instance_buffer.layout(scene_stages(vis_on), true),
                    ][usize::from(hw_accel.is_none())..],
                    label: Some("scene layout"),
                });
            let mut scene_entries = vec![
                sphere_buffer.binding(),
                material_buffer.binding(),
                texture_buffer.binding(),
                light_buffer.binding(),
                mesh_buffer.binding(),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureViewArray(&texture_views),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::SamplerArray(&samplers),
                },
                sphere_prev_buffer.binding(),
                instance_buffer.binding(),
            ];
            if let Some(accel) = &hw_accel {
                scene_entries.push(wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::AccelerationStructure(&accel.tlas),
                });
            }
            let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &scene_bind_group_layout,
                entries: &scene_entries,
                label: Some("scene bind group"),
            });

            (
                scene_bind_group_layout,
                scene_bind_group,
                light_indices.len() as f64,
                sphere_buffer.handle().clone(),
            )
        };

        let mut temporal_source = temporal::model_source(device);
        // An f16 model needs `enable f16;` at the start of the module.
        let model_f16 = temporal_source
            .as_deref()
            .is_some_and(|source| source.lines().any(|line| line == "enable f16;"));
        if let Some(source) = temporal_source.as_mut().filter(|_| model_f16) {
            *source = source.replacen("enable f16;\n", "", 1);
        }
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
        if model_f16 && !crate::has_flag("--fp16-state") {
            trace_source = format!("enable f16;\n{trace_source}");
        }
        // Hardware ray queries need the WGSL extension and the real implementation;
        // otherwise a stub keeps the (dead) HW_RT branches compiling.
        if vis_on {
            let start = trace_source.find("// VISIBILITY STUBS").unwrap();
            let end = trace_source.find("// END VISIBILITY STUBS").unwrap();
            trace_source.replace_range(start..end, "");
            trace_source = format!("enable primitive_index;\n{trace_source}");
        }
        let hw_source = if mesh::hw_rt() {
            trace_source = format!("enable wgpu_ray_query;\n{trace_source}");
            include_str!("hwrt.wgsl").to_owned()
        } else {
            String::new()
        };
        // The module is trace source + mesh source (varies with the module configuration)
        // + the rest; the parts are kept so calibration can rebuild the trace kernel.
        let module_config = default_module_config();
        let source_suffix = format!(
            "{}\n{}\n{}\n{}",
            hw_source,
            include_str!("wavefront.wgsl"),
            temporal_source.as_deref().unwrap_or(""),
            if vis_on {
                include_str!("visibility.wgsl")
            } else {
                ""
            }
        );
        let source = format!(
            "{}\n{}\n{}",
            trace_source,
            mesh_source(module_config),
            source_suffix
        );
        // naga guards every loop with an iteration counter. Our loops all terminate, and a
        // runaway loop would hang the GPU with or without the guard, so the guards only
        // cost time: 3.9% geomean over the benchmark views (up to 8%), bit-identical
        // output, tests and denoiser parity unchanged. `--loop-guards` restores them.
        // (The 4-wide pop loop was miscompiled without its guard; it is straight-line code now.)
        let shader = create_module(device, source);

        let vis_resolve_layout = vis_on.then(|| visibility::resolve_layout(device));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[
                Some(&vertex_uniform_bind_group_layout),
                Some(&images.trace_layout),
                Some(&parameter_bind_group_layout),
                Some(&scene_bind_group_layout),
                None,
                None,
                vis_resolve_layout.as_ref(),
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
        let bloom_display_layout = bloom::Bloom::display_layout(device);
        let display_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    Some(&vertex_uniform_bind_group_layout),
                    Some(&images.display_layout),
                    Some(&bloom_display_layout),
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
        let trace_constants = [
            ("SPHERE_COUNT", scene.spheres.len() as f64),
            ("INSTANCE_COUNT", scene.mesh.objects.len() as f64),
            ("LIGHT_COUNT", light_count),
            ("TRACE_MESHES", f64::from(!scene.mesh.triangles.is_empty())),
            ("TRACE_SPHERES", f64::from(!scene.spheres.is_empty())),
            (
                "RUSSIAN_ROULETTE",
                f64::from(!std::env::args().any(|a| a == "--no-russian-roulette")),
            ),
            (
                "SUN",
                f64::from(render_params.sky.sun && !render_params.sky.night),
            ),
            ("LAMPS", f64::from(scene.mesh.lamp_count() > 0)),
            ("DENOISER_INPUT", f64::from(temporal_source.is_some())),
            ("RESTIR", f64::from(scene.mesh.lamp_count() > 0)),
            ("RESTIR_BLOCKS", f64::from(restir_blocks())),
            ("GI_BLOCKS", f64::from(gi_blocks())),
            (
                "VIS_OUTPUT",
                f64::from(vis_on && upscale::output_guides_requested()),
            ),
            ("DI_SHARED", f64::from(di_shared())),
            (
                "GI_VISIBILITY_REUSE",
                std::env::var("RT_GI_VISIBILITY_REUSE").map_or(1., |v| v.parse().unwrap()),
            ),
            (
                "GI_TEMPORAL_TRUST",
                f64::from(std::env::var_os("RT_GI_TEMPORAL_RECHECK").is_none()),
            ),
            (
                "RESTIR_VISIBILITY_REUSE",
                f64::from(std::env::var_os("RT_RESTIR_VISIBILITY_REUSE").is_some()),
            ),
            (
                "GI_SPATIAL",
                std::env::var("RT_GI_SPATIAL")
                    .map_or(if gi_blocks() { 8. } else { 4. }, |v| v.parse().unwrap()),
            ),
            (
                "RESTIR_GI",
                f64::from(!scene.mesh.triangles.is_empty() && !crate::has_flag("--no-restir-gi")),
            ),
            // Experiments: RT_RESTIR_SPATIAL, RT_RESTIR_RADIUS, RT_RESTIR_CAP.
            (
                "RESTIR_SPATIAL",
                std::env::var("RT_RESTIR_SPATIAL").map_or(6., |v| v.parse().unwrap()),
            ),
            (
                "RESTIR_RADIUS",
                std::env::var("RT_RESTIR_RADIUS").map_or(16., |v| v.parse().unwrap()),
            ),
            (
                "RESTIR_CAP",
                std::env::var("RT_RESTIR_CAP").map_or(20., |v| v.parse().unwrap()),
            ),
            (
                "LAMP_CANDIDATES",
                lamp_candidates(0, "RT_LAMP_CANDIDATES", 4.),
            ),
            (
                "LAMP_CANDIDATES_DEEP",
                lamp_candidates(1, "RT_LAMP_CANDIDATES_DEEP", 2.),
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
                // Default: bounded visible normals. Sampling the normal distribution
                // (`--ndf-sampling`) wastes the specular samples that reflect below the
                // surface: 40% of shaded Sponza pixels got no light in a 1-spp ReSTIR
                // frame, 19% with this, at the same frame time (2026-11-18).
                f64::from(!crate::has_flag("--ndf-sampling") && !crate::has_flag("--ggx-vndf")),
            ),
            (
                "BLUE_NOISE_JITTER",
                f64::from(crate::has_flag("--blue-noise-jitter")),
            ),
            (
                "CENTER_JITTER",
                f64::from(crate::has_flag("--center-jitter")),
            ),
            (
                "SHARED_JITTER",
                f64::from(upscaling || crate::has_flag("--upscale-capture")),
            ),
            (
                "CONES_PRIMARY_ONLY",
                f64::from(crate::has_flag("--cones-primary-only")),
            ),
            ("VIS_PRIMARY", 0.),
            (
                "BLUE_NOISE_PATH",
                f64::from(crate::has_flag("--blue-noise-path")),
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
            ("PAIR_BVH", f64::from(mesh::pair_bvh())),
            ("QUAD_BVH", f64::from(mesh::quad_bvh())),
            ("OCT_BVH", f64::from(mesh::oct_bvh())),
            ("CW_BVH", f64::from(mesh::cw_bvh())),
            ("HW_RT", f64::from(mesh::hw_rt())),
            ("HW_CLOSEST", f64::from(module_config.hw.closest())),
            ("HW_SHADOWS", f64::from(module_config.hw.shadows())),
            (
                "HW_FLIP_FACING",
                f64::from(std::env::var("RT_HW_FLIP").is_ok()),
            ),
            (
                "QUAD_CULL",
                f64::from(std::env::var("RT_QUAD_CULL").is_ok()),
            ),
            ("QUAD_LEAF_COUNT", f64::from(mesh::quad_leaf_count())),
            ("QUAD_Q8", f64::from(mesh::quad_q8())),
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
            (
                "TRAVERSAL_STATS",
                f64::from(std::env::var("RT_STATS").is_ok()),
            ),
            ("GBUFFER", f64::from(gbuffer_enabled())),
            (
                "REFERENCE_GUIDES",
                f64::from(crate::has_flag("--reference-output")),
            ),
            (
                "COHERENCE_TEST",
                f64::from(std::env::var("RT_COHERENCE_TEST").is_ok()),
            ),
            ("TRACE_WIDTH", f64::from(module_config.workgroup[0])),
            ("TRACE_HEIGHT", f64::from(module_config.workgroup[1])),
        ];
        // One trace pipeline per variant; the auto-tuner picks one per frame.
        let trace_constants: Vec<(&'static str, f64)> = trace_constants.to_vec();
        let trace_pipelines = build_trace_pipelines(
            device,
            &shader,
            &pipeline_layout,
            &trace_constants,
            &variants,
            module_config,
        );
        // ReSTIR's first pass (lamp candidates, temporal reuse) and, with DI_SHARED, the
        // per-block spatial merge and shadow ray.
        let restir_pipeline = trace_constants
            .iter()
            .any(|&(name, value)| name == "RESTIR" && value != 0.)
            .then(|| {
                // Primary hits from the visibility raster when there is one (it costs a
                // fraction of re-tracing them).
                let constants: Vec<(&str, f64)> = trace_constants
                    .iter()
                    .map(|&(name, value)| match name {
                        "VIS_PRIMARY" => (name, f64::from(vis_on)),
                        _ => (name, value),
                    })
                    .collect();
                (
                    create_compute(
                        &pipeline_layout,
                        "csRestirInitial",
                        "ReSTIR initial reservoirs",
                        &constants,
                    ),
                    di_shared().then(|| {
                        create_compute(
                            &pipeline_layout,
                            "csRestirShared",
                            "ReSTIR shared block reservoirs",
                            &constants,
                        )
                    }),
                )
            });
        // Joint network: this frame's guides at output resolution (primary rays; the
        // visibility buffer is render-sized).
        let output_guides_pipeline = upscale::output_guides_requested().then(|| {
            let constants: Vec<(&str, f64)> = trace_constants
                .iter()
                .map(|&(name, value)| match name {
                    "VIS_PRIMARY" => (name, f64::from(vis_on)),
                    _ => (name, value),
                })
                .collect();
            create_compute(
                &pipeline_layout,
                "csOutputGuides",
                "output-resolution guides",
                &constants,
            )
        });
        let output_frame = UniformBuffer::new(device, 16, 0, Some("output frame data"));
        // ReSTIR GI reuse, after the path tracer (mesh scenes).
        let gi_pipeline = (!scene.mesh.triangles.is_empty() && !crate::has_flag("--no-restir-gi"))
            .then(|| {
                create_compute(
                    &pipeline_layout,
                    "csRestirGi",
                    "ReSTIR GI reuse",
                    &trace_constants,
                )
            });
        if variants.len() > 1 {
            eprintln!(
                "Auto-tune: {} trace variants {:?}",
                variants.len(),
                variants
            );
        }
        let denoising_pipeline = create_compute(
            &filter_pipeline_layout,
            "csDenoise",
            "denoising compute",
            &[(
                "SPARSE_DENOISE",
                f64::from(crate::has_flag("--sparse-denoise")),
            )],
        );
        let wavefront = (wavefront::enabled()).then(|| {
            assert!(
                !scene.mesh.triangles.is_empty() && scene.spheres.is_empty() && mesh::quad_bvh(),
                "--wavefront currently needs a mesh-only scene and --quad-bvh"
            );
            let layouts = wavefront::Wavefront::layouts(device);
            let pipeline_layouts = layouts.each_ref().map(|layout| {
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        Some(&vertex_uniform_bind_group_layout),
                        Some(&images.trace_layout),
                        Some(&parameter_bind_group_layout),
                        Some(&scene_bind_group_layout),
                        Some(layout),
                    ],
                    immediate_size: 0,
                    label: Some("wavefront layout"),
                })
            });
            let pipelines = [
                "csWfGenerate",
                "csWfArgs",
                "csWfTrace",
                "csWfShade",
                "csWfAccumulate",
            ]
            .map(|entry| {
                let layout = &pipeline_layouts[usize::from(entry == "csWfArgs")];
                create_compute(layout, entry, entry, &trace_constants)
            });
            let (width, height) = render_params.viewport_size;
            wavefront::Wavefront::new(device, layouts, pipelines, width * height)
        });
        let temporal = temporal_source.is_some().then(|| {
            let layout = temporal::TemporalDenoiser::layout(device);
            let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[
                    Some(&vertex_uniform_bind_group_layout),
                    Some(&images.trace_layout),
                    Some(&parameter_bind_group_layout),
                    Some(&scene_bind_group_layout),
                    Some(&empty),
                    Some(&layout),
                ],
                immediate_size: 0,
                label: Some("temporal denoiser layout"),
            });
            let model = temporal_source.as_deref().unwrap();
            // Entry names are built once per process; the pipeline helper takes &'static str.
            let pipelines = temporal::kernels(model)
                .into_iter()
                .map(|entry| {
                    let entry: &'static str = Box::leak(entry.into_boxed_str());
                    create_compute(&pipeline_layout, entry, entry, &trace_constants)
                })
                .collect();
            let (width, height) = render_params.viewport_size;
            temporal::TemporalDenoiser::new(
                device,
                layout,
                &empty,
                pipelines,
                images.filtered.handle(),
                width * height,
                model,
            )
        });
        let make_visibility = |size: (u32, u32)| {
            let raster_layout = visibility::raster_layout(device);
            let resolve_layout = vis_resolve_layout.clone().unwrap();
            let raster_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        None,
                        None,
                        Some(&parameter_bind_group_layout),
                        Some(&scene_bind_group_layout),
                        None,
                        None,
                        Some(&raster_layout),
                    ],
                    immediate_size: 0,
                    label: Some("visibility raster layout"),
                });
            let resolve_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        Some(&vertex_uniform_bind_group_layout),
                        Some(&images.trace_layout),
                        Some(&parameter_bind_group_layout),
                        Some(&scene_bind_group_layout),
                        None,
                        None,
                        Some(&resolve_layout),
                    ],
                    immediate_size: 0,
                    label: Some("visibility resolve layout"),
                });
            // RT_VIS_CONES=0 keeps the tracer's base-level texture sampling.
            let cones = std::env::var("RT_VIS_CONES").map_or(true, |v| v != "0");
            let resolve_constants: Vec<(&str, f64)> = trace_constants
                .iter()
                .map(|&(name, value)| {
                    (
                        name,
                        if name == "RAY_CONES" && cones {
                            1.
                        } else {
                            value
                        },
                    )
                })
                .collect();
            let resolve = create_compute(
                &resolve_pipeline_layout,
                "csVisibilityResolve",
                "visibility resolve",
                &resolve_constants,
            );
            visibility::Visibility::new(
                device,
                &scene.mesh,
                &shader,
                &raster_pipeline_layout,
                &raster_layout,
                resolve_layout,
                resolve,
                &trace_constants,
                size,
                &parameter_bind_group_layout,
            )
        };
        // Joint network: one raster at output resolution serves the tracer's primary
        // hits and the output guides (raytracer.wgsl VIS_OUTPUT).
        let visibility = vis_on.then(|| {
            make_visibility(if upscale::output_guides_requested() {
                output_size
            } else {
                render_params.viewport_size
            })
        });
        let exposure_pipeline = create_compute(
            &filter_pipeline_layout,
            "csExposure",
            "automatic exposure compute",
            &[],
        );
        let upscaler = upscaling.then(|| {
            upscale::Upscaler::new(
                device,
                images.filtered.handle(),
                &images.gbuffer,
                &images.display_layout,
                images.exposure.handle(),
                output_size,
                render_params.viewport_size,
                temporal_source.is_some(),
                joint_model.as_ref(),
                upscale::detail_requested(),
            )
        });
        // Bloom works on the image the display pass shows (the upscaler's output-size
        // image, or the render-size one).
        let bloom = bloom::Bloom::new(
            device,
            bloom_display_layout,
            upscaler
                .as_ref()
                .map_or(images.filtered.handle(), upscale::Upscaler::output),
            if upscaler.is_some() {
                output_size
            } else {
                render_params.viewport_size
            },
            render_params.tonemapper,
            render_params.bloom,
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

        let mut tracer = Self {
            vertex_uniform_bind_group,
            frame_data_buffer,
            images,
            camera_buffer,
            prev_camera_buffer,
            last_camera: initial_camera,
            sphere_buffer: sphere_handle,
            sphere_prev_buffer,
            instance_buffer,
            object_bounds,
            last_object_transforms: None,
            last_sphere_centers: sphere_prev_centers,
            sampling_parameter_buffer,
            hw_sky_state_buffer,
            parameter_bind_group,
            scene_bind_group,
            vertex_buffer,
            trace_pipelines,
            restir_pipeline,
            output_guides_group: None,
            output_guides_pipeline,
            output_frame,
            gi_pipeline,
            bloom,
            profiler: profiler::PassProfiler::new(device, queue),
            module_config,
            source_prefix: trace_source,
            source_suffix,
            trace_layout: pipeline_layout,
            trace_constants,
            scene_key: format!(
                "{}t{}v{}m{}s",
                scene.mesh.triangles.len(),
                scene.mesh.vertices.len(),
                scene.mesh.materials.len(),
                scene.spheres.len()
            ),
            tuner: tuner::AutoTuner::new(variants),
            timer: tuner::FrameTimer::new(device, queue),
            denoising_pipeline,
            exposure_pipeline,
            wavefront,
            temporal,
            visibility,
            upscaler,
            output_size,
            last_frame_index: 0,
            visibility_rastered: false,
            trace_region: None,
            skip_filter: false,
            filter_dirty: true,
            display_pipeline,
            latest_render_params: *render_params,
            render_progress,
            frame_number,
            restir_frames: 0,
            last_trace_ms: 0.,
        };
        tracer.output_guides_group = tracer
            .upscaler
            .as_ref()
            .and_then(|u| output_guides_group(device, &tracer.images, &tracer.output_frame, u));
        Ok(tracer)
    }

    /// Startup level of the auto-tuner: choose the trace kernel's traversal stack and
    /// workgroup shape for this GPU, driver and scene by timing the current view, and
    /// cache the choice (later runs apply it without measuring). `adapter` identifies
    /// the GPU and driver. Coordinate search: stacks at the default shape, then shapes.
    pub fn calibrate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, adapter: &str) {
        // Captures keep the per-frame tuner off (fixed variants) but calibrate the kernel.
        if (!auto_tune() && !crate::has_flag("--capture"))
            || self.wavefront.is_some()
            || self.timer.is_none()
            || crate::has_flag("--no-calibrate")
        {
            return;
        }
        // Revision 2: back-to-back measurement (earlier entries measured idle clocks);
        // RADV's wave size changes the best configuration.
        // The kernel's constants (ReSTIR, GI blocks, ...) and the sampling setup change
        // the best configuration too (review M7).
        let kernel = {
            use std::hash::{Hash, Hasher};
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            for (name, value) in &self.trace_constants {
                (name, value.to_bits()).hash(&mut hash);
            }
            let sampling = &self.latest_render_params.sampling;
            (sampling.num_bounces, sampling.restir).hash(&mut hash);
            hash.finish()
        };
        let key = format!(
            "{adapter}|{}|{}|c2{}{}|k{kernel:x}",
            self.scene_key,
            env!("CARGO_PKG_VERSION"),
            if mesh::hw_rt() { "|hw" } else { "" },
            std::env::var("RADV_PERFTEST").map_or(String::new(), |v| format!("|radv:{v}"))
        );
        let fresh = crate::has_flag("--recalibrate");
        let chosen = match tuner::load_cached(&key).filter(|_| !fresh) {
            Some(config) => config,
            None => {
                let start = std::time::Instant::now();
                let saved = (self.frame_number, self.last_camera);
                // Warm up: bring a power-managed GPU to its working clock first.
                let _ = self.measure_config(device, queue, self.module_config);
                let mut best = (
                    self.module_config,
                    self.measure_config(device, queue, self.module_config),
                );
                if crate::has_flag("--tuner-log") {
                    eprintln!("calibrate: {:?} {:.2} ms (default)", best.0, best.1);
                }
                let consider =
                    |this: &mut Self,
                     config: tuner::ModuleConfig,
                     best: &mut (tuner::ModuleConfig, f64)| {
                        if config != best.0 {
                            let ms = this.measure_config(device, queue, config);
                            if crate::has_flag("--tuner-log") {
                                eprintln!("calibrate: {config:?} {ms:.2} ms");
                            }
                            if ms < best.1 * 0.98 {
                                *best = (config, ms);
                            }
                        }
                    };
                // Hardware vs compute first (the stack and workgroup suit each differently);
                // compute is the baseline hardware queries must beat.
                if mesh::hw_rt() && std::env::var("RT_HW_MODE").is_err() {
                    for hw in [
                        tuner::HwRays::Compute,
                        tuner::HwRays::Shadows,
                        tuner::HwRays::All,
                    ] {
                        consider(self, tuner::ModuleConfig { hw, ..best.0 }, &mut best);
                    }
                }
                if self.module_config.stack.is_some() && !stack_override() {
                    for stack in [
                        tuner::Stack::Shared32,
                        tuner::Stack::Registers16,
                        tuner::Stack::Private64,
                    ] {
                        consider(
                            self,
                            tuner::ModuleConfig {
                                stack: Some(stack),
                                ..best.0
                            },
                            &mut best,
                        );
                    }
                }
                if std::env::var("RAYTRACER_WORKGROUP").is_err() {
                    for workgroup in [[8, 8], [16, 4], [32, 2], [64, 1]] {
                        consider(
                            self,
                            tuner::ModuleConfig {
                                workgroup,
                                ..best.0
                            },
                            &mut best,
                        );
                    }
                }
                (self.frame_number, self.last_camera) = saved;
                self.render_progress.reset();
                eprintln!(
                    "Calibrated trace kernel: {:?} ({:.2} ms), {:.1}s",
                    best.0,
                    best.1,
                    start.elapsed().as_secs_f64()
                );
                tuner::store_cached(&key, best.0);
                best.0
            }
        };
        if chosen != self.module_config {
            self.trace_pipelines =
                self.trace_pipelines_for(device, chosen, &self.tuner.variants.clone());
            self.module_config = chosen;
        }
        if crate::has_flag("--tuner-log") {
            eprintln!("calibrate: using {chosen:?}");
        }
    }

    fn trace_pipelines_for(
        &self,
        device: &wgpu::Device,
        config: tuner::ModuleConfig,
        variants: &[tuner::Variant],
    ) -> Vec<wgpu::ComputePipeline> {
        let module = create_module(
            device,
            format!(
                "{}\n{}\n{}",
                self.source_prefix,
                mesh_source(config),
                self.source_suffix
            ),
        );
        build_trace_pipelines(
            device,
            &module,
            &self.trace_layout,
            &self.trace_constants,
            variants,
            config,
        )
    }

    /// Median GPU ms of the trace pass (first tuner variant) with `config`.
    fn measure_config(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: tuner::ModuleConfig,
    ) -> f64 {
        // Measure the variant real frames usually settle on: primary hits from the
        // visibility raster when there is one (it prefers other stacks than tracing them).
        let variant = self
            .tuner
            .variants
            .iter()
            .position(|v| v.visibility_primary && self.visibility.is_some())
            .unwrap_or(0);
        let pipeline = self
            .trace_pipelines_for(device, config, &self.tuner.variants[variant..=variant])
            .remove(0);
        if self.tuner.variants[variant].visibility_primary
            && let Some(visibility) = &self.visibility
        {
            let (size, jitter) = self.visibility_frame(self.frame_number);
            visibility.set_frame(queue, jitter);
            let planes = GpuCamera::new(&self.latest_render_params.camera, size).frustum_planes();
            let mut encoder = device.create_command_encoder(&Default::default());
            visibility.raster(
                &mut encoder,
                &self.parameter_bind_group,
                &self.scene_bind_group,
                &planes,
            );
            queue.submit([encoder.finish()]);
        }
        // Frames are queued back to back and read once at the end: waiting after each
        // frame let the GPU idle, and power-managed GPUs (the Steam Deck) drop their
        // clock in the gaps, so timings measured the clock, not the configuration.
        const FRAMES: u32 = 24;
        let query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2 * FRAMES,
        });
        let bytes = 16 * u64::from(FRAMES);
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let (width, height) = self.latest_render_params.viewport_size;
        for frame in 0..FRAMES {
            // Keep tracing one sample per frame: never reach the sample target.
            self.render_progress.reset();
            self.prepare_frame(queue);
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&query, 2 * frame);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("calibration"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.images.trace, &[]);
                pass.set_bind_group(2, &self.parameter_bind_group, &[]);
                pass.set_bind_group(3, &self.scene_bind_group, &[]);
                if let Some(visibility) = &self.visibility {
                    pass.set_bind_group(6, visibility.trace_group(), &[]);
                }
                pass.dispatch_workgroups(
                    width.div_ceil(config.workgroup[0]),
                    height.div_ceil(config.workgroup[1]),
                    1,
                );
            }
            encoder.write_timestamp(&query, 2 * frame + 1);
            queue.submit([encoder.finish()]);
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(&query, 0..2 * FRAMES, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
        queue.submit([encoder.finish()]);
        read.slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mut times: Vec<f64> = {
            let data = read.slice(..).get_mapped_range().unwrap();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            // The first frames warm caches (and clocks).
            (4..FRAMES as usize)
                .map(|f| {
                    ticks[2 * f + 1].wrapping_sub(ticks[2 * f]) as f64
                        * f64::from(queue.get_timestamp_period())
                        / 1e6
                })
                .collect()
        };
        read.unmap();
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    }

    /// Sub-pixel jitter of the last traced frame when all pixels share one (upscaling,
    /// `--upscale-capture`), see `frame_jitter`.
    pub fn last_frame_jitter(&self) -> [f32; 2] {
        frame_jitter(self.last_frame_index)
    }

    /// Call after submitting a frame: starts reading back its timing for the auto-tuner.
    pub fn after_submit(&mut self, device: &wgpu::Device) {
        if let Some(profiler) = &mut self.profiler {
            profiler.collect(device);
        }
        if let Some(timer) = &mut self.timer {
            timer.after_submit();
            let _ = device.poll(wgpu::PollType::Poll);
        }
    }

    /// The trace variant the auto-tuner currently runs.
    pub fn trace_variant(&self) -> String {
        format!("{:?}", self.tuner.current())
    }

    /// Dispatch tracing, then filtering; presentation only reads the completed result.
    pub fn accumulate_frame(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        self.bloom.upload(queue);
        self.mark(encoder, "start");
        self.accumulate(queue, encoder);
        self.bloom.record(encoder);
        self.mark(encoder, "bloom");
        if let Some(profiler) = &mut self.profiler {
            profiler.finish(encoder);
        }
    }

    /// RT_PASS_TIMES: timestamp the end of a pass group.
    /// The visibility raster's size and jitter. Joint network: output resolution, with
    /// the render jitter scaled so that at integer ratios (render scale 0.5) every
    /// render-resolution sample lies exactly on a raster sample.
    fn visibility_frame(&self, frame: u32) -> ((u32, u32), [f32; 2]) {
        let render = self.latest_render_params.viewport_size;
        let jitter = frame_jitter(frame);
        if self.output_guides_pipeline.is_none() {
            return (render, jitter);
        }
        let out = self.output_size;
        let ratio = [
            out.0 as f32 / render.0 as f32,
            out.1 as f32 / render.1 as f32,
        ];
        (out, std::array::from_fn(|a| (jitter[a] * ratio[a]).fract()))
    }

    fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if let Some(profiler) = &mut self.profiler {
            profiler.mark(encoder, label);
        }
    }

    /// RT_PASS_TIMES: the average per-pass GPU times so far.
    pub fn pass_times(&self) -> Option<String> {
        self.profiler.as_ref().map(profiler::PassProfiler::report)
    }

    fn accumulate(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if self.temporal.is_some() || self.upscaler.is_some() {
            // Real-time mode: one fresh sample per frame; history lives in the denoiser
            // and/or the upscaler.
            self.render_progress.reset();
        }
        if let Some(temporal) = &mut self.temporal {
            temporal.prepare(queue);
        }
        if !self.filter_dirty
            && self.render_progress.accumulated_samples()
                >= self.latest_render_params.sampling.max_samples_per_pixel
        {
            // The cached image needs no uniform uploads or compute dispatches.
            // Keep random seeds aligned with redraws when rendering resumes.
            self.frame_number = self.frame_number.wrapping_add(1);
            return;
        }
        // The frame index the tracer sees (frameData.z); prepare_frame advances it.
        let frame_index = self.frame_number;
        self.last_frame_index = frame_index;
        let mut params = self.prepare_frame(queue);
        if let Some((x, y, _, _)) = self.trace_region {
            params.trace_origin = x | (y << 16);
            queue.write_buffer(
                self.sampling_parameter_buffer.handle(),
                0,
                bytemuck::cast_slice(&[params]),
            );
        }
        let (width, height) = self.latest_render_params.viewport_size;
        let groups = (width.div_ceil(8), height.div_ceil(8));
        for (variant, ms, switched) in self
            .timer
            .as_mut()
            .map(tuner::FrameTimer::collect)
            .unwrap_or_default()
        {
            self.tuner.record(variant, ms, switched);
            self.last_trace_ms = ms;
        }
        let variant = if params.num_samples_per_pixel != 0 {
            self.tuner.next()
        } else {
            0
        };
        let timing = if params.num_samples_per_pixel != 0 {
            let switched = self.tuner.switched(variant);
            self.timer
                .as_mut()
                .and_then(|timer| timer.begin(encoder, variant, switched))
        } else {
            None
        };
        if let Some(visibility) = self.visibility.as_ref().filter(|_| {
            // ReSTIR's first pass takes its primary hits from the raster too, and the
            // joint network's output guides.
            self.tuner.variants[variant].visibility_primary
                || (params.restir != 0 && self.restir_pipeline.is_some())
                || self.output_guides_pipeline.is_some()
        }) {
            // RT_VIS_RASTER_ONCE: keep the first frame's visibility buffer (static camera,
            // profiling the tracing pass alone).
            let once = std::env::var("RT_VIS_RASTER_ONCE").is_ok();
            if params.num_samples_per_pixel != 0 && (!once || !self.visibility_rastered) {
                self.visibility_rastered = true;
                let centred = crate::has_flag("--center-jitter") || once;
                let (size, jitter) = self.visibility_frame(frame_index);
                visibility.set_frame(queue, if centred { [0.5, 0.5] } else { jitter });
                let planes =
                    GpuCamera::new(&self.latest_render_params.camera, size).frustum_planes();
                visibility.raster(
                    encoder,
                    &self.parameter_bind_group,
                    &self.scene_bind_group,
                    &planes,
                );
            }
        }
        self.mark(encoder, "visibility raster");
        if params.num_samples_per_pixel != 0
            && params.restir != 0
            && let Some((initial, shared)) = &self.restir_pipeline
        {
            // One pixel per 2x2 block (raytracer.wgsl RESTIR_BLOCKS) unless RT_RESTIR_FULL.
            let (w, h) = if restir_blocks() {
                (width.div_ceil(2), height.div_ceil(2))
            } else {
                (width, height)
            };
            for (pipeline, label) in [(initial, "ReSTIR initial pass")]
                .into_iter()
                .chain(shared.as_ref().map(|p| (p, "ReSTIR shared pass")))
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.images.trace, &[]);
                pass.set_bind_group(2, &self.parameter_bind_group, &[]);
                pass.set_bind_group(3, &self.scene_bind_group, &[]);
                if let Some(visibility) = &self.visibility {
                    pass.set_bind_group(6, visibility.trace_group(), &[]);
                }
                pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
            }
        }
        self.mark(encoder, "restir");
        if params.num_samples_per_pixel != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("path tracing pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipelines[variant]);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.trace, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.set_bind_group(3, &self.scene_bind_group, &[]);
            if let Some(visibility) = &self.visibility {
                pass.set_bind_group(6, visibility.trace_group(), &[]);
            }
            if let Some(wavefront) = &self.wavefront {
                assert_eq!(
                    params.num_samples_per_pixel, 1,
                    "wavefront traces one sample per frame"
                );
                wavefront.record(
                    &mut pass,
                    (width, height),
                    trace_workgroup_size(),
                    self.latest_render_params.sampling.num_bounces,
                );
            } else {
                let (w, h) = self.trace_region.map_or((width, height), |r| (r.2, r.3));
                let [wx, wy] = self.module_config.workgroup;
                // GI blocks remap the thread grid into 2x2 planes: an even grid of
                // threads (a 64x1 workgroup over an odd height gets one more row).
                let even = |groups: u32, size: u32| {
                    if params.restir != 0 && gi_blocks() && groups * size % 2 == 1 {
                        groups + 1
                    } else {
                        groups
                    }
                };
                pass.dispatch_workgroups(even(w.div_ceil(wx), wx), even(h.div_ceil(wy), wy), 1);
            }
            self.filter_dirty = true;
        }
        if let (Some(timer), Some(slot)) = (&self.timer, timing) {
            timer.end(encoder, slot);
        }
        self.mark(encoder, "trace");
        if params.num_samples_per_pixel != 0
            && params.restir != 0
            && let Some(pipeline) = &self.gi_pipeline
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReSTIR GI pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.trace, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.set_bind_group(3, &self.scene_bind_group, &[]);
            if let Some(visibility) = &self.visibility {
                pass.set_bind_group(6, visibility.trace_group(), &[]);
            }
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.mark(encoder, "restir gi");
        if let Some(temporal) = &self.temporal {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temporal denoiser pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.trace, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.set_bind_group(3, &self.scene_bind_group, &[]);
            temporal.record(&mut pass, (width, height));
            drop(pass);
            self.mark(encoder, "denoiser");
            if self.latest_render_params.auto_exposure {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("exposure pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.exposure_pipeline);
                pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.images.filter, &[]);
                pass.set_bind_group(2, &self.parameter_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.filter_dirty = false;
        }
        self.mark(encoder, "exposure");
        if self.filter_dirty && !self.skip_filter {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("denoising pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.denoising_pipeline);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.images.filter, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
            if self.latest_render_params.auto_exposure {
                pass.set_pipeline(&self.exposure_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.filter_dirty = false;
        }
        self.mark(encoder, "spatial filter");
        // Joint network: this frame's guides at output resolution first.
        if params.num_samples_per_pixel != 0
            && let (Some(pipeline), Some(group), Some((w, h))) = (
                &self.output_guides_pipeline,
                &self.output_guides_group,
                self.upscaler.as_ref().map(upscale::Upscaler::output_size),
            )
        {
            queue.write_buffer(
                self.output_frame.handle(),
                0,
                bytemuck::cast_slice(&[
                    w,
                    h,
                    frame_index,
                    self.latest_render_params.exposure.to_bits(),
                ]),
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("output guides pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.set_bind_group(2, &self.parameter_bind_group, &[]);
            pass.set_bind_group(3, &self.scene_bind_group, &[]);
            if let Some(visibility) = &self.visibility {
                pass.set_bind_group(6, visibility.trace_group(), &[]);
            }
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        self.mark(encoder, "output guides");
        if let Some(upscaler) = &mut self.upscaler {
            if params.num_samples_per_pixel != 0 {
                upscaler.record(
                    queue,
                    encoder,
                    frame_jitter(frame_index),
                    frame_index,
                    self.latest_render_params.exposure.to_bits(),
                    self.profiler.as_mut(),
                );
            }
        }
        self.mark(encoder, "upscale");
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

    /// Fill the G-buffer by rasterized visibility (`--visibility`); call after a traced
    /// frame so frame data and camera are uploaded.
    pub fn encode_visibility(&self, encoder: &mut wgpu::CommandEncoder) {
        let visibility = self.visibility.as_ref().expect("--visibility");
        let planes = GpuCamera::new(
            &self.latest_render_params.camera,
            self.latest_render_params.viewport_size,
        )
        .frustum_planes();
        visibility.raster(
            encoder,
            &self.parameter_bind_group,
            &self.scene_bind_group,
            &planes,
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility resolve pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
        pass.set_bind_group(1, &self.images.trace, &[]);
        pass.set_bind_group(2, &self.parameter_bind_group, &[]);
        pass.set_bind_group(3, &self.scene_bind_group, &[]);
        visibility.resolve(&mut pass);
    }

    /// Median GPU ms of [raster, resolve] over 100 warmed repetitions.
    pub fn benchmark_visibility(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> [f64; 2] {
        let query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 3,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 24,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 24,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let visibility = self.visibility.as_ref().expect("--visibility");
        let planes = GpuCamera::new(
            &self.latest_render_params.camera,
            self.latest_render_params.viewport_size,
        )
        .frustum_planes();
        let mut times = [Vec::new(), Vec::new()];
        for i in 0..110 {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&query, 0);
            visibility.raster(
                &mut encoder,
                &self.parameter_bind_group,
                &self.scene_bind_group,
                &planes,
            );
            encoder.write_timestamp(&query, 1);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.vertex_uniform_bind_group, &[]);
                pass.set_bind_group(1, &self.images.trace, &[]);
                pass.set_bind_group(2, &self.parameter_bind_group, &[]);
                pass.set_bind_group(3, &self.scene_bind_group, &[]);
                visibility.resolve(&mut pass);
            }
            encoder.write_timestamp(&query, 2);
            encoder.resolve_query_set(&query, 0..3, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, 24);
            queue.submit([encoder.finish()]);
            read.slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            {
                let data = read.slice(..).get_mapped_range().unwrap();
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                let period = f64::from(queue.get_timestamp_period()) / 1e6;
                if i >= 10 {
                    times[0].push((ticks[1] - ticks[0]) as f64 * period);
                    times[1].push((ticks[2] - ticks[1]) as f64 * period);
                }
            }
            read.unmap();
        }
        times.map(|mut t| {
            t.sort_by(f64::total_cmp);
            t[t.len() / 2]
        })
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

    /// Move the scene's spheres; their previous positions feed motion vectors.
    /// Restarts accumulation, as the image content changed.
    pub fn set_spheres(&mut self, queue: &wgpu::Queue, spheres: &[Sphere]) {
        queue.write_buffer(&self.sphere_buffer, 0, bytemuck::cast_slice(spheres));
        queue.write_buffer(
            self.sphere_prev_buffer.handle(),
            0,
            bytemuck::cast_slice(&self.last_sphere_centers),
        );
        for (last, sphere) in self.last_sphere_centers.iter_mut().zip(spheres) {
            *last = sphere.center.into();
        }
        self.render_progress.reset();
        self.filter_dirty = true;
    }

    /// Move the scene's moving objects (object-to-world transforms, rigid with optional
    /// uniform scale); the previous call's transforms feed motion vectors. Restarts
    /// accumulation, as the image content changed.
    pub fn set_objects(&mut self, queue: &wgpu::Queue, transforms: &[glm::Mat4]) {
        assert_eq!(
            transforms.len(),
            self.object_bounds.len(),
            "one transform per moving object"
        );
        let previous = self
            .last_object_transforms
            .take()
            .unwrap_or_else(|| transforms.to_vec());
        queue.write_buffer(
            self.instance_buffer.handle(),
            0,
            bytemuck::cast_slice(&instance_words(transforms, &previous, &self.object_bounds)),
        );
        self.last_object_transforms = Some(transforms.to_vec());
        self.render_progress.reset();
        self.filter_dirty = true;
    }

    /// Trace only a rectangle of the frame (training captures); other pixels keep
    /// stale values. The wavefront tracer ignores this.
    pub fn set_trace_region(&mut self, region: Option<(u32, u32, u32, u32)>) {
        assert!(region.is_none_or(|r| r.0 < 65536 && r.1 < 65536));
        self.trace_region = region;
    }

    /// Skip the filter pass after tracing (captures that only read raw buffers).
    pub fn set_skip_filter(&mut self, skip: bool) {
        self.skip_filter = skip;
    }

    /// Restart progressive accumulation without changing parameters.
    pub fn reset_accumulation(&mut self) {
        self.render_progress.reset();
        self.filter_dirty = true;
    }

    /// Forget the ReSTIR DI/GI history (reservoirs and GI samples), e.g. when a capture
    /// starts a new sequence or trace region: stale samples would otherwise be reused.
    pub fn clear_history(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.images.reservoirs, 0, None);
        encoder.clear_buffer(&self.images.gi, 0, None);
        queue.submit([encoder.finish()]);
    }

    /// Primary-hit guides of the last traced frame, 12 floats per pixel.
    /// The primary-hit guides, 12 floats per pixel (without the emission slot).
    pub fn read_gbuffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let mut guides = read_buffer(device, queue, &self.images.gbuffer);
        guides.truncate(guides.len() / 16 * 12);
        guides
    }

    /// The displayed (filtered or denoised) radiance, 3 floats per pixel.
    pub fn read_filtered(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        read_buffer(device, queue, self.images.filtered.handle())
    }

    /// The image the display pass shows (before exposure and tonemapping): the
    /// upscaler's output, or the filtered render. 3 floats per pixel.
    pub fn read_display(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        self.upscaler.as_ref().map_or_else(
            || self.read_filtered(device, queue),
            |u| read_buffer(device, queue, u.output()),
        )
    }

    /// Output and render resolution.
    pub fn sizes(&self) -> ((u32, u32), (u32, u32)) {
        (self.output_size, self.latest_render_params.viewport_size)
    }

    /// Mean radiance per pixel accumulated so far, 3 floats per pixel.
    pub fn read_radiance(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let samples = self.render_progress.accumulated_samples().max(1) as f32;
        read_buffer(device, queue, self.images.accumulation.handle())
            .into_iter()
            .map(|v| v / samples)
            .collect()
    }

    fn prepare_frame(&mut self, queue: &wgpu::Queue) -> GpuSamplingParams {
        // Motion vectors compare against the camera traced in the previous frame.
        let camera = GpuCamera::new(
            &self.latest_render_params.camera,
            self.latest_render_params.viewport_size,
        );
        queue.write_buffer(
            self.prev_camera_buffer.handle(),
            0,
            bytemuck::bytes_of(&self.last_camera),
        );
        self.last_camera = camera;
        let mut gpu_sampling_params = self
            .render_progress
            .next_frame(&self.latest_render_params.sampling);
        if self.latest_render_params.sampling.restir {
            self.restir_frames = self.restir_frames.wrapping_add(1).max(1);
            gpu_sampling_params.restir = self.restir_frames;
        }
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
        let display = self
            .upscaler
            .as_ref()
            .map_or(&self.images.display, |u| &u.display);
        render_pass.set_bind_group(1, display, &[]);
        render_pass.set_bind_group(2, &self.bloom.display_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..VERTICES.len() as u32, 0..1);
    }

    pub fn set_render_params(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_params: &RenderParams,
    ) -> Result<(), RenderParamsValidationError> {
        // The caller's viewport is the output resolution (see Raytracer::new).
        let output_size = render_params.viewport_size;
        let scaled_params = RenderParams {
            viewport_size: upscale::render_size(output_size),
            ..*render_params
        };
        let render_params = &scaled_params;
        // Display settings apply without restarting accumulation.
        self.bloom
            .set(render_params.tonemapper, render_params.bloom);
        self.latest_render_params.tonemapper = render_params.tonemapper;
        self.latest_render_params.bloom = render_params.bloom;
        if *render_params == self.latest_render_params && output_size == self.output_size {
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
        if render_params.auto_exposure != self.latest_render_params.auto_exposure {
            set_auto_exposure(queue, &self.images, render_params.auto_exposure);
            // Recompute the key luminance from the current image.
            self.filter_dirty |= render_params.auto_exposure;
        }
        let mut previous = self.latest_render_params;
        previous.exposure = render_params.exposure;
        previous.auto_exposure = render_params.auto_exposure;
        if let Some(temporal) = &mut self.temporal {
            // Exposure alone does not restart the still-view convergence.
            if previous != *render_params || output_size != self.output_size {
                temporal.moved();
            }
        }
        if previous == *render_params {
            self.latest_render_params = *render_params;
            return Ok(());
        }
        let buffer_size =
            image_buffer_size(device, render_params.viewport_size, self.images.history)?;
        let sky_state = if render_params.sky != self.latest_render_params.sky {
            Some(render_params.sky.to_sky_state()?)
        } else {
            None
        };
        self.images
            .resize(device, &self.frame_data_buffer, buffer_size);
        if let Some(wavefront) = &mut self.wavefront {
            wavefront.resize(
                device,
                render_params.viewport_size.0 * render_params.viewport_size.1,
            );
        }
        if let Some(visibility) = &mut self.visibility {
            let joint = self.output_guides_pipeline.is_some();
            if (!joint && render_params.viewport_size != self.latest_render_params.viewport_size)
                || (joint && output_size != self.output_size)
            {
                visibility.resize(
                    device,
                    if joint {
                        output_size
                    } else {
                        render_params.viewport_size
                    },
                );
            }
        }
        if let Some(upscaler) = &mut self.upscaler {
            if render_params.viewport_size != self.latest_render_params.viewport_size
                || output_size != self.output_size
            {
                upscaler.resize(
                    device,
                    self.images.filtered.handle(),
                    &self.images.gbuffer,
                    &self.images.display_layout,
                    self.images.exposure.handle(),
                    output_size,
                    render_params.viewport_size,
                );
            }
        }
        if render_params.viewport_size != self.latest_render_params.viewport_size
            || output_size != self.output_size
        {
            let (source, size) = self.upscaler.as_ref().map_or(
                (self.images.filtered.handle(), render_params.viewport_size),
                |u| (u.output(), output_size),
            );
            self.bloom.resize(device, source, size);
        }
        // The image buffers or the upscaler's may have been reallocated.
        self.output_guides_group = self
            .upscaler
            .as_ref()
            .and_then(|u| output_guides_group(device, &self.images, &self.output_frame, u));
        self.output_size = output_size;
        if let Some(temporal) = &mut self.temporal {
            if render_params.viewport_size != self.latest_render_params.viewport_size {
                temporal.resize(
                    device,
                    self.images.filtered.handle(),
                    render_params.viewport_size.0 * render_params.viewport_size.1,
                );
            }
        }

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

/// Bytes of the 12-byte-per-pixel image buffers; an error when the largest per-pixel
/// buffer (guides or ReSTIR history) would exceed the device's buffer limits.
fn image_buffer_size(
    device: &wgpu::Device,
    (width, height): (u32, u32),
    history: image_buffers::History,
) -> Result<u64, RenderParamsValidationError> {
    let limits = device.limits();
    let max_bytes = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size));
    let pixels = u64::from(width) * u64::from(height);
    pixels
        .checked_mul(history.max_bytes_per_pixel())
        .filter(|&largest| largest <= max_bytes)
        .map(|_| pixels * std::mem::size_of::<[f32; 3]>() as u64)
        .ok_or(RenderParamsValidationError::ViewportBufferTooLarge {
            width,
            height,
            max_bytes,
        })
}

/// csOutputGuides' trace-images group: output-sized frame data, the output guides in
/// place of the G-buffer (joint network only).
fn output_guides_group(
    device: &wgpu::Device,
    images: &image_buffers::ImageBuffers,
    frame: &UniformBuffer,
    upscaler: &upscale::Upscaler,
) -> Option<wgpu::BindGroup> {
    let guides = upscaler.out_guides()?;
    fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }
    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("output guides bindings"),
        layout: &images.trace_layout,
        entries: &[
            frame.binding(),
            images.accumulation.binding(),
            entry(7, guides),
            entry(8, &images.reservoirs),
            entry(9, &images.gi),
        ],
    }))
}

fn set_auto_exposure(queue: &wgpu::Queue, images: &image_buffers::ImageBuffers, enabled: bool) {
    queue.write_buffer(
        images.exposure.handle(),
        4,
        bytemuck::bytes_of(&f32::from(u8::from(enabled))),
    );
}

/// Mesh traversal source with experiment substitutions (RT_STACK=N stack entries).
/// Trace pipelines (one per tuner variant) from `module` for a module configuration.
fn build_trace_pipelines(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    trace_constants: &[(&'static str, f64)],
    variants: &[tuner::Variant],
    config: tuner::ModuleConfig,
) -> Vec<wgpu::ComputePipeline> {
    variants
        .iter()
        .map(|variant| {
            let constants: Vec<(&str, f64)> = trace_constants
                .iter()
                .map(|&(name, value)| match name {
                    "RAY_CONES" => (name, f64::from(variant.cones != tuner::Cones::Off)),
                    "CONES_PRIMARY_ONLY" => {
                        (name, f64::from(variant.cones == tuner::Cones::Primary))
                    }
                    "VIS_PRIMARY" => (name, f64::from(variant.visibility_primary)),
                    "TRACE_WIDTH" => (name, f64::from(config.workgroup[0])),
                    "HW_CLOSEST" => (name, f64::from(config.hw.closest())),
                    "HW_SHADOWS" => (name, f64::from(config.hw.shadows())),
                    "TRACE_HEIGHT" => (name, f64::from(config.workgroup[1])),
                    _ => (name, value),
                })
                .collect();
            // RT_SPIRV_TRACE=FILE.spv: the trace kernel from raw SPIR-V (constants already
            // specialized), for comparing naga's output against other compilers/optimizers.
            if let Ok(path) = std::env::var("RT_SPIRV_TRACE") {
                let bytes = std::fs::read(&path).expect("RT_SPIRV_TRACE file");
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let module = unsafe {
                    device.create_shader_module_passthrough(
                        wgpu::ShaderModuleDescriptorPassthrough {
                            label: Some("trace kernel (SPIR-V)"),
                            entry_points: std::borrow::Cow::Owned(vec![
                                wgpu::PassthroughShaderEntryPoint {
                                    name: std::borrow::Cow::Borrowed("csAccumulate"),
                                    workgroup_size: (config.workgroup[0], config.workgroup[1], 1),
                                },
                            ]),
                            spirv: Some(std::borrow::Cow::Owned(words)),
                            ..Default::default()
                        },
                    )
                };
                eprintln!("Trace kernel from {path}");
                return device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("path tracing compute (SPIR-V)"),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some("csAccumulate"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            }
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("path tracing compute"),
                layout: Some(layout),
                module,
                entry_point: Some("csAccumulate"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        })
        .collect()
}

/// RT_* stack experiments choose the traversal stack themselves (no calibration).
fn stack_override() -> bool {
    [
        "RT_STACK",
        "RT_QUAD_STACK",
        "RT_PRIVATE_STACK",
        "RT_QUAD_SLM",
    ]
    .iter()
    .any(|v| std::env::var(v).is_ok())
}

/// The trace kernel's module configuration before calibration.
fn default_module_config() -> tuner::ModuleConfig {
    let workgroup = trace_workgroup_size();
    let stack = (mesh::quad_bvh() && !stack_override() && workgroup[0] * workgroup[1] == 64)
        .then_some(tuner::Stack::Shared32);
    // RT_HW_MODE=closest|shadows|compute fixes the hardware query split (experiments).
    let hw = match std::env::var("RT_HW_MODE").as_deref() {
        Ok("shadows") => tuner::HwRays::Shadows,
        Ok("compute") => tuner::HwRays::Compute,
        _ => tuner::HwRays::All,
    };
    tuner::ModuleConfig {
        stack,
        workgroup,
        hw,
    }
}

fn mesh_source(config: tuner::ModuleConfig) -> String {
    use tuner::Stack;
    let mut source = include_str!("mesh.wgsl").to_owned();
    if mesh::hw_rt() {
        source = source.replace(
            "fn meshHwCandidate(ray: Ray, limit: f32, anyHit: bool, outBest: ptr<function, MeshCandidate>) -> bool { return false; }",
            "",
        );
    }
    let mut source = match std::env::var("RT_STACK") {
        Ok(n) => source.replace("array<u32,64>", &format!("array<u32,{n}>")),
        Err(_) => source,
    };
    if let Ok(n) = std::env::var("RT_QUAD_STACK") {
        source = source.replace("const QUAD_STACK=64u;", &format!("const QUAD_STACK={n}u;"));
    }
    let lanes = config.workgroup[0] * config.workgroup[1];
    // 4-wide traversal stack: workgroup memory (exact, 32 entries; 64-lane workgroups),
    // registers (16 entries) or a module-scope array (64 entries), as configured.
    // Compressed 8-wide BVH: its stack takes the workgroup memory (16 entries, 8 KB) and
    // the (unused) 4-wide traversal keeps a private one.
    if mesh::cw_bvh() && config.stack == Some(Stack::Shared32) && lanes == 64 {
        let start = source.find("fn meshCwCandidate(").unwrap();
        let end = start + source[start..].find("\n}\n").unwrap();
        let body = source[start..end]
            .replace("var cwEntries:array<vec2<u32>,CW_STACK>;", "")
            .replace("cwEntries[depth]", "cwShared[depth*64u+quadLane]");
        source.replace_range(start..end, &body);
        source.push_str("var<workgroup> cwShared: array<vec2<u32>,1024>;\n");
        return source;
    }
    let slm = std::env::var("RT_QUAD_SLM").ok().or_else(|| {
        (config.stack == Some(Stack::Shared32))
            .then(|| if mesh::oct_bvh() { "64" } else { "32" }.to_owned())
    });
    if let Some(n) = slm {
        assert_eq!(
            lanes, 64,
            "the shared-memory traversal stack needs 64-invocation workgroups"
        );
        let start = source.find("fn meshQuadCandidate(").unwrap();
        let end = start + source[start..].find("\n}\n").unwrap();
        let body = source[start..end]
            .replace("var stack:array<u32,QUAD_STACK>;", "")
            .replace("stack[size]", "quadStack[size*64u+quadLane]");
        source.replace_range(start..end, &body);
        source = source.replace("const QUAD_STACK=64u;", &format!("const QUAD_STACK={n}u;"));
        source.push_str(&format!(
            "var<workgroup> quadStack: array<u32,{}>;\n",
            64 * n.parse::<u32>().unwrap()
        ));
    }
    if config.stack == Some(Stack::Registers16) {
        source = source.replace("const QUAD_STACK=64u;", "const QUAD_STACK=16u;");
    }
    // Module-scope stack: zero-initialized once per invocation, not per ray.
    if std::env::var("RT_PRIVATE_STACK").is_ok() || config.stack == Some(Stack::Private64) {
        source = source.replace("var stack:array<u32,64>;", "");
        source = source.replace("var stack:array<u32,QUAD_STACK>;", "");
        source.push_str("var<private> stack:array<u32,64>;\n");
    }
    source
}

/// The renderer's shader module (loop guards off unless `--loop-guards`, see Raytracer::new).
fn create_module(device: &wgpu::Device, source: String) -> wgpu::ShaderModule {
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some("raytracer shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    };
    if crate::has_flag("--loop-guards") {
        device.create_shader_module(descriptor)
    } else {
        let checks = wgpu::ShaderRuntimeChecks {
            force_loop_bounding: false,
            ..wgpu::ShaderRuntimeChecks::checked()
        };
        unsafe { device.create_shader_module_trusted(descriptor, checks) }
    }
}

/// Primary-hit guide output (`--gbuffer`; implied by capture and temporal denoising).
/// Lamp candidates per light sample: `--lamp-candidates FIRST[,DEEP]` (at the primary
/// hit, and at deeper hits) or the environment variable; more cost time, not rays.
fn lamp_candidates(index: usize, variable: &str, default: f64) -> f64 {
    crate::argument_value("--lamp-candidates")
        .and_then(|v| {
            v.split(',')
                .nth(index)
                .map(|n| n.parse().expect("--lamp-candidates N[,M]"))
        })
        .or_else(|| std::env::var(variable).ok().map(|v| v.parse().unwrap()))
        .unwrap_or(default)
}

pub fn gbuffer_enabled() -> bool {
    crate::has_flag("--gbuffer")
        || crate::has_flag("--capture")
        || crate::has_flag("--reference-output")
        || crate::has_flag("--temporal-model")
        || crate::has_flag("--restir")
        || upscale::joint_requested()
        || upscale::render_scale() < 1.
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let size = source.size();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
    queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&data).to_vec()
}

/// Parameter and scene groups: tracing stages plus the visibility raster's.
fn scene_stages(visibility: bool) -> wgpu::ShaderStages {
    tracing_stages() | visibility::stages(visibility)
}

/// Auto-tuning is on unless disabled, in captures (reproducible training data) and tests.
/// ReSTIR DI's first pass for one pixel per 2x2 block (`--restir-full` or
/// RT_RESTIR_FULL: every pixel). Night Bistro on the Steam Deck: 60.2 -> 45.3 ms per
/// frame, inputs 0.2 dB (per pixel) / 1 dB (4x4 averages) noisier.
/// ReSTIR GI's bounce traced by one pixel per 2x2 block, the rest from reuse
/// (raytracer.wgsl GI_BLOCKS), unless `--restir-full` or RT_GI_FULL. Steam Deck, 0.67
/// render scale, Medium: night Bistro 135 -> 82 ms per frame, Sponza 51 -> 32 ms
/// (no GI: 76 and 30 ms); Sponza 4x4-averaged PSNR 25.8 -> 24.6 dB (no GI: 20.8).
fn gi_blocks() -> bool {
    std::env::var_os("RT_GI_FULL").is_none() && !crate::has_flag("--restir-full")
}

/// ReSTIR DI's spatial merge and shadow ray once per 2x2 block (raytracer.wgsl
/// DI_SHARED; `--di-shared` or RT_DI_SHARED, with the blocks). Steam Deck, night Bistro
/// Medium: 75.7 -> 60.1 ms per frame; captures: 4x4 PSNR -0.5 to -0.6 dB and visibly
/// 2x2-correlated noise, which the current denoisers were not trained on. Opt-in.
fn di_shared() -> bool {
    restir_blocks()
        && (crate::has_flag("--di-shared") || std::env::var_os("RT_DI_SHARED").is_some())
}

fn restir_blocks() -> bool {
    std::env::var_os("RT_RESTIR_FULL").is_none() && !crate::has_flag("--restir-full")
}

fn auto_tune() -> bool {
    !crate::has_flag("--no-auto-tune")
        && !crate::has_flag("--capture")
        && !crate::has_flag("--reference-output")
        && !cfg!(test)
}

/// Trace-pass variants worth trying for this scene and device. Explicit flags force a
/// setting: --ray-cones / --no-ray-cones, --visibility-primary / --no-visibility-primary.
fn trace_variants(device: &wgpu::Device, scene: &Scene) -> Vec<tuner::Variant> {
    let auto = auto_tune() && !crate::has_flag("--wavefront");
    // Textured scenes: mips at the primary hit were never slower than none (1-2% faster,
    // 1.15x on textured Bistro); cones along the whole path win when bounce rays also
    // thrash the texture cache (1.37x on Bistro) and lose 1-6% elsewhere.
    use tuner::Cones;
    let cones: &[Cones] = if crate::has_flag("--no-ray-cones") || scene.mesh.textures.is_empty() {
        &[Cones::Off]
    } else if crate::has_flag("--cones-primary-only") {
        &[Cones::Primary]
    } else if crate::has_flag("--ray-cones") {
        &[Cones::All]
    } else if !auto {
        &[Cones::Off]
    } else {
        &[Cones::Primary, Cones::All]
    };
    // The visibility hook serves the 4-wide BVH query and hardware closest-hit queries.
    let available = device.features().contains(wgpu::Features::PRIMITIVE_INDEX)
        && !scene.mesh.triangles.is_empty()
        && mesh::quad_bvh();
    let primary: &[bool] = if crate::has_flag("--visibility-primary") {
        assert!(
            available,
            "--visibility-primary needs primitive_index, a mesh, --quad-bvh and no --hw-rt"
        );
        &[true]
    } else if !auto || crate::has_flag("--no-visibility-primary") || !available {
        &[false]
    } else {
        &[false, true]
    };
    cones
        .iter()
        .flat_map(|&cones| {
            primary
                .iter()
                .map(move |&visibility_primary| tuner::Variant {
                    cones,
                    visibility_primary,
                })
        })
        .collect()
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
    pub fn center(&self) -> glm::Vec3 {
        self.center.xyz()
    }
    pub fn set_center(&mut self, center: glm::Vec3) {
        self.center = glm::vec4(center.x, center.y, center.z, 0.);
    }
    pub fn material_index(&self) -> u32 {
        self.material_idx
    }

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
    /// Linear multiplier before tone mapping; compensation when `auto_exposure` is set.
    pub exposure: f32,
    /// Normalize the image's log-average luminance to middle grey before `exposure`.
    pub auto_exposure: bool,
    /// Display transform (a display setting: changing it keeps accumulated samples).
    pub tonemapper: Tonemapper,
    /// Bloom strength, 0 = off (a display setting).
    pub bloom: f32,
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
    /// A sun disk (sampled directly on mesh surfaces) next to the atmosphere.
    pub sun: bool,
    /// Faint moonlit atmosphere, no sun: emissive materials light the scene.
    pub night: bool,
}

impl Default for SkyParams {
    fn default() -> Self {
        Self {
            azimuth_degrees: 0_f32,
            zenith_degrees: 85_f32,
            turbidity: 4_f32,
            albedo: [1_f32; 3],
            sun: true,
            night: false,
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
            elevation: (FRAC_PI_2 - zenith).max(0.),
            turbidity: self.turbidity,
            albedo: self.albedo,
        })?;

        let (params_data, mut radiance_data) = state.raw();
        if self.night {
            for (radiance, scale) in radiance_data.iter_mut().zip(NIGHT_SKY) {
                *radiance *= scale;
            }
        }

        let sampling = sky_sampling_table(&params_data, &radiance_data, sun_direction);
        let sun = sun_radiance(self);
        Ok(GpuSkyState {
            sampling,
            params: params_data,
            radiances: radiance_data,
            _padding: [0_u32, 2],
            sun_direction: [
                sun_direction[0],
                sun_direction[1],
                sun_direction[2],
                SUN_ANGULAR_RADIUS.cos(),
            ],
            sun_radiance: [sun[0], sun[1], sun[2], SUN_SOLID_ANGLE],
        })
    }
}

/// The sun's angular radius (radians) and the solid angle of its disk.
const SUN_ANGULAR_RADIUS: f32 = 0.2666 * std::f32::consts::PI / 180.;
pub(crate) const SUN_SOLID_ANGLE: f32 = 6.806e-5; // 2π(1 − cos SUN_ANGULAR_RADIUS)
/// Night atmosphere: the day sky dimmed to faint, bluish moonlight.
const NIGHT_SKY: [f32; 3] = [0.00006, 0.0001, 0.00016];

/// Direct-beam transmittance of the atmosphere for R, G and B (680, 550, 440 nm):
/// Rayleigh and turbidity-dependent aerosol optical depths (Preetham et al. 1999)
/// times the Kasten–Young air mass for this sun zenith angle.
fn sun_transmittance(zenith_degrees: f32, turbidity: f32) -> [f64; 3] {
    let z = f64::from(zenith_degrees);
    let air_mass = 1. / (z.to_radians().cos() + 0.50572 * (96.07995 - z).powf(-1.6364));
    let beta = 0.04608 * f64::from(turbidity) - 0.04586;
    [0.68_f64, 0.55, 0.44].map(|micrometres| {
        let depth = 0.008735 * micrometres.powf(-4.08) + beta * micrometres.powf(-1.3);
        (-depth * air_mass).exp()
    })
}

/// Sun disk radiance (linear RGB, the sky model's units). Scaled so that at 45°
/// elevation and turbidity 4 the sun gives 4× the sky's horizontal illuminance;
/// other elevations and turbidities follow the atmosphere's transmittance.
pub(crate) fn sun_radiance(params: &SkyParams) -> [f32; 3] {
    if !params.sun || params.night || params.zenith_degrees >= 90. {
        return [0.; 3];
    }
    static SCALE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *SCALE.get_or_init(|| {
        let reference = SkyParams {
            zenith_degrees: 45.,
            turbidity: 4.,
            ..SkyParams::default()
        };
        let t = sun_transmittance(45., 4.);
        let luminance = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
        4. * sky_illuminance(&reference)
            / (luminance * f64::from(SUN_SOLID_ANGLE) * std::f64::consts::FRAC_1_SQRT_2)
    });
    sun_transmittance(params.zenith_degrees, params.turbidity).map(|t| (scale * t) as f32)
}

/// Horizontal illuminance (luminance-weighted irradiance) from the atmosphere alone.
pub(crate) fn sky_illuminance(params: &SkyParams) -> f64 {
    let zenith = Angle::degrees(params.zenith_degrees.min(90.)).as_radians();
    let azimuth = Angle::degrees(params.azimuth_degrees).as_radians();
    let sun = [
        zenith.sin() * azimuth.cos(),
        zenith.cos(),
        zenith.sin() * azimuth.sin(),
    ];
    let state = hw_skymodel::rgb::SkyState::new(&hw_skymodel::rgb::SkyParams {
        elevation: FRAC_PI_2 - zenith,
        turbidity: params.turbidity,
        albedo: params.albedo,
    })
    .expect("valid sky parameters");
    let (sky, radiances) = state.raw();
    // Equal-area cells: y uniform in (0, 1], φ uniform.
    let (rows, columns) = (128, 128);
    let cell = std::f64::consts::TAU / f64::from(rows * columns);
    let mut sum = 0.;
    for row in 0..rows {
        let y = (f64::from(row) + 0.5) / f64::from(rows);
        for column in 0..columns {
            let phi = std::f64::consts::TAU * (f64::from(column) + 0.5) / f64::from(columns);
            let r = (1. - y * y).sqrt();
            let direction = [r * phi.cos(), y, r * phi.sin()];
            let rgb = atmosphere_rgb(&sky, &radiances, direction, sun);
            sum += (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) * y * cell;
        }
    }
    sum
}

/// The analytic sky (same formula as WGSL `atmosphereRadiance`, mirrored below the horizon).
fn atmosphere_rgb(
    params: &[f32; 27],
    radiances: &[f32; 3],
    direction: [f64; 3],
    sun: [f32; 3],
) -> [f64; 3] {
    let y = direction[1];
    let cg = (direction[0] * f64::from(sun[0])
        + y * f64::from(sun[1])
        + direction[2] * f64::from(sun[2]))
    .clamp(-1.0, 1.0);
    let gamma = cg.acos();
    std::array::from_fn(|channel| {
        let p: [f64; 9] = std::array::from_fn(|i| f64::from(params[channel * 9 + i]));
        (f64::from(radiances[channel])
            * (1.0 + p[0] * (p[1] / (y.abs() + 0.01)).exp())
            * (p[2]
                + p[3] * (p[4] * gamma).exp()
                + p[5] * cg * cg
                + p[6] * (1.0 + cg * cg) / (1.0 + p[8] * p[8] - 2.0 * p[8] * cg).powf(1.5)
                + p[7] * y.abs().sqrt()))
        .max(0.0)
    })
}

#[derive(Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub max_samples_per_pixel: u32,
    pub num_samples_per_pixel: u32,
    pub num_bounces: u32,
    pub denoising_enabled: bool,
    pub denoising_radius: u32, // 0 means effectively disabled, 1 for 3x3, 2 for 5x5 etc.
    pub denoising_sigma_color: f32,
    /// Real-time frames: reuse lamp samples across pixels and frames (ReSTIR DI at the
    /// primary surface). Biased, so off for progressive accumulation and references.
    pub restir: bool,
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
            restir: false,
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
                trace_origin: 0,
                restir: 0,
                _pad: [0; 3],
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
                trace_origin: 0,
                restir: 0,
                _pad: [0; 3],
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
                trace_origin: 0,
                restir: 0,
                _pad: [0; 3],
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
    /// Inward-facing frustum side planes (normal, offset): dot(n, p) + d >= 0 inside.
    pub fn frustum_planes(&self) -> [[f32; 4]; 4] {
        let corners = [
            self.lower_left_corner,
            self.lower_left_corner + self.horizontal,
            self.lower_left_corner + self.horizontal + self.vertical,
            self.lower_left_corner + self.vertical,
        ];
        let center = self.lower_left_corner + 0.5 * (self.horizontal + self.vertical);
        std::array::from_fn(|i| {
            let mut n = glm::cross(&(corners[i] - self.eye), &(corners[(i + 1) % 4] - self.eye));
            if glm::dot(&n, &(center - self.eye)) < 0. {
                n = -n;
            }
            [n.x, n.y, n.z, -glm::dot(&n, &self.eye)]
        })
    }

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
    sun_direction: [f32; 4],    // 128 byte offset; w = cos(sun angular radius)
    sun_radiance: [f32; 4],     // 144 byte offset; rgb (0 = no sun), w = solid angle
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
        let rgb = atmosphere_rgb(
            params,
            radiances,
            [r * phi.cos(), y, r * phi.sin()],
            [sun[0], sun[1], sun[2]],
        );
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
    /// Origin of the traced region, x | y << 16 (see `Raytracer::set_trace_region`).
    trace_origin: u32,
    /// 0: no ReSTIR; else the ReSTIR frame count (its low bit picks the reservoir copy).
    restir: u32,
    _pad: [u32; 3],
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
