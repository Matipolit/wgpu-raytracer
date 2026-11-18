mod mesh_cases;
use super::*;

// Run explicitly with `cargo test --release -- --ignored`; these require a GPU.
fn gpu() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU regression tests require an available adapter");
        let (required_features, required_limits) = gpu_textures::device_requirements(&adapter);
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits,
                experimental_features: crate::raytracer::gpu_textures::experimental_features(),
                ..Default::default()
            })
            .await
            .unwrap()
    })
}

fn config((width, height): (u32, u32)) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

fn params() -> RenderParams {
    RenderParams {
        camera: Camera {
            eye_pos: glm::vec3(0., 0., 0.),
            eye_dir: glm::vec3(0., 0., -1.),
            up: glm::vec3(0., 1., 0.),
            vfov: Angle::degrees(30.),
            aperture: 0.,
            focus_distance: 10.,
        },
        exposure: 0.246,
        auto_exposure: false,
        sky: SkyParams {
            sun: false,
            ..SkyParams::default()
        },
        // The fragment reference tonemaps with the original curve, no bloom.
        tonemapper: crate::raytracer::Tonemapper::Hable,
        bloom: 0.,
        sampling: SamplingParams {
            max_samples_per_pixel: 8,
            ..Default::default()
        },
        viewport_size: (23, 17),
    }
}

fn uniform_scene() -> Scene {
    Scene {
        spheres: vec![Sphere::new(glm::vec3(0., 0., 0.), 50., 0)],
        materials: vec![Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(1., 1., 1.)),
        }],
        ..Default::default()
    }
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, tracer: &mut Raytracer) -> Vec<u8> {
    render_with_ui(device, queue, tracer, None)
}

fn render_with_ui(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tracer: &mut Raytracer,
    ui: Option<&mut crate::gui::Gui>,
) -> Vec<u8> {
    render_with_reference(device, queue, tracer, ui, None)
}

fn render_with_reference(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tracer: &mut Raytracer,
    ui: Option<&mut crate::gui::Gui>,
    reference: Option<&FragmentReference>,
) -> Vec<u8> {
    let (width, height) = tracer.latest_render_params.viewport_size;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("regression test output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config((width, height)).format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let stride = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("regression test readback"),
        size: u64::from(stride) * u64::from(height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut ui = ui;
    let ui_commands = ui.as_deref_mut().map(|gui| {
        let mut params = tracer.latest_render_params;
        let output = gui.run(
            egui::RawInput {
                // Advance a controlled clock so entrance animations settle.
                time: Some(f64::from(tracer.frame_number)),
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                )),
                ..Default::default()
            },
            &mut params,
            &mut crate::FlyCameraController::default(),
            60.,
            tracer.progress(),
        );
        gui.prepare(device, queue, &mut encoder, output, [width, height])
    });
    if let Some(reference) = reference {
        reference.accumulate(queue, &mut encoder, &view, tracer);
    } else {
        tracer.accumulate_frame(queue, &mut encoder);
    }
    {
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("regression test display"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            })
            .forget_lifetime();
        if let Some(reference) = reference {
            reference.display(&mut pass, tracer);
        } else {
            tracer.render_frame(&mut pass);
        }
        if let Some(gui) = &ui {
            gui.render(&mut pass);
        }
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    queue.submit(
        ui_commands
            .unwrap_or_default()
            .into_iter()
            .chain([encoder.finish()]),
    );
    if let Some(gui) = &mut ui {
        gui.free_textures();
    }
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = readback.slice(..).get_mapped_range().unwrap();
    data.chunks(stride as usize)
        .flat_map(|row| row[..(width * 4) as usize].iter().copied())
        .collect()
}

#[test]
#[ignore = "requires a GPU"]
fn textured_scene_and_egui_share_the_display_pass() {
    let (device, queue) = gpu();
    let mut params = params();
    params.viewport_size = (640, 480);
    params.sampling.num_samples_per_pixel = 8;
    let surface_config = config(params.viewport_size);
    let mut tracer =
        Raytracer::new(&device, &queue, &surface_config, &crate::scene(), &params).unwrap();
    let scene_only = render(&device, &queue, &mut tracer);
    assert!(scene_only.chunks_exact(4).any(|pixel| pixel[..3] != [0; 3]));

    let mut gui = crate::gui::Gui::new(&device, surface_config.format);
    // Egui measures a newly opened window before making it visible.
    render_with_ui(&device, &queue, &mut tracer, Some(&mut gui));
    let with_ui = render_with_ui(&device, &queue, &mut tracer, Some(&mut gui));
    assert!(with_ui != scene_only, "egui must paint over the scene");
    // Painting UI must leave the scene outside the panel unchanged.
    let last_row = ((params.viewport_size.1 - 1) * params.viewport_size.0 * 4) as usize;
    assert!(
        with_ui[last_row..] == scene_only[last_row..],
        "UI changed pixels outside its panel"
    );
    assert_eq!(tracer.progress(), 1.);
    let mut reused = false;
    for _ in 0..8 {
        let previous_pass = gui.context.cumulative_pass_nr();
        render_with_ui(&device, &queue, &mut tracer, Some(&mut gui));
        if gui.context.cumulative_pass_nr() == previous_pass {
            reused = true;
            break;
        }
    }
    assert!(reused, "idle UI should reuse its painted geometry");
    let output = gui.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(640., 480.),
            )),
            events: vec![egui::Event::PointerMoved(egui::pos2(500., 450.))],
            ..Default::default()
        },
        &mut params,
        &mut crate::FlyCameraController::default(),
        60.,
        tracer.progress(),
    );
    assert!(output.is_some(), "input must bypass the idle UI cache");
    let mut encoder = device.create_command_encoder(&Default::default());
    let commands = gui.prepare(&device, &queue, &mut encoder, output, [640, 480]);
    queue.submit(commands.into_iter().chain([encoder.finish()]));
    gui.free_textures();
    params.viewport_size = (800, 600);
    tracer.set_render_params(&device, &queue, &params).unwrap();
    let resized = render_with_ui(&device, &queue, &mut tracer, Some(&mut gui));
    assert_eq!(resized.len(), 800 * 600 * 4);
}

#[test]
#[ignore = "requires a GPU"]
fn resizing_and_accumulation_keep_every_pixel_at_the_same_brightness() {
    let (device, queue) = gpu();
    let mut params = params();
    let mut tracer = Raytracer::new(
        &device,
        &queue,
        &config(params.viewport_size),
        &uniform_scene(),
        &params,
    )
    .unwrap();
    let reference = render(&device, &queue, &mut tracer)[..4].to_vec();
    assert!(reference[0] > 0);

    // Grow far beyond the startup allocation, shrink, and change aspect ratio
    // without changing the pixel count (which reuses storage but must reset it).
    for size in [(2560, 1600), (31, 19), (19, 31), (23, 17)] {
        params.viewport_size = size;
        tracer.set_render_params(&device, &queue, &params).unwrap();
        assert_eq!(tracer.progress(), 0.);
        assert_eq!(
            tracer.images.accumulation.handle().size(),
            u64::from(size.0) * u64::from(size.1) * 12
        );
        for _ in 0..10 {
            let pixels = render(&device, &queue, &mut tracer);
            assert!(pixels.chunks_exact(4).all(|pixel| pixel == reference));
        }
        assert_eq!(tracer.progress(), 1.);
    }

    params.viewport_size = (u32::MAX, u32::MAX);
    assert!(matches!(
        tracer.set_render_params(&device, &queue, &params),
        Err(RenderParamsValidationError::ViewportBufferTooLarge { .. })
    ));
    assert_eq!(tracer.latest_render_params.viewport_size, (23, 17));
    assert!(
        render(&device, &queue, &mut tracer)
            .chunks_exact(4)
            .all(|pixel| pixel == reference)
    );
}

#[test]
#[ignore = "requires a GPU"]
fn denoiser_enable_radius_and_sigma_control_the_display() {
    let (device, queue) = gpu();
    let mut params = params();
    params.viewport_size = (63, 47);
    params.sampling.num_samples_per_pixel = 8;
    params.sampling.denoising_enabled = false;
    let mut scene = uniform_scene();
    scene.materials.push(Material::Emissive {
        emit: Texture::new_from_color(glm::vec3(0.2, 0.2, 0.2)),
    });
    scene
        .spheres
        .push(Sphere::new(glm::vec3(0., 0., -3.), 0.5, 1));
    let mut tracer = Raytracer::new(
        &device,
        &queue,
        &config(params.viewport_size),
        &scene,
        &params,
    )
    .unwrap();
    let unfiltered = render(&device, &queue, &mut tracer);

    // Reuse the seed so each setting is tested against identical path samples.
    params.sampling.denoising_enabled = true;
    params.sampling.denoising_radius = 0;
    tracer.set_render_params(&device, &queue, &params).unwrap();
    tracer.frame_number = 1;
    assert_eq!(render(&device, &queue, &mut tracer), unfiltered);

    params.sampling.denoising_radius = 1;
    params.sampling.denoising_sigma_color = 1.;
    tracer.set_render_params(&device, &queue, &params).unwrap();
    tracer.frame_number = 1;
    let filtered = render(&device, &queue, &mut tracer);
    assert_ne!(filtered, unfiltered);

    params.sampling.denoising_radius = 3;
    tracer.set_render_params(&device, &queue, &params).unwrap();
    tracer.frame_number = 1;
    let wider_filter = render(&device, &queue, &mut tracer);
    assert_ne!(wider_filter, filtered);

    params.sampling.denoising_sigma_color = 0.01;
    tracer.set_render_params(&device, &queue, &params).unwrap();
    tracer.frame_number = 1;
    assert_ne!(render(&device, &queue, &mut tracer), wider_filter);
}

// Frozen pre-compute renderer, used only in tests on the same GPU as compute.
struct FragmentReference {
    accumulation: wgpu::RenderPipeline,
    display: wgpu::RenderPipeline,
}

impl FragmentReference {
    fn new(device: &wgpu::Device, tracer: &Raytracer) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frozen fragment reference"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tests/fragment_reference.wgsl").into()),
        });
        let vertex = tracer.display_pipeline.get_bind_group_layout(0);
        let parameters = tracer.trace_pipelines[0].get_bind_group_layout(2);
        let scene = tracer.trace_pipelines[0].get_bind_group_layout(3);
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reference trace layout"),
            bind_group_layouts: &[
                Some(&vertex),
                Some(&tracer.images.trace_layout),
                Some(&parameters),
                Some(&scene),
            ],
            immediate_size: 0,
        });
        let display_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reference display layout"),
            bind_group_layouts: &[
                Some(&vertex),
                Some(&tracer.images.filter_layout),
                Some(&parameters),
            ],
            immediate_size: 0,
        });
        let pipeline = |layout: &wgpu::PipelineLayout, entry, mask| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fragment parity reference"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vsMain"),
                    compilation_options: Default::default(),
                    buffers: &[Some(Vertex::desc())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: mask,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            accumulation: pipeline(&trace_layout, "fsAccumulate", wgpu::ColorWrites::empty()),
            display: pipeline(&display_layout, "fsMain", wgpu::ColorWrites::ALL),
        }
    }

    fn accumulate(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        tracer: &mut Raytracer,
    ) {
        tracer.prepare_frame(queue);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reference accumulation"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.accumulation);
        pass.set_bind_group(0, &tracer.vertex_uniform_bind_group, &[]);
        pass.set_bind_group(1, &tracer.images.trace, &[]);
        pass.set_bind_group(2, &tracer.parameter_bind_group, &[]);
        pass.set_bind_group(3, &tracer.scene_bind_group, &[]);
        pass.set_vertex_buffer(0, tracer.vertex_buffer.slice(..));
        pass.draw(0..VERTICES.len() as u32, 0..1);
    }

    fn display(&self, pass: &mut wgpu::RenderPass<'_>, tracer: &Raytracer) {
        pass.set_pipeline(&self.display);
        pass.set_bind_group(0, &tracer.vertex_uniform_bind_group, &[]);
        pass.set_bind_group(1, &tracer.images.filter, &[]);
        pass.set_bind_group(2, &tracer.parameter_bind_group, &[]);
        pass.set_vertex_buffer(0, tracer.vertex_buffer.slice(..));
        pass.draw(0..VERTICES.len() as u32, 0..1);
    }
}

#[test]
#[ignore = "requires a GPU"]
fn compute_matches_fragment_reference_through_accumulation_and_resize() {
    let (device, queue) = gpu();
    let mut params = params();
    params.camera = crate::FlyCameraController::default().renderer_camera();
    let scene = crate::scene();
    let mut compute = Raytracer::new(
        &device,
        &queue,
        &config(params.viewport_size),
        &scene,
        &params,
    )
    .unwrap();
    let mut fragment = Raytracer::new(
        &device,
        &queue,
        &config(params.viewport_size),
        &scene,
        &params,
    )
    .unwrap();
    let reference = FragmentReference::new(&device, &fragment);
    let mut largest_error = 0;
    for size in [(13, 9), (63, 47), (128, 96)] {
        for (enabled, radius) in [
            (false, 1),
            (true, 0),
            (true, 1),
            (true, 3),
            (true, 5),
            (true, 6),
        ] {
            params.viewport_size = size;
            params.sampling.denoising_enabled = enabled;
            params.sampling.denoising_radius = radius;
            compute.set_render_params(&device, &queue, &params).unwrap();
            fragment
                .set_render_params(&device, &queue, &params)
                .unwrap();
            for frame in 1..=9 {
                let actual = render(&device, &queue, &mut compute);
                let expected =
                    render_with_reference(&device, &queue, &mut fragment, None, Some(&reference));
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(&a, &b)| a.abs_diff(b))
                    .max()
                    .unwrap();
                largest_error = largest_error.max(error);
                assert!(
                    error <= 1,
                    "image mismatch: size={size:?} enabled={enabled} radius={radius} frame={frame} max_error={error}"
                );
            }
        }
    }
    eprintln!("Compute/fragment image comparison: maximum channel error {largest_error}/255");
}

#[test]
fn sky_distribution_has_full_support_and_normalized_solid_angle_pdf() {
    let sky = SkyParams::default().to_sky_state().unwrap();
    let mut previous = 0.0;
    let mut integral = 0.0;
    for cell in sky.sampling {
        assert!(cell[0] > previous && cell[1].is_finite() && cell[1] > 0.0);
        let mass = cell[1] * (4.0 * std::f32::consts::PI / 2048.0);
        assert!((mass - (cell[0] - previous)).abs() < 1e-7);
        previous = cell[0];
        integral += mass;
    }
    assert_eq!(previous, 1.0);
    assert!((integral - 1.0).abs() < 1e-5);
}

fn luminance(rgb: [f32; 3]) -> f64 {
    0.2126 * f64::from(rgb[0]) + 0.7152 * f64::from(rgb[1]) + 0.0722 * f64::from(rgb[2])
}

#[test]
fn sun_is_four_times_sky_at_reference() {
    let sky = SkyParams {
        zenith_degrees: 45.,
        turbidity: 4.,
        ..SkyParams::default()
    };
    let sun = luminance(super::sun_radiance(&sky))
        * f64::from(super::SUN_SOLID_ANGLE)
        * std::f64::consts::FRAC_1_SQRT_2;
    let ratio = sun / super::sky_illuminance(&sky);
    assert!(
        (3.6..4.4).contains(&ratio),
        "sun/sky horizontal illuminance {ratio}"
    );
}

#[test]
fn sun_state_handles_horizon_and_below() {
    let at = |zenith_degrees| {
        super::sun_radiance(&SkyParams {
            zenith_degrees,
            ..SkyParams::default()
        })
    };
    let (low, high) = (at(89.), at(30.));
    assert!(low.iter().all(|c| c.is_finite() && *c > 0.), "{low:?}");
    assert!(low[0] / low[2] > high[0] / high[2], "{low:?} vs {high:?}");
    assert_eq!(at(95.), [0.; 3]);
    let state = SkyParams {
        zenith_degrees: 95.,
        ..SkyParams::default()
    }
    .to_sky_state()
    .unwrap();
    assert_eq!(state.sun_radiance[..3], [0.; 3]);
}

#[test]
fn night_sky_is_dark_and_sunless() {
    let day = SkyParams::default().to_sky_state().unwrap();
    let night = SkyParams {
        night: true,
        ..SkyParams::default()
    }
    .to_sky_state()
    .unwrap();
    assert_eq!(night.sun_radiance[..3], [0.; 3]);
    assert!(day.sun_radiance[1] > 0.);
    for c in 0..3 {
        assert!(night.radiances[c] < 0.01 * day.radiances[c]);
    }
}

#[test]
#[ignore = "requires a GPU"]
fn tiny_viewports_render_with_bloom() {
    // The bloom pyramid of a 1x1 or 2x2 image has single-texel levels.
    let (device, queue) = gpu();
    for size in [(1, 1), (2, 1), (2, 2), (3, 2)] {
        let mut render_params = params();
        render_params.viewport_size = size;
        render_params.bloom = 0.04;
        let mut tracer = Raytracer::new(
            &device,
            &queue,
            &config(size),
            &uniform_scene(),
            &render_params,
        )
        .unwrap();
        let pixels = render(&device, &queue, &mut tracer);
        assert_eq!(pixels.len() as u32, size.0 * size.1 * 4, "{size:?}");
    }
}

#[test]
#[ignore = "requires a GPU"]
fn joint_network_matches_pytorch() {
    // training/upscale/parity_vector.py: model.step on a small random scene; the WGSL
    // (training/upscale/export.py) must agree over several frames of history.
    use wgpu::util::DeviceExt;
    let (device, queue) = gpu();
    let f16 = device.features().contains(wgpu::Features::SHADER_F16);
    let variants = ["tiny", "medium"]
        .into_iter()
        .flat_map(|tier| [(tier, ""), (tier, ".f16")])
        .filter(|&(_, variant)| variant.is_empty() || f16);
    for (tier, variant) in variants {
        let data = std::fs::read(format!("tests/assets/joint-parity-{tier}.bin")).unwrap();
        let words: Vec<u32> = bytemuck::pod_collect_to_vec(&data[..20]);
        let [w, h, ow, oh, frames] = words[..] else {
            panic!()
        };
        let floats: Vec<f32> = bytemuck::pod_collect_to_vec(&data[20..]);
        let model = upscale::JointModel {
            source: std::fs::read_to_string(format!("models/jn-{tier}{variant}.wgsl")).unwrap(),
            weights: std::fs::read(format!("models/jn-{tier}{variant}.bin")).unwrap(),
        };
        let (low, high) = ((w * h) as usize, (ow * oh) as usize);
        let storage = |size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let image = storage(low * 12);
        let gbuffer = storage(low * 48);
        let exposure = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let display_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0, 4, 6].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let mut upscaler = upscale::Upscaler::new(
            &device,
            &image,
            &gbuffer,
            &display_layout,
            &exposure,
            (ow, oh),
            (w, h),
            false,
            Some(&model),
        );
        let mut at = 0;
        let mut take = |n: usize| {
            let slice = &floats[at..at + n];
            at += n;
            slice.to_vec()
        };
        for frame in 0..frames {
            let jitter = take(2);
            queue.write_buffer(&image, 0, bytemuck::cast_slice(&take(low * 3)));
            queue.write_buffer(&gbuffer, 0, bytemuck::cast_slice(&take(low * 12)));
            queue.write_buffer(
                upscaler.out_guides().unwrap(),
                0,
                bytemuck::cast_slice(&take(high * 12)),
            );
            let expected = take(high * 3);
            let mut encoder = device.create_command_encoder(&Default::default());
            upscaler.record(&queue, &mut encoder, [jitter[0], jitter[1]], frame, 0, None);
            queue.submit([encoder.finish()]);
            let got = read_buffer(&device, &queue, upscaler.output());
            // Intermediates are f16: compare in log space.
            let errors: Vec<f32> = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a.max(0.).ln_1p() - b.max(0.).ln_1p()).abs())
                .collect();
            let max = errors.iter().copied().fold(0f32, f32::max);
            let mean = errors.iter().sum::<f32>() / errors.len() as f32;
            assert!(
                max < 2e-2 && mean < 2e-3,
                "{tier}{variant} frame {frame}: max {max}, mean {mean}"
            );
        }
    }
}
