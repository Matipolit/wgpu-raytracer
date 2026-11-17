#![deny(
    clippy::pedantic,
    rust_2018_idioms,
    future_incompatible,
    nonstandard_style
)]

pub extern crate nalgebra_glm as glm;

use fly_camera::FlyCameraController;
use raytracer::{
    Material, Raytracer, RenderParams, SamplingParams, Scene, SkyParams, Sphere, Texture,
};
use std::{collections::VecDeque, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle},
    window::{Window, WindowId},
};

mod benchmark;
mod capture;
mod fly_camera;
mod gui;
mod offline;
mod raytracer;
mod shapes;

/// GPU, backend and driver, identifying the auto-tuner's calibration results.
pub(crate) fn adapter_key(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{}|{:?}|{}|{}",
        info.name, info.backend, info.driver, info.driver_info
    )
}

pub(crate) fn experiment_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    if has_flag("--fp16-state") {
        assert!(
            adapter.features().contains(wgpu::Features::SHADER_F16),
            "GPU does not support native shader f16"
        );
        wgpu::Features::SHADER_F16
    } else {
        wgpu::Features::empty()
    }
}

fn main() {
    if has_flag("--gpu-info") {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            println!(
                "{:?} f16={} limits={:?}",
                adapter.get_info(),
                adapter.features().contains(wgpu::Features::SHADER_F16),
                adapter.limits()
            );
        });
        return;
    }
    if std::env::args().any(|argument| argument == "--help") {
        println!(
            "Usage: cargo run --release -- [--sponza | --model FILE.gltf|FILE.glb] [--inspect-model] [--render-output FILE.png --samples N --size WIDTHxHEIGHT] [--russian-roulette | --no-russian-roulette] [--bounces N] [--sky-importance] [--ggx-vndf] [--exposure X | --exposure-compensation X] [--no-auto-exposure] [--camera x,y,z,dx,dy,dz,fov] [--move-speed UNITS_PER_S] [--sample-offset N] [--timing-output FILE.json] [--benchmark]\nDownload Sponza first with: python scripts/download_sponza.py\nLocal experiments: --bvh4 --bvh8 --quantized-bvh --bvh-leaf N --sah32 --sah-stop --packed-shading --stochastic-color --stochastic-all --fp16-state --packed-directions --skip-final-scatter --classify-opacity --approximate-alpha --alpha-cap4 --blue-noise-jitter --bounded-vndf --ray-cones --scene-cache"
        );
        return;
    }
    if std::env::args().any(|argument| argument == "--inspect-model") {
        let _ = selected_scene();
        return;
    }
    if has_flag("--capture") {
        capture::run();
        return;
    }
    if std::env::args().any(|argument| argument == "--render-output") {
        offline::run();
        return;
    }
    if std::env::args().any(|argument| argument == "--benchmark") {
        benchmark::run();
        return;
    }
    let event_loop = EventLoop::new().expect("Event loop should be available on desktop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        // Exercise the normal CloseRequested path in native lifecycle tests.
        smoke_frames_remaining: std::env::args()
            .any(|argument| argument == "--smoke-test-close")
            .then_some(3),
        ..Default::default()
    };
    event_loop.run_app(&mut app).expect("Event loop failed");
}

#[derive(Default)]
struct App {
    state: Option<AppState>,
    smoke_frames_remaining: Option<u32>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("weekend-raytracer-wgpu")
                        .with_inner_size(winit::dpi::PhysicalSize::new(800, 600)),
                )
                .expect("Window creation should succeed on desktop"),
        );
        self.state = Some(AppState::new(window, event_loop.owned_display_handle()));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };
        if window_id != state.window.id() {
            return;
        }
        let event = if matches!(event, WindowEvent::RedrawRequested)
            && self
                .smoke_frames_remaining
                .as_mut()
                .is_some_and(|remaining| {
                    *remaining -= 1;
                    *remaining == 0
                }) {
            WindowEvent::CloseRequested
        } else {
            event
        };
        let response = state.egui_state.on_window_event(&state.window, &event);
        // Always deliver releases, even when a widget takes focus mid-gesture.
        let release = matches!(&event,
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Released)
            || matches!(
                &event,
                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    ..
                }
            );
        if !response.consumed
            || release
            || (matches!(event, WindowEvent::CursorMoved { .. })
                && state.fly_camera_controller.look_pressed)
        {
            state.fly_camera_controller.handle_event(&event);
        }
        if matches!(event, WindowEvent::Focused(false)) {
            state.fly_camera_controller.release_input();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            // Winit sends Resized with the physical dimensions after a scale change.
            WindowEvent::RedrawRequested => state.render(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }

    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.state = None;
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        // Clipboard workers and GPU surfaces borrow the native display. Stop
        // them while winit's event loop (and its display connection) still lives.
        self.state = None;
    }
}

struct AppState {
    // Fields drop in declaration order: join clipboard workers first, then
    // release rendering resources, and keep the display alive through all of it.
    egui_state: egui_winit::State,
    gui: gui::Gui,
    raytracer: Raytracer,
    context: GpuContext,
    window: Arc<Window>,
    fly_camera_controller: FlyCameraController,
    /// WASD speed in scene units per second.
    move_speed: f32,
    render_params: RenderParams,
    last_time: Instant,
    fps_counter: FpsCounter,
    _display_handle: OwnedDisplayHandle,
}

impl AppState {
    fn new(window: Arc<Window>, display_handle: OwnedDisplayHandle) -> Self {
        let context =
            pollster::block_on(GpuContext::new(Arc::clone(&window), display_handle.clone()));
        let size = window.inner_size();
        let scene = selected_scene();
        let fly_camera_controller = scene_camera(&scene);
        let render_params = RenderParams {
            camera: fly_camera_controller.renderer_camera(),
            exposure: if has_flag("--no-auto-exposure") {
                scene_exposure(&scene)
            } else {
                AUTO_EXPOSURE_COMPENSATION
            },
            auto_exposure: !has_flag("--no-auto-exposure"),
            sky: scene_sky(&scene),
            sampling: SamplingParams {
                num_bounces: selected_bounces(),
                ..Default::default()
            },
            viewport_size: (size.width.max(1), size.height.max(1)),
        };
        let mut raytracer = Raytracer::new(
            &context.device,
            &context.queue,
            &context.surface_config,
            &scene,
            &render_params,
        )
        .expect("The default values should be selected correctly");
        raytracer.calibrate(&context.device, &context.queue, &context.adapter_key);
        let gui = gui::Gui::new(&context.device, context.surface_config.format);
        let egui_state = egui_winit::State::new(
            gui.context.clone(),
            egui::ViewportId::ROOT,
            &display_handle,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(context.device.limits().max_texture_dimension_2d as usize),
        );
        Self {
            window,
            context,
            fly_camera_controller,
            move_speed: scene_move_speed(&scene),
            render_params,
            raytracer,
            gui,
            egui_state,
            last_time: Instant::now(),
            fps_counter: FpsCounter::new(),
            _display_handle: display_handle,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let mut params = self.render_params;
        params.viewport_size = (size.width, size.height);
        if let Err(error) =
            self.raytracer
                .set_render_params(&self.context.device, &self.context.queue, &params)
        {
            eprintln!("Error resizing renderer: {error}");
            return;
        }
        self.render_params = params;
        self.context.surface_config.width = size.width;
        self.context.surface_config.height = size.height;
        self.context
            .surface
            .configure(&self.context.device, &self.context.surface_config);
    }

    fn render(&mut self, event_loop: &ActiveEventLoop) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        let (frame, suboptimal) = match self.context.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(size);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.context.surface = self
                    .context
                    .instance
                    .create_surface(Arc::clone(&self.window))
                    .expect("Surface recreation should succeed on desktop");
                self.resize(size);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface acquisition failed validation");
                event_loop.exit();
                return;
            }
        };
        let now = Instant::now();
        let dt = now - self.last_time;
        self.last_time = now;
        self.fps_counter.update(dt.as_secs_f32());
        self.fly_camera_controller.after_events(
            self.render_params.viewport_size,
            self.move_speed * dt.as_secs_f32(),
        );
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut output = self.gui.run(
            raw_input,
            &mut self.render_params,
            &mut self.fly_camera_controller,
            self.fps_counter.average_fps(),
            self.raytracer.progress(),
        );
        if let Some(output) = &mut output {
            self.egui_state.handle_platform_output_with_event_loop(
                &self.window,
                event_loop,
                std::mem::take(&mut output.platform_output),
            );
        }
        self.render_params.camera = self.fly_camera_controller.renderer_camera();
        if let Err(error) = self.raytracer.set_render_params(
            &self.context.device,
            &self.context.queue,
            &self.render_params,
        ) {
            eprintln!("Error setting render params: {error}");
        }
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let ui_commands = self.gui.prepare(
            &self.context.device,
            &self.context.queue,
            &mut encoder,
            output,
            [size.width, size.height],
        );
        self.raytracer
            .accumulate_frame(&self.context.queue, &mut encoder);
        {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("display and UI pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.raytracer.render_frame(&mut render_pass);
            self.gui.render(&mut render_pass);
        }
        self.context
            .queue
            .submit(ui_commands.into_iter().chain([encoder.finish()]));
        self.raytracer.after_submit(&self.context.device);
        self.gui.free_textures();
        self.window.pre_present_notify();
        self.context.queue.present(frame);
        if suboptimal {
            self.resize(self.window.inner_size());
        }
    }
}

struct GpuContext {
    /// GPU, backend and driver, for the auto-tuner's calibration cache.
    adapter_key: String,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    async fn new(window: Arc<Window>, display_handle: OwnedDisplayHandle) -> Self {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display_handle)),
        );
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Surface creation should succeed on desktop");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Adapter should be available on desktop");
        let (features, limits) = raytracer::gpu_textures::device_requirements(&adapter);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: experiment_features(&adapter) | features,
                required_limits: limits,
                experimental_features: crate::raytracer::gpu_textures::experimental_features(),
                ..Default::default()
            })
            .await
            .expect("Device should be available on desktop");
        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Surface should have a supported configuration");
        if let Some(format) = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(wgpu::TextureFormat::is_srgb)
        {
            surface_config.format = format;
        }
        surface.configure(&device, &surface_config);
        Self {
            adapter_key: adapter_key(&adapter.get_info()),
            instance,
            device,
            queue,
            surface,
            surface_config,
        }
    }
}

struct FpsCounter {
    frame_times: VecDeque<f32>,
}

impl FpsCounter {
    const MAX_FRAME_TIMES: usize = 8;

    pub fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(Self::MAX_FRAME_TIMES),
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.frame_times.push_back(dt);
        if self.frame_times.len() > Self::MAX_FRAME_TIMES {
            self.frame_times.pop_front();
        }
    }

    pub fn average_fps(&self) -> f32 {
        let sum: f32 = self.frame_times.iter().sum();
        self.frame_times.len() as f32 / sum
    }
}

fn scene() -> Scene {
    let materials = vec![
        Material::Checkerboard {
            even: Texture::new_from_color(glm::vec3(0.5_f32, 0.7_f32, 0.8_f32)),
            odd: Texture::new_from_color(glm::vec3(0.9_f32, 0.9_f32, 0.9_f32)),
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/moon.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::Metal {
            albedo: Texture::new_from_color(glm::vec3(1_f32, 0.85_f32, 0.57_f32)),
            fuzz: 0.4_f32,
        },
        Material::Dielectric {
            refraction_index: 1.5_f32,
        },
        Material::Lambertian {
            albedo: Texture::new_from_image("assets/earthmap.jpeg")
                .expect("Hardcoded path should be valid"),
        },
        Material::Emissive {
            emit: Texture::new_from_scaled_image("assets/sun.jpeg", 50.0)
                .expect("Hardcoded path should be valid"),
        },
        Material::Lambertian {
            albedo: Texture::new_from_color(glm::vec3(0.3_f32, 0.9_f32, 0.9_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(50.0_f32, 0.0_f32, 0.0_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(0.0_f32, 50.0_f32, 0.0_f32)),
        },
        Material::Emissive {
            emit: Texture::new_from_color(glm::vec3(0.0, 0.0, 50.0)),
        },
    ];

    let spheres = vec![
        Sphere::new(glm::vec3(0.0, -500.0, -1.0), 500.0, 0_u32),
        // left row
        Sphere::new(glm::vec3(-5.0, 1.0, -4.0), 1.0, 7_u32),
        Sphere::new(glm::vec3(0.0, 1.0, -4.0), 1.0, 8_u32),
        Sphere::new(glm::vec3(5.0, 1.0, -4.0), 1.0, 9_u32),
        // middle row
        Sphere::new(glm::vec3(-5.0, 1.0, 0.0), 1.0, 2_u32),
        Sphere::new(glm::vec3(0.0, 1.0, 0.0), 1.0, 3_u32),
        Sphere::new(glm::vec3(5.0, 1.0, 0.0), 1.0, 6_u32),
        // right row
        Sphere::new(glm::vec3(-5.0, 0.8, 4.0), 0.8, 1_u32),
        Sphere::new(glm::vec3(0.0, 1.2, 4.0), 1.2, 4_u32),
        Sphere::new(glm::vec3(5.0, 2.0, 4.0), 2.0, 5_u32),
    ];

    Scene {
        spheres,
        materials,
        ..Default::default()
    }
}

/// `--fastest` turns on every measured-best tracer option (exact or visually
/// equivalent); see performance/scenes/README.md.
const FASTEST: [&str; 4] = ["--sah-stop", "--skip-final-scatter", "--quad-bvh", "--sbvh"];

pub(crate) fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
        || (FASTEST.contains(&name) && std::env::args().any(|a| a == "--fastest"))
        // Paths starting from the visibility buffer need the visibility raster.
        || (name == "--visibility" && std::env::args().any(|a| a == "--visibility-primary"))
}

fn argument_value(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            return Some(args.next().unwrap_or_else(|| {
                eprintln!("{name} requires a value");
                std::process::exit(2)
            }));
        }
    }
    None
}

fn selected_bounces() -> u32 {
    let bounces =
        argument_value("--bounces").map_or(SamplingParams::default().num_bounces, |value| {
            value
                .parse()
                .expect("--bounces must be a non-negative integer")
        });
    // 0 traces nothing: a frame's fixed cost, for overhead measurements.
    bounces
}

fn selected_scene() -> Scene {
    let sponza = std::env::args().any(|arg| arg == "--sponza");
    let model = argument_value("--model");
    if sponza && model.is_some() {
        eprintln!("Choose either --sponza or --model");
        std::process::exit(2);
    }
    let path = model.or_else(|| sponza.then(|| "assets/sponza/sponza-png.glb".to_owned()));
    let Some(path) = path else {
        return scene();
    };
    let mut mesh =
        raytracer::mesh::Mesh::load(std::path::Path::new(&path)).unwrap_or_else(|error| {
            eprintln!("Cannot load {path}: {error}");
            std::process::exit(1)
        });
    if sponza {
        mesh.camera = Some(raytracer::Camera {
            eye_pos: glm::vec3(-8., 1.6, 0.),
            eye_dir: glm::normalize(&glm::vec3(1., 0.08, -0.08)),
            up: glm::vec3(0., 1., 0.),
            vfov: raytracer::Angle::degrees(65.),
            aperture: 0.,
            focus_distance: 10.,
        });
    }
    Scene {
        mesh,
        ..Default::default()
    }
}

fn scene_camera(scene: &Scene) -> FlyCameraController {
    let mut controller = FlyCameraController::default();
    if let Some(camera) = scene.mesh.camera {
        let direction = glm::normalize(&camera.eye_dir);
        controller.position = camera.eye_pos;
        controller.yaw = raytracer::Angle::radians(direction.z.atan2(direction.x));
        controller.pitch = raytracer::Angle::radians(direction.y.asin());
        controller.vfov_degrees = camera.vfov.as_degrees();
        controller.aperture = camera.aperture;
        controller.focus_distance = camera.focus_distance;
    } else if !scene.mesh.triangles.is_empty() {
        // Imported models without a preset view: no depth of field.
        controller.aperture = 0.;
    }
    if let Some((position, direction, vfov)) = camera_argument() {
        controller.position = position;
        controller.yaw = raytracer::Angle::radians(direction.z.atan2(direction.x));
        controller.pitch = raytracer::Angle::radians(direction.y.asin());
        controller.vfov_degrees = vfov;
        controller.aperture = 0.;
    }
    controller
}

/// `--camera x,y,z,dx,dy,dz,fov`: eye position, normalized view direction, vertical fov (degrees).
pub(crate) fn camera_argument() -> Option<(glm::Vec3, glm::Vec3, f32)> {
    let value = argument_value("--camera")?;
    let v: Vec<f32> = value
        .split(',')
        .map(|v| v.parse().expect("--camera expects numbers"))
        .collect();
    assert_eq!(v.len(), 7, "--camera x,y,z,dx,dy,dz,fov");
    Some((
        glm::vec3(v[0], v[1], v[2]),
        glm::normalize(&glm::vec3(v[3], v[4], v[5])),
        v[6],
    ))
}

/// `--move-speed`, else a fifteenth of the scene's bounding-box diagonal per
/// second (about Sponza's historical 2 units/s), else 2 for the sphere scene.
fn scene_move_speed(scene: &Scene) -> f32 {
    if let Some(speed) = argument_value("--move-speed") {
        return speed.parse().expect("--move-speed must be a number");
    }
    scene.mesh.nodes.first().map_or(2., |root| {
        glm::distance(&glm::Vec3::from(root.min), &glm::Vec3::from(root.max)) / 15.
    })
}

/// Exposure multiplier applied after automatic exposure maps the scene's
/// log-average luminance to middle grey; 2 is Hable's usual Uncharted 2 bias.
pub(crate) const AUTO_EXPOSURE_COMPENSATION: f32 = 2.;

fn scene_exposure(scene: &Scene) -> f32 {
    if scene.mesh.triangles.is_empty() {
        0.246
    } else {
        2.
    }
}
fn scene_sky(scene: &Scene) -> SkyParams {
    if scene.mesh.triangles.is_empty() {
        SkyParams::default()
    } else {
        SkyParams {
            zenith_degrees: 45.,
            ..SkyParams::default()
        }
    }
}
