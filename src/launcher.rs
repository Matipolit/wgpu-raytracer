//! Launcher menu (`weekend-raytracer-wgpu` with no arguments, or `--launcher`): pick a
//! scene, a quality preset and graphics settings, then restart the renderer with the
//! matching command line. Most settings are read from flags and baked into pipelines,
//! so launching a fresh process is simpler than switching them at runtime.
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct SceneEntry {
    pub name: String,
    pub args: Vec<String>,
    /// Uses the sky's sun (mesh scenes, not night).
    pub sun: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Low,
    Medium,
    High,
    Reference,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Denoiser {
    /// Progressive accumulation, no filter.
    Off,
    /// The edge-aware filter over the accumulated image.
    Spatial,
    /// A temporal neural denoiser: `models/<stem>.wgsl`.
    Neural(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Textures {
    /// The auto-tuner picks texture filtering per frame.
    Auto,
    /// Ray-cone filtering on every bounce.
    Full,
    /// Base mip only (fastest, aliased).
    Off,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub scene: String,
    pub preset: Preset,
    pub size: (u32, u32),
    pub fullscreen: bool,
    pub render_scale: f32,
    pub denoiser: Denoiser,
    pub bounces: u32,
    /// ReSTIR GI (reused bounce lighting; real-time denoised modes).
    pub gi: bool,
    pub max_samples: u32,
    pub hw_rt: bool,
    pub auto_tune: bool,
    pub textures: Textures,
    pub sun_elevation: f32,
    pub sun_azimuth: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            scene: "Bistro (day)".to_owned(),
            preset: Preset::Medium,
            size: (1280, 720),
            fullscreen: false,
            render_scale: 0.67,
            denoiser: Denoiser::Neural("td2-half24".to_owned()),
            bounces: 2,
            gi: true,
            max_samples: 1,
            hw_rt: false,
            auto_tune: true,
            textures: Textures::Auto,
            sun_elevation: 45.,
            sun_azimuth: 0.,
        }
    }
}

impl Preset {
    pub const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Reference];

    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Reference => "Reference",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Low => "half resolution, 2 bounces, small neural denoiser",
            Self::Medium => "67% resolution, 2 bounces + ReSTIR GI, best neural denoiser",
            Self::High => "full resolution, 2 bounces + ReSTIR GI, best neural denoiser",
            Self::Reference => "full resolution, 8 bounces, progressive (hold the camera still)",
        }
    }
}

impl Textures {
    pub const ALL: [Self; 3] = [Self::Auto, Self::Full, Self::Off];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto (tuner decides)",
            Self::Full => "Filtered on every bounce",
            Self::Off => "Unfiltered (fastest)",
        }
    }
}

impl Denoiser {
    pub fn label(&self) -> String {
        match self {
            Self::Off => "Off (progressive accumulation)".to_owned(),
            Self::Spatial => "Spatial filter".to_owned(),
            Self::Neural(stem) => format!("Neural: {stem}"),
        }
    }
}

impl Settings {
    /// Applies a preset's render scale, bounces, denoiser and sample budget; the
    /// neural denoisers fall back to the spatial filter when their files are missing.
    pub fn apply_preset(&mut self, preset: Preset, models: &[String]) {
        let neural = |stem: &str| {
            if models.iter().any(|m| m == stem) {
                Denoiser::Neural(stem.to_owned())
            } else {
                models
                    .last()
                    .map_or(Denoiser::Spatial, |m| Denoiser::Neural(m.clone()))
            }
        };
        self.preset = preset;
        // 2 bounces + ReSTIR GI beats 3 plain bounces in quality and cost (Deck:
        // Sponza 33 vs 36 ms per frame); Low drops GI for speed.
        (
            self.render_scale,
            self.bounces,
            self.gi,
            self.denoiser,
            self.max_samples,
        ) = match preset {
            Preset::Low => (0.5, 2, false, neural("td-half16"), 1),
            Preset::Medium => (0.67, 2, true, neural("td2-half24"), 1),
            Preset::High => (1., 2, true, neural("td2-half24"), 1),
            Preset::Reference => (1., 8, false, Denoiser::Off, 4096),
        };
    }

    /// The renderer's command line for these settings.
    pub fn args(&self, scenes: &[SceneEntry]) -> Vec<String> {
        let scene = scenes
            .iter()
            .find(|s| s.name == self.scene)
            .or_else(|| scenes.first());
        let mut args: Vec<String> = scene.map(|s| s.args.clone()).unwrap_or_default();
        let mut push = |flag: &str, value: Option<String>| {
            args.push(flag.to_owned());
            args.extend(value);
        };
        push("--fastest", None);
        if self.render_scale < 1. {
            push("--render-scale", Some(format!("{}", self.render_scale)));
        }
        match &self.denoiser {
            Denoiser::Off => push("--no-denoise", None),
            Denoiser::Spatial => {}
            Denoiser::Neural(stem) => push("--temporal-model", Some(format!("models/{stem}.wgsl"))),
        }
        push("--bounces", Some(self.bounces.to_string()));
        if !self.gi {
            push("--no-restir-gi", None);
        }
        if self.max_samples > 1 {
            push("--max-samples", Some(self.max_samples.to_string()));
        }
        if self.hw_rt {
            push("--hw-rt", None);
        }
        if !self.auto_tune {
            push("--no-auto-tune", None);
        }
        match self.textures {
            Textures::Auto => {}
            Textures::Full => push("--ray-cones", None),
            Textures::Off => push("--no-ray-cones", None),
        }
        if scene.is_some_and(|s| s.sun) {
            push("--sun-elevation", Some(format!("{}", self.sun_elevation)));
            push("--sun-azimuth", Some(format!("{}", self.sun_azimuth)));
        }
        push("--size", Some(format!("{}x{}", self.size.0, self.size.1)));
        if self.fullscreen {
            push("--fullscreen", None);
        }
        args
    }

    pub fn serialize(&self) -> String {
        let denoiser = match &self.denoiser {
            Denoiser::Off => "off".to_owned(),
            Denoiser::Spatial => "spatial".to_owned(),
            Denoiser::Neural(stem) => format!("neural:{stem}"),
        };
        format!(
            "scene={}\npreset={}\nsize={}x{}\nfullscreen={}\nrender_scale={}\ndenoiser={denoiser}\n\
             bounces={}\ngi={}\nmax_samples={}\nhw_rt={}\nauto_tune={}\ntextures={:?}\n\
             sun_elevation={}\nsun_azimuth={}\n",
            self.scene,
            self.preset.label(),
            self.size.0,
            self.size.1,
            self.fullscreen,
            self.render_scale,
            self.bounces,
            self.gi,
            self.max_samples,
            self.hw_rt,
            self.auto_tune,
            self.textures,
            self.sun_elevation,
            self.sun_azimuth,
        )
    }

    /// Settings from `key=value` lines; unknown keys and bad values keep the defaults.
    pub fn parse(text: &str) -> Self {
        let mut s = Self::default();
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "scene" if !value.is_empty() => s.scene = value.to_owned(),
                "preset" => {
                    if let Some(p) = Preset::ALL.into_iter().find(|p| p.label() == value) {
                        s.preset = p;
                    }
                }
                "size" => {
                    if let Some((w, h)) = value.split_once('x')
                        && let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>())
                        && w > 0
                        && h > 0
                    {
                        s.size = (w, h);
                    }
                }
                "fullscreen" => s.fullscreen = value.parse().unwrap_or(s.fullscreen),
                "render_scale" => {
                    if let Ok(v) = value.parse::<f32>()
                        && v > 0.
                        && v <= 1.
                    {
                        s.render_scale = v;
                    }
                }
                "denoiser" => match value {
                    "off" => s.denoiser = Denoiser::Off,
                    "spatial" => s.denoiser = Denoiser::Spatial,
                    _ => {
                        if let Some(stem) = value.strip_prefix("neural:")
                            && !stem.is_empty()
                        {
                            s.denoiser = Denoiser::Neural(stem.to_owned());
                        }
                    }
                },
                "bounces" => s.bounces = value.parse().unwrap_or(s.bounces),
                "gi" => s.gi = value.parse().unwrap_or(s.gi),
                "max_samples" => s.max_samples = value.parse().unwrap_or(s.max_samples),
                "hw_rt" => s.hw_rt = value.parse().unwrap_or(s.hw_rt),
                "auto_tune" => s.auto_tune = value.parse().unwrap_or(s.auto_tune),
                "textures" => {
                    if let Some(t) = Textures::ALL
                        .into_iter()
                        .find(|t| format!("{t:?}") == value)
                    {
                        s.textures = t;
                    }
                }
                "sun_elevation" => s.sun_elevation = value.parse().unwrap_or(s.sun_elevation),
                "sun_azimuth" => s.sun_azimuth = value.parse().unwrap_or(s.sun_azimuth),
                _ => {}
            }
        }
        s
    }
}

/// `sponza-atrium` → `Sponza atrium`.
fn display_name(key: &str) -> String {
    match key {
        "bistro-street-textured" => return "Bistro (day)".to_owned(),
        "bistro-street" => return "Bistro (untextured)".to_owned(),
        _ => {}
    }
    let words = key.replace('-', " ");
    let mut chars = words.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// Scenes from `performance/scenes/scenes.json` whose model exists, the Bistro night
/// variant, and the built-in sphere demo.
pub fn scenes(root: &Path) -> Vec<SceneEntry> {
    let mut found = Vec::new();
    let text =
        std::fs::read_to_string(root.join("performance/scenes/scenes.json")).unwrap_or_default();
    // One scene per line: "key": {"model": "...", "camera": "..."}
    for line in text.lines() {
        let fields: Vec<&str> = line.split('"').collect();
        let value = |name: &str| {
            fields
                .iter()
                .position(|f| *f == name)
                .and_then(|i| fields.get(i + 2))
                .copied()
        };
        let (Some(key), Some(model), Some(camera)) =
            (fields.get(1), value("model"), value("camera"))
        else {
            continue;
        };
        if !root.join(model).exists() {
            continue;
        }
        let args: Vec<String> = ["--model", model, "--camera", camera]
            .map(str::to_owned)
            .to_vec();
        found.push(SceneEntry {
            name: display_name(key),
            args: args.clone(),
            sun: true,
        });
        if *key == "bistro-street-textured" {
            found.push(SceneEntry {
                name: "Bistro (night)".to_owned(),
                args: [args, vec!["--night".to_owned()]].concat(),
                sun: false,
            });
        }
    }
    found.push(SceneEntry {
        name: "Sphere demo".to_owned(),
        args: Vec::new(),
        sun: false,
    });
    found
}

/// Temporal denoisers in `models/` (a `.wgsl` with its `.bin`), by file stem.
pub fn models(root: &Path) -> Vec<String> {
    let mut stems: Vec<String> = std::fs::read_dir(root.join("models"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|e| e == "wgsl") && path.with_extension("bin").exists()
        })
        .filter_map(|path| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    stems.sort();
    stems
}

/// The repository root: the working directory if it holds the scene list, else the
/// nearest ancestor of the executable that does (`target/release/..`).
pub fn root() -> PathBuf {
    let marker = Path::new("performance/scenes/scenes.json");
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join(marker).exists() {
        return cwd;
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.ancestors()
                .find(|dir| dir.join(marker).exists())
                .map(Path::to_path_buf)
        })
        .unwrap_or(cwd)
}

fn settings_path() -> Option<PathBuf> {
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config.join("wgpu-raytracer/launcher.txt"))
}

/// Starts the renderer (this executable) with `args` from the repository root. On
/// Unix the process replaces itself (the terminal, Ctrl+C and Steam keep tracking it);
/// elsewhere it starts a child process. Returns only if that failed or on spawn.
pub fn relaunch(args: &[String]) {
    let exe = std::env::current_exe().expect("current executable");
    println!("launching: {} {}", exe.display(), args.join(" "));
    let mut command = std::process::Command::new(exe);
    command.args(args).current_dir(root());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let error = command.exec();
        eprintln!("cannot start the renderer: {error}");
    }
    #[cfg(not(unix))]
    if let Err(error) = command.spawn() {
        eprintln!("cannot start the renderer: {error}");
    }
}

/// Running in Steam's Game Mode / on a Steam Deck (gamescope session).
fn handheld() -> bool {
    std::env::var_os("SteamDeck").is_some()
        || std::env::var("XDG_CURRENT_DESKTOP").is_ok_and(|d| d.eq_ignore_ascii_case("gamescope"))
        || std::env::var_os("GAMESCOPE_WAYLAND_DISPLAY").is_some()
}

const RESOLUTIONS: [(u32, u32); 7] = [
    (960, 540),
    (1280, 800),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];
const RENDER_SCALES: [(f32, &str); 5] = [
    (1., "100% (native)"),
    (0.75, "75%"),
    (0.67, "67%"),
    (0.5, "50%"),
    (0.33, "33%"),
];

pub fn run() {
    let event_loop = winit::event_loop::EventLoop::new().expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    let mut app = LauncherApp {
        state: None,
        gamepad: crate::gamepad::Gamepad::new(),
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

struct LauncherApp {
    state: Option<LauncherState>,
    gamepad: crate::gamepad::Gamepad,
}

struct LauncherState {
    // Drop order: egui state and UI before GPU resources and the window.
    egui_state: egui_winit::State,
    gui: crate::gui::Gui,
    context: crate::GpuContext,
    window: std::sync::Arc<winit::window::Window>,
    settings: Settings,
    scenes: Vec<SceneEntry>,
    models: Vec<String>,
    gpu_name: String,
    /// Controller A: launch at the next redraw.
    launch_requested: bool,
    _display_handle: winit::event_loop::OwnedDisplayHandle,
}

impl winit::application::ApplicationHandler for LauncherApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = std::sync::Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_title("weekend-raytracer-wgpu launcher")
                        .with_inner_size(winit::dpi::LogicalSize::new(860, 620)),
                )
                .expect("window"),
        );
        let display = event_loop.owned_display_handle();
        let context = pollster::block_on(crate::GpuContext::new(
            std::sync::Arc::clone(&window),
            display.clone(),
        ));
        let gui = crate::gui::Gui::new(&context.device, context.surface_config.format);
        let egui_state = egui_winit::State::new(
            gui.context.clone(),
            egui::ViewportId::ROOT,
            &display,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(context.device.limits().max_texture_dimension_2d as usize),
        );
        let root = root();
        let scenes = scenes(&root);
        let models = models(&root);
        let saved = settings_path().and_then(|path| std::fs::read_to_string(path).ok());
        let mut settings = saved.as_deref().map_or_else(
            || {
                let mut s = Settings::default();
                if handheld() {
                    // Steam Deck class: native 1280x800, fullscreen, cheapest preset.
                    s.apply_preset(Preset::Low, &models);
                    (s.size, s.fullscreen) = ((1280, 800), true);
                    s.scene = "Sponza atrium".to_owned();
                } else {
                    s.apply_preset(Preset::Medium, &models);
                }
                // With hardware RT on, the startup calibration still picks compute,
                // hardware shadows or hardware for all rays per GPU and scene. On the
                // Deck it is never faster (RDNA2 traverses in the shader), so the
                // structure is not even built there by default.
                s.hw_rt = context.ray_query && !handheld();
                s
            },
            Settings::parse,
        );
        settings.hw_rt &= context.ray_query;
        if !scenes.iter().any(|s| s.name == settings.scene) {
            settings.scene.clone_from(&scenes[0].name);
        }
        if let Denoiser::Neural(stem) = &settings.denoiser
            && !models.contains(stem)
        {
            settings.denoiser = Denoiser::Spatial;
        }
        let gpu_name = context
            .adapter_key
            .split('|')
            .next()
            .unwrap_or_default()
            .to_owned();
        window.request_redraw();
        self.state = Some(LauncherState {
            egui_state,
            gui,
            context,
            window,
            settings,
            scenes,
            models,
            gpu_name,
            launch_requested: false,
            _display_handle: display,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        use winit::event::WindowEvent;
        let Some(state) = &mut self.state else {
            return;
        };
        let response = state.egui_state.on_window_event(&state.window, &event);
        if response.repaint {
            state.window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                state.context.surface_config.width = size.width;
                state.context.surface_config.height = size.height;
                state
                    .context
                    .surface
                    .configure(&state.context.device, &state.context.surface_config);
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if state.redraw(event_loop) {
                    if let Some(path) = settings_path() {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = std::fs::write(path, state.settings.serialize());
                    }
                    relaunch(&state.settings.args(&state.scenes));
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Controller: D-pad up/down scene, left/right preset, A launches, B quits.
        let pad = self.gamepad.poll();
        if let Some(state) = &mut self.state
            && !pad.pressed.is_empty()
        {
            let names: Vec<&str> = state.scenes.iter().map(|s| s.name.as_str()).collect();
            let scene = names
                .iter()
                .position(|n| *n == state.settings.scene)
                .unwrap_or(0);
            let preset = Preset::ALL
                .iter()
                .position(|p| *p == state.settings.preset)
                .unwrap_or(0);
            for button in &pad.pressed {
                use gilrs::Button;
                match button {
                    Button::DPadUp => {
                        state.settings.scene =
                            names[(scene + names.len() - 1) % names.len()].to_owned();
                    }
                    Button::DPadDown => {
                        state.settings.scene = names[(scene + 1) % names.len()].to_owned();
                    }
                    Button::DPadLeft | Button::DPadRight => {
                        let step = if *button == Button::DPadLeft {
                            Preset::ALL.len() - 1
                        } else {
                            1
                        };
                        let next = Preset::ALL[(preset + step) % Preset::ALL.len()];
                        state.settings.apply_preset(next, &state.models);
                    }
                    Button::South => state.launch_requested = true,
                    Button::East => event_loop.exit(),
                    _ => {}
                }
            }
            state.window.request_redraw();
        }
        event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
            std::time::Instant::now() + std::time::Duration::from_millis(30),
        ));
    }

    fn exiting(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        self.state = None;
    }
}

impl LauncherState {
    /// Draws one frame; true when Launch was pressed.
    fn redraw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) -> bool {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return false;
        }
        let frame = match self.context.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            _ => {
                self.context
                    .surface
                    .configure(&self.context.device, &self.context.surface_config);
                self.window.request_redraw();
                return false;
            }
        };
        let input = self.egui_state.take_egui_input(&self.window);
        let mut launch = std::mem::take(&mut self.launch_requested);
        let (settings, scenes, models) = (&mut self.settings, &self.scenes, &self.models);
        let (gpu, ray_query) = (&self.gpu_name, self.context.ray_query);
        let mut output = self.gui.run_ui(input, |ui| {
            launch |= draw(ui, settings, scenes, models, gpu, ray_query);
        });
        self.egui_state.handle_platform_output_with_event_loop(
            &self.window,
            event_loop,
            std::mem::take(&mut output.platform_output),
        );
        let repaint = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|v| v.repaint_delay.is_zero());
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let commands = self.gui.prepare(
            &self.context.device,
            &self.context.queue,
            &mut encoder,
            Some(output),
            [size.width, size.height],
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("launcher"),
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
            self.gui.render(&mut pass);
        }
        self.context
            .queue
            .submit(commands.into_iter().chain([encoder.finish()]));
        self.gui.free_textures();
        self.window.pre_present_notify();
        self.context.queue.present(frame);
        if repaint {
            self.window.request_redraw();
        }
        launch
    }
}

/// The launcher UI; true when Launch was pressed.
fn draw(
    ui: &mut egui::Ui,
    settings: &mut Settings,
    scenes: &[SceneEntry],
    models: &[String],
    gpu: &str,
    ray_query: bool,
) -> bool {
    let mut launch = false;
    egui::CentralPanel::default().show(ui, |ui| {
        ui.heading("weekend-raytracer-wgpu");
        ui.label(format!("GPU: {gpu}"));
        ui.separator();
        ui.columns(2, |columns| {
            let ui = &mut columns[0];
            ui.strong("Scene");
            egui::ScrollArea::vertical()
                .id_salt("scenes")
                .show(ui, |ui| {
                    for scene in scenes {
                        ui.selectable_value(&mut settings.scene, scene.name.clone(), &scene.name);
                    }
                });
            let ui = &mut columns[1];
            ui.strong("Quality preset");
            for preset in Preset::ALL {
                let label = format!("{}: {}", preset.label(), preset.description());
                if ui.radio(settings.preset == preset, label).clicked() {
                    settings.apply_preset(preset, models);
                }
            }
            ui.add_space(8.);
            ui.strong("Graphics");
            egui::Grid::new("graphics").num_columns(2).show(ui, |ui| {
                graphics(ui, settings, scenes, models, ray_query);
            });
            ui.add_space(12.);
            let button = egui::Button::new(egui::RichText::new("Launch").heading());
            launch = ui.add_sized([ui.available_width(), 40.], button).clicked();
            ui.small(
                "In the renderer: hold the right mouse button to look, WASD to move, Q/E \
                 down/up; \"Back to launcher\" is in the parameter panel.",
            );
            ui.small(
                "Controller: D-pad scene/preset, A launch, B quit. In the renderer: sticks \
                 move/look, triggers down/up, L1/R1 slower/faster, Select hides the panel, \
                 Start returns here.",
            );
        });
    });
    launch
}

fn graphics(
    ui: &mut egui::Ui,
    settings: &mut Settings,
    scenes: &[SceneEntry],
    models: &[String],
    ray_query: bool,
) {
    ui.label("Resolution");
    egui::ComboBox::from_id_salt("resolution")
        .selected_text(format!("{}x{}", settings.size.0, settings.size.1))
        .show_ui(ui, |ui| {
            for size in RESOLUTIONS {
                let label = format!("{}x{}", size.0, size.1);
                ui.selectable_value(&mut settings.size, size, label);
            }
        });
    ui.end_row();
    ui.label("Fullscreen");
    ui.checkbox(&mut settings.fullscreen, "");
    ui.end_row();
    ui.label("Render scale");
    let scale_label = RENDER_SCALES
        .iter()
        .find(|(s, _)| (*s - settings.render_scale).abs() < 1e-3)
        .map_or_else(
            || format!("{:.0}%", settings.render_scale * 100.),
            |(_, l)| (*l).to_owned(),
        );
    egui::ComboBox::from_id_salt("scale")
        .selected_text(scale_label)
        .show_ui(ui, |ui| {
            for (scale, label) in RENDER_SCALES {
                ui.selectable_value(&mut settings.render_scale, scale, label);
            }
        });
    ui.end_row();
    ui.label("Denoiser");
    egui::ComboBox::from_id_salt("denoiser")
        .selected_text(settings.denoiser.label())
        .show_ui(ui, |ui| {
            let options = [Denoiser::Off, Denoiser::Spatial]
                .into_iter()
                .chain(models.iter().cloned().map(Denoiser::Neural));
            for option in options {
                let label = option.label();
                ui.selectable_value(&mut settings.denoiser, option, label);
            }
        });
    ui.end_row();
    ui.label("Bounces");
    ui.add(egui::Slider::new(&mut settings.bounces, 1..=8));
    ui.end_row();
    ui.label("ReSTIR GI");
    ui.checkbox(
        &mut settings.gi,
        "reused bounce light (neural denoiser modes)",
    );
    ui.end_row();
    // A temporal denoiser renders one fresh sample per frame; otherwise a still camera
    // accumulates up to this many samples per pixel.
    ui.label("Samples to accumulate");
    ui.add_enabled(
        !matches!(settings.denoiser, Denoiser::Neural(_)),
        egui::Slider::new(&mut settings.max_samples, 1..=4096).logarithmic(true),
    );
    ui.end_row();
    ui.label("Texture filtering");
    egui::ComboBox::from_id_salt("textures")
        .selected_text(settings.textures.label())
        .show_ui(ui, |ui| {
            for t in Textures::ALL {
                ui.selectable_value(&mut settings.textures, t, t.label());
            }
        });
    ui.end_row();
    ui.label("Hardware ray tracing");
    let note = if ray_query {
        "used where calibration finds it faster"
    } else {
        "not supported"
    };
    ui.add_enabled(ray_query, egui::Checkbox::new(&mut settings.hw_rt, note));
    ui.end_row();
    ui.label("Auto-tuner");
    ui.checkbox(&mut settings.auto_tune, "");
    ui.end_row();
    let sun = scenes
        .iter()
        .find(|s| s.name == settings.scene)
        .is_some_and(|s| s.sun);
    ui.label("Sun elevation");
    let elevation = egui::Slider::new(&mut settings.sun_elevation, 2.0..=90.0).suffix("°");
    ui.add_enabled(sun, elevation);
    ui.end_row();
    ui.label("Sun direction");
    let azimuth = egui::Slider::new(&mut settings.sun_azimuth, 0.0..=360.0).suffix("°");
    ui.add_enabled(sun, azimuth);
    ui.end_row();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("launcher-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    #[test]
    fn presets_map_to_flags() {
        let scenes = scenes(&repo());
        let models = vec!["td-half16".to_owned(), "td2-half24".to_owned()];
        let mut settings = Settings {
            scene: "Sponza atrium".to_owned(),
            ..Settings::default()
        };
        settings.apply_preset(Preset::High, &models);
        let args = settings.args(&scenes);
        let has = |flag: &str| args.iter().any(|a| a == flag);
        assert!(has("--fastest") && has("--temporal-model") && has("models/td2-half24.wgsl"));
        assert!(!has("--render-scale"), "{args:?}");
        assert!(
            has("--sponza") || has("assets/sponza/sponza-png.glb"),
            "{args:?}"
        );

        settings.apply_preset(Preset::Low, &models);
        let args = settings.args(&scenes);
        let at = args.iter().position(|a| a == "--render-scale").unwrap();
        assert_eq!(args[at + 1], "0.5");
        assert!(args.iter().any(|a| a == "models/td-half16.wgsl"));

        settings.apply_preset(Preset::Reference, &models);
        let args = settings.args(&scenes);
        assert!(args.iter().any(|a| a == "--no-denoise"));
        assert!(!args.iter().any(|a| a == "--temporal-model"));
        let at = args.iter().position(|a| a == "--max-samples").unwrap();
        assert_eq!(args[at + 1], "4096");

        settings.apply_preset(Preset::High, &[]);
        assert_eq!(settings.denoiser, Denoiser::Spatial);
    }

    #[test]
    fn night_and_sphere_scenes() {
        let scenes = scenes(&repo());
        let night = scenes.iter().find(|s| s.name == "Bistro (night)").unwrap();
        assert!(night.args.iter().any(|a| a == "--night") && !night.sun);
        let demo = scenes.iter().find(|s| s.name == "Sphere demo").unwrap();
        assert!(demo.args.is_empty() && !demo.sun);
        let settings = Settings {
            scene: "Sphere demo".to_owned(),
            ..Settings::default()
        };
        // The sphere demo still gets a flag, so the renderer does not reopen the launcher.
        assert!(!settings.args(&scenes).is_empty());
    }

    #[test]
    fn settings_roundtrip() {
        let settings = Settings {
            scene: "Bistro (night)".to_owned(),
            preset: Preset::Medium,
            size: (1920, 1080),
            fullscreen: true,
            render_scale: 0.67,
            denoiser: Denoiser::Neural("td2-half24".to_owned()),
            bounces: 4,
            gi: false,
            max_samples: 1,
            hw_rt: true,
            auto_tune: false,
            textures: Textures::Full,
            sun_elevation: 20.,
            sun_azimuth: 135.,
        };
        assert_eq!(Settings::parse(&settings.serialize()), settings);
    }

    #[test]
    fn settings_parse_ignores_garbage() {
        let parsed =
            Settings::parse("scene=Chess\nbounces=abc\nfoo=1\n=\nsize=12\nfullscreen=true\n");
        let defaults = Settings::default();
        assert_eq!(parsed.scene, "Chess");
        assert!(parsed.fullscreen);
        assert_eq!(parsed.bounces, defaults.bounces);
        assert_eq!(parsed.size, defaults.size);
    }

    #[test]
    fn missing_models_are_hidden() {
        let root = temp_root("empty");
        let found = scenes(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Sphere demo");
        assert!(models(&root).is_empty());
        std::fs::create_dir_all(root.join("models")).unwrap();
        std::fs::write(root.join("models/a.wgsl"), "").unwrap();
        std::fs::write(root.join("models/b.wgsl"), "").unwrap();
        std::fs::write(root.join("models/b.bin"), "").unwrap();
        assert_eq!(models(&root), vec!["b".to_owned()]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
